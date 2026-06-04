"""
live_trader.py — Bar-by-bar live trading loop for XAUUSDm.

Runs every candle close, fetches fresh OHLCV from the DB (populated by
MT5-Collector), calls the ML model, and places an order if a signal fires.

Modes:
  --mode paper   Simulate trades locally (no MT5 needed). Default.
  --mode mt5     Send real orders to MT5 demo/live account.

Usage:
  python live_trader.py --timeframe 15min --mode paper
  python live_trader.py --timeframe 15min --mode mt5 --risk-pct 1.0
  python live_trader.py --timeframe 15min --mode mt5 --lot-size 0.01

Signal logic: defined by the Z&Z strategy pipeline (zone_detection_Z&Z.py,
entry_logic_Z&Z.py). evaluate_bar() is a stub until those files are built.
"""

import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from data.loader import get_connection
from data.feature_engineer import build_features
from models.trainer import MODEL_DIR, MODEL_FILE, METADATA_FILE
from utils.trade_logger import TradeLogger
from utils.db_writer import (
    write_market_context, write_ml_signal, update_ml_entry, update_ml_exit,
    utcnow as _db_utcnow,
)

# ── Timeframe → sleep seconds (wait for next candle close) ───────────────────
TF_SECONDS = {
    "1min":  60,   "2min":  120,  "3min":  180,  "4min":  240,
    "5min":  300,  "10min": 600,  "15min": 900,  "30min": 1800,
    "1H":    3600, "4H":    14400, "1D":   86400,
}

# ── Timeframe string → MT5 timeframe constant ─────────────────────────────────
TF_MT5 = {
    "1min":  1,
    "2min":  2,
    "3min":  3,
    "4min":  4,
    "5min":  5,
    "10min": 10,
    "15min": 15,
    "30min": 30,
    "1H":    16385,
    "4H":    16388,
    "1D":    16408,
}

# ── Defaults ──────────────────────────────────────────────────────────────────
SYMBOL                   = "XAUUSDm"
LOOKBACK_BARS            = 350
RISK_PCT                 = 1.0
DEFAULT_LOTS             = 0.01
MAX_CONCURRENT_POSITIONS = 2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("live_trader")


# ─────────────────────────────────────────────────────────────────────────────
#  Model bundle loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model_bundle() -> dict:
    model_path    = os.path.join(MODEL_DIR, MODEL_FILE)
    metadata_path = os.path.join(MODEL_DIR, METADATA_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run: python -m models.trainer --timeframes 15min --tune"
        )

    model = joblib.load(model_path)
    meta  = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

    feature_columns = meta.get("feature_columns", [])
    scaler          = meta.get("scaler")
    threshold       = float(meta.get("optimal_threshold", 0.5))
    tf_build_params = meta.get("tf_build_params", {})

    logger.info(
        "Model loaded | threshold=%.3f | features=%d | trained=%s",
        threshold, len(feature_columns), meta.get("trained_at", "?"),
    )
    return {
        "model":           model,
        "scaler":          scaler,
        "feature_columns": feature_columns,
        "threshold":       threshold,
        "tf_build_params": tf_build_params,
        "metadata":        meta,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Lot size calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_lot_size(
    broker,
    symbol: str,
    sl_distance: float,
    risk_pct: float,
    fallback_lots: float = DEFAULT_LOTS,
) -> float:
    try:
        import MetaTrader5 as mt5
        info    = mt5.symbol_info(symbol)
        account = mt5.account_info()
        if info is None or account is None:
            return fallback_lots
        balance       = float(account.balance)
        risk_amount   = balance * risk_pct / 100.0
        tick_value    = float(info.trade_tick_value)
        tick_size     = float(info.trade_tick_size)
        sl_ticks      = sl_distance / tick_size
        risk_per_lot  = sl_ticks * tick_value
        if risk_per_lot <= 0:
            return fallback_lots
        lots = risk_amount / risk_per_lot
        lots = max(lots, info.volume_min)
        lots = min(lots, info.volume_max)
        step = info.volume_step
        lots = round(round(lots / step) * step, 8)
        return lots
    except Exception:
        return fallback_lots


# ─────────────────────────────────────────────────────────────────────────────
#  Single-bar signal evaluation (STUB — implement in Z&Z pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bar(
    latest_bars: pd.DataFrame,
    bundle: dict,
    timeframe: str,
    include_london_ny: bool = True,
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
) -> dict:
    """
    Build features and run the ML model on the latest bar.
    Signal direction logic is delegated to the Z&Z pipeline —
    this stub returns no-trade until entry_logic_Z&Z.py is wired in.
    """
    no_trade = {
        "signal": 0, "sl": None, "tp": None,
        "rr": None, "prob": 0.0, "grade": None,
        "reason": "", "feat_row": None,
    }

    feat_df = build_features(
        latest_bars,
        h1_df=h1_df,
        h4_df=h4_df,
        impulse_atr_multiplier=bundle["tf_build_params"].get(
            timeframe, {}).get("impulse_atr_multiplier", 0.5),
        include_london_ny=include_london_ny,
    )
    if feat_df.empty:
        return {**no_trade, "reason": "build_features returned empty"}

    row = feat_df.iloc[-1]
    no_trade["feat_row"] = row

    # Build feature vector for model inference
    feature_columns = bundle["feature_columns"]
    if not feature_columns:
        return {**no_trade, "reason": "no feature columns defined (Z&Z pipeline not wired yet)"}

    last_row = feat_df.iloc[[-1]].copy()
    for col in feature_columns:
        if col not in last_row.columns:
            last_row[col] = 0.0
    X = last_row[feature_columns].fillna(0)

    scaler = bundle["scaler"]
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    model     = bundle["model"]
    threshold = bundle["threshold"]
    proba     = model.predict_proba(X)[0]
    classes   = list(model.classes_)
    prob_win  = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])

    # TODO: wire Z&Z entry logic here (zone detection, HTF bias, BOS/CHoCH filter,
    #       SL/TP from H4 zone levels). See entry_logic_Z&Z.py.
    return {**no_trade, "reason": f"Z&Z signal logic not yet implemented (prob={prob_win:.3f})"}


# ─────────────────────────────────────────────────────────────────────────────
#  Main trading loop
# ─────────────────────────────────────────────────────────────────────────────

class LiveTrader:

    def __init__(
        self,
        broker,
        mode: str,
        timeframe: str,
        symbol: str = SYMBOL,
        risk_pct: float = RISK_PCT,
        lot_size: float = 0.0,
    ):
        self.broker    = broker
        self.mode      = mode
        self.timeframe = timeframe
        self.symbol    = symbol
        self.risk_pct  = risk_pct
        self.lot_size  = lot_size
        self.db        = None if mode == "mt5" else get_connection()
        self.bundle    = load_model_bundle()

        self._open_positions: list = []

        _tf_bp = self.bundle["metadata"].get("tf_build_params", {}).get(timeframe, {})
        self.include_london_ny = bool(_tf_bp.get("include_london_ny", timeframe != "15min"))

        self.tlog = TradeLogger()

        logger.info(
            "LiveTrader | mode=%s tf=%s symbol=%s include_london_ny=%s",
            mode, timeframe, symbol, self.include_london_ny,
        )

        acct    = self.broker.get_account_info()
        balance = float(acct.get("balance", 0.0))
        self.tlog.log_session_start(
            mode=mode,
            timeframe=timeframe,
            symbol=symbol,
            threshold=self.bundle["threshold"],
            balance=balance,
        )

    def _fetch_bars(self):
        """Return (primary_df, h1_df, h4_df). HTF frames may be None in paper/db mode."""
        if self.mode == "mt5":
            return self._fetch_bars_from_mt5()
        primary = self._fetch_bars_from_db()
        return primary, None, None

    def _fetch_bars_from_mt5(self):
        import MetaTrader5 as mt5
        tf_const = TF_MT5.get(self.timeframe)
        if tf_const is None:
            raise ValueError(f"Unknown timeframe: {self.timeframe}")
        if not mt5.symbol_select(self.symbol, True):
            logger.warning("symbol_select(%s) failed", self.symbol)
        rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, LOOKBACK_BARS)
        if rates is None or len(rates) == 0:
            raise ConnectionError(
                f"mt5.copy_rates_from_pos returned nothing for {self.symbol} {self.timeframe}. "
                f"Error: {mt5.last_error()}"
            )
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df["hour"]  = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month
        body_top    = df[["open", "close"]].max(axis=1)
        body_bot    = df[["open", "close"]].min(axis=1)
        df["candle_size"] = df["high"] - df["low"]
        df["body_size"]   = (df["close"] - df["open"]).abs()
        df["wick_upper"]  = df["high"] - body_top
        df["wick_lower"]  = body_bot   - df["low"]
        df.drop(columns=["time"], inplace=True)
        primary = df.reset_index(drop=True)

        m5_df  = (primary if self.timeframe == "5min"
                  else self._fetch_htf_bars_mt5(mt5, TF_MT5["5min"],  n_bars=100))
        m15_df = (primary if self.timeframe == "15min"
                  else self._fetch_htf_bars_mt5(mt5, TF_MT5["15min"], n_bars=100))
        h1_df  = (primary if self.timeframe == "1H"
                  else self._fetch_htf_bars_mt5(mt5, TF_MT5["1H"],   n_bars=200))
        h4_df  = (primary if self.timeframe == "4H"
                  else self._fetch_htf_bars_mt5(mt5, TF_MT5["4H"],   n_bars=100))

        self._raw_tf_dfs = {"5min": m5_df, "15min": m15_df, "1H": h1_df, "4H": h4_df}
        self._write_market_context_bars(self._raw_tf_dfs, h4_df=h4_df)

        return primary, h1_df, h4_df

    def _write_market_context_bars(self, tf_dfs: dict, h4_df=None) -> None:
        import math

        def _safe(val):
            try:
                f = float(val)
                return None if (math.isnan(f) or math.isinf(f)) else f
            except (TypeError, ValueError):
                return None

        def _atr(df: pd.DataFrame, period: int = 14) -> tuple:
            if df is None or len(df) < period + 1:
                return None, None, None
            hi = df["high"].values
            lo = df["low"].values
            cl = df["close"].values
            tr = np.maximum(hi[1:] - lo[1:],
                            np.maximum(np.abs(hi[1:] - cl[:-1]),
                                       np.abs(lo[1:] - cl[:-1])))
            if len(tr) < period:
                return None, None, None
            atr_series = np.convolve(tr, np.ones(period) / period, mode="valid")
            current = float(atr_series[-1])
            avg     = float(atr_series[-min(5, len(atr_series)):].mean())
            ratio   = round(current / avg, 3) if avg > 0 else None
            return round(current, 5), round(avg, 5), ratio

        def _htf_bias_from_df(df: pd.DataFrame, n: int = 20) -> Optional[int]:
            if df is None or len(df) < n:
                return None
            closes = df["close"].values[-n:]
            recent = float(closes[-5:].mean())
            older  = float(closes[:5].mean())
            diff   = recent - older
            atr_est = float(np.mean(df["high"].values[-n:] - df["low"].values[-n:]))
            if atr_est <= 0:
                return 0
            if diff > 0.5 * atr_est:
                return 1
            if diff < -0.5 * atr_est:
                return -1
            return 0

        def _session(ts: pd.Timestamp) -> str:
            h = ts.hour
            if 12 <= h < 16:
                return "Overlap"
            if 7 <= h < 16:
                return "London"
            if 16 <= h < 21:
                return "NY"
            if h >= 23 or h < 9:
                return "Asian"
            return "Off"

        _h4 = h4_df if h4_df is not None else tf_dfs.get("4H")
        htf_bias_val = _htf_bias_from_df(_h4)
        market_structure_val = (
            "bullish" if htf_bias_val == 1 else
            ("bearish" if htf_bias_val == -1 else
             ("neutral" if htf_bias_val == 0 else None))
        )

        rows = []
        for tf_name, df in tf_dfs.items():
            if df is None or df.empty:
                continue
            last   = df.iloc[-1]
            ts_raw = last.get("timestamp")
            if ts_raw is None:
                continue
            ts = pd.to_datetime(ts_raw)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            vol_cur   = _safe(last.get("volume"))
            vol_avg   = _safe(df["volume"].tail(14).mean()) if "volume" in df.columns else None
            vol_ratio = round(vol_cur / vol_avg, 3) if (vol_cur and vol_avg and vol_avg > 0) else None

            atr_cur, atr_avg, atr_ratio = _atr(df)

            body   = _safe(last.get("body_size",   abs(float(last["close"]) - float(last["open"]))))
            candle = _safe(last.get("candle_size", float(last["high"]) - float(last["low"])))
            body_ratio = round(body / candle, 4) if (body is not None and candle and candle > 0) else None

            mom = None
            if len(df) > 5:
                try:
                    mom = round(float(df["close"].iloc[-1]) / float(df["close"].iloc[-6]) - 1, 5)
                except Exception:
                    pass

            rows.append({
                "timestamp":         ts.to_pydatetime(),
                "symbol":            self.symbol,
                "timeframe":         tf_name,
                "open":              _safe(last["open"]),
                "high":              _safe(last["high"]),
                "low":               _safe(last["low"]),
                "close":             _safe(last["close"]),
                "volume":            vol_cur,
                "htf_bias":          htf_bias_val,
                "market_structure":  market_structure_val,
                "session":           _session(ts),
                "atr_current":       atr_cur,
                "atr_average":       atr_avg,
                "atr_ratio":         atr_ratio,
                "volume_current":    vol_cur,
                "volume_average":    vol_avg,
                "volume_ratio":      vol_ratio,
                "candle_body_ratio": body_ratio,
                "momentum_score":    mom,
                "mins_to_news":      None,
                "news_impact":       None,
            })

        if rows:
            write_market_context(rows)
            logger.debug("market_context updated for %d timeframes", len(rows))

    def _fetch_htf_bars_mt5(self, mt5, tf_const: int, n_bars: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, n_bars)
        if rates is None or len(rates) == 0:
            logger.warning("Could not fetch HTF bars (tf=%d)", tf_const)
            return None
        htf = pd.DataFrame(rates)
        htf["timestamp"] = pd.to_datetime(htf["time"], unit="s")
        htf.rename(columns={"tick_volume": "volume"}, inplace=True)
        htf.drop(columns=["time"], inplace=True)
        return htf.reset_index(drop=True)

    def _fetch_bars_from_db(self) -> pd.DataFrame:
        if not self.db.connect():
            raise ConnectionError("DB connection failed")
        query = """
            SELECT timestamp, open, high, low, close, volume,
                   hour, day_of_week, month, year, session,
                   candle_size, body_size, wick_upper, wick_lower
            FROM xauusd_ohlcv
            WHERE symbol = %s AND timeframe = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = self.db.fetch_dataframe(query, (self.symbol, self.timeframe, LOOKBACK_BARS))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.iloc[::-1].reset_index(drop=True)

    def _sync_open_positions(self) -> None:
        if self.mode == "paper":
            still_open = []
            for pos in self._open_positions:
                if self.broker.is_position_open(pos["ticket"]):
                    still_open.append(pos)
                else:
                    logger.info("Position ticket=%s closed (SL/TP hit)", pos["ticket"])
                    self.tlog.log_exit(
                        ticket=pos["ticket"],
                        close_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        exit_price=0.0,
                        pnl=0.0,
                        close_reason="unknown (paper)",
                        balance_after=0.0,
                    )
            self._open_positions = still_open
            return

        try:
            import MetaTrader5 as mt5
            still_open = []
            for pos in self._open_positions:
                positions = mt5.positions_get(ticket=pos["ticket"])
                if positions:
                    still_open.append(pos)
                else:
                    logger.info("Position ticket=%d closed (SL/TP hit)", pos["ticket"])
                    deal = self.broker.get_closed_deal_info(pos["ticket"])
                    acct = self.broker.get_account_info()
                    self.tlog.log_exit(
                        ticket=pos["ticket"],
                        close_time=deal.get("close_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
                        exit_price=deal.get("exit_price", 0.0),
                        pnl=deal.get("pnl", 0.0),
                        close_reason=deal.get("close_reason", "unknown"),
                        balance_after=float(acct.get("balance", 0.0)),
                    )
            self._open_positions = still_open
        except Exception as e:
            logger.warning("Could not sync positions: %s", e)

    def _get_lots(self) -> float:
        if self.lot_size > 0:
            return self.lot_size
        return DEFAULT_LOTS

    def run_once(self) -> None:
        now = datetime.utcnow()
        logger.info("── Bar evaluation at %s UTC ──", now.strftime("%Y-%m-%d %H:%M"))

        self._sync_open_positions()
        open_count = len(self._open_positions)
        logger.info("Open positions: %d / %d", open_count, MAX_CONCURRENT_POSITIONS)

        if open_count >= MAX_CONCURRENT_POSITIONS:
            tickets = [p["ticket"] for p in self._open_positions]
            logger.info("Max positions reached (tickets=%s) — skipping entry", tickets)
            return

        try:
            bars, h1_df, h4_df = self._fetch_bars()
        except Exception as e:
            logger.error("Failed to fetch bars: %s", e)
            return

        if len(bars) < 50:
            logger.warning("Only %d bars — need more warmup", len(bars))
            return

        logger.info(
            "HTF context: h1=%s bars  h4=%s bars",
            len(h1_df) if h1_df is not None else "None",
            len(h4_df) if h4_df is not None else "None",
        )

        sig = evaluate_bar(
            bars, self.bundle, self.timeframe,
            self.include_london_ny, h1_df=h1_df, h4_df=h4_df,
        )

        bar_time = str(bars.iloc[-1].get("timestamp", now))
        self.tlog.log_signal(bar_time, sig, sig.get("feat_row"))

        if sig["signal"] == 0:
            logger.info("No signal: %s", sig["reason"])
            return

        direction_str = "buy" if sig["signal"] == 1 else "sell"

        if self._open_positions:
            existing_dir = self._open_positions[0]["direction"]
            if direction_str != existing_dir:
                logger.info(
                    "Skipping entry — signal is %s but open positions are %s (directional lock)",
                    direction_str, existing_dir,
                )
                return

        entry = float(bars.iloc[-1]["close"])
        lots  = self._get_lots()
        grade = sig.get("grade", "C")

        logger.info(
            "SIGNAL %s | entry=%.5f sl=%.5f tp=%.5f rr=%.2f prob=%.3f lots=%.2f",
            direction_str.upper(),
            entry, sig["sl"], sig["tp"], sig["rr"], sig["prob"], lots,
        )

        ticket = self.broker.place_order(
            symbol    = self.symbol,
            direction = direction_str,
            volume    = lots,
            sl        = sig["sl"],
            tp        = sig["tp"],
            comment   = f"lt_{self.timeframe}_{sig['prob']:.2f}",
        )

        if ticket is not None:
            self._open_positions.append({"ticket": ticket, "direction": direction_str})
            logger.info(
                "Order placed: ticket=%s | now %d/%d positions open",
                ticket, len(self._open_positions), MAX_CONCURRENT_POSITIONS,
            )
            acct = self.broker.get_account_info()
            self.tlog.log_entry(
                ticket=ticket,
                bar_time=bar_time,
                symbol=self.symbol,
                direction=direction_str,
                grade=grade,
                entry_price=entry,
                sl=sig["sl"],
                tp=sig["tp"],
                rr=sig["rr"],
                lots=lots,
                prob=sig["prob"],
                balance_before=float(acct.get("balance", 0.0)),
                feat_row=sig.get("feat_row"),
            )
        else:
            logger.error("Order placement failed")

    def run(self, interval_secs: Optional[int] = None) -> None:
        interval = interval_secs or TF_SECONDS.get(self.timeframe, 900)
        logger.info("Starting live trader | interval=%ds | Ctrl-C to stop", interval)

        try:
            while True:
                try:
                    self.run_once()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error("Unhandled error in run_once: %s", e, exc_info=True)

                logger.info("Sleeping %ds until next bar...", interval)
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            self.tlog.log_session_end()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live bar-by-bar trading loop for XAUUSDm ML model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_trader.py --timeframe 15min --mode paper
  python live_trader.py --timeframe 15min --mode mt5 --login 12345 --password MyPass --server Exness-MT5Trial
  python live_trader.py --timeframe 15min --mode mt5 --lot-size 0.01
  python live_trader.py --timeframe 15min --mode paper --once
        """,
    )
    parser.add_argument("--timeframe", default="15min",  help="Timeframe to trade")
    parser.add_argument("--symbol",    default=SYMBOL,   help="Symbol (default: XAUUSDm)")
    parser.add_argument("--mode",      default="paper",  choices=["paper", "mt5"])
    parser.add_argument("--login",     type=int,         help="MT5 account login number")
    parser.add_argument("--password",                    help="MT5 account password")
    parser.add_argument("--server",                      help="MT5 broker server name")
    parser.add_argument("--risk-pct",  type=float, default=RISK_PCT,
                        help="%% of balance to risk per trade (mt5 mode)")
    parser.add_argument("--lot-size",  type=float, default=0.0,
                        help="Fixed lot size (0 = auto from risk-pct)")
    parser.add_argument("--once",      action="store_true",
                        help="Evaluate one bar then exit (for testing)")
    args = parser.parse_args()

    if args.mode == "mt5":
        login    = args.login    or int(os.environ.get("MT5_LOGIN",    0))
        password = args.password or os.environ.get("MT5_PASSWORD", "")
        server   = args.server   or os.environ.get("MT5_SERVER",   "")
        if not all([login, password, server]):
            parser.error(
                "--mode mt5 requires --login, --password, and --server "
                "(or MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in .env)"
            )
        from execution.mt5_connector import MT5Connector
        from execution.mt5_executor  import MT5Executor
        connector = MT5Connector(login=login, password=password, server=server)
        if not connector.connect():
            raise SystemExit("Failed to connect to MT5 terminal")
        broker = MT5Executor(connector)
    else:
        from execution.paper_trader import PaperTrader
        broker = PaperTrader(starting_equity=10_000.0)
        broker.connect()

    trader = LiveTrader(
        broker    = broker,
        mode      = args.mode,
        timeframe = args.timeframe,
        symbol    = args.symbol,
        risk_pct  = args.risk_pct,
        lot_size  = args.lot_size,
    )

    if args.once:
        trader.run_once()
    else:
        trader.run()


if __name__ == "__main__":
    main()
