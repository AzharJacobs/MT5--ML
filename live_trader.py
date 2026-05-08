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

Signal logic (mirrors backtest engine exactly):
  1. build_features() on last 300 bars
  2. Model predicts winner probability on latest bar
  3. Probability >= saved threshold AND in_session → check direction
  4. in_demand_zone=1 → buy, in_supply_zone=1 → sell
  5. SL/TP computed from zone levels + ATR (same as signal_generator.py)
  6. One position at a time — skip if MT5/paper position already open
"""

import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Optional, Tuple

from data.loader import get_connection
from data.feature_engineer import build_features
from models.trainer import MODEL_DIR, MODEL_FILE, METADATA_FILE

# ── Timeframe → sleep seconds (wait for next candle close) ───────────────────
TF_SECONDS = {
    "1min":  60,   "2min":  120,  "3min":  180,  "4min":  240,
    "5min":  300,  "10min": 600,  "15min": 900,  "30min": 1800,
    "1H":    3600, "4H":    14400, "1D":   86400,
}

# ── Defaults ──────────────────────────────────────────────────────────────────
SYMBOL        = "XAUUSDm"
LOOKBACK_BARS = 350     # need enough for warmup in build_features()
RISK_PCT      = 1.0     # % of account balance risked per trade (when lot-size is 0)
DEFAULT_LOTS  = 0.01    # fallback if account-based sizing fails

# ── Grade system (mirrors backtest engine.py exactly) ─────────────────────────
GRADE_A_MIN_ZONE_QUALITY = 3.5
GRADE_A_MIN_CONFIDENCE   = 0.42
GRADE_B_MIN_ZONE_QUALITY = 3.0
GRADE_B_MIN_CONFIDENCE   = 0.40
# B=0 means skip; scale A multiplier as account grows (A=2x at $150, A=3x at $300+)
GRADE_MULTIPLIERS = {"A": 1, "B": 0, "C": 1}

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
    # trainer.py saves self.metadata flat (not nested under a "metadata" key)
    meta = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

    feature_columns  = meta.get("feature_columns", [])
    scaler           = meta.get("scaler")
    threshold        = float(meta.get("optimal_threshold", 0.5))
    tf_build_params  = meta.get("tf_build_params", {})

    logger.info(
        "Model loaded | threshold=%.3f | features=%d | trained=%s",
        threshold, len(feature_columns), meta.get("trained_at", "?"),
    )
    return {
        "model":            model,
        "scaler":           scaler,
        "feature_columns":  feature_columns,
        "threshold":        threshold,
        "tf_build_params":  tf_build_params,
        "metadata":         meta,    # kept for convenience (full dict)
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SL / TP computation (mirrors signal_generator.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _f(val) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def compute_sl_tp(
    row: pd.Series,
    direction: int,          # 1=buy, -1=sell
    sl_buffer_atr: float = 0.5,
    min_rr: float = 1.5,
    use_midline_tp: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (sl, tp, rr) using zone levels + ATR, matching signal_generator.py.
    Returns (None, None, None) if levels are missing or RR too low.
    """
    close = float(row["close"])
    atr   = _f(row.get("atr_14"))
    if atr is None or np.isnan(atr) or atr <= 0:
        return None, None, None

    if direction == 1:  # buy
        d_bottom = _f(row.get("demand_zone_bottom"))
        s_bottom = _f(row.get("supply_zone_bottom"))
        if np.isnan(d_bottom):
            return None, None, None
        sl   = d_bottom - sl_buffer_atr * atr
        risk = close - sl
        if risk <= 0:
            return None, None, None
        if not np.isnan(s_bottom) and s_bottom > close:
            tp = close + (s_bottom - close) * (0.5 if use_midline_tp else 1.0)
        else:
            tp = close + max(min_rr * risk, 3.0 * atr)
        reward = tp - close

    else:  # sell
        s_top = _f(row.get("supply_zone_top"))
        d_top = _f(row.get("demand_zone_top"))
        if np.isnan(s_top):
            return None, None, None
        sl   = s_top + sl_buffer_atr * atr
        risk = sl - close
        if risk <= 0:
            return None, None, None
        if not np.isnan(d_top) and d_top < close:
            tp = close - (close - d_top) * (0.5 if use_midline_tp else 1.0)
        else:
            tp = close - max(min_rr * risk, 3.0 * atr)
        reward = close - tp

    if reward <= 0:
        return None, None, None

    rr = reward / risk
    if rr < min_rr:
        return None, None, None

    return round(sl, 5), round(tp, 5), round(rr, 2)


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
    """
    Risk-based position sizing.
    sl_distance: price distance from entry to SL (in quote currency).
    """
    try:
        import MetaTrader5 as mt5
        info    = mt5.symbol_info(symbol)
        account = mt5.account_info()
        if info is None or account is None:
            return fallback_lots
        balance       = float(account.balance)
        risk_amount   = balance * risk_pct / 100.0
        contract_size = float(info.trade_contract_size)   # e.g. 10 for XAUUSDm
        tick_value    = float(info.trade_tick_value)       # value of 1 tick in account currency
        tick_size     = float(info.trade_tick_size)        # e.g. 0.01
        # sl_in_ticks * tick_value_per_lot = risk per lot
        sl_ticks    = sl_distance / tick_size
        risk_per_lot = sl_ticks * tick_value
        if risk_per_lot <= 0:
            return fallback_lots
        lots = risk_amount / risk_per_lot
        # Clamp to broker min/max
        lots = max(lots, info.volume_min)
        lots = min(lots, info.volume_max)
        # Round to broker step
        step = info.volume_step
        lots = round(round(lots / step) * step, 8)
        return lots
    except Exception:
        return fallback_lots


# ─────────────────────────────────────────────────────────────────────────────
#  Single-bar signal evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_bar(
    latest_bars: pd.DataFrame,
    bundle: dict,
    timeframe: str,
    include_london_ny: bool = True,
) -> dict:
    """
    Run model on the latest completed bar and return a signal dict.

    Returns:
        {
          "signal":    1 (buy) | -1 (sell) | 0 (no trade),
          "sl":        float or None,
          "tp":        float or None,
          "rr":        float or None,
          "prob":      float,
          "reason":    str,
        }
    """
    no_trade = {"signal": 0, "sl": None, "tp": None, "rr": None, "prob": 0.0, "grade": None, "reason": ""}

    feat_df = build_features(
        latest_bars,
        impulse_atr_multiplier=bundle["tf_build_params"].get(
            timeframe, {}).get("impulse_atr_multiplier", 0.5),
        include_london_ny=include_london_ny,
    )
    if feat_df.empty:
        return {**no_trade, "reason": "build_features returned empty"}

    row = feat_df.iloc[-1]

    # Session gate — must be in a valid trading window
    if float(row.get("in_session", 0) or 0) != 1.0:
        return {**no_trade, "reason": "outside session"}

    # Direction from features
    in_demand = float(row.get("in_demand_zone", 0) or 0) == 1.0
    in_supply = float(row.get("in_supply_zone", 0) or 0) == 1.0
    if not in_demand and not in_supply:
        return {**no_trade, "reason": "not in zone"}

    # Between-zones guard
    if float(row.get("between_zones", 0) or 0) == 1.0:
        return {**no_trade, "reason": "between zones"}

    # Build feature vector
    feature_columns = bundle["feature_columns"]
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
    # Class order: model.classes_ — find index of class 1 (winner)
    classes = list(model.classes_)
    prob_win  = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])

    if prob_win < threshold:
        return {**no_trade, "reason": f"prob={prob_win:.3f} < threshold={threshold:.3f}"}

    direction = 1 if in_demand else -1

    # Grade filter — mirrors backtest engine grade system
    zone_quality = float(row.get("active_zone_quality", 0) or 0)
    if zone_quality >= GRADE_A_MIN_ZONE_QUALITY and prob_win >= GRADE_A_MIN_CONFIDENCE:
        grade = "A"
    elif zone_quality >= GRADE_B_MIN_ZONE_QUALITY and prob_win >= GRADE_B_MIN_CONFIDENCE:
        grade = "B"
    else:
        grade = "C"

    if GRADE_MULTIPLIERS.get(grade, 1) == 0:
        return {**no_trade, "reason": f"Grade B skipped (zone_quality={zone_quality:.2f} prob={prob_win:.3f})"}

    sl, tp, rr = compute_sl_tp(row, direction)
    if sl is None:
        return {**no_trade, "reason": "SL/TP computation failed (missing zone levels or RR<1.5)"}

    return {
        "signal":    direction,
        "sl":        sl,
        "tp":        tp,
        "rr":        rr,
        "prob":      prob_win,
        "grade":     grade,
        "reason":    f"{'buy' if direction==1 else 'sell'} Grade={grade} zone_quality={zone_quality:.2f} prob={prob_win:.3f} rr={rr}",
    }


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
        lot_size: float = 0.0,   # 0 = auto-calculate from risk_pct
    ):
        self.broker     = broker
        self.mode       = mode
        self.timeframe  = timeframe
        self.symbol     = symbol
        self.risk_pct   = risk_pct
        self.lot_size   = lot_size
        self.db         = get_connection()
        self.bundle     = load_model_bundle()
        self._ticket: Optional[int] = None   # open position ticket

        # Mirror training: 15min was trained without H16
        _tf_bp = self.bundle["metadata"].get("tf_build_params", {}).get(timeframe, {})
        self.include_london_ny = bool(_tf_bp.get("include_london_ny", timeframe != "15min"))

        logger.info(
            "LiveTrader | mode=%s tf=%s symbol=%s include_london_ny=%s",
            mode, timeframe, symbol, self.include_london_ny,
        )

    def _fetch_bars(self) -> pd.DataFrame:
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
        return df.iloc[::-1].reset_index(drop=True)   # chronological

    def _is_position_open(self) -> bool:
        if self.mode == "paper":
            return self._ticket is not None
        # For MT5: check if our ticket is still open
        if self._ticket is None:
            return False
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(ticket=self._ticket)
            if positions:
                return True
            # Ticket gone — position closed by SL/TP
            logger.info("Position ticket=%d was closed (SL/TP hit)", self._ticket)
            self._ticket = None
            return False
        except Exception:
            return False

    def _get_lots(self, entry: float, sl: float) -> float:
        if self.lot_size > 0:
            return self.lot_size
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return DEFAULT_LOTS
        return calculate_lot_size(self.broker, self.symbol, sl_distance, self.risk_pct)

    def run_once(self) -> None:
        """Evaluate one bar and place/skip order."""
        now = datetime.utcnow()
        logger.info("── Bar evaluation at %s UTC ──", now.strftime("%Y-%m-%d %H:%M"))

        # Skip if position already open
        if self._is_position_open():
            logger.info("Position open (ticket=%s) — skipping entry", self._ticket)
            return

        # Fetch data
        try:
            bars = self._fetch_bars()
        except Exception as e:
            logger.error("Failed to fetch bars: %s", e)
            return

        if len(bars) < 50:
            logger.warning("Only %d bars — need more warmup", len(bars))
            return

        # Evaluate
        sig = evaluate_bar(bars, self.bundle, self.timeframe, self.include_london_ny)

        if sig["signal"] == 0:
            logger.info("No signal: %s", sig["reason"])
            return

        entry = float(bars.iloc[-1]["close"])
        direction_str = "buy" if sig["signal"] == 1 else "sell"
        lots = self._get_lots(entry, sig["sl"])

        logger.info(
            "SIGNAL %s | Grade=%s entry=%.5f sl=%.5f tp=%.5f rr=%.2f prob=%.3f lots=%.2f",
            direction_str.upper(), sig.get("grade", "?"),
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
            self._ticket = ticket
            logger.info("Order placed: ticket=%s", ticket)
        else:
            logger.error("Order placement failed")

    def run(self, interval_secs: Optional[int] = None) -> None:
        """
        Run indefinitely, evaluating one bar per interval.
        interval_secs defaults to the candle size for the timeframe.
        """
        interval = interval_secs or TF_SECONDS.get(self.timeframe, 900)
        logger.info(
            "Starting live trader | interval=%ds | Ctrl-C to stop", interval
        )

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
            except Exception as e:
                logger.error("Unhandled error in run_once: %s", e, exc_info=True)

            # Sleep until next candle close
            logger.info("Sleeping %ds until next bar...", interval)
            time.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live bar-by-bar trading loop for XAUUSDm ML model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading — no MT5 needed, simulates entries locally
  python live_trader.py --timeframe 15min --mode paper

  # MT5 demo — sends real orders to your demo account
  python live_trader.py --timeframe 15min --mode mt5 --login 12345 --password MyPass --server Exness-MT5Trial

  # Fixed 0.01 lot size instead of risk-based sizing
  python live_trader.py --timeframe 15min --mode mt5 --lot-size 0.01

  # Run once (useful for cron / testing)
  python live_trader.py --timeframe 15min --mode paper --once
        """,
    )
    parser.add_argument("--timeframe",  default="15min",  help="Timeframe to trade")
    parser.add_argument("--symbol",     default=SYMBOL,   help="Symbol (default: XAUUSDm)")
    parser.add_argument("--mode",       default="paper",  choices=["paper", "mt5"])
    parser.add_argument("--login",      type=int,         help="MT5 account login number")
    parser.add_argument("--password",                     help="MT5 account password")
    parser.add_argument("--server",                       help="MT5 broker server name")
    parser.add_argument("--risk-pct",   type=float, default=RISK_PCT,  help="% of balance to risk per trade (mt5 mode)")
    parser.add_argument("--lot-size",   type=float, default=0.0,        help="Fixed lot size (0 = auto from risk-pct)")
    parser.add_argument("--once",       action="store_true",            help="Evaluate one bar then exit (for testing)")
    args = parser.parse_args()

    if args.mode == "mt5":
        if not all([args.login, args.password, args.server]):
            parser.error("--mode mt5 requires --login, --password, and --server")
        from execution.mt5_connector import MT5Connector
        from execution.mt5_executor  import MT5Executor
        connector = MT5Connector(login=args.login, password=args.password, server=args.server)
        if not connector.connect():
            raise SystemExit("Failed to connect to MT5 terminal")
        broker = MT5Executor(connector)
    else:
        from execution.paper_trader import PaperTrader
        broker = PaperTrader(starting_equity=10_000.0)
        broker.connect()

    trader = LiveTrader(
        broker     = broker,
        mode       = args.mode,
        timeframe  = args.timeframe,
        symbol     = args.symbol,
        risk_pct   = args.risk_pct,
        lot_size   = args.lot_size,
    )

    if args.once:
        trader.run_once()
    else:
        trader.run()


if __name__ == "__main__":
    main()
