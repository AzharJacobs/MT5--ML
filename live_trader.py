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
import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
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

# ── Optional RL shadow (loads only when --rl-shadow flag is passed) ───────────
try:
    from rl.shadow import RLShadow as _RLShadow
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False

# ── Timeframe → sleep seconds (wait for next candle close) ───────────────────
TF_SECONDS = {
    "1min":  60,   "2min":  120,  "3min":  180,  "4min":  240,
    "5min":  300,  "10min": 600,  "15min": 900,  "30min": 1800,
    "1H":    3600, "4H":    14400, "1D":   86400,
}

# ── Timeframe string → MT5 timeframe constant ─────────────────────────────────
TF_MT5 = {
    "1min":  1,      # TIMEFRAME_M1
    "2min":  2,      # TIMEFRAME_M2
    "3min":  3,      # TIMEFRAME_M3
    "4min":  4,      # TIMEFRAME_M4
    "5min":  5,      # TIMEFRAME_M5
    "10min": 10,     # TIMEFRAME_M10
    "15min": 15,     # TIMEFRAME_M15
    "30min": 30,     # TIMEFRAME_M30
    "1H":    16385,  # TIMEFRAME_H1
    "4H":    16388,  # TIMEFRAME_H4
    "1D":    16408,  # TIMEFRAME_D1
}

# ── Defaults ──────────────────────────────────────────────────────────────────
SYMBOL                   = "XAUUSDm"
LOOKBACK_BARS            = 350     # need enough for warmup in build_features()
RISK_PCT                 = 1.0     # % of account balance risked per trade (when lot-size is 0)
DEFAULT_LOTS             = 0.01    # fallback if account-based sizing fails
MAX_CONCURRENT_POSITIONS = 1       # one trade at a time — safe for $50 account

# ── Grade system (mirrors backtest engine.py exactly) ─────────────────────────
GRADE_A_MIN_ZONE_QUALITY = 3.5
GRADE_A_MIN_CONFIDENCE   = 0.42
GRADE_B_MIN_ZONE_QUALITY = 3.0
GRADE_B_MIN_CONFIDENCE   = 0.40
# B=0 means skip; A=0.02 lots, C=0.01 lots (fixed sizing matching backtest)
GRADE_MULTIPLIERS = {"A": 1, "B": 0, "C": 1}
GRADE_LOTS        = {"A": 0.02, "C": 0.01}   # fixed lot sizes per grade

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
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
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
    no_trade = {"signal": 0, "sl": None, "tp": None, "rr": None, "prob": 0.0, "grade": None, "reason": "", "feat_row": None}

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
    no_trade["feat_row"] = row   # attach row even on early exits so caller can log it

    # No hard session gate — in_session is a model feature; model decides.
    # (Matches backtest engine.py where session gate was removed.)

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
        "feat_row":  row,
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
        enable_rl_shadow: bool = False,
    ):
        self.broker     = broker
        self.mode       = mode
        self.timeframe  = timeframe
        self.symbol     = symbol
        self.risk_pct   = risk_pct
        self.lot_size   = lot_size
        self.db         = None if mode == "mt5" else get_connection()
        self.bundle     = load_model_bundle()

        # Optional RL shadow observer (never executes orders)
        self.rl_shadow = None
        if enable_rl_shadow:
            if not _RL_AVAILABLE:
                logger.warning("--rl-shadow requested but rl.shadow import failed (check stable-baselines3)")
            else:
                try:
                    self.rl_shadow = _RLShadow.load(
                        feature_columns=self.bundle["feature_columns"],
                        timeframe=timeframe,
                        symbol=symbol,
                    )
                    logger.info("RL shadow loaded (%s)", "multi-TF" if self.rl_shadow.is_multi_tf else "single-TF")
                except FileNotFoundError as exc:
                    logger.warning("RL shadow disabled — no trained model found: %s", exc)
        # Each entry: {"ticket": int, "direction": str ("buy"|"sell")}
        self._open_positions: list = []

        # Mirror training: 15min was trained without H16
        _tf_bp = self.bundle["metadata"].get("tf_build_params", {}).get(timeframe, {})
        self.include_london_ny = bool(_tf_bp.get("include_london_ny", timeframe != "15min"))

        self.tlog = TradeLogger()

        logger.info(
            "LiveTrader | mode=%s tf=%s symbol=%s include_london_ny=%s",
            mode, timeframe, symbol, self.include_london_ny,
        )

        # Log session start — captures threshold and starting balance
        acct = self.broker.get_account_info()
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
        # Ensure symbol is visible in Market Watch
        if not mt5.symbol_select(self.symbol, True):
            logger.warning("symbol_select(%s) failed — symbol may not exist on this broker", self.symbol)
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

        h1_df = self._fetch_htf_bars_mt5(mt5, TF_MT5["1H"],  n_bars=200)
        h4_df = self._fetch_htf_bars_mt5(mt5, TF_MT5["4H"],  n_bars=100)
        return primary, h1_df, h4_df

    def _fetch_htf_bars_mt5(self, mt5, tf_const: int, n_bars: int) -> pd.DataFrame:
        """Fetch H1 or H4 bars from MT5 for HTF trend context."""
        rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, n_bars)
        if rates is None or len(rates) == 0:
            logger.warning("Could not fetch HTF bars (tf=%d) — htf_4h_bias will be 0", tf_const)
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
        return df.iloc[::-1].reset_index(drop=True)   # chronological

    def _sync_open_positions(self) -> None:
        """Remove any positions that have been closed by SL/TP and log their outcome."""
        if self.mode == "paper":
            still_open = []
            for pos in self._open_positions:
                if self.broker.is_position_open(pos["ticket"]):
                    still_open.append(pos)
                else:
                    logger.info("Position ticket=%s closed (SL/TP hit)", pos["ticket"])
                    # Paper mode has no price simulation — log exit as unknown
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

    def _get_lots(self, grade: str, sl_distance: float = 0.0) -> float:
        """Risk-based lot sizing — risks RISK_PCT% of balance per trade.
        Grade A gets 1.5× the base risk allocation; Grade C gets 1.0×.
        Falls back to fixed DEFAULT_LOTS if MT5 data is unavailable."""
        if self.lot_size > 0:
            return self.lot_size
        if sl_distance > 0:
            grade_risk_multiplier = 1.5 if grade == "A" else 1.0
            adjusted_risk_pct = self.risk_pct * grade_risk_multiplier
            return calculate_lot_size(
                self.broker, self.symbol, sl_distance,
                adjusted_risk_pct, fallback_lots=DEFAULT_LOTS,
            )
        return DEFAULT_LOTS

    def run_once(self) -> None:
        """Evaluate one bar and place/skip order."""
        now = datetime.utcnow()
        logger.info("── Bar evaluation at %s UTC ──", now.strftime("%Y-%m-%d %H:%M"))

        # Sync open positions (remove any closed by SL/TP)
        self._sync_open_positions()
        open_count = len(self._open_positions)
        logger.info("Open positions: %d / %d", open_count, MAX_CONCURRENT_POSITIONS)

        if open_count >= MAX_CONCURRENT_POSITIONS:
            tickets = [p["ticket"] for p in self._open_positions]
            logger.info("Max positions reached (tickets=%s) — skipping entry", tickets)
            return

        # Fetch data
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

        # Evaluate signal
        sig = evaluate_bar(bars, self.bundle, self.timeframe, self.include_london_ny, h1_df=h1_df, h4_df=h4_df)

        # ── RL shadow: observe same bar, log what it would do (no execution) ──
        if self.rl_shadow is not None:
            feat_df = sig.get("feat_row")
            # feat_row is a Series; wrap back to a 1-row DataFrame for the shadow
            if feat_df is not None and not isinstance(feat_df, pd.DataFrame):
                feat_df = pd.DataFrame([feat_df])
            rl_sig = self.rl_shadow.observe(feat_df, ml_signal=sig)
            rl_dir = {1: "BUY", -1: "SELL", 0: "hold"}.get(rl_sig["signal"], "hold")
            logger.info(
                "RL shadow: %s  sl=%.5f  tp=%.5f  rr=%s  (vpos=%d veq=%.2f)",
                rl_dir,
                rl_sig["sl"] or 0, rl_sig["tp"] or 0, rl_sig["rr"],
                self.rl_shadow._position, self.rl_shadow._equity,
            )

        bar_time = str(bars.iloc[-1].get("timestamp", now))
        self.tlog.log_signal(bar_time, sig, sig.get("feat_row"))

        if sig["signal"] == 0:
            logger.info("No signal: %s", sig["reason"])
            return

        direction_str = "buy" if sig["signal"] == 1 else "sell"

        # Directional consistency: only add trades in same direction as open positions
        # (Backtrader netting + live MT5 netting don't support simultaneous long+short)
        if self._open_positions:
            existing_dir = self._open_positions[0]["direction"]
            if direction_str != existing_dir:
                logger.info(
                    "Skipping entry — signal is %s but open positions are %s (directional lock)",
                    direction_str, existing_dir,
                )
                return

        entry = float(bars.iloc[-1]["close"])
        grade = sig.get("grade", "C")
        sl_distance = abs(entry - sig["sl"]) if sig.get("sl") else 0.0
        lots  = self._get_lots(grade, sl_distance=sl_distance)

        logger.info(
            "SIGNAL %s | Grade=%s entry=%.5f sl=%.5f tp=%.5f rr=%.2f prob=%.3f lots=%.2f",
            direction_str.upper(), grade,
            entry, sig["sl"], sig["tp"], sig["rr"], sig["prob"], lots,
        )

        ticket = self.broker.place_order(
            symbol    = self.symbol,
            direction = direction_str,
            volume    = lots,
            sl        = sig["sl"],
            tp        = sig["tp"],
            comment   = f"lt_{self.timeframe}_{grade}_{sig['prob']:.2f}",
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
        """
        Run indefinitely, evaluating one bar per interval.
        interval_secs defaults to the candle size for the timeframe.
        """
        interval = interval_secs or TF_SECONDS.get(self.timeframe, 900)
        logger.info(
            "Starting live trader | interval=%ds | Ctrl-C to stop", interval
        )

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
    parser.add_argument("--risk-pct",   type=float, default=RISK_PCT,  help="%% of balance to risk per trade (mt5 mode)")
    parser.add_argument("--lot-size",   type=float, default=0.0,        help="Fixed lot size (0 = auto from risk-pct)")
    parser.add_argument("--once",       action="store_true",            help="Evaluate one bar then exit (for testing)")
    parser.add_argument("--rl-shadow",  action="store_true",
                        help="Run the trained RL agent in shadow mode (logs signals, no execution)")
    args = parser.parse_args()

    if args.mode == "mt5":
        # Fall back to .env values if CLI args not provided
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
        broker            = broker,
        mode              = args.mode,
        timeframe         = args.timeframe,
        symbol            = args.symbol,
        risk_pct          = args.risk_pct,
        lot_size          = args.lot_size,
        enable_rl_shadow  = args.rl_shadow,
    )

    if args.once:
        trader.run_once()
    else:
        trader.run()


if __name__ == "__main__":
    main()
