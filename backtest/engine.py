"""
backtest_backtrader.py - Proper trading backtest with Backtrader
===============================================================

Usage (examples):
  python backtest_backtrader.py --timeframe 1H --cash 10000 --stake 0.15
  python backtest_backtrader.py --timeframe 5min --confidence 0.52 --stake 0.15

CHANGES (trade frequency fix):
  - MIN_ZONE_QUALITY lowered from 3.5 → 2.0.
    At 3.5 only 9 trades fired across 46,702 bars (15min). The model was
    essentially never trading. 2.0 opens the gate to more zone encounters
    while still requiring a real zone (score 0 = no zone at all).

  - confidence default lowered from 0.55 → 0.52 in main().
    With so few trades the model never had enough samples to build high
    confidence. 0.52 keeps the signal meaningful while tripling trade count.

  - Diagnostic counters added to MLSignalStrategy and printed after results
    so you can see exactly which gate is filtering most bars.
    Gates: no_row | zone_quality | confidence | neutral_label | bad_sl_tp

  - Zone quality distribution printed before backtest runs so you can
    see the score spread and tune MIN_ZONE_QUALITY intelligently.

PREVIOUS CHANGES:
  - features_by_dt stores "raw" key with unscaled zone boundaries for SL/TP.
  - Default stake raised from 0.10 → 0.15.
  - MAX_CONCURRENT_POSITIONS = 2 guard added.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import backtrader as bt

from data.loader import get_connection
from data.pipeline import DataPreparator
from models.trainer import ModelTrainer, MODEL_DIR
from models.calibration import _PlattWrapper  # noqa: F401 — needed for joblib unpickling
from strategy.base_strategy import calculate_stop_loss, calculate_take_profit
from config.pipeline_config import MIN_ZONE_QUALITY, HTF_EXTREME_THRESHOLD, MIN_RR

MAX_CONCURRENT_POSITIONS = 2

# ── Signal grade thresholds ────────────────────────────────────────────────
# A = strong setup  (zone>=3.5, conf>=0.42) — sized up when account allows
# B = skipped       (zone 3.0-3.5) — 47% WR, not worth trading
# C = base setup    (everything else passing gates) — 67% WR bread-and-butter
#
# Multipliers: flat 1x across A and C for $50 account safety.
# Raise A multiplier as account grows (e.g. A=2 at $150, A=3 at $300+).
GRADE_A_MIN_ZONE_QUALITY = 3.5
GRADE_A_MIN_CONFIDENCE   = 0.58
GRADE_B_MIN_ZONE_QUALITY = 3.0
GRADE_B_MIN_CONFIDENCE   = 0.50
GRADE_MULTIPLIERS = {"A": 1, "B": 0, "C": 1}   # B=0 = skip; scale A when account grows

# Raw zone columns we need at execution time (unscaled, real price levels)
RAW_ZONE_COLS = [
    "demand_zone_bottom", "demand_zone_top",
    "supply_zone_bottom", "supply_zone_top",
    "htf_demand_zone_top", "htf_demand_zone_bottom",
    "htf_supply_zone_top", "htf_supply_zone_bottom",
    "atr_14",
    "htf_4h_bias",   # HTF context fed to model (unscaled: 1.0 / -1.0)
    "htf_1h_bias",   # HTF context fed to model
    "in_demand_zone", "in_supply_zone",  # direction source — model predicts winner/loser, not buy/sell
]


@dataclass(frozen=True)
class BacktestResult:
    final_value: float
    pnl: float
    max_drawdown_pct: float
    winrate_pct: float
    total_trades: int
    entries_submitted: int
    skipped_no_margin: int
    skipped_max_positions: int
    trail_activations: int
    filtered_no_row: int
    filtered_session: int
    filtered_zone_quality: int
    filtered_confidence: int
    filtered_neutral: int
    filtered_bad_sltp: int
    filtered_low_rr: int
    filtered_htf_filter: int = 0
    filtered_risk_atr: int = 0
    # Extended P&L fields
    start_cash: float = 10000.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    buy_trades: int = 0
    buy_wins: int = 0
    sell_trades: int = 0
    sell_wins: int = 0
    monthly_pnl: tuple = ()   # ((year, month, pnl_dollars), ...)
    # Grade breakdown
    grade_a_trades: int = 0
    grade_a_wins:   int = 0
    grade_b_trades: int = 0
    grade_b_wins:   int = 0
    grade_c_trades: int = 0
    grade_c_wins:   int = 0
    be_lock_count:  int = 0
    trade_log: tuple = ()  # raw per-trade entries for breakdown analysis


def _load_model_bundle(model_dir: str = MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
    model, metadata = ModelTrainer.load_model(model_dir=model_dir)
    return model, metadata


def _build_feature_matrix_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    metadata_bundle: Dict[str, Any],
    include_london_ny: bool = True,
    impulse_atr_multiplier: float = 0.5,
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
) -> pd.DataFrame:
    from data.feature_engineer import build_features

    saved_scaler    = metadata_bundle.get("scaler")
    feature_columns = metadata_bundle.get("feature_columns") or []

    if saved_scaler is None:
        raise ValueError("Saved scaler not found in model metadata. Retrain first.")
    if not feature_columns:
        raise ValueError("Saved feature_columns not found in model metadata. Retrain first.")

    data = df.copy()
    data = build_features(data, h1_df=h1_df, h4_df=h4_df,
                          include_london_ny=include_london_ny,
                          impulse_atr_multiplier=impulse_atr_multiplier)

    if "direction" in data.columns:
        data = data.drop(columns=["direction"])

    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    }
    session_map = {
        "asian": 0, "london": 1, "london_ny_overlap": 2,
        "new_york": 3, "off_hours": 4, "daily": 5, "unknown": -1,
    }

    if "day_of_week" in data.columns:
        data["day_of_week"] = data["day_of_week"].map(day_map).fillna(0).astype(float)
    if "session" in data.columns:
        data["session"]     = data["session"].map(session_map).fillna(-1).astype(float)

    tf_dummies = pd.get_dummies(
        pd.Series([timeframe] * len(data)), prefix="tf"
    ).astype(float)
    tf_dummies.index = data.index
    data = pd.concat([data, tf_dummies], axis=1)

    X = data.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_columns].fillna(0)

    X_scaled = pd.DataFrame(
        saved_scaler.transform(X),
        columns=feature_columns,
        index=data.index
    )

    X_scaled["timestamp"] = pd.to_datetime(data["timestamp"])
    X_scaled["close"]     = pd.to_numeric(data["close"], errors="coerce")
    X_scaled["timeframe"] = timeframe
    return X_scaled


class MLSignalStrategy(bt.Strategy):
    params = dict(
        confidence=0.52,
        max_confidence=1.0,       # upper cap — skip signals above this (overconfident bucket)
        stake=0.15,
        use_pct_stake=True,
        trail_trigger_pts=1500.0,
        trail_dist_atr=1.0,
        trail_dist_pts=1000.0,
        include_london_ny=True,   # match signal_generator: False for 15min, True for 5min
        min_zone_quality=MIN_ZONE_QUALITY,
        breakeven_trigger_pts=0.0,  # >0: move SL to entry once profit reaches this level
        timeframe="1H",           # used to select 15min-specific BE/trail parameters
    )

    def __init__(self):
        self._wins                  = 0
        self._losses                = 0
        self._trade_count           = 0
        self._entries_submitted     = 0
        self._skipped_margin        = 0
        self._skipped_max_pos       = 0
        self._open_position_count   = 0
        self._trail_activations     = 0
        self._be_lock_count: int    = 0

        # Per-trade state — each dict: {sl, tp, side, entry_price, size, grade,
        #   trail_active, best_price, entry_atr, be_locked, trade_ref}
        self._open_trades: list     = []
        self._any_exit_pending: bool = False

        self._trade_log: list = []   # [{date, side, pnl, entry_price, size, grade}, ...]
        self._current_grade: str = "C"

        # Diagnostic counters — tells you exactly which gate kills most bars
        self._diag = {
            "no_row":       0,  # bar has no feature row (warmup or timestamp mismatch)
            "session":      0,  # bar outside trading session window
            "zone_quality": 0,  # zone quality score below MIN_ZONE_QUALITY
            "confidence":   0,  # model confidence below threshold
            "neutral":      0,  # model predicted neutral/hold label
            "bad_sltp":     0,  # SL/TP geometrically invalid after calculation
            "low_rr":       0,  # R:R ratio below MIN_RR after geometry check
            "htf_filter":   0,  # counter-trend signal rejected by HTF soft gate
            "risk_atr":     0,  # SL distance < 0.5 ATR after zone-quality buffer
        }

        self.model          = getattr(self, "model", None)
        self.features_by_dt = getattr(self, "features_by_dt", None)

        if self.model is None or self.features_by_dt is None:
            raise RuntimeError("Strategy missing injected `model` or `features_by_dt`.")

    def notify_trade(self, trade):
        if trade.isopen:
            self._open_position_count += 1
            # Tag first unregistered entry with this trade.ref
            for t in self._open_trades:
                if "trade_ref" not in t:
                    t["trade_ref"] = trade.ref
                    break
            return
        if not trade.isclosed:
            return

        self._open_position_count = max(0, self._open_position_count - 1)
        self._trade_count += 1

        # Match by trade.ref; fallback FIFO
        matched_idx = None
        for i, t in enumerate(self._open_trades):
            if t.get("trade_ref") == trade.ref:
                matched_idx = i
                break
        if matched_idx is None and self._open_trades:
            matched_idx = 0

        if matched_idx is not None:
            t = self._open_trades.pop(matched_idx)
            side         = t["side"]
            entry_price  = t["entry_price"]
            grade        = t["grade"]
            size         = t["size"]
            with_trend   = t.get("with_trend", True)
            initial_risk = t.get("initial_risk", 0.0)
            prob         = t.get("prob", 0.0)
        else:
            side         = None
            entry_price  = 0.0
            grade        = "C"
            size         = 0.0
            with_trend   = True
            initial_risk = 0.0
            prob         = 0.0

        self._trade_log.append({
            "date":         self.data.datetime.datetime(0),
            "side":         side,
            "pnl":          float(trade.pnlcomm),
            "entry_price":  entry_price,
            "size":         size,
            "grade":        grade,
            "with_trend":   with_trend,
            "initial_risk": initial_risk,
            "prob":         prob,
        })

        # Backtrader netting mode: 2 buys = 1 Trade object at double size.
        # When that single Trade closes, notify_trade fires once but we may
        # have 2 entries in _open_trades. Clear all orphans when flat.
        if self.position.size == 0:
            # Log any remaining orphaned entries with zero PnL
            for orphan in self._open_trades:
                self._trade_log.append({
                    "date":         self.data.datetime.datetime(0),
                    "side":         orphan.get("side"),
                    "pnl":          0.0,
                    "entry_price":  orphan.get("entry_price", 0.0),
                    "size":         orphan.get("size", 0.0),
                    "grade":        orphan.get("grade", "C"),
                    "with_trend":   orphan.get("with_trend", True),
                    "initial_risk": orphan.get("initial_risk", 0.0),
                    "prob":         orphan.get("prob", 0.0),
                })
            self._open_trades.clear()
            self._any_exit_pending = False

        if trade.pnlcomm > 0:
            self._wins += 1
        else:
            self._losses += 1

    def notify_order(self, order):
        if order.status in (order.Canceled, order.Rejected, order.Margin):
            status_name = {
                order.Canceled: "Canceled",
                order.Rejected: "Rejected",
                order.Margin:   "Margin/Insufficient funds",
            }.get(order.status, str(order.status))
            print(f"  [WARN] Order failed ({status_name})")

    def _calc_grade(self, zone_quality: float, winner_proba: float) -> str:
        if (zone_quality >= GRADE_A_MIN_ZONE_QUALITY and
                winner_proba >= GRADE_A_MIN_CONFIDENCE):
            return "A"
        if (zone_quality >= GRADE_B_MIN_ZONE_QUALITY and
                winner_proba >= GRADE_B_MIN_CONFIDENCE):
            return "B"
        return "C"

    def _calc_size(self, price: float, grade: str = "C") -> float:
        if grade == "A":
            return 0.02
        return 0.01

    def _trail_dist_for(self, t: dict) -> float:
        if float(self.p.trail_dist_atr) > 0 and t.get("entry_atr"):
            return float(self.p.trail_dist_atr) * t["entry_atr"]
        return float(self.p.trail_dist_pts)

    def next(self):
        bar_high = float(self.data.high[0])
        bar_low  = float(self.data.low[0])

        # ── Per-trade in-trade management ─────────────────────────────
        if self._open_trades and not self._any_exit_pending:
            for t in self._open_trades:
                side  = t["side"]
                entry = t["entry_price"]

                _ir  = t.get("initial_risk", 0.0)
                _atr = t.get("entry_atr") or 0.0

                # Timeframe-specific parameters
                _is_15m      = (self.p.timeframe == "15min")
                _be_r        = 0.8 if _is_15m else 1.0   # BE trigger: 0.8R on 15min, 1R otherwise
                _trail_r     = 1.5                         # Trail trigger: 1.5R for all timeframes
                _trail_dist  = 0.8 if _is_15m else 1.5    # Trail distance ATR: tighter on 15min

                # Breakeven lock at _be_r × initial risk
                if _ir > 0 and _atr > 0 and not t["be_locked"]:
                    if side == "buy" and (bar_high - entry) >= _be_r * _ir:
                        t["sl"] = entry + 0.1 * _atr
                        t["be_locked"] = True
                        self._be_lock_count += 1
                    elif side == "sell" and (entry - bar_low) >= _be_r * _ir:
                        t["sl"] = entry - 0.1 * _atr
                        t["be_locked"] = True
                        self._be_lock_count += 1
                else:
                    # Fallback: params-based breakeven when no ATR/risk data
                    be_trigger = float(self.p.breakeven_trigger_pts)
                    if be_trigger > 0 and not t["be_locked"]:
                        if side == "buy" and (bar_high - entry) >= be_trigger:
                            t["sl"] = entry
                            t["be_locked"] = True
                            self._be_lock_count += 1
                        elif side == "sell" and (entry - bar_low) >= be_trigger:
                            t["sl"] = entry
                            t["be_locked"] = True
                            self._be_lock_count += 1

                # Trail at 1.5R, distance = _trail_dist × ATR
                if _ir > 0 and _atr > 0:
                    if side == "buy":
                        if not t["trail_active"] and (bar_high - entry) >= _trail_r * _ir:
                            t["trail_active"] = True
                            t["best_price"]   = bar_high
                            self._trail_activations += 1
                        if t["trail_active"]:
                            if bar_high > t["best_price"]:
                                t["best_price"] = bar_high
                            new_sl = t["best_price"] - _trail_dist * _atr
                            if new_sl > t["sl"]:
                                t["sl"] = new_sl
                    elif side == "sell":
                        if not t["trail_active"] and (entry - bar_low) >= _trail_r * _ir:
                            t["trail_active"] = True
                            t["best_price"]   = bar_low
                            self._trail_activations += 1
                        if t["trail_active"]:
                            if bar_low < t["best_price"]:
                                t["best_price"] = bar_low
                            new_sl = t["best_price"] + _trail_dist * _atr
                            if new_sl < t["sl"]:
                                t["sl"] = new_sl
                else:
                    # Fallback: params-based trail when no ATR/risk data
                    trail_dist = self._trail_dist_for(t)
                    trigger    = float(self.p.trail_trigger_pts)
                    if not t["be_locked"]:
                        if side == "buy":
                            profit_pts = bar_high - entry
                            if not t["trail_active"] and profit_pts >= trigger:
                                t["trail_active"] = True
                                t["best_price"]   = bar_high
                                self._trail_activations += 1
                            if t["trail_active"]:
                                if bar_high > t["best_price"]:
                                    t["best_price"] = bar_high
                                new_sl = t["best_price"] - trail_dist
                                if new_sl > t["sl"]:
                                    t["sl"] = new_sl
                        elif side == "sell":
                            profit_pts = entry - bar_low
                            if not t["trail_active"] and profit_pts >= trigger:
                                t["trail_active"] = True
                                t["best_price"]   = bar_low
                                self._trail_activations += 1
                            if t["trail_active"]:
                                if bar_low < t["best_price"]:
                                    t["best_price"] = bar_low
                                new_sl = t["best_price"] + trail_dist
                                if new_sl < t["sl"]:
                                    t["sl"] = new_sl

                sl = t["sl"]
                tp = t["tp"]
                hit = False
                if side == "buy":
                    hit = (bar_low <= sl) or (bar_high >= tp)
                elif side == "sell":
                    hit = (bar_high >= sl) or (bar_low <= tp)
                if hit:
                    self._any_exit_pending = True
                    self.close()  # close all open positions
                    break         # no need to check remaining trades

        # ── New entry gate ─────────────────────────────────────────────
        if self._any_exit_pending or len(self._open_trades) >= MAX_CONCURRENT_POSITIONS:
            if not self._any_exit_pending:
                self._skipped_max_pos += 1
            return

        # ── Gate 1: feature row lookup ─────────────────────────────────
        dt  = self.data.datetime.datetime(0).replace(tzinfo=None, microsecond=0)
        row = self.features_by_dt.get(dt)
        if row is None:
            self._diag["no_row"] += 1
            return

        _raw = row.get("raw")

        X_row        = row["X"]
        zone_quality = row.get("zone_quality", float("nan"))

        # ── Gate 2: zone quality ────────────────────────────────────────
        if not (isinstance(zone_quality, float) and zone_quality >= self.p.min_zone_quality):
            self._diag["zone_quality"] += 1
            return

        # ── Gate 3: model confidence ───────────────────────────────────
        proba   = self.model.predict_proba(X_row)[0]
        classes = getattr(self.model, "classes_", np.array([0, 1]))
        winner_class_idx = int(np.where(classes == 1)[0][0]) \
            if 1 in classes else 1
        winner_proba = float(proba[winner_class_idx])

        if winner_proba < float(self.p.confidence):
            self._diag["confidence"] += 1
            return
        if winner_proba > float(self.p.max_confidence):
            self._diag["confidence"] += 1
            return

        # ── Gate 4: direction from zone ────────────────────────────────
        if _raw is not None:
            try:
                in_demand = float(_raw.get("in_demand_zone", 0))
                in_supply = float(_raw.get("in_supply_zone", 0))
            except Exception:
                in_demand = in_supply = 0.0
        else:
            in_demand = in_supply = 0.0

        if in_demand == 1.0:
            pred_label = "buy"
        elif in_supply == 1.0:
            pred_label = "sell"
        else:
            self._diag["neutral"] += 1
            return

        # ── HTF soft filter: counter-trend trades require higher confidence ─
        htf_4h_bias = float(_raw.get("htf_4h_bias", 0.0)) if _raw is not None else 0.0
        is_counter  = (pred_label == "buy" and htf_4h_bias < 0) or \
                      (pred_label == "sell" and htf_4h_bias > 0)
        if is_counter:
            abs_bias = abs(htf_4h_bias)
            if abs_bias > 0.8 and winner_proba < 0.70:
                self._diag["htf_filter"] += 1
                return
            if 0.3 <= abs_bias <= 0.8 and winner_proba < float(self.p.confidence) + 0.05:
                self._diag["htf_filter"] += 1
                return

        # ── Directional consistency: only add trades in same direction ──
        # Backtrader uses a netting account — a sell against an open buy
        # would close it rather than open a new short position.
        if self._open_trades:
            existing_side = self._open_trades[0]["side"]
            if pred_label != existing_side:
                return

        close_price = float(self.data.close[0])

        grade = self._calc_grade(zone_quality, winner_proba)
        if GRADE_MULTIPLIERS.get(grade, 1) == 0:
            self._diag["grade_skip"] = self._diag.get("grade_skip", 0) + 1
            return
        size      = self._calc_size(close_price, grade)
        required  = close_price * size
        available = self.broker.getcash()
        if required > available * 0.99:
            self._skipped_margin += 1
            return

        lookback_n = 20
        if len(self.data) < lookback_n:
            return

        opens  = np.asarray([float(self.data.open[-i])   for i in range(lookback_n - 1, -1, -1)], dtype=float)
        highs  = np.asarray([float(self.data.high[-i])   for i in range(lookback_n - 1, -1, -1)], dtype=float)
        lows   = np.asarray([float(self.data.low[-i])    for i in range(lookback_n - 1, -1, -1)], dtype=float)
        closes = np.asarray([float(self.data.close[-i])  for i in range(lookback_n - 1, -1, -1)], dtype=float)
        vols   = np.asarray([float(self.data.volume[-i]) for i in range(lookback_n - 1, -1, -1)], dtype=float)

        lookback_df = pd.DataFrame(
            np.column_stack([opens, highs, lows, closes, vols]),
            columns=pd.Index(["open", "high", "low", "close", "volume"], dtype="object"),
        )

        candle_size = highs - lows
        body_size   = np.abs(closes - opens)
        wick_upper  = highs - np.maximum(closes, opens)
        wick_lower  = np.minimum(closes, opens) - lows

        lookback_df["candle_size"] = candle_size
        lookback_df["body_size"]   = body_size
        lookback_df["wick_upper"]  = wick_upper
        lookback_df["wick_lower"]  = wick_lower

        raw_feat_series = row.get("raw")

        sl = calculate_stop_loss(
            close_price, pred_label, lookback_df,
            feature_row=raw_feat_series,
        )
        tp = calculate_take_profit(
            close_price, pred_label, lookback_df,
            feature_row=raw_feat_series,
        )

        # ── Gate 5: SL/TP geometry sanity ─────────────────────────────
        if pred_label == "buy":
            if sl is None: sl = close_price * 0.997
            if tp is None: tp = close_price * 1.006
            if sl >= close_price or tp <= close_price:
                self._diag["bad_sltp"] += 1
                return
        else:
            if sl is None: sl = close_price * 1.003
            if tp is None: tp = close_price * 0.994
            if sl <= close_price or tp >= close_price:
                self._diag["bad_sltp"] += 1
                return

        # ── SL buffer: enforce zone-quality-scaled minimum SL distance ──
        atr_14 = float(_raw.get("atr_14", 0.0)) if _raw is not None else 0.0
        if atr_14 > 0:
            sl_buf_atr = 0.3 if zone_quality >= 3.5 else (0.5 if zone_quality >= 2.0 else 0.7)
            min_sl_dist = sl_buf_atr * atr_14
            if pred_label == "buy":
                sl = min(float(sl), close_price - min_sl_dist)
            else:
                sl = max(float(sl), close_price + min_sl_dist)
            if abs(close_price - float(sl)) / atr_14 < 0.5:
                self._diag["risk_atr"] += 1
                return

        # ── Gate 6: minimum risk-reward ratio ─────────────────────────
        sl_dist = abs(close_price - float(sl))
        tp_dist = abs(float(tp) - close_price)
        if sl_dist == 0 or (tp_dist / sl_dist) < MIN_RR:
            self._diag["low_rr"] += 1
            return

        self._current_grade = grade

        if pred_label == "buy":
            self.buy(size=size)
        else:
            self.sell(size=size)

        self._entries_submitted += 1

        atr_val = atr_14 if atr_14 > 0 else None
        if atr_val is None:
            try:
                atr_val = float(lookback_df["candle_size"].rolling(14).mean().iloc[-1])
            except Exception:
                pass

        self._open_trades.append({
            "sl":           float(sl),
            "tp":           float(tp),
            "side":         pred_label,
            "entry_price":  float(close_price),
            "size":         float(size),
            "grade":        grade,
            "trail_active": False,
            "best_price":   None,
            "entry_atr":    atr_val if atr_val and atr_val > 0 else None,
            "be_locked":    False,
            "exit_pending": False,
            "initial_risk": abs(close_price - float(sl)),
            "with_trend":   not is_counter,
            "prob":         winner_proba,
        })

    @property
    def wins(self)                  -> int:  return self._wins
    @property
    def losses(self)                -> int:  return self._losses
    @property
    def trades(self)                -> int:  return self._trade_count
    @property
    def entries_submitted(self)     -> int:  return self._entries_submitted
    @property
    def skipped_no_margin(self)     -> int:  return self._skipped_margin
    @property
    def skipped_max_positions(self) -> int:  return self._skipped_max_pos
    @property
    def trail_activations(self)     -> int:  return self._trail_activations
    @property
    def be_lock_count(self)         -> int:  return self._be_lock_count
    @property
    def diag(self)                  -> dict: return self._diag
    @property
    def trade_log(self)             -> list: return self._trade_log


# ======================================================================
def run_backtest(
    timeframe:   str,
    start_date:  Optional[str],
    end_date:    Optional[str],
    cash:        float,
    stake:       float,
    use_pct_stake: bool,
    confidence:  float,
    commission:  float,
    trail_trigger_pts:  float = 10.0,
    trail_dist_atr:     float = 1.0,
    trail_dist_pts:     float = 1000.0,
    include_london_ny:  bool  = True,
    model_dir:   str = MODEL_DIR,
    min_zone_quality: float = MIN_ZONE_QUALITY,
    breakeven_trigger_pts: float = 0.0,
    max_confidence: float = 1.0,
) -> BacktestResult:

    db = get_connection()
    if not db.connect():
        raise ConnectionError("Failed to connect to database")

    query  = "SELECT * FROM xauusd_ohlcv WHERE timeframe = %s"
    params = [timeframe]
    if start_date:
        query  += " AND date >= %s"; params.append(start_date)
    if end_date:
        query  += " AND date <= %s"; params.append(end_date)
    query += " ORDER BY timestamp ASC"

    df = db.fetch_dataframe(query, tuple(params))

    # Load H1 and H4 for HTF trend context (same date range)
    h1_df = h4_df = None
    for htf, attr in [("1H", "h1_df"), ("4H", "h4_df")]:
        htf_query  = "SELECT * FROM xauusd_ohlcv WHERE timeframe = %s"
        htf_params = [htf]
        if start_date:
            htf_query  += " AND date >= %s"; htf_params.append(start_date)
        if end_date:
            htf_query  += " AND date <= %s"; htf_params.append(end_date)
        htf_query += " ORDER BY timestamp ASC"
        htf_df = db.fetch_dataframe(htf_query, tuple(htf_params))
        if not htf_df.empty:
            htf_df["timestamp"] = pd.to_datetime(htf_df["timestamp"])
            if attr == "h1_df":
                h1_df = htf_df
            else:
                h4_df = htf_df

    db.disconnect()

    if df.empty:
        raise ValueError(f"No rows returned for timeframe={timeframe}.")

    df_bt = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    df_bt = df_bt.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    model, metadata_bundle = _load_model_bundle(model_dir=model_dir)

    # Read per-TF build params saved at training time so backtest matches training exactly
    _tf_bp = metadata_bundle.get("tf_build_params", {}).get(timeframe, {})
    if _tf_bp:
        saved_impulse_atr    = float(_tf_bp.get("impulse_atr_multiplier", 0.5))
        saved_include_london = bool(_tf_bp.get("include_london_ny", include_london_ny))
        if saved_include_london != include_london_ny:
            print(f"  [meta] include_london_ny overridden by saved metadata: {saved_include_london}")
            include_london_ny = saved_include_london
    else:
        saved_impulse_atr = 0.5

    # Auto-detect: 15min model was trained with H16 (London/NY overlap) excluded (WR=24%)
    if timeframe == "15min" and include_london_ny:
        include_london_ny = False
        print("  [auto] 15min: forcing include_london_ny=False (H16 excluded at training)")

    # Use optimal threshold saved during training if confidence not overridden
    saved_threshold = float(metadata_bundle.get("optimal_threshold", confidence))
    if confidence == 0.52:  # default — use saved threshold
        confidence = saved_threshold
        print(f"  Using saved optimal threshold: {confidence:.3f}")

    X_scaled = _build_feature_matrix_for_timeframe(df, timeframe, metadata_bundle,
                                                    include_london_ny=include_london_ny,
                                                    impulse_atr_multiplier=saved_impulse_atr,
                                                    h1_df=h1_df, h4_df=h4_df)

    feature_cols = [c for c in X_scaled.columns if c not in {"timestamp", "close", "timeframe"}]

    # Build scaled feature lookup
    features_by_dt: Dict[Any, Any] = {}
    for _, r in X_scaled.iterrows():
        dt    = r["timestamp"].to_pydatetime().replace(tzinfo=None, microsecond=0)
        X_row = pd.DataFrame([r[feature_cols].to_numpy()], columns=feature_cols)
        features_by_dt[dt] = {
            "X":           X_row,
            "zone_quality": float("nan"),
            "raw":         None,
        }

    # Overlay raw (unscaled) zone values and zone quality
    try:
        from data.feature_engineer import build_features as _bf
        _raw_feat = _bf(df.copy(), h1_df=h1_df, h4_df=h4_df,
                        include_london_ny=include_london_ny,
                        impulse_atr_multiplier=saved_impulse_atr)

        if "timestamp" in _raw_feat.columns:
            _raw_feat["timestamp"] = pd.to_datetime(_raw_feat["timestamp"])

            available_raw_cols = ["timestamp"] + [
                c for c in RAW_ZONE_COLS + ["active_zone_quality"]
                if c in _raw_feat.columns
            ]

            for _, rr in _raw_feat[available_raw_cols].iterrows():
                _dt = rr["timestamp"].to_pydatetime().replace(tzinfo=None, microsecond=0)
                if _dt in features_by_dt:
                    if "active_zone_quality" in rr:
                        features_by_dt[_dt]["zone_quality"] = float(rr["active_zone_quality"])
                    features_by_dt[_dt]["raw"] = rr

    except Exception as _e:
        print(f"  [WARN] Could not attach raw zone features: {_e}")

    print(f"  Price bars:    {len(df_bt):,}")
    print(f"  Feature rows:  {len(features_by_dt):,}")

    raw_count = sum(1 for v in features_by_dt.values() if v.get("raw") is not None)
    print(f"  Raw zone rows: {raw_count:,} ({raw_count/max(len(features_by_dt),1)*100:.1f}% coverage)")

    # Zone quality distribution — tune MIN_ZONE_QUALITY intelligently
    zq_values = [
        v["zone_quality"] for v in features_by_dt.values()
        if isinstance(v.get("zone_quality"), float) and not np.isnan(v["zone_quality"])
    ]
    if zq_values:
        zq = np.array(zq_values)
        above_zero = (zq > 0).sum()
        print(f"  Zone quality  | mean={zq.mean():.2f} min={zq.min():.2f} max={zq.max():.2f}")
        print(f"  Bars in zone  | >0: {above_zero:,} | >=2.0: {(zq>=2.0).sum():,} | "
              f">=3.0: {(zq>=3.0).sum():,} | >=3.5: {(zq>=3.5).sum():,}")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=float(commission))
    if not use_pct_stake:
        cerebro.addsizer(bt.sizers.FixedSize, stake=float(stake))

    cerebro.adddata(bt.feeds.PandasData(dataname=df_bt))
    cerebro.addanalyzer(bt.analyzers.DrawDown,   _name="dd")
    cerebro.addanalyzer(bt.analyzers.TimeReturn,
                        timeframe=bt.TimeFrame.Months, _name="monthly")

    MLSignalStrategy.model          = model
    MLSignalStrategy.features_by_dt = features_by_dt

    cerebro.addstrategy(
        MLSignalStrategy,
        confidence=float(confidence),
        max_confidence=float(max_confidence),
        stake=float(stake),
        use_pct_stake=use_pct_stake,
        trail_trigger_pts=float(trail_trigger_pts),
        trail_dist_atr=float(trail_dist_atr),
        trail_dist_pts=float(trail_dist_pts),
        include_london_ny=bool(include_london_ny),
        min_zone_quality=float(min_zone_quality),
        breakeven_trigger_pts=float(breakeven_trigger_pts),
        timeframe=timeframe,
    )

    start_value = cerebro.broker.getvalue()
    results     = cerebro.run()
    strat_inst  = results[0]
    end_value   = cerebro.broker.getvalue()

    dd     = strat_inst.analyzers.dd.get_analysis()
    max_dd = float(dd.get("max", {}).get("drawdown", 0.0))

    trades      = int(getattr(strat_inst, "trades", 0))
    wins        = int(getattr(strat_inst, "wins", 0))
    entries     = int(getattr(strat_inst, "entries_submitted", 0))
    skipped     = int(getattr(strat_inst, "skipped_no_margin", 0))
    skp_pos     = int(getattr(strat_inst, "skipped_max_positions", 0))
    trail_acts  = int(getattr(strat_inst, "trail_activations", 0))
    be_locks    = int(getattr(strat_inst, "be_lock_count", 0))
    winrate     = (wins / trades * 100.0) if trades > 0 else 0.0
    diag        = getattr(strat_inst, "diag", {})
    trade_log   = getattr(strat_inst, "trade_log", [])

    # Extended P&L stats from trade log
    win_pnls  = [t["pnl"] for t in trade_log if t["pnl"] > 0]
    loss_pnls = [t["pnl"] for t in trade_log if t["pnl"] <= 0]
    buy_log   = [t for t in trade_log if t.get("side") == "buy"]
    sell_log  = [t for t in trade_log if t.get("side") == "sell"]

    grade_a_log = [t for t in trade_log if t.get("grade") == "A"]
    grade_b_log = [t for t in trade_log if t.get("grade") == "B"]
    grade_c_log = [t for t in trade_log if t.get("grade") == "C"]

    gross_profit = sum(win_pnls)
    gross_loss   = sum(loss_pnls)
    avg_win      = float(np.mean(win_pnls))  if win_pnls  else 0.0
    avg_loss     = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    largest_win  = float(max(win_pnls))      if win_pnls  else 0.0
    largest_loss = float(min(loss_pnls))     if loss_pnls else 0.0
    buy_wins_n   = sum(1 for t in buy_log  if t["pnl"] > 0)
    sell_wins_n  = sum(1 for t in sell_log if t["pnl"] > 0)

    # Monthly P&L from TimeReturn analyzer (returns → dollar P&L)
    monthly_ret  = strat_inst.analyzers.monthly.get_analysis()
    running_val  = float(start_value)
    monthly_pnl_list = []
    for dt_key in sorted(monthly_ret.keys()):
        ret = float(monthly_ret[dt_key])
        month_pnl = running_val * ret
        running_val *= (1 + ret)
        monthly_pnl_list.append((dt_key.year, dt_key.month, round(month_pnl, 2)))

    return BacktestResult(
        final_value=float(end_value),
        pnl=float(end_value - start_value),
        max_drawdown_pct=max_dd,
        winrate_pct=float(winrate),
        total_trades=trades,
        entries_submitted=entries,
        skipped_no_margin=skipped,
        skipped_max_positions=skp_pos,
        trail_activations=trail_acts,
        filtered_no_row=diag.get("no_row", 0),
        filtered_session=diag.get("session", 0),
        filtered_zone_quality=diag.get("zone_quality", 0),
        filtered_confidence=diag.get("confidence", 0),
        filtered_neutral=diag.get("neutral", 0),
        filtered_bad_sltp=diag.get("bad_sltp", 0),
        filtered_low_rr=diag.get("low_rr", 0),
        filtered_htf_filter=diag.get("htf_filter", 0),
        filtered_risk_atr=diag.get("risk_atr", 0),
        start_cash=float(start_value),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        largest_win=round(largest_win, 2),
        largest_loss=round(largest_loss, 2),
        buy_trades=len(buy_log),
        buy_wins=buy_wins_n,
        sell_trades=len(sell_log),
        sell_wins=sell_wins_n,
        monthly_pnl=tuple(monthly_pnl_list),
        grade_a_trades=len(grade_a_log),
        grade_a_wins=sum(1 for t in grade_a_log if t["pnl"] > 0),
        grade_b_trades=len(grade_b_log),
        grade_b_wins=sum(1 for t in grade_b_log if t["pnl"] > 0),
        grade_c_trades=len(grade_c_log),
        grade_c_wins=sum(1 for t in grade_c_log if t["pnl"] > 0),
        be_lock_count=be_locks,
        trade_log=tuple(trade_log),
    )


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe",          default="1min")
    parser.add_argument("--start-date",         default=None)
    parser.add_argument("--end-date",           default=None)
    parser.add_argument("--cash",               type=float, default=10000.0)
    parser.add_argument("--stake",              type=float, default=0.15)
    parser.add_argument("--pct-stake",          dest="use_pct_stake", action="store_true", default=True)
    parser.add_argument("--no-pct-stake",       dest="use_pct_stake", action="store_false")
    parser.add_argument("--confidence",         type=float, default=0.52)
    parser.add_argument("--max-confidence",     type=float, default=1.0,
                        help="Skip signals above this probability (overconfident cap)")
    parser.add_argument("--commission",         type=float, default=0.0)
    parser.add_argument("--trail-trigger-pts",  type=float, default=10.0)
    parser.add_argument("--trail-dist-atr",     type=float, default=1.0)
    parser.add_argument("--trail-dist-pts",     type=float, default=1000.0)
    parser.add_argument("--model-dir",          default=MODEL_DIR)
    parser.add_argument("--no-london-ny",        dest="include_london_ny",
                        action="store_false", default=True,
                        help="Exclude London/NY overlap (H16). Use for 15min.")
    parser.add_argument("--min-zone-quality",    type=float, default=MIN_ZONE_QUALITY,
                        help=f"Minimum zone quality score to trade (default={MIN_ZONE_QUALITY})")
    parser.add_argument("--breakeven-trigger-pts", type=float, default=0.0,
                        help="Move SL to entry once profit reaches this level (0=disabled)")
    args = parser.parse_args()

    res = run_backtest(
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        cash=args.cash,
        stake=args.stake,
        use_pct_stake=args.use_pct_stake,
        confidence=args.confidence,
        commission=args.commission,
        trail_trigger_pts=args.trail_trigger_pts,
        trail_dist_atr=args.trail_dist_atr,
        trail_dist_pts=args.trail_dist_pts,
        include_london_ny=args.include_london_ny,
        model_dir=args.model_dir,
        min_zone_quality=args.min_zone_quality,
        breakeven_trigger_pts=args.breakeven_trigger_pts,
        max_confidence=args.max_confidence,
    )

    W  = 62
    print("\n" + "=" * W)
    print(f"  BACKTEST REPORT - {args.timeframe}  |  "
          f"{args.start_date or 'ALL'} to {args.end_date or 'ALL'}")
    print("=" * W)

    # ── Capital summary ───────────────────────────────────────────────
    roi = res.pnl / res.start_cash * 100
    pf  = res.gross_profit / abs(res.gross_loss) if res.gross_loss != 0 else float("inf")
    print(f"\n  CAPITAL")
    print(f"    Starting capital   : ${res.start_cash:>12,.2f}")
    print(f"    Final value        : ${res.final_value:>12,.2f}")
    print(f"    Net profit         : ${res.pnl:>+12,.2f}   ({roi:+.2f}%)")
    print(f"    Gross profit       : ${res.gross_profit:>12,.2f}")
    print(f"    Gross loss         : ${res.gross_loss:>12,.2f}")
    print(f"    Profit factor      :  {pf:>11.3f}   (gross profit / gross loss)")

    # ── Risk ──────────────────────────────────────────────────────────
    max_dd_dollar = res.start_cash * res.max_drawdown_pct / 100
    print(f"\n  RISK")
    print(f"    Max drawdown       : ${max_dd_dollar:>12,.2f}   ({res.max_drawdown_pct:.2f}%)")

    # ── Trade statistics ──────────────────────────────────────────────
    losses = res.total_trades - int(res.winrate_pct * res.total_trades / 100 + 0.5)
    print(f"\n  TRADES")
    print(f"    Total trades       : {res.total_trades:>8,}")
    print(f"    Winners            : {int(res.winrate_pct*res.total_trades/100+0.5):>8,}   ({res.winrate_pct:.1f}%)")
    print(f"    Losers             : {res.total_trades - int(res.winrate_pct*res.total_trades/100+0.5):>8,}")
    print(f"    Avg win            : ${res.avg_win:>+11,.2f}")
    print(f"    Avg loss           : ${res.avg_loss:>+11,.2f}")
    print(f"    Largest win        : ${res.largest_win:>+11,.2f}")
    print(f"    Largest loss       : ${res.largest_loss:>+11,.2f}")
    avg_trade = res.pnl / res.total_trades if res.total_trades else 0
    print(f"    Avg per trade      : ${avg_trade:>+11,.4f}")
    print(f"    Trail activations  : {res.trail_activations:>8,}   "
          f"(trigger={args.trail_trigger_pts:.0f}pts, dist={args.trail_dist_atr:.1f}ATR)")
    print(f"    Breakeven locks    : {res.be_lock_count:>8,}   "
          f"(1R ATR-based + params fallback)")

    # ── By direction ──────────────────────────────────────────────────
    buy_wr  = res.buy_wins  / res.buy_trades  * 100 if res.buy_trades  else 0
    sell_wr = res.sell_wins / res.sell_trades * 100 if res.sell_trades else 0
    print(f"\n  BY DIRECTION")
    print(f"    {'':6}  {'Trades':>7}  {'Wins':>6}  {'WR':>6}")
    print(f"    {'Buy':<6}  {res.buy_trades:>7,}  {res.buy_wins:>6,}  {buy_wr:>5.1f}%")
    print(f"    {'Sell':<6}  {res.sell_trades:>7,}  {res.sell_wins:>6,}  {sell_wr:>5.1f}%")

    # ── By grade ──────────────────────────────────────────────────────
    a_wr = res.grade_a_wins / res.grade_a_trades * 100 if res.grade_a_trades else 0
    b_wr = res.grade_b_wins / res.grade_b_trades * 100 if res.grade_b_trades else 0
    c_wr = res.grade_c_wins / res.grade_c_trades * 100 if res.grade_c_trades else 0
    print(f"\n  BY SIGNAL GRADE  (A=zone>=3.5+conf>=0.42 | B=zone>=3.0+conf>=0.40 | C=rest)")
    print(f"    {'Grade':<6}  {'Lots':>5}  {'Trades':>7}  {'Wins':>6}  {'WR':>6}")
    def _fmt_mult(g):
        m = GRADE_MULTIPLIERS.get(g, 0)
        return "SKIP" if m == 0 else f"{m}x"
    print(f"    {'A':<6}  {_fmt_mult('A'):>5}  {res.grade_a_trades:>7,}  {res.grade_a_wins:>6,}  {a_wr:>5.1f}%")
    print(f"    {'B':<6}  {_fmt_mult('B'):>5}  {res.grade_b_trades:>7,}  {res.grade_b_wins:>6,}  {b_wr:>5.1f}%")
    print(f"    {'C':<6}  {_fmt_mult('C'):>5}  {res.grade_c_trades:>7,}  {res.grade_c_wins:>6,}  {c_wr:>5.1f}%")

    # ── Monthly P&L ───────────────────────────────────────────────────
    if res.monthly_pnl:
        print(f"\n  MONTHLY P&L")
        print(f"    {'Month':<10}  {'P&L':>10}  {'Bar'}")
        print(f"    {'-'*10}  {'-'*10}")
        MONTH_NAMES = ["","Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        year_totals: dict = {}
        for year, month, mpnl in res.monthly_pnl:
            bar = "#" * int(abs(mpnl) / max(abs(p) for _,_,p in res.monthly_pnl) * 20 + 0.5) if mpnl else ""
            sign = "+" if mpnl >= 0 else ""
            print(f"    {MONTH_NAMES[month]} {year}    {sign}${mpnl:>8,.2f}  {bar}")
            year_totals[year] = year_totals.get(year, 0) + mpnl
        print(f"    {'-'*10}  {'-'*10}")
        for year, ytotal in sorted(year_totals.items()):
            sign = "+" if ytotal >= 0 else ""
            print(f"    {year} TOTAL   {sign}${ytotal:>8,.2f}")

    # ── Segment comparison table ──────────────────────────────────────
    def _seg_stats(trades):
        n = len(trades)
        if n == 0:
            return 0, 0, 0.0, 0.0, 0.0
        wins      = sum(1 for t in trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in trades)
        rrs = [
            t["pnl"] / (t["initial_risk"] * t["size"])
            for t in trades
            if t.get("initial_risk", 0.0) > 0 and t.get("size", 0.0) > 0
        ]
        avg_rr = float(np.mean(rrs)) if rrs else 0.0
        return n, wins, wins / n * 100, avg_rr, total_pnl

    tl = list(res.trade_log)
    seg_a  = [t for t in tl if t.get("grade") == "A"]
    seg_b  = [t for t in tl if t.get("grade") == "B"]
    seg_c  = [t for t in tl if t.get("grade") == "C"]
    seg_wt = [t for t in tl if t.get("with_trend", True)]
    seg_ct = [t for t in tl if not t.get("with_trend", True)]

    print(f"\n  SEGMENT COMPARISON")
    print(f"  {'Segment':<16} {'Trades':>7}  {'WR%':>6}  {'Avg RR':>7}  {'Total PnL':>11}")
    print(f"  {'-'*16} {'-'*7}  {'-'*6}  {'-'*7}  {'-'*11}")
    for label, seg in [("Grade A", seg_a), ("Grade B", seg_b), ("Grade C", seg_c),
                       ("-"*16, None),
                       ("With-trend", seg_wt), ("Counter-trend", seg_ct)]:
        if seg is None:
            print(f"  {label}")
            continue
        n, w, wr, avg_rr, pnl = _seg_stats(seg)
        if n == 0:
            print(f"  {label:<16} {'0':>7}  {'  --':>6}  {'   --':>7}  {'         --':>11}")
        else:
            sign = "+" if pnl >= 0 else ""
            print(f"  {label:<16} {n:>7,}  {wr:>5.1f}%  {avg_rr:>7.2f}  {sign}${pnl:>9,.2f}")

    # ── Confidence bucket breakdown ───────────────────────────────────
    def _conf_bucket(p):
        if p >= 0.65: return "0.65+"
        if p >= 0.55: return "0.55-0.65"
        if p >= 0.45: return "0.45-0.55"
        return "0.35-0.45"

    buck_labels = ["0.35-0.45", "0.45-0.55", "0.55-0.65", "0.65+"]
    buck = {b: [] for b in buck_labels}
    for t in tl:
        p = t.get("prob", 0.0)
        if p and p > 0:
            buck[_conf_bucket(p)].append(t)

    print(f"\n  CONFIDENCE BUCKET BREAKDOWN")
    print(f"  {'Bucket':<12} {'Trades':>7}  {'WR%':>6}  {'Avg RR':>7}  {'Total PnL':>11}")
    print(f"  {'-'*12} {'-'*7}  {'-'*6}  {'-'*7}  {'-'*11}")
    for label in buck_labels:
        n, w, wr, avg_rr, pnl = _seg_stats(buck[label])
        if n == 0:
            print(f"  {label:<12} {'0':>7}  {'  --':>6}  {'   --':>7}  {'         --':>11}")
        else:
            sign = "+" if pnl >= 0 else ""
            print(f"  {label:<12} {n:>7,}  {wr:>5.1f}%  {avg_rr:>7.2f}  {sign}${pnl:>9,.2f}")

    # ── Filter breakdown ──────────────────────────────────────────────
    print(f"\n  FILTER BREAKDOWN (bars rejected per gate)")

    gate_counts = {
        "no_feature_row":    res.filtered_no_row,
        "session (inactive)": res.filtered_session,
        f"zone_quality<{args.min_zone_quality}": res.filtered_zone_quality,
        "low_confidence":    res.filtered_confidence,
        "neutral_prediction": res.filtered_neutral,
        "bad_sltp_geometry": res.filtered_bad_sltp,
        f"low_rr<{MIN_RR:.1f}":     res.filtered_low_rr,
        "htf_soft_filter":   res.filtered_htf_filter,
        "risk_atr<0.5":      res.filtered_risk_atr,
    }
    bars_total = (
        res.filtered_no_row + res.filtered_session + res.filtered_zone_quality +
        res.filtered_confidence + res.filtered_neutral + res.filtered_bad_sltp +
        res.filtered_low_rr + res.filtered_htf_filter + res.filtered_risk_atr +
        res.entries_submitted
    )
    print(f"  {'Gate':<26} {'Count':>8}  {'% of total':>10}")
    print(f"  {'-'*26} {'-'*8}  {'-'*10}")
    for gate_name, count in gate_counts.items():
        pct = count / max(bars_total, 1) * 100
        print(f"  {gate_name:<26} {count:>8,}  {pct:>9.1f}%")
    print(f"  {'entries_submitted':<26} {res.entries_submitted:>8,}  "
          f"{res.entries_submitted / max(bars_total, 1) * 100:>9.1f}%")
    print(f"  {'TOTAL':<26} {bars_total:>8,}")

    GATE_WARN_THRESHOLD = 60.0
    GATE_SUGGESTIONS = {
        "no_feature_row":     "feature matrix timestamp alignment — check build_features() output",
        f"zone_quality<{args.min_zone_quality}": f"lower --min-zone-quality or MIN_ZONE_QUALITY in config/pipeline_config.py",
        "low_confidence":     "lower DEFAULT_CONFIDENCE_THRESHOLD or retrain model",
        "neutral_prediction": "more bars in both zones — check detect_zones() lookback",
        "bad_sltp_geometry":  "SL/TP calculation — check calculate_stop_loss/take_profit()",
        f"low_rr<{MIN_RR:.1f}":      f"zone spacing too tight — raise MIN_RR or lower MIN_ZONE_QUALITY",
        "htf_soft_filter":    "most signals are counter-trend — check HTF bias alignment or lower confidence requirement",
        "risk_atr<0.5":       "SL too close relative to ATR — zones may be too tight or ATR too large for this timeframe",
    }
    print()
    for gate_name, count in gate_counts.items():
        pct = count / max(bars_total, 1) * 100
        if pct > GATE_WARN_THRESHOLD and gate_name in GATE_SUGGESTIONS:
            print(f"  WARNING: '{gate_name}' rejected {pct:.1f}% of bars — "
                  f"suggested fix: {GATE_SUGGESTIONS[gate_name]}")


if __name__ == "__main__":
    main()