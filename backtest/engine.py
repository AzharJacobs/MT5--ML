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
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import backtrader as bt

from data.loader import get_connection
from data.pipeline import DataPreparator
from models.trainer import ModelTrainer, MODEL_DIR
from strategy.base_strategy import calculate_stop_loss, calculate_take_profit
from config.pipeline_config import MIN_ZONE_QUALITY, HTF_EXTREME_THRESHOLD

MAX_CONCURRENT_POSITIONS = 2

# Raw zone columns we need at execution time (unscaled, real price levels)
RAW_ZONE_COLS = [
    "demand_zone_bottom", "demand_zone_top",
    "supply_zone_bottom", "supply_zone_top",
    "htf_demand_zone_top", "htf_demand_zone_bottom",
    "htf_supply_zone_top", "htf_supply_zone_bottom",
    "atr_14",
    "htf_4h_bias",   # needed for hard HTF trend gate (unscaled: 1.0 / -1.0)
    "htf_1h_bias",   # needed for hard HTF trend gate
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


def _load_model_bundle(model_dir: str = MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
    model, metadata = ModelTrainer.load_model(model_dir=model_dir)
    return model, metadata


def _build_feature_matrix_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    metadata_bundle: Dict[str, Any]
) -> pd.DataFrame:
    from data.feature_engineer import build_features

    saved_scaler    = metadata_bundle.get("scaler")
    feature_columns = metadata_bundle.get("feature_columns") or []

    if saved_scaler is None:
        raise ValueError("Saved scaler not found in model metadata. Retrain first.")
    if not feature_columns:
        raise ValueError("Saved feature_columns not found in model metadata. Retrain first.")

    data = df.copy()
    data = build_features(data)

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
        stake=0.15,
        use_pct_stake=True,
        trail_trigger_pts=1500.0,
        trail_dist_atr=1.0,
        trail_dist_pts=1000.0,
        include_london_ny=True,   # match signal_generator: False for 15min, True for 5min
        min_zone_quality=MIN_ZONE_QUALITY,
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
        self._in_trade              = False
        self._exit_pending          = False

        self._sl: Optional[float]          = None
        self._tp: Optional[float]          = None
        self._side: Optional[str]          = None
        self._entry_price: Optional[float] = None
        self._trail_active: bool           = False
        self._best_price: Optional[float]  = None
        self._entry_atr: Optional[float]   = None

        # Diagnostic counters — tells you exactly which gate kills most bars
        self._diag = {
            "no_row":       0,  # bar has no feature row (warmup or timestamp mismatch)
            "session":      0,  # bar outside trading session window
            "zone_quality": 0,  # zone quality score below MIN_ZONE_QUALITY
            "confidence":   0,  # model confidence below threshold
            "neutral":      0,  # model predicted neutral/hold label
            "bad_sltp":     0,  # SL/TP geometrically invalid after calculation
        }

        self.model          = getattr(self, "model", None)
        self.features_by_dt = getattr(self, "features_by_dt", None)

        if self.model is None or self.features_by_dt is None:
            raise RuntimeError("Strategy missing injected `model` or `features_by_dt`.")

    def notify_trade(self, trade):
        if trade.isopen:
            self._open_position_count += 1
            return
        if not trade.isclosed:
            return
        self._open_position_count = max(0, self._open_position_count - 1)
        self._trade_count   += 1
        self._in_trade       = False
        self._exit_pending   = False
        self._sl             = None
        self._tp             = None
        self._side           = None
        self._entry_price    = None
        self._trail_active   = False
        self._best_price     = None
        self._entry_atr      = None
        if trade.pnlcomm > 0:
            self._wins += 1
        else:
            self._losses += 1

    def notify_order(self, order):
        if order.status in (order.Canceled, order.Rejected, order.Margin):
            if self.position.size == 0:
                self._in_trade     = False
                self._exit_pending = False
                self._sl           = None
                self._tp           = None
                self._side         = None
                status_name = {
                    order.Canceled: "Canceled",
                    order.Rejected: "Rejected",
                    order.Margin:   "Margin/Insufficient funds",
                }.get(order.status, str(order.status))
                print(f"  [WARN] Order failed ({status_name}) — resetting trade flag")
        elif order.status == order.Completed:
            if self.position.size == 0:
                self._exit_pending = False

    def _calc_size(self, price: float) -> float:
        if self.p.use_pct_stake:
            cash      = self.broker.getcash()
            trade_val = cash * float(self.p.stake)
            size      = trade_val / price
        else:
            size = float(self.p.stake)
        return max(size, 1e-8)

    def _trail_distance(self) -> float:
        if float(self.p.trail_dist_atr) > 0 and self._entry_atr:
            return float(self.p.trail_dist_atr) * self._entry_atr
        return float(self.p.trail_dist_pts)

    def next(self):
        # ── In-trade management ────────────────────────────────────────
        if self.position.size != 0:
            if self._exit_pending:
                return

            side     = self._side
            sl       = self._sl
            tp       = self._tp
            entry    = self._entry_price
            bar_high = float(self.data.high[0])
            bar_low  = float(self.data.low[0])

            if side and sl is not None and tp is not None and entry is not None:

                trail_dist = self._trail_distance()
                trigger    = float(self.p.trail_trigger_pts)

                if side == "buy":
                    profit_pts = bar_high - entry
                    if not self._trail_active and profit_pts >= trigger:
                        self._trail_active      = True
                        self._best_price        = bar_high
                        self._trail_activations += 1
                    if self._trail_active:
                        if bar_high > self._best_price:
                            self._best_price = bar_high
                        new_sl = self._best_price - trail_dist
                        if new_sl > self._sl:
                            self._sl = new_sl
                        sl = self._sl

                elif side == "sell":
                    profit_pts = entry - bar_low
                    if not self._trail_active and profit_pts >= trigger:
                        self._trail_active      = True
                        self._best_price        = bar_low
                        self._trail_activations += 1
                    if self._trail_active:
                        if bar_low < self._best_price:
                            self._best_price = bar_low
                        new_sl = self._best_price + trail_dist
                        if new_sl < self._sl:
                            self._sl = new_sl
                        sl = self._sl

                hit = False
                if side == "buy":
                    hit = (bar_low <= sl) or (bar_high >= tp)
                elif side == "sell":
                    hit = (bar_high >= sl) or (bar_low <= tp)
                if hit:
                    self._exit_pending = True
                    self.close()
            return

        if self._in_trade:
            return

        if self._open_position_count >= MAX_CONCURRENT_POSITIONS:
            self._skipped_max_pos += 1
            return

        # ── Gate 1: feature row lookup ─────────────────────────────────
        dt  = self.data.datetime.datetime(0).replace(tzinfo=None, microsecond=0)
        row = self.features_by_dt.get(dt)
        if row is None:
            self._diag["no_row"] += 1
            return

        # Resolve raw early — needed for direction gate and HTF soft filter.
        _raw = row.get("raw")

        # Session gate removed: in_session is now a model feature so the model
        # learns session importance itself. Hard-blocking here was preventing
        # valid out-of-session setups from being evaluated.

        X_row        = row["X"]
        zone_quality = row.get("zone_quality", float("nan"))

        # ── Gate 2: zone quality ────────────────────────────────────────
        if not (isinstance(zone_quality, float) and zone_quality >= self.p.min_zone_quality):
            self._diag["zone_quality"] += 1
            return

        # ── Gate 3: model confidence ───────────────────────────────────
        # Model is binary: class 0 = loser, class 1 = winner.
        # We want the probability of being a WINNER (class 1), not max(proba).
        # Bug was: conf = np.max(proba) which is always ≥ 0.50 in a binary
        # classifier — the confidence gate never fired.
        proba   = self.model.predict_proba(X_row)[0]
        classes = getattr(self.model, "classes_", np.array([0, 1]))
        winner_class_idx = int(np.where(classes == 1)[0][0]) \
            if 1 in classes else 1
        winner_proba = float(proba[winner_class_idx])

        if winner_proba < float(self.p.confidence):
            self._diag["confidence"] += 1
            return

        # ── Gate 4: direction from zone, not from model class ──────────
        # Model predicts winner (1) vs loser (0) — NOT buy vs sell.
        # Direction comes from which zone price is in at this bar.
        # _raw already resolved above in Gate 1b.
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
            # Not clearly in a zone — skip
            self._diag["neutral"] += 1
            return

        # ── Soft HTF trend filter ──────────────────────────────────────
        # htf_4h_bias and htf_1h_bias are already model features — the model
        # already penalises counter-trend trades. The old hard gate blocked
        # ALL sells when 4H was bullish even when model confidence was high.
        # New rule: only skip if HTF is extreme (|bias| > threshold) AND
        # the model is not confident enough to override it.
        if _raw is not None:
            try:
                _htf = float(_raw.get("htf_4h_bias", 0) or 0)
                if abs(_htf) > HTF_EXTREME_THRESHOLD:
                    if winner_proba < float(self.p.confidence):
                        if pred_label == "sell" and _htf > 0:
                            return
                        if pred_label == "buy" and _htf < 0:
                            return
            except Exception:
                pass

        close_price = float(self.data.close[0])

        size      = self._calc_size(close_price)
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

        # Raw feature row for zone-aligned SL/TP (features.py boundaries)
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
            self.buy(size=size)
        else:
            if sl is None: sl = close_price * 1.003
            if tp is None: tp = close_price * 0.994
            if sl <= close_price or tp >= close_price:
                self._diag["bad_sltp"] += 1
                return
            self.sell(size=size)

        self._in_trade          = True
        self._entries_submitted += 1
        self._exit_pending      = False
        self._sl                = float(sl)
        self._tp                = float(tp)
        self._side              = pred_label
        self._entry_price       = float(close_price)
        self._trail_active      = False
        self._best_price        = None

        try:
            atr_val = float(lookback_df["candle_size"].rolling(14).mean().iloc[-1])
        except Exception:
            atr_val = None
        self._entry_atr = atr_val if atr_val and atr_val > 0 else None

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
    def diag(self)                  -> dict: return self._diag


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
    trail_trigger_pts:  float = 1500.0,
    trail_dist_atr:     float = 1.0,
    trail_dist_pts:     float = 1000.0,
    include_london_ny:  bool  = True,
    model_dir:   str = MODEL_DIR,
    min_zone_quality: float = MIN_ZONE_QUALITY,
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
    db.disconnect()

    if df.empty:
        raise ValueError(f"No rows returned for timeframe={timeframe}.")

    df_bt = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    df_bt = df_bt.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    model, metadata_bundle = _load_model_bundle(model_dir=model_dir)

    # Use optimal threshold saved during training if confidence not overridden
    saved_threshold = float(metadata_bundle.get("optimal_threshold", confidence))
    if confidence == 0.52:  # default — use saved threshold
        confidence = saved_threshold
        print(f"  Using saved optimal threshold: {confidence:.3f}")

    X_scaled = _build_feature_matrix_for_timeframe(df, timeframe, metadata_bundle)

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
        _raw_feat = _bf(df.copy())

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
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    MLSignalStrategy.model          = model
    MLSignalStrategy.features_by_dt = features_by_dt

    cerebro.addstrategy(
        MLSignalStrategy,
        confidence=float(confidence),
        stake=float(stake),
        use_pct_stake=use_pct_stake,
        trail_trigger_pts=float(trail_trigger_pts),
        trail_dist_atr=float(trail_dist_atr),
        trail_dist_pts=float(trail_dist_pts),
        include_london_ny=bool(include_london_ny),
        min_zone_quality=float(min_zone_quality),
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
    winrate     = (wins / trades * 100.0) if trades > 0 else 0.0
    diag        = getattr(strat_inst, "diag", {})

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
    parser.add_argument("--commission",         type=float, default=0.0)
    parser.add_argument("--trail-trigger-pts",  type=float, default=1500.0)
    parser.add_argument("--trail-dist-atr",     type=float, default=1.0)
    parser.add_argument("--trail-dist-pts",     type=float, default=1000.0)
    parser.add_argument("--model-dir",          default=MODEL_DIR)
    parser.add_argument("--no-london-ny",        dest="include_london_ny",
                        action="store_false", default=True,
                        help="Exclude London/NY overlap (H16). Use for 15min.")
    parser.add_argument("--min-zone-quality",    type=float, default=MIN_ZONE_QUALITY,
                        help=f"Minimum zone quality score to trade (default={MIN_ZONE_QUALITY})")
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
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Backtrader)")
    print("=" * 60)
    print(f"Timeframe:             {args.timeframe}")
    print(f"Date range:            {args.start_date or 'ALL'} -> {args.end_date or 'ALL'}")
    print(f"Final portfolio value: {res.final_value:,.2f}")
    print(f"PnL:                   {res.pnl:,.2f}")
    print(f"Max drawdown:          {res.max_drawdown_pct:.2f}%")
    print(f"Trades completed:      {res.total_trades}")
    print(f"Entries submitted:     {res.entries_submitted}")
    print(f"Skipped (low margin):  {res.skipped_no_margin}")
    print(f"Skipped (max pos):     {res.skipped_max_positions}")
    print(f"Trail activations:     {res.trail_activations}  "
          f"(trigger={args.trail_trigger_pts:.0f}pts, dist={args.trail_dist_atr:.1f}ATR)")
    print(f"Winrate:               {res.winrate_pct:.2f}%")
    print()
    print("-- Filter breakdown (bars rejected per gate) --")

    gate_counts = {
        "no_feature_row":    res.filtered_no_row,
        "session (inactive)": res.filtered_session,
        f"zone_quality<{args.min_zone_quality}": res.filtered_zone_quality,
        "low_confidence":    res.filtered_confidence,
        "neutral_prediction": res.filtered_neutral,
        "bad_sltp_geometry": res.filtered_bad_sltp,
    }
    bars_total = (
        res.filtered_no_row + res.filtered_session + res.filtered_zone_quality +
        res.filtered_confidence + res.filtered_neutral + res.filtered_bad_sltp +
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
    }
    print()
    for gate_name, count in gate_counts.items():
        pct = count / max(bars_total, 1) * 100
        if pct > GATE_WARN_THRESHOLD and gate_name in GATE_SUGGESTIONS:
            print(f"  WARNING: '{gate_name}' rejected {pct:.1f}% of bars — "
                  f"suggested fix: {GATE_SUGGESTIONS[gate_name]}")


if __name__ == "__main__":
    main()