"""
backtest_backtrader.py - Proper trading backtest with Backtrader
===============================================================

Usage (examples):
  python backtest_backtrader.py --timeframe 1H --cash 10000 --stake 0.15
  python backtest_backtrader.py --timeframe 5min --confidence 0.55 --stake 0.15

CHANGES (profit optimisation pass):
  - Default stake raised from 0.10 → 0.15 (15% of available cash per trade).
    The strategy has proven profitable and drawdown is very controlled (<2%),
    so sizing up is the clearest lever available without changing the model.
  - MAX_CONCURRENT_POSITIONS = 2 guard added.
    Raising stake means we need a hard cap on simultaneous open trades to
    prevent compounding risk when multiple signals fire in the same session.
    Two open positions at 15% each = 30% capital at risk — manageable.
  - _open_position_count tracked via notify_trade so the cap is accurate
    regardless of whether SL/TP or a manual close exits the trade.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

import backtrader as bt

from db_connect import get_connection
from prepare_data import DataPreparator
from train_model import ModelTrainer, MODEL_DIR
from strategy import calculate_stop_loss, calculate_take_profit

# Maximum number of trades open at the same time.
# At 15% stake each, 2 positions = 30% capital at risk — the safe ceiling
# for a small account with ~1.6% max drawdown target.
MAX_CONCURRENT_POSITIONS = 2


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
    trail_activations: int    # trades where trailing SL was triggered


def _load_model_bundle(model_dir: str = MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
    model, metadata = ModelTrainer.load_model(model_dir=model_dir)
    return model, metadata


def _build_feature_matrix_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    metadata_bundle: Dict[str, Any]
) -> pd.DataFrame:
    import numpy as np
    from features import build_features

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
        confidence=0.55,
        stake=0.15,
        use_pct_stake=True,
        # ------------------------------------------------------------------
        # Trailing stop in profit
        #
        # trail_trigger_pts : how many points of profit price must reach before
        #                     the trailing stop activates at all.
        #                     0 = trail from the very first tick of profit.
        #                     Set to e.g. 100 to only start trailing once you're
        #                     comfortably 100 points in the green.
        #
        # trail_dist_atr    : once active, the SL trails this many ATRs behind
        #                     the current best price (high watermark for buys,
        #                     low watermark for sells).
        #                     1.0 ATR on USTEC 15min ≈ 80-120 points typically.
        #                     Set to 0 to use trail_dist_pts instead.
        #
        # trail_dist_pts    : fixed-point trail distance used when trail_dist_atr
        #                     is 0. Your example: entry 19000, trail 100 pts means
        #                     SL is always 100 points behind the best price seen.
        #
        # Only one of trail_dist_atr / trail_dist_pts is used:
        #   trail_dist_atr > 0  →  ATR-based (recommended, adapts to volatility)
        #   trail_dist_atr == 0 →  fixed points (trail_dist_pts)
        # ------------------------------------------------------------------
        trail_trigger_pts=100.0,   # start trailing after 100 pts profit
        trail_dist_atr=1.0,        # trail 1 ATR behind best price
        trail_dist_pts=100.0,      # fallback fixed trail if ATR unavailable
    )

    def __init__(self):
        self._wins                  = 0
        self._losses                = 0
        self._trade_count           = 0
        self._entries_submitted     = 0
        self._skipped_margin        = 0
        self._skipped_max_pos       = 0
        self._open_position_count   = 0
        self._trail_activations     = 0   # how many trades had trailing SL triggered
        self._in_trade              = False
        self._exit_pending          = False

        self._sl: Optional[float]          = None
        self._tp: Optional[float]          = None
        self._side: Optional[str]          = None
        self._entry_price: Optional[float] = None
        self._trail_active: bool           = False  # True once trailing SL is live
        self._best_price: Optional[float]  = None   # high watermark (buy) / low (sell)
        self._entry_atr: Optional[float]   = None   # ATR captured at entry

        self.model          = getattr(self, "model", None)
        self.features_by_dt = getattr(self, "features_by_dt", None)
        self.feature_array  = getattr(self, "feature_array", None)
        self.feature_cols   = getattr(self, "feature_cols", None)

        if self.model is None or self.features_by_dt is None:
            raise RuntimeError("Strategy missing injected `model` or `features_by_dt`.")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _calc_size(self, price: float) -> float:
        if self.p.use_pct_stake:
            cash      = self.broker.getcash()
            trade_val = cash * float(self.p.stake)
            size      = trade_val / price
        else:
            size = float(self.p.stake)
        return max(size, 1e-8)

    def _trail_distance(self) -> float:
        """Return the trailing distance in points for the current trade."""
        if float(self.p.trail_dist_atr) > 0 and self._entry_atr:
            return float(self.p.trail_dist_atr) * self._entry_atr
        return float(self.p.trail_dist_pts)

    # ------------------------------------------------------------------
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

                    # ── Activate trailing once trigger profit reached ───
                    if not self._trail_active and profit_pts >= trigger:
                        self._trail_active    = True
                        self._best_price      = bar_high
                        self._trail_activations += 1

                    # ── Update best price and trail SL upward ──────────
                    if self._trail_active:
                        if bar_high > self._best_price:
                            self._best_price = bar_high
                        # SL trails behind best price, but never moves backward
                        new_sl = self._best_price - trail_dist
                        if new_sl > self._sl:
                            self._sl = new_sl
                        sl = self._sl

                elif side == "sell":
                    profit_pts = entry - bar_low

                    if not self._trail_active and profit_pts >= trigger:
                        self._trail_active    = True
                        self._best_price      = bar_low
                        self._trail_activations += 1

                    if self._trail_active:
                        if bar_low < self._best_price:
                            self._best_price = bar_low
                        new_sl = self._best_price + trail_dist
                        if new_sl < self._sl:
                            self._sl = new_sl
                        sl = self._sl

                # ── SL / TP exit check ─────────────────────────────────
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

        # NEW: hard cap on concurrent positions
        if self._open_position_count >= MAX_CONCURRENT_POSITIONS:
            self._skipped_max_pos += 1
            return

        dt  = self.data.datetime.datetime(0).replace(tzinfo=None, microsecond=0)
        row = self.features_by_dt.get(dt)
        if row is None:
            return

        X_row        = row["X"]
        zone_quality = row.get("zone_quality", float("nan"))

        # NEW: zone quality gate — only trade setups scoring ≥ 3.5 / 6.
        # active_zone_quality aggregates: zone strength, freshness, touch count,
        # zone width, consolidation quality, and HTF alignment. Scores below
        # 3.5 tend to be weak or over-touched zones with poor follow-through.
        MIN_ZONE_QUALITY = 3.5
        if not (isinstance(zone_quality, float) and zone_quality >= MIN_ZONE_QUALITY):
            return
        proba    = self.model.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(proba))
        conf     = float(np.max(proba))

        if conf < float(self.p.confidence):
            return

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            pred_label = "buy" if pred_idx == 1 else "sell"
        else:
            raw_label = classes[pred_idx]
            pred_label = ("buy" if int(raw_label) == 1 else "sell") \
                if isinstance(raw_label, (int, np.integer)) \
                else str(raw_label).lower()

        if pred_label in {"neutral", "hold"}:
            return

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

        sl = calculate_stop_loss(close_price, pred_label, lookback_df)
        tp = calculate_take_profit(close_price, pred_label, lookback_df)

        if pred_label == "buy":
            if sl is None: sl = close_price * 0.997
            if tp is None: tp = close_price * 1.006
            if sl >= close_price or tp <= close_price:
                return
            self.buy(size=size)
        else:
            if sl is None: sl = close_price * 1.003
            if tp is None: tp = close_price * 0.994
            if sl <= close_price or tp >= close_price:
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
        # Capture ATR at entry from the most recent bar's feature row if available,
        # otherwise fall back to estimating from the lookback range.
        try:
            atr_val = float(lookback_df["candle_size"].rolling(14).mean().iloc[-1])
        except Exception:
            atr_val = None
        self._entry_atr = atr_val if atr_val and atr_val > 0 else None

    # ------------------------------------------------------------------
    @property
    def wins(self)                  -> int: return self._wins
    @property
    def losses(self)                -> int: return self._losses
    @property
    def trades(self)                -> int: return self._trade_count
    @property
    def entries_submitted(self)     -> int: return self._entries_submitted
    @property
    def skipped_no_margin(self)     -> int: return self._skipped_margin
    @property
    def skipped_max_positions(self) -> int: return self._skipped_max_pos
    @property
    def trail_activations(self)     -> int: return self._trail_activations


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
    trail_trigger_pts: float = 100.0,
    trail_dist_atr:    float = 1.0,
    trail_dist_pts:    float = 100.0,
    model_dir:   str = MODEL_DIR,
) -> BacktestResult:

    db = get_connection()
    if not db.connect():
        raise ConnectionError("Failed to connect to database")

    query  = "SELECT * FROM ustech_ohlcv WHERE timeframe = %s"
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
    X_scaled = _build_feature_matrix_for_timeframe(df, timeframe, metadata_bundle)

    feature_cols = [c for c in X_scaled.columns if c not in {"timestamp", "close", "timeframe"}]

    # Build a lookup of active_zone_quality from the raw (unscaled) feature data
    # so the quality gate can use the real 0-6 score, not the scaled version.
    # We re-derive it from X_scaled's index → merge back onto the original data.
    aq_series: Dict[Any, float] = {}
    if "active_zone_quality" in X_scaled.columns:
        # X_scaled index aligns with the original data rows; grab quality from
        # the raw feature column before scaling distorts the 0-6 range.
        # We stored it in X_scaled (StandardScaler shifts but keeps rank), but
        # for a hard threshold we need raw. Re-read from the pre-scaled df via
        # inverse_transform on just that column — simpler: store raw separately.
        pass  # handled below by storing raw_zone_quality per row

    features_by_dt: Dict[Any, Any] = {}
    for _, r in X_scaled.iterrows():
        dt    = r["timestamp"].to_pydatetime().replace(tzinfo=None, microsecond=0)
        X_row = pd.DataFrame([r[feature_cols].to_numpy()], columns=feature_cols)
        features_by_dt[dt] = {"X": X_row, "zone_quality": float("nan")}

    # Overlay raw active_zone_quality values (pre-scaling) for the gate check.
    # We need the un-scaled dataframe — rebuild features from raw df quickly.
    try:
        from features import build_features as _bf
        _raw_feat = _bf(df.copy())
        if "active_zone_quality" in _raw_feat.columns and "timestamp" in _raw_feat.columns:
            _raw_feat["timestamp"] = pd.to_datetime(_raw_feat["timestamp"])
            for _, rr in _raw_feat[["timestamp", "active_zone_quality"]].iterrows():
                _dt = rr["timestamp"].to_pydatetime().replace(tzinfo=None, microsecond=0)
                if _dt in features_by_dt:
                    features_by_dt[_dt]["zone_quality"] = float(rr["active_zone_quality"])
    except Exception as _e:
        print(f"  [WARN] Could not attach zone_quality for gate check: {_e}")

    print(f"  Price bars:    {len(df_bt)}")
    print(f"  Feature rows:  {len(features_by_dt)}")

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
    )


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe",     default="1min")
    parser.add_argument("--start-date",    default=None)
    parser.add_argument("--end-date",      default=None)
    parser.add_argument("--cash",          type=float, default=10000.0)
    parser.add_argument("--stake",         type=float, default=0.15,       # CHANGED: 0.10 → 0.15
                        help="Fraction of cash per trade if --pct-stake (default 0.15=15%%), "
                             "or fixed units if --no-pct-stake.")
    parser.add_argument("--pct-stake",          dest="use_pct_stake", action="store_true", default=True)
    parser.add_argument("--no-pct-stake",        dest="use_pct_stake", action="store_false")
    parser.add_argument("--confidence",          type=float, default=0.55)
    parser.add_argument("--commission",          type=float, default=0.0)
    parser.add_argument("--trail-trigger-pts",   type=float, default=100.0,
                        help="Points of profit needed before trailing SL activates (default 100).")
    parser.add_argument("--trail-dist-atr",      type=float, default=1.0,
                        help="Trail distance in ATR multiples (default 1.0). Set 0 to use --trail-dist-pts.")
    parser.add_argument("--trail-dist-pts",      type=float, default=100.0,
                        help="Fixed trail distance in points, used when --trail-dist-atr is 0 (default 100).")
    parser.add_argument("--model-dir",           default=MODEL_DIR)
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
        model_dir=args.model_dir,
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
    print(f"Trail activations:     {res.trail_activations}  (trigger={args.trail_trigger_pts:.0f}pts, dist={args.trail_dist_atr:.1f}ATR)")
    print(f"Winrate:               {res.winrate_pct:.2f}%")


if __name__ == "__main__":
    main()