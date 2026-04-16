"""
backtest_backtrader.py - Proper trading backtest with Backtrader
===============================================================

Usage (examples):
  python backtest_backtrader.py --timeframe 1H --cash 10000 --stake 0.1
  python backtest_backtrader.py --timeframe 5min --confidence 0.55 --stake 0.01
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


@dataclass(frozen=True)
class BacktestResult:
    final_value: float
    pnl: float
    max_drawdown_pct: float
    winrate_pct: float
    total_trades: int
    entries_submitted: int
    skipped_no_margin: int


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

    # Build features using the new pipeline
    data = df.copy()
    data = build_features(data)

    # Remove direction — excluded from model features
    if "direction" in data.columns:
        data = data.drop(columns=["direction"])

    # Encode categoricals — must match prepare_data.py exactly
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

    # Timeframe one-hot — match columns from training
    tf_dummies = pd.get_dummies(
        pd.Series([timeframe] * len(data)), prefix="tf"
    ).astype(float)
    tf_dummies.index = data.index
    data = pd.concat([data, tf_dummies], axis=1)

    # Align to training feature columns exactly
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
        stake=0.1,           # fraction of available cash per trade (0.1 = 10%)
        use_pct_stake=True,  # True  → stake = % of cash  (recommended for indices)
                             # False → stake = fixed units (original behaviour)
    )

    def __init__(self):
        self._wins             = 0
        self._losses           = 0
        self._trade_count      = 0
        self._entries_submitted = 0
        self._skipped_margin   = 0
        self._in_trade         = False
        self._exit_pending     = False

        # Manual SL/TP management (more reliable than bracket + Market parent)
        self._sl: Optional[float] = None
        self._tp: Optional[float] = None
        self._side: Optional[str] = None  # "buy" | "sell"

        self.model          = getattr(self, "model", None)
        self.features_by_dt = getattr(self, "features_by_dt", None)
        # Pre-built numpy array for fast prediction: index → feature row
        self.feature_array  = getattr(self, "feature_array", None)
        self.feature_cols   = getattr(self, "feature_cols", None)

        if self.model is None or self.features_by_dt is None:
            raise RuntimeError("Strategy missing injected `model` or `features_by_dt`.")

    # ------------------------------------------------------------------
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._trade_count += 1
        self._in_trade = False
        self._exit_pending = False
        self._sl = None
        self._tp = None
        self._side = None
        if trade.pnlcomm > 0:
            self._wins += 1
        else:
            self._losses += 1

    def notify_order(self, order):
        if order.status in (order.Canceled, order.Rejected, order.Margin):
            if self.position.size == 0:
                self._in_trade = False
                self._exit_pending = False
                self._sl = None
                self._tp = None
                self._side = None
                status_name = {
                    order.Canceled: "Canceled",
                    order.Rejected: "Rejected",
                    order.Margin:   "Margin/Insufficient funds",
                }.get(order.status, str(order.status))
                print(f"  [WARN] Order failed ({status_name}) — resetting trade flag")
        elif order.status == order.Completed:
            # If an exit order completed, clear pending flag; trade stats handled in notify_trade
            if self.position.size == 0:
                self._exit_pending = False

    # ------------------------------------------------------------------
    def _calc_size(self, price: float) -> float:
        """Return position size (units) based on stake setting."""
        if self.p.use_pct_stake:
            cash      = self.broker.getcash()
            trade_val = cash * float(self.p.stake)   # e.g. 10% of cash
            size      = trade_val / price
        else:
            size = float(self.p.stake)
        return max(size, 1e-8)

    # ------------------------------------------------------------------
    def next(self):
        # Manual exit management: close on next bar if SL/TP touched in-bar
        if self.position.size != 0:
            if self._exit_pending:
                return
            side = self._side
            sl   = self._sl
            tp   = self._tp
            if side and sl is not None and tp is not None:
                bar_high = float(self.data.high[0])
                bar_low  = float(self.data.low[0])
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

        dt  = self.data.datetime.datetime(0).replace(tzinfo=None, microsecond=0)
        row = self.features_by_dt.get(dt)
        if row is None:
            return

        # --- fast prediction using pre-built array row ---
        X_row = row["X"]   # already a (1, n_features) DataFrame
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

        # --- margin check before submitting ---
        size      = self._calc_size(close_price)
        required  = close_price * size
        available = self.broker.getcash()
        if required > available * 0.99:          # leave 1% buffer
            self._skipped_margin += 1
            return

        # --- lookback for SL/TP ---
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

        self._in_trade = True
        self._entries_submitted += 1
        self._exit_pending = False
        self._sl = float(sl)
        self._tp = float(tp)
        self._side = pred_label

    # ------------------------------------------------------------------
    @property
    def wins(self)              -> int: return self._wins
    @property
    def losses(self)            -> int: return self._losses
    @property
    def trades(self)            -> int: return self._trade_count
    @property
    def entries_submitted(self) -> int: return self._entries_submitted
    @property
    def skipped_no_margin(self) -> int: return self._skipped_margin


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

    # Pre-build feature dict: datetime → {"X": DataFrame(1, n_features)}
    features_by_dt: Dict[Any, Any] = {}
    for _, r in X_scaled.iterrows():
        dt    = r["timestamp"].to_pydatetime().replace(tzinfo=None, microsecond=0)
        X_row = pd.DataFrame([r[feature_cols].to_numpy()], columns=feature_cols)
        features_by_dt[dt] = {"X": X_row}

    # Warn if the feature dict is sparse relative to the price feed
    print(f"  Price bars:    {len(df_bt)}")
    print(f"  Feature rows:  {len(features_by_dt)}")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=float(commission))
    # Note: sizer is overridden per-trade inside next() when use_pct_stake=True
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
    )

    start_value = cerebro.broker.getvalue()
    results     = cerebro.run()
    strat_inst  = results[0]
    end_value   = cerebro.broker.getvalue()

    dd     = strat_inst.analyzers.dd.get_analysis()
    max_dd = float(dd.get("max", {}).get("drawdown", 0.0))

    trades  = int(getattr(strat_inst, "trades", 0))
    wins    = int(getattr(strat_inst, "wins", 0))
    entries = int(getattr(strat_inst, "entries_submitted", 0))
    skipped = int(getattr(strat_inst, "skipped_no_margin", 0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0

    return BacktestResult(
        final_value=float(end_value),
        pnl=float(end_value - start_value),
        max_drawdown_pct=max_dd,
        winrate_pct=float(winrate),
        total_trades=trades,
        entries_submitted=entries,
        skipped_no_margin=skipped,
    )


# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe",     default="1min")
    parser.add_argument("--start-date",    default=None)
    parser.add_argument("--end-date",      default=None)
    parser.add_argument("--cash",          type=float, default=10000.0)
    parser.add_argument("--stake",         type=float, default=0.1,
                        help="Fraction of cash per trade if --pct-stake (default 0.1=10%%), "
                             "or fixed units if --no-pct-stake.")
    parser.add_argument("--pct-stake",     dest="use_pct_stake", action="store_true",  default=True,
                        help="Use stake as %% of available cash (default, recommended for indices).")
    parser.add_argument("--no-pct-stake",  dest="use_pct_stake", action="store_false",
                        help="Use stake as fixed unit count.")
    parser.add_argument("--confidence",    type=float, default=0.55)
    parser.add_argument("--commission",    type=float, default=0.0)
    parser.add_argument("--model-dir",     default=MODEL_DIR)
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
    print(f"Winrate:               {res.winrate_pct:.2f}%")


if __name__ == "__main__":
    main()