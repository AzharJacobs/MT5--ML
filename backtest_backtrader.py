"""
backtest_backtrader.py - Proper trading backtest with Backtrader
===============================================================

Runs a simple long/short backtest using your saved ML model predictions.

What it does:
- Pulls historical candles from PostgreSQL for a chosen timeframe
- Rebuilds the same engineered/lagged features used in training
- Uses the saved model + saved scaler/feature_columns to generate per-bar signals
- Simulates trades in Backtrader and prints PnL / drawdown / winrate

Usage (examples):
  python backtest_backtrader.py --timeframe 1min --cash 10000 --stake 1
  python backtest_backtrader.py --timeframe 5min --confidence 0.55
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


def _load_model_bundle(model_dir: str = MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
    model, metadata = ModelTrainer.load_model(model_dir=model_dir)
    return model, metadata


def _build_feature_matrix_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    metadata_bundle: Dict[str, Any]
) -> pd.DataFrame:
    """
    Rebuild features aligned with the model's expected `feature_columns`,
    scaling them using the *saved* scaler from training metadata.
    """
    saved_scaler = metadata_bundle.get("scaler")
    feature_columns = metadata_bundle.get("feature_columns") or []

    if saved_scaler is None:
        raise ValueError("Saved scaler not found in model metadata. Retrain and save the model first.")
    if not feature_columns:
        raise ValueError("Saved feature_columns not found in model metadata. Retrain and save the model first.")

    prep = DataPreparator()

    data = df.copy()
    data = prep.apply_strategy_signals(data)
    data = prep.engineer_features(data)
    # match training default lag_periods=5 (train_model uses DataPreparator.prepare_data default)
    data = prep.create_lagged_features(data, lag_periods=5)

    # Ensure timeframe_encoded exists (engineer_features fits encoder on data)
    # Feature selection + missing fill
    X = data.copy()
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    X_scaled = pd.DataFrame(
        saved_scaler.transform(X),
        columns=feature_columns,
        index=data.index
    )

    # Keep timestamp to align with Backtrader feed
    X_scaled["timestamp"] = pd.to_datetime(data["timestamp"])
    X_scaled["close"] = pd.to_numeric(data["close"], errors="coerce")
    X_scaled["timeframe"] = timeframe

    return X_scaled


class MLSignalStrategy(bt.Strategy):
    params = dict(
        confidence=0.55,  # trade only when max(proba) >= confidence
        commission=0.0,
    )

    def __init__(self):
        # Track any submitted/accepted orders (including bracket legs).
        self._pending_orders = []
        self._wins = 0
        self._losses = 0
        self._trade_count = 0
        self._entries_submitted = 0

        # Injected externally
        self.model = getattr(self, "model", None)
        self.features_by_dt = getattr(self, "features_by_dt", None)

        if self.model is None or self.features_by_dt is None:
            raise RuntimeError("Strategy missing injected `model` or `features_by_dt`.")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            try:
                self._pending_orders.remove(order)
            except ValueError:
                pass

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._trade_count += 1
        if trade.pnlcomm > 0:
            self._wins += 1
        else:
            self._losses += 1

    def next(self):
        if self._pending_orders:
            return

        # Backtrader datetimes and pandas datetimes can differ subtly (tzinfo, microseconds).
        # Normalize to a tz-naive, second-resolution datetime for stable dict lookup.
        dt = self.data.datetime.datetime(0).replace(tzinfo=None, microsecond=0)
        row = self.features_by_dt.get(dt)
        if row is None:
            return

        X_row = row["X"]
        proba = self.model.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(proba))
        conf = float(np.max(proba))

        if conf < float(self.p.confidence):
            return

        # Use explicit class labels (supports 2-class or 3-class: buy/sell/neutral).
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            # Fallback to old convention if model doesn't expose classes_
            pred_label = "buy" if pred_idx == 1 else "sell"
        else:
            raw_label = classes[pred_idx]
            # Common training setups:
            # - binary: {0,1} meaning {sell,buy}
            # - ternary: {"sell","buy","neutral"} (or similar strings)
            if isinstance(raw_label, (int, np.integer)):
                pred_label = "buy" if int(raw_label) == 1 else "sell"
            else:
                pred_label = str(raw_label).lower()

        if pred_label in {"neutral", "hold"}:
            return

        close_price = float(self.data.close[0])

        # Build OHLCV lookback (oldest -> newest) for SL/TP calculators.
        lookback_n = 20
        if len(self.data) < lookback_n:
            return
        lookback_df = pd.DataFrame({
            "open":   [float(self.data.open[-i]) for i in range(lookback_n - 1, -1, -1)],
            "high":   [float(self.data.high[-i]) for i in range(lookback_n - 1, -1, -1)],
            "low":    [float(self.data.low[-i]) for i in range(lookback_n - 1, -1, -1)],
            "close":  [float(self.data.close[-i]) for i in range(lookback_n - 1, -1, -1)],
            "volume": [float(self.data.volume[-i]) for i in range(lookback_n - 1, -1, -1)],
        })
        lookback_df["candle_size"] = lookback_df["high"] - lookback_df["low"]
        lookback_df["body_size"] = (lookback_df["close"] - lookback_df["open"]).abs()
        lookback_df["wick_upper"] = lookback_df["high"] - lookback_df[["close", "open"]].max(axis=1)
        lookback_df["wick_lower"] = lookback_df[["close", "open"]].min(axis=1) - lookback_df["low"]

        # Simple regime:
        # - "buy": be long
        # - "sell": be short
        if pred_label == "buy":
            if self.position.size <= 0:
                if self.position.size < 0:
                    o = self.close()
                    if o is not None:
                        self._pending_orders.append(o)
                    return  # wait for close to complete
                sl = calculate_stop_loss(close_price, "buy", lookback_df)
                tp = calculate_take_profit(close_price, "buy", lookback_df)
                # Fallback: if zone-based SL/TP can't be computed from the 20-bar lookback,
                # use a simple fixed-percent bracket so the backtest can still trade.
                if sl is None:
                    sl = close_price * (1.0 - 0.003)
                if tp is None:
                    tp = close_price * (1.0 + 0.006)
                orders = self.buy_bracket(
                    price=close_price,
                    exectype=bt.Order.Market,
                    stopprice=float(sl),
                    limitprice=float(tp),
                )
                for o in orders:
                    if o is not None:
                        self._pending_orders.append(o)
                self._entries_submitted += 1
        else:
            if self.position.size >= 0:
                if self.position.size > 0:
                    o = self.close()
                    if o is not None:
                        self._pending_orders.append(o)
                    return  # wait for close to complete
                sl = calculate_stop_loss(close_price, "sell", lookback_df)
                tp = calculate_take_profit(close_price, "sell", lookback_df)
                if sl is None:
                    sl = close_price * (1.0 + 0.003)
                if tp is None:
                    tp = close_price * (1.0 - 0.006)
                orders = self.sell_bracket(
                    price=close_price,
                    exectype=bt.Order.Market,
                    stopprice=float(sl),
                    limitprice=float(tp),
                )
                for o in orders:
                    if o is not None:
                        self._pending_orders.append(o)
                self._entries_submitted += 1

    @property
    def wins(self) -> int:
        return self._wins

    @property
    def losses(self) -> int:
        return self._losses

    @property
    def trades(self) -> int:
        return self._trade_count

    @property
    def entries_submitted(self) -> int:
        return self._entries_submitted


def run_backtest(
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    cash: float,
    stake: float,
    confidence: float,
    commission: float,
    model_dir: str = MODEL_DIR
) -> BacktestResult:
    db = get_connection()
    if not db.connect():
        raise ConnectionError("Failed to connect to database")

    query = """
        SELECT *
        FROM ustech_ohlcv
        WHERE timeframe = %s
    """
    params = [timeframe]
    if start_date:
        query += " AND date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND date <= %s"
        params.append(end_date)
    query += " ORDER BY timestamp ASC"

    df = db.fetch_dataframe(query, tuple(params))
    db.disconnect()

    if df.empty:
        raise ValueError(f"No rows returned for timeframe={timeframe}.")

    # Backtrader expects columns: open/high/low/close/volume + datetime index/column
    df_bt = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    df_bt = df_bt.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_bt = df_bt.set_index("timestamp")

    model, metadata_bundle = _load_model_bundle(model_dir=model_dir)
    X_scaled = _build_feature_matrix_for_timeframe(df, timeframe, metadata_bundle)

    # Map datetime -> feature row for fast lookup inside next()
    feature_cols = [c for c in X_scaled.columns if c not in {"timestamp", "close", "timeframe"}]
    features_by_dt = {}
    for _, r in X_scaled.iterrows():
        dt = r["timestamp"].to_pydatetime().replace(tzinfo=None, microsecond=0)
        X_row = pd.DataFrame([r[feature_cols].to_numpy()], columns=feature_cols)
        features_by_dt[dt] = {"X": X_row}

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(cash))
    cerebro.broker.setcommission(commission=float(commission))
    cerebro.addsizer(bt.sizers.FixedSize, stake=float(stake))

    datafeed = bt.feeds.PandasData(dataname=df_bt)
    cerebro.adddata(datafeed)

    # analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    strat = cerebro.addstrategy(
        MLSignalStrategy,
        confidence=float(confidence),
        commission=float(commission),
    )

    # Inject model + features map after instantiation via "cheat": set on class,
    # Backtrader copies to instance attrs before __init__
    MLSignalStrategy.model = model  # type: ignore[attr-defined]
    MLSignalStrategy.features_by_dt = features_by_dt  # type: ignore[attr-defined]

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat_inst = results[0]
    end_value = cerebro.broker.getvalue()

    dd = strat_inst.analyzers.dd.get_analysis()
    max_dd = float(dd.get("max", {}).get("drawdown", 0.0))

    wins = int(getattr(strat_inst, "wins", 0))
    losses = int(getattr(strat_inst, "losses", 0))
    trades = int(getattr(strat_inst, "trades", 0))
    entries = int(getattr(strat_inst, "entries_submitted", 0))
    winrate = (wins / trades * 100.0) if trades > 0 else 0.0

    return BacktestResult(
        final_value=float(end_value),
        pnl=float(end_value - start_value),
        max_drawdown_pct=max_dd,
        winrate_pct=float(winrate),
        total_trades=trades,
        entries_submitted=entries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Backtrader backtest using saved ML model predictions.")
    parser.add_argument("--timeframe", default="1min", help="Timeframe to backtest (e.g. 1min, 5min, 1H).")
    parser.add_argument("--start-date", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--cash", type=float, default=10000.0, help="Starting cash.")
    parser.add_argument("--stake", type=float, default=1.0, help="Fixed stake per trade.")
    parser.add_argument("--confidence", type=float, default=0.55, help="Min probability to take a trade.")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission (e.g. 0.001 = 0.1 percent).")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory containing saved model files.")

    args = parser.parse_args()

    res = run_backtest(
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        cash=args.cash,
        stake=args.stake,
        confidence=args.confidence,
        commission=args.commission,
        model_dir=args.model_dir
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Backtrader)")
    print("=" * 60)
    print(f"Timeframe:            {args.timeframe}")
    # Use ASCII to avoid Windows cp1252 console encoding errors.
    print(f"Date range:           {args.start_date or 'ALL'} -> {args.end_date or 'ALL'}")
    print(f"Final portfolio value:{res.final_value:,.2f}")
    print(f"PnL:                  {res.pnl:,.2f}")
    print(f"Max drawdown:         {res.max_drawdown_pct:.2f}%")
    print(f"Trades:               {res.total_trades}")
    print(f"Entries submitted:    {res.entries_submitted}")
    print(f"Winrate:              {res.winrate_pct:.2f}%")


if __name__ == "__main__":
    main()

