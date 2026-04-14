"""
backtest_backtrader.py - Proper trading backtest with Backtrader
===============================================================

Runs a simple long/short backtest using your saved ML model predictions.

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
    data = prep.create_lagged_features(data, lag_periods=5)

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

    X_scaled["timestamp"] = pd.to_datetime(data["timestamp"])
    X_scaled["close"] = pd.to_numeric(data["close"], errors="coerce")
    X_scaled["timeframe"] = timeframe

    return X_scaled


class MLSignalStrategy(bt.Strategy):
    params = dict(
        confidence=0.55,
        commission=0.0,
    )

    def __init__(self):
        self._wins = 0
        self._losses = 0
        self._trade_count = 0
        self._entries_submitted = 0

        # True while a bracket is live (entry pending or position open)
        self._in_trade = False

        self.model = getattr(self, "model", None)
        self.features_by_dt = getattr(self, "features_by_dt", None)

        if self.model is None or self.features_by_dt is None:
            raise RuntimeError("Strategy missing injected `model` or `features_by_dt`.")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self._trade_count += 1
        self._in_trade = False          # trade fully closed — allow new entries
        if trade.pnlcomm > 0:
            self._wins += 1
        else:
            self._losses += 1

    def notify_order(self, order):
        # Release flag if entry order fails before opening a position
        if order.status in (order.Canceled, order.Rejected, order.Margin):
            if self.position.size == 0:
                self._in_trade = False
                print(f"  [WARN] Order failed ({order.status}) — resetting trade flag")

    def next(self):
        # Only enter when flat AND no bracket currently live
        if self._in_trade:
            return
        if self.position.size != 0:
            return

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

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            pred_label = "buy" if pred_idx == 1 else "sell"
        else:
            raw_label = classes[pred_idx]
            if isinstance(raw_label, (int, np.integer)):
                pred_label = "buy" if int(raw_label) == 1 else "sell"
            else:
                pred_label = str(raw_label).lower()

        if pred_label in {"neutral", "hold"}:
            return

        close_price = float(self.data.close[0])

        lookback_n = 20
        if len(self.data) < lookback_n:
            return

        lookback_df = pd.DataFrame({
            "open":   [float(self.data.open[-i])   for i in range(lookback_n - 1, -1, -1)],
            "high":   [float(self.data.high[-i])   for i in range(lookback_n - 1, -1, -1)],
            "low":    [float(self.data.low[-i])    for i in range(lookback_n - 1, -1, -1)],
            "close":  [float(self.data.close[-i])  for i in range(lookback_n - 1, -1, -1)],
            "volume": [float(self.data.volume[-i]) for i in range(lookback_n - 1, -1, -1)],
        })
        lookback_df["candle_size"] = lookback_df["high"] - lookback_df["low"]
        lookback_df["body_size"]   = (lookback_df["close"] - lookback_df["open"]).abs()
        lookback_df["wick_upper"]  = lookback_df["high"] - lookback_df[["close", "open"]].max(axis=1)
        lookback_df["wick_lower"]  = lookback_df[["close", "open"]].min(axis=1) - lookback_df["low"]

        sl = calculate_stop_loss(close_price, pred_label, lookback_df)
        tp = calculate_take_profit(close_price, pred_label, lookback_df)

        if pred_label == "buy":
            if sl is None:
                sl = close_price * 0.997
            if tp is None:
                tp = close_price * 1.006
            # Guard: SL must be below entry, TP must be above entry
            if sl >= close_price or tp <= close_price:
                return
            self.buy_bracket(
                price=close_price,
                exectype=bt.Order.Market,
                stopprice=float(sl),
                limitprice=float(tp),
            )
        else:  # sell
            if sl is None:
                sl = close_price * 1.003
            if tp is None:
                tp = close_price * 0.994
            # Guard: SL must be above entry, TP must be below entry
            if sl <= close_price or tp >= close_price:
                return
            self.sell_bracket(
                price=close_price,
                exectype=bt.Order.Market,
                stopprice=float(sl),
                limitprice=float(tp),
            )

        self._in_trade = True
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

    df_bt = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    df_bt = df_bt.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_bt = df_bt.set_index("timestamp")

    model, metadata_bundle = _load_model_bundle(model_dir=model_dir)
    X_scaled = _build_feature_matrix_for_timeframe(df, timeframe, metadata_bundle)

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
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")

    MLSignalStrategy.model = model  # type: ignore[attr-defined]
    MLSignalStrategy.features_by_dt = features_by_dt  # type: ignore[attr-defined]

    cerebro.addstrategy(
        MLSignalStrategy,
        confidence=float(confidence),
        commission=float(commission),
    )

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat_inst = results[0]
    end_value = cerebro.broker.getvalue()

    dd = strat_inst.analyzers.dd.get_analysis()
    max_dd = float(dd.get("max", {}).get("drawdown", 0.0))

    wins   = int(getattr(strat_inst, "wins", 0))
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
    parser.add_argument("--timeframe",  default="1min",  help="Timeframe to backtest (e.g. 1min, 5min, 1H).")
    parser.add_argument("--start-date", default=None,    help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date",   default=None,    help="End date (YYYY-MM-DD).")
    parser.add_argument("--cash",       type=float, default=10000.0, help="Starting cash.")
    parser.add_argument("--stake",      type=float, default=1.0,     help="Fixed stake per trade.")
    parser.add_argument("--confidence", type=float, default=0.55,    help="Min probability to take a trade.")
    parser.add_argument("--commission", type=float, default=0.0,     help="Commission (e.g. 0.001 = 0.1%).")
    parser.add_argument("--model-dir",  default=MODEL_DIR,           help="Directory containing saved model files.")

    args = parser.parse_args()

    res = run_backtest(
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        cash=args.cash,
        stake=args.stake,
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
    print(f"Trades:                {res.total_trades}")
    print(f"Entries submitted:     {res.entries_submitted}")
    print(f"Winrate:               {res.winrate_pct:.2f}%")


if __name__ == "__main__":
    main()