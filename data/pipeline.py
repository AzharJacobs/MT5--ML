"""
prepare_data.py — Data Preparation (New Pipeline)
==================================================

CHANGES (zone-feature variance fix):
  - _strip_to_zone_touches() replaced with _build_model_dataset().

    The old filter kept ONLY zone-touch rows. This made in_demand_zone
    and in_supply_zone have zero variance (always 1) — XGBoost dropped
    them and fell back to hour/session/ema_trend_bias as its only signal.
    Result: model learned time-of-day trading, not zone trading.

    New approach: keep ALL zone-touch rows + a random sample of non-zone
    rows as negative context (controlled by CONTEXT_RATIO). This gives
    zone features real variance — the model can now learn "price in zone
    AND high quality AND right session = winner" instead of just "right
    session = winner".

    CONTEXT_RATIO = 2.0 means we keep 2x as many non-zone rows as zone
    rows. This keeps the dataset manageable while providing enough
    contrast for XGBoost to learn zone-specific patterns.

PREVIOUS CHANGES:
  - Default training timeframes: ["5min","15min"].
  - min_rr: 1.2 across all timeframes.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from data.loader import get_connection
from data.feature_engineer import build_features, FEATURE_COLUMNS
from strategy.signal_generator import generate_labels, get_class_weights

logger = logging.getLogger("mt5_collector.prepare_data")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

LTF_TIMEFRAMES = {"1min","2min","3min","4min","5min","10min","15min","30min"}

TF_PARAMS = {
    # include_london_ny: H16 kept for 5min (wr=31.8%), dropped for 15min (wr=24.4%).
    # min_sl_atr: 15min zones <0.5 ATR get stopped out by noise (wr=12.8%).
    # max_rr: 15min RR>4 signals have wr=0-8% — cap prevents unreachable TPs.
    # Session hours 13-14 (NY open) added globally via signal_generator._in_trading_session().
    "1min":  {"impulse_atr": 0.4, "max_bars": 40,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "2min":  {"impulse_atr": 0.4, "max_bars": 40,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "3min":  {"impulse_atr": 0.4, "max_bars": 50,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "4min":  {"impulse_atr": 0.4, "max_bars": 50,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "5min":  {"impulse_atr": 0.5, "max_bars": 60,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "10min": {"impulse_atr": 0.5, "max_bars": 60,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "15min": {"impulse_atr": 0.3, "max_bars": 80,  "min_rr": 1.2, "max_rr": 4.0,  "use_midline_tp": True,  "min_sl_atr": 0.3, "include_london_ny": False},
    "30min": {"impulse_atr": 0.6, "max_bars": 80,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "1H":    {"impulse_atr": 0.6, "max_bars": 120, "min_rr": 1.2, "max_rr": None, "use_midline_tp": True,  "min_sl_atr": 0.0, "include_london_ny": True},
    "4H":    {"impulse_atr": 0.7, "max_bars": 60,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": False, "min_sl_atr": 0.0, "include_london_ny": True},
    "1D":    {"impulse_atr": 0.8, "max_bars": 30,  "min_rr": 1.2, "max_rr": None, "use_midline_tp": False, "min_sl_atr": 0.0, "include_london_ny": True},
}

CATEGORICAL_COLS = ["session", "day_of_week"]

# How many non-zone rows to keep per zone row as negative context.
# 2.0 = keep 2x as many non-zone rows as zone rows.
# This gives zone features real variance so XGBoost learns zone patterns.
CONTEXT_RATIO = 0.0


class DataPreparator:

    def __init__(self):
        self.scaler           = StandardScaler()
        self.feature_columns: List[str] = []
        self._db              = get_connection()

    def _load_ohlcv(self, timeframe: str, symbol: str) -> pd.DataFrame:
        if not self._db.connect():
            raise ConnectionError("Cannot connect to database")
        query = """
            SELECT timestamp, open, high, low, close, volume,
                   hour, day_of_week, month, year, session,
                   candle_size, body_size, wick_upper, wick_lower
            FROM xauusd_ohlcv
            WHERE symbol = %s AND timeframe = %s
            ORDER BY timestamp ASC
        """
        df = self._db.fetch_dataframe(query, (symbol, timeframe))
        logger.info(f"Loaded {len(df):,} rows for {timeframe}")
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
        }
        session_map = {
            "asian": 0, "london": 1, "london_ny_overlap": 2,
            "new_york": 3, "off_hours": 4, "daily": 5, "unknown": -1,
        }
        if "day_of_week" in df.columns:
            df["day_of_week"] = df["day_of_week"].map(day_map).fillna(0).astype(float)
        if "session" in df.columns:
            df["session"]     = df["session"].map(session_map).fillna(-1).astype(float)
        return df

    def _build_model_dataset(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """
        Build model training dataset from signal rows only.

        PREVIOUS APPROACH (broken): kept ALL zone-touch rows as training data.
        This created a 0.4% / 99.6% class split — the model predicted everything
        as 0 and achieved 98% accuracy while identifying essentially zero winners
        (F1=0.026, recall=0.06). Training on 87K zone rows with 358 winners is
        an unsolvable needle-in-haystack problem.

        CURRENT APPROACH: filter to rows where signal != 0 (actual trade entries
        that passed all filters: session, volume, ATR, SL/TP caps). This produces
        a ~38% / 62% split (winners vs losers among valid signals), which XGBoost
        can learn from. The model now answers: "given a valid zone entry, will it
        hit TP or SL?" instead of "find the 0.4% winners in all zone candles."

        Features still vary between rows:
          - Zone quality (strength, freshness, touches) differs per signal
          - HTF bias and confirmation patterns differ
          - in_demand_zone / in_supply_zone encode direction (buy vs sell)
          - Hour, session, RR ratio differ per signal
        """
        if "signal" not in df.columns:
            # Fallback: no signal column — use all zone-touch rows
            in_demand = df.get("in_demand_zone", pd.Series(0.0, index=df.index))
            in_supply = df.get("in_supply_zone", pd.Series(0.0, index=df.index))
            signal_mask = (in_demand.fillna(0) == 1.0) | (in_supply.fillna(0) == 1.0)
        else:
            signal_mask = df["signal"] != 0

        result_df = df[signal_mask].copy()

        n_winners = (result_df["label"] != 0).sum() if "label" in result_df.columns else 0
        n_total   = len(result_df)

        logger.info(
            f"Model dataset | signal_rows={n_total:,} | winners={n_winners:,} "
            f"({n_winners/max(n_total,1)*100:.1f}%)"
        )
        return result_df

    def prepare_data(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date:   str = None,
        symbol:     str = "XAUUSDm",
        test_size:  float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:

        if timeframes is None:
            timeframes = ["5min", "15min"]

        all_frames: List[pd.DataFrame] = []

        for tf in timeframes:
            params = TF_PARAMS.get(tf, {
                "impulse_atr": 0.5, "max_bars": 50,
                "min_rr": 1.2, "use_midline_tp": True
            })
            logger.info(f"\n{'='*50}\nProcessing {tf} | params={params}\n{'='*50}")

            raw = self._load_ohlcv(tf, symbol)
            if raw.empty:
                logger.warning(f"No data for {tf} — skipping")
                continue

            raw["timestamp"] = pd.to_datetime(raw["timestamp"])
            if start_date:
                raw = raw[raw["timestamp"] >= pd.to_datetime(start_date)]
            if end_date:
                raw = raw[raw["timestamp"] <= pd.to_datetime(end_date)]
            if raw.empty:
                continue

            h1_df = h4_df = None
            if tf in LTF_TIMEFRAMES:
                h1_df = self._load_ohlcv("1H", symbol)
                h4_df = self._load_ohlcv("4H", symbol)
                if h1_df.empty: h1_df = None
                if h4_df.empty: h4_df = None

            feat_df = build_features(
                raw,
                h1_df=h1_df,
                h4_df=h4_df,
                impulse_atr_multiplier=params["impulse_atr"],
            )

            labeled = generate_labels(
                feat_df,
                max_bars=params["max_bars"],
                min_rr=params["min_rr"],
                max_rr=params.get("max_rr"),
                sl_buffer_atr=0.5,
                min_sl_atr=params.get("min_sl_atr", 0.0),
                use_midline_tp=params["use_midline_tp"],
                include_london_ny=params.get("include_london_ny", True),
                timeframe=tf,
            )

            labeled["timeframe"] = tf
            all_frames.append(labeled)

            signals = (labeled["signal"] != 0).sum()
            winners = (labeled["label"] != 0).sum()
            logger.info(f"{tf} — rows={len(labeled):,} signals={signals:,} winners={winners:,}")

        if not all_frames:
            raise ValueError("No data prepared — check DB connection and timeframes")

        full = pd.concat(all_frames, ignore_index=True)
        full = full.sort_values("timestamp").reset_index(drop=True)

        full = self._encode_categoricals(full)

        self.feature_columns = [c for c in FEATURE_COLUMNS if c in full.columns]

        tf_dummies = pd.get_dummies(full["timeframe"], prefix="tf").astype(float)
        full = pd.concat([full, tf_dummies], axis=1)
        self.feature_columns += [c for c in tf_dummies.columns
                                  if c not in self.feature_columns]

        # Chronological split — raw_train/raw_test stay complete for backtest
        split_idx = int(len(full) * (1 - test_size))
        train_df  = full.iloc[:split_idx].copy()
        test_df   = full.iloc[split_idx:].copy()

        logger.info(
            f"\nSplit | train={len(train_df):,} "
            f"({train_df['timestamp'].min().date()} → "
            f"{train_df['timestamp'].max().date()}) | "
            f"test={len(test_df):,} "
            f"({test_df['timestamp'].min().date()} → "
            f"{test_df['timestamp'].max().date()})"
        )

        # Build model datasets with zone rows + context (replaces strip_to_zone_touches)
        train_model_df = self._build_model_dataset(train_df, seed=42)
        test_model_df  = self._build_model_dataset(test_df,  seed=42)

        logger.info(
            f"Model rows after build | "
            f"train={len(train_model_df):,} | test={len(test_model_df):,}"
        )

        X_train = train_model_df[self.feature_columns].fillna(0)
        X_test  = test_model_df[self.feature_columns].fillna(0)
        y_train = train_model_df["label"]
        y_test  = test_model_df["label"]

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_columns, index=X_test.index
        )

        self._log_label_dist(y_train, "Train")
        self._log_label_dist(y_test,  "Test")

        return X_train_scaled, y_train, train_df, X_test_scaled, y_test, test_df

    def _log_label_dist(self, y: pd.Series, name: str) -> None:
        counts = y.value_counts().to_dict()
        total  = len(y)
        parts  = [f"{k}={v} ({v/total*100:.1f}%)"
                  for k, v in sorted(counts.items())]
        logger.info(f"{name} labels: {' | '.join(parts)}")

    def get_scaler(self) -> StandardScaler:
        return self.scaler

    def get_feature_columns(self) -> List[str]:
        return self.feature_columns

    def apply_strategy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return build_features(df)

    def create_lagged_features(self, df: pd.DataFrame,
                               lag_periods: int = 5) -> pd.DataFrame:
        return df


def prepare_data(
    timeframes: List[str] = None,
    start_date: str = None,
    end_date:   str = None,
    symbol:     str = "XAUUSDm",
):
    prep = DataPreparator()
    return prep.prepare_data(timeframes, start_date, end_date, symbol)