"""
prepare_data.py — Data Preparation (New Pipeline)
==================================================

CHANGES (class-imbalance fix pass):
  - Default training timeframes: ["5min","15min","1H"] → ["5min","15min"].
    1H had only 34.5% win rate — adding it hurt more than it helped.

  - min_rr: 1.5 → 1.2 across all timeframes.
    At 1.5 only 106 winners survived across 110k rows (0.1%). That ratio
    is too extreme for XGBoost — it just predicts neutral every bar.
    1.2 RR roughly triples the winner count while still requiring real RR.

  - between_zones rows stripped from feature matrices before training.
    No-man's-land rows are all label=0 and make the imbalance worse.
    They're dropped from X_train/X_test only — raw_train/raw_test keep
    them so the backtest can iterate the full price feed without gaps.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

from db_connect import get_connection
from features import build_features, FEATURE_COLUMNS
from labels import generate_labels, get_class_weights

logger = logging.getLogger("mt5_collector.prepare_data")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

LTF_TIMEFRAMES = {"1min","2min","3min","4min","5min","10min","15min","30min"}

TF_PARAMS = {
    "1min":  {"impulse_atr": 0.4, "max_bars": 40,  "min_rr": 1.2, "use_midline_tp": True},
    "2min":  {"impulse_atr": 0.4, "max_bars": 40,  "min_rr": 1.2, "use_midline_tp": True},
    "3min":  {"impulse_atr": 0.4, "max_bars": 50,  "min_rr": 1.2, "use_midline_tp": True},
    "4min":  {"impulse_atr": 0.4, "max_bars": 50,  "min_rr": 1.2, "use_midline_tp": True},
    "5min":  {"impulse_atr": 0.5, "max_bars": 60,  "min_rr": 1.2, "use_midline_tp": True},
    "10min": {"impulse_atr": 0.5, "max_bars": 60,  "min_rr": 1.2, "use_midline_tp": True},
    "15min": {"impulse_atr": 0.5, "max_bars": 80,  "min_rr": 1.2, "use_midline_tp": True},
    "30min": {"impulse_atr": 0.6, "max_bars": 80,  "min_rr": 1.2, "use_midline_tp": True},
    "1H":    {"impulse_atr": 0.6, "max_bars": 120, "min_rr": 1.2, "use_midline_tp": True},
    "4H":    {"impulse_atr": 0.7, "max_bars": 60,  "min_rr": 1.2, "use_midline_tp": False},
    "1D":    {"impulse_atr": 0.8, "max_bars": 30,  "min_rr": 1.2, "use_midline_tp": False},
}

CATEGORICAL_COLS = ["session", "day_of_week"]


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
            FROM ustech_ohlcv
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

    def _strip_no_mans_land(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop between_zones=1 rows from the model input.
        The strategy never trades there — they're all label=0 and make
        the class imbalance worse. Raw DataFrames are kept intact for backtesting.
        """
        if "between_zones" not in df.columns:
            return df
        before = len(df)
        df = df[df["between_zones"].fillna(0) != 1.0].copy()
        after = len(df)
        logger.info(
            f"Stripped {before - after:,} between-zones rows "
            f"({(before - after) / before * 100:.1f}% of total)"
        )
        return df

    def prepare_data(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date:   str = None,
        symbol:     str = "USTECm",
        test_size:  float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:

        # CHANGED: default is now 5min + 15min only
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
                use_midline_tp=params["use_midline_tp"],
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

        # Chronological split on FULL data (raw_train/raw_test stay complete)
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

        # Strip no-mans-land from model inputs only
        train_model_df = self._strip_no_mans_land(train_df)
        test_model_df  = self._strip_no_mans_land(test_df)

        logger.info(
            f"Model rows after strip | "
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
    symbol:     str = "USTECm",
):
    prep = DataPreparator()
    return prep.prepare_data(timeframes, start_date, end_date, symbol)