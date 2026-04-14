"""
prepare_data.py - Data Preparation Module
==========================================

Pulls OHLCV data directly from PostgreSQL and prepares it for ML training.
Applies trading strategy signals as additional features.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from db_connect import get_connection, DatabaseConnection
from strategy import (
    apply_strategy,
    LOOKBACK_PERIODS,
    detect_demand_zone,
    detect_supply_zone,
    price_in_zone,
    ZONE_TOUCH_TOLERANCE,
)


class DataPreparator:
    """
    Handles all data preparation for the ML model.
    Pulls data from PostgreSQL and applies strategy signals.
    """

    def __init__(self):
        """Initialize data preparator with database connection."""
        self.db: DatabaseConnection = get_connection()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.timeframe_encoder = LabelEncoder()

    def fetch_training_data(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch training data from PostgreSQL database.

        Args:
            timeframes: List of timeframes to include (None = all)
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)

        Returns:
            Raw DataFrame from database
        """
        print("Fetching training data from PostgreSQL...")

        if not self.db.connect():
            raise ConnectionError("Failed to connect to database")

        # Build query for multiple timeframes
        query_parts = ["SELECT * FROM ustech_ohlcv WHERE 1=1"]
        params = []

        if timeframes:
            placeholders = ', '.join(['%s'] * len(timeframes))
            query_parts.append(f"AND timeframe IN ({placeholders})")
            params.extend(timeframes)

        if start_date:
            query_parts.append("AND date >= %s")
            params.append(start_date)

        if end_date:
            query_parts.append("AND date <= %s")
            params.append(end_date)

        query_parts.append("ORDER BY timeframe, timestamp ASC")
        query = " ".join(query_parts)

        df = self.db.fetch_dataframe(query, tuple(params) if params else None)

        print(f"[OK] Fetched {len(df)} records from database")
        return df

    def apply_strategy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trading strategy to each candle and add signal column.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added 'strategy_signal' column
        """
        print("Applying trading strategy signals...")

        # Sort by timeframe and timestamp to ensure correct order
        df = df.sort_values(['timeframe', 'timestamp']).reset_index(drop=True)

        strategy_signals = []

        # Process each timeframe separately
        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe].copy()

            for i in range(len(tf_data)):
                current_candle = tf_data.iloc[i]

                # Get lookback data (previous N candles)
                if i >= LOOKBACK_PERIODS:
                    lookback_start = i - LOOKBACK_PERIODS
                    lookback_data = tf_data.iloc[lookback_start:i]
                elif i > 0:
                    lookback_data = tf_data.iloc[0:i]
                else:
                    lookback_data = None

                # Apply strategy
                signal = apply_strategy(current_candle, lookback_data)
                strategy_signals.append({
                    'index': tf_data.index[i],
                    'signal': signal
                })

        # Create signal mapping
        signal_map = {item['index']: item['signal'] for item in strategy_signals}
        df['strategy_signal'] = df.index.map(signal_map)

        # Count signals
        signal_counts = df['strategy_signal'].value_counts()
        # Use ASCII to avoid Windows cp1252 console encoding errors.
        print("[OK] Strategy signals applied:")
        for signal, count in signal_counts.items():
            print(f"  - {signal}: {count}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for ML model.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")

        df = df.copy()

        # Price-based features
        df['price_range'] = df['high'] - df['low']
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100

        # Body and wick ratios
        df['body_to_range_ratio'] = np.where(
            df['candle_size'] > 0,
            df['body_size'] / df['candle_size'],
            0
        )
        df['upper_wick_ratio'] = np.where(
            df['candle_size'] > 0,
            df['wick_upper'] / df['candle_size'],
            0
        )
        df['lower_wick_ratio'] = np.where(
            df['candle_size'] > 0,
            df['wick_lower'] / df['candle_size'],
            0
        )

        # Volume features (if volume > 0)
        df['volume_normalized'] = df.groupby('timeframe')['volume'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

        # Time-based cyclical features (for better pattern recognition)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if not pd.api.types.is_numeric_dtype(df['day_of_week']):
            day_map = {
                'monday': 0,
                'tuesday': 1,
                'wednesday': 2,
                'thursday': 3,
                'friday': 4,
                'saturday': 5,
                'sunday': 6
            }
            df['day_of_week'] = (
                df['day_of_week']
                .astype(str)
                .str.strip()
                .str.lower()
                .map(day_map)
            )
            df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(0).astype(int)

        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Encode timeframe
        df['timeframe_encoded'] = self.timeframe_encoder.fit_transform(df['timeframe'])

        # Binary direction encoding
        df['is_bullish'] = (df['direction'] == 'buy').astype(int)

        # Strategy signal encoding
        signal_map = {'buy': 1, 'sell': -1, 'neutral': 0}
        df['strategy_signal_encoded'] = df['strategy_signal'].map(signal_map).fillna(0)

        print(f"[OK] Engineered {len(df.columns)} features")
        return df

    def create_zone_outcome_labels(
        self,
        df: pd.DataFrame,
        zone_lookback: int = 30,
        horizon: int = 20,
        zone_touch_tolerance: float = ZONE_TOUCH_TOLERANCE,
    ) -> pd.DataFrame:
        """
        Create forward-looking zone-to-zone outcome labels with zero lookahead in zone detection.

        Label semantics:
          -  1 (buy): price returns into a past-identified demand zone, then reaches the next supply zone within N candles.
          - -1 (sell): price returns into a past-identified supply zone, then reaches the next demand zone within N candles.
          -  0 (neutral): outcome not achieved within horizon, or missing required zones.

        Important:
          - Zones are detected using ONLY past candles (lookback window ending at t-1).
          - Outcome is evaluated using ONLY future candles (t+1 ... t+horizon).
        """
        print(
            "Creating forward-looking zone outcome labels "
            f"(lookback={zone_lookback}, horizon={horizon})..."
        )

        df = df.sort_values(['timeframe', 'timestamp']).reset_index(drop=True).copy()
        df['zone_label'] = 0

        labeled_dfs: List[pd.DataFrame] = []

        for timeframe in df['timeframe'].unique():
            tf = df[df['timeframe'] == timeframe].copy().reset_index(drop=True)

            if len(tf) < (zone_lookback + horizon + 2):
                labeled_dfs.append(tf)
                continue

            labels = np.zeros(len(tf), dtype=int)

            for i in range(zone_lookback, len(tf) - horizon - 1):
                lookback_data = tf.iloc[i - zone_lookback:i]
                current = tf.iloc[i]
                close_price = float(current['close'])

                demand_zone = detect_demand_zone(lookback_data)
                supply_zone = detect_supply_zone(lookback_data)

                in_demand = demand_zone is not None and price_in_zone(
                    close_price, demand_zone, tolerance=zone_touch_tolerance
                )
                in_supply = supply_zone is not None and price_in_zone(
                    close_price, supply_zone, tolerance=zone_touch_tolerance
                )

                if not in_demand and not in_supply:
                    continue

                future = tf.iloc[i + 1:i + 1 + horizon]
                if future.empty:
                    continue

                # BUY setup: return to demand -> reach next supply
                if in_demand and supply_zone is not None:
                    target_price = float(supply_zone['low']) * (1 - zone_touch_tolerance)
                    if float(future['high'].max()) >= target_price:
                        labels[i] = 1
                    else:
                        labels[i] = 0
                    continue

                # SELL setup: return to supply -> reach next demand
                if in_supply and demand_zone is not None:
                    target_price = float(demand_zone['high']) * (1 + zone_touch_tolerance)
                    if float(future['low'].min()) <= target_price:
                        labels[i] = -1
                    else:
                        labels[i] = 0

            tf['zone_label'] = labels
            labeled_dfs.append(tf)

        df = pd.concat(labeled_dfs, ignore_index=True)

        vc = df['zone_label'].value_counts().to_dict()
        print(
            "✓ Zone labels created: "
            f"buy={vc.get(1, 0)}, sell={vc.get(-1, 0)}, neutral={vc.get(0, 0)}"
        )

        return df

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        lag_periods: int = 5
    ) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.

        Args:
            df: DataFrame with features
            lag_periods: Number of lag periods to create

        Returns:
            DataFrame with lagged features
        """
        print(f"Creating lagged features (periods={lag_periods})...")

        df = df.copy()
        lag_columns = [
            # Core candle features
            'candle_size', 'body_size', 'wick_upper', 'wick_lower',
            # Engineered price/shape features
            'price_change_pct', 'body_to_range_ratio',
            'upper_wick_ratio', 'lower_wick_ratio', 'volume_normalized',
            # Time cyclical features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            # Strategy features (derived from past candles via apply_strategy)
            'strategy_signal_encoded',
        ]

        # Process each timeframe separately
        lagged_dfs = []

        for timeframe in df['timeframe'].unique():
            tf_data = df[df['timeframe'] == timeframe].copy()
            tf_data = tf_data.sort_values('timestamp')

            for col in lag_columns:
                if col in tf_data.columns:
                    for lag in range(1, lag_periods + 1):
                        tf_data[f'{col}_lag{lag}'] = tf_data[col].shift(lag)

            lagged_dfs.append(tf_data)

        df = pd.concat(lagged_dfs, ignore_index=True)

        # Drop rows with NaN from lagging
        initial_count = len(df)
        df = df.dropna()
        dropped = initial_count - len(df)

        print(f"[OK] Created lagged features (dropped {dropped} rows with NaN)")
        return df

    def prepare_features_and_target(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature matrix (X) and target vector (y).

        Args:
            df: Prepared DataFrame

        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        print("Preparing features and target...")

        # Define feature columns
        # IMPORTANT: To avoid leakage, we only use lagged features (t-1, t-2, ...).
        # No current-candle features are included in X.
        self.feature_columns = ['timeframe_encoded']

        # Add lagged feature columns if they exist
        lag_columns = [col for col in df.columns if "_lag" in col]
        self.feature_columns.extend(lag_columns)

        # Filter to only existing columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]

        # Create feature matrix
        X = df[self.feature_columns].copy()

        # Target: zone-to-zone outcome (binary for classifier): 1=buy success, 0=sell success
        # Neutral (0) is excluded before this function is called.
        if 'zone_label' not in df.columns:
            raise ValueError("Missing 'zone_label'. Run create_zone_outcome_labels() first.")
        y = (df['zone_label'] == 1).astype(int).copy()

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )

        print(f"[OK] Prepared {len(self.feature_columns)} features for {len(X)} samples")
        print(f"  Target distribution: buy={int(y.sum())}, sell={int(len(y) - y.sum())}")

        return X_scaled, y

    def _save_dataframe_with_fallback(self, df: pd.DataFrame, file_path: Path) -> Path:
        """Save a DataFrame to parquet if available, otherwise fall back to CSV."""
        try:
            df.to_parquet(file_path, index=False)
            return file_path
        except (ImportError, ValueError) as exc:
            csv_path = file_path.with_suffix('.csv')
            print(f"  - Parquet save failed ({exc}). Falling back to CSV: {csv_path.name}")
            df.to_csv(csv_path, index=False)
            return csv_path

    def save_timeframe_parquets(
        self,
        df: pd.DataFrame,
        output_dir: str = 'data'
    ) -> None:
        """
        Save each timeframe's prepared data into parquet files, with a CSV fallback.

        Args:
            df: Prepared DataFrame with a 'timeframe' column.
            output_dir: Relative output folder name.
        """
        project_root = Path(__file__).resolve().parent
        output_path = project_root / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving prepared data to: {output_path}")

        for timeframe in sorted(df['timeframe'].dropna().unique()):
            timeframe_safe = str(timeframe).replace('/', '_').replace(' ', '_')
            file_path = output_path / f"{timeframe_safe}.parquet"
            tf_data = df[df['timeframe'] == timeframe].copy()
            saved_path = self._save_dataframe_with_fallback(tf_data, file_path)
            print(f"  - Saved {len(tf_data)} rows for timeframe '{timeframe}' to {saved_path.name}")

        combined_file = output_path / "all_timeframes.parquet"
        saved_combined = self._save_dataframe_with_fallback(df, combined_file)
        print(f"  - Saved combined dataset to {saved_combined.name}")

    def prepare_data(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        lag_periods: int = 5,
        zone_lookback: int = 30,
        zone_horizon: int = 20,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Full data preparation pipeline.

        Args:
            timeframes: List of timeframes to include
            start_date: Start date filter
            end_date: End date filter
            lag_periods: Number of lag periods for features

        Returns:
            Tuple of (X features, y target, raw data for reference)
        """
        # Step 1: Fetch data from PostgreSQL
        raw_data = self.fetch_training_data(timeframes, start_date, end_date)

        if raw_data.empty:
            raise ValueError("No data fetched from database")

        # Step 2: Apply strategy signals
        data = self.apply_strategy_signals(raw_data)

        # Step 3: Engineer features
        data = self.engineer_features(data)

        # Step 4: Create forward-looking outcome labels (uses past zones + future reach)
        data = self.create_zone_outcome_labels(
            data,
            zone_lookback=zone_lookback,
            horizon=zone_horizon,
        )

        # Step 5: Create lagged features (ALL ML features are lagged; no current candle leakage)
        data = self.create_lagged_features(data, lag_periods)

        # Step 6: Keep only actionable samples (exclude neutral / unknown outcomes)
        initial_count = len(data)
        data = data[data['zone_label'] != 0].copy()
        print(f"[OK] Filtered neutral labels: {initial_count - len(data)} rows removed")

        # Save prepared timeframe data to parquet files
        self.save_timeframe_parquets(data)

        # Step 7: Prepare features and target
        X, y = self.prepare_features_and_target(data)

        return X, y, data

    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_columns

    def get_scaler(self) -> StandardScaler:
        """Return fitted scaler for prediction use."""
        return self.scaler

    def get_timeframe_encoder(self) -> LabelEncoder:
        """Return fitted timeframe encoder."""
        return self.timeframe_encoder


def prepare_data(
    timeframes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    lag_periods: int = 5,
    zone_lookback: int = 30,
    zone_horizon: int = 20,
) -> Tuple[pd.DataFrame, pd.Series, DataPreparator]:
    """
    Convenience function for data preparation.

    Returns:
        Tuple of (X features, y target, DataPreparator instance)
    """
    preparator = DataPreparator()
    X, y, _ = preparator.prepare_data(
        timeframes, start_date, end_date, lag_periods, zone_lookback, zone_horizon
    )
    return X, y, preparator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OHLCV data and save parquet files for specified timeframes.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5min"],
        help="Timeframes to prepare (e.g. 5min 10min 1h). Default: 10min"
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--lag-periods", type=int, default=5, help="Number of lag periods for features")

    args = parser.parse_args()

    print("=" * 60)
    print("DATA PREPARATION TEST")
    print("=" * 60)

    try:
        preparator = DataPreparator()

        X, y, raw_data = preparator.prepare_data(
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            lag_periods=args.lag_periods
        )

        print("\n" + "=" * 60)
        print("DATA PREPARATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(X)}")
        print(f"Features: {len(preparator.get_feature_columns())}")
        print(f"Feature columns: {preparator.get_feature_columns()[:10]}...")
        print(f"\nSample features (first 5 rows):")
        print(X.head())
        print(f"\nTarget distribution:")
        print(y.value_counts())

    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise
