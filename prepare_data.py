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
from strategy import apply_strategy, LOOKBACK_PERIODS


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

        print(f"✓ Fetched {len(df)} records from database")
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
        print(f"✓ Strategy signals applied:")
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

        print(f"✓ Engineered {len(df.columns)} features")
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
            'price_change_pct', 'body_to_range_ratio',
            'volume_normalized', 'is_bullish', 'strategy_signal_encoded'
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

        print(f"✓ Created lagged features (dropped {dropped} rows with NaN)")
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
        self.feature_columns = [
            # Core OHLCV features
            'candle_size', 'body_size', 'wick_upper', 'wick_lower',
            # Engineered features
            'price_change_pct', 'body_to_range_ratio',
            'upper_wick_ratio', 'lower_wick_ratio', 'volume_normalized',
            # Time features
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos',
            # Timeframe encoding
            'timeframe_encoded',
            # Strategy signal
            'strategy_signal_encoded'
        ]

        # Add lagged feature columns if they exist
        lag_columns = [col for col in df.columns if '_lag' in col]
        self.feature_columns.extend(lag_columns)

        # Filter to only existing columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]

        # Create feature matrix
        X = df[self.feature_columns].copy()

        # Target: predict next candle direction (0 = sell, 1 = buy)
        y = df['is_bullish'].copy()

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )

        print(f"✓ Prepared {len(self.feature_columns)} features for {len(X)} samples")
        print(f"  Target distribution: buy={y.sum()}, sell={len(y) - y.sum()}")

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
        lag_periods: int = 5
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

        # Step 4: Create lagged features
        data = self.create_lagged_features(data, lag_periods)

        # Save prepared timeframe data to parquet files
        self.save_timeframe_parquets(data)

        # Step 5: Prepare features and target
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
    lag_periods: int = 5
) -> Tuple[pd.DataFrame, pd.Series, DataPreparator]:
    """
    Convenience function for data preparation.

    Returns:
        Tuple of (X features, y target, DataPreparator instance)
    """
    preparator = DataPreparator()
    X, y, _ = preparator.prepare_data(timeframes, start_date, end_date, lag_periods)
    return X, y, preparator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OHLCV data and save parquet files for specified timeframes.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["10min"],
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
