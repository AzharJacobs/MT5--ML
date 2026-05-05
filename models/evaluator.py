"""
predict.py - Prediction Module
===============================

Loads saved ML model and makes predictions on live database data.
Supports various queries about direction patterns and predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from data.loader import get_connection, DatabaseConnection
from strategy.base_strategy import apply_strategy, LOOKBACK_PERIODS
from models.trainer import MODEL_DIR, MODEL_FILE, METADATA_FILE


class Predictor:
    """
    Handles loading saved model and making predictions on live data.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        """
        Initialize predictor by loading saved model.

        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        self.scaler = None
        self.timeframe_encoder = None
        self.feature_columns = None
        self.db: DatabaseConnection = get_connection()

        # Load model on initialization
        self._load_model()

    def _load_model(self) -> None:
        """Load the saved model and metadata."""
        model_path = os.path.join(self.model_dir, MODEL_FILE)
        metadata_path = os.path.join(self.model_dir, METADATA_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run train_model.py first to train and save a model."
            )

        # Load model
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")

        # Load metadata
        if os.path.exists(metadata_path):
            saved_metadata = joblib.load(metadata_path)
            self.metadata = saved_metadata.get('metadata', {})
            self.scaler = saved_metadata.get('scaler')
            self.timeframe_encoder = saved_metadata.get('timeframe_encoder')
            self.feature_columns = saved_metadata.get('feature_columns', [])
            print(f"✓ Metadata loaded from: {metadata_path}")
            print(f"  Model trained on: {self.metadata.get('training_date', 'unknown')}")
            print(f"  Training accuracy: {self.metadata.get('results', {}).get('train_accuracy', 'N/A'):.4f}")

    def predict_next_candle(
        self,
        timeframe: str,
        use_latest: bool = True,
        custom_data: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Predict the direction of the next candle.

        Args:
            timeframe: Timeframe to predict for
            use_latest: Whether to use latest data from database
            custom_data: Optional custom candle data for prediction

        Returns:
            Dictionary with prediction results
        """
        print(f"\nPredicting next candle direction for {timeframe}...")

        if use_latest:
            # Fetch latest candles from database for feature calculation
            if not self.db.connect():
                raise ConnectionError("Failed to connect to database")

            # Get last N candles for feature calculation
            lookback = LOOKBACK_PERIODS + 1
            query = """
                SELECT * FROM xauusd_ohlcv
                WHERE timeframe = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """
            latest_data = self.db.fetch_dataframe(query, (timeframe, lookback))

            if latest_data.empty:
                return {'error': f'No data found for timeframe {timeframe}'}

            # Reverse to chronological order
            latest_data = latest_data.iloc[::-1].reset_index(drop=True)
            current_candle = latest_data.iloc[-1]
            lookback_data = latest_data.iloc[:-1] if len(latest_data) > 1 else None
        else:
            current_candle = custom_data
            lookback_data = None

        # Apply strategy signal
        strategy_signal = apply_strategy(current_candle, lookback_data)

        # Prepare features
        features = self._prepare_features(current_candle, lookback_data, timeframe)

        # Make prediction
        prediction = self.model.predict(features)[0]
        prediction_proba = self.model.predict_proba(features)[0]

        direction = 'buy' if prediction == 1 else 'sell'
        confidence = max(prediction_proba)

        result = {
            'timeframe': timeframe,
            'prediction': direction,
            'confidence': float(confidence),
            'buy_probability': float(prediction_proba[1]),
            'sell_probability': float(prediction_proba[0]),
            'strategy_signal': strategy_signal,
            'latest_candle': {
                'timestamp': str(current_candle.get('timestamp', '')),
                'open': float(current_candle['open']),
                'high': float(current_candle['high']),
                'low': float(current_candle['low']),
                'close': float(current_candle['close']),
                'direction': current_candle.get('direction', 'unknown')
            },
            'prediction_time': datetime.now().isoformat()
        }

        print(f"\n{'=' * 40}")
        print(f"PREDICTION RESULT")
        print(f"{'=' * 40}")
        print(f"Predicted Direction: {direction.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Strategy Signal: {strategy_signal}")
        print(f"Buy Probability: {prediction_proba[1]:.2%}")
        print(f"Sell Probability: {prediction_proba[0]:.2%}")

        return result

    def _prepare_features(
        self,
        current_candle: pd.Series,
        lookback_data: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Prepare features for a single prediction.

        Args:
            current_candle: Current candle data
            lookback_data: Previous candles for lagged features
            timeframe: Timeframe string

        Returns:
            Feature DataFrame ready for prediction
        """
        # IMPORTANT (no lookahead):
        # The model is trained on lag-only features. For a prediction at time t, we use
        # candle features from t-1 (the most recent completed candle in lookback_data).
        if lookback_data is None or len(lookback_data) < 1:
            raise ValueError("Not enough lookback data to build lagged features (need >= 1 candle).")

        prev_candle = lookback_data.iloc[-1]

        candle_size = prev_candle.get('candle_size', prev_candle['high'] - prev_candle['low'])
        body_size = prev_candle.get('body_size', abs(prev_candle['close'] - prev_candle['open']))

        features = {
            # Non-lag features used in training
            'timeframe_encoded': 0,  # set below
        }

        # Timeframe encoding
        try:
            features['timeframe_encoded'] = self.timeframe_encoder.transform([timeframe])[0]
        except:
            features['timeframe_encoded'] = 0

        # Strategy signal (computed at t using current candle + past lookback), but we store it as lagged inputs
        # by sourcing it from the previous candle's perspective.
        signal_map = {'buy': 1, 'sell': -1, 'neutral': 0}

        # Build lagged features exactly like training: *_lag{lag}
        max_lag_needed = 0
        for col in self.feature_columns:
            if "_lag" in col:
                try:
                    lag_n = int(col.split("_lag")[-1])
                    max_lag_needed = max(max_lag_needed, lag_n)
                except Exception:
                    continue

        max_lag = max(1, min(max_lag_needed or LOOKBACK_PERIODS, len(lookback_data)))

        for lag in range(1, max_lag + 1):
            prev = lookback_data.iloc[-lag]
            prev_candle_size = prev.get('candle_size', prev['high'] - prev['low'])
            prev_body_size = prev.get('body_size', abs(prev['close'] - prev['open']))
            prev_open = float(prev['open']) if float(prev['open']) != 0 else 1.0

            # Candle/shape
            features[f'candle_size_lag{lag}'] = float(prev_candle_size)
            features[f'body_size_lag{lag}'] = float(prev_body_size)
            features[f'wick_upper_lag{lag}'] = float(prev.get('wick_upper', 0))
            features[f'wick_lower_lag{lag}'] = float(prev.get('wick_lower', 0))

            # Price/ratios
            features[f'price_change_pct_lag{lag}'] = float((prev['close'] - prev['open']) / prev_open * 100)
            features[f'body_to_range_ratio_lag{lag}'] = float(prev_body_size / prev_candle_size) if prev_candle_size > 0 else 0.0
            features[f'upper_wick_ratio_lag{lag}'] = float(prev.get('wick_upper', 0) / prev_candle_size) if prev_candle_size > 0 else 0.0
            features[f'lower_wick_ratio_lag{lag}'] = float(prev.get('wick_lower', 0) / prev_candle_size) if prev_candle_size > 0 else 0.0

            # Volume normalization using history strictly before that candle
            hist = lookback_data.iloc[:len(lookback_data) - lag]
            vol_norm = 0.0
            if hist is not None and len(hist) > 10:
                vol_mean = float(hist['volume'].mean())
                vol_std = float(hist['volume'].std())
                if vol_std > 0:
                    vol_norm = float((prev.get('volume', 0) - vol_mean) / vol_std)
            features[f'volume_normalized_lag{lag}'] = vol_norm

            # Time cyclical (based on candle timestamp fields, lagged)
            features[f'hour_sin_lag{lag}'] = float(np.sin(2 * np.pi * prev.get('hour', 0) / 24))
            features[f'hour_cos_lag{lag}'] = float(np.cos(2 * np.pi * prev.get('hour', 0) / 24))
            features[f'day_sin_lag{lag}'] = float(np.sin(2 * np.pi * prev.get('day_of_week', 0) / 7))
            features[f'day_cos_lag{lag}'] = float(np.cos(2 * np.pi * prev.get('day_of_week', 0) / 7))
            features[f'month_sin_lag{lag}'] = float(np.sin(2 * np.pi * prev.get('month', 1) / 12))
            features[f'month_cos_lag{lag}'] = float(np.cos(2 * np.pi * prev.get('month', 1) / 12))

            # Strategy signal encoded, lagged (compute using the candle at that lag as "current")
            # and candles before it as lookback.
            prev_lb = lookback_data.iloc[:len(lookback_data) - lag]
            sig = apply_strategy(prev, prev_lb if len(prev_lb) > 0 else None)
            features[f'strategy_signal_encoded_lag{lag}'] = signal_map.get(sig, 0)

        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])

        # Ensure all required columns exist (fill missing with 0)
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0

        # Select only the columns the model expects
        feature_df = feature_df[self.feature_columns]

        # Scale features
        if self.scaler is not None:
            feature_df = pd.DataFrame(
                self.scaler.transform(feature_df),
                columns=self.feature_columns
            )

        return feature_df

    def get_direction_counts(
        self,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        group_by: str = 'day'
    ) -> pd.DataFrame:
        """
        Get buy/sell counts for analysis.

        Args:
            timeframe: Timeframe to analyze
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            group_by: How to group counts ('day', 'hour', 'day_of_week', 'month')

        Returns:
            DataFrame with direction counts
        """
        if not self.db.connect():
            raise ConnectionError("Failed to connect to database")

        if group_by == 'day':
            return self.db.get_direction_counts_by_day(timeframe, start_date, end_date)
        elif group_by == 'hour':
            return self.db.get_direction_counts_by_hour(timeframe, start_date, end_date)
        elif group_by == 'day_of_week':
            return self.db.get_direction_counts_by_day_of_week(timeframe, start_date, end_date)
        elif group_by == 'month':
            return self.db.get_direction_counts_by_month(timeframe)
        else:
            raise ValueError(f"Invalid group_by value: {group_by}")

    def query_buy_candles_per_day(
        self,
        timeframe: str,
        month: int,
        year: int
    ) -> pd.DataFrame:
        """
        Query: How many buy candles per day in a specific month?

        Example: "In February 2025 on 1min timeframe, how many buy candles per day?"

        Args:
            timeframe: Timeframe to query
            month: Month number (1-12)
            year: Year

        Returns:
            DataFrame with date and buy_count
        """
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"

        # Adjust end date to last day of month
        end_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"\nQuerying buy candles per day for {timeframe} in {month}/{year}...")

        counts = self.get_direction_counts(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            group_by='day'
        )

        if not counts.empty:
            print(f"\nResults:")
            print(counts[['date', 'buy_count', 'sell_count']].to_string(index=False))
            print(f"\nTotal buy candles: {counts['buy_count'].sum()}")
            print(f"Total sell candles: {counts['sell_count'].sum()}")
            print(f"Average buy candles per day: {counts['buy_count'].mean():.1f}")

        return counts

    def query_best_trading_hours(
        self,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        direction: str = 'buy'
    ) -> pd.DataFrame:
        """
        Query: Which hours have the most buy/sell activity?

        Args:
            timeframe: Timeframe to analyze
            start_date: Optional start date
            end_date: Optional end date
            direction: 'buy' or 'sell'

        Returns:
            DataFrame with hours ranked by activity
        """
        print(f"\nFinding best {direction} hours for {timeframe}...")

        counts = self.get_direction_counts(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            group_by='hour'
        )

        if not counts.empty:
            count_col = 'buy_count' if direction == 'buy' else 'sell_count'
            counts = counts.sort_values(count_col, ascending=False)

            print(f"\nHours ranked by {direction} activity:")
            for _, row in counts.head(10).iterrows():
                hour = int(row['hour'])
                count = int(row[count_col])
                print(f"  {hour:02d}:00 - {count} {direction}s")

        return counts

    def query_best_trading_days(
        self,
        timeframe: str,
        start_date: str = None,
        end_date: str = None,
        direction: str = 'buy'
    ) -> pd.DataFrame:
        """
        Query: Which days of the week have the most buy/sell activity?

        Args:
            timeframe: Timeframe to analyze
            start_date: Optional start date
            end_date: Optional end date
            direction: 'buy' or 'sell'

        Returns:
            DataFrame with days ranked by activity
        """
        print(f"\nFinding best {direction} days for {timeframe}...")

        counts = self.get_direction_counts(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            group_by='day_of_week'
        )

        if not counts.empty:
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            count_col = 'buy_count' if direction == 'buy' else 'sell_count'
            counts = counts.sort_values(count_col, ascending=False)

            print(f"\nDays ranked by {direction} activity:")
            for _, row in counts.iterrows():
                day_idx = int(row['day_of_week'])
                day_name = day_names[day_idx] if day_idx < len(day_names) else f"Day {day_idx}"
                count = int(row[count_col])
                print(f"  {day_name}: {count} {direction}s")

        return counts

    def query_monthly_patterns(
        self,
        timeframe: str,
        year: int = None
    ) -> pd.DataFrame:
        """
        Query: Which months have the most buy/sell activity?

        Args:
            timeframe: Timeframe to analyze
            year: Optional specific year

        Returns:
            DataFrame with monthly activity
        """
        print(f"\nAnalyzing monthly patterns for {timeframe}...")

        counts = self.get_direction_counts(
            timeframe=timeframe,
            group_by='month'
        )

        if year is not None:
            counts = counts[counts['year'] == year]

        if not counts.empty:
            month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            print(f"\nMonthly activity:")
            for _, row in counts.iterrows():
                month_idx = int(row['month'])
                month_name = month_names[month_idx] if month_idx < len(month_names) else f"Month {month_idx}"
                year_val = int(row['year'])
                buy_count = int(row['buy_count'])
                sell_count = int(row['sell_count'])
                print(f"  {month_name} {year_val}: {buy_count} buys, {sell_count} sells")

        return counts


def predict(timeframe: str = '1min') -> Dict[str, Any]:
    """
    Convenience function for making a prediction.

    Args:
        timeframe: Timeframe to predict for

    Returns:
        Prediction result dictionary
    """
    predictor = Predictor()
    return predictor.predict_next_candle(timeframe)


def query(
    query_type: str,
    timeframe: str,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function for running queries.

    Args:
        query_type: Type of query ('buy_per_day', 'best_hours', 'best_days', 'monthly')
        timeframe: Timeframe to query
        **kwargs: Additional arguments for the query

    Returns:
        Query results DataFrame
    """
    predictor = Predictor()

    if query_type == 'buy_per_day':
        return predictor.query_buy_candles_per_day(
            timeframe=timeframe,
            month=kwargs.get('month', datetime.now().month),
            year=kwargs.get('year', datetime.now().year)
        )
    elif query_type == 'best_hours':
        return predictor.query_best_trading_hours(
            timeframe=timeframe,
            direction=kwargs.get('direction', 'buy')
        )
    elif query_type == 'best_days':
        return predictor.query_best_trading_days(
            timeframe=timeframe,
            direction=kwargs.get('direction', 'buy')
        )
    elif query_type == 'monthly':
        return predictor.query_monthly_patterns(
            timeframe=timeframe,
            year=kwargs.get('year')
        )
    else:
        raise ValueError(f"Unknown query type: {query_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTION MODULE TEST")
    print("=" * 60)

    try:
        # Initialize predictor
        predictor = Predictor()

        # Test prediction
        print("\n--- Testing Prediction ---")
        result = predictor.predict_next_candle('1min')

        # Test queries
        print("\n--- Testing Query: Buy Candles Per Day (Feb 2025) ---")
        predictor.query_buy_candles_per_day('1min', month=2, year=2025)

        print("\n--- Testing Query: Best Trading Hours ---")
        predictor.query_best_trading_hours('1min', direction='buy')

        print("\n--- Testing Query: Best Trading Days ---")
        predictor.query_best_trading_days('1min', direction='buy')

        print("\n--- Testing Query: Monthly Patterns ---")
        predictor.query_monthly_patterns('1min', year=2025)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run train_model.py first to train and save a model.")
    except Exception as e:
        print(f"\nError: {e}")
        raise
