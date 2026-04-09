"""
strategy.py - Personal Trading Strategy Rules
============================================

This file is where you define YOUR personal trading strategy rules.
The ML engine will apply these rules to historical data and learn from them.

HOW TO ADD YOUR STRATEGY:
1. Modify the `apply_strategy()` function below
2. Use the available candle data to define your buy/sell conditions
3. Return 'buy', 'sell', or 'neutral' based on your rules
4. The ML model will train on these strategy signals

AVAILABLE DATA FOR EACH CANDLE (passed as a pandas Series):
- open: Opening price
- high: Highest price
- low: Lowest price
- close: Closing price
- volume: Trading volume
- direction: Actual candle direction ('buy' or 'sell')
- candle_size: Total size of candle (high - low)
- body_size: Size of candle body (abs(close - open))
- wick_upper: Upper wick size
- wick_lower: Lower wick size
- hour: Hour of the day (0-23)
- day_of_week: Day of week (0=Monday, 6=Sunday)
- month: Month (1-12)
- year: Year

LOOKBACK DATA (passed as a DataFrame):
- Previous N candles for pattern recognition
- Use lookback_data.iloc[-1] for most recent previous candle
- Use lookback_data.iloc[-2] for second most recent, etc.

EXAMPLE STRATEGIES ARE PROVIDED BELOW - MODIFY OR REPLACE THEM WITH YOUR OWN!
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

# Number of previous candles to consider for pattern recognition
LOOKBACK_PERIODS = 5

# Strategy name (for logging purposes)
STRATEGY_NAME = "Custom Trading Strategy"

# Enable/disable specific strategy components
USE_CANDLESTICK_PATTERNS = True
USE_TIME_FILTERS = True
USE_MOMENTUM_RULES = True
USE_VOLUME_FILTERS = True


# =============================================================================
# MAIN STRATEGY FUNCTION - MODIFY THIS!
# =============================================================================

def apply_strategy(
    current_candle: pd.Series,
    lookback_data: pd.DataFrame = None
) -> str:
    """
    Apply your personal trading strategy to determine buy/sell signal.

    THIS IS THE MAIN FUNCTION TO MODIFY WITH YOUR TRADING RULES!

    Args:
        current_candle: pandas Series containing current candle data
        lookback_data: DataFrame containing previous N candles (oldest to newest)

    Returns:
        'buy' - Strategy signals a buy
        'sell' - Strategy signals a sell
        'neutral' - No clear signal
    """

    # Initialize signal
    signal = 'neutral'

    # ==========================================================================
    # EXAMPLE STRATEGY RULES - REPLACE WITH YOUR OWN!
    # ==========================================================================

    # Extract current candle data
    open_price = current_candle['open']
    high_price = current_candle['high']
    low_price = current_candle['low']
    close_price = current_candle['close']
    volume = current_candle.get('volume', 0)
    candle_size = current_candle.get('candle_size', high_price - low_price)
    body_size = current_candle.get('body_size', abs(close_price - open_price))
    wick_upper = current_candle.get('wick_upper', 0)
    wick_lower = current_candle.get('wick_lower', 0)
    hour = current_candle.get('hour', 0)
    day_of_week = current_candle.get('day_of_week', 0)

    # --------------------------------------------------------------------------
    # RULE 1: Candlestick Pattern Recognition
    # --------------------------------------------------------------------------
    if USE_CANDLESTICK_PATTERNS and candle_size > 0:
        body_ratio = body_size / candle_size if candle_size > 0 else 0

        # Bullish Engulfing Pattern Check
        if lookback_data is not None and len(lookback_data) >= 1:
            prev_candle = lookback_data.iloc[-1]
            prev_close = prev_candle['close']
            prev_open = prev_candle['open']

            # Bullish engulfing: previous was bearish, current is bullish and engulfs
            if (prev_close < prev_open and  # Previous was bearish
                close_price > open_price and  # Current is bullish
                open_price < prev_close and  # Opens below previous close
                close_price > prev_open):  # Closes above previous open
                signal = 'buy'

            # Bearish engulfing: previous was bullish, current is bearish and engulfs
            elif (prev_close > prev_open and  # Previous was bullish
                  close_price < open_price and  # Current is bearish
                  open_price > prev_close and  # Opens above previous close
                  close_price < prev_open):  # Closes below previous open
                signal = 'sell'

        # Hammer Pattern (bullish reversal)
        if signal == 'neutral':
            if (wick_lower > body_size * 2 and  # Long lower wick
                wick_upper < body_size * 0.5 and  # Small upper wick
                body_ratio < 0.4):  # Small body
                signal = 'buy'

        # Shooting Star Pattern (bearish reversal)
        if signal == 'neutral':
            if (wick_upper > body_size * 2 and  # Long upper wick
                wick_lower < body_size * 0.5 and  # Small lower wick
                body_ratio < 0.4):  # Small body
                signal = 'sell'

    # --------------------------------------------------------------------------
    # RULE 2: Time-Based Filters
    # --------------------------------------------------------------------------
    if USE_TIME_FILTERS and signal != 'neutral':
        # Avoid trading during low-activity hours (adjust to your preference)
        low_activity_hours = [0, 1, 2, 3, 4, 5, 22, 23]  # Late night/early morning
        if hour in low_activity_hours:
            signal = 'neutral'  # Override signal during low activity

        # Weekend filter (if your data includes weekends)
        if day_of_week in [5, 6]:  # Saturday, Sunday
            signal = 'neutral'

    # --------------------------------------------------------------------------
    # RULE 3: Momentum Rules
    # --------------------------------------------------------------------------
    if USE_MOMENTUM_RULES and lookback_data is not None and len(lookback_data) >= 3:
        # Check for momentum (3 consecutive candles in same direction)
        recent_candles = lookback_data.tail(3)

        if signal == 'neutral':
            # Bullish momentum: 3 consecutive bullish candles
            bullish_count = sum(1 for _, c in recent_candles.iterrows()
                              if c['close'] > c['open'])
            if bullish_count == 3 and close_price > open_price:
                signal = 'buy'

            # Bearish momentum: 3 consecutive bearish candles
            bearish_count = sum(1 for _, c in recent_candles.iterrows()
                              if c['close'] < c['open'])
            if bearish_count == 3 and close_price < open_price:
                signal = 'sell'

    # --------------------------------------------------------------------------
    # RULE 4: Volume Confirmation
    # --------------------------------------------------------------------------
    if USE_VOLUME_FILTERS and lookback_data is not None and len(lookback_data) >= 5:
        avg_volume = lookback_data['volume'].mean()

        # Require above-average volume for signal confirmation
        if signal != 'neutral' and volume < avg_volume * 0.5:
            # Low volume - reduce confidence in signal
            pass  # Keep signal but you could set to 'neutral' here

    return signal


# =============================================================================
# ADDITIONAL STRATEGY HELPER FUNCTIONS
# =============================================================================

def calculate_sma(data: pd.DataFrame, column: str, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data[column].rolling(window=period).mean()


def calculate_ema(data: pd.DataFrame, column: str, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_doji(candle: pd.Series, threshold: float = 0.1) -> bool:
    """Detect if candle is a Doji (small body relative to range)."""
    if candle['candle_size'] == 0:
        return False
    body_ratio = candle['body_size'] / candle['candle_size']
    return body_ratio < threshold


def detect_marubozu(candle: pd.Series, threshold: float = 0.95) -> str:
    """Detect Marubozu candle (full body, no wicks)."""
    if candle['candle_size'] == 0:
        return 'none'
    body_ratio = candle['body_size'] / candle['candle_size']
    if body_ratio >= threshold:
        return 'bullish' if candle['close'] > candle['open'] else 'bearish'
    return 'none'


def is_trend_up(lookback_data: pd.DataFrame, periods: int = 5) -> bool:
    """Check if recent trend is upward."""
    if lookback_data is None or len(lookback_data) < periods:
        return False
    recent = lookback_data.tail(periods)
    return recent['close'].iloc[-1] > recent['close'].iloc[0]


def is_trend_down(lookback_data: pd.DataFrame, periods: int = 5) -> bool:
    """Check if recent trend is downward."""
    if lookback_data is None or len(lookback_data) < periods:
        return False
    recent = lookback_data.tail(periods)
    return recent['close'].iloc[-1] < recent['close'].iloc[0]


# =============================================================================
# STRATEGY VALIDATION
# =============================================================================

def validate_strategy() -> bool:
    """
    Validate that the strategy is properly configured.
    Run this before training to ensure strategy is working.
    """
    print(f"Strategy Name: {STRATEGY_NAME}")
    print(f"Lookback Periods: {LOOKBACK_PERIODS}")
    print(f"Candlestick Patterns: {'Enabled' if USE_CANDLESTICK_PATTERNS else 'Disabled'}")
    print(f"Time Filters: {'Enabled' if USE_TIME_FILTERS else 'Disabled'}")
    print(f"Momentum Rules: {'Enabled' if USE_MOMENTUM_RULES else 'Disabled'}")
    print(f"Volume Filters: {'Enabled' if USE_VOLUME_FILTERS else 'Disabled'}")

    # Test with dummy data
    test_candle = pd.Series({
        'open': 100.0,
        'high': 105.0,
        'low': 99.0,
        'close': 104.0,
        'volume': 1000,
        'candle_size': 6.0,
        'body_size': 4.0,
        'wick_upper': 1.0,
        'wick_lower': 1.0,
        'hour': 10,
        'day_of_week': 2,
        'month': 6
    })

    try:
        result = apply_strategy(test_candle, None)
        if result in ['buy', 'sell', 'neutral']:
            print(f"✓ Strategy validation passed (test signal: {result})")
            return True
        else:
            print(f"✗ Strategy returned invalid signal: {result}")
            return False
    except Exception as e:
        print(f"✗ Strategy validation failed: {e}")
        return False


if __name__ == "__main__":
    validate_strategy()
