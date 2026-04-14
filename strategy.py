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
 
# Number of previous candles to consider for zone detection and pattern recognition
LOOKBACK_PERIODS = 20
 
# Strategy name (for logging purposes)
STRATEGY_NAME = "Zone-to-Zone Trading Strategy"
 
# Enable/disable specific strategy components
USE_ZONE_DETECTION = True        # Core: identify supply/demand zones
USE_CONFIRMATION_PATTERNS = True # Look for engulfing, pin bars, rejection wicks
USE_TIME_FILTERS = True          # Avoid low-activity hours
USE_ZONE_FRESHNESS = True        # Prefer zones that haven't been touched before
USE_MIDLINE_TP = True            # Use 50% midpoint between zones as take-profit
 
# Zone detection thresholds
STRONG_MOVE_THRESHOLD = 0.0010    # Minimum % move to qualify as strong departure (0.3%)
ZONE_BODY_RATIO_MIN = 0.50       # Minimum body/candle ratio for a strong zone candle
ZONE_TOUCH_TOLERANCE = 0.001     # How close price needs to be to zone boundary (0.1%)
MAX_ZONE_AGE = 60                # Discard zones older than this many candles
MIN_ZONE_CANDLES = 2             # Minimum consecutive strong candles to form a zone
 
 
# =============================================================================
# ZONE DETECTION HELPERS
# =============================================================================
 
def detect_demand_zone(lookback_data: pd.DataFrame) -> dict | None:
    """
    Detect the most recent valid demand zone from lookback data.

    A demand zone is identified by:
    - Strong bullish candles (long bodies, sharp move up)
    - Break of structure upward after the zone
    - Sometimes a V-shape reversal

    Returns a dict with zone 'high' and 'low' price levels, or None.
    """
    if lookback_data is None or len(lookback_data) < MIN_ZONE_CANDLES + 1:
        return None
    
    demand_zones = []
    
    # Scan from oldest to newest, looking for strong bullish moves
    for i in range(len(lookback_data) - 1):
        current = lookback_data.iloc[i]
        
        # Check if this is a strong bullish candle
        if current['close'] <= current['open']:  # Not bullish
            continue
        
        body_size = current['close'] - current['open']
        candle_size = current['high'] - current['low']
        
        # Check if body is substantial relative to candle
        if candle_size == 0 or (body_size / candle_size) < ZONE_BODY_RATIO_MIN:
            continue
        
        # Check if this is a strong move (at least STRONG_MOVE_THRESHOLD% up)
        if body_size / current['open'] < STRONG_MOVE_THRESHOLD:
            continue
        
        # Look ahead to confirm break of structure upward
        consecutive_strength = 1
        zone_low = current['low']
        zone_high = current['high']
        
        # Check next candles for continuation or higher lows (demand zone characteristics)
        for j in range(i + 1, min(i + MIN_ZONE_CANDLES + 1, len(lookback_data))):
            next_candle = lookback_data.iloc[j]
            
            # Expand zone bounds
            zone_low = min(zone_low, next_candle['low'])
            zone_high = max(zone_high, next_candle['high'])
            
            # Count consecutive bullish or consolidating candles
            if next_candle['close'] >= next_candle['open']:
                consecutive_strength += 1
            elif (next_candle['high'] - next_candle['low']) < (current['candle_size'] * 0.5):
                # Small consolidation candle is acceptable
                consecutive_strength += 0.5
        
        # Valid demand zone if we have enough consecutive strength
        if consecutive_strength >= MIN_ZONE_CANDLES:
            age = len(lookback_data) - i - 1
            
            if age <= MAX_ZONE_AGE:
                demand_zones.append({
                    'low': zone_low,
                    'high': zone_high,
                    'age': age,
                    'strength': consecutive_strength,
                    'index': i
                })
    
    # Return the freshest (most recent) demand zone
    if demand_zones:
        # Sort by age (freshest first) and then by strength
        demand_zones.sort(key=lambda x: (x['age'], -x['strength']))
        return {
            'low': demand_zones[0]['low'],
            'high': demand_zones[0]['high'],
            'age': demand_zones[0]['age'],
            'strength': demand_zones[0]['strength']
        }
    
    return None
 
 
def detect_supply_zone(lookback_data: pd.DataFrame) -> dict | None:
    """
    Detect the most recent valid supply zone from lookback data.

    A supply zone is identified by:
    - Strong bearish candles (long bodies, sharp move down)
    - Break of structure downward after the zone
    - Clear rejection wick or engulfing pattern

    Returns a dict with zone 'high' and 'low' price levels, or None.
    """
    # Use the same thresholds as demand zone detection.
    supply_strong_move_threshold = STRONG_MOVE_THRESHOLD
    supply_min_zone_candles = MIN_ZONE_CANDLES

    if lookback_data is None or len(lookback_data) < supply_min_zone_candles + 1:
        return None
    
    supply_zones = []
    
    # Scan from oldest to newest, looking for strong bearish moves
    for i in range(len(lookback_data) - 1):
        current = lookback_data.iloc[i]
        
        # Check if this is a strong bearish candle
        if current['close'] >= current['open']:  # Not bearish
            continue
        
        body_size = current['open'] - current['close']
        candle_size = current['high'] - current['low']
        
        # Check if body is substantial relative to candle
        if candle_size == 0 or (body_size / candle_size) < ZONE_BODY_RATIO_MIN:
            continue
        
        # Check if this is a strong move (at least supply_strong_move_threshold% down)
        if body_size / current['open'] < supply_strong_move_threshold:
            continue
        
        # Look ahead to confirm break of structure downward
        consecutive_strength = 1
        zone_low = current['low']
        zone_high = current['high']
        
        # Check next candles for continuation or lower highs (supply zone characteristics)
        for j in range(i + 1, min(i + supply_min_zone_candles + 1, len(lookback_data))):
            next_candle = lookback_data.iloc[j]
            
            # Expand zone bounds
            zone_low = min(zone_low, next_candle['low'])
            zone_high = max(zone_high, next_candle['high'])
            
            # Count consecutive bearish or consolidating candles
            if next_candle['close'] <= next_candle['open']:
                consecutive_strength += 1
            elif (next_candle['high'] - next_candle['low']) < (current['candle_size'] * 0.5):
                # Small consolidation candle is acceptable
                consecutive_strength += 0.5
        
        # Valid supply zone if we have enough consecutive strength
        if consecutive_strength >= supply_min_zone_candles:
            age = len(lookback_data) - i - 1
            
            if age <= MAX_ZONE_AGE:
                supply_zones.append({
                    'low': zone_low,
                    'high': zone_high,
                    'age': age,
                    'strength': consecutive_strength,
                    'index': i
                })
    
    # Return the freshest (most recent) supply zone
    if supply_zones:
        # Sort by age (freshest first) and then by strength
        supply_zones.sort(key=lambda x: (x['age'], -x['strength']))
        return {
            'low': supply_zones[0]['low'],
            'high': supply_zones[0]['high'],
            'age': supply_zones[0]['age'],
            'strength': supply_zones[0]['strength']
        }
    
    return None
 
 
def price_in_zone(price: float, zone: dict, tolerance: float = ZONE_TOUCH_TOLERANCE) -> bool:
    """Check if current price is within or touching a zone (with tolerance buffer)."""
    zone_low  = zone['low']  * (1 - tolerance)
    zone_high = zone['high'] * (1 + tolerance)
    return zone_low <= price <= zone_high
 
 
def zone_is_fresh(zone: dict, fresh_age: int = 20) -> bool:
    """
    Zones that haven't been touched before are more reliable.
    Uses zone age as a proxy — fresher zones score higher.
    Age <= 20 candles = fresh. Zones up to 50 are detected but only
    the freshest ones (<=20) are traded.
    """
    return zone['age'] <= fresh_age
 
 
# =============================================================================
# CONFIRMATION PATTERN HELPERS
# =============================================================================
 
def is_bullish_engulfing(current_candle: pd.Series, prev_candle: pd.Series) -> bool:
    """Previous bearish, current bullish and fully engulfs previous body."""
    return (
        prev_candle['close'] < prev_candle['open'] and
        current_candle['close'] > current_candle['open'] and
        current_candle['open'] < prev_candle['close'] and
        current_candle['close'] > prev_candle['open']
    )
 
 
def is_bearish_engulfing(current_candle: pd.Series, prev_candle: pd.Series) -> bool:
    """Previous bullish, current bearish and fully engulfs previous body."""
    return (
        prev_candle['close'] > prev_candle['open'] and
        current_candle['close'] < current_candle['open'] and
        current_candle['open'] > prev_candle['close'] and
        current_candle['close'] < prev_candle['open']
    )
 
 
def is_pin_bar_bullish(candle: pd.Series) -> bool:
    """
    Bullish pin bar / hammer: long lower wick, small body, small upper wick.
    Signals rejection of lower prices — buy confirmation in demand zone.
    """
    candle_size = candle.get('candle_size', candle['high'] - candle['low'])
    body_size   = candle.get('body_size', abs(candle['close'] - candle['open']))
    wick_lower  = candle.get('wick_lower', 0)
    wick_upper  = candle.get('wick_upper', 0)
 
    if candle_size == 0 or body_size == 0:
        return False
 
    return (
        wick_lower > body_size * 2 and
        wick_upper < body_size * 0.5 and
        body_size / candle_size < 0.4
    )
 
 
def is_pin_bar_bearish(candle: pd.Series) -> bool:
    """
    Bearish pin bar / shooting star: long upper wick, small body, small lower wick.
    Signals rejection of higher prices — sell confirmation in supply zone.
    """
    candle_size = candle.get('candle_size', candle['high'] - candle['low'])
    body_size   = candle.get('body_size', abs(candle['close'] - candle['open']))
    wick_lower  = candle.get('wick_lower', 0)
    wick_upper  = candle.get('wick_upper', 0)
 
    if candle_size == 0 or body_size == 0:
        return False
 
    return (
        wick_upper > body_size * 2 and
        wick_lower < body_size * 0.5 and
        body_size / candle_size < 0.4
    )
 
 
def is_higher_low(lookback_data: pd.DataFrame, periods: int = 3) -> bool:
    """Recent lows forming a higher low — bullish market structure."""
    if lookback_data is None or len(lookback_data) < periods:
        return False
    lows = lookback_data.tail(periods)['low'].values
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
 
 
def is_lower_high(lookback_data: pd.DataFrame, periods: int = 3) -> bool:
    """Recent highs forming a lower high — bearish market structure."""
    if lookback_data is None or len(lookback_data) < periods:
        return False
    highs = lookback_data.tail(periods)['high'].values
    return all(highs[i] < highs[i - 1] for i in range(1, len(highs)))
 
 
# =============================================================================
# MAIN STRATEGY FUNCTION — ZONE-TO-ZONE
# =============================================================================
 
def apply_strategy(
    current_candle: pd.Series,
    lookback_data: pd.DataFrame = None
) -> str:
    """
    Apply the Zone-to-Zone Trading Strategy to determine buy/sell signal.
 
    Core logic (5 steps from the strategy document):
      1. Identify strong supply and demand zones from historical candles.
      2. Wait for price to RETURN to a zone — do not chase mid-range moves.
      3. Look for confirmation inside the zone (engulfing, pin bar, rejection wick,
         higher low / lower high structure).
      4. Enter in the direction of the zone; target the NEXT opposite zone.
      5. Never trade in no-man's land (between zones).
 
    Args:
        current_candle: pandas Series with current candle OHLCV data.
        lookback_data:  DataFrame of previous N candles (oldest → newest).
 
    Returns:
        'buy'     — Price is in a demand zone with bullish confirmation.
        'sell'    — Price is in a supply zone with bearish confirmation.
        'neutral' — Price is between zones, no confirmation, or time-filtered.
    """
 
    signal = 'neutral'
 
    if lookback_data is None or len(lookback_data) < LOOKBACK_PERIODS:
        return signal
 
    close_price = current_candle['close']
    open_price  = current_candle['open']
    hour        = current_candle.get('hour', 12)
    day_of_week = current_candle.get('day_of_week', 1)
 
    # --------------------------------------------------------------------------
    # TIME FILTER — avoid low-activity sessions
    # --------------------------------------------------------------------------
    if USE_TIME_FILTERS:
        low_activity_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        if hour in low_activity_hours or day_of_week in [5, 6]:
            return 'neutral'
 
    # --------------------------------------------------------------------------
    # STEP 1 — Identify supply and demand zones
    # --------------------------------------------------------------------------
    if not USE_ZONE_DETECTION:
        return signal
 
    demand_zone = detect_demand_zone(lookback_data)
    supply_zone = detect_supply_zone(lookback_data)
 
    # --------------------------------------------------------------------------
    # STEP 2 — Check if price has RETURNED to a zone (never trade mid-range)
    # --------------------------------------------------------------------------
    in_demand_zone = demand_zone is not None and price_in_zone(close_price, demand_zone)
    in_supply_zone = supply_zone is not None and price_in_zone(close_price, supply_zone)
 
    if not in_demand_zone and not in_supply_zone:
        return 'neutral'  # Price is between zones — no trade
 
    # --------------------------------------------------------------------------
    # ZONE FRESHNESS — fresher zones are more reliable
    # --------------------------------------------------------------------------
    if USE_ZONE_FRESHNESS:
        if in_demand_zone and not zone_is_fresh(demand_zone):
            in_demand_zone = False
        if in_supply_zone and not zone_is_fresh(supply_zone):
            in_supply_zone = False
 
    if not in_demand_zone and not in_supply_zone:
        return 'neutral'
 
    # --------------------------------------------------------------------------
    # STEP 3 — Look for confirmation inside the zone
    # --------------------------------------------------------------------------
    prev_candle = lookback_data.iloc[-1] if len(lookback_data) >= 1 else None
 
    if USE_CONFIRMATION_PATTERNS and prev_candle is not None:
 
        # BUY confirmation inside demand zone
        buy_score = 0
        sell_score = 0
        buy_confirmed = False
        sell_confirmed = False

        if in_demand_zone:
            bullish_engulf = is_bullish_engulfing(current_candle, prev_candle)
            bullish_pin = is_pin_bar_bullish(current_candle)
            higher_low = is_higher_low(lookback_data, periods=3)
            current_bullish = close_price > open_price

            buy_confirmed = bool(bullish_engulf or bullish_pin or (higher_low and current_bullish))
            buy_score = int(bullish_engulf) + int(bullish_pin) + int(higher_low and current_bullish)

        if in_supply_zone:
            bearish_engulf = is_bearish_engulfing(current_candle, prev_candle)
            bearish_pin = is_pin_bar_bearish(current_candle)
            lower_high = is_lower_high(lookback_data, periods=3)
            current_bearish = close_price < open_price

            sell_confirmed = bool(bearish_engulf or bearish_pin or (lower_high and current_bearish))
            sell_score = int(bearish_engulf) + int(bearish_pin) + int(lower_high and current_bearish)

        if buy_confirmed and not sell_confirmed:
            signal = 'buy'
        elif sell_confirmed and not buy_confirmed:
            signal = 'sell'
        elif buy_confirmed and sell_confirmed:
            if buy_score > sell_score:
                signal = 'buy'
            elif sell_score > buy_score:
                signal = 'sell'
            else:
                # Tie-breaker: pick the zone price is deeper inside
                eps = 1e-12
                demand_depth = -1.0
                supply_depth = -1.0
                if in_demand_zone and demand_zone is not None:
                    dz_span = max(float(demand_zone['high'] - demand_zone['low']), eps)
                    demand_depth = float(demand_zone['high'] - close_price) / dz_span  # 0 at top, 1 at bottom
                if in_supply_zone and supply_zone is not None:
                    sz_span = max(float(supply_zone['high'] - supply_zone['low']), eps)
                    supply_depth = float(close_price - supply_zone['low']) / sz_span  # 0 at bottom, 1 at top
                signal = 'sell' if supply_depth >= demand_depth else 'buy'
 
    else:
        # Aggressive entry: enter at zone boundary without waiting for confirmation
        if in_demand_zone and not in_supply_zone:
            signal = 'buy'
        elif in_supply_zone and not in_demand_zone:
            signal = 'sell'
        elif in_demand_zone and in_supply_zone:
            eps = 1e-12
            demand_depth = -1.0
            supply_depth = -1.0
            if demand_zone is not None:
                dz_span = max(float(demand_zone['high'] - demand_zone['low']), eps)
                demand_depth = float(demand_zone['high'] - close_price) / dz_span
            if supply_zone is not None:
                sz_span = max(float(supply_zone['high'] - supply_zone['low']), eps)
                supply_depth = float(close_price - supply_zone['low']) / sz_span
            signal = 'sell' if supply_depth >= demand_depth else 'buy'
 
    # --------------------------------------------------------------------------
    # STEP 4 & 5 — Signal is set; SL/TP are calculated externally via helpers
    # --------------------------------------------------------------------------
    return signal
 
 
# =============================================================================
# TAKE PROFIT & STOP LOSS CALCULATORS
# =============================================================================
 
def calculate_take_profit(
    entry_price: float,
    direction: str,
    lookback_data: pd.DataFrame,
    use_midline: bool = USE_MIDLINE_TP
) -> float | None:
    """
    Calculate take-profit price using Zone-to-Zone logic.
 
    - Buy  → TP at the next supply zone (or midline = 50% of the distance).
    - Sell → TP at the next demand zone (or midline = 50% of the distance).
 
    The midline TP (50% level) provides a safer target since price often
    stalls or pulls back halfway before reaching the next zone.
    """
    if lookback_data is None:
        return None
 
    if direction == 'buy':
        supply_zone = detect_supply_zone(lookback_data)
        if supply_zone is None:
            return None
        target = supply_zone['low']
        if use_midline:
            return entry_price + (target - entry_price) * 0.5
        return target
 
    elif direction == 'sell':
        demand_zone = detect_demand_zone(lookback_data)
        if demand_zone is None:
            return None
        target = demand_zone['high']
        if use_midline:
            return entry_price - (entry_price - target) * 0.5
        return target
 
    return None
 
 
def calculate_stop_loss(
    entry_price: float,
    direction: str,
    lookback_data: pd.DataFrame
) -> float | None:
    """
    Calculate stop-loss price using Zone-to-Zone logic.
 
    - Buy  → SL below the demand zone (never inside the zone).
    - Sell → SL above the supply zone (never inside the zone).
 
    A small buffer is added so the stop sits just outside the zone boundary.
    """
    if lookback_data is None:
        return None
 
    buffer = entry_price * 0.001  # 0.1% buffer beyond zone edge
 
    if direction == 'buy':
        demand_zone = detect_demand_zone(lookback_data)
        if demand_zone is None:
            return None
        return demand_zone['low'] - buffer
 
    elif direction == 'sell':
        supply_zone = detect_supply_zone(lookback_data)
        if supply_zone is None:
            return None
        return supply_zone['high'] + buffer
 
    return None
 
 
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
    gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
 
 
def is_trend_up(lookback_data: pd.DataFrame, periods: int = 5) -> bool:
    """Check if recent trend is upward based on closing prices."""
    if lookback_data is None or len(lookback_data) < periods:
        return False
    recent = lookback_data.tail(periods)
    return recent['close'].iloc[-1] > recent['close'].iloc[0]
 
 
def is_trend_down(lookback_data: pd.DataFrame, periods: int = 5) -> bool:
    """Check if recent trend is downward based on closing prices."""
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
    Run this before training to ensure the strategy is working correctly.
    """
    print(f"Strategy Name:          {STRATEGY_NAME}")
    print(f"Lookback Periods:       {LOOKBACK_PERIODS}")
    print(f"Zone Detection:         {'Enabled' if USE_ZONE_DETECTION else 'Disabled'}")
    print(f"Confirmation Patterns:  {'Enabled' if USE_CONFIRMATION_PATTERNS else 'Disabled'}")
    print(f"Time Filters:           {'Enabled' if USE_TIME_FILTERS else 'Disabled'}")
    print(f"Zone Freshness Check:   {'Enabled' if USE_ZONE_FRESHNESS else 'Disabled'}")
    print(f"Midline Take Profit:    {'Enabled' if USE_MIDLINE_TP else 'Disabled'}")
    print()
 
    # Build synthetic lookback with a clear demand zone
    n = 25
    prices = [100.0 + i * 0.1 for i in range(n)]
 
    lookback = pd.DataFrame({
        'open':   [p - 0.5 for p in prices],
        'high':   [p + 1.0 for p in prices],
        'low':    [p - 1.0 for p in prices],
        'close':  prices,
        'volume': [1000] * n,
    })
 
    # Insert a strong bullish zone candle near the start
    lookback.iloc[5, lookback.columns.get_loc('open')]  = 98.0
    lookback.iloc[5, lookback.columns.get_loc('close')] = 101.5
    lookback.iloc[5, lookback.columns.get_loc('high')]  = 102.0
    lookback.iloc[5, lookback.columns.get_loc('low')]   = 97.5
 
    lookback['candle_size'] = lookback['high'] - lookback['low']
    lookback['body_size']   = (lookback['close'] - lookback['open']).abs()
    lookback['wick_upper']  = lookback['high'] - lookback[['close', 'open']].max(axis=1)
    lookback['wick_lower']  = lookback[['close', 'open']].min(axis=1) - lookback['low']
 
    # Current candle with bullish engulfing inside demand zone
    prev = lookback.iloc[-1]
    test_candle = pd.Series({
        'open':        prev['close'] - 0.3,
        'high':        prev['open'] + 0.5,
        'low':         prev['close'] - 0.5,
        'close':       prev['open'] + 0.3,
        'volume':      1500,
        'candle_size': 1.0,
        'body_size':   0.6,
        'wick_upper':  0.2,
        'wick_lower':  0.2,
        'hour':        10,
        'day_of_week': 2,
        'month':       6,
    })
 
    try:
        result = apply_strategy(test_candle, lookback)
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