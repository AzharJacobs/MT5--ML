"""
strategy.py - Personal Trading Strategy Rules
============================================

CHANGES (zone alignment fix):
  - calculate_stop_loss() and calculate_take_profit() now accept an optional
    `feature_row` parameter (pd.Series from features.py raw zone columns).
  - When feature_row is provided, SL/TP use the SAME zone boundaries the model
    was trained on (features.py zones) instead of re-detecting zones via
    strategy.py logic. This eliminates the train/execute mismatch.
  - HTF zone boundaries are used first (institutional levels), LTF as secondary,
    original strategy.py detection as final fallback.
  - All other strategy logic is unchanged.
"""

import pandas as pd
import numpy as np
from typing import Optional

# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

LOOKBACK_PERIODS = 20
STRATEGY_NAME = "Zone-to-Zone Trading Strategy"
USE_ZONE_DETECTION = True
USE_CONFIRMATION_PATTERNS = True
USE_TIME_FILTERS = True
USE_ZONE_FRESHNESS = True
USE_MIDLINE_TP = True

STRONG_MOVE_THRESHOLD = 0.0010
ZONE_BODY_RATIO_MIN = 0.50
ZONE_TOUCH_TOLERANCE = 0.001
MAX_ZONE_AGE = 60
MIN_ZONE_CANDLES = 2


# =============================================================================
# ZONE DETECTION HELPERS
# =============================================================================

def detect_demand_zone(lookback_data: pd.DataFrame) -> dict | None:
    if lookback_data is None or len(lookback_data) < MIN_ZONE_CANDLES + 1:
        return None

    demand_zones = []

    for i in range(len(lookback_data) - 1):
        current = lookback_data.iloc[i]

        if current['close'] <= current['open']:
            continue

        body_size = current['close'] - current['open']
        candle_size = current['high'] - current['low']

        if candle_size == 0 or (body_size / candle_size) < ZONE_BODY_RATIO_MIN:
            continue

        if body_size / current['open'] < STRONG_MOVE_THRESHOLD:
            continue

        consecutive_strength = 1
        zone_low = current['low']
        zone_high = current['high']

        for j in range(i + 1, min(i + MIN_ZONE_CANDLES + 1, len(lookback_data))):
            next_candle = lookback_data.iloc[j]
            zone_low = min(zone_low, next_candle['low'])
            zone_high = max(zone_high, next_candle['high'])

            if next_candle['close'] >= next_candle['open']:
                consecutive_strength += 1
            elif (next_candle['high'] - next_candle['low']) < (current['candle_size'] * 0.5):
                consecutive_strength += 0.5

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

    if demand_zones:
        demand_zones.sort(key=lambda x: (x['age'], -x['strength']))
        return {
            'low': demand_zones[0]['low'],
            'high': demand_zones[0]['high'],
            'age': demand_zones[0]['age'],
            'strength': demand_zones[0]['strength']
        }

    return None


def detect_supply_zone(lookback_data: pd.DataFrame) -> dict | None:
    supply_strong_move_threshold = STRONG_MOVE_THRESHOLD
    supply_min_zone_candles = MIN_ZONE_CANDLES

    if lookback_data is None or len(lookback_data) < supply_min_zone_candles + 1:
        return None

    supply_zones = []

    for i in range(len(lookback_data) - 1):
        current = lookback_data.iloc[i]

        if current['close'] >= current['open']:
            continue

        body_size = current['open'] - current['close']
        candle_size = current['high'] - current['low']

        if candle_size == 0 or (body_size / candle_size) < ZONE_BODY_RATIO_MIN:
            continue

        if body_size / current['open'] < supply_strong_move_threshold:
            continue

        consecutive_strength = 1
        zone_low = current['low']
        zone_high = current['high']

        for j in range(i + 1, min(i + supply_min_zone_candles + 1, len(lookback_data))):
            next_candle = lookback_data.iloc[j]
            zone_low = min(zone_low, next_candle['low'])
            zone_high = max(zone_high, next_candle['high'])

            if next_candle['close'] <= next_candle['open']:
                consecutive_strength += 1
            elif (next_candle['high'] - next_candle['low']) < (current['candle_size'] * 0.5):
                consecutive_strength += 0.5

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

    if supply_zones:
        supply_zones.sort(key=lambda x: (x['age'], -x['strength']))
        return {
            'low': supply_zones[0]['low'],
            'high': supply_zones[0]['high'],
            'age': supply_zones[0]['age'],
            'strength': supply_zones[0]['strength']
        }

    return None


def price_in_zone(price: float, zone: dict, tolerance: float = ZONE_TOUCH_TOLERANCE) -> bool:
    zone_low  = zone['low']  * (1 - tolerance)
    zone_high = zone['high'] * (1 + tolerance)
    return zone_low <= price <= zone_high


def zone_is_fresh(zone: dict, fresh_age: int = 20) -> bool:
    return zone['age'] <= fresh_age


# =============================================================================
# CONFIRMATION PATTERN HELPERS
# =============================================================================

def is_bullish_engulfing(current_candle: pd.Series, prev_candle: pd.Series) -> bool:
    return (
        prev_candle['close'] < prev_candle['open'] and
        current_candle['close'] > current_candle['open'] and
        current_candle['open'] < prev_candle['close'] and
        current_candle['close'] > prev_candle['open']
    )


def is_bearish_engulfing(current_candle: pd.Series, prev_candle: pd.Series) -> bool:
    return (
        prev_candle['close'] > prev_candle['open'] and
        current_candle['close'] < current_candle['open'] and
        current_candle['open'] > prev_candle['close'] and
        current_candle['close'] < prev_candle['open']
    )


def is_pin_bar_bullish(candle: pd.Series) -> bool:
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
    if lookback_data is None or len(lookback_data) < periods:
        return False
    lows = lookback_data.tail(periods)['low'].values
    return all(lows[i] > lows[i - 1] for i in range(1, len(lows)))


def is_lower_high(lookback_data: pd.DataFrame, periods: int = 3) -> bool:
    if lookback_data is None or len(lookback_data) < periods:
        return False
    highs = lookback_data.tail(periods)['high'].values
    return all(highs[i] < highs[i - 1] for i in range(1, len(highs)))


# =============================================================================
# MAIN STRATEGY FUNCTION
# =============================================================================

def apply_strategy(
    current_candle: pd.Series,
    lookback_data: pd.DataFrame = None
) -> str:
    signal = 'neutral'

    if lookback_data is None or len(lookback_data) < LOOKBACK_PERIODS:
        return signal

    close_price = current_candle['close']
    open_price  = current_candle['open']
    hour        = current_candle.get('hour', 12)
    day_of_week = current_candle.get('day_of_week', 1)

    # Session gate removed from live execution path.
    # in_session is a model feature — the model learns session importance.
    # Hard-blocking here was preventing valid out-of-session setups.

    if not USE_ZONE_DETECTION:
        return signal

    demand_zone = detect_demand_zone(lookback_data)
    supply_zone = detect_supply_zone(lookback_data)

    in_demand_zone = demand_zone is not None and price_in_zone(close_price, demand_zone)
    in_supply_zone = supply_zone is not None and price_in_zone(close_price, supply_zone)

    if not in_demand_zone and not in_supply_zone:
        return 'neutral'

    if USE_ZONE_FRESHNESS:
        if in_demand_zone and not zone_is_fresh(demand_zone):
            in_demand_zone = False
        if in_supply_zone and not zone_is_fresh(supply_zone):
            in_supply_zone = False

    if not in_demand_zone and not in_supply_zone:
        return 'neutral'

    prev_candle = lookback_data.iloc[-1] if len(lookback_data) >= 1 else None

    if USE_CONFIRMATION_PATTERNS and prev_candle is not None:
        buy_score = 0
        sell_score = 0
        buy_confirmed = False
        sell_confirmed = False

        if in_demand_zone:
            bullish_engulf = is_bullish_engulfing(current_candle, prev_candle)
            bullish_pin    = is_pin_bar_bullish(current_candle)
            higher_low     = is_higher_low(lookback_data, periods=3)
            current_bullish = close_price > open_price
            buy_confirmed  = bool(bullish_engulf or bullish_pin or (higher_low and current_bullish))
            buy_score      = int(bullish_engulf) + int(bullish_pin) + int(higher_low and current_bullish)

        if in_supply_zone:
            bearish_engulf  = is_bearish_engulfing(current_candle, prev_candle)
            bearish_pin     = is_pin_bar_bearish(current_candle)
            lower_high      = is_lower_high(lookback_data, periods=3)
            current_bearish = close_price < open_price
            sell_confirmed  = bool(bearish_engulf or bearish_pin or (lower_high and current_bearish))
            sell_score      = int(bearish_engulf) + int(bearish_pin) + int(lower_high and current_bearish)

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
                eps = 1e-12
                demand_depth = -1.0
                supply_depth = -1.0
                if in_demand_zone and demand_zone is not None:
                    dz_span = max(float(demand_zone['high'] - demand_zone['low']), eps)
                    demand_depth = float(demand_zone['high'] - close_price) / dz_span
                if in_supply_zone and supply_zone is not None:
                    sz_span = max(float(supply_zone['high'] - supply_zone['low']), eps)
                    supply_depth = float(close_price - supply_zone['low']) / sz_span
                signal = 'sell' if supply_depth >= demand_depth else 'buy'
    else:
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

    return signal


# =============================================================================
# TAKE PROFIT & STOP LOSS — UPDATED TO USE features.py ZONES
# =============================================================================

def calculate_stop_loss(
    entry_price: float,
    direction: str,
    lookback_data: pd.DataFrame,
    feature_row: pd.Series = None,
) -> float | None:
    """
    Calculate stop-loss price.

    Priority:
      1. features.py demand/supply zone boundaries (same zones model trained on)
      2. Original strategy.py zone detection (fallback)

    feature_row: raw (unscaled) pd.Series stored in features_by_dt["raw"]
                 Must contain: demand_zone_bottom, supply_zone_top, atr_14
    """
    # --- Derive ATR buffer from feature row ---
    atr = None
    if feature_row is not None:
        try:
            atr_val = feature_row.get("atr_14") if hasattr(feature_row, "get") else feature_row.get("atr_14", None)
            if atr_val is not None and not pd.isna(float(atr_val)):
                atr = float(atr_val)
        except Exception:
            atr = None

    buffer = (atr * 0.5) if (atr and atr > 0) else (entry_price * 0.001)

    # --- Primary: use features.py zone boundaries ---
    if feature_row is not None:
        try:
            if direction == "buy":
                zone_bottom = feature_row.get("demand_zone_bottom") \
                    if hasattr(feature_row, "get") else None
                if zone_bottom is not None and not pd.isna(float(zone_bottom)):
                    sl = float(zone_bottom) - buffer
                    if sl < entry_price:   # sanity: SL must be below entry for buy
                        return sl

            elif direction == "sell":
                zone_top = feature_row.get("supply_zone_top") \
                    if hasattr(feature_row, "get") else None
                if zone_top is not None and not pd.isna(float(zone_top)):
                    sl = float(zone_top) + buffer
                    if sl > entry_price:   # sanity: SL must be above entry for sell
                        return sl
        except Exception:
            pass  # fall through to strategy.py fallback

    # --- Fallback: original strategy.py zone detection ---
    buffer_pct = entry_price * 0.001

    if direction == "buy":
        demand_zone = detect_demand_zone(lookback_data)
        if demand_zone is None:
            return None
        return demand_zone["low"] - buffer_pct

    elif direction == "sell":
        supply_zone = detect_supply_zone(lookback_data)
        if supply_zone is None:
            return None
        return supply_zone["high"] + buffer_pct

    return None


def calculate_take_profit(
    entry_price: float,
    direction: str,
    lookback_data: pd.DataFrame,
    feature_row: pd.Series = None,
    use_midline: bool = USE_MIDLINE_TP,
) -> float | None:
    """
    Calculate take-profit price.

    Priority:
      1. HTF supply/demand zone boundaries from features.py (institutional levels)
      2. LTF supply/demand zone boundaries from features.py
      3. Original strategy.py zone detection (fallback)

    Midline TP = 50% of distance to next zone (safer, hits more often on small accounts).

    feature_row: raw (unscaled) pd.Series stored in features_by_dt["raw"]
                 Must contain: htf_supply_zone_bottom, htf_demand_zone_top,
                               supply_zone_bottom, demand_zone_top
    """
    if feature_row is not None:
        try:
            if direction == "buy":
                # 1. HTF supply zone bottom
                htf_val = feature_row.get("htf_supply_zone_bottom") \
                    if hasattr(feature_row, "get") else None
                if htf_val is not None and not pd.isna(float(htf_val)):
                    target = float(htf_val)
                    if target > entry_price:
                        return entry_price + (target - entry_price) * (0.5 if use_midline else 1.0)

                # 2. LTF supply zone bottom
                ltf_val = feature_row.get("supply_zone_bottom") \
                    if hasattr(feature_row, "get") else None
                if ltf_val is not None and not pd.isna(float(ltf_val)):
                    target = float(ltf_val)
                    if target > entry_price:
                        return entry_price + (target - entry_price) * (0.5 if use_midline else 1.0)

            elif direction == "sell":
                # 1. HTF demand zone top
                htf_val = feature_row.get("htf_demand_zone_top") \
                    if hasattr(feature_row, "get") else None
                if htf_val is not None and not pd.isna(float(htf_val)):
                    target = float(htf_val)
                    if target < entry_price:
                        return entry_price - (entry_price - target) * (0.5 if use_midline else 1.0)

                # 2. LTF demand zone top
                ltf_val = feature_row.get("demand_zone_top") \
                    if hasattr(feature_row, "get") else None
                if ltf_val is not None and not pd.isna(float(ltf_val)):
                    target = float(ltf_val)
                    if target < entry_price:
                        return entry_price - (entry_price - target) * (0.5 if use_midline else 1.0)

        except Exception:
            pass  # fall through to strategy.py fallback

    # --- Fallback: original strategy.py zone detection ---
    if direction == "buy":
        supply_zone = detect_supply_zone(lookback_data)
        if supply_zone is None:
            return None
        target = supply_zone["low"]
        return entry_price + (target - entry_price) * (0.5 if use_midline else 1.0)

    elif direction == "sell":
        demand_zone = detect_demand_zone(lookback_data)
        if demand_zone is None:
            return None
        target = demand_zone["high"]
        return entry_price - (entry_price - target) * (0.5 if use_midline else 1.0)

    return None


# =============================================================================
# ADDITIONAL HELPERS
# =============================================================================

def calculate_sma(data: pd.DataFrame, column: str, period: int) -> pd.Series:
    return data[column].rolling(window=period).mean()


def calculate_ema(data: pd.DataFrame, column: str, period: int) -> pd.Series:
    return data[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
    delta = data[column].diff()
    gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def is_trend_up(lookback_data: pd.DataFrame, periods: int = 5) -> bool:
    if lookback_data is None or len(lookback_data) < periods:
        return False
    recent = lookback_data.tail(periods)
    return recent['close'].iloc[-1] > recent['close'].iloc[0]


def is_trend_down(lookback_data: pd.DataFrame, periods: int = 5) -> bool:
    if lookback_data is None or len(lookback_data) < periods:
        return False
    recent = lookback_data.tail(periods)
    return recent['close'].iloc[-1] < recent['close'].iloc[0]


# =============================================================================
# STRATEGY VALIDATION
# =============================================================================

def validate_strategy() -> bool:
    print(f"Strategy Name:          {STRATEGY_NAME}")
    print(f"Lookback Periods:       {LOOKBACK_PERIODS}")
    print(f"Zone Detection:         {'Enabled' if USE_ZONE_DETECTION else 'Disabled'}")
    print(f"Confirmation Patterns:  {'Enabled' if USE_CONFIRMATION_PATTERNS else 'Disabled'}")
    print(f"Time Filters:           {'Enabled' if USE_TIME_FILTERS else 'Disabled'}")
    print(f"Zone Freshness Check:   {'Enabled' if USE_ZONE_FRESHNESS else 'Disabled'}")
    print(f"Midline Take Profit:    {'Enabled' if USE_MIDLINE_TP else 'Disabled'}")
    print()

    n = 25
    prices = [100.0 + i * 0.1 for i in range(n)]

    lookback = pd.DataFrame({
        'open':   [p - 0.5 for p in prices],
        'high':   [p + 1.0 for p in prices],
        'low':    [p - 1.0 for p in prices],
        'close':  prices,
        'volume': [1000] * n,
    })

    lookback.iloc[5, lookback.columns.get_loc('open')]  = 98.0
    lookback.iloc[5, lookback.columns.get_loc('close')] = 101.5
    lookback.iloc[5, lookback.columns.get_loc('high')]  = 102.0
    lookback.iloc[5, lookback.columns.get_loc('low')]   = 97.5

    lookback['candle_size'] = lookback['high'] - lookback['low']
    lookback['body_size']   = (lookback['close'] - lookback['open']).abs()
    lookback['wick_upper']  = lookback['high'] - lookback[['close', 'open']].max(axis=1)
    lookback['wick_lower']  = lookback[['close', 'open']].min(axis=1) - lookback['low']

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