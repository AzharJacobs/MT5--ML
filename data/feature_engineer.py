"""
features.py — Zone-to-Zone Feature Engineering
===============================================
Builds ML-ready features from raw OHLCV data.

MAJOR CHANGE (rule-based features):
  Added add_strategy_rules() which encodes the actual strategy conditions
  as binary features. Previously the model saw raw numbers (in_supply_zone=1,
  htf_4h_bias=1) and had to discover the rules itself. It failed — it learned
  time-of-day patterns instead of zone logic.

  Now the model sees pre-computed rule columns that directly answer:
    "Is this a valid buy setup according to the strategy?"
    "Is this a valid sell setup according to the strategy?"
    "Is HTF trend aligned with the zone direction?"
    "Is this a fresh untouched zone?"
    "Is there a confirmation pattern at this zone?"

  These rule columns are added to FEATURE_COLUMNS and replace the raw
  indicators that were confusing the model. The model now learns:
    rule_valid_buy_setup=1 + rule_confirmed_buy=1 + rule_fresh_zone=1 = winner
  Instead of finding spurious time/trend correlations.

PREVIOUS CHANGES:
  - Zone expiry fix: stale zones expire after MAX_ZONE_AGE_BARS + ZONE_EXPIRE_ATR.
  - Zone replacement guard: new zone only replaces old when price clearly broke out.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.features")

# Zone expiry settings
MAX_ZONE_AGE_BARS = 500   # expire zone if older than this many bars
ZONE_EXPIRE_ATR   = 5.0   # AND price is more than this many ATR away

# Session hours in broker time (Exness GMT+3)
# London open: 10:00-12:30, London/NY overlap: 16:00-19:00
GOOD_SESSION_HOURS = {10, 11, 16, 17, 18}
VALID_HOURS        = set(range(6, 22))   # exclude dead hours 22-05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Zone Detection
# ---------------------------------------------------------------------------

def detect_zones(
    df: pd.DataFrame,
    lookback: int = 30,
    impulse_atr_multiplier: float = 0.5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Detect supply and demand zones.

    Zone replacement rules:
      1. New zone only replaces active one when price broke out by 1 ATR.
      2. Zone expires when older than MAX_ZONE_AGE_BARS AND price is
         more than ZONE_EXPIRE_ATR away.
    """
    df   = df.copy().reset_index(drop=True)
    atr  = _atr(df, atr_period)

    zone_cols = [
        "demand_zone_top", "demand_zone_bottom", "demand_zone_strength",
        "demand_zone_fresh", "demand_zone_touches", "demand_zone_consolidation",
        "supply_zone_top", "supply_zone_bottom", "supply_zone_strength",
        "supply_zone_fresh", "supply_zone_touches", "supply_zone_consolidation",
        "nearest_demand_dist_atr", "nearest_supply_dist_atr",
        "in_demand_zone", "in_supply_zone", "between_zones",
    ]
    for col in zone_cols:
        df[col] = np.nan

    active_demand     = None
    active_demand_age = 0
    active_supply     = None
    active_supply_age = 0

    for i in range(lookback, len(df)):
        cur       = df.iloc[i]
        cur_atr   = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 1.0
        body      = float(abs(cur["close"] - cur["open"]))
        cur_close = float(cur["close"])

        if active_demand is not None:
            active_demand_age += 1
        if active_supply is not None:
            active_supply_age += 1

        # Expire stale zones
        if active_demand is not None and active_demand_age > MAX_ZONE_AGE_BARS:
            if abs(cur_close - active_demand["top"]) / cur_atr > ZONE_EXPIRE_ATR:
                active_demand     = None
                active_demand_age = 0

        if active_supply is not None and active_supply_age > MAX_ZONE_AGE_BARS:
            if abs(cur_close - active_supply["bottom"]) / cur_atr > ZONE_EXPIRE_ATR:
                active_supply     = None
                active_supply_age = 0

        # Consolidation score
        base_candles = df.iloc[max(0, i-3):i]
        if len(base_candles) > 0:
            base_bodies         = (base_candles["close"] - base_candles["open"]).abs()
            avg_base_body       = float(base_bodies.mean()) if len(base_bodies) > 0 else cur_atr
            consolidation_score = float(max(0.0, 1.0 - (avg_base_body / max(cur_atr, 1e-10))))
        else:
            consolidation_score = 0.0

        # Demand zone: bullish impulse
        if cur["close"] > cur["open"] and body > impulse_atr_multiplier * cur_atr:
            strength   = float(min(body / cur_atr, 5.0))
            new_demand = {
                "top":           float(max(cur["open"], cur["close"])),
                "bottom":        float(cur["low"]),
                "strength":      strength,
                "touches":       0,
                "fresh":         True,
                "consolidation": consolidation_score,
            }
            if active_demand is None:
                active_demand     = new_demand
                active_demand_age = 0
            elif cur_close > active_demand["top"] + cur_atr:
                active_demand     = new_demand
                active_demand_age = 0

        # Supply zone: bearish impulse
        if cur["close"] < cur["open"] and body > impulse_atr_multiplier * cur_atr:
            strength   = float(min(body / cur_atr, 5.0))
            new_supply = {
                "top":           float(cur["high"]),
                "bottom":        float(min(cur["open"], cur["close"])),
                "strength":      strength,
                "touches":       0,
                "fresh":         True,
                "consolidation": consolidation_score,
            }
            if active_supply is None:
                active_supply     = new_supply
                active_supply_age = 0
            elif cur_close < active_supply["bottom"] - cur_atr:
                active_supply     = new_supply
                active_supply_age = 0

        # Write demand zone
        if active_demand is not None:
            df.at[i, "demand_zone_top"]          = active_demand["top"]
            df.at[i, "demand_zone_bottom"]        = active_demand["bottom"]
            df.at[i, "demand_zone_strength"]      = active_demand["strength"]
            df.at[i, "demand_zone_fresh"]         = float(active_demand["fresh"])
            df.at[i, "demand_zone_touches"]       = float(active_demand["touches"])
            df.at[i, "demand_zone_consolidation"] = float(active_demand.get("consolidation", 0.0))

            dist = (cur_close - active_demand["top"]) / cur_atr
            df.at[i, "nearest_demand_dist_atr"] = float(dist)

            in_d = active_demand["bottom"] <= float(cur["low"]) <= active_demand["top"]
            df.at[i, "in_demand_zone"] = float(in_d)
            if in_d:
                active_demand["touches"] += 1
                active_demand["fresh"]    = False

        # Write supply zone
        if active_supply is not None:
            df.at[i, "supply_zone_top"]          = active_supply["top"]
            df.at[i, "supply_zone_bottom"]        = active_supply["bottom"]
            df.at[i, "supply_zone_strength"]      = active_supply["strength"]
            df.at[i, "supply_zone_fresh"]         = float(active_supply["fresh"])
            df.at[i, "supply_zone_touches"]       = float(active_supply["touches"])
            df.at[i, "supply_zone_consolidation"] = float(active_supply.get("consolidation", 0.0))

            dist = (active_supply["bottom"] - cur_close) / cur_atr
            df.at[i, "nearest_supply_dist_atr"] = float(dist)

            in_s = active_supply["bottom"] <= float(cur["high"]) <= active_supply["top"]
            df.at[i, "in_supply_zone"] = float(in_s)
            if in_s:
                active_supply["touches"] += 1
                active_supply["fresh"]    = False

        # Between zones
        if active_demand is not None and active_supply is not None:
            between = active_demand["top"] < cur_close < active_supply["bottom"]
            df.at[i, "between_zones"] = float(between)
        else:
            df.at[i, "between_zones"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Strategy Rule Features  ← NEW
# ---------------------------------------------------------------------------

def add_strategy_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the actual strategy rules as binary features.

    This replaces raw number features with pre-computed conditions that
    directly answer the model's key questions:

      rule_valid_buy_setup   : all buy conditions met (zone + HTF + session)
      rule_valid_sell_setup  : all sell conditions met (zone + HTF + session)
      rule_htf_aligned_buy   : HTF trend supports buying (4h bias >= 0)
      rule_htf_aligned_sell  : HTF trend supports selling (4h bias <= 0)
      rule_fresh_zone        : zone has never been touched (touches == 0)
      rule_confirmed_buy     : bullish confirmation pattern present
      rule_confirmed_sell    : bearish confirmation pattern present
      rule_good_session      : london or london/NY overlap session
      rule_valid_hour        : not dead hours (22:00-05:00)
      rule_zone_has_room     : enough distance to next opposing zone for TP
      rule_buy_score         : sum of buy rule conditions (0-5)
      rule_sell_score        : sum of sell rule conditions (0-5)
    """
    df = df.copy()

    # --- Session rules (mirror labels.py _in_trading_session) ---
    hour = df["hour"].fillna(-1).astype(float)
    rule_good_session = hour.isin(GOOD_SESSION_HOURS).astype(float)
    rule_valid_hour   = hour.isin(VALID_HOURS).astype(float)

    # --- Zone freshness ---
    d_touches = df.get("demand_zone_touches", pd.Series(1.0, index=df.index)).fillna(1)
    s_touches = df.get("supply_zone_touches", pd.Series(1.0, index=df.index)).fillna(1)
    rule_fresh_demand = (d_touches == 0).astype(float)
    rule_fresh_supply = (s_touches == 0).astype(float)
    rule_fresh_zone   = (
        (df.get("in_demand_zone", pd.Series(0, index=df.index)).fillna(0) == 1) & (d_touches == 0) |
        (df.get("in_supply_zone", pd.Series(0, index=df.index)).fillna(0) == 1) & (s_touches == 0)
    ).astype(float)

    # --- HTF alignment rules ---
    htf_4h = df.get("htf_4h_bias", pd.Series(0.0, index=df.index)).fillna(0)
    htf_1h = df.get("htf_1h_bias", pd.Series(0.0, index=df.index)).fillna(0)

    # Buy: HTF should be bullish or neutral (not bearish)
    rule_htf_aligned_buy  = (htf_4h >= 0).astype(float)
    # Sell: HTF should be bearish or neutral (not bullish)
    rule_htf_aligned_sell = (htf_4h <= 0).astype(float)
    # Strong alignment: both 1H and 4H agree
    rule_htf_strong_buy   = ((htf_4h > 0) & (htf_1h > 0)).astype(float)
    rule_htf_strong_sell  = ((htf_4h < 0) & (htf_1h < 0)).astype(float)

    # --- Confirmation patterns ---
    bull_eng  = df.get("bullish_engulfing",  pd.Series(0, index=df.index)).fillna(0)
    bear_eng  = df.get("bearish_engulfing",  pd.Series(0, index=df.index)).fillna(0)
    pin_bull  = df.get("pin_bar_bullish",    pd.Series(0, index=df.index)).fillna(0)
    pin_bear  = df.get("pin_bar_bearish",    pd.Series(0, index=df.index)).fillna(0)
    hi_low    = df.get("higher_low",         pd.Series(0, index=df.index)).fillna(0)
    lo_high   = df.get("lower_high",         pd.Series(0, index=df.index)).fillna(0)
    bos_bull  = df.get("bos_bullish",        pd.Series(0, index=df.index)).fillna(0)
    bos_bear  = df.get("bos_bearish",        pd.Series(0, index=df.index)).fillna(0)

    rule_confirmed_buy  = ((bull_eng == 1) | (pin_bull == 1) | ((hi_low == 1) & (bos_bull == 1))).astype(float)
    rule_confirmed_sell = ((bear_eng == 1) | (pin_bear == 1) | ((lo_high == 1) & (bos_bear == 1))).astype(float)

    # --- Zone entry flags ---
    in_demand = df.get("in_demand_zone", pd.Series(0, index=df.index)).fillna(0)
    in_supply = df.get("in_supply_zone", pd.Series(0, index=df.index)).fillna(0)

    # --- Room to next zone (TP viability) ---
    atr       = df.get("atr_14", pd.Series(1.0, index=df.index)).fillna(1).replace(0, 1)
    d_top     = df.get("demand_zone_top",    pd.Series(np.nan, index=df.index))
    s_bottom  = df.get("supply_zone_bottom", pd.Series(np.nan, index=df.index))
    close     = df["close"].fillna(0)

    # Room for buy: supply zone bottom is at least 2 ATR above current price
    room_buy  = ((s_bottom - close) / atr > 2.0).astype(float)
    # Room for sell: demand zone top is at least 2 ATR below current price
    room_sell = ((close - d_top) / atr > 2.0).astype(float)
    rule_zone_has_room = (
        ((in_demand == 1) & room_buy) |
        ((in_supply == 1) & room_sell)
    ).astype(float)

    # --- Full setup rules (all conditions) ---
    rule_valid_buy_setup = (
        (in_demand == 1) &
        (rule_htf_aligned_buy == 1) &
        (rule_good_session == 1)
    ).astype(float)

    rule_valid_sell_setup = (
        (in_supply == 1) &
        (rule_htf_aligned_sell == 1) &
        (rule_good_session == 1)
    ).astype(float)

    # --- Composite scores (how many conditions met) ---
    rule_buy_score = (
        (in_demand == 1).astype(float) +
        rule_htf_aligned_buy +
        rule_htf_strong_buy +
        rule_confirmed_buy +
        rule_fresh_demand +
        rule_good_session +
        room_buy
    ).clip(0, 7)

    rule_sell_score = (
        (in_supply == 1).astype(float) +
        rule_htf_aligned_sell +
        rule_htf_strong_sell +
        rule_confirmed_sell +
        rule_fresh_supply +
        rule_good_session +
        room_sell
    ).clip(0, 7)

    # Write all rule columns
    df["rule_valid_buy_setup"]   = rule_valid_buy_setup
    df["rule_valid_sell_setup"]  = rule_valid_sell_setup
    df["rule_htf_aligned_buy"]   = rule_htf_aligned_buy
    df["rule_htf_aligned_sell"]  = rule_htf_aligned_sell
    df["rule_htf_strong_buy"]    = rule_htf_strong_buy
    df["rule_htf_strong_sell"]   = rule_htf_strong_sell
    df["rule_fresh_zone"]        = rule_fresh_zone
    df["rule_fresh_demand"]      = rule_fresh_demand
    df["rule_fresh_supply"]      = rule_fresh_supply
    df["rule_confirmed_buy"]     = rule_confirmed_buy
    df["rule_confirmed_sell"]    = rule_confirmed_sell
    df["rule_good_session"]      = rule_good_session
    df["rule_valid_hour"]        = rule_valid_hour
    df["rule_zone_has_room"]     = rule_zone_has_room
    df["rule_buy_score"]         = rule_buy_score
    df["rule_sell_score"]        = rule_sell_score

    return df


# ---------------------------------------------------------------------------
# Zone Quality Scoring
# ---------------------------------------------------------------------------

def add_zone_quality(df: pd.DataFrame) -> pd.DataFrame:
    df       = df.copy()
    safe_atr = df["atr_14"].replace(0, np.nan) if "atr_14" in df.columns \
               else pd.Series(np.nan, index=df.index)

    demand_score = pd.Series(0.0, index=df.index)
    if "demand_zone_strength" in df.columns:
        demand_score += (df["demand_zone_strength"].fillna(0) / 5.0).clip(0, 1)
    if "demand_zone_fresh" in df.columns:
        demand_score += df["demand_zone_fresh"].fillna(0)
    if "demand_zone_touches" in df.columns:
        demand_score += (1.0 - (df["demand_zone_touches"].fillna(0) * 0.5)).clip(0, 1)
    if "demand_zone_top" in df.columns and "demand_zone_bottom" in df.columns:
        zone_width    = (df["demand_zone_top"] - df["demand_zone_bottom"]).fillna(0)
        demand_score += (zone_width / safe_atr.fillna(1)).clip(0, 2) / 2.0
    if "demand_zone_consolidation" in df.columns:
        demand_score += df["demand_zone_consolidation"].fillna(0).clip(0, 1)
    if "htf_1h_bias" in df.columns:
        demand_score += (df["htf_1h_bias"].fillna(0) == 1.0).astype(float)
    df["demand_zone_quality"] = demand_score.clip(0, 6)

    supply_score = pd.Series(0.0, index=df.index)
    if "supply_zone_strength" in df.columns:
        supply_score += (df["supply_zone_strength"].fillna(0) / 5.0).clip(0, 1)
    if "supply_zone_fresh" in df.columns:
        supply_score += df["supply_zone_fresh"].fillna(0)
    if "supply_zone_touches" in df.columns:
        supply_score += (1.0 - (df["supply_zone_touches"].fillna(0) * 0.5)).clip(0, 1)
    if "supply_zone_top" in df.columns and "supply_zone_bottom" in df.columns:
        zone_width    = (df["supply_zone_top"] - df["supply_zone_bottom"]).fillna(0)
        supply_score += (zone_width / safe_atr.fillna(1)).clip(0, 2) / 2.0
    if "supply_zone_consolidation" in df.columns:
        supply_score += df["supply_zone_consolidation"].fillna(0).clip(0, 1)
    if "htf_1h_bias" in df.columns:
        supply_score += (df["htf_1h_bias"].fillna(0) == -1.0).astype(float)
    df["supply_zone_quality"] = supply_score.clip(0, 6)

    df["active_zone_quality"] = np.where(
        df["in_demand_zone"].fillna(0) == 1.0, df["demand_zone_quality"],
        np.where(
            df["in_supply_zone"].fillna(0) == 1.0, df["supply_zone_quality"],
            0.0
        )
    )
    return df


# ---------------------------------------------------------------------------
# Confirmation Signals
# ---------------------------------------------------------------------------

def add_confirmation_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    body       = (df["close"] - df["open"]).abs()
    wick_upper = df["high"]  - df[["open", "close"]].max(axis=1)
    wick_lower = df[["open", "close"]].min(axis=1) - df["low"]
    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    df["bullish_engulfing"] = (
        (prev_close < prev_open) & (df["close"] > df["open"]) &
        (df["open"] < prev_close) & (df["close"] > prev_open)
    ).astype(float)

    df["bearish_engulfing"] = (
        (prev_close > prev_open) & (df["close"] < df["open"]) &
        (df["open"] > prev_close) & (df["close"] < prev_open)
    ).astype(float)

    safe_body = body.clip(lower=1e-10)
    df["pin_bar_bullish"] = (
        (wick_lower > 2.0 * safe_body) & (wick_lower > 2.0 * wick_upper)
    ).astype(float)

    df["pin_bar_bearish"] = (
        (wick_upper > 2.0 * safe_body) & (wick_upper > 2.0 * wick_lower)
    ).astype(float)

    df["higher_low"] = (df["low"]  > df["low"].shift(1)).astype(float)
    df["lower_high"] = (df["high"] < df["high"].shift(1)).astype(float)

    swing_high = df["high"].rolling(5).max().shift(1)
    swing_low  = df["low"].rolling(5).min().shift(1)
    df["bos_bullish"] = (df["close"] > swing_high).astype(float)
    df["bos_bearish"] = (df["close"] < swing_low).astype(float)

    df["buy_confirmation_score"]  = (
        df["bullish_engulfing"] + df["pin_bar_bullish"] +
        df["higher_low"]        + df["bos_bullish"]
    )
    df["sell_confirmation_score"] = (
        df["bearish_engulfing"] + df["pin_bar_bearish"] +
        df["lower_high"]        + df["bos_bearish"]
    )
    return df


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["atr_14"]  = _atr(df, 14)
    df["rsi_14"]  = _rsi(df["close"], 14)
    df["ema_20"]  = _ema(df["close"], 20)
    df["ema_50"]  = _ema(df["close"], 50)
    df["ema_200"] = _ema(df["close"], 200)

    safe_atr = df["atr_14"].replace(0, np.nan)
    df["ema_spread_atr"]     = (df["ema_20"] - df["ema_50"]) / safe_atr
    df["price_above_ema20"]  = (df["close"] > df["ema_20"]).astype(float)
    df["price_above_ema50"]  = (df["close"] > df["ema_50"]).astype(float)
    df["price_above_ema200"] = (df["close"] > df["ema_200"]).astype(float)

    df["ema_trend_bias"] = np.where(
        (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]),  1,
        np.where(
            (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]), -1, 0
        )
    ).astype(float)

    bb_mid   = df["close"].rolling(20).mean()
    bb_std   = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"]  = (df["close"] - bb_lower) / bb_width
    df["bb_width_atr"] = bb_width / safe_atr

    vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"]   = df["volume"] / vol_ma
    df["body_atr_ratio"] = (df["close"] - df["open"]).abs() / safe_atr
    df["momentum_5"]     = (df["close"] - df["close"].shift(5))  / safe_atr
    df["momentum_10"]    = (df["close"] - df["close"].shift(10)) / safe_atr

    return df


# ---------------------------------------------------------------------------
# HTF Context
# ---------------------------------------------------------------------------

def _extract_htf_zones(htf_df: pd.DataFrame, impulse_atr_multiplier: float = 0.5) -> pd.DataFrame:
    htf_df = htf_df.copy().reset_index(drop=True)
    htf_df["timestamp"] = pd.to_datetime(htf_df["timestamp"])
    atr = _atr(htf_df, 14)

    records       = []
    active_demand = None
    active_supply = None
    demand_age    = 0
    supply_age    = 0

    for i in range(len(htf_df)):
        cur       = htf_df.iloc[i]
        catr      = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 1.0
        body      = float(abs(cur["close"] - cur["open"]))
        cur_close = float(cur["close"])

        if active_demand is not None: demand_age += 1
        if active_supply is not None: supply_age += 1

        if active_demand is not None and demand_age > MAX_ZONE_AGE_BARS:
            if abs(cur_close - active_demand["top"]) / catr > ZONE_EXPIRE_ATR:
                active_demand = None; demand_age = 0

        if active_supply is not None and supply_age > MAX_ZONE_AGE_BARS:
            if abs(cur_close - active_supply["bottom"]) / catr > ZONE_EXPIRE_ATR:
                active_supply = None; supply_age = 0

        if cur["close"] > cur["open"] and body > impulse_atr_multiplier * catr:
            new_demand = {"top": float(max(cur["open"], cur["close"])), "bottom": float(cur["low"])}
            if active_demand is None:
                active_demand = new_demand; demand_age = 0
            elif cur_close > active_demand["top"] + catr:
                active_demand = new_demand; demand_age = 0

        if cur["close"] < cur["open"] and body > impulse_atr_multiplier * catr:
            new_supply = {"top": float(cur["high"]), "bottom": float(min(cur["open"], cur["close"]))}
            if active_supply is None:
                active_supply = new_supply; supply_age = 0
            elif cur_close < active_supply["bottom"] - catr:
                active_supply = new_supply; supply_age = 0

        records.append({
            "timestamp":              cur["timestamp"],
            "htf_demand_zone_top":    active_demand["top"]    if active_demand else np.nan,
            "htf_demand_zone_bottom": active_demand["bottom"] if active_demand else np.nan,
            "htf_supply_zone_top":    active_supply["top"]    if active_supply else np.nan,
            "htf_supply_zone_bottom": active_supply["bottom"] if active_supply else np.nan,
            "htf_1h_bias": (
                1.0  if active_demand and not active_supply else
                -1.0 if active_supply and not active_demand else
                0.0
            ),
        })
    return pd.DataFrame(records)


def add_htf_context(ltf_df, h1_df, h4_df):
    ltf_df = ltf_df.copy()
    ltf_df["timestamp"] = pd.to_datetime(ltf_df["timestamp"])

    if h1_df is not None and len(h1_df) > 0:
        h1_zones = _extract_htf_zones(h1_df, impulse_atr_multiplier=0.5)
        h1_zones["timestamp"] = pd.to_datetime(h1_zones["timestamp"])
        drop_cols = [c for c in ["htf_demand_zone_top","htf_demand_zone_bottom",
                                  "htf_supply_zone_top","htf_supply_zone_bottom",
                                  "htf_1h_bias"] if c in ltf_df.columns]
        ltf_df = ltf_df.drop(columns=drop_cols)
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h1_zones.sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
    else:
        for c in ["htf_demand_zone_top","htf_demand_zone_bottom",
                  "htf_supply_zone_top","htf_supply_zone_bottom"]:
            ltf_df[c] = np.nan
        ltf_df["htf_1h_bias"] = 0.0

    if h4_df is not None and len(h4_df) > 0:
        h4_df = h4_df.copy()
        h4_df["timestamp"] = pd.to_datetime(h4_df["timestamp"])
        h4_atr  = _atr(h4_df, 14)
        h4_body = (h4_df["close"] - h4_df["open"]).abs()
        h4_imp  = h4_body > 0.5 * h4_atr
        h4_bias = np.where(h4_imp & (h4_df["close"] > h4_df["open"]),  1.0,
                  np.where(h4_imp & (h4_df["close"] < h4_df["open"]), -1.0, np.nan))
        h4_df["htf_4h_bias"] = pd.Series(h4_bias).ffill().fillna(0.0).values
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h4_df[["timestamp","htf_4h_bias"]].sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
    else:
        ltf_df["htf_4h_bias"] = 0.0

    ltf_df["htf_aligned"] = (
        (ltf_df["htf_1h_bias"] == ltf_df["htf_4h_bias"]) &
        (ltf_df["htf_1h_bias"] != 0)
    ).astype(float)

    return ltf_df


# ---------------------------------------------------------------------------
# Master Builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
    zone_lookback: int = 30,
    impulse_atr_multiplier: float = 0.5,
) -> pd.DataFrame:
    logger.info(f"Building features for {len(df)} rows...")

    df = detect_zones(df, lookback=zone_lookback, impulse_atr_multiplier=impulse_atr_multiplier)
    df = add_confirmation_signals(df)
    df = add_indicators(df)

    if h1_df is not None or h4_df is not None:
        df = add_htf_context(df, h1_df, h4_df)

    df = add_zone_quality(df)
    df = add_strategy_rules(df)   # ← NEW: encode strategy rules as features

    warmup = max(200, zone_lookback)
    df = df.iloc[warmup:].reset_index(drop=True)

    logger.info(f"Features built — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature column list for ML
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # ── Strategy rule features (primary signal) ──────────────────────
    # These encode the actual strategy conditions directly.
    # rule_valid_buy/sell_setup already includes session so we don't
    # need raw hour/session as separate features — they just distract.
    "rule_valid_buy_setup",
    "rule_valid_sell_setup",
    "rule_htf_aligned_buy",
    "rule_htf_aligned_sell",
    "rule_htf_strong_buy",
    "rule_htf_strong_sell",
    "rule_fresh_zone",
    "rule_fresh_demand",
    "rule_fresh_supply",
    "rule_confirmed_buy",
    "rule_confirmed_sell",
    "rule_zone_has_room",
    "rule_buy_score",
    "rule_sell_score",

    # ── Zone context (supporting) ─────────────────────────────────────
    "in_demand_zone", "in_supply_zone", "between_zones",
    "demand_zone_strength", "demand_zone_fresh", "demand_zone_touches",
    "demand_zone_consolidation",
    "supply_zone_strength", "supply_zone_fresh", "supply_zone_touches",
    "supply_zone_consolidation",
    "nearest_demand_dist_atr", "nearest_supply_dist_atr",
    "demand_zone_quality", "supply_zone_quality", "active_zone_quality",

    # ── Confirmation patterns ─────────────────────────────────────────
    "bullish_engulfing", "bearish_engulfing",
    "pin_bar_bullish", "pin_bar_bearish",
    "higher_low", "lower_high",
    "bos_bullish", "bos_bearish",
    "buy_confirmation_score", "sell_confirmation_score",

    # ── HTF context ───────────────────────────────────────────────────
    "htf_1h_bias", "htf_4h_bias", "htf_aligned",
    "htf_demand_zone_top", "htf_demand_zone_bottom",
    "htf_supply_zone_top", "htf_supply_zone_bottom",

    # ── Market conditions ─────────────────────────────────────────────
    "atr_14", "rsi_14",
    "volume_ratio", "body_atr_ratio",
    "momentum_5", "momentum_10",
    "bb_position", "bb_width_atr",

    # ── Candle context ────────────────────────────────────────────────
    # month kept — seasonal patterns are real
    # hour/session removed — already encoded in rule_valid_buy/sell_setup
    "month",
    "candle_size", "body_size", "wick_upper", "wick_lower",
]