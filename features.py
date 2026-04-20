"""
features.py — Zone-to-Zone Feature Engineering
===============================================
Builds ML-ready features from raw OHLCV data.

Zone detection is intentionally lenient here — the goal is to detect
as many valid zones as possible and let the ML model learn which ones
are high-probability. Filtering happens via feature scores, not hard gates.

CRITICAL FIX (supply zone stuck at 16,248 while price at 25,000+):
  The zone replacement guard added previously prevented phantom mid-trade
  zone switches, but it was too strict — it never expired old zones that
  price had left far behind.

  Symptom: supply_zone_top = 16,248 across 46,613 of 46,702 bars while
  current price was at 25,000+. in_supply_zone = 57 total hits.
  Result: zero sell signals in labels, model only learned buys.

  Fix: zones now expire when BOTH conditions are true:
    1. Zone age > MAX_ZONE_AGE_BARS (500 bars, ~5 days on 15min)
    2. Price is more than ZONE_EXPIRE_ATR (5.0 ATR) away from the zone

  This preserves the replacement guard for active/nearby zones while
  allowing stale far-away zones to clear so new relevant ones can form.

PREVIOUS CHANGES:
  - Zone replacement guard added: active zone only replaced when price
    has clearly broken out by 1 ATR. Prevents phantom mid-trade switches.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.features")

# Zone expiry settings
MAX_ZONE_AGE_BARS  = 500   # expire zone if older than this many bars
ZONE_EXPIRE_ATR    = 5.0   # AND price is more than this many ATR away


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
      1. A new zone only replaces the active one when price has clearly
         broken out (1 ATR beyond zone boundary). Prevents phantom switches.
      2. A zone expires when it is older than MAX_ZONE_AGE_BARS AND price
         is more than ZONE_EXPIRE_ATR away. Prevents stale zones from
         persisting forever while price trends far away.
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

    active_demand      = None
    active_demand_age  = 0
    active_supply      = None
    active_supply_age  = 0

    for i in range(lookback, len(df)):
        cur       = df.iloc[i]
        cur_atr   = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 1.0
        body      = float(abs(cur["close"] - cur["open"]))
        cur_close = float(cur["close"])

        # --- Increment zone ages ---
        if active_demand is not None:
            active_demand_age += 1
        if active_supply is not None:
            active_supply_age += 1

        # --- Expire stale zones ---
        # A zone is expired when it is both OLD and FAR AWAY from current price.
        # This prevents zones from persisting forever during long trends.
        if active_demand is not None and active_demand_age > MAX_ZONE_AGE_BARS:
            dist_atr = abs(cur_close - active_demand["top"]) / cur_atr
            if dist_atr > ZONE_EXPIRE_ATR:
                active_demand     = None
                active_demand_age = 0

        if active_supply is not None and active_supply_age > MAX_ZONE_AGE_BARS:
            dist_atr = abs(cur_close - active_supply["bottom"]) / cur_atr
            if dist_atr > ZONE_EXPIRE_ATR:
                active_supply     = None
                active_supply_age = 0

        # --- Consolidation check ---
        base_candles = df.iloc[max(0, i-3):i]
        if len(base_candles) > 0:
            base_bodies       = (base_candles["close"] - base_candles["open"]).abs()
            avg_base_body     = float(base_bodies.mean()) if len(base_bodies) > 0 else cur_atr
            consolidation_score = float(max(0.0, 1.0 - (avg_base_body / max(cur_atr, 1e-10))))
        else:
            consolidation_score = 0.0

        # --- Demand zone: bullish impulse ---
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
                # Price has clearly broken above old zone — replace it
                active_demand     = new_demand
                active_demand_age = 0
            # else: keep old zone — price still near it

        # --- Supply zone: bearish impulse ---
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
                # Price has clearly broken below old zone — replace it
                active_supply     = new_supply
                active_supply_age = 0
            # else: keep old zone — price still near it

        # --- Write demand zone to row ---
        if active_demand is not None:
            df.at[i, "demand_zone_top"]           = active_demand["top"]
            df.at[i, "demand_zone_bottom"]         = active_demand["bottom"]
            df.at[i, "demand_zone_strength"]       = active_demand["strength"]
            df.at[i, "demand_zone_fresh"]          = float(active_demand["fresh"])
            df.at[i, "demand_zone_touches"]        = float(active_demand["touches"])
            df.at[i, "demand_zone_consolidation"]  = float(active_demand.get("consolidation", 0.0))

            dist = (cur_close - active_demand["top"]) / cur_atr
            df.at[i, "nearest_demand_dist_atr"] = float(dist)

            in_d = active_demand["bottom"] <= float(cur["low"]) <= active_demand["top"]
            df.at[i, "in_demand_zone"] = float(in_d)
            if in_d:
                active_demand["touches"] += 1
                active_demand["fresh"]    = False

        # --- Write supply zone to row ---
        if active_supply is not None:
            df.at[i, "supply_zone_top"]           = active_supply["top"]
            df.at[i, "supply_zone_bottom"]         = active_supply["bottom"]
            df.at[i, "supply_zone_strength"]       = active_supply["strength"]
            df.at[i, "supply_zone_fresh"]          = float(active_supply["fresh"])
            df.at[i, "supply_zone_touches"]        = float(active_supply["touches"])
            df.at[i, "supply_zone_consolidation"]  = float(active_supply.get("consolidation", 0.0))

            dist = (active_supply["bottom"] - cur_close) / cur_atr
            df.at[i, "nearest_supply_dist_atr"] = float(dist)

            in_s = active_supply["bottom"] <= float(cur["high"]) <= active_supply["top"]
            df.at[i, "in_supply_zone"] = float(in_s)
            if in_s:
                active_supply["touches"] += 1
                active_supply["fresh"]    = False

        # Between zones: only when BOTH zones exist and price is clearly between them
        if active_demand is not None and active_supply is not None:
            between = (
                active_demand["top"] < cur_close < active_supply["bottom"]
            )
            df.at[i, "between_zones"] = float(between)
        else:
            df.at[i, "between_zones"] = 0.0

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
        touch_score   = (1.0 - (df["demand_zone_touches"].fillna(0) * 0.5)).clip(0, 1)
        demand_score += touch_score
    if "demand_zone_top" in df.columns and "demand_zone_bottom" in df.columns:
        zone_width    = (df["demand_zone_top"] - df["demand_zone_bottom"]).fillna(0)
        width_score   = (zone_width / safe_atr.fillna(1)).clip(0, 2) / 2.0
        demand_score += width_score
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
        touch_score   = (1.0 - (df["supply_zone_touches"].fillna(0) * 0.5)).clip(0, 1)
        supply_score += touch_score
    if "supply_zone_top" in df.columns and "supply_zone_bottom" in df.columns:
        zone_width    = (df["supply_zone_top"] - df["supply_zone_bottom"]).fillna(0)
        width_score   = (zone_width / safe_atr.fillna(1)).clip(0, 2) / 2.0
        supply_score += width_score
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
        (prev_close < prev_open) &
        (df["close"] > df["open"]) &
        (df["open"]  < prev_close) &
        (df["close"] > prev_open)
    ).astype(float)

    df["bearish_engulfing"] = (
        (prev_close > prev_open) &
        (df["close"] < df["open"]) &
        (df["open"]  > prev_close) &
        (df["close"] < prev_open)
    ).astype(float)

    safe_body = body.clip(lower=1e-10)
    df["pin_bar_bullish"] = (
        (wick_lower > 2.0 * safe_body) &
        (wick_lower > 2.0 * wick_upper)
    ).astype(float)

    df["pin_bar_bearish"] = (
        (wick_upper > 2.0 * safe_body) &
        (wick_upper > 2.0 * wick_lower)
    ).astype(float)

    df["higher_low"]  = (df["low"]  > df["low"].shift(1)).astype(float)
    df["lower_high"]  = (df["high"] < df["high"].shift(1)).astype(float)

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

    df["atr_14"]   = _atr(df, 14)
    df["rsi_14"]   = _rsi(df["close"], 14)
    df["ema_20"]   = _ema(df["close"], 20)
    df["ema_50"]   = _ema(df["close"], 50)
    df["ema_200"]  = _ema(df["close"], 200)

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

        if active_demand is not None:
            demand_age += 1
        if active_supply is not None:
            supply_age += 1

        # Expire stale HTF zones (same logic as LTF)
        if active_demand is not None and demand_age > MAX_ZONE_AGE_BARS:
            if abs(cur_close - active_demand["top"]) / catr > ZONE_EXPIRE_ATR:
                active_demand = None
                demand_age    = 0

        if active_supply is not None and supply_age > MAX_ZONE_AGE_BARS:
            if abs(cur_close - active_supply["bottom"]) / catr > ZONE_EXPIRE_ATR:
                active_supply = None
                supply_age    = 0

        if cur["close"] > cur["open"] and body > impulse_atr_multiplier * catr:
            new_demand = {
                "top":    float(max(cur["open"], cur["close"])),
                "bottom": float(cur["low"]),
            }
            if active_demand is None:
                active_demand = new_demand
                demand_age    = 0
            elif cur_close > active_demand["top"] + catr:
                active_demand = new_demand
                demand_age    = 0

        if cur["close"] < cur["open"] and body > impulse_atr_multiplier * catr:
            new_supply = {
                "top":    float(cur["high"]),
                "bottom": float(min(cur["open"], cur["close"])),
            }
            if active_supply is None:
                active_supply = new_supply
                supply_age    = 0
            elif cur_close < active_supply["bottom"] - catr:
                active_supply = new_supply
                supply_age    = 0

        records.append({
            "timestamp":              cur["timestamp"],
            "htf_demand_zone_top":    active_demand["top"]    if active_demand else np.nan,
            "htf_demand_zone_bottom": active_demand["bottom"] if active_demand else np.nan,
            "htf_supply_zone_top":    active_supply["top"]    if active_supply else np.nan,
            "htf_supply_zone_bottom": active_supply["bottom"] if active_supply else np.nan,
            "htf_1h_bias": (
                1.0 if active_demand and not active_supply else
                -1.0 if active_supply and not active_demand else
                0.0
            ),
        })

    return pd.DataFrame(records)


def add_htf_context(
    ltf_df: pd.DataFrame,
    h1_df:  pd.DataFrame,
    h4_df:  pd.DataFrame,
) -> pd.DataFrame:
    ltf_df = ltf_df.copy()
    ltf_df["timestamp"] = pd.to_datetime(ltf_df["timestamp"])

    if h1_df is not None and len(h1_df) > 0:
        h1_zones = _extract_htf_zones(h1_df, impulse_atr_multiplier=0.5)
        h1_zones["timestamp"] = pd.to_datetime(h1_zones["timestamp"])
        drop_cols = [c for c in ["htf_demand_zone_top", "htf_demand_zone_bottom",
                                  "htf_supply_zone_top", "htf_supply_zone_bottom",
                                  "htf_1h_bias"] if c in ltf_df.columns]
        ltf_df = ltf_df.drop(columns=drop_cols)
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h1_zones.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )
    else:
        ltf_df["htf_demand_zone_top"]    = np.nan
        ltf_df["htf_demand_zone_bottom"] = np.nan
        ltf_df["htf_supply_zone_top"]    = np.nan
        ltf_df["htf_supply_zone_bottom"] = np.nan
        ltf_df["htf_1h_bias"]            = 0.0

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
            h4_df[["timestamp", "htf_4h_bias"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward"
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

    warmup = max(200, zone_lookback)
    df = df.iloc[warmup:].reset_index(drop=True)

    logger.info(f"Features built — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature column list for ML
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Zone
    "demand_zone_strength", "demand_zone_fresh", "demand_zone_touches", "demand_zone_consolidation",
    "supply_zone_strength", "supply_zone_fresh", "supply_zone_touches", "supply_zone_consolidation",
    "nearest_demand_dist_atr", "nearest_supply_dist_atr",
    "in_demand_zone", "in_supply_zone", "between_zones",
    # Confirmations
    "bullish_engulfing", "bearish_engulfing",
    "pin_bar_bullish", "pin_bar_bearish",
    "higher_low", "lower_high",
    "bos_bullish", "bos_bearish",
    "buy_confirmation_score", "sell_confirmation_score",
    # Indicators
    "atr_14", "rsi_14",
    "ema_spread_atr",
    "price_above_ema20", "price_above_ema50", "price_above_ema200",
    "ema_trend_bias",
    "bb_position", "bb_width_atr",
    "volume_ratio", "body_atr_ratio",
    "momentum_5", "momentum_10",
    # Zone quality
    "demand_zone_quality", "supply_zone_quality", "active_zone_quality",
    # HTF
    "htf_demand_zone_top", "htf_demand_zone_bottom",
    "htf_supply_zone_top", "htf_supply_zone_bottom",
    "htf_1h_bias", "htf_4h_bias", "htf_aligned",
    # Candle context
    "hour", "month",
    "candle_size", "body_size", "wick_upper", "wick_lower",
    "session",
]