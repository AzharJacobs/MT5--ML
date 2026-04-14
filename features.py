"""
features.py — Zone-to-Zone Feature Engineering
===============================================
Builds ML-ready features from raw OHLCV data.

Zone detection is intentionally lenient here — the goal is to detect
as many valid zones as possible and let the ML model learn which ones
are high-probability. Filtering happens via feature scores, not hard gates.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.features")


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

    Deliberately lenient (0.5x ATR default) so the model sees many zone
    encounters and learns which are high-probability from the feature set.
    Freshness, touch count, and strength score give the model enough signal
    to discriminate good zones from weak ones.
    """
    df   = df.copy().reset_index(drop=True)
    atr  = _atr(df, atr_period)

    # Initialize all zone columns as float from the start (avoids dtype errors)
    zone_cols = [
        "demand_zone_top", "demand_zone_bottom", "demand_zone_strength",
        "demand_zone_fresh", "demand_zone_touches",
        "supply_zone_top", "supply_zone_bottom", "supply_zone_strength",
        "supply_zone_fresh", "supply_zone_touches",
        "nearest_demand_dist_atr", "nearest_supply_dist_atr",
        "in_demand_zone", "in_supply_zone", "between_zones",
    ]
    for col in zone_cols:
        df[col] = np.nan

    active_demand = None
    active_supply = None

    for i in range(lookback, len(df)):
        cur     = df.iloc[i]
        cur_atr = float(atr.iloc[i]) if not np.isnan(atr.iloc[i]) else 1.0
        body    = float(abs(cur["close"] - cur["open"]))

        # --- Demand zone: bullish impulse ---
        if cur["close"] > cur["open"] and body > impulse_atr_multiplier * cur_atr:
            strength = float(min(body / cur_atr, 5.0))
            active_demand = {
                "top":     float(max(cur["open"], cur["close"])),
                "bottom":  float(cur["low"]),
                "strength": strength,
                "touches": 0,
                "fresh":   True,
            }

        # --- Supply zone: bearish impulse ---
        if cur["close"] < cur["open"] and body > impulse_atr_multiplier * cur_atr:
            strength = float(min(body / cur_atr, 5.0))
            active_supply = {
                "top":     float(cur["high"]),
                "bottom":  float(min(cur["open"], cur["close"])),
                "strength": strength,
                "touches": 0,
                "fresh":   True,
            }

        # --- Write demand zone to row ---
        if active_demand is not None:
            df.at[i, "demand_zone_top"]      = active_demand["top"]
            df.at[i, "demand_zone_bottom"]   = active_demand["bottom"]
            df.at[i, "demand_zone_strength"] = active_demand["strength"]
            df.at[i, "demand_zone_fresh"]    = float(active_demand["fresh"])
            df.at[i, "demand_zone_touches"]  = float(active_demand["touches"])

            dist = (float(cur["close"]) - active_demand["top"]) / cur_atr
            df.at[i, "nearest_demand_dist_atr"] = float(dist)

            in_d = active_demand["bottom"] <= float(cur["low"]) <= active_demand["top"]
            df.at[i, "in_demand_zone"] = float(in_d)
            if in_d:
                active_demand["touches"] += 1
                active_demand["fresh"]    = False

        # --- Write supply zone to row ---
        if active_supply is not None:
            df.at[i, "supply_zone_top"]      = active_supply["top"]
            df.at[i, "supply_zone_bottom"]   = active_supply["bottom"]
            df.at[i, "supply_zone_strength"] = active_supply["strength"]
            df.at[i, "supply_zone_fresh"]    = float(active_supply["fresh"])
            df.at[i, "supply_zone_touches"]  = float(active_supply["touches"])

            dist = (active_supply["bottom"] - float(cur["close"])) / cur_atr
            df.at[i, "nearest_supply_dist_atr"] = float(dist)

            in_s = active_supply["bottom"] <= float(cur["high"]) <= active_supply["top"]
            df.at[i, "in_supply_zone"] = float(in_s)
            if in_s:
                active_supply["touches"] += 1
                active_supply["fresh"]    = False

        # Between zones (no-man's land)
        if active_demand is not None and active_supply is not None:
            between = (
                active_demand["top"] < float(cur["close"]) < active_supply["bottom"]
            )
            df.at[i, "between_zones"] = float(between)

    return df


# ---------------------------------------------------------------------------
# Confirmation Signals
# ---------------------------------------------------------------------------

def add_confirmation_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick confirmation signals as numeric features.
    These are FEATURES for the model — not hard gates on signal generation.
    """
    df = df.copy()

    body       = (df["close"] - df["open"]).abs()
    wick_upper = df["high"]  - df[["open", "close"]].max(axis=1)
    wick_lower = df[["open", "close"]].min(axis=1) - df["low"]

    prev_open  = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    # Bullish engulfing
    df["bullish_engulfing"] = (
        (prev_close < prev_open) &
        (df["close"] > df["open"]) &
        (df["open"]  < prev_close) &
        (df["close"] > prev_open)
    ).astype(float)

    # Bearish engulfing
    df["bearish_engulfing"] = (
        (prev_close > prev_open) &
        (df["close"] < df["open"]) &
        (df["open"]  > prev_close) &
        (df["close"] < prev_open)
    ).astype(float)

    # Pin bar bullish (long lower wick)
    safe_body = body.clip(lower=1e-10)
    df["pin_bar_bullish"] = (
        (wick_lower > 2.0 * safe_body) &
        (wick_lower > 2.0 * wick_upper)
    ).astype(float)

    # Pin bar bearish (long upper wick)
    df["pin_bar_bearish"] = (
        (wick_upper > 2.0 * safe_body) &
        (wick_upper > 2.0 * wick_lower)
    ).astype(float)

    # Market structure
    df["higher_low"]  = (df["low"]  > df["low"].shift(1)).astype(float)
    df["lower_high"]  = (df["high"] < df["high"].shift(1)).astype(float)

    # Break of structure
    swing_high = df["high"].rolling(5).max().shift(1)
    swing_low  = df["low"].rolling(5).min().shift(1)
    df["bos_bullish"] = (df["close"] > swing_high).astype(float)
    df["bos_bearish"] = (df["close"] < swing_low).astype(float)

    # Composite scores (0–4) — used as features, not gates
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
    df["ema_spread_atr"]    = (df["ema_20"] - df["ema_50"]) / safe_atr
    df["price_above_ema20"] = (df["close"] > df["ema_20"]).astype(float)
    df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(float)
    df["price_above_ema200"]= (df["close"] > df["ema_200"]).astype(float)

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

def add_htf_context(
    ltf_df: pd.DataFrame,
    h1_df:  pd.DataFrame,
    h4_df:  pd.DataFrame,
) -> pd.DataFrame:
    ltf_df = ltf_df.copy()
    ltf_df["timestamp"] = pd.to_datetime(ltf_df["timestamp"])

    def _bias(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        atr  = _atr(df, 14)
        body = (df["close"] - df["open"]).abs()
        impulse = body > 0.5 * atr
        bias = np.where(impulse & (df["close"] > df["open"]),  1,
               np.where(impulse & (df["close"] < df["open"]), -1, 0))
        df["htf_bias"] = pd.Series(bias, index=df.index)
        df["htf_bias"] = df["htf_bias"].replace(0, np.nan).ffill().fillna(0).astype(float)
        return df[["timestamp", "htf_bias"]]

    if h1_df is not None and len(h1_df) > 0:
        h1b = _bias(h1_df).rename(columns={"htf_bias": "htf_1h_bias"})
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h1b.sort_values("timestamp"),
            on="timestamp", direction="backward"
        )
    else:
        ltf_df["htf_1h_bias"] = 0.0

    if h4_df is not None and len(h4_df) > 0:
        h4b = _bias(h4_df).rename(columns={"htf_bias": "htf_4h_bias"})
        ltf_df = pd.merge_asof(
            ltf_df.sort_values("timestamp"),
            h4b.sort_values("timestamp"),
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
    """
    Full feature pipeline. Lower impulse_atr_multiplier = more zones detected.
    """
    logger.info(f"Building features for {len(df)} rows...")

    df = detect_zones(df, lookback=zone_lookback, impulse_atr_multiplier=impulse_atr_multiplier)
    df = add_confirmation_signals(df)
    df = add_indicators(df)

    if h1_df is not None or h4_df is not None:
        df = add_htf_context(df, h1_df, h4_df)

    warmup = max(200, zone_lookback)
    df = df.iloc[warmup:].reset_index(drop=True)

    logger.info(f"Features built — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature column list for ML
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Zone
    "demand_zone_strength", "demand_zone_fresh", "demand_zone_touches",
    "supply_zone_strength", "supply_zone_fresh", "supply_zone_touches",
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
    # HTF
    "htf_1h_bias", "htf_4h_bias", "htf_aligned",
    # Candle context
    "hour", "month",
    "direction", "candle_size", "body_size", "wick_upper", "wick_lower",
    "session",
]