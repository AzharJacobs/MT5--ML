"""
feature_engineer.py — Technical indicator feature engineering.

Zone-to-Zone specific features (detect_zones, add_zone_quality,
add_strategy_rules, add_confirmation_signals, HTF zone extraction)
have been removed. They will be rebuilt from scratch in
zone_detection_Z&Z.py and related Z&Z pipeline files.
"""

import pandas as pd
import numpy as np
import logging

from config.pipeline_config import REQUIRED_FEATURE_COLUMNS

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
# Master Builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    h1_df: pd.DataFrame = None,
    h4_df: pd.DataFrame = None,
    zone_lookback: int = 30,
    impulse_atr_multiplier: float = 0.5,
    include_london_ny: bool = True,
) -> pd.DataFrame:
    logger.info(f"Building features for {len(df)} rows...")

    df = add_indicators(df)

    # Cyclical hour-of-day encoding
    if "timestamp" in df.columns:
        hour_f = pd.to_datetime(df["timestamp"]).dt.hour.astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * hour_f / 24.0).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * hour_f / 24.0).astype(np.float32)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 1.0

    if "hour" in df.columns:
        hour_s   = df["hour"].fillna(-1).astype(float)
        minute_s = (pd.to_datetime(df["timestamp"]).dt.minute
                    if "timestamp" in df.columns
                    else pd.Series(0, index=df.index))

        london_open_mask = (hour_s == 10) | (hour_s == 11) | ((hour_s == 12) & (minute_s < 30))
        ny_open_mask     = (hour_s == 13) | (hour_s == 14)
        overlap_mask     = (hour_s == 16)

        session_mask = london_open_mask | ny_open_mask
        if include_london_ny:
            session_mask = session_mask | overlap_mask
        df["in_session"] = session_mask.astype(float)

        df["session_id"] = np.where(
            london_open_mask, 1.0,
            np.where(ny_open_mask, 2.0,
            np.where(overlap_mask & include_london_ny, 3.0, 0.0))
        ).astype(float)
    else:
        df["in_session"] = 0.0
        df["session_id"] = 0.0

    warmup = max(200, zone_lookback)
    df = df.iloc[warmup:].reset_index(drop=True)

    logger.info(f"Features built — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature column list for ML
# Re-exported from config so callers using `from data.feature_engineer import
# FEATURE_COLUMNS` continue to work unchanged.
# The Z&Z pipeline will populate REQUIRED_FEATURE_COLUMNS in pipeline_config.py.
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = REQUIRED_FEATURE_COLUMNS
