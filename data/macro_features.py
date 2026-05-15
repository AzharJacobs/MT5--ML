"""
data/macro_features.py — Macro/regime features for RL training only.

Gives the RL agent awareness of broader market context that the ML bot
lacks: trend structure (HH/HL vs LH/LL), trend strength (ADX), EMA
regime, volatility regime, and range position. These are what caused
the ML bot to buy into downtrends — it had no way to see the macro picture.

NOT added to REQUIRED_FEATURE_COLUMNS — the live trading bot is untouched.

Usage in RL training pipeline:
    from data.feature_engineer import build_features
    from data.macro_features import add_macro_features, RL_MACRO_FEATURE_COLUMNS

    df = build_features(raw_df, ...)
    df = add_macro_features(df)
    # df now has 51 + 15 = 66 features per timeframe
"""

import numpy as np
import pandas as pd

RL_MACRO_FEATURE_COLUMNS = [
    # Trend displacement — how far has price moved, ATR-normalised
    "macro_trend_20",           # 20-bar price displacement / ATR
    "macro_trend_50",           # 50-bar price displacement / ATR

    # Candle structure
    "macro_bull_pct_20",        # fraction of last 20 bars that were bullish candles

    # Swing structure — is market making HH/HL or LH/LL?
    "macro_hh_hl",              # 1 if higher high AND higher low vs 10 bars ago
    "macro_lh_ll",              # 1 if lower high AND lower low vs 10 bars ago

    # Trend strength (ADX)
    "macro_adx_14",             # ADX(14) normalised to [0, 1]
    "macro_trending",           # 1 if ADX > 25 (market is trending, not ranging)

    # EMA regime — where is price relative to medium/long-term mean?
    "macro_price_vs_ema50",     # (close - EMA50) / ATR
    "macro_price_vs_ema200",    # (close - EMA200) / ATR
    "macro_ema_alignment",      # 1 = full bull (EMA20>50>200), -1 = full bear, 0 = mixed

    # Volatility regime — is the market expanding or contracting?
    "macro_atr_ratio",          # ATR(14) / ATR(14).rolling(50).mean()

    # Range context — where is price in the recent swing?
    "macro_swing_high_dist",    # (20-bar high - close) / ATR (overhead resistance)
    "macro_swing_low_dist",     # (close - 20-bar low) / ATR (below support)
    "macro_range_pos_50",       # (close - 50-bar low) / (50-bar high - low): 0=bottom, 1=top

    # Longer-term momentum
    "macro_momentum_20",        # (close - close[20]) / ATR
]


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX. Returns raw 0-100 values."""
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    close = df["close"].astype(float)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean().replace(0, np.nan)

    h_diff = high.diff()
    l_diff = -low.diff()

    plus_dm  = np.where((h_diff > l_diff) & (h_diff > 0), h_diff, 0.0)
    minus_dm = np.where((l_diff > h_diff) & (l_diff > 0), l_diff, 0.0)

    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.fillna(0.0)


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro/regime features to an already-built feature DataFrame.
    Requires: close, high, low, open, atr_14.
    Call after build_features().
    """
    df    = df.copy()
    close = df["close"].astype(float)
    high  = df["high"].astype(float)
    low   = df["low"].astype(float)
    atr   = df["atr_14"].replace(0, np.nan).astype(float)

    # ── EMA backbone ──────────────────────────────────────────────────────────
    ema20  = _ema(close, 20)
    ema50  = _ema(close, 50)
    ema200 = _ema(close, 200)

    # ── Trend displacement ────────────────────────────────────────────────────
    df["macro_trend_20"]    = ((close - close.shift(20)) / atr).clip(-10, 10).fillna(0.0)
    df["macro_trend_50"]    = ((close - close.shift(50)) / atr).clip(-10, 10).fillna(0.0)
    df["macro_momentum_20"] = df["macro_trend_20"]

    # ── Candle direction fraction ─────────────────────────────────────────────
    bull_bar = (close > df["open"].astype(float)).astype(float)
    df["macro_bull_pct_20"] = bull_bar.rolling(20).mean().fillna(0.5)

    # ── Swing structure ───────────────────────────────────────────────────────
    df["macro_hh_hl"] = (
        (high > high.shift(10)) & (low > low.shift(10))
    ).astype(float).fillna(0.0)

    df["macro_lh_ll"] = (
        (high < high.shift(10)) & (low < low.shift(10))
    ).astype(float).fillna(0.0)

    # ── ADX ───────────────────────────────────────────────────────────────────
    adx = _adx(df, 14)
    df["macro_adx_14"]   = (adx / 100.0).clip(0.0, 1.0)
    df["macro_trending"] = (adx > 25).astype(float)

    # ── EMA regime ────────────────────────────────────────────────────────────
    df["macro_price_vs_ema50"]  = ((close - ema50)  / atr).clip(-10, 10).fillna(0.0)
    df["macro_price_vs_ema200"] = ((close - ema200) / atr).clip(-10, 10).fillna(0.0)

    alignment = np.where(
        (ema20 > ema50) & (ema50 > ema200),  1.0,
        np.where((ema20 < ema50) & (ema50 < ema200), -1.0, 0.0)
    )
    df["macro_ema_alignment"] = pd.Series(alignment, index=df.index).fillna(0.0)

    # ── Volatility regime ─────────────────────────────────────────────────────
    atr_ma50 = atr.rolling(50).mean().replace(0, np.nan)
    df["macro_atr_ratio"] = (atr / atr_ma50).clip(0.0, 3.0).fillna(1.0)

    # ── Range context ─────────────────────────────────────────────────────────
    high20  = high.rolling(20).max()
    low20   = low.rolling(20).min()
    high50  = high.rolling(50).max()
    low50   = low.rolling(50).min()
    range50 = (high50 - low50).replace(0, np.nan)

    df["macro_swing_high_dist"] = ((high20 - close) / atr).clip(0.0, 10.0).fillna(0.0)
    df["macro_swing_low_dist"]  = ((close - low20)  / atr).clip(0.0, 10.0).fillna(0.0)
    df["macro_range_pos_50"]    = ((close - low50)  / range50).clip(0.0, 1.0).fillna(0.5)

    return df
