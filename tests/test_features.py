"""
test_features.py — Tests that feature engineering produces correct output.
"""

import pandas as pd
import numpy as np
import pytest
from data.feature_engineer import build_features, FEATURE_COLUMNS


def _make_ohlcv(n: int = 400) -> pd.DataFrame:
    """
    Minimal OHLCV DataFrame that includes all DB-sourced columns
    build_features() expects to receive from the caller.
    n=400 so the 200-bar warmup slice leaves a non-empty result.
    """
    rng   = pd.date_range("2024-01-02 10:00", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    op    = close - 0.05
    hi    = close + 0.1
    lo    = close - 0.1
    return pd.DataFrame({
        "timestamp":    rng,
        "open":         op,
        "high":         hi,
        "low":          lo,
        "close":        close,
        "volume":       np.random.randint(100, 1000, n),
        # DB columns that build_features() passes through
        "hour":         rng.hour,
        "month":        rng.month,
        "candle_size":  hi - lo,
        "body_size":    np.abs(close - op),
        "wick_upper":   hi - np.maximum(close, op),
        "wick_lower":   np.minimum(close, op) - lo,
    }, index=rng)


def test_build_features_returns_dataframe():
    df = _make_ohlcv()
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)


def test_feature_columns_present():
    df = _make_ohlcv()
    result = build_features(df)
    missing = [c for c in FEATURE_COLUMNS if c not in result.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_no_all_nan_columns():
    df = _make_ohlcv()
    result = build_features(df)
    # These columns are legitimately NaN in minimal test data:
    #   - HTF zone boundaries require h1_df/h4_df input
    #   - Zone detection columns are NaN when no impulse candles form
    #     (synthetic flat data has body << 0.5*ATR so no zones are created)
    # The meaningful assertion is that always-computed columns are not NaN.
    skip_if_no_zones = {
        "htf_demand_zone_top", "htf_demand_zone_bottom",
        "htf_supply_zone_top", "htf_supply_zone_bottom",
        "in_demand_zone", "in_supply_zone", "between_zones",
        "demand_zone_top", "demand_zone_bottom", "demand_zone_strength",
        "demand_zone_fresh", "demand_zone_touches", "demand_zone_consolidation",
        "supply_zone_top", "supply_zone_bottom", "supply_zone_strength",
        "supply_zone_fresh", "supply_zone_touches", "supply_zone_consolidation",
        "nearest_demand_dist_atr", "nearest_supply_dist_atr",
        "demand_zone_quality", "supply_zone_quality", "active_zone_quality",
    }
    for col in FEATURE_COLUMNS:
        if col in result.columns and col not in skip_if_no_zones:
            assert not result[col].isna().all(), f"Column {col!r} is all NaN"
