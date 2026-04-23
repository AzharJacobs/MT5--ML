"""
test_features.py — Tests that feature engineering produces correct output.
"""

import pandas as pd
import numpy as np
import pytest
from data.feature_engineer import build_features, FEATURE_COLUMNS


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
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
    for col in FEATURE_COLUMNS:
        if col in result.columns:
            assert not result[col].isna().all(), f"Column {col!r} is all NaN"
