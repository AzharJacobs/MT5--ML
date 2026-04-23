"""
test_strategy.py — Tests that strategy signals are generated correctly.
"""

import pandas as pd
import numpy as np
import pytest
from strategy.signal_generator import generate_labels


def _make_featured_df(n: int = 300) -> pd.DataFrame:
    rng = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "volume": np.random.randint(100, 1000, n),
        "atr": np.full(n, 0.2),
        "in_demand_zone": np.random.randint(0, 2, n),
        "in_supply_zone": np.random.randint(0, 2, n),
    }, index=rng)
    return df


def test_generate_labels_returns_series():
    df = _make_featured_df()
    labels = generate_labels(df)
    assert isinstance(labels, pd.Series)


def test_labels_are_binary():
    df = _make_featured_df()
    labels = generate_labels(df)
    unique = set(labels.dropna().unique())
    assert unique.issubset({0, 1}), f"Unexpected label values: {unique}"
