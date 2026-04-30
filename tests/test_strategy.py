"""
test_strategy.py — Tests that strategy signals are generated correctly.
"""

import pandas as pd
import numpy as np
import pytest
from strategy.signal_generator import generate_labels


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_featured_df(n: int = 300) -> pd.DataFrame:
    """Minimal featured DataFrame with all columns generate_labels() needs."""
    rng    = pd.date_range("2024-01-02 10:00", periods=n, freq="5min")
    close  = 100 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "timestamp":        rng,
        "open":             close - 0.05,
        "high":             close + 0.1,
        "low":              close - 0.1,
        "close":            close,
        "volume":           np.random.randint(100, 1000, n),
        "hour":             10,                              # always in session
        "atr_14":           np.full(n, 0.2),
        "volume_ratio":     np.full(n, 1.0),
        "between_zones":    np.zeros(n),
        "in_demand_zone":   np.zeros(n),
        "in_supply_zone":   np.zeros(n),
        "demand_zone_bottom": np.full(n, np.nan),
        "demand_zone_top":    np.full(n, np.nan),
        "supply_zone_bottom": np.full(n, np.nan),
        "supply_zone_top":    np.full(n, np.nan),
    })


def _make_sell_win_df() -> pd.DataFrame:
    """
    Synthetic DataFrame guaranteed to contain at least one sell winner.

    Setup (row 50):
      close           = 1000
      supply_zone_top = 1002   →  SL = 1002 + 0.5*10 = 1007  (risk = 7)
      demand_zone_top = 940    →  TP midline = 1000 - 30 = 970 (reward = 30)
      RR = 30 / 7 = 4.28  ≥ min_rr=1.5  ✓
      reward/atr = 3.0  ≤ MAX_TP_ATR=6.0  ✓
      risk/atr   = 0.7  ≤ MAX_SL_ATR=3.0  ✓

    Rows 51-55: low = 965 (below TP=970) → outcome = 1 → label = 1
    """
    n   = 200
    atr = 10.0

    entry_close = 1000.0
    close_arr   = np.full(n, entry_close)
    high_arr    = close_arr + 5.0
    low_arr     = close_arr - 5.0

    # Forward bars: price drops below TP
    for i in range(51, 56):
        low_arr[i] = 965.0

    df = pd.DataFrame({
        "timestamp":          pd.date_range("2024-01-02 10:00", periods=n, freq="5min"),
        "open":               close_arr - 2.0,
        "high":               high_arr,
        "low":                low_arr,
        "close":              close_arr,
        "volume":             1000,
        "hour":               10,
        "atr_14":             atr,
        "volume_ratio":       1.0,
        "between_zones":      0.0,
        "in_demand_zone":     0.0,
        "in_supply_zone":     0.0,
        "demand_zone_bottom": np.nan,
        "demand_zone_top":    np.nan,
        "supply_zone_bottom": np.nan,
        "supply_zone_top":    np.nan,
    })

    # Sell entry row
    df.at[50, "in_supply_zone"] = 1.0
    df.at[50, "supply_zone_top"] = 1002.0
    df.at[50, "demand_zone_top"] = 940.0

    return df


def _make_buy_win_df() -> pd.DataFrame:
    """
    Synthetic DataFrame guaranteed to contain at least one buy winner.

    Setup (row 50):
      close              = 1000
      demand_zone_bottom = 990     →  SL = 990 - 0.5*10 = 985   (risk = 15)
      supply_zone_bottom = 1100    →  TP midline = 1000 + 50 = 1050 (reward = 50)
      RR = 50/15 = 3.33  ✓
    Rows 51-55: high = 1055 (above TP=1050) → outcome = 1
    """
    n   = 200
    atr = 10.0

    entry_close = 1000.0
    close_arr   = np.full(n, entry_close)
    high_arr    = close_arr + 5.0
    low_arr     = close_arr - 5.0

    for i in range(51, 56):
        high_arr[i] = 1055.0

    df = pd.DataFrame({
        "timestamp":          pd.date_range("2024-01-02 10:00", periods=n, freq="5min"),
        "open":               close_arr - 2.0,
        "high":               high_arr,
        "low":                low_arr,
        "close":              close_arr,
        "volume":             1000,
        "hour":               10,
        "atr_14":             atr,
        "volume_ratio":       1.0,
        "between_zones":      0.0,
        "in_demand_zone":     0.0,
        "in_supply_zone":     0.0,
        "demand_zone_bottom": np.nan,
        "demand_zone_top":    np.nan,
        "supply_zone_bottom": np.nan,
        "supply_zone_top":    np.nan,
    })

    df.at[50, "in_demand_zone"]     = 1.0
    df.at[50, "demand_zone_bottom"] = 990.0
    df.at[50, "supply_zone_bottom"] = 1100.0

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_labels_returns_dataframe():
    """generate_labels() must return a DataFrame (not a Series)."""
    df     = _make_featured_df()
    result = generate_labels(df)
    assert isinstance(result, pd.DataFrame), (
        f"Expected pd.DataFrame, got {type(result)}"
    )


def test_labels_are_binary():
    """All label values must be in {0, 1}."""
    df     = _make_featured_df()
    result = generate_labels(df)
    unique = set(result["label"].dropna().unique())
    assert unique.issubset({0, 1}), f"Unexpected label values: {unique}"


def test_sell_wins_nonzero():
    """
    Regression guard for the sell_wins=0 bug.

    With correct binary labels (1=winner regardless of direction),
    a DataFrame that has valid sell setups followed by declining price
    must produce at least one sell winner.

    If this test fails it means the label fix was reverted or broken:
    sell winners are being assigned label=-1 instead of label=1.
    """
    df     = _make_sell_win_df()
    result = generate_labels(df, include_london_ny=True)

    sell_wins = ((result["label"] == 1) & (result["signal_direction"] == -1)).sum()
    assert sell_wins > 0, (
        f"sell_wins=0 — binary label fix may be broken. "
        f"Signal counts: {result['signal'].value_counts().to_dict()}, "
        f"label counts: {result['label'].value_counts().to_dict()}"
    )


def test_buy_wins_nonzero():
    """Mirror of test_sell_wins_nonzero() for buy direction."""
    df     = _make_buy_win_df()
    result = generate_labels(df, include_london_ny=True)

    buy_wins = ((result["label"] == 1) & (result["signal_direction"] == 1)).sum()
    assert buy_wins > 0, (
        f"buy_wins=0 — check label generation. "
        f"Signal counts: {result['signal'].value_counts().to_dict()}, "
        f"label counts: {result['label'].value_counts().to_dict()}"
    )
