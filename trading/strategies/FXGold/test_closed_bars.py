#!/usr/bin/env python3
"""
Unit test for engine.closed_bars() — the look-ahead fix from Task 4.

Run directly:
    python -X utf8 trading/strategies/FXGold/test_closed_bars.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.FXGold.engine import closed_bars


def test_excludes_still_forming_bar() -> None:
    """
    Synthetic 4H bars, open-stamped (matches the verified MT5/DB convention).
    The last bar opened at 20:00 and only fully closes at 00:00 the next day.
    Evaluating at ts_now == 22:00 (2h into that bar) must exclude it; the
    prior bar, which closed at 20:00, must be included.
    """
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2025-01-01 08:00", "2025-01-01 12:00", "2025-01-01 16:00", "2025-01-01 20:00",
        ]),
        "open":  [100.0, 101.0, 102.0, 103.0],
        "high":  [100.5, 101.5, 102.5, 103.5],
        "low":   [99.5, 100.5, 101.5, 102.5],
        "close": [101.0, 102.0, 103.0, 103.8],
    })

    ts_now = pd.Timestamp("2025-01-01 22:00")   # bar at 20:00 is still open (closes at 00:00)
    result = closed_bars(df, "4H", ts_now)

    assert len(result) == 3, f"expected the still-forming 20:00 bar excluded, got {len(result)} rows"
    assert result["timestamp"].max() == pd.Timestamp("2025-01-01 16:00")

    # A naive `timestamp <= ts_now` filter WOULD have leaked the 20:00 bar in —
    # confirm that's exactly the bug this replaces.
    naive = df[df["timestamp"] <= ts_now]
    assert len(naive) == 4, "sanity check: the naive filter should leak the forming bar (that's the bug)"

    # Now advance ts_now to the moment the 20:00 bar actually closes.
    ts_closed = pd.Timestamp("2025-01-02 00:00")
    result_closed = closed_bars(df, "4H", ts_closed)
    assert len(result_closed) == 4, "bar should be included once it has fully closed"

    print("PASS: closed_bars excludes a still-forming higher-TF bar and includes it once closed")


def test_unknown_timeframe_raises() -> None:
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01"]),
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
    })
    try:
        closed_bars(df, "3W", pd.Timestamp("2025-01-01"))
    except ValueError:
        print("PASS: unknown timeframe raises ValueError")
        return
    raise AssertionError("expected ValueError for unknown timeframe")


if __name__ == "__main__":
    test_excludes_still_forming_bar()
    test_unknown_timeframe_raises()
    print("\nAll closed_bars tests passed.")
