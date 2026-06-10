"""
XAUUSD (Gold) feature engineering.

The core ZZ strategy operates on raw OHLCV without ML features.
This module re-exports the shared feature engineering layer and the
ATR helper used internally by the gold engine (Fix 3a/3b).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.feature_engineer import build_features  # noqa: F401


def h4_atr14(df_h4: pd.DataFrame) -> float:
    """14-period ATR on an H4 window (Wilder TR). Used by Fix 3a/3b."""
    hi = df_h4["high"].values
    lo = df_h4["low"].values
    cl = df_h4["close"].values
    if len(hi) < 2:
        return float("nan")
    tr = np.maximum(
        hi[1:] - lo[1:],
        np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])),
    )
    n = min(14, len(tr))
    return float(np.mean(tr[-n:]))


__all__ = ["build_features", "h4_atr14"]
