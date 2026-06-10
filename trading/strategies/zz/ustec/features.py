"""
USTEC feature engineering.

The core ZZ strategy (zones + confirmations + trade_setup) operates on raw
OHLCV without ML features.  This module re-exports the shared feature
engineering layer for use in analysis scripts or future ML integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.feature_engineer import build_features  # noqa: F401

__all__ = ["build_features"]
