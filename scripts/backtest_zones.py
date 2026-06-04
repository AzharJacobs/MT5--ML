#!/usr/bin/env python3
"""
Backtest entry point: Zone-Based Strategy (Steps 1–4).

Examples:
    python scripts/backtest_zones.py --start 2023-01-01 --end 2024-01-01
    python scripts/backtest_zones.py --start 2024-01-01 --end 2025-01-01
    python scripts/backtest_zones.py --start 2025-01-01 --end 2026-01-01
    python scripts/backtest_zones.py --min_rr 2.0 --min_confirmations 2
    python scripts/backtest_zones.py --midline_tp --chart
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.engine_zones import main

if __name__ == "__main__":
    main()
