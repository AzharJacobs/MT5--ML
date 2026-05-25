#!/usr/bin/env python3
"""
Backtest entry point: Market Structure Trading Plan (PDF strategy).

Examples:
    python scripts/backtest_market_structure.py
    python scripts/backtest_market_structure.py --start 2023-01-01 --end 2024-12-31
    python scripts/backtest_market_structure.py --cash 50 --min_rr 2.0
    python scripts/backtest_market_structure.py --save results/ms_backtest.json
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.engine_market_structure import main

if __name__ == "__main__":
    main()
