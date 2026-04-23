#!/usr/bin/env python3
"""
backtest.py — Entry point: runs walk-forward backtest via Backtrader.

Usage:
    python backtest.py --timeframe 5min --cash 10000 --stake 0.15
    python backtest.py --timeframe 1H --confidence 0.52
"""

from backtest.engine import main

if __name__ == "__main__":
    main()
