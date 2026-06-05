#!/usr/bin/env python3
"""
backtest_gold_ZZ.py — CLI for the gold-specific Zone-to-Zone backtest.

All three gold fixes are on by default. Every fix parameter is individually
overridable from the command line for tuning.

Examples:
    python scripts/backtest_gold_ZZ.py --start 2023-01-01 --end 2024-01-01
    python scripts/backtest_gold_ZZ.py --start 2025-01-01 --end 2026-01-01
    python scripts/backtest_gold_ZZ.py --min_zone_atr_frac 0.20 --sl_atr_buffer 0.15
    python scripts/backtest_gold_ZZ.py --no_failed_zone_filter --start 2023-01-01
    python scripts/backtest_gold_ZZ.py --active_signals engulfing rejection_wick
"""

import sys
import os
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from strategy_v2.config_ZZ import GoldZZConfig
from strategy_v2.engine_ZZ import run_backtest_gold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: Gold Z&Z v2 (XAUUSD, three gold-specific fixes)"
    )
    # Run params
    parser.add_argument("--start",  default="2023-01-01")
    parser.add_argument("--end",    default="2024-01-01")
    parser.add_argument("--cash",   type=float, default=10_000.0)
    parser.add_argument("--save",   default=None)
    parser.add_argument("--chart",  action="store_true")
    # Fix 1
    parser.add_argument("--no_failed_zone_filter", action="store_true",
                        help="Disable permanent post-loss zone blacklist (revert to 48-bar cooldown)")
    # Fix 2
    parser.add_argument("--active_signals", nargs="+",
                        default=["engulfing", "rejection_wick", "choch"],
                        help="Signals that count toward min_confirmations for entry "
                             "(default: engulfing rejection_wick choch)")
    parser.add_argument("--min_confirmations", type=int, default=1)
    # Fix 3
    parser.add_argument("--min_zone_atr_frac", type=float, default=0.30,
                        help="Skip zones narrower than this × H4 ATR(14) (default 0.30)")
    parser.add_argument("--sl_atr_buffer",     type=float, default=0.20,
                        help="Widen SL by this × H4 ATR(14) beyond zone edge (default 0.20)")
    # Passthrough strategy params
    parser.add_argument("--min_rr",    type=float, default=1.5)
    parser.add_argument("--spread",    type=float, default=0.0)
    parser.add_argument("--fixed_lot", type=float, default=0.0)
    parser.add_argument("--midline_tp",  action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--no_leave_return", action="store_true")
    parser.add_argument("--cooldown_bars",   type=int, default=15)

    args = parser.parse_args()

    cfg = GoldZZConfig(
        failed_zone_filter=not args.no_failed_zone_filter,
        active_signals=frozenset(args.active_signals),
        min_confirmations=args.min_confirmations,
        min_zone_atr_frac=args.min_zone_atr_frac,
        sl_atr_buffer=args.sl_atr_buffer,
        min_rr=args.min_rr,
        spread=args.spread,
        fixed_lot=args.fixed_lot,
        midline_tp=args.midline_tp,
        midline_pct=args.midline_pct,
        require_leave_and_return=not args.no_leave_return,
        cooldown_bars=args.cooldown_bars,
    )

    run_backtest_gold(
        cfg=cfg,
        start=args.start,
        end=args.end,
        cash=args.cash,
        save_path=args.save,
        chart=args.chart,
    )


if __name__ == "__main__":
    main()
