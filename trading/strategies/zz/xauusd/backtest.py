#!/usr/bin/env python3
"""
XAUUSD (Gold) Zone-to-Zone backtest — config-driven CLI.

All three gold fixes are on by default (loaded from config.yaml).
Override any param on the command line for tuning runs.

Examples:
    python trading/strategies/zz/xauusd/backtest.py --start 2023-01-01 --end 2024-01-01
    python trading/strategies/zz/xauusd/backtest.py --start 2025-01-01 --end 2026-01-01
    python trading/strategies/zz/xauusd/backtest.py --min_zone_atr_frac 0.20 --sl_atr_buffer 0.15
    python trading/strategies/zz/xauusd/backtest.py --no_failed_zone_filter --start 2023-01-01
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.xauusd.strategy import make_gold_config, load_raw, MIN_RR
from trading.strategies.zz.xauusd.engine import run_backtest_gold


def main() -> None:
    _raw = load_raw()
    _gf  = _raw["gold_fixes"]
    _c   = _raw["confirmations"]
    _cool = _raw["cooldown"]

    parser = argparse.ArgumentParser(
        description="Backtest: Gold Z&Z v2 (XAUUSD, three gold fixes — defaults from config.yaml)"
    )
    # Run params
    parser.add_argument("--start",  default="2023-01-01")
    parser.add_argument("--end",    default="2024-01-01")
    parser.add_argument("--cash",   type=float, default=10_000.0)
    parser.add_argument("--save",   default=None)
    parser.add_argument("--chart",  action="store_true")
    parser.add_argument("--data_source", default="db", choices=["db", "mt5"])
    # Fix 1
    parser.add_argument("--no_failed_zone_filter", action="store_true",
                        help="Disable permanent post-loss zone blacklist (revert to cooldown)")
    # Fix 2
    parser.add_argument("--active_signals", nargs="+",
                        default=_c["active_signals"],
                        help="Signals that count toward min_confirmations for entry")
    parser.add_argument("--min_confirmations", type=int, default=int(_c["min_confirmations"]))
    # Fix 3
    parser.add_argument("--min_zone_atr_frac", type=float, default=float(_gf["min_zone_atr_frac"]),
                        help="Skip zones narrower than this × H4 ATR(14)")
    parser.add_argument("--sl_atr_buffer",     type=float, default=float(_gf["sl_atr_buffer"]),
                        help="Widen SL by this × H4 ATR(14) beyond zone edge")
    # Strategy params
    parser.add_argument("--min_rr",    type=float, default=MIN_RR)
    parser.add_argument("--spread",    type=float, default=float(_raw["spread_pts"]))
    parser.add_argument("--fixed_lot", type=float, default=float(_raw["fixed_lot"]))
    parser.add_argument("--midline_tp",  action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--no_leave_return", action="store_true")
    parser.add_argument("--cooldown_bars",   type=int, default=int(_cool["cooldown_bars"]))
    # D1 trend filter / session window (ported from USTEC; off by default)
    parser.add_argument("--d1_trend_filter", action="store_true")
    parser.add_argument("--trading_hours", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="e.g. --trading_hours 8 18")

    args = parser.parse_args()

    # Start from the yaml-driven defaults then apply CLI overrides
    cfg = make_gold_config()
    from dataclasses import replace
    cfg = replace(
        cfg,
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
        d1_trend_filter=args.d1_trend_filter,
        trading_hours=tuple(args.trading_hours) if args.trading_hours else None,
    )

    run_backtest_gold(
        cfg=cfg,
        start=args.start,
        end=args.end,
        cash=args.cash,
        save_path=args.save,
        chart=args.chart,
        data_source=args.data_source,
    )


if __name__ == "__main__":
    main()
