#!/usr/bin/env python3
"""
USTEC Zone-to-Zone backtest — config-driven CLI.

Defaults are loaded from config.yaml so this runner stays in sync with
live_bot_zz.py automatically.  Override any param on the command line for
tuning runs without touching the config.

Examples:
    python trading/strategies/zz/ustec/backtest.py --start 2023-01-01 --end 2024-01-01
    python trading/strategies/zz/ustec/backtest.py --start 2024-01-01 --end 2025-01-01
    python trading/strategies/zz/ustec/backtest.py --min_rr 2.0 --min_confirmations 2
    python trading/strategies/zz/ustec/backtest.py --midline_tp --chart
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.ustec.strategy import MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS
from trading.strategies.zz.ustec.engine import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: USTEC Zone-to-Zone (defaults from config.yaml)"
    )
    parser.add_argument("--start",  default="2023-01-01")
    parser.add_argument("--end",    default="2024-01-01")
    parser.add_argument("--cash",   type=float, default=10_000.0)
    parser.add_argument("--min_rr", type=float, default=MIN_RR)
    parser.add_argument("--max_bars", type=int, default=MAX_FORWARD_BARS)
    parser.add_argument("--spread",   type=float, default=SPREAD_PTS)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    # Step 2
    parser.add_argument("--no_directional_filter", action="store_true")
    parser.add_argument("--no_neutral",            action="store_true")
    parser.add_argument("--h4_swing_left",  type=int, default=2)
    parser.add_argument("--h4_swing_right", type=int, default=2)
    # Step 3
    parser.add_argument("--min_confirmations", type=int, default=1)
    parser.add_argument("--aggressive_boundary", action="store_true")
    parser.add_argument("--bos_msb_min1", action="store_true",
                        help="Allow single-confirmation entries when that confirmation is bos_msb")
    # Step 4
    parser.add_argument("--aggressive_entry", action="store_true")
    parser.add_argument("--midline_tp",  action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--sl_buffer",   type=float, default=0.002)
    # Cooldown
    parser.add_argument("--no_leave_return", action="store_true")
    parser.add_argument("--cooldown_bars",   type=int, default=15)
    # Trade management (R-multiple based; ATR/swing trailing is config.yaml-only, not CLI-exposed)
    parser.add_argument("--be_trigger_r",   type=float, default=0.0)
    parser.add_argument("--lock_trigger_r", type=float, default=0.0)
    parser.add_argument("--be_buffer_pts",  type=float, default=5.0)
    # Data source / zones
    parser.add_argument("--data_source", default="db", choices=["db", "mt5"])
    parser.add_argument("--dual_tf",  action="store_true",
                        help="Watch 1H and 4H zones in parallel; 4H takes priority on overlap")
    parser.add_argument("--max_rr", type=float, default=0.0,
                        help="Skip setups with theoretical RR above this (0 = off)")
    # Output
    parser.add_argument("--save",  default=None)
    parser.add_argument("--chart", action="store_true")

    args = parser.parse_args()

    run_backtest(
        start=args.start,
        end=args.end,
        cash=args.cash,
        min_rr=args.min_rr,
        max_forward_bars=args.max_bars,
        symbol="ustech",
        spread=args.spread,
        fixed_lot=args.fixed_lot,
        directional_filter=not args.no_directional_filter,
        allow_neutral=not args.no_neutral,
        h4_swing_left=args.h4_swing_left,
        h4_swing_right=args.h4_swing_right,
        min_confirmations=args.min_confirmations,
        aggressive_boundary=args.aggressive_boundary,
        bos_msb_min1=args.bos_msb_min1,
        aggressive_entry=args.aggressive_entry,
        midline_tp=args.midline_tp,
        midline_pct=args.midline_pct,
        sl_buffer_pct=args.sl_buffer,
        require_leave_and_return=not args.no_leave_return,
        cooldown_bars=args.cooldown_bars,
        be_trigger_r=args.be_trigger_r,
        lock_trigger_r=args.lock_trigger_r,
        be_buffer_pts=args.be_buffer_pts,
        data_source=args.data_source,
        use_1h_zones=args.dual_tf,
        max_rr=args.max_rr,
        save_path=args.save,
        chart=args.chart,
    )


if __name__ == "__main__":
    main()
