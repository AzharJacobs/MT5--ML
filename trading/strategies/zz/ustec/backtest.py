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

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT,
)
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
    parser.add_argument("--exclude_signals",
                        default=",".join(EXCLUDED_FROM_COUNT),
                        help="Comma-separated signal names to exclude from count")
    parser.add_argument("--zone_max_losses", type=int, default=ZONE_MAX_LOSSES)
    parser.add_argument("--dir_max_losses",   type=int, default=0)
    parser.add_argument("--dir_cooldown",     type=int, default=48)
    parser.add_argument("--h4_regime_filter",    dest="h4_regime_filter", action="store_true",  default=H4_REGIME_FILTER)
    parser.add_argument("--no_h4_regime_filter", dest="h4_regime_filter", action="store_false")
    # Step 4
    parser.add_argument("--aggressive_entry", action="store_true")
    parser.add_argument("--midline_tp",  action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--sl_buffer",   type=float, default=0.002)
    parser.add_argument("--min_sl_pct",  type=float, default=MIN_SL_PCT)
    # Cooldown
    parser.add_argument("--no_leave_return", action="store_true")
    parser.add_argument("--cooldown_bars",   type=int, default=15)
    # Trailing stop
    parser.add_argument("--trailing",    dest="enable_trailing", action="store_true",  default=ENABLE_TRAILING)
    parser.add_argument("--no_trailing", dest="enable_trailing", action="store_false")
    parser.add_argument("--be_trigger",  type=float, default=BE_TRIGGER_PTS)
    parser.add_argument("--be_buffer",   type=float, default=BE_BUFFER_PTS)
    parser.add_argument("--atr_mult",    type=float, default=ATR_TRAIL_MULT)
    # Output
    parser.add_argument("--save",  default=None)
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--dual_tf",          action="store_true",
                        help="Watch 1H and 4H zones in parallel; 4H takes priority on overlap")
    parser.add_argument("--h4_bias_gate_1h", action="store_true",
                        help="Require 4H bias to agree before taking a 1H-sourced signal")

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
        excluded_from_count=[s.strip() for s in args.exclude_signals.split(",") if s.strip()],
        aggressive_entry=args.aggressive_entry,
        midline_tp=args.midline_tp,
        midline_pct=args.midline_pct,
        sl_buffer_pct=args.sl_buffer,
        require_leave_and_return=not args.no_leave_return,
        cooldown_bars=args.cooldown_bars,
        zone_max_losses=args.zone_max_losses,
        dir_max_losses=args.dir_max_losses,
        dir_cooldown_bars=args.dir_cooldown,
        h4_regime_filter=args.h4_regime_filter,
        min_sl_pct=args.min_sl_pct,
        enable_trailing=args.enable_trailing,
        be_trigger_pts=args.be_trigger,
        be_buffer_pts=args.be_buffer,
        atr_trail_mult=args.atr_mult,
        realistic=args.realistic,
        save_path=args.save,
        chart=args.chart,
        dual_tf=args.dual_tf,
        h4_bias_gate_1h=args.h4_bias_gate_1h,
    )


if __name__ == "__main__":
    main()
