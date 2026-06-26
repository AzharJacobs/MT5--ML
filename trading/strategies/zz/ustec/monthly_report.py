"""
Single backtest run with full monthly breakdown: PnL, WR%, RR, trade count.
Usage:
    python trading/strategies/zz/ustec/monthly_report.py \
        --start 2023-01-01 --end 2025-12-31 --fixed_lot 0.01 --cash 150
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT,
)
from trading.strategies.zz.ustec.engine import run_backtest

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    parser.add_argument("--dual_tf",          action="store_true")
    parser.add_argument("--h4_bias_gate_1h",  action="store_true")
    args = parser.parse_args()

    print(f"\nRunning backtest {args.start} to {args.end} ...")
    result = run_backtest(
        start=args.start,
        end=args.end,
        cash=args.cash,
        min_rr=MIN_RR,
        max_forward_bars=MAX_FORWARD_BARS,
        symbol="ustech",
        spread=SPREAD_PTS,
        fixed_lot=args.fixed_lot,
        directional_filter=True,
        allow_neutral=True,
        h4_swing_left=2,
        h4_swing_right=2,
        min_confirmations=1,
        excluded_from_count=list(EXCLUDED_FROM_COUNT),
        zone_max_losses=ZONE_MAX_LOSSES,
        h4_regime_filter=H4_REGIME_FILTER,
        min_sl_pct=MIN_SL_PCT,
        enable_trailing=ENABLE_TRAILING,
        be_trigger_pts=BE_TRIGGER_PTS,
        be_buffer_pts=BE_BUFFER_PTS,
        atr_trail_mult=ATR_TRAIL_MULT,
        dual_tf=args.dual_tf,
        h4_bias_gate_1h=args.h4_bias_gate_1h,
        silent=True,
    )

    if not result or isinstance(result, dict):
        print("ERROR: backtest returned no trades.")
        return

    metrics, df = result

    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["year"]    = df["exit_dt"].dt.year
    df["month"]   = df["exit_dt"].dt.month

    # Overall summary
    total   = len(df)
    wins    = (df["outcome"] == 1).sum()
    losses  = df[df["pnl"] < 0]
    winners = df[df["pnl"] > 0]
    wr_tot  = wins / total * 100 if total else 0
    avg_w   = winners["pnl"].mean() if len(winners) else 0
    avg_l   = losses["pnl"].mean()  if len(losses)  else 0
    rr_tot  = avg_w / abs(avg_l) if avg_l != 0 else 0
    net_tot = df["pnl"].sum()

    W = 9
    print()
    print("=" * 72)
    print(f"  USTEC Zone-to-Zone  |  {args.start} to {args.end}")
    print(f"  Cash: ${args.cash:,.0f}  |  Lot: {args.fixed_lot}"
          + ("  |  dual_tf" if args.dual_tf else "")
          + ("  |  bias_gate" if args.h4_bias_gate_1h else ""))
    print("=" * 72)
    print(f"  {'Month':<10} {'Trades':>{W}} {'Wins':>{W}} {'WR%':>{W}} "
          f"{'RR':>{W}} {'Net PnL':>{W+2}}")
    print("  " + "-" * 60)

    prev_year = None
    year_pnl  = {}
    year_tr   = {}
    year_wins = {}

    all_months = sorted(df.groupby(["year", "month"]).groups.keys())

    for y, m in all_months:
        g     = df[(df["year"] == y) & (df["month"] == m)]
        n     = len(g)
        w     = int((g["outcome"] == 1).sum())
        wr    = w / n * 100 if n else 0
        gw    = g[g["pnl"] > 0]
        gl    = g[g["pnl"] < 0]
        aw    = gw["pnl"].mean() if len(gw) else 0
        al    = gl["pnl"].mean() if len(gl) else 0
        rr    = aw / abs(al) if al != 0 else 0
        pnl   = g["pnl"].sum()

        year_pnl[y]  = year_pnl.get(y, 0)  + pnl
        year_tr[y]   = year_tr.get(y, 0)   + n
        year_wins[y] = year_wins.get(y, 0) + w

        if y != prev_year:
            if prev_year is not None:
                # Print year subtotal
                _yw  = year_wins.get(prev_year, 0)
                _yn  = year_tr.get(prev_year, 0)
                _ywr = _yw / _yn * 100 if _yn else 0
                _yp  = year_pnl.get(prev_year, 0)
                print("  " + "-" * 60)
                print(f"  {str(prev_year) + ' Total':<10} {_yn:>{W}} {_yw:>{W}} "
                      f"{_ywr:>{W}.1f} {'':>{W}} {'+' if _yp >= 0 else ''}{_yp:>{W}.2f}")
                print()
            prev_year = y
            print(f"  --- {y} ---")

        rr_str = f"{rr:.2f}" if rr > 0 else "-"
        pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.2f}"
        print(f"  {MONTH_NAMES[m] + ' ' + str(y):<10} {n:>{W}} {w:>{W}} "
              f"{wr:>{W}.1f} {rr_str:>{W}} {pnl_str:>{W+2}}")

    # Last year subtotal
    if prev_year is not None:
        _yw  = year_wins.get(prev_year, 0)
        _yn  = year_tr.get(prev_year, 0)
        _ywr = _yw / _yn * 100 if _yn else 0
        _yp  = year_pnl.get(prev_year, 0)
        print("  " + "-" * 60)
        print(f"  {str(prev_year) + ' Total':<10} {_yn:>{W}} {_yw:>{W}} "
              f"{_ywr:>{W}.1f} {'':>{W}} {'+' if _yp >= 0 else ''}{_yp:>{W}.2f}")

    # Grand total
    print()
    print("  " + "=" * 60)
    rr_str = f"{rr_tot:.2f}" if rr_tot > 0 else "-"
    net_str = f"{'+' if net_tot >= 0 else ''}{net_tot:.2f}"
    print(f"  {'TOTAL':<10} {total:>{W}} {int(wins):>{W}} "
          f"{wr_tot:>{W}.1f} {rr_str:>{W}} {net_str:>{W+2}}")
    print(f"  {'Start cash':<10} {'':>{W}} {'':>{W}} {'':>{W}} "
          f"{'':>{W}} {'$' + f'{args.cash:,.2f}':>{W+2}}")
    print(f"  {'Final equity':<10} {'':>{W}} {'':>{W}} {'':>{W}} "
          f"{'':>{W}} {metrics['final_equity']:>{W+2}}")
    print("  " + "=" * 60)
    print()


if __name__ == "__main__":
    main()
