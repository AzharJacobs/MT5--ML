"""
Monthly equity report — running balance carries over each month.

Usage:
    python trading/strategies/zz/ustec/equity_report.py \
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

BAD_BUY_HOURS  = [0, 2, 9, 11, 16, 21, 23]
BAD_SELL_HOURS = [1, 2, 4, 6, 11, 15, 19, 21]
MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=150.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    parser.add_argument("--year",      type=int,   default=None,
                        help="Only print this year's months (balance still carries from start)")
    parser.add_argument("--max_sl_pts", type=float, default=80.0,
                        help="Skip trades where SL distance exceeds this many points")
    args = parser.parse_args()

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
        block_gradual_long_hours=BAD_BUY_HOURS,
        block_gradual_short_hours=BAD_SELL_HOURS,
        split_exit=True,
        scalper_tp_pct=0.45,
        silent=True,
    )

    if not result or isinstance(result, dict):
        print("ERROR: no trades returned.")
        return

    metrics, df = result

    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["year"]    = df["exit_dt"].dt.year
    df["month"]   = df["exit_dt"].dt.month
    df = df.sort_values("exit_dt").reset_index(drop=True)

    total  = len(df)
    wins   = int((df["outcome"] == 1).sum())
    net    = df["pnl"].sum()

    # Build month table
    months = []
    for (y, m), g in df.groupby(["year", "month"]):
        mw   = int((g["outcome"] == 1).sum())
        mn   = len(g)
        mpnl = g["pnl"].sum()
        months.append({"year": y, "month": m, "trades": mn, "wins": mw, "pnl": mpnl})

    months.sort(key=lambda x: (x["year"], x["month"]))

    running = args.cash
    rows    = []
    for mo in months:
        running += mo["pnl"]
        rows.append({**mo, "balance": running})

    # Year subtotals
    from itertools import groupby as _gb

    # Header
    C1, C2, C3, C4, C5, C6, C7 = 12, 7, 5, 7, 12, 12, 12
    div  = "-" * (C1 + C2 + C3 + C4 + C5 + C6 + C7 + 14)
    hdiv = "=" * (C1 + C2 + C3 + C4 + C5 + C6 + C7 + 14)

    def hdr_row(m, tr, w, l, pnl, ret, bal, bold=False):
        wr = f"{w/max(tr,1)*100:.0f}%" if tr else "-"
        return (f"  {m:<{C1}} {tr:>{C2}} {w:>{C3}} {l:>{C4}} "
                f"  {pnl:>+{C5}.2f}   {ret:>+{C6}.2f}   {bal:>{C7}.2f}")

    print()
    print(hdiv)
    print(f"  USTEC Split-Exit + Dual-Side Hour Filter  |  {args.start} to {args.end}  |  Start: ${args.cash:.2f}  |  Lot: {args.fixed_lot}")
    print(f"  {total} trades  |  {wins}W / {total-wins}L  |  WR {wins/total*100:.1f}%  |  Net ${net:+.2f}  |  Final ${args.cash+net:.2f}")
    print(hdiv)
    print(f"  {'Month':<{C1}} {'Trades':>{C2}} {'W':>{C3}} {'L':>{C4}}   {'Month PnL':>{C5}}   {'Cumul PnL':>{C6}}   {'Balance':>{C7}}")
    print(div)

    # If year filter: only display that year's rows; balance still reflects full run
    display_rows = rows if not args.year else [r for r in rows if r["year"] == args.year]

    yr_trades = sum(r["trades"] for r in display_rows)
    yr_wins   = sum(r["wins"]   for r in display_rows)
    yr_pnl    = sum(r["pnl"]    for r in display_rows)

    for r in display_rows:
        lbl      = f"{MONTH_NAMES[r['month']]} {r['year']}"
        cumul    = r["balance"] - args.cash
        l        = r["trades"] - r["wins"]
        pnl_flag = ""
        if r["pnl"] > 200:    pnl_flag = "  <<"
        elif r["pnl"] < -200: pnl_flag = "  !!"
        print(f"  {lbl:<{C1}} {r['trades']:>{C2}} {r['wins']:>{C3}} {l:>{C4}}"
              f"   {r['pnl']:>+{C5}.2f}   {cumul:>+{C6}.2f}   {r['balance']:>{C7}.2f}{pnl_flag}")

    print(div)
    yr_l    = yr_trades - yr_wins
    yr_cum  = display_rows[-1]["balance"] - args.cash if display_rows else 0.0
    print(f"  {'  TOTAL':<{C1}} {yr_trades:>{C2}} {yr_wins:>{C3}} {yr_l:>{C4}}"
          f"   {yr_pnl:>+{C5}.2f}   {yr_cum:>+{C6}.2f}   {'':>{C7}}")

    print()
    print(hdiv)
    final_bal = args.cash + net
    print(f"  GRAND TOTAL  |  {total} trades  |  {wins}W/{total-wins}L  |  WR {wins/total*100:.1f}%  |  "
          f"Net ${net:+.2f}  |  Final balance ${final_bal:.2f}  |  "
          f"Max SL filter: {args.max_sl_pts:.0f}pts")
    print(hdiv)
    print()


if __name__ == "__main__":
    main()
