#!/usr/bin/env python3
"""
monthly_backtest.py — Month-by-month equity walk on USTEC ZZ baseline.

Runs a single continuous backtest (equity compounds across months).
Slices df_t by the month trades were ENTERED to report per-month stats.

Sizing : 1% fixed-fractional  (lot = equity×0.01 ÷ SL_pts×contract)
         At $150 this clamps to the 0.01-lot floor until equity grows.

Usage:
  python trading/strategies/zz/ustec/monthly_backtest.py
  python trading/strategies/zz/ustec/monthly_backtest.py --cash 150 --start 2023-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.ustec.strategy import MIN_RR, MIN_SL_PCT, SPREAD_PTS, MAX_FORWARD_BARS
from trading.strategies.zz.ustec.engine import run_backtest


def run_monthly(start: str, end: str, cash: float, spread: float, risk_pct: float) -> None:

    print(f"\nMonthly backtest  {start} → {end}")
    print(f"  Cash=${cash:.2f}  risk={risk_pct*100:.2f}%  spread={spread}pts")
    print(f"  Single continuous run — equity compounds across months …\n")

    m, df = run_backtest(
        start=start, end=end, cash=cash,
        symbol="ustech", spread=spread,
        fixed_lot=0, risk_pct=risk_pct,
        min_rr=MIN_RR, min_sl_pct=MIN_SL_PCT, max_forward_bars=MAX_FORWARD_BARS,
        gradual_filter="all",
        silent=True,
    )

    if df.empty:
        print("  No trades generated.")
        return

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    all_months = sorted(df["month"].unique())

    # Track running equity: start of each month = equity after last closed trade
    # df["equity"] = equity after each trade closes (in chronological order)
    # We step through chronologically and assign start/end equity per month.
    df_sorted = df.sort_values("date").reset_index(drop=True)

    # Build a dict: month → list of rows
    from collections import defaultdict
    monthly = defaultdict(list)
    for _, row in df_sorted.iterrows():
        monthly[row["month"]].append(row)

    W   = 12
    SEP = "─" * (12 + W * 7 + 6)

    print(SEP)
    print(f"  {'Month':<10} {'Trades':>{W}} {'Wins':>{W}} {'WR%':>{W}} "
          f"{'Month PnL':>{W}} {'End Equity':>{W}} {'DD%':>{W}}")
    print(SEP)

    running_eq = cash
    peak_eq    = cash
    total_trades = 0
    total_wins   = 0

    for period in all_months:
        rows = monthly[period]
        month_pnl   = sum(r["pnl"] for r in rows)
        trades      = len(rows)
        wins        = sum(1 for r in rows if r["outcome"] == 1)
        wr          = wins / trades * 100 if trades else 0.0
        start_eq    = running_eq
        end_eq      = running_eq + month_pnl
        if end_eq < 0:
            end_eq = 0.0
        running_eq  = end_eq
        peak_eq     = max(peak_eq, running_eq)
        dd_pct      = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0
        total_trades += trades
        total_wins   += wins

        blown = " ⚠BLOWN" if running_eq <= 0.01 else ""
        print(f"  {str(period):<10} {trades:>{W}} {wins:>{W}} {wr:>{W-1}.1f}% "
              f"{month_pnl:>+{W}.2f} {running_eq:>{W},.2f} {dd_pct:>{W-1}.2f}%{blown}")

    print(SEP)
    total_wr  = total_wins / total_trades * 100 if total_trades else 0.0
    total_pnl = running_eq - cash
    overall_dd = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0

    dd_str = str(m.get("max_drawdown_%", "0"))
    max_dd = float(dd_str)

    print(f"  {'TOTAL':<10} {total_trades:>{W}} {total_wins:>{W}} {total_wr:>{W-1}.1f}% "
          f"{total_pnl:>+{W}.2f} {running_eq:>{W},.2f} {overall_dd:>{W-1}.2f}%")
    print(SEP)
    print(f"\n  Starting equity : ${cash:,.2f}")
    print(f"  Final equity    : ${running_eq:,.2f}")
    print(f"  Net PnL         : ${total_pnl:+,.2f}")
    print(f"  Peak equity     : ${peak_eq:,.2f}")
    print(f"  Lowest equity   : ${m.get('lowest_equity', cash):,.2f}")
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly backtest — USTEC ZZ")
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2025-01-01")
    parser.add_argument("--cash",     type=float, default=150.0)
    parser.add_argument("--spread",   type=float, default=SPREAD_PTS)
    parser.add_argument("--risk_pct", type=float, default=0.01)
    args = parser.parse_args()
    run_monthly(
        start=args.start, end=args.end,
        cash=args.cash, spread=args.spread,
        risk_pct=args.risk_pct,
    )


if __name__ == "__main__":
    main()
