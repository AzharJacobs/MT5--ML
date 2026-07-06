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

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, MAX_FORWARD_BARS, RISK_PCT, ALLOW_NEUTRAL, MIDLINE_TP, MIDLINE_PCT,
    SL_BUFFER_PCT,
)
from trading.strategies.zz.ustec.engine import run_backtest
from trading.Reports.report_html import save_html_report

REPORTS_DIR = _ROOT / "trading" / "Reports"


def run_monthly(start: str, end: str, cash: float, spread: float, min_confirmations: int,
                 out_path: str | None = None, bos_msb_min1: bool = False,
                 dual_tf: bool = False) -> None:

    print(f"\nMonthly backtest  {start} -> {end}")
    print(f"  Cash=${cash:.2f}  risk={RISK_PCT*100:.2f}% (from config.yaml)  spread={spread}pts")
    print(f"  Single continuous run - equity compounds across months...\n")

    m, df = run_backtest(
        start=start, end=end, cash=cash,
        symbol="ustech", spread=spread,
        fixed_lot=0,
        min_rr=MIN_RR, max_forward_bars=MAX_FORWARD_BARS,
        min_confirmations=min_confirmations,
        bos_msb_min1=bos_msb_min1,
        allow_neutral=ALLOW_NEUTRAL,
        midline_tp=MIDLINE_TP,
        midline_pct=MIDLINE_PCT,
        sl_buffer_pct=SL_BUFFER_PCT,
        use_1h_zones=dual_tf,
        silent=True,
    )

    if df.empty:
        print("  No trades generated.")
        return

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    full_range = pd.period_range(start=pd.Period(start, freq="M"), end=pd.Period(end, freq="M"), freq="M")
    all_months = list(full_range)

    # Track running equity: start of each month = equity after last closed trade
    # df["equity"] = equity after each trade closes (in chronological order)
    # We step through chronologically and assign start/end equity per month.
    df_sorted = df.sort_values("date").reset_index(drop=True)

    # Build a dict: month → list of rows
    from collections import defaultdict
    monthly = defaultdict(list)
    for _, row in df_sorted.iterrows():
        monthly[row["month"]].append(row)

    trade_cols = ["date", "side", "h4_bias", "signals", "confirmations",
                  "zone_fresh", "zone_kind", "entry", "sl", "tp", "exit", "pnl", "outcome"]
    trade_cols = [c for c in trade_cols if c in df_sorted.columns]

    running_eq = cash
    peak_eq    = cash
    lowest_eq  = cash
    total_trades = 0
    total_wins   = 0

    for period in all_months:
        rows = monthly[period]
        month_df    = pd.DataFrame(rows)
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
        lowest_eq   = min(lowest_eq, running_eq)
        dd_pct      = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0
        total_trades += trades
        total_wins   += wins

        print("=" * 100)
        print(f"  {period}  ({trades} trades)")
        print("=" * 100)
        if trades:
            print(month_df[trade_cols].to_string(index=False))
        else:
            print("  No trades this month.")
        blown = "  !! BLOWN" if running_eq <= 0.01 else ""
        print(f"\n  Trades: {trades}   Wins: {wins}   WR: {wr:.1f}%   "
              f"Month PnL: {month_pnl:+.2f}   Start Eq: {start_eq:,.2f}   "
              f"End Eq: {running_eq:,.2f}   DD: {dd_pct:.2f}%{blown}\n")

    total_wr  = total_wins / total_trades * 100 if total_trades else 0.0
    total_pnl = running_eq - cash
    overall_dd = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0

    dd_str = str(m.get("max_drawdown_%", "0"))
    max_dd = float(dd_str)

    print("=" * 100)
    print("  OVERALL (for reference - see each month above for the per-month breakdown)")
    print("=" * 100)
    print(f"  Starting equity : ${cash:,.2f}")
    print(f"  Final equity    : ${running_eq:,.2f}")
    print(f"  Net PnL         : ${total_pnl:+,.2f}")
    print(f"  Total trades    : {total_trades}   Wins: {total_wins}   WR: {total_wr:.1f}%")
    print(f"  Peak equity     : ${peak_eq:,.2f}")
    print(f"  Lowest equity   : ${lowest_eq:,.2f}")
    print(f"  Max drawdown    : {max_dd:.2f}%")
    print()

    report_path = out_path or str(REPORTS_DIR / "ustec_monthly_report.html")
    save_html_report(
        df, start=start, end=end, cash=cash, out_path=report_path,
        meta={
            "risk": f"{RISK_PCT*100:.2f}%/trade",
            "spread": f"{spread} pts",
            "min confirmations": str(min_confirmations),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly backtest - USTEC ZZ")
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2025-01-01")
    parser.add_argument("--cash",     type=float, default=150.0)
    parser.add_argument("--spread",   type=float, default=SPREAD_PTS)
    parser.add_argument("--min_confirmations", type=int, default=1)
    parser.add_argument("--out", default=None,
                        help="HTML report output path (default: trading/Reports/ustec_monthly_report.html)")
    parser.add_argument("--bos_msb_min1", action="store_true",
                        help="Allow single-confirmation entries when that confirmation is bos_msb")
    parser.add_argument("--dual_tf", action="store_true",
                        help="Watch 1H and 4H zones in parallel; 4H takes priority on overlap")
    args = parser.parse_args()
    run_monthly(
        start=args.start, end=args.end,
        cash=args.cash, spread=args.spread,
        min_confirmations=args.min_confirmations,
        out_path=args.out,
        bos_msb_min1=args.bos_msb_min1,
        dual_tf=args.dual_tf,
    )


if __name__ == "__main__":
    main()
