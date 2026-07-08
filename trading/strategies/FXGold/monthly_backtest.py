#!/usr/bin/env python3
"""
monthly_backtest.py — Month-by-month equity walk on the FXGold zone strategy.

Runs a single continuous backtest (equity compounds across months) and
renders the same monthly HTML report used by the USTEC ZZ strategy
(trading/Reports/report_html.py), adapted for FXGold's trade columns.

Usage:
  python trading/strategies/FXGold/monthly_backtest.py
  python trading/strategies/FXGold/monthly_backtest.py --data_source mt5 --cash 150 --start 2023-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.engine import run_backtest
from trading.Reports.report_html import save_html_report

REPORTS_DIR = _ROOT / "trading" / "Reports"


def run_monthly(
    start: str, end: str, cash: float, fixed_lot: float, data_source: str,
    out_path: str | None = None,
) -> None:
    print(f"\nFXGold monthly backtest  {start} -> {end}")
    print(f"  Cash=${cash:.2f}  fixed_lot={fixed_lot}  data_source={data_source}")
    print(f"  Single continuous run - equity compounds across months...\n")

    cfg = FXGoldConfig()
    trades, equity_curve, _df_h1 = run_backtest(
        cfg, start=start, end=end, cash=cash, fixed_lot=fixed_lot, data_source=data_source,
    )

    if not trades:
        print("  No trades generated.")
        return

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    full_range = pd.period_range(start=pd.Period(start, freq="M"), end=pd.Period(end, freq="M"), freq="M")
    all_months = list(full_range)

    df_sorted = df.sort_values("date").reset_index(drop=True)

    monthly = defaultdict(list)
    for _, row in df_sorted.iterrows():
        monthly[row["month"]].append(row)

    trade_cols = ["date", "side", "pattern", "zone_tf", "zone_kind",
                  "entry", "sl", "tp", "exit", "pnl", "outcome"]
    trade_cols = [c for c in trade_cols if c in df_sorted.columns]

    running_eq = cash
    peak_eq    = cash
    lowest_eq  = cash
    total_trades = 0
    total_wins   = 0

    for period in all_months:
        rows = monthly[period]
        month_df  = pd.DataFrame(rows)
        month_pnl = sum(r["pnl"] for r in rows)
        trades_n  = len(rows)
        wins      = sum(1 for r in rows if r["outcome"] == 1)
        wr        = wins / trades_n * 100 if trades_n else 0.0
        start_eq  = running_eq
        end_eq    = running_eq + month_pnl
        if end_eq < 0:
            end_eq = 0.0
        running_eq = end_eq
        peak_eq    = max(peak_eq, running_eq)
        lowest_eq  = min(lowest_eq, running_eq)
        dd_pct     = (running_eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0.0
        total_trades += trades_n
        total_wins   += wins

        print("=" * 100)
        print(f"  {period}  ({trades_n} trades)")
        print("=" * 100)
        if trades_n:
            print(month_df[trade_cols].to_string(index=False))
        else:
            print("  No trades this month.")
        blown = "  !! BLOWN" if running_eq <= 0.01 else ""
        print(f"\n  Trades: {trades_n}   Wins: {wins}   WR: {wr:.1f}%   "
              f"Month PnL: {month_pnl:+.2f}   Start Eq: {start_eq:,.2f}   "
              f"End Eq: {running_eq:,.2f}   DD: {dd_pct:.2f}%{blown}\n")

    total_wr  = total_wins / total_trades * 100 if total_trades else 0.0
    total_pnl = running_eq - cash
    eq_series = pd.Series([cash] + df_sorted["equity"].tolist())
    max_dd    = float(((eq_series - eq_series.cummax()) / eq_series.cummax()).min() * 100)

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

    report_path = out_path or str(REPORTS_DIR / "fxgold_monthly_report.html")
    save_html_report(
        df, start=start, end=end, cash=cash, out_path=report_path,
        title="FXGold Monthly Trade Report",
        eyebrow="ZONE-TO-ZONE · XAUUSD · BACKTEST",
        meta={
            "fixed lot":   str(fixed_lot),
            "data source": data_source,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly backtest - FXGold")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2025-01-01")
    parser.add_argument("--cash",  type=float, default=150.0)
    parser.add_argument("--fixed_lot", type=float, default=0.01)
    parser.add_argument("--data_source", default="db", choices=["db", "mt5"])
    parser.add_argument("--out", default=None,
                        help="HTML report output path (default: trading/Reports/fxgold_monthly_report.html)")
    args = parser.parse_args()
    run_monthly(
        start=args.start, end=args.end, cash=args.cash, fixed_lot=args.fixed_lot,
        data_source=args.data_source, out_path=args.out,
    )


if __name__ == "__main__":
    main()
