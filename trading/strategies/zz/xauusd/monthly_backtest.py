#!/usr/bin/env python3
"""
monthly_backtest.py — Month-by-month equity walk on XAUUSD (Gold) ZZ baseline.

Mirrors trading/strategies/zz/ustec/monthly_backtest.py — runs a single
continuous backtest (equity compounds across months), slices df_t by the
month trades were ENTERED to report per-month stats, and writes the same
standing HTML report format to trading/Reports/xauusd_monthly_report.html.

Usage:
  python trading/strategies/zz/xauusd/monthly_backtest.py
  python trading/strategies/zz/xauusd/monthly_backtest.py --cash 150 --start 2023-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace as _dc_replace
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.xauusd.strategy import make_gold_config, RISK_PCT, SPREAD_PTS
from trading.strategies.zz.xauusd.engine import run_backtest_gold
from trading.Reports.report_html import save_html_report

REPORTS_DIR = _ROOT / "trading" / "Reports"


def run_monthly(start: str, end: str, cash: float, min_confirmations: int | None = None,
                 out_path: str | None = None) -> None:

    cfg = make_gold_config()
    if min_confirmations is not None:
        cfg = _dc_replace(cfg, min_confirmations=min_confirmations)

    print(f"\nMonthly backtest (Gold)  {start} -> {end}")
    print(f"  Cash=${cash:.2f}  risk={RISK_PCT*100:.2f}% (from config.yaml)  spread={cfg.spread}pts")
    print(f"  D1 trend filter={cfg.d1_trend_filter}  trading_hours={cfg.trading_hours}")
    print(f"  Single continuous run - equity compounds across months...\n")

    m, df = run_backtest_gold(
        cfg=cfg, start=start, end=end, cash=cash, silent=True,
    )

    if df.empty:
        print("  No trades generated.")
        return

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    full_range = pd.period_range(start=pd.Period(start, freq="M"), end=pd.Period(end, freq="M"), freq="M")
    all_months = list(full_range)

    df_sorted = df.sort_values("date").reset_index(drop=True)

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

    report_path = out_path or str(REPORTS_DIR / "xauusd_monthly_report.html")
    save_html_report(
        df, start=start, end=end, cash=cash, out_path=report_path,
        title="Monthly Trade Report — XAUUSD",
        eyebrow="ZONE-TO-ZONE · XAUUSD · BACKTEST",
        meta={
            "risk": f"{RISK_PCT*100:.2f}%/trade",
            "spread": f"{cfg.spread} pts",
            "min confirmations": str(cfg.min_confirmations),
            "d1 trend filter": str(cfg.d1_trend_filter),
            "trading hours": str(cfg.trading_hours),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly backtest - XAUUSD (Gold) ZZ")
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2025-01-01")
    parser.add_argument("--cash",     type=float, default=150.0)
    parser.add_argument("--min_confirmations", type=int, default=None,
                        help="Override config.yaml's min_confirmations")
    parser.add_argument("--out", default=None,
                        help="HTML report output path (default: trading/Reports/xauusd_monthly_report.html)")
    args = parser.parse_args()
    run_monthly(
        start=args.start, end=args.end,
        cash=args.cash,
        min_confirmations=args.min_confirmations,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
