#!/usr/bin/env python3
"""
ZZ USTEC backtest: 2023, 2024, 2025 — month-by-month breakdown.
Runs as one continuous simulation (equity carries between years).
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _print_year_table(df_year: pd.DataFrame, year: int, prev_equity: float) -> float:
    df = df_year.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month

    rows = []
    month_start_eq = prev_equity

    for m in range(1, 13):
        grp = df[df["month"] == m]
        if grp.empty:
            continue
        trades       = len(grp)
        wins         = int((grp["outcome"] == 1).sum())
        wr           = wins / trades * 100
        month_end_eq = float(grp["equity"].iloc[-1])
        pnl          = month_end_eq - month_start_eq
        avg_lot      = grp["lot"].mean()
        min_lot      = grp["lot"].min()
        max_lot      = grp["lot"].max()
        rows.append({
            "month":   MONTH_ABBR[m - 1],
            "trades":  trades,
            "wins":    wins,
            "wr":      wr,
            "pnl":     pnl,
            "equity":  month_end_eq,
            "avg_lot": avg_lot,
            "min_lot": min_lot,
            "max_lot": max_lot,
        })
        month_start_eq = month_end_eq

    total_trades = sum(r["trades"] for r in rows)
    total_wins   = sum(r["wins"]   for r in rows)
    total_wr     = total_wins / max(total_trades, 1) * 100
    year_end_eq  = rows[-1]["equity"] if rows else prev_equity
    total_pnl    = year_end_eq - prev_equity

    W = 82
    print(f"\n{'='*W}")
    print(f"  ZZ Strategy  |  USTEC  |  {year}  |  Start: ${prev_equity:,.2f}")
    print(f"{'─'*W}")
    print(f"  {'Month':<6} {'Trades':>7} {'Wins':>5} {'WR%':>7} {'PnL':>12} "
          f"{'Running Eq':>12} {'Avg Lot':>9} {'Min-Max Lot':>14}")
    print(f"{'─'*W}")
    for r in rows:
        sign    = "+" if r["pnl"] >= 0 else "-"
        pnl_s   = f"{sign}${abs(r['pnl']):.2f}"
        eq_s    = f"${r['equity']:,.2f}"
        lot_rng = f"{r['min_lot']:.2f}-{r['max_lot']:.2f}"
        print(f"  {r['month']:<6} {r['trades']:>7} {r['wins']:>5} {r['wr']:>6.1f}% "
              f"{pnl_s:>12} {eq_s:>12} {r['avg_lot']:>9.3f} {lot_rng:>14}")
    print(f"{'─'*W}")
    sign  = "+" if total_pnl >= 0 else "-"
    pnl_s = f"{sign}${abs(total_pnl):.2f}"
    print(f"  {'Total':<6} {total_trades:>7} {total_wins:>5} {total_wr:>6.1f}% {pnl_s:>12}")
    print(f"{'='*W}")

    return year_end_eq


def _print_trade_detail(df_all: pd.DataFrame) -> None:
    W = 105
    print(f"\n{'='*W}")
    print(f"  PER-TRADE DETAIL  |  USTEC")
    print(f"{'─'*W}")
    print(f"  {'#':>4}  {'Date':<17} {'Side':<5} {'Lot':>6} {'Entry':>9} "
          f"{'SL':>9} {'TP':>9} {'Exit':>9} {'PnL':>9} {'Equity':>10} {'Result':<8}")
    print(f"{'─'*W}")
    for idx, row in df_all.iterrows():
        result_str = "WIN" if row["outcome"] == 1 else ("LOSS" if row["outcome"] == -1 else "EXPIRED")
        sign     = "+" if row["pnl"] >= 0 else ""
        date_str = str(row["date"])[:16]
        print(f"  {idx+1:>4}  {date_str:<17} {row['side']:<5} {row['lot']:>6.3f} "
              f"{row['entry']:>9.2f} {row['sl']:>9.2f} {row['tp']:>9.2f} "
              f"{row['exit']:>9.2f} {sign}{row['pnl']:>8.2f} "
              f"${row['equity']:>9.2f} {result_str:<8}")
    print(f"{'='*W}")


def main():
    cash = 150.0

    print("\n" + "="*82)
    print(f"  ZZ Strategy Backtest  |  USTEC  |  2023–2025  |  ${cash:.2f} starting cash")
    print(f"  Lot sizing: dynamic 1% risk (fixed_lot=0 → scales with equity)")
    print("="*82)
    print("  Running simulation... (this may take a few minutes)")

    result = run_backtest(
        start="2023-01-01",
        end="2025-12-31",
        cash=cash,
        symbol="ustech",
        fixed_lot=0.0,   # dynamic 1% risk sizing — lot grows as equity grows
    )

    if not isinstance(result, tuple) or len(result) < 2:
        print("\nNo trades generated or backtest failed.")
        return

    _, df_all = result

    if df_all is None or (isinstance(df_all, pd.DataFrame) and df_all.empty):
        print("\nNo trades generated.")
        return

    df_all = df_all.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["year"] = df_all["date"].dt.year
    df_all = df_all.reset_index(drop=True)

    equity = cash
    for year in [2023, 2024, 2025]:
        df_year = df_all[df_all["year"] == year].copy()
        if df_year.empty:
            print(f"\n  (No trades in {year})")
            continue
        equity = _print_year_table(df_year, year, equity)

    total_trades = len(df_all)
    total_wins   = int((df_all["outcome"] == 1).sum())
    total_wr     = total_wins / max(total_trades, 1) * 100
    total_pnl    = equity - cash
    sign         = "+" if total_pnl >= 0 else "-"

    print(f"\n{'='*82}")
    print(f"  GRAND TOTAL  |  2023–2025")
    print(f"{'─'*82}")
    print(f"  Trades: {total_trades}  |  Wins: {total_wins}  |  WR: {total_wr:.1f}%  |  "
          f"Net PnL: {sign}${abs(total_pnl):.2f}")
    print(f"  Start: ${cash:.2f}  →  End: ${equity:,.2f}")
    print(f"  Avg lot: {df_all['lot'].mean():.3f}  |  "
          f"Min lot: {df_all['lot'].min():.3f}  |  Max lot: {df_all['lot'].max():.3f}")
    print(f"{'='*82}")

    _print_trade_detail(df_all)


if __name__ == "__main__":
    main()
