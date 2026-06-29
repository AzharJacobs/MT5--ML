#!/usr/bin/env python3
"""
3-year month-by-month backtest: XAUUSD Gold Z&Z v2
2023 | 2024 | 2025 — equity carries across every month and year.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.xauusd.strategy import make_gold_config
from trading.strategies.zz.xauusd.engine import run_backtest_gold

START_CASH = 150.0
START      = "2023-01-01"
END        = "2026-01-01"

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]


def _div(label="", char="─", width=72):
    if label:
        pad = width - len(label) - 2
        print(f"{'─'*(pad//2)} {label} {'─'*(pad - pad//2)}")
    else:
        print(char * width)


def main():
    cfg = make_gold_config()

    print(f"\nRunning 3-year backtest  {START} → {END}  (start equity: ${START_CASH:,.2f})\n")
    _, df = run_backtest_gold(cfg, start=START, end=END, cash=START_CASH)

    if df is None or df.empty:
        print("No trades returned — check DB connection.")
        return

    df = df.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    equity = START_CASH
    grand_trades = grand_wins = grand_losses = 0
    grand_pnl = 0.0

    print("\n")
    _div("3-YEAR MONTH-BY-MONTH REPORT")

    for year in [2023, 2024, 2025]:
        _div(f"  {year}  ")
        hdr = f"{'Month':<8} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR%':>6} {'PnL':>10} {'Equity':>12}"
        print(hdr)
        _div(char="·")

        yr_df = df[df["year"] == year]
        yr_trades = yr_wins = yr_losses = 0
        yr_pnl    = 0.0
        yr_start_equity = equity

        for m in range(1, 13):
            mo_df  = yr_df[yr_df["month"] == m]
            t      = len(mo_df)
            w      = int((mo_df["outcome"] == 1).sum())
            l      = int((mo_df["outcome"] == -1).sum())
            pnl    = mo_df["pnl"].sum()
            wr     = (w / t * 100) if t > 0 else 0.0
            equity += pnl

            yr_trades  += t
            yr_wins    += w
            yr_losses  += l
            yr_pnl     += pnl

            if t == 0:
                row = f"{MONTHS[m-1]:<8} {'—':>7} {'—':>6} {'—':>7} {'—':>6} {'—':>10} {equity:>11,.2f}"
            else:
                sign = "+" if pnl >= 0 else ""
                row  = (f"{MONTHS[m-1]:<8} {t:>7} {w:>6} {l:>7} {wr:>5.1f}% "
                        f"{sign}{pnl:>9,.2f} {equity:>11,.2f}")
            print(row)

        _div(char="·")
        yr_wr  = yr_wins / yr_trades * 100 if yr_trades else 0.0
        yr_ret = (equity - yr_start_equity) / yr_start_equity * 100
        sign   = "+" if yr_pnl >= 0 else ""
        print(f"{'YEAR TOTAL':<8} {yr_trades:>7} {yr_wins:>6} {yr_losses:>7} "
              f"{yr_wr:>5.1f}% {sign}{yr_pnl:>9,.2f} {equity:>11,.2f}  "
              f"(year return: {sign}{yr_ret:.1f}%)")
        _div()
        print()

        grand_trades  += yr_trades
        grand_wins    += yr_wins
        grand_losses  += yr_losses
        grand_pnl     += yr_pnl

    grand_wr  = grand_wins / grand_trades * 100 if grand_trades else 0.0
    net_ret   = (equity - START_CASH) / START_CASH * 100
    sign      = "+" if grand_pnl >= 0 else ""

    _div("GRAND TOTAL — 3 YEARS (2023–2025)")
    print(f"  Total trades : {grand_trades}")
    print(f"  Wins         : {grand_wins}")
    print(f"  Losses       : {grand_losses}")
    print(f"  Win rate     : {grand_wr:.1f}%")
    print(f"  Net PnL      : {sign}${grand_pnl:,.2f}")
    print(f"  Start equity : ${START_CASH:,.2f}")
    print(f"  Final equity : ${equity:,.2f}")
    print(f"  Total return : {sign}{net_ret:.1f}%")
    _div()
    print()


if __name__ == "__main__":
    main()
