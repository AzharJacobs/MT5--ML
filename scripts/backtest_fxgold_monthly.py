#!/usr/bin/env python3
"""
FXGold strategy backtest — 2023-2025, month-by-month report.

Usage:
    python -X utf8 scripts/backtest_fxgold_monthly.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.engine import run_backtest

CASH      = 150.0
LOT       = 0.01
START     = "2023-01-01"
END       = "2025-12-31"

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
W = 82


# ─── Report helpers ───────────────────────────────────────────────────────────

def _monthly_table(df_year: pd.DataFrame, year: int, start_eq: float) -> float:
    df      = df_year.copy()
    df["m"] = pd.to_datetime(df["date"]).dt.month
    rows    = []
    eq      = start_eq

    for m in range(1, 13):
        grp = df[df["m"] == m]
        if grp.empty:
            continue
        trades = len(grp)
        wins   = int((grp["outcome"] == 1).sum())
        losses = int((grp["outcome"] == -1).sum())
        wr     = wins / trades * 100
        end_eq = float(grp["equity"].iloc[-1])
        pnl    = end_eq - eq
        rows.append(dict(month=MONTH_ABBR[m - 1], trades=trades,
                         wins=wins, losses=losses, wr=wr,
                         pnl=pnl, equity=end_eq))
        eq = end_eq

    total_t = sum(r["trades"] for r in rows)
    total_w = sum(r["wins"]   for r in rows)
    yr_pnl  = (rows[-1]["equity"] if rows else start_eq) - start_eq

    print(f"\n{'='*W}")
    print(f"  FXGold | XAUUSD | {year} | Start equity: ${start_eq:,.2f}")
    print(f"{'─'*W}")
    print(f"  {'Month':<6} {'Trades':>7} {'W':>4} {'L':>4} {'WR%':>7} "
          f"{'PnL':>11} {'Running Eq':>12}")
    print(f"{'─'*W}")
    for r in rows:
        sign  = "+" if r["pnl"] >= 0 else "-"
        pnl_s = f"{sign}${abs(r['pnl']):.2f}"
        print(f"  {r['month']:<6} {r['trades']:>7} {r['wins']:>4} {r['losses']:>4} "
              f"{r['wr']:>6.1f}%  {pnl_s:>10}  ${r['equity']:>10,.2f}")
    print(f"{'─'*W}")
    sign  = "+" if yr_pnl >= 0 else "-"
    pnl_s = f"{sign}${abs(yr_pnl):.2f}"
    print(f"  {'Total':<6} {total_t:>7} {total_w:>4} "
          f"{total_t-total_w:>4} {total_w/max(total_t,1)*100:>6.1f}%  {pnl_s:>10}")
    print(f"{'='*W}")

    return rows[-1]["equity"] if rows else start_eq


def _trade_detail(df: pd.DataFrame) -> None:
    print(f"\n{'='*W}")
    print(f"  PER-TRADE DETAIL")
    print(f"{'─'*W}")
    print(f"  {'#':>4}  {'Date':<17} {'Side':<5} {'TF':<3} {'Pattern':<13} "
          f"{'Entry':>8} {'SL':>8} {'TP':>8} {'Exit':>8} {'PnL':>8}  Result")
    print(f"{'─'*W}")
    for idx, r in df.iterrows():
        res  = "WIN" if r["outcome"] == 1 else ("LOSS" if r["outcome"] == -1 else "EXPD")
        sign = "+" if r["pnl"] >= 0 else ""
        dt   = str(r["date"])[:16]
        print(f"  {idx+1:>4}  {dt:<17} {r['side']:<5} {r['zone_tf']:<3} "
              f"{r['pattern']:<13} {r['entry']:>8.2f} {r['sl']:>8.2f} "
              f"{r['tp']:>8.2f} {r['exit']:>8.2f} {sign}{r['pnl']:>7.2f}  {res}")
    print(f"{'='*W}")


def _pattern_breakdown(df: pd.DataFrame) -> None:
    print(f"\n  Pattern breakdown:")
    print(f"  {'Pattern':<15} {'Trades':>7} {'WR%':>7} {'Net PnL':>10}")
    print(f"  {'─'*42}")
    for pat, grp in df.groupby("pattern"):
        wr  = (grp["outcome"] == 1).mean() * 100
        net = grp["pnl"].sum()
        print(f"  {pat:<15} {len(grp):>7} {wr:>6.1f}%  ${net:>+9.2f}")


def _zone_tf_breakdown(df: pd.DataFrame) -> None:
    print(f"\n  Zone TF breakdown:")
    print(f"  {'TF':<6} {'Trades':>7} {'WR%':>7} {'Net PnL':>10}")
    print(f"  {'─'*33}")
    for tf, grp in df.groupby("zone_tf"):
        wr  = (grp["outcome"] == 1).mean() * 100
        net = grp["pnl"].sum()
        print(f"  {tf:<6} {len(grp):>7} {wr:>6.1f}%  ${net:>+9.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = FXGoldConfig()

    print("\n" + "=" * W)
    print(f"  FXGold Backtest | XAUUSD | {START} → {END}")
    print(f"  Cash: ${CASH:.2f} | Lot: {LOT} | bias_mode: {cfg.bias_mode} "
          f"| min_rr: {cfg.min_rr} | fractal_window: {cfg.fractal_window}")
    print("=" * W)

    trades, equity_curve = run_backtest(
        cfg=cfg,
        start=START,
        end=END,
        cash=CASH,
        fixed_lot=LOT,
    )

    if not trades:
        print("\n  No trades generated. Check bias_mode, min_rr, or data availability.")
        return

    df_all = pd.DataFrame(trades)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["year"] = df_all["date"].dt.year
    df_all = df_all.reset_index(drop=True)

    # Monthly tables per year
    equity = CASH
    for year in [2023, 2024, 2025]:
        df_y = df_all[df_all["year"] == year]
        if df_y.empty:
            print(f"\n  (No trades in {year})")
            continue
        equity = _monthly_table(df_y, year, equity)

    # Grand total
    total_t = len(df_all)
    total_w = int((df_all["outcome"] == 1).sum())
    total_l = int((df_all["outcome"] == -1).sum())
    net_pnl = equity - CASH
    sign    = "+" if net_pnl >= 0 else "-"

    eq_s    = pd.Series(equity_curve)
    max_dd  = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100

    print(f"\n{'='*W}")
    print(f"  GRAND TOTAL  |  2023–2025")
    print(f"{'─'*W}")
    print(f"  Trades: {total_t}  |  Wins: {total_w}  |  Losses: {total_l}  "
          f"|  WR: {total_w/max(total_t,1)*100:.1f}%")
    print(f"  Net PnL: {sign}${abs(net_pnl):.2f}  |  "
          f"Start: ${CASH:.2f}  →  End: ${equity:,.2f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"{'='*W}")

    _pattern_breakdown(df_all)
    _zone_tf_breakdown(df_all)
    _trade_detail(df_all)


if __name__ == "__main__":
    main()
