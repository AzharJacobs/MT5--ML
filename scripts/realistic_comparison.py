#!/usr/bin/env python3
"""
realistic_comparison.py — Exness-realistic backtest comparison.

Two passes:
  BASELINE     — realistic mode, no min_sl_pct
  SL_FILTER    — realistic mode, min_sl_pct=0.25

Account model:
  $100 start | 0.05 lot | contract_size=1 | 1% margin | MC@60% | SO@0%
  Spread: 4pts (8pts Asian, 12pts daily break) | Slippage ≤3pts on SL
  Swap: -1.5pts/lot/night long, +0.8pts/lot/night short

Run:
    python -X utf8 scripts/realistic_comparison.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.zz.ustec.engine import run_backtest

YEARS = [("2023-01-01", "2024-01-01", "2023"),
         ("2024-01-01", "2025-01-01", "2024"),
         ("2025-01-01", "2026-01-01", "2025"),
         ("2023-01-01", "2026-01-01", "3-YR")]

BASE_KWARGS = dict(
    symbol              = "ustech",
    fixed_lot           = 0.05,
    cash                = 100.0,
    realistic           = True,
    contract_size_override = 1.0,
    base_spread_pts     = 4.0,
    swap_long_pts       = -1.5,
    swap_short_pts      = 0.8,
    slippage_max_pts    = 3.0,
    margin_rate         = 0.01,
    margin_call_pct     = 0.60,
)

PASSES = [
    ("BASELINE",    {}),
    ("SL≥0.25%",   {"min_sl_pct": 0.25}),
]


def run(start, end, extra):
    m, _ = run_backtest(start=start, end=end, **BASE_KWARGS, **extra)
    net   = float(m["net_pnl"].replace("$", "").replace(",", ""))
    wr    = float(m["win_rate_%"])
    tr    = int(m["total_trades"])
    dd    = float(m["max_drawdown_%"])
    eq    = float(m["final_equity"].replace("$", "").replace(",", ""))
    so    = m.get("stopout_occurred", False)
    swap  = m.get("total_swap_pnl", "n/a")
    slip  = m.get("total_slippage_cost", "n/a")
    return net, wr, tr, dd, eq, so, swap, slip


print("\n" + "=" * 90)
print("  REALISTIC EXNESS BACKTEST — USTECm  |  $100 / 0.05 lot / contract=1")
print("  Spread: 4pts (8 Asian / 12 break)  |  Slippage ≤3pts  |  Swap included")
print("=" * 90)

HDR = (f"\n{'Config':<14} | "
       f"{'2023':>8} {'WR':>5} {'Tr':>4} {'DD':>7} {'Eq$':>8} |"
       f" {'2024':>8} {'WR':>5} {'Tr':>4} {'DD':>7} {'Eq$':>8} |"
       f" {'2025':>8} {'WR':>5} {'Tr':>4} {'DD':>7} {'Eq$':>8} |"
       f" {'3-YR':>8} {'Tr':>4} {'SO':>4}")
print(HDR)
print("-" * 110)

for label, extra in PASSES:
    results = {}
    for start, end, yr in YEARS:
        results[yr] = run(start, end, extra)

    def fmt(yr):
        net, wr, tr, dd, eq, so, *_ = results[yr]
        return f"{net:>+8.1f} {wr:>4.1f}% {tr:>4} {dd:>6.1f}% {eq:>8.2f}"

    so_3yr = "YES" if results["3-YR"][5] else " no"
    net_3yr = results["3-YR"][0]
    tr_3yr  = results["3-YR"][2]

    row = (f"{label:<14} | "
           f" {fmt('2023')} |"
           f" {fmt('2024')} |"
           f" {fmt('2025')} |"
           f" {net_3yr:>+8.1f} {tr_3yr:>4} {so_3yr:>4}")
    print(row)

    # Extra detail row: swap + slippage for 3-yr run
    swap_3yr = results["3-YR"][6]
    slip_3yr = results["3-YR"][7]
    print(f"  {'':14}  swap={swap_3yr}  slippage_cost={slip_3yr}")

print()
print("Legend: Net=net P&L ($)  WR=win rate  Tr=trades  DD=max drawdown  Eq=final equity  SO=stop-out hit")
print()
