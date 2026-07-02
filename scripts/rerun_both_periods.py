#!/usr/bin/env python3
"""
Rerun both periods after LH+HL bias fix.
  OOS  : 2021-01-01 → 2023-01-01
  IS   : 2023-01-01 → 2025-01-01
Settings: conf=2, RR cap 5.0, $150, MT5 data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.zz.ustec.engine import run_backtest

COMMON = dict(
    cash=150.0,
    symbol="ustech",
    min_confirmations=2,
    data_source="mt5",
    max_rr=5.0,
    allow_neutral=False,
    silent=True,
)

print("Running 2021-2022 (OOS) …")
res_oos = run_backtest(start="2021-01-01", end="2023-01-01", **COMMON)

print("Running 2023-2024 (in-sample) …")
res_is  = run_backtest(start="2023-01-01", end="2025-01-01", **COMMON)

DIV = "─" * 74

def summarise(label, result):
    if not result:
        print(f"  {label}: no data")
        return
    m, df = result
    import pandas as pd
    avg_rr = (abs(df["tp"] - df["entry"]) / abs(df["entry"] - df["sl"])).mean()
    print(f"\n  {label}")
    print(f"  {DIV}")
    print(f"  Trades          : {m['total_trades']}")
    print(f"  Wins/Loss/Exp   : {m['tp_hits']}W / {m['sl_hits']}L / {m['expired']}E")
    print(f"  Win rate        : {m['win_rate_%']}%")
    print(f"  Avg RR (theory) : {avg_rr:.2f}")
    print(f"  Net PnL         : {m['net_pnl']}")
    print(f"  Final equity    : {m['final_equity']}  (from {m['start_cash']})")
    print(f"  Max drawdown    : {m['max_drawdown_%']}%")
    print(f"  Buys  (W)       : {m['buy_trades']} ({m['buy_wins']}W)")
    print(f"  Sells (W)       : {m['sell_trades']} ({m['sell_wins']}W)")
    print()
    print("  H4 bias breakdown:")
    for bias, grp in df.groupby("h4_bias"):
        wr = (grp["outcome"] == 1).mean() * 100
        print(f"    {bias:<16} {len(grp):>3} trades  WR={wr:.0f}%")
    print()
    print("  Per-trade list:")
    print(f"  {'Date':<22} {'Side':<5} {'Bias':<16} {'Signals':<38} {'RR':>5}  Out")
    print("  " + "─" * 90)
    for _, r in df.sort_values("date").iterrows():
        sigs = r["signals"].replace("|", " + ")
        out  = "WIN " if r["outcome"] == 1 else "LOSS"
        rr   = abs(r["tp"] - r["entry"]) / abs(r["entry"] - r["sl"])
        print(f"  {str(r['date']):<22} {r['side']:<5} {r['h4_bias']:<16} {sigs:<38} {rr:>5.2f}  {out}")

print()
print(DIV)
print("  RESULTS  |  allow_neutral=False  |  conf=2  RR<=5.0")
print(DIV)
summarise("OOS 2021-2022  (settings were NOT built on this)", res_oos)
summarise("IN-SAMPLE 2023-2024  (settings were built on this)", res_is)
