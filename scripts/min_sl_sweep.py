#!/usr/bin/env python3
"""
min_sl_sweep.py — Threshold sweep for min_sl_pct filter.

For each threshold, reports trades skipped vs kept and their net P&L.
The optimal floor is the highest threshold where skipped group is still net-negative.

Run:
    python -X utf8 scripts/min_sl_sweep.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.zz.ustec.engine import run_backtest
import pandas as pd

print("Loading 3-year baseline (2023-2025)...")
_, df = run_backtest(start="2023-01-01", end="2026-01-01", fixed_lot=0.05, symbol="ustech")

df["sl_dist_pct"] = (abs(df["entry"] - df["sl"]) / df["entry"] * 100)
df["win"] = df["outcome"] == 1

total_trades = len(df)
total_net    = df["pnl"].sum()

print(f"\nAll trades: {total_trades}  |  WR: {df['win'].mean()*100:.1f}%  |  Net P&L: ${total_net:+,.0f}\n")

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40]

print(f"{'Threshold':>10} | {'Skipped':>8} {'Skip WR%':>9} {'Skip Net$':>11} | {'Kept':>6} {'Kept WR%':>9} {'Kept Net$':>11}")
print("-" * 78)

for t in thresholds:
    skipped = df[df["sl_dist_pct"] < t]
    kept    = df[df["sl_dist_pct"] >= t]

    skip_wr  = skipped["win"].mean() * 100 if len(skipped) else 0
    skip_net = skipped["pnl"].sum()
    kept_wr  = kept["win"].mean() * 100 if len(kept) else 0
    kept_net = kept["pnl"].sum()

    flag = "  ← floor?" if skip_net < 0 else ""

    print(f"  ≥{t:.2f}%   | {len(skipped):>8} {skip_wr:>8.1f}% {skip_net:>+11,.0f} | "
          f"{len(kept):>6} {kept_wr:>8.1f}% {kept_net:>+11,.0f}{flag}")

print()
