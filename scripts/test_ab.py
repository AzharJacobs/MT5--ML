#!/usr/bin/env python3
"""
Test A vs Test B comparison.

  Test A — RR cap ≤ 5.0 only          (H4 zones, conf=2, $150, MT5)
  Test B — RR cap ≤ 5.0 + 1H zones   (H4 + 1H zones, conf=2, $150, MT5)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.zz.ustec.engine import run_backtest

COMMON = dict(
    start="2023-01-01",
    end="2025-01-01",
    cash=150.0,
    symbol="ustech",
    min_confirmations=2,
    data_source="mt5",
    max_rr=5.0,
    silent=True,
)

print("Running Test A  (H4 zones + RR cap) …")
res_a = run_backtest(**COMMON, use_1h_zones=False)

print("Running Test B  (H4 + 1H zones + RR cap) …")
res_b = run_backtest(**COMMON, use_1h_zones=True)

# ── Side-by-side summary ──────────────────────────────────────────────────────
W = "─" * 72

def _row(label, va, vb):
    print(f"  {label:<28} {str(va):>18}   {str(vb):>18}")

print(f"\n{W}")
print("  SIDE-BY-SIDE SUMMARY")
print(f"  {'Metric':<28} {'Test A (RR cap)':>18}   {'Test B (+1H zones)':>18}")
print(W)

if not res_a or not res_b:
    print("  One or both runs returned no data — check MT5 connection and date range.")
    sys.exit(1)

ma, dfa = res_a
mb, dfb = res_b

import pandas as pd

def wr(df):
    return f"{(df['outcome']==1).mean()*100:.1f}%"

def avg_rr(df):
    rr = abs(df['tp'] - df['entry']) / abs(df['entry'] - df['sl'])
    return f"{rr.mean():.2f}"

_row("Trades",         ma["total_trades"],       mb["total_trades"])
_row("Wins / Losses / Expired",
     f"{ma['tp_hits']}W/{ma['sl_hits']}L/{ma['expired']}E",
     f"{mb['tp_hits']}W/{mb['sl_hits']}L/{mb['expired']}E")
_row("Win rate",       f"{ma['win_rate_%']}%",   f"{mb['win_rate_%']}%")
_row("Avg RR (theory)", avg_rr(dfa),             avg_rr(dfb))
_row("Net PnL",        ma["net_pnl"],             mb["net_pnl"])
_row("Start equity",   ma["start_cash"],          mb["start_cash"])
_row("Final equity",   ma["final_equity"],        mb["final_equity"])
_row("Max drawdown",   f"{ma['max_drawdown_%']}%", f"{mb['max_drawdown_%']}%")
_row("Avg win $",      ma["avg_win_$"],           mb["avg_win_$"])
_row("Avg loss $",     ma["avg_loss_$"],          mb["avg_loss_$"])
_row("Buys (W)",       f"{ma['buy_trades']} ({ma['buy_wins']}W)",
                       f"{mb['buy_trades']} ({mb['buy_wins']}W)")
_row("Sells (W)",      f"{ma['sell_trades']} ({ma['sell_wins']}W)",
                       f"{mb['sell_trades']} ({mb['sell_wins']}W)")

print(W)

# ── Zone freshness breakdown ──────────────────────────────────────────────────
print(f"\n  Zone freshness:")
for lbl, df, tag in [("A", dfa, "Test A"), ("B", dfb, "Test B")]:
    for fresh_val, grp in df.groupby("zone_fresh"):
        n  = len(grp)
        wr_f = (grp["outcome"] == 1).mean() * 100
        ftag = "fresh" if fresh_val else "tapped"
        print(f"    {tag}  {ftag:<7} : {n:>3} trades  WR={wr_f:.0f}%")

# ── H4 bias breakdown ─────────────────────────────────────────────────────────
print(f"\n  H4 bias at entry:")
for lbl, df, tag in [("A", dfa, "Test A"), ("B", dfb, "Test B")]:
    for bias_val, grp in df.groupby("h4_bias"):
        wr_b = (grp["outcome"] == 1).mean() * 100
        print(f"    {tag}  {bias_val:<14} : {len(grp):>3} trades  WR={wr_b:.0f}%")

# ── Trades filtered by RR cap ─────────────────────────────────────────────────
print(f"\n  RR cap filter (baseline conf=2 = 34 trades):")
print(f"    Test A removed  : {34 - ma['total_trades']} trades above RR 5.0")
print(f"    Test B total    : {mb['total_trades']} trades (H4 + 1H zones, RR ≤ 5.0)")
print()
