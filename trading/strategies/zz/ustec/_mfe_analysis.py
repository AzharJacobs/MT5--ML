"""
MFE (Maximum Favorable Excursion) analysis on losing trades.
Shows how far losing trades moved in the right direction before hitting SL.
This tells us whether a trailing stop / breakeven stop would rescue capital.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np
from trading.strategies.zz.ustec.engine import run_backtest

start = sys.argv[1] if len(sys.argv) > 1 else "2023-01-01"
end   = sys.argv[2] if len(sys.argv) > 2 else "2025-12-31"

_, df = run_backtest(start=start, end=end, cash=150, fixed_lot=0.05, silent=True)

losers = df[df["outcome"] == -1].copy()
total  = len(df)
n_loss = len(losers)

print(f"\n{'='*62}")
print(f"  MFE on Losing Trades — {start} to {end}")
print(f"  Total trades: {total}  |  Losses: {n_loss}  ({n_loss/total*100:.0f}%)")
print(f"{'='*62}")

# MFE in points
losers["mfe_pts"] = losers["max_favour"]
losers["sl_dist"] = abs(losers["entry"] - losers["sl"])
losers["mfe_pct_of_sl"] = losers["mfe_pts"] / losers["sl_dist"] * 100
losers["mfe_pnl"]  = losers["mfe_pts"] * 0.05 * 100   # $ value of MFE

# Bracket by how far MFE went
buckets = [
    ("0 pts  (straight to SL)", 0, 1),
    ("1–10 pts",                 1, 10),
    ("10–25 pts",               10, 25),
    ("25–50 pts",               25, 50),
    ("50–100 pts",              50, 100),
    ("100+ pts",               100, 9999),
]

print(f"\n  {'MFE range':<30} {'Trades':>7} {'% of losers':>12} {'Avg loss $':>11}")
print("  " + "-"*63)
for label, lo, hi in buckets:
    grp = losers[(losers["mfe_pts"] >= lo) & (losers["mfe_pts"] < hi)]
    if len(grp) == 0:
        continue
    avg_l = grp["pnl"].mean()
    print(f"  {label:<30} {len(grp):>7} {len(grp)/n_loss*100:>11.1f}% {avg_l:>+11.2f}")

print()
print(f"  Median MFE on losers : {losers['mfe_pts'].median():.1f} pts  (${losers['mfe_pnl'].median():.0f})")
print(f"  Mean   MFE on losers : {losers['mfe_pts'].mean():.1f} pts  (${losers['mfe_pnl'].mean():.0f})")
print(f"  MFE > 20 pts         : {(losers['mfe_pts'] > 20).sum()} trades ({(losers['mfe_pts'] > 20).mean()*100:.0f}%)")
print(f"  MFE > 50 pts         : {(losers['mfe_pts'] > 50).sum()} trades ({(losers['mfe_pts'] > 50).mean()*100:.0f}%)")
print(f"  MFE > 75 pts         : {(losers['mfe_pts'] > 75).sum()} trades ({(losers['mfe_pts'] > 75).mean()*100:.0f}%)")

print(f"\n  --- If we moved SL to BE once MFE > 20 pts ---")
saveable = losers[losers["mfe_pts"] > 20]
full_sl_loss = saveable["pnl"].sum()
# BE exit saves the trade at ~0 (spread cost only)
spread_cost = 2 * 0.05 * 100  # $10
rescued = len(saveable) * (-spread_cost)  # worst-case BE = -$10 per trade
saved = rescued - full_sl_loss
print(f"  Trades that moved 20+ pts: {len(saveable)}")
print(f"  Their current total loss:  ${full_sl_loss:+.2f}")
print(f"  If BE'd at entry (worst):  ${rescued:+.2f}")
print(f"  Capital saved:             ${saved:+.2f}")

print(f"\n  --- If we moved SL to BE once MFE > 40 pts ---")
saveable2 = losers[losers["mfe_pts"] > 40]
full_sl_loss2 = saveable2["pnl"].sum()
rescued2 = len(saveable2) * (-spread_cost)
saved2 = rescued2 - full_sl_loss2
print(f"  Trades that moved 40+ pts: {len(saveable2)}")
print(f"  Their current total loss:  ${full_sl_loss2:+.2f}")
print(f"  If BE'd at entry (worst):  ${rescued2:+.2f}")
print(f"  Capital saved:             ${saved2:+.2f}")

print(f"\n{'='*62}\n")
