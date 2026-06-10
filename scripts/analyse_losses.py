#!/usr/bin/env python3
"""
analyse_losses.py — Dig into losing trades to find avoidable patterns.

Four lenses:
  1. Zone exhaustion    — zones tapped 3+ times that keep losing
  2. Same-session re-entry — re-entering same zone within 8h of a loss
  3. SL too tight       — SL distance % distribution: winners vs losers
  4. Counter-trend      — win rate when trading with vs against H4 bias

Run:
    python -X utf8 scripts/analyse_losses.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.zz.ustec.engine import run_backtest
import pandas as pd

print("Running 3-year baseline (2023-2025)...")
_, df = run_backtest(start="2023-01-01", end="2026-01-01", fixed_lot=0.05, symbol="ustech")

df["date"]      = pd.to_datetime(df["date"])
df["exit_date"] = pd.to_datetime(df["exit_date"])
df["zone_id"]   = df.apply(
    lambda r: f"{r['zone_bottom']:.2f}_{r['zone_top']:.2f}_{r['zone_kind']}", axis=1
)
df["sl_dist_pct"] = (abs(df["entry"] - df["sl"]) / df["entry"] * 100).round(3)
df["win"]  = df["outcome"] == 1
df["loss"] = df["outcome"] == -1

# H4 alignment: does the trade direction agree with the H4 bias?
def alignment(row):
    b, s = row["h4_bias"], row["side"]
    if (b == "bullish" and s == "buy") or (b == "bearish" and s == "sell"):
        return "aligned"
    if b == "neutral_up":
        return "neutral"
    return "counter"

df["alignment"] = df.apply(alignment, axis=1)

# Sort by time so shift() gives true previous trade on that zone
df = df.sort_values("date").reset_index(drop=True)
df["prev_trade_time"]    = df.groupby("zone_id")["date"].shift(1)
df["prev_outcome"]       = df.groupby("zone_id")["outcome"].shift(1)
df["hours_since_last"]   = ((df["date"] - df["prev_trade_time"]).dt.total_seconds() / 3600).round(1)

losses = df[df["loss"]]
wins   = df[df["win"]]
total  = len(df)

print(f"\nTotal trades: {total}  |  Wins: {len(wins)}  |  Losses: {len(losses)}  |  WR: {len(wins)/total*100:.1f}%")

SEP = "=" * 72

# ═══════════════════════════════════════════════════════════════════════
# 1. ZONE EXHAUSTION
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("1. ZONE EXHAUSTION — zones tapped 3+ times")
print(SEP)

zs = df.groupby("zone_id").agg(
    trades  = ("outcome", "count"),
    wins    = ("win",     "sum"),
    losses  = ("loss",    "sum"),
    net_pnl = ("pnl",     "sum"),
).reset_index()
zs["wr"] = (zs["wins"] / zs["trades"] * 100).round(1)

multi = zs[zs["trades"] >= 3].sort_values("net_pnl")
print(f"\nZones with 3+ trades: {len(multi)}  (of {len(zs)} total zones)\n")
print(f"{'zone_id':<42} {'Tr':>4} {'W':>4} {'L':>4} {'WR%':>6} {'Net$':>9}")
print("-" * 72)
for _, z in multi.iterrows():
    flag = "  ← DEAD" if z["wr"] == 0 else ("  ← POOR" if z["wr"] < 25 else "")
    print(f"{z['zone_id']:<42} {z['trades']:>4.0f} {z['wins']:>4.0f} {z['losses']:>4.0f} "
          f"{z['wr']:>5.1f}% {z['net_pnl']:>+9.0f}{flag}")

dead = multi[multi["wr"] == 0]
poor = multi[(multi["wr"] > 0) & (multi["wr"] < 25)]
print(f"\n  Dead zones (0% WR, 3+ trades)  : {len(dead)}  |  net P&L: ${dead['net_pnl'].sum():+,.0f}")
print(f"  Poor zones (<25% WR, 3+ trades): {len(poor)}  |  net P&L: ${poor['net_pnl'].sum():+,.0f}")
avoided = dead["net_pnl"].sum() + poor["net_pnl"].sum()
print(f"  If we'd stopped after 2 losses on any zone: ~${avoided:+,.0f} recovered")

# ═══════════════════════════════════════════════════════════════════════
# 2. SAME-SESSION RE-ENTRY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("2. SAME-SESSION RE-ENTRY — back into same zone within 8h of a loss")
print(SEP)

quick = df[(df["hours_since_last"] <= 8) & (df["prev_outcome"] == -1)].copy()
print(f"\nQuick re-entries (prev trade on zone was a loss, within 8h): {len(quick)}")

if len(quick):
    print(f"  Win rate : {quick['win'].mean()*100:.1f}%")
    print(f"  Net P&L  : ${quick['pnl'].sum():+,.0f}\n")
    print(f"  {'Window':<12} {'Trades':>7} {'WR%':>7} {'Net$':>10}")
    print(f"  {'-'*38}")
    for hrs, lbl in [(2, "≤2h"), (4, "≤4h"), (6, "≤6h"), (8, "≤8h")]:
        sub = quick[quick["hours_since_last"] <= hrs]
        if len(sub):
            print(f"  {lbl:<12} {len(sub):>7} {sub['win'].mean()*100:>6.1f}% {sub['pnl'].sum():>+10.0f}")
else:
    print("  None found.")

# All re-entries regardless of loss/win before
any_quick = df[(df["hours_since_last"] <= 8) & df["prev_trade_time"].notna()]
print(f"\n  All re-entries ≤8h (regardless of prior outcome): {len(any_quick)}")
if len(any_quick):
    print(f"  Win rate: {any_quick['win'].mean()*100:.1f}%  |  Net: ${any_quick['pnl'].sum():+,.0f}")

# ═══════════════════════════════════════════════════════════════════════
# 3. SL DISTANCE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("3. SL DISTANCE — winners vs losers")
print(SEP)

print(f"\n  {'Bucket':<14} {'Trades':>7} {'WR%':>7} {'Net$':>10}")
print(f"  {'-'*42}")
buckets = [
    (0.00, 0.20, "≤0.20%"),
    (0.20, 0.35, "0.20–0.35%"),
    (0.35, 0.50, "0.35–0.50%"),
    (0.50, 0.75, "0.50–0.75%"),
    (0.75, 1.00, "0.75–1.00%"),
    (1.00, 99.0, ">1.00%"),
]
for lo, hi, lbl in buckets:
    sub = df[(df["sl_dist_pct"] >= lo) & (df["sl_dist_pct"] < hi)]
    if len(sub):
        print(f"  {lbl:<14} {len(sub):>7} {sub['win'].mean()*100:>6.1f}% {sub['pnl'].sum():>+10.0f}")

print(f"\n  Median SL%  — Winners: {wins['sl_dist_pct'].median():.3f}%  |  Losers: {losses['sl_dist_pct'].median():.3f}%")
print(f"  Mean   SL%  — Winners: {wins['sl_dist_pct'].mean():.3f}%  |  Losers: {losses['sl_dist_pct'].mean():.3f}%")

tight = df[df["sl_dist_pct"] < 0.3]
if len(tight):
    print(f"\n  Tight SL (<0.3%): {len(tight)} trades  WR={tight['win'].mean()*100:.1f}%  net=${tight['pnl'].sum():+,.0f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. COUNTER-TREND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("4. TREND ALIGNMENT — trading with vs against H4 bias")
print(SEP)

print(f"\n  {'Alignment':<12} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'Net$':>10}")
print(f"  {'-'*46}")
for aln in ["aligned", "neutral", "counter"]:
    sub = df[df["alignment"] == aln]
    if len(sub):
        flag = "  ← GATE?" if aln == "counter" and sub["win"].mean() < 0.25 else ""
        print(f"  {aln:<12} {len(sub):>7} {int(sub['win'].sum()):>6} {sub['win'].mean()*100:>6.1f}% {sub['pnl'].sum():>+10.0f}{flag}")

print(f"\n  Per-bias breakdown (buy vs sell):")
for bias in sorted(df["h4_bias"].unique()):
    row_b = df[(df["h4_bias"] == bias) & (df["side"] == "buy")]
    row_s = df[(df["h4_bias"] == bias) & (df["side"] == "sell")]
    b_str = f"buy {len(row_b):>3} WR={row_b['win'].mean()*100:.0f}%" if len(row_b) else "buy   0"
    s_str = f"sell {len(row_s):>3} WR={row_s['win'].mean()*100:.0f}%" if len(row_s) else "sell   0"
    print(f"    {bias:<14}: {b_str}  |  {s_str}")

print()
