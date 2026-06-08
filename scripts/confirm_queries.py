#!/usr/bin/env python3
"""
confirm_queries.py — Two confirmation queries before implementing fixes.

Q1: Dead zones — did any winning trade appear AFTER the 2nd consecutive loss?
    (Would a "stop after 2 losses" rule have blocked a real winner?)

Q2: Tight-SL losers — are losses with SL <0.3% from entry concentrated in
    specific zones or sessions, or uniformly spread?

Run:
    python -X utf8 scripts/confirm_queries.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine_zones import run_backtest
import pandas as pd

print("Loading 3-year baseline...")
_, df = run_backtest(start="2023-01-01", end="2026-01-01", fixed_lot=0.05, symbol="ustech")

df["date"]     = pd.to_datetime(df["date"])
df["zone_id"]  = df.apply(lambda r: f"{r['zone_bottom']:.2f}_{r['zone_top']:.2f}_{r['zone_kind']}", axis=1)
df["sl_dist_pct"] = (abs(df["entry"] - df["sl"]) / df["entry"] * 100).round(3)
df["hour"]     = df["date"].dt.hour
df = df.sort_values("date").reset_index(drop=True)

SEP = "=" * 72

# ═══════════════════════════════════════════════════════════════════════
# Q1: Dead zones — wins AFTER 2nd consecutive loss
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("Q1: DEAD ZONES — wins that arrived AFTER the 2nd consecutive loss")
print(f"{SEP}")
print("     (i.e. trades a 'stop at 2 consec losses' rule would have blocked)\n")

rescued_wins = []   # wins we'd have missed by stopping at 2 consec losses
safe_stops   = []   # zones where stopping at 2 would have saved money with no missed wins

for zone_id, grp in df.groupby("zone_id"):
    grp = grp.sort_values("date").reset_index(drop=True)
    if len(grp) < 3:
        continue

    # Walk through trades, track consecutive losses
    consec = 0
    blocked_after = False  # did we cross the 2-loss threshold?
    wins_after_block = []
    losses_after_block = []

    for _, row in grp.iterrows():
        if blocked_after:
            if row["outcome"] == 1:
                wins_after_block.append(row)
            elif row["outcome"] == -1:
                losses_after_block.append(row)
        else:
            if row["outcome"] == -1:
                consec += 1
                if consec >= 2:
                    blocked_after = True
            elif row["outcome"] == 1:
                consec = 0  # reset on win

    if blocked_after:
        if wins_after_block:
            rescued_wins.append({
                "zone_id":   zone_id,
                "wins_missed": len(wins_after_block),
                "losses_avoided": len(losses_after_block),
                "win_pnl":   sum(r["pnl"] for r in wins_after_block),
                "loss_pnl":  sum(r["pnl"] for r in losses_after_block),
                "net_after": sum(r["pnl"] for r in wins_after_block) + sum(r["pnl"] for r in losses_after_block),
                "first_blocked": wins_after_block[0]["date"] if wins_after_block else None,
            })
        else:
            safe_stops.append({
                "zone_id": zone_id,
                "losses_avoided": len(losses_after_block),
                "loss_pnl": sum(r["pnl"] for r in losses_after_block),
            })

print(f"Zones where stopping at 2 consec losses would have BLOCKED a winner: {len(rescued_wins)}")
if rescued_wins:
    print(f"\n{'zone_id':<42} {'W_miss':>7} {'L_avoid':>8} {'W_PnL':>9} {'L_PnL':>9} {'Net_after':>10}")
    print("-" * 90)
    for r in sorted(rescued_wins, key=lambda x: x["net_after"]):
        print(f"{r['zone_id']:<42} {r['wins_missed']:>7} {r['losses_avoided']:>8} "
              f"{r['win_pnl']:>+9.0f} {r['loss_pnl']:>+9.0f} {r['net_after']:>+10.0f}")
    total_win_pnl  = sum(r["win_pnl"]  for r in rescued_wins)
    total_loss_pnl = sum(r["loss_pnl"] for r in rescued_wins)
    print(f"\n  Total wins we'd miss: ${total_win_pnl:+,.0f}")
    print(f"  Total losses we'd avoid (same zones, post-block): ${total_loss_pnl:+,.0f}")
    print(f"  Net of 'blocked' period across these zones: ${total_win_pnl+total_loss_pnl:+,.0f}")

print(f"\nZones where stopping at 2 consec losses is purely beneficial (no missed wins): {len(safe_stops)}")
pure_save = sum(r["loss_pnl"] for r in safe_stops)
print(f"  Losses avoided (pure saves): ${pure_save:+,.0f}")

# ═══════════════════════════════════════════════════════════════════════
# Q2: Tight-SL losers — zone concentration + session concentration
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("Q2: TIGHT-SL LOSERS (SL < 0.3%) — zone vs session concentration")
print(f"{SEP}\n")

tight_loss = df[(df["sl_dist_pct"] < 0.3) & (df["outcome"] == -1)].copy()
tight_win  = df[(df["sl_dist_pct"] < 0.3) & (df["outcome"] ==  1)].copy()
wide_loss  = df[(df["sl_dist_pct"] >= 0.3) & (df["outcome"] == -1)].copy()

print(f"Tight-SL (<0.3%) losses : {len(tight_loss)}  |  wins: {len(tight_win)}")
print(f"Wide-SL  (≥0.3%) losses : {len(wide_loss)}\n")

# --- Zone concentration ---
print("── Zone breakdown (tight-SL losses per zone, top 15) ──")
zone_tight = tight_loss.groupby("zone_id").agg(
    losses=("pnl", "count"),
    net=("pnl", "sum")
).sort_values("losses", ascending=False).head(15)
total_zones_hit = tight_loss["zone_id"].nunique()
top5_zones      = zone_tight.head(5)
top5_losses     = top5_zones["losses"].sum()
print(f"Tight-SL losses spread across {total_zones_hit} distinct zones  "
      f"(top 5 zones account for {top5_losses}/{len(tight_loss)} losses)\n")
print(f"{'zone_id':<42} {'Losses':>7} {'Net$':>9}")
print("-" * 60)
for zid, row in zone_tight.iterrows():
    print(f"{zid:<42} {row['losses']:>7.0f} {row['net']:>+9.0f}")

# --- Session concentration ---
print(f"\n── Session breakdown (tight-SL losses by hour-of-day UTC) ──")
session_tight = tight_loss.groupby("hour").agg(
    losses=("pnl", "count"),
    net=("pnl", "sum")
).sort_values("losses", ascending=False)
session_wide = wide_loss.groupby("hour").agg(
    losses=("pnl", "count"),
).rename(columns={"losses": "wide_losses"})
combined = session_tight.join(session_wide, how="left").fillna(0)
combined["tight_pct"] = (combined["losses"] / (combined["losses"] + combined["wide_losses"]) * 100).round(1)

print(f"\n{'Hour(UTC)':>10} {'T_Losses':>9} {'Net$':>9} {'W_Losses':>9} {'%Tight':>8}")
print("-" * 50)
for hour, row in combined.iterrows():
    flag = "  ← spike" if row["tight_pct"] > 50 and row["losses"] >= 3 else ""
    print(f"{hour:>10} {row['losses']:>9.0f} {row['net']:>+9.0f} {row['wide_losses']:>9.0f} {row['tight_pct']:>7.1f}%{flag}")

# Summary verdict
print(f"\n── Verdict ──")
top1_zone_count = zone_tight.iloc[0]["losses"] if len(zone_tight) else 0
concentration_ratio = top5_losses / len(tight_loss) * 100 if len(tight_loss) else 0
print(f"  Zone concentration  : top 5 zones = {concentration_ratio:.0f}% of tight-SL losses")
uniform_threshold = 100 / max(total_zones_hit, 1)
if concentration_ratio > uniform_threshold * 3:
    print(f"  → CONCENTRATED in specific zones (>{uniform_threshold*3:.0f}% in top 5)")
else:
    print(f"  → UNIFORM — spread across many zones, not zone-specific")

top3_hours = combined.nlargest(3, "losses")
top3_count = top3_hours["losses"].sum()
hour_concentration = top3_count / len(tight_loss) * 100 if len(tight_loss) else 0
print(f"  Session concentration: top 3 hours = {hour_concentration:.0f}% of tight-SL losses")
if hour_concentration > 40:
    print(f"  → CONCENTRATED in specific sessions")
else:
    print(f"  → UNIFORM — spread across the day")

print()
