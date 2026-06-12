#!/usr/bin/env python3
"""
backtest_zz_2023_2025_detailed.py — USTEC Zone-to-Zone full 2023–2025 analysis.

Sections:
  1. Random sample month — total trades
  2. Per-month breakdown  — full-TP hits vs 50%-then-SL
  3. Losing-trades report — 50%-then-SL count, contra-trend count, per-month
  4. Stale-price / geometry flags
  5. Overall 3-year summary

Run:
    python -X utf8 scripts/backtest_zz_2023_2025_detailed.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

SEP  = "═" * 68
SEP2 = "─" * 68

print("Running USTEC Zone-to-Zone backtest  2023-01-01 → 2026-01-01 ...")
result = run_backtest(
    start="2023-01-01",
    end="2026-01-01",
    symbol="ustech",
    fixed_lot=0.05,
    directional_filter=True,
    allow_neutral=True,
    min_confirmations=1,
    min_sl_pct=0.25,
)

if not result or len(result) < 2:
    print("No trades returned — check DB connection and date range.")
    sys.exit(1)

_, df = result

df["date"]      = pd.to_datetime(df["date"])
df["exit_date"] = pd.to_datetime(df["exit_date"])
df["month"]     = df["date"].dt.to_period("M")
df["year"]      = df["date"].dt.year

# ── Derived columns ──────────────────────────────────────────────────────────

# 50% TP threshold in price points from entry
df["tp_dist"]        = abs(df["tp"] - df["entry"])
# max_favour is already in price-point terms (fh-entry for buy, entry-fl for sell)
df["reached_half_tp"]= df["max_favour"] >= 0.5 * df["tp_dist"]
df["half_tp_then_sl"]= df["reached_half_tp"] & (df["outcome"] == -1)
df["full_tp_hit"]    = df["outcome"] == 1
df["expired"]        = df["outcome"] == 0

# Trend alignment
def _align(row):
    b, s = row["h4_bias"], row["side"]
    if (b == "bullish" and s == "buy") or (b == "bearish" and s == "sell"):
        return "aligned"
    if b in ("neutral_up", "neutral_down", "neutral"):
        return "neutral"
    return "counter"

df["alignment"]    = df.apply(_align, axis=1)
df["against_trend"]= df["alignment"] == "counter"

# Stale-price / geometry flags:
#   A) Zone height > 1% of entry price — oversized zone, precision degraded
#   B) Entry outside zone boundary at entry bar
#      (buy demand zone: entry > zone_top means price already blew through)
#      (sell supply zone: entry < zone_bottom)
df["zone_height"]      = df["zone_top"] - df["zone_bottom"]
df["zone_height_pct"]  = df["zone_height"] / df["entry"] * 100
df["wide_zone"]        = df["zone_height_pct"] > 1.0

df["entry_outside_zone"] = (
    ((df["side"] == "buy")  & (df["entry"] > df["zone_top"]   * 1.002)) |
    ((df["side"] == "sell") & (df["entry"] < df["zone_bottom"] * 0.998))
)
df["stale_flag"] = df["wide_zone"] | df["entry_outside_zone"]

win  = df["outcome"] == 1
loss = df["outcome"] == -1

# ── Helpers ──────────────────────────────────────────────────────────────────

def month_summary(grp):
    n       = len(grp)
    wins    = int((grp["outcome"] == 1).sum())
    wr      = wins / max(n, 1) * 100
    net_pnl = grp["pnl"].sum()
    h_sl    = int(grp["half_tp_then_sl"].sum())
    ftp     = int(grp["full_tp_hit"].sum())
    eq      = grp["equity"].reset_index(drop=True)
    if len(eq) > 1:
        dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    else:
        dd = 0.0
    return n, wins, wr, net_pnl, dd, h_sl, ftp

# ════════════════════════════════════════════════════════════════════════════
# 1. RANDOM SAMPLE MONTH
# ════════════════════════════════════════════════════════════════════════════

months  = sorted(df["month"].unique().tolist())
pick    = random.choice(months)
m_df    = df[df["month"] == pick]
n, wins, wr, net_pnl, dd, h_sl, ftp = month_summary(m_df)

print(f"\n{SEP}")
print(f"  1. RANDOM SAMPLE MONTH — {pick}")
print(SEP)
print(f"  Total trades         : {n}")
print(f"  Full TP hits         : {ftp}")
print(f"  Reached 50% → then SL: {h_sl}")
print(f"  Win rate             : {wr:.1f}%")
print(f"  Net PnL              : ${net_pnl:+.2f}")
print(f"  Max intra-month DD   : {dd:.2f}%")
print()

# ════════════════════════════════════════════════════════════════════════════
# 2. PER-MONTH BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print(f"  2. PER-MONTH TRADE OUTCOME BREAKDOWN")
print(SEP)
print(f"  {'Month':<10}  {'Trades':>6}  {'Full-TP':>7}  {'50%→SL':>7}  {'Expired':>7}  {'WR%':>6}  {'Net PnL':>10}  {'MaxDD%':>8}")
print(SEP2)

yearly_block = None
for m, grp in df.groupby("month"):
    yr = m.year
    if yearly_block != yr:
        if yearly_block is not None:
            # year subtotal
            ydf = df[df["year"] == yearly_block]
            yn, yw, ywr, ypnl, ydd, yh, yftp = month_summary(ydf)
            print(SEP2)
            print(f"  {'  ' + str(yearly_block) + ' TOTAL':<10}  {yn:>6}  {yftp:>7}  {yh:>7}"
                  f"  {int(ydf['expired'].sum()):>7}  {ywr:>5.1f}%  ${ypnl:>9.2f}  {ydd:>7.2f}%")
            print(SEP2)
        yearly_block = yr
        print(f"  ── {yr} ──")

    n, wins, wr, net_pnl, dd, h_sl, ftp = month_summary(grp)
    exp = int(grp["expired"].sum())
    marker = " ◄ sampled" if m == pick else ""
    print(f"  {str(m):<10}  {n:>6}  {ftp:>7}  {h_sl:>7}  {exp:>7}  {wr:>5.1f}%  ${net_pnl:>9.2f}  {dd:>7.2f}%{marker}")

# final year subtotal
if yearly_block is not None:
    ydf = df[df["year"] == yearly_block]
    yn, yw, ywr, ypnl, ydd, yh, yftp = month_summary(ydf)
    print(SEP2)
    print(f"  {'  ' + str(yearly_block) + ' TOTAL':<10}  {yn:>6}  {yftp:>7}  {yh:>7}"
          f"  {int(ydf['expired'].sum()):>7}  {ywr:>5.1f}%  ${ypnl:>9.2f}  {ydd:>7.2f}%")
    print(SEP2)
print()

# ════════════════════════════════════════════════════════════════════════════
# 3. LOSING-TRADES REPORT
# ════════════════════════════════════════════════════════════════════════════

losers      = df[df["outcome"] == -1]
half_tp_sl  = df[df["half_tp_then_sl"]]
contra_loss = losers[losers["against_trend"]]

print(f"\n{SEP}")
print(f"  3. LOSING-TRADES REPORT")
print(SEP)
print(f"  Total SL trades                      : {len(losers)}")
print(f"  ├─ Reached 50% TP then reversed → SL : {len(half_tp_sl)}"
      f"  ({len(half_tp_sl)/max(len(losers),1)*100:.1f}% of losses)")
print(f"  └─ Against prevailing H4 trend       : {len(contra_loss)}"
      f"  ({len(contra_loss)/max(len(losers),1)*100:.1f}% of losses)")
print()

# Per-month losing breakdown
print(f"  {'Month':<10}  {'Total L':>7}  {'50%→SL':>7}  {'Contra':>7}  "
      f"{'WR%':>6}  {'Net PnL':>10}  {'MaxDD%':>8}")
print(SEP2)
for m, grp in df.groupby("month"):
    n, wins, wr, net_pnl, dd, h_sl, ftp = month_summary(grp)
    lg   = int((grp["outcome"] == -1).sum())
    ct   = int(grp[grp["outcome"] == -1]["against_trend"].sum())
    if n == 0:
        continue
    print(f"  {str(m):<10}  {lg:>7}  {h_sl:>7}  {ct:>7}  {wr:>5.1f}%  ${net_pnl:>9.2f}  {dd:>7.2f}%")
print()

# Counter-trend loss profile
print(f"\n  Counter-trend loss detail (all years):")
print(f"  {'H4 Bias':<16}  {'Side':<6}  {'Trades':>6}  {'WR%':>6}  {'Net$':>10}")
print(f"  {'-'*50}")
for (bias, side), grp in contra_loss.groupby(["h4_bias", "side"]):
    all_contra = df[(df["h4_bias"] == bias) & (df["side"] == side) & (df["against_trend"])]
    wr = (all_contra["outcome"] == 1).mean() * 100
    print(f"  {bias:<16}  {side:<6}  {len(all_contra):>6}  {wr:>5.1f}%  {all_contra['pnl'].sum():>+10.2f}")
print()

# ════════════════════════════════════════════════════════════════════════════
# 4. STALE-PRICE / GEOMETRY FLAGS
# ════════════════════════════════════════════════════════════════════════════

flagged = df[df["stale_flag"]].copy()
wide    = df[df["wide_zone"]]
outside = df[df["entry_outside_zone"]]

print(f"\n{SEP}")
print(f"  4. STALE-PRICE / GEOMETRY FLAGS")
print(SEP)
print(f"  Total flagged trades : {len(flagged)} / {len(df)}")
print(f"  ├─ Wide zone (>1% of entry)          : {len(wide)}")
print(f"  └─ Entry outside zone boundary        : {len(outside)}")

if len(flagged):
    print(f"\n  Flagged trade outcomes:")
    print(f"  {'Month':<10}  {'Side':<6}  {'H4 Bias':<14}  {'ZoneH%':>7}  "
          f"{'OutsideZ':>9}  {'Outcome':>8}  {'PnL':>8}")
    print(f"  {'-'*68}")
    for _, r in flagged.iterrows():
        out_str  = "yes" if r["entry_outside_zone"] else "no"
        out_code = {1: "TP", -1: "SL", 0: "exp"}.get(int(r["outcome"]), "?")
        print(f"  {str(r['month']):<10}  {r['side']:<6}  {r['h4_bias']:<14}  "
              f"{r['zone_height_pct']:>6.2f}%  {out_str:>9}  {out_code:>8}  {r['pnl']:>+8.2f}")
    print()

    # WR for flagged vs clean
    clean = df[~df["stale_flag"]]
    f_wr  = (flagged["outcome"] == 1).mean() * 100
    c_wr  = (clean["outcome"]   == 1).mean() * 100
    print(f"  Flagged WR: {f_wr:.1f}%  |  Clean WR: {c_wr:.1f}%  "
          f"(delta: {f_wr - c_wr:+.1f}pp)")
    print(f"  Flagged net PnL: ${flagged['pnl'].sum():+.2f}  |  "
          f"Clean net PnL: ${clean['pnl'].sum():+.2f}")
else:
    print("  No geometry/stale issues detected.")
print()

# ════════════════════════════════════════════════════════════════════════════
# 5. OVERALL 2023–2025 SUMMARY
# ════════════════════════════════════════════════════════════════════════════

total    = len(df)
tp_hits  = int((df["outcome"] == 1).sum())
sl_hits  = int((df["outcome"] == -1).sum())
exp_hits = int((df["outcome"] == 0).sum())
wr       = tp_hits / max(total, 1) * 100
net_pnl  = df["pnl"].sum()
eq       = df["equity"]
max_dd   = ((eq - eq.cummax()) / eq.cummax()).min() * 100

print(f"\n{SEP}")
print(f"  5. OVERALL 2023–2025 SUMMARY  (USTEC, fixed lot 0.05)")
print(SEP)
print(f"  {'Total trades':<30}: {total}")
print(f"  {'TP hits':<30}: {tp_hits}")
print(f"  {'SL hits':<30}: {sl_hits}")
print(f"  {'Expired (time-out)':<30}: {exp_hits}")
print(f"  {'Win rate':<30}: {wr:.1f}%")
print(f"  {'Net PnL':<30}: ${net_pnl:+.2f}")
print(f"  {'Max drawdown':<30}: {max_dd:.2f}%")
print()
print(f"  ── Specific metrics ──")
print(f"  {'50%-TP then SL':<30}: {int(df['half_tp_then_sl'].sum())} trades"
      f"  ({df['half_tp_then_sl'].sum()/max(sl_hits,1)*100:.1f}% of all losses)")
print(f"  {'Counter-trend entries':<30}: {int(df['against_trend'].sum())} total"
      f"  ({int(contra_loss.shape[0])} resulted in SL)")
print(f"  {'Stale/geometry flagged':<30}: {len(flagged)} trades")
print()

print(f"  ── Alignment split ──")
print(f"  {'Alignment':<12}  {'Trades':>7}  {'Wins':>6}  {'WR%':>7}  {'Net$':>10}")
print(f"  {'-'*46}")
for aln in ("aligned", "neutral", "counter"):
    sub = df[df["alignment"] == aln]
    if len(sub):
        flag = "  ← review" if aln == "counter" and sub["full_tp_hit"].mean() < 0.30 else ""
        print(f"  {aln:<12}  {len(sub):>7}  {int(sub['full_tp_hit'].sum()):>6}"
              f"  {sub['full_tp_hit'].mean()*100:>6.1f}%  {sub['pnl'].sum():>+10.2f}{flag}")
print()
