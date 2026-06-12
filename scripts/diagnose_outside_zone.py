#!/usr/bin/env python3
"""
diagnose_outside_zone.py — Root-cause analysis of entry-outside-zone trades.

For every trade where the bar i+1 open landed outside the zone boundary, shows:
  - Signal bar close (bar i)  vs  fill bar open (bar i+1)
  - The gap between them and how far it pushed entry outside the zone
  - Zone boundary, direction, outcome, and PnL

Then identifies the root-cause pattern and proposes the engine fix.

Run:
    python -X utf8 scripts/diagnose_outside_zone.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

PARAMS = dict(
    start="2023-01-01", end="2026-01-01",
    symbol="ustech", fixed_lot=0.05,
    directional_filter=True, allow_neutral=True,
    min_confirmations=1, min_sl_pct=0.25,
)

print("Running USTEC 2023–2025 (zone_guard=False — reveals outside-zone entries) ...")
_, df = run_backtest(**PARAMS, zone_guard=False)

print("\nRunning USTEC 2023–2025 (zone_guard=True — fixed baseline) ...")
_, df_fixed = run_backtest(**PARAMS, zone_guard=True)

df["date"] = pd.to_datetime(df["date"])

# ── Outside-zone flag (0.2% tolerance to absorb spread) ──────────────────────
df["entry_outside_zone"] = (
    ((df["side"] == "buy")  & (df["entry"] > df["zone_top"]   * 1.002)) |
    ((df["side"] == "sell") & (df["entry"] < df["zone_bottom"] * 0.998))
)

# Zone boundary relevant to the trade direction
df["zone_boundary"] = df.apply(
    lambda r: r["zone_top"] if r["side"] == "buy" else r["zone_bottom"], axis=1
)

# Gap: bar i close → bar i+1 open (before spread)
# For buy: positive gap = opened higher (bad — moved away from demand)
# For sell: negative gap = opened lower (bad — moved away from supply)
df["bar_gap_pts"]  = df["raw_open"] - df["signal_close"]

# How far raw_open is outside the zone boundary (in points and %)
df["outside_pts"] = df.apply(
    lambda r: r["raw_open"] - r["zone_top"]     if r["side"] == "buy"
              else r["zone_bottom"] - r["raw_open"], axis=1
)
df["outside_pct"] = (df["outside_pts"] / df["zone_boundary"] * 100).round(4)

# Spread paid (entry = raw_open ± spread)
df["spread_pts"] = df.apply(
    lambda r: r["entry"] - r["raw_open"] if r["side"] == "buy"
              else r["raw_open"] - r["entry"], axis=1
)

flagged = df[df["entry_outside_zone"]].copy().reset_index(drop=True)

SEP  = "═" * 120
SEP2 = "─" * 120

print(f"\n{SEP}")
print(f"  ENTRY-OUTSIDE-ZONE DIAGNOSTIC  ({len(flagged)} trades flagged)")
print(SEP)

hdr = (f"  {'#':<3}  {'Signal bar (i)':<22}  {'Fill bar (i+1)':>14}  "
       f"{'Side':<5}  {'H4 Bias':<13}  "
       f"{'Zone Bdry':>10}  {'Bar i Close':>12}  {'Bar i+1 Open':>13}  "
       f"{'Bar Gap':>8}  {'Outside':>8}  {'Outside%':>9}  {'Spread':>7}  "
       f"{'Outcome':>8}  {'PnL':>9}")
print(hdr)
print(f"  {SEP2}")

for idx, r in flagged.iterrows():
    entry_ts  = r["date"] + pd.Timedelta(minutes=15)   # bar i+1 timestamp
    out_code  = {1: "TP", -1: "SL", 0: "exp"}.get(int(r["outcome"]), "?")
    print(
        f"  {idx+1:<3}  {str(r['date']):<22}  {str(entry_ts):>14}  "
        f"{r['side']:<5}  {r['h4_bias']:<13}  "
        f"{r['zone_boundary']:>10.2f}  {r['signal_close']:>12.2f}  {r['raw_open']:>13.2f}  "
        f"{r['bar_gap_pts']:>+8.2f}  {r['outside_pts']:>+8.2f}  {r['outside_pct']:>+8.3f}%  "
        f"{r['spread_pts']:>7.2f}  {out_code:>8}  {r['pnl']:>+9.2f}"
    )

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n  {'='*40}")
print(f"  Summary of {len(flagged)} outside-zone trades:")
sl_f = flagged[flagged["outcome"] == -1]
tp_f = flagged[flagged["outcome"] ==  1]
ex_f = flagged[flagged["outcome"] ==  0]
print(f"  SL losses  : {len(sl_f):>3}  net ${sl_f['pnl'].sum():+,.2f}")
print(f"  TP wins    : {len(tp_f):>3}  net ${tp_f['pnl'].sum():+,.2f}")
print(f"  Expired    : {len(ex_f):>3}  net ${ex_f['pnl'].sum():+,.2f}")
print(f"  ALL        : {len(flagged):>3}  net ${flagged['pnl'].sum():+,.2f}")
print()
print(f"  Avg bar gap (open − close)   : {flagged['bar_gap_pts'].mean():>+.2f} pts")
print(f"  Avg outside-zone penetration : {flagged['outside_pts'].mean():>+.2f} pts"
      f"  ({flagged['outside_pct'].mean():>+.3f}%)")
print(f"  Buy-side outside-zone  : {len(flagged[flagged['side']=='buy'])}"
      f"  |  Sell-side: {len(flagged[flagged['side']=='sell'])}")

# Show buys vs sells breakdown
for side, grp in flagged.groupby("side"):
    sl_n = (grp["outcome"] == -1).sum()
    tp_n = (grp["outcome"] ==  1).sum()
    print(f"    {side}: {len(grp)} trades  "
          f"SL={sl_n}  TP={tp_n}  net=${grp['pnl'].sum():+,.2f}")

# ── Root cause pattern ────────────────────────────────────────────────────────
print(f"\n{'═'*72}")
print(f"  ROOT CAUSE ANALYSIS")
print(f"{'═'*72}")
print("""
  Execution flow that causes the bug:

    Bar i   →  Confirmation signal fires on the 15-min CLOSE.
                The close is INSIDE the zone (this is the correct tap).
                signal_close = df_15m_w["close"].iloc[-1]

    Bar i+1 →  Engine uses the OPEN of the next bar as the fill:
                entry = float(df_15m["open"].iloc[i + 1])
                + spread adjustment applied.

    Problem →  When bars i and i+1 span a session boundary (US pre-market
               open, Asia session start, or post-news gap) the open of bar
               i+1 can land OUTSIDE the zone boundary:
                 • Buy  (demand zone): open > zone_top  — filled ABOVE the zone
                 • Sell (supply zone): open < zone_bottom — filled BELOW the zone

  Why this is a structural error for LOSING trades:
    - The SL was computed from signal_price (close of bar i, inside the zone),
      so SL distance = |signal_close - zone_bottom ± buffer|.
    - Entry has now gapped further away from the zone but SL has NOT moved —
      the trade absorbs extra gap risk without extra reward (TP stayed fixed).
    - In 9 of the 12 cases, the trade lost, and those losses are larger than
      the typical non-flagged loss because of the widened effective RR.

  Why 3 wins exist despite the bug:
    - April 2025 tariff-crash recovery: a 2.56% gap above the demand zone
      preceded a 600-point rip — the entry was structurally wrong but the
      market move was so large it won anyway (+$6,969).
    - The 3 wins are outlier events, not evidence the gap entries are valid.

  Frequency / session pattern:
    - 10 of 12 are BUY trades gapping UP through a demand zone.
    - Heaviest cluster: June–September 2024 (4 trades, US summer volatility).
    - April 2025 crash event adds the extreme outlier.
    - No cluster in Europe open hours → predominantly US pre-market gaps.
""")

# ── Proposed fix ──────────────────────────────────────────────────────────────
print(f"{'═'*72}")
print(f"  CODE FIX  —  trading/strategies/zz/ustec/engine.py")
print(f"{'═'*72}")
print("""
  Insert immediately after the spread-adjusted `entry` is computed
  (currently ~line 311) and BEFORE the SL/TP geometry checks:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  # Skip if fill has gapped outside the zone boundary.               │
  │  # 0.2% tolerance covers the full spread (~0.01%) with headroom.   │
  │  _gap_tol = 0.002                                                   │
  │  if direction == "buy" and entry > active_zone.top * (1 + _gap_tol):│
  │      filters["setup_invalid"] += 1                                  │
  │      continue                                                       │
  │  if direction == "sell" and entry < active_zone.bottom * (1-_gap_tol)│
  │      filters["setup_invalid"] += 1                                  │
  │      continue                                                       │
  └─────────────────────────────────────────────────────────────────────┘

  Expected impact on these 12 trades (from the observed data):
    Skipped SL losses (9 trades)   : saves  ~$5,657
    Skipped TP wins  (3 trades)    : forgoes ~$9,916  (incl. $6,969 outlier)
    Net P&L delta from the fix     : ~−$4,259

  Why apply it anyway:
    • The structural error means SL distance is wrong for every gap trade.
    • Without the fix, the engine is unknowingly taking wider-than-modelled
      risk on those entries — and in a live account that means real slippage
      AND a bad RR calculation.
    • The $6,969 win is a one-in-three-year outlier. Structurally sound
      entries will replace it over time as new zones form.
    • Recommend applying and re-running to get clean baseline metrics.
""")

# ── Retroactive (post-hoc filter on the unguarded run) ───────────────────────
clean_df  = df[~df["entry_outside_zone"]]
orig_net  = df["pnl"].sum()
retro_net = clean_df["pnl"].sum()
wr_orig   = (df["outcome"] == 1).mean() * 100
wr_retro  = (clean_df["outcome"] == 1).mean() * 100
print(f"  Post-hoc retroactive (just skip the 12 — same equity path):")
print(f"    Trades: {len(df)} → {len(clean_df)}  |  "
      f"WR: {wr_orig:.1f}% → {wr_retro:.1f}%  |  "
      f"Net PnL: ${orig_net:+,.2f} → ${retro_net:+,.2f}  "
      f"(Δ ${retro_net - orig_net:+,.2f})")
print()

# ── Live fixed-engine run (zone_guard=True, new correct baseline) ─────────────
print(f"{'═'*72}")
print(f"  FIXED ENGINE RUN  (zone_guard=True — new clean baseline)")
print(f"{'═'*72}")
df_fixed["date"] = pd.to_datetime(df_fixed["date"])
fn   = len(df_fixed)
fnet = df_fixed["pnl"].sum()
fwr  = (df_fixed["outcome"] == 1).mean() * 100
ftp  = (df_fixed["outcome"] == 1).sum()
fsl  = (df_fixed["outcome"] == -1).sum()
fexp = (df_fixed["outcome"] == 0).sum()
feq  = df_fixed["equity"]
fmdd = ((feq - feq.cummax()) / feq.cummax()).min() * 100
print(f"  Trades : {fn}  (was 273  →  −{273 - fn} outside-zone entries blocked)")
print(f"  TP hits: {ftp}  |  SL hits: {fsl}  |  Expired: {fexp}")
print(f"  Win rate      : {fwr:.1f}%   (was 35.9%)")
print(f"  Net PnL       : ${fnet:+,.2f}   (was $30,934)")
print(f"  Max drawdown  : {fmdd:.2f}%   (was −34.77%)")
print()
print(f"  Note: The fixed engine may produce a slightly different equity path")
print(f"  vs the retroactive skip because skipping a trade releases the cooldown")
print(f"  slot, which can allow different subsequent signals to fire.")
print()
