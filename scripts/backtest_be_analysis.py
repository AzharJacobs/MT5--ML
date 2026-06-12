#!/usr/bin/env python3
"""
backtest_be_analysis.py — Breakeven SL rule impact analysis, USTEC 2023–2025.

NOTE: The BE rule exits trades earlier (at entry level) which unblocks later
signals, so the BE run produces more trades (286 vs 273).  We therefore:
  - Analyse the 27 "50%→SL" trades post-hoc from the baseline max_favour data
  - Report BE run aggregate stats directly from that independent run
  - Use the BE run's own be_stop / be_tp_recovered flags for downside analysis

Run:
    python -X utf8 scripts/backtest_be_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

SEP  = "═" * 68
SEP2 = "─" * 68

PARAMS = dict(
    start="2023-01-01", end="2026-01-01",
    symbol="ustech", fixed_lot=0.05,
    directional_filter=True, allow_neutral=True,
    min_confirmations=1, min_sl_pct=0.25,
)

print("Running BASELINE (no BE rule) ...")
_, df_base = run_backtest(**PARAMS, breakeven_sl=False)

print("\nRunning WITH BREAKEVEN SL rule ...")
_, df_be   = run_backtest(**PARAMS, breakeven_sl=True)

for df in (df_base, df_be):
    df["date"]    = pd.to_datetime(df["date"])
    df["tp_dist"] = abs(df["tp"] - df["entry"])
    df["reached_half_tp"] = df["max_favour"] >= 0.5 * df["tp_dist"]

# ════════════════════════════════════════════════════════════════════════════
# 1.  POST-HOC ANALYSIS OF THE 27 "50%→SL" BASELINE TRADES
# ════════════════════════════════════════════════════════════════════════════
half_sl = df_base[(df_base["reached_half_tp"]) & (df_base["outcome"] == -1)]

print(f"\n{SEP}")
print(f"  1.  POST-HOC: {len(half_sl)} '50%→SL' TRADES UNDER BE RULE")
print(SEP)
print(f"""
  Methodology: max_favour is already captured for each baseline trade.
  Any trade where max_favour >= 0.5 × (tp − entry) AND outcome = SL
  would have been exited at entry (PnL ≈ $0) under the BE rule.
""")

# All of these have max_favour >= 0.5 * tp_dist by definition, so BE fires
# for all of them.  The only question is whether the actual BE run also
# triggered them (it may not — earlier BE exits unlock new entries that
# shift the trade universe).
gross_loss_saved = half_sl["pnl"].sum()     # was negative
print(f"  Modelled BE outcome for all {len(half_sl)} trades:")
print(f"  ├─ Exit price      : entry  (PnL ≈ $0 each)")
print(f"  ├─ Gross loss eliminated : ${-gross_loss_saved:+,.2f}")
print(f"  └─ Average loss per trade (saved): ${-gross_loss_saved/len(half_sl):,.2f}")
print()

print(f"  {'Date':<22}  {'Side':<5}  {'H4 Bias':<14}  "
      f"{'Orig PnL':>10}  {'max_favour':>11}  {'50% dist':>9}")
print(f"  {SEP2}")
for _, r in half_sl.sort_values("date").iterrows():
    half_d = 0.5 * abs(r["tp"] - r["entry"])
    print(f"  {str(r['date']):<22}  {r['side']:<5}  {r['h4_bias']:<14}  "
          f"{r['pnl']:>+10.2f}  {r['max_favour']:>11.2f}  {half_d:>9.2f}")

# ════════════════════════════════════════════════════════════════════════════
# 2.  DOWNSIDE FROM THE BE RUN: BE STOPS WHERE TP LATER OCCURRED
# ════════════════════════════════════════════════════════════════════════════
be_stops      = df_be[df_be["be_stop"]]
be_tp_missed  = be_stops[be_stops["be_tp_recovered"]]   # stopped at BE, then TP hit
be_tp_saved   = be_stops[~be_stops["be_tp_recovered"]]  # stopped at BE, TP never came

print(f"\n{SEP}")
print(f"  2.  DOWNSIDE — BE STOPS WHERE TP WAS HIT AFTER THE STOP")
print(SEP)
print(f"""
  "be_tp_recovered = True" means: price reached 50% TP → BE triggered →
  price returned to entry → BE stop fired → price continued to full TP.
  Without the BE rule the trade would have won (original SL still intact).
""")
print(f"  Total BE stops in BE run              : {len(be_stops)}")
print(f"  ├─ TP hit AFTER BE stop (missed win)  : {len(be_tp_missed)}")
print(f"  └─ TP NOT hit after BE stop (saved SL): {len(be_tp_saved)}")
print()

if len(be_tp_missed):
    print(f"  Missed wins (BE rule cost us these):")
    print(f"  {'Date':<22}  {'Side':<5}  {'H4 Bias':<14}  {'BE PnL':>8}  {'TP (forgone)':>13}")
    print(f"  {SEP2}")
    for _, r in be_tp_missed.sort_values("date").iterrows():
        tp_val = r["tp"]
        forgone = abs(tp_val - r["entry"]) * r["lot"] * 100
        print(f"  {str(r['date']):<22}  {r['side']:<5}  {r['h4_bias']:<14}  "
              f"{r['pnl']:>+8.2f}  ~${forgone:>12,.2f}")
    print(f"\n  Estimated forgone profit (missed TP wins): "
          f"~${sum(abs(r['tp']-r['entry'])*r['lot']*100 for _,r in be_tp_missed.iterrows()):,.2f}")
else:
    print("  None — no BE stops were followed by a TP hit in the remaining window.")

# ════════════════════════════════════════════════════════════════════════════
# 3.  OVERALL METRIC COMPARISON
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print(f"  3.  OVERALL METRIC COMPARISON")
print(SEP)

def _stats(df):
    n   = len(df)
    tp  = int((df["outcome"] ==  1).sum())
    sl  = int((df["outcome"] == -1).sum())
    exp = int((df["outcome"] ==  0).sum())
    be  = int(df.get("be_stop", pd.Series([False]*n)).sum()) if "be_stop" in df.columns else 0
    wr  = tp / max(n, 1) * 100
    net = df["pnl"].sum()
    eq  = df["equity"]
    mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return n, tp, sl, exp, be, wr, net, mdd

bn, btp, bsl, bexp, bbe, bwr, bnet, bmdd = _stats(df_base)
en, etp, esl, eexp, ebe, ewr, enet, emdd = _stats(df_be)

print(f"\n  {'Metric':<28}  {'Baseline':>12}  {'With BE':>12}  {'Delta':>10}")
print(f"  {SEP2}")

def _d0(a, b): s=b-a; return f"+{s:.0f}" if s>=0 else f"{s:.0f}"
def _d1(a, b): s=b-a; return f"+{s:.1f}" if s>=0 else f"{s:.1f}"
def _d2(a, b): s=b-a; return f"+{s:.2f}" if s>=0 else f"{s:.2f}"

rows = [
    ("Total trades",   bn, en),
    ("TP hits",        btp, etp),
    ("SL losses",      bsl, esl),
    ("Expired",        bexp, eexp),
    ("BE scratches",   bbe, ebe),
]
for lbl, vb, ve in rows:
    print(f"  {lbl:<28}  {vb:>12}  {ve:>12}  {_d0(vb,ve):>10}")

print(f"  {'Win rate %':<28}  {bwr:>11.1f}%  {ewr:>11.1f}%  {_d1(bwr,ewr):>9}pp")
print(f"  {'Net PnL $':<28}  ${bnet:>10,.2f}  ${enet:>10,.2f}  ${_d2(bnet,enet):>9}")
print(f"  {'Max drawdown %':<28}  {bmdd:>11.2f}%  {emdd:>11.2f}%  {_d1(bmdd,emdd):>9}pp")
print()

# ── BE activity breakdown ─────────────────────────────────────────────────────
be_trig    = df_be[df_be["be_triggered"]]
be_tp_wins = df_be[df_be["be_triggered"] & (df_be["outcome"] == 1)]

print(f"  ── BE rule activity in BE run ──")
print(f"  Trades where 50% TP was reached (BE triggered) : {len(be_trig)}")
print(f"  ├─ Continued to full TP (BE never fired)        : {len(be_tp_wins)}")
print(f"  └─ Stopped at BE entry level (scratch)          : {len(be_stops)}")
print(f"     ├─ TP hit after BE stop (missed win)          : {len(be_tp_missed)}")
print(f"     └─ TP not hit after BE stop (true saves)      : {len(be_tp_saved)}")

# ── P&L attribution ──────────────────────────────────────────────────────────
print(f"\n  ── P&L attribution (modelled) ──")
print(f"  Baseline 50%→SL losses eliminated (27 trades)  : ${-gross_loss_saved:+,.2f}")
print(f"  Net change in strategy P&L                      : ${enet - bnet:+,.2f}")
print(f"  Extra trades from BE (unlocked by earlier exits): {en - bn:+d}")
print()
print(f"  Note: The BE run generates {en - bn} extra trades because BE stops exit")
print(f"  trades earlier, which frees up the position slot sooner for new signals.")
print(f"  These extra trades contribute ${enet - bnet - (-gross_loss_saved):+,.2f} net")
print(f"  (can be positive or negative depending on market conditions).")
print()
