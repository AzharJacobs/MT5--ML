#!/usr/bin/env python3
"""
backtest_reentry_analysis.py — Re-entry rule analysis, USTEC 2023–2025.

Rule: after a BE stop fires (price returned to entry after hitting 50% TP),
scan forward for the same confirmation signal with same H4 bias and zone.
If found, enter a second trade (Entry 2) targeting the same TP at the same SL.

Run:
    python -X utf8 scripts/backtest_reentry_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

SEP  = "═" * 70
SEP2 = "─" * 70

PARAMS = dict(
    start="2023-01-01", end="2026-01-01",
    symbol="ustech", fixed_lot=0.05,
    directional_filter=True, allow_neutral=True,
    min_confirmations=1, min_sl_pct=0.25,
)

# ── Run 1: Baseline (no BE, no re-entry) ──────────────────────────────────────
print("Running BASELINE ...")
_, df_base = run_backtest(**PARAMS, breakeven_sl=False, reentry_rule=False)

# ── Run 2: BE + re-entry rule ─────────────────────────────────────────────────
print("\nRunning BE + RE-ENTRY RULE ...")
_, df_re   = run_backtest(**PARAMS, breakeven_sl=True,  reentry_rule=True)

for df in (df_base, df_re):
    df["date"]    = pd.to_datetime(df["date"])
    df["tp_dist"] = abs(df["tp"] - df["entry"])

# ── Pull out the 61 BE stops from the re-entry run ───────────────────────────
be_stops = df_re[df_re["be_stop"]].copy().reset_index(drop=True)
tp_recovered   = be_stops[be_stops["be_tp_recovered"]]   # 42 — would hit TP
tp_not_recover = be_stops[~be_stops["be_tp_recovered"]]  # 19 — would fail

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  BE-STOP UNIVERSE  (total: {len(be_stops)})")
print(SEP)
print(f"  ├─ TP eventually hit  (potential wins to capture): {len(tp_recovered)}")
print(f"  └─ TP not hit         (potential losses to avoid): {len(tp_not_recover)}")

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  1.  TRUE POSITIVES — re-entry on the {len(tp_recovered)} TP-recovered trades")
print(SEP)

tp_re_fired  = tp_recovered[tp_recovered["re2_triggered"]]
tp_re_won    = tp_re_fired[tp_re_fired["re2_outcome"] ==  1]
tp_re_lost   = tp_re_fired[tp_re_fired["re2_outcome"] == -1]
tp_re_exp    = tp_re_fired[tp_re_fired["re2_outcome"] ==  0]
tp_no_signal = tp_recovered[~tp_recovered["re2_triggered"]]

print(f"""
  Of the {len(tp_recovered)} TP-recovered trades (price went 50% TP → BE → TP):
  ├─ Re-entry signal fired  : {len(tp_re_fired):>3}  ({len(tp_re_fired)/len(tp_recovered)*100:.0f}%)
  │   ├─ Entry 2 hit TP     : {len(tp_re_won):>3}  → true positive wins
  │   ├─ Entry 2 hit SL     : {len(tp_re_lost):>3}  → re-entered but reversed again
  │   └─ Entry 2 expired    : {len(tp_re_exp):>3}
  └─ No re-entry signal     : {len(tp_no_signal):>3}  ({len(tp_no_signal)/len(tp_recovered)*100:.0f}%)  ← missed captures
""")

if len(tp_re_fired):
    gross_tp  = tp_re_won["re2_pnl"].sum()
    gross_sl  = tp_re_lost["re2_pnl"].sum()
    net_tp_re = tp_re_fired["re2_pnl"].sum()
    print(f"  Entry 2 P&L on TP-recovered group:")
    print(f"  ├─ TP wins  ({len(tp_re_won):>2} trades): ${gross_tp:>+,.2f}")
    print(f"  ├─ SL losses({len(tp_re_lost):>2} trades): ${gross_sl:>+,.2f}")
    print(f"  └─ Net Entry 2 P&L: ${net_tp_re:>+,.2f}")

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  2.  FALSE POSITIVES — re-entry on the {len(tp_not_recover)} failure trades")
print(SEP)

fp_fired  = tp_not_recover[tp_not_recover["re2_triggered"]]
fp_won    = fp_fired[fp_fired["re2_outcome"] ==  1]
fp_lost   = fp_fired[fp_fired["re2_outcome"] == -1]
fp_exp    = fp_fired[fp_fired["re2_outcome"] ==  0]
fp_no_sig = tp_not_recover[~tp_not_recover["re2_triggered"]]

print(f"""
  Of the {len(tp_not_recover)} trades where TP was NOT recovered (genuine failures):
  ├─ Re-entry signal fired  : {len(fp_fired):>3}  ({len(fp_fired)/max(len(tp_not_recover),1)*100:.0f}%)  ← false positives
  │   ├─ Entry 2 hit TP     : {len(fp_won):>3}  → lucky recoveries
  │   ├─ Entry 2 hit SL     : {len(fp_lost):>3}  → double loss
  │   └─ Entry 2 expired    : {len(fp_exp):>3}
  └─ Correctly filtered     : {len(fp_no_sig):>3}  ({len(fp_no_sig)/max(len(tp_not_recover),1)*100:.0f}%)  ← no signal = no re-entry
""")

if len(fp_fired):
    fp_net = fp_fired["re2_pnl"].sum()
    print(f"  Entry 2 P&L on failure group:")
    print(f"  ├─ TP wins  ({len(fp_won):>2} trades): ${fp_won['re2_pnl'].sum():>+,.2f}")
    print(f"  ├─ SL losses({len(fp_lost):>2} trades): ${fp_lost['re2_pnl'].sum():>+,.2f}")
    print(f"  └─ Net false-positive P&L: ${fp_net:>+,.2f}")

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  3.  ENTRY 2 SIGNAL ANALYSIS — confirmation filter quality")
print(SEP)

all_re2 = be_stops[be_stops["re2_triggered"]]
all_no_re2 = be_stops[~be_stops["re2_triggered"]]

precision = len(tp_re_won) / max(len(all_re2), 1) * 100
recall    = len(tp_re_won) / max(len(tp_recovered), 1) * 100
fp_rate   = len(fp_fired) / max(len(tp_not_recover), 1) * 100

print(f"""
  Filter quality on Entry 2 signals:
  ├─ Total re-entries triggered : {len(all_re2)} / {len(be_stops)}
  ├─ Precision (TP-wins / all re2)  : {precision:.0f}%
  │    → of all re-entries taken, this % produced a TP hit
  ├─ Recall (captured / 42 winners) : {recall:.0f}%
  │    → of the 42 TP-recoverable trades, this % were captured by re-entry
  └─ False-positive rate (FP / 19 failures): {fp_rate:.0f}%
       → of the 19 genuine failures, this % still triggered a re-entry
""")

# Signal types in re-entries vs no re-entries
if len(all_re2):
    from collections import Counter
    sig_counter = Counter()
    for sigs in all_re2["re2_signals"].dropna():
        for s in sigs.split("|"):
            sig_counter[s] += 1
    print(f"  Re-entry signal types (Entry 2):")
    for sig, cnt in sig_counter.most_common():
        print(f"    {sig:<30} : {cnt}")

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  4.  OVERALL METRIC COMPARISON")
print(SEP)

# Baseline metrics
bn   = len(df_base)
b_tp = int((df_base["outcome"] == 1).sum())
b_sl = int((df_base["outcome"] == -1).sum())
b_wr = b_tp / max(bn, 1) * 100
b_net = df_base["pnl"].sum()
b_eq  = df_base["equity"]
b_mdd = ((b_eq - b_eq.cummax()) / b_eq.cummax()).min() * 100

# Entry 1 (BE run — 286 trades, no re-entries counted)
e1n   = len(df_re)
e1_tp = int((df_re["outcome"] == 1).sum())
e1_sl = int((df_re["outcome"] == -1).sum())
e1_be = int(df_re["be_stop"].sum())
e1_wr = e1_tp / max(e1n, 1) * 100
e1_net = df_re["pnl"].sum()

# Entry 2 (re-entries only)
all_re2_trades = df_re[df_re["re2_triggered"]]
e2_tp  = int((all_re2_trades["re2_outcome"] == 1).sum())
e2_sl  = int((all_re2_trades["re2_outcome"] == -1).sum())
e2_exp = int((all_re2_trades["re2_outcome"] == 0).sum())
e2_wr  = e2_tp / max(len(all_re2_trades), 1) * 100
e2_net = all_re2_trades["re2_pnl"].sum()

# Combined (Entry 1 BE + Entry 2)
comb_net = e1_net + e2_net
comb_tp  = e1_tp + e2_tp
comb_sl  = e1_sl + e2_sl
comb_n   = e1n + len(all_re2_trades)
comb_wr  = comb_tp / max(comb_n, 1) * 100

# Combined equity curve (approximate — add re2 pnl on top of be run equity)
# Build cumulative re2 pnl aligned to trade order
df_re["re2_pnl_safe"] = df_re["re2_pnl"].fillna(0)
df_re["equity_combined"] = df_re["equity"] + df_re["re2_pnl_safe"].cumsum()
comb_eq  = df_re["equity_combined"]
comb_mdd = ((comb_eq - comb_eq.cummax()) / comb_eq.cummax()).min() * 100

print(f"""
  {'Metric':<30}  {'Baseline':>12}  {'E1+E2 (BE+RE)':>14}  {'Delta':>10}
  {SEP2}""")

def _d(a, b, fmt=".1f"):
    s = b - a
    fmts = f"+{s:{fmt}}" if s >= 0 else f"{s:{fmt}}"
    return fmts

rows = [
    ("Entry 1 trades",     bn,    e1n),
    ("Entry 2 trades",     0,     len(all_re2_trades)),
    ("Total trade events", bn,    comb_n),
    ("E1 TP hits",         b_tp,  e1_tp),
    ("E2 TP hits",         0,     e2_tp),
    ("Total TP hits",      b_tp,  comb_tp),
    ("E1 SL losses",       b_sl,  e1_sl),
    ("E2 SL losses",       0,     e2_sl),
    ("BE scratches (E1)",  0,     e1_be),
]
for lbl, vb, ve in rows:
    d = _d(vb, ve, ".0f")
    print(f"  {lbl:<30}  {vb:>12}  {ve:>14}  {d:>10}")

print(f"  {'Win rate %':<30}  {b_wr:>11.1f}%  {comb_wr:>13.1f}%  {_d(b_wr, comb_wr):>9}pp")
print(f"  {'Net PnL $':<30}  ${b_net:>10,.2f}  ${comb_net:>13,.2f}  ${_d(b_net, comb_net, '.2f'):>9}")
print(f"  {'Max drawdown %':<30}  {b_mdd:>11.2f}%  {comb_mdd:>13.2f}%  {_d(b_mdd, comb_mdd):>9}pp")

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  5.  ENTRY 2 TRADE LIST (all {len(all_re2_trades)} re-entries)")
print(SEP)
print(f"\n  {'Date':<22}  {'Side':<5}  {'H4 Bias':<14}  "
      f"{'be_tp_rec':>10}  {'E2 Sig':<30}  {'E2 Out':>7}  {'E2 PnL':>9}")
print(f"  {SEP2}")
for _, r in all_re2_trades.sort_values("date").iterrows():
    out_code = {1: "TP", -1: "SL", 0: "exp"}.get(int(r["re2_outcome"]), "?")
    tp_rec = "TP-recov" if r["be_tp_recovered"] else "FAILED  "
    print(
        f"  {str(r['date']):<22}  {r['side']:<5}  {r['h4_bias']:<14}  "
        f"{tp_rec:>10}  {r['re2_signals']:<30}  {out_code:>7}  {r['re2_pnl']:>+9.2f}"
    )

print(f"\n  Entry 2 net P&L: ${e2_net:+,.2f}  |  "
      f"Entry 2 WR: {e2_wr:.1f}%  |  "
      f"Trades: {len(all_re2_trades)}")

# ────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  6.  VERDICT — is same-confirmation filter strict enough?")
print(SEP)
print(f"""
  False positive rate (re-entry on 19 failures) : {fp_rate:.0f}%
  TP capture rate (re-entry on 42 TP-recoveries): {recall:.0f}%
  Entry 2 net PnL (all re-entries)              : ${e2_net:+,.2f}
  Combined net PnL vs baseline                   : ${_d(b_net, comb_net, '.2f')}

  Interpretation:
  ├─ FP rate ≤ 20%  → filter is adequate as-is
  ├─ FP rate 20–40% → consider adding M15-close requirement
  └─ FP rate > 40%  → strict filter needed (e.g. rejection-wick only)
""")
