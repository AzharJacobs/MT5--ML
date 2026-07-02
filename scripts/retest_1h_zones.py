#!/usr/bin/env python3
"""
1H zone retest: 2023-2024
Settings: LH+HL fix, conf=2, RR<=5.0, neutral=OFF, H4_WINDOW=150, 1H zones ON.
Baseline (same settings, 1H OFF) is hardcoded from prior run for comparison.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

COMMON = dict(
    start="2023-01-01",
    end="2025-01-01",
    cash=150.0,
    symbol="ustech",
    min_confirmations=2,
    data_source="mt5",
    max_rr=5.0,
    allow_neutral=False,
    silent=True,
)

print("Running 2023-2024  (1H zones ON) …")
result = run_backtest(**COMMON, use_1h_zones=True)

if not result:
    print("No data returned.")
    sys.exit(1)

m, df = result
df["rr"] = abs(df["tp"] - df["entry"]) / abs(df["entry"] - df["sl"])

DIV  = "─" * 92
DIV2 = "─" * 56

# ── 1. Per-trade list ────────────────────────────────────────────────────────
print()
print(DIV)
print("  PER-TRADE LIST  |  2023-2024  |  H4 + 1H zones  |  conf=2  RR<=5.0  neutral=OFF")
print(DIV)
print(f"  {'Date':<22} {'Side':<5} {'Bias':<16} {'ZoneTF':<7} {'Signals':<38} {'RR':>5}  Out")
print(DIV)
for _, r in df.sort_values("date").iterrows():
    sigs = r["signals"].replace("|", " + ")
    out  = "WIN " if r["outcome"] == 1 else "LOSS"
    ztf  = r.get("zone_tf", "H4")
    print(f"  {str(r['date']):<22} {r['side']:<5} {r['h4_bias']:<16} {ztf:<7} {sigs:<38} {r['rr']:>5.2f}  {out}")

# ── 2. H4 vs 1H split ───────────────────────────────────────────────────────
print()
print(DIV2)
print("  H4 vs 1H ZONE SPLIT")
print(DIV2)
for tf_label in ["H4", "1H"]:
    grp = df[df.get("zone_tf", pd.Series(["H4"]*len(df), index=df.index)) == tf_label] \
        if "zone_tf" in df.columns else df[df["zone_tf"] == tf_label]
    if grp.empty:
        print(f"  {tf_label}: no trades")
        continue
    wr  = (grp["outcome"] == 1).mean() * 100
    avg = grp["rr"].mean()
    wins = int((grp["outcome"] == 1).sum())
    loss = int((grp["outcome"] == -1).sum())
    print(f"  {tf_label} zones  : {len(grp):>3} trades  {wins}W/{loss}L  WR={wr:.0f}%  AvgRR={avg:.2f}")

print()
print(DIV2)
print("  OVERALL  (H4 + 1H combined)")
print(DIV2)
total = len(df)
wr_all = (df["outcome"] == 1).mean() * 100
print(f"  Trades   : {total}")
print(f"  Win rate : {wr_all:.1f}%")
print(f"  Net PnL  : {m['net_pnl']}")
print(f"  Equity   : {m['start_cash']} → {m['final_equity']}")
print(f"  Max DD   : {m['max_drawdown_%']}%")

print()
print("  Bias breakdown:")
for bias, grp in df.groupby("h4_bias"):
    wr = (grp["outcome"] == 1).mean() * 100
    print(f"    {bias:<16} {len(grp):>3} trades  WR={wr:.0f}%")

# ── 3. Monthly summary ───────────────────────────────────────────────────────
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
all_months  = pd.period_range("2023-01", "2024-12", freq="M")

print()
print(DIV)
print("  MONTHLY SUMMARY")
print(DIV)
print(f"  {'Month':<9} {'Tr':>3} {'W':>3} {'L':>3}  {'WR%':>6}  {'Net PnL':>10}  {'ZoneTF'}")
print(DIV)

for mo in all_months:
    grp = df[df["month"] == mo]
    if grp.empty:
        print(f"  {str(mo):<9}   0   -   -     N/A         -")
        continue
    w   = int((grp["outcome"] == 1).sum())
    l   = int((grp["outcome"] == -1).sum())
    wr  = w / len(grp) * 100
    pnl = grp["pnl"].sum()
    tfs = grp["zone_tf"].value_counts().to_dict() if "zone_tf" in grp.columns else {}
    tf_str = "  ".join(f"{k}:{v}" for k, v in sorted(tfs.items()))
    print(f"  {str(mo):<9} {len(grp):>3} {w:>3} {l:>3}  {wr:>5.0f}%  ${pnl:>+9.2f}  {tf_str}")

print(DIV)
print()

# ── 4. Baseline comparison ───────────────────────────────────────────────────
print("  COMPARISON vs BASELINE (1H zones OFF, same other settings)")
print(DIV2)
print(f"  {'Metric':<22} {'Baseline (H4 only)':>20}  {'1H zones ON':>14}")
print(DIV2)
rows = [
    ("Trades",    "10",                    str(total)),
    ("Win rate",  "50.0%",                 f"{wr_all:.1f}%"),
    ("Net PnL",   "+$15.48",               m["net_pnl"]),
    ("Max DD",    "-2.75%",                f"{m['max_drawdown_%']}%"),
    ("Avg win $", "$6.70",                 m["avg_win_$"]),
    ("Avg loss $","$-2.18",                m["avg_loss_$"]),
]
for label, base, new in rows:
    print(f"  {label:<22} {base:>20}  {new:>14}")
print(DIV2)
