#!/usr/bin/env python3
"""
Test: bos_msb_min1=True — single bos_msb entry allowed, all others need conf>=2.
Runs 2023-2024 (in-sample) + 2021-2022 (OOS).
Config: neutral=OFF, RR<=5.0, H4 only, H4_WINDOW=150, LH+HL fix.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

COMMON = dict(
    cash=150.0,
    symbol="ustech",
    min_confirmations=2,
    data_source="mt5",
    max_rr=5.0,
    allow_neutral=False,
    bos_msb_min1=True,
    silent=True,
)

print("Running 2023-2024 (in-sample) …")
res_is  = run_backtest(start="2023-01-01", end="2025-01-01", **COMMON)

print("Running 2021-2022 (OOS) …")
res_oos = run_backtest(start="2021-01-01", end="2023-01-01", **COMMON)

DIV  = "─" * 96
DIV2 = "─" * 58


def report(label, result, period):
    if not result:
        print(f"  {label}: no data")
        return
    m, df = result
    df = df.copy()
    df["rr"] = abs(df["tp"] - df["entry"]) / abs(df["entry"] - df["sl"])
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    # ── Per-trade list ────────────────────────────────────────────────────────
    print()
    print(DIV)
    print(f"  {label}  |  bos_msb_min1=True  |  conf=2 (or 1×bos_msb)")
    print(DIV)
    print(f"  {'Date':<22} {'Side':<5} {'Bias':<16} {'N':>2} {'Signals':<42} {'RR':>5}  Out")
    print(DIV)
    for _, r in df.sort_values("date").iterrows():
        sigs = r["signals"].replace("|", " + ")
        out  = "WIN " if r["outcome"] == 1 else "LOSS"
        n    = r["confirmations"]
        flag = " ← 1×bos" if n == 1 else ""
        print(f"  {str(r['date']):<22} {r['side']:<5} {r['h4_bias']:<16} {n:>2} {sigs:<42} {r['rr']:>5.2f}  {out}{flag}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total = len(df)
    wr    = (df["outcome"] == 1).mean() * 100
    print()
    print(f"  Trades   : {total}  ({m['tp_hits']}W / {m['sl_hits']}L / {m['expired']}E)")
    print(f"  Win rate : {wr:.1f}%")
    print(f"  Net PnL  : {m['net_pnl']}  ({m['start_cash']} → {m['final_equity']})")
    print(f"  Max DD   : {m['max_drawdown_%']}%")

    # Split: 1-conf bos_msb vs conf>=2
    single = df[df["confirmations"] == 1]
    multi  = df[df["confirmations"] >= 2]
    print()
    print(f"  1-conf bos_msb entries : {len(single):>3} trades  "
          f"{int((single['outcome']==1).sum())}W/{int((single['outcome']==-1).sum())}L  "
          f"WR={((single['outcome']==1).mean()*100) if len(single) else 0:.0f}%")
    print(f"  conf>=2 entries        : {len(multi):>3} trades  "
          f"{int((multi['outcome']==1).sum())}W/{int((multi['outcome']==-1).sum())}L  "
          f"WR={((multi['outcome']==1).mean()*100) if len(multi) else 0:.0f}%")

    # H4 bias breakdown
    print()
    print(f"  H4 bias breakdown:")
    for bias, grp in df.groupby("h4_bias"):
        wr_b = (grp["outcome"] == 1).mean() * 100
        print(f"    {bias:<16} {len(grp):>3} trades  WR={wr_b:.0f}%")

    # Monthly summary
    start_p, end_p = period
    all_months = pd.period_range(start_p, end_p, freq="M")
    print()
    print(f"  Monthly summary:")
    print(f"  {'Month':<9} {'Tr':>3} {'W':>3} {'L':>3}  {'WR%':>6}  {'Net PnL':>10}  1×bos")
    print("  " + "─" * 50)
    for mo in all_months:
        grp = df[df["month"] == mo]
        if grp.empty:
            print(f"  {str(mo):<9}   0   -   -     N/A           -")
            continue
        w   = int((grp["outcome"] == 1).sum())
        l   = int((grp["outcome"] == -1).sum())
        pnl = grp["pnl"].sum()
        n1  = int((grp["confirmations"] == 1).sum())
        wr_m = w / len(grp) * 100
        flag = f"  {n1}×" if n1 else ""
        print(f"  {str(mo):<9} {len(grp):>3} {w:>3} {l:>3}  {wr_m:>5.0f}%  ${pnl:>+9.2f}{flag}")


report("IN-SAMPLE 2023-2024", res_is,  ("2023-01", "2024-12"))
report("OOS 2021-2022",       res_oos, ("2021-01", "2022-12"))

# ── Baseline comparison (hardcoded from prior run) ────────────────────────────
print()
print(DIV2)
print("  COMPARISON vs BASELINE (conf=2 strict, same other settings)")
print(DIV2)
print(f"  {'Metric':<22} {'Baseline conf=2':>18}  {'bos_msb_min1':>14}")
print(DIV2)

if res_is:
    m_is, df_is = res_is
    wr_is = (df_is["outcome"] == 1).mean() * 100
    rows_is = [
        ("IS trades",    "10",      str(len(df_is))),
        ("IS win rate",  "50.0%",   f"{wr_is:.1f}%"),
        ("IS net PnL",   "+$15.48", m_is["net_pnl"]),
        ("IS max DD",    "-2.75%",  f"{m_is['max_drawdown_%']}%"),
    ]
    for label, base, new in rows_is:
        print(f"  {label:<22} {base:>18}  {new:>14}")

print(DIV2)
if res_oos:
    m_oos, df_oos = res_oos
    wr_oos = (df_oos["outcome"] == 1).mean() * 100
    rows_oos = [
        ("OOS trades",   "5",       str(len(df_oos))),
        ("OOS win rate", "20.0%",   f"{wr_oos:.1f}%"),
        ("OOS net PnL",  "+$0.92",  m_oos["net_pnl"]),
        ("OOS max DD",   "-3.80%",  f"{m_oos['max_drawdown_%']}%"),
    ]
    for label, base, new in rows_oos:
        print(f"  {label:<22} {base:>18}  {new:>14}")
print(DIV2)
