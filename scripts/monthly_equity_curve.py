#!/usr/bin/env python3
"""Monthly equity curve — zone_guard=True baseline, $100 start, 0.05 lot."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest

result = run_backtest(
    start="2023-01-01",
    end="2026-01-01",
    symbol="ustech",
    fixed_lot=0.05,
    cash=100.0,
    directional_filter=True,
    allow_neutral=True,
    min_confirmations=1,
    min_sl_pct=0.25,
    zone_guard=True,
)

if not result or len(result) < 2:
    print("No trades returned.")
    sys.exit(1)

_, df = result
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M")

# Recompute equity from $100 start + cumulative pnl
df = df.sort_values("date").reset_index(drop=True)
df["equity_running"] = 100.0 + df["pnl"].cumsum()

W = 12   # column widths

HDR = (f"{'Month':<8}  {'Start Bal':>10}  {'Trades':>6}  "
       f"{'Gross Win':>10}  {'Gross Loss':>11}  {'Net P&L':>10}  "
       f"{'End Bal':>10}  {'DD%':>7}  {'Note':<6}")
SEP = "-" * len(HDR)

print()
print(HDR)
print(SEP)

balance = 100.0
worst_month_pnl = None
worst_month_str = ""
dd_streak = []          # (month_str, cumulative_dd_from_peak)
running_peak = 100.0
max_dd_stretch = []
cur_stretch = []

months = sorted(df["month"].unique())
for m in months:
    grp = df[df["month"] == m].sort_values("date")

    start_bal = balance
    trades = len(grp)

    gross_win  = grp.loc[grp["pnl"] > 0, "pnl"].sum()
    gross_loss = grp.loc[grp["pnl"] < 0, "pnl"].sum()
    net_pnl    = grp["pnl"].sum()
    end_bal    = start_bal + net_pnl

    # Intra-month drawdown (on running equity within month)
    eq = start_bal + grp["pnl"].cumsum()
    peak_m = eq.cummax()
    dd_pct = ((eq - peak_m) / peak_m * 100).min() if len(eq) > 1 else 0.0

    # Track worst losing month
    if worst_month_pnl is None or net_pnl < worst_month_pnl:
        worst_month_pnl = net_pnl
        worst_month_str = str(m)

    # Track deepest dd stretch
    running_peak = max(running_peak, end_bal)
    if net_pnl < 0:
        cur_stretch.append((str(m), net_pnl, end_bal, running_peak))
    else:
        if cur_stretch:
            if not max_dd_stretch or len(cur_stretch) > len(max_dd_stretch):
                max_dd_stretch = cur_stretch[:]
            cur_stretch = []

    note = ""
    if str(m) == worst_month_str and trades > 0:
        note = "WORST"

    dd_str = f"{dd_pct:.1f}%" if net_pnl < 0 and trades > 0 else ""

    print(f"{str(m):<8}  {start_bal:>10.2f}  {trades:>6}  "
          f"{gross_win:>10.2f}  {gross_loss:>11.2f}  {net_pnl:>+10.2f}  "
          f"{end_bal:>10.2f}  {dd_str:>7}  {note:<6}")

    balance = end_bal

# flush last stretch
if cur_stretch:
    if not max_dd_stretch or len(cur_stretch) > len(max_dd_stretch):
        max_dd_stretch = cur_stretch[:]

print(SEP)
print(f"\nFinal balance : ${balance:.2f}   (started $100.00,  net ${balance-100:.2f})")
print(f"Total trades  : {len(df)}")
print(f"\nWorst month   : {worst_month_str}  (${worst_month_pnl:+.2f})")
if max_dd_stretch:
    s_month = max_dd_stretch[0][0]
    e_month = max_dd_stretch[-1][0]
    stretch_pnl = sum(r[1] for r in max_dd_stretch)
    low_bal = min(r[2] for r in max_dd_stretch)
    peak_b  = max_dd_stretch[0][3]
    dd_depth = (low_bal - peak_b) / peak_b * 100
    print(f"Deepest DD stretch: {s_month} → {e_month}  "
          f"({len(max_dd_stretch)} consecutive losing months, "
          f"stretch P&L ${stretch_pnl:+.2f}, trough DD {dd_depth:.1f}%)")
