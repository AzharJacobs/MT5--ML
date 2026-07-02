#!/usr/bin/env python3
"""Out-of-sample validation: 2021-2022 (period the settings were NOT built on)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.zz.ustec.engine import run_backtest

result = run_backtest(
    start="2021-01-01",
    end="2023-01-01",
    cash=150.0,
    symbol="ustech",
    min_confirmations=2,
    data_source="mt5",
    max_rr=5.0,
    silent=True,
)

if not result:
    print("No data returned — check MT5 connection and date range.")
    sys.exit(1)

m, df = result

DIV = "─" * 65
print()
print("  OUT-OF-SAMPLE  |  2021-01-01 to 2023-01-01")
print("  conf=2  |  RR cap 5.0  |  fresh zones  |  MT5 data")
print(DIV)
print(f"  Trades          : {m['total_trades']}")
print(f"  Wins/Loss/Exp   : {m['tp_hits']}W / {m['sl_hits']}L / {m['expired']}E")
print(f"  Win rate        : {m['win_rate_%']}%")
avg_rr = (abs(df["tp"] - df["entry"]) / abs(df["entry"] - df["sl"])).mean()
print(f"  Avg RR (theory) : {avg_rr:.2f}")
print(f"  Net PnL         : {m['net_pnl']}")
print(f"  Start equity    : {m['start_cash']}")
print(f"  Final equity    : {m['final_equity']}")
print(f"  Max drawdown    : {m['max_drawdown_%']}%")
print(f"  Avg win $       : {m['avg_win_$']}")
print(f"  Avg loss $      : {m['avg_loss_$']}")
print(f"  Buys  (W)       : {m['buy_trades']} ({m['buy_wins']}W)")
print(f"  Sells (W)       : {m['sell_trades']} ({m['sell_wins']}W)")

print()
print("  H4 bias breakdown:")
for bias, grp in df.groupby("h4_bias"):
    wr = (grp["outcome"] == 1).mean() * 100
    print(f"    {bias:<16} {len(grp):>3} trades  WR={wr:.0f}%")

print()
print("  Zone freshness:")
for fv, grp in df.groupby("zone_fresh"):
    wr = (grp["outcome"] == 1).mean() * 100
    lbl = "fresh" if fv else "tapped"
    print(f"    {lbl:<8} {len(grp):>3} trades  WR={wr:.0f}%")

print()
print("  Signal pair win rates:")
combos = {}
for _, r in df.iterrows():
    k = tuple(sorted(r["signals"].split("|")))
    if k not in combos:
        combos[k] = {"w": 0, "l": 0}
    if r["outcome"] == 1:
        combos[k]["w"] += 1
    else:
        combos[k]["l"] += 1
for k, v in sorted(combos.items(), key=lambda x: -(x[1]["w"] + x[1]["l"])):
    n = v["w"] + v["l"]
    wr = v["w"] / n * 100
    print(f"    {' + '.join(k):<44} {v['w']}W/{v['l']}L  WR={wr:.0f}%  n={n}")

print()
print("  Per-trade list:")
print(f"  {'Date':<22} {'Side':<5} {'Signals':<42} {'RR':>5}  Out")
print("  " + "─" * 78)
for _, r in df.sort_values("date").iterrows():
    sigs = r["signals"].replace("|", " + ")
    out  = "WIN " if r["outcome"] == 1 else "LOSS"
    rr   = abs(r["tp"] - r["entry"]) / abs(r["entry"] - r["sl"])
    print(f"  {str(r['date']):<22} {r['side']:<5} {sigs:<42} {rr:>5.2f}  {out}")

print()
