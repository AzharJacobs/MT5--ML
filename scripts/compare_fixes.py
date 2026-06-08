#!/usr/bin/env python3
"""
compare_fixes.py — Compare baseline vs three targeted fixes across 2023/2024/2025.

Fixes tested:
  Fix1  zone_max_losses=2  : blacklist zone after 2 consecutive losses
  Fix2  excl rejection_wick: rejection_wick fires+logs but doesn't gate entries alone
  Fix3  min_rr=2.0         : require tighter RR filter

Run:
    python scripts/compare_fixes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine_zones import run_backtest

CONFIGS = [
    ("BASELINE",         {}),
    ("Fix1: zone_bl=2",  {"zone_max_losses": 2}),
    ("Fix2: excl_wick",  {"excluded_from_count": ["rejection_wick"]}),
    ("Fix3: min_rr=2",   {"min_rr": 2.0}),
    ("ALL 3 FIXES",      {"zone_max_losses": 2, "excluded_from_count": ["rejection_wick"], "min_rr": 2.0}),
]

BASE = {"fixed_lot": 0.05, "symbol": "ustech"}

def run(yr_start, yr_end, extra):
    m, _ = run_backtest(start=yr_start, end=yr_end, **BASE, **extra)
    net = float(m["net_pnl"].replace("$", "").replace(",", ""))
    wr  = float(m["win_rate_%"])
    tr  = int(m["total_trades"])
    dd  = float(m["max_drawdown_%"])
    bl  = m.get("zones_blacklisted", 0)
    return net, wr, tr, dd, bl

HDR = (
    f"{'Config':<22} |"
    f" {'2023':>9} {'WR':>6} {'Tr':>4} {'DD':>7} |"
    f" {'2024':>9} {'WR':>6} {'Tr':>4} {'DD':>7} |"
    f" {'2025':>9} {'WR':>6} {'Tr':>4} {'DD':>7} |"
    f" {'3-YR':>9} {'WR':>6} {'Tr':>4}"
)

print()
print(HDR)
print("-" * 130)

for label, extra in CONFIGS:
    r = {}
    for yr in [2023, 2024, 2025]:
        r[yr] = run(f"{yr}-01-01", f"{yr+1}-01-01", extra)
    r3 = run("2023-01-01", "2026-01-01", extra)

    bl_note = f" (bl={r3[4]})" if extra.get("zone_max_losses") else ""
    row = (
        f"{label:<22} |"
        f" {r[2023][0]:>+9.0f} {r[2023][1]:>5.1f}% {r[2023][2]:>4} {r[2023][3]:>6.1f}% |"
        f" {r[2024][0]:>+9.0f} {r[2024][1]:>5.1f}% {r[2024][2]:>4} {r[2024][3]:>6.1f}% |"
        f" {r[2025][0]:>+9.0f} {r[2025][1]:>5.1f}% {r[2025][2]:>4} {r[2025][3]:>6.1f}% |"
        f" {r3[0]:>+9.0f} {r3[1]:>5.1f}% {r3[2]:>4}{bl_note}"
    )
    print(row)

print()
