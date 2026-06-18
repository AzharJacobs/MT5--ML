#!/usr/bin/env python3
"""
compare_arrival.py — Compare three gradual-filter modes over the 3-year USTEC backtest.

Modes
-----
  baseline         — all trades (arrival_type = retest | gradual)
  full_filter      — retest only (all gradual entries dropped)
  tightened        — retest + gradual with 2+ confirmations AND fresh zone

Tables printed
--------------
  1. Side-by-side: trade count, WR%, net PnL, avg PnL, max DD%
  2. Gradual breakdown (from baseline): 1-conf vs 2+conf, fresh vs tapped

Usage:
  python trading/strategies/zz/ustec/compare_arrival.py
  python trading/strategies/zz/ustec/compare_arrival.py --start 2022-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.ustec.strategy import MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS
from trading.strategies.zz.ustec.engine import run_backtest


def _mode_stats(metrics: dict, df_t) -> dict:
    import pandas as pd
    n   = len(df_t)
    wr  = (df_t["outcome"] == 1).mean() * 100 if n else 0.0
    net = df_t["pnl"].sum() if n else 0.0
    avg = df_t["pnl"].mean() if n else 0.0
    dd  = float(metrics.get("max_drawdown_%", "0").replace("%", "")) if isinstance(metrics.get("max_drawdown_%"), str) else float(metrics.get("max_drawdown_%", 0))
    return {"trades": n, "wr": wr, "net": net, "avg": avg, "dd": dd}


def run_comparison(start: str, end: str, cash: float, spread: float, fixed_lot: float) -> None:

    shared = dict(
        start=start, end=end, cash=cash,
        symbol="ustech", spread=spread, fixed_lot=fixed_lot,
        min_rr=MIN_RR, max_forward_bars=MAX_FORWARD_BARS,
        silent=True,
    )

    print(f"\nRunning 3-year backtest  {start} → {end}  (3 modes) …")

    print("  [1/3] baseline …", end="", flush=True)
    m_base, df_base = run_backtest(gradual_filter="all", **shared)
    print(" done")

    print("  [2/3] full_filter (retest only) …", end="", flush=True)
    m_full, df_full = run_backtest(gradual_filter="retest_only", **shared)
    print(" done")

    print("  [3/3] tightened (retest + gradual 2+conf & fresh) …", end="", flush=True)
    m_tight, df_tight = run_backtest(gradual_filter="tightened", **shared)
    print(" done\n")

    s_base  = _mode_stats(m_base,  df_base)
    s_full  = _mode_stats(m_full,  df_full)
    s_tight = _mode_stats(m_tight, df_tight)

    # ── Table 1: side-by-side comparison ─────────────────────────────────────
    W = 16
    sep = "─" * (W * 4 + 3)
    print(sep)
    print(f"  Mode Comparison  ({start} → {end})")
    print(sep)
    print(f"  {'Metric':<22} {'Baseline':>{W}} {'Full Filter':>{W}} {'Tightened':>{W}}")
    print(sep)

    rows = [
        ("Trades",    f"{s_base['trades']}",          f"{s_full['trades']}",          f"{s_tight['trades']}"),
        ("Win Rate",  f"{s_base['wr']:.1f}%",         f"{s_full['wr']:.1f}%",         f"{s_tight['wr']:.1f}%"),
        ("Net PnL",   f"${s_base['net']:+.2f}",       f"${s_full['net']:+.2f}",       f"${s_tight['net']:+.2f}"),
        ("Avg PnL",   f"${s_base['avg']:+.2f}",       f"${s_full['avg']:+.2f}",       f"${s_tight['avg']:+.2f}"),
        ("Max DD%",   f"{s_base['dd']:.2f}%",         f"{s_full['dd']:.2f}%",         f"{s_tight['dd']:.2f}%"),
    ]
    for label, v1, v2, v3 in rows:
        print(f"  {label:<22} {v1:>{W}} {v2:>{W}} {v3:>{W}}")
    print(sep)
    print()

    # ── Table 2: gradual breakdown (from baseline df) ─────────────────────────
    g_all = df_base[df_base["arrival_type"] == "gradual"].copy()

    if g_all.empty:
        print("  No gradual trades in baseline — breakdown not applicable.")
        return

    def _grp_stats(sub):
        if sub.empty:
            return 0, 0.0, 0.0
        wr  = (sub["outcome"] == 1).mean() * 100
        net = sub["pnl"].sum()
        return len(sub), wr, net

    conf_1   = g_all[g_all["confirmations"] < 2]
    conf_2p  = g_all[g_all["confirmations"] >= 2]
    fresh    = g_all[g_all["zone_fresh"] == True]
    tapped   = g_all[g_all["zone_fresh"] == False]

    # cross-cut: conf × freshness (4 cells)
    c1_fresh  = g_all[(g_all["confirmations"] < 2)  & (g_all["zone_fresh"] == True)]
    c1_tapped = g_all[(g_all["confirmations"] < 2)  & (g_all["zone_fresh"] == False)]
    c2_fresh  = g_all[(g_all["confirmations"] >= 2) & (g_all["zone_fresh"] == True)]
    c2_tapped = g_all[(g_all["confirmations"] >= 2) & (g_all["zone_fresh"] == False)]

    sep2 = "─" * 68
    print(sep2)
    print(f"  Gradual trades breakdown  (baseline, n={len(g_all)})")
    print(sep2)
    print(f"  {'Slice':<28} {'Count':>6} {'WR%':>7} {'Net PnL':>12} {'Avg PnL':>10}")
    print(sep2)

    slices = [
        ("1 confirmation",   *_grp_stats(conf_1)),
        ("2+ confirmations", *_grp_stats(conf_2p)),
        ("─── by freshness", None, None, None),
        ("fresh zone",       *_grp_stats(fresh)),
        ("tapped zone",      *_grp_stats(tapped)),
        ("─── cross-cut",    None, None, None),
        ("1-conf + fresh",   *_grp_stats(c1_fresh)),
        ("1-conf + tapped",  *_grp_stats(c1_tapped)),
        ("2+-conf + fresh",  *_grp_stats(c2_fresh)),
        ("2+-conf + tapped", *_grp_stats(c2_tapped)),
    ]
    for row in slices:
        label, cnt, wr, net = row
        if cnt is None:
            print(f"  {label}")
            continue
        avg = net / cnt if cnt > 0 else 0.0
        print(f"  {label:<28} {cnt:>6} {wr:>6.1f}% {net:>+12.2f} {avg:>+10.2f}")
    print(sep2)
    print()

    # ── Tightened: which graduals survive ─────────────────────────────────────
    surviving = g_all[(g_all["confirmations"] >= 2) & (g_all["zone_fresh"] == True)]
    dropped   = g_all[~((g_all["confirmations"] >= 2) & (g_all["zone_fresh"] == True))]
    sv_n, sv_wr, sv_net = _grp_stats(surviving)
    dr_n, dr_wr, dr_net = _grp_stats(dropped)
    print(f"  Tightened filter — of {len(g_all)} gradual trades:")
    print(f"    kept  (2+conf AND fresh): {sv_n:>3}  WR={sv_wr:.1f}%  net=${sv_net:+.2f}")
    print(f"    dropped                 : {dr_n:>3}  WR={dr_wr:.1f}%  net=${dr_net:+.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Arrival-mode comparison — USTEC ZZ")
    parser.add_argument("--start",     default="2022-01-01")
    parser.add_argument("--end",       default="2025-01-01")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--spread",    type=float, default=SPREAD_PTS)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    args = parser.parse_args()
    run_comparison(
        start=args.start, end=args.end,
        cash=args.cash, spread=args.spread, fixed_lot=args.fixed_lot,
    )


if __name__ == "__main__":
    main()
