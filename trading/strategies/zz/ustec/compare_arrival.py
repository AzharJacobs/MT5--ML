#!/usr/bin/env python3
"""
compare_arrival.py — Compare gradual-filter modes over the 3-year USTEC backtest.

Modes
-----
  baseline         — all trades (arrival_type = retest | gradual)
  full_filter      — retest only (all gradual entries dropped)
  no_multi_conf    — retest + gradual with 1 conf only (drop gradual 2+conf)
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

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, MIN_SL_PCT, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS,
)
from trading.strategies.zz.ustec.engine import run_backtest


def _mode_stats(metrics: dict, df_t) -> dict:
    n   = len(df_t)
    wr  = (df_t["outcome"] == 1).mean() * 100 if n else 0.0
    net = df_t["pnl"].sum() if n else 0.0
    avg = df_t["pnl"].mean() if n else 0.0
    dd  = float(str(metrics.get("max_drawdown_%", "0")).replace("%", ""))
    return {"trades": n, "wr": wr, "net": net, "avg": avg, "dd": dd}


def run_comparison(start: str, end: str, cash: float, spread: float, fixed_lot: float) -> None:

    shared = dict(
        start=start, end=end, cash=cash,
        symbol="ustech", spread=spread, fixed_lot=fixed_lot,
        min_rr=MIN_RR, min_sl_pct=MIN_SL_PCT, max_forward_bars=MAX_FORWARD_BARS,
        silent=True,
    )

    modes = [
        ("baseline",      "all"),
        ("full_filter",   "retest_only"),
        ("no_multi_conf", "no_multi_conf"),
        ("tightened",     "tightened"),
    ]

    print(f"\nRunning backtest  {start} → {end}  ({len(modes)} modes)  min_sl_pct={MIN_SL_PCT} …")
    results = {}
    for idx, (label, gf) in enumerate(modes, 1):
        print(f"  [{idx}/{len(modes)}] {label} …", end="", flush=True)
        m, df = run_backtest(gradual_filter=gf, **shared)
        results[label] = (m, df)
        print(" done")
    print()

    stats = {label: _mode_stats(m, df) for label, (m, df) in results.items()}

    # ── Table 1: side-by-side comparison ─────────────────────────────────────
    cols   = [label for label, _ in modes]
    W      = 15
    sep    = "─" * (22 + W * len(cols) + len(cols))
    header = f"  {'Metric':<22}" + "".join(f" {c:>{W}}" for c in cols)

    print(sep)
    print(f"  Mode Comparison  ({start} → {end})  min_sl_pct={MIN_SL_PCT}")
    print(sep)
    print(header)
    print(sep)

    metric_rows = [
        ("Trades",   lambda s: f"{s['trades']}"),
        ("Win Rate", lambda s: f"{s['wr']:.1f}%"),
        ("Net PnL",  lambda s: f"${s['net']:+.2f}"),
        ("Avg PnL",  lambda s: f"${s['avg']:+.2f}"),
        ("Max DD%",  lambda s: f"{s['dd']:.2f}%"),
    ]
    for row_label, fn in metric_rows:
        line = f"  {row_label:<22}" + "".join(f" {fn(stats[c]):>{W}}" for c in cols)
        print(line)
    print(sep)
    print()

    # ── Table 2: gradual breakdown from baseline ──────────────────────────────
    df_base = results["baseline"][1]
    g_all   = df_base[df_base["arrival_type"] == "gradual"].copy()

    if g_all.empty:
        print("  No gradual trades in baseline — breakdown not applicable.")
        return

    def _grp(sub):
        if sub.empty:
            return 0, 0.0, 0.0
        return len(sub), (sub["outcome"] == 1).mean() * 100, sub["pnl"].sum()

    conf_1    = g_all[g_all["confirmations"] < 2]
    conf_2p   = g_all[g_all["confirmations"] >= 2]
    fresh     = g_all[g_all["zone_fresh"] == True]
    tapped    = g_all[g_all["zone_fresh"] == False]
    c1_fresh  = g_all[(g_all["confirmations"] < 2)  & (g_all["zone_fresh"] == True)]
    c1_tapped = g_all[(g_all["confirmations"] < 2)  & (g_all["zone_fresh"] == False)]
    c2_fresh  = g_all[(g_all["confirmations"] >= 2) & (g_all["zone_fresh"] == True)]
    c2_tapped = g_all[(g_all["confirmations"] >= 2) & (g_all["zone_fresh"] == False)]

    sep2 = "─" * 68
    print(sep2)
    print(f"  Gradual breakdown  (baseline, n={len(g_all)})")
    print(sep2)
    print(f"  {'Slice':<28} {'Count':>6} {'WR%':>7} {'Net PnL':>12} {'Avg PnL':>10}")
    print(sep2)

    slices = [
        ("1 confirmation",   *_grp(conf_1)),
        ("2+ confirmations", *_grp(conf_2p)),
        ("─── by freshness", None, None, None),
        ("fresh zone",       *_grp(fresh)),
        ("tapped zone",      *_grp(tapped)),
        ("─── cross-cut",    None, None, None),
        ("1-conf + fresh",   *_grp(c1_fresh)),
        ("1-conf + tapped",  *_grp(c1_tapped)),
        ("2+-conf + fresh",  *_grp(c2_fresh)),
        ("2+-conf + tapped", *_grp(c2_tapped)),
    ]
    for row in slices:
        label, cnt, wr, net = row
        if cnt is None:
            print(f"  {label}")
            continue
        avg = net / cnt if cnt > 0 else 0.0
        print(f"  {label:<28} {cnt:>6} {wr:>6.1f}% {net:>+12.2f} {avg:>+10.2f}")
    print(sep2)

    # ── no_multi_conf filter summary ──────────────────────────────────────────
    kept_nmc    = g_all[g_all["confirmations"] < 2]
    dropped_nmc = g_all[g_all["confirmations"] >= 2]
    kn, kwr, knet = _grp(kept_nmc)
    dn, dwr, dnet = _grp(dropped_nmc)
    print()
    print(f"  no_multi_conf filter — of {len(g_all)} gradual trades:")
    print(f"    kept  (1 conf)  : {kn:>3}  WR={kwr:.1f}%  net=${knet:+.2f}  avg=${knet/kn:+.2f}" if kn else f"    kept  (1 conf)  :   0")
    print(f"    dropped (2+conf): {dn:>3}  WR={dwr:.1f}%  net=${dnet:+.2f}  avg=${dnet/dn:+.2f}" if dn else f"    dropped (2+conf):   0")
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
