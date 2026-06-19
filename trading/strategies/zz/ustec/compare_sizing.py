#!/usr/bin/env python3
"""
compare_sizing.py — Fixed-fractional risk % comparison over the 3-year USTEC backtest.

No entry filters. Baseline logic unchanged. Only position sizing varies.
Lot = (equity × risk_pct) ÷ (SL_distance × contract_size)

Usage:
  python trading/strategies/zz/ustec/compare_sizing.py
  python trading/strategies/zz/ustec/compare_sizing.py --start 2022-01-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trading.strategies.zz.ustec.strategy import MIN_RR, MIN_SL_PCT, SPREAD_PTS, MAX_FORWARD_BARS
from trading.strategies.zz.ustec.engine import run_backtest

RISK_LEVELS = [0.0025, 0.005, 0.01, 0.02]   # 0.25%, 0.5%, 1%, 2%


def run_sizing_comparison(start: str, end: str, cash: float, spread: float) -> None:

    shared = dict(
        start=start, end=end, cash=cash,
        symbol="ustech", spread=spread,
        fixed_lot=0,                      # force fractional sizing
        min_rr=MIN_RR, min_sl_pct=MIN_SL_PCT, max_forward_bars=MAX_FORWARD_BARS,
        gradual_filter="all",             # no entry filters
        silent=True,
    )

    print(f"\nFixed-fractional sizing comparison  {start} → {end}  cash=${cash:,.0f}")
    print(f"  Lot = (equity × risk%) ÷ (SL_pts × contract_size)\n")

    results = {}
    for rp in RISK_LEVELS:
        label = f"{rp*100:.2f}%".rstrip("0").rstrip(".")
        print(f"  running {label} …", end="", flush=True)
        m, df = run_backtest(risk_pct=rp, **shared)
        results[label] = (m, df)
        print(" done")
    print()

    # ── Build table ───────────────────────────────────────────────────────────
    labels = [f"{rp*100:.2f}%".rstrip("0").rstrip(".") for rp in RISK_LEVELS]
    W   = 14
    sep = "─" * (26 + W * len(labels) + len(labels))

    print(sep)
    print(f"  Sizing Comparison  ({start} → {end})")
    print(sep)
    hdr = f"  {'Metric':<24}" + "".join(f" {lbl:>{W}}" for lbl in labels)
    print(hdr)
    print(sep)

    for label in labels:
        m, df = results[label]
        n      = len(df)
        wr     = (df["outcome"] == 1).mean() * 100 if n else 0.0
        net    = df["pnl"].sum() if n else 0.0
        eq_fin = m.get("lowest_equity", cash)          # raw float now
        dd_str = str(m.get("max_drawdown_%", "0"))
        dd     = float(dd_str)
        final_eq_str = m.get("final_equity", f"${cash:,.2f}")
        # parse final equity from metrics (stored as formatted string)
        final_eq = float(str(final_eq_str).replace("$", "").replace(",", ""))
        low_eq   = float(eq_fin)
        ruined   = dd <= -99.9

        results[label] = results[label] + (
            {"trades": n, "wr": wr, "net": net, "final_eq": final_eq,
             "low_eq": low_eq, "dd": dd, "ruined": ruined},
        )

    metric_rows = [
        ("Trades",          lambda s: f"{s['trades']}"),
        ("Win Rate",        lambda s: f"{s['wr']:.1f}%"),
        ("Net PnL",         lambda s: f"${s['net']:+,.2f}"),
        ("Final Equity",    lambda s: f"${s['final_eq']:,.2f}"),
        ("Lowest Equity",   lambda s: f"${s['low_eq']:,.2f}"),
        ("Max DD%",         lambda s: f"{s['dd']:.2f}%" + (" ⚠ BLOWN" if s['ruined'] else "")),
    ]

    for row_label, fn in metric_rows:
        row = f"  {row_label:<24}" + "".join(f" {fn(results[lbl][2]):>{W}}" for lbl in labels)
        print(row)
    print(sep)
    print()

    blown = [lbl for lbl in labels if results[lbl][2]["ruined"]]
    if blown:
        print(f"  ⚠  Account blown (-100% DD) at risk levels: {', '.join(blown)}")
    else:
        print("  All risk levels survived without blowing the account.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-fractional sizing comparison — USTEC ZZ")
    parser.add_argument("--start",  default="2022-01-01")
    parser.add_argument("--end",    default="2025-01-01")
    parser.add_argument("--cash",   type=float, default=10_000.0)
    parser.add_argument("--spread", type=float, default=SPREAD_PTS)
    args = parser.parse_args()
    run_sizing_comparison(
        start=args.start, end=args.end,
        cash=args.cash, spread=args.spread,
    )


if __name__ == "__main__":
    main()
