#!/usr/bin/env python3
"""
compare_gold_ZZ.py — Side-by-side comparison: XAUUSD baseline vs Gold Z&Z v2.

Runs both engines silently for 2023, 2024, 2025, then prints a clean table
showing whether the three gold fixes turn a net -$118 into positive territory.

Usage:
    python scripts/compare_gold_ZZ.py
    python scripts/compare_gold_ZZ.py --min_zone_atr_frac 0.20 --sl_atr_buffer 0.15
"""

import sys
import os
import io
import contextlib
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from backtest.engine_zones import run_backtest
from strategy_v2.config_ZZ import GoldZZConfig
from strategy_v2.engine_ZZ import run_backtest_gold

YEARS = [
    ("2023-01-01", "2024-01-01", "2023"),
    ("2024-01-01", "2025-01-01", "2024"),
    ("2025-01-01", "2026-01-01", "2025"),
]

SEP  = "=" * 80
SEP2 = "-" * 80


def _payoff(df: pd.DataFrame) -> float:
    wins   = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] < 0]["pnl"]
    if losses.empty or wins.empty:
        return float("nan")
    return abs(wins.mean() / losses.mean())


def _row(label: str, df: pd.DataFrame, start_cash: float) -> dict:
    total   = len(df)
    tp_hits = int((df["outcome"] == 1).sum())
    sl_hits = int((df["outcome"] == -1).sum())
    exp     = int((df["outcome"] == 0).sum())
    wr      = tp_hits / max(total, 1) * 100
    net     = df["pnl"].sum()
    eq      = [start_cash] + list(df["equity"])
    eq_s    = pd.Series(eq)
    max_dd  = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100
    return dict(label=label, trades=total, tp=tp_hits, sl=sl_hits, exp=exp,
                wr=wr, payoff=_payoff(df), net=net, max_dd=max_dd)


def _print_table(rows: list, title: str) -> None:
    print(f"\n{title}")
    print(SEP2)
    hdr = (f"{'PERIOD':<16} {'TRADES':>7} {'TP':>5} {'SL':>5} {'EXP':>4} "
           f"{'WR%':>7} {'PAYOFF':>7} {'NET P&L':>11} {'MAX DD%':>9}")
    print(hdr)
    print(SEP2)
    for r in rows:
        print(f"{r['label']:<16} {r['trades']:>7} {r['tp']:>5} {r['sl']:>5} "
              f"{r['exp']:>4} {r['wr']:>6.1f}% {r['payoff']:>7.2f} "
              f"${r['net']:>10.2f} {r['max_dd']:>8.2f}%")
    print()


def run_comparison(args) -> None:
    cfg = GoldZZConfig(
        failed_zone_filter=not args.no_failed_zone_filter,
        active_signals=frozenset(args.active_signals),
        min_confirmations=args.min_confirmations,
        min_zone_atr_frac=args.min_zone_atr_frac,
        sl_atr_buffer=args.sl_atr_buffer,
        min_rr=args.min_rr,
        spread=args.spread,
    )

    print(f"\n{SEP}")
    print("  XAUUSD Z&Z  Baseline vs Gold v2  —  2023–2025")
    print(f"  Gold v2 config: failed_zone_filter={cfg.failed_zone_filter}  "
          f"active_signals={sorted(cfg.active_signals)}")
    print(f"                  min_zone_atr_frac={cfg.min_zone_atr_frac}  "
          f"sl_atr_buffer={cfg.sl_atr_buffer}")
    print(SEP)

    base_rows  = []
    v2_rows    = []
    base_dfs   = []
    v2_dfs     = []

    start_cash    = 10_000.0
    base_cash     = start_cash
    v2_cash       = start_cash

    for start, end, label in YEARS:
        print(f"\n  [{label}] Running baseline ...", end=" ", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            base_result = run_backtest(
                start=start, end=end, cash=base_cash, symbol="xauusd", min_rr=cfg.min_rr,
            )
        if isinstance(base_result, tuple):
            _, base_df = base_result
        else:
            base_df = pd.DataFrame()
        if not base_df.empty:
            r = _row(label + " baseline", base_df, base_cash)
            base_rows.append(r)
            base_dfs.append(base_df)
            base_cash = float(base_df["equity"].iloc[-1])
            print(f"done  (WR={r['wr']:.1f}%  net=${r['net']:+.0f})", flush=True)
        else:
            print("no trades", flush=True)

        print(f"  [{label}] Running gold v2  ...", end=" ", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            v2_metrics, v2_df = run_backtest_gold(
                cfg=cfg, start=start, end=end, cash=v2_cash,
            )
        if not v2_df.empty:
            r = _row(label + " gold_v2 ", v2_df, v2_cash)
            v2_rows.append(r)
            v2_dfs.append(v2_df)
            v2_cash = float(v2_df["equity"].iloc[-1])
            print(f"done  (WR={r['wr']:.1f}%  net=${r['net']:+.0f})", flush=True)
        else:
            print("no trades", flush=True)

    # 3-year combined rows
    if base_dfs:
        df_all_base = pd.concat(base_dfs, ignore_index=True)
        base_rows.append(_row("3YR baseline ", df_all_base, start_cash))
    if v2_dfs:
        df_all_v2 = pd.concat(v2_dfs, ignore_index=True)
        v2_rows.append(_row("3YR gold_v2  ", df_all_v2, start_cash))

    # ── Print comparison ──────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("  COMPARISON TABLE")
    print(SEP)

    # Interleave baseline and v2 rows per year for easy visual comparison
    all_rows = []
    for i in range(min(len(base_rows), len(v2_rows)) - 1):
        all_rows.append(base_rows[i])
        all_rows.append(v2_rows[i])

    # Then the 3yr combined at the end
    if base_rows:
        all_rows.append(base_rows[-1])
    if v2_rows:
        all_rows.append(v2_rows[-1])

    _print_table(all_rows, "  Per-year and 3yr combined:")

    # Delta summary
    if base_rows and v2_rows:
        b3 = base_rows[-1]
        v3 = v2_rows[-1]
        print(f"  Delta (gold_v2 minus baseline — 3yr combined):")
        print(f"    Trades  : {v3['trades'] - b3['trades']:+d}"
              f"  ({v3['trades']} vs {b3['trades']})")
        print(f"    WR      : {v3['wr'] - b3['wr']:+.1f}pp"
              f"  ({v3['wr']:.1f}% vs {b3['wr']:.1f}%)")
        print(f"    Payoff  : {v3['payoff'] - b3['payoff']:+.2f}x"
              f"  ({v3['payoff']:.2f}x vs {b3['payoff']:.2f}x)")
        print(f"    Net P&L : ${v3['net'] - b3['net']:+.2f}"
              f"  (${v3['net']:+.2f} vs ${b3['net']:+.2f})")
        print(f"    Max DD  : {v3['max_dd'] - b3['max_dd']:+.2f}pp"
              f"  ({v3['max_dd']:.2f}% vs {b3['max_dd']:.2f}%)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare XAUUSD baseline vs Gold Z&Z v2")
    parser.add_argument("--no_failed_zone_filter", action="store_true")
    parser.add_argument("--active_signals", nargs="+",
                        default=["engulfing", "rejection_wick", "choch"])
    parser.add_argument("--min_confirmations", type=int,  default=1)
    parser.add_argument("--min_zone_atr_frac", type=float, default=0.30)
    parser.add_argument("--sl_atr_buffer",     type=float, default=0.20)
    parser.add_argument("--min_rr",            type=float, default=1.5)
    parser.add_argument("--spread",            type=float, default=0.0)
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
