"""
Deep-dive on losing long (buy) trades.
Runs the backtest silently then breaks down every tracked dimension
to find what losing longs have in common.

Usage:
    python trading/strategies/zz/ustec/analyse_losing_longs.py \
        --start 2023-01-01 --end 2025-12-31 --fixed_lot 0.01 --cash 150
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT,
)
from trading.strategies.zz.ustec.engine import run_backtest

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def pct(n, d):
    return f"{n/d*100:.1f}%" if d else "n/a"


def breakdown(title, series, df_all, df_loss):
    print(f"\n  [{title}]")
    cats = series.value_counts().index.tolist()
    print(f"  {'Value':<22} {'All buys':>9} {'Losers':>9} {'Loss rate':>10} {'Avg loss $':>11}")
    print("  " + "-" * 65)
    for c in cats:
        n_all  = (series == c).sum()
        sub    = df_loss[series[df_loss.index] == c] if len(df_loss) else df_loss
        n_loss = len(sub)
        avg_l  = sub["pnl"].mean() if n_loss else 0
        print(f"  {str(c):<22} {n_all:>9} {n_loss:>9} {pct(n_loss, n_all):>10} {avg_l:>+11.2f}")


def signal_breakdown(df_all, df_loss):
    """Expand pipe-delimited signal column and count per signal."""
    print(f"\n  [Signal presence — losing longs vs all longs]")
    all_sigs = {}
    loss_sigs = {}
    for _, row in df_all.iterrows():
        for s in str(row["signals"]).split("|"):
            if s:
                all_sigs[s] = all_sigs.get(s, 0) + 1
    for _, row in df_loss.iterrows():
        for s in str(row["signals"]).split("|"):
            if s:
                loss_sigs[s] = loss_sigs.get(s, 0) + 1

    all_signals = sorted(all_sigs.keys())
    print(f"  {'Signal':<22} {'In all buys':>12} {'In losers':>10} {'Loss rate':>10}")
    print("  " + "-" * 58)
    for s in all_signals:
        n_all  = all_sigs.get(s, 0)
        n_loss = loss_sigs.get(s, 0)
        print(f"  {s:<22} {n_all:>12} {n_loss:>10} {pct(n_loss, n_all):>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    args = parser.parse_args()

    print(f"\nRunning backtest {args.start} to {args.end} ...")
    result = run_backtest(
        start=args.start,
        end=args.end,
        cash=args.cash,
        min_rr=MIN_RR,
        max_forward_bars=MAX_FORWARD_BARS,
        symbol="ustech",
        spread=SPREAD_PTS,
        fixed_lot=args.fixed_lot,
        directional_filter=True,
        allow_neutral=True,
        h4_swing_left=2,
        h4_swing_right=2,
        min_confirmations=1,
        excluded_from_count=list(EXCLUDED_FROM_COUNT),
        zone_max_losses=ZONE_MAX_LOSSES,
        h4_regime_filter=H4_REGIME_FILTER,
        min_sl_pct=MIN_SL_PCT,
        enable_trailing=ENABLE_TRAILING,
        be_trigger_pts=BE_TRIGGER_PTS,
        be_buffer_pts=BE_BUFFER_PTS,
        atr_trail_mult=ATR_TRAIL_MULT,
        silent=True,
    )

    if not result or isinstance(result, dict):
        print("ERROR: no trades returned.")
        return

    metrics, df = result

    df_buys = df[df["side"] == "buy"].copy()
    df_loss = df_buys[df_buys["pnl"] < 0].copy()
    df_win  = df_buys[df_buys["pnl"] > 0].copy()

    n_buys = len(df_buys)
    n_loss = len(df_loss)
    n_win  = len(df_win)
    total_loss_pnl = df_loss["pnl"].sum()
    avg_loss       = df_loss["pnl"].mean() if n_loss else 0

    print()
    print("=" * 70)
    print("  LOSING LONG TRADE ANALYSIS")
    print(f"  Period : {args.start} to {args.end}")
    print("=" * 70)
    print(f"\n  Total buy trades  : {n_buys}")
    print(f"  Winning buys      : {n_win}  ({pct(n_win, n_buys)})")
    print(f"  Losing buys       : {n_loss}  ({pct(n_loss, n_buys)})")
    print(f"  Total loss bleed  : ${total_loss_pnl:+.2f}")
    print(f"  Avg loss per trade: ${avg_loss:+.2f}")

    # ── 1. H4 bias at entry ───────────────────────────────────────────────────
    breakdown("H4 Bias at entry", df_buys["h4_bias"], df_buys, df_loss)

    # ── 2. Arrival type ───────────────────────────────────────────────────────
    breakdown("Arrival type (gradual=first touch, retest=bounce back)",
              df_buys["arrival_type"], df_buys, df_loss)

    # ── 3. Zone freshness ─────────────────────────────────────────────────────
    breakdown("Zone freshness at entry", df_buys["zone_fresh"], df_buys, df_loss)

    # ── 4. Prior bucket (zone history) ───────────────────────────────────────
    breakdown("Prior outcome bucket", df_buys["prior_bucket"], df_buys, df_loss)

    # ── 5. Confirmation count ─────────────────────────────────────────────────
    breakdown("Confirmation count", df_buys["confirmations"], df_buys, df_loss)

    # ── 6. Signals present ───────────────────────────────────────────────────
    signal_breakdown(df_buys, df_loss)

    # ── 7. Entry mode ────────────────────────────────────────────────────────
    breakdown("Entry mode", df_buys["entry_mode"], df_buys, df_loss)

    # ── 8. Zone strength bucket ───────────────────────────────────────────────
    df_buys["str_bucket"] = pd.cut(df_buys["zone_strength"],
                                   bins=[0, 2, 3, 4, 999],
                                   labels=["1.5-2", "2-3", "3-4", "4+"])
    df_loss["str_bucket"] = pd.cut(df_loss["zone_strength"],
                                   bins=[0, 2, 3, 4, 999],
                                   labels=["1.5-2", "2-3", "3-4", "4+"])
    breakdown("Zone strength bucket", df_buys["str_bucket"], df_buys, df_loss)

    # ── 9. Hour of entry ─────────────────────────────────────────────────────
    df_buys["hour"] = pd.to_datetime(df_buys["date"]).dt.hour
    df_loss["hour"] = pd.to_datetime(df_loss["date"]).dt.hour

    print(f"\n  [Hour of entry (UTC) — all buys vs losing buys]")
    print(f"  {'Hour':>6} {'All buys':>9} {'Losers':>9} {'Loss rate':>10} {'Avg loss $':>11}")
    print("  " + "-" * 50)
    for h in sorted(df_buys["hour"].unique()):
        n_all  = (df_buys["hour"] == h).sum()
        sub    = df_loss[df_loss["hour"] == h]
        n_l    = len(sub)
        avg_l  = sub["pnl"].mean() if n_l else 0
        flag   = " <--" if n_l >= 3 and n_l / n_all > 0.6 else ""
        print(f"  {h:>6}   {n_all:>9} {n_l:>9} {pct(n_l, n_all):>10} {avg_l:>+11.2f}{flag}")

    # ── 10. Month-by-month losing buy PnL ────────────────────────────────────
    df_loss["exit_dt"] = pd.to_datetime(df_loss["exit_date"])
    df_loss["year"]    = df_loss["exit_dt"].dt.year
    df_loss["month"]   = df_loss["exit_dt"].dt.month

    print(f"\n  [Monthly losing-long bleed]")
    print(f"  {'Month':<12} {'N losers':>9} {'Total bleed':>12} {'Avg loss':>10}")
    print("  " + "-" * 47)
    for (y, m), g in df_loss.groupby(["year", "month"]):
        lbl = f"{MONTH_NAMES[m]} {y}"
        print(f"  {lbl:<12} {len(g):>9} {g['pnl'].sum():>+12.2f} {g['pnl'].mean():>+10.2f}")

    # ── 11. Max adverse excursion vs outcome ─────────────────────────────────
    print(f"\n  [Max Adverse Excursion — losers vs winners]")
    print(f"  {'Group':<14} {'Avg MAE':>10} {'Med MAE':>10} {'Avg MFE':>10}")
    print("  " + "-" * 48)
    for lbl, grp in [("Losers", df_loss), ("Winners", df_win)]:
        if len(grp):
            print(f"  {lbl:<14} {grp['max_adverse'].mean():>10.1f} "
                  f"{grp['max_adverse'].median():>10.1f} "
                  f"{grp['max_favour'].mean():>10.1f}")

    # ── 12. Print the actual losing long trades ───────────────────────────────
    print(f"\n  [All {n_loss} losing long trades]")
    cols = ["date", "h4_bias", "arrival_type", "zone_fresh", "confirmations",
            "signals", "prior_bucket", "zone_strength", "entry", "sl", "tp",
            "exit", "pnl", "max_favour", "max_adverse"]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(df_loss[cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
