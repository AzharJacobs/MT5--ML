#!/usr/bin/env python3
"""
Straight-to-SL loss analysis.

Segments SL hits into:
  - Straight-to-SL  : MFE < 5% of TP range (price went directly to SL)
  - Moved losers    : MFE >= 5% (price moved your way, then reversed)

For each dimension, shows how often each bucket produces straight-to-SL
so you can identify filterable traits.
"""
import io
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

DASH = "─" * 68


def _run_silent():
    """Run backtest suppressing engine stdout."""
    from trading.strategies.zz.ustec.engine import run_backtest
    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = run_backtest(
            start="2023-01-01",
            end="2025-12-31",
            cash=150.0,
            symbol="ustech",
            fixed_lot=0.01,
        )
    finally:
        sys.stdout = _real_stdout
    return result


def _pct_row(label, s, m, width=20):
    tot = s + m
    pct = s / tot * 100 if tot else 0
    flag = "  ◄" if pct > 35 else ""
    print(f"  {label:<{width}} {s:>8} {m:>8} {tot:>7}  {pct:>6.1f}%{flag}")


def main():
    print("\nLoading backtest trades (engine output suppressed)...")
    result = _run_silent()
    if not isinstance(result, tuple) or len(result) < 2:
        print("Backtest failed or returned no trades.")
        return

    _, df = result
    df = df.copy()
    df["tp_range"] = abs(df["tp"] - df["entry"])
    df["mfe_pct"]  = df["max_favour"] / df["tp_range"].replace(0, float("nan")) * 100

    losers   = df[df["outcome"] == -1].copy()
    straight = losers[losers["mfe_pct"] < 5.0].copy()
    moved    = losers[losers["mfe_pct"] >= 5.0].copy()

    n_sl  = len(losers)
    n_str = len(straight)
    n_mov = len(moved)

    print(f"\n{'='*68}")
    print(f"  STRAIGHT-TO-SL ANALYSIS  |  2023-2025")
    print(f"  SL hits total : {n_sl}")
    print(f"  Straight (<5%): {n_str}  ({n_str/n_sl*100:.1f}%)")
    print(f"  Moved   (>=5%): {n_mov}  ({n_mov/n_sl*100:.1f}%)")
    print(f"  Total PnL straight: ${straight['pnl'].sum():+.2f}  "
          f"| avg/trade: ${straight['pnl'].mean():.2f}")
    print(f"  Total PnL moved:    ${moved['pnl'].sum():+.2f}  "
          f"| avg/trade: ${moved['pnl'].mean():.2f}")
    print(f"{'='*68}")

    hdr = f"  {'Bucket':<20} {'Str':>8} {'Mov':>8} {'Total':>7}  {'Str%':>6}"

    # ── 1. Zone kind ──────────────────────────────────────────────────────
    print(f"\n  1. ZONE KIND")
    print(hdr); print(f"  {DASH}")
    for k in sorted(losers["zone_kind"].dropna().unique()):
        s = (straight["zone_kind"] == k).sum()
        m = (moved["zone_kind"]   == k).sum()
        _pct_row(str(k), s, m)

    # ── 2. Zone strength buckets ──────────────────────────────────────────
    print(f"\n  2. ZONE STRENGTH  (avg straight={straight['zone_strength'].mean():.2f}  "
          f"avg moved={moved['zone_strength'].mean():.2f})")
    print(hdr); print(f"  {DASH}")
    bins = [(0, 1.5, "<1.5"), (1.5, 2.0, "1.5-2.0"), (2.0, 2.5, "2.0-2.5"),
            (2.5, 3.0, "2.5-3.0"), (3.0, 99, "3.0+")]
    for lo, hi, label in bins:
        s = ((straight["zone_strength"] >= lo) & (straight["zone_strength"] < hi)).sum()
        m = ((moved["zone_strength"]    >= lo) & (moved["zone_strength"]    < hi)).sum()
        if s + m == 0:
            continue
        _pct_row(label, s, m)

    # ── 3. Fresh vs retest ────────────────────────────────────────────────
    print(f"\n  3. FRESH vs RETEST  (is_retest = zone previously won)")
    print(hdr); print(f"  {DASH}")
    for flag, label in [(False, "fresh"), (True, "retest")]:
        s = (straight["is_retest"] == flag).sum()
        m = (moved["is_retest"]    == flag).sum()
        _pct_row(label, s, m)

    # ── 4. Prior bucket ───────────────────────────────────────────────────
    print(f"\n  4. PRIOR OUTCOME BUCKET")
    print(hdr); print(f"  {DASH}")
    for pb in ["first_attempt", "post_win", "post_loss"]:
        s = (straight["prior_bucket"] == pb).sum()
        m = (moved["prior_bucket"]    == pb).sum()
        if s + m == 0:
            continue
        _pct_row(pb, s, m)

    # ── 5. Signal type ────────────────────────────────────────────────────
    print(f"\n  5. SIGNAL TYPE  (each signal counted once per trade it appears in)")
    print(hdr); print(f"  {DASH}")
    sig_s, sig_m = {}, {}
    for sig_str in straight["signals"]:
        for sig in str(sig_str).split("|"):
            sig = sig.strip()
            sig_s[sig] = sig_s.get(sig, 0) + 1
    for sig_str in moved["signals"]:
        for sig in str(sig_str).split("|"):
            sig = sig.strip()
            sig_m[sig] = sig_m.get(sig, 0) + 1
    all_sigs = sorted(set(list(sig_s) + list(sig_m)))
    for sig in all_sigs:
        s = sig_s.get(sig, 0)
        m = sig_m.get(sig, 0)
        if s + m < 5:
            continue
        _pct_row(sig, s, m)

    # ── 6. H4 bias ────────────────────────────────────────────────────────
    print(f"\n  6. H4 BIAS AT ENTRY")
    print(hdr); print(f"  {DASH}")
    for bias in sorted(losers["h4_bias"].unique()):
        s = (straight["h4_bias"] == bias).sum()
        m = (moved["h4_bias"]    == bias).sum()
        _pct_row(str(bias), s, m)

    # ── 7. Side ───────────────────────────────────────────────────────────
    print(f"\n  7. TRADE SIDE")
    print(hdr); print(f"  {DASH}")
    for side in ["buy", "sell"]:
        s = (straight["side"] == side).sum()
        m = (moved["side"]    == side).sum()
        _pct_row(side, s, m)

    # ── 8. MFE distribution for straight-to-SL trades ───────────────────
    print(f"\n  8. MFE FINE-GRAIN  (straight-to-SL bucket only)")
    print(f"  {'Range':<20} {'Count':>8} {'%':>8}")
    print(f"  {DASH}")
    fine_bins = [(0, 1, "<1%"), (1, 2, "1-2%"), (2, 3, "2-3%"), (3, 4, "3-4%"), (4, 5, "4-5%")]
    for lo, hi, label in fine_bins:
        c = ((straight["mfe_pct"] >= lo) & (straight["mfe_pct"] < hi)).sum()
        pct = c / n_str * 100 if n_str else 0
        print(f"  {label:<20} {c:>8} {pct:>7.1f}%")
    truly_zero = (straight["mfe_pct"] < 0.1).sum()
    print(f"  {'(of which ~0%)':<20} {truly_zero:>8}  (price hit SL almost immediately)")

    # ── 9. Combined: zone_kind × h4_bias for straight-to-SL ─────────────
    print(f"\n  9. ZONE KIND × H4 BIAS  (straight-to-SL only, n={n_str})")
    print(f"  {'Kind × Bias':<22} {'n':>6} {'% of str':>10}")
    print(f"  {DASH}")
    for k in sorted(straight["zone_kind"].dropna().unique()):
        for bias in sorted(straight["h4_bias"].unique()):
            c = ((straight["zone_kind"] == k) & (straight["h4_bias"] == bias)).sum()
            if c == 0:
                continue
            pct = c / n_str * 100
            label = f"{k} × {bias}"
            flag = "  ◄" if pct > 10 else ""
            print(f"  {label:<22} {c:>6} {pct:>9.1f}%{flag}")

    print(f"\n{'='*68}")
    print(f"  ◄ = Straight% > 35% (worth investigating as a filter candidate)")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
