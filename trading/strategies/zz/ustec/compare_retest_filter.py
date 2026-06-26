"""
Baseline vs retest_buys_only comparison.
2023-01-01 to 2025-12-31, fixed_lot 0.01, cash 150.

Usage:
    python trading/strategies/zz/ustec/compare_retest_filter.py \
        --start 2023-01-01 --end 2025-12-31 --fixed_lot 0.01 --cash 150
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT,
)
from trading.strategies.zz.ustec.engine import run_backtest

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _run(label, retest_buys_only, args):
    print(f"\n{'='*55}")
    print(f"  Running: {label}")
    print(f"{'='*55}")
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
        retest_buys_only=retest_buys_only,
        silent=True,
    )
    if not result or isinstance(result, dict):
        print(f"  ERROR: no trades returned for {label}")
        return None, None
    return result


def _arrival_stats(df, side=None):
    g = df if side is None else df[df["side"] == side]
    out = {}
    for at in ("gradual", "retest"):
        sub = g[g["arrival_type"] == at]
        n   = len(sub)
        w   = int((sub["outcome"] == 1).sum())
        pnl = sub["pnl"].sum()
        wr  = w / n * 100 if n else 0
        out[at] = {"n": n, "wins": w, "wr": wr, "pnl": pnl}
    return out


def _pnl_by_year(df, years):
    df = df.copy()
    df["year"] = pd.to_datetime(df["exit_date"]).dt.year
    return {y: df[df["year"] == y]["pnl"].sum() for y in years}


def _pnl_by_month(df):
    df = df.copy()
    ts = pd.to_datetime(df["exit_date"])
    df["year"]  = ts.dt.year
    df["month"] = ts.dt.month
    rows = {}
    for (y, m), g in df.groupby(["year", "month"]):
        rows[(y, m)] = g["pnl"].sum()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    args = parser.parse_args()

    r_base  = _run("Baseline (gradual+retest buys)", retest_buys_only=False, args=args)
    r_retest = _run("Retest buys only",               retest_buys_only=True,  args=args)

    if r_base[0] is None or r_retest[0] is None:
        return

    m_b, df_b = r_base
    m_r, df_r = r_retest

    years = [2023, 2024, 2025]
    py_b  = _pnl_by_year(df_b, years)
    py_r  = _pnl_by_year(df_r, years)
    pm_b  = _pnl_by_month(df_b)
    pm_r  = _pnl_by_month(df_r)

    W  = 20
    L  = 26
    S  = "-" * (L + W * 2 + 4)

    def row(label, v1, v2):
        print(f"  {label:<{L}} {str(v1):>{W}}   {str(v2):>{W}}")

    def section(title):
        print(f"\n  {title}")
        print("  " + S)
        row("", "Baseline", "Retest-Only Buys")
        print("  " + S)

    print("\n" + "=" * (L + W * 2 + 8))
    print("  USTEC Zone-to-Zone: Retest Filter Impact")
    print(f"  Period   : {args.start} to {args.end}")
    print(f"  Cash     : ${args.cash:,.2f}  |  Lot : {args.fixed_lot}")
    print(f"  Rule     : gradual long entries blocked when retest_buys_only=True")
    print(f"  Sells    : unchanged in both runs (gradual sells still allowed)")
    print("=" * (L + W * 2 + 8))

    # Key metrics
    section("Key Metrics")
    row("Net PnL",         m_b["net_pnl"],                m_r["net_pnl"])
    row("Final Equity",    m_b["final_equity"],            m_r["final_equity"])
    row("Win Rate %",      m_b["win_rate_%"] + "%",        m_r["win_rate_%"] + "%")
    row("Total Trades",    m_b["total_trades"],            m_r["total_trades"])
    row("Max Drawdown %",  m_b["max_drawdown_%"] + "%",   m_r["max_drawdown_%"] + "%")
    row("Largest Loss",    m_b["largest_loss_$"],          m_r["largest_loss_$"])
    row("Largest Win",     m_b["largest_win_$"],           m_r["largest_win_$"])
    row("Avg Win",         m_b["avg_win_$"],               m_r["avg_win_$"])
    row("Avg Loss",        m_b["avg_loss_$"],              m_r["avg_loss_$"])

    # Arrival breakdown — buys
    section("Buy Trades: Arrival Breakdown")
    ab_b = _arrival_stats(df_b, "buy")
    ab_r = _arrival_stats(df_r, "buy")
    for at in ("gradual", "retest"):
        d_b = ab_b[at]
        d_r = ab_r[at]
        row(f"  {at} trades",
            f"{d_b['n']} ({d_b['wr']:.1f}% WR, ${d_b['pnl']:+.2f})",
            f"{d_r['n']} ({d_r['wr']:.1f}% WR, ${d_r['pnl']:+.2f})")

    # Sell breakdown (should be identical)
    section("Sell Trades: Arrival Breakdown (sanity check)")
    as_b = _arrival_stats(df_b, "sell")
    as_r = _arrival_stats(df_r, "sell")
    for at in ("gradual", "retest"):
        d_b = as_b[at]
        d_r = as_r[at]
        row(f"  {at} trades",
            f"{d_b['n']} ({d_b['wr']:.1f}% WR)",
            f"{d_r['n']} ({d_r['wr']:.1f}% WR)")

    # Skipped gradual long counter
    print()
    skip_b = df_b[df_b["arrival_type"] == "gradual"].shape[0]
    skip_r = df_r[df_r["arrival_type"] == "gradual"].shape[0]
    skipped_count = skip_b - skip_r   # how many gradual buys were blocked
    print(f"  Gradual buy entries taken    : Baseline={skip_b}  |  Retest-only={skip_r}")
    print(f"  Gradual long entries BLOCKED : {skipped_count} trades skipped by retest_buys_only filter")
    retest_b = df_b[df_b["arrival_type"] == "retest"].shape[0]
    retest_r = df_r[df_r["arrival_type"] == "retest"].shape[0]
    print(f"  Retest entries taken         : Baseline={retest_b}  |  Retest-only={retest_r}")

    # Year-by-year
    section("Year-by-Year PnL")
    for y in years:
        row(str(y), f"${py_b.get(y, 0):+.2f}", f"${py_r.get(y, 0):+.2f}")
    row("Total", f"${sum(py_b.values()):+.2f}", f"${sum(py_r.values()):+.2f}")

    # Month-by-month
    section("Month-by-Month PnL")
    all_ym = sorted(set(list(pm_b.keys()) + list(pm_r.keys())))
    prev_y = None
    for y, m in all_ym:
        if y != prev_y:
            if prev_y is not None:
                print()
            prev_y = y
        lbl = f"{MONTH_NAMES[m]} {y}"
        v_b = pm_b.get((y, m), 0.0)
        v_r = pm_r.get((y, m), 0.0)
        flag = "  <--" if abs(v_r - v_b) > 50 else ""
        row(lbl, f"${v_b:+.2f}", f"${v_r:+.2f}" + flag)

    print("\n" + "=" * (L + W * 2 + 8) + "\n")


if __name__ == "__main__":
    main()
