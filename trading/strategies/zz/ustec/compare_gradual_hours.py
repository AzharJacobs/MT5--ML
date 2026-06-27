"""
Baseline vs gradual-long bad-hour filter comparison.

Usage:
    python trading/strategies/zz/ustec/compare_gradual_hours.py \
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

BAD_HOURS = [0, 2, 9, 11, 16, 21, 23]

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _run(label, block_hours, args):
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
        block_gradual_long_hours=block_hours,
        silent=True,
    )
    if not result or isinstance(result, dict):
        print(f"  ERROR: no trades for {label}")
        return None, None
    return result


def _pnl_by_year(df, years):
    df = df.copy()
    df["year"] = pd.to_datetime(df["exit_date"]).dt.year
    return {y: df[df["year"] == y]["pnl"].sum() for y in years}


def _pnl_by_month(df):
    df = df.copy()
    ts = pd.to_datetime(df["exit_date"])
    df["year"]  = ts.dt.year
    df["month"] = ts.dt.month
    return {(y, m): g["pnl"].sum() for (y, m), g in df.groupby(["year", "month"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    args = parser.parse_args()

    r_base   = _run("Baseline",                       block_hours=None,      args=args)
    r_filter = _run("Bad-hour gradual long blocked",  block_hours=BAD_HOURS, args=args)

    if r_base[0] is None or r_filter[0] is None:
        return

    m_b, df_b = r_base
    m_f, df_f = r_filter

    years = [2023, 2024, 2025]
    py_b  = _pnl_by_year(df_b, years)
    py_f  = _pnl_by_year(df_f, years)
    pm_b  = _pnl_by_month(df_b)
    pm_f  = _pnl_by_month(df_f)

    W  = 22
    L  = 24
    S  = "-" * (L + W * 2 + 4)

    def row(label, v1, v2, flag=""):
        print(f"  {label:<{L}} {str(v1):>{W}}   {str(v2):>{W}}{flag}")

    def section(title):
        print(f"\n  {title}")
        print("  " + S)
        row("", "Baseline", "Bad-Hour Filter")
        print("  " + S)

    print("\n" + "=" * (L + W * 2 + 8))
    print("  USTEC Zone-to-Zone: Gradual Long Bad-Hour Filter")
    print(f"  Period  : {args.start} to {args.end}")
    print(f"  Cash    : ${args.cash:,.2f}  |  Lot : {args.fixed_lot}")
    print(f"  Blocked : gradual long entries at UTC hours {BAD_HOURS}")
    print(f"  Safe    : retest buys + all sells untouched regardless of hour")
    print("=" * (L + W * 2 + 8))

    section("Key Metrics")
    row("Net PnL",        m_b["net_pnl"],               m_f["net_pnl"])
    row("Final Equity",   m_b["final_equity"],           m_f["final_equity"])
    row("Win Rate %",     m_b["win_rate_%"] + "%",       m_f["win_rate_%"] + "%")
    row("Total Trades",   m_b["total_trades"],           m_f["total_trades"])
    row("Max DD %",       m_b["max_drawdown_%"] + "%",   m_f["max_drawdown_%"] + "%")
    row("Largest Loss",   m_b["largest_loss_$"],         m_f["largest_loss_$"])
    row("Largest Win",    m_b["largest_win_$"],          m_f["largest_win_$"])
    row("Avg Win",        m_b["avg_win_$"],              m_f["avg_win_$"])
    row("Avg Loss",       m_b["avg_loss_$"],             m_f["avg_loss_$"])

    # Skipped counter
    print()
    # Compute from trade tables: gradual buys at bad hours in baseline that don't appear in filter
    df_b_gb = df_b[(df_b["side"] == "buy") & (df_b["arrival_type"] == "gradual")].copy()
    df_b_gb["hour"] = pd.to_datetime(df_b_gb["date"]).dt.hour
    blocked_n = df_b_gb[df_b_gb["hour"].isin(BAD_HOURS)]
    safe_n    = df_b_gb[~df_b_gb["hour"].isin(BAD_HOURS)]
    print(f"  Gradual long bad-hour entries blocked : {len(blocked_n)}")
    print(f"    Of those: {int((blocked_n['outcome']==1).sum())} wins / {int((blocked_n['outcome']==-1).sum())} losses"
          f"  (WR {int((blocked_n['outcome']==1).sum())/max(len(blocked_n),1)*100:.1f}%)"
          f"  net ${blocked_n['pnl'].sum():+.2f}")
    print(f"  Gradual long good-hour entries kept   : {len(safe_n)}")
    print(f"    Of those: {int((safe_n['outcome']==1).sum())} wins / {int((safe_n['outcome']==-1).sum())} losses"
          f"  (WR {int((safe_n['outcome']==1).sum())/max(len(safe_n),1)*100:.1f}%)"
          f"  net ${safe_n['pnl'].sum():+.2f}")

    section("Year-by-Year PnL")
    for y in years:
        delta = py_f.get(y, 0) - py_b.get(y, 0)
        flag  = f"  ({delta:+.2f})" if abs(delta) > 1 else ""
        row(str(y), f"${py_b.get(y,0):+.2f}", f"${py_f.get(y,0):+.2f}", flag)
    tot_b = sum(py_b.values())
    tot_f = sum(py_f.values())
    row("Total", f"${tot_b:+.2f}", f"${tot_f:+.2f}", f"  ({tot_f-tot_b:+.2f})")

    section("Month-by-Month PnL")
    all_ym = sorted(set(list(pm_b.keys()) + list(pm_f.keys())))
    prev_y = None
    for y, m in all_ym:
        if y != prev_y:
            if prev_y is not None:
                print()
            prev_y = y
        lbl  = f"{MONTH_NAMES[m]} {y}"
        v_b  = pm_b.get((y, m), 0.0)
        v_f  = pm_f.get((y, m), 0.0)
        delta = v_f - v_b
        flag  = f"  ({delta:+.2f})" if abs(delta) > 10 else ""
        row(lbl, f"${v_b:+.2f}", f"${v_f:+.2f}", flag)

    print("\n" + "=" * (L + W * 2 + 8) + "\n")


if __name__ == "__main__":
    main()
