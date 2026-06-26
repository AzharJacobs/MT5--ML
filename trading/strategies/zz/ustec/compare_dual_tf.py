"""
Side-by-side comparison: 4H-only vs dual (1H+4H) vs dual+bias-gate backtests.

Usage:
    python trading/strategies/zz/ustec/compare_dual_tf.py \
        --start 2023-01-01 --end 2025-12-31 --fixed_lot 0.01 --cash 150

Overlap priority: 4H wins when both TFs fire with overlapping zones.
Bias gate: when enabled, 1H signals are only taken when 4H macro bias agrees.
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

MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _run(label: str, dual_tf: bool, h4_bias_gate_1h: bool, args) -> tuple:
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"{'='*60}")
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
        dual_tf=dual_tf,
        h4_bias_gate_1h=h4_bias_gate_1h,
        silent=True,
    )
    if not result or isinstance(result, dict):
        print(f"ERROR: no trades for {label}")
        return None, None
    metrics, df_t = result
    return metrics, df_t


def _pnl_by_year(df_t: pd.DataFrame, years: list) -> dict:
    df_t = df_t.copy()
    df_t["year"] = pd.to_datetime(df_t["exit_date"]).dt.year
    return {y: df_t[df_t["year"] == y]["pnl"].sum() for y in years}


def _pnl_by_month(df_t: pd.DataFrame, year_months: list) -> dict:
    """Return {(year, month): pnl} for every (year, month) in year_months."""
    df_t = df_t.copy()
    ts = pd.to_datetime(df_t["exit_date"])
    df_t["year"]  = ts.dt.year
    df_t["month"] = ts.dt.month
    result = {}
    for y, m in year_months:
        g = df_t[(df_t["year"] == y) & (df_t["month"] == m)]
        result[(y, m)] = g["pnl"].sum() if len(g) else 0.0
    return result


def _fmt(v):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    return f"${v:+.2f}"


def _generate_year_months(start: str, end: str) -> list:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    cur = pd.Timestamp(year=s.year, month=s.month, day=1)
    out = []
    while cur <= e:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare 4H-only vs dual (1H+4H) vs dual+bias-gate backtests"
    )
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    args = parser.parse_args()

    m_4h,   df_4h   = _run("4H Only",          dual_tf=False, h4_bias_gate_1h=False, args=args)
    m_dual, df_dual  = _run("Dual (1H+4H)",     dual_tf=True,  h4_bias_gate_1h=False, args=args)
    m_gate, df_gate  = _run("Dual + Bias Gate", dual_tf=True,  h4_bias_gate_1h=True,  args=args)

    runs = [
        ("4H Only",          m_4h,   df_4h),
        ("Dual (1H+4H)",     m_dual, df_dual),
        ("Dual+BiasGate",    m_gate, df_gate),
    ]

    if any(m is None for _, m, _ in runs):
        print("\nOne or more runs failed - cannot produce comparison.")
        return

    years      = [2023, 2024, 2025]
    year_months = _generate_year_months(args.start, args.end)

    py   = {lbl: _pnl_by_year(df, years)       for lbl, _, df in runs}
    pm   = {lbl: _pnl_by_month(df, year_months) for lbl, _, df in runs}

    # Layout
    W   = 17   # column width per run
    L   = 22   # label width
    SEP = "-" * (L + W * 3 + 4)

    hdrs = [lbl for lbl, _, _ in runs]

    def row(label, vals):
        cells = "  ".join(f"{str(v):>{W}}" for v in vals)
        print(f"  {label:<{L}}  {cells}")

    def section(title):
        print(f"\n  {title}")
        print("  " + SEP)
        row("", hdrs)
        print("  " + SEP)

    print("\n" + "=" * (L + W * 3 + 8))
    print("  USTEC Zone-to-Zone: 3-Way Comparison")
    print(f"  Period   : {args.start} to {args.end}")
    print(f"  Cash     : ${args.cash:,.2f}  |  Fixed lot : {args.fixed_lot}")
    print(f"  4H>1H    : 4H wins on zone overlap or same-bar tie")
    print(f"  BiasGate : 1H signals blocked when 4H bias disagrees")
    print("=" * (L + W * 3 + 8))

    # Key metrics
    section("Key Metrics")
    row("Net PnL",         [m["net_pnl"]          for _, m, _ in runs])
    row("Win Rate %",      [m["win_rate_%"] + "%"  for _, m, _ in runs])
    row("Total Trades",    [m["total_trades"]       for _, m, _ in runs])
    row("Max Drawdown %",  [m["max_drawdown_%"]+"%"  for _, m, _ in runs])
    row("Largest Loss",    [m["largest_loss_$"]     for _, m, _ in runs])
    row("Largest Win",     [m["largest_win_$"]      for _, m, _ in runs])
    row("Avg Win",         [m["avg_win_$"]           for _, m, _ in runs])
    row("Avg Loss",        [m["avg_loss_$"]          for _, m, _ in runs])

    # TF source split for dual runs
    print()
    for lbl, _, df in runs[1:]:
        if "tf_source" in df.columns:
            sc = df["tf_source"].value_counts()
            h4n = int(sc.get("4H", 0))
            h1n = int(sc.get("1H", 0))
            print(f"    {lbl}: {h4n} from 4H zones, {h1n} from 1H zones")

    # Year-by-year
    section("Year-by-Year PnL")
    for y in years:
        row(str(y), [_fmt(py[lbl].get(y, 0.0)) for lbl, _, _ in runs])
    row("Total", [_fmt(sum(py[lbl].values())) for lbl, _, _ in runs])

    # Month-by-month
    section("Month-by-Month PnL")
    prev_year = None
    for y, m in year_months:
        if y != prev_year:
            if prev_year is not None:
                print()
            prev_year = y
        lbl_m = f"{MONTH_NAMES[m]} {y}"
        vals  = [_fmt(pm[run_lbl].get((y, m), 0.0)) for run_lbl, _, _ in runs]
        row(lbl_m, vals)

    print("\n" + "=" * (L + W * 3 + 8) + "\n")


if __name__ == "__main__":
    main()
