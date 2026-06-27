"""
Bad-hour filter  vs  bad-hour filter + H4 structure gate comparison.

Usage:
    python trading/strategies/zz/ustec/compare_h4_structure.py \
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

FOCUS_MONTHS = {(2023, 8), (2024, 11), (2025, 3), (2025, 11)}


def _run(label, h4_gate, args):
    print(f"\n{'='*55}")
    print(f"  Running: {label}")
    print(f"{'='*55}")
    return run_backtest(
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
        block_gradual_long_hours=BAD_HOURS,
        h4_structure_gate=h4_gate,
        silent=True,
    )


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


def _trades_by_month(df):
    df = df.copy()
    ts = pd.to_datetime(df["exit_date"])
    df["year"]  = ts.dt.year
    df["month"] = ts.dt.month
    result = {}
    for (y, m), g in df.groupby(["year", "month"]):
        result[(y, m)] = {
            "n": len(g),
            "w": int((g["outcome"] == 1).sum()),
            "pnl": g["pnl"].sum(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    args = parser.parse_args()

    r_base = _run("Bad-hour filter only (baseline)",    h4_gate=False, args=args)
    r_gate = _run("Bad-hour + H4 structure gate",       h4_gate=True,  args=args)

    if not r_base or isinstance(r_base, dict):
        print("ERROR: baseline returned no trades."); return
    if not r_gate or isinstance(r_gate, dict):
        print("ERROR: gated run returned no trades."); return

    m_b, df_b = r_base
    m_g, df_g = r_gate

    years = [2023, 2024, 2025]
    py_b  = _pnl_by_year(df_b, years)
    py_g  = _pnl_by_year(df_g, years)
    tm_b  = _trades_by_month(df_b)
    tm_g  = _trades_by_month(df_g)

    W  = 26
    L  = 24
    S  = "-" * (L + W * 2 + 4)

    def row(label, v1, v2, flag=""):
        print(f"  {label:<{L}} {str(v1):>{W}}   {str(v2):>{W}}{flag}")

    def section(title):
        print(f"\n  {title}")
        print("  " + S)
        row("", "Bad-hour filter", "Bad-hour + H4 gate")
        print("  " + S)

    print("\n" + "=" * (L + W * 2 + 8))
    print("  USTEC Zone-to-Zone: H4 Structure Gate")
    print(f"  Period  : {args.start} to {args.end}")
    print(f"  Cash    : ${args.cash:,.2f}  |  Lot : {args.fixed_lot}")
    print(f"  Gate    : block buy when price < H4 20-MA AND last swing low = LL")
    print(f"  Also on : bad-hour gradual long block {BAD_HOURS}")
    print("=" * (L + W * 2 + 8))

    section("Key Metrics")
    row("Net PnL",       m_b["net_pnl"],               m_g["net_pnl"])
    row("Final Equity",  m_b["final_equity"],           m_g["final_equity"])
    row("Win Rate %",    m_b["win_rate_%"] + "%",       m_g["win_rate_%"] + "%")
    row("Total Trades",  m_b["total_trades"],           m_g["total_trades"])
    row("Buy Trades",    m_b["buy_trades"],             m_g["buy_trades"])
    row("Buy Wins",      m_b["buy_wins"],               m_g["buy_wins"])
    row("Max DD %",      m_b["max_drawdown_%"] + "%",   m_g["max_drawdown_%"] + "%")
    row("Largest Loss",  m_b["largest_loss_$"],         m_g["largest_loss_$"])
    row("Largest Win",   m_b["largest_win_$"],          m_g["largest_win_$"])
    row("Avg Win",       m_b["avg_win_$"],              m_g["avg_win_$"])
    row("Avg Loss",      m_b["avg_loss_$"],             m_g["avg_loss_$"])

    # What got blocked
    print()
    # Blocked = buy trades in baseline that are absent from gated (rough proxy via trade count diff)
    # More accurate: find buy entries in df_b that don't exist in df_g by matching date+side
    buy_b = df_b[df_b["side"] == "buy"].copy()
    buy_g = df_g[df_g["side"] == "buy"].copy()
    buy_b["dt_key"] = pd.to_datetime(buy_b["date"]).astype(str)
    buy_g["dt_key"] = pd.to_datetime(buy_g["date"]).astype(str)
    blocked = buy_b[~buy_b["dt_key"].isin(set(buy_g["dt_key"]))]
    print(f"  Buys blocked by H4 structure gate : {len(blocked)}")
    if len(blocked):
        bw = int((blocked["outcome"] == 1).sum())
        bl = int((blocked["outcome"] == -1).sum())
        print(f"    Of those: {bw} wins / {bl} losses"
              f"  (WR {bw/max(len(blocked),1)*100:.1f}%)"
              f"  net ${blocked['pnl'].sum():+.2f}")

    section("Year-by-Year PnL")
    for y in years:
        delta = py_g.get(y, 0) - py_b.get(y, 0)
        flag  = f"  ({delta:+.2f})" if abs(delta) > 1 else ""
        row(str(y), f"${py_b.get(y,0):+.2f}", f"${py_g.get(y,0):+.2f}", flag)
    tot_b = sum(py_b.values())
    tot_g = sum(py_g.values())
    row("Total", f"${tot_b:+.2f}", f"${tot_g:+.2f}", f"  ({tot_g-tot_b:+.2f})")

    section("Month-by-Month PnL  (* = focus month)")
    all_ym = sorted(set(list(tm_b.keys()) + list(tm_g.keys())))
    prev_y = None
    for y, m in all_ym:
        if y != prev_y:
            if prev_y is not None:
                print()
            prev_y = y
        b_info = tm_b.get((y, m), {"n": 0, "w": 0, "pnl": 0.0})
        g_info = tm_g.get((y, m), {"n": 0, "w": 0, "pnl": 0.0})
        delta  = g_info["pnl"] - b_info["pnl"]
        lbl    = f"{MONTH_NAMES[m]} {y}"
        focus  = " *" if (y, m) in FOCUS_MONTHS else ""
        v_b    = f"${b_info['pnl']:+.2f} ({b_info['w']}W/{b_info['n']-b_info['w']}L)"
        v_g    = f"${g_info['pnl']:+.2f} ({g_info['w']}W/{g_info['n']-g_info['w']}L)"
        flag   = f"  ({delta:+.2f}){focus}" if abs(delta) > 10 or focus else focus
        row(lbl, v_b, v_g, flag)

    print("\n" + "=" * (L + W * 2 + 8) + "\n")


if __name__ == "__main__":
    main()
