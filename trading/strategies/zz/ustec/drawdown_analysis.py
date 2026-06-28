"""
Drawdown analysis — identifies every drawdown period, the trades inside each,
and the single deepest trough.

Usage:
    python trading/strategies/zz/ustec/drawdown_analysis.py \
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

BAD_HOURS = [0, 2, 9, 11, 16, 21, 23]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=150.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    parser.add_argument("--top_dd",    type=int,   default=5)
    args = parser.parse_args()

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
        block_gradual_long_hours=BAD_HOURS,
        silent=True,
    )

    if not result or isinstance(result, dict):
        print("ERROR: no trades returned.")
        return

    metrics, df = result
    df = df.sort_values("exit_date").reset_index(drop=True)

    # Reconstruct trade-level equity curve (unclamped for real drawdown maths)
    equity = args.cash
    rows   = []
    for _, t in df.iterrows():
        equity += t["pnl"]
        rows.append({
            "trade_n":    len(rows) + 1,
            "exit_date":  t["exit_date"],
            "entry_date": t["date"],
            "side":       t["side"],
            "arrival":    t["arrival_type"],
            "signals":    t["signals"],
            "pnl":        t["pnl"],
            "outcome":    t["outcome"],
            "equity":     equity,
        })
    eq_df = pd.DataFrame(rows)
    eq_df["peak"] = eq_df["equity"].cummax().clip(lower=args.cash)

    # Prepend starting point
    start_row = pd.DataFrame([{
        "trade_n": 0, "exit_date": args.start, "entry_date": args.start,
        "side": "-", "arrival": "-", "signals": "-",
        "pnl": 0.0, "outcome": 0, "equity": args.cash, "peak": args.cash,
    }])
    eq_df = pd.concat([start_row, eq_df], ignore_index=True)
    eq_df["peak"] = eq_df["equity"].cummax().clip(lower=args.cash)
    eq_df["dd_pts"] = eq_df["equity"] - eq_df["peak"]
    eq_df["dd_pct"] = eq_df["dd_pts"] / eq_df["peak"] * 100

    # ── Identify drawdown periods (peak → trough → recovery) ─────────────────
    in_dd       = False
    dd_start    = None
    dd_peak_eq  = args.cash
    periods     = []

    for i, r in eq_df.iterrows():
        if not in_dd:
            if r["dd_pts"] < -0.01:
                in_dd      = True
                dd_start   = i
                dd_peak_eq = r["peak"]
        else:
            if r["equity"] >= r["peak"] - 0.01:
                # recovery
                trough_i = eq_df.loc[dd_start:i, "equity"].idxmin()
                periods.append({
                    "start_i":   dd_start,
                    "trough_i":  trough_i,
                    "end_i":     i,
                    "peak_eq":   dd_peak_eq,
                    "trough_eq": eq_df.loc[trough_i, "equity"],
                    "depth_pts": eq_df.loc[trough_i, "equity"] - dd_peak_eq,
                    "depth_pct": (eq_df.loc[trough_i, "equity"] - dd_peak_eq) / dd_peak_eq * 100,
                    "trades_in": i - dd_start,
                    "recovered": True,
                })
                in_dd = False

    if in_dd:
        trough_i = eq_df.loc[dd_start:, "equity"].idxmin()
        periods.append({
            "start_i":   dd_start,
            "trough_i":  trough_i,
            "end_i":     len(eq_df) - 1,
            "peak_eq":   dd_peak_eq,
            "trough_eq": eq_df.loc[trough_i, "equity"],
            "depth_pts": eq_df.loc[trough_i, "equity"] - dd_peak_eq,
            "depth_pct": (eq_df.loc[trough_i, "equity"] - dd_peak_eq) / dd_peak_eq * 100,
            "trades_in": len(eq_df) - 1 - dd_start,
            "recovered": False,
        })

    periods.sort(key=lambda x: x["depth_pts"])

    hdiv = "=" * 90
    div  = "-" * 90

    print()
    print(hdiv)
    print(f"  DRAWDOWN ANALYSIS  |  {args.start} to {args.end}  |  Start: ${args.cash:.2f}")
    print(hdiv)
    print(f"  Total drawdown periods : {len(periods)}")
    print(f"  Max drawdown           : {eq_df['dd_pct'].min():.1f}%   ({eq_df['dd_pts'].min():+.2f} pts)")
    print()

    # ── Top N deepest periods ─────────────────────────────────────────────────
    print(f"  Top {args.top_dd} deepest drawdown periods")
    print(div)
    print(f"  {'#':<3} {'Peak $':>8} {'Trough $':>10} {'Depth':>8} {'Depth%':>8}  "
          f"{'Trades':>7}  {'From':<12}  {'To':<12}")
    print(div)
    for rank, p in enumerate(periods[:args.top_dd], 1):
        t_date = str(eq_df.loc[p["trough_i"], "exit_date"])[:10]
        s_date = str(eq_df.loc[p["start_i"],  "exit_date"])[:10]
        rec    = "" if p["recovered"] else "  (open)"
        print(f"  {rank:<3} {p['peak_eq']:>8.2f} {p['trough_eq']:>10.2f} "
              f"{p['depth_pts']:>+8.2f} {p['depth_pct']:>7.1f}%  "
              f"{p['trades_in']:>7}  {s_date:<12}  {t_date:<12}{rec}")

    # ── Deep-dive into each top period ───────────────────────────────────────
    for rank, p in enumerate(periods[:args.top_dd], 1):
        rows_in = eq_df.iloc[p["start_i"]:p["end_i"] + 1]
        rows_in = rows_in[rows_in["trade_n"] > 0]  # skip the phantom start row

        s_date  = str(eq_df.loc[p["start_i"],  "exit_date"])[:10]
        tr_date = str(eq_df.loc[p["trough_i"], "exit_date"])[:10]

        print()
        print(f"  {'-'*88}")
        print(f"  DD #{rank}  |  Peak ${p['peak_eq']:.2f}  ->  Trough ${p['trough_eq']:.2f}"
              f"  ({p['depth_pct']:.1f}%)  |  {s_date} to {tr_date}"
              + ("  [still open]" if not p["recovered"] else ""))
        print(f"  {'-'*88}")
        print(f"  {'#':<4} {'Date':<12} {'Side':<5} {'Arr':<8} {'Signals':<30} "
              f"{'PnL':>8}  {'Equity':>8}  {'DD%':>7}")
        print(f"  {'-'*88}")

        for _, tr in rows_in.iterrows():
            dd_pct = (tr["equity"] - p["peak_eq"]) / p["peak_eq"] * 100
            marker = " <<" if tr["equity"] == p["trough_eq"] else ""
            sigs   = str(tr["signals"])[:30]
            print(f"  {int(tr['trade_n']):<4} {str(tr['exit_date'])[:10]:<12} "
                  f"{tr['side']:<5} {tr['arrival'][:7]:<8} {sigs:<30} "
                  f"{tr['pnl']:>+8.2f}  {tr['equity']:>8.2f}  {dd_pct:>6.1f}%{marker}")

        # summary of what type of trades caused it
        if len(rows_in):
            losing = rows_in[rows_in["pnl"] < 0]
            buy_l  = losing[losing["side"] == "buy"]
            sell_l = losing[losing["side"] == "sell"]
            grad_l = losing[losing["arrival"] == "gradual"]
            ret_l  = losing[losing["arrival"] == "retest"]
            print(f"  {'-'*88}")
            print(f"  Losses in this period: {len(losing)}"
                  f"  |  buy {len(buy_l)} / sell {len(sell_l)}"
                  f"  |  gradual {len(grad_l)} / retest {len(ret_l)}"
                  f"  |  total bleed ${losing['pnl'].sum():.2f}")

    # ── Running equity table ──────────────────────────────────────────────────
    print()
    print(hdiv)
    print(f"  Full trade-by-trade equity curve")
    print(hdiv)
    print(f"  {'#':<4} {'Exit':<12} {'Side':<5} {'Arr':<8} {'PnL':>8}  {'Equity':>9}  {'DD%':>7}")
    print(div)
    for _, r in eq_df[eq_df["trade_n"] > 0].iterrows():
        marker = " <-- ruin" if r["equity"] <= 0 else (
                 " <<" if r["pnl"] > 200 else (
                 " !!" if r["pnl"] < -150 else ""))
        print(f"  {int(r['trade_n']):<4} {str(r['exit_date'])[:10]:<12} "
              f"{r['side']:<5} {r['arrival'][:7]:<8} "
              f"{r['pnl']:>+8.2f}  {r['equity']:>9.2f}  {r['dd_pct']:>6.1f}%{marker}")
    print(div)
    print(f"  Final equity: ${eq_df['equity'].iloc[-1]:.2f}  |  "
          f"Max DD: {eq_df['dd_pct'].min():.1f}%  |  "
          f"Deepest trough: ${eq_df['equity'].min():.2f}")
    print(hdiv)
    print()


if __name__ == "__main__":
    main()
