"""
Month-by-month trade breakdown with every trade listed.
Usage:
    python trading/strategies/zz/ustec/monthly_trade_breakdown.py \
        --start 2023-01-01 --end 2025-12-31 --fixed_lot 0.01 --cash 150 \
        --bad_hours 0,2,9,11,16,21,23
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=10_000.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    parser.add_argument("--bad_hours", default="",
                        help="Comma-separated UTC hours to block gradual long entries")
    args = parser.parse_args()

    bad_hours = [int(h) for h in args.bad_hours.split(",") if h.strip()] or None

    label = f"bad-hour filter {bad_hours}" if bad_hours else "baseline"
    print(f"\nRunning backtest ({label}) ...")

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
        block_gradual_long_hours=bad_hours,
        silent=True,
    )

    if not result or isinstance(result, dict):
        print("ERROR: no trades returned.")
        return

    metrics, df = result

    df["entry_dt"] = pd.to_datetime(df["date"])
    df["exit_dt"]  = pd.to_datetime(df["exit_date"])
    df["year"]     = df["exit_dt"].dt.year
    df["month"]    = df["exit_dt"].dt.month
    df["hour"]     = df["entry_dt"].dt.hour

    total   = len(df)
    wins    = int((df["outcome"] == 1).sum())
    net     = df["pnl"].sum()

    print()
    print("=" * 108)
    print(f"  USTEC Zone-to-Zone  |  {args.start} to {args.end}  |  {label}")
    print(f"  Total: {total} trades  |  {wins}W / {total-wins}L  |  WR {wins/total*100:.1f}%  |  Net ${net:+.2f}")
    print("=" * 108)

    hdr = (f"  {'Date/Time':<18} {'Side':<5} {'Arr':<8} {'Confs':<6} {'Signals':<32} "
           f"{'Entry':>8} {'Exit':>8} {'PnL':>8}  {'Out'}")
    div = "  " + "-" * 104

    prev_ym = None

    for _, t in df.sort_values("exit_dt").iterrows():
        ym = (int(t["year"]), int(t["month"]))

        if ym != prev_ym:
            if prev_ym is not None:
                # print month summary
                g = df[(df["year"] == prev_ym[0]) & (df["month"] == prev_ym[1])]
                mw = int((g["outcome"] == 1).sum())
                mn = len(g)
                print(div)
                print(f"  {'':18} {'':5} {'':8} {'':6} {'':32} "
                      f"  {mw}W/{mn-mw}L  WR {mw/mn*100:.0f}%   "
                      f"Net ${g['pnl'].sum():+.2f}")
                print()

            prev_ym = ym
            print(f"\n  --- {MONTH_NAMES[ym[1]]} {ym[0]} ---")
            print(hdr)
            print(div)

        outcome_str = "WIN " if t["outcome"] == 1 else ("LOSS" if t["pnl"] < 0 else "exp ")
        arr = t["arrival_type"][:7]
        sigs = str(t["signals"])[:32]
        dt_str = t["entry_dt"].strftime("%Y-%m-%d %H:%M")

        print(f"  {dt_str:<18} {t['side']:<5} {arr:<8} {int(t['confirmations']):<6} "
              f"{sigs:<32} {t['entry']:>8.1f} {t['exit']:>8.1f} "
              f"{t['pnl']:>+8.2f}  {outcome_str}")

    # last month summary
    if prev_ym:
        g = df[(df["year"] == prev_ym[0]) & (df["month"] == prev_ym[1])]
        mw = int((g["outcome"] == 1).sum())
        mn = len(g)
        print(div)
        print(f"  {'':18} {'':5} {'':8} {'':6} {'':32} "
              f"  {mw}W/{mn-mw}L  WR {mw/mn*100:.0f}%   "
              f"Net ${g['pnl'].sum():+.2f}")

    # Grand summary
    print()
    print("=" * 108)
    winners = df[df["pnl"] > 0]
    losers  = df[df["pnl"] < 0]
    rr = winners["pnl"].mean() / abs(losers["pnl"].mean()) if len(losers) else 0
    print(f"  TOTAL  {total} trades  |  {wins}W / {total-wins}L  |  "
          f"WR {wins/total*100:.1f}%  |  RR {rr:.2f}  |  Net ${net:+.2f}  |  "
          f"Final equity {metrics['final_equity']}")
    print("=" * 108)
    print()


if __name__ == "__main__":
    main()
