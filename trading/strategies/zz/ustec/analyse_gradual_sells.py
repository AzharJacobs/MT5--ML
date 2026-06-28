"""
Analyse gradual sell entries by hour — find bad hours driving losses.

Usage:
    python trading/strategies/zz/ustec/analyse_gradual_sells.py \
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=150.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
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
        print("ERROR: no trades."); return

    metrics, df = result
    df["entry_dt"] = pd.to_datetime(df["date"])
    df["hour"]     = df["entry_dt"].dt.hour

    gs = df[(df["side"] == "sell") & (df["arrival_type"] == "gradual")].copy()
    gr = df[(df["side"] == "sell") & (df["arrival_type"] == "retest")].copy()
    total_sells = df[df["side"] == "sell"]

    hdiv = "=" * 72
    div  = "-" * 72

    print()
    print(hdiv)
    print(f"  Gradual Sell Analysis  |  {args.start} to {args.end}")
    print(hdiv)
    print(f"  Total sells      : {len(total_sells)}  "
          f"({int((total_sells['outcome']==1).sum())}W / {int((total_sells['outcome']!=1).sum())}L)  "
          f"WR {(total_sells['outcome']==1).mean()*100:.1f}%  "
          f"net ${total_sells['pnl'].sum():+.2f}")
    print(f"  Gradual sells    : {len(gs)}  "
          f"({int((gs['outcome']==1).sum())}W / {int((gs['outcome']!=1).sum())}L)  "
          f"WR {(gs['outcome']==1).mean()*100:.1f}%  "
          f"net ${gs['pnl'].sum():+.2f}")
    print(f"  Retest sells     : {len(gr)}  "
          f"({int((gr['outcome']==1).sum())}W / {int((gr['outcome']!=1).sum())}L)  "
          f"WR {(gr['outcome']==1).mean()*100:.1f}%  "
          f"net ${gr['pnl'].sum():+.2f}")

    # ── Hour breakdown for gradual sells ─────────────────────────────────────
    print()
    print(f"  Gradual Sell — by Entry Hour (UTC)")
    print(div)
    print(f"  {'Hour':>5}  {'Trades':>7}  {'W':>4}  {'L':>4}  {'WR%':>6}  {'Net PnL':>10}  {'Avg PnL':>9}  {'Status'}")
    print(div)

    bad_sell_hours = []
    hour_data = []
    for h in range(24):
        g = gs[gs["hour"] == h]
        if len(g) == 0:
            continue
        w   = int((g["outcome"] == 1).sum())
        l   = int((g["outcome"] != 1).sum())
        wr  = w / len(g) * 100
        net = g["pnl"].sum()
        avg = g["pnl"].mean()
        hour_data.append((h, len(g), w, l, wr, net, avg))

    # Sort by net PnL ascending (worst first)
    hour_data.sort(key=lambda x: x[5])

    for h, n, w, l, wr, net, avg in hour_data:
        status = ""
        if net < -50 and wr < 25:
            status = "BAD"
            bad_sell_hours.append(h)
        elif net > 50 and wr > 40:
            status = "good"
        print(f"  {h:>5}h  {n:>7}  {w:>4}  {l:>4}  {wr:>5.1f}%  {net:>+10.2f}  {avg:>+9.2f}  {status}")

    print(div)
    print(f"  Bad hours (net < -$50 AND WR < 25%) : {sorted(bad_sell_hours)}")

    # ── Same hours as buy bad-hours for comparison ────────────────────────────
    buy_bad = set(BAD_HOURS)
    sell_bad = set(bad_sell_hours)
    overlap = buy_bad & sell_bad
    sell_only_bad = sell_bad - buy_bad
    print(f"  Overlap with buy bad hours           : {sorted(overlap)}")
    print(f"  Sell-only bad hours                  : {sorted(sell_only_bad)}")

    # ── What would blocking bad-hour gradual sells save? ─────────────────────
    print()
    print(f"  If we blocked gradual sells at hours {sorted(bad_sell_hours)}:")
    would_block = gs[gs["hour"].isin(bad_sell_hours)]
    would_keep  = gs[~gs["hour"].isin(bad_sell_hours)]
    if len(would_block):
        bw = int((would_block["outcome"] == 1).sum())
        bl = int((would_block["outcome"] != 1).sum())
        print(f"    Blocked: {len(would_block)} trades  {bw}W/{bl}L  "
              f"WR {bw/len(would_block)*100:.1f}%  net ${would_block['pnl'].sum():+.2f}")
    if len(would_keep):
        kw = int((would_keep["outcome"] == 1).sum())
        kl = int((would_keep["outcome"] != 1).sum())
        print(f"    Kept   : {len(would_keep)} trades  {kw}W/{kl}L  "
              f"WR {kw/len(would_keep)*100:.1f}%  net ${would_keep['pnl'].sum():+.2f}")

    # ── Full list of gradual sell losses ──────────────────────────────────────
    print()
    print(f"  All gradual sell trades (sorted by PnL)")
    print(div)
    print(f"  {'Date':<12} {'Hr':>3}  {'Signals':<32}  {'PnL':>8}  {'Out':<5}  {'Bad?'}")
    print(div)
    for _, t in gs.sort_values("pnl").iterrows():
        bad = "BAD" if t["hour"] in bad_sell_hours else ""
        out = "WIN" if t["outcome"] == 1 else ("LOSS" if t["pnl"] < 0 else "exp")
        dt  = str(t["entry_dt"])[:10]
        sig = str(t["signals"])[:32]
        print(f"  {dt:<12} {int(t['hour']):>3}h  {sig:<32}  {t['pnl']:>+8.2f}  {out:<5}  {bad}")
    print(div)
    print(f"  Gradual sells total: {len(gs)}  net ${gs['pnl'].sum():+.2f}")
    print(hdiv)
    print()


if __name__ == "__main__":
    main()
