"""
3-Year USTEC Monthly Pattern Analysis — 2023, 2024, 2025

Runs the full backtest using all current config.yaml settings and outputs:
  1. Month-by-month table with running balance
  2. Seasonality heat (same calendar month averaged across all 3 years)
  3. Buy/sell and retest/gradual breakdown by year
  4. Weak month deep-dive with worst trades
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from collections import defaultdict

from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT, load_raw,
)
from trading.strategies.zz.ustec.engine import run_backtest

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

START = "2023-01-01"
END   = "2026-01-01"
CASH  = 10_000.0


def main():
    _cfg      = load_raw()
    trade_cfg = _cfg.get("trade_setup", {})
    use_m15_sl       = bool(trade_cfg.get("use_m15_sl", False))
    m15_sl_atr_floor = float(trade_cfg.get("m15_sl_atr_floor_mult", 0.5))
    max_zone_ht_atr  = float(_cfg.get("zone", {}).get("max_zone_height_atr", 0.0))

    print(f"\nRunning 3-year backtest ({START} → {END}) ...")
    print(f"  lot={FIXED_LOTS}  spread={SPREAD_PTS}pts  min_rr={MIN_RR}  "
          f"trailing={ENABLE_TRAILING}  be_trigger={BE_TRIGGER_PTS}pts  "
          f"use_m15_sl={use_m15_sl}  max_zone_ht={max_zone_ht_atr}x  zone_max_losses={ZONE_MAX_LOSSES}")

    result = run_backtest(
        start=START,
        end=END,
        cash=CASH,
        min_rr=MIN_RR,
        max_forward_bars=MAX_FORWARD_BARS,
        symbol="ustech",
        spread=SPREAD_PTS,
        fixed_lot=FIXED_LOTS,
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
        use_m15_sl=use_m15_sl,
        m15_sl_atr_floor_mult=m15_sl_atr_floor,
        max_zone_height_atr=max_zone_ht_atr,
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
    df = df.sort_values("exit_dt").reset_index(drop=True)

    total = len(df)
    wins  = int((df["outcome"] == 1).sum())
    net   = df["pnl"].sum()

    # ── 1. Month-by-month table ───────────────────────────────────────────────
    HDR = "  {:<13} {:>4} {:>4} {:>4} {:>6}  {:>10}  {:>10}  {:>10}"
    ROW = "  {:<13} {:>4} {:>4} {:>4} {:>5.1f}%  {:>+10.2f}  {:>+10.2f}  {:>10.2f}{}"
    W = 82

    print()
    print("=" * W)
    print(f"  USTEC ZZ  —  3-Year Monthly Breakdown  ({START} → {END})")
    print(f"  lot={FIXED_LOTS}  cash=${CASH:,.0f}  spread={SPREAD_PTS}pts  "
          f"use_m15_sl={use_m15_sl}  zone_max_losses={ZONE_MAX_LOSSES}")
    print("=" * W)
    print(HDR.format("Month", "Tr", "W", "L", "WR%", "Net PnL", "Cum PnL", "Balance"))
    print("  " + "-" * (W - 2))

    cum       = 0.0
    month_rows = []
    year_acc  = defaultdict(lambda: {"tr": 0, "w": 0, "l": 0, "pnl": 0.0})
    prev_year = None

    for (y, m), g in df.groupby(["year", "month"]):
        tr  = len(g)
        w   = int((g["outcome"] == 1).sum())
        l   = int((g["pnl"] < 0).sum())
        wr  = w / tr * 100 if tr else 0.0
        pnl = g["pnl"].sum()
        cum += pnl
        year_acc[y]["tr"]  += tr
        year_acc[y]["w"]   += w
        year_acc[y]["l"]   += l
        year_acc[y]["pnl"] += pnl
        month_rows.append({"year": y, "month": m, "tr": tr, "w": w, "l": l,
                            "wr": wr, "pnl": pnl, "cum": cum, "bal": CASH + cum})

        if prev_year is not None and y != prev_year:
            ya  = year_acc[prev_year]
            ywr = ya["w"] / ya["tr"] * 100 if ya["tr"] else 0.0
            print("  " + "·" * (W - 2))
            print(f"  {str(prev_year)+' TOTAL':<13} {ya['tr']:>4} {ya['w']:>4} {ya['l']:>4} "
                  f"{ywr:>5.1f}%  {ya['pnl']:>+10.2f}")
            print()
        prev_year = y

        flag = ""
        if wr < 35 and pnl < 0:
            flag = "  ◄ WEAK"
        elif wr < 40:
            flag = "  ◄ low WR"
        elif wr >= 60 and pnl > 0:
            flag = "  ★"

        print(ROW.format(MONTH_NAMES[m] + " " + str(y), tr, w, l, wr, pnl, cum, CASH + cum, flag))

    if prev_year:
        ya  = year_acc[prev_year]
        ywr = ya["w"] / ya["tr"] * 100 if ya["tr"] else 0.0
        print("  " + "·" * (W - 2))
        print(f"  {str(prev_year)+' TOTAL':<13} {ya['tr']:>4} {ya['w']:>4} {ya['l']:>4} "
              f"{ywr:>5.1f}%  {ya['pnl']:>+10.2f}")

    print()
    print("  " + "=" * (W - 2))
    print(ROW.format("GRAND TOTAL", total, wins, total - wins, wins / total * 100,
                     net, net, CASH + net, ""))
    print("  " + "=" * (W - 2))

    # ── 2. Seasonality ────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  SEASONALITY  —  Same Calendar Month Averaged Across All Years")
    print("=" * W)
    print(f"  {'Mth':<6} {'AvgWR%':>7} {'AvgPnL':>10}  {'Pos/Yrs':>8}  {'Trades':>7}   Notes")
    print("  " + "-" * 70)

    for m_num in range(1, 13):
        rs = [r for r in month_rows if r["month"] == m_num]
        if not rs:
            continue
        avg_wr  = sum(r["wr"]  for r in rs) / len(rs)
        avg_pnl = sum(r["pnl"] for r in rs) / len(rs)
        pos     = sum(1 for r in rs if r["pnl"] > 0)
        tot_tr  = sum(r["tr"]  for r in rs)
        note = ""
        if avg_wr < 40 and avg_pnl < 0:
            note = "CONSISTENTLY WEAK"
        elif avg_wr < 40:
            note = "low WR"
        elif pos == len(rs) and avg_wr >= 55:
            note = "STRONG — profitable every year"
        elif pos == len(rs):
            note = "profitable every year"
        elif avg_wr >= 60:
            note = "STRONG"
        print(f"  {MONTH_NAMES[m_num]:<6} {avg_wr:>6.1f}%  {avg_pnl:>+10.2f}   {pos}/{len(rs)}       {tot_tr:>7}   {note}")

    # ── 3. Side breakdown per year ────────────────────────────────────────────
    print()
    print("=" * W)
    print("  SIDE BREAKDOWN  —  Buys vs Sells per Year")
    print("=" * W)
    print(f"  {'Year / Side':<16} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'Net PnL':>12}")
    print("  " + "-" * 55)
    for y in sorted(df["year"].unique()):
        ydf = df[df["year"] == y]
        for side in ("buy", "sell"):
            sdf = ydf[ydf["side"] == side]
            if sdf.empty:
                continue
            sw  = int((sdf["outcome"] == 1).sum())
            sn  = len(sdf)
            swr = sw / sn * 100
            sp  = sdf["pnl"].sum()
            print(f"  {str(y)+' '+side:<16} {sn:>7} {sw:>6} {swr:>6.0f}%  {sp:>+12.2f}")
        print()

    # ── 4. Arrival type breakdown per year ────────────────────────────────────
    print("=" * W)
    print("  ARRIVAL TYPE  —  Retest vs Gradual per Year")
    print("=" * W)
    print(f"  {'Year / Type':<16} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'Net PnL':>12}")
    print("  " + "-" * 55)
    for y in sorted(df["year"].unique()):
        ydf = df[df["year"] == y]
        for atype in ("retest", "gradual"):
            adf = ydf[ydf["arrival_type"] == atype]
            if adf.empty:
                continue
            aw  = int((adf["outcome"] == 1).sum())
            an  = len(adf)
            awr = aw / an * 100
            ap  = adf["pnl"].sum()
            print(f"  {str(y)+' '+atype:<16} {an:>7} {aw:>6} {awr:>6.0f}%  {ap:>+12.2f}")
        print()

    # ── 5. Weak months deep-dive ──────────────────────────────────────────────
    weak = [r for r in month_rows if r["wr"] < 40 or r["pnl"] < -150]
    if weak:
        print("=" * W)
        print("  WEAK MONTHS  —  Deep-dive  (WR < 40% OR Net PnL < -$150)")
        print("=" * W)
        for r in weak:
            y, m = r["year"], r["month"]
            mdf = df[(df["year"] == y) & (df["month"] == m)]
            print(f"\n  ◄ {MONTH_NAMES[m]} {y}  —  {r['tr']} trades  "
                  f"{r['w']}W/{r['l']}L  WR {r['wr']:.0f}%  Net ${r['pnl']:+.2f}")
            for side in ("buy", "sell"):
                sdf = mdf[mdf["side"] == side]
                if not sdf.empty:
                    sw = int((sdf["outcome"] == 1).sum())
                    sn = len(sdf)
                    print(f"      {side.capitalize():<6}: {sn:>3} trades  {sw}W/{sn-sw}L  "
                          f"WR {sw/sn*100:.0f}%  Net ${sdf['pnl'].sum():+.2f}")
            for atype in ("retest", "gradual"):
                adf = mdf[mdf["arrival_type"] == atype]
                if not adf.empty:
                    aw = int((adf["outcome"] == 1).sum())
                    an = len(adf)
                    print(f"      {atype.capitalize():<9}: {an:>3} trades  {aw}W/{an-aw}L  "
                          f"WR {aw/an*100:.0f}%  Net ${adf['pnl'].sum():+.2f}")
            # Three worst trades this month
            worst3 = mdf.nsmallest(3, "pnl")
            print(f"      Worst trades this month:")
            for _, t in worst3.iterrows():
                print(f"        {t['entry_dt'].strftime('%Y-%m-%d %H:%M')}  "
                      f"{t['side']:<5}  {t['arrival_type']:<8}  "
                      f"${t['pnl']:+.2f}  [{t['signals']}]")

    print()
    print("=" * W)
    print(f"  TOTAL: {total} trades  |  {wins}W/{total-wins}L  |  WR {wins/total*100:.1f}%  |  "
          f"Net ${net:+.2f}  |  Final ${CASH+net:,.2f}  |  Max DD {metrics['max_drawdown_%']}")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
