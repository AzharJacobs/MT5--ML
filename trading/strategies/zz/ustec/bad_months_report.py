"""
Detailed trade report for Aug 2023, Mar 2025, May 2025.
Shows every trade: timing, prices, zone, signals, duration, and why it lost.
"""

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
    EXCLUDED_FROM_COUNT, load_raw,
)
from trading.strategies.zz.ustec.engine import run_backtest

TARGET_MONTHS = [
    (2023, 8, "Aug 2023"),
    (2025, 3, "Mar 2025"),
    (2025, 5, "May 2025"),
]


def main():
    _cfg      = load_raw()
    trade_cfg = _cfg.get("trade_setup", {})
    use_m15_sl       = bool(trade_cfg.get("use_m15_sl", False))
    m15_sl_atr_floor = float(trade_cfg.get("m15_sl_atr_floor_mult", 0.5))
    max_zone_ht_atr  = float(_cfg.get("zone", {}).get("max_zone_height_atr", 0.0))

    print("\nRunning 3-year backtest to extract bad-month trades ...")
    result = run_backtest(
        start="2023-01-01",
        end="2026-01-01",
        cash=10_000.0,
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

    _, df = result
    df["entry_dt"] = pd.to_datetime(df["date"])
    df["exit_dt"]  = pd.to_datetime(df["exit_date"])
    df["year"]     = df["entry_dt"].dt.year
    df["month"]    = df["entry_dt"].dt.month
    df["duration_h"] = (df["exit_dt"] - df["entry_dt"]).dt.total_seconds() / 3600

    for year, month, label in TARGET_MONTHS:
        mdf = df[(df["year"] == year) & (df["month"] == month)].sort_values("entry_dt")
        if mdf.empty:
            print(f"\n  {label}: no trades found.")
            continue

        total  = len(mdf)
        wins   = int((mdf["outcome"] == 1).sum())
        losses = int((mdf["pnl"] < 0).sum())
        net    = mdf["pnl"].sum()

        W = 90
        print()
        print("=" * W)
        print(f"  {label}  —  {total} trades  |  {wins}W / {losses}L  |  "
              f"WR {wins/total*100:.0f}%  |  Net ${net:+.2f}")
        print("=" * W)

        for idx, (_, t) in enumerate(mdf.iterrows(), 1):
            outcome_str = ("WIN " if t["outcome"] == 1
                           else "LOSS" if t["pnl"] < 0
                           else "BE  ")

            sl_dist  = abs(t["entry"] - t["sl"])
            tp_dist  = abs(t["tp"]   - t["entry"])
            rr_plan  = tp_dist / sl_dist if sl_dist > 0 else 0
            exit_dist = abs(t["exit"] - t["entry"])
            move_pct  = (t["exit"] - t["entry"]) / t["entry"] * 100

            zone_h   = t["zone_top"] - t["zone_bottom"]
            bias     = t["h4_bias"]
            signals  = t["signals"]
            arr      = t["arrival_type"]

            # How far into the zone was entry?
            if t["side"] == "buy":
                zone_pct = (t["zone_top"] - t["entry"]) / zone_h * 100 if zone_h > 0 else 0
                sl_below_zone = t["entry"] - t["sl"] > zone_h   # SL outside zone
            else:
                zone_pct = (t["entry"] - t["zone_bottom"]) / zone_h * 100 if zone_h > 0 else 0
                sl_below_zone = t["sl"] - t["entry"] > zone_h

            print(f"\n  Trade {idx}/{total}  ──────────────────────────────────────────────────────────────────")
            print(f"    Entry   : {t['entry_dt'].strftime('%Y-%m-%d  %H:%M UTC')}   ({t['side'].upper()}  {arr})")
            print(f"    Exit    : {t['exit_dt'].strftime('%Y-%m-%d  %H:%M UTC')}   ({outcome_str}  after {t['duration_h']:.1f}h)")
            print(f"    Prices  : entry={t['entry']:.1f}  SL={t['sl']:.1f}  TP={t['tp']:.1f}  exit={t['exit']:.1f}")
            print(f"    Geometry: SL={sl_dist:.1f}pts  TP={tp_dist:.1f}pts  planned RR={rr_plan:.2f}:1")
            print(f"    Zone    : {t['zone_bottom']:.1f} – {t['zone_top']:.1f}  (height={zone_h:.1f}pts)"
                  f"  fresh={t['zone_fresh']}  strength={t['zone_strength']:.2f}  kind={t['zone_kind']}")
            print(f"    Signals : [{signals}]   confirmations={int(t['confirmations'])}")
            print(f"    H4 bias : {bias}   lot={t['lot']:.2f}   PnL=${t['pnl']:+.2f}")
            print(f"    MFE/MAE : max_favour={t['max_favour']:.1f}pts  max_adverse={t['max_adverse']:.1f}pts")

            # Diagnosis
            reasons = []
            if t["pnl"] < 0:
                if t["max_favour"] < 10:
                    reasons.append("price never moved in our favour — immediate reversal from entry")
                elif t["max_favour"] < sl_dist * 0.3:
                    reasons.append(f"small bounce ({t['max_favour']:.0f}pts) then reversed — zone rejected weakly")
                if t["max_adverse"] > sl_dist * 0.9:
                    reasons.append(f"full SL hit — price drove {t['max_adverse']:.0f}pts adverse")
                if arr == "gradual" and t["zone_fresh"]:
                    reasons.append("first-touch entry on fresh zone — no prior zone reaction to confirm")
                if t["duration_h"] < 2:
                    reasons.append(f"stopped out fast ({t['duration_h']:.1f}h) — no momentum at all")
                if sl_dist > 150:
                    reasons.append(f"very wide SL ({sl_dist:.0f}pts) → large dollar loss even at min size")
            elif t["outcome"] == 0:
                reasons.append(f"expired after {t['duration_h']:.1f}h — closed at BE by trailing stop")

            if reasons:
                print(f"    WHY    : " + reasons[0])
                for r in reasons[1:]:
                    print(f"             " + r)

        # Month summary
        print()
        print(f"  {'─'*88}")
        buy_df  = mdf[mdf["side"] == "buy"]
        sell_df = mdf[mdf["side"] == "sell"]
        grad_df = mdf[mdf["arrival_type"] == "gradual"]
        ret_df  = mdf[mdf["arrival_type"] == "retest"]

        if len(buy_df):
            bw = int((buy_df["outcome"] == 1).sum())
            print(f"  Buys  : {len(buy_df):>2} trades  {bw}W/{len(buy_df)-bw}L  "
                  f"Net ${buy_df['pnl'].sum():+.2f}  avg_sl={buy_df.apply(lambda r: abs(r['entry']-r['sl']), axis=1).mean():.0f}pts")
        if len(sell_df):
            sw = int((sell_df["outcome"] == 1).sum())
            print(f"  Sells : {len(sell_df):>2} trades  {sw}W/{len(sell_df)-sw}L  "
                  f"Net ${sell_df['pnl'].sum():+.2f}  avg_sl={sell_df.apply(lambda r: abs(r['entry']-r['sl']), axis=1).mean():.0f}pts")
        if len(grad_df):
            gw = int((grad_df["outcome"] == 1).sum())
            print(f"  Gradual: {len(grad_df):>2} trades  {gw}W/{len(grad_df)-gw}L  Net ${grad_df['pnl'].sum():+.2f}")
        if len(ret_df):
            rw = int((ret_df["outcome"] == 1).sum())
            print(f"  Retest : {len(ret_df):>2} trades  {rw}W/{len(ret_df)-rw}L  Net ${ret_df['pnl'].sum():+.2f}")

        avg_mfe = mdf["max_favour"].mean()
        avg_mae = mdf["max_adverse"].mean()
        avg_sl  = mdf.apply(lambda r: abs(r["entry"] - r["sl"]), axis=1).mean()
        print(f"  Avg MFE: {avg_mfe:.0f}pts   Avg MAE: {avg_mae:.0f}pts   Avg SL dist: {avg_sl:.0f}pts")
        print(f"  {'─'*88}")

    print()


if __name__ == "__main__":
    main()
