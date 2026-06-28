"""
Trade forensics — classifies each loss in target months as:
  ROUND-TRIP   : price went meaningfully into profit then reversed to SL
  TIGHT-SL     : adverse excursion barely exceeded SL (wiggle-out candidate)
  STRAIGHT-LOSS: went adverse from the start, never meaningful positive excursion

Also shows whether a BE trigger at midpoint would have saved round-trips,
and whether a wider ATR-based SL would have avoided tight-SL exits.

Usage:
    python trading/strategies/zz/ustec/trade_forensics.py \
        --start 2023-01-01 --end 2025-12-31 --fixed_lot 0.01 --cash 150 \
        --focus "2023-08,2025-11"
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

BAD_BUY_HOURS  = [0, 2, 9, 11, 16, 21, 23]
BAD_SELL_HOURS = [1, 2, 4, 6, 11, 15, 19, 21]

# Thresholds for classification
ROUND_TRIP_MIN_FAVOUR = 25.0   # pts — trade went at least this far in profit before reversing
TIGHT_SL_RATIO        = 1.25   # adverse excursion was within this multiple of SL distance


def classify(row):
    sl_dist = abs(row["entry"] - row["sl"])
    mfe     = row["max_favour"]   # max pts in our favour
    mae     = row["max_adverse"]  # max pts against us

    if mfe >= ROUND_TRIP_MIN_FAVOUR:
        return "ROUND-TRIP"
    if sl_dist > 0 and mae <= sl_dist * TIGHT_SL_RATIO and mfe < 15:
        return "TIGHT-SL"
    return "STRAIGHT-LOSS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",     default="2023-01-01")
    parser.add_argument("--end",       default="2025-12-31")
    parser.add_argument("--cash",      type=float, default=150.0)
    parser.add_argument("--fixed_lot", type=float, default=FIXED_LOTS)
    parser.add_argument("--focus",     default="2023-08,2025-11",
                        help="Comma-separated YYYY-MM months to deep-dive")
    args = parser.parse_args()

    focus_months = []
    for s in args.focus.split(","):
        s = s.strip()
        if s:
            y, m = s.split("-")
            focus_months.append((int(y), int(m)))

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
        block_gradual_long_hours=BAD_BUY_HOURS,
        block_gradual_short_hours=BAD_SELL_HOURS,
        silent=True,
    )

    if not result or isinstance(result, dict):
        print("ERROR: no trades."); return

    _, df = result
    df["entry_dt"] = pd.to_datetime(df["date"])
    df["exit_dt"]  = pd.to_datetime(df["exit_date"])
    df["year"]     = df["exit_dt"].dt.year
    df["month"]    = df["exit_dt"].dt.month
    df["hour"]     = df["entry_dt"].dt.hour
    df["sl_dist"]  = abs(df["entry"] - df["sl"])
    df["sl_pct"]   = df["sl_dist"] / df["entry"] * 100
    df["class"]    = df.apply(classify, axis=1)

    hdiv = "=" * 96
    div  = "-" * 96

    print()
    print(hdiv)
    print(f"  TRADE FORENSICS  |  {args.start} to {args.end}  |  Focus: {args.focus}")
    print(f"  Thresholds: round-trip MFE >= {ROUND_TRIP_MIN_FAVOUR}pts  |  tight-SL MAE <= {TIGHT_SL_RATIO}x SL dist")
    print(hdiv)

    # Overall loss classification summary
    losses = df[df["pnl"] < 0]
    print(f"\n  ALL LOSSES ({len(losses)} total):")
    for cls in ("ROUND-TRIP", "TIGHT-SL", "STRAIGHT-LOSS"):
        g = losses[losses["class"] == cls]
        if len(g):
            print(f"    {cls:<15} : {len(g):>3} trades  net ${g['pnl'].sum():+.2f}"
                  f"  |  avg MFE {g['max_favour'].mean():.1f}pts  avg MAE {g['max_adverse'].mean():.1f}pts")

    for ym in focus_months:
        y, m = ym
        month_df = df[(df["year"] == y) & (df["month"] == m)]
        month_losses = month_df[month_df["pnl"] < 0]

        import calendar
        mname = calendar.month_abbr[m]

        print()
        print(f"  {mname} {y}  |  {len(month_df)} trades  "
              f"{int((month_df['outcome']==1).sum())}W/{int((month_df['outcome']!=1).sum())}L  "
              f"net ${month_df['pnl'].sum():+.2f}")
        print(div)
        print(f"  {'Date':<12} {'Hr':>3}  {'Side':<5} {'Arr':<8} {'Signals':<30}  "
              f"{'SL dist':>8}  {'MFE':>7}  {'MAE':>7}  {'PnL':>8}  {'Class'}")
        print(div)

        for _, t in month_df.sort_values("entry_dt").iterrows():
            is_loss = t["pnl"] < 0
            cls     = t["class"] if is_loss else "-"
            marker  = ""
            if cls == "ROUND-TRIP":   marker = " <RT>"
            elif cls == "TIGHT-SL":   marker = " <TS>"
            elif cls == "STRAIGHT-LOSS": marker = " <SL>"
            sigs = str(t["signals"])[:30]
            print(f"  {str(t['entry_dt'])[:10]:<12} {int(t['hour']):>3}h  "
                  f"{t['side']:<5} {t['arrival_type'][:7]:<8} {sigs:<30}  "
                  f"{t['sl_dist']:>7.1f}p  {t['max_favour']:>6.1f}p  {t['max_adverse']:>6.1f}p  "
                  f"{t['pnl']:>+8.2f}  {cls}{marker}")

        if len(month_losses) == 0:
            print("  No losses this month.")
            continue

        # ── Per-class deep dive ───────────────────────────────────────────────
        for cls in ("ROUND-TRIP", "TIGHT-SL", "STRAIGHT-LOSS"):
            g = month_losses[month_losses["class"] == cls]
            if len(g) == 0:
                continue

            print()
            print(f"  >> {cls} ({len(g)} trades)  net ${g['pnl'].sum():+.2f}")

            if cls == "ROUND-TRIP":
                print(f"     Price went {g['max_favour'].mean():.0f}pts in profit on avg before reversing to SL.")
                # Would BE at midpoint have saved any?
                saved = []
                for _, t in g.iterrows():
                    midpoint = abs(t["tp"] - t["entry"]) / 2
                    if t["max_favour"] >= midpoint:
                        # BE would have triggered — at worst exit at entry (net ~0 minus spread)
                        saved.append(abs(t["pnl"]))
                if saved:
                    print(f"     BE trigger at midpoint would have saved {len(saved)}/{len(g)} trades"
                          f"  (~${sum(saved):.2f} recovered)")
                else:
                    print(f"     BE trigger at midpoint would NOT have helped"
                          f" — MFE never reached the midpoint before reversing.")
                print(f"     Signals : {', '.join(g['signals'].unique())}")
                print(f"     Hours   : {sorted(g['hour'].tolist())}")

            elif cls == "TIGHT-SL":
                avg_ratio = (g["max_adverse"] / g["sl_dist"]).mean()
                print(f"     MAE was only {avg_ratio:.2f}x the SL distance on avg — price barely clipped SL.")
                print(f"     A SL 30% wider would need MAE <= {avg_ratio:.2f}x * 1.3 = {avg_ratio*1.3:.2f}x SL.")
                print(f"     BUT: without bar-by-bar data we cannot confirm price recovered after stopping out.")
                print(f"     Recommendation: raise min_sl_pct or use ATR-based SL floor.")
                wider_sl_cost = g["sl_dist"].mean() * 0.3 * g["lot"].mean() * 100
                print(f"     Wider SL cost per trade: ~${wider_sl_cost:.2f} extra risk")
                print(f"     Signals : {', '.join(g['signals'].unique())}")
                print(f"     Hours   : {sorted(g['hour'].tolist())}")

            elif cls == "STRAIGHT-LOSS":
                print(f"     MFE avg {g['max_favour'].mean():.1f}pts — price went adverse almost immediately.")
                print(f"     No mechanical fix helps here — these are wrong-direction reads.")
                # Check for common signals
                all_sigs = []
                for sigs in g["signals"]:
                    all_sigs.extend(str(sigs).split("|"))
                from collections import Counter
                sig_counts = Counter(all_sigs)
                print(f"     Common signals: {dict(sig_counts.most_common(4))}")
                print(f"     Hours   : {sorted(g['hour'].tolist())}")
                print(f"     H4 bias : {sorted(g['h4_bias'].unique())}")
                # Check if raising min_confirmations would filter any
                low_conf = g[g["confirmations"] < 2]
                if len(low_conf):
                    print(f"     {len(low_conf)}/{len(g)} had only 1 confirmation"
                          f" — raising min_confirmations to 2 would block these.")
                    print(f"     But check: would that also kill good trades this month?")
                    good = month_df[month_df["pnl"] > 0]
                    good_1conf = good[good["confirmations"] < 2]
                    print(f"     Good trades with 1 conf in same month: {len(good_1conf)}"
                          f" (net ${good_1conf['pnl'].sum():+.2f})")

    # ── Cross-month pattern: do straight losses share a fingerprint? ──────────
    print()
    print(hdiv)
    print("  CROSS-MONTH: STRAIGHT-LOSS fingerprint across all focus months")
    print(div)
    focus_losses_sl = pd.DataFrame()
    for ym in focus_months:
        y, m = ym
        g = df[(df["year"] == y) & (df["month"] == m) &
               (df["pnl"] < 0) & (df["class"] == "STRAIGHT-LOSS")]
        focus_losses_sl = pd.concat([focus_losses_sl, g])

    if len(focus_losses_sl):
        print(f"  {len(focus_losses_sl)} straight losses across focus months  net ${focus_losses_sl['pnl'].sum():+.2f}")
        print(f"  Avg MFE : {focus_losses_sl['max_favour'].mean():.1f}pts")
        print(f"  Avg MAE : {focus_losses_sl['max_adverse'].mean():.1f}pts")
        print(f"  Avg SL dist: {focus_losses_sl['sl_dist'].mean():.1f}pts")
        print()
        print(f"  Confirmations breakdown:")
        for c, cg in focus_losses_sl.groupby("confirmations"):
            print(f"    {int(c)} conf : {len(cg)} trades  net ${cg['pnl'].sum():+.2f}")
        print()
        print(f"  H4 bias at entry:")
        for b, bg in focus_losses_sl.groupby("h4_bias"):
            print(f"    {b:<16} : {len(bg)} trades  net ${bg['pnl'].sum():+.2f}")
        print()
        print(f"  Side:")
        for s, sg in focus_losses_sl.groupby("side"):
            print(f"    {s:<6} : {len(sg)} trades  net ${sg['pnl'].sum():+.2f}")
        print()
        print(f"  Arrival:")
        for a, ag in focus_losses_sl.groupby("arrival_type"):
            print(f"    {a:<8} : {len(ag)} trades  net ${ag['pnl'].sum():+.2f}")

    print()
    print(hdiv)
    print()


if __name__ == "__main__":
    main()
