"""
Show every trade that lost more than $300 and explain why.
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from collections import Counter
from trading.strategies.zz.ustec.strategy import (
    MIN_RR, SPREAD_PTS, FIXED_LOTS, MAX_FORWARD_BARS, MIN_SL_PCT,
    ZONE_MAX_LOSSES, H4_REGIME_FILTER,
    ENABLE_TRAILING, BE_TRIGGER_PTS, BE_BUFFER_PTS, ATR_TRAIL_MULT,
    EXCLUDED_FROM_COUNT,
)
from trading.strategies.zz.ustec.engine import run_backtest

BAD_BUY_HOURS  = [0, 2, 9, 11, 16, 21, 23]
BAD_SELL_HOURS = [1, 2, 4, 6, 11, 15, 19, 21]
THRESHOLD      = -300.0

result = run_backtest(
    start="2023-01-01", end="2025-12-31",
    cash=150.0, min_rr=MIN_RR, max_forward_bars=MAX_FORWARD_BARS,
    symbol="ustech", spread=SPREAD_PTS, fixed_lot=FIXED_LOTS,
    directional_filter=True, allow_neutral=True,
    h4_swing_left=2, h4_swing_right=2, min_confirmations=1,
    excluded_from_count=list(EXCLUDED_FROM_COUNT),
    zone_max_losses=ZONE_MAX_LOSSES, h4_regime_filter=H4_REGIME_FILTER,
    min_sl_pct=MIN_SL_PCT, enable_trailing=ENABLE_TRAILING,
    be_trigger_pts=BE_TRIGGER_PTS, be_buffer_pts=BE_BUFFER_PTS,
    atr_trail_mult=ATR_TRAIL_MULT,
    block_gradual_long_hours=BAD_BUY_HOURS,
    block_gradual_short_hours=BAD_SELL_HOURS,
    split_exit=True, scalper_tp_pct=0.45,
    silent=True,
)

_, df = result
df["entry_dt"] = pd.to_datetime(df["date"])
df["hour"]     = df["entry_dt"].dt.hour
df["sl_dist"]  = (df["entry"] - df["sl"]).abs()

big = df[df["pnl"] < THRESHOLD].sort_values("pnl").reset_index(drop=True)

div  = "-" * 100
hdiv = "=" * 100

print()
print(hdiv)
print(f"  Trades with PnL < ${THRESHOLD:.0f}  |  {len(big)} found out of {len(df)} total")
print(hdiv)

for _, t in big.iterrows():
    sl_dist  = t["sl_dist"]
    mfe      = t["max_favour"]
    mae      = t["max_adverse"]

    # Classify why it lost big
    reasons = []
    if mae > sl_dist * 2:
        reasons.append("LARGE SL: stop was wide and got hit hard")
    if mfe < 15:
        reasons.append("STRAIGHT LOSS: went adverse from bar 1, no meaningful bounce")
    elif mfe >= 25:
        reasons.append(f"ROUND-TRIP: reached +{mfe:.0f}pts profit then reversed fully to SL")
    if sl_dist > 100:
        reasons.append(f"SL DISTANCE {sl_dist:.0f}pts — zone was very deep, risk was outsized")
    if t["signals"] in ("rejection_wick", "engulfing") and t["confirmations"] == 1:
        reasons.append("WEAK CONFIRMATION: single weak signal, no structural backing")
    if t["h4_bias"] in ("neutral", "neutral_up") and t["side"] == "sell":
        reasons.append("BIAS CONFLICT: selling into neutral/upward H4 bias")
    if t["h4_bias"] in ("neutral", "neutral_down") and t["side"] == "buy":
        reasons.append("BIAS CONFLICT: buying into neutral/downward H4 bias")

    print()
    print(f"  {str(t['entry_dt'])[:16]}  |  {t['side'].upper()}  {t['arrival_type']}  "
          f"|  {t['hour']}h UTC  |  PnL ${t['pnl']:+.2f}")
    print(f"  Entry {t['entry']:.1f}  SL {t['sl']:.1f}  TP {t['tp']:.1f}  "
          f"Exit {t['exit']:.1f}")
    print(f"  SL dist {sl_dist:.1f}pts  |  MFE +{mfe:.1f}pts  |  MAE -{mae:.1f}pts")
    print(f"  H4 bias: {t['h4_bias']}  |  Signals: {t['signals']}  |  Confs: {int(t['confirmations'])}")
    print(f"  Zone: {t['zone_bottom']:.1f} - {t['zone_top']:.1f}  "
          f"(height {t['zone_top']-t['zone_bottom']:.1f}pts)  "
          f"strength {t['zone_strength']:.2f}")
    if reasons:
        for r in reasons:
            print(f"  >> {r}")
    else:
        print(f"  >> No clear single cause — large SL combined with full adverse move")
    print(div)

# Summary
print()
print(f"  COMMON PATTERNS across {len(big)} large losses:")
print(div)

# Side
for s, g in big.groupby("side"):
    print(f"  {s.upper():<6}: {len(g)} trades  net ${g['pnl'].sum():+.2f}  "
          f"avg PnL ${g['pnl'].mean():+.2f}")

print()
print(f"  Avg SL distance : {big['sl_dist'].mean():.1f}pts  "
      f"(vs all losses: {df[df['pnl']<0]['sl_dist'].mean():.1f}pts)")
print(f"  Avg MFE         : {big['max_favour'].mean():.1f}pts")
print(f"  Avg MAE         : {big['max_adverse'].mean():.1f}pts")

print()
print(f"  H4 bias breakdown:")
for b, g in big.groupby("h4_bias"):
    print(f"    {b:<16}: {len(g)} trades  net ${g['pnl'].sum():+.2f}")

print()
print(f"  Signals breakdown:")
all_sigs = []
for s in big["signals"]: all_sigs.extend(str(s).split("|"))
for sig, cnt in Counter(all_sigs).most_common():
    print(f"    {sig:<25}: {cnt}")

print()
print(f"  Arrival type:")
for a, g in big.groupby("arrival_type"):
    print(f"    {a:<10}: {len(g)} trades  net ${g['pnl'].sum():+.2f}")

print()
print(f"  Confirmations:")
for c, g in big.groupby("confirmations"):
    print(f"    {int(c)} conf : {len(g)} trades  net ${g['pnl'].sum():+.2f}")

print()
print(f"  Zone height stats:")
big2 = big.copy()
big2["zone_h"] = big2["zone_top"] - big2["zone_bottom"]
print(f"    Min zone height : {big2['zone_h'].min():.1f}pts")
print(f"    Max zone height : {big2['zone_h'].max():.1f}pts")
print(f"    Avg zone height : {big2['zone_h'].mean():.1f}pts")
print(f"    (vs all trades avg: {(df['zone_top']-df['zone_bottom']).mean():.1f}pts)")

print()
print(f"  Could max_sl_pct filter help?")
print(f"  Current max_sl_pct setting: {getattr(__import__('trading.strategies.zz.ustec.strategy', fromlist=['MAX_SL_PCT']), 'MAX_SL_PCT', 'N/A')}")
pct_sl = (big["sl_dist"] / big["entry"] * 100)
print(f"  SL % of entry on big losses: min {pct_sl.min():.2f}%  max {pct_sl.max():.2f}%  avg {pct_sl.mean():.2f}%")
all_pct_sl = (df["sl_dist"] / df["entry"] * 100)
print(f"  SL % across ALL trades:      min {all_pct_sl.min():.2f}%  max {all_pct_sl.max():.2f}%  avg {all_pct_sl.mean():.2f}%")

print(hdiv)
print()
