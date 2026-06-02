"""
diagnose_2024.py — Two diagnostic tables on the 2024 backtest.

BREAKDOWN 1: Expectancy by zone quality grade (A/B/C/D buckets)
BREAKDOWN 2: htf_4h_bias agreement with EMA200 trend direction on 4H
"""

import os, sys
sys.path.insert(0, r"f:\MT5- ML")
os.chdir(r"f:\MT5- ML")

import numpy as np
import pandas as pd
from data.loader import get_connection
from data.feature_engineer import _atr, _ema
from backtest.engine import run_backtest

# ── Run 2024 backtest with combined settings ──────────────────────────────────
print("Running 2024 backtest (combined settings — NEW htf_4h_bias)...")
res = run_backtest(
    timeframe="15min",
    start_date="2024-01-01",
    end_date="2024-12-31",
    cash=50.0,
    stake=0.15,
    use_pct_stake=True,
    confidence=0.52,      # uses saved threshold
    commission=0.0,
    include_london_ny=False,
    be_r=999.0,
    no_trail=True,
    require_confirmation=True,
    trend_only=True,
)

trades = list(res.trade_log)
print(f"Captured {len(trades)} trade-log entries\n")


# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWN 1 — Expectancy by zone quality grade
# ─────────────────────────────────────────────────────────────────────────────

def zq_grade(zq):
    if pd.isna(zq) or zq is None:
        return "? (no zq)"
    zq = float(zq)
    if zq >= 3.5: return "A (>=3.5)"
    if zq >= 3.0: return "B (3.0-3.49)"
    if zq >= 2.0: return "C (2.0-2.99)"
    return               "D (<2.0)"

rows = []
for t in trades:
    rows.append({
        "grade_bucket": zq_grade(t.get("zone_quality")),
        "zq":           t.get("zone_quality", float("nan")),
        "pnl":          float(t["pnl"]),
        "side":         t.get("side", ""),
        "close_reason": t.get("close_reason", ""),
        "entry_date":   t.get("entry_date"),
    })

df = pd.DataFrame(rows)

GRADE_ORDER = ["A (>=3.5)", "B (3.0-3.49)", "C (2.0-2.99)", "D (<2.0)", "? (no zq)"]

W = 90
print("=" * W)
print("  BREAKDOWN 1 — Expectancy by zone quality grade")
print("=" * W)
print(f"\n  {'Grade':<14} {'Trades':>7} {'WR%':>7} {'AvgWin':>9} {'AvgLoss':>9}"
      f" {'TP%':>6} {'TotalPnL':>10} {'Expect/trade':>14}")
print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*6} {'-'*10} {'-'*14}")

for g in GRADE_ORDER:
    sub = df[df["grade_bucket"] == g]
    if len(sub) == 0:
        continue
    n     = len(sub)
    wins  = sub[sub["pnl"] > 0]
    losses = sub[sub["pnl"] <= 0]
    wr    = len(wins) / n
    lr    = 1 - wr
    avg_w = wins["pnl"].mean()  if len(wins)   else 0.0
    avg_l = losses["pnl"].mean() if len(losses) else 0.0
    tp_pct = (sub["close_reason"] == "tp").sum() / n * 100
    total = sub["pnl"].sum()
    expect = avg_w * wr + avg_l * lr    # avg_l is already negative
    print(f"  {g:<14} {n:>7,} {wr*100:>6.1f}% {avg_w:>+9.4f} {avg_l:>+9.4f}"
          f" {tp_pct:>5.1f}% {total:>+10.4f} {expect:>+14.5f}")

print(f"\n  Note: higher expectancy/trade and TP% in A/B vs C/D = zone-quality has edge.")
print(f"  Flat or inverted = zone-quality score is noise.")


# ─────────────────────────────────────────────────────────────────────────────
# BREAKDOWN 2 — htf_4h_bias vs EMA200 trend alignment
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * W)
print("  BREAKDOWN 2 — htf_4h_bias vs 4H EMA200 trend direction")
print("=" * W)

# Load ALL 4H data (need pre-2024 bars for EMA200 warmup)
db = get_connection()
db.connect()
q4h = "SELECT timestamp, open, high, low, close FROM xauusd_ohlcv WHERE timeframe='4H' ORDER BY timestamp ASC"
h4 = db.fetch_dataframe(q4h)
db.disconnect()

h4["timestamp"] = pd.to_datetime(h4["timestamp"])
h4 = h4.sort_values("timestamp").reset_index(drop=True)

# Compute EMA200 on 4H close
h4["ema200"] = _ema(h4["close"], 200)

# Compute htf_4h_bias using NEW EMA200-based logic (matches updated add_htf_context)
if len(h4) >= 200:
    h4_ema200 = _ema(h4["close"], 200)
    h4_slope  = h4_ema200.diff()
    h4_bias_raw = np.where(
        (h4["close"].values > h4_ema200.values) & (h4_slope.values > 0),  1.0,
        np.where(
            (h4["close"].values < h4_ema200.values) & (h4_slope.values < 0), -1.0, 0.0
        )
    )
    h4["htf_4h_bias"] = h4_bias_raw.astype(float)
else:
    h4["htf_4h_bias"] = 0.0

# EMA200 trend: +1 if close > EMA200, -1 if below, 0 if EMA not ready
h4["ema200_trend"] = np.where(
    h4["ema200"].isna(), 0,
    np.where(h4["close"] > h4["ema200"], 1, -1)
)

# Filter to 2024 only for analysis
h4_2024 = h4[
    (h4["timestamp"] >= "2024-01-01") &
    (h4["timestamp"] <  "2025-01-01") &
    (h4["ema200_trend"] != 0)
].copy()

n_bars = len(h4_2024)

# Agreement: htf_4h_bias and ema200_trend have the same sign (both >0 or both <0)
# Neutral (bias=0) is counted separately
bias_nonzero = h4_2024[h4_2024["htf_4h_bias"] != 0]
n_nonzero    = len(bias_nonzero)

agree    = (bias_nonzero["htf_4h_bias"] * bias_nonzero["ema200_trend"] > 0).sum()
disagree = (bias_nonzero["htf_4h_bias"] * bias_nonzero["ema200_trend"] < 0).sum()
neutral  = (h4_2024["htf_4h_bias"] == 0).sum()

print(f"\n  4H bars in 2024 (EMA200 ready): {n_bars:,}")
print(f"  htf_4h_bias = 0 (neutral/no impulse): {neutral:,}  ({neutral/n_bars*100:.1f}%)")
print(f"  htf_4h_bias != 0 bars:               {n_nonzero:,}  ({n_nonzero/n_bars*100:.1f}%)")
print(f"\n  Of those {n_nonzero:,} directional bars:")
print(f"    Agrees with EMA200:    {agree:>5,}  ({agree/max(n_nonzero,1)*100:.1f}%)")
print(f"    Disagrees with EMA200: {disagree:>5,}  ({disagree/max(n_nonzero,1)*100:.1f}%)")

# Directional breakdown
ema_up   = h4_2024[h4_2024["ema200_trend"] == 1]
ema_down = h4_2024[h4_2024["ema200_trend"] == -1]
print(f"\n  When EMA200 = UPTREND  ({len(ema_up):,} bars):")
print(f"    htf_4h_bias = +1 (agree bullish): {(ema_up['htf_4h_bias']==1).sum():,}  "
      f"({(ema_up['htf_4h_bias']==1).sum()/max(len(ema_up),1)*100:.1f}%)")
print(f"    htf_4h_bias =  0 (neutral):       {(ema_up['htf_4h_bias']==0).sum():,}  "
      f"({(ema_up['htf_4h_bias']==0).sum()/max(len(ema_up),1)*100:.1f}%)")
print(f"    htf_4h_bias = -1 (disagree):      {(ema_up['htf_4h_bias']==-1).sum():,}  "
      f"({(ema_up['htf_4h_bias']==-1).sum()/max(len(ema_up),1)*100:.1f}%)")

print(f"\n  When EMA200 = DOWNTREND ({len(ema_down):,} bars):")
print(f"    htf_4h_bias = -1 (agree bearish): {(ema_down['htf_4h_bias']==-1).sum():,}  "
      f"({(ema_down['htf_4h_bias']==-1).sum()/max(len(ema_down),1)*100:.1f}%)")
print(f"    htf_4h_bias =  0 (neutral):       {(ema_down['htf_4h_bias']==0).sum():,}  "
      f"({(ema_down['htf_4h_bias']==0).sum()/max(len(ema_down),1)*100:.1f}%)")
print(f"    htf_4h_bias = +1 (disagree):      {(ema_down['htf_4h_bias']==1).sum():,}  "
      f"({(ema_down['htf_4h_bias']==1).sum()/max(len(ema_down),1)*100:.1f}%)")

# Losing sell trades: what % were in EMA200 uptrend?
sell_losses = [t for t in trades if t.get("side") == "sell" and float(t["pnl"]) <= 0]
print(f"\n  Losing SELL trades in 2024: {len(sell_losses):,}")

if sell_losses and h4_2024 is not None:
    # For each losing sell, find the nearest prior 4H bar's ema200_trend
    h4_lookup = h4_2024[["timestamp", "ema200_trend", "htf_4h_bias"]].copy()
    _ts = h4_lookup["timestamp"]
    h4_lookup["timestamp"] = _ts.dt.tz_convert(None) if _ts.dt.tz is not None else _ts
    h4_lookup = h4_lookup.sort_values("timestamp")

    in_uptrend  = 0
    in_downtrend = 0
    no_match    = 0

    for t in sell_losses:
        ed = t.get("entry_date")
        if ed is None:
            no_match += 1
            continue
        ed = pd.Timestamp(ed)
        # Find the last 4H bar at or before the entry date
        prior = h4_lookup[h4_lookup["timestamp"] <= ed]
        if prior.empty:
            no_match += 1
            continue
        row4h = prior.iloc[-1]
        if row4h["ema200_trend"] == 1:
            in_uptrend += 1
        elif row4h["ema200_trend"] == -1:
            in_downtrend += 1
        else:
            no_match += 1

    total_matched = in_uptrend + in_downtrend
    print(f"    In EMA200 uptrend  (bearish bias disagrees with trend): "
          f"{in_uptrend:>4,}  ({in_uptrend/max(total_matched,1)*100:.1f}%)")
    print(f"    In EMA200 downtrend (bearish bias agrees with trend):   "
          f"{in_downtrend:>4,}  ({in_downtrend/max(total_matched,1)*100:.1f}%)")
    print(f"    No EMA200 match:                                        "
          f"{no_match:>4,}")

    # Same for winning sells (for comparison)
    sell_wins = [t for t in trades if t.get("side") == "sell" and float(t["pnl"]) > 0]
    win_up = win_down = 0
    for t in sell_wins:
        ed = t.get("entry_date")
        if ed is None: continue
        ed = pd.Timestamp(ed)
        prior = h4_lookup[h4_lookup["timestamp"] <= ed]
        if prior.empty: continue
        row4h = prior.iloc[-1]
        if row4h["ema200_trend"] == 1: win_up += 1
        elif row4h["ema200_trend"] == -1: win_down += 1

    print(f"\n  Winning SELL trades for comparison ({len(sell_wins):,}):")
    wm = win_up + win_down
    print(f"    In EMA200 uptrend:   {win_up:>4,}  ({win_up/max(wm,1)*100:.1f}%)")
    print(f"    In EMA200 downtrend: {win_down:>4,}  ({win_down/max(wm,1)*100:.1f}%)")
