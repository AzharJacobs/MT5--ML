#!/usr/bin/env python3
"""
Diagnosis: why did H4 bias read neutral_up/bullish on buy entries in 2022 bear?
Prints each trade's H4 swing label sequence at entry time.
NO code changes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from trading.strategies.zz.ustec.engine import run_backtest, H4_WINDOW
from trading.strategies.zz.core.timeframe_structure import detect_h4_bias
from trading.strategies.zz.core.swing_structure import (
    detect_swings as _detect_swings,
    label_structure as _label_structure,
)
from trading.shared.mt5_loader import fetch_ohlcv, disconnect

# ── 1. Run backtest to get the 11 trades ─────────────────────────────────────
print("Fetching data and running backtest …")
result = run_backtest(
    start="2021-01-01",
    end="2023-01-01",
    cash=150.0,
    symbol="ustech",
    min_confirmations=2,
    data_source="mt5",
    max_rr=5.0,
    silent=True,
)
if not result:
    print("No trades returned.")
    sys.exit(1)

_, df_trades = result

# ── 2. Pull full 4H data for the period ──────────────────────────────────────
df_4h = fetch_ohlcv("ustech", "4H", "2020-07-01", "2023-01-01", silent=True)
disconnect()
df_4h["timestamp"] = pd.to_datetime(df_4h["timestamp"])
df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)
print(f"4H bars loaded: {len(df_4h)}  ({df_4h['timestamp'].iloc[0]} → {df_4h['timestamp'].iloc[-1]})\n")

DIV = "─" * 80

# ── 3. For each trade, replay H4 swing structure at entry time ────────────────
print(DIV)
print(f"  {'Date':<22} {'Side':<5} {'H4 Bias':<16} {'Last 6 swing labels (oldest→newest)'}")
print(DIV)

for _, tr in df_trades.sort_values("date").iterrows():
    ts = pd.to_datetime(tr["date"])
    win = df_4h[df_4h["timestamp"] <= ts].tail(H4_WINDOW).reset_index(drop=True)

    bias = detect_h4_bias(win, left=2, right=2)

    # Replay swing labels
    swings = _detect_swings(win, left=2, right=2)
    labeled = _label_structure(swings)
    # last 8 labels
    recent = labeled[-8:] if len(labeled) >= 8 else labeled
    label_seq = "  ".join(
        f"{lbl}@{win['timestamp'].iloc[idx].strftime('%Y-%m-%d') if idx < len(win) else '?'}"
        for idx, price, lbl in recent
    )

    side_flag = " ←BUY-IN-BEAR" if tr["side"] == "buy" and bias in ("neutral_up", "bullish") else ""
    print(f"  {str(tr['date']):<22} {tr['side']:<5} {bias:<16} {label_seq}{side_flag}")

print(DIV)

# ── 4. Detailed drill-down on buy trades with non-bearish bias ────────────────
buys_wrong = df_trades[(df_trades["side"] == "buy") & (df_trades["h4_bias"].isin(["neutral_up", "bullish"]))]

print(f"\n  BUY entries with non-bearish bias: {len(buys_wrong)}")
print("  Detailed swing sequence for each:\n")

for _, tr in buys_wrong.sort_values("date").iterrows():
    ts = pd.to_datetime(tr["date"])
    win = df_4h[df_4h["timestamp"] <= ts].tail(H4_WINDOW).reset_index(drop=True)

    swings  = _detect_swings(win, left=2, right=2)
    labeled = _label_structure(swings)

    sh = [(i, p, l) for i, p, l in labeled if l in ("HH", "LH")]
    sl = [(i, p, l) for i, p, l in labeled if l in ("HL", "LL")]

    last_sh = sh[-1] if sh else None
    last_sl = sl[-1] if sl else None
    prev_sh = sh[-2] if len(sh) >= 2 else None
    prev_sl = sl[-2] if len(sl) >= 2 else None

    print(f"  Trade: {tr['date']}  side={tr['side']}  bias={tr['h4_bias']}")
    print(f"    Last swing HIGH: {last_sh[2] if last_sh else 'none'} "
          f"price={last_sh[1]:.1f if last_sh else 0:.1f} "
          f"bar={last_sh[0] if last_sh else '?'}  "
          f"date={win['timestamp'].iloc[last_sh[0]].strftime('%Y-%m-%d') if last_sh and last_sh[0] < len(win) else '?'}")
    print(f"    Prev swing HIGH: {prev_sh[2] if prev_sh else 'none'} "
          f"price={prev_sh[1]:.1f if prev_sh else 0:.1f}  "
          f"date={win['timestamp'].iloc[prev_sh[0]].strftime('%Y-%m-%d') if prev_sh and prev_sh[0] < len(win) else '?'}")
    print(f"    Last swing LOW : {last_sl[2] if last_sl else 'none'} "
          f"price={last_sl[1]:.1f if last_sl else 0:.1f}  "
          f"date={win['timestamp'].iloc[last_sl[0]].strftime('%Y-%m-%d') if last_sl and last_sl[0] < len(win) else '?'}")
    print(f"    Prev swing LOW : {prev_sl[2] if prev_sl else 'none'} "
          f"price={prev_sl[1]:.1f if prev_sl else 0:.1f}  "
          f"date={win['timestamp'].iloc[prev_sl[0]].strftime('%Y-%m-%d') if prev_sl and prev_sl[0] < len(win) else '?'}")

    # What the 150-bar window is actually covering
    w_start = win["timestamp"].iloc[0]
    w_end   = win["timestamp"].iloc[-1]
    w_range = float(win["high"].max() - win["low"].min())
    print(f"    H4 window: {w_start.strftime('%Y-%m-%d')} → {w_end.strftime('%Y-%m-%d')}  "
          f"range={w_range:.0f} pts  bars={len(win)}")
    print()
