"""
Quick diagnostic: show current bar's full zone picture via MT5 live data.
Usage: python scripts/diag_live_now.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from trading.strategies.zz.ustec.strategy import (
    SYMBOL, make_configs, H4_WINDOW, M15_WINDOW,
)
from trading.strategies.zz.core.timeframe_structure import analyse_timeframes
from trading.strategies.zz.core.confirmations import check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import setup_from_analysis

login    = int(os.environ.get("MT5_LOGIN", 0))
password = os.environ.get("MT5_PASSWORD", "")
server   = os.environ.get("MT5_SERVER", "")

assert mt5.initialize(), "MT5 init failed"
assert mt5.login(login, password, server), "MT5 login failed"

TF_MAP = {"15min": mt5.TIMEFRAME_M15, "4H": mt5.TIMEFRAME_H4}

def load(tf, n):
    rates = mt5.copy_rates_from_pos(SYMBOL, TF_MAP[tf], 1, n)
    df = pd.DataFrame(rates)
    df["timestamp"] = (
        pd.to_datetime(df["time"], unit="s")
        .dt.tz_localize("Etc/GMT-3").dt.tz_convert("UTC").dt.tz_localize(None)
    )
    for c in ("open","high","low","close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)

df_15m = load("15min", M15_WINDOW + 10).tail(M15_WINDOW).reset_index(drop=True)
df_4h  = load("4H",    H4_WINDOW  + 10).tail(H4_WINDOW).reset_index(drop=True)
df_4h["ema50"]  = df_4h["close"].ewm(span=50,  adjust=False).mean()
df_4h["ema200"] = df_4h["close"].ewm(span=200, adjust=False).mean()

tf_cfg, conf_cfg, setup_cfg = make_configs()
tf_result = analyse_timeframes(df_4h, df_15m, cfg=tf_cfg, h4_up_to_bar=len(df_4h)-1)

now = datetime.now(timezone.utc)
price = float(df_15m["close"].iloc[-1])
last_ts = df_15m["timestamp"].iloc[-1]

print(f"\n{'='*60}")
print(f"  USTEC Live Diagnostic — {now.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Last 15M bar: {last_ts}  |  Close: {price:.2f}")
print(f"{'='*60}")

bias = tf_result["h4_bias"]
print(f"\nH4 Bias        : {bias}")
print(f"Signal         : {tf_result['signal']}")
print(f"Reason         : {tf_result['reason']}")

h4_zones = tf_result["h4_zones"]
print(f"\nAll H4 zones ({len(h4_zones)} total):")
for z in sorted(h4_zones, key=lambda z: z.bottom):
    dist = z.mid - price
    marker = " ← ACTIVE" if tf_result.get("active_zone") is z else ""
    print(f"  {z.kind:8s} {z.bottom:.2f}-{z.top:.2f}  str={z.strength:.1f}  "
          f"{'fresh' if z.fresh else 'tapped':6s}  dist={dist:+.1f}{marker}")

az = tf_result.get("active_zone")
if az:
    print(f"\nActive zone    : {az.kind} {az.bottom:.2f}–{az.top:.2f}  str={az.strength:.1f}")
    direction = tf_result["direction"]
    conf = check_confirmations_at_last_bar(df_15m, az, direction, conf_cfg)
    print(f"Confirmations  : count={conf.count}  signals={conf.signals}  confirmed={conf.confirmed}")

    setup = setup_from_analysis(tf_result, price, setup_cfg)
    print(f"\nSetup valid    : {setup.valid}")
    if setup.valid:
        print(f"  Entry={setup.entry:.2f}  SL={setup.sl:.2f}  TP={setup.tp:.2f}  RR={setup.rr:.2f}")
        print(f"  Risk={setup.risk:.1f} pts  Reward={setup.reward:.1f} pts")
        print(f"  TP zone: {setup.tp_zone}")
    else:
        print(f"  Reason: {setup.reason}")
        # Show geometry manually
        if az:
            if direction == "buy":
                sl = az.bottom
                print(f"  Manual geometry: entry={price:.2f}  sl={sl:.2f}  risk={price-sl:.1f} pts")
            else:
                sl = az.top
                print(f"  Manual geometry: entry={price:.2f}  sl={sl:.2f}  risk={sl-price:.1f} pts")
            # Find TP zones manually
            if direction == "buy":
                tp_cands = [z for z in h4_zones if z.kind=="supply" and z.bottom > price]
                print(f"  Supply zones above price ({price:.2f}):")
                for z in sorted(tp_cands, key=lambda z: z.bottom)[:5]:
                    rr = (z.bottom - price) / (price - az.bottom) if (price - az.bottom) > 0 else 0
                    print(f"    {z.bottom:.2f}–{z.top:.2f}  dist={z.bottom-price:.1f} pts  RR≈{rr:.2f}  {'fresh' if z.fresh else 'tapped'}")
            else:
                tp_cands = [z for z in h4_zones if z.kind=="demand" and z.top < price]
                print(f"  Demand zones below price ({price:.2f}):")
                for z in sorted(tp_cands, key=lambda z: -z.top)[:5]:
                    rr = (price - z.top) / (az.top - price) if (az.top - price) > 0 else 0
                    print(f"    {z.bottom:.2f}–{z.top:.2f}  dist={price-z.top:.1f} pts  RR≈{rr:.2f}  {'fresh' if z.fresh else 'tapped'}")

print(f"\n{'='*60}\n")
mt5.shutdown()
