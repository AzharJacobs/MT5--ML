import os, sys
sys.path.insert(0, r"f:\MT5- ML")
os.chdir(r"f:\MT5- ML")
from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
import pandas as pd
from data.feature_engineer import build_features

mt5.initialize(login=435541691, password="Mt5_ML#1", server="Exness-MT5Trial9")
acc = mt5.account_info()
print(f"Account balance: {acc.balance}  Equity: {acc.equity}")

bars = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M15, 0, 350)
df = pd.DataFrame(bars)
df["timestamp"] = pd.to_datetime(df["time"], unit="s")
df.rename(columns={"tick_volume": "volume"}, inplace=True)

# Also fetch H1 and H4 for HTF context
bars_h1 = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_H1, 0, 200)
df_h1 = pd.DataFrame(bars_h1)
df_h1["timestamp"] = pd.to_datetime(df_h1["time"], unit="s")
df_h1.rename(columns={"tick_volume": "volume"}, inplace=True)

bars_h4 = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_H4, 0, 100)
df_h4 = pd.DataFrame(bars_h4)
df_h4["timestamp"] = pd.to_datetime(df_h4["time"], unit="s")
df_h4.rename(columns={"tick_volume": "volume"}, inplace=True)

def prep(d):
    d = d.copy()
    d["hour"]        = d["timestamp"].dt.hour
    d["day_of_week"] = d["timestamp"].dt.dayofweek
    d["month"]       = d["timestamp"].dt.month
    d["year"]        = d["timestamp"].dt.year
    body_top = d[["open","close"]].max(axis=1)
    body_bot = d[["open","close"]].min(axis=1)
    d["candle_size"] = d["high"] - d["low"]
    d["body_size"]   = (d["close"] - d["open"]).abs()
    d["wick_upper"]  = d["high"] - body_top
    d["wick_lower"]  = body_bot  - d["low"]
    return d

df    = prep(df)
df_h1 = prep(df_h1)
df_h4 = prep(df_h4)
feat = build_features(df, h1_df=df_h1, h4_df=df_h4)
last = feat.iloc[-1]
prev = feat.iloc[-2]

print(f"\n=== CURRENT BAR (latest closed 15min) ===")
print(f"Time:           {last['timestamp']}")
print(f"Close:          {last['close']:.3f}")
print(f"in_demand_zone: {last.get('in_demand_zone')}")
print(f"in_supply_zone: {last.get('in_supply_zone')}")
print(f"between_zones:  {last.get('between_zones')}")
print(f"")
print(f"=== LTF ZONE LEVELS ===")
print(f"Demand zone:  {last.get('demand_zone_bottom'):.3f} - {last.get('demand_zone_top'):.3f}")
print(f"Supply zone:  {last.get('supply_zone_bottom'):.3f} - {last.get('supply_zone_top'):.3f}")
print(f"")
print(f"=== HTF ZONE LEVELS (1H) ===")
htf_db = last.get("htf_demand_zone_bottom")
htf_dt = last.get("htf_demand_zone_top")
htf_sb = last.get("htf_supply_zone_bottom")
htf_st = last.get("htf_supply_zone_top")
print(f"HTF Demand:  {htf_db} - {htf_dt}")
print(f"HTF Supply:  {htf_sb} - {htf_st}")
print(f"")
print(f"=== SIGNAL QUALITY ===")
print(f"zone_quality: {last.get('zone_quality')}")
print(f"htf_4h_bias:  {last.get('htf_4h_bias')}")
print(f"htf_1h_bias:  {last.get('htf_1h_bias')}")
print(f"atr_14:       {last.get('atr_14'):.4f}")
print(f"volume_ratio: {last.get('volume_ratio'):.3f}")
print(f"momentum:     {last.get('momentum_5')}")
print(f"bos_bullish:             {last.get('bos_bullish')}")
print(f"bos_bearish:             {last.get('bos_bearish')}")
print(f"choch_bullish:           {last.get('choch_bullish')}")
print(f"choch_bearish:           {last.get('choch_bearish')}")
print(f"buy_confirmation_score:  {last.get('buy_confirmation_score')}")
print(f"sell_confirmation_score: {last.get('sell_confirmation_score')}")

# Also show last 10 bars with zone status
print(f"\n=== LAST 10 BARS — ZONE MAP ===")
cols = ["timestamp","close","in_demand_zone","in_supply_zone","between_zones","demand_zone_bottom","demand_zone_top","supply_zone_bottom","supply_zone_top","zone_quality","bos_bullish","bos_bearish","choch_bullish","choch_bearish","buy_confirmation_score","sell_confirmation_score"]
available = [c for c in cols if c in feat.columns]
print(feat[available].tail(10).to_string(index=False))

mt5.shutdown()
