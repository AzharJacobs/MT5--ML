"""
MT5 data loader — fetches OHLCV directly from a running MT5 terminal.

MT5 must be open and logged in before calling any function here.
Credentials are read from .env (MT5_LOGIN / MT5_PASSWORD / MT5_SERVER).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    raise ImportError("MetaTrader5 package not installed. Run: pip install MetaTrader5")

# Timeframe string → MT5 constant
_TF_MAP = {
    "1min":  mt5.TIMEFRAME_M1,
    "5min":  mt5.TIMEFRAME_M5,
    "15min": mt5.TIMEFRAME_M15,
    "30min": mt5.TIMEFRAME_M30,
    "1H":    mt5.TIMEFRAME_H1,
    "4H":    mt5.TIMEFRAME_H4,
    "1D":    mt5.TIMEFRAME_D1,
}

# Symbol name in MT5 for each strategy symbol key
MT5_SYMBOL_MAP = {
    "ustech": "USTECm",
}

_connected = False


def connect(silent: bool = False) -> bool:
    global _connected
    if _connected:
        return True

    login    = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server   = os.getenv("MT5_SERVER", "")

    if not mt5.initialize():
        print(f"[MT5] initialize() failed — is MT5 terminal open? error: {mt5.last_error()}")
        return False

    if not mt5.login(login, password=password, server=server):
        print(f"[MT5] login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    if not silent:
        print(f"[OK] Connected to MT5 — {info.name}  |  {info.server}  |  balance={info.balance}")

    _connected = True
    return True


def disconnect():
    global _connected
    mt5.shutdown()
    _connected = False


def fetch_ohlcv(
    symbol_key: str,
    timeframe: str,
    start: str,
    end: str,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from MT5 and return a DataFrame matching the engine's
    expected schema: timestamp, open, high, low, close, volume.

    symbol_key: strategy key e.g. "ustech"
    timeframe:  engine string e.g. "15min", "4H"
    start/end:  "YYYY-MM-DD" strings (UTC assumed)
    """
    if not connect(silent=silent):
        return pd.DataFrame()

    mt5_symbol = MT5_SYMBOL_MAP.get(symbol_key.lower())
    if not mt5_symbol:
        print(f"[MT5] No symbol mapping for '{symbol_key}'. Add it to MT5_SYMBOL_MAP.")
        return pd.DataFrame()

    tf = _TF_MAP.get(timeframe)
    if tf is None:
        print(f"[MT5] Unknown timeframe '{timeframe}'. Options: {list(_TF_MAP)}")
        return pd.DataFrame()

    date_from = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    rates = mt5.copy_rates_range(mt5_symbol, tf, date_from, date_to)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        if not silent:
            print(f"[MT5] No data for {mt5_symbol} {timeframe} {start}→{end}  error={err}")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if not silent:
        print(f"[MT5] {mt5_symbol} {timeframe}: {len(df)} bars  "
              f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")

    return df
