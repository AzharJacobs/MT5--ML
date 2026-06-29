"""
FXGold backtest engine.

Iterates on H1 bars (entry TF).  For each bar:
  1. Build H4 + D1 zone windows (re-detected only when a new bar closes on
     that TF — cached to avoid O(n²) penalty on every H1 bar).
  2. Evaluate each strong zone via evaluate_entry (bias gate + pattern gate).
  3. Enter at the open of the next H1 bar; simulate forward to SL or TP.
  4. One trade at a time (skip_until blocks new entries while in position).

Contract size : 100  (XAUUSD standard)
Load start    : 180 days before trade_start for zone/bias warmup.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.zones import detect_zones
from trading.strategies.FXGold.entry import evaluate_entry, TouchTracker
from trading.shared.data_loader import get_connection

_CONTRACT = 100.0


def _load_ohlcv(db, table: str, tf: str, start: str, end: str) -> pd.DataFrame:
    query = (
        f"SELECT * FROM {table} WHERE timeframe = %s "
        f"AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC"
    )
    df = db.fetch_dataframe(query, (tf, start, end))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def run_backtest(
    cfg: FXGoldConfig,
    start: str = "2023-01-01",
    end:   str = "2025-12-31",
    cash:  float = 150.0,
    fixed_lot: float = 0.01,
) -> Tuple[List[dict], List[float]]:
    """
    Run the FXGold strategy backtest.

    Args:
        cfg:       Strategy configuration.
        start:     First date trades are recorded (YYYY-MM-DD).
        end:       Last date (inclusive).
        cash:      Starting equity.
        fixed_lot: Lot size per trade.

    Returns:
        (trades, equity_curve)
        trades       — list of trade dicts
        equity_curve — equity value after each trade (index 0 = start cash)
    """
    load_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    trade_start_ts = pd.Timestamp(start)

    # ── Load data ────────────────────────────────────────────────────────────
    db = get_connection()
    db.database = cfg.db_name
    db.connect()

    print(f"  Loading {cfg.tf_entry} data  {load_start} → {end} ...")
    df_h1 = _load_ohlcv(db, cfg.table, cfg.tf_entry, load_start, end)
    print(f"  Loading {cfg.tf_mid}  data  {load_start} → {end} ...")
    df_h4 = _load_ohlcv(db, cfg.table, cfg.tf_mid,   load_start, end)
    print(f"  Loading {cfg.tf_high}  data  {load_start} → {end} ...")
    df_d1 = _load_ohlcv(db, cfg.table, cfg.tf_high,  load_start, end)

    try:
        db.connection.close()
    except Exception:
        pass

    if df_h1.empty or df_h4.empty or df_d1.empty:
        print("  ERROR: one or more TFs returned no data — check DB and date range.")
        return [], []

    print(f"  {cfg.tf_entry} bars: {len(df_h1)} | {cfg.tf_mid} bars: {len(df_h4)} "
          f"| {cfg.tf_high} bars: {len(df_d1)}")

    # ── State ────────────────────────────────────────────────────────────────
    tracker        = TouchTracker()
    trades: list   = []
    equity         = cash
    equity_curve   = [cash]
    skip_until     = -1

    cached_h4_zones: list = []
    cached_d1_zones: list = []
    last_h4_ts: Optional[pd.Timestamp] = None
    last_d1_ts: Optional[pd.Timestamp] = None

    n      = len(df_h1)
    warmup = cfg.h1_window + 10

    min_h4_bars = 2 * cfg.fractal_window + 1
    min_d1_bars = 2 * cfg.fractal_window + 1

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(warmup, n - cfg.max_forward_bars):

        if i <= skip_until:
            continue

        ts_now = df_h1["timestamp"].iloc[i]

        if ts_now < trade_start_ts:
            continue

        # H4 and D1 slices up to current timestamp
        df_h4_w = df_h4[df_h4["timestamp"] <= ts_now].tail(cfg.h4_window).reset_index(drop=True)
        df_d1_w = df_d1[df_d1["timestamp"] <= ts_now].tail(cfg.d1_window).reset_index(drop=True)

        if len(df_h4_w) < min_h4_bars or len(df_d1_w) < min_d1_bars:
            continue

        # Zone detection — re-run only when higher-TF bar changes
        h4_last = df_h4_w["timestamp"].iloc[-1]
        d1_last = df_d1_w["timestamp"].iloc[-1]

        if h4_last != last_h4_ts:
            cached_h4_zones = detect_zones(df_h4_w, cfg.tf_mid, cfg)
            last_h4_ts = h4_last

        if d1_last != last_d1_ts:
            cached_d1_zones = detect_zones(df_d1_w, cfg.tf_high, cfg)
            last_d1_ts = d1_last

        all_zones = cached_h4_zones + cached_d1_zones
        if not all_zones:
            continue

        # H1 window for entry confirmation and bias
        h1_start = max(0, i - cfg.h1_window + 1)
        df_h1_w  = df_h1.iloc[h1_start: i + 1].reset_index(drop=True)

        bar_high  = float(df_h1["high"].iloc[i])
        bar_low   = float(df_h1["low"].iloc[i])
        bar_close = float(df_h1["close"].iloc[i])

        # Evaluate zones — strongest (highest score) first
        for zone in all_zones:
            signal = evaluate_entry(
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                zone=zone,
                tracker=tracker,
                df_entry_tf=df_h1_w,
                opposing_zones=all_zones,
                cfg=cfg,
                df_d1=df_d1_w,
                df_h4=df_h4_w,
                df_h1=df_h1_w,
            )

            if not signal.valid:
                continue

            # Enter at next H1 bar's open
            entry     = float(df_h1["open"].iloc[i + 1])
            sl        = signal.sl
            tp        = signal.tp
            direction = signal.direction

            # Simulate forward bar-by-bar
            outcome     = 0
            exit_price  = entry
            exit_bar    = i + cfg.max_forward_bars

            for j in range(i + 1, min(i + 1 + cfg.max_forward_bars, n)):
                fh = float(df_h1["high"].iloc[j])
                fl = float(df_h1["low"].iloc[j])
                if direction == "buy":
                    if fh >= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                    if fl <= sl:  outcome = -1; exit_price = sl; exit_bar = j; break
                else:
                    if fl <= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                    if fh >= sl:  outcome = -1; exit_price = sl; exit_bar = j; break

            if direction == "buy":
                pnl = (exit_price - entry) * fixed_lot * _CONTRACT
            else:
                pnl = (entry - exit_price) * fixed_lot * _CONTRACT

            equity += pnl
            equity_curve.append(equity)
            skip_until = exit_bar

            exit_ts = df_h1["timestamp"].iloc[min(exit_bar, n - 1)]

            trades.append({
                "date":        ts_now,
                "exit_date":   exit_ts,
                "side":        direction,
                "entry":       round(entry, 2),
                "sl":          round(sl, 2),
                "tp":          round(tp, 2),
                "exit":        round(exit_price, 2),
                "outcome":     outcome,
                "lot":         fixed_lot,
                "pnl":         round(pnl, 2),
                "equity":      round(equity, 2),
                "pattern":     signal.pattern or "",
                "tp_mode":     signal.tp_mode or "",
                "zone_tf":     zone.origin_tf,
                "zone_kind":   zone.kind,
                "zone_top":    round(zone.top, 2),
                "zone_bottom": round(zone.bottom, 2),
                "zone_score":  round(zone.score, 2),
            })

            break  # one trade at a time; strongest zone wins

    return trades, equity_curve
