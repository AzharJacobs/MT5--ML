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
from trading.strategies.FXGold.zones import detect_zones, is_rejection_touch, Zone
from trading.strategies.FXGold.entry import evaluate_entry, TouchTracker
from trading.strategies.FXGold.bias import get_aligned_bias, get_bias, d1_bias
from trading.shared.data_loader import get_connection
from trading.shared.mt5_loader import fetch_ohlcv as _mt5_fetch, disconnect as _mt5_disconnect

_CONTRACT = 100.0

# Bar-duration lookup for the timeframe strings used by tf_high/tf_mid/tf_entry
# across both config presets.
_TF_DURATIONS = {
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "1H":    pd.Timedelta(hours=1),
    "4H":    pd.Timedelta(hours=4),
    "1D":    pd.Timedelta(days=1),
}


def closed_bars(df: pd.DataFrame, tf: str, ts_now: pd.Timestamp) -> pd.DataFrame:
    """
    Return only the bars in `df` that have FULLY CLOSED by ts_now.

    MT5 (and this project's DB mirror, sourced from MT5) stamp each bar with
    its OPEN time — verified empirically, not assumed: aggregating 1H bars
    into 4H buckets and comparing OHLC against the real 4H bars, bucket-START
    alignment matched every complete bucket checked (45/45) while bucket-END
    alignment matched none (0/58), on both the MT5 feed and the DB mirror.

    A bar at timestamp T therefore only finishes forming at T + tf_duration.
    A naive `df["timestamp"] <= ts_now` filter leaks the still-forming bar
    in one bar early — this excludes it correctly.
    """
    duration = _TF_DURATIONS.get(tf)
    if duration is None:
        raise ValueError(f"Unknown timeframe '{tf}' — add its duration to _TF_DURATIONS.")
    return df[df["timestamp"] + duration <= ts_now]


def _reset_touches_on_flip_or_death(
    tracker: TouchTracker, old_zones: List[Zone], new_zones: List[Zone],
) -> None:
    """
    Reset the live touch count for any zone that flipped kind or dropped out
    of the strong/non-dead list (detect_zones only returns strong, non-dead
    zones) between two consecutive detect_zones() calls on the same higher
    TF. Matches the PDF: after a flip, the retest count restarts from zero.

    zone_key is (bottom, top) only — it does NOT change when a zone flips
    kind, so without this the tracker would silently carry a support zone's
    touch count over to the resistance zone it flipped into.
    """
    new_kind_by_key = {z.zone_key: z.kind for z in new_zones}
    for old_zone in old_zones:
        new_kind = new_kind_by_key.get(old_zone.zone_key)
        if new_kind is None or new_kind != old_zone.kind:
            tracker.reset(old_zone)


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
    data_source: str = "db",   # "db" = PostgreSQL  |  "mt5" = live MT5 terminal
    load_start: Optional[str] = None,  # override the default 180-day warmup start (YYYY-MM-DD)
) -> Tuple[List[dict], List[float], pd.DataFrame]:
    """
    Run the FXGold strategy backtest.

    Args:
        cfg:         Strategy configuration.
        start:       First date trades are recorded (YYYY-MM-DD).
        end:         Last date (inclusive).
        cash:        Starting equity.
        fixed_lot:   Lot size per trade.
        data_source: "db" (PostgreSQL, default) or "mt5" (live MT5 terminal —
                     MT5 must be open/logged in; symbol XAUUSDm).
        load_start:  Data is loaded from this date for zone/bias warmup, so
                     `start` itself trades clean. Defaults to 180 days before
                     `start`; pass an explicit date to override.

    Returns:
        (trades, equity_curve, df_h1)
        trades       — list of trade dicts
        equity_curve — equity value after each trade (index 0 = start cash)
        df_h1        — entry-TF OHLCV frame used for the run (for charting)
    """
    if load_start is None:
        load_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
    trade_start_ts = pd.Timestamp(start)

    # ── Load data ────────────────────────────────────────────────────────────
    if data_source == "mt5":
        print(f"  Loading {cfg.tf_entry} data from MT5  {load_start} → {end} ...")
        df_h1 = _mt5_fetch("xauusd", cfg.tf_entry, load_start, end)
        print(f"  Loading {cfg.tf_mid}  data from MT5  {load_start} → {end} ...")
        df_h4 = _mt5_fetch("xauusd", cfg.tf_mid,   load_start, end)
        print(f"  Loading {cfg.tf_high}  data from MT5  {load_start} → {end} ...")
        df_d1 = _mt5_fetch("xauusd", cfg.tf_high,  load_start, end)
        _mt5_disconnect()
    else:
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
        src = "MT5 terminal" if data_source == "mt5" else "DB"
        print(f"  ERROR: one or more TFs returned no data from {src} — check connection and date range.")
        return [], [], pd.DataFrame()

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
    cached_bias: Optional[str] = None
    cached_d1_bias: Optional[str] = None
    cached_h4_bias: Optional[str] = None

    n      = len(df_h1)
    warmup = cfg.h1_window + 10

    min_h4_bars = 2 * cfg.fractal_window + 1
    min_d1_bars = 2 * cfg.fractal_window + 1

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(warmup, n - cfg.max_forward_bars):

        ts_now = df_h1["timestamp"].iloc[i]

        if ts_now < trade_start_ts:
            continue

        # H4 and D1 slices of bars that have FULLY CLOSED by ts_now (see
        # closed_bars docstring — a naive timestamp <= ts_now filter would
        # leak in the still-forming higher-TF bar, since bars are open-stamped).
        #
        # This block, and the touch-recording pass below, run on EVERY bar —
        # including bars we're holding a position through (skip_until is
        # only checked further down). Previously the whole loop body was
        # skipped while in a position, so real retests against a zone during
        # an open trade went uncounted (Task 5 fix).
        df_h4_w = closed_bars(df_h4, cfg.tf_mid,  ts_now).tail(cfg.h4_window).reset_index(drop=True)
        df_d1_w = closed_bars(df_d1, cfg.tf_high, ts_now).tail(cfg.d1_window).reset_index(drop=True)

        if len(df_h4_w) < min_h4_bars or len(df_d1_w) < min_d1_bars:
            continue

        # Zone detection — re-run only when higher-TF bar changes. Any zone
        # that flipped kind or dropped out of the strong/non-dead list since
        # the last detection has its live touch count reset (PDF: a flip
        # restarts the retest count).
        h4_last = df_h4_w["timestamp"].iloc[-1]
        d1_last = df_d1_w["timestamp"].iloc[-1]

        h4_changed = h4_last != last_h4_ts
        if h4_changed:
            new_h4_zones = detect_zones(df_h4_w, cfg.tf_mid, cfg)
            _reset_touches_on_flip_or_death(tracker, cached_h4_zones, new_h4_zones)
            cached_h4_zones = new_h4_zones
            last_h4_ts = h4_last

        d1_changed = d1_last != last_d1_ts
        if d1_changed:
            new_d1_zones = detect_zones(df_d1_w, cfg.tf_high, cfg)
            _reset_touches_on_flip_or_death(tracker, cached_d1_zones, new_d1_zones)
            cached_d1_zones = new_d1_zones
            last_d1_ts = d1_last

        all_zones = cached_h4_zones + cached_d1_zones
        if not all_zones:
            continue

        bar_high  = float(df_h1["high"].iloc[i])
        bar_low   = float(df_h1["low"].iloc[i])
        bar_close = float(df_h1["close"].iloc[i])

        # Touch recording — runs for every zone on every bar, regardless of
        # position status or bias. evaluate_entry() below only READS the
        # count via tracker.get_count(); it never increments it (Task 5).
        for zone in all_zones:
            if is_rejection_touch(bar_high, bar_low, bar_close, zone):
                tracker.record_touch(zone)

        # One trade at a time — entries are only evaluated while flat.
        if i <= skip_until:
            continue

        # H1 window for entry confirmation and bias
        h1_start = max(0, i - cfg.h1_window + 1)
        df_h1_w  = df_h1.iloc[h1_start: i + 1].reset_index(drop=True)

        # Bias only depends on df_d1_w/df_h4_w (and df_h1_w in "strict" mode, which
        # shifts every bar). Recompute only when those windows actually changed —
        # otherwise every zone at every bar would redundantly re-scan the full
        # bias structure (this was the backtest's dominant cost).
        if cfg.bias_mode == "strict" or cached_bias is None or h4_changed or d1_changed:
            cached_bias    = get_aligned_bias(df_d1_w, df_h4_w, df_h1_w, cfg)
            cached_d1_bias = d1_bias(df_d1_w, cfg)
            cached_h4_bias = get_bias(df_h4_w, cfg)
        bar_bias    = cached_bias
        bar_d1_bias = cached_d1_bias
        bar_h4_bias = cached_h4_bias

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
                precomputed_bias=bar_bias,
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
                "pattern":         signal.pattern or "",
                "tp_mode":         signal.tp_mode or "",
                "zone_tf":         zone.origin_tf,
                "zone_kind":       zone.kind,
                "zone_top":        round(zone.top, 2),
                "zone_bottom":     round(zone.bottom, 2),
                "zone_score":      round(zone.score, 2),
                "live_touch_count": signal.live_count,
                "d1_bias":         bar_d1_bias,
                "h4_bias":         bar_h4_bias,
            })

            break  # one trade at a time; strongest zone wins

    return trades, equity_curve, df_h1
