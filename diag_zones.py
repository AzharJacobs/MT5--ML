"""
diag_zones.py — Read-only H4 zone diagnostic for the Z&Z live bot.

Connects to MT5, fetches exactly the H4 window the bot uses, runs the same
detection + analysis pipeline, and prints every zone plus the active zone,
direction, bias, and the raw swing highs/lows the bias logic detected.

Usage:
  python -X utf8 diag_zones.py
  python -X utf8 diag_zones.py --login 12345 --password xxx --server "Exness-MT5Trial6"

No orders are placed.  Credentials are read from .env first; flags override.
All timestamps printed in broker time (GMT+3 = UTC+3h), matching the MT5 chart.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# ── resolve project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ── re-use bot constants + helpers verbatim ───────────────────────────────────
from live_bot_zz import (
    H4_WINDOW,
    M15_WINDOW,
    SYMBOL,
    TAP_TOL,
    _load_ohlcv_mt5,
    _make_configs,
)

# ── strategy modules (same imports as the bot) ────────────────────────────────
from strategy.zones import detect_zones, update_freshness
from strategy.timeframe_structure import (
    TFConfig,
    analyse_timeframes,
    detect_h4_bias,
)

# ── load swing_structure_Z&Z.py the same way timeframe_structure.py does ─────
_zz_path = ROOT / "strategy" / "swing_structure_Z&Z.py"
_zz_spec = importlib.util.spec_from_file_location("swing_structure_zz", _zz_path)
_zz_mod  = importlib.util.module_from_spec(_zz_spec)
_zz_spec.loader.exec_module(_zz_mod)
_detect_swings = _zz_mod.detect_swings
_label_structure = _zz_mod.label_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BROKER_OFFSET = timedelta(hours=3)  # Exness GMT+3


def _to_broker(ts) -> str:
    """Convert a naive UTC pandas Timestamp / datetime to broker time string."""
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    broker_ts = ts.astimezone(timezone(BROKER_OFFSET))
    return broker_ts.strftime("%Y-%m-%d %H:%M [+3]")


def _sep(width: int = 110) -> None:
    print("─" * width)


def _connect_mt5(login: int, password: str, server: str) -> None:
    """Initialise MT5 terminal.  Raises SystemExit on failure."""
    import MetaTrader5 as mt5

    if not mt5.initialize(login=login, password=password, server=server):
        raise SystemExit(f"MT5 initialize() failed: {mt5.last_error()}")

    info = mt5.account_info()
    if info is None:
        raise SystemExit(f"account_info() returned None after connect: {mt5.last_error()}")

    print(f"\nMT5 connected  login={info.login}  server={info.server}  "
          f"balance={info.balance:.2f} {info.currency}")


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(login: int, password: str, server: str) -> None:
    import MetaTrader5 as mt5

    _connect_mt5(login, password, server)

    # ── Fetch data exactly as the bot does ────────────────────────────────────
    df_4h_raw  = _load_ohlcv_mt5("4H",    H4_WINDOW  + 10)
    df_15m_raw = _load_ohlcv_mt5("15min", M15_WINDOW + 10)

    if df_4h_raw.empty or df_15m_raw.empty:
        raise SystemExit("Failed to fetch data — check symbol name and MT5 connection")

    # Trim to exact window (identical to run_once)
    df_4h  = df_4h_raw.tail(H4_WINDOW).reset_index(drop=True)
    df_15m = df_15m_raw.tail(M15_WINDOW).reset_index(drop=True)

    now_utc = datetime.now(timezone.utc)
    print(f"\nDiagnostic run at: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}  "
          f"(broker: {_to_broker(now_utc)})")
    print(f"H4 window : {len(df_4h)} bars  "
          f"[{_to_broker(df_4h['timestamp'].iloc[0])}  →  "
          f"{_to_broker(df_4h['timestamp'].iloc[-1])}]")
    print(f"M15 window: {len(df_15m)} bars  "
          f"[{_to_broker(df_15m['timestamp'].iloc[0])}  →  "
          f"{_to_broker(df_15m['timestamp'].iloc[-1])}]")
    print(f"Current M15 close price: {df_15m['close'].iloc[-1]:.2f}")

    # ── Build configs exactly as the bot does ─────────────────────────────────
    tf_cfg, _conf_cfg, _setup_cfg = _make_configs()
    up_to_bar = len(df_4h) - 1

    # ── Section 1: all H4 zones ───────────────────────────────────────────────
    h4_zones = detect_zones(df_4h, cfg=tf_cfg.h4_zone_cfg, up_to_bar=up_to_bar)

    if h4_zones:
        update_freshness(
            h4_zones,
            df_4h["high"].values,
            df_4h["low"].values,
            current_bar=up_to_bar,
            tap_tol=tf_cfg.h4_zone_cfg.tap_tolerance_pct,
        )

    print(f"\n{'═'*110}")
    print(f"  ALL H4 ZONES  ({len(h4_zones)} detected)  —  sorted by price (bottom)")
    print(f"{'═'*110}")
    col = "{:<10}  {:>10}  {:>10}  {:>9}  {:>7}  {}"
    print(col.format("KIND", "BOTTOM", "TOP", "STRENGTH", "FRESH", "ORIGIN BAR (broker time)"))
    _sep()

    sorted_zones = sorted(h4_zones, key=lambda z: z.bottom)
    for z in sorted_zones:
        origin_ts = _to_broker(df_4h["timestamp"].iloc[z.origin_bar]) \
                    if z.origin_bar < len(df_4h) else "n/a"
        fresh_str = "YES" if z.fresh else f"no (×{z.tap_count})"
        print(col.format(
            z.kind,
            f"{z.bottom:.2f}",
            f"{z.top:.2f}",
            f"{z.strength:.2f}",
            fresh_str,
            origin_ts,
        ))

    if not h4_zones:
        print("  (none found)")
    _sep()

    # ── Section 2: full analyse_timeframes result ─────────────────────────────
    tf_result = analyse_timeframes(
        df_4h, df_15m,
        cfg=tf_cfg,
        h4_up_to_bar=up_to_bar,
    )

    h4_bias     = tf_result["h4_bias"]
    active_zone = tf_result["active_zone"]
    direction   = tf_result["direction"]
    signal      = tf_result["signal"]
    reason      = tf_result["reason"]
    eligible    = tf_result["eligible_zones"]
    zone_tapped = tf_result["zone_tapped"]
    m15_ok      = tf_result["m15_confirmed"]

    print(f"\n{'═'*110}")
    print("  ANALYSE_TIMEFRAMES RESULT")
    print(f"{'═'*110}")
    print(f"  H4 bias        : {h4_bias}")
    print(f"  Signal         : {signal}")
    print(f"  Direction      : {direction}")
    print(f"  Eligible zones : {len(eligible)}")
    print(f"  Zone tapped M15: {zone_tapped}")
    print(f"  M15 dir close  : {m15_ok}")
    print(f"  Reason         : {reason}")

    if active_zone is not None:
        fresh_str = "YES" if active_zone.fresh else f"no (×{active_zone.tap_count})"
        origin_ts = _to_broker(df_4h["timestamp"].iloc[active_zone.origin_bar]) \
                    if active_zone.origin_bar < len(df_4h) else "n/a"
        print(f"\n  ┌─ ACTIVE ZONE ─────────────────────────────────────────────────────────")
        print(f"  │  kind      : {active_zone.kind}")
        print(f"  │  bottom    : {active_zone.bottom:.2f}")
        print(f"  │  top       : {active_zone.top:.2f}")
        print(f"  │  mid       : {active_zone.mid:.2f}")
        print(f"  │  strength  : {active_zone.strength:.2f}")
        print(f"  │  fresh     : {fresh_str}")
        print(f"  │  origin bar: {origin_ts}")
        print(f"  │  zone_id   : {active_zone.zone_id}")
        print(f"  └───────────────────────────────────────────────────────────────────────")
    else:
        print("\n  Active zone: NONE")

    _sep()

    # ── Section 3: raw H4 swing structure ─────────────────────────────────────
    print(f"\n{'═'*110}")
    print(f"  H4 SWING STRUCTURE  (left={tf_cfg.h4_swing_left}  right={tf_cfg.h4_swing_right})  "
          f"→ h4_bias = {h4_bias}")
    print(f"{'═'*110}")

    swings  = _detect_swings(df_4h.reset_index(drop=True),
                              left=tf_cfg.h4_swing_left,
                              right=tf_cfg.h4_swing_right)
    labeled = _label_structure(swings)

    if not labeled:
        print("  (no confirmed swings in the H4 window)")
    else:
        sw_col = "{:<6}  {:>6}  {:>10}  {:>8}  {}"
        print(sw_col.format("#", "LABEL", "PRICE", "BAR IDX", "BROKER TIME"))
        _sep()

        # Print ALL labeled swings; flag the last 3 highs and 3 lows
        sh_indices = [i for i, (_, __, lbl) in enumerate(labeled) if lbl in ("HH", "LH")]
        sl_indices = [i for i, (_, __, lbl) in enumerate(labeled) if lbl in ("HL", "LL")]
        recent_sh = set(sh_indices[-3:]) if sh_indices else set()
        recent_sl = set(sl_indices[-3:]) if sl_indices else set()

        for row_i, (bar_idx, price, lbl) in enumerate(labeled):
            ts_str = _to_broker(df_4h["timestamp"].iloc[bar_idx]) \
                     if bar_idx < len(df_4h) else "n/a"
            marker = " ◄" if (row_i in recent_sh or row_i in recent_sl) else ""
            print(sw_col.format(
                row_i + 1,
                lbl,
                f"{price:.2f}",
                bar_idx,
                ts_str + marker,
            ))

        print()
        # Summary: last swing high and last swing low used for bias decision
        sh_labeled = [(i, p, l) for i, p, l in labeled if l in ("HH", "LH")]
        sl_labeled = [(i, p, l) for i, p, l in labeled if l in ("HL", "LL")]
        if sh_labeled:
            bi, pr, lb = sh_labeled[-1]
            ts = _to_broker(df_4h["timestamp"].iloc[bi]) if bi < len(df_4h) else "n/a"
            print(f"  Last swing HIGH : {lb}  price={pr:.2f}  bar={bi}  {ts}")
        if sl_labeled:
            bi, pr, lb = sl_labeled[-1]
            ts = _to_broker(df_4h["timestamp"].iloc[bi]) if bi < len(df_4h) else "n/a"
            print(f"  Last swing LOW  : {lb}  price={pr:.2f}  bar={bi}  {ts}")

    _sep()

    # ── Section 4: eligible zone summary ─────────────────────────────────────
    if eligible:
        print(f"\n{'═'*110}")
        print(f"  ELIGIBLE ZONES (pass directional filter for bias={h4_bias})")
        print(f"{'═'*110}")
        col2 = "{:<10}  {:>10}  {:>10}  {:>9}  {:>7}  {}"
        print(col2.format("KIND", "BOTTOM", "TOP", "STRENGTH", "FRESH", "ORIGIN BAR (broker time)"))
        _sep()
        for z in sorted(eligible, key=lambda z: z.bottom):
            origin_ts = _to_broker(df_4h["timestamp"].iloc[z.origin_bar]) \
                        if z.origin_bar < len(df_4h) else "n/a"
            fresh_str = "YES" if z.fresh else f"no (×{z.tap_count})"
            arrow = "  ◄ ACTIVE" if active_zone and z.zone_id == active_zone.zone_id else ""
            print(col2.format(
                z.kind,
                f"{z.bottom:.2f}",
                f"{z.top:.2f}",
                f"{z.strength:.2f}",
                fresh_str,
                origin_ts + arrow,
            ))
        _sep()

    mt5.shutdown()
    print("\nMT5 disconnected.  Diagnostic complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="H4 zone diagnostic — read-only")
    parser.add_argument("--login",    type=int, help="MT5 account number")
    parser.add_argument("--password",           help="MT5 password")
    parser.add_argument("--server",             help="MT5 server name")
    args = parser.parse_args()

    login    = args.login    or int(os.environ.get("MT5_LOGIN",    0))
    password = args.password or os.environ.get("MT5_PASSWORD", "")
    server   = args.server   or os.environ.get("MT5_SERVER",   "")

    if not all([login, password, server]):
        parser.error(
            "Credentials required: --login/--password/--server "
            "or MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in .env"
        )

    run_diagnostic(login=login, password=password, server=server)


if __name__ == "__main__":
    main()
