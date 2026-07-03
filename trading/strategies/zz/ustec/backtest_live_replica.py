#!/usr/bin/env python3
"""
backtest_live_replica.py — Bar-by-bar replay of live_bot_zz.py's EXACT decision
and trade-management logic, for historical validation.

Unlike engine.py (a separate, simplified simulator that has drifted from the
live bot — no regime filter, no weekly-trend filter, no zone blacklist, no
gradual/retest arrival gate, no BE-midline trailing, and a hardcoded
contract_size=1.0 bug), this script re-implements live_bot_zz.py's ZZLiveBot
loop verbatim against historical M15/H4 bars pulled from the DB, calling the
identical core functions (analyse_timeframes / check_confirmations_at_last_bar
/ setup_from_analysis) with the identical config.yaml-sourced configs.

Position sizing mirrors the live bot's paper-mode path: 1% equity risk per
trade (RISK_PCT from config.yaml), not the unused fixed_lot value.

Usage:
  python -X utf8 trading/strategies/zz/ustec/backtest_live_replica.py
  python -X utf8 trading/strategies/zz/ustec/backtest_live_replica.py --start 2023-01-01 --end 2024-01-01 --equity 50
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.ustec.strategy import (
    SYMBOL, DB_NAME, TABLE, CONTRACT_SIZE, SPREAD_PTS, RISK_PCT,
    MIN_SL_PCT, MAX_SL_PCT, MAX_POSITIONS, H4_WINDOW, M15_WINDOW,
    COOLDOWN_LOSS_H, COOLDOWN_WIN_FLOOR_H, TAP_TOL, ZONE_MAX_LOSSES,
    ENABLE_TRAILING, BE_BUFFER_PTS, H4_REGIME_FILTER,
    make_configs,
)
from trading.strategies.zz.core.timeframe_structure import analyse_timeframes
from trading.strategies.zz.core.confirmations import check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import setup_from_analysis
from trading.shared.data_loader import get_connection

_STRONG_SIGNALS = {"rejection_wick", "engulfing", "bos_msb", "choch"}
_WEEKLY_TREND_BARS = 80


def _load_ohlcv(db, timeframe: str, start: str, end: str) -> pd.DataFrame:
    q = (f"SELECT * FROM {TABLE} WHERE timeframe = %s "
         f"AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC")
    df = db.fetch_dataframe(q, (timeframe, start, end))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _zk(zone) -> tuple:
    return (round(zone.bottom, 1), round(zone.top, 1))


def _lot_size(equity: float, sl_dist: float) -> float:
    if sl_dist <= 0:
        return 0.01
    lot = (equity * RISK_PCT) / (sl_dist * CONTRACT_SIZE)
    return max(round(lot, 2), 0.01)


def run(report_start: str, report_end: str, starting_equity: float) -> pd.DataFrame:
    ts_report_start = pd.Timestamp(report_start)
    ts_report_end   = pd.Timestamp(report_end)
    data_start = (ts_report_start - timedelta(days=75)).strftime("%Y-%m-%d")
    data_end   = (ts_report_end   + timedelta(days=200)).strftime("%Y-%m-%d")

    db = get_connection()
    db.database = DB_NAME
    db.connect()
    print(f"Loading {SYMBOL} data  {data_start} -> {data_end}  (padded around report window {report_start}..{report_end}) ...")
    df_15m = _load_ohlcv(db, "15min", data_start, data_end)
    df_4h  = _load_ohlcv(db, "4H",    data_start, data_end)
    try:
        db.connection.close()
    except Exception:
        pass

    if df_15m.empty or df_4h.empty:
        raise SystemExit("No data returned — check DB connection / date range.")

    _ohlcv = ["open", "high", "low", "close"]
    df_4h = df_4h[df_4h[_ohlcv].ne(df_4h[_ohlcv].shift()).any(axis=1)].reset_index(drop=True)
    print(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")

    tf_cfg, conf_cfg, setup_cfg = make_configs()

    n = len(df_15m)
    h4_ts = df_4h["timestamp"].values
    h4_close_time = df_4h["timestamp"] + pd.Timedelta(hours=4)

    zone_cooldown: dict = {}
    zone_reentry: dict = {}
    zone_outcome_history: dict = {}
    zone_blacklist: set = set()
    zone_consec_losses: dict = {}
    zone_touch_tracker: dict = {}

    open_positions: list = []
    closed_trades: list = []

    equity = starting_equity
    blown_at = None  # timestamp equity first hit 0 — no new entries after this
    h4_idx = -1  # pointer: index of last FULLY CLOSED H4 bar as of ts_now

    # Find starting loop index: first M15 bar >= report_start, with enough
    # warm-up bars already available before it.
    start_i = int(df_15m["timestamp"].searchsorted(ts_report_start))
    start_i = max(start_i, M15_WINDOW)

    for i in range(start_i, n):
        ts_now = df_15m["timestamp"].iloc[i]
        bar_h = float(df_15m["high"].iloc[i])
        bar_l = float(df_15m["low"].iloc[i])
        bar_c = float(df_15m["close"].iloc[i])

        # advance H4 pointer to the last bar whose close time <= ts_now
        while h4_idx + 1 < len(df_4h) and h4_close_time.iloc[h4_idx + 1] <= ts_now:
            h4_idx += 1
        if h4_idx < 20:
            continue

        # ── zone touch tracker: mark exited/violation using this bar ─────────
        for _zks in zone_touch_tracker.values():
            if not _zks["exited"]:
                if _zks["kind"] == "demand":
                    if bar_l > _zks["top"]:
                        _zks["exited"] = True; _zks["exit_kind"] = "favorable"
                    elif bar_h < _zks["bottom"]:
                        _zks["exited"] = True; _zks["exit_kind"] = "violation"
                else:
                    if bar_h < _zks["bottom"]:
                        _zks["exited"] = True; _zks["exit_kind"] = "favorable"
                    elif bar_l > _zks["top"]:
                        _zks["exited"] = True; _zks["exit_kind"] = "violation"

        # ── zone reentry (leave-and-return) update ────────────────────────────
        if zone_reentry:
            done = []
            for _zkey, state in zone_reentry.items():
                z_top = state["top"] * (1 + TAP_TOL)
                z_bot = state["bottom"] * (1 - TAP_TOL)
                if state["phase"] == "exit":
                    if bar_h < z_bot or bar_l > z_top:
                        state["phase"] = "return"
                else:
                    if bar_l <= z_top and bar_h >= z_bot and ts_now >= state["earliest"]:
                        done.append(_zkey)
            for _zkey in done:
                del zone_reentry[_zkey]

        # ── check open positions for TP/SL (with BE-midline trailing) ────────
        for pos in list(open_positions):
            direction = pos["direction"]
            if ENABLE_TRAILING and not pos["be_triggered"]:
                midline = (pos["entry"] + pos["tp"]) / 2
                if direction == "buy" and bar_c >= midline:
                    pos["be_triggered"] = True
                    pos["sl"] = max(pos["sl"], pos["entry"] + BE_BUFFER_PTS)
                elif direction == "sell" and bar_c <= midline:
                    pos["be_triggered"] = True
                    pos["sl"] = min(pos["sl"], pos["entry"] - BE_BUFFER_PTS)

            outcome = None; close_price = None; exit_reason = None
            if direction == "buy":
                if bar_h >= pos["tp"]:
                    outcome = 1;  close_price = pos["tp"]; exit_reason = "TP"
                elif bar_l <= pos["sl"]:
                    outcome = -1; close_price = pos["sl"]; exit_reason = "SL"
            else:
                if bar_l <= pos["tp"]:
                    outcome = 1;  close_price = pos["tp"]; exit_reason = "TP"
                elif bar_h >= pos["sl"]:
                    outcome = -1; close_price = pos["sl"]; exit_reason = "SL"

            if outcome is not None:
                pnl = ((close_price - pos["entry"]) * pos["lots"] * CONTRACT_SIZE
                       if direction == "buy"
                       else (pos["entry"] - close_price) * pos["lots"] * CONTRACT_SIZE)

                zid, zkey = pos["zid"], pos["zk"]
                zone_outcome_history.setdefault(zid, []).append(outcome)
                if outcome == 1:
                    if ZONE_MAX_LOSSES > 0:
                        zone_consec_losses[zid] = 0
                    zone_reentry[zkey] = {
                        "phase": "exit", "bottom": pos["zone_bottom"], "top": pos["zone_top"],
                        "earliest": ts_now + timedelta(hours=COOLDOWN_WIN_FLOOR_H),
                    }
                else:
                    zone_cooldown[zkey] = ts_now + timedelta(hours=COOLDOWN_LOSS_H)
                    if pnl < 0 and ZONE_MAX_LOSSES > 0:
                        new_count = zone_consec_losses.get(zid, 0) + 1
                        zone_consec_losses[zid] = new_count
                        if new_count >= ZONE_MAX_LOSSES:
                            zone_blacklist.add(zid)
                zone_touch_tracker.pop(zkey, None)

                equity = max(equity + pnl, 0.0)
                if equity <= 0.0 and blown_at is None:
                    blown_at = ts_now
                hours_held = (ts_now - pos["open_time"]).total_seconds() / 3600.0
                closed_trades.append({
                    **pos["record"],
                    "exit_time": ts_now, "exit": round(close_price, 2),
                    "exit_reason": exit_reason, "be_triggered": pos["be_triggered"],
                    "hours_held": round(hours_held, 2),
                    "outcome": "WIN" if outcome == 1 else "LOSS",
                    "pnl": round(pnl, 2), "equity_after": round(equity, 2),
                })
                open_positions.remove(pos)

        if len(open_positions) >= MAX_POSITIONS:
            continue
        if ts_now > ts_report_end:
            continue  # stop opening new trades once report window is over
        if blown_at is not None:
            continue  # account blown — no new entries, only let existing positions resolve

        # ── build windows (no lookahead: only fully-closed bars) ─────────────
        h4_lo = max(0, h4_idx - H4_WINDOW + 1)
        df_h4_w = df_4h.iloc[h4_lo: h4_idx + 1].reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue
        df_h4_w["ema50"] = df_h4_w["close"].ewm(span=50, adjust=False).mean()
        df_h4_w["ema200"] = df_h4_w["close"].ewm(span=200, adjust=False).mean()

        m15_lo = max(0, i - M15_WINDOW + 1)
        df_15m_w = df_15m.iloc[m15_lo: i + 1].reset_index(drop=True)
        if len(df_15m_w) < M15_WINDOW:
            continue

        tf_result = analyse_timeframes(
            df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=len(df_h4_w) - 1,
        )
        if tf_result["signal"] == "neutral":
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = _zk(active_zone)
        zid         = active_zone.zone_id

        if H4_REGIME_FILTER and direction == "buy":
            _price = float(df_15m_w["close"].iloc[-1])
            _ema50 = float(df_h4_w["ema50"].iloc[-1]); _ema200 = float(df_h4_w["ema200"].iloc[-1])
            if _price < _ema50 and _price < _ema200:
                continue
        if H4_REGIME_FILTER and direction == "sell":
            _price = float(df_15m_w["close"].iloc[-1])
            _ema50 = float(df_h4_w["ema50"].iloc[-1]); _ema200 = float(df_h4_w["ema200"].iloc[-1])
            if _price > _ema50 and _price > _ema200:
                continue

        if h4_idx > _WEEKLY_TREND_BARS:
            _cur  = float(df_4h["close"].iloc[h4_idx])
            _past = float(df_4h["close"].iloc[h4_idx - _WEEKLY_TREND_BARS])
            if direction == "buy" and _cur < _past:
                continue
            if direction == "sell" and _cur > _past:
                continue

        if zk not in zone_touch_tracker:
            zone_touch_tracker[zk] = {
                "bottom": active_zone.bottom, "top": active_zone.top,
                "kind": active_zone.kind, "exited": False, "exit_kind": None,
            }

        _until = zone_cooldown.get(zk)
        if (_until is not None and ts_now < _until) or (zk in zone_reentry):
            continue
        if ZONE_MAX_LOSSES > 0 and zid in zone_blacklist:
            continue

        conf = check_confirmations_at_last_bar(df_15m_w, active_zone, direction, conf_cfg)
        if not conf.confirmed:
            continue

        signal_price = float(df_15m_w["close"].iloc[-1])
        if direction == "buy" and signal_price > active_zone.top:
            continue
        if direction == "sell" and signal_price < active_zone.bottom:
            continue

        _atr_vals = []
        for _ak in range(max(1, len(df_15m_w) - 14), len(df_15m_w)):
            _fh = float(df_15m_w["high"].iloc[_ak]); _fl = float(df_15m_w["low"].iloc[_ak])
            _fpc = float(df_15m_w["close"].iloc[_ak - 1])
            _atr_vals.append(max(_fh - _fl, abs(_fh - _fpc), abs(_fl - _fpc)))
        m15_atr14 = sum(_atr_vals) / max(len(_atr_vals), 1) if _atr_vals else 0.0

        setup = setup_from_analysis(tf_result, signal_price, setup_cfg,
                                     m15_sl_anchor=conf.m15_sl_anchor, m15_atr14=m15_atr14)
        if not setup.valid:
            continue

        if direction == "buy" and setup.tp < signal_price + 30.0:
            continue
        if direction == "sell" and setup.tp > signal_price - 30.0:
            continue

        _tracker = zone_touch_tracker.get(zk, {})
        if _tracker.get("exited", False):
            if _tracker.get("exit_kind") == "violation":
                continue
            arrival_type = "retest"
        else:
            arrival_type = "gradual"

        if arrival_type == "gradual" and not _STRONG_SIGNALS.intersection(conf.signals):
            continue

        # ── enter trade (paper-mode fill: signal price + spread) ─────────────
        entry = setup.entry
        sl = setup.sl; tp = setup.tp
        if SPREAD_PTS > 0:
            entry = entry + SPREAD_PTS if direction == "buy" else entry - SPREAD_PTS
        sl_dist = abs(setup.entry - sl)
        sl = (entry - sl_dist) if direction == "buy" else (entry + sl_dist)

        if direction == "buy" and (sl >= entry or tp <= entry):
            continue
        if direction == "sell" and (sl <= entry or tp >= entry):
            continue

        sl_dist_pct = abs(entry - sl) / entry * 100.0
        if sl_dist_pct < MIN_SL_PCT:
            continue
        if MAX_SL_PCT > 0.0 and sl_dist_pct > MAX_SL_PCT:
            continue

        sl_dist = abs(entry - sl)
        lots = _lot_size(equity, sl_dist)
        rr = abs(tp - entry) / sl_dist if sl_dist > 0 else 0.0
        touch_count = len(zone_outcome_history.get(zid, [])) + 1

        record = {
            "entry_time": ts_now, "direction": direction,
            "zone_kind": active_zone.kind,
            "zone_range": f"{active_zone.bottom:.2f}-{active_zone.top:.2f}",
            "zone_strength": round(active_zone.strength, 2),
            "arrival_type": arrival_type, "touch_count": touch_count,
            "signals": "|".join(conf.signals), "h4_bias": tf_result["h4_bias"],
            "entry": round(entry, 2), "sl": round(sl, 2), "tp": round(tp, 2),
            "rr": round(rr, 2), "lots": lots,
        }

        open_positions.append({
            "direction": direction, "entry": entry, "sl": sl, "tp": tp, "lots": lots,
            "zk": zk, "zid": zid, "zone_bottom": active_zone.bottom, "zone_top": active_zone.top,
            "open_time": ts_now, "record": record, "be_triggered": False,
        })

    # ── any positions still open when data runs out ──────────────────────────
    for pos in open_positions:
        last_close = float(df_15m["close"].iloc[-1])
        direction = pos["direction"]
        pnl = ((last_close - pos["entry"]) * pos["lots"] * CONTRACT_SIZE
               if direction == "buy" else (pos["entry"] - last_close) * pos["lots"] * CONTRACT_SIZE)
        hours_held = (df_15m["timestamp"].iloc[-1] - pos["open_time"]).total_seconds() / 3600.0
        closed_trades.append({
            **pos["record"], "exit_time": df_15m["timestamp"].iloc[-1],
            "exit": round(last_close, 2), "exit_reason": "other",
            "be_triggered": pos["be_triggered"], "hours_held": round(hours_held, 2),
            "outcome": "OPEN", "pnl": round(pnl, 2), "equity_after": round(equity, 2),
        })

    if blown_at is not None:
        print(f"\n*** ACCOUNT BLOWN at {blown_at} — equity hit $0, no new entries taken after this point. ***")

    df_t = pd.DataFrame(closed_trades)
    if df_t.empty:
        return df_t
    df_t = df_t[(df_t["entry_time"] >= ts_report_start) & (df_t["entry_time"] <= ts_report_end)].reset_index(drop=True)
    return df_t


def print_report(df_t: pd.DataFrame, starting_equity: float) -> None:
    if df_t.empty:
        print("No trades in report window.")
        return

    df_t = df_t.sort_values("entry_time").reset_index(drop=True)
    df_t["month"] = df_t["entry_time"].dt.to_period("M")

    cols = ["entry_time", "direction", "zone_kind", "zone_range", "touch_count",
            "arrival_type", "signals", "entry", "sl", "tp", "exit", "exit_reason",
            "hours_held", "outcome", "pnl", "equity_after"]
    cols = [c for c in cols if c in df_t.columns]

    grand_trades = 0; grand_wins = 0; grand_losses = 0; grand_pnl = 0.0

    for period, grp in df_t.groupby("month"):
        grp = grp.reset_index(drop=True)
        decided = grp[grp["outcome"].isin(["WIN", "LOSS"])]
        trades = len(decided); wins = int((decided["outcome"] == "WIN").sum())
        losses = trades - wins
        wr = wins / trades * 100 if trades else 0.0
        month_pnl = grp["pnl"].sum()

        print(f"\n{'='*30} {period} {'='*30}")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(grp[cols].to_string(index=True))
        print(f"-- {period}: trades={trades}  wins={wins}  losses={losses}  "
              f"WR={wr:.1f}%  month_pnl=${month_pnl:+.2f}  end_equity=${grp['equity_after'].iloc[-1]:,.2f}")

        grand_trades += trades; grand_wins += wins; grand_losses += losses; grand_pnl += month_pnl

    grand_wr = grand_wins / grand_trades * 100 if grand_trades else 0.0
    print(f"\n{'='*70}")
    print(f"TOTAL  trades={grand_trades}  wins={grand_wins}  losses={grand_losses}  "
          f"WR={grand_wr:.1f}%  net_pnl=${grand_pnl:+.2f}  "
          f"start_equity=${starting_equity:.2f}  end_equity=${df_t['equity_after'].iloc[-1]:,.2f}")
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest replicating live_bot_zz.py exactly")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--equity", type=float, default=50.0)
    args = parser.parse_args()

    df_t = run(args.start, args.end, args.equity)
    print_report(df_t, args.equity)


if __name__ == "__main__":
    main()
