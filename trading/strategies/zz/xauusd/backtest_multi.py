#!/usr/bin/env python3
"""
XAUUSD (Gold) Zone-to-Zone — multi-position backtest (up to N concurrent trades).

Applies all three gold fixes:
  Fix 1: permanent post-loss zone blacklist
  Fix 2: active-signals confirmation gate
  Fix 3a/b: ATR thin-zone filter + ATR SL buffer

Optional improvements (CLI flags):
  --split_tp        Each signal opens 2 half-lot legs: one exits at 50% TP, one runs full TP.
  --max_sl_dist N   Skip any trade whose SL distance exceeds N dollars (0 = disabled).
  --min_rr N        Minimum risk-reward ratio (overrides config).
  --min_conf N      Minimum active confirmations (overrides config).

Reports:
  - Overall summary
  - Monthly breakdown (net PnL incl. losses + winners-only)
  - Yearly breakdown
  - Losing trade detail table
  - (split_tp mode) Signal-level summary table

Usage:
    python trading/strategies/zz/xauusd/backtest_multi.py
    python trading/strategies/zz/xauusd/backtest_multi.py --start 2022-01-01 --end 2026-01-01
    python trading/strategies/zz/xauusd/backtest_multi.py --split_tp
    python trading/strategies/zz/xauusd/backtest_multi.py --split_tp --max_sl_dist 80
    python trading/strategies/zz/xauusd/backtest_multi.py --split_tp --max_sl_dist 80 --min_conf 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.xauusd.strategy import (
    make_gold_config,
    MAX_FORWARD_BARS,
    H4_WINDOW,
    M15_WINDOW,
    SPREAD_PTS,
    CONTRACT_SIZE,
    FIXED_LOTS,
    COOLDOWN_BARS,
)
from trading.strategies.zz.core.timeframe_structure import TFConfig, analyse_timeframes
from trading.strategies.zz.core.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import TradeSetupConfig, setup_from_analysis
from trading.strategies.zz.core.zones import ZoneConfig
from trading.shared.data_loader import get_connection

_DB_NAME   = "XAUUSD"
_TABLE     = "xauusd_ohlcv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_ohlcv(db, table: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    query = (
        f"SELECT * FROM {table} WHERE timeframe = %s "
        f"AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC"
    )
    df = db.fetch_dataframe(query, (timeframe, start, end))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _zone_key(zone) -> tuple:
    return (round(zone.bottom, 1), round(zone.top, 1))


def _h4_atr14(df_h4_w: pd.DataFrame) -> float:
    hi = df_h4_w["high"].values
    lo = df_h4_w["low"].values
    cl = df_h4_w["close"].values
    if len(hi) < 2:
        return float("nan")
    tr = np.maximum(
        hi[1:] - lo[1:],
        np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])),
    )
    n = min(14, len(tr))
    return float(np.mean(tr[-n:]))


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_multi_backtest_gold(
    start: str = "2022-01-01",
    end: str = "2026-01-01",
    max_positions: int = 3,
    lot: float = 0.04,
    spread: float = SPREAD_PTS,
    split_tp: bool = False,
    max_sl_dist: float = 0.0,
    min_rr_override: float = 0.0,
    min_conf_override: int = 0,
) -> None:
    cfg = make_gold_config()

    if min_rr_override > 0:
        from dataclasses import replace
        cfg = replace(cfg, min_rr=min_rr_override)
    if min_conf_override > 0:
        from dataclasses import replace
        cfg = replace(cfg, min_confirmations=min_conf_override)

    # In split_tp mode max_positions refers to concurrent signals (each = 2 legs)
    # so we track slots by unique signal_id, not raw leg count.
    leg_lot = lot / 2 if split_tp else lot

    tf_cfg = TFConfig(
        directional_filter=cfg.directional_filter,
        allow_neutral_up=cfg.allow_neutral,
        allow_neutral_down=cfg.allow_neutral,
        h4_zone_cfg=ZoneConfig(
            impulse_atr_mult=cfg.zone_impulse_atr_mult,
            body_ratio_min=cfg.zone_body_ratio_min,
            min_departure_candles=cfg.zone_min_departure_candles,
            departure_window=cfg.zone_departure_window,
            base_lookback=cfg.zone_base_lookback,
            min_strength=cfg.zone_min_strength,
        ),
        m15_tap_lookback=20,
        require_m15_directional_close=True,
    )
    conf_cfg = ConfirmationConfig(
        min_confirmations=cfg.min_confirmations,
        aggressive_boundary=cfg.aggressive_boundary,
        bos_lookback=cfg.bos_lookback,
        structure_lookback=cfg.structure_lookback,
    )
    setup_cfg = TradeSetupConfig(
        aggressive_entry=cfg.aggressive_entry,
        sl_buffer_pct=cfg.sl_buffer_pct,
        midline_tp=cfg.midline_tp,
        midline_pct=cfg.midline_pct,
        min_rr=cfg.min_rr,
    )

    db = get_connection()
    db.database = _DB_NAME
    db.connect()

    sl_cap_label = f"${max_sl_dist:.0f}" if max_sl_dist > 0 else "off"
    print(f"\n{'='*72}")
    print(f"  XAUUSD (Gold) Multi-Position Backtest")
    print(f"  Period      : {start}  →  {end}")
    print(f"  Lot size    : {lot}  ({'split: 2×'+str(leg_lot) if split_tp else 'single leg'})"
          f"  |  Max concurrent signals: {max_positions}")
    print(f"  Spread      : {spread} pts  |  Contract size: ${CONTRACT_SIZE}/pt/lot")
    print(f"  Gold fixes  : zone_blacklist={cfg.failed_zone_filter}  "
          f"atr_frac={cfg.min_zone_atr_frac}  sl_buf={cfg.sl_atr_buffer}")
    print(f"  Active sigs : {sorted(cfg.active_signals)}  min={cfg.min_confirmations}")
    print(f"  Extras      : split_tp={split_tp}  max_sl_dist={sl_cap_label}  min_rr={cfg.min_rr}")
    print(f"{'='*72}")

    print("\nLoading data ...")
    df_15m = _load_ohlcv(db, _TABLE, "15min", start, end)
    df_4h  = _load_ohlcv(db, _TABLE, "4H",    start, end)

    try:
        db.connection.close()
    except Exception:
        pass

    if df_15m.empty or df_4h.empty:
        print("ERROR: no data returned — check DB and date range.")
        return

    _ohlcv = ["open", "high", "low", "close"]
    _before = len(df_4h)
    df_4h = df_4h[df_4h[_ohlcv].ne(df_4h[_ohlcv].shift()).any(axis=1)].reset_index(drop=True)
    dupes = _before - len(df_4h)
    if dupes:
        print(f"  (removed {dupes} duplicate 4H rows)")
    print(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")

    n      = len(df_15m)
    warmup = max(M15_WINDOW, 30)

    open_positions: list[dict] = []
    closed_trades:  list[dict] = []

    zone_cooldown: dict = {}
    zone_reentry:  dict = {}
    failed_zones:  set  = set()
    signal_counter: int = 0

    for i in range(warmup, n - 1):
        ts_now = df_15m["timestamp"].iloc[i]
        bar_h  = float(df_15m["high"].iloc[i])
        bar_l  = float(df_15m["low"].iloc[i])

        # ── Step 1: update leave-and-return tracking ──────────────────────────
        if zone_reentry:
            tol   = 0.001
            ready = []
            for zk, state in zone_reentry.items():
                z_bot = state["bottom"] * (1 - tol)
                z_top = state["top"]    * (1 + tol)
                if state["phase"] == "exit":
                    if bar_h < z_bot or bar_l > z_top:
                        state["phase"] = "return"
                else:
                    if z_bot <= bar_h and bar_l <= z_top:
                        if i >= state["earliest_reentry"]:
                            ready.append(zk)
            for zk in ready:
                del zone_reentry[zk]

        # ── Step 2: update open positions against this bar ────────────────────
        still_open: list[dict] = []
        for pos in open_positions:
            direction = pos["side"]
            entry     = pos["entry"]
            sl        = pos["sl"]
            tp        = pos["tp"]
            half_p    = entry + 0.5 * (tp - entry)   # works for buy and sell

            # Track 50% TP milestone (for full-TP legs; 50% legs auto-hit on close)
            if not pos["half_tp_hit"]:
                if direction == "buy"  and bar_h >= half_p:
                    pos["half_tp_hit"] = True
                elif direction == "sell" and bar_l <= half_p:
                    pos["half_tp_hit"] = True

            outcome    = 0
            exit_price = float(df_15m["close"].iloc[i])
            if direction == "buy":
                if bar_h >= tp:
                    outcome = 1;  exit_price = tp
                elif bar_l <= sl:
                    outcome = -1; exit_price = sl
            else:
                if bar_l <= tp:
                    outcome = 1;  exit_price = tp
                elif bar_h >= sl:
                    outcome = -1; exit_price = sl

            bars_held = i - pos["entry_bar"]
            resolved  = (outcome != 0) or (bars_held >= MAX_FORWARD_BARS)

            if resolved:
                pnl = (
                    (exit_price - entry) * pos["lot"] * CONTRACT_SIZE
                    if direction == "buy"
                    else (entry - exit_price) * pos["lot"] * CONTRACT_SIZE
                )
                zk  = pos["zone_key"]
                zid = pos["zone_id"]
                pos.update({
                    "outcome":    outcome,
                    "exit_price": round(exit_price, 2),
                    "exit_bar":   i,
                    "exit_date":  ts_now,
                    "pnl":        round(pnl, 2),
                })
                closed_trades.append(pos)

                # Only the full-TP leg (or single leg) drives zone state
                if pos["leg"] in ("full", "single"):
                    if outcome == 1:
                        if cfg.require_leave_and_return:
                            zone_reentry[zk] = {
                                "phase":            "exit",
                                "bottom":           pos["zone_bottom"],
                                "top":              pos["zone_top"],
                                "earliest_reentry": i + COOLDOWN_BARS,
                            }
                        else:
                            zone_cooldown[zk] = i + COOLDOWN_BARS
                    elif outcome == -1:
                        if cfg.failed_zone_filter:
                            failed_zones.add(zid)
                        else:
                            zone_cooldown[zk] = i + 48
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── Step 3: look for new entry if signal slot is free ─────────────────
        active_signal_count = len({p["signal_id"] for p in open_positions})
        if active_signal_count >= max_positions or i >= n - 5:
            continue

        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        m15_start = max(0, i - M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)
        h4_up_to  = len(df_h4_w) - 1

        h4_atr = _h4_atr14(df_h4_w)

        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=h4_up_to)
        if tf_result["signal"] == "neutral":
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = _zone_key(active_zone)
        zid         = active_zone.zone_id

        # Fix 3a: skip thin zones
        if h4_atr > 0:
            zone_height = active_zone.top - active_zone.bottom
            if zone_height < cfg.min_zone_atr_frac * h4_atr:
                continue

        # Cooldown / blacklist gates
        if zone_cooldown.get(zk, -1) >= i:
            continue
        if cfg.require_leave_and_return and zk in zone_reentry:
            continue
        if cfg.failed_zone_filter and zid in failed_zones:
            continue

        # Don't stack two signals on the same zone
        if zk in {p["zone_key"] for p in open_positions}:
            continue

        # Fix 2: active-signals confirmation gate
        conf = check_confirmations_at_last_bar(df_15m_w, active_zone, direction, conf_cfg)
        active_sigs  = [s for s in conf.signals if s in cfg.active_signals]
        active_count = len(active_sigs)
        if active_count < cfg.min_confirmations:
            continue

        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            continue

        next_open   = float(df_15m["open"].iloc[i + 1])
        entry_price = next_open + spread if direction == "buy" else next_open - spread

        sl_dist = abs(signal_price - setup.sl)
        # Fix 3b: widen SL by ATR buffer
        if h4_atr > 0:
            sl_dist += cfg.sl_atr_buffer * h4_atr

        # Max SL distance cap — skip if total signal dollar risk exceeds threshold
        # Denominator uses full lot so the cap is consistent regardless of split_tp
        if max_sl_dist > 0 and sl_dist * lot * CONTRACT_SIZE > max_sl_dist:
            continue

        sl = entry_price - sl_dist if direction == "buy" else entry_price + sl_dist
        tp = setup.tp

        if direction == "buy":
            if sl >= entry_price or tp <= entry_price:
                continue
        else:
            if sl <= entry_price or tp >= entry_price:
                continue

        rr = abs(tp - entry_price) / sl_dist if sl_dist > 0 else 0.0
        if rr < cfg.min_rr:
            continue

        signal_counter += 1
        sig_id = signal_counter

        common = {
            "entry_bar":   i,
            "entry_date":  ts_now,
            "exit_date":   None,
            "side":        direction,
            "entry":       round(entry_price, 2),
            "sl":          round(sl, 2),
            "tp_full":     round(tp, 2),
            "zone_key":    zk,
            "zone_id":     zid,
            "zone_bottom": active_zone.bottom,
            "zone_top":    active_zone.top,
            "half_tp_hit": False,
            "outcome":     None,
            "exit_price":  None,
            "exit_bar":    None,
            "pnl":         None,
            "confirmations": active_count,
            "signals":       "|".join(active_sigs),
            "h4_bias":       tf_result.get("h4_bias", ""),
            "h4_atr":        round(h4_atr, 2) if h4_atr > 0 else 0.0,
            "signal_id":     sig_id,
        }

        if split_tp:
            half_tp = round(entry_price + 0.5 * (tp - entry_price), 2)
            open_positions.append({**common, "lot": leg_lot, "tp": half_tp, "leg": "50pct",
                                   "half_tp_hit": True})   # 50% leg always "hits" milestone
            open_positions.append({**common, "lot": leg_lot, "tp": tp,      "leg": "full"})
        else:
            open_positions.append({**common, "lot": lot, "tp": tp, "leg": "single"})

    # Close any positions still open at end of data
    last_idx   = n - 1
    last_close = float(df_15m["close"].iloc[last_idx])
    last_ts    = df_15m["timestamp"].iloc[last_idx]
    for pos in open_positions:
        direction = pos["side"]
        entry     = pos["entry"]
        pnl = (
            (last_close - entry) * pos["lot"] * CONTRACT_SIZE
            if direction == "buy"
            else (entry - last_close) * pos["lot"] * CONTRACT_SIZE
        )
        pos.update({
            "outcome":    0,
            "exit_price": round(last_close, 2),
            "exit_bar":   last_idx,
            "exit_date":  last_ts,
            "pnl":        round(pnl, 2),
        })
        closed_trades.append(pos)

    if not closed_trades:
        print("\nNo trades generated. Try widening the date range or relaxing filters.")
        return

    df_t = pd.DataFrame(closed_trades)
    df_t["entry_month"] = pd.to_datetime(df_t["entry_date"]).dt.to_period("M")
    df_t["entry_year"]  = pd.to_datetime(df_t["entry_date"]).dt.year

    # ── Overall summary ───────────────────────────────────────────────────────
    total         = len(df_t)
    tp_hits       = int((df_t["outcome"] == 1).sum())
    sl_hits       = int((df_t["outcome"] == -1).sum())
    expired       = int((df_t["outcome"] == 0).sum())
    win_rate      = tp_hits / max(total, 1) * 100
    half_tp_total = int(df_t["half_tp_hit"].sum())
    net_pnl       = df_t["pnl"].sum()
    winners_pnl   = df_t[df_t["pnl"] > 0]["pnl"].sum()
    losers_pnl    = df_t[df_t["pnl"] < 0]["pnl"].sum()
    avg_win       = df_t[df_t["pnl"] > 0]["pnl"].mean() if tp_hits  else 0.0
    avg_loss      = df_t[df_t["pnl"] < 0]["pnl"].mean() if sl_hits  else 0.0

    n_signals = df_t["signal_id"].nunique()

    print(f"\n{'='*72}")
    print(f"  OVERALL SUMMARY  ({start}  →  {end})")
    print(f"{'='*72}")
    if split_tp:
        print(f"  Signals fired    : {n_signals}")
        print(f"  Total legs       : {total}  (2 per signal)")
    else:
        print(f"  Total trades     : {total}")
    print(f"  Full TP hits     : {tp_hits}  ({win_rate:.1f}%)")
    print(f"  SL hits          : {sl_hits}")
    print(f"  Expired          : {expired}")
    print(f"  50% TP reached   : {half_tp_total}  ({half_tp_total/max(total,1)*100:.1f}%)")
    print(f"  Net PnL          : ${net_pnl:+,.2f}")
    print(f"  Winners only     : ${winners_pnl:+,.2f}")
    print(f"  Losers total     : ${losers_pnl:+,.2f}")
    print(f"  Avg win          : ${avg_win:+,.2f}")
    print(f"  Avg loss         : ${avg_loss:+,.2f}")
    print()

    # ── Signal-level summary (split_tp mode) ─────────────────────────────────
    if split_tp and n_signals > 0:
        sig_rows = []
        for sid, grp in df_t.groupby("signal_id"):
            half_leg = grp[grp["leg"] == "50pct"].iloc[0] if len(grp[grp["leg"] == "50pct"]) else None
            full_leg = grp[grp["leg"] == "full"].iloc[0]   if len(grp[grp["leg"] == "full"])  else None
            half_out = int(half_leg["outcome"]) if half_leg is not None else None
            full_out = int(full_leg["outcome"]) if full_leg is not None else None
            if   half_out == 1 and full_out == 1:  result = "both_tp"
            elif half_out == 1 and full_out == -1: result = "half_win_full_loss"
            elif half_out == 1 and full_out == 0:  result = "half_win_full_expired"
            elif half_out == -1:                   result = "both_sl"
            else:                                  result = "other"
            sig_rows.append({"signal_id": sid, "result": result,
                             "pnl": grp["pnl"].sum()})

        df_sig = pd.DataFrame(sig_rows)
        print(f"{'='*72}")
        print(f"  SIGNAL-LEVEL OUTCOMES  ({n_signals} signals)")
        print(f"{'─'*72}")
        for cat in ("both_tp", "half_win_full_loss", "half_win_full_expired",
                    "both_sl", "other"):
            g = df_sig[df_sig["result"] == cat]
            if len(g) == 0:
                continue
            pct = len(g) / n_signals * 100
            net = g["pnl"].sum()
            print(f"  {cat:<26} : {len(g):>4} signals  ({pct:.1f}%)  net=${net:+,.2f}")
        print(f"{'─'*72}")
        full_wins = len(df_sig[df_sig["result"] == "both_tp"])
        partial   = len(df_sig[df_sig["result"] == "half_win_full_loss"])
        print(f"  Full wins (both legs TP)    : {full_wins}  ({full_wins/n_signals*100:.1f}%)")
        print(f"  Partial wins (50% only)     : {partial}   ({partial/n_signals*100:.1f}%)")
        print(f"{'='*72}")
        print()

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    W = 98
    print(f"{'='*W}")
    print(f"  MONTHLY BREAKDOWN  (Net PnL includes losses | Win-Only excludes losses)")
    print(f"{'─'*W}")
    hdr = (f"  {'Month':<10} {'Legs':>5} {'FullTP':>7} {'50%TP':>6} "
           f"{'Losses':>7} {'Exprd':>6}  {'Net PnL':>11}  {'Win-Only':>11}  {'WR':>6}")
    print(hdr)
    print(f"{'─'*W}")

    for month, grp in df_t.groupby("entry_month"):
        m_total   = len(grp)
        m_tp      = int((grp["outcome"] == 1).sum())
        m_sl      = int((grp["outcome"] == -1).sum())
        m_exp     = int((grp["outcome"] == 0).sum())
        m_half    = int(grp["half_tp_hit"].sum())
        m_net     = grp["pnl"].sum()
        m_win_pnl = grp[grp["pnl"] > 0]["pnl"].sum()
        m_wr      = m_tp / max(m_total, 1) * 100
        print(
            f"  {str(month):<10} {m_total:>5} {m_tp:>7} {m_half:>6} {m_sl:>7} {m_exp:>6} "
            f" ${m_net:>+10,.2f}  ${m_win_pnl:>+10,.2f}  {m_wr:>5.1f}%"
        )

    print(f"{'─'*W}")
    print(
        f"  {'TOTAL':<10} {total:>5} {tp_hits:>7} {half_tp_total:>6} {sl_hits:>7} {expired:>6} "
        f" ${net_pnl:>+10,.2f}  ${winners_pnl:>+10,.2f}  {win_rate:>5.1f}%"
    )
    print(f"{'='*W}")

    # ── Yearly breakdown ──────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  YEARLY BREAKDOWN")
    print(f"{'─'*W}")
    hdr2 = (f"  {'Year':<6} {'Legs':>5} {'FullTP':>7} {'50%TP':>6} "
            f"{'Losses':>7} {'Exprd':>6}  {'Net PnL':>11}  {'Win-Only':>11}  {'WR':>6}")
    print(hdr2)
    print(f"{'─'*W}")

    for year, grp in df_t.groupby("entry_year"):
        y_total   = len(grp)
        y_tp      = int((grp["outcome"] == 1).sum())
        y_sl      = int((grp["outcome"] == -1).sum())
        y_exp     = int((grp["outcome"] == 0).sum())
        y_half    = int(grp["half_tp_hit"].sum())
        y_net     = grp["pnl"].sum()
        y_win_pnl = grp[grp["pnl"] > 0]["pnl"].sum()
        y_wr      = y_tp / max(y_total, 1) * 100
        print(
            f"  {year:<6} {y_total:>5} {y_tp:>7} {y_half:>6} {y_sl:>7} {y_exp:>6} "
            f" ${y_net:>+10,.2f}  ${y_win_pnl:>+10,.2f}  {y_wr:>5.1f}%"
        )

    print(f"{'─'*W}")
    print(
        f"  {'TOTAL':<6} {total:>5} {tp_hits:>7} {half_tp_total:>6} {sl_hits:>7} {expired:>6} "
        f" ${net_pnl:>+10,.2f}  ${winners_pnl:>+10,.2f}  {win_rate:>5.1f}%"
    )
    print(f"{'='*W}")

    # ── Losing trade breakdown ────────────────────────────────────────────────
    losers_df = df_t[df_t["outcome"] == -1].copy()
    if len(losers_df):
        print(f"\n{'='*W}")
        print(f"  LOSING TRADES  ({len(losers_df)} legs | "
              f"Total: ${losers_df['pnl'].sum():+,.2f} | "
              f"Avg: ${losers_df['pnl'].mean():+,.2f})")
        print(f"{'─'*W}")
        print(f"  {'Entry Date':<20} {'Leg':<6} {'Side':<5} {'Entry':>8} {'SL':>8} "
              f"{'TP':>8} {'Exit':>8} {'PnL':>10}  {'Signals'}")
        print(f"{'─'*W}")
        for _, row in losers_df.sort_values("entry_date").iterrows():
            print(
                f"  {str(row['entry_date'])[:19]:<20} {row['leg']:<6} {row['side']:<5} "
                f"{row['entry']:>8.1f} {row['sl']:>8.1f} {row['tp']:>8.1f} "
                f"{row['exit_price']:>8.1f} ${row['pnl']:>+9,.2f}  {row['signals']}"
            )
        print(f"{'─'*W}")
        print(
            f"  Largest single loss: ${losers_df['pnl'].min():+,.2f}  |  "
            f"Avg: ${losers_df['pnl'].mean():+,.2f}  |  "
            f"Total: ${losers_df['pnl'].sum():+,.2f}"
        )
        print(f"{'='*W}")
    else:
        print("\n  No losing trades in this period.")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAUUSD (Gold) multi-position backtest (up to N concurrent signals)"
    )
    parser.add_argument("--start",         default="2022-01-01")
    parser.add_argument("--end",           default="2026-01-01")
    parser.add_argument("--max_positions", type=int,   default=3,
                        help="Max concurrent signals (each signal = 2 legs in split_tp mode)")
    parser.add_argument("--lot",           type=float, default=0.04,
                        help="Total lot per signal (split evenly across legs in split_tp mode)")
    parser.add_argument("--spread",        type=float, default=SPREAD_PTS)
    parser.add_argument("--split_tp",      action="store_true",
                        help="Open 2 half-lot legs per signal: one exits at 50%% TP, one at full TP")
    parser.add_argument("--max_sl_dist",   type=float, default=0.0,
                        help="Skip trades whose total signal dollar risk exceeds this (0=off, e.g. 160)")
    parser.add_argument("--min_rr",        type=float, default=0.0,
                        help="Override min risk-reward ratio (0=use config default)")
    parser.add_argument("--min_conf",      type=int,   default=0,
                        help="Override min active confirmations (0=use config default)")
    args = parser.parse_args()

    run_multi_backtest_gold(
        start=args.start,
        end=args.end,
        max_positions=args.max_positions,
        lot=args.lot,
        spread=args.spread,
        split_tp=args.split_tp,
        max_sl_dist=args.max_sl_dist,
        min_rr_override=args.min_rr,
        min_conf_override=args.min_conf,
    )


if __name__ == "__main__":
    main()
