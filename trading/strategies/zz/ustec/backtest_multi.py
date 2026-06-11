#!/usr/bin/env python3
"""
USTEC Zone-to-Zone — multi-group backtest.

Each signal fires a GROUP of 3 positions:
  Tier 1 : TP at 50% of range  |  normal SL
  Tier 2 : TP at 70% of range  |  SL → 30% mark when 50% hit
  Tier 3 : same as Tier 2

Usage:
    python trading/strategies/zz/ustec/backtest_multi.py
    python trading/strategies/zz/ustec/backtest_multi.py --start 2025-01-01 --end 2025-12-31
    python trading/strategies/zz/ustec/backtest_multi.py --max_groups 1 --lot 0.04
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.ustec.strategy import (
    make_configs,
    MAX_FORWARD_BARS,
    H4_WINDOW,
    M15_WINDOW,
    SPREAD_PTS,
    CONTRACT_SIZE,
    FIXED_LOTS,
    MIN_SL_PCT,
    COOLDOWN_BARS,
    COOLDOWN_LOSS_H,
)
from trading.strategies.zz.core.timeframe_structure import analyse_timeframes
from trading.strategies.zz.core.confirmations import check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import setup_from_analysis
from trading.shared.data_loader import get_connection


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


def run_multi_backtest(
    start: str = "2025-12-01",
    end: str = "2026-06-10",
    max_groups: int = 1,
    lot: float = 0.04,
    spread: float = SPREAD_PTS,
) -> None:
    tf_cfg, conf_cfg, setup_cfg = make_configs()
    COOLDOWN_LOSS_BARS = int(COOLDOWN_LOSS_H * 4)

    db = get_connection()
    db.database = "ustech_ohlcv"
    db.connect()

    print(f"\n{'='*70}")
    print(f"  USTEC Multi-Group Backtest  (3 positions per signal)")
    print(f"  Period        : {start}  →  {end}")
    print(f"  Lot size      : {lot}  |  Max concurrent groups: {max_groups}")
    print(f"  Spread        : {spread} pts  |  Contract size: ${CONTRACT_SIZE}/pt/lot")
    print(f"  Tier 1        : TP @ 50% of range, normal SL")
    print(f"  Tier 2 & 3    : TP @ 70% of range, SL → 30% mark when 50% hit")
    print(f"{'='*70}")

    print("\nLoading data ...")
    df_15m = _load_ohlcv(db, "ustech_ohlcv", "15min", start, end)
    df_4h  = _load_ohlcv(db, "ustech_ohlcv", "4H",    start, end)

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

    open_positions:     list[dict] = []
    closed_trades:      list[dict] = []

    zone_cooldown:  dict = {}
    zone_reentry:   dict = {}
    pending_groups: dict = {}   # group_id → {zone info, outcomes, trail_states}
    group_counter:  int  = 0

    for i in range(warmup, n - 1):
        ts_now = df_15m["timestamp"].iloc[i]
        bar_h  = float(df_15m["high"].iloc[i])
        bar_l  = float(df_15m["low"].iloc[i])

        # ── Step 1: leave-and-return tracking ────────────────────────────────
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

        # ── Step 2: process open positions ───────────────────────────────────
        still_open: list[dict] = []
        for pos in open_positions:
            direction = pos["side"]
            entry     = pos["entry"]
            tp        = pos["tp"]
            tier      = pos["tier"]

            # Trailing SL logic for Tier 2 & 3:
            # Once price hits 50% mark → SL moves to 30% mark (locks in 30% profit).
            # Price then either continues to 70% TP or exits at 30% — never a loss.
            if tier in (2, 3):
                zone_tp = pos["full_tp"]   # original zone-to-zone TP (reference range)
                p50 = entry + 0.50 * (zone_tp - entry)
                p30 = entry + 0.30 * (zone_tp - entry)
                trail = pos["trail_state"]

                if trail == "none":
                    hit50 = bar_h >= p50 if direction == "buy" else bar_l <= p50
                    if hit50:
                        pos["sl"]          = round(p30, 2)
                        pos["trail_state"] = "locked30"

            sl = pos["sl"]

            # Check TP / SL
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
                if direction == "buy":
                    pnl = (exit_price - entry) * lot * CONTRACT_SIZE
                else:
                    pnl = (entry - exit_price) * lot * CONTRACT_SIZE

                pos.update({
                    "outcome":    outcome,
                    "exit_price": round(exit_price, 2),
                    "exit_bar":   i,
                    "exit_date":  ts_now,
                    "pnl":        round(pnl, 2),
                })
                closed_trades.append(pos)

                # Accumulate group completion state
                gid = pos["group_id"]
                if gid not in pending_groups:
                    pending_groups[gid] = {
                        "zone_key":    pos["zone_key"],
                        "zone_bottom": pos["zone_bottom"],
                        "zone_top":    pos["zone_top"],
                        "outcomes":    [],
                        "trail_states": [],
                    }
                pending_groups[gid]["outcomes"].append(outcome)
                pending_groups[gid]["trail_states"].append(pos.get("trail_state", "none"))

                # Process zone management when all 3 positions in group close
                if len(pending_groups[gid]["outcomes"]) == 3:
                    zk          = pending_groups[gid]["zone_key"]
                    outcomes    = pending_groups[gid]["outcomes"]
                    trail_states = pending_groups[gid]["trail_states"]

                    any_win     = any(o == 1 for o in outcomes)
                    any_trailed = any(ts == "locked30" for ts in trail_states)

                    if any_win or any_trailed:
                        # Trade went in our direction — leave-and-return
                        zone_reentry[zk] = {
                            "phase":            "exit",
                            "bottom":           pending_groups[gid]["zone_bottom"],
                            "top":              pending_groups[gid]["zone_top"],
                            "earliest_reentry": i + COOLDOWN_BARS,
                        }
                    else:
                        # All 3 hit original SL — genuine bad trade
                        zone_cooldown[zk] = i + COOLDOWN_LOSS_BARS
                    del pending_groups[gid]
            else:
                still_open.append(pos)

        open_positions = still_open

        # ── Step 3: look for a new signal group ──────────────────────────────
        open_group_count = len({p["group_id"] for p in open_positions})
        if open_group_count >= max_groups or i >= n - 5:
            continue

        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        m15_start = max(0, i - M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)
        h4_up_to  = len(df_h4_w) - 1

        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=h4_up_to)
        if tf_result["signal"] == "neutral":
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = _zone_key(active_zone)

        if zone_cooldown.get(zk, -1) >= i:
            continue
        if zk in zone_reentry:
            continue
        if zk in {p["zone_key"] for p in open_positions}:
            continue

        conf = check_confirmations_at_last_bar(df_15m_w, active_zone, direction, conf_cfg)
        if not conf.confirmed:
            continue

        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            continue

        next_open   = float(df_15m["open"].iloc[i + 1])
        entry_price = next_open + spread if direction == "buy" else next_open - spread

        sl_dist = abs(signal_price - setup.sl)
        sl      = entry_price - sl_dist if direction == "buy" else entry_price + sl_dist
        tp      = setup.tp

        if direction == "buy":
            if sl >= entry_price or tp <= entry_price:
                continue
        else:
            if sl <= entry_price or tp >= entry_price:
                continue

        if MIN_SL_PCT > 0:
            if abs(entry_price - sl) / entry_price * 100 < MIN_SL_PCT:
                continue

        range_pts = tp - entry_price            # signed range (negative for sells)
        tp_t1  = entry_price + 0.50 * range_pts  # Tier 1  → 50% of range
        tp_t23 = entry_price + 0.70 * range_pts  # Tier 2/3 → 70% of range

        gid = group_counter
        group_counter += 1

        for tier in (1, 2, 3):
            open_positions.append({
                "group_id":    gid,
                "tier":        tier,
                "entry_bar":   i,
                "entry_date":  ts_now,
                "exit_date":   None,
                "side":        direction,
                "entry":       round(entry_price, 2),
                "sl":          round(sl, 2),
                "tp":          round(tp_t1 if tier == 1 else tp_t23, 2),
                "full_tp":     round(tp, 2),   # zone-to-zone reference (for SL trail calc)
                "lot":         lot,
                "zone_key":    zk,
                "zone_bottom": active_zone.bottom,
                "zone_top":    active_zone.top,
                "trail_state": "none",
                "outcome":     None,
                "exit_price":  None,
                "exit_bar":    None,
                "pnl":         None,
                "confirmations": conf.count,
                "signals":       "|".join(conf.signals),
                "h4_bias":       tf_result.get("h4_bias", ""),
            })

    # Force-close any positions still open at end of data
    last_idx   = n - 1
    last_close = float(df_15m["close"].iloc[last_idx])
    last_ts    = df_15m["timestamp"].iloc[last_idx]
    for pos in open_positions:
        direction = pos["side"]
        entry     = pos["entry"]
        pnl = (last_close - entry) * lot * CONTRACT_SIZE if direction == "buy" else (entry - last_close) * lot * CONTRACT_SIZE
        pos.update({
            "outcome":    0,
            "exit_price": round(last_close, 2),
            "exit_bar":   last_idx,
            "exit_date":  last_ts,
            "pnl":        round(pnl, 2),
        })
        closed_trades.append(pos)

    if not closed_trades:
        print("\nNo trades generated. Try widening the date range.")
        return

    df_t = pd.DataFrame(closed_trades)

    # ── Build group-level summary ─────────────────────────────────────────────
    def _group_stats(g):
        t1   = g[g["tier"] == 1]
        t23  = g[g["tier"].isin([2, 3])]
        return pd.Series({
            "entry_date":  g["entry_date"].min(),
            "side":        g["side"].iloc[0],
            "entry":       g["entry"].iloc[0],
            "orig_sl":     g[g["tier"] == 2]["sl"].iloc[0] if len(g[g["tier"] == 2]) else g["sl"].iloc[0],
            "full_tp":     g["full_tp"].iloc[0],
            "net_pnl":     g["pnl"].sum(),
            "t1_win":      bool(len(t1) and (t1["outcome"] == 1).any()),
            "any_full_tp": bool(len(t23) and (t23["outcome"] == 1).any()),
            "all_sl":      bool((g["outcome"] == -1).all()),
            "signals":     g["signals"].iloc[0],
        })

    df_groups = df_t.groupby("group_id").apply(_group_stats).reset_index()
    df_groups["entry_month"] = pd.to_datetime(df_groups["entry_date"]).dt.to_period("M")

    total_groups  = len(df_groups)
    full_tp_count = int(df_groups["any_full_tp"].sum())
    t1_wins       = int(df_groups["t1_win"].sum())
    all_sl_count  = int(df_groups["all_sl"].sum())
    any_profit    = int((df_groups["net_pnl"] > 0).sum())
    net_pnl       = df_groups["net_pnl"].sum()
    winners_pnl   = df_groups[df_groups["net_pnl"] > 0]["net_pnl"].sum()
    losers_pnl    = df_groups[df_groups["net_pnl"] < 0]["net_pnl"].sum()
    avg_win       = df_groups[df_groups["net_pnl"] > 0]["net_pnl"].mean() if any_profit else 0.0
    avg_loss      = df_groups[df_groups["net_pnl"] < 0]["net_pnl"].mean() if all_sl_count else 0.0
    win_rate      = any_profit / max(total_groups, 1) * 100

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OVERALL SUMMARY  (1 group = 1 signal = 3 positions)")
    print(f"{'='*70}")
    print(f"  Signal groups    : {total_groups}  ({len(df_t)} individual positions)")
    print(f"  Tier 1 wins      : {t1_wins}  ({t1_wins/max(total_groups,1)*100:.1f}%)  [hit 50% TP target]")
    print(f"  Full TP (T2/T3)  : {full_tp_count}  ({full_tp_count/max(total_groups,1)*100:.1f}%)  [T2 or T3 reached 70% TP]")
    print(f"  All 3 hit SL     : {all_sl_count}  ({all_sl_count/max(total_groups,1)*100:.1f}%)  [genuine bad trades]")
    print(f"  Groups profitable: {any_profit}  ({win_rate:.1f}%)")
    print(f"  Net PnL          : ${net_pnl:+,.2f}")
    print(f"  Winners only     : ${winners_pnl:+,.2f}")
    print(f"  Losers total     : ${losers_pnl:+,.2f}")
    print(f"  Avg winning grp  : ${avg_win:+,.2f}")
    print(f"  Avg losing grp   : ${avg_loss:+,.2f}")
    print()

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    W = 102
    print(f"{'='*W}")
    print(f"  MONTHLY BREAKDOWN  (per signal group | Net = all 3 positions combined)")
    print(f"{'─'*W}")
    print(f"  {'Month':<10} {'Groups':>6} {'T1Win':>6} {'FullTP':>7} {'AllSL':>6} {'Profit%':>8}   {'Net PnL':>10}  {'Win-Only':>10}  {'WR%':>6}")
    print(f"{'─'*W}")

    for month, grp in df_groups.groupby("entry_month"):
        m_total   = len(grp)
        m_t1      = int(grp["t1_win"].sum())
        m_ftp     = int(grp["any_full_tp"].sum())
        m_allsl   = int(grp["all_sl"].sum())
        m_prof    = int((grp["net_pnl"] > 0).sum())
        m_net     = grp["net_pnl"].sum()
        m_win_pnl = grp[grp["net_pnl"] > 0]["net_pnl"].sum()
        m_wr      = m_prof / max(m_total, 1) * 100
        print(
            f"  {str(month):<10} {m_total:>6} {m_t1:>6} {m_ftp:>7} {m_allsl:>6} {m_prof:>6}/{m_total:<4}"
            f"  ${m_net:>+9,.2f}  ${m_win_pnl:>+9,.2f}  {m_wr:>5.1f}%"
        )

    print(f"{'─'*W}")
    print(
        f"  {'TOTAL':<10} {total_groups:>6} {t1_wins:>6} {full_tp_count:>7} {all_sl_count:>6} {any_profit:>6}/{total_groups:<4}"
        f"  ${net_pnl:>+9,.2f}  ${winners_pnl:>+9,.2f}  {win_rate:>5.1f}%"
    )
    print(f"{'='*W}")

    # ── Losing groups (all 3 hit SL) ──────────────────────────────────────────
    losing_groups = df_groups[df_groups["all_sl"]].copy()
    if len(losing_groups):
        print(f"\n{'='*W}")
        print(f"  LOSING GROUPS  ({len(losing_groups)} groups where all 3 positions hit SL)")
        print(f"{'─'*W}")
        print(f"  {'Entry Date':<20} {'Side':<5} {'Entry':>8} {'SL':>8} {'FullTP':>8}  {'T1':>9}  {'T2':>9}  {'T3':>9}  {'Net':>9}  Signals")
        print(f"{'─'*W}")

        for _, row in losing_groups.sort_values("entry_date").iterrows():
            gid  = row["group_id"]
            gpos = df_t[df_t["group_id"] == gid].sort_values("tier")
            t1p  = float(gpos[gpos["tier"] == 1]["pnl"].sum())
            t2p  = float(gpos[gpos["tier"] == 2]["pnl"].sum())
            t3p  = float(gpos[gpos["tier"] == 3]["pnl"].sum())
            net  = t1p + t2p + t3p
            print(
                f"  {str(row['entry_date'])[:19]:<20} {row['side']:<5} "
                f"{row['entry']:>8.1f} {row['orig_sl']:>8.1f} {row['full_tp']:>8.1f} "
                f" ${t1p:>+8,.2f}  ${t2p:>+8,.2f}  ${t3p:>+8,.2f}  ${net:>+8,.2f}  {row['signals']}"
            )

        print(f"{'─'*W}")
        print(
            f"  Largest group loss : ${losing_groups['net_pnl'].min():+,.2f}  |  "
            f"Avg : ${losing_groups['net_pnl'].mean():+,.2f}  |  "
            f"Total : ${losing_groups['net_pnl'].sum():+,.2f}"
        )
        print(f"{'='*W}")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="USTEC multi-group backtest (3-tier positions per signal)"
    )
    parser.add_argument("--start",          default="2025-12-01")
    parser.add_argument("--end",            default="2026-06-10")
    parser.add_argument("--max_groups",     type=int,   default=1,
                        help="Max concurrent signal groups (each group = 3 positions)")
    parser.add_argument("--lot",    type=float, default=0.04)
    parser.add_argument("--spread", type=float, default=SPREAD_PTS)
    args = parser.parse_args()

    run_multi_backtest(
        start=args.start,
        end=args.end,
        max_groups=args.max_groups,
        lot=args.lot,
        spread=args.spread,
    )


if __name__ == "__main__":
    main()
