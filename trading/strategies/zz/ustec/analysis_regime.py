#!/usr/bin/env python3
"""
USTEC Regime Analysis
Covers: regime breakdown, MFE distribution, time-in-trade, zone quality,
signal clustering, and counterfactual comparisons.

Usage:
    python trading/strategies/zz/ustec/analysis_regime.py
    python trading/strategies/zz/ustec/analysis_regime.py --start 2023-01-01 --end 2026-06-10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.ustec.strategy import (
    make_configs, MAX_FORWARD_BARS, H4_WINDOW, M15_WINDOW,
    SPREAD_PTS, CONTRACT_SIZE, MIN_SL_PCT,
    COOLDOWN_BARS, COOLDOWN_LOSS_H,
)
from trading.strategies.zz.core.timeframe_structure import analyse_timeframes
from trading.strategies.zz.core.confirmations import check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import setup_from_analysis
from trading.shared.data_loader import get_connection

SEP  = "=" * 90
DASH = "─" * 90


def _load_ohlcv(db, table, timeframe, start, end):
    q  = (f"SELECT * FROM {table} WHERE timeframe = %s "
          f"AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC")
    df = db.fetch_dataframe(q, (timeframe, start, end))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _zone_key(zone):
    return (round(zone.bottom, 1), round(zone.top, 1))


def _run_backtest(
    df_15m, df_4h,
    tf_cfg, conf_cfg, setup_cfg,
    lot, spread,
    mode="grouped",   # "grouped" or "independent"
    ema_filter=None,  # None | "h4" | "d1" | "both"
):
    COOLDOWN_LOSS_BARS = int(COOLDOWN_LOSS_H * 4)

    # ── Precompute EMAs ───────────────────────────────────────────────────────
    df_4h = df_4h.copy()
    _ohlcv = ["open", "high", "low", "close"]
    df_4h  = df_4h[df_4h[_ohlcv].ne(df_4h[_ohlcv].shift()).any(axis=1)].reset_index(drop=True)
    df_4h["ema200"] = df_4h["close"].ewm(span=200, adjust=False).mean()

    # D1 EMA200: resample H4, shift 1 day (no lookahead)
    df_4h_ts = df_4h.set_index("timestamp")
    d1_close = df_4h_ts["close"].resample("D").last().dropna()
    d1_ema   = d1_close.ewm(span=200, adjust=False).mean().shift(1).dropna()
    # align to date (timezone-naive)
    d1_ema.index = d1_ema.index.normalize()

    n      = len(df_15m)
    warmup = max(M15_WINDOW, 30)

    open_positions = []
    closed_trades  = []
    pending_groups = {}
    group_counter  = 0

    zone_cooldown      = {}
    zone_reentry       = {}
    zone_consec_losses = {}
    zone_blacklist     = set()

    for i in range(warmup, n - 1):
        ts_now = df_15m["timestamp"].iloc[i]
        bar_h  = float(df_15m["high"].iloc[i])
        bar_l  = float(df_15m["low"].iloc[i])

        # ── leave-and-return ──────────────────────────────────────────────────
        if zone_reentry:
            tol, ready = 0.001, []
            for zk, state in zone_reentry.items():
                z_bot = state["bottom"] * (1 - tol)
                z_top = state["top"]    * (1 + tol)
                if state["phase"] == "exit":
                    if bar_h < z_bot or bar_l > z_top:
                        state["phase"] = "return"
                elif z_bot <= bar_h and bar_l <= z_top and i >= state["earliest_reentry"]:
                    ready.append(zk)
            for zk in ready:
                del zone_reentry[zk]

        # ── process open positions ────────────────────────────────────────────
        still_open = []
        for pos in open_positions:
            direction = pos["side"]
            entry     = pos["entry"]
            tp        = pos["tp"]
            tier      = pos.get("tier", 0)

            # MFE tracking
            favorable = (bar_h - entry) if direction == "buy" else (entry - bar_l)
            pos["mfe"] = max(pos.get("mfe", 0.0), favorable)

            # Trailing SL — grouped Tier 2/3 only
            if mode == "grouped" and tier in (2, 3):
                zone_tp = pos["full_tp"]
                p50 = entry + 0.50 * (zone_tp - entry)
                p30 = entry + 0.30 * (zone_tp - entry)
                if pos["trail_state"] == "none":
                    hit50 = (bar_h >= p50) if direction == "buy" else (bar_l <= p50)
                    if hit50:
                        pos["sl"]          = round(p30, 2)
                        pos["trail_state"] = "locked30"

            sl = pos["sl"]

            # TP / SL check
            outcome    = 0
            exit_price = float(df_15m["close"].iloc[i])
            if direction == "buy":
                if bar_h >= tp:   outcome = 1;  exit_price = tp
                elif bar_l <= sl: outcome = -1; exit_price = sl
            else:
                if bar_l <= tp:   outcome = 1;  exit_price = tp
                elif bar_h >= sl: outcome = -1; exit_price = sl

            bars_held = i - pos["entry_bar"]
            resolved  = (outcome != 0) or (bars_held >= MAX_FORWARD_BARS)

            if resolved:
                pnl = ((exit_price - entry) if direction == "buy" else (entry - exit_price)) * lot * CONTRACT_SIZE
                full_range = abs(pos.get("full_tp", tp) - entry)
                mfe_pct    = pos["mfe"] / full_range if full_range > 0 else 0.0
                pos.update({
                    "outcome":    outcome,
                    "exit_price": round(exit_price, 2),
                    "exit_bar":   i,
                    "exit_date":  ts_now,
                    "pnl":        round(pnl, 2),
                    "bars_held":  bars_held,
                    "mfe_pct":    round(mfe_pct, 3),
                })
                closed_trades.append(pos)

                # Zone management
                gid = pos["group_id"]
                zk  = pos["zone_key"]
                if mode == "grouped":
                    pg = pending_groups.setdefault(gid, {
                        "zone_key": zk, "zone_bottom": pos["zone_bottom"],
                        "zone_top": pos["zone_top"], "outcomes": [], "trail_states": [],
                    })
                    pg["outcomes"].append(outcome)
                    pg["trail_states"].append(pos.get("trail_state", "none"))
                    if len(pg["outcomes"]) == 3:
                        any_win     = any(o == 1 for o in pg["outcomes"])
                        any_trailed = any(t == "locked30" for t in pg["trail_states"])
                        if any_win or any_trailed:
                            zone_consec_losses[zk] = 0
                            zone_reentry[zk] = {"phase": "exit", "bottom": pg["zone_bottom"],
                                                "top": pg["zone_top"], "earliest_reentry": i + COOLDOWN_BARS}
                        else:
                            zone_cooldown[zk] = i + COOLDOWN_LOSS_BARS
                        del pending_groups[gid]
                else:
                    if outcome == 1:
                        zone_consec_losses[zk] = 0
                        zone_reentry[zk] = {"phase": "exit", "bottom": pos["zone_bottom"],
                                            "top": pos["zone_top"], "earliest_reentry": i + COOLDOWN_BARS}
                    elif outcome == -1:
                        zone_cooldown[zk] = i + COOLDOWN_LOSS_BARS
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── entry gate ────────────────────────────────────────────────────────
        open_count = len({p["group_id"] for p in open_positions}) if mode == "grouped" else len(open_positions)
        max_open   = 1 if mode == "grouped" else 3
        if open_count >= max_open or i >= n - 5:
            continue

        df_h4_w  = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        df_15m_w = df_15m.iloc[max(0, i - M15_WINDOW + 1): i + 1].reset_index(drop=True)
        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=len(df_h4_w) - 1)
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
        sl_dist     = abs(signal_price - setup.sl)
        sl          = entry_price - sl_dist if direction == "buy" else entry_price + sl_dist
        tp          = setup.tp

        if direction == "buy":
            if sl >= entry_price or tp <= entry_price:
                continue
        else:
            if sl <= entry_price or tp >= entry_price:
                continue
        if MIN_SL_PCT > 0 and abs(entry_price - sl) / entry_price * 100 < MIN_SL_PCT:
            continue

        # ── EMA lookup ────────────────────────────────────────────────────────
        h4_ema_val = float(df_h4_w["ema200"].iloc[-1]) if "ema200" in df_h4_w.columns and len(df_h4_w) else None
        today_norm = ts_now.normalize()
        d1_prev    = d1_ema[d1_ema.index < today_norm]
        d1_ema_val = float(d1_prev.iloc[-1]) if len(d1_prev) > 0 else None

        # ── EMA filter ────────────────────────────────────────────────────────
        if ema_filter in ("h4", "both") and h4_ema_val is not None:
            if direction == "buy"  and entry_price < h4_ema_val: continue
            if direction == "sell" and entry_price > h4_ema_val: continue
        if ema_filter in ("d1", "both") and d1_ema_val is not None:
            if direction == "buy"  and entry_price < d1_ema_val: continue
            if direction == "sell" and entry_price > d1_ema_val: continue

        # ── Regime tag ────────────────────────────────────────────────────────
        regime_h4 = "up"   if (h4_ema_val and entry_price > h4_ema_val) else "down"
        regime_d1 = "up"   if (d1_ema_val and entry_price > d1_ema_val) else "down"

        gid = group_counter
        group_counter += 1

        base = dict(
            group_id=gid, entry_bar=i, entry_date=ts_now, side=direction,
            entry=round(entry_price, 2), sl=round(sl, 2), full_tp=round(tp, 2),
            zone_key=zk, zone_bottom=active_zone.bottom, zone_top=active_zone.top,
            trail_state="none", mfe=0.0, regime_h4=regime_h4, regime_d1=regime_d1,
            conf_count=conf.count, signals="|".join(conf.signals),
            outcome=None, exit_price=None, exit_bar=None, exit_date=None,
            pnl=None, bars_held=None, mfe_pct=None,
        )

        if mode == "grouped":
            r = tp - entry_price
            for tier, tp_this in ((1, entry_price + 0.50 * r), (2, entry_price + 0.70 * r), (3, entry_price + 0.70 * r)):
                open_positions.append({**base, "tier": tier, "tp": round(tp_this, 2)})
        else:
            open_positions.append({**base, "tier": 0, "tp": round(tp, 2)})

    # force-close
    last_close = float(df_15m["close"].iloc[-1])
    last_ts    = df_15m["timestamp"].iloc[-1]
    for pos in open_positions:
        entry = pos["entry"]; direction = pos["side"]
        pnl   = ((last_close - entry) if direction == "buy" else (entry - last_close)) * lot * CONTRACT_SIZE
        fr    = abs(pos.get("full_tp", pos["tp"]) - entry)
        pos.update(dict(outcome=0, exit_price=round(last_close, 2), exit_bar=len(df_15m)-1,
                        exit_date=last_ts, pnl=round(pnl, 2),
                        bars_held=len(df_15m)-1-pos["entry_bar"],
                        mfe_pct=round(pos["mfe"]/fr if fr > 0 else 0, 3)))
        closed_trades.append(pos)

    return closed_trades


def _build_groups(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame()

    def _agg(g):
        t1  = g[g["tier"] == 1]
        t23 = g[g["tier"].isin([2, 3])]
        r   = g.iloc[0]
        return pd.Series({
            "entry_date":  g["entry_date"].min(),
            "exit_date":   g["exit_date"].max(),
            "side":        r["side"],
            "entry":       r["entry"],
            "full_tp":     r["full_tp"],
            "net_pnl":     g["pnl"].sum(),
            "t1_win":      bool(len(t1)  and (t1["outcome"] == 1).any()),
            "any_full_tp": bool(len(t23) and (t23["outcome"] == 1).any()),
            "all_sl":      bool((g["outcome"] == -1).all()),
            "profitable":  bool(g["pnl"].sum() > 0),
            "regime_h4":   r["regime_h4"],
            "regime_d1":   r["regime_d1"],
            "conf_count":  r["conf_count"],
            "signals":     r["signals"],
            "mfe_pct":     g["mfe_pct"].max(),
            "bars_held":   (g["exit_bar"].max() or 0) - (g["entry_bar"].min() or 0),
        })

    return df.groupby("group_id").apply(_agg).reset_index()


# ── Analysis sections ─────────────────────────────────────────────────────────

def _regime_table(groups, label="H4"):
    col = "regime_h4" if label == "H4" else "regime_d1"
    print(f"\n  Regime basis: {label} EMA200  (up = entry_price > EMA200)")
    print(f"  {'Regime':<10} {'Groups':>7} {'T1Win%':>8} {'FullTP%':>8} {'AllSL%':>8} {'Profit%':>9}  {'Net PnL':>12}  {'WR%':>6}")
    print(f"  {DASH[:80]}")
    totals = {"n":0,"t1":0,"ftp":0,"sl":0,"prof":0,"pnl":0.0}
    for regime in ("up", "down"):
        sub = groups[groups[col] == regime]
        if sub.empty:
            continue
        n    = len(sub)
        t1   = sub["t1_win"].sum()
        ftp  = sub["any_full_tp"].sum()
        sl   = sub["all_sl"].sum()
        prof = sub["profitable"].sum()
        pnl  = sub["net_pnl"].sum()
        print(f"  {regime:<10} {n:>7} {t1/n*100:>7.1f}% {ftp/n*100:>7.1f}% {sl/n*100:>7.1f}% {prof/n*100:>8.1f}%  ${pnl:>+10,.2f}  {prof/n*100:>5.1f}%")
        totals["n"]+=n; totals["t1"]+=t1; totals["ftp"]+=ftp; totals["sl"]+=sl; totals["prof"]+=prof; totals["pnl"]+=pnl
    n = totals["n"]
    print(f"  {DASH[:80]}")
    print(f"  {'TOTAL':<10} {n:>7} {totals['t1']/n*100:>7.1f}% {totals['ftp']/n*100:>7.1f}% {totals['sl']/n*100:>7.1f}% {totals['prof']/n*100:>8.1f}%  ${totals['pnl']:>+10,.2f}  {totals['prof']/n*100:>5.1f}%")

    sl_groups = groups[groups["all_sl"]]
    print(f"\n  All-SL groups by {label} regime:")
    for regime in ("up", "down"):
        cnt = (sl_groups[col] == regime).sum()
        pct = cnt / max(len(sl_groups), 1) * 100
        print(f"    {regime:>4}: {cnt:>3} groups  ({pct:.1f}% of all losing groups)")


def _mfe_distribution(groups):
    sl_groups = groups[groups["all_sl"]]
    print(f"\n  MFE distribution for ALL-SL groups ({len(sl_groups)} groups)")
    print(f"  (How far price moved favourably before reversing to SL)\n")
    buckets = [
        ("Never reached 5%  ", lambda x: x < 0.05),
        ("Reached  5–14%    ", lambda x: 0.05 <= x < 0.15),
        ("Reached 15–24%    ", lambda x: 0.15 <= x < 0.25),
        ("Reached 25–49%    ", lambda x: 0.25 <= x < 0.50),
        ("Reached 50%+      ", lambda x: x >= 0.50),
    ]
    for label, fn in buckets:
        mask  = sl_groups["mfe_pct"].apply(fn)
        cnt   = mask.sum()
        pct   = cnt / max(len(sl_groups), 1) * 100
        bar   = "█" * int(pct / 2)
        print(f"    {label}  {cnt:>4} ({pct:>5.1f}%)  {bar}")

    med = sl_groups["mfe_pct"].median() * 100
    avg = sl_groups["mfe_pct"].mean()   * 100
    print(f"\n    Median MFE: {med:.1f}% of range  |  Avg MFE: {avg:.1f}% of range")


def _time_analysis(groups):
    hours = lambda bars: bars * 15 / 60
    win  = groups[groups["profitable"]]
    lose = groups[~groups["profitable"]]
    print(f"\n  {'Category':<22} {'Groups':>7} {'Avg bars':>9} {'Avg hours':>10} {'Median h':>9}")
    print(f"  {DASH[:65]}")
    for label, sub in (("All SL (losing)", groups[groups["all_sl"]]),
                        ("Any profit",      win),
                        ("ALL groups",      groups)):
        if sub.empty:
            continue
        bh = sub["bars_held"].dropna()
        print(f"  {label:<22} {len(sub):>7} {bh.mean():>9.1f} {hours(bh.mean()):>10.1f} {hours(bh.median()):>9.1f}")


def _zone_quality(groups):
    print(f"\n  Win rate by confirmation count × regime (H4 EMA200)\n")
    print(f"  {'Confs':>6} {'Regime':>8} {'n':>6} {'Profitable':>11} {'WR%':>6}  {'Net PnL':>12}  {'Avg$/trade':>11}")
    print(f"  {DASH[:72]}")
    for confs in sorted(groups["conf_count"].unique()):
        for regime in ("up", "down"):
            sub = groups[(groups["conf_count"] == confs) & (groups["regime_h4"] == regime)]
            if sub.empty:
                continue
            n    = len(sub)
            prof = sub["profitable"].sum()
            pnl  = sub["net_pnl"].sum()
            avg  = pnl / n
            flag = "  [LOW SAMPLE]" if n < 15 else ""
            print(f"  {confs:>6}  {regime:>8} {n:>6} {prof:>11} {prof/n*100:>5.1f}%  ${pnl:>+10,.2f}  ${avg:>+9,.2f}{flag}")

    # ── Regime × confirmation interaction ─────────────────────────────────────
    print(f"\n  Regime × confirmation rule test")
    print(f"  Rule: keep up+1conf and down+2+conf; drop up+2+conf and down+1conf\n")
    print(f"  {'Bucket':<30} {'n':>6} {'WR%':>6}  {'Net PnL':>12}  {'Avg$/trade':>11}")
    print(f"  {DASH[:72]}")

    buckets = [
        ("KEEP  up   + 1 conf",  (groups["regime_h4"] == "up")   & (groups["conf_count"] == 1)),
        ("KEEP  down + 2+ conf", (groups["regime_h4"] == "down") & (groups["conf_count"] >= 2)),
        ("DROP  up   + 2+ conf", (groups["regime_h4"] == "up")   & (groups["conf_count"] >= 2)),
        ("DROP  down + 1 conf",  (groups["regime_h4"] == "down") & (groups["conf_count"] == 1)),
    ]
    keep_pnl = drop_pnl = 0.0
    keep_n   = drop_n   = 0
    keep_prof = drop_prof = 0
    for label, mask in buckets:
        sub = groups[mask]
        if sub.empty:
            print(f"  {label:<30} {'0':>6} {'—':>6}  {'$0':>12}  {'$0':>11}")
            continue
        n    = len(sub)
        prof = sub["profitable"].sum()
        pnl  = sub["net_pnl"].sum()
        avg  = pnl / n
        flag = "  [LOW SAMPLE]" if n < 15 else ""
        print(f"  {label:<30} {n:>6} {prof/n*100:>5.1f}%  ${pnl:>+10,.2f}  ${avg:>+9,.2f}{flag}")
        if label.startswith("KEEP"):
            keep_pnl  += pnl;  keep_n  += n;  keep_prof += prof
        else:
            drop_pnl  += pnl;  drop_n  += n;  drop_prof += prof

    print(f"  {DASH[:72]}")
    if keep_n:
        print(f"  {'KEEP total':<30} {keep_n:>6} {keep_prof/keep_n*100:>5.1f}%  ${keep_pnl:>+10,.2f}  ${keep_pnl/keep_n:>+9,.2f}")
    if drop_n:
        print(f"  {'DROP total':<30} {drop_n:>6} {drop_prof/drop_n*100:>5.1f}%  ${drop_pnl:>+10,.2f}  ${drop_pnl/drop_n:>+9,.2f}")


def _clustering(groups, df_15m, bad_periods):
    print(f"\n  Buy-signal clustering during bad stretches")
    print(f"  (All-SL buy groups firing within 96 bars = 24h of each other)\n")
    buy_sl = groups[(groups["all_sl"]) & (groups["side"] == "buy")].sort_values("entry_date").copy()
    buy_sl["entry_bar_approx"] = buy_sl["entry_date"].apply(
        lambda ts: int(df_15m[df_15m["timestamp"] <= ts].index[-1]) if len(df_15m[df_15m["timestamp"] <= ts]) else 0
    )
    for period_label, start, end in bad_periods:
        sub = buy_sl[(buy_sl["entry_date"] >= pd.Timestamp(start)) &
                     (buy_sl["entry_date"] <= pd.Timestamp(end))].reset_index(drop=True)
        print(f"  Period: {period_label}  ({len(sub)} all-SL buy groups)")
        if sub.empty:
            print("    (no data)\n")
            continue
        runs, cur_run = [], [sub.iloc[0]["entry_date"]]
        for k in range(1, len(sub)):
            gap = (sub.iloc[k]["entry_bar_approx"] - sub.iloc[k-1]["entry_bar_approx"])
            if gap <= 96:
                cur_run.append(sub.iloc[k]["entry_date"])
            else:
                if len(cur_run) >= 2:
                    runs.append(cur_run)
                cur_run = [sub.iloc[k]["entry_date"]]
        if len(cur_run) >= 2:
            runs.append(cur_run)

        for run in runs:
            span_h = (run[-1] - run[0]).total_seconds() / 3600
            print(f"    Run of {len(run):>2}: {str(run[0])[:16]} → {str(run[-1])[:16]}  ({span_h:.0f}h span)")
        if not runs:
            print("    No clustered runs found.")
        print()


def _counterfactual_table(results):
    print(f"\n  {'Mode':<36} {'Groups':>7} {'AllSL%':>8} {'Net PnL':>12} {'WR%':>6}")
    print(f"  {DASH[:73]}")
    for label, groups in results:
        if groups.empty:
            print(f"  {label:<36}  (no data)")
            continue
        n    = len(groups)
        sl   = groups["all_sl"].sum() if "all_sl" in groups.columns else groups[groups["pnl"] < 0]["pnl"].count()
        prof = groups["profitable"].sum() if "profitable" in groups.columns else (groups["pnl"] > 0).sum()
        pnl  = groups["net_pnl"].sum()   if "net_pnl"    in groups.columns else groups["pnl"].sum()
        sl_pct = sl / n * 100
        wr     = prof / n * 100
        print(f"  {label:<36} {n:>7} {sl_pct:>7.1f}% ${pnl:>+10,.2f}  {wr:>5.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2026-06-10")
    parser.add_argument("--lot",   type=float, default=0.04)
    args = parser.parse_args()

    tf_cfg, conf_cfg, setup_cfg = make_configs()

    db = get_connection()
    db.database = "ustech_ohlcv"
    db.connect()
    print(f"Loading data {args.start} → {args.end} ...")
    df_15m = _load_ohlcv(db, "ustech_ohlcv", "15min", args.start, args.end)
    df_4h  = _load_ohlcv(db, "ustech_ohlcv", "4H",    args.start, args.end)
    try: db.connection.close()
    except: pass

    if df_15m.empty or df_4h.empty:
        print("ERROR: no data."); return
    print(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")

    common = dict(df_15m=df_15m, df_4h=df_4h, tf_cfg=tf_cfg, conf_cfg=conf_cfg,
                  setup_cfg=setup_cfg, lot=args.lot, spread=SPREAD_PTS)

    # ── Run all 5 modes ───────────────────────────────────────────────────────
    MODES = [
        ("Grouped T1@50% T2/3@70%+lock30",      dict(mode="grouped",      ema_filter=None)),
        ("Grouped + H4 EMA200 filter",           dict(mode="grouped",      ema_filter="h4")),
        ("Grouped + D1 EMA200 filter",           dict(mode="grouped",      ema_filter="d1")),
        ("Grouped + H4+D1 EMA200 (both)",        dict(mode="grouped",      ema_filter="both")),
        ("Independent 3-pos @ 100% TP",          dict(mode="independent",  ema_filter=None)),
        ("Independent + H4 EMA200 filter",       dict(mode="independent",  ema_filter="h4")),
    ]

    all_results = {}
    for label, kwargs in MODES:
        print(f"\nRunning: {label} ...")
        trades = _run_backtest(**common, **kwargs)
        if kwargs["mode"] == "grouped":
            groups = _build_groups(trades)
        else:
            df = pd.DataFrame(trades)
            if not df.empty:
                df["net_pnl"]    = df["pnl"]
                df["all_sl"]     = df["outcome"] == -1
                df["profitable"] = df["pnl"] > 0
                df["t1_win"]     = df["outcome"] == 1
                df["any_full_tp"]= df["outcome"] == 1
            groups = df if not df.empty else pd.DataFrame()
        all_results[label] = groups

    # ── Print results ─────────────────────────────────────────────────────────
    base_groups = all_results["Grouped T1@50% T2/3@70%+lock30"]

    print(f"\n\n{SEP}")
    print(f"  SECTION 1 — COUNTERFACTUAL SUMMARY  ({args.start} → {args.end})")
    print(SEP)
    _counterfactual_table([(lbl, all_results[lbl]) for lbl, _ in MODES])

    print(f"\n\n{SEP}")
    print(f"  SECTION 2 — REGIME BREAKDOWN  (Grouped current system)")
    print(SEP)
    _regime_table(base_groups, "H4")
    print()
    _regime_table(base_groups, "D1")

    print(f"\n\n{SEP}")
    print(f"  SECTION 3 — MFE DISTRIBUTION  (losing groups — how far price moved before SL)")
    print(SEP)
    _mfe_distribution(base_groups)

    print(f"\n\n{SEP}")
    print(f"  SECTION 4 — TIME IN TRADE  (bars × 15min = hours)")
    print(SEP)
    _time_analysis(base_groups)

    print(f"\n\n{SEP}")
    print(f"  SECTION 5 — ZONE QUALITY  (confirmation count as quality proxy)")
    print(SEP)
    _zone_quality(base_groups)

    print(f"\n\n{SEP}")
    print(f"  SECTION 6 — SIGNAL CLUSTERING  (buy SL runs in bad stretches)")
    print(SEP)
    bad_periods = [
        ("Aug 2023",       "2023-08-01", "2023-08-31"),
        ("Mar–Apr 2026",   "2026-03-01", "2026-04-30"),
    ]
    _clustering(base_groups, df_15m, bad_periods)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
