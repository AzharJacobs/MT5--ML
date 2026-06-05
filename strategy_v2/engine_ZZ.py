"""
engine_ZZ.py — Gold-specific (XAUUSD) Zone-to-Zone backtest engine.

Forked from backtest/engine_zones.py run_backtest().  The bar-by-bar loop is
the only code that cannot be injected into without re-implementing it; every
strategy object (zones, timeframes, confirmations, trade_setup) is still
imported unchanged from strategy/.

Three gold-specific modifications vs the baseline engine:

  Fix 1 (failed_zone_filter)
    After outcome == -1, add active_zone.zone_id to failed_zones (set).
    The cooldown gate permanently rejects any zid in failed_zones.
    When disabled, falls back to the legacy 48-bar COOLDOWN_LOSS.

  Fix 2 (active_signals)
    After check_confirmations_at_last_bar(), filter conf.signals to
    cfg.active_signals to get active_count.  Entry requires
    active_count >= cfg.min_confirmations.  All fired signals are still
    stored in 'signals_all' for post-analysis; 'signals' holds only the
    active subset that gated entry.

  Fix 3a (min_zone_atr_frac)
    After Step 2 returns an active zone, compute H4 ATR(14) once and skip
    the zone if its height < cfg.min_zone_atr_frac * h4_atr.

  Fix 3b (sl_atr_buffer)
    After SL distance is rebased to actual entry, widen sl_dist by
    cfg.sl_atr_buffer * h4_atr — same ATR value as fix 3a.

Usage (via scripts/backtest_gold_ZZ.py):
    python scripts/backtest_gold_ZZ.py --start 2023-01-01 --end 2024-01-01
    python scripts/backtest_gold_ZZ.py --min_zone_atr_frac 0.20 --sl_atr_buffer 0.15
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ── Reuse helpers and constants from the baseline engine ─────────────────────
from backtest.engine_zones import (
    _load_ohlcv,
    _zone_key,
    _lot_size,
    SYMBOL_CONFIG,
    RISK_PCT,
    COOLDOWN_LOSS,
)

# ── All strategy modules imported unchanged ───────────────────────────────────
from strategy.zones import ZoneConfig
from strategy.timeframe_structure import TFConfig, analyse_timeframes
from strategy.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from strategy.trade_setup import TradeSetupConfig, setup_from_analysis
from backtest.report import print_report, save_report
from backtest.chart_market_structure import plot_trades
from data.loader import get_connection

from strategy_v2.config_ZZ import GoldZZConfig


# ── ATR helper ────────────────────────────────────────────────────────────────

def _h4_atr14(df_h4_w: pd.DataFrame) -> float:
    """14-period ATR from the current H4 window (classic Wilder TR)."""
    hi = df_h4_w["high"].values
    lo = df_h4_w["low"].values
    cl = df_h4_w["close"].values
    if len(hi) < 2:
        return float("nan")
    tr = np.maximum(
        hi[1:] - lo[1:],
        np.maximum(np.abs(hi[1:] - cl[:-1]),
                   np.abs(lo[1:] - cl[:-1])),
    )
    n = min(14, len(tr))
    return float(np.mean(tr[-n:]))


# ── Constants ─────────────────────────────────────────────────────────────────

_H4_WINDOW  = 150
_M15_WINDOW = 80
_SYMBOL     = "xauusd"
_DB_NAME, _TABLE, _CONTRACT = SYMBOL_CONFIG[_SYMBOL]


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest_gold(
    cfg: GoldZZConfig,
    start: str = "2023-01-01",
    end:   str = "2024-01-01",
    cash:  float = 10_000.0,
    save_path: Optional[str] = None,
    chart: bool = False,
) -> tuple:
    """
    Run the gold-specific Zone-to-Zone backtest for [start, end).

    Returns (metrics: dict, df_trades: pd.DataFrame).
    """
    eff_spread = cfg.spread

    db = get_connection()
    db.database = _DB_NAME
    db.connect()

    print(f"\nSymbol : XAUUSD (gold v2)  |  DB: {_DB_NAME}  "
          f"|  Contract: ${_CONTRACT}/pt/lot  |  Spread: {eff_spread} pts")
    print(f"Loading 15M data  {start} → {end} ...")
    df_15m = _load_ohlcv(db, _TABLE, "15min", start, end)
    print(f"Loading 4H  data  {start} → {end} ...")
    df_4h  = _load_ohlcv(db, _TABLE, "4H",    start, end)

    try:
        db.connection.close()
    except Exception:
        pass

    if df_15m.empty or df_4h.empty:
        print("ERROR: no data returned — check DB connection and date range.")
        return {}, pd.DataFrame()

    print(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")
    print(f"Gold fixes: failed_zone_filter={cfg.failed_zone_filter}  "
          f"active_signals={sorted(cfg.active_signals)}  "
          f"min_zone_atr_frac={cfg.min_zone_atr_frac}  "
          f"sl_atr_buffer={cfg.sl_atr_buffer}\n")

    # ── Step config objects (all from strategy/, unchanged) ───────────────────
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

    # ── State ─────────────────────────────────────────────────────────────────
    equity        = cash
    equity_curve  = [cash]
    trades: list  = []
    skip_until    = -1
    zone_cooldown: dict = {}
    zone_reentry:  dict = {}
    won_zones:     set  = set()
    zone_outcome_history: dict = {}
    # Fix 1: permanent blacklist — keyed on zone_id (kind-aware string)
    failed_zones: set = set()

    n      = len(df_15m)
    warmup = max(_M15_WINDOW, 30)

    filters = {
        "in_position":   0,
        "tf_neutral":    0,
        "thin_zone":     0,   # fix 3a
        "conf_failed":   0,
        "setup_invalid": 0,
        "zone_cooldown": 0,
    }

    # ── Bar-by-bar loop ───────────────────────────────────────────────────────
    for i in range(warmup, n - cfg.max_forward_bars):

        # Leave-and-return state update (runs on every bar, even while in position)
        if cfg.require_leave_and_return and zone_reentry:
            bar_h = float(df_15m["high"].iloc[i])
            bar_l = float(df_15m["low"].iloc[i])
            tol   = 0.001
            ready = []
            for zk, state in zone_reentry.items():
                z_bot = state["bottom"] * (1 - tol)
                z_top = state["top"]    * (1 + tol)
                if state["phase"] == "exit":
                    if bar_h < z_bot or bar_l > z_top:
                        state["phase"] = "return"
                else:
                    if bar_l <= z_top and bar_h >= z_bot:
                        if i >= state["earliest_reentry"]:
                            ready.append(zk)
            for zk in ready:
                del zone_reentry[zk]

        if i <= skip_until:
            filters["in_position"] += 1
            continue

        ts_now = df_15m["timestamp"].iloc[i]

        # H4 window — no lookahead
        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(_H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        # M15 window
        m15_start = max(0, i - _M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)

        # Compute H4 ATR once — used for fix 3a (zone filter) and fix 3b (SL buffer)
        h4_atr = _h4_atr14(df_h4_w)

        # ── Step 2: timeframe analysis ────────────────────────────────────────
        h4_up_to  = len(df_h4_w) - 1
        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg,
                                       h4_up_to_bar=h4_up_to)

        if tf_result["signal"] == "neutral":
            filters["tf_neutral"] += 1
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = _zone_key(active_zone)
        zid         = active_zone.zone_id   # kind-aware stable id

        # ── Fix 3a: ATR-based thin-zone filter ───────────────────────────────
        if h4_atr > 0:
            zone_height = active_zone.top - active_zone.bottom
            if zone_height < cfg.min_zone_atr_frac * h4_atr:
                filters["thin_zone"] += 1
                continue

        # ── Step 3: confirmations (all 5 signals computed and logged) ─────────
        conf = check_confirmations_at_last_bar(
            df_15m_w, active_zone, direction, conf_cfg
        )

        # Fix 2: filter to active signals for the entry gate only.
        # conf.signals (all fired) is preserved in 'signals_all' on the trade.
        active_sigs  = [s for s in conf.signals if s in cfg.active_signals]
        active_count = len(active_sigs)
        if active_count < cfg.min_confirmations:
            filters["conf_failed"] += 1
            continue

        # ── Zone cooldown / blacklist gate ────────────────────────────────────
        if zone_cooldown.get(zk, -1) >= i:
            filters["zone_cooldown"] += 1
            continue
        if cfg.require_leave_and_return and zk in zone_reentry:
            filters["zone_cooldown"] += 1
            continue
        # Fix 1: permanent failed-zone blacklist (keyed on kind-aware zid)
        if cfg.failed_zone_filter and zid in failed_zones:
            filters["zone_cooldown"] += 1
            continue

        # Prior-outcome bucket (exact zone_id, same as baseline)
        is_retest = zk in won_zones
        history   = zone_outcome_history.get(zid, [])
        if not history:
            prior_bucket = "first_attempt"
        elif history[-1] == 1:
            prior_bucket = "post_win"
        elif history[-1] == -1:
            prior_bucket = "post_loss"
        else:
            prior_bucket = "post_expired"

        # ── Step 4: trade setup geometry ─────────────────────────────────────
        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            filters["setup_invalid"] += 1
            continue

        # Realistic entry: open of next bar + spread
        entry = float(df_15m["open"].iloc[i + 1])
        if eff_spread > 0:
            entry = entry + eff_spread if direction == "buy" else entry - eff_spread

        # Rebase SL to actual entry (keep risk distance)
        sl_dist = abs(signal_price - setup.sl)

        # Fix 3b: widen SL by ATR buffer (same h4_atr as fix 3a)
        if h4_atr > 0:
            sl_dist += cfg.sl_atr_buffer * h4_atr

        sl = entry - sl_dist if direction == "buy" else entry + sl_dist

        # TP (midline or zone edge)
        if setup.tp_mode == "midline" and setup.tp_zone is not None:
            full_tp = (setup.tp_zone.bottom if direction == "buy"
                       else setup.tp_zone.top)
            tp = entry + cfg.midline_pct * (full_tp - entry)
        else:
            tp = setup.tp

        # Geometry validation at actual entry
        if direction == "buy":
            if sl >= entry or tp <= entry:
                filters["setup_invalid"] += 1
                continue
        else:
            if sl <= entry or tp >= entry:
                filters["setup_invalid"] += 1
                continue

        # RR check after ATR-widened SL
        rr_check = abs(tp - entry) / sl_dist if sl_dist > 0 else 0.0
        if rr_check < cfg.min_rr:
            filters["setup_invalid"] += 1
            continue

        lot = (cfg.fixed_lot if cfg.fixed_lot > 0
               else _lot_size(equity, sl_dist, _CONTRACT))

        # ── Simulate forward price action ─────────────────────────────────────
        outcome        = 0
        exit_price     = entry
        exit_bar       = i + cfg.max_forward_bars
        max_favourable = 0.0
        max_adverse    = 0.0

        for j in range(i + 1, min(i + 1 + cfg.max_forward_bars, n)):
            fh = float(df_15m["high"].iloc[j])
            fl = float(df_15m["low"].iloc[j])

            favour  = (fh - entry) if direction == "buy" else (entry - fl)
            adverse = (entry - fl) if direction == "buy" else (fh - entry)
            max_favourable = max(max_favourable, favour)
            max_adverse    = max(max_adverse,    adverse)

            if direction == "buy":
                if fh >= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                if fl <= sl:  outcome = -1; exit_price = sl; exit_bar = j; break
            else:
                if fl <= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                if fh >= sl:  outcome = -1; exit_price = sl; exit_bar = j; break

        # ── P&L ───────────────────────────────────────────────────────────────
        if direction == "buy":
            pnl = (exit_price - entry) * lot * _CONTRACT
        else:
            pnl = (entry - exit_price) * lot * _CONTRACT

        equity += pnl
        equity_curve.append(equity)
        skip_until = exit_bar

        # Zone outcome bookkeeping
        zone_outcome_history.setdefault(zid, []).append(outcome)

        if outcome == 1:
            won_zones.add(zk)
            if cfg.require_leave_and_return:
                zone_reentry[zk] = {
                    "phase":            "exit",
                    "bottom":           active_zone.bottom,
                    "top":              active_zone.top,
                    "earliest_reentry": exit_bar + cfg.cooldown_bars,
                }
            else:
                zone_cooldown[zk] = exit_bar + cfg.cooldown_bars

        elif outcome == -1:
            # Fix 1: permanently blacklist the zone (kind-aware zid), else legacy cooldown
            if cfg.failed_zone_filter:
                failed_zones.add(zid)
            else:
                zone_cooldown[zk] = exit_bar + COOLDOWN_LOSS

        exit_ts = df_15m["timestamp"].iloc[min(exit_bar, n - 1)]
        trades.append({
            "date":          ts_now,
            "exit_date":     exit_ts,
            "side":          direction,
            "entry":         entry,
            "sl":            sl,
            "tp":            tp,
            "exit":          exit_price,
            "outcome":       outcome,
            "lot":           lot,
            "pnl":           round(pnl, 2),
            "equity":        round(equity, 2),
            "max_favour":    round(max_favourable, 2),
            "max_adverse":   round(max_adverse, 2),
            # Fix 2: 'signals' = active subset (gated entry), 'signals_all' = everything fired
            "confirmations": active_count,
            "signals":       "|".join(active_sigs),
            "signals_all":   "|".join(conf.signals),
            "h4_bias":       tf_result["h4_bias"],
            "h4_atr":        round(h4_atr, 2),
            "entry_mode":    setup.entry_mode,
            "tp_mode":       setup.tp_mode,
            "is_retest":     is_retest,
            "prior_bucket":  prior_bucket,
            "zone_bottom":   active_zone.bottom,
            "zone_top":      active_zone.top,
            "zone_strength": active_zone.strength,
            "zone_kind":     active_zone.kind,
            "zone_height_$": round(active_zone.top - active_zone.bottom, 2),
            "zone_ht_atr":   round((active_zone.top - active_zone.bottom) / h4_atr, 3)
                             if h4_atr > 0 else float("nan"),
            "structure":     {"zone": (active_zone.bottom, active_zone.top)},
        })

    # ── Report ────────────────────────────────────────────────────────────────
    if not trades:
        print("No trades generated. Lower min_zone_atr_frac or reduce min_confirmations.")
        return {}, pd.DataFrame()

    df_t = pd.DataFrame(trades)

    total    = len(df_t)
    tp_hits  = int((df_t["outcome"] ==  1).sum())
    sl_hits  = int((df_t["outcome"] == -1).sum())
    expired  = int((df_t["outcome"] ==  0).sum())
    win_rate = tp_hits / max(total, 1) * 100
    winners  = df_t[df_t["pnl"] > 0]
    losers   = df_t[df_t["pnl"] < 0]

    eq_s       = pd.Series(equity_curve)
    max_dd_pct = round(((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100, 2)

    buy_df  = df_t[df_t["side"] == "buy"]
    sell_df = df_t[df_t["side"] == "sell"]

    metrics = {
        "strategy":           "Zone-to-Zone GOLD v2",
        "symbol":             "XAUUSD",
        "period":             f"{start} to {end}",
        "start_cash":         f"${cash:,.2f}",
        "final_equity":       f"${equity:,.2f}",
        "net_pnl":            f"${equity - cash:,.2f}",
        "total_trades":       total,
        "tp_hits":            tp_hits,
        "sl_hits":            sl_hits,
        "expired":            expired,
        "win_rate_%":         f"{win_rate:.1f}",
        "buy_trades":         len(buy_df),
        "buy_wins":           int((buy_df["outcome"] == 1).sum()),
        "sell_trades":        len(sell_df),
        "sell_wins":          int((sell_df["outcome"] == 1).sum()),
        "avg_win_$":          f"${winners['pnl'].mean():.2f}" if len(winners) else "$0.00",
        "avg_loss_$":         f"${losers['pnl'].mean():.2f}"  if len(losers)  else "$0.00",
        "largest_win_$":      f"${df_t['pnl'].max():.2f}",
        "largest_loss_$":     f"${df_t['pnl'].min():.2f}",
        "max_drawdown_%":     f"{max_dd_pct:.2f}",
        # Gold-fix params (self-documenting)
        "failed_zone_filter": cfg.failed_zone_filter,
        "active_signals":     sorted(cfg.active_signals),
        "min_zone_atr_frac":  cfg.min_zone_atr_frac,
        "sl_atr_buffer":      cfg.sl_atr_buffer,
        "min_rr":             cfg.min_rr,
        "spread_pts":         eff_spread,
    }

    print_report(metrics, run_label="Gold Z&Z v2")

    # Filter breakdown
    evaluated = n - warmup - cfg.max_forward_bars
    print(f"\n{'─'*52}")
    print(f"  Filter breakdown  ({evaluated:,} bars evaluated)")
    print(f"{'─'*52}")
    for label, count in sorted(filters.items(), key=lambda x: -x[1]):
        if count == 0:
            continue
        pct = count / max(evaluated, 1) * 100
        print(f"  {label:<25} {count:>8,}  ({pct:.1f}%)")
    print(f"  {'signals fired':<25} {total:>8,}")
    print(f"{'─'*52}")

    # Confirmation stacking (active signals only)
    stack_counts = df_t["confirmations"].value_counts().sort_index()
    print(f"\n  Confirmation stacking (active signals only):")
    for cnt, freq in stack_counts.items():
        wr_s = (df_t[df_t["confirmations"] == cnt]["outcome"] == 1).mean() * 100
        print(f"    {cnt} confirmations : {freq:>4} trades  WR={wr_s:.0f}%")

    # Signal breakdown — active vs disabled
    print(f"\n  Signal breakdown (all fired, active=entry-eligible):")
    all_fired = []
    for row in df_t["signals_all"]:
        all_fired.extend(row.split("|") if row else [])
    for sig in sorted(set(s for s in all_fired if s)):
        mask = df_t["signals_all"].str.contains(sig, na=False)
        grp  = df_t[mask]
        wr_s = (grp["outcome"] == 1).mean() * 100
        tag  = "ACTIVE  " if sig in cfg.active_signals else "disabled"
        print(f"    [{tag}] {sig:<18}: {len(grp):>3} trades  WR={wr_s:.0f}%  "
              f"net=${grp['pnl'].sum():+.2f}")

    # H4 bias breakdown
    print(f"\n  H4 bias at entry:")
    for bias_val, grp in df_t.groupby("h4_bias"):
        wr_b = (grp["outcome"] == 1).mean() * 100
        print(f"    {bias_val:<14} : {len(grp):>4} trades  WR={wr_b:.0f}%")

    # Prior-outcome bucket (post_loss should be empty when failed_zone_filter=True)
    print(f"\n  Prior-outcome split:")
    for pb in ("first_attempt", "post_win", "post_loss", "post_expired"):
        g = df_t[df_t["prior_bucket"] == pb]
        if len(g) == 0:
            continue
        wr  = (g["outcome"] == 1).mean() * 100
        net = g["pnl"].sum()
        print(f"    {pb:<16} : {len(g):>3} trades  "
              f"WR={wr:.0f}%  net=${net:+.2f}")

    # Zone stats
    print(f"\n  Zone quality (traded zones):")
    print(f"    avg zone height $  : ${df_t['zone_height_$'].mean():.2f}")
    print(f"    avg zone height ATR: {df_t['zone_ht_atr'].mean():.3f}×")
    print(f"    avg zone strength  : {df_t['zone_strength'].mean():.2f}")
    print(f"    avg H4 ATR at entry: ${df_t['h4_atr'].mean():.2f}")
    print(f"    demand zones       : {(df_t['zone_kind']=='demand').sum()}"
          f"  ({(df_t['zone_kind']=='demand').mean()*100:.0f}%)")
    print(f"    supply zones       : {(df_t['zone_kind']=='supply').sum()}"
          f"  ({(df_t['zone_kind']=='supply').mean()*100:.0f}%)")
    print()

    cols = ["date", "side", "h4_bias", "signals", "signals_all",
            "confirmations", "is_retest", "prior_bucket",
            "entry", "sl", "tp", "exit", "pnl", "outcome"]
    print(df_t[cols].to_string(index=False))
    print()

    if save_path:
        save_report(metrics, save_path)

    if chart:
        title = (
            f"Gold Z&Z v2 — XAUUSD  |  {start} → {end}  |  "
            f"{tp_hits}W / {sl_hits}L / {expired}E  |  WR {win_rate:.1f}%"
        )
        fig = plot_trades(df_15m, trades, title=title, start_cash=cash)
        fig.show()

    return metrics, df_t
