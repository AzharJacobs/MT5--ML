"""
XAUUSD (Gold) Zone-to-Zone backtest engine.

Full implementation — no longer depends on strategy_v2/.
Three gold-specific modifications vs the baseline engine:

  Fix 1 (failed_zone_filter)  — permanent post-loss zone blacklist
  Fix 2 (active_signals)      — only named signals gate entry
  Fix 3a (min_zone_atr_frac)  — skip zones narrower than frac × H4 ATR
  Fix 3b (sl_atr_buffer)      — widen SL by frac × H4 ATR
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.core.zones import ZoneConfig
from trading.strategies.zz.core.timeframe_structure import TFConfig, analyse_timeframes
from trading.strategies.zz.core.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import TradeSetupConfig, setup_from_analysis
from trading.shared.backtest.report import print_report, save_report
from trading.shared.backtest.chart import plot_trades
from trading.shared.data_loader import get_connection
from trading.shared.mt5_loader import fetch_ohlcv as _mt5_fetch, disconnect as _mt5_disconnect


# ── Inlined helpers (previously in backtest/engine_zones.py) ─────────────────

_RISK_PCT     = 0.01
_COOLDOWN_LOSS = 48   # M15 bars (12 h)


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


def _lot_size(equity: float, sl_dist: float, contract_size: float) -> float:
    if sl_dist <= 0:
        return 0.01
    lot = (equity * _RISK_PCT) / (sl_dist * contract_size)
    return max(round(lot, 2), 0.01)

# Gold-specific config — lives in this package now
from trading.strategies.zz.xauusd.strategy import GoldZZConfig


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
        np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])),
    )
    n = min(14, len(tr))
    return float(np.mean(tr[-n:]))


# ── Constants ─────────────────────────────────────────────────────────────────

_H4_WINDOW  = 150
_M15_WINDOW = 80
_DB_NAME    = "XAUUSD"
_TABLE      = "xauusd_ohlcv"
_CONTRACT   = 100.0


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest_gold(
    cfg: GoldZZConfig,
    start: str = "2023-01-01",
    end:   str = "2024-01-01",
    cash:  float = 10_000.0,
    save_path: Optional[str] = None,
    chart: bool = False,
    silent: bool = False,
    data_source: str = "db",       # "db" = PostgreSQL  |  "mt5" = live MT5 terminal
) -> tuple:
    """
    Run the gold-specific Zone-to-Zone backtest for [start, end).
    Returns (metrics: dict, df_trades: pd.DataFrame).
    """
    eff_spread = cfg.spread
    _pr = (lambda *a, **kw: None) if silent else print

    _pr(f"\nSymbol : XAUUSD (gold v2)  |  source: {data_source}  "
        f"|  Contract: ${_CONTRACT}/pt/lot  |  Spread: {eff_spread} pts")

    if data_source == "mt5":
        _pr(f"Loading 15M data from MT5 terminal  {start} -> {end} ...")
        df_15m = _mt5_fetch("xauusd", "15min", start, end, silent=silent)
        _pr(f"Loading 4H  data from MT5 terminal  {start} -> {end} ...")
        df_4h  = _mt5_fetch("xauusd", "4H",    start, end, silent=silent)
        _mt5_disconnect()
    else:
        db = get_connection()
        db.database = _DB_NAME
        db.connect()

        _pr(f"Loading 15M data  {start} -> {end} ...")
        df_15m = _load_ohlcv(db, _TABLE, "15min", start, end)
        _pr(f"Loading 4H  data  {start} -> {end} ...")
        df_4h  = _load_ohlcv(db, _TABLE, "4H",    start, end)

        try:
            db.connection.close()
        except Exception:
            pass

    if df_15m.empty or df_4h.empty:
        src = "MT5 terminal" if data_source == "mt5" else "DB"
        print(f"ERROR: no data returned from {src} — check connection and date range.")
        return {}, pd.DataFrame()

    _pr(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")
    _pr(f"Gold fixes: failed_zone_filter={cfg.failed_zone_filter}  "
        f"active_signals={sorted(cfg.active_signals)}  "
        f"min_zone_atr_frac={cfg.min_zone_atr_frac}  "
        f"sl_atr_buffer={cfg.sl_atr_buffer}")
    _pr(f"D1 trend filter: {cfg.d1_trend_filter}  |  Session window: {cfg.trading_hours}\n")

    df_d1 = None
    if cfg.d1_trend_filter:
        # D1 = H4 resampled to calendar days (6 H4 bars per day). EMA computed once
        # on the full closed-day series — no lookahead since each day's OHLC only
        # depends on bars up to and including that day.
        df_d1 = (
            df_4h.set_index("timestamp")
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna(subset=["close"])
            .reset_index()
        )
        df_d1["ema"] = df_d1["close"].ewm(span=cfg.d1_ema_period, adjust=False).mean()
        _pr(f"D1 bars: {len(df_d1)}  (EMA{cfg.d1_ema_period})")

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

    equity        = cash
    equity_curve  = [cash]
    trades: list  = []
    skip_until    = -1
    zone_cooldown: dict = {}
    zone_reentry:  dict = {}
    won_zones:     set  = set()
    zone_outcome_history: dict = {}
    failed_zones: set = set()   # Fix 1: permanent post-loss blacklist

    n      = len(df_15m)
    warmup = max(_M15_WINDOW, 30)

    filters = {
        "in_position":     0,
        "tf_neutral":      0,
        "thin_zone":       0,
        "conf_failed":     0,
        "setup_invalid":   0,
        "zone_cooldown":   0,
        "trading_hours":   0,
        "d1_trend_filter": 0,
    }

    for i in range(warmup, n - cfg.max_forward_bars):

        # Leave-and-return state update (every bar, even while in position)
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

        # Session window — blocks NEW entries outside trading_hours. Positions already
        # open are managed above (skip_until/leave-and-return), unaffected by this gate.
        if cfg.trading_hours is not None and not (cfg.trading_hours[0] <= ts_now.hour < cfg.trading_hours[1]):
            filters["trading_hours"] += 1
            continue

        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(_H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        m15_start = max(0, i - _M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)

        h4_atr = _h4_atr14(df_h4_w)

        h4_up_to  = len(df_h4_w) - 1
        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=h4_up_to)

        if tf_result["signal"] == "neutral":
            filters["tf_neutral"] += 1
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = _zone_key(active_zone)
        zid         = active_zone.zone_id

        # D1 trend filter — highest priority: buy only above D1 EMA, sell only below.
        if cfg.d1_trend_filter and df_d1 is not None:
            _d1_idx = df_d1["timestamp"].searchsorted(ts_now.normalize(), side="left") - 1
            _d1_blocked = True
            if _d1_idx >= 0:
                _d1_ema     = df_d1["ema"].iloc[_d1_idx]
                _price_now  = float(df_15m["close"].iloc[i])
                _d1_blocked = (
                    (_price_now > _d1_ema and direction != "buy") or
                    (_price_now < _d1_ema and direction != "sell")
                )
            if _d1_blocked:
                filters["d1_trend_filter"] += 1
                continue

        # Fix 3a: skip thin zones
        if h4_atr > 0:
            zone_height = active_zone.top - active_zone.bottom
            if zone_height < cfg.min_zone_atr_frac * h4_atr:
                filters["thin_zone"] += 1
                continue

        conf = check_confirmations_at_last_bar(df_15m_w, active_zone, direction, conf_cfg)

        # Fix 2: filter to active signals for the entry gate
        active_sigs  = [s for s in conf.signals if s in cfg.active_signals]
        active_count = len(active_sigs)
        # Fresh-zone relaxation (ported from USTEC): a fresh H4 zone only needs 1
        # active-signal confirmation; tapped zones keep the configured min_confirmations.
        _required_conf = 1 if active_zone.fresh else cfg.min_confirmations
        if active_count < _required_conf:
            filters["conf_failed"] += 1
            continue

        # Zone cooldown / blacklist gate
        if zone_cooldown.get(zk, -1) >= i:
            filters["zone_cooldown"] += 1
            continue
        if cfg.require_leave_and_return and zk in zone_reentry:
            filters["zone_cooldown"] += 1
            continue
        # Fix 1: permanent failed-zone blacklist
        if cfg.failed_zone_filter and zid in failed_zones:
            filters["zone_cooldown"] += 1
            continue

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

        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            filters["setup_invalid"] += 1
            continue

        entry = float(df_15m["open"].iloc[i + 1])
        if eff_spread > 0:
            entry = entry + eff_spread if direction == "buy" else entry - eff_spread

        sl_dist = abs(signal_price - setup.sl)

        # Fix 3b: widen SL by ATR buffer
        if h4_atr > 0:
            sl_dist += cfg.sl_atr_buffer * h4_atr

        sl = entry - sl_dist if direction == "buy" else entry + sl_dist

        if setup.tp_mode == "midline" and setup.tp_zone is not None:
            full_tp = (setup.tp_zone.bottom if direction == "buy" else setup.tp_zone.top)
            tp = entry + cfg.midline_pct * (full_tp - entry)
        else:
            tp = setup.tp

        if direction == "buy":
            if sl >= entry or tp <= entry:
                filters["setup_invalid"] += 1
                continue
        else:
            if sl <= entry or tp >= entry:
                filters["setup_invalid"] += 1
                continue

        rr_check = abs(tp - entry) / sl_dist if sl_dist > 0 else 0.0
        if rr_check < cfg.min_rr:
            filters["setup_invalid"] += 1
            continue

        lot = (cfg.fixed_lot if cfg.fixed_lot > 0 else _lot_size(equity, sl_dist, _CONTRACT))

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

        if direction == "buy":
            pnl = (exit_price - entry) * lot * _CONTRACT
        else:
            pnl = (entry - exit_price) * lot * _CONTRACT

        equity += pnl
        equity_curve.append(equity)
        skip_until = exit_bar

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
            if cfg.failed_zone_filter:
                failed_zones.add(zid)   # Fix 1
            else:
                zone_cooldown[zk] = exit_bar + _COOLDOWN_LOSS

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
            "zone_fresh":    active_zone.fresh,
            "zone_height_$": round(active_zone.top - active_zone.bottom, 2),
            "zone_ht_atr":   round((active_zone.top - active_zone.bottom) / h4_atr, 3)
                             if h4_atr > 0 else float("nan"),
            "structure":     {"zone": (active_zone.bottom, active_zone.top)},
        })

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
        "failed_zone_filter": cfg.failed_zone_filter,
        "active_signals":     sorted(cfg.active_signals),
        "min_zone_atr_frac":  cfg.min_zone_atr_frac,
        "sl_atr_buffer":      cfg.sl_atr_buffer,
        "min_rr":             cfg.min_rr,
        "spread_pts":         eff_spread,
    }

    if not silent:
        print_report(metrics, run_label="Gold Z&Z v2")

        evaluated = n - warmup - cfg.max_forward_bars
        print(f"\n{'-'*52}")
        print(f"  Filter breakdown  ({evaluated:,} bars evaluated)")
        print(f"{'-'*52}")
        for label, count in sorted(filters.items(), key=lambda x: -x[1]):
            if count == 0:
                continue
            pct = count / max(evaluated, 1) * 100
            print(f"  {label:<25} {count:>8,}  ({pct:.1f}%)")
        print(f"  {'signals fired':<25} {total:>8,}")
        print(f"{'-'*52}")

        stack_counts = df_t["confirmations"].value_counts().sort_index()
        print(f"\n  Confirmation stacking (active signals only):")
        for cnt, freq in stack_counts.items():
            wr_s = (df_t[df_t["confirmations"] == cnt]["outcome"] == 1).mean() * 100
            print(f"    {cnt} confirmations : {freq:>4} trades  WR={wr_s:.0f}%")

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

        print(f"\n  H4 bias at entry:")
        for bias_val, grp in df_t.groupby("h4_bias"):
            wr_b = (grp["outcome"] == 1).mean() * 100
            print(f"    {bias_val:<14} : {len(grp):>4} trades  WR={wr_b:.0f}%")

        print(f"\n  Prior-outcome split:")
        for pb in ("first_attempt", "post_win", "post_loss", "post_expired"):
            g = df_t[df_t["prior_bucket"] == pb]
            if len(g) == 0:
                continue
            wr  = (g["outcome"] == 1).mean() * 100
            net = g["pnl"].sum()
            print(f"    {pb:<16} : {len(g):>3} trades  WR={wr:.0f}%  net=${net:+.2f}")

        print(f"\n  Zone quality:")
        print(f"    avg zone height $  : ${df_t['zone_height_$'].mean():.2f}")
        print(f"    avg zone height ATR: {df_t['zone_ht_atr'].mean():.3f}x")
        print(f"    avg zone strength  : {df_t['zone_strength'].mean():.2f}")
        print(f"    avg H4 ATR at entry: ${df_t['h4_atr'].mean():.2f}")
        print()

        cols = ["date", "side", "h4_bias", "signals", "signals_all",
                "confirmations", "is_retest", "prior_bucket",
                "entry", "sl", "tp", "exit", "pnl", "outcome"]
        print(df_t[cols].to_string(index=False))
        print()

    if save_path:
        save_report(metrics, save_path)

    if chart:
        title = (f"Gold Z&Z v2 - XAUUSD  |  {start} -> {end}  |  "
                 f"{tp_hits}W / {sl_hits}L / {expired}E  |  WR {win_rate:.1f}%")
        fig = plot_trades(df_15m, trades, title=title, start_cash=cash)
        fig.show()

    return metrics, df_t
