"""
engine_zones.py — Backtest engine for the zone-based strategy (Steps 1–4).

Bar-by-bar M15 simulation:
  Step 1  detect_zones on H4 (bounded to current bar — no lookahead)
  Step 2  analyse_timeframes → H4 bias + active zone + M15 pre-filter
  Step 3  check_confirmations_at_last_bar → entry pattern gate
  Step 4  build_trade_setup → entry / SL / TP geometry

Usage (via scripts/backtest_zones.py):
    python scripts/backtest_zones.py --start 2023-01-01 --end 2024-01-01
    python scripts/backtest_zones.py --start 2024-01-01 --end 2025-01-01 --min_rr 2.0
    python scripts/backtest_zones.py --aggressive_entry --midline_tp --midline_pct 0.5
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from data.loader import get_connection
from strategy.zones import ZoneConfig
from strategy.timeframe_structure import TFConfig, analyse_timeframes
from strategy.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from strategy.trade_setup import TradeSetupConfig, setup_from_analysis
from backtest.report import print_report, save_report
from backtest.chart_market_structure import plot_trades


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H4_WINDOW  = 150   # H4 bars fed to zone detection + bias (needs 4+ swings)
M15_WINDOW = 80    # M15 bars fed to confirmation checks (covers all lookbacks)

MAX_FORWARD_BARS = 350   # 350 × 15 min ≈ 87 h ≈ 4 trading days

RISK_PCT = 0.01   # 1 % of equity risked per trade

# Zone re-entry cooldown — outcome-based (M15 bars).
# COOLDOWN_WIN replaced by config-driven logic (cooldown_bars / require_leave_and_return).
COOLDOWN_LOSS    = 48   # loss retry window: 48 bars = 12 h
COOLDOWN_EXPIRED =  0   # expired: no cooldown

SYMBOL_CONFIG = {
    "ustech": ("ustech_ohlcv", "ustech_ohlcv", 100.0),
    "xauusd": ("XAUUSD",       "xauusd_ohlcv", 100.0),
}
SYMBOL_SPREAD = {
    "ustech": 2.0,
    "xauusd": 0.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    lot = (equity * RISK_PCT) / (sl_dist * contract_size)
    return max(round(lot, 2), 0.01)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    start: str = "2023-01-01",
    end:   str = "2024-01-01",
    cash:  float = 10_000.0,
    min_rr: float = 1.5,
    max_forward_bars: int = MAX_FORWARD_BARS,
    symbol: str = "ustech",
    spread: Optional[float] = None,
    # Step 2 toggles
    directional_filter: bool = True,
    allow_neutral: bool = True,
    # Step 3 toggles
    min_confirmations: int = 1,
    aggressive_boundary: bool = False,
    excluded_from_count: Optional[list] = None,
    # Step 4 toggles
    aggressive_entry: bool = False,
    midline_tp: bool = False,
    midline_pct: float = 0.50,
    sl_buffer_pct: float = 0.002,
    fixed_lot: float = 0.0,         # 0 = dynamic 1% risk sizing; >0 = fixed lot every trade
    # Win-cooldown mode (two toggleable modes, not mutually exclusive)
    require_leave_and_return: bool = True,
    cooldown_bars: int = 15,
    # Zone loss blacklist: permanently skip a zone after this many consecutive losses.
    # 0 = disabled (default baseline behaviour).
    zone_max_losses: int = 0,
    # Output
    save_path: Optional[str] = None,
    chart: bool = False,
) -> dict:
    """
    Run the zone-based strategy backtest for [start, end).

    Returns the metrics dict (also saved to save_path as JSON when provided).
    """
    sym = symbol.lower()
    if sym not in SYMBOL_CONFIG:
        raise ValueError(f"Unknown symbol '{symbol}'. Choose from: {list(SYMBOL_CONFIG)}")

    db_name, table, contract_size = SYMBOL_CONFIG[sym]
    eff_spread = spread if spread is not None else SYMBOL_SPREAD.get(sym, 0.0)

    db = get_connection()
    db.database = db_name
    db.connect()

    print(f"\nSymbol : {sym.upper()}  |  DB: {db_name}  |  Contract: ${contract_size}/pt/lot  |  Spread: {eff_spread} pts")
    print(f"Loading 15M data  {start} → {end} ...")
    df_15m = _load_ohlcv(db, table, "15min", start, end)
    print(f"Loading 4H  data  {start} → {end} ...")
    df_4h  = _load_ohlcv(db, table, "4H",    start, end)

    try:
        db.connection.close()
    except Exception:
        pass

    if df_15m.empty or df_4h.empty:
        print("ERROR: no data returned — check DB connection and date range.")
        return {}

    # Drop consecutive 4H rows with identical OHLCV — historical data was stored with
    # each bar duplicated (bar open + bar close timestamps, same prices). Dedup preserves
    # the last (close-time) row of each pair so timestamps align with bar boundaries.
    _ohlcv = ["open", "high", "low", "close"]
    _before = len(df_4h)
    df_4h = df_4h[df_4h[_ohlcv].ne(df_4h[_ohlcv].shift()).any(axis=1)].reset_index(drop=True)
    _dupes = _before - len(df_4h)
    if _dupes:
        print(f"  (removed {_dupes} duplicate 4H rows — historical data artifact)")

    print(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")

    # ── Step config objects ────────────────────────────────────────────────
    tf_cfg = TFConfig(
        directional_filter=directional_filter,
        allow_neutral_up=allow_neutral,
        allow_neutral_down=allow_neutral,
        h4_zone_cfg=ZoneConfig(
            impulse_atr_mult=2.0,
            body_ratio_min=0.50,
            min_departure_candles=2,
            departure_window=6,
            base_lookback=5,
            min_strength=1.5,
        ),
        m15_tap_lookback=20,
        require_m15_directional_close=True,
    )
    conf_cfg = ConfirmationConfig(
        min_confirmations=min_confirmations,
        aggressive_boundary=aggressive_boundary,
        bos_lookback=15,
        structure_lookback=25,
        excluded_from_count=excluded_from_count or [],
    )
    setup_cfg = TradeSetupConfig(
        aggressive_entry=aggressive_entry,
        sl_buffer_pct=sl_buffer_pct,
        midline_tp=midline_tp,
        midline_pct=midline_pct,
        min_rr=min_rr,
    )

    print(
        f"Config : directional_filter={directional_filter}  min_confirmations={min_confirmations}"
        f"  aggressive_entry={aggressive_entry}  midline_tp={midline_tp}\n"
    )

    # ── State ─────────────────────────────────────────────────────────────
    equity       = cash
    equity_curve = [cash]
    trades:  list = []
    skip_until   = -1
    zone_cooldown: dict = {}
    # zone_reentry: tracks leave-and-return state after a win.
    # key = zone_key tuple; value = {'phase': 'exit'|'return', 'bottom': float, 'top': float}
    zone_reentry: dict = {}
    won_zones:    set  = set()   # zone keys that have had at least one win
    # zone_outcome_history: exact zone_id → list of outcomes in entry order.
    # Used to determine prior_bucket for each trade (first / post_win / post_loss).
    zone_outcome_history: dict = {}
    # zone_consec_losses: consecutive loss count per zone_id for blacklist logic.
    zone_consec_losses: dict = {}
    zone_blacklist:      set  = set()   # permanently blocked zone_ids
    n     = len(df_15m)
    warmup = max(M15_WINDOW, 30)

    filters = {
        "in_position":       0,
        "tf_neutral":        0,    # Step 2 rejected (bias/zone/pre-filter)
        "conf_failed":       0,    # Step 3 no confirmation
        "setup_invalid":     0,    # Step 4 geometry/RR failure
        "zone_cooldown":     0,
        "zone_blacklist":    0,    # zone permanently blocked after zone_max_losses
    }

    # ── Bar-by-bar loop ────────────────────────────────────────────────────
    for i in range(warmup, n - max_forward_bars):

        # ── Leave-and-return zone state update (runs on EVERY bar) ───────────
        # Must happen before skip_until so state advances even while in position.
        if require_leave_and_return and zone_reentry:
            bar_h = float(df_15m["high"].iloc[i])
            bar_l = float(df_15m["low"].iloc[i])
            tol   = 0.001
            ready = []
            for zk, state in zone_reentry.items():
                z_bot = state["bottom"] * (1 - tol)
                z_top = state["top"]    * (1 + tol)
                if state["phase"] == "exit":
                    # Price has fully left the zone when the bar's range
                    # is entirely above zone top or entirely below zone bottom
                    if bar_h < z_bot or bar_l > z_top:
                        state["phase"] = "return"
                else:  # "return"
                    # Price has returned when any part of bar overlaps zone
                    if bar_l <= z_top and bar_h >= z_bot:
                        # Also enforce minimum bar cooldown as a floor
                        if i >= state["earliest_reentry"]:
                            ready.append(zk)
            for zk in ready:
                del zone_reentry[zk]

        if i <= skip_until:
            filters["in_position"] += 1
            continue

        ts_now = df_15m["timestamp"].iloc[i]

        # H4 window: only bars with timestamp ≤ current M15 bar (no lookahead)
        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        # M15 window: last M15_WINDOW bars including current bar
        m15_start = max(0, i - M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)

        # ── Step 2: timeframe analysis ────────────────────────────────────
        h4_up_to  = len(df_h4_w) - 1
        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg,
                                       h4_up_to_bar=h4_up_to)

        if tf_result["signal"] == "neutral":
            filters["tf_neutral"] += 1
            continue

        # ── Step 3: entry confirmations ───────────────────────────────────
        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        conf = check_confirmations_at_last_bar(
            df_15m_w, active_zone, direction, conf_cfg
        )
        if not conf.confirmed:
            filters["conf_failed"] += 1
            continue

        # ── Zone cooldown gate ────────────────────────────────────────────
        zk = _zone_key(active_zone)
        if zone_cooldown.get(zk, -1) >= i:
            filters["zone_cooldown"] += 1
            continue
        if require_leave_and_return and zk in zone_reentry:
            filters["zone_cooldown"] += 1
            continue

        # ── Zone blacklist gate (consecutive-loss limit) ──────────────────
        zid = active_zone.zone_id
        if zone_max_losses > 0 and zid in zone_blacklist:
            filters["zone_blacklist"] += 1
            continue

        # Flag: has this zone been won before? (re-test trade)
        is_retest = zk in won_zones

        # Prior-outcome bucket using exact zone_id (kind + rounded edges, stable)
        # (zid already set above)
        history = zone_outcome_history.get(zid, [])
        if not history:
            prior_bucket = "first_attempt"
        elif history[-1] == 1:
            prior_bucket = "post_win"
        elif history[-1] == -1:
            prior_bucket = "post_loss"
        else:  # prior was expired (0)
            prior_bucket = "post_expired"

        # ── Step 4: trade setup geometry ──────────────────────────────────
        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            filters["setup_invalid"] += 1
            continue

        # ── Realistic entry: open of next bar ─────────────────────────────
        entry = float(df_15m["open"].iloc[i + 1])

        # Apply spread cost
        if eff_spread > 0:
            entry = entry + eff_spread if direction == "buy" else entry - eff_spread

        # Rebase SL to actual entry (keep risk distance, adjust anchor)
        sl_dist = abs(signal_price - setup.sl)
        sl = entry - sl_dist if direction == "buy" else entry + sl_dist

        # TP: zone edge is absolute; midline recomputed from actual entry
        if setup.tp_mode == "midline" and setup.tp_zone is not None:
            full_tp = (setup.tp_zone.bottom if direction == "buy"
                       else setup.tp_zone.top)
            tp = entry + midline_pct * (full_tp - entry)
        else:
            tp = setup.tp

        # Re-validate geometry at actual entry
        if direction == "buy":
            if sl >= entry or tp <= entry:
                filters["setup_invalid"] += 1
                continue
        else:
            if sl <= entry or tp >= entry:
                filters["setup_invalid"] += 1
                continue

        lot = fixed_lot if fixed_lot > 0 else _lot_size(equity, abs(entry - sl), contract_size)

        # ── Simulate forward price action ─────────────────────────────────
        outcome        = 0
        exit_price     = entry
        exit_bar       = i + max_forward_bars
        max_favourable = 0.0
        max_adverse    = 0.0

        for j in range(i + 1, min(i + 1 + max_forward_bars, n)):
            fh = float(df_15m["high"].iloc[j])
            fl = float(df_15m["low"].iloc[j])

            favour  = (fh - entry) if direction == "buy" else (entry - fl)
            adverse = (entry - fl) if direction == "buy" else (fh - entry)
            max_favourable = max(max_favourable, favour)
            max_adverse    = max(max_adverse,    adverse)

            if direction == "buy":
                if fh >= tp:   outcome =  1; exit_price = tp; exit_bar = j; break
                if fl <= sl:   outcome = -1; exit_price = sl; exit_bar = j; break
            else:
                if fl <= tp:   outcome =  1; exit_price = tp; exit_bar = j; break
                if fh >= sl:   outcome = -1; exit_price = sl; exit_bar = j; break

        # ── P&L ──────────────────────────────────────────────────────────
        if direction == "buy":
            pnl = (exit_price - entry) * lot * contract_size
        else:
            pnl = (entry - exit_price) * lot * contract_size

        equity += pnl
        equity_curve.append(equity)
        skip_until = exit_bar

        # Zone cooldown (outcome-based)
        # Record outcome to exact zone history (used for prior_bucket tagging)
        zone_outcome_history.setdefault(zid, []).append(outcome)

        if outcome == 1:
            won_zones.add(zk)
            # Win resets consecutive loss count for this zone
            zone_consec_losses[zid] = 0
            if require_leave_and_return:
                zone_reentry[zk] = {
                    "phase":           "exit",
                    "bottom":          active_zone.bottom,
                    "top":             active_zone.top,
                    "earliest_reentry": exit_bar + cooldown_bars,
                }
            else:
                zone_cooldown[zk] = exit_bar + cooldown_bars
        elif outcome == -1:
            zone_cooldown[zk] = exit_bar + COOLDOWN_LOSS
            # Consecutive loss tracking for blacklist
            if zone_max_losses > 0:
                new_count = zone_consec_losses.get(zid, 0) + 1
                zone_consec_losses[zid] = new_count
                if new_count >= zone_max_losses:
                    zone_blacklist.add(zid)
        else:  # expired
            zone_consec_losses[zid] = 0

        exit_ts = df_15m["timestamp"].iloc[min(exit_bar, n - 1)]
        trades.append({
            "date":         ts_now,
            "exit_date":    exit_ts,
            "side":         direction,
            "entry":        entry,
            "sl":           sl,
            "tp":           tp,
            "exit":         exit_price,
            "outcome":      outcome,
            "lot":          lot,
            "pnl":          round(pnl, 2),
            "equity":       round(equity, 2),
            "max_favour":   round(max_favourable, 2),
            "max_adverse":  round(max_adverse, 2),
            "confirmations": conf.count,
            "signals":      "|".join(conf.signals),
            "h4_bias":      tf_result["h4_bias"],
            "entry_mode":   setup.entry_mode,
            "tp_mode":      setup.tp_mode,
            "is_retest":    is_retest,
            "prior_bucket": prior_bucket,
            "zone_bottom":  active_zone.bottom,
            "zone_top":     active_zone.top,
            "zone_strength": active_zone.strength,
            "zone_kind":    active_zone.kind,
            "structure":    {"zone": (active_zone.bottom, active_zone.top)},
        })

    # ── Report ────────────────────────────────────────────────────────────
    if not trades:
        print("No trades generated. Widen date range, lower min_rr, or reduce min_confirmations.")
        return {}

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

    # Confirmation stacking breakdown
    stack_counts = df_t["confirmations"].value_counts().sort_index()

    metrics = {
        "strategy":        "Zone-Based Strategy (Steps 1–4)",
        "symbol":          sym.upper(),
        "period":          f"{start} to {end}",
        "start_cash":      f"${cash:,.2f}",
        "final_equity":    f"${equity:,.2f}",
        "net_pnl":         f"${equity - cash:,.2f}",
        "total_trades":    total,
        "tp_hits":         tp_hits,
        "sl_hits":         sl_hits,
        "expired":         expired,
        "win_rate_%":      f"{win_rate:.1f}",
        "buy_trades":      len(buy_df),
        "buy_wins":        int((buy_df["outcome"] == 1).sum()),
        "sell_trades":     len(sell_df),
        "sell_wins":       int((sell_df["outcome"] == 1).sum()),
        "avg_win_$":       f"${winners['pnl'].mean():.2f}" if len(winners) else "$0.00",
        "avg_loss_$":      f"${losers['pnl'].mean():.2f}"  if len(losers)  else "$0.00",
        "largest_win_$":   f"${df_t['pnl'].max():.2f}",
        "largest_loss_$":  f"${df_t['pnl'].min():.2f}",
        "max_drawdown_%":  f"{max_dd_pct:.2f}",
        "spread_pts":      eff_spread,
        "min_rr":          min_rr,
        "min_confirmations": min_confirmations,
        "entry_mode":      "zone_boundary" if aggressive_entry else "confirmation",
        "tp_mode":         f"midline_{midline_pct:.0%}" if midline_tp else "zone_edge",
        "directional_filter":  directional_filter,
        "win_cooldown":        f"leave_and_return (floor={cooldown_bars}b)" if require_leave_and_return else f"{cooldown_bars} bars",
        "excluded_from_count": ",".join(excluded_from_count) if excluded_from_count else "none",
        "zone_max_losses":     zone_max_losses if zone_max_losses > 0 else "disabled",
        "zones_blacklisted":   len(zone_blacklist),
    }

    print_report(metrics, run_label="Zone Strategy")

    # Filter breakdown
    evaluated = n - warmup - max_forward_bars
    print(f"\n{'─'*48}")
    print(f"  Filter breakdown  ({evaluated:,} bars evaluated)")
    print(f"{'─'*48}")
    for label, count in sorted(filters.items(), key=lambda x: -x[1]):
        if count == 0:
            continue
        pct = count / max(evaluated, 1) * 100
        print(f"  {label:<25} {count:>8,}  ({pct:.1f}%)")
    print(f"  {'signals fired':<25} {total:>8,}")
    print(f"{'─'*48}")

    # Confirmation stacking report
    print(f"\n  Confirmation stacking (out of 5):")
    for cnt, freq in stack_counts.items():
        wr_stack = df_t[df_t["confirmations"] == cnt]
        wr_pct   = (wr_stack["outcome"] == 1).mean() * 100
        print(f"    {cnt} confirmations : {freq:>4} trades  WR={wr_pct:.0f}%")

    # Bias breakdown
    print(f"\n  H4 bias at entry:")
    for bias_val, grp in df_t.groupby("h4_bias"):
        wr_b = (grp["outcome"] == 1).mean() * 100
        print(f"    {bias_val:<14} : {len(grp):>4} trades  WR={wr_b:.0f}%")
    print()

    # Full trade log
    # Prior-outcome bucket breakdown (exact zone_id — no approximation)
    def _pb_stats(bucket):
        g = df_t[df_t["prior_bucket"] == bucket]
        if len(g) == 0:
            return 0, 0, 0.0, 0.0
        wr  = (g["outcome"] == 1).mean() * 100
        net = g["pnl"].sum()
        return len(g), int((g["outcome"] == 1).sum()), wr, net

    print(f"\n  Prior-outcome split (exact zone_id):")
    for pb in ("first_attempt", "post_win", "post_loss", "post_expired"):
        n_pb, w_pb, wr_pb, net_pb = _pb_stats(pb)
        if n_pb == 0:
            continue
        print(f"    {pb:<16} : {n_pb:>3} trades  W={w_pb:>2}  "
              f"WR={wr_pb:.0f}%  net=${net_pb:+.2f}")
    print()

    # Sanity: first_attempt + post_win should match earlier first-tap vs re-test
    n_ft = len(df_t[df_t["is_retest"] == False])
    n_rt = len(df_t[df_t["is_retest"] == True])
    n_fa = len(df_t[df_t["prior_bucket"] == "first_attempt"])
    n_pl = len(df_t[df_t["prior_bucket"] == "post_loss"])
    n_pw = len(df_t[df_t["prior_bucket"] == "post_win"])
    print(f"  Sanity check:  is_retest=False={n_ft}  vs  first_attempt+post_loss="
          f"{n_fa}+{n_pl}={n_fa+n_pl}  |  "
          f"is_retest=True={n_rt}  vs  post_win={n_pw}")
    print()

    cols = ["date", "side", "h4_bias", "signals", "confirmations",
            "is_retest", "prior_bucket", "entry", "sl", "tp", "exit", "pnl", "outcome"]
    print(df_t[cols].to_string(index=False))
    print()

    if save_path:
        save_report(metrics, save_path)

    if chart:
        title = (
            f"Zone Strategy — {sym.upper()}  |  {start} → {end}  |  "
            f"{tp_hits}W / {sl_hits}L / {expired}E  |  WR {win_rate:.1f}%"
        )
        fig = plot_trades(df_15m, trades, title=title, start_cash=cash)
        fig.show()

    return metrics, df_t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest: Zone-Based Strategy (Steps 1–4)"
    )
    parser.add_argument("--symbol",   default="ustech",
                        choices=list(SYMBOL_CONFIG))
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2024-01-01")
    parser.add_argument("--cash",     type=float, default=10_000.0)
    parser.add_argument("--min_rr",   type=float, default=1.5)
    parser.add_argument("--max_bars", type=int,   default=MAX_FORWARD_BARS)
    parser.add_argument("--spread",   type=float, default=None)
    # Step 2
    parser.add_argument("--no_directional_filter", action="store_true")
    parser.add_argument("--no_neutral",            action="store_true",
                        help="Block neutral_up/neutral_down bias (require strict bullish/bearish)")
    # Step 3
    parser.add_argument("--min_confirmations", type=int,   default=1)
    parser.add_argument("--aggressive_boundary", action="store_true",
                        help="Enter on zone touch, no confirmation pattern required")
    parser.add_argument("--exclude_signals", default="",
                        help="Comma-separated signal names excluded from confirmation count "
                             "(still logged). Example: --exclude_signals rejection_wick,choch")
    parser.add_argument("--zone_max_losses", type=int, default=0,
                        help="Permanently blacklist a zone after this many consecutive losses "
                             "(0 = disabled). Recommended: 2.")
    # Step 4
    parser.add_argument("--aggressive_entry", action="store_true",
                        help="Entry at zone boundary instead of confirmation bar close")
    parser.add_argument("--midline_tp",  action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--sl_buffer",   type=float, default=0.002)
    parser.add_argument("--fixed_lot",   type=float, default=0.0,
                        help="Fixed lot size for every trade (0 = dynamic 1%% risk sizing)")
    # Output
    parser.add_argument("--save",  default=None)
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--cooldown_bars",          type=int,  default=15,
                        help="Bars blocked after a TP hit (default 15). "
                             "Used as bar-count cooldown when --no_leave_return, "
                             "and as minimum floor in leave-and-return mode.")
    parser.add_argument("--no_leave_return",        action="store_true",
                        help="Disable leave-and-return mode; use bar-count cooldown only.")
    args = parser.parse_args()

    run_backtest(
        start=args.start,
        end=args.end,
        cash=args.cash,
        min_rr=args.min_rr,
        max_forward_bars=args.max_bars,
        symbol=args.symbol,
        spread=args.spread,
        directional_filter=not args.no_directional_filter,
        allow_neutral=not args.no_neutral,
        min_confirmations=args.min_confirmations,
        aggressive_boundary=args.aggressive_boundary,
        excluded_from_count=[s.strip() for s in args.exclude_signals.split(",") if s.strip()],
        aggressive_entry=args.aggressive_entry,
        midline_tp=args.midline_tp,
        midline_pct=args.midline_pct,
        sl_buffer_pct=args.sl_buffer,
        fixed_lot=args.fixed_lot,
        require_leave_and_return=not args.no_leave_return,
        cooldown_bars=args.cooldown_bars,
        zone_max_losses=args.zone_max_losses,
        save_path=args.save,
        chart=args.chart,
    )  # returns (metrics, df_t) — ignored here


if __name__ == "__main__":
    main()
