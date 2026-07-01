"""
USTEC Zone-to-Zone backtest engine.

Core strategy (all steps execute in order):
  Step 1  detect_zones on H4 (bounded to current bar — no lookahead)
  Step 2  analyse_timeframes → H4 bias + zone selection + M15 tap
          No-man's-land rule: clean edge reaction required (not mid-zone drift)
  Step 3  check_confirmations → engulfing / rejection wick / BOS / CHoCH
  Step 4  build_trade_setup → entry / SL / TP geometry
          Buy demand, sell supply — directional only
          SL below demand zone / above supply zone
          TP at next opposite zone (or midline 50%)
"""

from __future__ import annotations

import argparse
from dataclasses import replace as _dc_replace
from typing import Optional

import pandas as pd

from trading.strategies.zz.core.timeframe_structure import TFConfig, analyse_timeframes
from trading.strategies.zz.core.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import TradeSetupConfig, setup_from_analysis
from trading.shared.backtest.report import print_report, save_report
from trading.shared.backtest.chart import plot_trades
from trading.shared.data_loader import get_connection

from trading.strategies.zz.ustec.strategy import make_configs as _make_base_configs


H4_WINDOW        = 150
M15_WINDOW       = 80
MAX_FORWARD_BARS = 350
RISK_PCT         = 0.01
COOLDOWN_LOSS    = 48   # M15 bars cooldown after a loss (~12 h)

SYMBOL_CONFIG = {
    "ustech": ("ustech_ohlcv", "ustech_ohlcv", 100.0),
}
SYMBOL_SPREAD = {
    "ustech": 2.0,
}


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


def run_backtest(
    start:  str   = "2023-01-01",
    end:    str   = "2024-01-01",
    cash:   float = 10_000.0,
    min_rr: float = 1.5,
    max_forward_bars: int   = MAX_FORWARD_BARS,
    symbol: str   = "ustech",
    spread: Optional[float] = None,
    directional_filter: bool  = True,
    allow_neutral:      bool  = True,
    h4_swing_left:  int = 2,
    h4_swing_right: int = 2,
    min_confirmations:   int  = 1,
    aggressive_boundary: bool = False,
    aggressive_entry:    bool = False,
    midline_tp:    bool  = False,
    midline_pct:   float = 0.50,
    sl_buffer_pct: float = 0.002,
    fixed_lot:     float = 0.0,
    require_leave_and_return: bool = True,
    cooldown_bars: int = 15,
    save_path: Optional[str] = None,
    chart:  bool = False,
    silent: bool = False,
) -> dict:
    sym = symbol.lower()
    if sym not in SYMBOL_CONFIG:
        raise ValueError(f"Unknown symbol '{symbol}'. Choose from: {list(SYMBOL_CONFIG)}")

    db_name, table, contract_size = SYMBOL_CONFIG[sym]
    eff_spread = spread if spread is not None else SYMBOL_SPREAD.get(sym, 0.0)

    db = get_connection()
    db.database = db_name
    db.connect()

    _pr = (lambda *a, **kw: None) if silent else print
    _pr(f"\n{sym.upper()}  |  {start} → {end}  |  spread={eff_spread} pts")

    df_15m = _load_ohlcv(db, table, "15min", start, end)
    df_4h  = _load_ohlcv(db, table, "4H",    start, end)
    try:
        db.connection.close()
    except Exception:
        pass

    if df_15m.empty or df_4h.empty:
        print("ERROR: no data returned — check DB connection and date range.")
        return {}

    _ohlcv  = ["open", "high", "low", "close"]
    _before = len(df_4h)
    df_4h   = df_4h[df_4h[_ohlcv].ne(df_4h[_ohlcv].shift()).any(axis=1)].reset_index(drop=True)
    _dupes  = _before - len(df_4h)
    if _dupes:
        _pr(f"  (removed {_dupes} duplicate 4H rows)")

    _pr(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")

    _base_tf_cfg, _base_conf_cfg, _base_setup_cfg = _make_base_configs()

    tf_cfg = _dc_replace(
        _base_tf_cfg,
        directional_filter=directional_filter,
        allow_neutral_up=allow_neutral,
        allow_neutral_down=allow_neutral,
        h4_swing_left=h4_swing_left,
        h4_swing_right=h4_swing_right,
    )
    conf_cfg = _dc_replace(
        _base_conf_cfg,
        min_confirmations=min_confirmations,
        aggressive_boundary=aggressive_boundary,
        excluded_from_count=[],
    )
    setup_cfg = _dc_replace(
        _base_setup_cfg,
        aggressive_entry=aggressive_entry,
        sl_buffer_pct=sl_buffer_pct,
        midline_tp=midline_tp,
        midline_pct=midline_pct,
        min_rr=min_rr,
    )

    equity       = cash
    equity_curve = [cash]
    trades: list = []
    _pending: list = []

    zone_cooldown: dict = {}
    zone_reentry:  dict = {}

    n      = len(df_15m)
    warmup = max(M15_WINDOW, 30)

    for i in range(warmup, n - max_forward_bars):
        bar_h = float(df_15m["high"].iloc[i])
        bar_l = float(df_15m["low"].iloc[i])

        # ── Update leave-and-return zone states ───────────────────────────────
        # After a win: wait for price to fully leave the zone, then return before
        # allowing another entry. This enforces "wait for price to return — don't chase."
        if require_leave_and_return and zone_reentry:
            tol   = 0.001
            ready = []
            for zk, state in zone_reentry.items():
                z_bot = state["bottom"] * (1 - tol)
                z_top = state["top"]    * (1 + tol)
                if state["phase"] == "exit":
                    if bar_h < z_bot or bar_l > z_top:
                        state["phase"] = "return"
                else:
                    if bar_l <= z_top and bar_h >= z_bot and i >= state["earliest_reentry"]:
                        ready.append(zk)
            for zk in ready:
                del zone_reentry[zk]

        # ── Close positions whose exit bar has been reached ───────────────────
        _still_open = []
        for _p in sorted(_pending, key=lambda x: x["exit_bar"]):
            if _p["exit_bar"] <= i:
                equity += _p["pnl"]
                if equity < 0:
                    equity = 0.0
                _p["trade_data"]["equity"] = round(equity, 2)
                equity_curve.append(equity)
                trades.append(_p["trade_data"])
                _zk, _outcome, _eb = _p["zk"], _p["outcome"], _p["exit_bar"]
                if _outcome == 1:
                    if require_leave_and_return:
                        zone_reentry[_zk] = {
                            "phase":            "exit",
                            "bottom":           _p["zone_bottom"],
                            "top":              _p["zone_top"],
                            "earliest_reentry": _eb + cooldown_bars,
                        }
                    else:
                        zone_cooldown[_zk] = _eb + cooldown_bars
                elif _outcome == -1:
                    zone_cooldown[_zk] = _eb + COOLDOWN_LOSS
            else:
                _still_open.append(_p)
        _pending = _still_open

        if _pending:
            continue

        ts_now = df_15m["timestamp"].iloc[i]

        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        m15_start = max(0, i - M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)

        # ── Step 2: H4 bias + zone selection + M15 tap ───────────────────────
        h4_up_to  = len(df_h4_w) - 1
        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=h4_up_to)

        if tf_result["signal"] == "neutral":
            continue

        # No-man's land rule: skip if M15 tap is mid-zone drift, not edge reaction.
        # A clean tap requires price to reach the zone edge and close the correct side
        # of the zone midpoint. Anything else is between zones — skip it.
        if not tf_result.get("clean_tap", True):
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = _zone_key(active_zone)

        if zone_cooldown.get(zk, -1) >= i:
            continue
        if require_leave_and_return and zk in zone_reentry:
            continue

        # ── Step 3: confirmation at zone ─────────────────────────────────────
        conf = check_confirmations_at_last_bar(df_15m_w, active_zone, direction, conf_cfg)
        if not conf.confirmed:
            continue

        signal_price = float(df_15m_w["close"].iloc[-1])

        # ── Step 4: trade geometry ────────────────────────────────────────────
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            continue

        # Fill at next bar open + spread
        raw_open = float(df_15m["open"].iloc[i + 1])
        entry    = raw_open + eff_spread if direction == "buy" else raw_open - eff_spread

        # Skip if fill bar opened outside the zone (gap)
        _gap_tol = 0.002
        if direction == "buy"  and entry > active_zone.top    * (1 + _gap_tol):
            continue
        if direction == "sell" and entry < active_zone.bottom * (1 - _gap_tol):
            continue

        # Translate SL distance from signal price to actual fill
        sl_dist = abs(signal_price - setup.sl)
        sl = entry - sl_dist if direction == "buy" else entry + sl_dist
        tp = setup.tp

        if direction == "buy"  and (sl >= entry or tp <= entry):
            continue
        if direction == "sell" and (sl <= entry or tp >= entry):
            continue

        lot = fixed_lot if fixed_lot > 0 else _lot_size(equity, abs(entry - sl), contract_size)

        # ── Simulate trade bar-by-bar ─────────────────────────────────────────
        outcome        = 0
        exit_price     = entry
        exit_bar       = i + max_forward_bars
        max_favourable = 0.0
        max_adverse    = 0.0

        for j in range(i + 1, min(i + 1 + max_forward_bars, n)):
            fh = float(df_15m["high"].iloc[j])
            fl = float(df_15m["low"].iloc[j])
            max_favourable = max(max_favourable, (fh - entry) if direction == "buy" else (entry - fl))
            max_adverse    = max(max_adverse,    (entry - fl) if direction == "buy" else (fh - entry))
            if direction == "buy":
                if fh >= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                if fl <= sl:  outcome = -1; exit_price = sl; exit_bar = j; break
            else:
                if fl <= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                if fh >= sl:  outcome = -1; exit_price = sl; exit_bar = j; break

        if direction == "buy":
            pnl = (exit_price - entry) * lot * contract_size
        else:
            pnl = (entry - exit_price) * lot * contract_size

        exit_ts = df_15m["timestamp"].iloc[min(exit_bar, n - 1)]
        _pending.append({
            "exit_bar":    exit_bar,
            "pnl":         pnl,
            "outcome":     outcome,
            "direction":   direction,
            "zk":          zk,
            "zone_bottom": active_zone.bottom,
            "zone_top":    active_zone.top,
            "trade_data": {
                "date":          ts_now,
                "exit_date":     exit_ts,
                "side":          direction,
                "entry":         round(entry, 5),
                "sl":            round(sl, 5),
                "tp":            round(tp, 5),
                "exit":          round(exit_price, 5),
                "outcome":       outcome,
                "lot":           lot,
                "pnl":           round(pnl, 2),
                "equity":        None,
                "max_favour":    round(max_favourable, 2),
                "max_adverse":   round(max_adverse, 2),
                "confirmations": conf.count,
                "signals":       "|".join(conf.signals),
                "h4_bias":       tf_result["h4_bias"],
                "zone_fresh":    active_zone.fresh,
                "zone_bottom":   active_zone.bottom,
                "zone_top":      active_zone.top,
                "zone_strength": active_zone.strength,
                "zone_kind":     active_zone.kind,
                "entry_mode":    setup.entry_mode,
                "tp_mode":       setup.tp_mode,
            },
        })

    # ── Flush positions still open at end of loop ─────────────────────────────
    for _p in sorted(_pending, key=lambda x: x["exit_bar"]):
        equity += _p["pnl"]
        if equity < 0:
            equity = 0.0
        _p["trade_data"]["equity"] = round(equity, 2)
        equity_curve.append(equity)
        trades.append(_p["trade_data"])

    if not trades:
        print("No trades generated. Widen date range, lower min_rr, or reduce min_confirmations.")
        return {}

    df_t     = pd.DataFrame(trades)
    total    = len(df_t)
    tp_hits  = int((df_t["outcome"] ==  1).sum())
    sl_hits  = int((df_t["outcome"] == -1).sum())
    expired  = int((df_t["outcome"] ==  0).sum())
    win_rate = tp_hits / max(total, 1) * 100
    winners  = df_t[df_t["pnl"] > 0]
    losers   = df_t[df_t["pnl"] < 0]
    eq_s     = pd.Series(equity_curve)
    max_dd   = round(((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100, 2)
    buy_df   = df_t[df_t["side"] == "buy"]
    sell_df  = df_t[df_t["side"] == "sell"]

    metrics = {
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
        "max_drawdown_%":  f"{max_dd:.2f}",
        "spread_pts":      eff_spread,
        "min_rr":          min_rr,
        "min_confirmations": min_confirmations,
        "directional_filter": directional_filter,
        "entry_mode":      "zone_boundary" if aggressive_entry else "confirmation",
        "tp_mode":         f"midline_{midline_pct:.0%}" if midline_tp else "zone_edge",
    }

    if not silent:
        print_report(metrics, run_label="Zone Strategy")

        print(f"\n  Confirmation stacking:")
        for cnt, freq in df_t["confirmations"].value_counts().sort_index().items():
            wr_s = (df_t[df_t["confirmations"] == cnt]["outcome"] == 1).mean() * 100
            print(f"    {cnt} confirmations : {freq:>4} trades  WR={wr_s:.0f}%")

        print(f"\n  H4 bias at entry:")
        for bias_val, grp in df_t.groupby("h4_bias"):
            wr_b = (grp["outcome"] == 1).mean() * 100
            print(f"    {bias_val:<14} : {len(grp):>4} trades  WR={wr_b:.0f}%")

        print(f"\n  Zone freshness:")
        for fresh_val, grp in df_t.groupby("zone_fresh"):
            wr_f = (grp["outcome"] == 1).mean() * 100
            lbl  = "fresh" if fresh_val else "tapped"
            print(f"    {lbl:<8} : {len(grp):>4} trades  WR={wr_f:.0f}%")

        print()
        cols = ["date", "side", "h4_bias", "signals", "confirmations",
                "zone_fresh", "zone_kind", "entry", "sl", "tp", "exit", "pnl", "outcome"]
        _present = [c for c in cols if c in df_t.columns]
        print(df_t[_present].to_string(index=False))
        print()

    if save_path:
        save_report(metrics, save_path)

    if chart:
        title = (
            f"Zone Strategy — {sym.upper()}  {start}→{end}  "
            f"{tp_hits}W/{sl_hits}L/{expired}E  WR{win_rate:.1f}%"
        )
        fig = plot_trades(df_15m, trades, title=title, start_cash=cash)
        fig.show()

    return metrics, df_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest: USTEC Zone-to-Zone")
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2024-01-01")
    parser.add_argument("--cash",     type=float, default=10_000.0)
    parser.add_argument("--min_rr",   type=float, default=1.5)
    parser.add_argument("--max_bars", type=int,   default=MAX_FORWARD_BARS)
    parser.add_argument("--spread",   type=float, default=None)
    parser.add_argument("--no_directional_filter", action="store_true")
    parser.add_argument("--no_neutral",            action="store_true")
    parser.add_argument("--h4_swing_left",  type=int, default=2)
    parser.add_argument("--h4_swing_right", type=int, default=2)
    parser.add_argument("--min_confirmations",   type=int,  default=1)
    parser.add_argument("--aggressive_boundary", action="store_true")
    parser.add_argument("--aggressive_entry",    action="store_true")
    parser.add_argument("--midline_tp",          action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--sl_buffer",   type=float, default=0.002)
    parser.add_argument("--fixed_lot",   type=float, default=0.0)
    parser.add_argument("--no_leave_return", action="store_true")
    parser.add_argument("--cooldown_bars",   type=int, default=15)
    parser.add_argument("--save",  default=None)
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    run_backtest(
        start=args.start,
        end=args.end,
        cash=args.cash,
        min_rr=args.min_rr,
        max_forward_bars=args.max_bars,
        spread=args.spread,
        directional_filter=not args.no_directional_filter,
        allow_neutral=not args.no_neutral,
        h4_swing_left=args.h4_swing_left,
        h4_swing_right=args.h4_swing_right,
        min_confirmations=args.min_confirmations,
        aggressive_boundary=args.aggressive_boundary,
        aggressive_entry=args.aggressive_entry,
        midline_tp=args.midline_tp,
        midline_pct=args.midline_pct,
        sl_buffer_pct=args.sl_buffer,
        fixed_lot=args.fixed_lot,
        require_leave_and_return=not args.no_leave_return,
        cooldown_bars=args.cooldown_bars,
        save_path=args.save,
        chart=args.chart,
        silent=args.silent,
    )


if __name__ == "__main__":
    main()
