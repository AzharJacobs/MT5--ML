"""
USTEC Zone-to-Zone backtest engine.

Bar-by-bar M15 simulation:
  Step 1  detect_zones on H4 (bounded to current bar — no lookahead)
  Step 2  analyse_timeframes → H4 bias + active zone + M15 pre-filter
  Step 3  check_confirmations_at_last_bar → entry pattern gate
  Step 4  build_trade_setup → entry / SL / TP geometry
"""

from __future__ import annotations

import argparse
import random
from typing import Optional

import numpy as np
import pandas as pd

from trading.strategies.zz.core.zones import ZoneConfig
from trading.strategies.zz.core.timeframe_structure import TFConfig, analyse_timeframes
from trading.strategies.zz.core.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from trading.strategies.zz.core.swing_structure import (
    detect_swings as _detect_swings_h4,
    label_structure as _label_structure_h4,
)
from trading.strategies.zz.core.trade_setup import TradeSetupConfig, setup_from_analysis
from trading.shared.backtest.report import print_report, save_report
from trading.shared.backtest.chart import plot_trades
from trading.shared.data_loader import get_connection

# Pull USTEC config defaults so run_backtest() parameters match config.yaml out-of-the-box
from trading.strategies.zz.ustec.strategy import (
    ZONE_MAX_LOSSES  as _CFG_ZONE_MAX_LOSSES,
    MAX_SL_PCT       as _CFG_MAX_SL_PCT,
    ENABLE_TRAILING  as _CFG_ENABLE_TRAILING,
    BE_BUFFER_PTS    as _CFG_BE_BUFFER_PTS,
    BE_TRIGGER_PTS   as _CFG_BE_TRIGGER_PTS,
    ATR_TRAIL_MULT   as _CFG_ATR_TRAIL_MULT,
    H4_REGIME_FILTER as _CFG_H4_REGIME_FILTER,
)


# ── Constants ─────────────────────────────────────────────────────────────────

H4_WINDOW  = 150
M15_WINDOW = 80
MAX_FORWARD_BARS = 350

RISK_PCT      = 0.01
COOLDOWN_LOSS = 48   # M15 bars (12 h)

SYMBOL_CONFIG = {
    "ustech": ("ustech_ohlcv", "ustech_ohlcv", 100.0),
    "xauusd": ("XAUUSD",       "xauusd_ohlcv", 100.0),
}
SYMBOL_SPREAD = {
    "ustech": 2.0,
    "xauusd": 0.0,
}


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


def _dynamic_spread(ts: pd.Timestamp, base: float = 4.0) -> float:
    h = ts.hour + ts.minute / 60.0
    if 21.0 <= h < 22.1:
        return base * 3.0
    if h < 6.0:
        return base * 2.0
    return base


def _count_overnights(ts_entry: pd.Timestamp, ts_exit: pd.Timestamp) -> int:
    return max(0, (ts_exit.date() - ts_entry.date()).days)


def _lot_size(equity: float, sl_dist: float, contract_size: float, risk_pct: float = RISK_PCT) -> float:
    if sl_dist <= 0:
        return 0.01
    lot = (equity * risk_pct) / (sl_dist * contract_size)
    return max(round(lot, 2), 0.01)


def _h4_bearish_regime(
    df_h4: pd.DataFrame,
    n_consec_ll: int = 2,
    swing_left: int = 2,
    swing_right: int = 2,
) -> bool:
    """
    True when the last n_consec_ll confirmed H4 swing lows are all Lower Lows,
    indicating a sustained bearish structural regime.  No lookahead beyond df_h4.
    swing_right bars of confirmation lag is inherent and expected.
    """
    if len(df_h4) < swing_left + swing_right + 2:
        return False
    swings  = _detect_swings_h4(df_h4, left=swing_left, right=swing_right)
    labeled = _label_structure_h4(swings)
    sl_seq  = [lbl for _, _, lbl in labeled if lbl in ("HL", "LL")]
    if len(sl_seq) < n_consec_ll:
        return False
    return all(lbl == "LL" for lbl in sl_seq[-n_consec_ll:])


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    start: str = "2023-01-01",
    end:   str = "2024-01-01",
    cash:  float = 10_000.0,
    min_rr: float = 1.5,
    max_forward_bars: int = MAX_FORWARD_BARS,
    symbol: str = "ustech",
    spread: Optional[float] = None,
    directional_filter: bool = True,
    allow_neutral: bool = True,
    h4_swing_left: int = 2,
    h4_swing_right: int = 2,
    min_confirmations: int = 1,
    aggressive_boundary: bool = False,
    excluded_from_count: Optional[list] = None,
    aggressive_entry: bool = False,
    midline_tp: bool = False,
    midline_pct: float = 0.50,
    sl_buffer_pct: float = 0.002,
    fixed_lot: float = 0.0,
    require_leave_and_return: bool = True,
    cooldown_bars: int = 15,
    zone_max_losses: int = _CFG_ZONE_MAX_LOSSES,
    dir_max_losses: int = 0,
    dir_cooldown_bars: int = 48,
    h4_regime_filter: bool = _CFG_H4_REGIME_FILTER,
    regime_conf_filter: bool = False,
    min_sl_pct: float = 0.0,
    max_sl_pct: float = _CFG_MAX_SL_PCT,
    enable_trailing:   bool  = _CFG_ENABLE_TRAILING,
    be_buffer_pts:     float = _CFG_BE_BUFFER_PTS,
    be_trigger_pts:    float = _CFG_BE_TRIGGER_PTS,
    atr_trail_mult:    float = _CFG_ATR_TRAIL_MULT,
    trail_swing_left:  int   = 2,
    trail_swing_right: int   = 2,
    realistic: bool = False,
    contract_size_override: Optional[float] = None,
    base_spread_pts: float = 4.0,
    swap_long_pts: float = -1.5,
    swap_short_pts: float = 0.8,
    slippage_max_pts: float = 3.0,
    margin_rate: float = 0.01,
    margin_call_pct: float = 0.60,
    save_path: Optional[str] = None,
    zone_guard: bool = True,
    chart: bool = False,
    gradual_filter: str = "all",   # "all" | "retest_only" | "tightened" | "no_multi_conf"
    silent: bool = False,
    risk_pct: Optional[float] = None,   # overrides RISK_PCT; ignored when fixed_lot > 0
) -> dict:
    sym = symbol.lower()
    if sym not in SYMBOL_CONFIG:
        raise ValueError(f"Unknown symbol '{symbol}'. Choose from: {list(SYMBOL_CONFIG)}")

    db_name, table, contract_size = SYMBOL_CONFIG[sym]
    if contract_size_override is not None:
        contract_size = contract_size_override
    eff_spread = spread if spread is not None else SYMBOL_SPREAD.get(sym, 0.0)
    if realistic and spread is None:
        eff_spread = base_spread_pts
    _risk_pct = risk_pct if risk_pct is not None else RISK_PCT

    db = get_connection()
    db.database = db_name
    db.connect()

    _pr = (lambda *a, **kw: None) if silent else print
    _pr(f"\nSymbol : {sym.upper()}  |  DB: {db_name}  |  Contract: ${contract_size}/pt/lot  |  Spread: {eff_spread} pts")
    _pr(f"Loading 15M data  {start} → {end} ...")
    df_15m = _load_ohlcv(db, table, "15min", start, end)
    _pr(f"Loading 4H  data  {start} → {end} ...")
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
        _pr(f"  (removed {_dupes} duplicate 4H rows — historical data artifact)")

    _pr(f"15M bars: {len(df_15m)} | 4H bars: {len(df_4h)}")
    df_4h["ema50"]  = df_4h["close"].ewm(span=50,  adjust=False).mean()
    df_4h["ema200"] = df_4h["close"].ewm(span=200, adjust=False).mean()

    tf_cfg = TFConfig(
        directional_filter=directional_filter,
        allow_neutral_up=allow_neutral,
        allow_neutral_down=allow_neutral,
        h4_swing_left=h4_swing_left,
        h4_swing_right=h4_swing_right,
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

    _pr(
        f"Config : directional_filter={directional_filter}  min_confirmations={min_confirmations}"
        f"  aggressive_entry={aggressive_entry}  midline_tp={midline_tp}\n"
    )

    equity         = cash
    equity_curve   = [cash]
    trades: list   = []
    skip_until     = -1
    zone_cooldown: dict  = {}
    zone_reentry:  dict  = {}
    won_zones:     set   = set()
    zone_outcome_history: dict = {}
    zone_consec_losses:   dict = {}
    zone_blacklist:       set  = set()
    zone_touch_tracker:   dict = {}   # zk → {bottom, top, exited}
    dir_consec_losses: dict = {"buy": 0, "sell": 0}
    dir_cooldown:      dict = {"buy": None, "sell": None}
    stopout_occurred:     bool = False
    total_swap_pnl:       float = 0.0
    total_slippage:       float = 0.0
    n      = len(df_15m)
    warmup = max(M15_WINDOW, 30)

    filters = {
        "in_position":   0,
        "tf_neutral":    0,
        "conf_failed":   0,
        "setup_invalid": 0,
        "zone_cooldown": 0,
        "zone_blacklist": 0,
        "dir_breaker":   0,
        "regime_filter": 0,
        "regime_conf_skip": 0,
        "sl_too_tight":   0,
        "sl_too_wide":    0,
        "margin_call":    0,
        "gradual_filter": 0,
    }

    for i in range(warmup, n - max_forward_bars):

        bar_h = float(df_15m["high"].iloc[i])
        bar_l = float(df_15m["low"].iloc[i])

        for _zkt, _zks in zone_touch_tracker.items():
            if not _zks["exited"]:
                if bar_h < _zks["bottom"] or bar_l > _zks["top"]:
                    _zks["exited"] = True

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
                    if bar_l <= z_top and bar_h >= z_bot:
                        if i >= state["earliest_reentry"]:
                            ready.append(zk)
            for zk in ready:
                del zone_reentry[zk]

        if i <= skip_until:
            filters["in_position"] += 1
            continue

        ts_now = df_15m["timestamp"].iloc[i]

        df_h4_w = df_4h[df_4h["timestamp"] <= ts_now].tail(H4_WINDOW).reset_index(drop=True)
        if len(df_h4_w) < 20:
            continue

        m15_start = max(0, i - M15_WINDOW + 1)
        df_15m_w  = df_15m.iloc[m15_start: i + 1].reset_index(drop=True)

        h4_up_to  = len(df_h4_w) - 1
        tf_result = analyse_timeframes(df_h4_w, df_15m_w, cfg=tf_cfg, h4_up_to_bar=h4_up_to)

        if tf_result["signal"] == "neutral":
            filters["tf_neutral"] += 1
            continue

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]

        if h4_regime_filter and direction == "buy":
            _price  = float(df_15m_w["close"].iloc[-1])
            _ema50  = float(df_h4_w["ema50"].iloc[-1])
            _ema200 = float(df_h4_w["ema200"].iloc[-1])
            if _price < _ema50 and _price < _ema200:
                filters["regime_filter"] += 1
                continue

        if dir_max_losses > 0:
            _dcd = dir_cooldown[direction]
            if _dcd is not None:
                if i >= _dcd:
                    dir_cooldown[direction] = None
                    dir_consec_losses[direction] = 0
                else:
                    filters["dir_breaker"] += 1
                    continue

        conf = check_confirmations_at_last_bar(df_15m_w, active_zone, direction, conf_cfg)
        if not conf.confirmed:
            filters["conf_failed"] += 1
            continue

        # Close-containment: signal bar (bar i) must close inside the zone.
        # Catches Pattern A — bars that wicked into the zone but closed outside.
        # zone_guard below handles Pattern B (genuine between-bar session gaps).
        _sig_close = float(df_15m_w["close"].iloc[-1])
        _close_tol = 0.001
        if direction == "buy"  and _sig_close > active_zone.top    * (1 + _close_tol):
            filters["conf_failed"] += 1
            continue
        if direction == "sell" and _sig_close < active_zone.bottom * (1 - _close_tol):
            filters["conf_failed"] += 1
            continue

        if regime_conf_filter and conf.count >= 2:
            _h4_ema200 = float(df_h4_w["ema200"].iloc[-1])
            if _sig_close > _h4_ema200:
                filters["regime_conf_skip"] += 1
                continue

        zk = _zone_key(active_zone)
        if zk not in zone_touch_tracker:
            zone_touch_tracker[zk] = {
                "bottom": active_zone.bottom,
                "top":    active_zone.top,
                "exited": False,
            }

        if zone_cooldown.get(zk, -1) >= i:
            filters["zone_cooldown"] += 1
            continue
        if require_leave_and_return and zk in zone_reentry:
            filters["zone_cooldown"] += 1
            continue

        zid = active_zone.zone_id
        if zone_max_losses > 0 and zid in zone_blacklist:
            filters["zone_blacklist"] += 1
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

        arrival_type = "retest" if zone_touch_tracker.get(zk, {}).get("exited", False) else "gradual"

        # Gradual (first-touch) entries are only allowed when at least one
        # strong reversal signal confirms the zone is reacting — otherwise wait
        # for a proper retest of the zone.
        _STRONG_SIGNALS = {"rejection_wick", "engulfing", "bos_msb", "choch"}
        if arrival_type == "gradual" and not _STRONG_SIGNALS.intersection(conf.signals):
            filters["gradual_filter"] += 1
            continue

        if arrival_type == "gradual" and gradual_filter != "all":
            if (
                gradual_filter == "retest_only"
                or (gradual_filter == "tightened" and (conf.count < 2 or not active_zone.fresh))
                or (gradual_filter == "no_multi_conf" and conf.count >= 2)
            ):
                filters["gradual_filter"] += 1
                continue

        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, setup_cfg)
        if not setup.valid:
            filters["setup_invalid"] += 1
            continue

        # Change 4: TP headroom — require at least 30 pts of reward vs current price
        if direction == "buy" and setup.tp < signal_price + 30.0:
            filters["setup_invalid"] += 1
            continue
        if direction == "sell" and setup.tp > signal_price - 30.0:
            filters["setup_invalid"] += 1
            continue

        raw_open = float(df_15m["open"].iloc[i + 1])
        entry    = raw_open
        bar_spread = _dynamic_spread(ts_now, base_spread_pts) if realistic else eff_spread
        if bar_spread > 0:
            entry = entry + bar_spread if direction == "buy" else entry - bar_spread

        # Skip if the fill bar opened outside the zone boundary.
        # 0.2% tolerance absorbs normal spread without being loose enough to
        # allow genuine gap-outside entries.  Pass zone_guard=False to bypass
        # (used only by the outside-zone diagnostic script).
        if zone_guard:
            _gap_tol = 0.002
            if direction == "buy" and entry > active_zone.top * (1 + _gap_tol):
                filters["setup_invalid"] += 1
                continue
            if direction == "sell" and entry < active_zone.bottom * (1 - _gap_tol):
                filters["setup_invalid"] += 1
                continue

        sl_dist = abs(signal_price - setup.sl)
        sl = entry - sl_dist if direction == "buy" else entry + sl_dist

        if setup.tp_mode == "midline" and setup.tp_zone is not None:
            full_tp = (setup.tp_zone.bottom if direction == "buy" else setup.tp_zone.top)
            tp = entry + midline_pct * (full_tp - entry)
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

        if min_sl_pct > 0.0:
            sl_dist_pct = abs(entry - sl) / entry * 100.0
            if sl_dist_pct < min_sl_pct:
                filters["sl_too_tight"] += 1
                continue

        if max_sl_pct > 0.0:
            _sl_pct = abs(entry - sl) / entry * 100.0
            if _sl_pct > max_sl_pct:
                filters["sl_too_wide"] += 1
                continue

        lot = fixed_lot if fixed_lot > 0 else _lot_size(equity, abs(entry - sl), contract_size, _risk_pct)

        if realistic:
            margin_needed = lot * contract_size * entry * margin_rate
            if equity < margin_needed * margin_call_pct:
                filters["margin_call"] += 1
                continue

        outcome         = 0
        exit_price      = entry
        exit_bar        = i + max_forward_bars
        max_favourable  = 0.0
        max_adverse     = 0.0

        if realistic and lot * contract_size > 0:
            price_to_zero      = equity / (lot * contract_size)
            stopout_price_buy  = entry - price_to_zero
            stopout_price_sell = entry + price_to_zero
        else:
            stopout_price_buy = stopout_price_sell = None

        # ── Trailing stop state ───────────────────────────────────────────────
        _be_triggered = False
        _trail_sl     = sl
        if enable_trailing:
            _midline = (entry + tp) / 2
            # be_trigger_pts > 0 → trigger BE at fixed pts offset; else use midline
            if be_trigger_pts > 0:
                _be_trig_buy  = entry + be_trigger_pts
                _be_trig_sell = entry - be_trigger_pts
            else:
                _be_trig_buy  = _midline
                _be_trig_sell = _midline
            _atr_trs = []
            for _ak in range(max(1, i - 13), i + 1):
                _h  = float(df_15m["high"].iloc[_ak])
                _l  = float(df_15m["low"].iloc[_ak])
                _pc = float(df_15m["close"].iloc[_ak - 1])
                _atr_trs.append(max(_h - _l, abs(_h - _pc), abs(_l - _pc)))
            _atr14 = sum(_atr_trs) / max(len(_atr_trs), 1)
        else:
            _be_trig_buy = _be_trig_sell = _midline = 0.0
            _atr14 = 0.0

        for j in range(i + 1, min(i + 1 + max_forward_bars, n)):
            fh = float(df_15m["high"].iloc[j])
            fl = float(df_15m["low"].iloc[j])
            favour  = (fh - entry) if direction == "buy" else (entry - fl)
            adverse = (entry - fl) if direction == "buy" else (fh - entry)
            max_favourable = max(max_favourable, favour)
            max_adverse    = max(max_adverse,    adverse)
            # ── Trailing stop update ──────────────────────────────────────────
            if enable_trailing:
                fc = float(df_15m["close"].iloc[j])
                # 1. Break-even at midline: trigger on bar close, not high/low
                if not _be_triggered:
                    if direction == "buy"  and fc >= _midline:
                        _be_triggered = True
                        _trail_sl = max(_trail_sl, entry + be_buffer_pts)
                    elif direction == "sell" and fc <= _midline:
                        _be_triggered = True
                        _trail_sl = min(_trail_sl, entry - be_buffer_pts)
                if _be_triggered and be_trigger_pts == 0:
                    # 2. Ratchet behind newly confirmed M15 swing — legacy midline mode only.
                    # When be_trigger_pts > 0 (pure BE stop), we skip swing trailing so the
                    # SL stays fixed at entry+buffer and doesn't generate premature extra exits.
                    _sb = j - trail_swing_right
                    if _sb > i + trail_swing_left:
                        _sb_l = float(df_15m["low"].iloc[_sb])
                        _sb_h = float(df_15m["high"].iloc[_sb])
                        _ll = all(float(df_15m["low"].iloc[_sb - m])  > _sb_l for m in range(1, trail_swing_left  + 1))
                        _rl = all(float(df_15m["low"].iloc[_sb + m])  > _sb_l for m in range(1, trail_swing_right + 1))
                        _lh = all(float(df_15m["high"].iloc[_sb - m]) < _sb_h for m in range(1, trail_swing_left  + 1))
                        _rh = all(float(df_15m["high"].iloc[_sb + m]) < _sb_h for m in range(1, trail_swing_right + 1))
                        if direction == "buy" and _ll and _rl:
                            _cand = _sb_l - be_buffer_pts
                            if _cand > _trail_sl:
                                _trail_sl = _cand
                        elif direction == "sell" and _lh and _rh:
                            _cand = _sb_h + be_buffer_pts
                            if _cand < _trail_sl:
                                _trail_sl = _cand
                    # 3. ATR backstop: SL can't lag more than mult×ATR from price
                    if atr_trail_mult > 0 and _atr14 > 0:
                        if direction == "buy":
                            _trail_sl = max(_trail_sl, fc - atr_trail_mult * _atr14)
                        else:
                            _trail_sl = min(_trail_sl, fc + atr_trail_mult * _atr14)
            # ─────────────────────────────────────────────────────────────────
            if direction == "buy":
                if stopout_price_buy is not None and fl <= stopout_price_buy:
                    outcome = -1; exit_price = stopout_price_buy; exit_bar = j
                    stopout_occurred = True; break
                if fh >= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                if fl <= _trail_sl:  outcome = -1; exit_price = _trail_sl; exit_bar = j; break
            else:
                if stopout_price_sell is not None and fh >= stopout_price_sell:
                    outcome = -1; exit_price = stopout_price_sell; exit_bar = j
                    stopout_occurred = True; break
                if fl <= tp:  outcome =  1; exit_price = tp; exit_bar = j; break
                if fh >= _trail_sl:  outcome = -1; exit_price = _trail_sl; exit_bar = j; break

        slip = 0.0
        if realistic and outcome == -1 and not stopout_occurred:
            slip = random.uniform(0.0, slippage_max_pts)
            if direction == "buy":
                exit_price -= slip
            else:
                exit_price += slip
            total_slippage += slip * lot * contract_size

        if direction == "buy":
            pnl = (exit_price - entry) * lot * contract_size
        else:
            pnl = (entry - exit_price) * lot * contract_size

        swap_pnl = 0.0
        if realistic and exit_bar > i + 1:
            ts_entry_bar = df_15m["timestamp"].iloc[i + 1]
            ts_exit_bar  = df_15m["timestamp"].iloc[min(exit_bar, n - 1)]
            nights = _count_overnights(ts_entry_bar, ts_exit_bar)
            if nights > 0:
                swap_rate = swap_long_pts if direction == "buy" else swap_short_pts
                swap_pnl  = swap_rate * lot * contract_size * nights
                pnl      += swap_pnl
                total_swap_pnl += swap_pnl

        equity += pnl
        if equity < 0:
            equity = 0.0
        equity_curve.append(equity)
        skip_until = exit_bar

        zone_outcome_history.setdefault(zid, []).append(outcome)

        if outcome == 1:
            won_zones.add(zk)
            zone_consec_losses[zid] = 0
            if dir_max_losses > 0:
                dir_consec_losses[direction] = 0
                dir_cooldown[direction] = None
            if require_leave_and_return:
                zone_reentry[zk] = {
                    "phase":            "exit",
                    "bottom":           active_zone.bottom,
                    "top":              active_zone.top,
                    "earliest_reentry": exit_bar + cooldown_bars,
                }
            else:
                zone_cooldown[zk] = exit_bar + cooldown_bars
        elif outcome == -1:
            zone_cooldown[zk] = exit_bar + COOLDOWN_LOSS
            # Only blacklist / count losses when the trade actually lost money.
            # A trailing-stop exit at BE+buffer is outcome=-1 but pnl > 0; don't penalise the zone.
            if pnl < 0:
                if zone_max_losses > 0:
                    new_count = zone_consec_losses.get(zid, 0) + 1
                    zone_consec_losses[zid] = new_count
                    if new_count >= zone_max_losses:
                        zone_blacklist.add(zid)
                if dir_max_losses > 0:
                    new_dir = dir_consec_losses[direction] + 1
                    dir_consec_losses[direction] = new_dir
                    if new_dir >= dir_max_losses:
                        dir_cooldown[direction] = exit_bar + dir_cooldown_bars
        else:
            zone_consec_losses[zid] = 0

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
            "confirmations": conf.count,
            "signals":       "|".join(conf.signals),
            "h4_bias":       tf_result["h4_bias"],
            "entry_mode":    setup.entry_mode,
            "tp_mode":       setup.tp_mode,
            "is_retest":     is_retest,
            "prior_bucket":  prior_bucket,
            "arrival_type":  arrival_type,
            "zone_fresh":    active_zone.fresh,
            "zone_bottom":   active_zone.bottom,
            "zone_top":      active_zone.top,
            "zone_strength": active_zone.strength,
            "zone_kind":     active_zone.kind,
            "structure":       {"zone": (active_zone.bottom, active_zone.top)},
            "signal_close":    signal_price,
            "raw_open":        raw_open,
        })
        zone_touch_tracker.pop(zk, None)

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
        "lowest_equity":   min(equity_curve),
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
        "dir_max_losses":      dir_max_losses if dir_max_losses > 0 else "disabled",
        "dir_cooldown_bars":   dir_cooldown_bars if dir_max_losses > 0 else "n/a",
        "h4_regime_filter":    h4_regime_filter,
        "regime_conf_filter":  regime_conf_filter,
        "enable_trailing":     enable_trailing,
        "be_trigger_pts":      be_trigger_pts if enable_trailing else "disabled",
        "be_buffer_pts":       be_buffer_pts  if enable_trailing else "disabled",
        "atr_trail_mult":      atr_trail_mult if enable_trailing else "disabled",
        "min_sl_pct":          min_sl_pct if min_sl_pct > 0 else "disabled",
        "realistic_mode":      realistic,
        "contract_size":       contract_size,
        "total_swap_pnl":      f"${total_swap_pnl:+.2f}" if realistic else "n/a",
        "total_slippage_cost": f"${-abs(total_slippage):.2f}" if realistic else "n/a",
        "stopout_occurred":    stopout_occurred if realistic else "n/a",
    }

    if not silent:
        print_report(metrics, run_label="Zone Strategy")

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

        print(f"\n  Confirmation stacking (out of 5):")
        for cnt, freq in stack_counts.items():
            wr_stack = df_t[df_t["confirmations"] == cnt]
            wr_pct   = (wr_stack["outcome"] == 1).mean() * 100
            print(f"    {cnt} confirmations : {freq:>4} trades  WR={wr_pct:.0f}%")

        print(f"\n  H4 bias at entry:")
        for bias_val, grp in df_t.groupby("h4_bias"):
            wr_b = (grp["outcome"] == 1).mean() * 100
            print(f"    {bias_val:<14} : {len(grp):>4} trades  WR={wr_b:.0f}%")
        print()

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

        # ── Arrival-type split ────────────────────────────────────────────────
        print(f"  {'─'*62}")
        print(f"  Arrival-type split")
        print(f"  {'─'*62}")
        print(f"  {'Type':<10} {'Trades':>7} {'Wins':>5} {'WR%':>6} {'Net PnL':>10} {'Avg PnL':>10}")
        print(f"  {'─'*62}")
        _at_rows = []
        for atype in ("retest", "gradual"):
            _g = df_t[df_t["arrival_type"] == atype]
            if len(_g) == 0:
                _at_rows.append((atype, 0, 0, 0.0, 0.0, 0.0))
                continue
            _wr  = (_g["outcome"] == 1).mean() * 100
            _net = _g["pnl"].sum()
            _avg = _g["pnl"].mean()
            _at_rows.append((atype, len(_g), int((_g["outcome"] == 1).sum()), _wr, _net, _avg))
            print(f"  {atype:<10} {len(_g):>7} {int((_g['outcome']==1).sum()):>5} "
                  f"{_wr:>5.1f}% {_net:>+10.2f} {_avg:>+10.2f}")
        _retest_net  = next((r[4] for r in _at_rows if r[0] == "retest"),  0.0)
        print(f"  {'─'*62}")
        print(f"  Net PnL excl. gradual : ${_retest_net:+.2f}")
        print(f"  {'─'*62}")
        print()

        cols = ["date", "side", "h4_bias", "signals", "confirmations",
                "arrival_type", "is_retest", "prior_bucket", "entry", "sl", "tp", "exit", "pnl", "outcome"]
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest: USTEC Zone-to-Zone")
    parser.add_argument("--symbol",   default="ustech", choices=list(SYMBOL_CONFIG))
    parser.add_argument("--start",    default="2023-01-01")
    parser.add_argument("--end",      default="2024-01-01")
    parser.add_argument("--cash",     type=float, default=10_000.0)
    parser.add_argument("--min_rr",   type=float, default=1.5)
    parser.add_argument("--max_bars", type=int,   default=MAX_FORWARD_BARS)
    parser.add_argument("--spread",   type=float, default=None)
    parser.add_argument("--no_directional_filter", action="store_true")
    parser.add_argument("--h4_swing_left",  type=int, default=2)
    parser.add_argument("--h4_swing_right", type=int, default=2)
    parser.add_argument("--no_neutral",            action="store_true")
    parser.add_argument("--min_confirmations", type=int,   default=1)
    parser.add_argument("--aggressive_boundary", action="store_true")
    parser.add_argument("--exclude_signals", default="")
    parser.add_argument("--zone_max_losses", type=int, default=0)
    parser.add_argument("--dir_max_losses",   type=int, default=0)
    parser.add_argument("--dir_cooldown",     type=int, default=48)
    parser.add_argument("--h4_regime_filter", action="store_true")
    parser.add_argument("--regime_conf_filter", action="store_true")
    parser.add_argument("--aggressive_entry", action="store_true")
    parser.add_argument("--midline_tp",  action="store_true")
    parser.add_argument("--midline_pct", type=float, default=0.50)
    parser.add_argument("--sl_buffer",   type=float, default=0.002)
    parser.add_argument("--fixed_lot",   type=float, default=0.0)
    parser.add_argument("--save",  default=None)
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--cooldown_bars",  type=int,  default=15)
    parser.add_argument("--no_leave_return", action="store_true")
    parser.add_argument("--min_sl_pct",  type=float, default=0.0)
    parser.add_argument("--realistic",   action="store_true")
    parser.add_argument("--trailing",    dest="enable_trailing", action="store_true",  default=_CFG_ENABLE_TRAILING)
    parser.add_argument("--no_trailing", dest="enable_trailing", action="store_false")
    parser.add_argument("--be_trigger",  type=float, default=_CFG_BE_TRIGGER_PTS)
    parser.add_argument("--be_buffer",   type=float, default=_CFG_BE_BUFFER_PTS)
    parser.add_argument("--atr_mult",    type=float, default=_CFG_ATR_TRAIL_MULT)
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
        h4_swing_left=args.h4_swing_left,
        h4_swing_right=args.h4_swing_right,
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
        dir_max_losses=args.dir_max_losses,
        dir_cooldown_bars=args.dir_cooldown,
        h4_regime_filter=args.h4_regime_filter,
        regime_conf_filter=args.regime_conf_filter,
        min_sl_pct=args.min_sl_pct,
        realistic=args.realistic,
        enable_trailing=args.enable_trailing,
        be_trigger_pts=args.be_trigger,
        be_buffer_pts=args.be_buffer,
        atr_trail_mult=args.atr_mult,
        save_path=args.save,
        chart=args.chart,
    )


if __name__ == "__main__":
    main()
