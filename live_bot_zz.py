"""
live_bot_zz.py — Zone-to-Zone (Z&Z) strategy live trading bot.

Instrument : USTECH (Exness DEMO only — hard-checked at startup, will refuse to run on live)
Timeframes : H4 (bias + zone marking)  +  M15 (entries + confirmation)

Strategy pipeline — exact match to the validated backtest baseline
(engine_zones.py, require_leave_and_return=True, min_confirmations=1):

  Step 1  detect_zones(H4)            — strategy.zones
  Step 2  analyse_timeframes          — strategy.timeframe_structure
  Step 3  check_confirmations_at_last_bar — strategy.confirmations
          CHoCH is logged in every trade record; not a gate beyond min_conf=1
  Step 4  setup_from_analysis         — strategy.trade_setup

Cooldown (identical to backtest):
  WIN  → require_leave_and_return=True with 15-bar (≈4h) minimum floor
  LOSS → 48-bar (12h) block on that zone

IMPORTS: ONLY Z&Z modules + connection/execution/DB infra.
         NO base_strategy, NO market_structure, NO ML models, NO feature_engineer.

Usage:
  python live_bot_zz.py --mode paper          # paper trade (default, safe)
  python live_bot_zz.py --mode mt5            # MT5 demo — must be trial/demo server
  python live_bot_zz.py --mode mt5 --once     # evaluate one bar then exit
  python live_bot_zz.py --mode paper --once
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Z&Z strategy modules ONLY ─────────────────────────────────────────────────
from strategy.zones import ZoneConfig, detect_zones, update_freshness
from strategy.timeframe_structure import TFConfig, analyse_timeframes
from strategy.confirmations import ConfirmationConfig, check_confirmations_at_last_bar
from strategy.trade_setup import TradeSetupConfig, setup_from_analysis

# ── Infrastructure (connection / execution / DB) ──────────────────────────────
from data.loader import get_connection
from execution.broker_interface import BrokerInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_zz")


# ---------------------------------------------------------------------------
# Validated baseline constants — must match engine_zones.py exactly
# ---------------------------------------------------------------------------

SYMBOL          = "USTECm"
DB_NAME         = "ustech_ohlcv"
TABLE           = "ustech_ohlcv"
CONTRACT_SIZE   = 100.0       # $100/pt/lot (Exness USTEC)
SPREAD_PTS      = 2.0         # applied at fill for paper mode; MT5 uses real fill
RISK_PCT        = 0.01        # 1% of equity per trade (unused — fixed lot mode)
FIXED_LOTS      = 0.05        # fixed lot size per trade
MAX_POSITIONS   = 2           # maximum concurrent open positions
MIN_RR          = 1.5
MIN_CONF        = 1
MIN_SL_PCT      = 0.25        # skip trade if SL is closer than 0.25% from entry
H4_WINDOW       = 150         # H4 bars for zone + bias
M15_WINDOW      = 80          # M15 bars for confirmations
COOLDOWN_LOSS_H = 12          # hours to block zone after a loss (≈48 × 15M bars)
COOLDOWN_WIN_FLOOR_H = 3.75   # minimum hours after a win before leave-and-return eligible
                               # (≈15 × 15M bars)
TAP_TOL         = 0.001       # zone edge tolerance for leave-and-return tracking

DEMO_KEYWORDS   = ("trial", "demo", "test")   # server name must contain one of these


# ---------------------------------------------------------------------------
# Config factory — identical to engine_zones.py run_backtest() defaults
# ---------------------------------------------------------------------------

def _make_configs():
    tf_cfg = TFConfig(
        directional_filter=True,
        allow_neutral_up=True,
        allow_neutral_down=True,
        h4_swing_left=2,
        h4_swing_right=2,
        h4_zone_cfg=ZoneConfig(
            impulse_atr_mult=2.0,
            body_ratio_min=0.50,
            min_departure_candles=2,
            departure_window=6,
            base_lookback=5,
            min_strength=1.5,
            tap_tolerance_pct=TAP_TOL,
        ),
        prefer_fresh_h4=True,
        m15_tap_lookback=20,
        m15_tap_tolerance_pct=TAP_TOL,
        require_m15_directional_close=True,
    )
    conf_cfg = ConfirmationConfig(
        min_confirmations=MIN_CONF,
        aggressive_boundary=False,
        engulf_full_body=True,
        wick_ratio_min=0.60,
        body_ratio_max=0.35,
        close_in_upper_half=True,
        bos_lookback=15,
        bos_swing_n=2,
        structure_lookback=25,
        structure_swing_n=2,
        choch_lookback=20,
        choch_swing_left=2,
        choch_swing_right=2,
        choch_min_swings=2,
        zone_tolerance_pct=TAP_TOL,
    )
    setup_cfg = TradeSetupConfig(
        aggressive_entry=False,
        sl_buffer_pct=0.002,
        sl_min_points=0.0,
        midline_tp=False,
        midline_pct=0.50,
        tp_prefer_fresh=True,
        min_rr=MIN_RR,
    )
    return tf_cfg, conf_cfg, setup_cfg


# ---------------------------------------------------------------------------
# Logger — CSV + console, one file per day
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRADE_FIELDS = [
    "timestamp", "symbol", "mode",
    "zone_id", "zone_bottom", "zone_top", "zone_kind", "zone_strength", "zone_fresh",
    "h4_bias", "direction",
    "signals_fired", "confirmation_count", "choch_fired",
    "entry_mode", "tp_mode",
    "entry", "sl", "tp", "rr", "lots",
    "ticket", "signal_price", "fill_price", "outcome", "close_price", "pnl",
    "prior_bucket",
]

SKIP_FIELDS = [
    "timestamp", "symbol", "reason",
    "h4_bias", "zone_tapped", "active_zone",
]


def _csv_append(path: Path, fields: list, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


class ZZLogger:
    def __init__(self, mode: str):
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.trade_path = LOG_DIR / f"live_zz_trades_{date_str}.csv"
        self.skip_path  = LOG_DIR / f"live_zz_skipped_{date_str}.csv"
        self.mode = mode
        log.info("Trade log : %s", self.trade_path)
        log.info("Skip  log : %s", self.skip_path)

    def log_trade_open(self, record: dict) -> None:
        _csv_append(self.trade_path, TRADE_FIELDS, record)
        log.info(
            "TRADE OPEN | %s %s | zone=%s | bias=%s | sigs=%s(%d) | "
            "entry=%.2f sl=%.2f tp=%.2f RR=%.2f lots=%.2f",
            record["direction"].upper(), record["symbol"],
            record["zone_id"], record["h4_bias"],
            record["signals_fired"], record["confirmation_count"],
            record["entry"], record["sl"], record["tp"],
            record["rr"], record["lots"],
        )

    def log_trade_close(self, record: dict) -> None:
        _csv_append(self.trade_path, TRADE_FIELDS, {**record})
        outcome_str = "WIN" if record["outcome"] == 1 else ("LOSS" if record["outcome"] == -1 else "EXP")
        log.info(
            "TRADE CLOSE | %s | fill=%.2f pnl=%+.2f",
            outcome_str, record.get("close_price", 0), record.get("pnl", 0),
        )

    def log_skip(self, ts: datetime, reason: str, tf_result: dict) -> None:
        row = {
            "timestamp":   ts.isoformat(),
            "symbol":      SYMBOL,
            "reason":      reason,
            "h4_bias":     tf_result.get("h4_bias", ""),
            "zone_tapped": tf_result.get("zone_tapped", False),
            "active_zone": str(tf_result.get("active_zone")),
        }
        _csv_append(self.skip_path, SKIP_FIELDS, row)
        log.debug("SKIP | %s | %s", reason, tf_result.get("h4_bias", ""))


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

_MT5_TF_MAP = None  # populated on first call to _load_ohlcv_mt5

def _load_ohlcv_mt5(timeframe: str, n_bars: int) -> pd.DataFrame:
    """
    Fetch closed bars directly from the connected MT5 terminal.
    position=1 skips bar 0 (currently forming) so we only see closed bars.
    Timestamps are converted from Exness broker time (GMT+3) to UTC.
    """
    import MetaTrader5 as mt5
    global _MT5_TF_MAP
    if _MT5_TF_MAP is None:
        _MT5_TF_MAP = {
            "15min": mt5.TIMEFRAME_M15,
            "4H":    mt5.TIMEFRAME_H4,
        }
    tf_const = _MT5_TF_MAP.get(timeframe)
    if tf_const is None:
        log.error("_load_ohlcv_mt5: unknown timeframe '%s'", timeframe)
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(SYMBOL, tf_const, 1, n_bars)
    if rates is None or len(rates) == 0:
        log.error("_load_ohlcv_mt5: no data for %s %s — %s", SYMBOL, timeframe, mt5.last_error())
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    # Exness broker timestamps are GMT+3; convert to naive UTC
    df["timestamp"] = (
        pd.to_datetime(df["time"], unit="s")
        .dt.tz_localize("Etc/GMT-3")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def _load_ohlcv(db, timeframe: str, n_bars: int) -> pd.DataFrame:
    """Fetch bars from the PostgreSQL DB (used in paper mode)."""
    q = (
        f"SELECT * FROM {TABLE} WHERE timeframe = %s "
        f"ORDER BY timestamp DESC LIMIT %s"
    )
    df = db.fetch_dataframe(q, (timeframe, n_bars))
    if df is None or df.empty:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Lot sizing
# ---------------------------------------------------------------------------

def _lot_size_formula(equity: float, sl_dist: float) -> float:
    if sl_dist <= 0:
        return 0.01
    lot = (equity * RISK_PCT) / (sl_dist * CONTRACT_SIZE)
    return max(round(lot, 2), 0.01)


def _lot_size_mt5(symbol: str, sl_dist: float, fallback: float = 0.01) -> float:
    try:
        import MetaTrader5 as mt5
        info    = mt5.symbol_info(symbol)
        account = mt5.account_info()
        if info is None or account is None:
            return fallback
        balance      = float(account.balance)
        risk_amount  = balance * RISK_PCT
        tick_val     = float(info.trade_tick_value)
        tick_sz      = float(info.trade_tick_size)
        sl_ticks     = sl_dist / tick_sz
        risk_per_lot = sl_ticks * tick_val
        if risk_per_lot <= 0:
            return fallback
        lots = risk_amount / risk_per_lot
        lots = max(lots, info.volume_min)
        lots = min(lots, info.volume_max)
        step = info.volume_step
        lots = round(round(lots / step) * step, 8)
        return lots
    except Exception as exc:
        log.warning("lot_size_mt5 fallback: %s", exc)
        return fallback


# ---------------------------------------------------------------------------
# Main bot
# ---------------------------------------------------------------------------

class ZZLiveBot:

    def __init__(self, broker: BrokerInterface, mode: str):
        self.broker = broker
        self.mode   = mode
        self.zlog   = ZZLogger(mode)

        # ── Validate demo account ─────────────────────────────────────────
        self._demo_guard()

        # ── Build exact baseline configs ──────────────────────────────────
        self.tf_cfg, self.conf_cfg, self.setup_cfg = _make_configs()

        # ── Zone cooldown state (mirrors engine_zones.py exactly) ─────────
        self.zone_cooldown: dict  = {}   # key → datetime until which zone is blocked
        self.zone_reentry:  dict  = {}   # key → {phase, bottom, top, earliest}
        self.won_zones:     set   = set()
        self.zone_outcome_history: dict = {}

        # ── Open position tracking ────────────────────────────────────────
        self.open_positions: list = []  # up to MAX_POSITIONS concurrent

        # ── DB connection ─────────────────────────────────────────────────
        self.db = get_connection()
        self.db.database = DB_NAME
        self.db.connect()

        # ── Verify symbol exists on this MT5 server ───────────────────────
        self.contract_size = CONTRACT_SIZE  # fallback; updated from symbol_info below
        if mode == "mt5":
            import MetaTrader5 as mt5
            info = mt5.symbol_info(SYMBOL)
            if info is None:
                similar = [s.name for s in (mt5.symbols_get() or [])
                           if SYMBOL[:5] in s.name.upper() or "NAS" in s.name.upper()][:15]
                log.warning(
                    "SYMBOL '%s' not found on this server — check the symbol name. "
                    "Similar symbols available: %s", SYMBOL, similar
                )
            else:
                cs = float(getattr(info, "trade_contract_size", 0) or 0)
                if cs > 0:
                    self.contract_size = cs
                log.info(
                    "Symbol verified: %s | trade_mode=%d | digits=%d | contract_size=%.2f",
                    SYMBOL, info.trade_mode, info.digits, self.contract_size,
                )

        log.info("ZZLiveBot ready | mode=%s | symbol=%s | min_conf=%d | "
                 "leave_and_return=True | loss_cooldown=%dh | lots=%.2f | max_pos=%d",
                 mode, SYMBOL, MIN_CONF, COOLDOWN_LOSS_H, FIXED_LOTS, MAX_POSITIONS)
        log.info("Config: directional_filter=True allow_neutral=True "
                 "aggressive_entry=False midline_tp=False min_rr=%.1f "
                 "sl_buffer=0.002 spread=%.1fpts min_sl_pct=%.2f%%",
                 MIN_RR, SPREAD_PTS, MIN_SL_PCT)

    # ── Demo guard ────────────────────────────────────────────────────────

    def _demo_guard(self) -> None:
        server = os.environ.get("MT5_SERVER", "")
        if not any(kw in server.lower() for kw in DEMO_KEYWORDS):
            raise SystemExit(
                f"SAFETY STOP: MT5_SERVER='{server}' does not look like a demo account "
                f"(must contain one of {DEMO_KEYWORDS}). "
                f"This bot will NOT run on a live account."
            )
        log.info("Demo guard PASSED — server='%s'", server)

    # ── Leave-and-return zone state update ────────────────────────────────

    def _update_zone_reentry(self, now: datetime, bar_h: float, bar_l: float) -> None:
        tol  = TAP_TOL
        done = []
        for zk, state in self.zone_reentry.items():
            z_top = state["top"]    * (1 + tol)
            z_bot = state["bottom"] * (1 - tol)
            if state["phase"] == "exit":
                if bar_h < z_bot or bar_l > z_top:
                    state["phase"] = "return"
                    log.debug("Zone %s: price left zone, waiting for return", zk)
            else:  # "return"
                if bar_l <= z_top and bar_h >= z_bot:
                    if now >= state["earliest"]:
                        done.append(zk)
                        log.debug("Zone %s: leave-and-return complete — eligible", zk)
        for zk in done:
            del self.zone_reentry[zk]

    # ── Zone eligibility check ────────────────────────────────────────────

    def _zone_blocked(self, zk: tuple, now: datetime) -> bool:
        # Bar-count (time-based) cooldown — used for losses
        until = self.zone_cooldown.get(zk)
        if until is not None and now < until:
            return True
        # Leave-and-return — used for wins
        if zk in self.zone_reentry:
            return True
        return False

    # ── Zone key ──────────────────────────────────────────────────────────

    @staticmethod
    def _zk(zone) -> tuple:
        return (round(zone.bottom, 1), round(zone.top, 1))

    # ── Prior bucket (for logging) ────────────────────────────────────────

    def _prior_bucket(self, zid: str) -> str:
        history = self.zone_outcome_history.get(zid, [])
        if not history:
            return "first_attempt"
        last = history[-1]
        if last == 1:
            return "post_win"
        if last == -1:
            return "post_loss"
        return "post_expired"

    # ── Single bar evaluation ─────────────────────────────────────────────

    def run_once(self) -> None:
        now = datetime.now(timezone.utc)
        log.info("── Bar at %s UTC ──", now.strftime("%Y-%m-%d %H:%M"))

        # ── Fetch data ────────────────────────────────────────────────────
        try:
            if self.mode == "mt5":
                # Pull bars directly from the connected MT5 terminal — always
                # current, no DB lag, no collector dependency.
                import MetaTrader5 as mt5
                info = mt5.symbol_info(SYMBOL)
                if info is None:
                    similar = [s.name for s in (mt5.symbols_get() or [])
                               if SYMBOL[:5] in s.name.upper() or "NAS" in s.name.upper()][:10]
                    log.warning(
                        "symbol_info(%s) returned None — symbol may not exist on this server. "
                        "Similar symbols: %s", SYMBOL, similar
                    )
                    return
                if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                    log.info(
                        "Market not open for %s (trade_mode=%d) — skipping bar",
                        SYMBOL, info.trade_mode,
                    )
                    return
                df_15m = _load_ohlcv_mt5("15min", M15_WINDOW + 10)
                df_4h  = _load_ohlcv_mt5("4H",    H4_WINDOW  + 10)
            else:
                df_15m = _load_ohlcv(self.db, "15min", M15_WINDOW + 10)
                df_4h  = _load_ohlcv(self.db, "4H",    H4_WINDOW  + 10)
        except Exception as exc:
            log.error("Data fetch failed: %s", exc)
            return

        if df_15m.empty or df_4h.empty or len(df_15m) < M15_WINDOW:
            log.warning("Insufficient data — 15M=%d 4H=%d", len(df_15m), len(df_4h))
            return

        # Trim to window sizes
        df_15m_w = df_15m.tail(M15_WINDOW).reset_index(drop=True)
        df_4h_w  = df_4h.tail(H4_WINDOW).reset_index(drop=True)

        # Staleness guard — only applies to paper mode (DB-backed).
        # MT5 mode fetches directly from the terminal so data is always current.
        if self.mode == "paper":
            last_bar_ts = df_15m_w["timestamp"].iloc[-1]
            if last_bar_ts.tzinfo is None:
                last_bar_ts = last_bar_ts.replace(tzinfo=timezone.utc)
            data_age = (now - last_bar_ts).total_seconds() / 60
            if data_age > 30:
                log.warning(
                    "DB data stale — last 15M bar is %.0f min old (%s) — skipping bar. "
                    "Check that the MT5 collector is running.",
                    data_age, last_bar_ts.strftime("%H:%M UTC"),
                )
                return

        # Update leave-and-return state with the latest bar
        bar_h = float(df_15m_w["high"].iloc[-1])
        bar_l = float(df_15m_w["low"].iloc[-1])
        self._update_zone_reentry(now, bar_h, bar_l)

        # ── Check if any open positions have closed ───────────────────────
        for pos in list(self.open_positions):
            self._check_open_position(pos, bar_h, bar_l, now)

        # Cap at MAX_POSITIONS concurrent trades
        if len(self.open_positions) >= MAX_POSITIONS:
            log.info("Max positions (%d) reached — skipping entry evaluation", MAX_POSITIONS)
            return

        # ── Step 2: timeframe analysis ────────────────────────────────────
        tf_result = analyse_timeframes(
            df_4h_w, df_15m_w,
            cfg=self.tf_cfg,
            h4_up_to_bar=len(df_4h_w) - 1,
        )

        if tf_result["signal"] == "neutral":
            self.zlog.log_skip(now, tf_result["reason"], tf_result)
            log.info("No signal: %s", tf_result["reason"])
            return

        active_zone = tf_result["active_zone"]
        direction   = tf_result["direction"]
        zk          = self._zk(active_zone)
        zid         = active_zone.zone_id

        # Zone cooldown gate
        if self._zone_blocked(zk, now):
            self.zlog.log_skip(now, "zone_cooldown", tf_result)
            log.info("Zone %s blocked (cooldown)", zk)
            return

        # ── Step 3: confirmations ─────────────────────────────────────────
        conf = check_confirmations_at_last_bar(
            df_15m_w, active_zone, direction, self.conf_cfg
        )
        if not conf.confirmed:
            self.zlog.log_skip(now, f"conf_failed: {conf.signals}", tf_result)
            log.info("Confirmation failed (count=%d signals=%s)", conf.count, conf.signals)
            return

        # ── Step 4: trade setup ───────────────────────────────────────────
        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, self.setup_cfg)
        if not setup.valid:
            self.zlog.log_skip(now, f"setup_invalid: {setup.reason}", tf_result)
            log.info("Setup invalid: %s", setup.reason)
            return

        # ── Enter trade ───────────────────────────────────────────────────
        self._enter_trade(
            now=now,
            direction=direction,
            setup=setup,
            active_zone=active_zone,
            zk=zk,
            zid=zid,
            tf_result=tf_result,
            conf=conf,
        )

    # ── Enter trade ───────────────────────────────────────────────────────

    def _enter_trade(
        self, now, direction, setup, active_zone, zk, zid,
        tf_result, conf,
    ) -> None:
        signal_price = setup.entry          # raw M15 close — always logged for slippage audit
        entry = setup.entry
        sl    = setup.sl
        tp    = setup.tp

        # Apply spread cost in paper mode (MT5 gets real fill via bid/ask)
        if self.mode == "paper" and SPREAD_PTS > 0:
            if direction == "buy":
                entry += SPREAD_PTS
            else:
                entry -= SPREAD_PTS

        # Rebase SL to spread-adjusted entry
        sl_dist = abs(setup.entry - sl)
        sl = (entry - sl_dist) if direction == "buy" else (entry + sl_dist)

        # Re-validate geometry after spread adjustment (vs setup entry)
        if direction == "buy" and (sl >= entry or tp <= entry):
            log.warning("Geometry invalid after spread adjust — skip")
            return
        if direction == "sell" and (sl <= entry or tp >= entry):
            log.warning("Geometry invalid after spread adjust — skip")
            return

        # SL distance filter — skip if SL is too close to entry (noise risk)
        sl_dist_pct = abs(entry - sl) / entry * 100.0
        if sl_dist_pct < MIN_SL_PCT:
            log.info(
                "SL too tight (%.3f%% < %.2f%%) — skip  entry=%.2f sl=%.2f",
                sl_dist_pct, MIN_SL_PCT, entry, sl,
            )
            return

        # For MT5 mode: validate SL/TP against LIVE price — signal_price may be
        # stale if price moved significantly since the confirmation bar closed.
        if self.mode == "mt5":
            from execution.mt5_executor import MT5Executor
            tick = MT5Executor._get_tick(SYMBOL)
            if tick is None:
                log.warning(
                    "Live tick unavailable for %s — cannot validate setup, rejecting trade",
                    SYMBOL,
                )
                return
            live_price = tick.bid if direction == "sell" else tick.ask
            if direction == "buy" and (sl >= live_price or tp <= live_price):
                log.warning(
                    "Setup stale vs live price=%.2f — buy sl=%.2f tp=%.2f — skip",
                    live_price, sl, tp,
                )
                return
            if direction == "sell" and (sl <= live_price or tp >= live_price):
                log.warning(
                    "Setup stale vs live price=%.2f — sell sl=%.2f tp=%.2f — skip",
                    live_price, sl, tp,
                )
                return
            # Reject if live price has already travelled more than 50% of the
            # reward distance — setup is stale and trade would enter mid-move.
            reward = abs(tp - entry)
            price_drift = abs(live_price - entry)
            if price_drift > 0.5 * reward:
                log.warning(
                    "Setup stale — live price=%.2f drifted %.1f pts (>50%% of reward=%.1f) "
                    "from signal entry=%.2f — skip",
                    live_price, price_drift, reward, entry,
                )
                return

        lots = FIXED_LOTS

        rr = abs(tp - entry) / abs(entry - sl)

        # Place order
        ticket = self.broker.place_order(
            symbol    = SYMBOL,
            direction = direction,
            volume    = lots,
            sl        = round(sl, 2),
            tp        = round(tp, 2),
            comment   = f"zz_{zid[:12]}",
        )

        if ticket is None:
            log.error("Order placement failed")
            return

        # In MT5 mode: read the actual fill price from deal history.
        # fill_price is left None (blank in CSV) if the deal isn't readable yet,
        # so it's distinguishable from a true zero-slippage fill.
        if self.mode == "mt5":
            real_fill = self.broker.get_entry_deal_price(ticket)
            if real_fill is None:
                fill_price = None
                log.warning(
                    "Could not read entry deal for ticket=%d — fill_price will be blank in log",
                    ticket,
                )
            else:
                fill_price = real_fill
                # Adverse slippage: positive = worse fill than signal, for both directions.
                slippage = (fill_price - signal_price) if direction == "buy" else (signal_price - fill_price)
                log.info(
                    "Fill: ticket=%d signal=%.2f fill=%.2f adverse_slippage=%+.2f pts (%s)",
                    ticket, signal_price, fill_price, slippage, direction,
                )
        else:
            fill_price = entry  # paper mode: fill = spread-adjusted entry

        prior = self._prior_bucket(zid)

        record = {
            "timestamp":          now.isoformat(),
            "symbol":             SYMBOL,
            "mode":               self.mode,
            "zone_id":            zid,
            "zone_bottom":        active_zone.bottom,
            "zone_top":           active_zone.top,
            "zone_kind":          active_zone.kind,
            "zone_strength":      round(active_zone.strength, 2),
            "zone_fresh":         active_zone.fresh,
            "h4_bias":            tf_result["h4_bias"],
            "direction":          direction,
            "signals_fired":      "|".join(conf.signals),
            "confirmation_count": conf.count,
            "choch_fired":        conf.choch,
            "entry_mode":         setup.entry_mode,
            "tp_mode":            setup.tp_mode,
            "entry":              round(entry, 2),
            "sl":                 round(sl, 2),
            "tp":                 round(tp, 2),
            "rr":                 round(rr, 2),
            "lots":               lots,
            "ticket":             ticket,
            "signal_price":       round(signal_price, 2),
            "fill_price":         round(fill_price, 2) if fill_price is not None else None,
            "outcome":            None,
            "close_price":        None,
            "pnl":                None,
            "prior_bucket":       prior,
        }

        self.zlog.log_trade_open(record)

        self.open_positions.append({
            "ticket":    ticket,
            "direction": direction,
            "entry":     entry,
            "sl":        sl,
            "tp":        tp,
            "lots":      lots,
            "zk":        zk,
            "zid":       zid,
            "zone":      active_zone,
            "open_time": now,
            "record":    record,
        })

    # ── Check if open position has closed ─────────────────────────────────

    def _check_open_position(self, pos: dict, bar_h: float, bar_l: float, now: datetime) -> None:
        if self.mode == "mt5":
            self._check_position_mt5(pos, now)
            return

        # Paper mode: check TP/SL against current bar
        outcome    = None
        close_price = None
        direction   = pos["direction"]

        if direction == "buy":
            if bar_h >= pos["tp"]:
                outcome     = 1
                close_price = pos["tp"]
            elif bar_l <= pos["sl"]:
                outcome     = -1
                close_price = pos["sl"]
        else:
            if bar_l <= pos["tp"]:
                outcome     = 1
                close_price = pos["tp"]
            elif bar_h >= pos["sl"]:
                outcome     = -1
                close_price = pos["sl"]

        if outcome is not None:
            pnl = (
                (close_price - pos["entry"]) * pos["lots"] * self.contract_size
                if direction == "buy"
                else (pos["entry"] - close_price) * pos["lots"] * self.contract_size
            )
            self._on_position_close(pos, outcome, close_price, pnl, now)

    def _check_position_mt5(self, pos: dict, now: datetime) -> None:
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(ticket=pos["ticket"])
            if positions:
                return  # still open
            # Closed — get deal info
            deal = self.broker.get_closed_deal_info(pos["ticket"])
            close_price = float(deal.get("exit_price", pos["entry"]))
            pnl         = float(deal.get("pnl", 0.0))
            outcome = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
            self._on_position_close(pos, outcome, close_price, pnl, now)
        except Exception as exc:
            log.warning("MT5 position check failed: %s", exc)

    def _on_position_close(
        self, pos: dict, outcome: int, close_price: float, pnl: float, now: datetime
    ) -> None:
        zk  = pos["zk"]
        zid = pos["zid"]

        # Update zone history
        self.zone_outcome_history.setdefault(zid, []).append(outcome)

        # Update cooldown
        if outcome == 1:
            self.won_zones.add(zk)
            earliest = now + timedelta(hours=COOLDOWN_WIN_FLOOR_H)
            self.zone_reentry[zk] = {
                "phase":    "exit",
                "bottom":   pos["zone"].bottom,
                "top":      pos["zone"].top,
                "earliest": earliest,
            }
            log.info("WIN — zone %s enters leave-and-return state", zk)
        elif outcome == -1:
            until = now + timedelta(hours=COOLDOWN_LOSS_H)
            self.zone_cooldown[zk] = until
            log.info("LOSS — zone %s blocked until %s", zk, until.strftime("%H:%M UTC"))

        # Update trade log record
        record = {**pos["record"]}
        record["outcome"]     = outcome
        record["close_price"] = round(close_price, 2)
        record["pnl"]         = round(pnl, 2)

        self.zlog.log_trade_close(record)
        self.open_positions.remove(pos)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Starting Z&Z live bot — Ctrl-C to stop")
        try:
            while True:
                try:
                    self.run_once()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error("Unhandled error in run_once: %s", exc, exc_info=True)

                wait = _seconds_to_next_bar(tf_seconds=900)
                log.info("Sleeping %.0fs until next 15M close", wait)
                time.sleep(wait)
        except KeyboardInterrupt:
            log.info("Stopped by user")
        finally:
            try:
                self.db.connection.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Timer helper — align to 15M bar closes
# ---------------------------------------------------------------------------

def _seconds_to_next_bar(tf_seconds: int = 900) -> int:
    """Return seconds until the next 15M candle close, plus a 3s buffer."""
    now     = datetime.now(timezone.utc)
    elapsed = (now.minute * 60 + now.second) % tf_seconds
    return (tf_seconds - elapsed) + 3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z&Z strategy live bot — DEMO only"
    )
    parser.add_argument(
        "--mode", default="paper", choices=["paper", "mt5"],
        help="paper = simulate locally (default); mt5 = send orders to MT5 demo",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Evaluate one bar then exit (for testing)",
    )
    parser.add_argument("--login",    type=int, help="MT5 account number (or MT5_LOGIN in .env)")
    parser.add_argument("--password",           help="MT5 password (or MT5_PASSWORD in .env)")
    parser.add_argument("--server",             help="MT5 server (or MT5_SERVER in .env)")
    args = parser.parse_args()

    # ── Set up broker ─────────────────────────────────────────────────────
    if args.mode == "mt5":
        login    = args.login    or int(os.environ.get("MT5_LOGIN",    0))
        password = args.password or os.environ.get("MT5_PASSWORD", "")
        server   = args.server   or os.environ.get("MT5_SERVER",   "")
        if not all([login, password, server]):
            parser.error(
                "--mode mt5 requires --login/--password/--server "
                "or MT5_LOGIN/MT5_PASSWORD/MT5_SERVER in .env"
            )
        from execution.mt5_connector import MT5Connector
        from execution.mt5_executor  import MT5Executor
        connector = MT5Connector(login=login, password=password, server=server)
        if not connector.connect():
            raise SystemExit("Failed to connect to MT5 terminal")
        broker = MT5Executor(connector)
        log.info("MT5 connected | login=%d server=%s", login, server)
    else:
        from execution.paper_trader import PaperTrader
        broker = PaperTrader(starting_equity=10_000.0)
        broker.connect()
        log.info("Paper trader initialised (equity=$10,000)")

    # ── Launch bot ────────────────────────────────────────────────────────
    bot = ZZLiveBot(broker=broker, mode=args.mode)

    if args.once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()
