"""
live_bot_zz.py — Zone-to-Zone (Z&Z) strategy live trading bot.

Instrument : USTECH (Exness DEMO only — hard-checked at startup)
Timeframes : H4 (bias + zone marking)  +  M15 (entries + confirmation)

Strategy pipeline — exact match to the validated backtest baseline:
  Step 1  detect_zones(H4)
  Step 2  analyse_timeframes
  Step 3  check_confirmations_at_last_bar
  Step 4  setup_from_analysis

ALL symbol params and strategy config come from
trading/strategies/zz/ustec/config.yaml via strategy.py.
Neither this file nor the backtest hard-codes those values directly —
that is how live/backtest divergence is prevented.

Usage:
  python trading/live/live_bot_zz.py --mode paper          # paper trade (default)
  python trading/live/live_bot_zz.py --mode mt5            # MT5 demo only
  python trading/live/live_bot_zz.py --mode mt5 --once     # evaluate one bar then exit
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]   # trading/live → trading → MT5--ML
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

# ── All symbol constants + config factory from the single source of truth ─────
from trading.strategies.zz.ustec.strategy import (
    SYMBOL, DB_NAME, TABLE, CONTRACT_SIZE,
    SPREAD_PTS, FIXED_LOTS, RISK_PCT, MIN_RR, MIN_SL_PCT,
    MAX_POSITIONS, H4_WINDOW, M15_WINDOW,
    COOLDOWN_LOSS_H, COOLDOWN_WIN_FLOOR_H, TAP_TOL,
    make_configs,
)

# ── Z&Z strategy modules ───────────────────────────────────────────────────────
from trading.strategies.zz.core.timeframe_structure import analyse_timeframes
from trading.strategies.zz.core.confirmations import check_confirmations_at_last_bar
from trading.strategies.zz.core.trade_setup import setup_from_analysis

# ── Infrastructure ─────────────────────────────────────────────────────────────
from trading.shared.data_loader import get_connection
from trading.shared.broker_interface import BrokerInterface

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("live_zz")

DEMO_KEYWORDS = ("trial", "demo", "test")


# ---------------------------------------------------------------------------
# Logger — CSV + console, one file per day
# ---------------------------------------------------------------------------

LOG_DIR = _PROJECT_ROOT / "logs"
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

_MT5_TF_MAP = None


def _load_ohlcv_mt5(timeframe: str, n_bars: int) -> pd.DataFrame:
    """Fetch closed bars directly from the connected MT5 terminal."""
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

    # position=1 skips bar 0 (currently forming) — only closed bars
    rates = mt5.copy_rates_from_pos(SYMBOL, tf_const, 1, n_bars)
    if rates is None or len(rates) == 0:
        log.error("_load_ohlcv_mt5: no data for %s %s — %s", SYMBOL, timeframe, mt5.last_error())
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    # Exness timestamps are GMT+3; convert to naive UTC
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
    """Fetch bars from PostgreSQL (paper mode)."""
    q = f"SELECT * FROM {TABLE} WHERE timeframe = %s ORDER BY timestamp DESC LIMIT %s"
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

        self._demo_guard()

        # Config objects from the single source of truth (config.yaml)
        self.tf_cfg, self.conf_cfg, self.setup_cfg = make_configs()

        # Zone cooldown state (mirrors backtest/engine_zones.py exactly)
        self.zone_cooldown: dict = {}
        self.zone_reentry:  dict = {}
        self.won_zones:     set  = set()
        self.zone_outcome_history: dict = {}

        self.open_positions: list = []

        self.db = get_connection()
        self.db.database = DB_NAME
        self.db.connect()

        self.contract_size = CONTRACT_SIZE
        if mode == "mt5":
            import MetaTrader5 as mt5
            info = mt5.symbol_info(SYMBOL)
            if info is None:
                similar = [s.name for s in (mt5.symbols_get() or [])
                           if SYMBOL[:5] in s.name.upper() or "NAS" in s.name.upper()][:15]
                log.warning(
                    "SYMBOL '%s' not found — check the symbol name. Similar: %s",
                    SYMBOL, similar,
                )
            else:
                cs = float(getattr(info, "trade_contract_size", 0) or 0)
                if cs > 0:
                    self.contract_size = cs
                log.info(
                    "Symbol verified: %s | trade_mode=%d | digits=%d | contract_size=%.2f",
                    SYMBOL, info.trade_mode, info.digits, self.contract_size,
                )

        log.info(
            "ZZLiveBot ready | mode=%s | symbol=%s | min_conf=%d | "
            "leave_and_return=True | loss_cooldown=%.0fh | lots=%.2f | max_pos=%d",
            mode, SYMBOL, self.conf_cfg.min_confirmations,
            COOLDOWN_LOSS_H, FIXED_LOTS, MAX_POSITIONS,
        )

    def _demo_guard(self) -> None:
        server = os.environ.get("MT5_SERVER", "")
        if not any(kw in server.lower() for kw in DEMO_KEYWORDS):
            raise SystemExit(
                f"SAFETY STOP: MT5_SERVER='{server}' does not look like a demo account "
                f"(must contain one of {DEMO_KEYWORDS}). "
                f"This bot will NOT run on a live account."
            )
        log.info("Demo guard PASSED — server='%s'", server)

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
            else:
                if bar_l <= z_top and bar_h >= z_bot:
                    if now >= state["earliest"]:
                        done.append(zk)
                        log.debug("Zone %s: leave-and-return complete — eligible", zk)
        for zk in done:
            del self.zone_reentry[zk]

    def _zone_blocked(self, zk: tuple, now: datetime) -> bool:
        until = self.zone_cooldown.get(zk)
        if until is not None and now < until:
            return True
        if zk in self.zone_reentry:
            return True
        return False

    @staticmethod
    def _zk(zone) -> tuple:
        return (round(zone.bottom, 1), round(zone.top, 1))

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

    def run_once(self) -> None:
        now = datetime.now(timezone.utc)
        log.info("── Bar at %s UTC ──", now.strftime("%Y-%m-%d %H:%M"))

        try:
            if self.mode == "mt5":
                import MetaTrader5 as mt5
                info = mt5.symbol_info(SYMBOL)
                if info is None:
                    similar = [s.name for s in (mt5.symbols_get() or [])
                               if SYMBOL[:5] in s.name.upper() or "NAS" in s.name.upper()][:10]
                    log.warning("symbol_info(%s) returned None — similar: %s", SYMBOL, similar)
                    return
                if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
                    log.info("Market not open for %s (trade_mode=%d) — skipping", SYMBOL, info.trade_mode)
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

        df_15m_w = df_15m.tail(M15_WINDOW).reset_index(drop=True)
        df_4h_w  = df_4h.tail(H4_WINDOW).reset_index(drop=True)

        if self.mode == "paper":
            last_bar_ts = df_15m_w["timestamp"].iloc[-1]
            if last_bar_ts.tzinfo is None:
                last_bar_ts = last_bar_ts.replace(tzinfo=timezone.utc)
            data_age = (now - last_bar_ts).total_seconds() / 60
            if data_age > 30:
                log.warning(
                    "DB data stale — last 15M bar is %.0f min old (%s) — "
                    "check that the MT5 collector is running.",
                    data_age, last_bar_ts.strftime("%H:%M UTC"),
                )
                return

        bar_h = float(df_15m_w["high"].iloc[-1])
        bar_l = float(df_15m_w["low"].iloc[-1])
        self._update_zone_reentry(now, bar_h, bar_l)

        for pos in list(self.open_positions):
            self._check_open_position(pos, bar_h, bar_l, now)

        if len(self.open_positions) >= MAX_POSITIONS:
            log.info("Max positions (%d) reached — skipping entry evaluation", MAX_POSITIONS)
            return

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

        if self._zone_blocked(zk, now):
            self.zlog.log_skip(now, "zone_cooldown", tf_result)
            log.info("Zone %s blocked (cooldown)", zk)
            return

        conf = check_confirmations_at_last_bar(
            df_15m_w, active_zone, direction, self.conf_cfg
        )
        if not conf.confirmed:
            self.zlog.log_skip(now, f"conf_failed: {conf.signals}", tf_result)
            log.info("Confirmation failed (count=%d signals=%s)", conf.count, conf.signals)
            return

        signal_price = float(df_15m_w["close"].iloc[-1])
        setup = setup_from_analysis(tf_result, signal_price, self.setup_cfg)
        if not setup.valid:
            self.zlog.log_skip(now, f"setup_invalid: {setup.reason}", tf_result)
            log.info("Setup invalid: %s", setup.reason)
            return

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

    def _enter_trade(self, now, direction, setup, active_zone, zk, zid, tf_result, conf) -> None:
        signal_price = setup.entry
        entry = setup.entry
        sl    = setup.sl
        tp    = setup.tp

        if self.mode == "paper" and SPREAD_PTS > 0:
            entry = entry + SPREAD_PTS if direction == "buy" else entry - SPREAD_PTS

        sl_dist = abs(setup.entry - sl)
        sl = (entry - sl_dist) if direction == "buy" else (entry + sl_dist)

        if direction == "buy" and (sl >= entry or tp <= entry):
            log.warning("Geometry invalid after spread adjust — skip")
            return
        if direction == "sell" and (sl <= entry or tp >= entry):
            log.warning("Geometry invalid after spread adjust — skip")
            return

        sl_dist_pct = abs(entry - sl) / entry * 100.0
        if sl_dist_pct < MIN_SL_PCT:
            log.info("SL too tight (%.3f%% < %.2f%%) — skip", sl_dist_pct, MIN_SL_PCT)
            return

        if self.mode == "mt5":
            from trading.shared.mt5_executor import MT5Executor
            tick = MT5Executor._get_tick(SYMBOL)
            if tick is None:
                log.warning("Live tick unavailable — rejecting trade")
                return
            live_price = tick.bid if direction == "sell" else tick.ask
            if direction == "buy" and (sl >= live_price or tp <= live_price):
                log.warning("Setup stale vs live price=%.2f — buy sl=%.2f tp=%.2f — skip",
                            live_price, sl, tp)
                return
            if direction == "sell" and (sl <= live_price or tp >= live_price):
                log.warning("Setup stale vs live price=%.2f — sell sl=%.2f tp=%.2f — skip",
                            live_price, sl, tp)
                return
            reward = abs(tp - entry)
            price_drift = abs(live_price - entry)
            if price_drift > 0.5 * reward:
                log.warning(
                    "Setup stale — live price=%.2f drifted %.1f pts (>50%% of reward=%.1f) — skip",
                    live_price, price_drift, reward,
                )
                return

        lots = FIXED_LOTS
        rr   = abs(tp - entry) / abs(entry - sl)

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

        if self.mode == "mt5":
            real_fill = self.broker.get_entry_deal_price(ticket)
            if real_fill is None:
                fill_price = None
                log.warning("Could not read entry deal for ticket=%d", ticket)
            else:
                fill_price = real_fill
                slippage = (fill_price - signal_price) if direction == "buy" else (signal_price - fill_price)
                log.info("Fill: ticket=%d signal=%.2f fill=%.2f adverse_slippage=%+.2f pts (%s)",
                         ticket, signal_price, fill_price, slippage, direction)
        else:
            fill_price = entry

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
            "prior_bucket":       self._prior_bucket(zid),
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

    def _check_open_position(self, pos: dict, bar_h: float, bar_l: float, now: datetime) -> None:
        if self.mode == "mt5":
            self._check_position_mt5(pos, now)
            return

        outcome     = None
        close_price = None
        direction   = pos["direction"]

        if direction == "buy":
            if bar_h >= pos["tp"]:
                outcome = 1;  close_price = pos["tp"]
            elif bar_l <= pos["sl"]:
                outcome = -1; close_price = pos["sl"]
        else:
            if bar_l <= pos["tp"]:
                outcome = 1;  close_price = pos["tp"]
            elif bar_h >= pos["sl"]:
                outcome = -1; close_price = pos["sl"]

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
            if mt5.positions_get(ticket=pos["ticket"]):
                return
            deal = self.broker.get_closed_deal_info(pos["ticket"])
            close_price = float(deal.get("exit_price", pos["entry"]))
            pnl         = float(deal.get("pnl", 0.0))
            outcome = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
            self._on_position_close(pos, outcome, close_price, pnl, now)
        except Exception as exc:
            log.warning("MT5 position check failed: %s", exc)

    def _on_position_close(self, pos: dict, outcome: int, close_price: float, pnl: float, now: datetime) -> None:
        zk  = pos["zk"]
        zid = pos["zid"]

        self.zone_outcome_history.setdefault(zid, []).append(outcome)

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

        record = {**pos["record"]}
        record["outcome"]     = outcome
        record["close_price"] = round(close_price, 2)
        record["pnl"]         = round(pnl, 2)

        self.zlog.log_trade_close(record)
        self.open_positions.remove(pos)

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


def _seconds_to_next_bar(tf_seconds: int = 900) -> int:
    now     = datetime.now(timezone.utc)
    elapsed = (now.minute * 60 + now.second) % tf_seconds
    return (tf_seconds - elapsed) + 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Z&Z strategy live bot — DEMO only")
    parser.add_argument("--mode", default="paper", choices=["paper", "mt5"])
    parser.add_argument("--once", action="store_true",
                        help="Evaluate one bar then exit (for testing)")
    parser.add_argument("--login",    type=int)
    parser.add_argument("--password")
    parser.add_argument("--server")
    args = parser.parse_args()

    if args.mode == "mt5":
        login    = args.login    or int(os.environ.get("MT5_LOGIN",    0))
        password = args.password or os.environ.get("MT5_PASSWORD", "")
        server   = args.server   or os.environ.get("MT5_SERVER",   "")
        if not all([login, password, server]):
            parser.error("--mode mt5 requires --login/--password/--server or .env equivalents")
        from trading.shared.mt5_connector import MT5Connector
        from trading.shared.mt5_executor  import MT5Executor
        connector = MT5Connector(login=login, password=password, server=server)
        if not connector.connect():
            raise SystemExit("Failed to connect to MT5 terminal")
        broker = MT5Executor(connector)
        log.info("MT5 connected | login=%d server=%s", login, server)
    else:
        from trading.shared.paper_trader import PaperTrader
        broker = PaperTrader(starting_equity=10_000.0)
        broker.connect()
        log.info("Paper trader initialised (equity=$10,000)")

    bot = ZZLiveBot(broker=broker, mode=args.mode)

    if args.once:
        bot.run_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()
