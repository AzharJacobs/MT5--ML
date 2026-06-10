"""
trade_logger.py — Persistent signal and trade logging to the RL/ML PostgreSQL database.

Writes every bar evaluation (signal + trade state) to the ml_performance table.
All timestamps are UTC.  No CSV or flat-file output.

Table: ml_performance
  signal          SMALLINT   — 1=buy, -1=sell, 0=no trade
  confidence      FLOAT      — ML win probability
  triggered_rule  TEXT       — reason / rule description
  triggered_tf    VARCHAR    — timeframe that fired the signal
  entry_price     FLOAT      — filled once order is placed
  sl_price        FLOAT      — stop-loss price
  tp_price        FLOAT      — take-profit price
  sl_distance_pips FLOAT     — SL distance in pips
  tp_distance_pips FLOAT     — TP distance in pips
  rr_ratio        FLOAT      — risk/reward ratio
  exit_price      FLOAT      — filled on close
  exit_reason     VARCHAR    — "SL" | "TP" | "manual" | "unknown"
  actual_pnl_usd  FLOAT      — realised PnL in USD
  actual_pnl_pips FLOAT      — realised PnL in pips
  outcome         VARCHAR    — "win" | "loss" | "breakeven"
  closed_at       TIMESTAMPTZ — UTC close time
  trade_duration  INT        — bars held
  is_stalling     BOOL

Usage:
    tlog = TradeLogger(symbol="XAUUSDm", timeframe="15min")
    tlog.log_signal(bar_time, sig, feat_row)       # every bar
    tlog.log_entry(ticket, bar_time, ...)           # on order fill
    tlog.log_exit(ticket, close_time, exit_price, pnl, close_reason, balance_after)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from trading.db.writer import (
    write_ml_signal,
    update_ml_entry,
    update_ml_exit,
    utcnow,
)

logger = logging.getLogger("trade_logger")

# 1 pip for XAU/USD (standard Exness convention: 0.10)
_PIP = 0.10


def _to_pips(price_diff: float) -> Optional[float]:
    try:
        return round(abs(float(price_diff)) / _PIP, 1)
    except (TypeError, ValueError):
        return None


def _safe_float(val, ndigits: int = 5) -> Optional[float]:
    try:
        f = float(val)
        import math
        return round(f, ndigits) if not (math.isnan(f) or math.isinf(f)) else None
    except (TypeError, ValueError):
        return None


def _parse_ts(bar_time) -> datetime:
    """Convert bar_time string/datetime to UTC-aware datetime."""
    if isinstance(bar_time, datetime):
        if bar_time.tzinfo is None:
            return bar_time.replace(tzinfo=timezone.utc)
        return bar_time
    try:
        dt = datetime.fromisoformat(str(bar_time))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return utcnow()


class TradeLogger:

    def __init__(self, symbol: str = "XAUUSDm", timeframe: str = "15min"):
        self.symbol    = symbol
        self.timeframe = timeframe

        # ticket → {"db_id": int, "open_time": datetime, "entry_price": float,
        #            "bars_open": int}
        self._open_trades: dict = {}

        # DB id of the most recently logged signal row (used by log_entry on same bar)
        self._last_signal_id: Optional[int] = None
        self._last_signal_ts: Optional[datetime] = None

        logger.info("TradeLogger ready  symbol=%s  tf=%s", symbol, timeframe)

    # ── Public API ─────────────────────────────────────────────────────────────

    def log_signal(self, bar_time, sig: dict, feat_row=None) -> None:
        """
        Log every bar evaluation — including no-trade bars.
        Call after evaluate_bar() regardless of outcome.
        """
        signal_int = int(sig.get("signal", 0) or 0)
        sl_v  = _safe_float(sig.get("sl"))
        tp_v  = _safe_float(sig.get("tp"))
        close = _safe_float(feat_row.get("close") if feat_row is not None else None)

        sl_pips = _to_pips(close - sl_v) if (close and sl_v and signal_int == 1)  else (
                  _to_pips(sl_v - close) if (close and sl_v and signal_int == -1) else None)
        tp_pips = _to_pips(tp_v - close) if (close and tp_v and signal_int == 1)  else (
                  _to_pips(close - tp_v) if (close and tp_v and signal_int == -1) else None)

        ts = _parse_ts(bar_time)

        db_id = write_ml_signal(
            timestamp        = ts,
            symbol           = self.symbol,
            signal           = signal_int,
            confidence       = _safe_float(sig.get("prob", 0.0)) or 0.0,
            triggered_rule   = str(sig.get("reason", "") or ""),
            triggered_tf     = self.timeframe,
            sl_price         = sl_v,
            tp_price         = tp_v,
            sl_distance_pips = sl_pips,
            tp_distance_pips = tp_pips,
            rr_ratio         = _safe_float(sig.get("rr")),
        )
        self._last_signal_id = db_id
        self._last_signal_ts = ts

        logger.info(
            "SIGNAL  %-5s  signal=%-3s  prob=%.3f  grade=%s  reason=%s",
            self.timeframe,
            {1: "BUY", -1: "SELL", 0: "NONE"}.get(signal_int, "NONE"),
            float(sig.get("prob", 0) or 0),
            sig.get("grade") or "–",
            (sig.get("reason") or "")[:80],
        )

    def log_entry(
        self,
        ticket,
        bar_time,
        symbol:         str,
        direction:      str,
        grade:          str,
        entry_price:    float,
        sl:             float,
        tp:             float,
        rr:             float,
        lots:           float,
        prob:           float,
        balance_before: float,
        feat_row        = None,
    ) -> None:
        """
        Record a trade entry.  Updates the most recent signal row with entry_price
        and stores the DB id for later exit update.
        """
        ts = _parse_ts(bar_time)
        db_id = self._last_signal_id

        # If there's a matching signal row on the same bar, update it
        if db_id is not None:
            update_ml_entry(row_id=db_id, entry_price=float(entry_price))
        else:
            # Edge case: log_entry without prior log_signal (shouldn't happen)
            db_id = write_ml_signal(
                timestamp        = ts,
                symbol           = symbol,
                signal           = 1 if direction == "buy" else -1,
                confidence       = float(prob),
                triggered_rule   = f"entry ticket={ticket}",
                triggered_tf     = self.timeframe,
                sl_price         = _safe_float(sl),
                tp_price         = _safe_float(tp),
                sl_distance_pips = _to_pips(abs(float(entry_price) - float(sl))),
                tp_distance_pips = _to_pips(abs(float(tp) - float(entry_price))),
                rr_ratio         = _safe_float(rr),
            )
            if db_id:
                update_ml_entry(row_id=db_id, entry_price=float(entry_price))

        self._open_trades[ticket] = {
            "db_id":       db_id,
            "open_time":   ts,
            "entry_price": float(entry_price),
            "direction":   1 if direction == "buy" else -1,
            "bars_open":   0,
        }
        self._last_signal_id = None   # consumed

        logger.info(
            "ENTRY   ticket=%-8s  %-4s  entry=%.5f  SL=%.5f  TP=%.5f  "
            "RR=%.2f  prob=%.3f  grade=%s  lots=%.2f",
            ticket, direction.upper(),
            entry_price, sl, tp, rr, prob, grade, lots,
        )

    def log_exit(
        self,
        ticket,
        close_time,
        exit_price:    float,
        pnl:           float,
        close_reason:  str,
        balance_after: float,
    ) -> None:
        """
        Complete a trade record with exit data.
        Must be called after log_entry() for the same ticket.
        """
        trade = self._open_trades.pop(ticket, {})
        if not trade:
            logger.warning("log_exit: no open trade found for ticket=%s", ticket)
            return

        db_id      = trade.get("db_id")
        open_time  = trade.get("open_time")
        entry_px   = trade.get("entry_price", 0.0)
        direction  = trade.get("direction", 1)

        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        pnl_pips = _safe_float(
            (float(exit_price) - float(entry_px)) * direction / _PIP
        )
        closed_at = _parse_ts(close_time)

        trade_duration: Optional[int] = None
        if open_time and closed_at:
            try:
                # Map timeframe label → bar size in minutes
                _TF_MIN = {
                    "1min": 1, "2min": 2, "3min": 3, "4min": 4,
                    "5min": 5, "10min": 10, "15min": 15, "30min": 30,
                    "1H": 60, "4H": 240, "1D": 1440,
                }
                bar_mins = _TF_MIN.get(self.timeframe, 15)
                delta_secs = (closed_at - open_time).total_seconds()
                trade_duration = max(1, int(delta_secs // 60 // bar_mins))
            except Exception:
                pass

        if db_id is not None:
            update_ml_exit(
                row_id           = db_id,
                exit_price       = float(exit_price),
                exit_reason      = close_reason,
                actual_pnl_usd   = float(pnl),
                actual_pnl_pips  = pnl_pips or 0.0,
                outcome          = outcome,
                closed_at        = closed_at,
                trade_duration   = trade_duration,
            )

        logger.info(
            "EXIT    ticket=%-8s  exit=%.5f  pnl=%+.2f  %-9s via %-7s  balance=%.2f",
            ticket, exit_price, pnl, outcome.upper(), close_reason, balance_after,
        )

    def log_session_start(
        self, mode: str, timeframe: str, symbol: str, threshold: float, balance: float
    ) -> None:
        self.symbol    = symbol
        self.timeframe = timeframe
        logger.info(
            "═══ SESSION START  mode=%s  tf=%s  symbol=%s  threshold=%.3f  balance=%.2f ═══",
            mode, timeframe, symbol, threshold, balance,
        )

    def log_session_end(self) -> None:
        logger.info("═══ SESSION END ═══")
