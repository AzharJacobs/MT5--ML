"""
trade_logger.py — Persistent trade and signal logging for the live trader.

Writes two CSV files per trading day under logs/:
  logs/trades_YYYY-MM-DD.csv   One row per completed trade (entry + exit context)
  logs/signals_YYYY-MM-DD.csv  One row per bar evaluation (every outcome, including no-trades)

Also appends all entries to logs/live_trader.log in human-readable form.

Usage in live_trader.py:
    from utils.trade_logger import TradeLogger
    tlog = TradeLogger()
    tlog.log_signal(bar_time, sig, feat_row)
    tlog.log_entry(ticket, bar_time, ...)
    tlog.log_exit(ticket, close_time, exit_price, pnl, close_reason, balance_after)
"""

import os
import csv
import logging
from datetime import datetime, date
from typing import Optional

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# CSV columns — order matters for readability when opening in Excel/pandas
TRADE_FIELDS = [
    "ticket", "symbol", "direction", "grade",
    "open_time", "close_time",
    "entry_price", "exit_price",
    "sl", "tp", "rr",
    "lots", "pnl", "outcome", "close_reason",
    "prob", "zone_quality", "htf_4h_bias",
    "in_session", "session_id",
    "rule_good_session", "rule_buy_score", "rule_sell_score",
    "balance_before", "balance_after",
]

SIGNAL_FIELDS = [
    "bar_time", "signal", "reason",
    "prob", "grade", "sl", "tp", "rr",
    "zone_quality", "htf_4h_bias", "in_session", "session_id",
    "in_demand", "in_supply",
    "rule_good_session", "rule_buy_score", "rule_sell_score",
]


def _safe_float(val, default=None, ndigits: int = 4):
    try:
        v = float(val)
        return round(v, ndigits) if v == v else default  # nan check
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


class TradeLogger:

    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._open_trades: dict = {}   # ticket -> entry row dict

        # File handler for human-readable log (appends across restarts)
        self._flog = logging.getLogger("trade_file_log")
        if not self._flog.handlers:
            fh = logging.FileHandler(
                os.path.join(LOGS_DIR, "live_trader.log"), encoding="utf-8"
            )
            fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self._flog.addHandler(fh)
            self._flog.setLevel(logging.INFO)
            self._flog.propagate = False

    # ── Internal helpers ────────────────────────────────────────────────────

    def _trade_csv(self) -> str:
        return os.path.join(LOGS_DIR, f"trades_{date.today()}.csv")

    def _signal_csv(self) -> str:
        return os.path.join(LOGS_DIR, f"signals_{date.today()}.csv")

    def _append_csv(self, path: str, fields: list, row: dict):
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({k: (row.get(k, "") if row.get(k) is not None else "") for k in fields})

    def _extract_rule_cols(self, row) -> dict:
        """Pull the key diagnostic columns from a feature row (pd.Series or dict)."""
        if row is None:
            return {}
        return {
            "zone_quality":       _safe_float(row.get("active_zone_quality", 0), 0, 3),
            "htf_4h_bias":        _safe_float(row.get("htf_4h_bias", 0), 0, 1),
            "in_session":         _safe_int(row.get("in_session", 0)),
            "session_id":         _safe_int(row.get("session_id", 0)),
            "in_demand":          _safe_int(row.get("in_demand_zone", 0)),
            "in_supply":          _safe_int(row.get("in_supply_zone", 0)),
            "rule_good_session":  _safe_int(row.get("rule_good_session", 0)),
            "rule_buy_score":     _safe_float(row.get("rule_buy_score", 0), 0, 1),
            "rule_sell_score":    _safe_float(row.get("rule_sell_score", 0), 0, 1),
        }

    # ── Public API ──────────────────────────────────────────────────────────

    def log_signal(self, bar_time, sig: dict, feat_row=None):
        """
        Log every bar evaluation — including no-trade bars.
        Call this after evaluate_bar() regardless of outcome.
        feat_row: the pd.Series feature row (feat_df.iloc[-1]) for the bar.
        """
        signal_val = sig.get("signal", 0)
        direction_str = {1: "BUY", -1: "SELL", 0: "NONE"}.get(signal_val, "NONE")
        data = {
            "bar_time": bar_time,
            "signal":   direction_str,
            "reason":   sig.get("reason", ""),
            "prob":     _safe_float(sig.get("prob", 0.0), 0, 4),
            "grade":    sig.get("grade") or "",
            "sl":       _safe_float(sig.get("sl")),
            "tp":       _safe_float(sig.get("tp")),
            "rr":       _safe_float(sig.get("rr")),
        }
        data.update(self._extract_rule_cols(feat_row))
        self._append_csv(self._signal_csv(), SIGNAL_FIELDS, data)

    def log_entry(
        self,
        ticket,
        bar_time,
        symbol: str,
        direction: str,
        grade: str,
        entry_price: float,
        sl: float,
        tp: float,
        rr: float,
        lots: float,
        prob: float,
        balance_before: float,
        feat_row=None,
    ):
        """
        Log a trade entry. Stored in memory until log_exit() completes the row.
        feat_row: the pd.Series feature row so we capture what the model saw.
        """
        entry = {
            "ticket":        ticket,
            "symbol":        symbol,
            "direction":     direction,
            "grade":         grade,
            "open_time":     bar_time,
            "entry_price":   round(float(entry_price), 5),
            "sl":            round(float(sl), 5),
            "tp":            round(float(tp), 5),
            "rr":            round(float(rr), 2),
            "lots":          round(float(lots), 2),
            "prob":          round(float(prob), 4),
            "balance_before": round(float(balance_before), 2),
        }
        entry.update(self._extract_rule_cols(feat_row))

        self._open_trades[ticket] = entry

        self._flog.info(
            "ENTRY  ticket=%-8s %-4s %-8s Grade=%s  entry=%.5f  SL=%.5f  TP=%.5f  "
            "RR=%.2f  prob=%.3f  lots=%.2f  zone_q=%.2f  in_sess=%s  good_sess=%s",
            ticket, direction.upper(), symbol, grade,
            entry_price, sl, tp, rr, prob, lots,
            entry.get("zone_quality", 0),
            entry.get("in_session", "?"),
            entry.get("rule_good_session", "?"),
        )

    def log_exit(
        self,
        ticket,
        close_time,
        exit_price: float,
        pnl: float,
        close_reason: str,          # "SL" | "TP" | "manual" | "unknown"
        balance_after: float,
    ):
        """
        Complete a trade record with exit data and write to the trades CSV.
        Must be called after log_entry() for the same ticket.
        """
        entry = self._open_trades.pop(ticket, {})
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")

        row = {
            **entry,
            "close_time":   close_time,
            "exit_price":   round(float(exit_price), 5),
            "pnl":          round(float(pnl), 2),
            "outcome":      outcome,
            "close_reason": close_reason,
            "balance_after": round(float(balance_after), 2),
        }
        self._append_csv(self._trade_csv(), TRADE_FIELDS, row)

        self._flog.info(
            "EXIT   ticket=%-8s %-4s  exit=%.5f  pnl=%+.2f  %-9s via %-7s  balance=%.2f",
            ticket,
            entry.get("direction", "?").upper(),
            exit_price, pnl,
            outcome.upper(), close_reason,
            balance_after,
        )

    def log_session_start(self, mode: str, timeframe: str, symbol: str, threshold: float, balance: float):
        self._flog.info(
            "═══ SESSION START  mode=%s  tf=%s  symbol=%s  threshold=%.3f  balance=%.2f ═══",
            mode, timeframe, symbol, threshold, balance,
        )

    def log_session_end(self):
        self._flog.info("═══ SESSION END ═══")
