"""
utils/db_writer.py — SQLAlchemy write interface for the RL/ML PostgreSQL database.

Single source of truth for all persistent writes: market_context, ml_performance,
rl_shadow, training_data, retrain_log.  No CSV or flat-file logging anywhere.

Column names match the RL/ML schema exactly — do not rename without updating the DB.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

logger = logging.getLogger("db_writer")

# ── Engine singleton (RL/ML database) ─────────────────────────────────────────

_engine: Optional[Engine] = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        import psycopg2
        host     = os.getenv("RL_DB_HOST",     os.getenv("DB_HOST",     "localhost"))
        port     = int(os.getenv("RL_DB_PORT", os.getenv("DB_PORT",     "5432")))
        database = os.getenv("RL_DB_NAME",     "RL/ML")
        user     = os.getenv("RL_DB_USER",     os.getenv("DB_USER",     "postgres"))
        password = os.getenv("RL_DB_PASSWORD", os.getenv("DB_PASSWORD", ""))

        # Use a creator callable so the DB name with "/" is never URL-encoded
        def _creator():
            return psycopg2.connect(
                host=host, port=port, dbname=database,
                user=user, password=password,
            )

        _engine = create_engine(
            "postgresql+psycopg2://",
            creator=_creator,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=2,
        )
        logger.info("RL/ML engine ready → %s:%s/%s", host, port, database)
    return _engine


# ── Helpers ────────────────────────────────────────────────────────────────────

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clean(val: Any) -> Any:
    """Return None instead of NaN/inf; preserve booleans (do not convert to float)."""
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val          # preserve True/False for PostgreSQL BOOLEAN columns
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return val


def _row(d: dict) -> dict:
    """Clean every value in a dict for safe DB insertion."""
    return {k: _clean(v) for k, v in d.items()}


# ── market_context ─────────────────────────────────────────────────────────────
# Columns: id, timestamp, symbol, timeframe, open, high, low, close, volume,
#          htf_bias, market_structure, session, atr_current, atr_average, atr_ratio,
#          volume_current, volume_average, volume_ratio, candle_body_ratio,
#          momentum_score, mins_to_news, news_impact

def write_market_context(rows: List[Dict]) -> None:
    """
    Upsert latest bar for each (timestamp, symbol, timeframe) into market_context.
    Each dict must have at minimum: timestamp, symbol, timeframe, open, high, low, close.
    """
    if not rows:
        return
    sql = text("""
        INSERT INTO market_context
            (timestamp, symbol, timeframe, open, high, low, close, volume,
             htf_bias, market_structure, session,
             atr_current, atr_average, atr_ratio,
             volume_current, volume_average, volume_ratio,
             candle_body_ratio, momentum_score, mins_to_news, news_impact)
        VALUES
            (:timestamp, :symbol, :timeframe, :open, :high, :low, :close, :volume,
             :htf_bias, :market_structure, :session,
             :atr_current, :atr_average, :atr_ratio,
             :volume_current, :volume_average, :volume_ratio,
             :candle_body_ratio, :momentum_score, :mins_to_news, :news_impact)
    """)
    cleaned = [_row({
        "timestamp":       r.get("timestamp"),
        "symbol":          r.get("symbol"),
        "timeframe":       r.get("timeframe"),
        "open":            r.get("open"),
        "high":            r.get("high"),
        "low":             r.get("low"),
        "close":           r.get("close"),
        "volume":          r.get("volume"),
        "htf_bias":        r.get("htf_bias"),
        "market_structure": r.get("market_structure"),
        "session":         r.get("session"),
        "atr_current":     r.get("atr_current"),
        "atr_average":     r.get("atr_average"),
        "atr_ratio":       r.get("atr_ratio"),
        "volume_current":  r.get("volume_current"),
        "volume_average":  r.get("volume_average"),
        "volume_ratio":    r.get("volume_ratio"),
        "candle_body_ratio": r.get("candle_body_ratio"),
        "momentum_score":  r.get("momentum_score"),
        "mins_to_news":    r.get("mins_to_news"),
        "news_impact":     r.get("news_impact"),
    }) for r in rows]
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, cleaned)
    except Exception as exc:
        logger.error("write_market_context failed: %s", exc)


# ── ml_performance ─────────────────────────────────────────────────────────────
# Columns: id, timestamp, symbol, signal, confidence, triggered_rule, triggered_tf,
#          entry_price, sl_price, tp_price, sl_distance_pips, tp_distance_pips,
#          rr_ratio, trade_duration, max_profit_reached, max_drawdown_hit,
#          is_stalling, exit_price, exit_reason, actual_pnl_pips, actual_pnl_usd,
#          outcome, closed_at

def write_ml_signal(
    timestamp:        datetime,
    symbol:           str,
    signal:           int,            # 1=buy, -1=sell, 0=no trade
    confidence:       float,
    triggered_rule:   str,
    triggered_tf:     str,
    sl_price:         Optional[float],
    tp_price:         Optional[float],
    sl_distance_pips: Optional[float],
    tp_distance_pips: Optional[float],
    rr_ratio:         Optional[float],
) -> Optional[int]:
    """INSERT a bar evaluation row. Returns the DB id."""
    sql = text("""
        INSERT INTO ml_performance
            (timestamp, symbol, signal, confidence, triggered_rule, triggered_tf,
             sl_price, tp_price, sl_distance_pips, tp_distance_pips, rr_ratio)
        VALUES
            (:timestamp, :symbol, :signal, :confidence, :triggered_rule, :triggered_tf,
             :sl_price, :tp_price, :sl_distance_pips, :tp_distance_pips, :rr_ratio)
        RETURNING id
    """)
    data = _row({
        "timestamp": timestamp, "symbol": symbol,
        "signal": signal, "confidence": confidence,
        "triggered_rule": triggered_rule, "triggered_tf": triggered_tf,
        "sl_price": sl_price, "tp_price": tp_price,
        "sl_distance_pips": sl_distance_pips, "tp_distance_pips": tp_distance_pips,
        "rr_ratio": rr_ratio,
    })
    # timestamp and symbol must never be cleaned to None
    data["timestamp"] = timestamp
    data["symbol"]    = symbol
    try:
        with get_engine().begin() as conn:
            return conn.execute(sql, data).scalar()
    except Exception as exc:
        logger.error("write_ml_signal failed: %s", exc)
        return None


def update_ml_entry(
    row_id:      int,
    entry_price: float,
) -> None:
    """Update ml_performance row with entry price once the order is filled."""
    sql = text("UPDATE ml_performance SET entry_price = :entry_price WHERE id = :row_id")
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, {"entry_price": _clean(entry_price), "row_id": row_id})
    except Exception as exc:
        logger.error("update_ml_entry failed (id=%s): %s", row_id, exc)


def update_ml_exit(
    row_id:           int,
    exit_price:       float,
    exit_reason:      str,
    actual_pnl_usd:   float,
    actual_pnl_pips:  float,
    outcome:          str,
    closed_at:        datetime,
    trade_duration:   Optional[int]   = None,
    max_profit_reached: Optional[float] = None,
    max_drawdown_hit:   Optional[float] = None,
    is_stalling:      bool            = False,
) -> None:
    """UPDATE ml_performance row with full exit data."""
    sql = text("""
        UPDATE ml_performance SET
            exit_price         = :exit_price,
            exit_reason        = :exit_reason,
            actual_pnl_usd     = :actual_pnl_usd,
            actual_pnl_pips    = :actual_pnl_pips,
            outcome            = :outcome,
            closed_at          = :closed_at,
            trade_duration     = :trade_duration,
            max_profit_reached = :max_profit_reached,
            max_drawdown_hit   = :max_drawdown_hit,
            is_stalling        = :is_stalling
        WHERE id = :row_id
    """)
    data = _row({
        "exit_price": exit_price, "exit_reason": exit_reason,
        "actual_pnl_usd": actual_pnl_usd, "actual_pnl_pips": actual_pnl_pips,
        "outcome": outcome,
        "trade_duration": trade_duration,
        "max_profit_reached": max_profit_reached,
        "max_drawdown_hit": max_drawdown_hit,
        "is_stalling": is_stalling,
    })
    data["closed_at"] = closed_at
    data["row_id"]    = row_id
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, data)
    except Exception as exc:
        logger.error("update_ml_exit failed (id=%s): %s", row_id, exc)


# ── rl_shadow ──────────────────────────────────────────────────────────────────
# Columns: id, timestamp, symbol,
#   ml_signal, ml_confidence, ml_entry_price, ml_sl_price, ml_tp_price,
#   ml_rr_ratio, ml_triggered_rule, ml_triggered_tf,
#   htf_bias, market_structure, session, momentum_score, atr_ratio, volume_ratio,
#   candle_body_ratio, in_trade, trade_duration, unrealised_pnl, is_stalling,
#   better_setup_available, better_setup_direction, better_setup_quality,
#   rl_decision, rl_confidence, rl_reason, rl_suggested_entry, rl_suggested_sl,
#   rl_suggested_tp, rl_recommended_rotation,
#   price_1h_later, price_4h_later, price_24h_later,
#   max_favourable, max_adverse, ml_actual_pnl, rl_hypothetical_pnl,
#   pnl_difference, rl_was_correct

def write_rl_decision(
    timestamp:               datetime,
    symbol:                  str,
    ml_signal:               int,
    ml_confidence:           float,
    ml_entry_price:          Optional[float],
    ml_sl_price:             Optional[float],
    ml_tp_price:             Optional[float],
    ml_rr_ratio:             Optional[float],
    ml_triggered_rule:       str,
    ml_triggered_tf:         str,
    htf_bias:                Optional[int],       # -1 / 0 / 1
    market_structure:        Optional[str],        # "bullish"/"bearish"/"neutral"
    session:                 str,
    momentum_score:          Optional[float],
    atr_ratio:               Optional[float],
    volume_ratio:            Optional[float],
    candle_body_ratio:       Optional[float],
    in_trade:                bool,
    trade_duration:          int,
    unrealised_pnl:          float,
    is_stalling:             bool,
    better_setup_available:  bool,
    better_setup_direction:  Optional[int],
    better_setup_quality:    Optional[float],
    rl_decision:             int,                 # 0=agree, 1=skip, 2=rotate
    rl_confidence:           Optional[float],
    rl_reason:               str,
    rl_suggested_entry:      Optional[float],
    rl_suggested_sl:         Optional[float],
    rl_suggested_tp:         Optional[float],
    rl_recommended_rotation: bool,
) -> Optional[int]:
    """INSERT an RL shadow decision row. Returns DB id for later outcome update."""
    sql = text("""
        INSERT INTO rl_shadow
            (timestamp, symbol,
             ml_signal, ml_confidence, ml_entry_price, ml_sl_price, ml_tp_price,
             ml_rr_ratio, ml_triggered_rule, ml_triggered_tf,
             htf_bias, market_structure, session, momentum_score, atr_ratio,
             volume_ratio, candle_body_ratio,
             in_trade, trade_duration, unrealised_pnl, is_stalling,
             better_setup_available, better_setup_direction, better_setup_quality,
             rl_decision, rl_confidence, rl_reason,
             rl_suggested_entry, rl_suggested_sl, rl_suggested_tp,
             rl_recommended_rotation)
        VALUES
            (:timestamp, :symbol,
             :ml_signal, :ml_confidence, :ml_entry_price, :ml_sl_price, :ml_tp_price,
             :ml_rr_ratio, :ml_triggered_rule, :ml_triggered_tf,
             :htf_bias, :market_structure, :session, :momentum_score, :atr_ratio,
             :volume_ratio, :candle_body_ratio,
             :in_trade, :trade_duration, :unrealised_pnl, :is_stalling,
             :better_setup_available, :better_setup_direction, :better_setup_quality,
             :rl_decision, :rl_confidence, :rl_reason,
             :rl_suggested_entry, :rl_suggested_sl, :rl_suggested_tp,
             :rl_recommended_rotation)
        RETURNING id
    """)
    data = _row({
        "ml_signal": ml_signal, "ml_confidence": ml_confidence,
        "ml_entry_price": ml_entry_price, "ml_sl_price": ml_sl_price,
        "ml_tp_price": ml_tp_price, "ml_rr_ratio": ml_rr_ratio,
        "ml_triggered_rule": ml_triggered_rule, "ml_triggered_tf": ml_triggered_tf,
        "htf_bias": htf_bias, "market_structure": market_structure, "session": session,
        "momentum_score": momentum_score, "atr_ratio": atr_ratio,
        "volume_ratio": volume_ratio, "candle_body_ratio": candle_body_ratio,
        "in_trade": in_trade, "trade_duration": trade_duration,
        "unrealised_pnl": unrealised_pnl, "is_stalling": is_stalling,
        "better_setup_available": better_setup_available,
        "better_setup_direction": better_setup_direction,
        "better_setup_quality": better_setup_quality,
        "rl_decision": rl_decision, "rl_confidence": rl_confidence,
        "rl_reason": rl_reason,
        "rl_suggested_entry": rl_suggested_entry,
        "rl_suggested_sl": rl_suggested_sl, "rl_suggested_tp": rl_suggested_tp,
        "rl_recommended_rotation": rl_recommended_rotation,
    })
    data["timestamp"] = timestamp
    data["symbol"]    = symbol
    try:
        with get_engine().begin() as conn:
            return conn.execute(sql, data).scalar()
    except Exception as exc:
        logger.error("write_rl_decision failed: %s", exc)
        return None


def update_rl_outcome(
    row_id:              int,
    price_1h_later:      Optional[float],
    price_4h_later:      Optional[float],
    price_24h_later:     Optional[float],
    max_favourable:      Optional[float],
    max_adverse:         Optional[float],
    ml_actual_pnl:       Optional[float],
    rl_hypothetical_pnl: Optional[float],
    pnl_difference:      Optional[float],
    rl_was_correct:      Optional[bool],
) -> None:
    """UPDATE an rl_shadow row with resolved market outcome data."""
    sql = text("""
        UPDATE rl_shadow SET
            price_1h_later      = :price_1h_later,
            price_4h_later      = :price_4h_later,
            price_24h_later     = :price_24h_later,
            max_favourable      = :max_favourable,
            max_adverse         = :max_adverse,
            ml_actual_pnl       = :ml_actual_pnl,
            rl_hypothetical_pnl = :rl_hypothetical_pnl,
            pnl_difference      = :pnl_difference,
            rl_was_correct      = :rl_was_correct
        WHERE id = :row_id
    """)
    data = _row({
        "price_1h_later": price_1h_later,
        "price_4h_later": price_4h_later,
        "price_24h_later": price_24h_later,
        "max_favourable": max_favourable,
        "max_adverse": max_adverse,
        "ml_actual_pnl": ml_actual_pnl,
        "rl_hypothetical_pnl": rl_hypothetical_pnl,
        "pnl_difference": pnl_difference,
        "rl_was_correct": rl_was_correct,
    })
    data["row_id"] = row_id
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, data)
    except Exception as exc:
        logger.error("update_rl_outcome failed (id=%s): %s", row_id, exc)


# ── retrain_log ────────────────────────────────────────────────────────────────
# Columns: id, timestamp, model_version, training_rows, timesteps,
#          ml_win_rate, rl_accuracy, pnl_improvement

def write_retrain_log(
    model_version:  str,
    training_rows:  int,
    timesteps:      int,
    ml_win_rate:    Optional[float] = None,
    rl_accuracy:    Optional[float] = None,
    pnl_improvement: Optional[float] = None,
) -> None:
    """INSERT a model retrain event into retrain_log."""
    sql = text("""
        INSERT INTO retrain_log
            (timestamp, model_version, training_rows, timesteps,
             ml_win_rate, rl_accuracy, pnl_improvement)
        VALUES
            (:timestamp, :model_version, :training_rows, :timesteps,
             :ml_win_rate, :rl_accuracy, :pnl_improvement)
    """)
    data = _row({
        "model_version": model_version, "training_rows": training_rows,
        "timesteps": timesteps, "ml_win_rate": ml_win_rate,
        "rl_accuracy": rl_accuracy, "pnl_improvement": pnl_improvement,
    })
    data["timestamp"] = utcnow()
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, data)
    except Exception as exc:
        logger.error("write_retrain_log failed: %s", exc)
