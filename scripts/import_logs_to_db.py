"""
scripts/import_logs_to_db.py — One-shot import of historical log files into RL/ML DB.

Imports:
  logs/signals_*.csv  → ml_performance (signal evaluation rows)
  logs/trades_*.csv   → ml_performance (updates matched signal rows with trade data)

Matching logic:
  Each trade row is matched to a signal row by (open_time == bar_time) AND
  direction is consistent with signal.  Matched rows are updated with entry,
  exit, and PnL data.  Unmatched trades are inserted as standalone rows.

Safe to re-run — uses INSERT ... ON CONFLICT (timestamp, symbol, triggered_rule)
DO NOTHING, so duplicates are silently skipped.

Run:
  python scripts/import_logs_to_db.py
  python scripts/import_logs_to_db.py --dry-run   # preview without writing
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.db.writer import get_engine, _clean
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("import_logs")

LOGS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
PIP        = 0.10
DEFAULT_TF = "15min"
DEFAULT_SYM = "XAUUSDm"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sig_int(s) -> int:
    """Convert signal string BUY/SELL/NONE → 1/-1/0."""
    if pd.isna(s):
        return 0
    v = str(s).strip().upper()
    return 1 if v == "BUY" else (-1 if v == "SELL" else 0)


def _utc(val) -> Optional[datetime]:
    if pd.isna(val) if not isinstance(val, str) else not val:
        return None
    try:
        dt = pd.to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.to_pydatetime()
    except Exception:
        return None


def _f(val, ndigits: int = 5) -> Optional[float]:
    try:
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, ndigits)
    except (TypeError, ValueError):
        return None


def _pips(distance: Optional[float]) -> Optional[float]:
    return round(abs(distance) / PIP, 1) if distance is not None else None


# ── Load CSVs ─────────────────────────────────────────────────────────────────

def load_signals() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(LOGS_DIR, "signals_*.csv")))
    if not files:
        logger.warning("No signals_*.csv files found in %s", LOGS_DIR)
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d signal rows from %d files", len(df), len(files))
    return df


def load_trades() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(LOGS_DIR, "trades_*.csv")))
    if not files:
        logger.warning("No trades_*.csv files found in %s", LOGS_DIR)
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d trade rows from %d files", len(df), len(files))
    return df


# ── Import signals ─────────────────────────────────────────────────────────────

def import_signals(signals: pd.DataFrame, dry_run: bool = False) -> dict:
    """
    INSERT signal evaluation rows into ml_performance.
    Returns {timestamp_str: db_id} map for later trade matching.
    """
    if signals.empty:
        return {}

    sql = text("""
        INSERT INTO ml_performance
            (timestamp, symbol, signal, confidence, triggered_rule, triggered_tf,
             sl_price, tp_price, rr_ratio)
        VALUES
            (:timestamp, :symbol, :signal, :confidence, :triggered_rule, :triggered_tf,
             :sl_price, :tp_price, :rr_ratio)
        ON CONFLICT DO NOTHING
        RETURNING id, timestamp, signal
    """)

    inserted = 0
    skipped  = 0
    id_map   = {}   # "YYYY-MM-DD HH:MM:SS_signal" → db_id

    engine = get_engine()
    rows_to_insert = []

    for _, row in signals.iterrows():
        ts = _utc(row.get("bar_time"))
        if ts is None:
            skipped += 1
            continue

        sig_int = _sig_int(row.get("signal", "NONE"))
        sl_v    = _f(row.get("sl"))
        tp_v    = _f(row.get("tp"))

        rows_to_insert.append({
            "timestamp":      ts,
            "symbol":         DEFAULT_SYM,
            "signal":         sig_int,
            "confidence":     _f(row.get("prob"), 4) or 0.0,
            "triggered_rule": str(row.get("reason", "") or "")[:500],
            "triggered_tf":   DEFAULT_TF,
            "sl_price":       sl_v,
            "tp_price":       tp_v,
            "rr_ratio":       _f(row.get("rr"), 3),
        })

    if dry_run:
        logger.info("[DRY RUN] Would insert %d signal rows into ml_performance", len(rows_to_insert))
        return {}

    with engine.begin() as conn:
        for r in rows_to_insert:
            try:
                result = conn.execute(sql, r)
                row_result = result.fetchone()
                if row_result:
                    db_id = row_result[0]
                    ts_key = f"{r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}_{r['signal']}"
                    id_map[ts_key] = db_id
                    inserted += 1
                else:
                    skipped += 1
            except Exception as exc:
                logger.warning("Signal insert failed (%s): %s", r.get("timestamp"), exc)
                skipped += 1

    logger.info("Signals: %d inserted, %d skipped (duplicates/errors)", inserted, skipped)
    return id_map


# ── Import trades ─────────────────────────────────────────────────────────────

def import_trades(trades: pd.DataFrame, signal_id_map: dict, dry_run: bool = False) -> None:
    """
    For each trade, try to UPDATE the matching signal row.
    If no signal row found, INSERT a new complete row.
    """
    if trades.empty:
        return

    update_sql = text("""
        UPDATE ml_performance SET
            entry_price     = :entry_price,
            exit_price      = :exit_price,
            exit_reason     = :exit_reason,
            actual_pnl_usd  = :actual_pnl_usd,
            actual_pnl_pips = :actual_pnl_pips,
            outcome         = :outcome,
            closed_at       = :closed_at,
            sl_price        = :sl_price,
            tp_price        = :tp_price,
            rr_ratio        = :rr_ratio,
            confidence      = :confidence
        WHERE id = :row_id
    """)

    insert_sql = text("""
        INSERT INTO ml_performance
            (timestamp, symbol, signal, confidence, triggered_rule, triggered_tf,
             sl_price, tp_price, rr_ratio,
             entry_price, exit_price, exit_reason,
             actual_pnl_usd, actual_pnl_pips, outcome, closed_at)
        VALUES
            (:timestamp, :symbol, :signal, :confidence, :triggered_rule, :triggered_tf,
             :sl_price, :tp_price, :rr_ratio,
             :entry_price, :exit_price, :exit_reason,
             :actual_pnl_usd, :actual_pnl_pips, :outcome, :closed_at)
        ON CONFLICT DO NOTHING
    """)

    updated  = 0
    inserted = 0
    skipped  = 0

    engine = get_engine()

    for _, row in trades.iterrows():
        open_ts = _utc(row.get("open_time"))
        if open_ts is None:
            skipped += 1
            continue

        direction = str(row.get("direction", "") or "").strip().lower()
        sig_int   = 1 if direction == "buy" else (-1 if direction == "sell" else 0)
        symbol    = str(row.get("symbol", DEFAULT_SYM) or DEFAULT_SYM).strip()

        entry_px = _f(row.get("entry_price"))
        exit_px  = _f(row.get("exit_price"))
        sl_v     = _f(row.get("sl"))
        tp_v     = _f(row.get("tp"))
        pnl_usd  = _f(row.get("pnl"))
        pnl_pips = None
        if entry_px and exit_px:
            pnl_pips = round((exit_px - entry_px) * sig_int / PIP, 1)

        outcome    = str(row.get("outcome", "") or "").strip().lower()[:10]
        close_rsn  = str(row.get("close_reason", "") or "").strip()[:20]
        closed_at  = _utc(row.get("close_time"))
        confidence = _f(row.get("prob"), 4)

        trade_data = {
            "entry_price":     entry_px,
            "exit_price":      exit_px,
            "exit_reason":     close_rsn,
            "actual_pnl_usd":  pnl_usd,
            "actual_pnl_pips": pnl_pips,
            "outcome":         outcome,
            "closed_at":       closed_at,
            "sl_price":        sl_v,
            "tp_price":        tp_v,
            "rr_ratio":        _f(row.get("rr"), 3),
            "confidence":      confidence,
        }

        # Try to match a signal row by timestamp + direction
        ts_key = f"{open_ts.strftime('%Y-%m-%d %H:%M:%S')}_{sig_int}"
        matched_id = signal_id_map.get(ts_key)

        if dry_run:
            action = "UPDATE" if matched_id else "INSERT"
            logger.info("[DRY RUN] Trade %s %s @ %s → %s (id=%s)",
                        direction, symbol, open_ts, action, matched_id)
            continue

        with engine.begin() as conn:
            if matched_id:
                try:
                    conn.execute(update_sql, {**trade_data, "row_id": matched_id})
                    updated += 1
                    logger.debug("Updated signal row id=%d with trade data", matched_id)
                except Exception as exc:
                    logger.warning("Trade UPDATE failed (id=%s): %s", matched_id, exc)
                    skipped += 1
            else:
                # No matching signal row — insert standalone complete row
                try:
                    conn.execute(insert_sql, {
                        "timestamp":      open_ts,
                        "symbol":         symbol,
                        "signal":         sig_int,
                        "confidence":     confidence or 0.0,
                        "triggered_rule": f"{direction} entry (imported from trades log)",
                        "triggered_tf":   DEFAULT_TF,
                        **trade_data,
                    })
                    inserted += 1
                    logger.debug("Inserted standalone trade row @ %s", open_ts)
                except Exception as exc:
                    logger.warning("Trade INSERT failed @ %s: %s", open_ts, exc)
                    skipped += 1

    if not dry_run:
        logger.info("Trades: %d updated, %d inserted standalone, %d skipped", updated, inserted, skipped)


# ── Summary query ─────────────────────────────────────────────────────────────

def print_summary() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM ml_performance")).scalar()
        signals_only = conn.execute(text(
            "SELECT COUNT(*) FROM ml_performance WHERE entry_price IS NULL"
        )).scalar()
        with_entry = conn.execute(text(
            "SELECT COUNT(*) FROM ml_performance WHERE entry_price IS NOT NULL"
        )).scalar()
        with_exit = conn.execute(text(
            "SELECT COUNT(*) FROM ml_performance WHERE exit_price IS NOT NULL"
        )).scalar()
        wins = conn.execute(text(
            "SELECT COUNT(*) FROM ml_performance WHERE outcome='win'"
        )).scalar()
        losses = conn.execute(text(
            "SELECT COUNT(*) FROM ml_performance WHERE outcome='loss'"
        )).scalar()
        total_pnl = conn.execute(text(
            "SELECT COALESCE(SUM(actual_pnl_usd),0) FROM ml_performance WHERE actual_pnl_usd IS NOT NULL"
        )).scalar()

    print("\n" + "=" * 55)
    print("ml_performance import summary")
    print("=" * 55)
    print(f"  Total rows         : {total}")
    print(f"  Signal-only (no entry): {signals_only}")
    print(f"  Trades (with entry): {with_entry}")
    print(f"  Closed trades      : {with_exit}")
    print(f"  Wins               : {wins}")
    print(f"  Losses             : {losses}")
    print(f"  Total realised PnL : ${total_pnl:.2f}")
    print("=" * 55)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import historical log CSVs into RL/ML DB")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be imported without writing to DB")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN mode — no data will be written")

    signals = load_signals()
    trades  = load_trades()

    # Step 1: import all signal rows, capture DB ids
    signal_id_map = import_signals(signals, dry_run=args.dry_run)

    # Step 2: import trades, matching against signal rows where possible
    import_trades(trades, signal_id_map, dry_run=args.dry_run)

    if not args.dry_run:
        print_summary()
    else:
        logger.info("Dry run complete. Run without --dry-run to write to DB.")


if __name__ == "__main__":
    main()
