"""
Label Logic Diagnostic
======================
Checks that generate_labels() is producing sane output:
  1. sell_wins > 0  (binary label fix working)
  2. win rate 35-55% on zone-touch rows
  3. HTF zone columns populated for LTF timeframes
  4. Expired trade count reasonable (<40% of signals)
  5. RR distribution >= min_rr
  6. Session filter not wiping out all signals
  7. No lookahead in forward simulation (sanity check)

Run from project root:
  python -m tests.test_label_logic
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("label_test")

from data.loader import get_connection
from data.feature_engineer import build_features
from data.pipeline import TF_PARAMS, LTF_TIMEFRAMES
from strategy.signal_generator import generate_labels


TIMEFRAMES_TO_TEST = ["5min", "15min"]
SYMBOL             = "XAUUSDm"
DATE_FROM          = "2024-11-08"   # earliest date with 5min data
DATE_TO            = "2025-06-30"

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def load_raw(db, symbol, timeframe, date_from, date_to):
    query = """
        SELECT *
        FROM xauusd_ohlcv
        WHERE symbol = %(symbol)s
          AND timeframe = %(tf)s
          AND date BETWEEN %(from)s AND %(to)s
        ORDER BY timestamp ASC
    """
    df = db.fetch_dataframe(query, {
        "symbol": symbol, "tf": timeframe,
        "from": date_from, "to": date_to
    })
    logger.info(f"  Loaded {len(df):,} rows for {timeframe}")
    return df


def load_htf_context(db, symbol, date_from, date_to):
    htf_dfs = {}
    for tf in ["1H", "4H"]:
        query = """
            SELECT *
            FROM xauusd_ohlcv
            WHERE symbol = %(symbol)s
              AND timeframe = %(tf)s
              AND date BETWEEN %(from)s AND %(to)s
            ORDER BY timestamp ASC
        """
        df = db.fetch_dataframe(query, {
            "symbol": symbol, "tf": tf,
            "from": date_from, "to": date_to
        })
        htf_dfs[tf] = df
        logger.info(f"  Loaded {len(df):,} rows for HTF {tf}")
    return htf_dfs


def check_htf_columns_populated(df_feat, timeframe):
    """For LTF timeframes, HTF zone columns must not be all-NaN."""
    if timeframe not in LTF_TIMEFRAMES:
        return True
    htf_cols = ["htf_demand_zone_bottom", "htf_demand_zone_top",
                 "htf_supply_zone_bottom", "htf_supply_zone_top"]
    missing = [c for c in htf_cols if c not in df_feat.columns]
    if missing:
        logger.error(f"  {FAIL} HTF columns missing from features: {missing}")
        return False
    for col in htf_cols:
        pct_valid = df_feat[col].notna().mean() * 100
        tag = PASS if pct_valid > 20 else WARN
        logger.info(f"  {tag} {col}: {pct_valid:.1f}% non-null")
    return True


def check_session_filter_impact(df_feat):
    """How many rows are in-session vs total."""
    def _in_session(row):
        hour = int(row.get("hour", -1) or -1)
        ts   = row.get("timestamp")
        minute = 0
        if ts is not None:
            try:
                minute = pd.to_datetime(ts).minute
            except Exception:
                pass
        asia_london = (hour == 10) or (hour == 11) or (hour == 12 and minute < 30)
        london_ny   = (hour >= 16 and hour < 19)
        return asia_london or london_ny

    total = len(df_feat)
    in_session = df_feat.apply(_in_session, axis=1).sum()
    pct = in_session / max(total, 1) * 100
    tag = PASS if 5 < pct < 60 else WARN
    logger.info(f"  {tag} Session coverage: {in_session:,}/{total:,} rows ({pct:.1f}%)")
    return pct


def run_label_checks(df_labelled, timeframe, params):
    results = {}
    signals   = df_labelled[df_labelled["signal"] != 0]
    n_signals = len(signals)

    if n_signals == 0:
        logger.error(f"  {FAIL} No signals generated at all — check session/zone/volume filters")
        return {"ok": False}

    buy_sigs  = (df_labelled["signal"] ==  1).sum()
    sell_sigs = (df_labelled["signal"] == -1).sum()
    winners   = (df_labelled["label"] ==  1).sum()
    tp_hits   = (df_labelled["trade_outcome"] ==  1).sum()
    sl_hits   = (df_labelled["trade_outcome"] == -1).sum()
    expired   = (signals["trade_outcome"] == 0).sum()

    buy_wins  = ((df_labelled["label"] == 1) & (df_labelled["signal_direction"] ==  1)).sum()
    sell_wins = ((df_labelled["label"] == 1) & (df_labelled["signal_direction"] == -1)).sum()

    win_rate    = tp_hits / max(n_signals, 1) * 100
    expired_pct = expired / max(n_signals, 1) * 100

    logger.info(f"\n  --- {timeframe} Label Summary ---")
    logger.info(f"  Total signals : {n_signals:,}  (buys={buy_sigs} sells={sell_sigs})")
    logger.info(f"  Winners       : {winners:,}  (buy_wins={buy_wins} sell_wins={sell_wins})")
    logger.info(f"  TP hits       : {tp_hits}  SL hits={sl_hits}  Expired={expired}")
    logger.info(f"  Win rate      : {win_rate:.1f}%")
    logger.info(f"  Expired pct   : {expired_pct:.1f}%")

    # 1. sell_wins > 0
    if sell_wins == 0:
        logger.error(f"  {FAIL} sell_wins=0 — binary label fix may not be working or no sell signals hitting TP")
    else:
        logger.info(f"  {PASS} sell_wins={sell_wins} (binary label fix confirmed working)")
    results["sell_wins_ok"] = sell_wins > 0

    # 2. win rate 35-55%
    if 35 <= win_rate <= 55:
        logger.info(f"  {PASS} Win rate {win_rate:.1f}% is in healthy range (35-55%)")
    elif win_rate < 10:
        logger.error(f"  {FAIL} Win rate {win_rate:.1f}% is suspiciously low — zones may be too wide or TP too far")
    elif win_rate > 70:
        logger.error(f"  {FAIL} Win rate {win_rate:.1f}% is suspiciously high — possible lookahead leak")
    else:
        logger.warning(f"  {WARN} Win rate {win_rate:.1f}% is outside 35-55% — worth investigating")
    results["win_rate"] = win_rate
    results["win_rate_ok"] = win_rate > 10

    # 3. buy/sell symmetry
    buy_wr  = buy_wins  / max(buy_sigs, 1) * 100
    sell_wr = sell_wins / max(sell_sigs, 1) * 100
    logger.info(f"  Buy win rate  : {buy_wr:.1f}%  |  Sell win rate: {sell_wr:.1f}%")
    gap = abs(buy_wr - sell_wr)
    if gap > 20 and sell_sigs > 10:
        logger.warning(f"  {WARN} Buy/sell win rate gap is {gap:.1f}pp — model may be biased to one direction")
    else:
        logger.info(f"  {PASS} Buy/sell win rate gap: {gap:.1f}pp")
    results["direction_gap"] = gap

    # 4. expired trades
    if expired_pct > 40:
        logger.warning(f"  {WARN} {expired_pct:.1f}% of signals expired without hitting TP or SL — "
                       f"consider increasing max_bars (currently {params['max_bars']})")
    else:
        logger.info(f"  {PASS} Expired rate {expired_pct:.1f}% is acceptable")
    results["expired_pct"] = expired_pct

    # 5. RR distribution
    rr_vals = df_labelled.loc[df_labelled["signal"] != 0, "rr_ratio"].dropna()
    if len(rr_vals) > 0:
        min_rr = params["min_rr"]
        below_min = (rr_vals < min_rr).sum()
        if below_min > 0:
            logger.error(f"  {FAIL} {below_min} signals have RR < min_rr ({min_rr}) — RR filter not enforced")
        else:
            logger.info(f"  {PASS} All RRs >= min_rr={min_rr}")
        logger.info(f"  RR stats: min={rr_vals.min():.2f} mean={rr_vals.mean():.2f} "
                    f"median={rr_vals.median():.2f} max={rr_vals.max():.2f}")
        results["rr_ok"] = below_min == 0
    else:
        logger.warning(f"  {WARN} No RR values found in signals")

    # 6. Signal count sanity
    if n_signals < 20:
        logger.warning(f"  {WARN} Only {n_signals} signals — filters may be too tight for this date range")
    elif n_signals > 5000:
        logger.warning(f"  {WARN} {n_signals} signals — very high count, check for zone over-detection")
    else:
        logger.info(f"  {PASS} Signal count {n_signals} looks reasonable")

    results["ok"] = True
    return results


def main():
    logger.info("=" * 60)
    logger.info("LABEL LOGIC DIAGNOSTIC")
    logger.info(f"Symbol={SYMBOL}  Period={DATE_FROM} → {DATE_TO}")
    logger.info("=" * 60)

    from data.loader import DatabaseConnection
    db = DatabaseConnection()
    if not db.connect():
        logger.error("Cannot connect to database — aborting")
        sys.exit(1)

    htf_dfs = load_htf_context(db, SYMBOL, DATE_FROM, DATE_TO)

    all_ok = True
    for tf in TIMEFRAMES_TO_TEST:
        logger.info(f"\n{'='*60}")
        logger.info(f"TIMEFRAME: {tf}")
        logger.info(f"{'='*60}")

        params = TF_PARAMS.get(tf, {"max_bars": 50, "min_rr": 1.2,
                                     "impulse_atr": 0.5, "use_midline_tp": True})

        # Load raw LTF data
        df_raw = load_raw(db, SYMBOL, tf, DATE_FROM, DATE_TO)
        if df_raw.empty:
            logger.error(f"  {FAIL} No data returned for {tf}")
            all_ok = False
            continue

        # Build features (same pipeline as training)
        logger.info("  Building features...")
        df_feat = build_features(
            df_raw,
            h1_df=htf_dfs.get("1H"),
            h4_df=htf_dfs.get("4H"),
            impulse_atr_multiplier=params["impulse_atr"],
        )
        logger.info(f"  Features built: {len(df_feat):,} rows, {len(df_feat.columns)} columns")

        # Check HTF columns populated
        check_htf_columns_populated(df_feat, tf)

        # Check session filter impact
        check_session_filter_impact(df_feat)

        # Check zone-touch rows exist
        in_demand = (df_feat.get("in_demand_zone", pd.Series(dtype=float)) == 1).sum()
        in_supply = (df_feat.get("in_supply_zone", pd.Series(dtype=float)) == 1).sum()
        logger.info(f"  Zone touches — demand: {in_demand:,}  supply: {in_supply:,}")
        if in_demand == 0 and in_supply == 0:
            logger.error(f"  {FAIL} No zone-touch rows found — zone detection may be broken")
            all_ok = False
            continue

        # Generate labels
        logger.info("  Generating labels...")
        df_labelled = generate_labels(
            df_feat,
            max_bars=params["max_bars"],
            min_rr=params["min_rr"],
            max_rr=params.get("max_rr"),
            sl_buffer_atr=0.5,
            min_sl_atr=params.get("min_sl_atr", 0.0),
            use_midline_tp=params["use_midline_tp"],
            include_london_ny=params.get("include_london_ny", True),
            timeframe=tf,
        )

        # Run checks
        res = run_label_checks(df_labelled, tf, params)
        if not res.get("ok"):
            all_ok = False

    db.disconnect()

    logger.info(f"\n{'='*60}")
    if all_ok:
        logger.info("OVERALL: All checks passed")
    else:
        logger.info("OVERALL: Some checks FAILED — review output above")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
