"""
Win Rate Breakdown Diagnostic
==============================
Answers: why is the win rate ~28-34%?

Checks:
  1. Real HTF-zone TP vs fallback ATR TP — does one path win more?
  2. RR bucket distribution — are most signals just scraping 1.20?
  3. SL tightness — is the SL being hit before price moves at all?
  4. Per-hour win rate — are certain sessions dragging the average?
  5. Zone freshness — fresh zones vs re-touched zones win rate
  6. Direction breakdown — does one side consistently lose?
  7. TP distance in ATR — how far is price being asked to travel?

Run from project root:
  python -m tests.test_wr_breakdown
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("wr_breakdown")

from data.loader import DatabaseConnection
from data.feature_engineer import build_features
from data.pipeline import TF_PARAMS, LTF_TIMEFRAMES
from strategy.signal_generator import generate_labels

SYMBOL    = "XAUUSDm"
DATE_FROM = "2024-11-08"
DATE_TO   = "2025-06-30"
TIMEFRAME = "15min"


def section(title):
    logger.info(f"\n{'─'*55}")
    logger.info(f"  {title}")
    logger.info(f"{'─'*55}")


def load_and_label():
    db = DatabaseConnection()
    db.connect()

    params = TF_PARAMS[TIMEFRAME]

    def fetch(tf, d_from, d_to):
        q = """
            SELECT * FROM xauusd_ohlcv
            WHERE symbol=%(s)s AND timeframe=%(tf)s
              AND date BETWEEN %(f)s AND %(t)s
            ORDER BY timestamp ASC
        """
        return db.fetch_dataframe(q, {"s": SYMBOL, "tf": tf, "f": d_from, "t": d_to})

    df_raw  = fetch(TIMEFRAME, DATE_FROM, DATE_TO)
    df_1h   = fetch("1H",      DATE_FROM, DATE_TO)
    df_4h   = fetch("4H",      DATE_FROM, DATE_TO)

    logger.info(f"Loaded: {len(df_raw):,} {TIMEFRAME}  |  {len(df_1h):,} 1H  |  {len(df_4h):,} 4H")

    df_feat = build_features(df_raw, h1_df=df_1h, h4_df=df_4h,
                             impulse_atr_multiplier=params["impulse_atr"])

    df_lab  = generate_labels(df_feat,
                              max_bars=params["max_bars"],
                              min_rr=params["min_rr"],
                              max_rr=params.get("max_rr"),
                              sl_buffer_atr=0.5,
                              min_sl_atr=params.get("min_sl_atr", 0.0),
                              use_midline_tp=params["use_midline_tp"],
                              include_london_ny=params.get("include_london_ny", True),
                              timeframe=TIMEFRAME)
    db.disconnect()
    return df_lab, params


def analyse(df: pd.DataFrame, params: dict):
    sigs = df[df["signal"] != 0].copy()
    if sigs.empty:
        logger.error("No signals found — nothing to analyse")
        return

    atr = df["atr_14"].median()
    sigs["tp_dist_atr"] = (sigs["tp_price"] - sigs["close"]).abs() / atr.clip(1e-6)
    sigs["sl_dist_atr"] = (sigs["sl_price"] - sigs["close"]).abs() / atr.clip(1e-6)

    # ── 1. HTF zone TP vs fallback ATR TP ───────────────────────────────────
    section("1. HTF-ZONE TP vs FALLBACK ATR TP")

    # A signal used a real HTF zone TP if the opposing HTF zone column is non-NaN
    # and the TP is close to that zone level.
    def _used_htf_tp(row):
        if row["signal"] == 1:   # buy → TP should be near htf_supply_zone_bottom
            ref = row.get("htf_supply_zone_bottom", np.nan)
        else:                    # sell → TP near htf_demand_zone_top
            ref = row.get("htf_demand_zone_top", np.nan)
        if pd.isna(ref) or pd.isna(row["tp_price"]):
            return "fallback"
        # TP is within 5% of the HTF zone reference (midline vs full varies)
        rel_diff = abs(row["tp_price"] - ref) / max(abs(ref), 1e-6)
        return "htf_zone" if rel_diff < 0.10 else "fallback"

    sigs["tp_source"] = sigs.apply(_used_htf_tp, axis=1)
    for src, grp in sigs.groupby("tp_source"):
        wr = (grp["label"] == 1).mean() * 100
        n  = len(grp)
        logger.info(f"  {src:12s}: n={n:4d}  win_rate={wr:.1f}%  "
                    f"rr_mean={grp['rr_ratio'].mean():.2f}  "
                    f"tp_dist_atr_mean={grp['tp_dist_atr'].mean():.2f}")

    # ── 2. RR bucket distribution ────────────────────────────────────────────
    section("2. RR BUCKET DISTRIBUTION")
    bins   = [0, 1.21, 1.5, 2.0, 3.0, 5.0, 999]
    labels = ["1.20 (floor)", "1.21-1.50", "1.51-2.00", "2.01-3.00", "3.01-5.00", ">5.00"]
    sigs["rr_bucket"] = pd.cut(sigs["rr_ratio"], bins=bins, labels=labels, right=True)
    for bucket, grp in sigs.groupby("rr_bucket", observed=True):
        wr   = (grp["label"] == 1).mean() * 100
        pct  = len(grp) / len(sigs) * 100
        logger.info(f"  RR {str(bucket):15s}: n={len(grp):4d} ({pct:4.1f}%)  win_rate={wr:.1f}%")

    # ── 3. SL tightness ─────────────────────────────────────────────────────
    section("3. SL DISTANCE FROM ENTRY (in ATR)")
    sl_med  = sigs["sl_dist_atr"].median()
    sl_mean = sigs["sl_dist_atr"].mean()
    sl_p10  = sigs["sl_dist_atr"].quantile(0.10)
    sl_p90  = sigs["sl_dist_atr"].quantile(0.90)
    logger.info(f"  SL dist: p10={sl_p10:.2f} ATR  median={sl_med:.2f} ATR  "
                f"mean={sl_mean:.2f} ATR  p90={sl_p90:.2f} ATR")

    # SL bins
    sl_bins   = [0, 0.5, 1.0, 2.0, 3.0, 999]
    sl_labels = ["<0.5 ATR", "0.5-1.0 ATR", "1.0-2.0 ATR", "2.0-3.0 ATR", ">3.0 ATR"]
    sigs["sl_bucket"] = pd.cut(sigs["sl_dist_atr"], bins=sl_bins, labels=sl_labels, right=True)
    for bucket, grp in sigs.groupby("sl_bucket", observed=True):
        wr   = (grp["label"] == 1).mean() * 100
        pct  = len(grp) / len(sigs) * 100
        logger.info(f"  SL {str(bucket):15s}: n={len(grp):4d} ({pct:4.1f}%)  win_rate={wr:.1f}%")

    # ── 4. Per-hour win rate ─────────────────────────────────────────────────
    section("4. WIN RATE BY HOUR (broker GMT+3)")
    sigs["hour_int"] = sigs["hour"].astype(int)
    hour_stats = (
        sigs.groupby("hour_int")["label"]
        .agg(["count", "sum"])
        .rename(columns={"count": "n", "sum": "wins"})
    )
    hour_stats["wr"] = hour_stats["wins"] / hour_stats["n"] * 100
    for h, row in hour_stats.iterrows():
        bar = "█" * int(row["wr"] / 5)
        logger.info(f"  Hour {h:02d}:00  n={int(row['n']):4d}  wr={row['wr']:5.1f}%  {bar}")

    # ── 5. Zone freshness ────────────────────────────────────────────────────
    section("5. ZONE FRESHNESS (fresh=never touched vs re-touched)")
    for direction, col in [(1, "demand_zone_fresh"), (-1, "supply_zone_fresh")]:
        grp = sigs[sigs["signal"] == direction]
        if grp.empty or col not in grp.columns:
            continue
        fresh   = grp[grp[col] == 1.0]
        touched = grp[grp[col] != 1.0]
        dir_str = "BUY (demand)" if direction == 1 else "SELL (supply)"
        wr_f = (fresh["label"] == 1).mean() * 100 if len(fresh) else 0
        wr_t = (touched["label"] == 1).mean() * 100 if len(touched) else 0
        logger.info(f"  {dir_str}")
        logger.info(f"    Fresh   zones: n={len(fresh):4d}  wr={wr_f:.1f}%")
        logger.info(f"    Touched zones: n={len(touched):4d}  wr={wr_t:.1f}%")

    # ── 6. TP distance distribution ─────────────────────────────────────────
    section("6. TP DISTANCE FROM ENTRY (in ATR)")
    tp_med  = sigs["tp_dist_atr"].median()
    tp_mean = sigs["tp_dist_atr"].mean()
    tp_p25  = sigs["tp_dist_atr"].quantile(0.25)
    tp_p75  = sigs["tp_dist_atr"].quantile(0.75)
    tp_p90  = sigs["tp_dist_atr"].quantile(0.90)
    logger.info(f"  TP dist: p25={tp_p25:.1f}  median={tp_med:.1f}  "
                f"mean={tp_mean:.1f}  p75={tp_p75:.1f}  p90={tp_p90:.1f}  (ATR units)")
    logger.info(f"  NOTE: max_bars={params['max_bars']} × 5min candle = "
                f"~{params['max_bars'] * 5 // 60}h {params['max_bars'] * 5 % 60}m forward window")

    tp_bins   = [0, 1, 2, 4, 8, 15, 999]
    tp_labels = ["<1 ATR", "1-2 ATR", "2-4 ATR", "4-8 ATR", "8-15 ATR", ">15 ATR"]
    sigs["tp_bucket"] = pd.cut(sigs["tp_dist_atr"], bins=tp_bins, labels=tp_labels, right=True)
    for bucket, grp in sigs.groupby("tp_bucket", observed=True):
        wr  = (grp["label"] == 1).mean() * 100
        pct = len(grp) / len(sigs) * 100
        logger.info(f"  TP {str(bucket):12s}: n={len(grp):4d} ({pct:4.1f}%)  win_rate={wr:.1f}%")

    # ── 7. Outcome summary ───────────────────────────────────────────────────
    section("7. OUTCOME BREAKDOWN")
    tp_hit  = (sigs["trade_outcome"] ==  1).sum()
    sl_hit  = (sigs["trade_outcome"] == -1).sum()
    expired = (sigs["trade_outcome"] ==  0).sum()
    n       = len(sigs)
    logger.info(f"  TP hit : {tp_hit:4d} ({tp_hit/n*100:.1f}%)  ← these are winners")
    logger.info(f"  SL hit : {sl_hit:4d} ({sl_hit/n*100:.1f}%)  ← losers — SL too tight?")
    logger.info(f"  Expired: {expired:4d} ({expired/n*100:.1f}%)  ← time ran out — TP too far?")

    # SL hit rate by direction
    for direction, label in [(1, "BUY"), (-1, "SELL")]:
        grp    = sigs[sigs["signal"] == direction]
        sl_pct = (grp["trade_outcome"] == -1).mean() * 100
        tp_pct = (grp["trade_outcome"] ==  1).mean() * 100
        ex_pct = (grp["trade_outcome"] ==  0).mean() * 100
        logger.info(f"  {label}: TP={tp_pct:.1f}%  SL={sl_pct:.1f}%  Expired={ex_pct:.1f}%  (n={len(grp)})")


def main():
    logger.info("=" * 55)
    logger.info(f"WIN RATE BREAKDOWN  |  {TIMEFRAME}  |  {DATE_FROM} → {DATE_TO}")
    logger.info("=" * 55)
    df, params = load_and_label()
    analyse(df, params)
    logger.info("\n" + "=" * 55)
    logger.info("Done.")


if __name__ == "__main__":
    main()
