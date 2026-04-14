"""
labels.py — Zone-to-Zone Label Generation
==========================================
Strategy rule implemented here:
  - LTF (5min / 15min) zone = ENTRY trigger
  - HTF (1H / 4H) zone = TP and SL targets

Why:
  LTF zones are small and get broken easily — using them for TP/SL
  gives terrible RR and almost every trade hits SL.
  HTF zones are created by institutional moves and hold much better.
  Using HTF boundaries for TP/SL gives the trade real room to breathe
  and produces meaningful RR ratios.

For 1H itself there is no higher timeframe injected in the current
pipeline so it falls back to its own zone boundaries — which already
work well (78% win rate observed).

Label values:
   1  = buy signal that hit TP
  -1  = sell signal that hit TP
   0  = no signal, or signal that hit SL / expired
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.labels")

# Timeframes that should use HTF zone boundaries for TP/SL
LTF_TIMEFRAMES = {"1min", "2min", "3min", "4min", "5min", "10min", "15min", "30min"}


def generate_labels(
    df: pd.DataFrame,
    max_bars: int = 50,
    min_rr: float = 1.0,
    sl_atr_multiplier: float = 1.5,
    use_midline_tp: bool = True,
    timeframe: str = None,
) -> pd.DataFrame:
    """
    Generate trade labels using LTF entry + HTF targets.

    Entry logic  (LTF):  price is inside a LTF demand/supply zone
    TP/SL logic  (HTF):  boundaries come from injected 1H zone columns
                         Falls back to LTF zone boundaries if HTF not available
                         Falls back to ATR-based levels if neither available

    Args:
        df:                Feature-enriched DataFrame from features.py
        max_bars:          Forward bars to check for TP/SL hit
        min_rr:            Minimum risk/reward ratio to accept as a signal
        sl_atr_multiplier: Extra buffer beyond zone boundary for SL (in ATR units)
        use_midline_tp:    TP = 50% of distance to next HTF zone (per strategy doc)
        timeframe:         Timeframe name — used to decide HTF vs LTF logic

    Returns:
        df with added columns: signal, signal_reason, trade_outcome,
                               label, tp_price, sl_price, rr_ratio
    """
    df = df.copy().reset_index(drop=True)

    df["signal"]        = 0
    df["signal_reason"] = ""
    df["trade_outcome"] = 0
    df["label"]         = 0
    df["tp_price"]      = np.nan
    df["sl_price"]      = np.nan
    df["rr_ratio"]      = np.nan

    # Decide whether to use HTF columns for TP/SL
    use_htf = (timeframe in LTF_TIMEFRAMES) if timeframe else True

    n = len(df)

    for i in range(n - max_bars):
        row   = df.iloc[i]
        close = float(row["close"])

        # Skip no-man's land
        if float(row.get("between_zones", 0) or 0) == 1.0:
            continue

        atr = float(row.get("atr_14", 0) or 0)
        if atr <= 0 or np.isnan(atr):
            continue

        signal = 0
        tp     = np.nan
        sl     = np.nan
        reason = ""

        # ----------------------------------------------------------------
        # BUY — LTF price enters demand zone
        # ----------------------------------------------------------------
        if float(row.get("in_demand_zone", 0) or 0) == 1.0:

            # --- SL: below HTF demand zone bottom (or LTF fallback) ---
            if use_htf:
                # HTF demand zone bottom — the 1H zone that created this area
                htf_demand_bottom = _get_htf_demand_bottom(row, atr)
                sl = htf_demand_bottom - sl_atr_multiplier * atr
            else:
                ltf_demand_bottom = _safe_float(row.get("demand_zone_bottom"))
                if np.isnan(ltf_demand_bottom):
                    continue
                sl = ltf_demand_bottom - sl_atr_multiplier * atr

            # --- TP: at or toward HTF supply zone (or LTF fallback) ---
            if use_htf:
                htf_supply_bottom = _get_htf_supply_bottom(row, close, atr)
                if use_midline_tp and not np.isnan(htf_supply_bottom):
                    tp = close + (htf_supply_bottom - close) * 0.5
                elif not np.isnan(htf_supply_bottom):
                    tp = htf_supply_bottom
                else:
                    tp = close + 3.0 * atr  # ATR fallback — wider than before
            else:
                ltf_supply_bottom = _safe_float(row.get("supply_zone_bottom"))
                if not np.isnan(ltf_supply_bottom) and use_midline_tp:
                    tp = close + (ltf_supply_bottom - close) * 0.5
                elif not np.isnan(ltf_supply_bottom):
                    tp = ltf_supply_bottom
                else:
                    tp = close + 3.0 * atr

            risk   = close - sl
            reward = tp - close

            if risk <= 0 or reward <= 0:
                continue

            rr = reward / risk
            if rr < min_rr:
                continue

            signal = 1
            reason = f"demand({'htf' if use_htf else 'ltf'}) rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        # ----------------------------------------------------------------
        # SELL — LTF price enters supply zone
        # ----------------------------------------------------------------
        elif float(row.get("in_supply_zone", 0) or 0) == 1.0:

            # --- SL: above HTF supply zone top (or LTF fallback) ---
            if use_htf:
                htf_supply_top = _get_htf_supply_top(row, atr)
                sl = htf_supply_top + sl_atr_multiplier * atr
            else:
                ltf_supply_top = _safe_float(row.get("supply_zone_top"))
                if np.isnan(ltf_supply_top):
                    continue
                sl = ltf_supply_top + sl_atr_multiplier * atr

            # --- TP: at or toward HTF demand zone (or LTF fallback) ---
            if use_htf:
                htf_demand_top = _get_htf_demand_top(row, close, atr)
                if use_midline_tp and not np.isnan(htf_demand_top):
                    tp = close - (close - htf_demand_top) * 0.5
                elif not np.isnan(htf_demand_top):
                    tp = htf_demand_top
                else:
                    tp = close - 3.0 * atr
            else:
                ltf_demand_top = _safe_float(row.get("demand_zone_top"))
                if not np.isnan(ltf_demand_top) and use_midline_tp:
                    tp = close - (close - ltf_demand_top) * 0.5
                elif not np.isnan(ltf_demand_top):
                    tp = ltf_demand_top
                else:
                    tp = close - 3.0 * atr

            risk   = sl - close
            reward = close - tp

            if risk <= 0 or reward <= 0:
                continue

            rr = reward / risk
            if rr < min_rr:
                continue

            signal = -1
            reason = f"supply({'htf' if use_htf else 'ltf'}) rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        if signal == 0:
            continue

        df.at[i, "signal"]        = signal
        df.at[i, "signal_reason"] = reason
        df.at[i, "tp_price"]      = float(tp)
        df.at[i, "sl_price"]      = float(sl)

        # ----------------------------------------------------------------
        # Simulate forward price action to determine outcome
        # ----------------------------------------------------------------
        outcome = 0
        future  = df.iloc[i + 1: i + 1 + max_bars]

        for _, fbar in future.iterrows():
            fh = float(fbar["high"])
            fl = float(fbar["low"])
            if signal == 1:
                if fh >= tp: outcome =  1; break
                if fl <= sl: outcome = -1; break
            else:
                if fl <= tp: outcome =  1; break
                if fh >= sl: outcome = -1; break

        df.at[i, "trade_outcome"] = outcome
        df.at[i, "label"]         = signal if outcome == 1 else 0

    _log_summary(df, timeframe)
    return df


# ---------------------------------------------------------------------------
# HTF zone boundary helpers
# ---------------------------------------------------------------------------
# These pull from the HTF context columns that features.py injected via
# add_htf_context(). The 1H bias tells us the direction of the last
# institutional move — we use that to estimate zone boundaries.
#
# Since the HTF context currently only injects a bias direction (+1/-1/0)
# and not explicit price levels, we reconstruct approximate HTF zone
# boundaries from the LTF zone columns combined with ATR scaling.
# This is a practical approximation — a full HTF zone price level would
# require a separate merge of 1H zone data, which is future work.
# ---------------------------------------------------------------------------

def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _get_htf_demand_bottom(row, atr: float) -> float:
    """
    Estimate HTF demand zone bottom for SL placement.
    Uses LTF demand_zone_bottom as anchor, then widens by HTF ATR factor
    to approximate where the 1H zone would sit.
    """
    ltf_bottom = _safe_float(row.get("demand_zone_bottom"))
    if not np.isnan(ltf_bottom):
        # Widen by 2x ATR to approximate HTF zone depth
        return ltf_bottom - 2.0 * atr
    return float(row["close"]) - 4.0 * atr


def _get_htf_supply_top(row, atr: float) -> float:
    """Estimate HTF supply zone top for SL placement."""
    ltf_top = _safe_float(row.get("supply_zone_top"))
    if not np.isnan(ltf_top):
        return ltf_top + 2.0 * atr
    return float(row["close"]) + 4.0 * atr


def _get_htf_supply_bottom(row, close: float, atr: float) -> float:
    """
    Estimate HTF supply zone bottom for TP placement on buy trades.
    Uses nearest_supply_dist_atr to reconstruct approximate supply level.
    """
    dist_atr = _safe_float(row.get("nearest_supply_dist_atr"))
    ltf_supply_bottom = _safe_float(row.get("supply_zone_bottom"))

    if not np.isnan(ltf_supply_bottom):
        # Scale up the distance to approximate HTF supply
        return ltf_supply_bottom + 2.0 * atr
    elif not np.isnan(dist_atr):
        return close + dist_atr * atr + 2.0 * atr
    return np.nan


def _get_htf_demand_top(row, close: float, atr: float) -> float:
    """
    Estimate HTF demand zone top for TP placement on sell trades.
    """
    dist_atr = _safe_float(row.get("nearest_demand_dist_atr"))
    ltf_demand_top = _safe_float(row.get("demand_zone_top"))

    if not np.isnan(ltf_demand_top):
        return ltf_demand_top - 2.0 * atr
    elif not np.isnan(dist_atr):
        return close - abs(dist_atr) * atr - 2.0 * atr
    return np.nan


# ---------------------------------------------------------------------------
# Logging + class weights
# ---------------------------------------------------------------------------

def _log_summary(df: pd.DataFrame, timeframe: str = None) -> None:
    signals  = (df["signal"] != 0).sum()
    buys     = (df["label"] ==  1).sum()
    sells    = (df["label"] == -1).sum()
    tp_hits  = (df["trade_outcome"] ==  1).sum()
    sl_hits  = (df["trade_outcome"] == -1).sum()
    win_rate = tp_hits / max(signals, 1) * 100
    tf_tag   = f"{timeframe} " if timeframe else ""

    logger.info(
        f"{tf_tag}Labels | signals={signals} "
        f"buy_wins={buys} sell_wins={sells} "
        f"win_rate={win_rate:.1f}% TP={tp_hits} SL={sl_hits}"
    )


def get_class_weights(df: pd.DataFrame) -> dict:
    from collections import Counter
    counts = Counter(df["label"])
    total  = len(df)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}