"""
labels.py — Zone-to-Zone Label Generation
==========================================
Entry:  LTF zone encounter (5min / 15min candle enters demand or supply zone)
TP/SL:  Real HTF (1H) zone price levels stored by features.py

Why this works:
  LTF zones are small and close together — using them for TP/SL produces
  tiny reward and wide risk, killing RR on almost every signal.
  HTF zones are institutional-level moves that are far apart and hold.
  Using real 1H zone prices gives the trade genuine room to breathe
  and produces valid RR ratios.

Label values:
   1  = buy signal that hit TP
  -1  = sell signal that hit TP
   0  = no signal, SL hit, or expired
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.labels")

LTF_TIMEFRAMES = {"1min","2min","3min","4min","5min","10min","15min","30min"}


def generate_labels(
    df: pd.DataFrame,
    max_bars: int = 50,
    min_rr: float = 1.0,
    sl_buffer_atr: float = 0.5,
    use_midline_tp: bool = True,
    timeframe: str = None,
) -> pd.DataFrame:
    """
    Generate trade labels.

    For LTF timeframes: entry from LTF zone, TP/SL from real 1H zone prices.
    For HTF timeframes: entry and TP/SL from own zone prices.

    Args:
        df:             Feature-enriched DataFrame from features.py
        max_bars:       Forward bars to simulate TP/SL outcome
        min_rr:         Minimum risk/reward ratio to accept signal
        sl_buffer_atr:  Small ATR buffer added beyond zone boundary for SL
        use_midline_tp: TP at 50% distance to next zone (per strategy doc)
        timeframe:      Timeframe name — determines LTF vs HTF logic
    """
    df = df.copy().reset_index(drop=True)

    df["signal"]        = 0
    df["signal_reason"] = ""
    df["trade_outcome"] = 0
    df["label"]         = 0
    df["tp_price"]      = np.nan
    df["sl_price"]      = np.nan
    df["rr_ratio"]      = np.nan

    use_htf = (timeframe in LTF_TIMEFRAMES) if timeframe else False
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
        # BUY — price enters demand zone
        # ----------------------------------------------------------------
        if float(row.get("in_demand_zone", 0) or 0) == 1.0:

            if use_htf:
                # SL: below real 1H demand zone bottom
                htf_d_bottom = _f(row.get("htf_demand_zone_bottom"))
                sl = (htf_d_bottom - sl_buffer_atr * atr) if not np.isnan(htf_d_bottom) \
                     else (close - 4.0 * atr)

                # TP: toward real 1H supply zone bottom
                htf_s_bottom = _f(row.get("htf_supply_zone_bottom"))
                if not np.isnan(htf_s_bottom) and htf_s_bottom > close:
                    tp = close + (htf_s_bottom - close) * (0.5 if use_midline_tp else 1.0)
                else:
                    tp = close + 4.0 * atr
            else:
                # HTF timeframe — use own zone levels
                d_bottom = _f(row.get("demand_zone_bottom"))
                s_bottom = _f(row.get("supply_zone_bottom"))
                if np.isnan(d_bottom):
                    continue
                sl = d_bottom - sl_buffer_atr * atr
                tp = (close + (s_bottom - close) * (0.5 if use_midline_tp else 1.0)) \
                     if not np.isnan(s_bottom) and s_bottom > close else close + 3.0 * atr

            risk   = close - sl
            reward = tp - close
            if risk <= 0 or reward <= 0:
                continue
            rr = reward / risk
            if rr < min_rr:
                continue

            signal = 1
            reason = f"demand_{'htf' if use_htf else 'ltf'} rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        # ----------------------------------------------------------------
        # SELL — price enters supply zone
        # ----------------------------------------------------------------
        elif float(row.get("in_supply_zone", 0) or 0) == 1.0:

            if use_htf:
                # SL: above real 1H supply zone top
                htf_s_top = _f(row.get("htf_supply_zone_top"))
                sl = (htf_s_top + sl_buffer_atr * atr) if not np.isnan(htf_s_top) \
                     else (close + 4.0 * atr)

                # TP: toward real 1H demand zone top
                htf_d_top = _f(row.get("htf_demand_zone_top"))
                if not np.isnan(htf_d_top) and htf_d_top < close:
                    tp = close - (close - htf_d_top) * (0.5 if use_midline_tp else 1.0)
                else:
                    tp = close - 4.0 * atr
            else:
                s_top   = _f(row.get("supply_zone_top"))
                d_top   = _f(row.get("demand_zone_top"))
                if np.isnan(s_top):
                    continue
                sl = s_top + sl_buffer_atr * atr
                tp = (close - (close - d_top) * (0.5 if use_midline_tp else 1.0)) \
                     if not np.isnan(d_top) and d_top < close else close - 3.0 * atr

            risk   = sl - close
            reward = close - tp
            if risk <= 0 or reward <= 0:
                continue
            rr = reward / risk
            if rr < min_rr:
                continue

            signal = -1
            reason = f"supply_{'htf' if use_htf else 'ltf'} rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        if signal == 0:
            continue

        df.at[i, "signal"]        = signal
        df.at[i, "signal_reason"] = reason
        df.at[i, "tp_price"]      = float(tp)
        df.at[i, "sl_price"]      = float(sl)

        # ----------------------------------------------------------------
        # Simulate forward price action
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


def _f(val) -> float:
    """Safe float conversion."""
    try:
        v = float(val)
        return v if not np.isnan(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


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