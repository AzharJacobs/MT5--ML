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
   1  = winner (buy OR sell signal that hit TP)
   0  = loser  (SL hit, expired, or no signal)

CRITICAL FIX (sell_wins=0 bug):
  Previously sell winners produced label=-1 because signal=-1 and the
  code did `label = signal if outcome == 1 else 0`. The XGBoost label
  map only contained {0, 1} so every -1 was silently treated as a
  separate third class — the model NEVER learned from sell winners.
  Result: sell_wins=0 across all timeframes, model predicted only buys.

  Fix: label is now BINARY — 1=winner (either direction), 0=loser.
  Direction is encoded in the features (in_supply_zone, in_demand_zone,
  confirmation scores) so the model still learns what conditions lead to
  wins. A separate `signal_direction` column is kept for diagnostics.

FILTER CHANGES (win-rate fix, diagnosed 2026-04-29):
  Three filters added after breakdown analysis showed WR=28% root causes:
  1. MAX_TP_ATR=6.0  — TP > 6 ATR unreachable in the 5h forward window;
     those signals had wr=9-17% and were dragging the average down.
  2. MAX_SL_ATR=3.0  — SL > 3 ATR means the zone is too wide; price wicks
     through without the move paying off. wr jumped to 43-49% in 1-3 ATR range.
  3. Session narrowed to hours 10-12:30 + 16:00 only (drop 17-18) — London/NY
     overlap hours 17-18 had wr=19% on 39% of all signals.
  - MIN_VOLUME_RATIO lowered from 0.8 → 0.6 (previous change).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("mt5_collector.labels")

LTF_TIMEFRAMES = {"1min","2min","3min","4min","5min","10min","15min","30min"}

# Candle volume must be at least 60% of the 20-bar rolling average to qualify.
MIN_VOLUME_RATIO = 0.6

# TP further than this many ATR from entry is unreachable in the forward window.
MAX_TP_ATR = 6.0

# SL further than this many ATR indicates the zone is too wide to trade cleanly.
MAX_SL_ATR = 3.0


def generate_labels(
    df: pd.DataFrame,
    max_bars: int = 50,
    min_rr: float = 1.5,
    max_rr: float = None,
    sl_buffer_atr: float = 0.5,
    min_sl_atr: float = 0.0,
    use_midline_tp: bool = True,
    include_london_ny: bool = True,
    timeframe: str = None,
) -> pd.DataFrame:
    """
    Generate trade labels.

    For LTF timeframes: entry from LTF zone, TP/SL from real 1H zone prices.
    For HTF timeframes: entry and TP/SL from own zone prices.

    Args:
        df:               Feature-enriched DataFrame from features.py
        max_bars:         Forward bars to simulate TP/SL outcome
        min_rr:           Minimum risk/reward ratio to accept signal
        max_rr:           Maximum RR cap — signals above this are skipped (None = no cap)
        sl_buffer_atr:    Small ATR buffer added beyond zone boundary for SL
        min_sl_atr:       Minimum SL distance in ATR — filters noise stops on tiny zones
        use_midline_tp:   TP at 50% distance to next zone. True for LTF, False for HTF.
        include_london_ny: Include London/NY overlap window (16:00). Disable for 15min
                           where H16 wr=24% vs London open H10-12 wr=37-42%.
        timeframe:        Timeframe name — determines LTF vs HTF logic

    Label output (binary):
        label=1  → zone entry that hit TP (winner, buy or sell)
        label=0  → not a signal, SL hit, or expired
    """
    df = df.copy().reset_index(drop=True)

    df["signal"]           = 0       # -1=sell entry, 0=none, 1=buy entry
    df["signal_direction"] = 0       # same as signal — kept for diagnostics only
    df["signal_reason"]    = ""
    df["trade_outcome"]    = 0       # 1=TP hit, -1=SL hit, 0=expired
    df["label"]            = 0       # BINARY target: 1=winner, 0=loser
    df["tp_price"]         = np.nan
    df["sl_price"]         = np.nan
    df["rr_ratio"]         = np.nan

    use_htf = (timeframe in LTF_TIMEFRAMES) if timeframe else False
    n = len(df)

    # Trading session windows in BROKER time (Exness GMT+3).
    # Hours 17-18 dropped for all TFs: wr=19% on 39% of 5min signals.
    # H16 kept only when include_london_ny=True (5min); dropped for 15min (wr=24%).
    def _in_trading_session(row) -> bool:
        hour = int(row.get("hour", -1) or -1)
        ts   = row.get("timestamp")
        minute = 0
        if ts is not None:
            try:
                minute = pd.to_datetime(ts).minute
            except Exception:
                minute = 0
        asia_london = (hour == 10) or (hour == 11) or (hour == 12 and minute < 30)
        ny_open     = (hour == 13) or (hour == 14)
        london_ny   = include_london_ny and (hour == 16)
        return asia_london or ny_open or london_ny

    for i in range(n - max_bars):
        row   = df.iloc[i]
        close = float(row["close"])

        # Session filter
        if not _in_trading_session(row):
            continue

        # No-man's land filter
        if float(row.get("between_zones", 0) or 0) == 1.0:
            continue

        # ATR guard
        atr = float(row.get("atr_14", 0) or 0)
        if atr <= 0 or np.isnan(atr):
            continue

        # Volume participation filter
        vol_ratio = float(row.get("volume_ratio", 1.0) or 1.0)
        if np.isnan(vol_ratio) or vol_ratio < MIN_VOLUME_RATIO:
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
                htf_d_bottom = _f(row.get("htf_demand_zone_bottom"))
                sl   = (htf_d_bottom - sl_buffer_atr * atr) if not np.isnan(htf_d_bottom) \
                       else (close - 4.0 * atr)
                risk = close - sl  # compute risk before TP so fallback can guarantee min RR

                htf_s_bottom = _f(row.get("htf_supply_zone_bottom"))
                if not np.isnan(htf_s_bottom) and htf_s_bottom > close:
                    tp = close + (htf_s_bottom - close) * (0.5 if use_midline_tp else 1.0)
                else:
                    tp = close + max(min_rr * risk, 2.0 * atr)
            else:
                d_bottom = _f(row.get("demand_zone_bottom"))
                s_bottom = _f(row.get("supply_zone_bottom"))
                if np.isnan(d_bottom):
                    continue
                sl   = d_bottom - sl_buffer_atr * atr
                risk = close - sl  # compute risk before TP so fallback can guarantee min RR
                tp   = (close + (s_bottom - close) * (0.5 if use_midline_tp else 1.0)) \
                       if not np.isnan(s_bottom) and s_bottom > close \
                       else close + max(min_rr * risk, 3.0 * atr)

            reward = tp - close
            if risk <= 0 or reward <= 0:
                continue
            rr = reward / risk
            if rr < min_rr:
                continue
            if max_rr is not None and rr > max_rr:
                continue
            if reward / atr > MAX_TP_ATR:
                continue
            if risk / atr > MAX_SL_ATR:
                continue
            if min_sl_atr > 0 and risk / atr < min_sl_atr:
                continue

            signal = 1
            reason = f"demand_{'htf' if use_htf else 'ltf'} rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        # ----------------------------------------------------------------
        # SELL — price enters supply zone
        # ----------------------------------------------------------------
        elif float(row.get("in_supply_zone", 0) or 0) == 1.0:

            if use_htf:
                htf_s_top = _f(row.get("htf_supply_zone_top"))
                sl   = (htf_s_top + sl_buffer_atr * atr) if not np.isnan(htf_s_top) \
                       else (close + 4.0 * atr)
                risk = sl - close  # compute risk before TP so fallback can guarantee min RR

                htf_d_top = _f(row.get("htf_demand_zone_top"))
                if not np.isnan(htf_d_top) and htf_d_top < close:
                    tp = close - (close - htf_d_top) * (0.5 if use_midline_tp else 1.0)
                else:
                    tp = close - max(min_rr * risk, 2.0 * atr)
            else:
                s_top = _f(row.get("supply_zone_top"))
                d_top = _f(row.get("demand_zone_top"))
                if np.isnan(s_top):
                    continue
                sl   = s_top + sl_buffer_atr * atr
                risk = sl - close  # compute risk before TP so fallback can guarantee min RR
                tp   = (close - (close - d_top) * (0.5 if use_midline_tp else 1.0)) \
                       if not np.isnan(d_top) and d_top < close \
                       else close - max(min_rr * risk, 3.0 * atr)

            reward = close - tp
            if risk <= 0 or reward <= 0:
                continue
            rr = reward / risk
            if rr < min_rr:
                continue
            if max_rr is not None and rr > max_rr:
                continue
            if reward / atr > MAX_TP_ATR:
                continue
            if risk / atr > MAX_SL_ATR:
                continue
            if min_sl_atr > 0 and risk / atr < min_sl_atr:
                continue

            signal = -1
            reason = f"supply_{'htf' if use_htf else 'ltf'} rr={rr:.2f}"
            df.at[i, "rr_ratio"] = float(rr)

        if signal == 0:
            continue

        df.at[i, "signal"]           = signal
        df.at[i, "signal_direction"] = signal
        df.at[i, "signal_reason"]    = reason
        df.at[i, "tp_price"]         = float(tp)
        df.at[i, "sl_price"]         = float(sl)

        # ----------------------------------------------------------------
        # Simulate forward price action
        # ----------------------------------------------------------------
        outcome = 0
        future  = df.iloc[i + 1: i + 1 + max_bars]

        for _, fbar in future.iterrows():
            fh = float(fbar["high"])
            fl = float(fbar["low"])
            if signal == 1:       # buy: TP above entry, SL below entry
                if fh >= tp: outcome =  1; break
                if fl <= sl: outcome = -1; break
            else:                 # sell: TP below entry, SL above entry
                if fl <= tp: outcome =  1; break
                if fh >= sl: outcome = -1; break

        df.at[i, "trade_outcome"] = outcome

        # FIXED: binary label — 1=winner regardless of direction
        # Old code: label = signal if outcome == 1 else 0
        #   → sell wins became label=-1, invisible to XGBoost's {0,1} classes
        #   → sell_wins=0 in training, model never learned sells
        # New code: label = 1 if winner, 0 if loser — direction lives in features
        df.at[i, "label"] = 1 if outcome == 1 else 0

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
    signals   = (df["signal"] != 0).sum()
    buy_sigs  = (df["signal"] ==  1).sum()
    sell_sigs = (df["signal"] == -1).sum()
    winners   = (df["label"]  ==  1).sum()
    tp_hits   = (df["trade_outcome"] ==  1).sum()
    sl_hits   = (df["trade_outcome"] == -1).sum()

    # Direction breakdown using signal_direction (not label)
    buy_wins  = ((df["label"] == 1) & (df["signal_direction"] ==  1)).sum()
    sell_wins = ((df["label"] == 1) & (df["signal_direction"] == -1)).sum()

    win_rate = tp_hits / max(signals, 1) * 100
    tf_tag   = f"{timeframe} " if timeframe else ""
    logger.info(
        f"{tf_tag}Labels | signals={signals} "
        f"(buys={buy_sigs} sells={sell_sigs}) | "
        f"winners={winners} (buy_wins={buy_wins} sell_wins={sell_wins}) | "
        f"win_rate={win_rate:.1f}% TP={tp_hits} SL={sl_hits}"
    )


def get_class_weights(df: pd.DataFrame) -> dict:
    from collections import Counter
    counts = Counter(df["label"])
    total  = len(df)
    return {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}