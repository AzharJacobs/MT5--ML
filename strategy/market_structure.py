"""
Market Structure Trading Plan — Backtest Signal Logic
======================================================
Strictly implements the rules from the PDF for backtesting only.

Sequence: BOS → Pullback into zone → 15M BOS confirmation → Entry

Playbook A (what we test):
  1. Detect 4H swing structure: HH/HL = bullish, LL/LH = bearish
  2. BOS only valid on candle CLOSE beyond prior swing (no wicks)
  3. Strong Swing Low = lowest CLOSE before the impulse (bullish)
     Strong Swing High = highest CLOSE before the impulse (bearish)
  4. Zone = price range between BOS candle and the Strong Swing point
  5. Zone active until price closes back through Strong Swing point
  6. Zone tapped when price retraces into the zone
  7. 15M BOS in same direction while zone is tapped = entry signal
  8. SL = below 15M Strong Swing Low | TP = 4H Weak Swing High
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SWING_LOOKBACK   = 10    # bars each side to confirm a swing point (window = 2*n+1)
H4_BARS          = 150   # needs 4+ swings × 21 bars each → 84 bars minimum; 150 gives headroom
M15_BARS         = 60    # how many 15M bars to load for execution analysis
ZONE_VISIT_BARS  = 20    # how many recent 15M bars to check for zone tap
ZONE_TOL         = 0.002 # ±0.2% tolerance on zone boundaries
SL_BUFFER_PCT    = 0.0005  # small buffer beyond 15M swing point for SL


# ---------------------------------------------------------------------------
# Step 1 — Swing detection with HH/HL/LH/LL labeling
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, n: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Detect swing highs and swing lows, label each as HH/LH or HL/LL.

    A swing high at bar i: high[i] is strictly the maximum in [i-n, i+n].
    A swing low  at bar i: low[i]  is strictly the minimum in [i-n, i+n].
    BOS rule (close-based) is applied separately; this just finds the pivots.

    Returns df with added columns:
      swing_high  — float (the high price) or NaN
      swing_low   — float (the low price) or NaN
      swing_close_high — close at swing high bar (for Strong Swing use)
      swing_close_low  — close at swing low bar
      swing_high_label — 'HH' | 'LH' | None
      swing_low_label  — 'HL' | 'LL' | None
    """
    df = df.copy().reset_index(drop=True)
    length = len(df)

    sh_price  = [np.nan] * length
    sl_price  = [np.nan] * length
    sh_close  = [np.nan] * length
    sl_close  = [np.nan] * length
    sh_label  = [None]   * length
    sl_label  = [None]   * length

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    prev_sh = None   # previous confirmed swing high price
    prev_sl = None   # previous confirmed swing low price

    for i in range(n, length - n):
        window_highs = highs[i - n: i + n + 1]
        window_lows  = lows[i  - n: i + n + 1]

        # Swing high: strict max in window and higher than immediate neighbours
        if (highs[i] == window_highs.max()
                and highs[i] > highs[i - 1]
                and highs[i] > highs[i + 1]):
            sh_price[i] = highs[i]
            sh_close[i] = closes[i]
            if prev_sh is None:
                sh_label[i] = "HH"  # first swing — label as HH by convention
            elif highs[i] > prev_sh:
                sh_label[i] = "HH"
            else:
                sh_label[i] = "LH"
            prev_sh = highs[i]

        # Swing low: strict min in window and lower than immediate neighbours
        if (lows[i] == window_lows.min()
                and lows[i] < lows[i - 1]
                and lows[i] < lows[i + 1]):
            sl_price[i] = lows[i]
            sl_close[i] = closes[i]
            if prev_sl is None:
                sl_label[i] = "HL"  # first swing — label as HL by convention
            elif lows[i] > prev_sl:
                sl_label[i] = "HL"
            else:
                sl_label[i] = "LL"
            prev_sl = lows[i]

    df["swing_high"]       = sh_price
    df["swing_low"]        = sl_price
    df["swing_close_high"] = sh_close
    df["swing_close_low"]  = sl_close
    df["swing_high_label"] = sh_label
    df["swing_low_label"]  = sl_label
    return df


# ---------------------------------------------------------------------------
# Step 2+3+4+5 — 4H structure: bias, BOS, zone, strong/weak swings
# ---------------------------------------------------------------------------

def analyse_4h_structure(df_4h: pd.DataFrame, n: int = SWING_LOOKBACK) -> dict:
    """
    Full 4H structure analysis. Returns a dict with:
      bias         — 'bullish' | 'bearish' | 'neutral'
      zone         — (zone_low, zone_high) or None
      zone_active  — bool: zone not yet invalidated by a close through Strong Swing
      zone_tapped  — bool: price has visited the zone (set by caller with 15M data)
      strong_swing — price of the Strong Swing point (invalidation level)
      weak_swing   — price of the Weak Swing target (TP level)
      bos_close    — close price of the BOS candle
    """
    result = {
        "bias":        "neutral",
        "zone":        None,
        "zone_active": False,
        "zone_tapped": False,
        "strong_swing": None,
        "weak_swing":   None,
        "bos_close":    None,
    }

    if df_4h is None or len(df_4h) < n * 2 + 4:
        return result

    df = detect_swings(df_4h, n=n)

    # Collect labeled swings in order
    swing_highs = [
        (i, row["swing_high"], row["swing_close_high"], row["swing_high_label"])
        for i, row in df.iterrows()
        if not np.isnan(row["swing_high"])
    ]
    swing_lows = [
        (i, row["swing_low"], row["swing_close_low"], row["swing_low_label"])
        for i, row in df.iterrows()
        if not np.isnan(row["swing_low"])
    ]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    # ---- Determine bias from last two swing highs and lows ----
    last_sh_label = swing_highs[-1][3]
    last_sl_label = swing_lows[-1][3]

    bullish = (last_sh_label == "HH" and last_sl_label == "HL")
    bearish = (last_sh_label == "LH" and last_sl_label == "LL")

    if not bullish and not bearish:
        return result

    closes = df["close"].values

    # ---- Confirm BOS on CLOSE (no wicks) ----
    if bullish:
        # BOS up: a 4H bar whose close > prior swing high price
        prior_sh_price = swing_highs[-2][1]   # second-to-last swing high
        bos_idx = None
        for i in range(int(swing_highs[-2][0]) + 1, len(df)):
            if closes[i] > prior_sh_price:
                bos_idx = i
                break
        if bos_idx is None:
            return result

        bos_close = closes[bos_idx]

        # Strong Swing Low = lowest CLOSE in bars between prior swing low and BOS candle
        prior_sl_idx = int(swing_lows[-2][0])
        segment_closes = closes[prior_sl_idx: bos_idx + 1]
        if len(segment_closes) == 0:
            return result
        strong_swing_price = float(segment_closes.min())

        # Weak Swing High = most recent swing high (the target liquidity level)
        weak_swing_price = swing_highs[-1][1]

        # Zone = from Strong Swing Low close to BOS candle close
        zone_low  = min(strong_swing_price, bos_close)
        zone_high = max(strong_swing_price, bos_close)

        # Zone is active if last 4H close has not gone back below Strong Swing Low
        last_close_4h = closes[-1]
        zone_active = last_close_4h > strong_swing_price

        result.update({
            "bias":         "bullish",
            "zone":         (zone_low, zone_high),
            "zone_active":  zone_active,
            "strong_swing": strong_swing_price,
            "weak_swing":   weak_swing_price,
            "bos_close":    bos_close,
        })

    else:  # bearish
        # BOS down: a 4H bar whose close < prior swing low price
        prior_sl_price = swing_lows[-2][1]
        bos_idx = None
        for i in range(int(swing_lows[-2][0]) + 1, len(df)):
            if closes[i] < prior_sl_price:
                bos_idx = i
                break
        if bos_idx is None:
            return result

        bos_close = closes[bos_idx]

        # Strong Swing High = highest CLOSE between prior swing high and BOS candle
        prior_sh_idx = int(swing_highs[-2][0])
        segment_closes = closes[prior_sh_idx: bos_idx + 1]
        if len(segment_closes) == 0:
            return result
        strong_swing_price = float(segment_closes.max())

        # Weak Swing Low = most recent swing low (TP target)
        weak_swing_price = swing_lows[-1][1]

        # Zone = from BOS candle close to Strong Swing High close
        zone_low  = min(strong_swing_price, bos_close)
        zone_high = max(strong_swing_price, bos_close)

        last_close_4h = closes[-1]
        zone_active = last_close_4h < strong_swing_price

        result.update({
            "bias":         "bearish",
            "zone":         (zone_low, zone_high),
            "zone_active":  zone_active,
            "strong_swing": strong_swing_price,
            "weak_swing":   weak_swing_price,
            "bos_close":    bos_close,
        })

    return result


# ---------------------------------------------------------------------------
# Step 5+6 — Zone tap check using 15M candles
# ---------------------------------------------------------------------------

def zone_tapped_recently(
    df_15m: pd.DataFrame,
    zone: Tuple[float, float],
    lookback: int = ZONE_VISIT_BARS,
    tol: float = ZONE_TOL,
) -> bool:
    """
    True if any of the last `lookback` 15M candles touched or overlapped the zone.
    Checks both low and high so we catch candles that wick through the zone.
    """
    recent = df_15m.tail(lookback)
    z_low  = zone[0] * (1 - tol)
    z_high = zone[1] * (1 + tol)
    touched = (recent["low"] <= z_high) & (recent["high"] >= z_low)
    return bool(touched.any())


# ---------------------------------------------------------------------------
# Step 4 — 15M BOS detection and SL anchor
# ---------------------------------------------------------------------------

def analyse_15m_structure(
    df_15m: pd.DataFrame,
    direction: str,
    n: int = SWING_LOOKBACK,
) -> dict:
    """
    Detect 15M BOS in `direction` ('bullish' or 'bearish') and return the
    15M Strong Swing point to use as SL anchor.

    Returns:
      bos_confirmed — bool
      sl_anchor     — float | None (15M strong swing low for buys, high for sells)
    """
    result = {"bos_confirmed": False, "sl_anchor": None}

    if df_15m is None or len(df_15m) < n * 2 + 4:
        return result

    df     = detect_swings(df_15m, n=n)
    closes = df["close"].values

    if direction == "bullish":
        # Find swing highs on 15M
        sh_rows = df[~df["swing_high"].isna()]
        if sh_rows.empty:
            return result
        prior_sh_price = float(sh_rows["swing_high"].iloc[-1])
        last_close     = closes[-1]
        # BOS up: 15M close breaks above the most recent 15M swing high
        if last_close > prior_sh_price:
            result["bos_confirmed"] = True
            # SL anchor = most recent 15M swing low (the Strong Swing Low on 15M)
            sl_rows = df[~df["swing_low"].isna()]
            if not sl_rows.empty:
                result["sl_anchor"] = float(sl_rows["swing_low"].iloc[-1])

    else:  # bearish
        sl_rows = df[~df["swing_low"].isna()]
        if sl_rows.empty:
            return result
        prior_sl_price = float(sl_rows["swing_low"].iloc[-1])
        last_close     = closes[-1]
        # BOS down: 15M close breaks below the most recent 15M swing low
        if last_close < prior_sl_price:
            result["bos_confirmed"] = True
            sh_rows = df[~df["swing_high"].isna()]
            if not sh_rows.empty:
                result["sl_anchor"] = float(sh_rows["swing_high"].iloc[-1])

    return result


# ---------------------------------------------------------------------------
# Main signal function
# ---------------------------------------------------------------------------

def apply_market_structure_signal(
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    min_rr: float = 1.5,
) -> dict:
    """
    Playbook A — Trend Continuation signal.

    Checks (in order):
      1. 4H structure is bullish/bearish with a confirmed close-based BOS
      2. Zone is still active (price hasn't closed through Strong Swing point)
      3. Price has tapped (visited) the zone in the last ZONE_VISIT_BARS 15M bars
      4. 15M forms a BOS in the same direction (close-based)
      5. SL below/above 15M Strong Swing, TP at 4H Weak Swing — RR >= min_rr

    Returns dict: {signal, sl, tp, reason, structure}
    """
    neutral = {"signal": "neutral", "sl": None, "tp": None, "reason": "", "structure": {}}

    if df_4h is None or df_15m is None or df_4h.empty or df_15m.empty:
        return {**neutral, "reason": "missing data"}

    # Step 1–5: 4H analysis
    s = analyse_4h_structure(df_4h)

    if s["bias"] == "neutral":
        return {**neutral, "reason": "4H bias neutral", "structure": s}

    if not s["zone_active"]:
        return {**neutral, "reason": "4H zone invalidated (price closed through strong swing)", "structure": s}

    zone = s["zone"]

    # Step 6: zone tapped check (using 15M candles)
    if not zone_tapped_recently(df_15m, zone):
        return {**neutral, "reason": "zone not tapped recently", "structure": s}

    # Step 4: 15M BOS confirmation
    m15 = analyse_15m_structure(df_15m, direction=s["bias"])
    if not m15["bos_confirmed"]:
        return {**neutral, "reason": f"no 15M {s['bias']} BOS", "structure": s}

    sl_anchor = m15["sl_anchor"]
    if sl_anchor is None:
        return {**neutral, "reason": "no 15M swing point for SL", "structure": s}

    price    = float(df_15m["close"].iloc[-1])
    weak_tp  = s["weak_swing"]

    if weak_tp is None:
        return {**neutral, "reason": "no 4H weak swing for TP", "structure": s}

    # Step 5: geometry + RR
    if s["bias"] == "bullish":
        sl     = sl_anchor * (1 - SL_BUFFER_PCT)
        tp     = weak_tp
        if sl >= price:
            return {**neutral, "reason": "SL at or above entry", "structure": s}
        if tp <= price:
            return {**neutral, "reason": "TP at or below entry", "structure": s}
        risk   = price - sl
        reward = tp - price

    else:  # bearish
        sl     = sl_anchor * (1 + SL_BUFFER_PCT)
        tp     = weak_tp
        if sl <= price:
            return {**neutral, "reason": "SL at or below entry", "structure": s}
        if tp >= price:
            return {**neutral, "reason": "TP at or above entry", "structure": s}
        risk   = sl - price
        reward = price - tp

    if risk <= 0:
        return {**neutral, "reason": "zero risk", "structure": s}

    rr = reward / risk
    if rr < min_rr:
        return {**neutral, "reason": f"RR {rr:.2f} below min {min_rr}", "structure": s}

    direction_label = "buy" if s["bias"] == "bullish" else "sell"
    return {
        "signal":    direction_label,
        "sl":        round(sl, 5),
        "tp":        round(tp, 5),
        "reason":    (
            f"4H {s['bias']} | zone {zone[0]:.2f}-{zone[1]:.2f} tapped | "
            f"15M BOS {s['bias']} | RR={rr:.2f}"
        ),
        "structure": s,
    }
