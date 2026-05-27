"""
Market Structure Trading Plan — Backtest Signal Logic
======================================================
Strictly implements the rules from the PDF for backtesting only.

Playbook A — Trend Continuation:
  Sequence: 4H BOS → pullback into 4H zone → 15M BOS same direction → entry
  1. Detect 4H swing structure: HH/HL = bullish, LL/LH = bearish
  2. BOS only valid on candle CLOSE beyond prior swing (no wicks)
  3. Strong Swing Low = candle with the lowest LOW before the impulse (bullish)
     Strong Swing High = candle with the highest HIGH before the impulse (bearish)
  4. Zone = body range (open/close, no wicks) from Strong Swing candle to BOS candle
  5. Zone active until price closes back through Strong Swing point
  6. Zone tapped when price retraces into the zone
  7. 15M BOS in same direction while zone is tapped = entry signal
  8. SL = beyond 15M Strong Swing | TP = first post-BOS 4H Weak Swing

Playbook C — Counter-Trend Fade into 4H Zone:
  Sequence: 4H macro bias confirmed → 15M turns counter-trend → ride 15M
            move back toward the 4H demand/supply zone → TP at zone edge
  1. Same 4H structure gate as Playbook A
  2. 4H zone active and NOT yet tapped (zone is the profit target)
  3. 4H internal structure still intact post-BOS
  4. 15M has confirmed a close-based BOS in the OPPOSITE direction
  5. Trade follows 15M direction (counter to 4H):
       4H bullish + 15M bearish → SELL  (ride pullback to 4H demand zone)
       4H bearish + 15M bullish → BUY   (ride bounce to 4H supply zone)
  6. SL beyond the 15M strong swing that formed when 15M turned counter-trend
     TP = near edge of the 4H zone (zone[1] for sells, zone[0] for buys)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SWING_LOOKBACK   = 3     # bars each side to confirm a swing point (window = 2*n+1)
ZONE_WIDE        = False  # False = origin candle body only (tighter, better backtest results)
H4_BARS          = 150   # needs 4+ swings × 21 bars each → 84 bars minimum; 150 gives headroom
M15_BARS         = 60    # how many 15M bars to load for execution analysis
ZONE_VISIT_BARS  = 20    # how many recent 15M bars to check for zone tap
ZONE_TOL         = 0.002 # ±0.2% tolerance on zone boundaries
SL_BUFFER_PCT    = 0.0005  # small buffer beyond 15M swing point for SL
ATR_PERIOD       = 14      # lookback for ATR calculation
BOS_ATR_MULT     = 0.5     # BOS impulse must be >= this many ATRs to be valid
MIN_SL_ATR_MULT  = 0.5     # minimum SL distance as multiple of 15M ATR (filters junk entries)
M15_BOS_MAX_AGE  = 20      # 15M BOS must be within this many bars to be considered fresh


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
# ATR helper
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Average True Range over the last `period` bars of df."""
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    prev_c = np.empty_like(closes)
    prev_c[0]  = closes[0]
    prev_c[1:] = closes[:-1]
    tr   = np.maximum(highs - lows,
           np.maximum(np.abs(highs - prev_c),
                      np.abs(lows  - prev_c)))
    tail = tr[-period:] if len(tr) >= period else tr
    return float(tail.mean()) if len(tail) > 0 else 0.0


# ---------------------------------------------------------------------------
# Zone definition helper
# ---------------------------------------------------------------------------

def detect_zone(df: pd.DataFrame, origin_idx: int, bos_idx: int, wide: bool = True) -> Tuple[float, float]:
    """
    wide=True  (default): full body range across the impulse (origin → BOS) — PDF spec.
    wide=False           : body of the origin candle only — legacy single-candle mode.
    """
    if wide:
        segment = df.iloc[origin_idx: bos_idx + 1]
        bodies_lo = segment[["open", "close"]].min(axis=1)
        bodies_hi = segment[["open", "close"]].max(axis=1)
        return (float(bodies_lo.min()), float(bodies_hi.max()))
    else:
        row = df.iloc[origin_idx]
        lo = float(min(row["open"], row["close"]))
        hi = float(max(row["open"], row["close"]))
        return (lo, hi)


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
        "bos_idx":      None,
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

        # Strong Swing Low = candle with the lowest LOW between prior swing low and BOS candle
        prior_sl_idx = int(swing_lows[-2][0])
        lows_arr = df["low"].values
        segment_lows = lows_arr[prior_sl_idx: bos_idx + 1]
        if len(segment_lows) == 0:
            return result
        origin_idx = int(np.argmin(segment_lows)) + prior_sl_idx
        strong_swing_price = float(lows_arr[origin_idx])

        # Weak Swing = first swing HIGH formed after the BOS candle (post-BOS retracement)
        post_bos_sh = [
            (i, row["swing_high"]) for i, row in df.iterrows()
            if i > bos_idx and not np.isnan(row["swing_high"])
        ]
        weak_swing_price = post_bos_sh[0][1] if post_bos_sh else None

        # Zone = body range from Strong Swing candle through BOS candle
        zone_low, zone_high = detect_zone(df, origin_idx, bos_idx, wide=ZONE_WIDE)

        # ATR impulse strength check — reject weak BOS
        atr = calc_atr(df.iloc[:bos_idx + 1])
        if atr > 0 and (bos_close - strong_swing_price) < BOS_ATR_MULT * atr:
            return result

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
            "bos_idx":      bos_idx,
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

        # Strong Swing High = candle with the highest HIGH between prior swing high and BOS candle
        prior_sh_idx = int(swing_highs[-2][0])
        highs_arr = df["high"].values
        segment_highs = highs_arr[prior_sh_idx: bos_idx + 1]
        if len(segment_highs) == 0:
            return result
        origin_idx = int(np.argmax(segment_highs)) + prior_sh_idx
        strong_swing_price = float(highs_arr[origin_idx])

        # Weak Swing = first swing LOW formed after the BOS candle (post-BOS retracement)
        post_bos_sl = [
            (i, row["swing_low"]) for i, row in df.iterrows()
            if i > bos_idx and not np.isnan(row["swing_low"])
        ]
        weak_swing_price = post_bos_sl[0][1] if post_bos_sl else None

        # Zone = body range from Strong Swing candle through BOS candle
        zone_low, zone_high = detect_zone(df, origin_idx, bos_idx, wide=ZONE_WIDE)

        # ATR impulse strength check — reject weak BOS
        atr = calc_atr(df.iloc[:bos_idx + 1])
        if atr > 0 and (strong_swing_price - bos_close) < BOS_ATR_MULT * atr:
            return result

        last_close_4h = closes[-1]
        zone_active = last_close_4h < strong_swing_price

        result.update({
            "bias":         "bearish",
            "zone":         (zone_low, zone_high),
            "zone_active":  zone_active,
            "strong_swing": strong_swing_price,
            "weak_swing":   weak_swing_price,
            "bos_close":    bos_close,
            "bos_idx":      bos_idx,
        })

    return result


# ---------------------------------------------------------------------------
# 4H internal structure — post-BOS sub-structure check
# ---------------------------------------------------------------------------

def analyse_4h_internal(
    df_4h: pd.DataFrame,
    bos_idx: int,
    direction: str,
    n: int = SWING_LOOKBACK,
) -> dict:
    """
    Examine 4H bars AFTER the BOS candle for internal swing progression.
    Returns valid=False when two confirmed swings exist and the second breaks
    the prior swing against the trade direction (LH after bullish BOS, or
    HL after bearish BOS), signalling the 4H trend has already reversed.
    Returns valid=True when: fewer than two swings have formed (early setup),
    or the confirmed swings are still progressing with the bias.

    Returns:
      valid              — bool
      last_internal_swing — float | None  (latest internal swing in direction)
    """
    result: dict = {"valid": True, "last_internal_swing": None}

    # detect_swings resets the index, so iloc[bos_idx + 1:] is safe
    post = pd.DataFrame(df_4h).iloc[bos_idx + 1:].copy().reset_index(drop=True)
    if len(post) < n * 2 + 3:
        return result  # not enough post-BOS bars yet — don't block

    df_post = detect_swings(post, n=n)

    if direction == "bullish":
        sh_list = df_post.loc[~df_post["swing_high"].isna(), "swing_high"].tolist()
        if len(sh_list) >= 2:
            if sh_list[-1] < sh_list[-2]:   # LH formed → internal structure turned bearish
                result["valid"] = False
                return result
            result["last_internal_swing"] = sh_list[-1]
        elif sh_list:
            result["last_internal_swing"] = sh_list[-1]

    else:  # bearish
        sl_list = df_post.loc[~df_post["swing_low"].isna(), "swing_low"].tolist()
        if len(sl_list) >= 2:
            if sl_list[-1] > sl_list[-2]:   # HL formed → internal structure turned bullish
                result["valid"] = False
                return result
            result["last_internal_swing"] = sl_list[-1]
        elif sl_list:
            result["last_internal_swing"] = sl_list[-1]

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
# Step 4 — Full 15M internal structure map
# ---------------------------------------------------------------------------

def analyse_15m_structure(
    df_15m: pd.DataFrame,
    direction: str,
    n: int = SWING_LOOKBACK,
) -> dict:
    """
    Full 15M internal structure map mirroring the 4H analysis.

    Finds the most recent 15M close-based BOS in `direction`, identifies the
    15M Strong Swing origin candle (actual low/high), Weak Swing (first
    post-BOS swing), 15M zone (origin candle body), and the overall 15M
    internal bias from HH/HL or LL/LH labeling.

    Returns:
      bos_confirmed    — bool
      bos_idx          — int | None
      sl_anchor        — float | None  (actual low/high of 15M origin candle)
      strong_swing     — float | None  (same as sl_anchor)
      weak_swing_15m   — float | None  (first post-BOS swing in direction)
      zone_15m         — (lo, hi) | None  (body of 15M origin candle)
      internal_bias    — 'bullish' | 'bearish' | 'neutral'
    """
    result: dict = {
        "bos_confirmed":  False,
        "bos_idx":        None,
        "sl_anchor":      None,
        "strong_swing":   None,
        "weak_swing_15m": None,
        "zone_15m":       None,
        "internal_bias":  "neutral",
        "post_bos_intact": True,
    }

    if df_15m is None or len(df_15m) < n * 2 + 4:
        return result

    df     = detect_swings(df_15m, n=n)
    closes = df["close"].values
    lows   = df["low"].values
    highs  = df["high"].values

    swing_highs = [
        (int(i), float(row["swing_high"]), row["swing_high_label"])
        for i, row in df.iterrows()
        if not np.isnan(row["swing_high"])
    ]
    swing_lows = [
        (int(i), float(row["swing_low"]), row["swing_low_label"])
        for i, row in df.iterrows()
        if not np.isnan(row["swing_low"])
    ]

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return result

    # Internal bias from 15M swing labels
    last_sh_lbl = swing_highs[-1][2]
    last_sl_lbl = swing_lows[-1][2]
    if last_sh_lbl == "HH" and last_sl_lbl == "HL":
        result["internal_bias"] = "bullish"
    elif last_sh_lbl == "LH" and last_sl_lbl == "LL":
        result["internal_bias"] = "bearish"

    if direction == "bullish":
        prior_sh_price = swing_highs[-2][1]
        search_start   = swing_highs[-2][0] + 1
        bos_idx = None
        for i in range(search_start, len(df)):
            if closes[i] > prior_sh_price:
                bos_idx = i
                break
        if bos_idx is None:
            return result

        # Strong Swing: candle with lowest LOW between last pre-BOS swing low and BOS
        pre_bos_sl = [s for s in swing_lows if s[0] < bos_idx]
        if not pre_bos_sl:
            return result
        anchor_idx   = pre_bos_sl[-1][0]
        seg_lows     = lows[anchor_idx: bos_idx + 1]
        origin_idx   = int(np.argmin(seg_lows)) + anchor_idx
        strong_swing = float(lows[origin_idx])
        zone_15m     = detect_zone(df, origin_idx, bos_idx, wide=ZONE_WIDE)

        # Weak Swing: first swing HIGH after BOS
        post_bos_sh    = [(i, p) for (i, p, _) in swing_highs if i > bos_idx]
        weak_swing_15m = post_bos_sh[0][1] if post_bos_sh else None

        # Fix 3: post-BOS structure check — LH after bullish BOS means trend is reversing
        lh_after_bos    = [s for s in swing_highs if s[0] > bos_idx and s[2] == "LH"]
        post_bos_intact = len(lh_after_bos) == 0

        result.update({
            "bos_confirmed":   True,
            "bos_idx":         bos_idx,
            "sl_anchor":       strong_swing,
            "strong_swing":    strong_swing,
            "weak_swing_15m":  weak_swing_15m,
            "zone_15m":        zone_15m,
            "post_bos_intact": post_bos_intact,
        })

    else:  # bearish
        prior_sl_price = swing_lows[-2][1]
        search_start   = swing_lows[-2][0] + 1
        bos_idx = None
        for i in range(search_start, len(df)):
            if closes[i] < prior_sl_price:
                bos_idx = i
                break
        if bos_idx is None:
            return result

        # Strong Swing: candle with highest HIGH between last pre-BOS swing high and BOS
        pre_bos_sh = [s for s in swing_highs if s[0] < bos_idx]
        if not pre_bos_sh:
            return result
        anchor_idx   = pre_bos_sh[-1][0]
        seg_highs    = highs[anchor_idx: bos_idx + 1]
        origin_idx   = int(np.argmax(seg_highs)) + anchor_idx
        strong_swing = float(highs[origin_idx])
        zone_15m     = detect_zone(df, origin_idx, bos_idx, wide=ZONE_WIDE)

        # Weak Swing: first swing LOW after BOS
        post_bos_sl    = [(i, p) for (i, p, _) in swing_lows if i > bos_idx]
        weak_swing_15m = post_bos_sl[0][1] if post_bos_sl else None

        # Fix 3: post-BOS structure check — HL after bearish BOS means trend is reversing
        hl_after_bos    = [s for s in swing_lows if s[0] > bos_idx and s[2] == "HL"]
        post_bos_intact = len(hl_after_bos) == 0

        result.update({
            "bos_confirmed":   True,
            "bos_idx":         bos_idx,
            "sl_anchor":       strong_swing,
            "strong_swing":    strong_swing,
            "weak_swing_15m":  weak_swing_15m,
            "zone_15m":        zone_15m,
            "post_bos_intact": post_bos_intact,
        })

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
    neutral = {"signal": "neutral", "sl": None, "tp": None, "reason": "", "structure": {}, "playbook": "A"}

    if df_4h is None or df_15m is None or df_4h.empty or df_15m.empty:
        return {**neutral, "reason": "missing data"}

    # Step 1: 4H structure (includes ATR impulse gate)
    s = analyse_4h_structure(df_4h)

    if s["bias"] == "neutral":
        return {**neutral, "reason": "4H bias neutral", "structure": s}

    if not s["zone_active"]:
        return {**neutral, "reason": "4H zone invalidated (price closed through strong swing)", "structure": s}

    # Step 2: 4H internal structure — confirm post-BOS trend intact
    s4h_int = analyse_4h_internal(df_4h, s["bos_idx"], s["bias"])
    if not s4h_int["valid"]:
        return {**neutral, "reason": "4H internal structure broken post-BOS", "structure": {**s, "4h_internal": s4h_int}}

    zone = s["zone"]

    # Step 3: zone tapped check (using 15M candles)
    if not zone_tapped_recently(df_15m, zone):
        return {**neutral, "reason": "zone not tapped recently", "structure": {**s, "4h_internal": s4h_int}}

    # Step 4: full 15M internal structure map
    m15 = analyse_15m_structure(df_15m, direction=s["bias"])
    if not m15["bos_confirmed"]:
        return {**neutral, "reason": f"no 15M {s['bias']} BOS", "structure": {**s, "4h_internal": s4h_int, "15m": m15}}

    # Fix 2: 15M internal bias must agree with 4H direction
    if m15["internal_bias"] != s["bias"]:
        return {**neutral, "reason": f"15M bias {m15['internal_bias']} conflicts with 4H {s['bias']}", "structure": {**s, "4h_internal": s4h_int, "15m": m15}}

    # Fix 3: 15M post-BOS structure must still be intact
    if not m15.get("post_bos_intact", True):
        return {**neutral, "reason": "15M post-BOS structure broken", "structure": {**s, "4h_internal": s4h_int, "15m": m15}}

    sl_anchor = m15["sl_anchor"]
    if sl_anchor is None:
        return {**neutral, "reason": "no 15M swing point for SL", "structure": {**s, "4h_internal": s4h_int, "15m": m15}}

    price    = float(df_15m["close"].iloc[-1])
    weak_tp  = s["weak_swing"]

    combined = {**s, "4h_internal": s4h_int, "15m": m15}

    if weak_tp is None:
        return {**neutral, "reason": "no 4H weak swing for TP", "structure": combined}

    # Step 5: geometry + RR
    if s["bias"] == "bullish":
        sl     = sl_anchor * (1 - SL_BUFFER_PCT)
        tp     = weak_tp
        if sl >= price:
            return {**neutral, "reason": "SL at or above entry", "structure": combined}
        if tp <= price:
            return {**neutral, "reason": "TP at or below entry", "structure": combined}
        risk   = price - sl
        reward = tp - price

    else:  # bearish
        sl     = sl_anchor * (1 + SL_BUFFER_PCT)
        tp     = weak_tp
        if sl <= price:
            return {**neutral, "reason": "SL at or below entry", "structure": combined}
        if tp >= price:
            return {**neutral, "reason": "TP at or above entry", "structure": combined}
        risk   = sl - price
        reward = price - tp

    if risk <= 0:
        return {**neutral, "reason": "zero risk", "structure": combined}

    # SL too tight: reject if risk < 0.5× 15M ATR (noise will instantly hit SL)
    atr_15m = calc_atr(df_15m)
    if atr_15m > 0 and risk < MIN_SL_ATR_MULT * atr_15m:
        return {**neutral, "reason": f"SL too tight ({risk:.1f} < {MIN_SL_ATR_MULT}×ATR {atr_15m:.1f})", "structure": combined}

    # Stale BOS: reject if 15M BOS formed more than M15_BOS_MAX_AGE bars ago
    m15_bos_age = (len(df_15m) - 1) - m15["bos_idx"]
    if m15_bos_age > M15_BOS_MAX_AGE:
        return {**neutral, "reason": f"15M BOS stale ({m15_bos_age} bars ago)", "structure": combined}

    rr = reward / risk
    if rr < min_rr:
        return {**neutral, "reason": f"RR {rr:.2f} below min {min_rr}", "structure": combined}

    direction_label = "buy" if s["bias"] == "bullish" else "sell"
    return {
        "signal":    direction_label,
        "sl":        round(sl, 5),
        "tp":        round(tp, 5),
        "reason":    (
            f"4H {s['bias']} | zone {zone[0]:.2f}-{zone[1]:.2f} tapped | "
            f"15M BOS {s['bias']} | 15M bias {m15['internal_bias']} | RR={rr:.2f}"
        ),
        "structure": combined,
        "playbook":  "A",
    }


# ---------------------------------------------------------------------------
# Playbook C — Counter-Trend Fade into 4H Zone
# ---------------------------------------------------------------------------

def apply_playbook_c_signal(
    df_15m: pd.DataFrame,
    df_4h: pd.DataFrame,
    min_rr: float = 1.5,
) -> dict:
    """
    Playbook C — Counter-Trend Fade into 4H Zone.

    The 4H macro trend is confirmed. The 15M has turned OPPOSITE to the 4H
    bias, meaning price is currently pulling back (4H bullish) or bouncing
    (4H bearish) toward the 4H demand/supply zone. We trade the 15M move
    toward that zone, using the zone edge as TP.

    Checks (in order):
      1. 4H structure confirmed with ATR-valid BOS (same gate as Playbook A)
      2. 4H zone still active
      3. 4H zone NOT yet tapped — it is the profit target, not the trigger
      4. 4H internal post-BOS structure still intact
      5. 15M has a close-based BOS in the OPPOSITE direction to 4H
      6. SL beyond the 15M strong swing that caused the counter-trend BOS
         TP = near edge of 4H zone (zone[1] for sells, zone[0] for buys)
         RR >= min_rr

    Returns dict: {signal, sl, tp, reason, structure, playbook}
    """
    neutral = {
        "signal": "neutral", "sl": None, "tp": None,
        "reason": "", "structure": {}, "playbook": "C",
    }

    if df_4h is None or df_15m is None or df_4h.empty or df_15m.empty:
        return {**neutral, "reason": "missing data"}

    # Step 1: 4H structure (includes ATR impulse gate)
    s = analyse_4h_structure(df_4h)
    if s["bias"] == "neutral":
        return {**neutral, "reason": "4H bias neutral", "structure": s}

    if not s["zone_active"]:
        return {**neutral, "reason": "4H zone invalidated", "structure": s}

    zone = s["zone"]
    if zone is None:
        return {**neutral, "reason": "no 4H zone", "structure": s}

    # Step 2: Zone must NOT be tapped yet — it is the TP target
    if zone_tapped_recently(df_15m, zone):
        return {**neutral, "reason": "C: 4H zone already reached (zone is TP target)", "structure": s}

    # Step 3: 4H internal post-BOS structure still intact
    s4h_int = analyse_4h_internal(df_4h, s["bos_idx"], s["bias"])
    if not s4h_int["valid"]:
        return {**neutral, "reason": "4H internal structure broken", "structure": {**s, "4h_internal": s4h_int}}

    # Step 4: 15M must have BOS in the OPPOSITE direction to 4H
    opposite = "bearish" if s["bias"] == "bullish" else "bullish"
    m15 = analyse_15m_structure(df_15m, direction=opposite)
    if not m15["bos_confirmed"]:
        return {**neutral, "reason": f"C: no 15M {opposite} BOS", "structure": {**s, "4h_internal": s4h_int, "15m": m15}}

    sl_anchor = m15["sl_anchor"]
    if sl_anchor is None:
        return {**neutral, "reason": "C: no 15M swing point for SL", "structure": {**s, "4h_internal": s4h_int, "15m": m15}}

    price    = float(df_15m["close"].iloc[-1])
    combined = {**s, "4h_internal": s4h_int, "15m": m15}

    # Step 5: geometry + RR
    # 4H bullish + 15M bearish → SELL; SL above the 15M swing HIGH origin;
    #   TP = zone[1] (top/near edge of 4H demand zone — first zone contact on the way down)
    # 4H bearish + 15M bullish → BUY;  SL below the 15M swing LOW origin;
    #   TP = zone[0] (bottom/near edge of 4H supply zone — first zone contact on the way up)
    if s["bias"] == "bullish":
        side = "sell"
        sl   = sl_anchor * (1 + SL_BUFFER_PCT)
        tp   = zone[1]
        if sl <= price:
            return {**neutral, "reason": "C: SL at or below entry", "structure": combined}
        if tp >= price:
            return {**neutral, "reason": "C: TP at or above entry", "structure": combined}
        risk   = sl - price
        reward = price - tp
    else:
        side = "buy"
        sl   = sl_anchor * (1 - SL_BUFFER_PCT)
        tp   = zone[0]
        if sl >= price:
            return {**neutral, "reason": "C: SL at or above entry", "structure": combined}
        if tp <= price:
            return {**neutral, "reason": "C: TP at or below entry", "structure": combined}
        risk   = price - sl
        reward = tp - price

    if risk <= 0:
        return {**neutral, "reason": "C: zero risk", "structure": combined}

    rr = reward / risk
    if rr < min_rr:
        return {**neutral, "reason": f"C: RR {rr:.2f} below min {min_rr}", "structure": combined}

    return {
        "signal":   side,
        "sl":       round(sl, 5),
        "tp":       round(tp, 5),
        "reason":   (
            f"4H {s['bias']} | 15M {opposite} BOS (counter-trend) | "
            f"zone TP {zone[0]:.2f}-{zone[1]:.2f} | RR={rr:.2f}"
        ),
        "structure": combined,
        "playbook":  "C",
    }
