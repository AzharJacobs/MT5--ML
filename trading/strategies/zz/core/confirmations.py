"""
confirmations.py — Step 3 entry confirmations for zone-based trading.

IMPORTANT SEPARATION FROM STEP 2:
  The M15 directional-close check in timeframe_structure.py (_m15_directional_close)
  is a pre-filter only.  It checks whether *any* recent bar has a directional close
  inside the zone.  It is NOT a confirmation and its result never increments the
  confirmation counter here.  Confirmations are evaluated independently at a
  specific signal bar (idx) and each increments the count on its own.

Zone precondition (enforced by check_all_confirmations, NOT the individual functions):
  The signal bar (or its range) must overlap the H4 zone.
  Individual functions are zone-agnostic so they can be unit-tested freely.

Buy confirmations (demand zone) — 5 signals:
  1. bullish_engulfing   — green body fully covers prior red body
  2. rejection_wick_bull — hammer/pin bar: long lower wick, closes near high
  3. bullish_bos         — close breaks above most recent swing high (continuation)
  4. higher_low          — second confirmed swing low above first in lookback
  5. bullish_choch       — prior M15 structure is bearish (LH/LL), price breaks
                           above the most recent lower high (first flip signal)

Sell confirmations (supply zone) — mirrored:
  1. bearish_engulfing   — red body fully covers prior green body
  2. rejection_wick_bear — shooting star: long upper wick, closes near low
  3. bearish_bos         — close breaks below most recent swing low (continuation)
  4. lower_high          — second confirmed swing high below first in lookback
  5. bearish_choch       — prior M15 structure is bullish (HH/HL), price breaks
                           below the most recent higher low (first flip signal)

BOS vs CHoCH distinction:
  BOS  = continuation: breaks structure in the SAME direction as prevailing trend.
         No trend prerequisite — just checks for a break of the most recent pivot.
  CHoCH = reversal: breaks AGAINST the prevailing local trend. Requires the prior
          M15 structure to be in the opposite direction before the break counts.
  Both are logged separately so CHoCH-tagged trades can be analysed independently.

Stacking: check_all_confirmations fires all five, logs which ones triggered,
and returns a ConfirmationResult with individual flags + total count (0–5).
The caller decides min_confirmations for trade entry.
"""

from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from trading.strategies.zz.core.market_structure import detect_swings
from trading.strategies.zz.core.zones import Zone

# Load swing_structure_Z&Z.py for CHoCH trend detection (& not valid in import names)
_zz_path = pathlib.Path(__file__).parent / "swing_structure.py"
_zz_spec = importlib.util.spec_from_file_location("swing_structure_zz", _zz_path)
_zz_mod  = importlib.util.module_from_spec(_zz_spec)
_zz_spec.loader.exec_module(_zz_mod)
_detect_swings_zz = _zz_mod.detect_swings
_label_structure  = _zz_mod.label_structure


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationConfig:
    # ── Entry gate ────────────────────────────────────────────────────────────
    min_confirmations: int = 1
    # Minimum number of confirmations required to enter.
    # Range 1–4; set to 0 for aggressive boundary entry (zone touch = entry).

    aggressive_boundary: bool = False
    # When True, ignore min_confirmations entirely.
    # Equivalent to min_confirmations=0 but semantically explicit.

    # ── Engulfing ─────────────────────────────────────────────────────────────
    engulf_full_body: bool = True
    # True  → current body must fully cover the prior body (strict).
    # False → partial body overlap is sufficient (more permissive).

    # ── Rejection wick (pin bar / shooting star) ──────────────────────────────
    wick_ratio_min: float = 0.60
    # Dominant wick / total candle range must be >= this.
    # E.g. 0.60 = lower wick is ≥ 60% of range for a hammer.

    body_ratio_max: float = 0.35
    # Body / total candle range must be <= this (mostly wick, small body).

    close_in_upper_half: bool = True
    # Bullish pin bar: require close in the upper 50% of range.
    # Bearish pin bar: require close in the lower 50% of range.

    # ── BOS / MSB ─────────────────────────────────────────────────────────────
    bos_lookback: int = 15
    # Bars back from the signal bar to search for the recent swing high/low to break.

    bos_swing_n: int = 2
    # Bars each side for swing detection in the BOS lookback window.

    # ── Higher low / lower high (structure shift) ─────────────────────────────
    structure_lookback: int = 20
    # Bars back to look for two consecutive confirmed swing lows (HL) or highs (LH).

    structure_swing_n: int = 2
    # Bars each side for swing detection in the structure lookback.

    # ── CHoCH (Change of Character) ───────────────────────────────────────────
    choch_lookback: int = 20
    # Bars back to examine for the prevailing M15 trend before the signal bar.

    choch_swing_left: int = 2
    # Left confirmation window for Z&Z pivots used in CHoCH trend detection.

    choch_swing_right: int = 2
    # Right confirmation window.  No lookahead beyond this many bars.

    choch_min_swings: int = 2
    # Minimum confirmed swing highs (bull CHoCH) or lows (bear CHoCH) required
    # to establish a prevailing trend.  Below this → CHoCH returns False.

    # ── Zone tap tolerance (for price_in_zone precondition) ──────────────────
    zone_tolerance_pct: float = 0.001
    # ±0.1% on zone edges when checking whether bar idx overlaps the zone.

    # ── Signals excluded from the confirmation count ─────────────────────────
    excluded_from_count: List[str] = field(default_factory=list)
    # Signal names here still FIRE and appear in trade logs but do NOT increment
    # the count toward min_confirmations.  Lets you keep a signal for analysis
    # without letting it gate entries on its own.
    # Example: ["rejection_wick"] → pin bar alone won't satisfy min_conf=1,
    # but is still logged when it co-fires with another confirmation.


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ConfirmationResult:
    direction: str          # 'buy' | 'sell'
    price_in_zone: bool     # zone precondition: signal bar range overlaps zone

    # ── Individual confirmation flags ─────────────────────────────────────────
    engulfing: bool         # bullish/bearish engulfing candle
    rejection_wick: bool    # hammer (buy) or shooting star (sell)
    bos_msb: bool           # break of structure in trend direction (continuation)
    structure_shift: bool   # higher low (buy) or lower high (sell)
    choch: bool             # change of character — first counter-trend structure break

    # ── Aggregate ─────────────────────────────────────────────────────────────
    count: int              # total confirmations fired (0–5)
    signals: List[str]      # names of fired confirmations, for logging
    confirmed: bool         # meets entry gate (count >= min or aggressive)

    def __repr__(self) -> str:
        gate = "CONFIRMED" if self.confirmed else "pending"
        return (
            f"ConfirmationResult({self.direction} {gate} "
            f"count={self.count}/5 {self.signals} "
            f"zone={'in' if self.price_in_zone else 'OUT'})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _swing_highs_in_window(df_window: pd.DataFrame, n: int) -> List[float]:
    """Confirmed swing high prices in df_window, oldest → newest."""
    sw = detect_swings(df_window.reset_index(drop=True), n=n)
    return sw.loc[~sw["swing_high"].isna(), "swing_high"].tolist()


def _swing_lows_in_window(df_window: pd.DataFrame, n: int) -> List[float]:
    """Confirmed swing low prices in df_window, oldest → newest."""
    sw = detect_swings(df_window.reset_index(drop=True), n=n)
    return sw.loc[~sw["swing_low"].isna(), "swing_low"].tolist()


def _candle_parts(df: pd.DataFrame, idx: int):
    """Return (o, h, l, c, body, rng, upper_wick, lower_wick) for bar idx."""
    o = float(df["open"].iloc[idx])
    h = float(df["high"].iloc[idx])
    l = float(df["low"].iloc[idx])
    c = float(df["close"].iloc[idx])
    body       = abs(c - o)
    rng        = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return o, h, l, c, body, rng, upper_wick, lower_wick


# ---------------------------------------------------------------------------
# Buy confirmations (demand zone)
# ---------------------------------------------------------------------------

def bullish_engulfing(
    df: pd.DataFrame,
    idx: int,
    full_body: bool = True,
) -> bool:
    """
    Bar at idx is a bullish candle whose body fully (or partially) covers
    the prior bar's red body.

    Conditions (strict / full_body=True):
      - current bar : close > open  (bullish)
      - prior bar   : close < open  (bearish)
      - current open  <= prior close  (engulfs bottom of prior body)
      - current close >= prior open   (engulfs top of prior body)

    full_body=False relaxes to partial overlap of bodies.
    """
    if idx < 1 or idx >= len(df):
        return False

    c_o, _, _, c_c, _, _, _, _ = _candle_parts(df, idx)
    p_o, _, _, p_c, _, _, _, _ = _candle_parts(df, idx - 1)

    if c_c <= c_o:   # current not bullish
        return False
    if p_c >= p_o:   # prior not bearish
        return False

    if full_body:
        return c_o <= p_c and c_c >= p_o
    else:
        # partial: bodies overlap in any direction
        cur_lo, cur_hi = min(c_o, c_c), max(c_o, c_c)
        pri_lo, pri_hi = min(p_o, p_c), max(p_o, p_c)
        return cur_lo <= pri_hi and cur_hi >= pri_lo


def rejection_wick_bull(
    df: pd.DataFrame,
    idx: int,
    wick_ratio_min: float = 0.60,
    body_ratio_max: float = 0.35,
    close_in_upper_half: bool = True,
) -> bool:
    """
    Bullish pin bar / hammer at bar idx.

    Criteria:
      - lower_wick / range >= wick_ratio_min  (dominant lower wick)
      - body / range       <= body_ratio_max  (small body)
      - close in upper 50% of range           (if close_in_upper_half=True)
    """
    if idx < 0 or idx >= len(df):
        return False

    _, h, l, c, body, rng, _, lower_wick = _candle_parts(df, idx)

    if rng < 1e-10:
        return False

    wick_ok  = (lower_wick / rng) >= wick_ratio_min
    body_ok  = (body / rng) <= body_ratio_max
    close_ok = (c >= l + 0.5 * rng) if close_in_upper_half else True

    return wick_ok and body_ok and close_ok


def bullish_bos(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 15,
    swing_n: int = 2,
) -> bool:
    """
    Bar at idx closes above the most recent confirmed swing high in the prior
    `lookback` bars — a bullish Break of Structure / Market Structure Break.

    In a demand zone pullback, this means price has broken through the most
    recent lower high formed during the retracement.
    """
    if idx < swing_n * 2 + 2:
        return False

    start  = max(0, idx - lookback)
    window = df.iloc[start:idx].reset_index(drop=True)   # bars BEFORE idx

    if len(window) < swing_n * 2 + 1:
        return False

    sh_prices = _swing_highs_in_window(window, swing_n)
    if not sh_prices:
        return False

    most_recent_sh = sh_prices[-1]
    return float(df["close"].iloc[idx]) > most_recent_sh


def higher_low(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    swing_n: int = 2,
) -> bool:
    """
    Within the `lookback` bars ending just before idx, two confirmed swing lows
    exist and the more recent one is higher than the earlier — a Higher Low,
    signalling recovering demand inside the zone.
    """
    if idx < swing_n * 2 + 2:
        return False

    start  = max(0, idx - lookback)
    window = df.iloc[start:idx].reset_index(drop=True)

    sl_prices = _swing_lows_in_window(window, swing_n)
    if len(sl_prices) < 2:
        return False

    return sl_prices[-1] > sl_prices[-2]


# ---------------------------------------------------------------------------
# Sell confirmations (supply zone) — mirrored
# ---------------------------------------------------------------------------

def bearish_engulfing(
    df: pd.DataFrame,
    idx: int,
    full_body: bool = True,
) -> bool:
    """
    Bar at idx is a bearish candle whose body fully (or partially) covers
    the prior bar's green body.

    Conditions (strict / full_body=True):
      - current bar : close < open  (bearish)
      - prior bar   : close > open  (bullish)
      - current open  >= prior close  (engulfs top of prior body)
      - current close <= prior open   (engulfs bottom of prior body)
    """
    if idx < 1 or idx >= len(df):
        return False

    c_o, _, _, c_c, _, _, _, _ = _candle_parts(df, idx)
    p_o, _, _, p_c, _, _, _, _ = _candle_parts(df, idx - 1)

    if c_c >= c_o:   # current not bearish
        return False
    if p_c <= p_o:   # prior not bullish
        return False

    if full_body:
        return c_o >= p_c and c_c <= p_o
    else:
        cur_lo, cur_hi = min(c_o, c_c), max(c_o, c_c)
        pri_lo, pri_hi = min(p_o, p_c), max(p_o, p_c)
        return cur_lo <= pri_hi and cur_hi >= pri_lo


def rejection_wick_bear(
    df: pd.DataFrame,
    idx: int,
    wick_ratio_min: float = 0.60,
    body_ratio_max: float = 0.35,
    close_in_lower_half: bool = True,
) -> bool:
    """
    Bearish pin bar / shooting star at bar idx.

    Criteria:
      - upper_wick / range >= wick_ratio_min  (dominant upper wick)
      - body / range       <= body_ratio_max  (small body)
      - close in lower 50% of range           (if close_in_lower_half=True)
    """
    if idx < 0 or idx >= len(df):
        return False

    _, h, l, c, body, rng, upper_wick, _ = _candle_parts(df, idx)

    if rng < 1e-10:
        return False

    wick_ok  = (upper_wick / rng) >= wick_ratio_min
    body_ok  = (body / rng) <= body_ratio_max
    close_ok = (c <= l + 0.5 * rng) if close_in_lower_half else True

    return wick_ok and body_ok and close_ok


def bearish_bos(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 15,
    swing_n: int = 2,
) -> bool:
    """
    Bar at idx closes below the most recent confirmed swing low in the prior
    `lookback` bars — a bearish Break of Structure / Market Structure Break.

    In a supply zone bounce, this means price has broken below the most recent
    higher low formed during the retracement up into the zone.
    """
    if idx < swing_n * 2 + 2:
        return False

    start  = max(0, idx - lookback)
    window = df.iloc[start:idx].reset_index(drop=True)

    if len(window) < swing_n * 2 + 1:
        return False

    sl_prices = _swing_lows_in_window(window, swing_n)
    if not sl_prices:
        return False

    most_recent_sl = sl_prices[-1]
    return float(df["close"].iloc[idx]) < most_recent_sl


def lower_high(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    swing_n: int = 2,
) -> bool:
    """
    Within the `lookback` bars ending just before idx, two confirmed swing highs
    exist and the more recent one is lower than the earlier — a Lower High,
    signalling weakening supply / momentum in the supply zone.
    """
    if idx < swing_n * 2 + 2:
        return False

    start  = max(0, idx - lookback)
    window = df.iloc[start:idx].reset_index(drop=True)

    sh_prices = _swing_highs_in_window(window, swing_n)
    if len(sh_prices) < 2:
        return False

    return sh_prices[-1] < sh_prices[-2]


# ---------------------------------------------------------------------------
# CHoCH — Change of Character (5th confirmation, independent of BOS)
# ---------------------------------------------------------------------------

def bullish_choch(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    swing_left: int = 2,
    swing_right: int = 2,
    min_swing_highs: int = 2,
) -> bool:
    """
    Bullish Change of Character at bar idx.

    Requires TWO conditions — both must hold:
      1. Prevailing M15 structure (in the prior `lookback` bars) is BEARISH:
         the most recent confirmed swing high is a Lower High (LH), meaning
         the local trend was making lower highs before this bar.
      2. The current bar closes ABOVE that most recent Lower High price —
         the first break upward against the bearish trend.

    This is strictly different from bullish_bos:
      - bullish_bos breaks the most recent swing high regardless of its label.
      - bullish_choch requires the trend to be bearish (last SH is LH) first.
      When the last SH is already a HH (uptrend), bullish_choch returns False
      even if price exceeds it (that's a BOS continuation, not a CHoCH).

    No lookahead: only bars df.iloc[max(0, idx-lookback) : idx] are examined.
    Uses swing_structure_Z&Z fractal pivots (same source as H4 bias detection).
    """
    if idx < swing_left + swing_right + 2:
        return False

    start  = max(0, idx - lookback)
    window = df.iloc[start:idx].reset_index(drop=True)   # bars BEFORE idx

    if len(window) < swing_left + swing_right + 1:
        return False

    swings  = _detect_swings_zz(window, left=swing_left, right=swing_right)
    labeled = _label_structure(swings)

    sh_labeled = [(i, p, l) for i, p, l in labeled if l in ("HH", "LH")]

    if len(sh_labeled) < min_swing_highs:
        return False

    last_sh_label = sh_labeled[-1][2]
    last_sh_price = sh_labeled[-1][1]

    # Prior structure must be bearish (most recent confirmed SH is a LH)
    if last_sh_label != "LH":
        return False

    # CHoCH: current close breaks above that Lower High
    return float(df["close"].iloc[idx]) > last_sh_price


def bearish_choch(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 20,
    swing_left: int = 2,
    swing_right: int = 2,
    min_swing_lows: int = 2,
) -> bool:
    """
    Bearish Change of Character at bar idx.

    Requires TWO conditions — both must hold:
      1. Prevailing M15 structure (in the prior `lookback` bars) is BULLISH:
         the most recent confirmed swing low is a Higher Low (HL), meaning
         the local trend was making higher lows before this bar.
      2. The current bar closes BELOW that most recent Higher Low price —
         the first break downward against the bullish trend.

    This is strictly different from bearish_bos:
      - bearish_bos breaks the most recent swing low regardless of its label.
      - bearish_choch requires the trend to be bullish (last SL is HL) first.
      When the last SL is already a LL (downtrend), bearish_choch returns False.

    No lookahead: only bars df.iloc[max(0, idx-lookback) : idx] are examined.
    """
    if idx < swing_left + swing_right + 2:
        return False

    start  = max(0, idx - lookback)
    window = df.iloc[start:idx].reset_index(drop=True)

    if len(window) < swing_left + swing_right + 1:
        return False

    swings  = _detect_swings_zz(window, left=swing_left, right=swing_right)
    labeled = _label_structure(swings)

    sl_labeled = [(i, p, l) for i, p, l in labeled if l in ("HL", "LL")]

    if len(sl_labeled) < min_swing_lows:
        return False

    last_sl_label = sl_labeled[-1][2]
    last_sl_price = sl_labeled[-1][1]

    # Prior structure must be bullish (most recent confirmed SL is a HL)
    if last_sl_label != "HL":
        return False

    # CHoCH: current close breaks below that Higher Low
    return float(df["close"].iloc[idx]) < last_sl_price


# ---------------------------------------------------------------------------
# Aggregate: check all confirmations at a specific bar
# ---------------------------------------------------------------------------

def check_all_confirmations(
    df_m15: pd.DataFrame,
    idx: int,
    zone: Zone,
    direction: str,
    cfg: Optional[ConfirmationConfig] = None,
) -> ConfirmationResult:
    """
    Evaluate all four confirmations for the given direction at bar `idx`.

    Preconditions enforced here (not in individual functions):
      1. The bar at idx must overlap the H4 zone (high/low range touches zone).
         If not, all confirmations return False and confirmed=False.
      2. The signal bar's close must be inside the zone (with tolerance).
         Buy: close must be <= zone.top * (1 + tol).
         Sell: close must be >= zone.bottom * (1 - tol).
         If not, price exited the zone on the signal bar; confirmed=False is
         returned with price_in_zone=True so callers can distinguish the two
         failure modes in logs.

    Parameters
    ----------
    df_m15    : M15 OHLCV dataframe.
    idx       : Signal bar index to evaluate (usually len(df_m15) - 1).
    zone      : Active H4 Zone from timeframe_structure.analyse_timeframes.
    direction : 'buy' (demand) | 'sell' (supply).
    cfg       : ConfirmationConfig (defaults applied when None).

    Returns
    -------
    ConfirmationResult with individual flags, count, stacked signals list,
    and confirmed boolean.
    """
    if cfg is None:
        cfg = ConfirmationConfig()

    _false = ConfirmationResult(
        direction=direction, price_in_zone=False,
        engulfing=False, rejection_wick=False,
        bos_msb=False, structure_shift=False, choch=False,
        count=0, signals=[], confirmed=False,
    )

    if idx < 0 or idx >= len(df_m15):
        return _false

    # ── Zone precondition ────────────────────────────────────────────────────
    bar_h = float(df_m15["high"].iloc[idx])
    bar_l = float(df_m15["low"].iloc[idx])
    z_top = zone.top    * (1 + cfg.zone_tolerance_pct)
    z_bot = zone.bottom * (1 - cfg.zone_tolerance_pct)
    in_zone = bar_l <= z_top and bar_h >= z_bot

    if not in_zone:
        return _false

    # ── Close-containment gate ───────────────────────────────────────────────
    # Catch the case where the bar range touched the zone but the close has
    # already exited: the fill on the next bar would be geometrically outside,
    # producing an invalid SL distance.  Overnight gaps are still caught by
    # zone_guard at fill time (signal close was valid; gap happened between bars).
    bar_c = float(df_m15["close"].iloc[idx])
    close_left_zone = (
        (direction == "buy"  and bar_c > zone.top    * (1 + cfg.zone_tolerance_pct)) or
        (direction == "sell" and bar_c < zone.bottom * (1 - cfg.zone_tolerance_pct))
    )
    if close_left_zone:
        return ConfirmationResult(
            direction=direction, price_in_zone=True,
            engulfing=False, rejection_wick=False,
            bos_msb=False, structure_shift=False, choch=False,
            count=0, signals=[], confirmed=False,
        )

    # ── Fire each confirmation independently (Step 2 pre-filter not included) ──
    if direction == "buy":
        eng  = bullish_engulfing(df_m15, idx,
                                 full_body=cfg.engulf_full_body)
        wick = rejection_wick_bull(df_m15, idx,
                                   wick_ratio_min=cfg.wick_ratio_min,
                                   body_ratio_max=cfg.body_ratio_max,
                                   close_in_upper_half=cfg.close_in_upper_half)
        bos  = bullish_bos(df_m15, idx,
                           lookback=cfg.bos_lookback,
                           swing_n=cfg.bos_swing_n)
        strk = higher_low(df_m15, idx,
                          lookback=cfg.structure_lookback,
                          swing_n=cfg.structure_swing_n)
        chch = bullish_choch(df_m15, idx,
                             lookback=cfg.choch_lookback,
                             swing_left=cfg.choch_swing_left,
                             swing_right=cfg.choch_swing_right,
                             min_swing_highs=cfg.choch_min_swings)
    else:
        eng  = bearish_engulfing(df_m15, idx,
                                 full_body=cfg.engulf_full_body)
        wick = rejection_wick_bear(df_m15, idx,
                                   wick_ratio_min=cfg.wick_ratio_min,
                                   body_ratio_max=cfg.body_ratio_max,
                                   close_in_lower_half=cfg.close_in_upper_half)
        bos  = bearish_bos(df_m15, idx,
                           lookback=cfg.bos_lookback,
                           swing_n=cfg.bos_swing_n)
        strk = lower_high(df_m15, idx,
                          lookback=cfg.structure_lookback,
                          swing_n=cfg.structure_swing_n)
        chch = bearish_choch(df_m15, idx,
                             lookback=cfg.choch_lookback,
                             swing_left=cfg.choch_swing_left,
                             swing_right=cfg.choch_swing_right,
                             min_swing_lows=cfg.choch_min_swings)

    fired: List[str] = []
    if eng:  fired.append("engulfing")
    if wick: fired.append("rejection_wick")
    if bos:  fired.append("bos_msb")
    if strk: fired.append("structure_shift")
    if chch: fired.append("choch")

    # count_effective excludes signals in excluded_from_count from the gate check.
    # count (total fired) is preserved for logging so trade records are unchanged.
    excluded = set(cfg.excluded_from_count or [])
    count_effective = sum(1 for s in fired if s not in excluded)
    count = len(fired)

    if cfg.aggressive_boundary:
        confirmed = True
    else:
        confirmed = count_effective >= cfg.min_confirmations

    return ConfirmationResult(
        direction=direction,
        price_in_zone=True,
        engulfing=eng,
        rejection_wick=wick,
        bos_msb=bos,
        structure_shift=strk,
        choch=chch,
        count=count,
        signals=fired,
        confirmed=confirmed,
    )


def check_confirmations_at_last_bar(
    df_m15: pd.DataFrame,
    zone: Zone,
    direction: str,
    cfg: Optional[ConfirmationConfig] = None,
) -> ConfirmationResult:
    """Convenience wrapper: evaluate confirmations at the last M15 bar."""
    return check_all_confirmations(df_m15, len(df_m15) - 1, zone, direction, cfg)
