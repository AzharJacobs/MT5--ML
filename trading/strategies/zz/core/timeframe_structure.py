"""
timeframe_structure.py — Two-timeframe zone + bias framework (Step 2).

Roles
-----
H4  : directional bias detection + strongest zone marking.
M15 : confirmation that price has entered the H4 zone with a directional close.

Directional filter (default ON, toggleable via TFConfig.directional_filter):
  Buy  from demand zone → requires H4 bias 'bullish'  or 'neutral_up'
  Sell from supply zone → requires H4 bias 'bearish'  or 'neutral_down'
  When disabled, all zones are eligible regardless of H4 bias.

Bias states
-----------
  'bullish'      — H4 swing structure shows HH + HL (both legs trending up)
  'bearish'      — H4 swing structure shows LH + LL (both legs trending down)
  'neutral_up'   — mixed structure, upward lean  (HH + LL, or last confirmed leg up)
  'neutral_down' — mixed structure, downward lean (LH + HL)
  'neutral'      — fewer than two confirmed swing highs/lows; no readable direction

This module builds on strategy.zones (Step 1) and strategy/swing_structure_Z&Z.py
for H4 bias.  Step 3's higher_low/lower_high use a separate M15 mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from trading.strategies.zz.core.swing_structure import detect_swings as _detect_swings_zz, label_structure as _label_structure

from trading.strategies.zz.core.zones import (
    Zone,
    ZoneConfig,
    detect_zones,
    update_freshness,
    nearest_zones,
    strongest_zones,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TFConfig:
    # ── Directional filter (explicit + toggleable) ───────────────────────────
    directional_filter: bool = True
    # True  → demand (buy) trades blocked unless H4 bias is bullish or neutral_up.
    #         supply (sell) trades blocked unless H4 bias is bearish or neutral_down.
    # False → any zone is tradeable regardless of H4 bias direction.

    allow_neutral_up: bool = True
    # When directional_filter=True, neutral_up bias still permits demand zone buys.

    allow_neutral_down: bool = True
    # When directional_filter=True, neutral_down bias still permits supply zone sells.

    # ── H4 bias detection (fractal pivot via swing_structure_Z&Z) ────────────
    h4_swing_left: int = 2
    # Bars to the LEFT of a pivot that must be lower/higher to confirm a swing.

    h4_swing_right: int = 2
    # Bars to the RIGHT that must be lower/higher.  Confirmed swings lag by
    # exactly h4_swing_right bars — no lookahead beyond that window.

    # ── H4 zone parameters ───────────────────────────────────────────────────
    h4_zone_cfg: ZoneConfig = field(default_factory=ZoneConfig)
    # Full ZoneConfig for H4 zone detection (impulse thresholds, departure window, etc.)

    prefer_fresh_h4: bool = True
    # Prefer fresh (un-tapped) H4 zones when selecting the active zone.

    max_h4_candidates: int = 5
    # Maximum number of H4 zones considered per kind when ranking candidates.

    # ── M15 confirmation ─────────────────────────────────────────────────────
    m15_tap_lookback: int = 20
    # Recent M15 bars to check for zone touch (bar range overlapping the zone).

    m15_tap_tolerance_pct: float = 0.001
    # ±0.1% tolerance on zone edges for M15 tap detection.

    require_m15_directional_close: bool = True
    # Require at least one M15 candle within the tap window to close with a
    # body in the trade direction (bullish body for demand, bearish for supply).

    # ── M15 zone detection (stacked-confluence filter) ────────────────────────
    m15_zone_cfg: ZoneConfig = field(
        default_factory=lambda: ZoneConfig(
            impulse_atr_mult=1.5,
            min_strength=1.0,
            min_departure_candles=2,
            departure_window=6,
        )
    )
    # M15 impulses are smaller than H4 so thresholds are lower.
    # Used by the engine's stacked-confluence filter; ignored when that filter is off.

    # ── 1H zone detection (optional stacked confluence) ───────────────────────
    h1_zone_cfg: ZoneConfig = field(
        default_factory=lambda: ZoneConfig(
            impulse_atr_mult=1.75,
            min_strength=1.2,
            min_departure_candles=2,
            departure_window=6,
        )
    )
    use_1h_zones: bool = False
    # When True, detect 1H zones and merge them into the H4 zone pool.
    # ZoneConfig is scaled between H4 (atr_mult=2.0, strength=1.5) and
    # M15 (atr_mult=1.5, strength=1.0) defaults.


# ---------------------------------------------------------------------------
# H4 bias detection
# ---------------------------------------------------------------------------

def detect_h4_bias(
    df_h4: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> str:
    """
    Determine H4 directional bias using the fractal pivot method from
    swing_structure_Z&Z.py.

    Detects swing highs/lows with the left/right confirmation window, labels
    each as HH/LH/HL/LL, then reads the most recent label pair.

    Returns
    -------
    'bullish'      — last SH is HH  AND  last SL is HL
    'bearish'      — last SH is LH  AND  last SL is LL
    'neutral_up'   — mixed (HH+LL or LH+HL where upward leg is present)
    'neutral_down' — mixed (downward leg present)
    'neutral'      — fewer than 2 confirmed swing highs or lows
    """
    if df_h4 is None or len(df_h4) < left + right + 4:
        return "neutral"

    swings  = _detect_swings_zz(df_h4.reset_index(drop=True), left=left, right=right)
    labeled = _label_structure(swings)

    sh = [(i, p, l) for i, p, l in labeled if l in ("HH", "LH")]
    sl = [(i, p, l) for i, p, l in labeled if l in ("HL", "LL")]

    if len(sh) < 2 or len(sl) < 2:
        return "neutral"

    last_sh_lbl = sh[-1][2]
    last_sl_lbl = sl[-1][2]

    if last_sh_lbl == "HH" and last_sl_lbl == "HL":
        return "bullish"
    if last_sh_lbl == "LH" and last_sl_lbl == "LL":
        return "bearish"
    # Mixed cases
    if last_sh_lbl == "HH":
        return "neutral_up"
    if last_sh_lbl == "LH":       # LH+HL = lower high dominates → neutral_down
        return "neutral_down"
    if last_sl_lbl == "HL":       # HH+HL already handled above; only LL+HL reaches here
        return "neutral_up"
    # last_sl_lbl == "LL"
    return "neutral_down"


# ---------------------------------------------------------------------------
# Directional filter
# ---------------------------------------------------------------------------

def bias_permits(bias: str, zone_kind: str, cfg: TFConfig) -> bool:
    """
    Return True when the H4 bias allows trading the given zone kind.

    When cfg.directional_filter is False this always returns True.
    Otherwise:
      demand (buy)  → needs 'bullish' or (neutral_up  if allow_neutral_up)
      supply (sell) → needs 'bearish' or (neutral_down if allow_neutral_down)
    """
    if not cfg.directional_filter:
        return True

    if zone_kind == "demand":
        return bias == "bullish" or (bias == "neutral_up" and cfg.allow_neutral_up)

    if zone_kind == "supply":
        return bias == "bearish" or (bias == "neutral_down" and cfg.allow_neutral_down)

    return False


# ---------------------------------------------------------------------------
# M15 helpers
# ---------------------------------------------------------------------------

def _m15_tapped(df_m15: pd.DataFrame, zone: Zone, lookback: int, tol: float) -> bool:
    """True if any of the last `lookback` M15 bars overlaps the zone (wicks included)."""
    recent = df_m15.tail(lookback)
    z_top = zone.top    * (1 + tol)
    z_bot = zone.bottom * (1 - tol)
    return bool(((recent["low"] <= z_top) & (recent["high"] >= z_bot)).any())


def _m15_directional_close(
    df_m15: pd.DataFrame,
    zone: Zone,
    direction: str,
    lookback: int,
    tol: float,
) -> bool:
    """
    True if any of the last `lookback` M15 bars simultaneously:
      (a) overlaps the zone, AND
      (b) closes with a body in the trade direction.

    direction='buy'  → close > open (bullish body) while touching demand zone.
    direction='sell' → close < open (bearish body) while touching supply zone.
    """
    recent = df_m15.tail(lookback)
    z_top = zone.top    * (1 + tol)
    z_bot = zone.bottom * (1 - tol)
    in_zone = (recent["low"] <= z_top) & (recent["high"] >= z_bot)

    if direction == "buy":
        directional = recent["close"] > recent["open"]
    else:
        directional = recent["close"] < recent["open"]

    return bool((in_zone & directional).any())


def _m15_clean_edge_tap(
    df_m15: pd.DataFrame,
    zone: Zone,
    direction: str,
    lookback: int,
    tol: float,
) -> bool:
    """
    True when the M15 tap is a clean edge reaction rather than mid-zone drift.

    A clean tap requires at least one M15 bar in the lookback window that:
      (a) overlaps the zone (same condition as _m15_tapped), AND
      (b) touched the critical edge of the zone:
            demand → bar low reached zone.bottom*(1+tol)  (touched bottom edge)
            supply → bar high reached zone.top*(1-tol)    (touched top edge)
      (c) closed on the correct side of the zone midpoint:
            demand → close >= zone.mid  (bounced back above the midpoint)
            supply → close <= zone.mid  (dropped back below the midpoint)

    Failure = price wandered into the middle of the zone without a clean
    edge approach → no-man's-land, skip.
    """
    recent = df_m15.tail(lookback)
    z_top  = zone.top    * (1 + tol)
    z_bot  = zone.bottom * (1 - tol)
    z_mid  = zone.mid

    in_zone = (recent["low"] <= z_top) & (recent["high"] >= z_bot)

    if direction == "buy":
        edge_touched  = recent["low"] <= zone.bottom * (1 + tol)
        close_reacted = recent["close"] >= z_mid
    else:
        edge_touched  = recent["high"] >= zone.top * (1 - tol)
        close_reacted = recent["close"] <= z_mid

    return bool((in_zone & edge_touched & close_reacted).any())


# ---------------------------------------------------------------------------
# H4 zone selection
# ---------------------------------------------------------------------------

def select_h4_zone(
    h4_zones: List[Zone],
    bias: str,
    cfg: TFConfig,
    current_price: float,
) -> Optional[Zone]:
    """
    Choose the single best H4 zone given the current bias and price.

    Selection logic:
      1. Filter zones by directional filter (bias_permits).
      2. Prefer fresh zones over tapped ones.
      3. Within the fresh pool, pick the one with the highest strength
         that is closest to current price (proximity × strength composite).
      4. Fall back to stale zones by the same composite if no fresh zones exist.

    Returns None when no eligible zones are found.
    """
    eligible = [z for z in h4_zones if bias_permits(bias, z.kind, cfg)]
    if not eligible:
        return None

    def score(z: Zone) -> float:
        dist = abs(z.mid - current_price)
        if dist < 1e-10:
            dist = 1e-10
        return z.strength / dist

    if cfg.prefer_fresh_h4:
        fresh = sorted([z for z in eligible if z.fresh],     key=score, reverse=True)
        stale = sorted([z for z in eligible if not z.fresh], key=score, reverse=True)
        ranked = fresh + stale
    else:
        ranked = sorted(eligible, key=score, reverse=True)

    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def analyse_timeframes(
    df_h4: pd.DataFrame,
    df_m15: pd.DataFrame,
    cfg: Optional[TFConfig] = None,
    h4_up_to_bar: Optional[int] = None,
    df_h1: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full two-timeframe structure analysis.

    Parameters
    ----------
    df_h4          : H4 OHLCV dataframe (caller pre-trims to the desired window).
    df_m15         : M15 OHLCV dataframe (last bar = current bar).
    cfg            : TFConfig (defaults applied when None).
    h4_up_to_bar   : If provided, zone detection on H4 is capped at this bar
                     index (inclusive) to prevent lookahead in backtesting.
                     Pass None to use the full df_h4.

    Returns
    -------
    dict with keys:
      h4_bias        — str: 'bullish'|'bearish'|'neutral_up'|'neutral_down'|'neutral'
      h4_zones       — List[Zone]: all zones found on H4 (freshness updated)
      eligible_zones — List[Zone]: zones that pass the directional filter
      active_zone    — Zone | None: selected zone (strongest + closest eligible)
      direction      — 'buy'|'sell'|None: trade direction implied by active zone
      zone_tapped    — bool: M15 has touched the active zone recently
      m15_confirmed  — bool: M15 shows a directional close inside active zone
      signal         — 'buy'|'sell'|'neutral'
      reason         — str: human-readable explanation
    """
    if cfg is None:
        cfg = TFConfig()

    result = {
        "h4_bias":        "neutral",
        "h4_zones":       [],
        "eligible_zones": [],
        "active_zone":    None,
        "direction":      None,
        "zone_tapped":    False,
        "m15_confirmed":  False,
        "clean_tap":      False,
        "signal":         "neutral",
        "reason":         "",
    }

    if df_h4 is None or df_m15 is None or df_h4.empty or df_m15.empty:
        result["reason"] = "missing data"
        return result

    # ── Step 1: H4 bias ──────────────────────────────────────────────────────
    bias = detect_h4_bias(df_h4, left=cfg.h4_swing_left, right=cfg.h4_swing_right)
    result["h4_bias"] = bias

    # ── Step 2: H4 zone detection ────────────────────────────────────────────
    h4_zones = detect_zones(df_h4, cfg=cfg.h4_zone_cfg, up_to_bar=h4_up_to_bar)

    # Update freshness against the full H4 price history available
    if h4_zones:
        highs_h4 = df_h4["high"].values
        lows_h4  = df_h4["low"].values
        cap = h4_up_to_bar if h4_up_to_bar is not None else len(df_h4) - 1
        update_freshness(h4_zones, highs_h4, lows_h4, current_bar=max(0, cap - 2),
                         tap_tol=cfg.h4_zone_cfg.tap_tolerance_pct)

    # ── 1H zone detection (merged into H4 pool when enabled) ────────────────
    h1_zones: list = []
    if cfg.use_1h_zones and df_h1 is not None and not df_h1.empty:
        h1_up = len(df_h1) - 1
        h1_zones = detect_zones(df_h1, cfg=cfg.h1_zone_cfg, up_to_bar=h1_up)
        if h1_zones:
            highs_h1 = df_h1["high"].values
            lows_h1  = df_h1["low"].values
            update_freshness(h1_zones, highs_h1, lows_h1,
                             current_bar=max(0, h1_up - 2),
                             tap_tol=cfg.h1_zone_cfg.tap_tolerance_pct)
            h4_zones = h4_zones + h1_zones

    result["h4_zones"] = h4_zones
    result["h1_zones"] = h1_zones   # kept separate for zone-TF tracking in engine

    # ── Step 3: directional filter ───────────────────────────────────────────
    eligible = [z for z in h4_zones if bias_permits(bias, z.kind, cfg)]
    result["eligible_zones"] = eligible

    if not eligible:
        result["reason"] = (
            f"H4 bias={bias} — no zones pass directional filter "
            f"(filter={'ON' if cfg.directional_filter else 'OFF'})"
        )
        return result

    # ── Step 4: select the active H4 zone ───────────────────────────────────
    current_price = float(df_m15["close"].iloc[-1])
    active_zone   = select_h4_zone(eligible, bias, cfg, current_price)
    if active_zone is None:
        result["reason"] = "no eligible H4 zone found after selection"
        return result

    result["active_zone"] = active_zone
    direction = "buy" if active_zone.kind == "demand" else "sell"
    result["direction"] = direction

    # ── Step 5: M15 zone tap ─────────────────────────────────────────────────
    tapped = _m15_tapped(df_m15, active_zone,
                         cfg.m15_tap_lookback, cfg.m15_tap_tolerance_pct)
    result["zone_tapped"] = tapped

    if tapped:
        result["clean_tap"] = _m15_clean_edge_tap(
            df_m15, active_zone, direction,
            cfg.m15_tap_lookback, cfg.m15_tap_tolerance_pct,
        )

    if not tapped:
        result["reason"] = (
            f"H4 bias={bias} | {active_zone.kind} zone {active_zone.bottom:.2f}–"
            f"{active_zone.top:.2f} (str={active_zone.strength:.1f}) not yet tapped on M15"
        )
        return result

    # ── Step 6: M15 directional confirmation ─────────────────────────────────
    m15_ok = True
    if cfg.require_m15_directional_close:
        m15_ok = _m15_directional_close(
            df_m15, active_zone, direction,
            cfg.m15_tap_lookback, cfg.m15_tap_tolerance_pct,
        )
    result["m15_confirmed"] = m15_ok

    if not m15_ok:
        result["reason"] = (
            f"H4 bias={bias} | zone tapped | "
            f"waiting for M15 {direction} confirmation candle"
        )
        return result

    # ── Signal confirmed ──────────────────────────────────────────────────────
    freshness = "fresh" if active_zone.fresh else f"tapped×{active_zone.tap_count}"
    result["signal"] = direction
    result["reason"]  = (
        f"H4 bias={bias} | {active_zone.kind} zone "
        f"{active_zone.bottom:.2f}–{active_zone.top:.2f} "
        f"(str={active_zone.strength:.1f} {freshness}) | "
        f"M15 tapped + {direction} confirmation"
    )
    return result


# ---------------------------------------------------------------------------
# Convenience wrapper: directional filter toggle at call time
# ---------------------------------------------------------------------------

def signal_with_filter(
    df_h4: pd.DataFrame,
    df_m15: pd.DataFrame,
    directional_filter: bool = True,
    cfg: Optional[TFConfig] = None,
    h4_up_to_bar: Optional[int] = None,
) -> dict:
    """
    One-call wrapper that overrides cfg.directional_filter at call time.
    Useful for quick comparison between filtered and unfiltered signals.
    """
    if cfg is None:
        cfg = TFConfig()
    cfg.directional_filter = directional_filter
    return analyse_timeframes(df_h4, df_m15, cfg=cfg, h4_up_to_bar=h4_up_to_bar)
