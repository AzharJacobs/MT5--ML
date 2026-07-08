"""
Zone detection and grading for the FXGold strategy.

Zones are formed from fractal swing highs/lows (5-bar pivot by default).
Boundaries follow the wick (shadow), not the body:

  Resistance zone (swing high):
    top    = candle high        (wick tip)
    bottom = max(open, close)   (body top)

  Support zone (swing low):
    top    = min(open, close)   (body bottom)
    bottom = candle low         (wick tip)

Grading rules:
  - Touch   : wick enters the zone AND bar closes outside  →  rejection count +1
  - Breakout: bar body closes THROUGH the zone             →  zone flips role (R↔S)
  - score   = rejection_count + tf_weight[origin_tf]
  - is_strong = rejections >= min_rejections_strong AND origin_tf in strong_tfs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from trading.strategies.FXGold.config import FXGoldConfig


# ─── Zone dataclass ───────────────────────────────────────────────────────────

@dataclass
class Zone:
    kind: str        # "support" or "resistance"
    top: float
    bottom: float
    origin_tf: str
    bar_index: int   # index of the fractal candle in the source DataFrame
    touches: int = 0
    score: float = 0.0
    is_strong: bool = False
    flipped: bool = False
    flip_origin: Optional[int] = None   # bar index where first breakout fired
    dead: bool = False                  # True after a second breakout (single-flip guard)

    @property
    def zone_key(self) -> Tuple[float, float]:
        return (round(self.bottom, 2), round(self.top, 2))


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_fractals(df: pd.DataFrame, window: int) -> Tuple[List[int], List[int]]:
    """
    Return (swing_high_indices, swing_low_indices).
    A pivot at index i requires the high/low to be strictly greater/less than
    all bars in the [i-window, i-1] and [i+1, i+window] windows.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)
    swing_hi: List[int] = []
    swing_lo: List[int] = []

    for i in range(window, n - window):
        if highs[i] > highs[i - window: i].max() and highs[i] > highs[i + 1: i + window + 1].max():
            swing_hi.append(i)
        if lows[i] < lows[i - window: i].min() and lows[i] < lows[i + 1: i + window + 1].min():
            swing_lo.append(i)

    return swing_hi, swing_lo


def _zone_from_high(df: pd.DataFrame, idx: int, tf: str) -> Zone:
    row = df.iloc[idx]
    return Zone(
        kind="resistance",
        top=float(row["high"]),
        bottom=float(max(row["open"], row["close"])),
        origin_tf=tf,
        bar_index=idx,
    )


def _zone_from_low(df: pd.DataFrame, idx: int, tf: str) -> Zone:
    row = df.iloc[idx]
    return Zone(
        kind="support",
        top=float(min(row["open"], row["close"])),
        bottom=float(row["low"]),
        origin_tf=tf,
        bar_index=idx,
    )


def _grade(zones: List[Zone], df: pd.DataFrame, cfg: FXGoldConfig) -> None:
    """
    Scan all bars AFTER each zone's origin candle to count touches and detect
    breakouts. Mutates each Zone in place.

    Reads OHLC once into numpy arrays — this runs on every H4/D1 bar close via
    detect_zones, and per-scalar df.iloc[i] access here was the dominant cost
    of a backtest (each pandas scalar lookup is ~100x a numpy array index).
    """
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    for zone in zones:
        for i in range(zone.bar_index + 1, n):
            bh = float(highs[i])
            bl = float(lows[i])
            bc = float(closes[i])

            if is_breakout(bc, zone):
                if zone.flipped:                  # single-flip guard: second breakout kills zone
                    zone.dead = True
                    break
                zone.flipped = True
                zone.kind = "support" if zone.kind == "resistance" else "resistance"
                zone.touches = 0                  # pre-breakout touches discarded
                zone.flip_origin = i              # touch counting restarts here
                continue                          # keep scanning for post-flip retests

            if is_rejection_touch(bh, bl, bc, zone):
                zone.touches += 1

        w = cfg.tf_weights.get(zone.origin_tf, 1.0)
        zone.score = zone.touches + w
        zone.is_strong = (
            zone.touches >= cfg.min_rejections_strong
            and zone.origin_tf in cfg.strong_tfs
        )


# ─── Public helpers (used by entry.py) ───────────────────────────────────────

def is_rejection_touch(bar_high: float, bar_low: float, bar_close: float, zone: Zone) -> bool:
    """
    True when price wicks INTO the zone but closes OUTSIDE — a rejection.

    Resistance: bar wick reaches zone.bottom or above; close stays below zone.bottom.
    Support:    bar wick reaches zone.top or below;    close stays above zone.top.
    """
    if zone.kind == "resistance":
        return bar_high >= zone.bottom and bar_close < zone.bottom
    else:
        return bar_low <= zone.top and bar_close > zone.top


def is_breakout(bar_close: float, zone: Zone) -> bool:
    """
    True when the bar body closes fully through the zone — triggers a role flip.

    Resistance: close above zone.top   (bull breakout through supply)
    Support:    close below zone.bottom (bear breakout through demand)
    """
    if zone.kind == "resistance":
        return bar_close > zone.top
    else:
        return bar_close < zone.bottom


def price_in_zone(bar_high: float, bar_low: float, zone: Zone) -> bool:
    """True when any part of the bar range overlaps the zone."""
    return bar_low <= zone.top and bar_high >= zone.bottom


# ─── Main detection entry point ───────────────────────────────────────────────

def detect_zones(df: pd.DataFrame, tf: str, cfg: FXGoldConfig) -> List[Zone]:
    """
    Full zone detection pipeline for a single-timeframe OHLCV DataFrame.

    Steps:
      1. Find all fractal swing highs/lows (5-bar pivot by default).
      2. Build a Zone for each pivot using wick boundaries.
      3. Grade each zone: count rejection touches, detect role flips.
      4. Return strong, non-flipped zones sorted by score (highest first).

    Args:
        df:  OHLCV DataFrame with columns open/high/low/close, sorted ascending.
             Must have at least 2 * cfg.fractal_window + 1 rows.
        tf:  Timeframe label matching keys in cfg.tf_weights (e.g. "H4", "Daily").
        cfg: Strategy configuration.

    Returns:
        List[Zone] — strong, active zones only.
    """
    if len(df) < 2 * cfg.fractal_window + 1:
        return []

    swing_hi, swing_lo = _find_fractals(df, window=cfg.fractal_window)

    zones: List[Zone] = []
    for idx in swing_hi:
        zones.append(_zone_from_high(df, idx, tf))
    for idx in swing_lo:
        zones.append(_zone_from_low(df, idx, tf))

    _grade(zones, df, cfg)

    strong = [z for z in zones if z.is_strong and not z.dead]
    return sorted(strong, key=lambda z: z.score, reverse=True)
