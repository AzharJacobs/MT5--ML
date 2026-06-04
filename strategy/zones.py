"""
zones.py — Demand / Supply zone detection engine.

Zone marking rules (demand — supply is mirrored):
  1. Detect an impulsive UP move: strong bullish departure candles where
     price breaks away sharply from a base.
  2. Step back to the BASE immediately before the impulse:
       the last red/bearish candle (or, if none in range, the prior bar).
  3. Zone box uses the base candle's full range (wicks):
       top    = high (wick high) of base candle/cluster
       bottom = low  (wick low)  of base candle/cluster
  4. Strength = net_impulse / ATR — higher = stronger zone.
  5. A zone is "fresh" until price taps back into it; prefer fresh zones.

All thresholds are tunable via ZoneConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ZoneConfig:
    # ── Impulse detection (all tunable) ─────────────────────────────────────
    impulse_atr_mult: float = 2.0
    # Net move of the departure window must be >= impulse_atr_mult × ATR.
    # Raise to require stronger explosions; lower to catch smaller moves.

    body_ratio_min: float = 0.5
    # Minimum body-to-range ratio: |close - open| / (high - low).
    # Departure candles below this are not counted toward min_departure_candles.

    min_departure_candles: int = 2
    # Minimum number of qualifying departure candles within departure_window.

    departure_window: int = 6
    # Number of bars to scan for the impulse after the base candle.

    # ── Base detection ───────────────────────────────────────────────────────
    base_lookback: int = 5
    # Bars back from the impulse start to search for the opposing-colour base.
    # Demand → last bearish candle; Supply → last bullish candle.
    # Falls back to the immediate prior bar if none found.

    # ── Over-marking guard ───────────────────────────────────────────────────
    min_strength: float = 1.5
    # Hard floor: impulse_net / ATR below this → zone discarded.
    # Prevents weak or ambiguous moves from registering zones.

    # ── Freshness / tap detection ────────────────────────────────────────────
    tap_tolerance_pct: float = 0.001
    # ±0.1 % around zone edges counts as a tap.
    # Price does not need to close inside; any bar whose range overlaps counts.


# ---------------------------------------------------------------------------
# Zone data object
# ---------------------------------------------------------------------------

@dataclass
class Zone:
    kind: str        # 'demand' | 'supply'
    top: float       # upper edge — wick high of base candle
    bottom: float    # lower edge — wick low  of base candle
    origin_bar: int  # bar index of the base candle in the source dataframe
    strength: float  # impulse_net / ATR at origin (higher = stronger)
    fresh: bool = True
    tap_count: int = 0

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float, tol: float = 0.0) -> bool:
        return self.bottom * (1 - tol) <= price <= self.top * (1 + tol)

    def __repr__(self) -> str:
        state = "fresh" if self.fresh else f"tapped×{self.tap_count}"
        return (
            f"Zone({self.kind} {self.bottom:.2f}–{self.top:.2f} "
            f"str={self.strength:.1f} bar={self.origin_bar} {state})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int,
) -> np.ndarray:
    """Per-bar rolling ATR array (Wilder-style simple rolling mean of TR)."""
    n = len(highs)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    prev = closes[:-1]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev), np.abs(lows[1:] - prev)),
    )
    atr = np.empty(n)
    for i in range(n):
        s = max(0, i - period + 1)
        atr[i] = tr[s : i + 1].mean()
    return atr


def _body_ratio(o: float, h: float, l: float, c: float) -> float:
    rng = h - l
    return abs(c - o) / rng if rng > 1e-10 else 0.0


def _find_base(
    opens: np.ndarray,
    closes: np.ndarray,
    impulse_start: int,
    kind: str,
    lookback: int,
) -> int:
    """
    Walk back from impulse_start to find the base candle.

    Demand zone → last bearish (close < open) candle in [impulse_start-lookback,
                  impulse_start-1].
    Supply zone → last bullish (close > open) candle in the same range.
    Falls back to impulse_start - 1 if no matching candle is found.
    """
    stop = max(0, impulse_start - lookback)
    if kind == "demand":
        for j in range(impulse_start - 1, stop - 1, -1):
            if closes[j] < opens[j]:
                return j
    else:
        for j in range(impulse_start - 1, stop - 1, -1):
            if closes[j] > opens[j]:
                return j
    return max(0, impulse_start - 1)


# ---------------------------------------------------------------------------
# Zone detection
# ---------------------------------------------------------------------------

def detect_zones(
    df: pd.DataFrame,
    cfg: Optional[ZoneConfig] = None,
    atr_period: int = 14,
    up_to_bar: Optional[int] = None,
) -> List[Zone]:
    """
    Detect all demand and supply zones in df.

    Parameters
    ----------
    df          : OHLCV dataframe with columns open/high/low/close.
    cfg         : ZoneConfig (defaults applied when None).
    atr_period  : ATR lookback period.
    up_to_bar   : Only detect zones using bars 0…up_to_bar (inclusive).
                  Zones require departure_window bars AFTER the base to confirm,
                  so the last detectable base is at up_to_bar - departure_window.
                  Pass None to scan the full dataframe.

    Returns
    -------
    List of Zone objects in chronological order (oldest origin first).
    """
    if cfg is None:
        cfg = ZoneConfig()

    df = df.reset_index(drop=True)
    n = len(df)
    limit = (up_to_bar + 1) if up_to_bar is not None else n

    opens  = df["open"].values[:limit]
    highs  = df["high"].values[:limit]
    lows   = df["low"].values[:limit]
    closes = df["close"].values[:limit]
    m = len(opens)

    min_bars = atr_period + cfg.departure_window + 2
    if m < min_bars:
        return []

    atr = _rolling_atr(highs, lows, closes, atr_period)

    zones: List[Zone] = []
    # Separate origin-bar sets per kind to allow (unlikely) dual-use of same bar
    used_demand: set = set()
    used_supply: set = set()

    # i = first bar of the impulse; bars [i, i+departure_window) = impulse window
    scan_end = m - cfg.departure_window
    for i in range(atr_period + 1, scan_end):
        cur_atr = atr[i - 1]   # ATR at the candle just before the impulse
        if cur_atr <= 0:
            continue

        ref_close = closes[i - 1]   # close of the bar immediately before the impulse
        w_end = i + cfg.departure_window
        if w_end > m:
            continue

        w_o = opens[i:w_end]
        w_h = highs[i:w_end]
        w_l = lows[i:w_end]
        w_c = closes[i:w_end]

        net_up   = w_c[-1] - ref_close
        net_down = ref_close - w_c[-1]

        bull_qual = sum(
            1 for k in range(len(w_o))
            if w_c[k] > w_o[k]
            and _body_ratio(w_o[k], w_h[k], w_l[k], w_c[k]) >= cfg.body_ratio_min
        )
        bear_qual = sum(
            1 for k in range(len(w_o))
            if w_c[k] < w_o[k]
            and _body_ratio(w_o[k], w_h[k], w_l[k], w_c[k]) >= cfg.body_ratio_min
        )

        # ── Demand zone (impulsive UP) ─────────────────────────────────────
        if (net_up >= cfg.impulse_atr_mult * cur_atr
                and bull_qual >= cfg.min_departure_candles):
            base_idx = _find_base(opens, closes, i, "demand", cfg.base_lookback)
            if base_idx not in used_demand:
                strength = net_up / cur_atr
                if strength >= cfg.min_strength:
                    z_top = float(highs[base_idx])
                    z_bot = float(lows[base_idx])
                    if z_top > z_bot:
                        zones.append(Zone(
                            kind="demand",
                            top=z_top,
                            bottom=z_bot,
                            origin_bar=base_idx,
                            strength=strength,
                        ))
                        used_demand.add(base_idx)

        # ── Supply zone (impulsive DOWN) ───────────────────────────────────
        if (net_down >= cfg.impulse_atr_mult * cur_atr
                and bear_qual >= cfg.min_departure_candles):
            base_idx = _find_base(opens, closes, i, "supply", cfg.base_lookback)
            if base_idx not in used_supply:
                strength = net_down / cur_atr
                if strength >= cfg.min_strength:
                    z_top = float(highs[base_idx])
                    z_bot = float(lows[base_idx])
                    if z_top > z_bot:
                        zones.append(Zone(
                            kind="supply",
                            top=z_top,
                            bottom=z_bot,
                            origin_bar=base_idx,
                            strength=strength,
                        ))
                        used_supply.add(base_idx)

    return zones


# ---------------------------------------------------------------------------
# Freshness tracking
# ---------------------------------------------------------------------------

def update_freshness(
    zones: List[Zone],
    highs: np.ndarray,
    lows: np.ndarray,
    current_bar: int,
    tap_tol: float = 0.001,
) -> None:
    """
    In-place update: mark zones tapped if price has returned into them
    between the zone's origin bar and current_bar (inclusive).

    A tap is any bar whose high/low range overlaps the zone (wicks included).
    """
    for zone in zones:
        if not zone.fresh:
            continue
        start = zone.origin_bar + 1
        end   = min(current_bar + 1, len(highs))
        if start >= end:
            continue
        z_top = zone.top    * (1 + tap_tol)
        z_bot = zone.bottom * (1 - tap_tol)
        for j in range(start, end):
            if lows[j] <= z_top and highs[j] >= z_bot:
                zone.fresh     = False
                zone.tap_count += 1
                break


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def price_in_zone(price: float, zone: Zone, tol: float = 0.001) -> bool:
    """True if price falls inside zone (with percentage tolerance on edges)."""
    return zone.bottom * (1 - tol) <= price <= zone.top * (1 + tol)


def nearest_zones(
    zones: List[Zone],
    price: float,
    kind: Optional[str] = None,
    prefer_fresh: bool = True,
    max_results: int = 3,
) -> List[Zone]:
    """
    Return up to max_results zones sorted by proximity to price.

    Parameters
    ----------
    kind          : 'demand', 'supply', or None (both).
    prefer_fresh  : Fresh zones sorted before stale ones (within each tier,
                    sorted by distance). Disable to rank purely by distance.
    max_results   : Cap on returned zones.
    """
    pool = [z for z in zones if kind is None or z.kind == kind]
    if prefer_fresh:
        fresh = sorted(
            [z for z in pool if z.fresh],
            key=lambda z: abs(z.mid - price),
        )
        stale = sorted(
            [z for z in pool if not z.fresh],
            key=lambda z: abs(z.mid - price),
        )
        ordered = fresh + stale
    else:
        ordered = sorted(pool, key=lambda z: abs(z.mid - price))
    return ordered[:max_results]


def strongest_zones(
    zones: List[Zone],
    kind: Optional[str] = None,
    fresh_only: bool = False,
    max_results: int = 5,
) -> List[Zone]:
    """Return zones sorted by strength (impulse / ATR), strongest first."""
    pool = [z for z in zones if kind is None or z.kind == kind]
    if fresh_only:
        pool = [z for z in pool if z.fresh]
    return sorted(pool, key=lambda z: -z.strength)[:max_results]
