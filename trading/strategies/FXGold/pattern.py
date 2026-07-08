"""
SRR / RSS "secret pattern" — Genesis FX Shadow Entry PDF, SOP 3.

A liquidity-sweep reversal read on the entry TF, checked as an extra gate
alongside the zone touch and candlestick confirmation:

  SRR (buy at a support zone):
    Price recently swept below at least `cfg.secret_pattern_min_breakouts`
    consecutive prior swing lows — sitting ABOVE the support zone, i.e. the
    levels price broke through on its way down into the zone — trapping late
    sellers before reversing back up into the zone.

  RSS (sell at a resistance zone):
    Mirror — price swept above at least `cfg.secret_pattern_min_breakouts`
    consecutive prior swing highs sitting BELOW the resistance zone, before
    reversing back down into the zone.

"Swept" = a swing pivot exists; "broken" = a later bar body-closed through
it (below for a swing low, above for a swing high).

Only bars strictly BEFORE the bar currently being evaluated for entry are
used — the confirming bar itself must never contribute to the swing/breakout
structure it is supposed to be confirming. See check_secret_pattern.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.zones import Zone, _find_fractals


def _swept_and_broken(
    df_prior: pd.DataFrame,
    side: str,           # "low" (SRR/buy) or "high" (RSS/sell)
    zone_bound: float,   # buy: zone.top (pivots must sit above it) | sell: zone.bottom (below it)
    min_breakouts: int,
    fractal_window: int,
) -> bool:
    """
    True when at least `min_breakouts` of the most recent consecutive swing
    pivots on `df_prior` (already filtered to the zone's far side) were each
    later body-broken.
    """
    n = len(df_prior)
    if n < 2 * fractal_window + 1:
        return False

    swing_hi, swing_lo = _find_fractals(df_prior, window=fractal_window)
    pivots: List[int] = swing_lo if side == "low" else swing_hi
    if not pivots:
        return False

    levels = df_prior["low" if side == "low" else "high"].to_numpy()
    closes = df_prior["close"].to_numpy()

    # Only pivots on the far side of the zone count as levels price swept
    # through on its way in — a swing low for SRR must sit above the support
    # zone; a swing high for RSS must sit below the resistance zone.
    if side == "low":
        candidates = [idx for idx in pivots if levels[idx] > zone_bound]
    else:
        candidates = [idx for idx in pivots if levels[idx] < zone_bound]

    if len(candidates) < min_breakouts:
        return False

    recent = candidates[-min_breakouts:]
    broken = 0
    for idx in recent:
        if idx + 1 >= n:
            continue
        after  = closes[idx + 1:]
        level  = levels[idx]
        hit    = (after < level).any() if side == "low" else (after > level).any()
        if hit:
            broken += 1

    return broken >= min_breakouts


def check_secret_pattern(
    df_entry_tf: pd.DataFrame,
    direction: str,
    zone: Zone,
    cfg: FXGoldConfig,
) -> bool:
    """
    SOP 3 gate — see module docstring for the SRR/RSS definition.

    Args:
        df_entry_tf: OHLCV bars at the entry TF, sorted ascending. The LAST
                     row is the bar currently being confirmed for entry and
                     is EXCLUDED before any pivot/breakout scanning — only
                     bars strictly before it are used, so the pattern can
                     never be built from the candle it is meant to confirm.
        direction:   "buy" (zone is support) or "sell" (zone is resistance).
        zone:        The Zone being evaluated — its top/bottom bounds which
                     side of the zone prior swing pivots must sit on.
        cfg:         Strategy configuration.

    Returns:
        True if the liquidity-sweep pattern is present in the bars leading
        up to (but not including) the current bar.
    """
    if len(df_entry_tf) < 2:
        return False

    df_prior = df_entry_tf.iloc[:-1].reset_index(drop=True)

    if direction == "buy":
        side, zone_bound = "low", zone.top
    else:
        side, zone_bound = "high", zone.bottom

    return _swept_and_broken(
        df_prior,
        side=side,
        zone_bound=zone_bound,
        min_breakouts=cfg.secret_pattern_min_breakouts,
        fractal_window=cfg.secret_pattern_fractal_window,
    )
