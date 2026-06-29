"""
Directional bias filter for the FXGold strategy.

Reads swing structure (HH/HL vs LH/LL) from H4/Daily/H1 DataFrames and
returns an overall direction that gates entry decisions in entry.py.

Reuses _find_fractals from zones.py — no separate pivot logic.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.zones import _find_fractals


def get_bias(df: pd.DataFrame, cfg: FXGoldConfig) -> str:
    """
    Determine trend bias for a single timeframe using swing structure.

    Gets the last cfg.bias_swing_count swing highs and swing lows (no
    look-ahead: uses bars up to the end of df as passed by the caller).
    Compares each consecutive pair:
      All higher highs AND all higher lows → "up"
      All lower highs  AND all lower lows  → "down"
      Anything mixed                       → "sideways"

    Args:
        df:  OHLCV DataFrame sorted ascending (only bars up to current time).
        cfg: Strategy configuration.

    Returns:
        "up", "down", or "sideways".
    """
    swing_hi, swing_lo = _find_fractals(df, window=cfg.fractal_window)

    n = cfg.bias_swing_count

    # Need at least 2 points to compare consecutive swings
    if len(swing_hi) < 2 or len(swing_lo) < 2:
        return "sideways"

    recent_hi: List[float] = [float(df["high"].iloc[i]) for i in swing_hi[-n:]]
    recent_lo: List[float] = [float(df["low"].iloc[i])  for i in swing_lo[-n:]]

    hh = all(recent_hi[i] > recent_hi[i - 1] for i in range(1, len(recent_hi)))
    lh = all(recent_hi[i] < recent_hi[i - 1] for i in range(1, len(recent_hi)))
    hl = all(recent_lo[i] > recent_lo[i - 1] for i in range(1, len(recent_lo)))
    ll = all(recent_lo[i] < recent_lo[i - 1] for i in range(1, len(recent_lo)))

    if hh and hl:
        return "up"
    if lh and ll:
        return "down"
    return "sideways"


def get_aligned_bias(
    df_d1: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    cfg: FXGoldConfig,
) -> str:
    """
    Combine bias across three timeframes according to cfg.bias_mode.

    Modes:
      "strict": D1, H4, and H1 must all return the same direction.
                Any mismatch or any "sideways" → return "sideways".
      "loose":  D1 sets the direction, H4 must agree, H1 is ignored.
                Any mismatch or either returning "sideways" → "sideways".

    Args:
        df_d1/h4/h1: OHLCV DataFrames for each timeframe, sorted ascending,
                     containing only bars up to the current evaluation point.
        cfg:         Strategy configuration.

    Returns:
        "up", "down", or "sideways".
    """
    d1 = get_bias(df_d1, cfg)
    h4 = get_bias(df_h4, cfg)

    if cfg.bias_mode == "strict":
        h1 = get_bias(df_h1, cfg)
        if d1 == h4 == h1 and d1 != "sideways":
            return d1
        return "sideways"
    elif cfg.bias_mode == "d1_only":
        if d1 != "sideways":
            return d1
        return "sideways"
    else:  # "loose"
        if d1 == h4 and d1 != "sideways":
            return d1
        return "sideways"
