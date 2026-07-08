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


def get_bias_ema(df: pd.DataFrame, period: int) -> str:
    """
    Softened bias read: last close vs an EMA, rather than requiring clean
    swing structure.

    Diagnosed against live data: swing-structure get_bias() on D1 read
    "sideways" 50.3% of the time across a 1440-candidate-touch sample
    (2024-01 to 2025-06, XAUUSD), starving the bias gate of any directional
    D1 read at all — "sideways" almost never occurs here (only an exact
    close == EMA tie), so this is the default D1 read (cfg.d1_bias_method).

    Args:
        df:     OHLCV DataFrame sorted ascending.
        period: EMA span, in bars of df's timeframe.

    Returns:
        "up", "down", or "sideways" (only on an exact tie, or too few bars).
    """
    if len(df) < period:
        return "sideways"
    ema = df["close"].ewm(span=period, adjust=False).mean()
    last_close = float(df["close"].iloc[-1])
    last_ema   = float(ema.iloc[-1])
    if last_close > last_ema:
        return "up"
    if last_close < last_ema:
        return "down"
    return "sideways"


def d1_bias(df_d1: pd.DataFrame, cfg: FXGoldConfig) -> str:
    """Dispatch D1's bias read per cfg.d1_bias_method — used by
    get_aligned_bias so all three bias_modes share the same (softened, by
    default) D1 read. Public so other callers (e.g. the engine, when logging
    a trade's individual D1/H4 bias readings) can use the same dispatch."""
    if cfg.d1_bias_method == "ema":
        return get_bias_ema(df_d1, cfg.d1_bias_ema_period)
    return get_bias(df_d1, cfg)


def get_aligned_bias(
    df_d1: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    cfg: FXGoldConfig,
) -> str:
    """
    Combine bias across three timeframes according to cfg.bias_mode.
    D1's own read is controlled separately by cfg.d1_bias_method (see
    d1_bias) — "ema" (default, softened) or "swing" (old swing-structure
    read, get_bias) — independent of which of the three modes below is active.

    Modes:
      "strict": D1, H4, and H1 must all return the same direction.
                Any mismatch or any "sideways" → return "sideways".
      "loose":  D1 sets the direction. H4 only BLOCKS when it actively
                opposes D1 (a different non-sideways direction) — H4 ==
                "sideways" no longer blocks, it simply doesn't confirm.
                D1 "sideways" always blocks. H1 is ignored.
                (Diagnosed against live data: requiring exact D1==H4
                agreement passed only 24.3% of candidates where D1 itself
                had a directional read, because H4 was "sideways" — not
                opposing — 42.7% of the time. This rule change alone lifts
                that pass rate to 67% while still blocking genuine
                opposition, which was a real, comparatively rare signal —
                16.4% of all candidate touches, and e.g. 79% of D1=down
                candidates had H4 actively reading up.)
      "d1_only": D1 alone, H4/H1 ignored entirely — kept for experiments.

    Args:
        df_d1/h4/h1: OHLCV DataFrames for each timeframe, sorted ascending,
                     containing only bars up to the current evaluation point.
        cfg:         Strategy configuration.

    Returns:
        "up", "down", or "sideways".
    """
    d1 = d1_bias(df_d1, cfg)
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
        if d1 == "sideways":
            return "sideways"
        if h4 == d1 or h4 == "sideways":
            return d1
        return "sideways"   # h4 actively opposes d1
