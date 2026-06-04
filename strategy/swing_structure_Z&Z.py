"""
swing_structure_Z&Z.py — Fractal / pivot swing detection and HH/HL/LH/LL labeling.

Used exclusively by Step 2 (detect_h4_bias in timeframe_structure.py).
Step 3's higher_low/lower_high use a separate mechanism — do NOT merge.

detect_swings(df, left, right)
    Williams-fractal style pivot: a bar qualifies as a swing high when its
    high strictly exceeds all `left` bars before it and all `right` bars
    after it.  Mirrored for swing lows.  `right` is the only look-ahead
    window, so confirmed swings lag by exactly `right` bars.

label_structure(swings)
    Walk the list of (index, price, kind) swings in order and assign:
      swing high above prior swing high → HH,  else → LH
      swing low  above prior swing low  → HL,  else → LL
    Returns list of (index, price, label).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


# type aliases
SwingList   = List[Tuple[int, float, str]]   # (bar_index, price, 'H'|'L')
LabeledList = List[Tuple[int, float, str]]   # (bar_index, price, 'HH'|'LH'|'HL'|'LL')


def detect_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> SwingList:
    """
    Identify swing highs and lows using a fractal / pivot approach.

    A swing high at bar i requires:
      high[i] > high[i-k]  for all k in 1..left   (strictly higher than left bars)
      high[i] > high[i+k]  for all k in 1..right  (strictly higher than right bars)

    A swing low at bar i requires:
      low[i]  < low[i-k]   for all k in 1..left
      low[i]  < low[i+k]   for all k in 1..right

    Parameters
    ----------
    df    : OHLCV DataFrame with at least 'high' and 'low' columns.
    left  : bars to the left that must be lower (default 2).
    right : bars to the right that must be lower (default 2).
              The last `right` bars cannot be confirmed — no lookahead beyond them.

    Returns
    -------
    List of (bar_index, price, kind) tuples where kind is 'H' or 'L',
    ordered chronologically.  bar_index corresponds to df.index values after
    reset_index(drop=True).
    """
    df = df.reset_index(drop=True)
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)

    swings: SwingList = []

    for i in range(left, n - right):
        h = highs[i]
        l = lows[i]

        # Swing high: strictly greater than all left and right neighbours
        is_sh = (
            all(h > highs[i - k] for k in range(1, left + 1))
            and all(h > highs[i + k] for k in range(1, right + 1))
        )
        # Swing low: strictly less than all left and right neighbours
        is_sl = (
            all(l < lows[i - k] for k in range(1, left + 1))
            and all(l < lows[i + k] for k in range(1, right + 1))
        )

        if is_sh:
            swings.append((i, float(h), "H"))
        if is_sl:
            swings.append((i, float(l), "L"))

    # Sort by bar index so highs and lows interleave chronologically
    swings.sort(key=lambda x: x[0])
    return swings


def label_structure(swings: SwingList) -> LabeledList:
    """
    Walk confirmed swings chronologically and assign HH/LH/HL/LL labels.

    Labeling rules:
      swing high (H) > previous swing high → HH, else → LH
      swing low  (L) > previous swing low  → HL, else → LL

    The very first swing high is labeled HH and first swing low is labeled HL
    by convention (no prior to compare against).

    Parameters
    ----------
    swings : output of detect_swings — list of (index, price, 'H'|'L').

    Returns
    -------
    LabeledList — same structure as swings but label replaces kind.
    """
    labeled: LabeledList = []
    prev_sh_price: float | None = None
    prev_sl_price: float | None = None

    for idx, price, kind in swings:
        if kind == "H":
            if prev_sh_price is None or price > prev_sh_price:
                label = "HH"
            else:
                label = "LH"
            prev_sh_price = price
            labeled.append((idx, price, label))

        else:  # 'L'
            if prev_sl_price is None or price > prev_sl_price:
                label = "HL"
            else:
                label = "LL"
            prev_sl_price = price
            labeled.append((idx, price, label))

    return labeled


def get_bias(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> str:
    """
    One-call convenience: detect swings → label → return bias string.

    Returns 'bullish' | 'bearish' | 'neutral_up' | 'neutral_down' | 'neutral'.
    Intended for quick inspection / unit tests.
    """
    swings  = detect_swings(df, left=left, right=right)
    labeled = label_structure(swings)

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
    if last_sh_lbl == "HH":
        return "neutral_up"
    if last_sl_lbl == "HL":
        return "neutral_up"
    if last_sh_lbl == "LH":
        return "neutral_down"
    # last_sl_lbl == "LL"
    return "neutral_down"
