"""
stop_loss.py — Dynamic stop-loss management.

Thin wrapper around strategy/base_strategy.py calculate_stop_loss() so that
risk/ and strategy/ always produce identical results for the same inputs.
The canonical logic lives in base_strategy.py.
"""

import pandas as pd
from typing import Optional

from strategy.base_strategy import calculate_stop_loss


def get_dynamic_stop_loss(
    entry_price: float,
    direction: str,
    atr: float,
    feature_row: Optional[pd.Series] = None,
    sl_buffer_atr: float = 0.5,
) -> float:
    """
    Return the stop-loss price.

    Delegates to calculate_stop_loss() (strategy/base_strategy.py) which uses
    features.py zone boundaries when available, then falls back to ATR-based.

    Args:
        entry_price:   Trade entry price.
        direction:     "buy" or "sell".
        atr:           Current ATR value (used only for the hard fallback).
        feature_row:   Raw (unscaled) pd.Series from features_by_dt["raw"].
                       Must contain demand_zone_bottom / supply_zone_top.
        sl_buffer_atr: Fraction of ATR to add beyond the zone boundary.
    """
    result = calculate_stop_loss(
        entry_price,
        direction,
        pd.DataFrame(),   # empty — only used when feature_row is None
        feature_row=feature_row,
    )
    if result is not None:
        return result
    # Hard ATR fallback when zone boundaries are unavailable
    buffer = atr * sl_buffer_atr
    return (entry_price - buffer) if direction == "buy" else (entry_price + buffer)
