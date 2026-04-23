"""
stop_loss.py — Dynamic stop-loss management logic.
Reads zone boundaries from feature_row (produced by data/feature_engineer.py).
"""

import pandas as pd
from typing import Optional


def get_dynamic_stop_loss(
    entry_price: float,
    direction: str,
    atr: float,
    feature_row: Optional[pd.Series] = None,
    sl_buffer_atr: float = 0.5,
) -> float:
    """
    Return the stop-loss price.
    Uses HTF zone boundary when available (from feature_row), else ATR-based.
    """
    buffer = atr * sl_buffer_atr
    if direction == "buy":
        zone_low = feature_row.get("htf_demand_low") if feature_row is not None else None
        base = zone_low if zone_low else entry_price
        return base - buffer
    else:
        zone_high = feature_row.get("htf_supply_high") if feature_row is not None else None
        base = zone_high if zone_high else entry_price
        return base + buffer
