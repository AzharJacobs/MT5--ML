"""
Entry logic for the FXGold strategy.

Flow (evaluated bar-by-bar after each candle closes):
  1. Is price retesting a strong zone this bar?  (wick in, body out = rejection touch)
  2. Does the live touch count allow entry?
       live == 1  →  guide's "2nd touch"  →  TAKE
       live == 2  →  guide's "3rd touch"  →  SKIP
       live >= 3  →  guide's "4th touch"  →  SKIP (breakout expected)
  3. Is there a reversal confirmation candle at the entry TF?
  4. Compute SL = wick tip ± sl_buffer_pct.
  5. Find TP: nearest opposing strong zone (near edge).
       Buy  → bottom of nearest resistance zone above entry.
       Sell → top of nearest support zone below entry.
       No zone found → fall back to fallback_rr × sl_dist.
       Opposing zone closer than min_rr × sl_dist → skip the trade.

D1/H4 bias alignment is the CALLER's responsibility — pass only zones whose
kind already matches the bias direction.

Enter at the OPEN of the next bar (signal fires on candle close).
signal_price = current bar close is used as the entry approximation for
R:R checks; the actual entry price (next-bar open) is recorded by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from trading.strategies.FXGold.config import FXGoldConfig
from trading.strategies.FXGold.zones import Zone, price_in_zone, is_rejection_touch
from trading.strategies.FXGold.confirmations import ConfirmationResult, check_confirmation
from trading.strategies.FXGold.bias import get_aligned_bias


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class EntrySignal:
    valid: bool
    direction: Optional[str] = None   # "buy" or "sell"
    sl: Optional[float] = None        # stop-loss level
    tp: Optional[float] = None        # take-profit level
    tp_mode: Optional[str] = None     # "zone" or "fallback"
    pattern: Optional[str] = None     # confirmation pattern name
    reason: str = ""                  # acceptance or rejection reason


# ─── Touch tracker ────────────────────────────────────────────────────────────

ZoneKey = Tuple[float, float]


class TouchTracker:
    """
    Tracks live touch counts per zone across bar-by-bar scanning.
    Instantiate once per backtest run or live session; pass the same instance
    to every evaluate_entry call.

    live_touch is incremented each time a rejection touch is recorded.
    Historical rejections (used for zone grading strength) are NOT counted here.
    """

    def __init__(self) -> None:
        self._live: Dict[ZoneKey, int] = {}

    def record_touch(self, zone: Zone) -> int:
        """Increment and return the updated live touch count for this zone."""
        key = zone.zone_key
        self._live[key] = self._live.get(key, 0) + 1
        return self._live[key]

    def get_count(self, zone: Zone) -> int:
        return self._live.get(zone.zone_key, 0)

    def reset(self, zone: Zone) -> None:
        """Call this when a zone is invalidated (flipped or broken)."""
        self._live.pop(zone.zone_key, None)


# ─── TP zone search ───────────────────────────────────────────────────────────

def find_tp_zone(
    signal_price: float,
    direction: str,
    opposing_zones: List[Zone],
    sl_dist: float,
    cfg: FXGoldConfig,
) -> Tuple[float, str]:
    """
    Find the take-profit level from the nearest opposing strong zone.

    TP edge used (the side price reaches FIRST):
      Buy  → zone.bottom of the nearest resistance zone above signal_price.
      Sell → zone.top    of the nearest support zone below signal_price.

    Fallback (no zone found): signal_price ± fallback_rr × sl_dist.

    Args:
        signal_price:   Current bar close (entry approximation).
        direction:      "buy" or "sell".
        opposing_zones: All strong, non-flipped zones from both TFs.
                        Caller passes the full list; this function filters by
                        kind and position.
        sl_dist:        Distance from signal_price to SL (positive float).
        cfg:            Strategy configuration.

    Returns:
        (tp_price, mode) where mode is "zone" or "fallback".
    """
    candidates: List[Tuple[float, float]] = []  # (distance, tp_price)

    if direction == "buy":
        for z in opposing_zones:
            if z.kind == "resistance" and z.bottom > signal_price:
                candidates.append((z.bottom - signal_price, z.bottom))
    else:
        for z in opposing_zones:
            if z.kind == "support" and z.top < signal_price:
                candidates.append((signal_price - z.top, z.top))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], "zone"

    # No opposing zone found — use fallback R:R
    if direction == "buy":
        return signal_price + cfg.fallback_rr * sl_dist, "fallback"
    else:
        return signal_price - cfg.fallback_rr * sl_dist, "fallback"


# ─── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_entry(
    bar_high: float,
    bar_low: float,
    bar_close: float,
    zone: Zone,
    tracker: TouchTracker,
    df_entry_tf: pd.DataFrame,
    opposing_zones: List[Zone],
    cfg: FXGoldConfig,
    df_d1: pd.DataFrame = None,
    df_h4: pd.DataFrame = None,
    df_h1: pd.DataFrame = None,
) -> EntrySignal:
    """
    Evaluate whether the just-closed bar presents a valid entry at the given zone.
    If valid, enter at the open of the NEXT bar.

    Args:
        bar_high/low/close: OHLC of the bar that just closed.
        zone:            Strong, non-flipped Zone being tested (support or resistance).
        tracker:         Persistent TouchTracker (shared across all bars/zones).
        df_entry_tf:     OHLCV bars at the entry TF, sorted ascending.
                         Last row = the bar that just closed.
        opposing_zones:  All strong zones from both TFs (used to find TP).
        cfg:             Strategy configuration.
        df_d1/h4/h1:     OHLCV bars up to current bar for each TF (bias check).
                         All three required when cfg.bias_mode == "strict";
                         only df_d1 and df_h4 required for "loose".
                         Pass None to skip the bias gate entirely.

    Returns:
        EntrySignal — valid=True with direction, sl, tp, tp_mode, and pattern.
    """
    # Gate 0: directional bias filter (D1 / H4 / H1 structure).
    if df_d1 is not None and df_h4 is not None:
        _h1 = df_h1 if df_h1 is not None else pd.DataFrame()
        bias = get_aligned_bias(df_d1, df_h4, _h1, cfg)
        required_bias = "up" if zone.kind == "support" else "down"
        if bias == "sideways":
            if not cfg.allow_sideways_trades:
                return EntrySignal(valid=False, reason="bias_sideways_not_allowed")
        elif bias != required_bias:
            return EntrySignal(valid=False, reason=f"bias_{bias}_not_aligned")

    # Gate 1: is any part of this bar inside the zone?
    if not price_in_zone(bar_high, bar_low, zone):
        return EntrySignal(valid=False, reason="price_not_in_zone")

    # Gate 2: rejection touch — wick enters, body closes outside.
    if not is_rejection_touch(bar_high, bar_low, bar_close, zone):
        return EntrySignal(valid=False, reason="waiting_for_close_outside_zone")

    # Gate 3: touch-count filter.
    live_count = tracker.record_touch(zone)
    if live_count > cfg.max_live_touches_entry:
        reason = "skip_3rd_touch" if live_count == 2 else "breakout_expected_4th_plus"
        return EntrySignal(valid=False, reason=reason)

    # Gate 4: direction from zone kind.
    direction = "buy" if zone.kind == "support" else "sell"

    # Gate 5: reversal confirmation on the entry TF.
    conf: ConfirmationResult = check_confirmation(df_entry_tf, direction, cfg)
    if not conf.valid:
        return EntrySignal(valid=False, reason="no_confirmation_pattern")

    # Gate 6: SL = beyond the wick tip with a small percentage buffer.
    if direction == "buy":
        sl = zone.bottom * (1.0 - cfg.sl_buffer_pct)
    else:
        sl = zone.top * (1.0 + cfg.sl_buffer_pct)

    sl_dist = abs(bar_close - sl)
    if sl_dist == 0:
        return EntrySignal(valid=False, reason="zero_sl_dist")

    # Gate 7: TP from nearest opposing zone (or fallback R:R).
    tp, tp_mode = find_tp_zone(bar_close, direction, opposing_zones, sl_dist, cfg)

    rr = abs(tp - bar_close) / sl_dist
    if rr < cfg.min_rr:
        return EntrySignal(valid=False, reason=f"sub_min_rr_{rr:.2f}")

    return EntrySignal(
        valid=True,
        direction=direction,
        sl=round(sl, 5),
        tp=round(tp, 5),
        tp_mode=tp_mode,
        pattern=conf.pattern,
        reason="entry_confirmed",
    )
