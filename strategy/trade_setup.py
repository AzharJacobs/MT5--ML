"""
trade_setup.py — Step 4: trade geometry (entry, SL, TP, RR).

Builds on:
  Step 1 (zones.py)               — Zone objects
  Step 2 (timeframe_structure.py) — active_zone, h4_zones, direction
  Step 3 (confirmations.py)       — signal_price (close of confirmation bar)

Entry modes (TradeSetupConfig.aggressive_entry):
  False (default) — entry at signal bar close (confirmation price).
  True            — entry at zone boundary: top of demand zone (buy) or
                    bottom of supply zone (sell). Useful for limit orders.

Stop loss — always placed OUTSIDE the zone:
  Buy  → sl = zone.bottom * (1 - sl_buffer_pct)   below demand zone
  Sell → sl = zone.top    * (1 + sl_buffer_pct)   above supply zone

Take profit — opposite zone:
  Buy  → nearest supply zone whose bottom is above entry → tp = that zone's bottom
  Sell → nearest demand zone whose top is below entry    → tp = that zone's top
  If no opposite zone is found, the setup is invalid.

  midline_tp=True → tp = entry + midline_pct × (full_tp − entry)
  midline_pct default 0.50 (50 % of the distance to the next opposite zone).

RR validation:
  Setup is marked invalid when RR < min_rr or geometry is wrong (SL/TP cross entry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from strategy.zones import Zone


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TradeSetupConfig:
    # ── Entry mode ────────────────────────────────────────────────────────────
    aggressive_entry: bool = False
    # False → entry at confirmation bar close (realistic market order).
    # True  → entry at zone boundary (limit order at zone edge):
    #           buy  = zone.top    (price enters demand from above)
    #           sell = zone.bottom (price enters supply from below)

    # ── Stop loss ─────────────────────────────────────────────────────────────
    sl_buffer_pct: float = 0.001
    # Percentage buffer placing SL outside the zone.
    # buy  SL = zone.bottom * (1 - sl_buffer_pct)
    # sell SL = zone.top    * (1 + sl_buffer_pct)
    # NEVER inside the zone — enforced with a hard check.

    sl_min_points: float = 0.0
    # Hard minimum SL distance in price points (0 = off).
    # Widens SL if the buffer alone is too tight (e.g. on thin zones).

    # ── Take profit ───────────────────────────────────────────────────────────
    midline_tp: bool = False
    # False → TP at the near edge of the next opposite zone.
    # True  → TP at midline_pct × distance from entry to that edge.

    midline_pct: float = 0.50
    # Fraction used when midline_tp=True.  0.5 = halfway to next zone.
    # Any value in (0, 1]; values > 1 overshoot the zone (usually undesirable).

    tp_prefer_fresh: bool = True
    # When finding the TP zone, rank fresh opposite zones before stale ones.
    # Both kinds are always searched — freshness only affects tie-breaking.

    # ── Validation ────────────────────────────────────────────────────────────
    min_rr: float = 1.5
    # Minimum risk:reward.  Setup is invalid when RR < min_rr.


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TradeSetup:
    # ── Core geometry ─────────────────────────────────────────────────────────
    valid: bool          # False when geometry or RR checks fail
    direction: str       # 'buy' | 'sell'
    entry: float         # intended entry price
    sl: float            # stop loss
    tp: float            # take profit
    rr: float            # risk : reward (0.0 when invalid)
    risk: float          # absolute SL distance from entry
    reward: float        # absolute TP distance from entry

    # ── Source zones ──────────────────────────────────────────────────────────
    entry_zone: Zone            # zone we're trading FROM (demand/supply)
    tp_zone: Optional[Zone]     # opposite zone used for TP (None on invalid)

    # ── Metadata ──────────────────────────────────────────────────────────────
    entry_mode: str      # 'confirmation' | 'zone_boundary'
    tp_mode: str         # 'zone_edge' | 'midline'
    reason: str          # empty when valid; explains failure when not

    def lot_size(
        self,
        equity: float,
        risk_pct: float = 0.01,
        contract_size: float = 100.0,
    ) -> float:
        """
        Compute position size that risks risk_pct of equity given this setup's
        SL distance.  contract_size = dollar value per point per lot.
        Returns minimum 0.01 lots.
        """
        if not self.valid or self.risk <= 0:
            return 0.01
        risk_dollars = equity * risk_pct
        lot = risk_dollars / (self.risk * contract_size)
        return max(round(lot, 2), 0.01)

    def __repr__(self) -> str:
        if not self.valid:
            return f"TradeSetup(INVALID — {self.reason})"
        return (
            f"TradeSetup({self.direction} entry={self.entry:.4f} "
            f"sl={self.sl:.4f} tp={self.tp:.4f} RR={self.rr:.2f} "
            f"[{self.entry_mode}/{self.tp_mode}])"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_tp_zone(
    h4_zones: List[Zone],
    direction: str,
    entry: float,
    prefer_fresh: bool,
) -> Optional[Zone]:
    """
    Find the nearest opposite zone in the trade direction.

    Buy  → nearest supply zone whose bottom is strictly above entry.
    Sell → nearest demand zone whose top is strictly below entry.

    prefer_fresh=True sorts fresh zones before stale within the candidate pool,
    then sorts by distance so the nearest fresh zone wins.  If only stale zones
    exist they are still used.
    """
    if direction == "buy":
        candidates = [
            z for z in h4_zones
            if z.kind == "supply" and z.bottom > entry
        ]
        key = lambda z: (not (z.fresh and prefer_fresh), z.bottom - entry)
    else:
        candidates = [
            z for z in h4_zones
            if z.kind == "demand" and z.top < entry
        ]
        key = lambda z: (not (z.fresh and prefer_fresh), entry - z.top)

    if not candidates:
        return None

    return sorted(candidates, key=key)[0]


def _calc_tp_price(
    direction: str,
    entry: float,
    tp_zone: Zone,
    midline_tp: bool,
    midline_pct: float,
) -> float:
    """
    Compute TP price from the opposite zone.

    Full TP  : near edge of tp_zone (buy → zone bottom, sell → zone top)
    Midline  : entry + midline_pct × (full_tp − entry)
    """
    if direction == "buy":
        full_tp = tp_zone.bottom   # near (lower) edge of supply zone
    else:
        full_tp = tp_zone.top      # near (upper) edge of demand zone

    if midline_tp:
        return entry + midline_pct * (full_tp - entry)
    return full_tp


def _invalid(
    direction: str,
    entry_zone: Zone,
    entry_mode: str,
    reason: str,
) -> TradeSetup:
    return TradeSetup(
        valid=False, direction=direction,
        entry=0.0, sl=0.0, tp=0.0,
        rr=0.0, risk=0.0, reward=0.0,
        entry_zone=entry_zone, tp_zone=None,
        entry_mode=entry_mode, tp_mode="",
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_trade_setup(
    direction: str,
    entry_zone: Zone,
    h4_zones: List[Zone],
    signal_price: float,
    cfg: Optional[TradeSetupConfig] = None,
) -> TradeSetup:
    """
    Build a fully-validated trade setup from zone-based analysis.

    Parameters
    ----------
    direction    : 'buy' | 'sell'
    entry_zone   : The active H4 zone from analyse_timeframes (demand or supply).
    h4_zones     : Full list of H4 zones (used to locate the TP opposite zone).
    signal_price : Close price of the M15 confirmation bar (from Step 3).
                   Used as entry in confirmation mode; zone boundary overrides it
                   in aggressive mode.
    cfg          : TradeSetupConfig (defaults applied when None).

    Returns
    -------
    TradeSetup with valid=True and filled geometry, or valid=False with reason.
    """
    if cfg is None:
        cfg = TradeSetupConfig()

    # ── Entry ─────────────────────────────────────────────────────────────────
    if cfg.aggressive_entry:
        entry      = entry_zone.top    if direction == "buy" else entry_zone.bottom
        entry_mode = "zone_boundary"
    else:
        entry      = signal_price
        entry_mode = "confirmation"

    # ── Stop loss — always outside the zone ──────────────────────────────────
    if direction == "buy":
        sl = entry_zone.bottom * (1.0 - cfg.sl_buffer_pct)
        # Widen if sl_min_points requires it
        if cfg.sl_min_points > 0:
            sl = min(sl, entry - cfg.sl_min_points)
        # Hard guard: SL must be below zone bottom
        if sl >= entry_zone.bottom:
            return _invalid(direction, entry_zone, entry_mode,
                            f"SL {sl:.4f} not below zone bottom {entry_zone.bottom:.4f}")
        if sl >= entry:
            return _invalid(direction, entry_zone, entry_mode,
                            f"SL {sl:.4f} >= entry {entry:.4f}")
    else:
        sl = entry_zone.top * (1.0 + cfg.sl_buffer_pct)
        if cfg.sl_min_points > 0:
            sl = max(sl, entry + cfg.sl_min_points)
        if sl <= entry_zone.top:
            return _invalid(direction, entry_zone, entry_mode,
                            f"SL {sl:.4f} not above zone top {entry_zone.top:.4f}")
        if sl <= entry:
            return _invalid(direction, entry_zone, entry_mode,
                            f"SL {sl:.4f} <= entry {entry:.4f}")

    # ── Take profit — next opposite zone ─────────────────────────────────────
    tp_zone = _find_tp_zone(h4_zones, direction, entry, cfg.tp_prefer_fresh)
    if tp_zone is None:
        return _invalid(direction, entry_zone, entry_mode,
                        f"no opposite zone found above/below entry {entry:.4f}")

    tp_mode = "midline" if cfg.midline_tp else "zone_edge"
    tp      = _calc_tp_price(direction, entry, tp_zone, cfg.midline_tp, cfg.midline_pct)

    # ── Geometry validation ───────────────────────────────────────────────────
    if direction == "buy":
        if tp <= entry:
            return _invalid(direction, entry_zone, entry_mode,
                            f"TP {tp:.4f} not above entry {entry:.4f}")
    else:
        if tp >= entry:
            return _invalid(direction, entry_zone, entry_mode,
                            f"TP {tp:.4f} not below entry {entry:.4f}")

    risk   = abs(entry - sl)
    reward = abs(tp - entry)

    if risk <= 0:
        return _invalid(direction, entry_zone, entry_mode, "zero risk")

    rr = reward / risk
    if rr < cfg.min_rr:
        return _invalid(direction, entry_zone, entry_mode,
                        f"RR {rr:.2f} below min {cfg.min_rr}")

    return TradeSetup(
        valid=True,
        direction=direction,
        entry=round(entry, 5),
        sl=round(sl, 5),
        tp=round(tp, 5),
        rr=round(rr, 3),
        risk=round(risk, 5),
        reward=round(reward, 5),
        entry_zone=entry_zone,
        tp_zone=tp_zone,
        entry_mode=entry_mode,
        tp_mode=tp_mode,
        reason="",
    )


# ---------------------------------------------------------------------------
# Convenience: build from analyse_timeframes + check_all_confirmations output
# ---------------------------------------------------------------------------

def setup_from_analysis(
    tf_result: dict,
    signal_price: float,
    cfg: Optional[TradeSetupConfig] = None,
) -> TradeSetup:
    """
    One-call wrapper for the common backtest loop pattern.

    tf_result must be the dict returned by analyse_timeframes() and must
    already have signal != 'neutral' (caller checks this).

    Parameters
    ----------
    tf_result    : dict from analyse_timeframes with keys: direction, active_zone, h4_zones
    signal_price : close of the M15 confirmation bar (from Step 3 check)
    cfg          : TradeSetupConfig (defaults applied when None)
    """
    direction  = tf_result["direction"]
    entry_zone = tf_result["active_zone"]
    h4_zones   = tf_result["h4_zones"]

    if direction is None or entry_zone is None:
        dummy_zone = Zone(kind="demand", top=0, bottom=0, origin_bar=0, strength=0)
        return _invalid("neutral", dummy_zone, "", "no active zone from timeframe analysis")

    return build_trade_setup(direction, entry_zone, h4_zones, signal_price, cfg)
