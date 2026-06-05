"""
config_ZZ.py — Gold-specific (XAUUSD) Zone-to-Zone strategy configuration.

Single source of truth for all three gold fixes and every tunable parameter.
The engine (engine_ZZ.py) reads exclusively from GoldZZConfig; nothing is
hard-coded in the loop.

Fix summary
-----------
Fix 1  failed_zone_filter   — zones that produce a SL are permanently blacklisted.
Fix 2  active_signals        — only engulfing / rejection_wick / choch gate entry;
                               structure_shift and bos_msb still fire and are logged.
Fix 3a min_zone_atr_frac     — skip zones narrower than this fraction of H4 ATR.
Fix 3b sl_atr_buffer         — widen SL by this fraction of H4 ATR beyond zone edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass
class GoldZZConfig:
    # ── Fix 1: Post-loss zone blacklist ───────────────────────────────────────
    failed_zone_filter: bool = True
    # True  → a zone that hits SL is added to failed_zones (permanent blacklist).
    #         No re-entry ever. Gold momentum breaks rarely recover to the same zone.
    # False → legacy 48-bar cooldown then re-eligible (USTECH default).

    # ── Fix 2: Active confirmation signals ────────────────────────────────────
    active_signals: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"engulfing", "rejection_wick", "choch"})
    )
    # Only these signal names count toward min_confirmations for entry.
    # Disabled: "structure_shift" (53% fire-rate, 24% WR, -$3,890 net),
    #           "bos_msb"         (31% fire-rate, 26% WR, -$1,017 net).
    # All signals are still computed and logged in the trade record for analysis.
    min_confirmations: int = 1

    # ── Fix 3a: ATR-based thin-zone filter ────────────────────────────────────
    min_zone_atr_frac: float = 0.30
    # Skip the active zone if (top − bottom) < min_zone_atr_frac × H4_ATR(14).
    # Gold H4 ATR ≈ $40–$90 at current prices:
    #   0.30 → minimum zone ≈ $12–$27  (filters ~40% of zones at baseline median $10.47)
    #   0.20 → minimum zone ≈ $8–$18   (softer; use if trade count drops too far)
    #   0.40 → minimum zone ≈ $16–$36  (tighter; use if SL spikes persist)

    # ── Fix 3b: ATR-based SL buffer ───────────────────────────────────────────
    sl_atr_buffer: float = 0.20
    # After setup_from_analysis() computes SL, widen sl_dist by sl_atr_buffer × H4_ATR.
    # At H4 ATR ≈ $50: adds ~$10 buffer on top of the base sl_buffer_pct margin.
    # Set to 0.0 to use only the original percentage buffer.
    # Both fix 3a and fix 3b use the same H4_ATR(14) computed once per bar.

    # ── Step 2: H4 / M15 timeframe config ────────────────────────────────────
    directional_filter: bool = True
    allow_neutral: bool = True

    # Zone detection thresholds (fed to ZoneConfig)
    zone_impulse_atr_mult: float = 2.0
    zone_body_ratio_min: float = 0.50
    zone_min_departure_candles: int = 2
    zone_departure_window: int = 6
    zone_base_lookback: int = 5
    zone_min_strength: float = 1.5   # ATR-height filter is separate; keep this at 1.5

    # ── Step 3: Confirmation sub-config ──────────────────────────────────────
    aggressive_boundary: bool = False
    bos_lookback: int = 15
    structure_lookback: int = 25

    # ── Step 4: Trade setup ───────────────────────────────────────────────────
    aggressive_entry: bool = False
    midline_tp: bool = False
    midline_pct: float = 0.50
    sl_buffer_pct: float = 0.002   # base % buffer; sl_atr_buffer adds on top

    # ── Re-entry / cooldown ───────────────────────────────────────────────────
    require_leave_and_return: bool = True
    cooldown_bars: int = 15
    # After a win: leave-and-return state machine + cooldown_bars floor.
    # After a loss with failed_zone_filter=True: zone permanently blacklisted
    #   (cooldown_bars / COOLDOWN_LOSS are irrelevant for that zone).

    # ── Simulation ────────────────────────────────────────────────────────────
    max_forward_bars: int = 350    # 350 × 15 min ≈ 87 h ≈ 4 trading days
    spread: float = 0.0
    fixed_lot: float = 0.0         # 0 = dynamic 1% risk sizing; >0 = fixed lot
    min_rr: float = 1.5
