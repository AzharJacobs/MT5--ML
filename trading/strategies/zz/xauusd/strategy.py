"""
XAUUSD (Gold) Zone-to-Zone strategy configuration.

Single source of truth for both backtest.py and the gold live bot.
All symbol-specific params — including the three gold fixes — live in
config.yaml next to this file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet

import yaml

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[4]   # trading/strategies/zz/xauusd → MT5--ML
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class GoldZZConfig:
    """Gold (XAUUSD) Zone-to-Zone strategy configuration."""

    # ── Fix 1: Post-loss zone blacklist ───────────────────────────────────────
    failed_zone_filter: bool = True
    # True  → a zone that hits SL is permanently blacklisted (gold behaviour).
    # False → legacy 48-bar cooldown then re-eligible (USTEC default).

    # ── Fix 2: Active confirmation signals ────────────────────────────────────
    active_signals: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"engulfing", "rejection_wick", "choch"})
    )
    min_confirmations: int = 1

    # ── Fix 3a: ATR-based thin-zone filter ────────────────────────────────────
    min_zone_atr_frac: float = 0.30

    # ── Fix 3b: ATR-based SL buffer ───────────────────────────────────────────
    sl_atr_buffer: float = 0.20

    # ── Step 2: H4 / M15 timeframe config ────────────────────────────────────
    directional_filter: bool = True
    allow_neutral: bool = True

    # Zone detection thresholds (fed to ZoneConfig)
    zone_impulse_atr_mult: float = 2.0
    zone_body_ratio_min: float = 0.50
    zone_min_departure_candles: int = 2
    zone_departure_window: int = 6
    zone_base_lookback: int = 5
    zone_min_strength: float = 1.5

    # ── Step 3: Confirmation sub-config ──────────────────────────────────────
    aggressive_boundary: bool = False
    bos_lookback: int = 15
    structure_lookback: int = 25

    # ── Step 4: Trade setup ───────────────────────────────────────────────────
    aggressive_entry: bool = False
    midline_tp: bool = False
    midline_pct: float = 0.50
    sl_buffer_pct: float = 0.002

    # ── Re-entry / cooldown ───────────────────────────────────────────────────
    require_leave_and_return: bool = True
    cooldown_bars: int = 15

    # ── Simulation ────────────────────────────────────────────────────────────
    max_forward_bars: int = 350
    spread: float = 0.0
    fixed_lot: float = 0.0
    min_rr: float = 1.5


def load_raw() -> dict:
    with open(_HERE / "config.yaml") as f:
        return yaml.safe_load(f)


_cfg = load_raw()

# ── Symbol-level constants ────────────────────────────────────────────────────
SYMBOL           = str(_cfg["symbol"])
DB_NAME          = str(_cfg["db_name"])
TABLE            = str(_cfg["table"])
CONTRACT_SIZE    = float(_cfg["contract_size"])
SPREAD_PTS       = float(_cfg["spread_pts"])
FIXED_LOTS       = float(_cfg["fixed_lot"])
RISK_PCT         = float(_cfg["risk_pct"])
MIN_RR           = float(_cfg["min_rr"])
MAX_POSITIONS    = int(_cfg["max_positions"])
H4_WINDOW        = int(_cfg["h4_window"])
M15_WINDOW       = int(_cfg["m15_window"])

_cool            = _cfg["cooldown"]
COOLDOWN_BARS    = int(_cool["cooldown_bars"])
COOLDOWN_LOSS_BARS = int(_cool["cooldown_loss_bars"])
MAX_FORWARD_BARS = int(_cool["max_forward_bars"])

TAP_TOL          = float(_cfg["zone"]["tap_tolerance_pct"])


def make_gold_config() -> GoldZZConfig:
    """
    Build GoldZZConfig from config.yaml.

    Called by both backtest.py and the gold live bot — guarantees identical
    parameters and eliminates live/backtest divergence.
    """
    _gf  = _cfg["gold_fixes"]
    _c   = _cfg["confirmations"]
    _tf  = _cfg["timeframe"]
    _z   = _cfg["zone"]
    _s   = _cfg["trade_setup"]
    _cool = _cfg["cooldown"]

    return GoldZZConfig(
        # Gold fixes
        failed_zone_filter=bool(_gf["failed_zone_filter"]),
        active_signals=frozenset(_c["active_signals"]),
        min_zone_atr_frac=float(_gf["min_zone_atr_frac"]),
        sl_atr_buffer=float(_gf["sl_atr_buffer"]),
        # Confirmations
        min_confirmations=int(_c["min_confirmations"]),
        aggressive_boundary=bool(_c.get("aggressive_boundary", False)),
        bos_lookback=int(_c["bos_lookback"]),
        structure_lookback=int(_c["structure_lookback"]),
        # Timeframe
        directional_filter=bool(_tf["directional_filter"]),
        allow_neutral=bool(_tf["allow_neutral"]),
        # Zone detection
        zone_impulse_atr_mult=float(_z["impulse_atr_mult"]),
        zone_body_ratio_min=float(_z["body_ratio_min"]),
        zone_min_departure_candles=int(_z["min_departure_candles"]),
        zone_departure_window=int(_z["departure_window"]),
        zone_base_lookback=int(_z["base_lookback"]),
        zone_min_strength=float(_z["min_strength"]),
        # Trade setup
        aggressive_entry=bool(_s.get("aggressive_entry", False)),
        sl_buffer_pct=float(_s["sl_buffer_pct"]),
        midline_tp=bool(_s.get("midline_tp", False)),
        midline_pct=float(_s.get("midline_pct", 0.50)),
        # Cooldown
        require_leave_and_return=bool(_cool["require_leave_and_return"]),
        cooldown_bars=int(_cool["cooldown_bars"]),
        max_forward_bars=int(_cool["max_forward_bars"]),
        # Execution
        spread=float(_cfg["spread_pts"]),
        fixed_lot=float(_cfg["fixed_lot"]),
        min_rr=float(_cfg["min_rr"]),
    )
