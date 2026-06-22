"""
USTEC Zone-to-Zone strategy configuration.

Single source of truth for both backtest.py and live_bot_zz.py.
All symbol-specific params live in config.yaml next to this file.
Neither the backtest nor the live bot hard-codes these values directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[4]   # trading/strategies/zz/ustec → MT5--ML
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from trading.strategies.zz.core.zones import ZoneConfig
from trading.strategies.zz.core.timeframe_structure import TFConfig
from trading.strategies.zz.core.confirmations import ConfirmationConfig
from trading.strategies.zz.core.trade_setup import TradeSetupConfig


def load_raw() -> dict:
    with open(_HERE / "config.yaml") as f:
        return yaml.safe_load(f)


_cfg = load_raw()

# ── Symbol-level constants ────────────────────────────────────────────────────
SYMBOL               = str(_cfg["symbol"])
DB_NAME              = str(_cfg["db_name"])
TABLE                = str(_cfg["table"])
CONTRACT_SIZE        = float(_cfg["contract_size"])
SPREAD_PTS           = float(_cfg["spread_pts"])
FIXED_LOTS           = float(_cfg["fixed_lot"])
RISK_PCT             = float(_cfg["risk_pct"])
MIN_RR               = float(_cfg["min_rr"])
MIN_SL_PCT           = float(_cfg.get("min_sl_pct", 0.0))
MAX_SL_PCT           = float(_cfg.get("max_sl_pct", 0.0))
MAX_POSITIONS        = int(_cfg["max_positions"])
H4_WINDOW            = int(_cfg["h4_window"])
M15_WINDOW           = int(_cfg["m15_window"])

_cool                = _cfg["cooldown"]
COOLDOWN_BARS        = int(_cool["cooldown_bars"])
COOLDOWN_LOSS_H      = float(_cool["cooldown_loss_h"])
COOLDOWN_WIN_FLOOR_H = float(_cool["cooldown_win_floor_h"])
MAX_FORWARD_BARS     = int(_cool["max_forward_bars"])
ZONE_MAX_LOSSES      = int(_cool.get("zone_max_losses", 0))

TAP_TOL              = float(_cfg["zone"]["tap_tolerance_pct"])
EXCLUDED_FROM_COUNT  = list(_cfg["confirmations"].get("excluded_from_count", []))

_trail               = _cfg.get("trailing", {})
ENABLE_TRAILING      = bool(_trail.get("enable_trailing", False))
BE_TRIGGER_PTS       = float(_trail.get("be_trigger_pts", 0.0))
BE_BUFFER_PTS        = float(_trail.get("be_buffer_pts", 5.0))
ATR_TRAIL_MULT       = float(_trail.get("atr_trail_mult", 1.5))

_regime              = _cfg.get("regime", {})
H4_REGIME_FILTER     = bool(_regime.get("h4_regime_filter", False))


def make_configs() -> tuple[TFConfig, ConfirmationConfig, TradeSetupConfig]:
    """
    Build validated config objects from config.yaml.

    Called by both backtest.py and live_bot_zz.py — guarantees they use
    identical parameters and that live/backtest divergence is impossible.
    """
    _z  = _cfg["zone"]
    _tf = _cfg["timeframe"]
    _c  = _cfg["confirmations"]
    _s  = _cfg["trade_setup"]

    tf_cfg = TFConfig(
        directional_filter=bool(_tf["directional_filter"]),
        allow_neutral_up=bool(_tf["allow_neutral"]),
        allow_neutral_down=bool(_tf["allow_neutral"]),
        h4_swing_left=int(_tf.get("h4_swing_left", 2)),
        h4_swing_right=int(_tf.get("h4_swing_right", 2)),
        h4_zone_cfg=ZoneConfig(
            impulse_atr_mult=float(_z["impulse_atr_mult"]),
            body_ratio_min=float(_z["body_ratio_min"]),
            min_departure_candles=int(_z["min_departure_candles"]),
            departure_window=int(_z["departure_window"]),
            base_lookback=int(_z["base_lookback"]),
            min_strength=float(_z["min_strength"]),
            tap_tolerance_pct=float(_z["tap_tolerance_pct"]),
            max_zone_height_atr=float(_z.get("max_zone_height_atr", 0.0)),
        ),
        prefer_fresh_h4=bool(_tf.get("prefer_fresh_h4", True)),
        m15_tap_lookback=int(_tf.get("m15_tap_lookback", 20)),
        m15_tap_tolerance_pct=TAP_TOL,
        require_m15_directional_close=bool(_tf.get("require_m15_directional_close", True)),
    )
    conf_cfg = ConfirmationConfig(
        min_confirmations=int(_c["min_confirmations"]),
        excluded_from_count=list(_c.get("excluded_from_count", [])),
        aggressive_boundary=bool(_c.get("aggressive_boundary", False)),
        engulf_full_body=bool(_c.get("engulf_full_body", True)),
        wick_ratio_min=float(_c.get("wick_ratio_min", 0.60)),
        body_ratio_max=float(_c.get("body_ratio_max", 0.35)),
        close_in_upper_half=bool(_c.get("close_in_upper_half", True)),
        bos_lookback=int(_c["bos_lookback"]),
        bos_swing_n=int(_c.get("bos_swing_n", 2)),
        structure_lookback=int(_c["structure_lookback"]),
        structure_swing_n=int(_c.get("structure_swing_n", 2)),
        choch_lookback=int(_c.get("choch_lookback", 20)),
        choch_swing_left=int(_c.get("choch_swing_left", 2)),
        choch_swing_right=int(_c.get("choch_swing_right", 2)),
        choch_min_swings=int(_c.get("choch_min_swings", 2)),
        zone_tolerance_pct=TAP_TOL,
    )
    setup_cfg = TradeSetupConfig(
        aggressive_entry=bool(_s.get("aggressive_entry", False)),
        sl_buffer_pct=float(_s["sl_buffer_pct"]),
        sl_min_points=float(_s.get("sl_min_points", 0.0)),
        midline_tp=bool(_s.get("midline_tp", False)),
        midline_pct=float(_s.get("midline_pct", 0.50)),
        tp_prefer_fresh=bool(_s.get("tp_prefer_fresh", True)),
        min_rr=MIN_RR,
        use_m15_sl=bool(_s.get("use_m15_sl", False)),
        m15_sl_atr_floor_mult=float(_s.get("m15_sl_atr_floor_mult", 0.5)),
    )
    return tf_cfg, conf_cfg, setup_cfg
