"""
pipeline_config.py — Single source of truth for all shared pipeline constants.

Every other module imports from here. No hardcoded values elsewhere.
"""

# ---------------------------------------------------------------------------
# Session hours (broker time, Exness GMT+3)
# Mirrors signal_generator.py _in_trading_session() exactly.
# ---------------------------------------------------------------------------
SESSION_HOURS_CORE = {10, 11}       # London open, always included
SESSION_HOUR_12_MAX_MINUTE = 30     # Hour 12 included only up to :30
SESSION_HOURS_NY_OPEN = {13, 14}    # NY open — high volume, added to expand signal count
SESSION_HOUR_LONDON_NY = 16         # London/NY overlap (conditional per TF)

# ---------------------------------------------------------------------------
# Zone quality
# ---------------------------------------------------------------------------
MIN_ZONE_QUALITY = 1.5              # Lowered from 3.0 — opens more valid zones

# ---------------------------------------------------------------------------
# HTF soft filter (normalised htf_4h_bias scale: -1 to +1)
# ---------------------------------------------------------------------------
HTF_EXTREME_THRESHOLD = 0.8         # |htf_4h_bias| > this → HTF is extreme

# ---------------------------------------------------------------------------
# Confidence / model
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.52  # overridden by saved optimal_threshold

# ---------------------------------------------------------------------------
# Label generation / forward simulation
# ---------------------------------------------------------------------------
MIN_RR           = 1.5    # minimum risk/reward to accept a signal
MAX_TP_ATR       = 6.0    # TP > 6 ATR is unreachable in the forward window
MAX_SL_ATR       = 3.0    # SL > 3 ATR means the zone is too wide
MAX_BARS_FORWARD = 50     # bars to simulate outcome

# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
SMOTE_RATIO = 0.4

# ---------------------------------------------------------------------------
# Required feature columns (single canonical list for the whole pipeline).
# data/feature_engineer.py imports this and re-exports it as FEATURE_COLUMNS
# so all downstream callers continue to work unchanged.
# ---------------------------------------------------------------------------
REQUIRED_FEATURE_COLUMNS = [
    # ── Strategy rule features (primary signal) ──────────────────────────
    "rule_valid_buy_setup",
    "rule_valid_sell_setup",
    "rule_htf_aligned_buy",
    "rule_htf_aligned_sell",
    "rule_htf_strong_buy",
    "rule_htf_strong_sell",
    "rule_fresh_zone",
    "rule_fresh_demand",
    "rule_fresh_supply",
    "rule_confirmed_buy",
    "rule_confirmed_sell",
    "rule_zone_has_room",
    "rule_buy_score",
    "rule_sell_score",

    # ── Zone context (supporting) ─────────────────────────────────────────
    "in_demand_zone", "in_supply_zone", "between_zones",
    "demand_zone_strength", "demand_zone_fresh", "demand_zone_touches",
    "demand_zone_consolidation",
    "supply_zone_strength", "supply_zone_fresh", "supply_zone_touches",
    "supply_zone_consolidation",
    "nearest_demand_dist_atr", "nearest_supply_dist_atr",
    "demand_zone_quality", "supply_zone_quality", "active_zone_quality",

    # ── Confirmation patterns ─────────────────────────────────────────────
    "bullish_engulfing", "bearish_engulfing",
    "pin_bar_bullish", "pin_bar_bearish",
    "higher_low", "lower_high",
    "bos_bullish", "bos_bearish",
    "buy_confirmation_score", "sell_confirmation_score",

    # ── HTF context ───────────────────────────────────────────────────────
    "htf_1h_bias", "htf_4h_bias", "htf_aligned",
    "htf_demand_zone_top", "htf_demand_zone_bottom",
    "htf_supply_zone_top", "htf_supply_zone_bottom",

    # ── Market conditions ─────────────────────────────────────────────────
    "atr_14", "rsi_14",
    "volume_ratio", "body_atr_ratio",
    "momentum_5", "momentum_10",
    "bb_position", "bb_width_atr",

    # ── Candle context ────────────────────────────────────────────────────
    "month",
    "candle_size", "body_size", "wick_upper", "wick_lower",

    # ── Session feature (model learns session importance itself) ──────────
    "in_session",

    # ── Session identity (1=London open, 2=NY open, 3=overlap, 0=off) ────
    "session_id",
]
