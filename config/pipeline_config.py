"""
pipeline_config.py — Single source of truth for all shared pipeline constants.
"""

# ---------------------------------------------------------------------------
# Confidence / model
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD = 0.52

# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
SMOTE_RATIO = 0.4

# ---------------------------------------------------------------------------
# Legacy constants kept for backtest/engine.py compatibility.
# Will be replaced when engine_Z&Z.py is built.
# ---------------------------------------------------------------------------
MIN_ZONE_QUALITY    = 1.5
HTF_EXTREME_THRESHOLD = 0.8
MIN_RR              = 1.5

# ---------------------------------------------------------------------------
# Required feature columns — will be populated by the new Z&Z pipeline.
# data/feature_engineer.py re-exports this as FEATURE_COLUMNS so existing
# callers remain importable without error.
# ---------------------------------------------------------------------------
REQUIRED_FEATURE_COLUMNS = []
