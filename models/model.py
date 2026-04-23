"""
model.py — Model architecture definition.
Centralises model construction so trainer.py and evaluator.py share the same spec.
"""

from typing import Any, Dict, Optional


def build_model(params: Optional[Dict[str, Any]] = None) -> Any:
    """Return an untrained XGBClassifier with the given (or default) params."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    defaults = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": 42,
    }
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)
