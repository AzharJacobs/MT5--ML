"""
check_model_health.py — Inspect saved model metadata and flag retraining needs.

Usage:
    python scripts/check_model_health.py
"""

import os
import sys
import joblib
from datetime import datetime

METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "runs", "model_metadata.joblib"
)

RETRAIN_BEFORE_DATE = datetime(2025, 1, 1)
MIN_RECALL_MINORITY = 0.20


def check_model_health() -> bool:
    """
    Load model metadata and print a health report.

    Returns True if model is healthy, False if retraining is recommended.
    """
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: Model metadata not found at {METADATA_PATH}")
        print("Run: python -m models.trainer")
        return False

    metadata = joblib.load(METADATA_PATH)

    trained_at         = metadata.get("trained_at", "unknown")
    optimal_threshold  = metadata.get("optimal_threshold", "unknown")
    smote_used         = metadata.get("smote_used", "unknown")
    label_map          = metadata.get("label_map", {})
    feature_columns    = metadata.get("feature_columns", [])

    # Current trainer stores metrics in metadata["results"]
    results         = metadata.get("results", {}) or {}
    recall_minority = results.get("recall_minority")
    f1_minority     = results.get("f1_minority")

    # Fallback: legacy format stored metrics in classification_report dict
    if recall_minority is None:
        report = metadata.get("classification_report", {})
        if isinstance(report, dict):
            minority = report.get("1") or report.get(1) or {}
            recall_minority = minority.get("recall")
            f1_minority     = minority.get("f1-score")

    # Sell-wins fix check: old code stored label=-1 for sells in label_map
    sell_wins_ok = -1 not in label_map

    print("=" * 56)
    print("  MODEL HEALTH CHECK")
    print("=" * 56)
    print(f"  trained_at          : {trained_at}")
    print(f"  recall_minority     : "
          f"{recall_minority:.4f}" if recall_minority is not None else "  recall_minority     : unknown")
    print(f"  f1_minority         : "
          f"{f1_minority:.4f}" if f1_minority is not None else "  f1_minority         : unknown")
    print(f"  optimal_threshold   : {optimal_threshold}")
    print(f"  sell_wins_fix       : {'OK — binary labels (1=winner)' if sell_wins_ok else 'BROKEN — old label=-1 for sells'}")
    print(f"  smote_used          : {smote_used}")
    print(f"  feature_count       : {len(feature_columns)}")
    print("=" * 56)

    # Evaluate warnings
    needs_retrain  = False
    warning_lines  = []

    if recall_minority is not None and recall_minority < MIN_RECALL_MINORITY:
        warning_lines.append(
            f"recall_minority={recall_minority:.3f} < {MIN_RECALL_MINORITY} "
            f"— model is missing too many winners"
        )
        needs_retrain = True

    if trained_at != "unknown":
        try:
            dt = trained_at if isinstance(trained_at, datetime) \
                 else datetime.fromisoformat(str(trained_at))
            if dt < RETRAIN_BEFORE_DATE:
                warning_lines.append(
                    f"trained_at={trained_at} is before "
                    f"{RETRAIN_BEFORE_DATE.date()} — data may be stale"
                )
                needs_retrain = True
        except (ValueError, TypeError):
            pass

    if not sell_wins_ok:
        warning_lines.append(
            "label_map contains -1 key — model trained with broken sell labels; "
            "sell winners were never learned"
        )
        needs_retrain = True

    if warning_lines:
        print()
        for w in warning_lines:
            print(f"  WARNING: {w}")
        print()
        print("  MODEL NEEDS RETRAINING — run: python -m models.trainer")
        print()
        return False
    else:
        print()
        print("  Model health: OK")
        print()
        return True


if __name__ == "__main__":
    healthy = check_model_health()
    sys.exit(0 if healthy else 1)
