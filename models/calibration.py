"""
Platt sigmoid calibration wrapper — kept in a standalone module so that
joblib can unpickle _PlattWrapper regardless of which script saved it.
"""
import numpy as np


class _PlattWrapper:
    """Wraps a fitted base model with a sigmoid (Logistic Regression) calibrator."""

    def __init__(self, base_model, platt_lr):
        self.base_model = base_model
        self.platt_lr   = platt_lr
        self.classes_   = getattr(base_model, "classes_", np.array([0, 1]))
        self.estimator  = base_model

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self.platt_lr.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
