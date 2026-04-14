"""
train_model.py — ML Model Trainer (Zone-to-Zone)
=================================================
Trains an XGBoost classifier on Zone-to-Zone features and labels.
Saves model + metadata for use by backtest_backtrader.py and predict.py.

Usage:
    python train_model.py --timeframes 5min 15min 1H
    python train_model.py --timeframes 5min --model-type xgboost
    python train_model.py --timeframes 15min --tune
"""

import os
import argparse
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from prepare_data import DataPreparator

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

logger = logging.getLogger("mt5_collector.train_model")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

MODEL_DIR      = "models"
MODEL_FILE     = "ustech_ml_model.joblib"
METADATA_FILE  = "model_metadata.joblib"


class ModelTrainer:

    def __init__(self, model_type: str = "xgboost"):
        self.model_type  = model_type
        self.model       = None
        self.preparator  = DataPreparator()
        self.metadata: Dict[str, Any] = {}
        self._init_model()

    def _init_model(self, params: dict = None) -> None:
        if self.model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost not installed: pip install xgboost")
            default = dict(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                min_child_weight=5,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="mlogloss",
                use_label_encoder=False,
            )
            if params:
                default.update(params)
            self.model = XGBClassifier(**default)

        elif self.model_type == "catboost":
            if CatBoostClassifier is None:
                raise ImportError("catboost not installed: pip install catboost")
            default = dict(
                iterations=600,
                depth=6,
                learning_rate=0.05,
                loss_function="MultiClass",
                eval_metric="Accuracy",
                random_seed=42,
                verbose=False,
                auto_class_weights="Balanced",
            )
            if params:
                default.update(params)
            self.model = CatBoostClassifier(**default)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        logger.info(f"Model initialized: {self.model_type}")

    # ------------------------------------------------------------------
    def train(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date:   str = None,
        symbol:     str = "USTECm",
        tune:       bool = False,
        tune_trials: int = 50,
    ) -> Dict[str, Any]:

        print("\n" + "=" * 60)
        print("  ZONE-TO-ZONE ML MODEL TRAINING")
        print("=" * 60)

        # Prepare data
        X_train, y_train, raw_train, X_test, y_test, raw_test = \
            self.preparator.prepare_data(
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                symbol=symbol,
            )

        print(f"\n  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
        print(f"  Features: {len(X_train.columns)}")

        # Label distribution
        for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
            dist = y_split.value_counts().to_dict()
            print(f"  {split_name} labels: {dist}")

        # Optional hyperparameter tuning
        if tune and optuna is not None:
            logger.info("Running Optuna hyperparameter search...")
            best_params = self._tune(X_train, y_train, n_trials=tune_trials)
            self._init_model(best_params)

        # Handle class imbalance for XGBoost
        # XGBoost doesn't support multi-class scale_pos_weight directly,
        # so we use sample_weight instead
        sample_weight = self._compute_sample_weights(y_train)

        # Train
        print(f"\n  Training {self.model_type}...")
        t0 = datetime.now()

        if self.model_type == "xgboost":
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Training complete in {elapsed:.1f}s")

        # Evaluate
        results = self._evaluate(X_train, y_train, X_test, y_test)

        # Cross-validation on full dataset
        X_full = pd.concat([X_train, X_test])
        y_full = pd.concat([y_train, y_test])
        tscv   = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_full, y_full, cv=tscv, scoring="accuracy")
        results["cv_mean"] = float(cv_scores.mean())
        results["cv_std"]  = float(cv_scores.std())
        print(f"\n  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

        # Store metadata
        self.metadata = {
            "model_type":      self.model_type,
            "trained_at":      datetime.now().isoformat(),
            "timeframes":      timeframes or ["5min","15min","1H"],
            "symbol":          symbol,
            "feature_columns": self.preparator.get_feature_columns(),
            "scaler":          self.preparator.get_scaler(),
            "results":         results,
            "train_rows":      len(X_train),
            "test_rows":       len(X_test),
        }

        return results

    # ------------------------------------------------------------------
    def _compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        """Compute per-sample weights to balance class distribution."""
        counts = y.value_counts().to_dict()
        total  = len(y)
        n_cls  = len(counts)
        weight_map = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
        return np.array([weight_map[yi] for yi in y])

    # ------------------------------------------------------------------
    def _evaluate(
        self,
        X_train, y_train,
        X_test,  y_test,
    ) -> Dict[str, Any]:

        y_pred_train = self.model.predict(X_train)
        y_pred_test  = self.model.predict(X_test)

        train_acc = float(accuracy_score(y_train, y_pred_train))
        test_acc  = float(accuracy_score(y_test,  y_pred_test))

        print(f"\n{'='*40}")
        print("  MODEL PERFORMANCE")
        print(f"{'='*40}")
        print(f"  Train Accuracy : {train_acc:.4f}")
        print(f"  Test  Accuracy : {test_acc:.4f}")
        print(f"\n  Classification Report (Test):")
        print(classification_report(y_test, y_pred_test, zero_division=0))

        cm = confusion_matrix(y_test, y_pred_test)
        print(f"  Confusion Matrix:\n{cm}")

        results: Dict[str, Any] = {
            "train_accuracy": train_acc,
            "test_accuracy":  test_acc,
            "confusion_matrix": cm.tolist(),
        }

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature":    self.preparator.get_feature_columns(),
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)
            results["feature_importance"] = fi.to_dict("records")
            print(f"\n  Top 10 Features:")
            for _, row in fi.head(10).iterrows():
                print(f"    {row['feature']:35s}: {row['importance']:.4f}")

        return results

    # ------------------------------------------------------------------
    def _tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
        """Optuna hyperparameter search."""
        skf = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            if self.model_type == "xgboost":
                params = dict(
                    n_estimators      = trial.suggest_int("n_estimators", 200, 1500),
                    max_depth         = trial.suggest_int("max_depth", 3, 10),
                    learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample         = trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight  = trial.suggest_int("min_child_weight", 1, 20),
                    reg_lambda        = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                    random_state=42, n_jobs=-1,
                    tree_method="hist", eval_metric="mlogloss",
                    use_label_encoder=False,
                )
                m = XGBClassifier(**params)
            else:
                params = dict(
                    iterations    = trial.suggest_int("iterations", 200, 1500),
                    depth         = trial.suggest_int("depth", 3, 10),
                    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
                    random_seed=42, verbose=False,
                    loss_function="MultiClass",
                )
                m = CatBoostClassifier(**params)

            sw = self._compute_sample_weights(y)
            scores = []
            for tr_idx, val_idx in skf.split(X, y):
                m.fit(X.iloc[tr_idx], y.iloc[tr_idx], sample_weight=sw[tr_idx])
                scores.append(accuracy_score(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(f"\n  Best params: {study.best_params}")
        print(f"  Best CV score: {study.best_value:.4f}")
        return study.best_params

    # ------------------------------------------------------------------
    def save_model(self, model_dir: str = MODEL_DIR) -> str:
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, MODEL_FILE)
        joblib.dump(self.model, model_path)

        meta_path = os.path.join(model_dir, METADATA_FILE)
        joblib.dump(self.metadata, meta_path)

        print(f"\n  Model saved    → {model_path}")
        print(f"  Metadata saved → {meta_path}")
        return model_path

    # ------------------------------------------------------------------
    @staticmethod
    def load_model(model_dir: str = MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
        model_path = os.path.join(model_dir, MODEL_FILE)
        meta_path  = os.path.join(model_dir, METADATA_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model    = joblib.load(model_path)
        metadata = joblib.load(meta_path) if os.path.exists(meta_path) else {}
        print(f"[OK] Model loaded from {model_path}")
        return model, metadata


from typing import Any  # noqa — needed for load_model return type hint


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Zone-to-Zone ML Model")
    parser.add_argument("--timeframes",  nargs="+", default=["5min","15min","1H"])
    parser.add_argument("--start-date",  default=None)
    parser.add_argument("--end-date",    default=None)
    parser.add_argument("--symbol",      default="USTECm")
    parser.add_argument("--model-type",  default="xgboost", choices=["xgboost","catboost"])
    parser.add_argument("--tune",        action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  ZONE-TO-ZONE ML TRAINER")
    print("=" * 60)
    print(f"  Timeframes : {args.timeframes}")
    print(f"  Symbol     : {args.symbol}")
    print(f"  Model      : {args.model_type}")
    print(f"  Tune       : {args.tune}")
    print("=" * 60)

    trainer = ModelTrainer(model_type=args.model_type)
    results = trainer.train(
        timeframes=args.timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        tune=args.tune,
        tune_trials=args.tune_trials,
    )

    if not args.no_save:
        trainer.save_model()

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Test Accuracy : {results['test_accuracy']:.4f}")
    if not args.no_save:
        print(f"  Saved to      : {MODEL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()