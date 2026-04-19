"""
train_model.py — ML Model Trainer (Zone-to-Zone)
=================================================

Usage:
    python train_model.py --timeframes 5min 15min
    python train_model.py --timeframes 5min 15min --no-smote
    python train_model.py --timeframes 5min 15min --tune

CHANGES (class-imbalance fix pass):
  - SMOTE oversampling added on the training set after the label map is applied.
    SMOTE (Synthetic Minority Oversampling TEchnique) generates synthetic
    examples of the minority class (winning trades) by interpolating between
    real winners in feature space. This directly addresses the 99.9% / 0.1%
    imbalance that caused the model to predict neutral on every bar.

    Target ratio = 0.3 (winners become 30% of training data after resampling).
    This is intentionally conservative — going to 50/50 on synthetic data
    tends to overfit on the generated examples. 30% gives the model enough
    winners to learn the pattern without hallucinating too many false positives.

    SMOTE is applied to X_train only, AFTER the chronological split, so the
    test set is never touched — evaluation remains on real, unseeen data.

    install: pip install imbalanced-learn

  - --no-smote flag added if you want to compare with/without.
  - scale_pos_weight removed from XGBoost defaults (SMOTE handles balance now).
  - Dynamic label remapping retained from previous fix.
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
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

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

# SMOTE target: minority class becomes this fraction of the training set.
# 0.3 = winners become ~30% of training data after oversampling.
# SMOTE target: minority class becomes this fraction of the training set.
# 0.1 = winners become ~10% of training data after oversampling.
# Kept conservative — too many synthetic examples (0.3 was 43k fake winners
# from only 102 real ones) teaches the model patterns that don't exist in
# real market data. 0.1 gives ~11k synthetic winners which is enough for
# XGBoost to learn the pattern without hallucinating.
SMOTE_RATIO = 0.2


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
                min_child_weight=3,   # lowered from 5 — helps on smaller minority class
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="mlogloss",
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
    def _build_label_map(self, y: pd.Series) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Dynamic label map — handles missing classes gracefully."""
        unique_labels = sorted(y.unique().tolist())
        label_map     = {orig: new for new, orig in enumerate(unique_labels)}
        label_map_rev = {new: orig for orig, new in label_map.items()}
        logger.info(f"Label map (dynamic): {label_map}")
        logger.info(f"Unique classes in y_train: {unique_labels}")
        return label_map, label_map_rev

    # ------------------------------------------------------------------
    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ratio: float = SMOTE_RATIO,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Oversample the minority class using SMOTE.

        ratio = desired minority fraction after resampling.
        E.g. ratio=0.3 with 110k neutral / 300 winners →
             SMOTE generates synthetic winners until winners = 30% of total.

        k_neighbors is capped at (n_minority_samples - 1) so SMOTE never
        crashes when the minority class is very small.
        """
        if not SMOTE_AVAILABLE:
            logger.warning(
                "imbalanced-learn not installed — skipping SMOTE. "
                "Run: pip install imbalanced-learn"
            )
            return X, y

        counts      = y.value_counts().to_dict()
        n_majority  = max(counts.values())
        n_minority  = min(counts.values())
        minority_cls = min(counts, key=counts.get)

        if n_minority < 2:
            logger.warning(f"Only {n_minority} minority samples — cannot SMOTE, skipping.")
            return X, y

        # Target count for minority class
        target_minority = int(n_majority * ratio / (1 - ratio))
        target_minority = max(target_minority, n_minority)  # never shrink

        # k_neighbors must be < n_minority
        k = min(5, n_minority - 1)

        sampling_strategy = {minority_cls: target_minority}

        logger.info(
            f"SMOTE | minority class={minority_cls} "
            f"original={n_minority:,} → target={target_minority:,} "
            f"(ratio={ratio}) k_neighbors={k}"
        )

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k,
            random_state=42,
        )

        X_res, y_res = smote.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)

        logger.info(
            f"SMOTE complete | "
            f"before={len(X):,} rows → after={len(X_res):,} rows | "
            f"class dist: {dict(y_res.value_counts())}"
        )
        return X_res, y_res

    # ------------------------------------------------------------------
    def train(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date:   str = None,
        symbol:     str = "USTECm",
        tune:       bool = False,
        tune_trials: int = 50,
        use_smote:  bool = True,
    ) -> Dict[str, Any]:

        print("\n" + "=" * 60)
        print("  ZONE-TO-ZONE ML MODEL TRAINING")
        print("=" * 60)

        X_train, y_train, raw_train, X_test, y_test, raw_test = \
            self.preparator.prepare_data(
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
                symbol=symbol,
            )

        print(f"\n  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
        print(f"  Features: {len(X_train.columns)}")

        for split_name, y_split in [("Train", y_train), ("Test", y_test)]:
            dist = y_split.value_counts().to_dict()
            print(f"  {split_name} labels: {dist}")

        if tune and optuna is not None:
            logger.info("Running Optuna hyperparameter search...")
            best_params = self._tune(X_train, y_train, n_trials=tune_trials)
            self._init_model(best_params)

        # Dynamic label map
        self.label_map, self.label_map_reverse = self._build_label_map(y_train)

        y_train_mapped = y_train.map(self.label_map)
        y_test_mapped  = y_test.map(self.label_map)

        neutral_mapped = self.label_map.get(0, 0)
        y_test_mapped  = y_test_mapped.fillna(neutral_mapped).astype(int)
        y_train_mapped = y_train_mapped.fillna(neutral_mapped).astype(int)

        print(f"\n  Mapped classes: {sorted(self.label_map.values())} "
              f"(from original: {sorted(self.label_map.keys())})")

        # SMOTE — oversample minority class on training data only
        if use_smote:
            X_train, y_train_mapped = self._apply_smote(X_train, y_train_mapped)
            print(f"\n  After SMOTE | train rows: {len(X_train):,} | "
                  f"labels: {dict(pd.Series(y_train_mapped).value_counts())}")
        else:
            # Fallback: sample weights only
            print("  SMOTE disabled — using sample weights only")

        # Convert to clean numpy arrays before fitting.
        # After SMOTE, y_train_mapped may be a numpy array; y_test_mapped
        # is a pandas Series with a stale index. XGBoost infers num_class
        # from the union of train+eval labels — a misaligned index causes
        # class 1 to go missing, raising "label must be in [0, num_class)".
        X_train_np = np.asarray(X_train,        dtype=np.float32)
        y_train_np = np.asarray(y_train_mapped, dtype=np.int32).ravel()
        X_test_np  = np.asarray(X_test,         dtype=np.float32)
        y_test_np  = np.asarray(y_test_mapped,  dtype=np.int32).ravel()

        sample_weight = self._compute_sample_weights(pd.Series(y_train_np))

        print(f"\n  Class distribution going into fit:")
        print(f"    train: {dict(zip(*np.unique(y_train_np, return_counts=True)))}")
        print(f"    test : {dict(zip(*np.unique(y_test_np,  return_counts=True)))}")

        print(f"\n  Training {self.model_type}...")
        t0 = datetime.now()

        if self.model_type == "xgboost":
            # eval_set removed — it caused persistent num_class=1 errors because
            # XGBoost's internal class scanner misreads the test label array when
            # the pandas index is non-contiguous after zone-touch filtering.
            # We use fixed n_estimators=600 so early stopping isn't needed.
            self.model.fit(
                X_train_np, y_train_np,
                sample_weight=sample_weight,
                verbose=False,
            )
        else:
            self.model.fit(X_train_np, y_train_np, sample_weight=sample_weight)

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Training complete in {elapsed:.1f}s")

        results = self._evaluate(X_train_np, y_train_np, X_test_np, y_test_np)

        # CV on test only (train has synthetic rows from SMOTE, not real time series)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.model, X_test_np, y_test_np, cv=tscv, scoring="accuracy"
        )
        results["cv_mean"] = float(cv_scores.mean())
        results["cv_std"]  = float(cv_scores.std())
        print(f"\n  CV Accuracy (test only): {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

        self.metadata = {
            "model_type":        self.model_type,
            "trained_at":        datetime.now().isoformat(),
            "timeframes":        timeframes or ["5min", "15min"],
            "symbol":            symbol,
            "feature_columns":   self.preparator.get_feature_columns(),
            "scaler":            self.preparator.get_scaler(),
            "results":           results,
            "label_map":         self.label_map,
            "label_map_reverse": self.label_map_reverse,
            "train_rows":        len(X_train),
            "test_rows":         len(X_test),
            "smote_used":        use_smote and SMOTE_AVAILABLE,
        }

        return results

    # ------------------------------------------------------------------
    def _compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        counts     = y.value_counts().to_dict()
        total      = len(y)
        n_cls      = len(counts)
        weight_map = {cls: total / (n_cls * cnt) for cls, cnt in counts.items()}
        return np.array([weight_map[yi] for yi in y])

    # ------------------------------------------------------------------
    def _evaluate(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
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
            "train_accuracy":   train_acc,
            "test_accuracy":    test_acc,
            "confusion_matrix": cm.tolist(),
        }

        if hasattr(self.model, "feature_importances_"):
            feat_cols = self.preparator.get_feature_columns()
            importances = self.model.feature_importances_
            # Guard against length mismatch when SMOTE changes column count
            min_len = min(len(feat_cols), len(importances))
            fi = pd.DataFrame({
                "feature":    feat_cols[:min_len],
                "importance": importances[:min_len],
            }).sort_values("importance", ascending=False)
            results["feature_importance"] = fi.to_dict("records")
            print(f"\n  Top 10 Features:")
            for _, row in fi.head(10).iterrows():
                print(f"    {row['feature']:35s}: {row['importance']:.4f}")

        return results

    # ------------------------------------------------------------------
    def _tune(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
        skf = TimeSeriesSplit(n_splits=5)

        def objective(trial):
            if self.model_type == "xgboost":
                params = dict(
                    n_estimators     = trial.suggest_int("n_estimators", 200, 1500),
                    max_depth        = trial.suggest_int("max_depth", 3, 10),
                    learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
                    reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                    random_state=42, n_jobs=-1,
                    tree_method="hist", eval_metric="mlogloss",
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

            label_map, _ = self._build_label_map(y)
            y_mapped = y.map(label_map).fillna(0).astype(int)
            sw = self._compute_sample_weights(y_mapped)
            scores = []
            for tr_idx, val_idx in skf.split(X, y_mapped):
                m.fit(X.iloc[tr_idx], y_mapped.iloc[tr_idx],
                      sample_weight=sw[tr_idx])
                scores.append(accuracy_score(
                    y_mapped.iloc[val_idx], m.predict(X.iloc[val_idx])
                ))
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
        meta_path  = os.path.join(model_dir, METADATA_FILE)
        joblib.dump(self.model, model_path)
        joblib.dump(self.metadata, meta_path)
        print(f"\n  Model saved    → {model_path}")
        print(f"  Metadata saved → {meta_path}")
        return model_path

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


from typing import Any  # noqa


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Zone-to-Zone ML Model")
    parser.add_argument("--timeframes",  nargs="+", default=["5min", "15min"])
    parser.add_argument("--start-date",  default=None)
    parser.add_argument("--end-date",    default=None)
    parser.add_argument("--symbol",      default="USTECm")
    parser.add_argument("--model-type",  default="xgboost",
                        choices=["xgboost", "catboost"])
    parser.add_argument("--tune",        action="store_true")
    parser.add_argument("--tune-trials", type=int, default=50)
    parser.add_argument("--no-smote",    action="store_true",
                        help="Disable SMOTE oversampling (use sample weights only)")
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    if args.no_smote is False and not SMOTE_AVAILABLE:
        print("\n  [WARN] imbalanced-learn not installed.")
        print("  Run:  pip install imbalanced-learn")
        print("  Continuing with sample weights only.\n")

    print("=" * 60)
    print("  ZONE-TO-ZONE ML TRAINER")
    print("=" * 60)
    print(f"  Timeframes : {args.timeframes}")
    print(f"  Symbol     : {args.symbol}")
    print(f"  Model      : {args.model_type}")
    print(f"  Tune       : {args.tune}")
    print(f"  SMOTE      : {'disabled' if args.no_smote else 'enabled'}")
    print("=" * 60)

    trainer = ModelTrainer(model_type=args.model_type)
    results = trainer.train(
        timeframes=args.timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        symbol=args.symbol,
        tune=args.tune,
        tune_trials=args.tune_trials,
        use_smote=not args.no_smote,
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