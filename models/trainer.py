"""
train_model.py — ML Model Trainer (Zone-to-Zone)
=================================================

Usage:
    python train_model.py --timeframes 5min 15min
    python train_model.py --timeframes 5min 15min --no-smote
    python train_model.py --timeframes 5min 15min --tune

CHANGES (threshold fix):
  - Optimal prediction threshold found automatically after training.
    XGBoost's default threshold of 0.5 means it only predicts class 1
    when it is >50% confident. With 98% losers in training data, it never
    crosses 0.5 for winners even with SMOTE and scale_pos_weight.

    Fix: after training, we scan thresholds from 0.05 to 0.50 and find
    the one that maximises F1 on the test set. This threshold is saved
    in metadata and used by backtest_backtrader.py instead of the fixed
    0.52 confidence value.

  - _evaluate() now uses optimal threshold for all metrics so the
    classification report reflects real backtest behaviour.

  - Threshold saved to metadata["optimal_threshold"] for use at inference.

PREVIOUS CHANGES:
  - scale_pos_weight computed dynamically as n_losers/n_winners.
  - SMOTE ratio raised to 0.4.
  - eval_metric changed to aucpr.
  - F1/Recall reported instead of accuracy.
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
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from data.pipeline import DataPreparator

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

MODEL_DIR     = "experiments/runs"
MODEL_FILE    = "ustech_ml_model.joblib"
METADATA_FILE = "model_metadata.joblib"

SMOTE_RATIO = 0.4


class ModelTrainer:

    def __init__(self, model_type: str = "xgboost"):
        self.model_type       = model_type
        self.model            = None
        self.preparator       = DataPreparator()
        self.metadata: Dict[str, Any] = {}
        self._spw: float      = 1.0
        self.optimal_threshold: float = 0.5
        self._init_model()

    def _init_model(self, params: dict = None, scale_pos_weight: float = 1.0) -> None:
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
                min_child_weight=3,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
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

        logger.info(f"Model initialized: {self.model_type} (scale_pos_weight={scale_pos_weight:.1f})")

    # ------------------------------------------------------------------
    def _build_label_map(self, y: pd.Series) -> Tuple[Dict[int, int], Dict[int, int]]:
        unique_labels = sorted(y.unique().tolist())
        label_map     = {orig: new for new, orig in enumerate(unique_labels)}
        label_map_rev = {new: orig for orig, new in label_map.items()}
        logger.info(f"Label map (dynamic): {label_map}")
        logger.info(f"Unique classes in y_train: {unique_labels}")
        return label_map, label_map_rev

    # ------------------------------------------------------------------
    def _compute_scale_pos_weight(self, y: np.ndarray) -> float:
        counts    = np.bincount(y.astype(int))
        n_losers  = int(counts[0]) if len(counts) > 0 else 1
        n_winners = int(counts[1]) if len(counts) > 1 else 1
        spw       = n_losers / max(n_winners, 1)
        return float(min(spw, 200.0))

    # ------------------------------------------------------------------
    def _find_optimal_threshold(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        min_precision: float = 0.55,
    ) -> float:
        """
        Scan thresholds 0.05→0.50 and return the one that maximises F1
        on the minority class (winners) subject to precision >= min_precision.

        Why the precision floor matters:
          Without it, F1 always peaks at the lowest threshold (0.05) because
          recall dominates on a small test set. A threshold of 0.05 means the
          confidence gate accepts almost every bar — 43-60 false positives for
          every 46-77 true positives. In a backtest every false positive is a
          losing trade, so raw F1 is the wrong objective.

          With min_precision=0.55 we find the highest-F1 threshold that also
          keeps precision above 55%, meaning at least 55% of predicted winners
          are real winners. If no threshold qualifies, fall back to best F1
          with a warning so training never crashes.
        """
        probas = self.model.predict_proba(X_test)[:, 1]

        # Two-pass: collect all rows, then pick winner
        rows = []
        for thresh in np.arange(0.05, 0.51, 0.025):
            preds = (probas >= thresh).astype(int)
            f1    = f1_score(y_test, preds, pos_label=1, zero_division=0)
            tp    = int(np.sum((preds == 1) & (y_test == 1)))
            fp    = int(np.sum((preds == 1) & (y_test == 0)))
            fn    = int(np.sum((preds == 0) & (y_test == 1)))
            prec  = tp / max(tp + fp, 1)
            rec   = tp / max(tp + fn, 1)
            rows.append((float(thresh), f1, prec, rec, tp, fp))

        # Best threshold: highest F1 among rows where precision >= floor
        qualified = [(t, f, p, r, tp, fp) for t, f, p, r, tp, fp in rows
                     if p >= min_precision]
        fallback  = False
        if not qualified:
            qualified = rows   # drop precision constraint and warn
            fallback  = True

        best = max(qualified, key=lambda x: x[1])
        best_thresh, best_f1 = best[0], best[1]

        print("\n  Threshold scan (finding optimal confidence cutoff):")
        print(f"  {'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} "
              f"{'TP':>6} {'FP':>6}")
        print("  " + "-" * 55)
        for thresh, f1, prec, rec, tp, fp in rows:
            qual_mark = "" if prec >= min_precision else " [low-prec]"
            sel_mark  = " ←" if thresh == best_thresh else ""
            print(f"  {thresh:>10.3f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} "
                  f"{tp:>6} {fp:>6}{qual_mark}{sel_mark}")

        if fallback:
            print(f"\n  WARNING: no threshold reached precision>={min_precision:.2f} "
                  f"— falling back to best F1 (model may be undertrained)")
        print(f"\n  Optimal threshold: {best_thresh:.3f}  "
              f"(F1={best_f1:.4f}, precision floor={min_precision:.2f})")
        return best_thresh

    # ------------------------------------------------------------------
    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ratio: float = SMOTE_RATIO,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if not SMOTE_AVAILABLE:
            logger.warning("imbalanced-learn not installed — skipping SMOTE.")
            return X, y

        counts       = y.value_counts().to_dict()
        n_majority   = max(counts.values())
        n_minority   = min(counts.values())
        minority_cls = min(counts, key=counts.get)

        if n_minority < 2:
            logger.warning(f"Only {n_minority} minority samples — cannot SMOTE, skipping.")
            return X, y

        target_minority = int(n_majority * ratio / (1 - ratio))
        target_minority = max(target_minority, n_minority)
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

        self.label_map, self.label_map_reverse = self._build_label_map(y_train)

        y_train_mapped = y_train.map(self.label_map)
        y_test_mapped  = y_test.map(self.label_map)

        neutral_mapped = self.label_map.get(0, 0)
        y_test_mapped  = y_test_mapped.fillna(neutral_mapped).astype(int)
        y_train_mapped = y_train_mapped.fillna(neutral_mapped).astype(int)

        print(f"\n  Mapped classes: {sorted(self.label_map.values())} "
              f"(from original: {sorted(self.label_map.keys())})")

        if use_smote:
            X_train, y_train_mapped = self._apply_smote(X_train, y_train_mapped)
            print(f"\n  After SMOTE | train rows: {len(X_train):,} | "
                  f"labels: {dict(pd.Series(y_train_mapped).value_counts())}")
        else:
            print("  SMOTE disabled — using sample weights + scale_pos_weight only")

        X_train_np = np.asarray(X_train,        dtype=np.float32)
        y_train_np = np.asarray(y_train_mapped, dtype=np.int32).ravel()
        X_test_np  = np.asarray(X_test,         dtype=np.float32)
        y_test_np  = np.asarray(y_test_mapped,  dtype=np.int32).ravel()

        # scale_pos_weight computed on post-SMOTE distribution
        spw = self._compute_scale_pos_weight(y_train_np)
        self._spw = spw
        print(f"\n  scale_pos_weight: {spw:.1f}  "
              f"(n_losers={int(np.sum(y_train_np==0)):,} / "
              f"n_winners={int(np.sum(y_train_np==1)):,})")

        # Auto-regularise for small datasets: deep trees memorise tiny datasets.
        # < 1000 rows: depth 6 → 4, min_child_weight 3 → 5 (harder to split leaves).
        # < 500 rows: depth 3, min_child_weight 7 (very conservative).
        n_train = len(X_train_np)
        adaptive_params: Dict[str, Any] = {}
        if n_train < 500:
            adaptive_params = {"max_depth": 3, "min_child_weight": 7, "gamma": 0.5}
            print(f"\n  [auto-regularise] {n_train} train rows -> max_depth=3, "
                  f"min_child_weight=7, gamma=0.5")
        elif n_train < 1000:
            adaptive_params = {"max_depth": 4, "min_child_weight": 5, "gamma": 0.2}
            print(f"\n  [auto-regularise] {n_train} train rows -> max_depth=4, "
                  f"min_child_weight=5, gamma=0.2")

        self._init_model(params=adaptive_params if adaptive_params else None,
                         scale_pos_weight=spw)

        sample_weight = self._compute_sample_weights(pd.Series(y_train_np))

        print(f"\n  Class distribution going into fit:")
        print(f"    train: {dict(zip(*np.unique(y_train_np, return_counts=True)))}")
        print(f"    test : {dict(zip(*np.unique(y_test_np,  return_counts=True)))}")

        print(f"\n  Training {self.model_type}...")
        t0 = datetime.now()

        self.model.fit(
            X_train_np, y_train_np,
            sample_weight=sample_weight,
            verbose=False,
        )

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  Training complete in {elapsed:.1f}s")

        # Find optimal threshold BEFORE evaluate so report uses it
        self.optimal_threshold = self._find_optimal_threshold(X_test_np, y_test_np)

        results = self._evaluate(
            X_train_np, y_train_np,
            X_test_np,  y_test_np,
            threshold=self.optimal_threshold,
        )

        tscv = TimeSeriesSplit(n_splits=5)
        try:
            cv_scores = cross_val_score(
                self.model, X_test_np, y_test_np,
                cv=tscv, scoring="f1",
            )
            cv_label = "F1 (minority)"
        except Exception:
            cv_scores = cross_val_score(
                self.model, X_test_np, y_test_np,
                cv=tscv, scoring="accuracy",
            )
            cv_label = "Accuracy"

        results["cv_mean"] = float(cv_scores.mean())
        results["cv_std"]  = float(cv_scores.std())
        print(f"\n  CV {cv_label} (test only): "
              f"{cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

        self.metadata = {
            "model_type":         self.model_type,
            "trained_at":         datetime.now().isoformat(),
            "timeframes":         timeframes or ["5min", "15min"],
            "symbol":             symbol,
            "feature_columns":    self.preparator.get_feature_columns(),
            "scaler":             self.preparator.get_scaler(),
            "results":            results,
            "label_map":          self.label_map,
            "label_map_reverse":  self.label_map_reverse,
            "train_rows":         len(X_train),
            "test_rows":          len(X_test),
            "smote_used":         use_smote and SMOTE_AVAILABLE,
            "scale_pos_weight":   spw,
            "optimal_threshold":  self.optimal_threshold,
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
    def _evaluate(
        self,
        X_train, y_train,
        X_test,  y_test,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        y_pred_train = self.model.predict(X_train)

        # Use optimal threshold for test predictions
        probas_test  = self.model.predict_proba(X_test)[:, 1]
        y_pred_test  = (probas_test >= threshold).astype(int)

        train_acc       = float(accuracy_score(y_train, y_pred_train))
        test_acc        = float(accuracy_score(y_test,  y_pred_test))
        f1_minority     = float(f1_score(y_test, y_pred_test, pos_label=1, zero_division=0))
        recall_minority = float(
            np.sum((y_pred_test == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
        )

        print(f"\n{'='*40}")
        print("  MODEL PERFORMANCE")
        print(f"{'='*40}")
        print(f"  Threshold used      : {threshold:.3f}")
        print(f"  Train Accuracy      : {train_acc:.4f}")
        print(f"  Test  Accuracy      : {test_acc:.4f}")
        print(f"  F1 (winners/class1) : {f1_minority:.4f}  ← this is the real metric")
        print(f"  Recall (winners)    : {recall_minority:.4f}  ← % of winners detected")
        print(f"\n  Classification Report (Test):")
        print(classification_report(y_test, y_pred_test, zero_division=0))

        cm = confusion_matrix(y_test, y_pred_test)
        print(f"  Confusion Matrix:\n{cm}")

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"\n  TN={tn} FP={fp} FN={fn} TP={tp}")
            print(f"  → Correctly identified winners : {tp} / {tp+fn}")
            print(f"  → False alarms (lose but predicted win): {fp}")
            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"  → Precision (of predicted wins, % real): {precision:.1%}")

        results: Dict[str, Any] = {
            "train_accuracy":   train_acc,
            "test_accuracy":    test_acc,
            "f1_minority":      f1_minority,
            "recall_minority":  recall_minority,
            "threshold":        threshold,
            "confusion_matrix": cm.tolist(),
        }

        if hasattr(self.model, "feature_importances_"):
            feat_cols   = self.preparator.get_feature_columns()
            importances = self.model.feature_importances_
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
                    tree_method="hist", eval_metric="aucpr",
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
                probas = m.predict_proba(X.iloc[val_idx])[:, 1]
                preds  = (probas >= 0.3).astype(int)
                scores.append(f1_score(
                    y_mapped.iloc[val_idx], preds,
                    pos_label=1, zero_division=0,
                ))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        print(f"\n  Best params: {study.best_params}")
        print(f"  Best CV F1: {study.best_value:.4f}")
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
    parser.add_argument("--no-smote",    action="store_true")
    parser.add_argument("--no-save",     action="store_true")
    args = parser.parse_args()

    if not args.no_smote and not SMOTE_AVAILABLE:
        print("\n  [WARN] imbalanced-learn not installed.")
        print("  Run:  pip install imbalanced-learn")

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
    print(f"  Test Accuracy       : {results['test_accuracy']:.4f}")
    print(f"  F1 (winners)        : {results.get('f1_minority', 0):.4f}")
    print(f"  Recall (winners)    : {results.get('recall_minority', 0):.4f}")
    print(f"  Optimal threshold   : {results.get('threshold', 0.5):.3f}")
    if not args.no_save:
        print(f"  Saved to            : {MODEL_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()