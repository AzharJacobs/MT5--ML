"""
train_model.py - ML Model Training Module
==========================================

Trains scikit-learn models on prepared OHLCV data with strategy signals.
Saves trained model to file for later prediction use.
"""

import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from prepare_data import DataPreparator

# Optional model deps (installed via requirements.txt updates)
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None  # type: ignore


# Model save directory
MODEL_DIR = "models"
MODEL_FILE = "ustech_ml_model.joblib"
METADATA_FILE = "model_metadata.joblib"


class ModelTrainer:
    """
    Handles ML model training, evaluation, and saving.
    """

    def __init__(self, model_type: str = "catboost"):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to use ('catboost', 'xgboost', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.preparator = DataPreparator()
        self.training_metadata: Dict[str, Any] = {}

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ML model based on model_type."""
        if self.model_type == "catboost":
            if CatBoostClassifier is None:
                raise ImportError(
                    "catboost is not installed. Install it (e.g. `pip install catboost`) "
                    "or use model_type='xgboost' / 'logistic'."
                )
            self.model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False
                # class_weights are set later after data is loaded
            )
        elif self.model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError(
                    "xgboost is not installed. Install it (e.g. `pip install xgboost`) "
                    "or use model_type='catboost' / 'logistic'."
                )
            self.model = XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss"
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(f"✓ Initialized {self.model_type} model")

    def train(
        self,
        timeframes: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        test_size: float = 0.2,
        lag_periods: int = 5
    ) -> Dict[str, Any]:
        """
        Train the ML model on historical data.

        Args:
            timeframes: List of timeframes to include in training
            start_date: Training data start date
            end_date: Training data end date
            test_size: Proportion of data for testing
            lag_periods: Number of lag periods for features

        Returns:
            Dictionary with training results and metrics
        """
        print("\n" + "=" * 60)
        print("STARTING MODEL TRAINING")
        print("=" * 60)

        # Step 1: Prepare data
        X, y, raw_data = self.preparator.prepare_data(
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            lag_periods=lag_periods
        )

        # Step 2: Chronological train/test split (no shuffling)
        print(f"\nSplitting data chronologically (test_size={test_size})...")
        if raw_data is None or 'timestamp' not in raw_data.columns:
            raise ValueError("Expected raw_data with a 'timestamp' column for chronological splitting.")

        sort_cols = ['timestamp'] + (['timeframe'] if 'timeframe' in raw_data.columns else [])
        order = raw_data.sort_values(sort_cols).index
        X = X.loc[order].reset_index(drop=True)
        y = y.loc[order].reset_index(drop=True)
        raw_data = raw_data.loc[order].reset_index(drop=True)

        split_idx = int(len(X) * (1 - test_size))
        if split_idx <= 0 or split_idx >= len(X):
            raise ValueError(f"Invalid split derived from test_size={test_size} for n={len(X)}")

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")

        # Step 3: Train model
        print(f"\nTraining {self.model_type} model...")
        start_time = datetime.now()

        # Add class weights (helps imbalance). For CatBoost, build the estimator
        # with class_weights baked in so sklearn clone() works reliably in CV.
        if self.model_type == "catboost":
            y_arr = np.asarray(y_train)
            n0 = int(np.sum(y_arr == 0))
            n1 = int(np.sum(y_arr == 1))
            if n0 == 0 or n1 == 0:
                raise ValueError(f"Cannot compute class weights (n0={n0}, n1={n1}).")
            self.model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False,
                class_weights={0: len(y_arr) / n0, 1: len(y_arr) / n1},
            )

        self.model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✓ Training completed in {training_time:.2f} seconds")

        # Step 4: Evaluate model
        print("\nEvaluating model...")
        results = self._evaluate_model(X_train, X_test, y_train, y_test)

        # Step 5: Cross-validation
        print("\nPerforming cross-validation...")
        tscv = TimeSeriesSplit(n_splits=5)
        cv_estimator = self.model
        cv_scores = cross_val_score(cv_estimator, X, y, cv=tscv, scoring='accuracy')
        results['cv_scores'] = cv_scores.tolist()
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Step 6: Store metadata
        self.training_metadata = {
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'timeframes': timeframes or ['all'],
            'date_range': {
                'start': start_date,
                'end': end_date
            },
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'feature_columns': self.preparator.get_feature_columns(),
            'lag_periods': lag_periods,
            'results': results
        }

        return results

    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on train and test sets.

        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Calculate metrics
        results = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0)
        }

        # Print results
        print(f"\n{'=' * 40}")
        print("MODEL PERFORMANCE METRICS")
        print(f"{'=' * 40}")
        print(f"Training Accuracy:  {results['train_accuracy']:.4f}")
        print(f"Testing Accuracy:   {results['test_accuracy']:.4f}")
        print(f"Precision:          {results['precision']:.4f}")
        print(f"Recall:             {results['recall']:.4f}")
        print(f"F1 Score:           {results['f1']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        results['confusion_matrix'] = cm.tolist()

        print(f"\nConfusion Matrix:")
        print(f"  Predicted Sell | Predicted Buy")
        print(f"  Actual Sell: {cm[0][0]:>6} | {cm[0][1]:>6}")
        print(f"  Actual Buy:  {cm[1][0]:>6} | {cm[1][1]:>6}")

        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.preparator.get_feature_columns(),
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            results['feature_importance'] = feature_importance.to_dict('records')

            print(f"\nTop 10 Feature Importances:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

        return results

    def save_model(self, model_dir: str = MODEL_DIR) -> str:
        """
        Save trained model and metadata to files.

        Args:
            model_dir: Directory to save model files

        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No trained model to save. Train the model first.")

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(model_dir, MODEL_FILE)
        joblib.dump(self.model, model_path)
        print(f"\n✓ Model saved to: {model_path}")

        # Save metadata and preprocessing objects
        metadata_path = os.path.join(model_dir, METADATA_FILE)
        metadata_to_save = {
            'metadata': self.training_metadata,
            'scaler': self.preparator.get_scaler(),
            'timeframe_encoder': self.preparator.get_timeframe_encoder(),
            'feature_columns': self.preparator.get_feature_columns()
        }
        joblib.dump(metadata_to_save, metadata_path)
        print(f"✓ Metadata saved to: {metadata_path}")

        return model_path

    @staticmethod
    def load_model(model_dir: str = MODEL_DIR) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model and its metadata.

        Args:
            model_dir: Directory containing model files

        Returns:
            Tuple of (model, metadata dict)
        """
        model_path = os.path.join(model_dir, MODEL_FILE)
        metadata_path = os.path.join(model_dir, METADATA_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")

        metadata = {}
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            print(f"✓ Metadata loaded from: {metadata_path}")

        return model, metadata

    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout_seconds: Optional[int] = None,
        cv_splits: int = 5,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of Optuna trials
            timeout_seconds: Optional timeout for tuning
            cv_splits: Stratified KFold splits
            random_seed: Random seed

        Returns:
            Best parameters and score
        """
        if optuna is None:
            raise ImportError(
                "optuna is not installed. Install it (e.g. `pip install optuna`) to tune hyperparameters."
            )

        print("\nPerforming hyperparameter tuning with Optuna...")
        print(f"  Model type: {self.model_type}")
        print(f"  Trials: {n_trials} | CV splits: {cv_splits} | Timeout: {timeout_seconds or 'none'}s")

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)

        def objective(trial: "optuna.Trial") -> float:
            if self.model_type == "catboost":
                if CatBoostClassifier is None:
                    raise ImportError("catboost not installed")
                params = {
                    "iterations": trial.suggest_int("iterations", 200, 1500),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                    "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "random_seed": random_seed,
                    "verbose": False,
                }
                model = CatBoostClassifier(**params)
            elif self.model_type == "xgboost":
                if XGBClassifier is None:
                    raise ImportError("xgboost not installed")
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "random_state": random_seed,
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "eval_metric": "logloss",
                }
                model = XGBClassifier(**params)
            elif self.model_type == "logistic":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
                    "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                    "max_iter": 2000,
                    "random_state": random_seed,
                    "n_jobs": -1,
                }
                model = LogisticRegression(**params)
            else:
                raise ValueError(f"Optuna tuning not configured for model type: {self.model_type}")

            scores = []
            for train_idx, valid_idx in skf.split(X, y):
                X_train_cv = X.iloc[train_idx]
                y_train_cv = y.iloc[train_idx]
                X_valid_cv = X.iloc[valid_idx]
                y_valid_cv = y.iloc[valid_idx]

                model.fit(X_train_cv, y_train_cv)
                scores.append(float(model.score(X_valid_cv, y_valid_cv)))

            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

        best_params = dict(study.best_params)
        best_score = float(study.best_value)

        print(f"\nBest parameters: {best_params}")
        print(f"Best CV score: {best_score:.4f}")

        # Rebuild model with best params
        if self.model_type == "catboost":
            self.model = CatBoostClassifier(
                **best_params,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_seed,
                verbose=False
            )
        elif self.model_type == "xgboost":
            self.model = XGBClassifier(
                **best_params,
                random_state=random_seed,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="logloss"
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                **best_params,
                max_iter=2000,
                random_state=random_seed,
                n_jobs=-1
            )

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": len(study.trials)
        }


def train_model(
    timeframes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    model_type: str = "catboost",
    save: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for training and saving a model.

    Args:
        timeframes: List of timeframes to train on
        start_date: Training data start date
        end_date: Training data end date
        model_type: Type of ML model to use
        save: Whether to save the trained model

    Returns:
        Training results dictionary
    """
    trainer = ModelTrainer(model_type=model_type)
    results = trainer.train(
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date
    )

    if save:
        trainer.save_model()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ML model on selected timeframes.")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=None,
        help="Timeframes to train on (e.g. --timeframes 5min 10min 1H). Default: all",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date filter (YYYY-MM-DD)")
    parser.add_argument(
        "--model-type",
        type=str,
        default="catboost",
        choices=["catboost", "xgboost", "logistic"],
        help="Model type to train (default: catboost)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save model artifacts")
    parser.add_argument("--lag-periods", type=int, default=5, help="Number of lag periods for features")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (default: 0.2)")

    args = parser.parse_args()

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    trainer = ModelTrainer(model_type=args.model_type)
    results = trainer.train(
        timeframes=args.timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        test_size=args.test_size,
        lag_periods=args.lag_periods,
    )

    if not args.no_save:
        trainer.save_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {results['test_accuracy']:.4f}")
    if not args.no_save:
        print(f"Model saved to: {MODEL_DIR}/{MODEL_FILE}")
