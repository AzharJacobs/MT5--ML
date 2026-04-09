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
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from prepare_data import DataPreparator


# Model save directory
MODEL_DIR = "models"
MODEL_FILE = "ustech_ml_model.joblib"
METADATA_FILE = "model_metadata.joblib"


class ModelTrainer:
    """
    Handles ML model training, evaluation, and saving.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.preparator = DataPreparator()
        self.training_metadata: Dict[str, Any] = {}

        # Initialize model based on type
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ML model based on model_type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
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

        # Step 2: Split data into training and testing sets
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")

        # Step 3: Train model
        print(f"\nTraining {self.model_type} model...")
        start_time = datetime.now()

        self.model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✓ Training completed in {training_time:.2f} seconds")

        # Step 4: Evaluate model
        print("\nEvaluating model...")
        results = self._evaluate_model(X_train, X_test, y_train, y_test)

        # Step 5: Cross-validation
        print("\nPerforming cross-validation...")
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
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
        param_grid: Dict[str, List] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Dictionary of parameters to search

        Returns:
            Best parameters and score
        """
        if param_grid is None:
            if self.model_type == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [5, 10, 15]
                }
            elif self.model_type == "gradient_boosting":
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1, 0.15]
                }
            else:
                param_grid = {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'saga']
                }

        print(f"\nPerforming hyperparameter tuning...")
        print(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }


def train_model(
    timeframes: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    model_type: str = "random_forest",
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
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # Train model with all available data
    results = train_model(
        timeframes=None,  # Use all timeframes
        start_date=None,  # Use all available data
        end_date=None,
        model_type="random_forest",
        save=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Model saved to: {MODEL_DIR}/{MODEL_FILE}")
