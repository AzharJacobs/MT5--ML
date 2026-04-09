#!/usr/bin/env python3
"""
main.py - Main Entry Point for USTech ML Engine
=================================================

This script runs all components of the ML engine in the correct order:
1. Tests database connection
2. Validates trading strategy
3. Trains or loads the ML model
4. Runs predictions and queries

Usage:
    python main.py              # Full pipeline (train + predict)
    python main.py --train      # Train model only
    python main.py --predict    # Predict only (uses saved model)
    python main.py --query      # Run sample queries
    python main.py --retrain    # Force retrain even if model exists
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional

# Local imports
from db_connect import get_connection
from strategy import validate_strategy, STRATEGY_NAME
from train_model import ModelTrainer, train_model, MODEL_DIR, MODEL_FILE
from predict import Predictor, predict, query
from prepare_data import DataPreparator


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def test_database_connection() -> bool:
    """Test database connectivity."""
    print_header("DATABASE CONNECTION TEST")

    db = get_connection()

    if db.connect():
        print(f"\nDatabase Info:")
        print(f"  Host: {db.host}")
        print(f"  Port: {db.port}")
        print(f"  Database: {db.database}")
        print(f"  User: {db.user}")

        # Get data info
        timeframes = db.get_available_timeframes()
        date_range = db.get_date_range()
        record_count = db.get_record_count()

        print(f"\nData Info:")
        print(f"  Available timeframes: {', '.join(timeframes)}")
        print(f"  Date range: {date_range['min_date']} to {date_range['max_date']}")
        print(f"  Total records: {record_count:,}")

        db.disconnect()
        return True
    else:
        print("\n✗ Database connection failed!")
        print("  Please check your .env configuration.")
        return False


def validate_trading_strategy() -> bool:
    """Validate the trading strategy configuration."""
    print_header("TRADING STRATEGY VALIDATION")

    print(f"\nStrategy: {STRATEGY_NAME}")
    return validate_strategy()


def check_model_exists() -> bool:
    """Check if a trained model already exists."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    return os.path.exists(model_path)


def run_training(
    timeframes: list = None,
    start_date: str = None,
    end_date: str = None,
    model_type: str = "random_forest",
    force_retrain: bool = False
) -> bool:
    """
    Run model training.

    Args:
        timeframes: List of timeframes to train on
        start_date: Training data start date
        end_date: Training data end date
        model_type: Type of model to train
        force_retrain: Force retrain even if model exists

    Returns:
        True if training successful
    """
    print_header("MODEL TRAINING")

    if check_model_exists() and not force_retrain:
        print("\n✓ Trained model already exists.")
        print("  Use --retrain flag to force retraining.")
        return True

    try:
        results = train_model(
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            save=True
        )

        print_section("Training Summary")
        print(f"  Model Type: {model_type}")
        print(f"  Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"  Testing Accuracy: {results['test_accuracy']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        print(f"  Cross-Validation: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")

        return True

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False


def run_predictions(timeframes: list = None) -> bool:
    """
    Run predictions using saved model.

    Args:
        timeframes: List of timeframes to predict for

    Returns:
        True if predictions successful
    """
    print_header("PREDICTIONS")

    if not check_model_exists():
        print("\n✗ No trained model found. Please train a model first.")
        return False

    try:
        predictor = Predictor()

        # If no timeframes specified, use common ones
        if timeframes is None:
            timeframes = ['1min', '5min', '1H']

        for tf in timeframes:
            print_section(f"Prediction for {tf}")
            try:
                result = predictor.predict_next_candle(tf)
                if 'error' not in result:
                    print(f"\n  Predicted Direction: {result['prediction'].upper()}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Strategy Signal: {result['strategy_signal']}")
                else:
                    print(f"  {result['error']}")
            except Exception as e:
                print(f"  Error: {e}")

        return True

    except Exception as e:
        print(f"\n✗ Predictions failed: {e}")
        return False


def run_sample_queries() -> bool:
    """Run sample analysis queries."""
    print_header("SAMPLE QUERIES")

    if not check_model_exists():
        print("\n✗ No trained model found. Please train a model first.")
        return False

    try:
        predictor = Predictor()

        # Query 1: Buy candles per day in recent month
        print_section("Query: Buy Candles Per Day (Recent Month)")
        current_date = datetime.now()
        predictor.query_buy_candles_per_day(
            timeframe='1min',
            month=current_date.month,
            year=current_date.year
        )

        # Query 2: Best trading hours
        print_section("Query: Best Trading Hours")
        predictor.query_best_trading_hours(
            timeframe='1min',
            direction='buy'
        )

        # Query 3: Best trading days
        print_section("Query: Best Trading Days of Week")
        predictor.query_best_trading_days(
            timeframe='1min',
            direction='buy'
        )

        # Query 4: Monthly patterns
        print_section("Query: Monthly Patterns")
        predictor.query_monthly_patterns(
            timeframe='1min',
            year=current_date.year
        )

        return True

    except Exception as e:
        print(f"\n✗ Queries failed: {e}")
        return False


def run_full_pipeline(
    timeframes: list = None,
    force_retrain: bool = False
) -> None:
    """Run the complete ML pipeline."""
    print_header("USTECH ML ENGINE - FULL PIPELINE")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Test database connection
    if not test_database_connection():
        print("\n✗ Pipeline aborted: Database connection failed.")
        sys.exit(1)

    # Step 2: Validate trading strategy
    if not validate_trading_strategy():
        print("\n✗ Pipeline aborted: Strategy validation failed.")
        sys.exit(1)

    # Step 3: Train model (or use existing)
    if not run_training(timeframes=timeframes, force_retrain=force_retrain):
        print("\n✗ Pipeline aborted: Training failed.")
        sys.exit(1)

    # Step 4: Run predictions
    if not run_predictions(timeframes=timeframes):
        print("\n⚠ Predictions encountered errors.")

    # Step 5: Run sample queries
    if not run_sample_queries():
        print("\n⚠ Queries encountered errors.")

    print_header("PIPELINE COMPLETE")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def interactive_mode() -> None:
    """Run in interactive mode with menu."""
    while True:
        print_header("USTECH ML ENGINE - INTERACTIVE MODE")
        print("""
Select an option:
  1. Test database connection
  2. Validate trading strategy
  3. Train model
  4. Make prediction
  5. Run queries
  6. Retrain model (force)
  7. Full pipeline
  0. Exit
        """)

        try:
            choice = input("Enter choice (0-7): ").strip()

            if choice == '0':
                print("\nGoodbye!")
                break
            elif choice == '1':
                test_database_connection()
            elif choice == '2':
                validate_trading_strategy()
            elif choice == '3':
                run_training()
            elif choice == '4':
                tf = input("Enter timeframe (default: 1min): ").strip() or '1min'
                run_predictions([tf])
            elif choice == '5':
                run_sample_queries()
            elif choice == '6':
                run_training(force_retrain=True)
            elif choice == '7':
                run_full_pipeline()
            else:
                print("Invalid choice. Please enter 0-7.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="USTech ML Engine - Machine Learning for trading predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --train            # Train model only
  python main.py --predict          # Predict using saved model
  python main.py --query            # Run analysis queries
  python main.py --retrain          # Force retrain model
  python main.py --interactive      # Interactive menu mode
  python main.py --timeframes 1min 5min 1H  # Specify timeframes
        """
    )

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the ML model'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make predictions using saved model'
    )
    parser.add_argument(
        '--query',
        action='store_true',
        help='Run sample analysis queries'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retrain even if model exists'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode with menu'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=None,
        help='Specify timeframes (e.g., --timeframes 1min 5min 1H)'
    )
    parser.add_argument(
        '--model-type',
        default='random_forest',
        choices=['random_forest', 'gradient_boosting', 'logistic'],
        help='ML model type to use'
    )

    args = parser.parse_args()

    # Interactive mode
    if args.interactive:
        interactive_mode()
        return

    # Specific actions
    if args.train:
        test_database_connection()
        validate_trading_strategy()
        run_training(
            timeframes=args.timeframes,
            force_retrain=args.retrain
        )
        return

    if args.predict:
        run_predictions(timeframes=args.timeframes)
        return

    if args.query:
        run_sample_queries()
        return

    # Default: run full pipeline
    run_full_pipeline(
        timeframes=args.timeframes,
        force_retrain=args.retrain
    )


if __name__ == "__main__":
    main()
