USTech ML Engine
A complete Machine Learning engine for predicting trading directions using historical OHLCV data from PostgreSQL. This engine connects directly to your database, learns from historical patterns, incorporates your personal trading strategy, and predicts the direction of the next candle.

Features
Direct PostgreSQL Connection: All data comes directly from the database, never from files
Multi-Timeframe Support: Train and predict across all available timeframes (1min to 1D)
Personal Strategy Integration: Your trading strategy is applied during training and prediction
Pattern Recognition: Learns time-based patterns (hours, days, months)
Flexible Queries: Answer questions like "How many buy candles in February 2025?"
Model Persistence: Save trained models to avoid retraining every time
Retraining Support: Easily retrain when strategy or data changes
Project Structure
ustech_ml/
├── .env                # Database credentials (configure this!)
├── requirements.txt    # Python dependencies
├── README.md           # This documentation
├── db_connect.py       # PostgreSQL connection handler
├── prepare_data.py     # Data fetching and feature engineering
├── strategy.py         # YOUR PERSONAL TRADING STRATEGY (edit this!)
├── train_model.py      # ML model training and saving
├── predict.py          # Load model and make predictions
├── main.py             # Main entry point (run this!)
└── models/             # Saved model files (created automatically)
    ├── ustech_ml_model.joblib
    └── model_metadata.joblib
Installation
1. Install Python Dependencies
cd ustech_ml

pip install -r requirements.txt
Or using pip3:

pip3 install -r requirements.txt
2. Configure Database Connection
Edit the .env file with your PostgreSQL credentials:

DB_HOST=localhost
DB_PORT=5432
DB_NAME=ustech_data
DB_USER=postgres
DB_PASSWORD=your_password_here
3. Verify Database Connection
python db_connect.py
This will test the connection and display available data information.

Adding Your Trading Strategy
The most important file is strategy.py. This is where you define YOUR personal trading rules.

How to Add Your Strategy
Open strategy.py
Find the apply_strategy() function
Modify the rules to match your trading strategy
The function must return: 'buy', 'sell', or 'neutral'
Available Data for Your Strategy
Each candle provides these fields:

open, high, low, close: Price data
volume: Trading volume
direction: Actual candle direction ('buy' or 'sell')
candle_size: Total size (high - low)
body_size: Body size (abs(close - open))
wick_upper, wick_lower: Wick sizes
hour: Hour of day (0-23)
day_of_week: Day (0=Monday, 6=Sunday)
month, year: Date components
Strategy Example
def apply_strategy(current_candle, lookback_data):

    """Your custom trading strategy."""


    # Example: Buy on bullish engulfing pattern

    if lookback_data is not None and len(lookback_data) >= 1:

        prev = lookback_data.iloc[-1]


        # Bullish engulfing

        if (prev['close'] < prev['open'] and  # Previous was bearish

            current_candle['close'] > current_candle['open'] and  # Current is bullish

            current_candle['open'] < prev['close'] and  # Opens below prev close

            current_candle['close'] > prev['open']):  # Closes above prev open

            return 'buy'


        # Bearish engulfing

        if (prev['close'] > prev['open'] and

            current_candle['close'] < current_candle['open'] and

            current_candle['open'] > prev['close'] and

            current_candle['close'] < prev['open']):

            return 'sell'


    return 'neutral'
Strategy Configuration
At the top of strategy.py, you can enable/disable strategy components:

LOOKBACK_PERIODS = 5           # How many previous candles to consider

USE_CANDLESTICK_PATTERNS = True  # Enable pattern recognition

USE_TIME_FILTERS = True          # Enable time-based filters

USE_MOMENTUM_RULES = True        # Enable momentum detection

USE_VOLUME_FILTERS = True        # Enable volume confirmation
Validate Your Strategy
python strategy.py
This tests your strategy with dummy data to ensure it works.

Training the Model
Full Pipeline (Recommended)
python main.py
This will:

Test database connection
Validate your strategy
Train the model (or use existing)
Run predictions
Run sample queries
Train Only
python main.py --train
Force Retrain (After Strategy Changes)
python main.py --retrain
Train with Specific Timeframes
python main.py --train --timeframes 1min 5min 1H
Choose Model Type
python main.py --train --model-type gradient_boosting
Available models:

random_forest (default, recommended)
gradient_boosting
logistic
Running Predictions
Predict Using Saved Model
python main.py --predict
Predict for Specific Timeframes
python main.py --predict --timeframes 1min 5min
Direct Prediction Script
python predict.py
Programmatic Prediction
from predict import Predictor


predictor = Predictor()

result = predictor.predict_next_candle('1min')


print(f"Prediction: {result['prediction']}")

print(f"Confidence: {result['confidence']:.2%}")

print(f"Strategy Signal: {result['strategy_signal']}")
Running Queries
Sample Queries
python main.py --query
Query: Buy Candles Per Day
from predict import Predictor


predictor = Predictor()


# How many buy candles per day in February 2025 on 1min timeframe?

predictor.query_buy_candles_per_day(

    timeframe='1min',

    month=2,

    year=2025

)
Query: Best Trading Hours
predictor.query_best_trading_hours(

    timeframe='1min',

    direction='buy'  # or 'sell'

)
Query: Best Trading Days of Week
predictor.query_best_trading_days(

    timeframe='1min',

    direction='buy'

)
Query: Monthly Patterns
predictor.query_monthly_patterns(

    timeframe='1min',

    year=2025

)
Retraining the Model
You should retrain when:

Strategy changes: You modified strategy.py
New data available: Fresh data added to database
Performance degradation: Model accuracy decreases
How to Retrain
# Force retrain with all data

python main.py --retrain


# Retrain with specific timeframes

python main.py --retrain --timeframes 1min 5min 1H


# Retrain with different model type

python main.py --retrain --model-type gradient_boosting
Programmatic Retraining
from train_model import train_model


# Retrain with current settings

results = train_model(

    timeframes=['1min', '5min', '1H'],

    model_type='random_forest',

    save=True

)


print(f"New accuracy: {results['test_accuracy']:.4f}")
Interactive Mode
For easier exploration:

python main.py --interactive
This provides a menu-driven interface for all operations.

Understanding the Output
Training Output
MODEL PERFORMANCE METRICS
========================================
Training Accuracy:  0.8234
Testing Accuracy:   0.7891
Precision:          0.7856
Recall:             0.8012
F1 Score:           0.7933

Confusion Matrix:
  Predicted Sell | Predicted Buy
  Actual Sell:   4521 |    678
  Actual Buy:     892 |   4109

Top 10 Feature Importances:
  strategy_signal_encoded: 0.1245
  price_change_pct: 0.0987
  body_to_range_ratio: 0.0876
  ...
Prediction Output
PREDICTION RESULT
========================================
Predicted Direction: BUY
Confidence: 73.45%
Strategy Signal: buy
Buy Probability: 73.45%
Sell Probability: 26.55%
Troubleshooting
Database Connection Failed
✗ Database connection failed
Solution: Check your .env file:

Verify host, port, database name
Ensure PostgreSQL is running
Check username and password
No Data Found
No data found for timeframe 1min
Solution: Verify data exists in database:

SELECT COUNT(*) FROM ustech_ohlcv WHERE timeframe = '1min';
Model Not Found
No trained model found. Please train a model first.
Solution: Train a model first:

python main.py --train
Strategy Validation Failed
✗ Strategy validation failed
Solution: Check your strategy.py:

Ensure apply_strategy() returns 'buy', 'sell', or 'neutral'
Fix any syntax errors
Database Schema Expected
The ML engine expects this table structure:

CREATE TABLE ustech_ohlcv (

    id SERIAL PRIMARY KEY,

    symbol VARCHAR(50),

    timeframe VARCHAR(10),

    timestamp TIMESTAMP,

    date DATE,

    time TIME,

    hour INTEGER,

    day_of_week INTEGER,

    month INTEGER,

    year INTEGER,

    open DECIMAL,

    high DECIMAL,

    low DECIMAL,

    close DECIMAL,

    volume DECIMAL,

    direction VARCHAR(10),  -- 'buy' or 'sell'

    candle_size DECIMAL,

    body_size DECIMAL,

    wick_upper DECIMAL,

    wick_lower DECIMAL

);
Performance Tips
Start with 1min data: It has the most samples for better learning
Use feature importance: Focus on features that matter most
Validate strategy first: Ensure your rules make sense historically
Cross-validate: Check CV scores, not just test accuracy
Retrain periodically: Market conditions change over time
Example Workflow
# 1. Configure database

nano .env


# 2. Test connection

python db_connect.py


# 3. Add your strategy

nano strategy.py


# 4. Validate strategy

python strategy.py


# 5. Train model

python main.py --train


# 6. Make predictions

python main.py --predict


# 7. After strategy changes, retrain

python main.py --retrain
API Reference
predict.py
# Make a prediction

from predict import predict, Predictor


# Quick prediction

result = predict(timeframe='1min')


# Full predictor with all methods

predictor = Predictor()

result = predictor.predict_next_candle('1min')

counts = predictor.query_buy_candles_per_day('1min', month=2, year=2025)

hours = predictor.query_best_trading_hours('1min', direction='buy')

days = predictor.query_best_trading_days('1min', direction='buy')

monthly = predictor.query_monthly_patterns('1min', year=2025)
train_model.py
from train_model import train_model, ModelTrainer


# Quick training

results = train_model(timeframes=['1min'], save=True)


# Full trainer with hyperparameter tuning

trainer = ModelTrainer(model_type='random_forest')

results = trainer.train(timeframes=['1min'])

tuning_results = trainer.hyperparameter_tuning(X, y)

trainer.save_model()
License
This ML engine is provided for personal trading analysis. Use at your own risk.

Remember: Past performance does not guarantee future results. Always validate predictions before making trading decisions.