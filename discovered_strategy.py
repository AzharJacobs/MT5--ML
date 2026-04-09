import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
from db_connect import get_connection

TIMEFRAMES = ['1min','2min','3min','4min','5min','10min','15min','30min','1H','4H','1D']

FEATURES = [
    'hour', 'day_of_week', 'month',
    'candle_size', 'body_size',
    'wick_upper', 'wick_lower',
    'open', 'high', 'low', 'close', 'volume'
]

def load_data():
    print("\nConnecting to PostgreSQL...")
    db = get_connection()
    print("Connection successful")

    print("Loading market data...")
    query = """
        SELECT timeframe, timestamp, open, high, low, close, volume,
               hour, day_of_week, month, year,
               candle_size, body_size, wick_upper, wick_lower, direction
        FROM ustech_ohlcv
        ORDER BY timeframe, timestamp ASC
    """
    db.cursor.execute(query)
    rows = db.cursor.fetchall()
    columns = [desc[0] for desc in db.cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    print(f"Total candles loaded: {len(df):,}")
    return df


def prepare_data(df):
    print("\nPreparing data for strategy discovery...")

    le_dow = LabelEncoder()
    le_dir = LabelEncoder()

    df['day_of_week_enc'] = le_dow.fit_transform(df['day_of_week'])
    df['direction_enc'] = le_dir.fit_transform(df['direction'])

    df['next_direction'] = df.groupby('timeframe')['direction_enc'].shift(-1)
    df = df.dropna(subset=['next_direction'])
    df['next_direction'] = df['next_direction'].astype(int)

    feature_cols = [
        'hour', 'day_of_week_enc', 'month',
        'candle_size', 'body_size',
        'wick_upper', 'wick_lower',
        'open', 'high', 'low', 'close', 'volume'
    ]

    return df, feature_cols, le_dow, le_dir


def discover_per_timeframe(df, feature_cols):
    all_discoveries = {}

    for tf in TIMEFRAMES:
        tf_df = df[df['timeframe'] == tf].copy()

        if len(tf_df) < 100:
            print(f"  Skipping {tf} — not enough data")
            continue

        print(f"\n  Discovering patterns for {tf}...")
        print(f"  Rows available: {len(tf_df):,}")

        X = tf_df[feature_cols]
        y = tf_df['next_direction']

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test) * 100

        importances = model.feature_importances_
        feature_importance = dict(zip(feature_cols, importances))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        # Best hour
        hour_win = tf_df.groupby('hour')['next_direction'].mean()
        best_hour = int(hour_win.idxmax())
        worst_hour = int(hour_win.idxmin())

        # Best day
        day_win = tf_df.groupby('day_of_week')['next_direction'].mean()
        best_day = day_win.idxmax()
        worst_day = day_win.idxmin()

        # Best month
        month_win = tf_df.groupby('month')['next_direction'].mean()
        best_month = int(month_win.idxmax())

        # Buy vs sell ratio
        direction_counts = tf_df['direction'].value_counts().to_dict()
        total = sum(direction_counts.values())
        buy_pct = round((direction_counts.get('buy', 0) / total) * 100, 2)
        sell_pct = round((direction_counts.get('sell', 0) / total) * 100, 2)

        all_discoveries[tf] = {
            'accuracy': round(accuracy, 2),
            'total_candles': len(tf_df),
            'buy_percentage': buy_pct,
            'sell_percentage': sell_pct,
            'best_hour_to_buy': best_hour,
            'worst_hour_to_buy': worst_hour,
            'best_day_to_buy': best_day,
            'worst_day_to_buy': worst_day,
            'best_month': best_month,
            'top_5_features': [f[0] for f in top_features]
        }

        print(f"  Accuracy      : {accuracy:.2f}%")
        print(f"  Best hour     : {best_hour}:00")
        print(f"  Best day      : {best_day}")
        print(f"  Buy %         : {buy_pct}%")
        print(f"  Sell %        : {sell_pct}%")

    return all_discoveries


def save_results(discoveries):
    print("\nSaving discovered strategy...")

    # Save as JSON
    with open('discovered_strategy.json', 'w') as f:
        json.dump(discoveries, f, indent=4)

    # Save as readable text
    with open('discovered_strategy.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("   USTECH ML DISCOVERED STRATEGY REPORT\n")
        f.write("=" * 60 + "\n\n")

        for tf, data in discoveries.items():
            f.write(f"TIMEFRAME: {tf}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Model Accuracy       : {data['accuracy']}%\n")
            f.write(f"  Total Candles        : {data['total_candles']:,}\n")
            f.write(f"  Buy Candles          : {data['buy_percentage']}%\n")
            f.write(f"  Sell Candles         : {data['sell_percentage']}%\n")
            f.write(f"  Best Hour to Trade   : {data['best_hour_to_buy']}:00\n")
            f.write(f"  Worst Hour to Trade  : {data['worst_hour_to_buy']}:00\n")
            f.write(f"  Best Day to Trade    : {data['best_day_to_buy']}\n")
            f.write(f"  Worst Day to Trade   : {data['worst_day_to_buy']}\n")
            f.write(f"  Best Month           : {data['best_month']}\n")
            f.write(f"  Top Predictive Features:\n")
            for feat in data['top_5_features']:
                f.write(f"    - {feat}\n")
            f.write("\n")

    print("Saved: discovered_strategy.txt")
    print("Saved: discovered_strategy.json")


def main():
    print("\n" + "=" * 60)
    print("   USTECH STRATEGY DISCOVERY ENGINE")
    print("=" * 60)

    df = load_data()
    df, feature_cols, le_dow, le_dir = prepare_data(df)

    print("\nRunning discovery across all timeframes...")
    discoveries = discover_per_timeframe(df, feature_cols)

    save_results(discoveries)

    print("\n" + "=" * 60)
    print("   DISCOVERY COMPLETE")
    print("   Review discovered_strategy.txt for full report")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()