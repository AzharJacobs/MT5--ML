import pandas as pd
from db_connect import get_connection
from features import build_features

db = get_connection()
db.connect()
df = db.fetch_dataframe(
    'SELECT * FROM xauusd_ohlcv WHERE timeframe = %s ORDER BY timestamp ASC',
    ('15min',)
)
db.disconnect()

feat = build_features(df)

print('in_demand_zone:', feat['in_demand_zone'].sum())
print('in_supply_zone:', feat['in_supply_zone'].sum())

print('\nLast 100 bars supply zone top unique values:')
print(feat['supply_zone_top'].tail(100).unique())

print('\nLast 100 bars close range:')
close_tail = feat['close'].tail(100)
print(f'  min={close_tail.min():.0f}  max={close_tail.max():.0f}')

supply_top = feat['supply_zone_top'].fillna(0)
above = (feat['close'] > supply_top).sum()
print(f'\nBars where close > supply_zone_top: {above}')
print(f'Total bars: {len(feat)}')

print('\nSupply zone top value counts (top 5):')
print(feat['supply_zone_top'].value_counts().head())

print('\nDemand zone bottom value counts (top 5):')
print(feat['demand_zone_bottom'].value_counts().head())