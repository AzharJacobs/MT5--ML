import pandas as pd
import numpy as np
from db_connect import get_connection
from features import build_features

db = get_connection()
db.connect()
df = db.fetch_dataframe(
    'SELECT * FROM ustech_verified WHERE is_verified = TRUE AND timeframe = %s ORDER BY timestamp ASC',
    ('15min',)
)
db.disconnect()

h1 = None
h4 = None
try:
    db2 = get_connection()
    db2.connect()
    h1 = db2.fetch_dataframe('SELECT * FROM ustech_verified WHERE is_verified = TRUE AND timeframe = %s ORDER BY timestamp ASC', ('1H',))
    h4 = db2.fetch_dataframe('SELECT * FROM ustech_verified WHERE is_verified = TRUE AND timeframe = %s ORDER BY timestamp ASC', ('4H',))
    db2.disconnect()
except:
    pass

feat = build_features(df, h1_df=h1, h4_df=h4)

print("htf_4h_bias value counts:")
print(feat['htf_4h_bias'].value_counts())
print(f"\nhtf_1h_bias value counts:")
print(feat['htf_1h_bias'].value_counts())
print(f"\nhtf_aligned value counts:")
print(feat['htf_aligned'].value_counts())
print(f"\nrule_htf_aligned_buy (should block sells in bull): {feat['rule_htf_aligned_buy'].sum():.0f} bars")
print(f"rule_htf_aligned_sell (should block buys in bear): {feat['rule_htf_aligned_sell'].sum():.0f} bars")
print(f"rule_valid_buy_setup: {feat['rule_valid_buy_setup'].sum():.0f} bars")
print(f"rule_valid_sell_setup: {feat['rule_valid_sell_setup'].sum():.0f} bars")