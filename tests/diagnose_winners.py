"""
diagnose_winners.py
===================
Checks whether winning trades look different from losing trades
in feature space. If they don't, no amount of SMOTE or scale_pos_weight
will fix the model — the features themselves need to change.
"""
import pandas as pd
import numpy as np
from prepare_data import DataPreparator

print("Loading and preparing data...")
prep = DataPreparator()
X_train, y_train, raw_train, X_test, y_test, raw_test = prep.prepare_data(
    timeframes=["15min"],
)

print(f"\nTrain rows: {len(X_train):,}")
print(f"Winners (label=1): {(y_train==1).sum()}")
print(f"Losers  (label=0): {(y_train==0).sum()}")

# Compare mean feature values for winners vs losers
winners = X_train[y_train == 1]
losers  = X_train[y_train == 0]

print(f"\n{'Feature':<35} {'Winners mean':>14} {'Losers mean':>14} {'Diff':>10}")
print("-" * 75)

diffs = []
for col in X_train.columns:
    w_mean = winners[col].mean()
    l_mean = losers[col].mean()
    diff   = abs(w_mean - l_mean)
    diffs.append((col, w_mean, l_mean, diff))

# Sort by absolute difference — features that separate winners from losers most
diffs.sort(key=lambda x: x[3], reverse=True)

for col, w, l, d in diffs[:20]:
    print(f"  {col:<33} {w:>14.4f} {l:>14.4f} {d:>10.4f}")

print("\n--- Zone-specific features ---")
zone_cols = [c for c in X_train.columns if "zone" in c.lower()]
for col, w, l, d in diffs:
    if col in zone_cols:
        print(f"  {col:<33} {w:>14.4f} {l:>14.4f} {d:>10.4f}")

print("\n--- In-zone flags ---")
for col in ["in_demand_zone", "in_supply_zone", "between_zones", "active_zone_quality"]:
    if col in X_train.columns:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        print(f"  {col:<33} winners={w_mean:.4f}  losers={l_mean:.4f}")

print("\n--- HTF alignment ---")
for col in ["htf_aligned", "htf_1h_bias", "htf_4h_bias"]:
    if col in X_train.columns:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        print(f"  {col:<33} winners={w_mean:.4f}  losers={l_mean:.4f}")

print("\nDone. If winners and losers have very similar feature values,")
print("the model cannot separate them — features need to be improved.")