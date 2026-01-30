"""
Debug why whale features show 0 importance.
"""
import sys
sys.path.insert(0, '.')

import pickle
import numpy as np

print("=" * 70)
print("DEBUGGING FEATURE IMPORTANCE")
print("=" * 70)

# Load the model
model_path = "models/quality_model.pkl"

with open(model_path, 'rb') as f:
    data = pickle.load(f)

model = data['model']
metrics = data['metrics']

print(f"\nModel type: {type(model).__name__}")

# Get feature columns from QualityModel class
from liquidity_hunter.quality_model import QualityModel
feature_cols = QualityModel.FEATURE_COLS

print(f"Number of features: {len(feature_cols)}")

# Check if model has feature_importances_
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    print(f"\nfeature_importances_ length: {len(importances)}")
    print(f"Sum of importances: {sum(importances):.4f}")
    print(f"Max importance: {max(importances):.4f}")
    print(f"Non-zero importances: {sum(1 for i in importances if i > 0.0001)}")

    # Show ALL feature importances (not just top 5)
    print("\n" + "=" * 70)
    print("ALL FEATURE IMPORTANCES (sorted)")
    print("=" * 70)

    feat_imp = list(zip(feature_cols, importances))
    feat_imp.sort(key=lambda x: x[1], reverse=True)

    for name, imp in feat_imp:
        bar = "█" * int(imp * 50) if imp > 0 else ""
        marker = "🆕" if "whale_delta_24h" in name or "whale_accel" in name or "whale_is_" in name else ""
        print(f"  {name:35} {imp:.6f} {bar} {marker}")

    # Check if AdaBoost has estimators we can inspect
    if hasattr(model, 'estimators_'):
        print(f"\n" + "=" * 70)
        print(f"AdaBoost has {len(model.estimators_)} estimators")
        print("=" * 70)

        # Check first few estimators
        for i, est in enumerate(model.estimators_[:3]):
            if hasattr(est, 'feature_importances_'):
                est_imp = est.feature_importances_
                non_zero = sum(1 for x in est_imp if x > 0.0001)
                max_feat_idx = np.argmax(est_imp)
                print(f"  Estimator {i}: {non_zero} non-zero features, max={feature_cols[max_feat_idx]}")
            elif hasattr(est, 'tree_'):
                # Decision stump
                feat_used = est.tree_.feature[0] if est.tree_.feature[0] >= 0 else None
                if feat_used is not None and feat_used < len(feature_cols):
                    print(f"  Estimator {i}: Decision stump using '{feature_cols[feat_used]}'")

else:
    print("\n⚠️ Model doesn't have feature_importances_ attribute")
    print(f"Model attributes: {dir(model)}")

# Check if the issue is with the training data
print("\n" + "=" * 70)
print("CHECKING TRAINING DATA VARIANCE")
print("=" * 70)

# Load a small sample to check feature variance
from core.data_fetcher import fetch_klines_parallel
from liquidity_hunter.quality_model import generate_quality_samples
import pandas as pd
import os
import sqlite3

# Quick test with 1 coin
print("\nFetching BTC data to check feature variance...")
klines = fetch_klines_parallel(['BTCUSDT'], '4h', 200, lambda c,t,s: None)

whale_history = {}
if os.path.exists("data/whale_history.db"):
    conn = sqlite3.connect("data/whale_history.db")
    df_whale = pd.read_sql(
        "SELECT symbol, timestamp, whale_long_pct FROM whale_snapshots WHERE symbol='BTCUSDT' ORDER BY timestamp",
        conn
    )
    conn.close()
    whale_history['BTCUSDT'] = df_whale.to_dict('records')

if 'BTCUSDT' in klines and klines['BTCUSDT'] is not None:
    samples = generate_quality_samples(klines['BTCUSDT'], 'BTCUSDT', whale_history.get('BTCUSDT', []))

    if samples:
        df = pd.DataFrame(samples)

        print(f"\nSamples: {len(df)}")
        print("\nFeature statistics (whale acceleration):")
        print("-" * 60)

        whale_feats = ['whale_delta_24h', 'whale_delta_7d', 'whale_daily_avg_7d',
                       'whale_acceleration_ratio', 'whale_is_fresh', 'whale_is_late_entry']

        for feat in whale_feats:
            if feat in df.columns:
                mean = df[feat].mean()
                std = df[feat].std()
                min_val = df[feat].min()
                max_val = df[feat].max()
                non_zero = (df[feat] != 0).sum()
                print(f"  {feat:30} mean={mean:+.4f} std={std:.4f} range=[{min_val:.2f}, {max_val:.2f}] non-zero={non_zero}/{len(df)}")

        # Check correlation with target
        if 'won' in df.columns:
            print("\nCorrelation with 'won' (target):")
            print("-" * 60)
            for feat in whale_feats:
                if feat in df.columns:
                    corr = df[feat].corr(df['won'].astype(float))
                    print(f"  {feat:30} corr={corr:+.4f}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
