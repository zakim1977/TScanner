"""
Quick test script to verify whale acceleration features are working.
Run in Anaconda prompt: python test_whale_features.py
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 70)
print("WHALE ACCELERATION FEATURE TEST")
print("=" * 70)

# Test with 2 coins, 90 days (faster)
TEST_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TEST_DAYS = 90

print(f"\nTesting with {len(TEST_SYMBOLS)} coins, {TEST_DAYS} days...")

# Step 1: Fetch data
print("\n[1/4] Fetching price data...")
try:
    from core.data_fetcher import fetch_klines_parallel

    klines_data = fetch_klines_parallel(
        symbols=TEST_SYMBOLS,
        interval='4h',
        limit=min(TEST_DAYS * 6, 500),
        progress_callback=lambda c, t, s: print(f"  Fetching {s}... {c}/{t}")
    )
    print(f"  ✅ Fetched data for {len(klines_data)} symbols")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Step 2: Load whale history
print("\n[2/4] Loading whale history...")
import os
import sqlite3

whale_history = {}
whale_db = "data/whale_history.db"

if os.path.exists(whale_db):
    conn = sqlite3.connect(whale_db)
    df_whale = pd.read_sql(
        "SELECT symbol, timestamp, whale_long_pct FROM whale_snapshots ORDER BY timestamp",
        conn
    )
    conn.close()

    for symbol in df_whale['symbol'].unique():
        sym_data = df_whale[df_whale['symbol'] == symbol]
        whale_history[symbol] = sym_data.to_dict('records')

    print(f"  ✅ Loaded whale data for {len(whale_history)} symbols")
    for sym in TEST_SYMBOLS:
        count = len(whale_history.get(sym, []))
        print(f"     {sym}: {count} snapshots")
else:
    print(f"  ⚠️ No whale history database found at {whale_db}")

# Step 3: Generate samples and check features
print("\n[3/4] Generating samples with whale acceleration features...")

from liquidity_hunter.quality_model import generate_quality_samples, QualityModel

all_samples = []
for symbol, df in klines_data.items():
    if df is None or len(df) < 100:
        print(f"  ⚠️ {symbol}: Not enough data")
        continue

    sym_whale = whale_history.get(symbol, [])
    samples = generate_quality_samples(df, symbol, sym_whale)
    all_samples.extend(samples)
    print(f"  {symbol}: {len(samples)} samples")

print(f"\n  Total samples: {len(all_samples)}")

# Step 4: Check if whale acceleration features exist
print("\n[4/4] Checking whale acceleration features in samples...")

whale_accel_features = [
    'whale_delta_24h', 'whale_delta_7d', 'whale_daily_avg_7d',
    'whale_acceleration_ratio', 'whale_accel_accelerating',
    'whale_accel_decelerating', 'whale_accel_reversing',
    'whale_accel_steady', 'whale_is_fresh', 'whale_is_late_entry'
]

if all_samples:
    sample = all_samples[0]

    print("\n  Feature check in first sample:")
    print("  " + "-" * 50)

    found_count = 0
    missing_count = 0

    for feat in whale_accel_features:
        if feat in sample:
            value = sample[feat]
            found_count += 1
            print(f"  ✅ {feat}: {value}")
        else:
            missing_count += 1
            print(f"  ❌ {feat}: NOT FOUND")

    print("  " + "-" * 50)
    print(f"  Found: {found_count}/10, Missing: {missing_count}/10")

    # Show distribution of key features across all samples
    print("\n  Feature distribution across all samples:")
    print("  " + "-" * 50)

    df_samples = pd.DataFrame(all_samples)

    for feat in ['whale_delta_24h', 'whale_delta_7d', 'whale_is_fresh', 'whale_is_late_entry']:
        if feat in df_samples.columns:
            non_zero = (df_samples[feat] != 0).sum()
            mean_val = df_samples[feat].mean()
            print(f"  {feat}:")
            print(f"    Mean: {mean_val:.4f}, Non-zero: {non_zero}/{len(df_samples)} ({100*non_zero/len(df_samples):.1f}%)")

    # Quick train test
    print("\n" + "=" * 70)
    print("QUICK TRAIN TEST (small sample)")
    print("=" * 70)

    if len(all_samples) >= 100:
        model = QualityModel()

        # Use subset for quick test
        test_samples = all_samples[:min(1000, len(all_samples))]

        print(f"\nTraining on {len(test_samples)} samples...")
        metrics = model.train(test_samples, training_days=TEST_DAYS)

        if 'error' not in metrics:
            print("\n✅ Training completed!")

            # Check feature importance
            result = model.get_feature_importance()

            print("\nWhale Acceleration Feature Importances:")
            print("-" * 50)

            total_whale_imp = 0
            for feat in whale_accel_features:
                imp = result['whale_accel_features'].get(feat, 'N/A')
                if isinstance(imp, (int, float)):
                    total_whale_imp += imp
                    bar = "█" * int(imp * 100) if imp > 0 else ""
                    print(f"  {feat:30} {imp:.4f} {bar}")
                else:
                    print(f"  {feat:30} {imp}")

            print("-" * 50)
            print(f"  Total whale accel importance: {total_whale_imp:.4f}")

            if total_whale_imp > 0.01:
                print("\n✅ Whale acceleration features ARE being used by the model!")
            else:
                print("\n⚠️ Whale acceleration features have very low importance.")
                print("   This could mean:")
                print("   1. Other features are more predictive")
                print("   2. Need more training data")
                print("   3. Features need adjustment")
        else:
            print(f"\n❌ Training error: {metrics.get('error')}")
    else:
        print(f"\n⚠️ Not enough samples ({len(all_samples)}) for training test")

else:
    print("  ❌ No samples generated!")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
