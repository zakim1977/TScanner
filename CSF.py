"""
Check what's ACTUALLY in the training data.
Run during or after training to verify SMC columns exist.
"""
import sys
sys.path.insert(0, '.')

import pickle
import os
import numpy as np

print("=" * 70)
print("CHECKING TRAINED MODEL FEATURES")
print("=" * 70)

model_path = 'ml/models/probabilistic/prob_model_swing_stock.pkl'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("\nAvailable models:")
    model_dir = 'ml/models/probabilistic'
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            print(f"   {f}")
    exit(1)

with open(model_path, 'rb') as f:
    bundle = pickle.load(f)

meta = bundle.get('metadata', {})
scaler = bundle.get('scaler')

print(f"\n📊 Model Metadata:")
print(f"   Samples: {meta.get('n_samples', '?')}")
print(f"   Features: {meta.get('n_features', '?')}")
print(f"   Trained: {meta.get('trained_at', '?')}")

# Check scaler to understand feature statistics
if scaler is not None:
    print(f"\n📈 Feature Statistics from Scaler:")
    means = scaler.mean_
    stds = scaler.scale_
    
    # Feature names (from probabilistic_ml.py)
    feature_names = [
        'return_1', 'return_3', 'return_5', 'return_10',
        'high_low_range', 'close_position', 'dist_from_high', 'dist_from_low',
        'bos_signal', 'bb_position', 'atr_normalized', 'volume_change',
        'rsi', 'macd_hist', 'trend_strength',
        # Features 15-26: Positioning (crypto=whale, stock=TA)
        'pos_15', 'pos_16', 'pos_17', 'pos_18', 'pos_19',
        'pos_20', 'pos_21', 'pos_22', 'pos_23', 'pos_24', 'pos_25', 'cmf',
        # Features 27-42: SMC
        'smc_liq_bull', 'smc_liq_bear',
        'smc_at_bull_ob', 'smc_at_bear_ob', 'smc_near_bull_ob', 'smc_near_bear_ob',
        'smc_fvg_bull', 'smc_fvg_bear',
        'smc_accum', 'smc_distrib',
        'smc_37', 'smc_38', 'smc_39', 'smc_40', 'smc_41', 'smc_42',
        # Remaining features
        'session', 'volatility', 'trend_momentum', 'price_acceleration',
        'higher_high', 'lower_low', 'inside_bar', 'btc_corr', 'fear_greed'
    ]
    
    print(f"\n   {'Feature':<25} {'Mean':>10} {'Std':>10} {'Status'}")
    print(f"   {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    
    smc_start = 27  # SMC features start at index 27
    smc_end = 43    # SMC features end around index 42
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
        
        # Check if feature has variance (useful for learning)
        if std < 0.001:
            status = "⚠️ NO VARIANCE"
        elif abs(mean) < 0.001 and std < 0.1:
            status = "⚠️ MOSTLY ZEROS"
        else:
            status = "✅"
        
        # Highlight SMC features
        if smc_start <= i <= smc_end:
            print(f"   {name:<25} {mean:>10.4f} {std:>10.4f} {status} [SMC]")
        elif i < 50:  # Only show first 50
            print(f"   {name:<25} {mean:>10.4f} {std:>10.4f} {status}")
    
    # Summary of SMC features
    print(f"\n📊 SMC Feature Summary (indices {smc_start}-{smc_end}):")
    smc_means = means[smc_start:smc_end+1]
    smc_stds = stds[smc_start:smc_end+1]
    
    non_zero_smc = sum(1 for m, s in zip(smc_means, smc_stds) if abs(m) > 0.001 or s > 0.01)
    print(f"   SMC features with variance: {non_zero_smc}/{len(smc_means)}")
    
    if non_zero_smc < 5:
        print(f"\n   ❌ PROBLEM: Most SMC features are zeros!")
        print(f"   This means SMC columns were NOT in training data.")
    else:
        print(f"\n   ✅ SMC features have variance - being used!")

print("\n" + "=" * 70)