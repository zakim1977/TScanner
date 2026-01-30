"""Check if whale acceleration features are being used by the ML model."""
import pickle
import os

model_path = "models/quality_model.pkl"

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("=" * 60)
    print("MODEL FEATURE ANALYSIS")
    print("=" * 60)
    
    # Check what's in the pickle
    if isinstance(model_data, dict):
        print(f"\nModel data keys: {list(model_data.keys())}")
        model = model_data.get('model')
        feature_cols = model_data.get('feature_cols', [])
    else:
        model = model_data
        feature_cols = []
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1}. {col}")
    
    # Check for whale acceleration features
    whale_accel_features = [
        'whale_delta_24h', 'whale_delta_7d', 'whale_daily_avg_7d',
        'whale_acceleration_ratio', 'whale_accel_accelerating',
        'whale_accel_decelerating', 'whale_accel_reversing',
        'whale_accel_steady', 'whale_is_fresh', 'whale_is_late_entry'
    ]
    
    print("\n" + "=" * 60)
    print("WHALE ACCELERATION FEATURES CHECK")
    print("=" * 60)
    
    found = []
    missing = []
    for feat in whale_accel_features:
        if feat in feature_cols:
            found.append(feat)
        else:
            missing.append(feat)
    
    print(f"\n✅ Found ({len(found)}):")
    for f in found:
        print(f"   {f}")
    
    if missing:
        print(f"\n❌ Missing ({len(missing)}):")
        for f in missing:
            print(f"   {f}")
    
    # Try to get feature importance
    if model is not None and hasattr(model, 'feature_importances_'):
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCES (Top 20)")
        print("=" * 60)
        
        importances = model.feature_importances_
        if len(feature_cols) == len(importances):
            feat_imp = list(zip(feature_cols, importances))
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, imp) in enumerate(feat_imp[:20]):
                bar = "█" * int(imp * 50)
                print(f"  {i+1:2}. {name:30} {imp:.4f} {bar}")
            
            # Show whale features specifically
            print("\n" + "=" * 60)
            print("WHALE ACCELERATION FEATURE IMPORTANCES")
            print("=" * 60)
            whale_imps = [(n, i) for n, i in feat_imp if 'whale' in n.lower()]
            for name, imp in whale_imps:
                bar = "█" * int(imp * 50)
                is_new = "🆕" if name in whale_accel_features else ""
                print(f"  {name:30} {imp:.4f} {bar} {is_new}")
else:
    print(f"Model not found at {model_path}")
