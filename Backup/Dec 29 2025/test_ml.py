#!/usr/bin/env python3
"""
Test ML Engine - Run this to verify ML is working
"""

import os
import sys

print("=" * 60)
print("🧪 TESTING ML ENGINE")
print("=" * 60)

# Check if models exist
model_dir = os.path.join(os.path.dirname(__file__), 'ml', 'models')
print(f"\n📂 Model directory: {model_dir}")

if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    print(f"   Files found: {files}")
else:
    print("   ❌ Directory does not exist!")
    sys.exit(1)

# Check each required file
required_files = ['direction_model.pkl', 'tp_sl_model.pkl', 'feature_scaler.pkl', 'model_metadata.json']
for f in required_files:
    path = os.path.join(model_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   ✅ {f}: {size:,} bytes")
    else:
        print(f"   ❌ {f}: MISSING!")

# Try to import and use ML engine
print("\n🔧 Testing ML Engine Import...")
try:
    from ml.ml_engine import MLEngine, get_ml_prediction, is_ml_available
    print("   ✅ Import successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Check if ML is available
print("\n🔍 Checking is_ml_available()...")
try:
    available = is_ml_available()
    print(f"   ML Available: {available}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Try a prediction
print("\n🎯 Testing prediction...")
try:
    prediction = get_ml_prediction(
        whale_pct=65,
        retail_pct=45,
        oi_change=5.2,
        price_change_24h=2.1,
        position_pct=35,
        current_price=95000,
        swing_high=98000,
        swing_low=90000,
        ta_score=62,
        trend='BULLISH',
        money_flow_phase='ACCUMULATION',
    )
    
    print(f"\n   📊 PREDICTION RESULT:")
    print(f"   Direction:  {prediction.direction}")
    print(f"   Confidence: {prediction.confidence:.1f}%")
    print(f"   Reasoning:  {prediction.reasoning}")
    print(f"   Top Features:")
    for name, value in prediction.top_features[:5]:
        print(f"      - {name}: {value:.3f}")
    
    print("\n✅ ML ENGINE IS WORKING!")
    
except Exception as e:
    import traceback
    print(f"   ❌ Prediction failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)