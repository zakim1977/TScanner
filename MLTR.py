"""
Diagnostic script to check F1 scores stored in trained ML models.
Run this from your TScanner directory:
    python check_f1_scores.py
"""

import sys
import os
import pickle

print("=" * 70)
print("F1 SCORES DIAGNOSTIC - Check what's actually stored in models")
print("=" * 70)

# CORRECT paths - models are in ml/models/probabilistic/
model_paths = [
    'ml/models/probabilistic/prob_model_daytrade_crypto.pkl',
    'ml/models/probabilistic/prob_model_scalp_crypto.pkl', 
    'ml/models/probabilistic/prob_model_swing_crypto.pkl',
    'ml/models/probabilistic/prob_model_investment_crypto.pkl',
    'ml/models/probabilistic/prob_model_daytrade_stock.pkl',
    'ml/models/probabilistic/prob_model_swing_stock.pkl',
]

for model_path in model_paths:
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_path}")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print("  [NOT FOUND] - Model not trained yet")
        continue
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Get metadata
        metadata = model_data.get('metadata', {})
        metrics_dict = metadata.get('metrics', {})
        
        print(f"\n  Trained at: {metadata.get('trained_at', 'Unknown')}")
        print(f"  Samples: {metadata.get('n_samples', 'Unknown')}")
        print(f"  Average F1: {metadata.get('avg_f1', 0)*100:.1f}%")
        
        # Extract F1 per label
        print(f"\n  F1 SCORES PER LABEL:")
        print("  " + "-" * 50)
        
        f1_scores = {}
        for label, m in metrics_dict.items():
            if isinstance(m, dict):
                f1 = m.get('f1', 0.0)
                precision = m.get('precision', 0.0)
                recall = m.get('recall', 0.0)
                f1_scores[label] = f1
                print(f"    {label:25s}: F1={f1*100:5.1f}%  Prec={precision*100:5.1f}%  Rec={recall*100:5.1f}%")
        
        # Check if all F1 are the same (suspicious!)
        unique_f1 = set(round(f*100, 1) for f in f1_scores.values())
        if len(unique_f1) == 1 and len(f1_scores) > 1:
            print(f"\n  WARNING: All labels have IDENTICAL F1 = {list(unique_f1)[0]}%")
            print("     This is suspicious - check training data quality!")
        elif len(unique_f1) <= 2 and len(f1_scores) > 2:
            print(f"\n  WARNING: Only {len(unique_f1)} unique F1 values across {len(f1_scores)} labels")
            print(f"     Values: {sorted(unique_f1)}")
        
        # Also check direct f1_scores key (old format)
        direct_f1 = metadata.get('f1_scores', {})
        if direct_f1:
            print(f"\n  [OLD FORMAT] Direct f1_scores key found:")
            for k, v in direct_f1.items():
                print(f"    {k}: {v*100:.1f}%")
                
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("""
If you see:
  - All F1 scores identical -> Training issue, needs retraining
  - Different F1 per label -> Model is fine, display bug needs fixing
  - "NOT FOUND" -> Model not trained for that mode
""")