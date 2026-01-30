"""
Run this to see what's actually in your model file.
Usage: python check_model.py
"""
import pickle
import os

# Find model files
model_dir = r"C:\Users\zakim\training\TScanner\ml\models\probabilistic"

print("=" * 60)
print("MODEL FILE DIAGNOSIS")
print("=" * 60)

# List all model files
if os.path.exists(model_dir):
    files = os.listdir(model_dir)
    print(f"\nFound {len(files)} files in {model_dir}:")
    for f in files:
        print(f"  - {f}")
else:
    print(f"ERROR: Directory not found: {model_dir}")
    exit(1)

# Check each .pkl file
for f in files:
    if f.endswith('.pkl'):
        path = os.path.join(model_dir, f)
        print(f"\n{'=' * 60}")
        print(f"FILE: {f}")
        print("=" * 60)
        
        try:
            with open(path, 'rb') as file:
                data = pickle.load(file)
            
            print(f"\nTop-level keys: {list(data.keys())}")
            
            # Check models
            models = data.get('models_per_label', data.get('models', {}))
            print(f"\nModels (labels trained): {list(models.keys())}")
            
            # Check metadata
            metadata = data.get('metadata', {})
            print(f"\nMetadata keys: {list(metadata.keys())}")
            
            # Check metrics (where F1 scores live)
            metrics = metadata.get('metrics', {})
            if metrics:
                print(f"\nMetrics (F1 scores):")
                for label, m in metrics.items():
                    if isinstance(m, dict):
                        f1 = m.get('f1', 'N/A')
                        print(f"  {label}: F1={f1}")
                    else:
                        print(f"  {label}: {m}")
            else:
                print("\n⚠️ NO METRICS FOUND IN METADATA!")
                
            # Check thresholds
            thresholds = data.get('thresholds', metadata.get('thresholds', {}))
            if thresholds:
                print(f"\nThresholds: {thresholds}")
            
            # Check labels list
            labels = data.get('labels', metadata.get('labels', []))
            if labels:
                print(f"\nLabels list: {labels}")
                
        except Exception as e:
            print(f"ERROR loading {f}: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)