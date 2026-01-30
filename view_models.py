"""
View all trained model metrics.
Run from TScanner directory:
    python view_models.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.probabilistic_ml import get_all_model_metrics, get_model_status

print("=" * 60)
print("TRAINED MODELS STATUS")
print("=" * 60)

metrics = get_all_model_metrics()

if not metrics:
    print("\n❌ No models found. Train models in ML Training tab.")
else:
    print(f"\n✅ Found {len(metrics)} trained models:\n")
    
    for key in sorted(metrics.keys()):
        m = metrics[key]
        mode = m.get('mode', '?').upper()
        market = m.get('market_type', '?').upper()
        avg_f1 = m.get('avg_f1', 0)
        if avg_f1 < 1:
            avg_f1 *= 100  # Convert to percentage if needed
        samples = m.get('n_samples', 0)
        trained = m.get('trained_at', 'Unknown')[:16]  # Date + time
        
        print(f"📊 {mode} {market}")
        print(f"   F1 Score: {avg_f1:.1f}%")
        print(f"   Samples:  {samples:,}")
        print(f"   Trained:  {trained}")
        
        # Show per-label metrics if available
        label_metrics = m.get('metrics', {})
        if label_metrics:
            print(f"   Labels:")
            for label, lm in label_metrics.items():
                f1 = lm.get('f1', 0)
                if f1 < 1:
                    f1 *= 100
                print(f"     • {label}: {f1:.1f}% F1")
        print()

print("=" * 60)
