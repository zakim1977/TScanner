"""
Diagnostic script to check if stock swing thresholds are correct.
Run this from your TScanner directory:
    python check_thresholds.py
"""

import sys
import os

# Add ml folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("THRESHOLD DIAGNOSTIC")
print("=" * 60)

try:
    # Force reload by clearing cache
    if 'ml.probabilistic_ml' in sys.modules:
        del sys.modules['ml.probabilistic_ml']
    if 'probabilistic_ml' in sys.modules:
        del sys.modules['probabilistic_ml']
    
    from ml.probabilistic_ml import MODE_LABELS_CRYPTO, MODE_LABELS_STOCK, get_mode_labels
    
    print("\n✅ Successfully imported probabilistic_ml")
    
    print("\n" + "=" * 60)
    print("CRYPTO SWING SETTINGS:")
    print("=" * 60)
    crypto_swing = MODE_LABELS_CRYPTO['swing']
    print(f"  Lookahead: {crypto_swing['lookahead']} candles")
    print(f"  Thresholds:")
    for k, v in crypto_swing['thresholds'].items():
        print(f"    {k}: {v}%")
    
    print("\n" + "=" * 60)
    print("STOCK SWING SETTINGS:")
    print("=" * 60)
    stock_swing = MODE_LABELS_STOCK['swing']
    print(f"  Lookahead: {stock_swing['lookahead']} candles")
    print(f"  Thresholds:")
    for k, v in stock_swing['thresholds'].items():
        print(f"    {k}: {v}%")
    
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    
    # Check if stock thresholds are correct (should be 2.0, 1.5, 1.5)
    expected_stock = {'trend_holds': 2.0, 'reversal': 1.5, 'drawdown': 1.5}
    
    all_correct = True
    for k, expected_v in expected_stock.items():
        actual_v = stock_swing['thresholds'].get(k, 'MISSING')
        status = "✅" if actual_v == expected_v else "❌"
        print(f"  {status} {k}: expected {expected_v}%, got {actual_v}%")
        if actual_v != expected_v:
            all_correct = False
    
    print("\n" + "=" * 60)
    if all_correct:
        print("✅ ALL THRESHOLDS ARE CORRECT!")
        print("   If training still shows wrong positive rates,")
        print("   try: rmdir /s /q ml\\__pycache__")
    else:
        print("❌ THRESHOLDS ARE WRONG!")
        print("   Please copy the new probabilistic_ml.py file.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()