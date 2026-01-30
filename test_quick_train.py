"""
Quick training test - 1 coin, 365 days
Verify price action features and F1 score before full training
"""
import sys
sys.path.insert(0, '.')

def run_quick_test():
    print("=" * 70)
    print("QUICK TRAINING TEST - 1 Coin, 365 Days")
    print("=" * 70)

    from liquidity_hunter.quality_model import train_quality_model, get_quality_model_status

    # Check current model status
    print("\n1. Current Model Status:")
    status = get_quality_model_status()
    print(f"   Is trained: {status.get('is_trained', False)}")
    if status.get('is_trained'):
        print(f"   Current F1: {status.get('f1', 0):.1%}")
        print(f"   Current samples: {status.get('total_samples', 0)}")

    # Train on single coin
    print("\n2. Training on BTCUSDT (365 days)...")
    print("   This will test Price Action feature extraction...")
    print("-" * 70)

    def progress(text, pct):
        print(f"   [{pct*100:5.1f}%] {text}")

    metrics = train_quality_model(
        symbols=['BTCUSDT'],  # Just 1 coin for quick test
        days=365,             # 365 days of data for more samples
        progress_callback=progress
    )

    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    if 'error' in metrics:
        print(f"\n❌ ERROR: {metrics['error']}")
        return metrics

    # Display key metrics
    print(f"\n📊 CORE METRICS:")
    print(f"   Total Samples: {metrics.get('total_samples', 0)}")
    print(f"   Features Used: {metrics.get('features_used', 0)}")
    print(f"   Base Win Rate: {metrics.get('base_win_rate', 0):.1%}")

    print(f"\n🎯 MODEL PERFORMANCE (Test Set):")
    print(f"   Accuracy:  {metrics.get('accuracy', 0):.1%}")
    print(f"   F1 Score:  {metrics.get('f1', 0):.1%}")
    print(f"   Precision: {metrics.get('precision', 0):.1%}")
    print(f"   Recall:    {metrics.get('recall', 0):.1%}")

    # OVERFITTING CHECK
    print(f"\n🔍 OVERFITTING CHECK:")
    overfit_status = metrics.get('overfit_status', 'UNKNOWN')
    if overfit_status == 'SEVERE':
        status_icon = '🚨 SEVERE'
    elif overfit_status == 'WARNING':
        status_icon = '⚠️ WARNING'
    elif overfit_status == 'MILD':
        status_icon = '⚡ MILD'
    else:
        status_icon = '✅ OK'
    print(f"   Status: {status_icon}")
    print(f"   Train Accuracy: {metrics.get('train_accuracy', 0):.1%} | Test: {metrics.get('test_accuracy', 0):.1%} | Gap: {metrics.get('overfit_gap_accuracy', 0):+.1%}")
    print(f"   Train F1:       {metrics.get('train_f1', 0):.1%} | Test: {metrics.get('test_f1', 0):.1%} | Gap: {metrics.get('overfit_gap_f1', 0):+.1%}")
    print(f"   5-Fold CV F1:   {metrics.get('cv_f1_mean', 0):.1%} +/- {metrics.get('cv_f1_std', 0):.1%}")
    print(f"   5-Fold CV Acc:  {metrics.get('cv_accuracy_mean', 0):.1%} +/- {metrics.get('cv_accuracy_std', 0):.1%}")
    print(f"   Train/Test:     {metrics.get('train_set_size', 0)}/{metrics.get('test_set_size', 0)} samples")

    # MODEL COMPARISON
    print(f"\n🤖 MODEL COMPARISON:")
    print(f"   Best Model: {metrics.get('best_model', 'Unknown')}")
    model_scores = metrics.get('model_scores', {})
    if model_scores:
        for name, scores in sorted(model_scores.items(), key=lambda x: x[1].get('cv_f1', 0), reverse=True):
            cv_f1 = scores.get('cv_f1', 0)
            overfit = scores.get('overfit', 0)
            marker = "🏆" if name == metrics.get('best_model') else "  "
            print(f"   {marker} {name:20s} CV F1: {cv_f1:.1%}  Overfit: {overfit:+.1%}")

    print(f"\n💰 WIN RATES BY CONFIDENCE:")
    print(f"   >50% confidence: {metrics.get('high_conf_win_rate', 0):.1%} ({metrics.get('high_conf_trades', 0)} trades)")
    print(f"   >60% confidence: {metrics.get('very_high_win_rate', 0):.1%} ({metrics.get('very_high_trades', 0)} trades)")

    # TRADE FREQUENCY PROJECTION
    print(f"\n📅 TRADE FREQUENCY (from 1 coin):")
    very_high_per_month = metrics.get('very_high_per_month', 0)
    high_conf_per_month = metrics.get('high_conf_per_month', 0)
    print(f"   Very High (>60%): {very_high_per_month:.1f} trades/month")
    print(f"   High Conf (>50%): {high_conf_per_month:.1f} trades/month")

    print(f"\n📊 PROJECTED WITH 50 COINS:")
    # Assume 60% of coins have similar patterns (conservative)
    effective_coins = 30
    projected_very_high = very_high_per_month * effective_coins
    projected_high = high_conf_per_month * effective_coins
    print(f"   Very High (>60%): ~{projected_very_high:.0f} trades/month")
    print(f"   High Conf (>50%): ~{projected_high:.0f} trades/month")
    if projected_very_high < 10:
        print(f"   ⚠️ May need to lower confidence threshold or add more coins")

    print(f"\n📈 TOP 5 FEATURES (importance):")
    top_features = metrics.get('top_features', [])[:5]
    for feat, importance in top_features:
        bar = "█" * int(importance * 50)
        print(f"   {feat:25s} {importance:.3f} {bar}")

    # Check if price action features are being used
    print(f"\n🔍 PRICE ACTION FEATURES CHECK:")
    pa_features = [f for f, _ in metrics.get('top_features', []) if f.startswith('pa_')]
    if pa_features:
        print(f"   ✅ Price Action features found in model: {len(pa_features)}")
        for feat, imp in [(f, i) for f, i in metrics.get('top_features', []) if f.startswith('pa_')]:
            print(f"      - {feat}: {imp:.3f}")
    else:
        print(f"   ⚠️ No Price Action features found - check if PA analyzer is working")

    print(f"\n💵 EXPECTED ROI (per trade at 1:1 R:R):")
    print(f"   Base:        {metrics.get('base_roi_per_trade', 0):+.1f}%")
    print(f"   >50% conf:   {metrics.get('high_conf_roi_per_trade', 0):+.1f}%")
    print(f"   >60% conf:   {metrics.get('very_high_roi_per_trade', 0):+.1f}%")

    print("\n" + "=" * 70)

    # Recommendation based on F1 AND overfitting status
    f1 = metrics.get('f1', 0)
    cv_f1 = metrics.get('cv_f1_mean', 0)
    overfit_status = metrics.get('overfit_status', 'UNKNOWN')

    print("\n📋 OVERALL ASSESSMENT:")

    # Check overfitting first (most important)
    if overfit_status == 'SEVERE':
        print("🚨 SEVERE OVERFITTING DETECTED!")
        print("   The model is memorizing data, not learning patterns.")
        print("   Recommendation: Add more coins to training (50+) for better generalization.")
    elif overfit_status == 'WARNING':
        print("⚠️ OVERFITTING WARNING - Results may not generalize!")
        print("   Train metrics are much better than test metrics.")
        print("   Recommendation: Train on more coins to improve generalization.")
    elif cv_f1 >= 0.55 and overfit_status in ['OK', 'MILD']:
        print("✅ GOOD: Model is learning well with acceptable overfitting!")
        print(f"   Cross-validated F1: {cv_f1:.1%} is robust")
        print("   Recommendation: Proceed with full 50+ coin training")
    elif cv_f1 >= 0.45:
        print("⚠️ OK: Model is learning but cross-validated F1 is modest")
        print("   Recommendation: Train on more coins for better patterns")
    else:
        print("❌ WEAK: Model may not be learning effectively")
        print("   Check: Are there enough samples? Are features informative?")

    print("=" * 70)

    return metrics


if __name__ == "__main__":
    run_quick_test()
