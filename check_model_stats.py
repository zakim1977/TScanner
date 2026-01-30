"""Check quality model stats for expected trades per month."""
import pickle

with open('models/quality_model.pkl', 'rb') as f:
    data = pickle.load(f)

metrics = data.get('metrics', {})

print("=" * 60)
print("QUALITY MODEL TRAINING STATS")
print("=" * 60)

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\n" + "=" * 60)
print("KEY METRICS FOR TRADING")
print("=" * 60)

# Extract key metrics
very_high_per_month = metrics.get('very_high_per_month', 0)
high_conf_per_month = metrics.get('high_conf_per_month', 0)
very_high_win_rate = metrics.get('very_high_win_rate', 0)
high_conf_win_rate = metrics.get('high_conf_win_rate', 0)

print(f"\nTrades per month (>60% ML): ~{very_high_per_month:.1f}")
print(f"Trades per month (>50% ML): ~{high_conf_per_month:.1f}")
print(f"\nWin rate (>60% ML): {very_high_win_rate:.1%}")
print(f"Win rate (>50% ML): {high_conf_win_rate:.1%}")

# ROI projection
rr = 2.0  # 2:1 R:R
if very_high_win_rate > 0:
    ev_very_high = (very_high_win_rate * rr) - ((1 - very_high_win_rate) * 1)
    monthly_roi_very_high = ev_very_high * very_high_per_month
    print(f"\nMonthly ROI (>60% only, 1% risk): {monthly_roi_very_high:+.1f}%")

if high_conf_win_rate > 0:
    ev_high = (high_conf_win_rate * rr) - ((1 - high_conf_win_rate) * 1)
    monthly_roi_high = ev_high * high_conf_per_month
    print(f"Monthly ROI (>50% only, 1% risk): {monthly_roi_high:+.1f}%")
