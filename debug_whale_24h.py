"""
Debug script to test trailing 24h whale data from Binance API.
Run: python debug_whale_24h.py
"""
import sys
sys.path.insert(0, '.')

import requests

print("=" * 70)
print("TESTING TRAILING 24h WHALE DATA (FROM API)")
print("=" * 70)

symbol = "BTCUSDT"
url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"

# Fetch 168 hours (7 days) of data
params = {'symbol': symbol, 'period': '1h', 'limit': 168}

print(f"\nFetching whale data for {symbol}...")
print(f"URL: {url}")
print(f"Params: {params}")

try:
    resp = requests.get(url, params=params, timeout=10)
    print(f"\nStatus: {resp.status_code}")

    if resp.status_code == 200 and resp.json():
        data = resp.json()
        print(f"Data points received: {len(data)}")

        if len(data) >= 2:
            # Show first and last few data points
            print(f"\nOldest data point (7d ago):")
            print(f"  Timestamp: {data[0].get('timestamp')}")
            print(f"  Whale Long %: {float(data[0].get('longAccount', 0.5)) * 100:.1f}%")

            print(f"\nMost recent data point:")
            print(f"  Timestamp: {data[-1].get('timestamp')}")
            print(f"  Whale Long %: {float(data[-1].get('longAccount', 0.5)) * 100:.1f}%")

            # Calculate deltas
            current_whale = float(data[-1].get('longAccount', 0.5)) * 100
            oldest_whale = float(data[0].get('longAccount', 0.5)) * 100

            print("\n" + "=" * 70)
            print("CALCULATED DELTAS")
            print("=" * 70)

            # 7d delta (using all available data)
            whale_delta_7d = current_whale - oldest_whale
            print(f"\n7d Delta: {whale_delta_7d:+.1f}%")
            print(f"  (From {oldest_whale:.1f}% to {current_whale:.1f}%)")

            # 24h delta (if we have at least 24 data points)
            if len(data) >= 24:
                whale_24h_ago = float(data[-24].get('longAccount', 0.5)) * 100
                whale_delta_24h = current_whale - whale_24h_ago
                print(f"\n24h Delta: {whale_delta_24h:+.1f}%")
                print(f"  (From {whale_24h_ago:.1f}% to {current_whale:.1f}%)")

                # Calculate acceleration
                daily_avg_7d = whale_delta_7d / 7
                print(f"\n7d Daily Average: {daily_avg_7d:+.2f}%")

                if abs(whale_delta_24h) > abs(daily_avg_7d) * 1.5:
                    if (whale_delta_24h > 0) == (whale_delta_7d > 0):
                        print(f"\nAcceleration: ACCELERATING")
                        print(f"  24h change ({whale_delta_24h:+.1f}%) is 1.5x faster than daily avg ({daily_avg_7d:+.2f}%)")
                    else:
                        print(f"\nAcceleration: REVERSING")
                        print(f"  24h direction differs from 7d trend!")
                elif abs(whale_delta_24h) < abs(daily_avg_7d) * 0.5:
                    print(f"\nAcceleration: DECELERATING")
                    print(f"  24h change ({whale_delta_24h:+.1f}%) is slower than daily avg ({daily_avg_7d:+.2f}%)")
                else:
                    print(f"\nAcceleration: STEADY")

                # Fresh vs Late entry
                is_fresh = whale_delta_24h > 0 and whale_delta_24h > daily_avg_7d
                is_late = whale_delta_7d > 5 and whale_delta_24h < daily_avg_7d * 0.5

                if is_fresh:
                    print(f"\nEntry Timing: FRESH ENTRY")
                elif is_late:
                    print(f"\nEntry Timing: LATE ENTRY (7d move done, 24h slowing)")
                else:
                    print(f"\nEntry Timing: NORMAL")

            else:
                print(f"\n⚠️ Only {len(data)} data points - need 24+ for 24h delta")

            print("\n" + "=" * 70)
            print("SUCCESS: API provides trailing 24h and 7d data directly!")
            print("No need to rely on stored snapshots.")
            print("=" * 70)

        else:
            print(f"\n⚠️ Not enough data points: {len(data)}")

    else:
        print(f"\n❌ API error: {resp.text[:200]}")

except Exception as e:
    print(f"\n❌ Error: {e}")
