# Save as test_connection.py and run it
import requests

print("Testing connections...")

# Test 1: Google (should always work)
try:
    r = requests.get("https://www.google.com", timeout=5)
    print(f"✅ Google: {r.status_code}")
except Exception as e:
    print(f"❌ Google FAILED: {e}")

# Test 2: Binance with SSL
try:
    r = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
    print(f"✅ Binance (SSL): {r.status_code}")
except Exception as e:
    print(f"❌ Binance (SSL) FAILED: {e}")

# Test 3: Binance without SSL verify
try:
    r = requests.get("https://api.binance.com/api/v3/ping", timeout=10, verify=False)
    print(f"✅ Binance (no SSL): {r.status_code}")
except Exception as e:
    print(f"❌ Binance (no SSL) FAILED: {e}")

# Test 4: Yahoo Finance
try:
    r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD", timeout=10)
    print(f"✅ Yahoo Finance: {r.status_code}")
except Exception as e:
    print(f"❌ Yahoo FAILED: {e}")
    