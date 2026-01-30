

import requests
import json
import urllib3
urllib3.disable_warnings()

API_KEY = "99e07660d21547c2ac87f535fb04a58d"

headers = {"Accept": "application/json", "CG-API-KEY": API_KEY}
url = "https://open-api-v4.coinglass.com/api/futures/global-long-short-account-ratio/history"
params = {"symbol": "BTCUSDT", "exchange": "Binance", "interval": "4h", "limit": 3}

response = requests.get(url, headers=headers, params=params, timeout=30, verify=False)
data = response.json()

print("RETAIL FIELDS:", list(data['data'][0].keys()))
print(json.dumps(data['data'][0], indent=2))