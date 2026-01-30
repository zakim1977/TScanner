import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Test GIGGLE whale data
url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
r = requests.get(url, params={'symbol': 'GIGGLEUSDT', 'period': '1h', 'limit': 1}, verify=False)
print("Whale (Top Trader) Data:")
print(r.json())

# Test GIGGLE retail data  
url2 = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
r2 = requests.get(url2, params={'symbol': 'GIGGLEUSDT', 'period': '1h', 'limit': 1}, verify=False)
print("\nRetail Data:")
print(r2.json())
