import requests
url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
r = requests.get(url, params={'symbol': 'GIGGLEUSDT', 'period': '1h', 'limit': 1})
print(r.json())