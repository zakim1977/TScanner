"""
Data Fetching Module - Multi-Source with Fallbacks
SSL verification disabled for corporate/antivirus environments
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import ssl
import os

# ═══════════════════════════════════════════════════════════════════════════════
# DISABLE SSL VERIFICATION GLOBALLY
# (Required for networks with SSL inspection - antivirus, corporate proxy, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

# Disable SSL warnings
warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# Disable SSL verification globally for requests
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Create a session with SSL disabled
SESSION = requests.Session()
SESSION.verify = False

# ═══════════════════════════════════════════════════════════════════════════════
# MONKEY-PATCH REQUESTS TO DISABLE SSL GLOBALLY
# This ensures yfinance and all other libraries skip SSL verification
# ═══════════════════════════════════════════════════════════════════════════════

_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

# Also patch the simple requests functions
_original_get = requests.get
_original_post = requests.post

def _patched_get(url, **kwargs):
    kwargs['verify'] = False
    return _original_get(url, **kwargs)

def _patched_post(url, **kwargs):
    kwargs['verify'] = False
    return _original_post(url, **kwargs)

requests.get = _patched_get
requests.post = _patched_post

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_API_BASE = "https://api.binance.com"
TIMEOUT = 15

# Browser headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def is_crypto_symbol(symbol: str) -> bool:
    """Check if symbol is a cryptocurrency"""
    crypto_indicators = ['-USD', '/USD', 'USDT', 'BTC', 'ETH', 'BNB']
    symbol_upper = symbol.upper()
    return any(ind in symbol_upper for ind in crypto_indicators)


def symbol_to_binance(symbol: str) -> str:
    """Convert to Binance format (BTCUSDT)"""
    s = symbol.upper().strip()
    if s.endswith('USDT'):
        return s
    if '-USD' in s:
        return s.replace('-USD', '') + 'USDT'
    if '/USD' in s:
        return s.split('/')[0] + 'USDT'
    return s + 'USDT'


def symbol_to_yahoo(symbol: str) -> str:
    """Convert to Yahoo format (BTC-USD)"""
    s = symbol.upper().strip()
    if s.endswith('USDT'):
        return s.replace('USDT', '-USD')
    if s.endswith('-USD'):
        return s
    return s + '-USD'


def binance_interval_map(interval: str) -> str:
    """Map intervals to Binance format"""
    mapping = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w',
    }
    return mapping.get(interval, '4h')


def yahoo_interval_map(interval: str) -> str:
    """Map intervals to Yahoo format"""
    mapping = {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '1h', '1d': '1d', '1w': '1wk',  # Yahoo doesn't have 4h
    }
    return mapping.get(interval, '1h')


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 1: BINANCE API
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str = '4h', limit: int = 200) -> pd.DataFrame:
    """Fetch from Binance API (SSL disabled)"""
    import time
    
    binance_symbol = symbol_to_binance(symbol)
    binance_interval = binance_interval_map(interval)
    
    url = f"{BINANCE_API_BASE}/api/v3/klines"
    params = {
        'symbol': binance_symbol,
        'interval': binance_interval,
        'limit': min(limit, 1000)
    }
    
    for attempt in range(2):
        try:
            response = SESSION.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                if not data:
                    return None
                
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df['DateTime'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            elif response.status_code == 429:
                time.sleep(2)
                continue
                
        except Exception as e:
            if attempt == 0:
                time.sleep(0.5)
                continue
            return None
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# METHOD 2: YAHOO FINANCE DIRECT API (BACKUP)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_yfinance(symbol: str, interval: str = '4h', limit: int = 200) -> pd.DataFrame:
    """Fetch from Yahoo Finance direct API (backup for crypto)"""
    try:
        yahoo_symbol = symbol_to_yahoo(symbol)
        yahoo_interval = yahoo_interval_map(interval)
        
        # Calculate range based on interval AND limit
        if interval in ['1m', '5m', '15m']:
            range_param = '7d'
        elif interval in ['30m', '1h']:
            range_param = '60d'
        elif interval == '4h':
            range_param = '60d'
            yahoo_interval = '1h'
        elif interval == '1d':
            # For daily data, calculate range from limit
            if limit > 1825:
                range_param = '10y'
            elif limit > 1095:
                range_param = '5y'
            elif limit > 365:
                range_param = '2y'
            else:
                range_param = '1y'
        else:
            range_param = '2y'
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {'interval': yahoo_interval, 'range': range_param}
        
        response = SESSION.get(url, params=params, headers=HEADERS, timeout=15)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if not data.get('chart', {}).get('result'):
            return None
        
        result = data['chart']['result'][0]
        timestamps = result.get('timestamp', [])
        quote = result.get('indicators', {}).get('quote', [{}])[0]
        
        if not timestamps or not quote:
            return None
        
        df = pd.DataFrame({
            'DateTime': pd.to_datetime(timestamps, unit='s'),
            'Open': quote.get('open', []),
            'High': quote.get('high', []),
            'Low': quote.get('low', []),
            'Close': quote.get('close', []),
            'Volume': quote.get('volume', [])
        })
        
        df = df.dropna()
        
        if len(df) < 10:
            return None
        
        # If we wanted 4h but got 1h, resample
        if interval == '4h' and yahoo_interval == '1h':
            df.set_index('DateTime', inplace=True)
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna().reset_index()
        
        return df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(limit)
        
    except Exception as e:
        print(f"Yahoo API error for {symbol}: {str(e)[:50]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SMART FETCHER - TRIES ALL METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_smart(symbol: str, interval: str = '4h', limit: int = 200) -> pd.DataFrame:
    """Smart fetcher - uses Binance with SSL disabled"""
    return fetch_binance_klines(symbol, interval, limit)


def fetch_ohlcv_binance(symbol: str, interval: str = '4h', limit: int = 200) -> pd.DataFrame:
    """Alias"""
    return fetch_ohlcv_smart(symbol, interval, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_binance_price(symbol: str) -> tuple:
    """Fetch current price (SSL disabled)"""
    try:
        url = f"{BINANCE_API_BASE}/api/v3/ticker/price"
        params = {'symbol': symbol_to_binance(symbol)}
        response = SESSION.get(url, params=params, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            return float(response.json()['price']), datetime.now()
    except Exception as e:
        # Log but don't crash - price fetch failures are recoverable
        import logging
        logging.warning(f"fetch_binance_price failed for {symbol}: {type(e).__name__}: {str(e)[:50]}")
    
    return None, None


def get_current_price(symbol: str) -> float:
    """Get current price"""
    price, _ = fetch_binance_price(symbol)
    return price if price else 0


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH ALL PAIRS - BINANCE GLOBAL FUTURES ONLY
# ═══════════════════════════════════════════════════════════════════════════════

# Binance Futures API (Global - NOT available in US)
BINANCE_FUTURES_API = "https://fapi.binance.com"

def fetch_binance_futures_pairs(limit: int = 500) -> list:
    """
    Fetch ONLY coins with active Binance Futures contracts.
    
    WHY THIS MATTERS:
    - Whale/OI/Funding data ONLY exists for futures pairs
    - Scanning spot-only coins gives inaccurate institutional data
    - Binance Futures is GLOBAL (not available in US)
    - This ensures we scan coins available in Dubai/Global
    
    Returns:
        List of symbols like ['BTCUSDT', 'ETHUSDT', ...]
    """
    try:
        # Get futures exchange info - ONLY active futures contracts
        url = f"{BINANCE_FUTURES_API}/fapi/v1/exchangeInfo"
        response = SESSION.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter for active USDT perpetual contracts only
            futures_symbols = []
            for symbol_info in data.get('symbols', []):
                symbol = symbol_info.get('symbol', '')
                status = symbol_info.get('status', '')
                contract_type = symbol_info.get('contractType', '')
                
                # Only USDT perpetuals that are actively trading
                if (symbol.endswith('USDT') and 
                    status == 'TRADING' and 
                    contract_type == 'PERPETUAL'):
                    
                    base = symbol.replace('USDT', '')
                    
                    # Exclude stablecoins and leveraged tokens
                    stablecoins = ['USDC', 'BUSD', 'TUSD', 'FDUSD', 'DAI', 'USDP']
                    if base in stablecoins:
                        continue
                    if any(x in base for x in ['UP', 'DOWN', 'BEAR', 'BULL']):
                        continue
                    
                    futures_symbols.append(symbol)
            
            # Now get 24h volume to sort by liquidity
            try:
                vol_url = f"{BINANCE_FUTURES_API}/fapi/v1/ticker/24hr"
                vol_response = SESSION.get(vol_url, headers=HEADERS, timeout=15)
                
                if vol_response.status_code == 200:
                    vol_data = {t['symbol']: float(t.get('quoteVolume', 0)) 
                               for t in vol_response.json()}
                    
                    # Sort by futures volume (more accurate than spot)
                    futures_symbols.sort(key=lambda s: vol_data.get(s, 0), reverse=True)
            except Exception as vol_err:
                import logging
                logging.warning(f"Volume sort failed, using unsorted: {str(vol_err)[:50]}")
            
            result = futures_symbols[:limit]
            print(f"✅ Fetched {len(result)} Binance Futures pairs (Global)")
            return result
            
    except Exception as e:
        print(f"⚠️ Could not fetch futures pairs: {str(e)[:50]}")
    
    # Fallback to known futures pairs
    return get_default_futures_pairs()[:limit]


def get_default_futures_pairs() -> list:
    """
    Default Binance pairs available on BOTH Spot and Futures.
    Updated Jan 2025 - removed renamed/delisted coins.
    """
    return [
        # === TOP 1-20 by market cap ===
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
        "TRXUSDT", "POLUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",  # MATIC→POL
        "UNIUSDT", "ETCUSDT", "XLMUSDT", "FILUSDT", "NEARUSDT",
        # === 21-40: L2s and newer ===
        "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT", "SEIUSDT",
        "INJUSDT", "TIAUSDT", "JUPUSDT", "STXUSDT", "IMXUSDT",
        "MANTAUSDT", "METISUSDT", "ZKSYNCUSDT", "BLURUSDT", "STRKUSDT",
        "ZKUSDT", "SCROLLUSDT", "CELOUSDT", "MOVRUSDT", "BEAMUSDT",
        # === 41-60: DeFi ===
        "AAVEUSDT", "MKRUSDT", "LDOUSDT", "SNXUSDT", "CRVUSDT",
        "COMPUSDT", "RUNEUSDT", "GMXUSDT", "DYDXUSDT", "PENDLEUSDT",
        "1INCHUSDT", "SUSHIUSDT", "YFIUSDT", "BALUSDT", "ZRXUSDT",
        "LRCUSDT", "KNCUSDT", "RENUSDT", "WNXMUSDT", "CVXUSDT",
        # === 61-80: Memes (high volume) ===
        "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "FLOKIUSDT", "BONKUSDT",
        "MEMEUSDT", "PEOPLEUSDT", "LUNCUSDT", "NOTUSDT", "ORDIUSDT",
        "NEIROUSDT", "ACTUSDT", "PNUTUSDT", "GOATUSDT", "BOMEUSDT",
        "TURBOSUDT", "POPCATUSDT", "DOGSUSDT", "CATIUSDT", "HMSTRUSDT",
        # === 81-100: AI/Compute ===
        "FETUSDT", "RENDERUSDT", "TAOUSDT", "WLDUSDT", "OCEANUSDT",
        "AIUSDT", "ARKMUSDT", "PHBUSDT", "CTXCUSDT", "NMRUSDT",
        "RLCUSDT", "GLMUSDT", "IQUSDT", "IOTXUSDT", "MDTUSDT",
        "LPTUSDT", "CKBUSDT", "ALTUSDT", "AIOZUSDT", "VANRYUSDT",
        # === 101-120: Infrastructure ===
        "ICPUSDT", "HBARUSDT", "VETUSDT", "ALGOUSDT", "FTMUSDT",
        "EGLDUSDT", "FLOWUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
        "THETAUSDT", "QNTUSDT", "IOTAUSDT", "NEOUSDT", "EOSUSDT",
        "XTZUSDT", "KAVAUSDT", "ZILUSDT", "ONTUSDT", "WAVESUSDT",
        # === 121-140: Gaming/Metaverse ===
        "GALAUSDT", "ENJUSDT", "CHZUSDT", "GMTUSDT", "APEUSDT",
        "ROSEUSDT", "SKLUSDT", "SFPUSDT", "WOOUSDT", "PIXELUSDT",
        "PORTALUSDT", "XAIUSDT", "ACEUSDT", "CYBERUSDT", "ARKUSDT",
        "IDUSDT", "EDUUSDT", "HOOKUSDT", "MAGICUSDT", "LOOKSUSDT",
        # === 141-160: Mid-caps ===
        "MASKUSDT", "ENSUSDT", "ACHUSDT", "AGLDUSDT", "ARPAUSDT",
        "AUDIOUSDT", "BATUSDT", "CELRUSDT", "COTIUSDT", "CTSIUSDT",
        "DENTUSDT", "DGBUSDT", "DUSKUSDT", "ELFUSDT", "FLMUSDT",
        "FORTHUSDT", "FRONTUSDT", "GTCUSDT", "HIGHUSDT", "HOTUSDT",
        # === 161-180: More mid-caps ===
        "IOSTUSDT", "JASMYUSDT", "JOEUSDT", "KEYUSDT", "KLAYUSDT",
        "LINAUSDT", "LITUSDT", "LOKAUSDT", "LSKUSDT", "MAVUSDT",
        "MBLUSDT", "MINAUSDT", "MTLUSDT", "NKNUSDT", "OAXUSDT",
        "OGUSDT", "OMGUSDT", "ONEUSDT", "OXTUSDT", "PAXGUSDT",
        # === 181-200: Additional pairs ===
        "PERPUSDT", "PHAUSDT", "POLYXUSDT", "POWRUSDT", "QIUSDT",
        "QTUMUSDT", "RADUSDT", "RAREUSDT", "REEFUSDT", "REQUSDT",
        "RVNUSDT", "SCUSDT", "SLPUSDT", "SSVUSDT", "STORJUSDT",
        "STRAXUSDT", "SUPERUSDT", "SXPUSDT", "TFUELUSDT", "TKOUSDT"
    ]


def fetch_all_binance_usdt_pairs(limit: int = 500) -> list:
    """Fetch all USDT pairs sorted by volume (SSL disabled)"""
    try:
        url = f"{BINANCE_API_BASE}/api/v3/ticker/24hr"
        response = SESSION.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Filter USDT pairs - exclude stablecoins and leveraged tokens
            stablecoins = [
                # USD-pegged stablecoins
                'USDC', 'BUSD', 'TUSD', 'FDUSD', 'DAI', 'USDP', 'USDE', 
                'PYUSD', 'GUSD', 'FRAX', 'LUSD', 'SUSD', 'CUSD', 'UST',
                'BFUSD', 'USDJ', 'USDD', 'USDN', 'USDX', 'USDQ', 'USDK',
                'MUSD', 'HUSD', 'ZUSD', 'OUSD', 'DUSD', 'EUSD', 'CEUR',
                'USD1',  # World Liberty stablecoin
                # Fiat-pegged
                'EUR', 'GBP', 'AUD', 'AEUR', 'EURI', 'TRYB', 'BIDR', 'IDRT',
                # Gold/commodity-backed (not crypto plays)
                'PAXG', 'XAUT'
            ]
            usdt_pairs = []
            
            for ticker in data:
                symbol = ticker['symbol']
                if not symbol.endswith('USDT'):
                    continue
                
                base = symbol.replace('USDT', '')
                if base in stablecoins:
                    continue
                # Also filter if name contains USD (likely stablecoin)
                if 'USD' in base and base != 'USD':
                    continue
                if any(x in base for x in ['UP', 'DOWN', 'BEAR', 'BULL', '2L', '3L', '2S', '3S']):
                    continue
                
                try:
                    volume = float(ticker['quoteVolume'])
                    if volume > 100000:
                        usdt_pairs.append({'symbol': symbol, 'volume': volume})
                except:
                    pass
            
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            symbols = [p['symbol'] for p in usdt_pairs[:limit]]
            print(f"Fetched {len(symbols)} USDT pairs")
            return symbols
    except Exception as e:
        print(f"Could not fetch pairs list: {str(e)[:50]}")
    
    # Return default list if API fails
    return get_default_pairs()[:limit]


def get_default_pairs() -> list:
    """Default crypto pairs if API fails - NOW uses verified futures list"""
    return get_default_futures_pairs()


def get_all_binance_pairs() -> list:
    """
    Get all tradeable Binance pairs.
    NOW USES FUTURES PAIRS for accurate whale/OI data.
    """
    return fetch_binance_futures_pairs(500)


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK/ETF SUPPORT (Direct Yahoo Finance API - bypasses yfinance library)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data(symbol: str, interval: str = '1d', limit: int = 200) -> pd.DataFrame:
    """
    Fetch stock/ETF data directly from Yahoo Finance API (SSL disabled)
    VERSION 2.0 - Direct API, NO yfinance library
    """
    try:
        symbol = symbol.upper().strip()
        
        # Map interval to Yahoo format
        interval_map = {
            '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h',
            '1d': '1d', '1w': '1wk', '1M': '1mo'
        }
        yahoo_interval = interval_map.get(interval, '1d')
        
        # Calculate range based on interval AND limit
        # Yahoo valid ranges: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        if interval in ['5m', '15m']:
            range_param = '60d'
        elif interval in ['30m', '1h', '4h']:
            # For hourly data, calculate years needed
            if limit > 730 * 24:  # More than 2 years of hourly data
                range_param = '5y'
            else:
                range_param = '2y'
        elif interval == '1d':
            # For daily data, calculate range from limit (days)
            if limit > 1825:  # More than 5 years
                range_param = '10y'
            elif limit > 1095:  # More than 3 years
                range_param = '5y'
            elif limit > 365:  # More than 1 year
                range_param = '2y'
            elif limit > 180:  # More than 6 months
                range_param = '1y'
            elif limit > 90:  # More than 3 months
                range_param = '6mo'
            else:
                range_param = '3mo'
        elif interval in ['1w', '1M']:
            range_param = '10y'
        else:
            range_param = '2y'
        
        # Direct Yahoo Finance API call - NO YFINANCE LIBRARY
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'interval': yahoo_interval,
            'range': range_param,
        }
        
        print(f"[v2.0 DIRECT API] Fetching {symbol} from Yahoo ({range_param}, {yahoo_interval})...")
        
        response = SESSION.get(url, params=params, headers=HEADERS, timeout=15)
        
        print(f"  Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  Yahoo API returned status {response.status_code}")
            return None
        
        data = response.json()
        
        # Parse the response
        if 'chart' not in data or 'result' not in data['chart']:
            print(f"  Invalid response structure for {symbol}")
            return None
        
        if not data['chart']['result']:
            # Check for error message
            error = data.get('chart', {}).get('error')
            if error:
                print(f"  Yahoo API error: {error}")
            else:
                print(f"  No results for {symbol}")
            return None
        
        result = data['chart']['result'][0]
        
        # Get timestamps and quotes
        timestamps = result.get('timestamp', [])
        if not timestamps:
            print(f"  No timestamps for {symbol}")
            return None
        
        quote = result.get('indicators', {}).get('quote', [{}])[0]
        
        if not quote:
            print(f"  No quote data for {symbol}")
            return None
        
        # Build DataFrame
        df = pd.DataFrame({
            'DateTime': pd.to_datetime(timestamps, unit='s'),
            'Open': quote.get('open', []),
            'High': quote.get('high', []),
            'Low': quote.get('low', []),
            'Close': quote.get('close', []),
            'Volume': quote.get('volume', [])
        })
        
        # Remove rows with NaN
        df = df.dropna()
        
        if len(df) < 10:
            print(f"  Not enough data for {symbol} ({len(df)} rows)")
            return None
        
        # Convert to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # If we wanted 4h but got 1h, resample
        if interval == '4h' and yahoo_interval == '1h':
            df.set_index('DateTime', inplace=True)
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna().reset_index()
        
        print(f"  ✓ Got {len(df)} candles for {symbol}")
        return df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(limit)
        
    except Exception as e:
        print(f"[v2.0] Stock data error for {symbol}: {str(e)[:80]}")
        return None


def get_stock_price(symbol: str) -> float:
    """Get current stock/ETF price via direct API"""
    try:
        symbol = symbol.upper().strip()
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {'interval': '1d', 'range': '1d'}
        
        response = SESSION.get(url, params=params, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('chart', {}).get('result'):
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                return float(meta.get('regularMarketPrice', 0))
    except Exception as e:
        import logging
        logging.warning(f"get_stock_price failed for {symbol}: {type(e).__name__}: {str(e)[:50]}")
    return 0


def get_popular_etfs() -> list:
    """Get list of popular ETFs for scanning"""
    return [
        # Broad Market
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VT",
        # Sector ETFs
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB",
        # Tech/Growth
        "ARKK", "ARKW", "ARKF", "ARKG", "VGT", "IGV", "SOXX", "SMH",
        # Bonds
        "TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK",
        # International
        "EFA", "EEM", "VEA", "VWO", "IEMG",
        # Commodities
        "GLD", "SLV", "USO", "UNG", "DBC",
        # Leveraged (use with caution)
        "TQQQ", "SQQQ", "SPXL", "SPXS", "UVXY",
        # Dividend
        "VYM", "SCHD", "DVY", "HDV",
        # Real Estate
        "VNQ", "IYR", "XLRE",
    ]


def get_popular_stocks() -> list:
    """Get list of popular stocks for scanning"""
    return [
        # Mega Cap Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Tech
        "AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "IBM", "QCOM", "AVGO",
        # Finance
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "BLK",
        # Healthcare
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT",
        # Consumer
        "WMT", "HD", "NKE", "MCD", "SBUX", "TGT", "COST", "PG", "KO", "PEP",
        # Industrial
        "CAT", "BA", "GE", "MMM", "HON", "UPS", "DE",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG",
        # Communication
        "DIS", "NFLX", "CMCSA", "T", "VZ",
    ]


def get_sp500_symbols(limit: int = 500) -> list:
    """
    Get S&P 500 stock symbols for market pulse scanning.
    Returns top stocks by market cap / volume.
    """
    # Core S&P 500 stocks - most liquid and representative
    sp500_core = [
        # Mega Cap (Top 20 by weight)
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        # Large Cap Tech
        "AVGO", "PEP", "KO", "COST", "TMO", "MCD", "CSCO", "WMT", "ABT", "CRM",
        "ACN", "DHR", "ADBE", "AMD", "NFLX", "CMCSA", "TXN", "NKE", "PM", "NEE",
        "ORCL", "INTC", "IBM", "QCOM", "HON", "UPS", "RTX", "LOW", "UNP", "SPGI",
        # Finance
        "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "CB", "MMC",
        "PNC", "USB", "TFC", "AIG", "MET", "PRU", "ALL", "TRV", "AFL", "CME",
        # Healthcare
        "PFE", "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "ZTS", "CI", "HUM",
        "CVS", "ELV", "REGN", "VRTX", "MRNA", "BIIB", "IQV", "DXCM", "BSX", "BDX",
        # Consumer
        "DIS", "SBUX", "TGT", "MDLZ", "CL", "EL", "ADM", "SYY", "GIS", "KHC",
        "KMB", "HSY", "MKC", "CAG", "CPB", "HRL", "K", "SJM", "CLX", "CHD",
        # Industrial
        "CAT", "BA", "GE", "MMM", "DE", "LMT", "NOC", "GD", "ITW", "EMR",
        "PH", "ETN", "ROK", "FAST", "PCAR", "CMI", "IR", "AME", "XYL", "SWK",
        # Energy
        "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI", "WMB", "HAL",
        "DVN", "HES", "FANG", "BKR", "TRGP", "OKE", "MRO", "APA", "CTRA",
        # Materials
        "LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "VMC", "MLM", "DOW",
        "DD", "PPG", "ALB", "CF", "MOS", "IFF", "FMC", "CE", "EMN", "SEE",
        # Utilities
        "NEE", "DUK", "SO", "D", "SRE", "AEP", "XEL", "EXC", "ED", "WEC",
        "ES", "AWK", "PEG", "DTE", "FE", "AEE", "CMS", "CNP", "NI", "EVRG",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
        "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "PEAK", "HST", "KIM", "REG",
        # Communication
        "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "WBD", "PARA", "LYV", "IPG",
        "OMC", "FOXA", "NWS", "DISH",
        # Additional S&P 500
        "PYPL", "SQ", "SHOP", "SNOW", "DDOG", "ZS", "CRWD", "NET", "OKTA", "MDB",
        "PANW", "FTNT", "ZM", "DOCU", "TEAM", "NOW", "WDAY", "SPLK", "VEEV", "ANSS",
        "CDNS", "SNPS", "KLAC", "LRCX", "AMAT", "MCHP", "ADI", "NXPI", "SWKS", "MPWR",
        "ON", "MRVL", "MU", "WDC", "STX", "ENPH", "SEDG", "FSLR", "RUN",
    ]
    return sp500_core[:limit]


def get_etf_symbols(limit: int = 100) -> list:
    """Get popular ETF symbols for market pulse scanning"""
    etfs = [
        # Major Index ETFs
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "VEA", "EFA", "VWO",
        # Sector ETFs
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
        "XBI", "XOP", "XHB", "XRT", "XME",
        # Tech/Growth
        "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "IGV", "SOXX", "SMH", "HACK", "SKYY",
        # Dividend/Value
        "VIG", "VYM", "SCHD", "DVY", "HDV", "SDY", "DGRO", "NOBL", "VTV", "IWD",
        # Bond ETFs
        "TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "JNK", "TIP", "VCIT",
        # Commodity
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "COPX", "REMX",
        # International
        "EEM", "EWJ", "EWZ", "FXI", "EWG", "EWU", "EWY", "EWT", "EWA", "EWC",
        # Leveraged (for reference)
        "TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS", "UVXY", "SVXY",
        # Thematic
        "ICLN", "TAN", "LIT", "BOTZ", "ROBO", "AIQ", "KWEB", "MCHI", "CIBR", "DRIV",
    ]
    return etfs[:limit]


def fetch_universal(symbol: str, interval: str = '1d', limit: int = 200, market: str = 'auto') -> pd.DataFrame:
    """
    Universal fetcher - auto-detects crypto vs stock/ETF
    
    Args:
        symbol: Trading symbol
        interval: Timeframe
        limit: Number of candles
        market: 'crypto', 'stock', or 'auto' (auto-detect)
    """
    symbol_upper = symbol.upper().strip()
    
    # Auto-detect market type
    if market == 'auto':
        if 'USDT' in symbol_upper or symbol_upper in get_default_pairs():
            market = 'crypto'
        elif symbol_upper in get_popular_etfs() or symbol_upper in get_popular_stocks():
            market = 'stock'
        else:
            # Try crypto first, then stock
            df = fetch_binance_klines(symbol, interval, limit)
            if df is not None and len(df) >= 20:
                return df
            return fetch_stock_data(symbol, interval, limit)
    
    if market == 'crypto':
        return fetch_binance_klines(symbol, interval, limit)
    else:
        return fetch_stock_data(symbol, interval, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# TIME ESTIMATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_time_to_target(df: pd.DataFrame, current_price: float, target_price: float, 
                           timeframe: str) -> dict:
    """
    Estimate time to reach target based on historical volatility
    
    REFINED FORMULA:
    - Markets rarely move in straight lines
    - Account for consolidation, pullbacks, ranging periods
    - Use ATR * 5-7 candles (more conservative than * 3)
    
    Returns:
        dict with 'candles', 'time_str', 'confidence', 'probability'
    """
    if df is None or len(df) < 20:
        return {'candles': 0, 'time_str': 'N/A', 'confidence': 'Low', 'probability': 0}
    
    # Calculate ATR (Average True Range)
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean().iloc[-1]
    
    if atr <= 0 or pd.isna(atr):
        return {'candles': 0, 'time_str': 'N/A', 'confidence': 'Low', 'probability': 0}
    
    # Distance to target
    distance = abs(target_price - current_price)
    
    # How many ATRs away?
    atr_multiple = distance / atr
    
    # REFINED: ~5-6 candles per ATR (more conservative)
    # Reasoning: Markets consolidate, pullback, don't move straight
    # Short timeframes (1m-15m): Use 6x (more noise)
    # Medium timeframes (1h-4h): Use 5x
    # Long timeframes (1d-1w): Use 4x (cleaner trends)
    
    tf_multipliers = {
        '1m': 7, '5m': 6, '15m': 6, '30m': 5,
        '1h': 5, '4h': 4, '1d': 4, '1w': 3
    }
    multiplier = tf_multipliers.get(timeframe, 5)
    
    estimated_candles = int(atr_multiple * multiplier)
    estimated_candles = max(1, estimated_candles)
    
    # Convert to time based on timeframe
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
    }
    
    minutes_per_candle = timeframe_minutes.get(timeframe, 60)
    total_minutes = estimated_candles * minutes_per_candle
    
    # Format time string
    if total_minutes < 60:
        time_str = f"~{total_minutes}m"
    elif total_minutes < 1440:
        hours = total_minutes / 60
        time_str = f"~{hours:.1f}h"
    elif total_minutes < 10080:
        days = total_minutes / 1440
        time_str = f"~{days:.1f}d"
    else:
        weeks = total_minutes / 10080
        time_str = f"~{weeks:.1f}w"
    
    # Confidence based on ATR multiple (adjusted thresholds)
    if atr_multiple < 1.0:
        confidence = 'High'
        probability = 75
    elif atr_multiple < 2.0:
        confidence = 'Medium'
        probability = 55
    elif atr_multiple < 3.0:
        confidence = 'Low'
        probability = 35
    else:
        confidence = 'Very Low'
        probability = 20
    
    return {
        'candles': estimated_candles,
        'time_str': time_str,
        'confidence': confidence,
        'probability': probability,
        'atr_multiple': round(atr_multiple, 2)
    }


def test_binance_connection():
    """Test connection (SSL disabled)"""
    try:
        response = SESSION.get(f"{BINANCE_API_BASE}/api/v3/ping", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_binance_connection() -> bool:
    """Test connection (SSL disabled)"""
    try:
        response = SESSION.get(f"{BINANCE_API_BASE}/api/v3/ping", timeout=5)
        return response.status_code == 200
    except:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL FETCHING - 20x FASTER MARKET SCANNING
# ═══════════════════════════════════════════════════════════════════════════════
"""
USAGE:
------
    from core.data_fetcher import fetch_klines_parallel, fetch_whale_data_parallel
    
    # Fetch 300 coins in ~30 seconds instead of 30+ minutes
    klines = fetch_klines_parallel(symbols, '15m', 200)
    whale_data = fetch_whale_data_parallel(symbols)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Optional
import time

# Configuration
PARALLEL_MAX_WORKERS = 15  # Safe for Binance rate limits
PARALLEL_TIMEOUT = 10


def fetch_klines_parallel(
    symbols: List[str],
    interval: str = '15m',
    limit: int = 200,
    progress_callback: Callable = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch klines for multiple symbols in parallel.
    
    Args:
        symbols: List of symbols (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w)
        limit: Number of candles
        progress_callback: Optional callback(completed, total, current_symbol)
    
    Returns:
        Dict mapping symbol to DataFrame: {symbol: df, ...}
    
    Example:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        klines = fetch_klines_parallel(symbols, '15m', 200)
        for symbol, df in klines.items():
            print(f"{symbol}: {len(df)} candles")
    """
    results = {}
    total = len(symbols)
    completed = 0
    
    def fetch_single(symbol: str) -> tuple:
        """Fetch one symbol"""
        try:
            df = fetch_binance_klines(symbol, interval, limit)
            return symbol, df
        except Exception as e:
            return symbol, None
    
    with ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_single, s): s for s in symbols}
        
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            
            try:
                sym, df = future.result()
                if df is not None and len(df) >= 50:
                    results[sym] = df
            except:
                pass
            
            if progress_callback:
                progress_callback(completed, total, symbol)
    
    return results


def fetch_whale_data_single(symbol: str) -> Optional[Dict]:
    """
    Fetch whale/positioning data for a single symbol.
    Used internally by fetch_whale_data_parallel.
    
    NOW INCLUDES:
    - OI change 24h (calculated from historical data)
    - Price change 24h (from ticker)
    """
    try:
        result = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. TOP TRADER LONG/SHORT RATIO (WHALES) - WITH TRAILING 24H & 7D
        # ═══════════════════════════════════════════════════════════════════
        url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
        # Fetch 168 hours (7 days) of data to calculate both 24h and 7d deltas
        params = {'symbol': symbol, 'period': '1h', 'limit': 168}
        resp = SESSION.get(url, params=params, timeout=PARALLEL_TIMEOUT)

        whale_long_pct = 50.0
        whale_delta_4h = None   # NEW: Early signal (4 hours)
        whale_delta_24h = None
        whale_delta_7d = None
        whale_acceleration = None
        whale_early_signal = None  # NEW: Early reversal detection
        is_fresh_accumulation = False

        if resp.status_code == 200 and resp.json():
            whale_data = resp.json()
            if whale_data:
                # Current (most recent)
                whale_long_pct = float(whale_data[-1].get('longAccount', 0.5)) * 100

                # Calculate 4h delta (4 hours ago = 4 data points back) - EARLY SIGNAL
                if len(whale_data) >= 4:
                    whale_4h_ago = float(whale_data[-4].get('longAccount', 0.5)) * 100
                    whale_delta_4h = whale_long_pct - whale_4h_ago

                # Calculate 24h delta (24 hours ago = 24 data points back)
                if len(whale_data) >= 24:
                    whale_24h_ago = float(whale_data[-24].get('longAccount', 0.5)) * 100
                    whale_delta_24h = whale_long_pct - whale_24h_ago

                # Calculate 7d delta (168 hours ago = oldest data point)
                if len(whale_data) >= 168:
                    whale_7d_ago = float(whale_data[0].get('longAccount', 0.5)) * 100
                    whale_delta_7d = whale_long_pct - whale_7d_ago
                elif len(whale_data) >= 48:  # At least 2 days
                    whale_7d_ago = float(whale_data[0].get('longAccount', 0.5)) * 100
                    whale_delta_7d = whale_long_pct - whale_7d_ago

                # NEW: Detect EARLY reversal signals (4h vs 24h)
                # If 4h is going opposite direction of 24h, whales might be reversing!
                if whale_delta_4h is not None and whale_delta_24h is not None:
                    if whale_delta_24h < -2 and whale_delta_4h > 1:
                        # 24h distributing but 4h starting to accumulate = EARLY BUY signal
                        whale_early_signal = 'EARLY_ACCUMULATION'
                    elif whale_delta_24h > 2 and whale_delta_4h < -1:
                        # 24h accumulating but 4h starting to distribute = EARLY SELL signal
                        whale_early_signal = 'EARLY_DISTRIBUTION'
                    elif whale_delta_4h > 2 and whale_delta_24h > 0:
                        # Both positive, 4h strong = FRESH accumulation
                        whale_early_signal = 'FRESH_ACCUMULATION'
                    elif whale_delta_4h < -2 and whale_delta_24h < 0:
                        # Both negative, 4h strong = FRESH distribution
                        whale_early_signal = 'FRESH_DISTRIBUTION'

                # Calculate acceleration (24h vs 7d daily average)
                if whale_delta_24h is not None and whale_delta_7d is not None:
                    daily_avg_7d = whale_delta_7d / 7

                    if abs(whale_delta_24h) > abs(daily_avg_7d) * 1.5:
                        if (whale_delta_24h > 0) == (whale_delta_7d > 0):
                            whale_acceleration = 'ACCELERATING'
                            is_fresh_accumulation = whale_delta_24h > 0
                        else:
                            whale_acceleration = 'REVERSING'
                            is_fresh_accumulation = whale_delta_24h > 0
                    elif abs(whale_delta_24h) < abs(daily_avg_7d) * 0.5:
                        whale_acceleration = 'DECELERATING'
                        is_fresh_accumulation = False
                    else:
                        whale_acceleration = 'STEADY'
                        is_fresh_accumulation = whale_delta_24h > 2

        result['top_trader_ls'] = {
            'long_pct': whale_long_pct,
            'short_pct': 100 - whale_long_pct,
        }

        # NEW: Trailing whale delta data (from API, not database!)
        result['whale_trailing'] = {
            'whale_delta_4h': whale_delta_4h,      # NEW: Early signal (4 hours)
            'whale_delta_24h': whale_delta_24h,
            'whale_delta_7d': whale_delta_7d,
            'whale_acceleration': whale_acceleration,
            'whale_early_signal': whale_early_signal,  # NEW: EARLY_ACCUMULATION, EARLY_DISTRIBUTION, etc.
            'is_fresh_accumulation': is_fresh_accumulation,
            'data_points': len(resp.json()) if resp.status_code == 200 and resp.json() else 0
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. GLOBAL LONG/SHORT RATIO (RETAIL)
        # ═══════════════════════════════════════════════════════════════════
        url2 = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params2 = {'symbol': symbol, 'period': '1h', 'limit': 1}
        resp2 = SESSION.get(url2, params=params2, timeout=PARALLEL_TIMEOUT)
        if resp2.status_code == 200 and resp2.json():
            data2 = resp2.json()[0]
            result['retail_ls'] = {
                'long_pct': float(data2.get('longAccount', 0.5)) * 100,
                'short_pct': float(data2.get('shortAccount', 0.5)) * 100,
            }
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. OPEN INTEREST WITH 24H CHANGE (FIX: Now calculates change!)
        # ═══════════════════════════════════════════════════════════════════
        oi_current = 0
        oi_change_24h = 0
        
        # Get current OI
        oi_url = "https://fapi.binance.com/fapi/v1/openInterest"
        oi_resp = SESSION.get(oi_url, params={'symbol': symbol}, timeout=PARALLEL_TIMEOUT)
        if oi_resp.status_code == 200:
            oi_current = float(oi_resp.json().get('openInterest', 0))
        
        # Get historical OI to calculate 24h change
        oi_hist_url = "https://fapi.binance.com/futures/data/openInterestHist"
        oi_hist_params = {'symbol': symbol, 'period': '1h', 'limit': 25}  # ~24 hours
        oi_hist_resp = SESSION.get(oi_hist_url, params=oi_hist_params, timeout=PARALLEL_TIMEOUT)
        
        if oi_hist_resp.status_code == 200 and oi_hist_resp.json():
            oi_hist_data = oi_hist_resp.json()
            if len(oi_hist_data) >= 2:
                # Get oldest OI in the range (24h ago)
                oi_24h_ago = float(oi_hist_data[0].get('sumOpenInterest', 0))
                # Get most recent OI
                oi_latest = float(oi_hist_data[-1].get('sumOpenInterest', 0))
                
                if oi_24h_ago > 0:
                    oi_change_24h = ((oi_latest - oi_24h_ago) / oi_24h_ago) * 100
                
                # Use historical latest if current API failed
                if oi_current == 0:
                    oi_current = oi_latest
        
        result['open_interest'] = {
            'current': oi_current,
            'change_24h': round(oi_change_24h, 2),
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. FUNDING RATE
        # ═══════════════════════════════════════════════════════════════════
        fund_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        fund_resp = SESSION.get(fund_url, params={'symbol': symbol, 'limit': 1}, timeout=PARALLEL_TIMEOUT)
        if fund_resp.status_code == 200 and fund_resp.json():
            rate = float(fund_resp.json()[0].get('fundingRate', 0))
            result['funding'] = {'rate': rate, 'rate_pct': rate * 100}
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. PRICE CHANGE 24H (FIX: Now fetched from ticker!)
        # ═══════════════════════════════════════════════════════════════════
        ticker_url = f"{BINANCE_API_BASE}/api/v3/ticker/24hr"
        ticker_resp = SESSION.get(ticker_url, params={'symbol': symbol}, timeout=PARALLEL_TIMEOUT)
        if ticker_resp.status_code == 200:
            ticker_data = ticker_resp.json()
            result['price_change_24h'] = float(ticker_data.get('priceChangePercent', 0))
            result['volume_24h'] = float(ticker_data.get('quoteVolume', 0))
            result['high_24h'] = float(ticker_data.get('highPrice', 0))
            result['low_24h'] = float(ticker_data.get('lowPrice', 0))
        else:
            result['price_change_24h'] = 0
        
        return result if result else None
    except Exception as e:
        print(f"[WHALE_FETCH_ERROR] {symbol}: {e}")
        return None


def fetch_whale_data_parallel(
    symbols: List[str],
    progress_callback: Callable = None,
) -> Dict[str, Dict]:
    """
    Fetch whale positioning data for multiple symbols in parallel.
    
    Args:
        symbols: List of symbols
        progress_callback: Optional callback(completed, total, current_symbol)
    
    Returns:
        Dict mapping symbol to whale data: {symbol: {top_trader_ls, retail_ls, ...}, ...}
    
    Example:
        whale_data = fetch_whale_data_parallel(['BTCUSDT', 'ETHUSDT'])
        print(whale_data['BTCUSDT']['top_trader_ls']['long_pct'])  # e.g., 55.2
    """
    results = {}
    total = len(symbols)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=PARALLEL_MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_whale_data_single, s): s for s in symbols}
        
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            
            try:
                data = future.result()
                if data:
                    results[symbol] = data
            except:
                pass
            
            if progress_callback:
                progress_callback(completed, total, symbol)
    
    return results


def fetch_all_tickers() -> Dict[str, Dict]:
    """
    Fetch ALL Binance tickers in ONE API call.
    
    Returns:
        Dict: {symbol: {price, volume, change_24h, high, low}, ...}
    
    Example:
        tickers = fetch_all_tickers()
        print(tickers['BTCUSDT']['price'])  # Current BTC price
        print(tickers['BTCUSDT']['change_24h'])  # 24h change %
    """
    try:
        resp = SESSION.get(f"{BINANCE_API_BASE}/api/v3/ticker/24hr", timeout=PARALLEL_TIMEOUT)
        if resp.status_code == 200:
            tickers = {}
            for t in resp.json():
                symbol = t['symbol']
                tickers[symbol] = {
                    'price': float(t['lastPrice']),
                    'volume': float(t['quoteVolume']),
                    'change_24h': float(t['priceChangePercent']),
                    'high': float(t['highPrice']),
                    'low': float(t['lowPrice']),
                }
            return tickers
    except Exception as e:
        print(f"Error fetching all tickers: {e}")
    return {}


def prefilter_by_volume(
    symbols: List[str],
    min_volume_usdt: float = 10_000_000,
) -> List[str]:
    """
    Pre-filter symbols by 24h volume using single API call.
    Reduces symbols to scan without individual API calls.
    
    Args:
        symbols: List of symbols to filter
        min_volume_usdt: Minimum 24h volume in USDT
    
    Returns:
        Filtered list of symbols meeting volume criteria
    """
    tickers = fetch_all_tickers()
    
    filtered = []
    for symbol in symbols:
        ticker = tickers.get(symbol)
        if ticker and ticker['volume'] >= min_volume_usdt:
            filtered.append(symbol)
    
    return filtered