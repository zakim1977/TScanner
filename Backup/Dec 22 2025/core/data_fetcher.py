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
        
        # Calculate range based on interval
        if interval in ['1m', '5m', '15m']:
            range_param = '7d'
        elif interval in ['30m', '1h']:
            range_param = '60d'
        elif interval == '4h':
            range_param = '60d'
            yahoo_interval = '1h'
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
    except:
        pass
    
    return None, None


def get_current_price(symbol: str) -> float:
    """Get current price"""
    price, _ = fetch_binance_price(symbol)
    return price if price else 0


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH ALL PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

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
    """Default crypto pairs if API fails"""
    return [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
        "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
        "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "APTUSDT",
        "NEARUSDT", "FILUSDT", "LDOUSDT", "ARBUSDT", "OPUSDT",
        "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "JUPUSDT",
        "WIFUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "ORDIUSDT",
        "FETUSDT", "RENDERUSDT", "TAOUSDT", "KASUSDT", "RUNEUSDT",
        "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "COMPUSDT",
        "ICPUSDT", "HBARUSDT", "VETUSDT", "ALGOUSDT", "FTMUSDT"
    ]


def get_all_binance_pairs() -> list:
    """Alias"""
    return fetch_all_binance_usdt_pairs(500)


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
        
        # Calculate range based on interval
        if interval in ['5m', '15m']:
            range_param = '60d'
        elif interval in ['30m', '1h', '4h']:
            range_param = '2y'
        elif interval == '1d':
            range_param = '2y'
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
    except:
        pass
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
    
    # Estimate candles: ~1 ATR per 2-4 candles on average
    estimated_candles = int(atr_multiple * 3)
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
    
    # Confidence based on ATR multiple
    if atr_multiple < 1.5:
        confidence = 'High'
        probability = 70
    elif atr_multiple < 3:
        confidence = 'Medium'
        probability = 50
    elif atr_multiple < 5:
        confidence = 'Low'
        probability = 30
    else:
        confidence = 'Very Low'
        probability = 15
    
    return {
        'candles': estimated_candles,
        'time_str': time_str,
        'confidence': confidence,
        'probability': probability,
        'atr_multiple': round(atr_multiple, 1)
    }
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
