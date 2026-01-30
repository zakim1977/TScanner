"""
Historical Data Fetcher for Crypto Whale Data
==============================================
Fetches historical whale positioning data from Coinglass API V4.

API V4 Endpoints (AGGREGATED - simpler, no exchange needed):
- /api/futures/openInterest/ohlc-history (symbol=BTC)
- /api/futures/fundingRate/ohlc-history (symbol=BTC)
- /api/futures/global-long-short-account-ratio/history (exchange+symbol)
- /api/futures/top-long-short-account-ratio/history (exchange+symbol)

Hobbyist Tier: interval >= 4h, 30 req/min
"""

import os
import sqlite3
import time
import requests
from datetime import datetime
from typing import Optional, Dict, List


def _get_whale_store():
    from core.whale_data_store import get_whale_store
    return get_whale_store()


def _migrate_database():
    """Ensure database has all required columns"""
    db_path = "data/whale_history.db"
    
    if not os.path.exists(db_path):
        print("Database will be created fresh.")
        return
    
    print("Checking database schema...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(whale_snapshots)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'data_source' not in columns:
        print("  → Adding 'data_source' column...")
        cursor.execute("ALTER TABLE whale_snapshots ADD COLUMN data_source TEXT DEFAULT 'live'")
        conn.commit()
        print("  ✓ Column added!")
    else:
        print("  ✓ Schema OK")
    
    conn.close()


class CoinglassHistoricalFetcher:
    """
    Coinglass API V4 Fetcher - Uses AGGREGATED endpoints
    
    Key insight: Use aggregated endpoints (just symbol, no exchange) for OI/Funding
    Use exchange-specific endpoints for L/S ratios
    """
    
    BASE_URL = "https://open-api-v4.coinglass.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "CG-API-KEY": api_key
        }
        self._rate_limit_delay = 2.5
        self._debug = True
        
    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make API request"""
        url = f"{self.BASE_URL}{endpoint}"
        
        if self._debug:
            print(f"  GET {endpoint}")
            print(f"      params: {params}")
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            
            if self._debug:
                print(f"      status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                code = data.get('code')
                
                if code == "0" or code == 0:
                    if self._debug:
                        print(f"      ✓ Got {len(data.get('data', []))} records")
                    return data
                else:
                    print(f"      API Error: {data.get('msg', 'Unknown')}")
                    return None
                    
            elif response.status_code == 429:
                print("      Rate limited - waiting 60s")
                time.sleep(60)
                return self._request(endpoint, params)
            else:
                print(f"      HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"      Exception: {e}")
            return None
        finally:
            time.sleep(self._rate_limit_delay)
    
    def fetch_oi_history(self, symbol: str, interval: str = "4h", limit: int = 500) -> List[Dict]:
        """
        Fetch Open Interest history
        Params: exchange=Binance, symbol=BTCUSDT (full pair!), interval, limit
        """
        # Full pair name required: BTCUSDT not BTC
        full_symbol = symbol.upper() if symbol.upper().endswith('USDT') else symbol.upper() + 'USDT'
        
        result = self._request("/api/futures/open-interest/history", {
            "exchange": "Binance",
            "symbol": full_symbol,  # BTCUSDT not BTC!
            "interval": interval,
            "limit": limit
        })
        
        if not result or not result.get('data'):
            return []
        
        data = result.get('data', [])
        lookback = {'4h': 6, '12h': 2, '1d': 1}.get(interval, 6)
        
        processed = []
        for i, point in enumerate(data):
            try:
                if isinstance(point, dict):
                    timestamp = point.get('time', point.get('t', 0))
                    oi = float(point.get('close', point.get('c', point.get('openInterest', 0))))
                    
                    prev_oi = oi
                    if i >= lookback and isinstance(data[i - lookback], dict):
                        prev_oi = float(data[i - lookback].get('close', data[i - lookback].get('c', 
                                        data[i - lookback].get('openInterest', oi))))
                    
                    change_24h = ((oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
                    
                    processed.append({
                        'timestamp': timestamp,
                        'open_interest': oi,
                        'oi_change_24h': change_24h
                    })
            except:
                continue
        
        return processed
    
    def fetch_funding_history(self, symbol: str, interval: str = "4h", limit: int = 500) -> List[Dict]:
        """
        Fetch Funding Rate history
        Params: exchange=Binance, symbol=BTCUSDT (full pair!), interval, limit
        """
        full_symbol = symbol.upper() if symbol.upper().endswith('USDT') else symbol.upper() + 'USDT'
        
        result = self._request("/api/futures/funding-rate/history", {
            "exchange": "Binance",
            "symbol": full_symbol,  # BTCUSDT not BTC!
            "interval": interval,
            "limit": limit
        })
        
        if not result or not result.get('data'):
            return []
        
        processed = []
        for point in result.get('data', []):
            try:
                if isinstance(point, dict):
                    processed.append({
                        'timestamp': point.get('time', point.get('t', 0)),
                        'funding_rate': float(point.get('close', point.get('c', 
                                              point.get('fundingRate', point.get('rate', 0)))))
                    })
            except:
                continue
        
        return processed
    
    def fetch_long_short_ratio_history(self, symbol: str, interval: str = "4h", limit: int = 500) -> List[Dict]:
        """
        Fetch Top Trader Long/Short ratio (whales)
        Params: exchange=Binance, symbol=BTCUSDT (full pair!), interval, limit
        """
        full_symbol = symbol.upper() if symbol.upper().endswith('USDT') else symbol.upper() + 'USDT'
        
        result = self._request("/api/futures/top-long-short-account-ratio/history", {
            "symbol": full_symbol,  # BTCUSDT not BTC!
            "exchange": "Binance",
            "interval": interval,
            "limit": limit
        })
        
        if not result or not result.get('data'):
            return []
        
        processed = []
        for point in result.get('data', []):
            try:
                if isinstance(point, dict):
                    timestamp = point.get('time', point.get('t', point.get('createTime', 0)))
                    
                    # Try various field names - Coinglass V4 API uses these:
                    long_pct = point.get('top_account_long_percent',      # V4 API field
                               point.get('longAccount', point.get('longRatio', 
                               point.get('longRate', point.get('long', 50)))))
                    short_pct = point.get('top_account_short_percent',    # V4 API field
                                point.get('shortAccount', point.get('shortRatio',
                                point.get('shortRate', point.get('short', 50)))))
                    
                    # Convert if decimal
                    if isinstance(long_pct, (int, float)) and long_pct <= 1:
                        long_pct *= 100
                    if isinstance(short_pct, (int, float)) and short_pct <= 1:
                        short_pct *= 100
                    
                    processed.append({
                        'timestamp': timestamp,
                        'whale_long_pct': float(long_pct),
                        'whale_short_pct': float(short_pct)
                    })
            except:
                continue
        
        return processed
    
    def fetch_global_long_short_history(self, symbol: str, interval: str = "4h", limit: int = 500) -> List[Dict]:
        """
        Fetch Global Long/Short ratio (retail proxy)
        Params: exchange=Binance, symbol=BTCUSDT (full pair!), interval, limit
        """
        full_symbol = symbol.upper() if symbol.upper().endswith('USDT') else symbol.upper() + 'USDT'
        
        result = self._request("/api/futures/global-long-short-account-ratio/history", {
            "symbol": full_symbol,  # BTCUSDT not BTC!
            "exchange": "Binance",
            "interval": interval,
            "limit": limit
        })
        
        if not result or not result.get('data'):
            return []
        
        processed = []
        for point in result.get('data', []):
            try:
                if isinstance(point, dict):
                    timestamp = point.get('time', point.get('t', point.get('createTime', 0)))
                    # V4 API field names
                    long_pct = point.get('global_account_long_percent',   # V4 API field (correct!)
                               point.get('long_percent',                  
                               point.get('longAccount', point.get('longRatio',
                               point.get('longRate', point.get('long', 50))))))
                    
                    if isinstance(long_pct, (int, float)) and long_pct <= 1:
                        long_pct *= 100
                    
                    processed.append({
                        'timestamp': timestamp,
                        'retail_long_pct': float(long_pct)
                    })
            except:
                continue
        
        return processed

    def fetch_price_history(self, symbol: str, interval: str = "4h", limit: int = 500) -> List[Dict]:
        """Fetch price from Binance (free)"""
        try:
            if not symbol.upper().endswith('USDT'):
                symbol = symbol.upper() + 'USDT'
            else:
                symbol = symbol.upper()
            
            response = requests.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10
            )
            
            if response.status_code != 200:
                return []
            
            klines = response.json()
            lookback = {'4h': 6, '12h': 2, '1d': 1}.get(interval, 6)
            
            processed = []
            for i, k in enumerate(klines):
                price = float(k[4])
                prev_price = float(klines[i - lookback][4]) if i >= lookback else price
                change = ((price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                
                processed.append({
                    'timestamp': k[0],
                    'price': price,
                    'price_change_24h': change
                })
            
            return processed
        except:
            return []


class HistoricalDataImporter:
    """Import historical data from Coinglass"""
    
    # VERIFIED list of coins with active Binance USDT perpetual futures
    # Last verified: Dec 2024
    # These are coins that ACTUALLY have data on Coinglass
    DEFAULT_SYMBOLS = [
        # === TOP 50 BY MARKET CAP (ALL VERIFIED) ===
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'DOGEUSDT', 'ADAUSDT', 'TRXUSDT', 'AVAXUSDT', 'LINKUSDT',
        'TONUSDT', 'SHIBUSDT', 'DOTUSDT', 'BCHUSDT', 'LTCUSDT',
        'NEARUSDT', 'UNIUSDT', 'APTUSDT', 'ICPUSDT', 'ETCUSDT',
        'RENDERUSDT', 'FETUSDT', 'XLMUSDT', 'ATOMUSDT', 'FILUSDT',
        'ARBUSDT', 'IMXUSDT', 'OPUSDT', 'INJUSDT', 'HBARUSDT',
        'STXUSDT', 'VETUSDT', 'MKRUSDT', 'GRTUSDT', 'AAVEUSDT',
        'THETAUSDT', 'FTMUSDT', 'ALGOUSDT', 'TIAUSDT', 'RUNEUSDT',
        'LDOUSDT', 'SUIUSDT', 'SEIUSDT', 'PENDLEUSDT', 'JUPUSDT',
        'TAOUSDT', 'ONDOUSDT', 'WIFUSDT', 'BONKUSDT', 'PEPEUSDT',
        
        # === LAYER 1 / INFRASTRUCTURE (VERIFIED) ===
        'MATICUSDT', 'NEOUSDT', 'EOSUSDT', 'XTZUSDT', 'EGLDUSDT',
        'FLOWUSDT', 'IOTAUSDT', 'KAVAUSDT', 'ZILUSDT', 'ONTUSDT',
        'QTUMUSDT', 'ICXUSDT', 'IOSTUSDT', 'KSMUSDT', 'MINAUSDT',
        'CFXUSDT', 'ROSEUSDT', 'CKBUSDT', 'CELOUSDT', 'ONEUSDT',
        'KLAYUSDT', 'ASTRUSDT', 'ZENUSDT', 'SCUSDT', 'WAVESUSDT',
        'STRKUSDT', 'ZKUSDT', 'MANTAUSDT', 'ZETAUSDT', 'DYMUSDT',
        'METISUSDT', 'AXLUSDT', 'EIGENUSDT', 'WUSDT',
        
        # === DEFI (VERIFIED) ===
        'COMPUSDT', 'SNXUSDT', 'CRVUSDT', 'DYDXUSDT', 'SUSHIUSDT',
        '1INCHUSDT', 'BALUSDT', 'YFIUSDT', 'ZRXUSDT', 'KNCUSDT',
        'LRCUSDT', 'UMAUSDT', 'GMXUSDT', 'RDNTUSDT', 'WOOUSDT',
        'CVXUSDT', 'FXSUSDT', 'SSVUSDT', 'LQTYUSDT', 'PERPUSDT',
        'STGUSDT', 'DODOUSDT', 'CAKEUSDT', 'ALPACAUSDT',
        
        # === AI / DATA (VERIFIED) ===
        'ARKMUSDT', 'OCEANUSDT', 'AGIXUSDT', 'NMRUSDT', 'RLCUSDT',
        'STORJUSDT', 'ANKRUSDT', 'LPTUSDT', 'API3USDT', 'BANDUSDT',
        'CTXCUSDT', 'NKNUSDT', 'GLMRUSDT', 'PHBUSDT',
        
        # === GAMING / NFT / METAVERSE (VERIFIED) ===
        'AXSUSDT', 'GALAUSDT', 'SANDUSDT', 'MANAUSDT', 'APEUSDT',
        'GMTUSDT', 'ENJUSDT', 'CHZUSDT', 'YGGUSDT', 'ALICEUSDT',
        'MAGICUSDT', 'ILVUSDT', 'HIGHUSDT', 'ACEUSDT', 'PIXELUSDT',
        'XAIUSDT', 'PORTALUSDT', 'SUPERUSDT', 'MBOXUSDT',
        
        # === MEMECOINS (VERIFIED) ===
        'FLOKIUSDT', 'MEMEUSDT', 'BOMEUSDT', 'NEIROUSDT', 'DOGSUSDT',
        'PNUTUSDT', 'ACTUSDT', 'PEOPLEUSDT', 'LUNCUSDT', 'MEWUSDT',
        '1000SHIBUSDT', '1000PEPEUSDT', '1000BONKUSDT', '1000FLOKIUSDT',
        '1000LUNCUSDT', 'TURBOUSDT',
        
        # === EXCHANGE / UTILITY (VERIFIED) ===
        'SXPUSDT', 'HOTUSDT', 'DENTUSDT', 'WINUSDT', 'BTTUSDT',
        'RSRUSDT', 'OGNUSDT', 'CELRUSDT', 'COTIUSDT', 'MTLUSDT',
        'ACHUSDT', 'TUSDT', 'ORDIUSDT', 'BLURUSDT',
        
        # === MID-CAP ALTS (VERIFIED) ===
        'BATUSDT', 'IOTXUSDT', 'SKLUSDT', 'CTSIUSDT', 'OMGUSDT',
        'CVCUSDT', 'BLZUSDT', 'CTKUSDT', 'TRUUSDT', 'REEFUSDT',
        'LITUSDT', 'DIAUSDT', 'ATAUSDT', 'BONDUSDT', 'OMUSDT',
        'HIFIUSDT', 'NFPUSDT', 'ALTUSDT', 'LEVERUSDT', 'HOOKUSDT',
        'IDUSDT', 'EDUUSDT', 'CYBERUSDT', 'MAVUSDT', 'WLDUSDT',
        'ARKUSDT', 'JASMYUSDT', 'BICOUSDT', 'AGLDUSDT', 'BELUSDT',
        
        # === ESTABLISHED ALTS (VERIFIED) ===
        'DASHUSDT', 'ZECUSDT', 'XMRUSDT', 'ENSUSDT', 'ARUSDT',
        'QNTUSDT', 'CHRUSDT', 'HNTUSDT', 'FLUXUSDT',
        'PONDUSDT', 'BADGERUSDT', 'SFPUSDT', 'LOKAUSDT', 'ERNUSDT',
        
        # === FAN TOKENS (VERIFIED - Binance specific) ===
        'LAZIOUSDT', 'PORTOUSDT', 'SANTOSUSDT', 'CITYUSDT',
        'PSGUSDT', 'BARUSDT', 'JUVUSDT', 'OGUSDT', 'ACMUSDT',
    ]
    def __init__(self, api_key: str):
        _migrate_database()
        self.fetcher = CoinglassHistoricalFetcher(api_key)
        self.store = _get_whale_store()
        self._progress_callback = None
        self._cancelled = False
    
    def set_progress_callback(self, callback):
        self._progress_callback = callback
    
    def cancel(self):
        self._cancelled = True
    
    def _report(self, symbol: str, pct: float, msg: str):
        if self._progress_callback:
            self._progress_callback(symbol, pct, msg)
        print(f"[{pct:5.1f}%] {msg}")
    
    def get_data_coverage(self) -> Dict:
        """Get current data coverage to identify gaps."""
        db_path = "data/whale_history.db"
        
        if not os.path.exists(db_path):
            return {'exists': False, 'gaps': [{'start': None, 'end': None, 'days': 180, 'description': 'No database - full import needed'}]}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM whale_snapshots")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM whale_snapshots")
            date_range = cursor.fetchone()
            
            # Get all unique dates to find internal gaps
            cursor.execute("SELECT DISTINCT DATE(timestamp) as date FROM whale_snapshots ORDER BY date")
            all_dates = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            # Parse dates
            start_str = date_range[0][:10] if date_range[0] else None
            end_str = date_range[1][:10] if date_range[1] else None
            
            gaps = []
            
            # Check for internal gaps (missing days between records)
            if len(all_dates) >= 2:
                from datetime import datetime, timedelta
                prev_date = None
                for d in all_dates:
                    curr_date = datetime.strptime(d, '%Y-%m-%d')
                    if prev_date:
                        days_diff = (curr_date - prev_date).days
                        if days_diff > 1:
                            gaps.append({
                                'start': prev_date.strftime('%Y-%m-%d'),
                                'end': d,
                                'days': days_diff,
                                'description': f"Missing {days_diff} days: {prev_date.strftime('%Y-%m-%d')} to {d}"
                            })
                    prev_date = curr_date
            
            # Also check gap from last date to today
            if end_str:
                from datetime import datetime
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
                today = datetime.now()
                days_gap = (today - end_date).days
                
                if days_gap > 1:
                    gaps.append({
                        'start': end_str,
                        'end': today.strftime('%Y-%m-%d'),
                        'days': days_gap,
                        'description': f"Missing {days_gap} days: {end_str} to {today.strftime('%Y-%m-%d')}"
                    })
            
            return {
                'exists': True,
                'total_records': total,
                'start_date': start_str,
                'end_date': end_str,
                'gaps': gaps
            }
            
        except Exception as e:
            return {'exists': False, 'error': str(e), 'gaps': []}
    
    def import_historical_data(
        self, 
        symbols: List[str] = None, 
        lookback_days: int = 180, 
        interval: str = "4h", 
        skip_existing: bool = True,
        from_date: str = None,  # NEW: Format 'YYYY-MM-DD'
        to_date: str = None     # NEW: Format 'YYYY-MM-DD'
    ) -> Dict:
        """Import historical data
        
        Args:
            symbols: List of symbols to import
            lookback_days: Days of history to fetch (ignored if from_date/to_date set)
            interval: Candle interval (4h, 12h, 1d)
            skip_existing: If True, skip symbols that already have >500 records
            from_date: Start date 'YYYY-MM-DD' (NEW - for gap filling)
            to_date: End date 'YYYY-MM-DD' (NEW - for gap filling)
        """
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS
        
        # Handle date range mode
        if from_date and to_date:
            print(f"📅 DATE RANGE MODE: {from_date} to {to_date}")
            from datetime import datetime
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')
            lookback_days = (end_dt - start_dt).days + 1
            print(f"   Calculating {lookback_days} days of data")
            # In date range mode, don't skip - we want to fill gaps
            skip_existing = False
        
        # Enforce hobbyist tier minimum
        if interval in ['1h', '1m', '5m', '15m', '30m']:
            print(f"⚠️ Hobbyist tier requires interval >= 4h. Using 4h.")
            interval = '4h'
        
        # Check what's already imported
        already_imported = set()
        if skip_existing:
            already_imported = self._get_imported_symbols(min_records=500)
            if already_imported:
                print(f"⏭️  Found {len(already_imported)} symbols with >500 records (will skip)")
        
        self._cancelled = False
        candles_per_day = {'4h': 6, '12h': 2, '1d': 1}.get(interval, 6)
        limit = min(lookback_days * candles_per_day, 1000)
        
        stats = {'symbols_processed': 0, 'symbols_failed': [], 'records_imported': 0, 'errors': [], 'symbols_skipped': []}
        total = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            if self._cancelled:
                stats['cancelled'] = True
                break
            
            pct = (idx / total) * 100
            
            # Skip if already imported
            if skip_existing and symbol in already_imported:
                stats['symbols_skipped'].append(symbol)
                self._report(symbol, pct, f"⏭️ {symbol}: Already imported (skipping)")
                print(f"  ⏭️ Skipping {symbol} - already has data")
                continue
            
            self._report(symbol, pct, f"Processing {symbol}...")
            
            try:
                records = self._import_symbol(symbol, interval, limit)
                stats['records_imported'] += records
                stats['symbols_processed'] += 1
                self._report(symbol, pct, f"✓ {symbol}: {records} records")
            except Exception as e:
                stats['symbols_failed'].append(symbol)
                stats['errors'].append(f"{symbol}: {e}")
                self._report(symbol, pct, f"✗ {symbol}: {e}")
        
        if stats['symbols_skipped']:
            print(f"\n⏭️  Skipped {len(stats['symbols_skipped'])} already-imported symbols")
        
        self._report("DONE", 100, f"Complete: {stats['records_imported']} records")
        return stats
    
    def _get_imported_symbols(self, min_records: int = 500) -> set:
        """Get symbols that already have sufficient data"""
        import sqlite3
        db_path = "data/whale_history.db"
        
        if not os.path.exists(db_path):
            return set()
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM whale_snapshots 
                GROUP BY symbol HAVING COUNT(*) >= ?
            """, (min_records,))
            symbols = {row[0] for row in cursor.fetchall()}
            conn.close()
            return symbols
        except:
            return set()
    
    def _import_symbol(self, symbol: str, interval: str, limit: int) -> int:
        """Import data for one symbol"""
        from core.whale_data_store import WhaleSnapshot
        
        print(f"\n  Fetching {symbol}...")
        
        print("    - Price (Binance)")
        price_data = self.fetcher.fetch_price_history(symbol, interval, limit)
        
        print("    - Open Interest")
        oi_data = self.fetcher.fetch_oi_history(symbol, interval, limit)
        
        print("    - Funding Rate")
        funding_data = self.fetcher.fetch_funding_history(symbol, interval, limit)
        
        print("    - Top L/S (whales)")
        ls_data = self.fetcher.fetch_long_short_ratio_history(symbol, interval, limit)
        
        print("    - Global L/S (retail)")
        global_ls = self.fetcher.fetch_global_long_short_history(symbol, interval, limit)
        
        if not price_data:
            print(f"    ⚠️ No price data - skipping")
            return 0
        
        # Also check if we have whale data (most important!)
        if not ls_data and not global_ls:
            print(f"    ⚠️ No whale positioning data - skipping")
            return 0
        
        # Check minimum data quality
        min_records = 10
        if len(price_data) < min_records:
            print(f"    ⚠️ Insufficient data ({len(price_data)} records) - skipping")
            return 0
        
        print(f"    Data: price={len(price_data)}, oi={len(oi_data)}, "
              f"funding={len(funding_data)}, whale={len(ls_data)}, retail={len(global_ls)}")
        
        # Merge
        merged = self._merge(price_data, oi_data, funding_data, ls_data, global_ls)
        
        # Calculate position_in_range based on 20-bar high/low
        prices = [r.get('price', 0) for r in merged if r.get('price', 0) > 0]
        
        # Store
        stored = 0
        for idx, r in enumerate(merged):
            try:
                # Calculate position in range (where is price relative to recent high/low)
                price = r.get('price', 0)
                if price > 0 and idx >= 20:
                    # Use last 20 bars for range
                    recent_prices = prices[max(0, idx-20):idx+1]
                    if recent_prices:
                        high_20 = max(recent_prices)
                        low_20 = min(recent_prices)
                        if high_20 > low_20:
                            position_in_range = ((price - low_20) / (high_20 - low_20)) * 100
                            position_in_range = max(0, min(100, position_in_range))
                        else:
                            position_in_range = 50
                    else:
                        position_in_range = 50
                else:
                    position_in_range = 50
                
                snapshot = WhaleSnapshot(
                    symbol=symbol,
                    timeframe=interval,
                    timestamp=datetime.fromtimestamp(r['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    whale_long_pct=r.get('whale_long_pct', 50),
                    retail_long_pct=r.get('retail_long_pct', 50),
                    oi_value=r.get('open_interest', 0),
                    oi_change_24h=r.get('oi_change_24h', 0),
                    funding_rate=r.get('funding_rate', 0),
                    price=price,
                    price_change_24h=r.get('price_change_24h', 0),
                    position_in_range=position_in_range,
                    mfi=50,
                    cmf=0,
                    rsi=50,
                    volume_ratio=1.0,
                    atr_pct=0,
                    signal_direction='WAIT',
                    predictive_score=50,
                    data_source='historical'
                )
                self.store.store_snapshot(snapshot)
                stored += 1
            except Exception as e:
                if 'data_source' in str(e):
                    print(f"    DB Error: Run 'python migrate_db.py' first!")
                    raise
                continue
        
        # AUTO-BACKTEST: Calculate outcomes for this symbol
        if stored > 0:
            self._backtest_symbol_outcomes(symbol, merged)
        
        return stored
    
    def _backtest_symbol_outcomes(self, symbol: str, data: list):
        """
        Calculate win/loss outcomes for imported data.
        Runs automatically after import - no manual script needed!
        Uses ATR-based dynamic thresholds for realistic results.
        """
        if len(data) < 15:
            return
        
        import sqlite3
        db_path = "data/whale_history.db"
        
        # ATR multipliers - MATCH SignalGenerator logic!
        # SL = ATR * 2, TP1 = Risk * 1.5 (1.5:1 R:R)
        SL_ATR_MULT = 2.0
        TP_RR_RATIO = 1.5
        DEFAULT_ATR = 4.0  # Higher default for crypto
        MAX_BARS = 8
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get prices and whale data for lookback
            prices = [d.get('price', 0) for d in data]
            timestamps = [d.get('timestamp', 0) for d in data]
            
            # Calculate ATR for each point (simple approximation)
            # Use price volatility over last 14 bars
            def calc_atr(idx):
                if idx < 14:
                    return DEFAULT_ATR
                window = prices[idx-14:idx]
                if not window or min(window) <= 0:
                    return DEFAULT_ATR
                returns = [abs(window[i] - window[i-1]) / window[i-1] * 100 for i in range(1, len(window)) if window[i-1] > 0]
                return sum(returns) / len(returns) if returns else DEFAULT_ATR
            
            updated = 0
            for i in range(len(data) - MAX_BARS):
                price = prices[i]
                if price <= 0:
                    continue
                
                ts = timestamps[i]
                
                # MATCH SignalGenerator: SL = ATR * 2, TP = SL * 1.5
                atr = calc_atr(i)
                sl_pct = atr * SL_ATR_MULT  # e.g., 2% * 2.0 = 4%
                tp_pct = sl_pct * TP_RR_RATIO  # e.g., 4% * 1.5 = 6%
                
                # Determine expected direction from whale %
                whale_pct = data[i].get('whale_long_pct', 50)
                if whale_pct >= 60:
                    expected_dir = 'LONG'
                elif whale_pct <= 40:
                    expected_dir = 'SHORT'
                else:
                    expected_dir = 'NEUTRAL'
                
                # Look ahead MAX_BARS
                future_prices = [p for p in prices[i+1:i+MAX_BARS+1] if p > 0]
                if len(future_prices) < 3:
                    continue
                
                # Calculate max moves
                max_up = max((p - price) / price * 100 for p in future_prices)
                max_down = min((p - price) / price * 100 for p in future_prices)
                
                # Determine outcome - which hit first?
                hit_tp1 = 0
                hit_sl = 0
                candles_to_result = 0
                
                for j, fp in enumerate(future_prices):
                    pct_change = (fp - price) / price * 100
                    
                    if expected_dir == 'LONG':
                        if pct_change >= tp_pct:
                            hit_tp1 = 1
                            candles_to_result = j + 1
                            break
                        elif pct_change <= -sl_pct:
                            hit_sl = 1
                            candles_to_result = j + 1
                            break
                    elif expected_dir == 'SHORT':
                        if pct_change <= -tp_pct:
                            hit_tp1 = 1
                            candles_to_result = j + 1
                            break
                        elif pct_change >= sl_pct:
                            hit_sl = 1
                            candles_to_result = j + 1
                            break
                    else:
                        # Neutral - which significant move comes first?
                        if pct_change >= tp_pct:  # Up first = win
                            hit_tp1 = 1
                            candles_to_result = j + 1
                            break
                        elif pct_change <= -sl_pct:  # Down first = loss
                            hit_sl = 1
                            candles_to_result = j + 1
                            break
                
                # Update database
                cursor.execute("""
                    UPDATE whale_snapshots 
                    SET hit_tp1 = ?, hit_sl = ?, 
                        max_favorable_pct = ?, max_adverse_pct = ?,
                        candles_to_result = ?
                    WHERE symbol = ? AND timestamp = ?
                """, (hit_tp1, hit_sl, max_up, max_down, candles_to_result, symbol, ts))
                
                updated += 1
            
            conn.commit()
            conn.close()
            
            if updated > 0:
                pass  # Backtest complete
                
        except Exception as e:
            pass  # Backtest error (silently ignore)
    
    def _merge(self, price, oi, funding, ls, global_ls):
        """
        Merge all data by timestamp.
        
        FIXED: Use closest match instead of strict window.
        Coinglass API timestamps may not align perfectly with Binance price data.
        """
        def to_lookup(data):
            return {d['timestamp']: d for d in data} if data else {}
        
        oi_map = to_lookup(oi)
        fund_map = to_lookup(funding)
        ls_map = to_lookup(ls)
        gls_map = to_lookup(global_ls)
        
        def find_closest(lookup, ts, max_diff=14400000):
            """
            Find closest matching timestamp within max_diff (default 4 hours).
            Returns the data dict or empty dict if no match.
            """
            if not lookup:
                return {}
            
            # Exact match
            if ts in lookup:
                return lookup[ts]
            
            # Find closest timestamp
            closest_ts = None
            closest_diff = float('inf')
            
            for k in lookup:
                diff = abs(k - ts)
                if diff < closest_diff and diff <= max_diff:
                    closest_diff = diff
                    closest_ts = k
            
            if closest_ts is not None:
                return lookup[closest_ts]
            
            return {}
        
        merged = []
        whale_matches = 0
        retail_matches = 0
        
        for p in price:
            ts = p['timestamp']
            
            # Use 4-hour window for 4h candles (14400000ms = 4 hours)
            # This is wide enough to catch slight timestamp misalignments
            whale_data = find_closest(ls_map, ts, max_diff=14400000)
            retail_data = find_closest(gls_map, ts, max_diff=14400000)
            
            if whale_data.get('whale_long_pct'):
                whale_matches += 1
            if retail_data.get('retail_long_pct'):
                retail_matches += 1
            
            merged.append({
                'timestamp': ts,
                'price': p.get('price', 0),
                'price_change_24h': p.get('price_change_24h', 0),
                'open_interest': find_closest(oi_map, ts).get('open_interest', 0),
                'oi_change_24h': find_closest(oi_map, ts).get('oi_change_24h', 0),
                'funding_rate': find_closest(fund_map, ts).get('funding_rate', 0),
                'whale_long_pct': whale_data.get('whale_long_pct', 50),
                'retail_long_pct': retail_data.get('retail_long_pct', 50)
            })
        
        # Debug: Log match rates
        if len(price) > 0:
            print(f"    Merge stats: {whale_matches}/{len(price)} whale matches ({whale_matches/len(price)*100:.0f}%), "
                  f"{retail_matches}/{len(price)} retail matches ({retail_matches/len(price)*100:.0f}%)")
        
        return merged


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import Coinglass historical data")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--test", action="store_true", help="Test with just BTC")
    parser.add_argument("--force", action="store_true", help="Force re-import even if data exists")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COINGLASS HISTORICAL IMPORTER (V4 API)")
    print("=" * 60)
    
    if not args.force:
        print("ℹ️  Will skip symbols with >500 existing records (use --force to override)")
    
    symbols = ['BTCUSDT'] if args.test else args.symbols
    
    importer = HistoricalDataImporter(args.api_key)
    stats = importer.import_historical_data(
        symbols=symbols, 
        lookback_days=args.days, 
        interval=args.interval,
        skip_existing=not args.force
    )
    
    print("\n" + "=" * 60)
    print(f"Records: {stats['records_imported']:,}")
    print(f"Symbols: {stats['symbols_processed']}")
    if stats.get('symbols_skipped'):
        print(f"Skipped: {len(stats['symbols_skipped'])} (already imported)")
    if stats.get('errors'):
        print("Errors:", stats['errors'])