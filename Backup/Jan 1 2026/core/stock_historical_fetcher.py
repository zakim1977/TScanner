"""
Stock Historical Data Fetcher
=============================
Fetches historical institutional data from Quiver API for stocks/ETFs.
This is the stock equivalent of historical_data_fetcher.py for crypto.

Quiver API provides:
- Congress trading history (transactions by members of Congress)
- Insider trading history (CEO, CFO, director buys/sells)
- Short interest history

Strategy:
---------
1. Fetch historical transactions (Congress trades, insider trades)
2. Calculate rolling institutional scores over time
3. Store as snapshots for historical validation
4. Match against current conditions using KNN
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# SSL BYPASS - Required for corporate/proxy environments with self-signed certs
# ═══════════════════════════════════════════════════════════════════════════════
import ssl
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Create unverified SSL context
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ═══════════════════════════════════════════════════════════════════════════════

# Import stores
def _get_stock_store():
    from core.stock_data_store import get_stock_store, StockSnapshot
    return get_stock_store()


QUIVER_BASE_URL = "https://api.quiverquant.com/beta"


@dataclass
class QuiverHistoricalConfig:
    """Configuration for Quiver historical data import"""
    api_key: str
    symbols: List[str]  # ['AAPL', 'MSFT', ...]
    lookback_days: int = 365
    

class QuiverHistoricalFetcher:
    """
    Fetches historical institutional data from Quiver API.
    
    Quiver API Endpoints:
    - /historical/congresstrading/{symbol} - Congress trades
    - /historical/insiders/{symbol} - Insider trades
    - /historical/shortinterest/{symbol} - Short interest over time
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._rate_limit_delay = 0.5  # Quiver has strict rate limits
        
    def _request(self, endpoint: str) -> Optional[List]:
        """Make API request with rate limiting"""
        try:
            url = f"{QUIVER_BASE_URL}{endpoint}"
            response = requests.get(url, headers=self.headers, timeout=30, verify=False)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Rate limited - waiting 60s")
                time.sleep(60)
                return self._request(endpoint)
            elif response.status_code == 404:
                return []  # No data for this symbol
            else:
                print(f"API Error {response.status_code}: {response.text[:100]}")
                return None
                
        except Exception as e:
            print(f"Request error: {e}")
            return None
        finally:
            time.sleep(self._rate_limit_delay)
    
    def fetch_congress_history(
        self, 
        symbol: str, 
        lookback_days: int = 365
    ) -> List[Dict]:
        """
        Fetch Congress trading history for a symbol.
        
        Returns list of transactions:
        {
            'date': str,
            'representative': str,
            'transaction': str (Purchase/Sale),
            'amount': str (range like $1,001-$15,000)
        }
        """
        endpoint = f"/historical/congresstrading/{symbol}"
        data = self._request(endpoint)
        
        if not data:
            return []
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        processed = []
        for trade in data:
            try:
                trade_date = datetime.strptime(trade.get('TransactionDate', ''), '%Y-%m-%d')
                if trade_date >= cutoff:
                    processed.append({
                        'date': trade.get('TransactionDate'),
                        'representative': trade.get('Representative', ''),
                        'transaction': trade.get('Transaction', ''),  # Purchase or Sale
                        'amount': trade.get('Amount', ''),
                        'party': trade.get('Party', ''),
                    })
            except:
                continue
        
        return processed
    
    def fetch_insider_history(
        self,
        symbol: str,
        lookback_days: int = 365
    ) -> List[Dict]:
        """
        Fetch insider trading history for a symbol.
        
        Returns list of transactions:
        {
            'date': str,
            'name': str,
            'title': str (CEO, CFO, Director, etc.),
            'transaction_type': str (Buy/Sell),
            'shares': int,
            'price': float
        }
        """
        endpoint = f"/historical/insiders/{symbol}"
        data = self._request(endpoint)
        
        if not data:
            return []
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        processed = []
        for trade in data:
            try:
                trade_date = datetime.strptime(trade.get('Date', ''), '%Y-%m-%d')
                if trade_date >= cutoff:
                    processed.append({
                        'date': trade.get('Date'),
                        'name': trade.get('Name', ''),
                        'title': trade.get('Title', ''),
                        'transaction_type': trade.get('TransactionType', ''),
                        'shares': trade.get('Shares', 0),
                        'price': trade.get('Price', 0),
                        'value': trade.get('Value', 0),
                    })
            except:
                continue
        
        return processed
    
    def fetch_short_interest_history(
        self,
        symbol: str,
        lookback_days: int = 365
    ) -> List[Dict]:
        """
        Fetch short interest history for a symbol.
        
        Returns list of snapshots:
        {
            'date': str,
            'short_interest': int,
            'short_percent': float,
            'days_to_cover': float
        }
        """
        endpoint = f"/historical/shortinterest/{symbol}"
        data = self._request(endpoint)
        
        if not data:
            return []
        
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        processed = []
        for record in data:
            try:
                record_date = datetime.strptime(record.get('Date', ''), '%Y-%m-%d')
                if record_date >= cutoff:
                    processed.append({
                        'date': record.get('Date'),
                        'short_interest': record.get('ShortInterest', 0),
                        'short_percent': record.get('ShortInterestPercent', 0),
                        'days_to_cover': record.get('DaysToCover', 0),
                    })
            except:
                continue
        
        return processed
    
    def fetch_price_history(
        self,
        symbol: str,
        lookback_days: int = 365
    ) -> List[Dict]:
        """
        Fetch price history from Yahoo Finance (free).
        
        Returns list of OHLCV data.
        """
        # Strip $ prefix if present (yfinance needs plain symbol)
        clean_symbol = symbol.lstrip('$').upper()
        
        # Try yfinance first
        try:
            import yfinance as yf
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            df = yf.download(
                clean_symbol, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )
            
            if not df.empty:
                processed = []
                for idx, row in df.iterrows():
                    try:
                        processed.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume']) if row['Volume'] else 0,
                        })
                    except:
                        continue
                
                if processed:
                    print(f"    ✓ Got {len(processed)} price records for {clean_symbol}")
                    return processed
                    
        except Exception as e:
            print(f"    ⚠️ yfinance error: {e}")
        
        # Fallback: Direct Yahoo Finance API (no yfinance dependency)
        try:
            print(f"    Trying direct Yahoo API for {clean_symbol}...")
            
            period1 = int((datetime.now() - timedelta(days=lookback_days)).timestamp())
            period2 = int(datetime.now().timestamp())
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_symbol}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': '1d',
                'events': 'history'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('chart', {}).get('result', [])
                
                if result:
                    timestamps = result[0].get('timestamp', [])
                    quotes = result[0].get('indicators', {}).get('quote', [{}])[0]
                    
                    processed = []
                    for i, ts in enumerate(timestamps):
                        try:
                            processed.append({
                                'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
                                'open': float(quotes.get('open', [])[i] or 0),
                                'high': float(quotes.get('high', [])[i] or 0),
                                'low': float(quotes.get('low', [])[i] or 0),
                                'close': float(quotes.get('close', [])[i] or 0),
                                'volume': int(quotes.get('volume', [])[i] or 0),
                            })
                        except:
                            continue
                    
                    if processed:
                        print(f"    ✓ Got {len(processed)} price records via direct API")
                        return processed
                        
        except Exception as e:
            print(f"    ⚠️ Direct API error: {e}")
        
        print(f"    ⚠️ No price data for {clean_symbol}")
        return []


class StockHistoricalDataImporter:
    """
    Orchestrates historical stock data import and storage.
    
    Usage:
    ------
    importer = StockHistoricalDataImporter(api_key="your_quiver_key")
    stats = importer.import_historical_data(
        symbols=['AAPL', 'MSFT', 'NVDA'],
        lookback_days=180
    )
    """
    
    # Popular stocks and ETFs with Congress/Insider trading activity
    DEFAULT_SYMBOLS = [
        # === BIG TECH ===
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
        'AVGO', 'ORCL', 'CRM', 'ADBE', 'CSCO', 'ACN', 'IBM', 'INTC', 'AMD',
        
        # === FINANCE ===
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP',
        'V', 'MA', 'PYPL', 'COF', 'USB',
        
        # === HEALTHCARE ===
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
        'BMY', 'AMGN', 'GILD', 'CVS', 'ISRG', 'VRTX', 'REGN', 'MRNA',
        
        # === ENERGY ===
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'VLO', 'PSX',
        
        # === CONSUMER ===
        'WMT', 'HD', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX',
        'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'EL',
        
        # === INDUSTRIAL ===
        'CAT', 'DE', 'BA', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'NOC',
        'MMM', 'UNP', 'FDX',
        
        # === TELECOM/MEDIA ===
        'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
        
        # === SEMIS ===
        'TSM', 'ASML', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC', 'ADI',
        
        # === POPULAR ETFs ===
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'ARKK', 'XLF',
        'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'GLD', 'SLV', 'TLT', 'HYG',
        
        # === HIGH CONGRESS ACTIVITY ===
        'PLTR', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'SQ',
        'SHOP', 'SNOW', 'DDOG', 'NET', 'ZS', 'CRWD', 'PANW', 'OKTA',
    ]
    
    def __init__(self, api_key: str):
        """
        Initialize importer with Quiver API credentials.
        
        Args:
            api_key: Your Quiver Quant API key
        """
        self.api_key = api_key
        self.fetcher = QuiverHistoricalFetcher(api_key)
        self.store = _get_stock_store()
        
        # Progress tracking
        self._progress_callback = None
        self._cancelled = False
    
    def set_progress_callback(self, callback):
        """Set callback for progress updates: callback(symbol, progress_pct, message)"""
        self._progress_callback = callback
    
    def cancel(self):
        """Cancel ongoing import"""
        self._cancelled = True
    
    def _report_progress(self, symbol: str, pct: float, message: str):
        """Report progress if callback is set"""
        if self._progress_callback:
            self._progress_callback(symbol, pct, message)
    
    def import_historical_data(
        self,
        symbols: List[str] = None,
        lookback_days: int = 180,
        skip_existing: bool = True,
    ) -> Dict:
        """
        Import historical institutional data for multiple stocks.
        
        Args:
            symbols: List of stock symbols (default: popular stocks)
            lookback_days: How far back to fetch (default: 6 months)
            skip_existing: Skip symbols that already have data
            
        Returns:
            Dict with import statistics
        """
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS
        
        self._cancelled = False
        
        # Check what's already imported
        already_imported = set()
        if skip_existing:
            already_imported = self._get_imported_symbols(min_records=10)
            if already_imported:
                print(f"⏭️  Found {len(already_imported)} symbols with data (will skip)")
        
        stats = {
            'symbols_processed': 0,
            'symbols_failed': [],
            'symbols_skipped': [],
            'records_imported': 0,
            'import_time': datetime.now().isoformat(),
            'errors': []
        }
        
        total_symbols = len(symbols)
        
        for idx, symbol in enumerate(symbols):
            if self._cancelled:
                stats['cancelled'] = True
                break
            
            progress = (idx / total_symbols) * 100
            
            # Skip if already imported
            if skip_existing and symbol in already_imported:
                stats['symbols_skipped'].append(symbol)
                self._report_progress(symbol, progress, f"⏭️ {symbol}: Already imported (skipping)")
                continue
            
            self._report_progress(symbol, progress, f"Processing {symbol}...")
            
            try:
                records = self._import_symbol(symbol, lookback_days)
                stats['records_imported'] += records
                stats['symbols_processed'] += 1
                
                self._report_progress(
                    symbol, progress,
                    f"✓ {symbol}: {records} snapshots"
                )
                
            except Exception as e:
                stats['symbols_failed'].append(symbol)
                stats['errors'].append(f"{symbol}: {str(e)}")
                
                self._report_progress(
                    symbol, progress,
                    f"✗ {symbol}: {str(e)}"
                )
        
        self._report_progress("DONE", 100, f"Import complete: {stats['records_imported']} snapshots")
        
        # AUTO-BACKTEST: Calculate outcomes for imported data
        if stats['records_imported'] > 0:
            try:
                self._report_progress("BACKTEST", 100, "Calculating outcomes...")
                updated = self.store.backtest_pending_outcomes(max_symbols=len(stats['symbols_processed']))
                if updated > 0:
                    # print(f"    ✓ Backtest: {updated} outcomes calculated")
            except Exception as e:
                # print(f"    ⚠️ Backtest error: {e}")
        
        return stats
    
    def _get_imported_symbols(self, min_records: int = 10) -> set:
        """Get symbols that already have data"""
        try:
            all_snapshots = self.store.get_all_symbols()
            # Filter to symbols with enough records
            imported = set()
            for symbol in all_snapshots:
                snapshots = self.store.get_snapshots(symbol)
                if len(snapshots) >= min_records:
                    imported.add(symbol)
            return imported
        except:
            return set()
    
    def _import_symbol(
        self,
        symbol: str,
        lookback_days: int
    ) -> int:
        """
        Import all historical data for a single stock.
        Creates weekly snapshots with rolling institutional scores.
        """
        from core.stock_data_store import StockSnapshot
        
        # Fetch all data types
        print(f"    Fetching Congress trades...")
        congress_trades = self.fetcher.fetch_congress_history(symbol, lookback_days)
        print(f"    Fetching Insider trades...")
        insider_trades = self.fetcher.fetch_insider_history(symbol, lookback_days)
        print(f"    Fetching Short interest...")
        short_history = self.fetcher.fetch_short_interest_history(symbol, lookback_days)
        
        # Check if we have ANY Quiver data
        has_quiver_data = bool(congress_trades or insider_trades or short_history)
        
        if not has_quiver_data:
            print(f"    ⚠️ No Quiver data for {symbol}")
            return 0
        
        print(f"    Data: congress={len(congress_trades)}, insider={len(insider_trades)}, short={len(short_history)}")
        
        # Try to get price data (optional - yfinance may fail)
        print(f"    Fetching price data...")
        price_history = self.fetcher.fetch_price_history(symbol, lookback_days)
        
        # If no price data, create weekly snapshots based on Quiver data dates
        if not price_history:
            print(f"    ⚠️ No price data - using Quiver dates only")
            return self._import_quiver_only(symbol, congress_trades, insider_trades, short_history, lookback_days)
        
        # Create weekly snapshots with rolling scores
        snapshots_created = 0
        
        # Group by week
        weeks = self._group_by_week(price_history)
        
        for week_date, week_prices in weeks.items():
            # Calculate rolling institutional scores for this week
            scores = self._calculate_rolling_scores(
                week_date,
                congress_trades,
                insider_trades,
                short_history,
                lookback_window=30  # 30-day rolling window
            )
            
            # Get price data for this week
            last_price = week_prices[-1]
            first_price = week_prices[0]
            
            price_change = ((last_price['close'] - first_price['close']) / first_price['close'] * 100) if first_price['close'] > 0 else 0
            
            # Calculate position in 52-week range
            year_prices = [p['close'] for p in price_history if p['date'] <= week_date][-252:]  # ~1 year
            if year_prices:
                year_high = max(year_prices)
                year_low = min(year_prices)
                position_pct = ((last_price['close'] - year_low) / (year_high - year_low) * 100) if year_high > year_low else 50
            else:
                position_pct = 50
            
            # Create snapshot
            snapshot = StockSnapshot(
                symbol=symbol,
                timeframe='historical_1w',
                timestamp=week_date,
                
                # Quiver institutional data
                congress_score=scores['congress_score'],
                insider_score=scores['insider_score'],
                short_interest_pct=scores['short_interest_pct'],
                combined_score=scores['combined_score'],
                
                # Technical
                price=last_price['close'],
                price_change_24h=price_change,
                rsi=50,  # Would need to calculate
                position_pct=position_pct,
                ta_score=50,  # Placeholder
                
                # No signal for historical
                signal_direction='WAIT',
                signal_name='historical_import',
                confidence='MEDIUM',
            )
            
            self.store.store_snapshot(snapshot)
            snapshots_created += 1
        
        return snapshots_created
    
    def _import_quiver_only(
        self,
        symbol: str,
        congress_trades: List[Dict],
        insider_trades: List[Dict],
        short_history: List[Dict],
        lookback_days: int
    ) -> int:
        """
        Import Quiver data without price data (fallback when yfinance fails).
        Creates monthly snapshots based on available Quiver data dates.
        """
        from core.stock_data_store import StockSnapshot
        
        # Collect all dates from Quiver data
        all_dates = set()
        
        for t in congress_trades:
            try:
                all_dates.add(t['date'][:7])  # YYYY-MM
            except:
                pass
        
        for t in insider_trades:
            try:
                all_dates.add(t['date'][:7])  # YYYY-MM
            except:
                pass
        
        for s in short_history:
            try:
                all_dates.add(s['date'][:7])  # YYYY-MM
            except:
                pass
        
        if not all_dates:
            return 0
        
        # Create monthly snapshots
        snapshots_created = 0
        
        for month in sorted(all_dates):
            # Use last day of month as snapshot date
            try:
                year, mon = month.split('-')
                snapshot_date = f"{year}-{mon}-28"  # Approximate end of month
            except:
                continue
            
            # Calculate scores for this month
            scores = self._calculate_rolling_scores(
                snapshot_date,
                congress_trades,
                insider_trades,
                short_history,
                lookback_window=60  # 60-day rolling window
            )
            
            # Create snapshot with placeholder price data
            snapshot = StockSnapshot(
                symbol=symbol,
                timeframe='historical_1m',
                timestamp=snapshot_date,
                
                # Quiver institutional data
                congress_score=scores['congress_score'],
                insider_score=scores['insider_score'],
                short_interest_pct=scores['short_interest_pct'],
                combined_score=scores['combined_score'],
                
                # Placeholder technical data (no price available)
                price=0,
                price_change_24h=0,
                rsi=50,
                position_pct=50,
                ta_score=50,
                
                # No signal for historical
                signal_direction='WAIT',
                signal_name='quiver_only_import',
                confidence='LOW',
            )
            
            self.store.store_snapshot(snapshot)
            snapshots_created += 1
        
        print(f"    ✓ Created {snapshots_created} monthly snapshots (Quiver only)")
        return snapshots_created
    
    def _group_by_week(self, price_history: List[Dict]) -> Dict[str, List[Dict]]:
        """Group price data by week (ending Sunday)"""
        weeks = {}
        
        for price in price_history:
            try:
                date = datetime.strptime(price['date'], '%Y-%m-%d')
                # Get Sunday of that week
                week_end = date + timedelta(days=(6 - date.weekday()))
                week_key = week_end.strftime('%Y-%m-%d')
                
                if week_key not in weeks:
                    weeks[week_key] = []
                weeks[week_key].append(price)
            except:
                continue
        
        return weeks
    
    def _calculate_rolling_scores(
        self,
        as_of_date: str,
        congress_trades: List[Dict],
        insider_trades: List[Dict],
        short_history: List[Dict],
        lookback_window: int = 30
    ) -> Dict:
        """
        Calculate rolling institutional scores as of a specific date.
        
        This simulates what the scores would have been on that date,
        allowing for proper backtesting.
        """
        try:
            as_of = datetime.strptime(as_of_date, '%Y-%m-%d')
        except:
            as_of = datetime.now()
        
        window_start = as_of - timedelta(days=lookback_window)
        
        # Congress score (0-100)
        congress_in_window = [
            t for t in congress_trades
            if window_start <= datetime.strptime(t['date'], '%Y-%m-%d') <= as_of
        ]
        
        if congress_in_window:
            buys = sum(1 for t in congress_in_window if 'Purchase' in t.get('transaction', ''))
            total = len(congress_in_window)
            congress_score = (buys / total * 100) if total > 0 else 50
        else:
            congress_score = 50
        
        # Insider score (0-100)
        insider_in_window = [
            t for t in insider_trades
            if window_start <= datetime.strptime(t['date'], '%Y-%m-%d') <= as_of
        ]
        
        if insider_in_window:
            # Weight C-suite buys more heavily
            c_suite_titles = ['CEO', 'CFO', 'COO', 'President', 'Chairman']
            
            buy_value = 0
            sell_value = 0
            
            for trade in insider_in_window:
                value = abs(trade.get('value', 0))
                is_c_suite = any(title.lower() in trade.get('title', '').lower() for title in c_suite_titles)
                multiplier = 2.0 if is_c_suite else 1.0
                
                if 'Buy' in trade.get('transaction_type', '') or 'Purchase' in trade.get('transaction_type', ''):
                    buy_value += value * multiplier
                else:
                    sell_value += value * multiplier
            
            total_value = buy_value + sell_value
            insider_score = (buy_value / total_value * 100) if total_value > 0 else 50
        else:
            insider_score = 50
        
        # Short interest (get closest to as_of_date)
        short_interest_pct = 0
        if short_history:
            closest_short = None
            min_diff = float('inf')
            
            for record in short_history:
                try:
                    record_date = datetime.strptime(record['date'], '%Y-%m-%d')
                    diff = abs((as_of - record_date).days)
                    if diff < min_diff and record_date <= as_of:
                        min_diff = diff
                        closest_short = record
                except:
                    continue
            
            if closest_short:
                short_interest_pct = closest_short.get('short_percent', 0)
        
        # Combined score (weighted average)
        # Higher short interest = contrarian bullish (squeeze potential)
        squeeze_score = min(100, short_interest_pct * 5) if short_interest_pct > 10 else 50
        
        combined_score = (
            congress_score * 0.25 +
            insider_score * 0.40 +
            squeeze_score * 0.35
        )
        
        return {
            'congress_score': congress_score,
            'insider_score': insider_score,
            'short_interest_pct': short_interest_pct,
            'combined_score': combined_score,
        }


# ============================================================================
# STREAMLIT INTEGRATION
# ============================================================================

def create_stock_import_ui():
    """
    Create Streamlit UI for stock historical data import.
    """
    import streamlit as st
    
    st.subheader("📈 Stock Historical Data Import")
    
    st.markdown("""
    Import historical institutional data from Quiver for stock backtesting.
    
    **Data Sources:**
    - Congress Trading (Nancy Pelosi, etc.)
    - Insider Trading (CEO/CFO buys)
    - Short Interest
    """)
    
    # API Key input
    api_key = st.text_input(
        "Quiver API Key",
        type="password",
        help="Get your API key from quiverquant.com"
    )
    
    # Symbols selection
    default_symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA']
    
    symbols = st.multiselect(
        "Stocks to Import",
        options=StockHistoricalDataImporter.DEFAULT_SYMBOLS,
        default=default_symbols
    )
    
    # Custom symbols
    custom = st.text_input(
        "Additional Symbols (comma-separated)",
        placeholder="AMD, INTC, PLTR"
    )
    
    if custom:
        custom_symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
        symbols = list(set(symbols + custom_symbols))
    
    # Lookback period
    lookback_days = st.slider(
        "Lookback Period (days)",
        min_value=30,
        max_value=365,
        value=180,
        step=30
    )
    
    # Import button
    if st.button("🚀 Start Stock Import", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Quiver API key")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(symbol, pct, msg):
            progress_bar.progress(int(pct))
            status_text.text(msg)
        
        try:
            importer = StockHistoricalDataImporter(api_key)
            importer.set_progress_callback(update_progress)
            
            stats = importer.import_historical_data(
                symbols=symbols,
                lookback_days=lookback_days
            )
            
            st.success(f"""
            ✅ Stock Import Complete!
            - Snapshots imported: {stats['records_imported']}
            - Symbols processed: {stats['symbols_processed']}
            """)
            
            if stats.get('errors'):
                with st.expander("⚠️ Errors"):
                    for err in stats['errors']:
                        st.text(err)
                        
        except Exception as e:
            st.error(f"Import failed: {str(e)}")
    
    # Show current database stats
    st.divider()
    st.subheader("📊 Current Stock Database Stats")
    
    try:
        store = _get_stock_store()
        stats = store.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Snapshots", f"{stats.get('total_snapshots', 0):,}")
            st.metric("With Outcomes", f"{stats.get('with_outcomes', 0):,}")
        with col2:
            st.metric("Symbols Tracked", stats.get('total_symbols', 0))
            
        if stats.get('symbols'):
            st.caption(f"Symbols: {', '.join(stats['symbols'][:10])}...")
            
    except Exception as e:
        st.warning(f"Could not load stats: {e}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import historical stock institutional data")
    parser.add_argument("--api-key", required=True, help="Quiver API key")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to import")
    parser.add_argument("--days", type=int, default=180, help="Lookback days")
    
    args = parser.parse_args()
    
    importer = StockHistoricalDataImporter(args.api_key)
    
    def print_progress(symbol, pct, msg):
        print(f"[{pct:.0f}%] {msg}")
    
    importer.set_progress_callback(print_progress)
    
    stats = importer.import_historical_data(
        symbols=args.symbols,
        lookback_days=args.days
    )
    
    print("\n" + "="*50)
    print("STOCK IMPORT COMPLETE")
    print("="*50)
    print(f"Snapshots: {stats['records_imported']}")
    print(f"Symbols: {stats['symbols_processed']}")
