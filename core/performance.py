"""
Performance Optimization Module for InvestorIQ
==============================================
Drop this file into your core/ folder and import the functions.

Usage:
    from core.performance import (
        get_cached_klines, 
        fetch_multiple_concurrent,
        PriceCache
    )
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import hashlib

# ═══════════════════════════════════════════════════════════════════════════════
# TTL CACHE FOR API DATA
# ═══════════════════════════════════════════════════════════════════════════════

class TTLCache:
    """Thread-safe cache with TTL (Time To Live) expiration"""
    
    def __init__(self, default_ttl: int = 30):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttls: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value if exists and not expired"""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check TTL
            ttl = self._ttls.get(key, self.default_ttl)
            age = time.time() - self._timestamps.get(key, 0)
            
            if age > ttl:
                # Expired - remove and return None
                del self._cache[key]
                del self._timestamps[key]
                if key in self._ttls:
                    del self._ttls[key]
                self.misses += 1
                return None
            
            self.hits += 1
            # Return copy for DataFrames to prevent mutation
            value = self._cache[key]
            if isinstance(value, pd.DataFrame):
                return value.copy()
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value with optional custom TTL"""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
            if ttl is not None:
                self._ttls[key] = ttl
    
    def invalidate(self, key: str):
        """Remove a specific key"""
        with self._lock:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            self._ttls.pop(key, None)
    
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._ttls.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'size': len(self._cache)
        }


# Global cache instances
_klines_cache = TTLCache(default_ttl=30)  # 30 seconds for OHLCV data
_whale_cache = TTLCache(default_ttl=60)   # 60 seconds for whale data
_price_cache = TTLCache(default_ttl=10)   # 10 seconds for spot prices


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def get_cached_klines(symbol: str, interval: str, limit: int = 200, 
                      fetch_func: Callable = None, ttl: int = 30) -> Optional[pd.DataFrame]:
    """
    Cached version of klines fetching.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe (e.g., '15m', '4h')
        limit: Number of candles
        fetch_func: The actual fetch function to call on cache miss
        ttl: Cache TTL in seconds
    
    Returns:
        DataFrame or None
    """
    cache_key = f"klines_{symbol}_{interval}_{limit}"
    
    # Try cache first
    cached = _klines_cache.get(cache_key)
    if cached is not None:
        return cached
    
    # Cache miss - fetch fresh data
    if fetch_func is None:
        # Import here to avoid circular imports
        from core.data_fetcher import fetch_binance_klines
        fetch_func = fetch_binance_klines
    
    df = fetch_func(symbol, interval, limit)
    
    if df is not None and len(df) > 0:
        _klines_cache.set(cache_key, df, ttl)
    
    return df


def get_cached_price(symbol: str, fetch_func: Callable = None) -> float:
    """
    Get current price with caching (10 second TTL).
    
    Args:
        symbol: Trading pair
        fetch_func: Function to fetch price on cache miss
    
    Returns:
        Current price or 0 on failure
    """
    cache_key = f"price_{symbol}"
    
    cached = _price_cache.get(cache_key)
    if cached is not None:
        return cached
    
    if fetch_func is None:
        from core.data_fetcher import get_current_price
        fetch_func = get_current_price
    
    try:
        price = fetch_func(symbol)
        if price and price > 0:
            _price_cache.set(cache_key, price)
            return price
    except:
        pass
    
    return 0


def get_cached_whale_data(symbol: str, fetch_func: Callable = None) -> Optional[Dict]:
    """
    Get whale analysis with caching (60 second TTL).
    Whale data changes less frequently than prices.
    """
    cache_key = f"whale_{symbol}"
    
    cached = _whale_cache.get(cache_key)
    if cached is not None:
        return cached
    
    if fetch_func is None:
        from core.whale_institutional import get_whale_analysis
        fetch_func = get_whale_analysis
    
    try:
        data = fetch_func(symbol)
        if data:
            _whale_cache.set(cache_key, data)
            return data
    except:
        pass
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CONCURRENT FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_multiple_concurrent(
    items: List[Dict],
    fetch_func: Callable,
    max_workers: int = 5,
    progress_callback: Callable = None
) -> Dict[str, Any]:
    """
    Fetch data for multiple items concurrently.
    
    Args:
        items: List of dicts with at least 'symbol' key
        fetch_func: Function that takes an item and returns (symbol, result, error)
        max_workers: Maximum concurrent threads
        progress_callback: Optional callback(current, total, symbol) for progress updates
    
    Returns:
        Dict mapping symbol to {'result': data, 'error': error_msg}
    """
    results = {}
    total = len(items)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_func, item): item for item in items}
        
        # Collect results as they complete
        for future in as_completed(futures):
            item = futures[future]
            symbol = item.get('symbol', 'UNKNOWN')
            
            try:
                symbol, result, error = future.result()
                results[symbol] = {'result': result, 'error': error}
            except Exception as e:
                results[symbol] = {'result': None, 'error': str(e)}
            
            completed += 1
            if progress_callback:
                progress_callback(completed, total, symbol)
    
    return results


def fetch_watchlist_data_concurrent(
    watchlist: List[Dict],
    analyze_func: Callable = None,
    max_workers: int = 5,
    use_cache: bool = True
) -> Dict[str, Dict]:
    """
    Fetch and analyze all watchlist items concurrently.
    
    Args:
        watchlist: List of watchlist items
        analyze_func: Optional analysis function
        max_workers: Concurrent threads
        use_cache: Whether to use cached klines
    
    Returns:
        Dict mapping symbol to analysis results
    """
    from core.data_fetcher import fetch_binance_klines
    
    def fetch_single(item):
        symbol = item.get('symbol', '')
        timeframe = item.get('timeframe', '15m')
        mode_name = item.get('mode_name', 'Day Trade')
        
        try:
            # Fetch data (with or without cache)
            if use_cache:
                df = get_cached_klines(symbol, timeframe, 200)
            else:
                df = fetch_binance_klines(symbol, timeframe, 200)
            
            if df is None or len(df) < 50:
                return (symbol, None, "Not enough data")
            
            # If analysis function provided, run it
            if analyze_func:
                market_type = "🪙 Crypto" if symbol.endswith('USDT') else "📈 Stock"
                mode_key = mode_name.lower().replace(' ', '')
                
                analysis = analyze_func(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    market_type=market_type,
                    trade_mode=mode_key,
                    fetch_whale_api=("Crypto" in market_type)
                )
                return (symbol, {'df': df, 'analysis': analysis}, None)
            else:
                return (symbol, {'df': df, 'analysis': None}, None)
                
        except Exception as e:
            return (symbol, None, str(e))
    
    return fetch_multiple_concurrent(
        items=watchlist,
        fetch_func=fetch_single,
        max_workers=max_workers
    )


def fetch_prices_concurrent(symbols: List[str], max_workers: int = 10) -> Dict[str, float]:
    """
    Fetch current prices for multiple symbols concurrently.
    
    Args:
        symbols: List of trading pairs
        max_workers: Concurrent threads (prices are fast, can use more)
    
    Returns:
        Dict mapping symbol to current price
    """
    def fetch_price(symbol_item):
        symbol = symbol_item if isinstance(symbol_item, str) else symbol_item.get('symbol', '')
        price = get_cached_price(symbol)
        return (symbol, price, None if price > 0 else "Failed to fetch")
    
    items = [{'symbol': s} if isinstance(s, str) else s for s in symbols]
    results = fetch_multiple_concurrent(
        items=items,
        fetch_func=fetch_price,
        max_workers=max_workers
    )
    
    return {symbol: data['result'] for symbol, data in results.items() if data['result']}


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class BatchAnalyzer:
    """
    Helper class for batch analysis with progress tracking.
    
    Usage:
        analyzer = BatchAnalyzer(watchlist, analyze_func)
        for symbol, result in analyzer.run_with_progress():
            # Process each result as it completes
            display_result(symbol, result)
    """
    
    def __init__(self, items: List[Dict], analyze_func: Callable, max_workers: int = 5):
        self.items = items
        self.analyze_func = analyze_func
        self.max_workers = max_workers
        self.results = {}
        self.errors = {}
        self.completed = 0
        self.total = len(items)
    
    def run_with_progress(self):
        """Generator that yields results as they complete"""
        results = fetch_watchlist_data_concurrent(
            watchlist=self.items,
            analyze_func=self.analyze_func,
            max_workers=self.max_workers,
            use_cache=True
        )
        
        for symbol, data in results.items():
            self.completed += 1
            if data.get('error'):
                self.errors[symbol] = data['error']
                yield (symbol, None, data['error'])
            else:
                result = data.get('result', {})
                self.results[symbol] = result
                yield (symbol, result, None)
    
    def get_progress(self) -> Dict:
        """Get current progress"""
        return {
            'completed': self.completed,
            'total': self.total,
            'percentage': (self.completed / self.total * 100) if self.total > 0 else 0,
            'errors': len(self.errors)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SMART REFRESH LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class SmartRefreshManager:
    """
    Manages refresh intervals based on item characteristics.
    
    - More volatile items (scalp timeframes) refresh more often
    - Stable items (investment timeframes) refresh less often
    - Items that haven't changed much skip refresh
    """
    
    # Refresh intervals by timeframe (seconds)
    REFRESH_INTERVALS = {
        '1m': 15,
        '5m': 30,
        '15m': 60,
        '30m': 120,
        '1h': 180,
        '4h': 300,
        '1d': 600,
        '1w': 1200,
    }
    
    def __init__(self):
        self._last_refresh: Dict[str, float] = {}
        self._last_values: Dict[str, Dict] = {}
    
    def should_refresh(self, symbol: str, timeframe: str) -> bool:
        """Check if item should be refreshed based on timeframe and last refresh"""
        key = f"{symbol}_{timeframe}"
        interval = self.REFRESH_INTERVALS.get(timeframe, 60)
        
        last = self._last_refresh.get(key, 0)
        return (time.time() - last) >= interval
    
    def mark_refreshed(self, symbol: str, timeframe: str, values: Dict = None):
        """Mark item as refreshed"""
        key = f"{symbol}_{timeframe}"
        self._last_refresh[key] = time.time()
        if values:
            self._last_values[key] = values
    
    def get_stale_items(self, items: List[Dict]) -> List[Dict]:
        """Get list of items that need refreshing"""
        return [
            item for item in items
            if self.should_refresh(item.get('symbol', ''), item.get('timeframe', '15m'))
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_cache_stats() -> Dict:
    """Get statistics for all caches"""
    return {
        'klines': _klines_cache.stats(),
        'whale': _whale_cache.stats(),
        'price': _price_cache.stats()
    }


def clear_all_caches():
    """Clear all cached data"""
    _klines_cache.clear()
    _whale_cache.clear()
    _price_cache.clear()


def invalidate_symbol(symbol: str):
    """Clear all cached data for a specific symbol"""
    # Find and invalidate all keys containing this symbol
    for cache in [_klines_cache, _whale_cache, _price_cache]:
        keys_to_remove = [k for k in cache._cache.keys() if symbol in k]
        for key in keys_to_remove:
            cache.invalidate(key)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def create_progress_callback(progress_bar, status_text=None):
    """
    Create a progress callback for Streamlit.
    
    Usage:
        progress_bar = st.progress(0)
        status = st.empty()
        callback = create_progress_callback(progress_bar, status)
        
        results = fetch_multiple_concurrent(items, fetch_func, progress_callback=callback)
    """
    def callback(current, total, symbol):
        progress = current / total
        progress_bar.progress(progress, text=f"Processing {symbol} ({current}/{total})")
        if status_text:
            status_text.text(f"Analyzing {symbol}...")
    
    return callback


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Demo usage
    print("Performance Module Demo")
    print("=" * 50)
    
    # Test cache
    cache = TTLCache(default_ttl=5)
    cache.set('test_key', 'test_value')
    print(f"Cache get (should hit): {cache.get('test_key')}")
    print(f"Cache stats: {cache.stats()}")
    
    # Test concurrent fetch (mock)
    test_items = [
        {'symbol': 'BTCUSDT', 'timeframe': '15m'},
        {'symbol': 'ETHUSDT', 'timeframe': '15m'},
        {'symbol': 'SOLUSDT', 'timeframe': '15m'},
    ]
    
    def mock_fetch(item):
        import random
        time.sleep(0.5)  # Simulate API call
        return (item['symbol'], {'price': random.uniform(100, 1000)}, None)
    
    print("\nFetching 3 items concurrently...")
    start = time.time()
    results = fetch_multiple_concurrent(test_items, mock_fetch, max_workers=3)
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f}s (sequential would be ~1.5s)")
    print(f"Results: {results}")
