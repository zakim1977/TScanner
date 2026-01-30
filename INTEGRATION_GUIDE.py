"""
INTEGRATION GUIDE: How to Add Performance Optimizations to app.py
=================================================================

This file shows the exact changes needed to integrate the performance module.
Copy these code blocks into the appropriate places in your app.py.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: ADD IMPORT AT TOP OF app.py (around line 30-50)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add this import block near your other imports:

from core.performance import (
    get_cached_klines,
    get_cached_price,
    get_cached_whale_data,
    fetch_watchlist_data_concurrent,
    fetch_prices_concurrent,
    SmartRefreshManager,
    get_cache_stats,
    clear_all_caches,
    create_progress_callback
)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: INITIALIZE SMART REFRESH MANAGER (around line 150, after session_state init)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Add this after your session_state initialization:

# Initialize smart refresh manager
if 'refresh_manager' not in st.session_state:
    st.session_state.refresh_manager = SmartRefreshManager()
"""


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: REPLACE WATCHLIST LOOP (around line 9118-9250)
# ═══════════════════════════════════════════════════════════════════════════════

# BEFORE (slow, sequential):
"""
for idx, item in sorted_watchlist:
    symbol = item.get('symbol', 'UNKNOWN')
    ...
    try:
        df_watch = fetch_binance_klines(symbol, timeframe, 200)
        ...
        analysis = analyze_symbol_full(...)
"""

# AFTER (fast, concurrent):
OPTIMIZED_WATCHLIST_CODE = '''
    if not st.session_state.watchlist:
        # Show empty state...
        pass
    else:
        # ═══════════════════════════════════════════════════════════════════
        # OPTIMIZED: Concurrent data fetching
        # ═══════════════════════════════════════════════════════════════════
        
        # Show progress while fetching
        progress_container = st.empty()
        status_container = st.empty()
        
        with progress_container:
            progress_bar = st.progress(0, text="Fetching watchlist data...")
        
        # Fetch ALL data concurrently (major speedup!)
        all_data = fetch_watchlist_data_concurrent(
            watchlist=st.session_state.watchlist,
            analyze_func=analyze_symbol_full,
            max_workers=5,
            use_cache=True
        )
        
        # Clear progress indicators
        progress_container.empty()
        status_container.empty()
        
        # Track items to remove
        items_to_remove = []
        
        # Sort watchlist by added_at (newest first)
        sorted_watchlist = sorted(
            enumerate(st.session_state.watchlist), 
            key=lambda x: x[1].get('added_at', ''), 
            reverse=True
        )
        
        for idx, item in sorted_watchlist:
            symbol = item.get('symbol', 'UNKNOWN')
            direction = item.get('direction', 'LONG')
            entry_price = item.get('price_at_add', 0)
            entry_score = item.get('score', item.get('score_at_add', 0))
            entry_whale = item.get('whale_pct_at_add', 50)
            entry_explosion = item.get('explosion_score', item.get('explosion_at_add', 0))
            entry_ml_direction = item.get('ml_direction_at_add', 'UNKNOWN')
            entry_ml_conf = item.get('ml_conf_at_add', 0)
            entry_vwap_type = item.get('vwap_bounce_type', '')
            added_at = item.get('added_at', '')
            timeframe = item.get('timeframe', '15m')
            mode_name = item.get('mode_name', 'Day Trade')
            
            # Get pre-fetched data (already fetched concurrently!)
            fetched = all_data.get(symbol, {})
            error = fetched.get('error')
            result = fetched.get('result', {})
            
            if error or not result:
                # Fetch failed - use entry values as fallback
                current_price = entry_price
                price_change = 0
                data_available = False
                current_whale = entry_whale
                current_explosion = entry_explosion
                current_score = entry_score
                current_ml = 'UNKNOWN'
                current_ml_conf = 0
                analysis = None
            else:
                # Use pre-fetched analysis
                analysis = result.get('analysis', {})
                df_watch = result.get('df')
                
                if analysis and 'error' not in analysis:
                    data_available = True
                    current_price = analysis['current_price']
                    price_change = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    current_whale = analysis.get('whale_pct', 50)
                    current_explosion = analysis.get('explosion', {}).get('score', 0) if analysis.get('explosion') else 0
                    
                    # ML prediction
                    ml_pred = analysis.get('ml_prediction')
                    if ml_pred is not None:
                        current_ml = ml_pred.direction
                        current_ml_conf = ml_pred.confidence
                    else:
                        current_ml = 'UNKNOWN'
                        current_ml_conf = 0
                    
                    # Predictive score
                    pred_result = analysis.get('predictive_result')
                    confidence_scores = analysis.get('confidence_scores', {})
                    combined_score = confidence_scores.get('combined_score', 50)
                    current_score = pred_result.final_score if pred_result else combined_score
                else:
                    # Analysis failed
                    data_available = False
                    current_price = entry_price
                    price_change = 0
                    current_whale = entry_whale
                    current_explosion = entry_explosion
                    current_score = entry_score
                    current_ml = 'UNKNOWN'
                    current_ml_conf = 0
            
            # ... rest of the watchlist item display code stays the same ...
'''


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: OPTIMIZE TRADE MONITOR PRICE FETCHING (around line 9742-9780)
# ═══════════════════════════════════════════════════════════════════════════════

# BEFORE (fetches prices one by one):
"""
def get_cached_prices(symbols: list, force_refresh: bool = False) -> dict:
    ...
    for symbol in symbols:
        try:
            price = get_current_price(symbol)
            if price > 0:
                new_prices[symbol] = price
        except:
            pass
"""

# AFTER (concurrent price fetching):
OPTIMIZED_TRADE_MONITOR_PRICES = '''
def get_cached_prices_optimized(symbols: list, force_refresh: bool = False) -> dict:
    """Get prices with caching. Fetches ALL prices concurrently when cache is stale."""
    now = datetime.now()
    cache_age = (now - st.session_state.price_cache_time).total_seconds() if st.session_state.price_cache_time else float('inf')
    
    # Check if we need to refresh
    need_refresh = force_refresh or cache_age > PRICE_CACHE_TTL or not st.session_state.price_cache
    
    if need_refresh:
        # OPTIMIZED: Fetch all prices concurrently!
        new_prices = fetch_prices_concurrent(symbols, max_workers=10)
        
        st.session_state.price_cache = new_prices
        st.session_state.price_cache_time = now
        st.session_state.force_price_refresh = False
    
    return st.session_state.price_cache
'''


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: ADD CACHE STATS DISPLAY (optional, for debugging)
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_STATS_DISPLAY = '''
# Add this in sidebar or settings area to monitor cache performance:

if st.sidebar.checkbox("Show Cache Stats"):
    stats = get_cache_stats()
    st.sidebar.json(stats)
    
    if st.sidebar.button("Clear All Caches"):
        clear_all_caches()
        st.sidebar.success("Caches cleared!")
'''


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: OPTIMIZE SCANNER VIEW (if needed)
# ═══════════════════════════════════════════════════════════════════════════════

OPTIMIZED_SCANNER_CODE = '''
# In scanner view, replace sequential analysis with concurrent:

def scan_symbols_optimized(symbols_list, timeframe, trade_mode):
    """Scan multiple symbols concurrently"""
    
    # Prepare items
    items = [
        {
            'symbol': symbol,
            'timeframe': timeframe,
            'mode_name': trade_mode
        }
        for symbol in symbols_list
    ]
    
    # Show progress
    progress_bar = st.progress(0, text="Scanning market...")
    
    # Fetch and analyze ALL concurrently
    results = fetch_watchlist_data_concurrent(
        watchlist=items,
        analyze_func=analyze_symbol_full,
        max_workers=10,  # Higher for scanner since we have many symbols
        use_cache=True
    )
    
    progress_bar.empty()
    
    # Process results
    signals = []
    for symbol, data in results.items():
        if data.get('error'):
            continue
        
        result = data.get('result', {})
        analysis = result.get('analysis', {})
        
        if analysis and 'error' not in analysis:
            # Extract signal...
            signal = analysis.get('signal')
            if signal:
                signals.append({
                    'symbol': symbol,
                    'signal': signal,
                    'analysis': analysis
                })
    
    return signals
'''


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK INTEGRATION SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

"""
QUICK START - 3 STEPS TO FASTER APP:

1. Copy core/performance.py to your core/ folder

2. Add this import at top of app.py:
   from core.performance import (
       fetch_watchlist_data_concurrent,
       fetch_prices_concurrent,
       get_cached_klines
   )

3. In Watchlist view (around line 9118), replace:
   
   for idx, item in sorted_watchlist:
       df_watch = fetch_binance_klines(symbol, timeframe, 200)
   
   With:
   
   # Fetch ALL data at once (concurrently)
   all_data = fetch_watchlist_data_concurrent(st.session_state.watchlist, analyze_symbol_full)
   
   for idx, item in sorted_watchlist:
       data = all_data.get(item['symbol'], {})
       analysis = data.get('result', {}).get('analysis')

That's it! You should see 3-5x speedup immediately.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE: EVEN SIMPLER - JUST ADD CACHING
# ═══════════════════════════════════════════════════════════════════════════════

"""
If you want the absolute minimum change, just replace fetch_binance_klines calls
with get_cached_klines in your existing code:

BEFORE:
    df = fetch_binance_klines(symbol, timeframe, 200)

AFTER:
    df = get_cached_klines(symbol, timeframe, 200, fetch_func=fetch_binance_klines)

This alone will give you ~2x speedup from avoiding redundant API calls.
"""
