"""
Liquidation Level Collector
============================
Collects REAL 25x/50x/100x liquidation levels for ML training.

Every scan automatically saves data → builds training dataset over time!

After 30-90 days of collection, we can train on REAL liquidation data
instead of proxy swing levels.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

# Database path
DB_PATH = "data/liquidation_levels.db"


def init_database():
    """Initialize the liquidation levels database."""
    os.makedirs("data", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Main table for liquidation level snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS liq_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            
            -- Liquidation levels (where positions get liquidated)
            liq_100x_above REAL,  -- 100x shorts liquidated here (above price)
            liq_100x_below REAL,  -- 100x longs liquidated here (below price)
            liq_50x_above REAL,
            liq_50x_below REAL,
            liq_25x_above REAL,
            liq_25x_below REAL,
            
            -- Liquidity amounts at each level (if available)
            liq_100x_above_amt REAL,
            liq_100x_below_amt REAL,
            liq_50x_above_amt REAL,
            liq_50x_below_amt REAL,
            liq_25x_above_amt REAL,
            liq_25x_below_amt REAL,
            
            -- Context data
            whale_pct REAL,
            whale_delta REAL,
            atr REAL,
            
            -- Outcome tracking (filled in later by backtest)
            swept_above INTEGER DEFAULT NULL,      -- Did price sweep above level?
            swept_below INTEGER DEFAULT NULL,      -- Did price sweep below level?
            swept_level TEXT DEFAULT NULL,         -- Which level was swept first? (100x_above, 50x_below, etc)
            bounce_after_sweep INTEGER DEFAULT NULL, -- Did price bounce after sweep?
            candles_to_sweep INTEGER DEFAULT NULL,
            
            -- Metadata
            data_source TEXT DEFAULT 'scanner',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(timestamp, symbol)
        )
    """)
    
    # Index for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_liq_symbol_ts ON liq_snapshots(symbol, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_liq_timestamp ON liq_snapshots(timestamp)")
    
    conn.commit()
    conn.close()
    
    print(f"[LIQ_COLLECTOR] Database initialized: {DB_PATH}")


def save_liquidation_snapshot(
    symbol: str,
    price: float,
    levels: Dict,
    whale_pct: float = None,
    whale_delta: float = None,
    atr: float = None,
    data_source: str = 'scanner',
    timestamp_override: int = None
) -> bool:
    """
    Save a liquidation level snapshot to the database.
    
    Called automatically when scanner/single analysis fetches liquidation data.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        price: Current price
        levels: Dict with liquidation levels:
            {
                '100x_above': 105200,
                '100x_below': 99800,
                '50x_above': 106500,
                ...
            }
        whale_pct: Current whale long %
        whale_delta: Whale positioning change
        atr: Current ATR
        data_source: Where data came from ('scanner', 'single_analysis', 'monitor')
        timestamp_override: Optional specific timestamp (for historical data)
    
    Returns:
        True if saved successfully
    """
    try:
        init_database()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = timestamp_override if timestamp_override else int(time.time() * 1000)
        
        cursor.execute("""
            INSERT OR REPLACE INTO liq_snapshots (
                timestamp, symbol, price,
                liq_100x_above, liq_100x_below,
                liq_50x_above, liq_50x_below,
                liq_25x_above, liq_25x_below,
                liq_100x_above_amt, liq_100x_below_amt,
                liq_50x_above_amt, liq_50x_below_amt,
                liq_25x_above_amt, liq_25x_below_amt,
                whale_pct, whale_delta, atr,
                data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, symbol, price,
            levels.get('100x_above'), levels.get('100x_below'),
            levels.get('50x_above'), levels.get('50x_below'),
            levels.get('25x_above'), levels.get('25x_below'),
            levels.get('100x_above_amt'), levels.get('100x_below_amt'),
            levels.get('50x_above_amt'), levels.get('50x_below_amt'),
            levels.get('25x_above_amt'), levels.get('25x_below_amt'),
            whale_pct, whale_delta, atr,
            data_source
        ))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"[LIQ_COLLECTOR] Error saving snapshot: {e}")
        return False


def save_from_liquidity_data(
    symbol: str,
    liq_data: Dict,
    current_price: float,
    whale_pct: float = None,
    whale_delta: float = None,
    atr: float = None,
    data_source: str = 'scanner'
) -> bool:
    """
    Save liquidation data in the format returned by liquidity_sequence_v2.
    
    This is the main function to call from scanner/single analysis.
    
    Args:
        symbol: Trading pair
        liq_data: Data from get_liquidation_levels() containing:
            {
                'levels': [
                    {'leverage': '100x', 'price': 99800, 'side': 'long', 'amount': 5000000},
                    {'leverage': '100x', 'price': 105200, 'side': 'short', 'amount': 4500000},
                    ...
                ]
            }
        current_price: Current market price
        whale_pct: Whale positioning %
        whale_delta: Whale change
        atr: ATR value
        data_source: Source identifier
    
    Returns:
        True if saved
    """
    try:
        levels_dict = {}
        
        # Parse the liquidation data
        if isinstance(liq_data, dict):
            level_list = liq_data.get('levels', [])
            
            for level in level_list:
                leverage = level.get('leverage', '').replace('x', '')
                price = level.get('price', 0)
                side = level.get('side', '').lower()
                amount = level.get('amount', 0)
                
                if not leverage or not price:
                    continue
                
                # Determine if above or below current price
                if price > current_price:
                    key = f"{leverage}x_above"
                    amt_key = f"{leverage}x_above_amt"
                else:
                    key = f"{leverage}x_below"
                    amt_key = f"{leverage}x_below_amt"
                
                # Keep the closest level to current price for each leverage
                if key not in levels_dict or abs(price - current_price) < abs(levels_dict[key] - current_price):
                    levels_dict[key] = price
                    levels_dict[amt_key] = amount
        
        # Also handle direct level format
        if 'liq_100x_above' in liq_data:
            levels_dict['100x_above'] = liq_data.get('liq_100x_above')
            levels_dict['100x_below'] = liq_data.get('liq_100x_below')
            levels_dict['50x_above'] = liq_data.get('liq_50x_above')
            levels_dict['50x_below'] = liq_data.get('liq_50x_below')
            levels_dict['25x_above'] = liq_data.get('liq_25x_above')
            levels_dict['25x_below'] = liq_data.get('liq_25x_below')
        
        if not levels_dict:
            return False
        
        return save_liquidation_snapshot(
            symbol=symbol,
            price=current_price,
            levels=levels_dict,
            whale_pct=whale_pct,
            whale_delta=whale_delta,
            atr=atr,
            data_source=data_source
        )
        
    except Exception as e:
        print(f"[LIQ_COLLECTOR] Error parsing liq data: {e}")
        return False


def get_collection_stats() -> Dict:
    """Get statistics about collected data."""
    try:
        if not os.path.exists(DB_PATH):
            return {'total_records': 0, 'symbols': [], 'date_range': None}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM liq_snapshots")
        total = cursor.fetchone()[0]
        
        # Unique symbols
        cursor.execute("SELECT DISTINCT symbol FROM liq_snapshots ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        
        # Records per symbol
        cursor.execute("""
            SELECT symbol, COUNT(*) as cnt 
            FROM liq_snapshots 
            GROUP BY symbol 
            ORDER BY cnt DESC
        """)
        per_symbol = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM liq_snapshots")
        min_ts, max_ts = cursor.fetchone()
        
        date_range = None
        if min_ts and max_ts:
            min_date = datetime.fromtimestamp(min_ts / 1000)
            max_date = datetime.fromtimestamp(max_ts / 1000)
            days = (max_date - min_date).days
            date_range = {
                'start': min_date.strftime('%Y-%m-%d'),
                'end': max_date.strftime('%Y-%m-%d'),
                'days': days
            }
        
        # Records with outcomes (backfilled)
        cursor.execute("SELECT COUNT(*) FROM liq_snapshots WHERE swept_level IS NOT NULL")
        with_outcomes = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_records': total,
            'symbols': symbols,
            'records_per_symbol': per_symbol,
            'date_range': date_range,
            'with_outcomes': with_outcomes,
            'ready_for_training': total >= 500 and with_outcomes >= 100
        }
        
    except Exception as e:
        print(f"[LIQ_COLLECTOR] Error getting stats: {e}")
        return {'total_records': 0, 'error': str(e)}


def backfill_outcomes(symbol: str = None, lookback_candles: int = 30) -> Dict:
    """
    Backfill sweep outcomes for collected data.
    
    For each snapshot, check if price later swept any liquidation level
    and whether it bounced.
    
    Args:
        symbol: Specific symbol to backfill (None = all)
        lookback_candles: How many candles forward to check
    
    Returns:
        Stats about backfilling
    """
    try:
        if not os.path.exists(DB_PATH):
            return {'error': 'Database not found'}
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get records without outcomes
        query = """
            SELECT id, timestamp, symbol, price,
                   liq_100x_above, liq_100x_below,
                   liq_50x_above, liq_50x_below,
                   liq_25x_above, liq_25x_below
            FROM liq_snapshots
            WHERE swept_level IS NULL
        """
        if symbol:
            query += f" AND symbol = '{symbol}'"
        
        cursor.execute(query)
        records = cursor.fetchall()
        
        if not records:
            conn.close()
            return {'updated': 0, 'message': 'No records to backfill'}
        
        # Try to import price fetcher
        try:
            from core.data_fetcher import fetch_binance_klines
        except:
            conn.close()
            return {'error': 'data_fetcher not available'}
        
        updated = 0
        
        for record in records:
            rec_id, timestamp, sym, price, l100_above, l100_below, l50_above, l50_below, l25_above, l25_below = record
            
            # Fetch price data after this timestamp
            try:
                df = fetch_binance_klines(sym, '4h', limit=lookback_candles + 5)
                if df is None or len(df) < 5:
                    continue
                
                # Find candle at or after timestamp
                rec_time = datetime.fromtimestamp(timestamp / 1000)
                df['datetime'] = pd.to_datetime(df['DateTime'])
                
                future_df = df[df['datetime'] >= rec_time].head(lookback_candles)
                if len(future_df) < 3:
                    continue
                
                # Check which level was swept first
                swept_level = None
                candles_to_sweep = None
                bounce = None
                
                levels_to_check = [
                    ('100x_above', l100_above),
                    ('100x_below', l100_below),
                    ('50x_above', l50_above),
                    ('50x_below', l50_below),
                    ('25x_above', l25_above),
                    ('25x_below', l25_below),
                ]
                
                for i, row in future_df.iterrows():
                    high = row['High']
                    low = row['Low']
                    close = row['Close']
                    
                    for level_name, level_price in levels_to_check:
                        if level_price is None:
                            continue
                        
                        # Check if swept
                        if 'above' in level_name and high >= level_price:
                            if swept_level is None:
                                swept_level = level_name
                                candles_to_sweep = i + 1
                                # Bounce = closed back below
                                bounce = 1 if close < level_price else 0
                                break
                        
                        elif 'below' in level_name and low <= level_price:
                            if swept_level is None:
                                swept_level = level_name
                                candles_to_sweep = i + 1
                                # Bounce = closed back above
                                bounce = 1 if close > level_price else 0
                                break
                    
                    if swept_level:
                        break
                
                # Update record
                if swept_level:
                    cursor.execute("""
                        UPDATE liq_snapshots
                        SET swept_level = ?, candles_to_sweep = ?, bounce_after_sweep = ?,
                            swept_above = ?, swept_below = ?
                        WHERE id = ?
                    """, (
                        swept_level, candles_to_sweep, bounce,
                        1 if 'above' in swept_level else 0,
                        1 if 'below' in swept_level else 0,
                        rec_id
                    ))
                    updated += 1
                else:
                    # No sweep within lookback - mark as checked
                    cursor.execute("""
                        UPDATE liq_snapshots
                        SET swept_level = 'NONE', swept_above = 0, swept_below = 0
                        WHERE id = ?
                    """, (rec_id,))
                    updated += 1
                    
            except Exception as e:
                continue
        
        conn.commit()
        conn.close()
        
        return {'updated': updated, 'total_checked': len(records)}
        
    except Exception as e:
        return {'error': str(e)}


def get_training_samples(min_records: int = 100) -> List[Dict]:
    """
    Get training samples from collected liquidation data.
    
    Only returns records with backfilled outcomes.
    
    Returns:
        List of training samples with features and labels
    """
    try:
        if not os.path.exists(DB_PATH):
            return []
        
        conn = sqlite3.connect(DB_PATH)
        
        df = pd.read_sql("""
            SELECT * FROM liq_snapshots
            WHERE swept_level IS NOT NULL AND swept_level != 'NONE'
            ORDER BY timestamp
        """, conn)
        
        conn.close()
        
        if len(df) < min_records:
            print(f"[LIQ_COLLECTOR] Only {len(df)} samples with outcomes. Need {min_records}+")
            return []
        
        samples = []
        
        for _, row in df.iterrows():
            # Calculate features
            price = row['price']
            atr = row['atr'] or (price * 0.02)  # Default 2% ATR
            
            # Distance to each level in ATR
            def calc_distance(level_price):
                if level_price and level_price > 0:
                    return abs(level_price - price) / atr
                return None
            
            sample = {
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'price': price,
                'atr': atr,
                
                # Level distances (features)
                'dist_100x_above': calc_distance(row['liq_100x_above']),
                'dist_100x_below': calc_distance(row['liq_100x_below']),
                'dist_50x_above': calc_distance(row['liq_50x_above']),
                'dist_50x_below': calc_distance(row['liq_50x_below']),
                'dist_25x_above': calc_distance(row['liq_25x_above']),
                'dist_25x_below': calc_distance(row['liq_25x_below']),
                
                # Whale context
                'whale_pct': row['whale_pct'] or 50,
                'whale_delta': row['whale_delta'] or 0,
                
                # Labels (outcomes)
                'swept_level': row['swept_level'],
                'swept_above': row['swept_above'],
                'swept_below': row['swept_below'],
                'bounce_after_sweep': row['bounce_after_sweep'],
                'candles_to_sweep': row['candles_to_sweep'],
                
                # Level type (for model)
                'level_type': f"{row['swept_level'].split('_')[0]}X_LIQ" if row['swept_level'] else 'UNKNOWN',
                'level_strength': 1.0 if '100x' in str(row['swept_level']) else 0.85 if '50x' in str(row['swept_level']) else 0.7,
            }
            
            samples.append(sample)
        
        print(f"[LIQ_COLLECTOR] Returning {len(samples)} training samples from real liquidation data!")
        return samples
        
    except Exception as e:
        print(f"[LIQ_COLLECTOR] Error getting training samples: {e}")
        return []


def generate_historical_liq_levels(
    symbols: List[str] = None,
    days: int = 365,
    progress_callback: callable = None
) -> Dict:
    """
    Generate historical liquidation levels from price data.
    
    Since Coinglass doesn't provide historical liquidation heatmap data,
    we ESTIMATE where levels would have been based on:
    - Historical prices
    - Leverage formulas (100x = 1%, 50x = 2%, 25x = 4%)
    
    This gives us training data for REAL liquidation levels!
    
    Args:
        symbols: List of symbols to process
        days: Days of history to process
        progress_callback: Optional callback(text, pct)
    
    Returns:
        Stats about generated data
    """
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
        ]
    
    def update_progress(text, pct):
        if progress_callback:
            progress_callback(text, pct)
        print(f"[LIQ_HISTORY] {text}")
    
    update_progress("⚡ Fetching historical price data...", 0.1)
    
    # Fetch price data
    try:
        from core.data_fetcher import fetch_klines_parallel
        
        def fetch_progress(completed, total, symbol):
            pct = 0.1 + (completed / total) * 0.3
            update_progress(f"⚡ Fetching {symbol}... {completed}/{total}", pct)
        
        klines_data = fetch_klines_parallel(
            symbols=symbols,
            interval='4h',
            limit=min(days * 6, 1500),
            progress_callback=fetch_progress
        )
    except ImportError:
        return {'error': 'data_fetcher not available'}
    
    if not klines_data:
        return {'error': 'No price data fetched'}
    
    update_progress("📊 Loading whale history...", 0.45)
    
    # Load whale history for context
    whale_history = {}
    try:
        import sqlite3
        whale_db = "data/whale_history.db"
        if os.path.exists(whale_db):
            conn = sqlite3.connect(whale_db)
            df_whale = pd.read_sql("""
                SELECT symbol, timestamp, whale_long_pct, retail_long_pct 
                FROM whale_snapshots 
                ORDER BY timestamp
            """, conn)
            conn.close()
            
            # Index by symbol and timestamp
            for _, row in df_whale.iterrows():
                sym = row['symbol']
                ts = row['timestamp']
                if sym not in whale_history:
                    whale_history[sym] = {}
                whale_history[sym][ts] = {
                    'whale_pct': row['whale_long_pct'],
                    'retail_pct': row['retail_long_pct']
                }
            update_progress(f"📊 Loaded whale data for {len(whale_history)} symbols", 0.5)
    except Exception as e:
        update_progress(f"⚠️ Could not load whale history: {e}", 0.5)
    
    update_progress("🔄 Generating liquidation levels...", 0.55)
    
    # Generate levels for each price point
    records_saved = 0
    total_symbols = len(klines_data)
    
    for idx, (symbol, df) in enumerate(klines_data.items()):
        pct = 0.55 + (idx / total_symbols) * 0.4
        update_progress(f"🔄 Processing {symbol}... {idx+1}/{total_symbols}", pct)
        
        if df is None or len(df) < 50:
            print(f"[LIQ_HISTORY] Skipping {symbol} - insufficient data ({len(df) if df is not None else 0} rows)")
            continue
        
        # Debug: show columns
        print(f"[LIQ_HISTORY] {symbol} columns: {list(df.columns)}")
        
        # Normalize columns to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Get timestamps - handle different column names
        timestamps = None
        if 'datetime' in df.columns:
            timestamps = df['datetime'].tolist()
        elif 'date' in df.columns:
            timestamps = df['date'].tolist()
        elif 'time' in df.columns:
            timestamps = df['time'].tolist()
        
        symbol_records = 0
        
        # Sample every 6 candles (once per day for 4h data)
        for i in range(14, len(df), 6):
            try:
                price = float(df['close'].iloc[i])
                atr_val = df['atr'].iloc[i]
                atr = float(atr_val) if pd.notna(atr_val) else price * 0.02
                
                if price <= 0:
                    continue
                
                # Get timestamp in milliseconds
                if timestamps is not None:
                    ts = timestamps[i]
                    if hasattr(ts, 'timestamp'):
                        ts_ms = int(ts.timestamp() * 1000)
                    elif isinstance(ts, (int, float)):
                        ts_ms = int(ts) if ts > 1e12 else int(ts * 1000)
                    else:
                        # Use index-based time
                        ts_ms = int(time.time() * 1000) - (len(df) - i) * 4 * 60 * 60 * 1000
                else:
                    # Estimate timestamp based on position
                    ts_ms = int(time.time() * 1000) - (len(df) - i) * 4 * 60 * 60 * 1000
                
                # Calculate liquidation levels based on leverage
                levels = {
                    '100x_above': price * 1.01,
                    '100x_below': price * 0.99,
                    '50x_above': price * 1.02,
                    '50x_below': price * 0.98,
                    '25x_above': price * 1.04,
                    '25x_below': price * 0.96,
                }
                
                # Get whale data if available
                whale_pct = 50
                sym_whale = whale_history.get(symbol, {})
                if sym_whale:
                    try:
                        closest_ts = min(sym_whale.keys(), key=lambda x: abs(x - ts_ms), default=None)
                        if closest_ts and abs(closest_ts - ts_ms) < 24 * 60 * 60 * 1000:
                            whale_pct = sym_whale[closest_ts].get('whale_pct', 50)
                    except:
                        pass
                
                # Save to database - use unique timestamp per record
                record_ts = ts_ms + i  # Add offset to avoid duplicates
                
                saved = save_liquidation_snapshot(
                    symbol=symbol,
                    price=price,
                    levels=levels,
                    whale_pct=whale_pct,
                    atr=atr,
                    data_source='historical',
                    timestamp_override=record_ts
                )
                
                if saved:
                    records_saved += 1
                    symbol_records += 1
                    
            except Exception as e:
                print(f"[LIQ_HISTORY] Error at {symbol} idx {i}: {e}")
                continue
        
        print(f"[LIQ_HISTORY] {symbol}: saved {symbol_records} records")
    
    update_progress("✅ Historical data generation complete!", 1.0)
    
    # Now backfill outcomes
    update_progress("🔄 Backfilling outcomes...", 0.95)
    backfill_result = backfill_outcomes()
    
    return {
        'records_saved': records_saved,
        'symbols_processed': len(klines_data),
        'backfill_result': backfill_result
    }


# Initialize on import
init_database()


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("LIQUIDATION LEVEL COLLECTOR - TEST")
    print("=" * 60)
    
    # Test save
    test_levels = {
        '100x_above': 105000,
        '100x_below': 99000,
        '50x_above': 107000,
        '50x_below': 97000,
        '25x_above': 110000,
        '25x_below': 94000,
    }
    
    saved = save_liquidation_snapshot(
        symbol='BTCUSDT',
        price=102000,
        levels=test_levels,
        whale_pct=65,
        data_source='test'
    )
    print(f"Save test: {'✅ Success' if saved else '❌ Failed'}")
    
    # Get stats
    stats = get_collection_stats()
    print(f"\nCollection Stats:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Symbols: {stats['symbols']}")
    print(f"  Date range: {stats['date_range']}")
    print(f"  Ready for training: {stats['ready_for_training']}")
