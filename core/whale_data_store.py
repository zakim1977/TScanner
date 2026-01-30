"""
Whale Data Store
================
Stores whale/institutional data as we fetch it for future historical validation.
Over time, this builds a database of OI, whale positioning, funding rate etc.

Storage Strategy:
- Whale data (OI, positioning, funding) is SYMBOL-LEVEL (same for all timeframes)
- Technical data (MFI, RSI) is TIMEFRAME-LEVEL (different per TF)
- Store whale data once per hour per symbol (it doesn't change faster than that)
- Technical data can be derived from raw OHLCV when needed for validation

Data Sources:
- LIVE: Real-time data from daily scans (ongoing collection)
- HISTORICAL: Imported from paid APIs (Coinglass, etc.)
- Both are stored in the same table with data_source flag
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
import threading

# Thread lock for database access
_db_lock = threading.Lock()

# Track last storage time per symbol (not per timeframe!)
# Whale data is symbol-level, not timeframe-specific
_last_stored: Dict[str, datetime] = {}

# Whale data refresh interval (minutes)
# OI/positioning doesn't change faster than this
WHALE_DATA_REFRESH_MINUTES = 60  # Store once per hour per symbol

# Data source constants
DATA_SOURCE_LIVE = "live"
DATA_SOURCE_HISTORICAL = "historical"


def _should_store(symbol: str) -> bool:
    """
    Check if we should store whale data for this symbol.
    Store once per hour regardless of timeframe being analyzed.
    (Whale data is symbol-level, not timeframe-specific)
    """
    now = datetime.now()
    
    if symbol not in _last_stored:
        _last_stored[symbol] = now
        return True
    
    time_since_last = (now - _last_stored[symbol]).total_seconds() / 60
    
    if time_since_last >= WHALE_DATA_REFRESH_MINUTES:
        _last_stored[symbol] = now
        return True
    
    return False


@dataclass
class WhaleSnapshot:
    """Single point-in-time whale data snapshot"""
    symbol: str
    timeframe: str
    timestamp: str  # ISO format
    
    # Whale positioning
    whale_long_pct: float
    retail_long_pct: float
    
    # Open Interest
    oi_value: float
    oi_change_24h: float
    
    # Funding
    funding_rate: float
    
    # Price context
    price: float
    price_change_24h: float
    
    # Technical context (stored together for easy matching)
    position_in_range: float
    mfi: float
    cmf: float
    rsi: float
    volume_ratio: float
    atr_pct: float
    
    # Our signal (if generated)
    signal_direction: Optional[str] = None  # LONG/SHORT/None
    predictive_score: Optional[int] = None
    
    # Outcome (filled later when we know what happened)
    hit_tp1: Optional[bool] = None
    hit_sl: Optional[bool] = None
    candles_to_result: Optional[int] = None
    max_favorable_pct: Optional[float] = None
    max_adverse_pct: Optional[float] = None
    
    # Data source tracking (NEW)
    data_source: str = DATA_SOURCE_LIVE  # 'live' or 'historical'


class WhaleDataStore:
    """
    SQLite-based store for whale data snapshots.
    Builds historical database over time for ML validation.
    """
    
    def __init__(self, db_path: str = "data/whale_history.db"):
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        with _db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='whale_snapshots'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Check if data_source column exists
                cursor.execute("PRAGMA table_info(whale_snapshots)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'data_source' not in columns:
                    print("Migrating database: Adding data_source column...")
                    cursor.execute("ALTER TABLE whale_snapshots ADD COLUMN data_source TEXT DEFAULT 'live'")
                    conn.commit()
                    print("Migration complete!")
            
            # Create table with all columns if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS whale_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    
                    -- Whale data
                    whale_long_pct REAL,
                    retail_long_pct REAL,
                    oi_value REAL,
                    oi_change_24h REAL,
                    funding_rate REAL,
                    
                    -- Price
                    price REAL,
                    price_change_24h REAL,
                    
                    -- Technical
                    position_in_range REAL,
                    mfi REAL,
                    cmf REAL,
                    rsi REAL,
                    volume_ratio REAL,
                    atr_pct REAL,
                    
                    -- Signal
                    signal_direction TEXT,
                    predictive_score INTEGER,
                    
                    -- Outcome (updated later)
                    hit_tp1 INTEGER,
                    hit_sl INTEGER,
                    candles_to_result INTEGER,
                    max_favorable_pct REAL,
                    max_adverse_pct REAL,
                    
                    -- Data source tracking (live vs historical import)
                    data_source TEXT DEFAULT 'live',
                    
                    -- Indexes
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Create indexes for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_tf 
                ON whale_snapshots(symbol, timeframe)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON whale_snapshots(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_source 
                ON whale_snapshots(data_source)
            """)
            
            conn.commit()
            conn.close()
    
    def store_snapshot(self, snapshot: WhaleSnapshot) -> bool:
        """
        Store a whale data snapshot.
        Only keeps ONE record per symbol/timeframe/day (latest wins).
        Returns True if stored, False if error.
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Extract date+hour for deduplication (allows multiple snapshots per day for 24h analysis)
                # Format: "2025-01-25 14:30:00" -> "2025-01-25 14"
                if ' ' in snapshot.timestamp:
                    parts = snapshot.timestamp.split(' ')
                    date_part = parts[0]
                    hour_part = parts[1].split(':')[0] if len(parts) > 1 else '00'
                    snapshot_hour = f"{date_part} {hour_part}"
                else:
                    snapshot_hour = snapshot.timestamp.split('T')[0] + " 00"

                # For historical data, don't delete existing - allow multiple per day
                data_source = getattr(snapshot, 'data_source', DATA_SOURCE_LIVE)

                if data_source == DATA_SOURCE_LIVE:
                    # Live data: Delete any existing record for same symbol/timeframe/HOUR (not day)
                    # This allows up to 24 snapshots per day for better 24h delta calculation
                    cursor.execute("""
                        DELETE FROM whale_snapshots
                        WHERE symbol = ? AND timeframe = ? AND timestamp LIKE ? AND data_source = 'live'
                    """, (snapshot.symbol, snapshot.timeframe, f"{snapshot_hour}%"))
                
                # Insert new record
                cursor.execute("""
                    INSERT OR REPLACE INTO whale_snapshots (
                        symbol, timeframe, timestamp,
                        whale_long_pct, retail_long_pct,
                        oi_value, oi_change_24h, funding_rate,
                        price, price_change_24h,
                        position_in_range, mfi, cmf, rsi, volume_ratio, atr_pct,
                        signal_direction, predictive_score,
                        hit_tp1, hit_sl, candles_to_result, max_favorable_pct, max_adverse_pct,
                        data_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.symbol, snapshot.timeframe, snapshot.timestamp,
                    snapshot.whale_long_pct, snapshot.retail_long_pct,
                    snapshot.oi_value, snapshot.oi_change_24h, snapshot.funding_rate,
                    snapshot.price, snapshot.price_change_24h,
                    snapshot.position_in_range, snapshot.mfi, snapshot.cmf, 
                    snapshot.rsi, snapshot.volume_ratio, snapshot.atr_pct,
                    snapshot.signal_direction, snapshot.predictive_score,
                    snapshot.hit_tp1, snapshot.hit_sl, snapshot.candles_to_result,
                    snapshot.max_favorable_pct, snapshot.max_adverse_pct,
                    data_source
                ))
                
                conn.commit()
                conn.close()
                return True
                
        except Exception as e:
            print(f"Error storing whale snapshot: {e}")
            return False
    
    def update_outcome(
        self,
        symbol: str,
        timeframe: str,
        timestamp: str,
        hit_tp1: bool,
        hit_sl: bool,
        candles_to_result: int,
        max_favorable_pct: float,
        max_adverse_pct: float
    ) -> bool:
        """Update outcome for a stored snapshot"""
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE whale_snapshots 
                    SET hit_tp1 = ?, hit_sl = ?, candles_to_result = ?,
                        max_favorable_pct = ?, max_adverse_pct = ?
                    WHERE symbol = ? AND timeframe = ? AND timestamp = ?
                """, (
                    1 if hit_tp1 else 0, 1 if hit_sl else 0, candles_to_result,
                    max_favorable_pct, max_adverse_pct,
                    symbol, timeframe, timestamp
                ))
                
                conn.commit()
                conn.close()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating outcome: {e}")
            return False
    
    def get_snapshots(
        self,
        symbol: str,
        timeframe: str,
        lookback_days: int = 365,
        with_outcomes_only: bool = False,
        any_timeframe: bool = False  # If True, get all TFs for this symbol
    ) -> List[Dict]:
        """
        Get historical snapshots for a symbol.
        
        Args:
            any_timeframe: If True, get snapshots from ALL timeframes for this symbol
                          (useful because whale data is symbol-level, not TF-specific)
            with_outcomes_only: If True, only return snapshots where outcome is known
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
                
                if any_timeframe:
                    # Get ALL timeframes for this symbol (whale data is TF-agnostic)
                    if with_outcomes_only:
                        cursor.execute("""
                            SELECT * FROM whale_snapshots
                            WHERE symbol = ? AND timestamp >= ?
                            AND hit_tp1 IS NOT NULL
                            ORDER BY timestamp ASC
                        """, (symbol, cutoff))
                    else:
                        cursor.execute("""
                            SELECT * FROM whale_snapshots
                            WHERE symbol = ? AND timestamp >= ?
                            ORDER BY timestamp ASC
                        """, (symbol, cutoff))
                else:
                    # Get specific timeframe only
                    if with_outcomes_only:
                        cursor.execute("""
                            SELECT * FROM whale_snapshots
                            WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                            AND hit_tp1 IS NOT NULL
                            ORDER BY timestamp ASC
                        """, (symbol, timeframe, cutoff))
                    else:
                        cursor.execute("""
                            SELECT * FROM whale_snapshots
                            WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                            ORDER BY timestamp ASC
                        """, (symbol, timeframe, cutoff))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"Error getting snapshots: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get overall statistics about stored data"""
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM whale_snapshots")
                total = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE hit_tp1 IS NOT NULL")
                with_outcomes = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM whale_snapshots")
                symbols = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM whale_snapshots")
                date_range = cursor.fetchone()
                
                # Data source breakdown
                cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE data_source = 'live'")
                live_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE data_source = 'historical'")
                historical_count = cursor.fetchone()[0]
                
                conn.close()
                
                return {
                    'total_snapshots': total,
                    'with_outcomes': with_outcomes,
                    'pending_outcomes': total - with_outcomes,
                    'symbols_tracked': symbols,
                    'oldest_data': date_range[0],
                    'newest_data': date_range[1],
                    'live_records': live_count,
                    'historical_records': historical_count,
                }
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def get_all_snapshots(
        self,
        lookback_days: int = 365,
        max_records: int = 5000
    ) -> List[Dict]:
        """
        Get historical snapshots from ALL symbols.
        
        Used for cross-symbol pattern matching - whale patterns are
        market-wide, so similar conditions on BTC validate setups on altcoins.
        
        Args:
            lookback_days: How far back to look
            max_records: Maximum records to return (for performance)
            
        Returns:
            List of snapshot dictionaries
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get diverse sample from all symbols (NO timestamp filter for now)
                cursor.execute("""
                    SELECT * FROM whale_snapshots
                    ORDER BY RANDOM()
                    LIMIT ?
                """, (max_records,))
                
                rows = cursor.fetchall()
                conn.close()
                
                # print(f"[DEBUG] get_all_snapshots returned {len(rows)} rows")
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"Error getting all snapshots: {e}")
            return []
    
    def cleanup_old_data(self, keep_days: int = 400):
        """Remove data older than keep_days"""
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
                
                cursor.execute("""
                    DELETE FROM whale_snapshots WHERE timestamp < ?
                """, (cutoff,))
                
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                
                return deleted
        except Exception as e:
            print(f"Error cleaning up: {e}")
            return 0
    
    def backtest_pending_outcomes(self, max_symbols: int = 50) -> int:
        """
        Calculate outcomes for records that don't have them yet.
        Called automatically on app startup.
        
        Args:
            max_symbols: Process at most this many symbols per call (for performance)
            
        Returns:
            Number of records updated
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Find symbols with pending outcomes
                cursor.execute("""
                    SELECT DISTINCT symbol FROM whale_snapshots 
                    WHERE hit_tp1 IS NULL 
                    LIMIT ?
                """, (max_symbols,))
                
                symbols = [row[0] for row in cursor.fetchall()]
                
                if not symbols:
                    conn.close()
                    return 0
                
                total_updated = 0
                
                for symbol in symbols:
                    # Get all data for this symbol, ordered by timestamp
                    # INCLUDE atr_pct and whale_long_pct for dynamic thresholds!
                    cursor.execute("""
                        SELECT id, timestamp, price, atr_pct, whale_long_pct FROM whale_snapshots
                        WHERE symbol = ?
                        ORDER BY timestamp ASC
                    """, (symbol,))
                    
                    rows = cursor.fetchall()
                    if len(rows) < 15:
                        continue
                    
                    # ATR multipliers - adjusted for Position Trade 1D
                    # For daily charts, crypto typically moves 2-4% per day
                    # We use tighter targets for more realistic outcomes
                    SL_ATR_MULT = 1.5      # SL = ATR * 1.5 (was 2.0)
                    TP_RR_RATIO = 2.0      # TP = SL * 2.0 for 2:1 R:R (was 1.5)
                    DEFAULT_ATR = 2.5      # More realistic for crypto (was 4.0)
                    MAX_BARS = 15          # 15 days lookahead for Position Trade (was 8)
                    
                    # Process each record
                    for i in range(len(rows) - MAX_BARS):
                        record_id, timestamp, price, atr_pct, whale_pct = rows[i]
                        
                        if not price or price <= 0:
                            continue
                        
                        # MATCH SignalGenerator: SL = ATR * 2, TP = SL * 1.5
                        atr = atr_pct if atr_pct and atr_pct > 0 else DEFAULT_ATR
                        sl_pct = atr * SL_ATR_MULT  # e.g., 2% * 2.0 = 4%
                        tp_pct = sl_pct * TP_RR_RATIO  # e.g., 4% * 1.5 = 6%
                        
                        # Determine expected direction from whale %
                        if whale_pct and whale_pct >= 60:
                            expected_dir = 'LONG'
                        elif whale_pct and whale_pct <= 40:
                            expected_dir = 'SHORT'
                        else:
                            expected_dir = 'NEUTRAL'
                        
                        # Look ahead MAX_BARS
                        future_prices = [rows[j][2] for j in range(i+1, min(i+MAX_BARS+1, len(rows))) if rows[j][2] and rows[j][2] > 0]
                        if len(future_prices) < 3:
                            continue
                        
                        # Calculate max moves
                        max_up = max((p - price) / price * 100 for p in future_prices)
                        max_down = min((p - price) / price * 100 for p in future_prices)
                        
                        # Determine outcome - which hit first?
                        hit_tp1 = 0
                        hit_sl = 0
                        candles = 0
                        
                        for j, fp in enumerate(future_prices):
                            pct = (fp - price) / price * 100
                            
                            if expected_dir == 'LONG':
                                if pct >= tp_pct:
                                    hit_tp1 = 1
                                    candles = j + 1
                                    break
                                elif pct <= -sl_pct:
                                    hit_sl = 1
                                    candles = j + 1
                                    break
                            elif expected_dir == 'SHORT':
                                if pct <= -tp_pct:  # Down is good for short
                                    hit_tp1 = 1
                                    candles = j + 1
                                    break
                                elif pct >= sl_pct:  # Up is bad for short
                                    hit_sl = 1
                                    candles = j + 1
                                    break
                            else:
                                # Neutral - which significant move comes first?
                                if pct >= tp_pct:  # Up first = win
                                    hit_tp1 = 1
                                    candles = j + 1
                                    break
                                elif pct <= -sl_pct:  # Down first = loss
                                    hit_sl = 1
                                    candles = j + 1
                                    break
                        
                        cursor.execute("""
                            UPDATE whale_snapshots 
                            SET hit_tp1 = ?, hit_sl = ?, 
                                max_favorable_pct = ?, max_adverse_pct = ?,
                                candles_to_result = ?
                            WHERE id = ?
                        """, (hit_tp1, hit_sl, max_up, max_down, candles, record_id))
                        
                        total_updated += 1
                
                conn.commit()
                conn.close()
                
                return total_updated
                
        except Exception as e:
            print(f"Backtest error: {e}")
            return 0
    
    def reset_all_outcomes(self) -> int:
        """
        Reset all outcomes to NULL and recalculate them.
        Use this after changing backtest parameters.
        
        Returns:
            Number of records reset
        """
        try:
            with _db_lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Reset all outcomes
                cursor.execute("""
                    UPDATE whale_snapshots 
                    SET hit_tp1 = NULL, hit_sl = NULL, candles_to_outcome = NULL
                """)
                
                reset_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                print(f"Reset {reset_count} records")
                return reset_count
                
        except Exception as e:
            print(f"Reset error: {e}")
            return 0
    
    def full_rebacktest(self, batch_size: int = 100) -> dict:
        """
        Reset and recalculate ALL outcomes.
        Call this after importing data or changing parameters.
        
        Returns:
            Dict with stats
        """
        print("Resetting all outcomes...")
        reset = self.reset_all_outcomes()
        
        print("Re-running backtest...")
        total_updated = 0
        iterations = 0
        max_iterations = 5000  # Safety limit
        
        while iterations < max_iterations:
            updated = self.backtest_pending_outcomes(max_symbols=batch_size)
            if updated == 0:
                break
            total_updated += updated
            iterations += 1
            if iterations % 10 == 0:
                print(f"  Progress: {total_updated} records updated...")
        
        print(f"Backtest complete: {total_updated} records updated")
        return {
            'reset': reset,
            'updated': total_updated,
            'iterations': iterations
        }


# Global instance
_whale_store: Optional[WhaleDataStore] = None


def get_whale_store() -> WhaleDataStore:
    """Get or create global whale data store"""
    global _whale_store
    if _whale_store is None:
        _whale_store = WhaleDataStore()
    return _whale_store


def store_current_whale_data(
    symbol: str,
    timeframe: str,
    whale_data: dict,
    technical_data: dict,
    signal_direction: Optional[str] = None,
    predictive_score: Optional[int] = None
) -> bool:
    """
    Store whale data from current analysis.
    
    Rate limiting: Once per hour per SYMBOL (not per timeframe).
    Why? Whale data (OI, positioning, funding) is symbol-level - same 
    whether you're looking at 1m or 1d chart.
    
    This means if you scan BTCUSDT on 15m, the whale data is also valid
    for 4h, 1d validation later.
    """
    
    # Check if we should store (rate limiting per symbol, not timeframe)
    if not _should_store(symbol):
        return False  # Skip - already stored this hour for this symbol
    
    store = get_whale_store()
    
    # Extract whale data
    top_trader = whale_data.get('top_trader_ls', {})
    retail = whale_data.get('retail_ls', {})
    oi_data = whale_data.get('open_interest', {})
    funding = whale_data.get('funding', {})
    
    snapshot = WhaleSnapshot(
        symbol=symbol,
        timeframe=timeframe,  # Store which TF triggered this, but whale data is TF-agnostic
        timestamp=datetime.now().isoformat(),
        
        # Whale data (SAME for all timeframes - this is symbol-level data)
        whale_long_pct=top_trader.get('long_pct', 50) if isinstance(top_trader, dict) else 50,
        retail_long_pct=retail.get('long_pct', 50) if isinstance(retail, dict) else 50,
        oi_value=oi_data.get('value', 0) if isinstance(oi_data, dict) else 0,
        oi_change_24h=oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0,
        funding_rate=funding.get('rate', 0) if isinstance(funding, dict) else whale_data.get('funding_rate', 0),
        
        # Price (same for all timeframes)
        price=technical_data.get('price', 0),
        price_change_24h=whale_data.get('real_whale_data', {}).get('price_change_24h', 0) if whale_data.get('real_whale_data') else 0,
        
        # Technical (this IS timeframe-specific, but we store it anyway)
        # For cross-TF validation, we'll primarily match on whale data
        position_in_range=technical_data.get('position_in_range', 50),
        mfi=technical_data.get('mfi', 50),
        cmf=technical_data.get('cmf', 0),
        rsi=technical_data.get('rsi', 50),
        volume_ratio=technical_data.get('volume_ratio', 1.0),
        atr_pct=technical_data.get('atr_pct', 2.0),
        
        # Signal
        signal_direction=signal_direction,
        predictive_score=predictive_score,
    )
    
    return store.store_snapshot(snapshot)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL LOOKUP FOR MASTER_RULES
# ═══════════════════════════════════════════════════════════════════════════════

def find_similar_setups(
    whale_pct: float,
    oi_change: float,
    position_pct: float,
    direction: str = None,  # LONG, SHORT, or None for any
    symbol: str = None,     # Specific symbol or None for all
    lookback_days: int = 90,
    db_path: str = "data/whale_history.db"
) -> Dict:
    """
    Find historical setups similar to current conditions and calculate win rate.
    
    This is THE KEY FUNCTION that powers data-driven decisions!
    
    Matching criteria (WIDENED for better matches):
    - whale_pct: ±15% of current (was ±10%)
    - oi_change: same general direction OR any if flat
    - position_pct: wider buckets (EARLY 0-40, MIDDLE 30-70, LATE 60-100)
    
    Args:
        whale_pct: Current whale long percentage
        oi_change: Current OI change 24h %
        position_pct: Current position in range (0-100)
        direction: Filter by signal direction (LONG/SHORT/None)
        symbol: Filter by symbol (None = all symbols)
        lookback_days: How far back to look
        
    Returns:
        {
            'win_rate': float (0-100),
            'avg_return': float (percentage),
            'sample_size': int,
            'wins': int,
            'losses': int,
            'avg_favorable': float,
            'avg_adverse': float,
            'confidence': str (HIGH/MEDIUM/LOW/INSUFFICIENT)
        }
    """
    
    if not os.path.exists(db_path):
        return {
            'win_rate': None,
            'avg_return': None,
            'sample_size': 0,
            'wins': 0,
            'losses': 0,
            'confidence': 'INSUFFICIENT',
            'message': 'No historical database found'
        }
    
    # Define buckets - WIDENED for better matching
    whale_min = whale_pct - 15  # Was 10
    whale_max = whale_pct + 15  # Was 10
    
    # OI direction bucket - more lenient
    if oi_change > 3:
        oi_direction = 'RISING'
        oi_min, oi_max = 1, 100  # Was 2, 100
    elif oi_change < -3:
        oi_direction = 'FALLING'
        oi_min, oi_max = -100, -1  # Was -100, -2
    else:
        oi_direction = 'FLAT'
        oi_min, oi_max = -100, 100  # Accept ANY OI for flat (was -2, 2)
    
    # Position bucket - MUCH WIDER with overlap
    if position_pct <= 40:
        pos_bucket = 'EARLY'
        pos_min, pos_max = 0, 50  # Was 0, 35
    elif position_pct >= 60:
        pos_bucket = 'LATE'
        pos_min, pos_max = 50, 100  # Was 65, 100
    else:
        pos_bucket = 'MIDDLE'
        pos_min, pos_max = 25, 75  # Was 35, 65
    
    try:
        with _db_lock:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Build query
            # IMPORTANT: Exclude records with whale_long_pct = 50 (fake placeholder data from historical imports)
            query = """
                SELECT 
                    hit_tp1, hit_sl, max_favorable_pct, max_adverse_pct,
                    signal_direction, whale_long_pct, oi_change_24h, position_in_range,
                    timestamp
                FROM whale_snapshots
                WHERE 
                    whale_long_pct BETWEEN ? AND ?
                    AND whale_long_pct != 50.0
                    AND oi_change_24h BETWEEN ? AND ?
                    AND position_in_range BETWEEN ? AND ?
                    AND timestamp >= datetime('now', ?)
            """
            
            params = [whale_min, whale_max, oi_min, oi_max, pos_min, pos_max, f'-{lookback_days} days']
            
            # Optional filters
            if direction:
                query += " AND signal_direction = ?"
                params.append(direction)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # FALLBACK 1: If no results, try without position filter
            if len(rows) == 0:
                query_fallback = """
                    SELECT 
                        hit_tp1, hit_sl, max_favorable_pct, max_adverse_pct,
                        signal_direction, whale_long_pct, oi_change_24h, position_in_range,
                        timestamp
                    FROM whale_snapshots
                    WHERE 
                        whale_long_pct BETWEEN ? AND ?
                        AND whale_long_pct != 50.0
                        AND timestamp >= datetime('now', ?)
                """
                params_fallback = [whale_min, whale_max, f'-{lookback_days} days']
                
                if direction:
                    query_fallback += " AND signal_direction = ?"
                    params_fallback.append(direction)
                
                cursor.execute(query_fallback, params_fallback)
                rows = cursor.fetchall()
                
                if rows:
                    pos_bucket = 'ANY (fallback)'
            
            # FALLBACK 2: If still no results, extend lookback to 365 days
            if len(rows) == 0 and lookback_days < 365:
                query_fallback = """
                    SELECT 
                        hit_tp1, hit_sl, max_favorable_pct, max_adverse_pct,
                        signal_direction, whale_long_pct, oi_change_24h, position_in_range,
                        timestamp
                    FROM whale_snapshots
                    WHERE 
                        whale_long_pct BETWEEN ? AND ?
                        AND timestamp >= datetime('now', '-365 days')
                """
                params_fallback = [whale_min, whale_max]
                cursor.execute(query_fallback, params_fallback)
                rows = cursor.fetchall()
            
            # FALLBACK 3: If STILL no results, try ANY whale% just to see if data exists
            if len(rows) == 0:
                cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM whale_snapshots")
                count_row = cursor.fetchone()
                total_count = count_row[0] if count_row else 0
                min_ts = count_row[1] if count_row else 'N/A'
                max_ts = count_row[2] if count_row else 'N/A'
                
                # Try completely unrestricted query
                cursor.execute("""
                    SELECT 
                        hit_tp1, hit_sl, max_favorable_pct, max_adverse_pct,
                        signal_direction, whale_long_pct, oi_change_24h, position_in_range,
                        timestamp
                    FROM whale_snapshots
                    WHERE whale_long_pct BETWEEN 40 AND 80
                    LIMIT 1000
                """)
                rows = cursor.fetchall()
                
                if len(rows) == 0:
                    conn.close()
                    return {
                        'win_rate': None,
                        'sample_size': 0,
                        'confidence': 'INSUFFICIENT',
                        'message': f'DB has {total_count} records ({min_ts} to {max_ts}) but no whale data matches. Check data quality.'
                    }
            
            conn.close()
        
        if len(rows) == 0:
            return {
                'win_rate': None,
                'avg_return': None,
                'sample_size': 0,
                'wins': 0,
                'losses': 0,
                'confidence': 'INSUFFICIENT',
                'message': f'No similar setups with real whale data. Run Market Pulse with 🐋 Real API to build database.'
            }
        
        # Calculate outcomes
        wins = 0
        losses = 0
        total_favorable = 0
        total_adverse = 0
        
        for row in rows:
            hit_tp1, hit_sl, max_fav, max_adv, sig_dir, w_pct, oi_chg, pos, ts = row
            
            # Count win/loss (hit_tp1 before hit_sl = win)
            if hit_tp1:
                wins += 1
            elif hit_sl:
                losses += 1
            # else: no outcome recorded yet, skip
            
            if max_fav is not None:
                total_favorable += max_fav
            if max_adv is not None:
                total_adverse += abs(max_adv)
        
        total_with_outcome = wins + losses
        
        if total_with_outcome == 0:
            return {
                'win_rate': None,
                'avg_return': None,
                'sample_size': len(rows),
                'wins': 0,
                'losses': 0,
                'confidence': 'INSUFFICIENT',
                'message': f'{len(rows)} similar setups found but no outcomes. Run: python backtest_outcomes.py'
            }
        
        win_rate = (wins / total_with_outcome) * 100
        avg_favorable = total_favorable / len(rows) if len(rows) > 0 else 0
        avg_adverse = total_adverse / len(rows) if len(rows) > 0 else 0
        
        # Estimate average return (simplified)
        avg_return = (win_rate / 100) * avg_favorable - ((100 - win_rate) / 100) * avg_adverse
        
        # Confidence based on sample size
        if total_with_outcome >= 20:
            confidence = 'HIGH'
        elif total_with_outcome >= 10:
            confidence = 'MEDIUM'
        elif total_with_outcome >= 5:
            confidence = 'LOW'
        else:
            confidence = 'INSUFFICIENT'
        
        return {
            'win_rate': round(win_rate, 1),
            'avg_return': round(avg_return, 2),
            'sample_size': total_with_outcome,
            'total_matches': len(rows),
            'wins': wins,
            'losses': losses,
            'avg_favorable': round(avg_favorable, 2),
            'avg_adverse': round(avg_adverse, 2),
            'confidence': confidence,
            'whale_bucket': f'{whale_min:.0f}-{whale_max:.0f}%',
            'oi_bucket': oi_direction,
            'position_bucket': pos_bucket,
            'message': f'{win_rate:.0f}% win rate ({wins}W/{losses}L) from {total_with_outcome} similar setups'
        }
        
    except Exception as e:
        return {
            'win_rate': None,
            'avg_return': None,
            'sample_size': 0,
            'wins': 0,
            'losses': 0,
            'confidence': 'ERROR',
            'message': f'Error querying historical data: {str(e)}'
        }