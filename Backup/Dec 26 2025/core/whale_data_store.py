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
                
                # Extract just the date portion for deduplication
                snapshot_date = snapshot.timestamp.split(' ')[0] if ' ' in snapshot.timestamp else snapshot.timestamp.split('T')[0]
                
                # Delete any existing record for same symbol/timeframe/date
                cursor.execute("""
                    DELETE FROM whale_snapshots 
                    WHERE symbol = ? AND timeframe = ? AND timestamp LIKE ?
                """, (snapshot.symbol, snapshot.timeframe, f"{snapshot_date}%"))
                
                # Insert new record (always latest for that day)
                cursor.execute("""
                    INSERT INTO whale_snapshots (
                        symbol, timeframe, timestamp,
                        whale_long_pct, retail_long_pct,
                        oi_value, oi_change_24h, funding_rate,
                        price, price_change_24h,
                        position_in_range, mfi, cmf, rsi, volume_ratio, atr_pct,
                        signal_direction, predictive_score,
                        hit_tp1, hit_sl, candles_to_result, max_favorable_pct, max_adverse_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.symbol, snapshot.timeframe, snapshot.timestamp,
                    snapshot.whale_long_pct, snapshot.retail_long_pct,
                    snapshot.oi_value, snapshot.oi_change_24h, snapshot.funding_rate,
                    snapshot.price, snapshot.price_change_24h,
                    snapshot.position_in_range, snapshot.mfi, snapshot.cmf, 
                    snapshot.rsi, snapshot.volume_ratio, snapshot.atr_pct,
                    snapshot.signal_direction, snapshot.predictive_score,
                    snapshot.hit_tp1, snapshot.hit_sl, snapshot.candles_to_result,
                    snapshot.max_favorable_pct, snapshot.max_adverse_pct
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
                
                conn.close()
                
                return {
                    'total_snapshots': total,
                    'with_outcomes': with_outcomes,
                    'pending_outcomes': total - with_outcomes,
                    'symbols_tracked': symbols,
                    'oldest_data': date_range[0],
                    'newest_data': date_range[1],
                }
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
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
