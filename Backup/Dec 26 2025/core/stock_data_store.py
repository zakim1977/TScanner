"""
Stock Data Store for Historical Validation
==========================================
Stores Quiver institutional data for stocks/ETFs over time.
This is the stock equivalent of whale_data_store.py for crypto.

Data stored:
- Congress trading score
- Insider trading score  
- Short interest
- Combined institutional score
- Outcomes (hit TP1, hit SL, time taken)
"""

import os
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time


@dataclass
class StockSnapshot:
    """Single point-in-time institutional data snapshot"""
    symbol: str
    timeframe: str
    timestamp: str
    
    # Quiver institutional data
    congress_score: float  # 0-100 (like whale_pct)
    insider_score: float   # 0-100
    short_interest_pct: float  # Short interest as % of float
    combined_score: float  # Overall institutional score
    
    # Technical context
    price: float
    price_change_24h: float
    rsi: float
    position_pct: float  # Position in range 0-100
    ta_score: float
    
    # Signal context
    signal_direction: str  # 'LONG', 'SHORT', 'WAIT'
    signal_name: str
    confidence: str
    
    # Outcome tracking (filled in later)
    hit_tp1: bool = False
    hit_sl: bool = False
    candles_to_result: int = 0
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0
    outcome_tracked: bool = False


class StockDataStore:
    """
    Store for stock institutional data snapshots.
    Uses simple JSON file storage (upgradeable to SQLite later).
    """
    
    def __init__(self, data_dir: str = "data/stock_history"):
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minute cache
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_file_path(self, symbol: str) -> str:
        """Get file path for symbol data"""
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        return os.path.join(self.data_dir, f"{safe_symbol}_history.json")
    
    def _load_symbol_data(self, symbol: str) -> List[Dict]:
        """Load data for a symbol"""
        # Check cache first
        cache_key = symbol
        if cache_key in self._cache:
            if time.time() - self._cache_time.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]
        
        file_path = self._get_file_path(symbol)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._cache[cache_key] = data
                    self._cache_time[cache_key] = time.time()
                    return data
            except Exception as e:
                print(f"Error loading stock data for {symbol}: {e}")
                return []
        return []
    
    def _save_symbol_data(self, symbol: str, data: List[Dict]):
        """Save data for a symbol"""
        file_path = self._get_file_path(symbol)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Update cache
            self._cache[symbol] = data
            self._cache_time[symbol] = time.time()
        except Exception as e:
            print(f"Error saving stock data for {symbol}: {e}")
    
    def store_snapshot(self, snapshot: StockSnapshot):
        """
        Store a new institutional data snapshot.
        Only keeps ONE record per symbol/timeframe/day (latest wins).
        """
        with self._lock:
            data = self._load_symbol_data(snapshot.symbol)
            
            # Extract just the date portion for deduplication
            snapshot_date = snapshot.timestamp.split(' ')[0] if ' ' in snapshot.timestamp else snapshot.timestamp.split('T')[0]
            
            # Remove any existing records for the same date and timeframe (keep latest)
            data = [d for d in data if not (
                d.get('timeframe') == snapshot.timeframe and 
                d.get('timestamp', '').startswith(snapshot_date)
            )]
            
            # Add new snapshot (always latest for that day)
            data.append(asdict(snapshot))
            
            # Trim old data (keep last 6 months)
            cutoff = datetime.now() - timedelta(days=180)
            cutoff_str = cutoff.strftime('%Y-%m-%d')
            data = [d for d in data if d.get('timestamp', '9999')[:10] >= cutoff_str]
            
            self._save_symbol_data(snapshot.symbol, data)
    
    def get_snapshots(
        self,
        symbol: str,
        timeframe: str = None,
        lookback_days: int = 90,
        with_outcomes_only: bool = False,
        any_timeframe: bool = False
    ) -> List[Dict]:
        """
        Get historical snapshots for validation.
        
        Args:
            symbol: Stock symbol
            timeframe: Filter by timeframe (optional)
            lookback_days: How far back to look
            with_outcomes_only: Only return snapshots with tracked outcomes
            any_timeframe: If True, return all timeframes for this symbol
        """
        data = self._load_symbol_data(symbol)
        
        if not data:
            return []
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=lookback_days)
        cutoff_str = cutoff.strftime('%Y-%m-%d')
        
        filtered = []
        for d in data:
            # Date filter
            if d.get('timestamp', '0000') < cutoff_str:
                continue
            
            # Timeframe filter
            if not any_timeframe and timeframe and d.get('timeframe') != timeframe:
                continue
            
            # Outcome filter
            if with_outcomes_only and not d.get('outcome_tracked'):
                continue
            
            filtered.append(d)
        
        return filtered
    
    def update_outcome(
        self,
        symbol: str,
        timestamp: str,
        hit_tp1: bool,
        hit_sl: bool,
        candles_to_result: int,
        max_favorable_pct: float,
        max_adverse_pct: float
    ):
        """Update a snapshot with its outcome"""
        with self._lock:
            data = self._load_symbol_data(symbol)
            
            for d in data:
                if d.get('timestamp') == timestamp:
                    d['hit_tp1'] = hit_tp1
                    d['hit_sl'] = hit_sl
                    d['candles_to_result'] = candles_to_result
                    d['max_favorable_pct'] = max_favorable_pct
                    d['max_adverse_pct'] = max_adverse_pct
                    d['outcome_tracked'] = True
                    break
            
            self._save_symbol_data(symbol, data)
    
    def get_stats(self, symbol: str = None) -> Dict:
        """Get statistics about stored data"""
        if symbol:
            data = self._load_symbol_data(symbol)
            return {
                'symbol': symbol,
                'total_snapshots': len(data),
                'with_outcomes': sum(1 for d in data if d.get('outcome_tracked')),
                'timeframes': list(set(d.get('timeframe') for d in data)),
            }
        else:
            # Get stats for all symbols
            total = 0
            with_outcomes = 0
            symbols = []
            
            for file in os.listdir(self.data_dir):
                if file.endswith('_history.json'):
                    symbol = file.replace('_history.json', '')
                    data = self._load_symbol_data(symbol)
                    total += len(data)
                    with_outcomes += sum(1 for d in data if d.get('outcome_tracked'))
                    symbols.append(symbol)
            
            return {
                'total_symbols': len(symbols),
                'total_snapshots': total,
                'with_outcomes': with_outcomes,
                'symbols': symbols[:20],  # First 20
            }


# Singleton instance
_stock_store: Optional[StockDataStore] = None
_stock_store_lock = threading.Lock()


def get_stock_store() -> StockDataStore:
    """Get or create the singleton stock data store"""
    global _stock_store
    
    if _stock_store is None:
        with _stock_store_lock:
            if _stock_store is None:
                _stock_store = StockDataStore()
    
    return _stock_store


def store_stock_snapshot(
    symbol: str,
    timeframe: str,
    quiver_data: Dict,
    technical_data: Dict,
    signal_data: Dict
):
    """
    Convenience function to store a stock snapshot.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        timeframe: Timeframe analyzed
        quiver_data: Data from Quiver API (congress, insider, short interest)
        technical_data: Technical analysis data (RSI, position, etc.)
        signal_data: Signal information (direction, name, confidence)
    """
    store = get_stock_store()
    
    snapshot = StockSnapshot(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Quiver data
        congress_score=quiver_data.get('congress_score', 50),
        insider_score=quiver_data.get('insider_score', 50),
        short_interest_pct=quiver_data.get('short_interest_pct', 0),
        combined_score=quiver_data.get('combined_score', 50),
        
        # Technical
        price=technical_data.get('price', 0),
        price_change_24h=technical_data.get('price_change_24h', 0),
        rsi=technical_data.get('rsi', 50),
        position_pct=technical_data.get('position_pct', 50),
        ta_score=technical_data.get('ta_score', 50),
        
        # Signal
        signal_direction=signal_data.get('direction', 'WAIT'),
        signal_name=signal_data.get('name', ''),
        confidence=signal_data.get('confidence', 'MEDIUM'),
    )
    
    store.store_snapshot(snapshot)
