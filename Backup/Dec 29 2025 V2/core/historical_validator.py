"""
Historical Validator
====================
Validates signals by finding similar historical conditions using KNN.
Uses ML for similarity search (not prediction) to find "what happened to similar setups".

Over time, as we store more whale data, validation becomes more accurate.
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import threading
import pickle

# Import whale store lazily to avoid circular imports
def _get_whale_store():
    from core.whale_data_store import get_whale_store
    return get_whale_store()


@dataclass
class HistoricalMatch:
    """Single historical match"""
    date: str
    similarity: float
    result: str  # WIN/LOSS/TIMEOUT
    candles: int
    time_str: str
    conditions: Dict = field(default_factory=dict)


@dataclass
class HistoricalValidation:
    """Result of historical pattern matching"""
    
    # Sample info
    matches_found: int
    lookback_period: str
    avg_similarity: float
    data_source: str  # "whale_db" or "technical_only"
    
    # Outcomes
    win_rate: float
    loss_rate: float
    timeout_rate: float
    
    # Timing
    avg_candles_to_tp1: float
    avg_candles_to_sl: float
    avg_time_to_tp1: str
    avg_time_to_sl: str
    
    # Risk metrics
    avg_max_favorable: float
    avg_max_adverse: float
    
    # Confidence & grading
    sample_confidence: str       # HIGH/MEDIUM/LOW
    history_grade: str           # A+, A, B, C, D, F
    alignment: str               # ALIGNED/OVER_CONFIDENT/UNDER_CONFIDENT
    
    # Scores
    our_score: int
    historical_score: int
    
    # Top matches for transparency
    top_matches: List[HistoricalMatch] = field(default_factory=list)
    
    # Status message
    message: str = ""


class HistoricalValidator:
    """
    Validates signals by finding similar historical conditions using KNN.
    
    Two data sources:
    1. Whale DB (preferred): Our stored snapshots with real whale data
    2. Technical fallback: Calculate from raw OHLCV (no whale data)
    """
    
    # Feature weights for similarity matching
    # Higher = more important for finding similar conditions
    WHALE_FEATURE_WEIGHTS = {
        'whale_long_pct': 2.0,       # Most important - leading indicator
        'oi_change_24h': 1.8,        # Very important
        'retail_long_pct': 1.2,
        'funding_rate': 1.0,
        'position_in_range': 1.5,    # Entry timing
        'mfi': 0.8,
        'cmf': 0.8,
        'rsi': 0.6,
        'volume_ratio': 0.5,
        'atr_pct': 0.4,
    }
    
    # When whale data not available, use only technical
    TECHNICAL_FEATURE_WEIGHTS = {
        'position_in_range': 2.0,
        'mfi': 1.5,
        'cmf': 1.5,
        'rsi': 1.2,
        'volume_ratio': 1.0,
        'atr_pct': 0.8,
        'price_vs_ema20': 1.0,
        'ema_alignment': 1.2,
    }
    
    # For stocks/ETFs using Quiver institutional data
    STOCK_FEATURE_WEIGHTS = {
        'congress_score': 2.5,      # Congress trading (like whale_pct)
        'insider_score': 2.5,       # CEO/insider buys (very predictive)
        'short_interest_pct': 2.0,  # Short interest
        'combined_score': 2.0,      # Overall institutional score
        'position_pct': 1.8,        # Position in range
        'rsi': 1.2,
        'ta_score': 1.0,
        'price_change_24h': 0.8,
    }
    
    def __init__(
        self,
        cache_dir: str = "data/validation_cache",
        lookback_months: int = 12,
        min_samples: int = 20,
        k_neighbors: int = 50,
    ):
        self.cache_dir = cache_dir
        self.lookback_months = lookback_months
        self.min_samples = min_samples
        self.k_neighbors = k_neighbors
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Scalers per symbol/timeframe
        self.scalers = {}
        self.knn_models = {}
        
        # Thread lock
        self._lock = threading.Lock()
    
    def validate_signal(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_conditions: Dict,
        our_score: int,
        tp_distance_pct: float,
        sl_distance_pct: float,
        df: pd.DataFrame = None,  # Optional: raw data for technical fallback
        market_type: str = 'crypto',  # 'crypto' or 'stock'
    ) -> Optional[HistoricalValidation]:
        """
        Validate a signal by finding similar historical conditions.
        
        Args:
            current_conditions: Dict with whale/stock + technical data
            our_score: Our predictive score (0-100)
            tp_distance_pct: Distance to TP1 as %
            sl_distance_pct: Distance to SL as %
            df: Raw OHLCV data for technical-only fallback
            market_type: 'crypto' (uses whale data) or 'stock' (uses Quiver data)
            
        Returns:
            HistoricalValidation or None if insufficient data
        """
        
        # Route based on market type
        if market_type == 'stock':
            # Try stock database (Quiver institutional data)
            result = self._validate_from_stock_db(
                symbol, timeframe, direction, current_conditions,
                our_score, tp_distance_pct, sl_distance_pct
            )
            
            if result and result.matches_found >= self.min_samples:
                return result
        else:
            # Try whale database first (preferred - has real whale data)
            result = self._validate_from_whale_db(
                symbol, timeframe, direction, current_conditions,
                our_score, tp_distance_pct, sl_distance_pct
            )
            
            if result and result.matches_found >= self.min_samples:
                return result
        
        # Fallback to technical-only validation from raw data
        if df is not None and len(df) >= 500:
            return self._validate_from_technical(
                symbol, timeframe, direction, current_conditions,
                our_score, tp_distance_pct, sl_distance_pct, df
            )
        
        return None
    
    def _validate_from_stock_db(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_conditions: Dict,
        our_score: int,
        tp_distance_pct: float,
        sl_distance_pct: float,
    ) -> Optional[HistoricalValidation]:
        """Validate using stored Quiver stock data"""
        
        try:
            from .stock_data_store import get_stock_store
            store = get_stock_store()
        except ImportError:
            return None
        
        # Get historical snapshots
        snapshots = store.get_snapshots(
            symbol, timeframe,
            lookback_days=self.lookback_months * 30,
            with_outcomes_only=True,
            any_timeframe=False
        )
        
        # Fallback: get all timeframes
        if len(snapshots) < self.min_samples:
            snapshots = store.get_snapshots(
                symbol, timeframe,
                lookback_days=self.lookback_months * 30,
                with_outcomes_only=False,
                any_timeframe=True
            )
        
        if len(snapshots) < self.min_samples:
            return None
        
        # Convert to DataFrame
        df_hist = pd.DataFrame(snapshots)
        
        # Features to use
        feature_cols = [c for c in self.STOCK_FEATURE_WEIGHTS.keys() if c in df_hist.columns]
        
        if len(feature_cols) < 4:
            return None
        
        # Build feature matrix
        X = df_hist[feature_cols].fillna(50).values
        
        # Apply weights
        weights = np.array([self.STOCK_FEATURE_WEIGHTS.get(c, 1.0) for c in feature_cols])
        X_weighted = X * weights
        
        # Scale
        cache_key = f"{symbol}_{timeframe}_stock"
        if cache_key not in self.scalers:
            self.scalers[cache_key] = StandardScaler()
            self.scalers[cache_key].fit(X_weighted)
        
        X_scaled = self.scalers[cache_key].transform(X_weighted)
        
        # Fit KNN
        k = min(self.k_neighbors, len(X))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_scaled)
        
        # Current conditions vector
        current_vector = np.array([[current_conditions.get(c, 50) for c in feature_cols]])
        current_weighted = current_vector * weights
        current_scaled = self.scalers[cache_key].transform(current_weighted)
        
        # Find neighbors
        distances, indices = knn.kneighbors(current_scaled)
        
        # Build matches
        matches = []
        max_dist = distances[0].max() if len(distances[0]) > 0 else 1
        
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - (dist / max_dist) if max_dist > 0 else 1
            
            if similarity > 0.4:
                row = df_hist.iloc[idx]
                matches.append({
                    'timestamp': row.get('timestamp', ''),
                    'similarity': similarity,
                    'hit_tp1': bool(row.get('hit_tp1')),
                    'hit_sl': bool(row.get('hit_sl')),
                    'candles_to_result': row.get('candles_to_result', 0),
                    'max_favorable_pct': row.get('max_favorable_pct', 0),
                    'max_adverse_pct': row.get('max_adverse_pct', 0),
                })
        
        if len(matches) < self.min_samples:
            return None
        
        return self._compile_validation(
            matches, our_score, timeframe, "stock_db", len(snapshots)
        )
    
    def _validate_from_whale_db(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_conditions: Dict,
        our_score: int,
        tp_distance_pct: float,
        sl_distance_pct: float,
    ) -> Optional[HistoricalValidation]:
        """
        Validate using stored whale data snapshots.
        
        Key insight: Whale data patterns (OI, positioning, funding) are 
        MARKET-WIDE signals that apply across all coins. A bullish whale
        setup on BTC works similarly on ETH, SOL, etc.
        
        So we search ALL symbols for similar conditions, not just the
        current symbol being analyzed.
        """
        
        store = _get_whale_store()
        
        # First try: Get snapshots for specific symbol (if we have enough)
        snapshots = store.get_snapshots(
            symbol, timeframe,
            lookback_days=self.lookback_months * 30,
            with_outcomes_only=False,
            any_timeframe=True
        )
        
        # KEY FIX: If not enough data for this symbol, search ALL symbols!
        # Whale patterns are market-wide, so similar conditions on BTC
        # can validate a setup on TAOUSDT
        if len(snapshots) < self.min_samples:
            snapshots = store.get_all_snapshots(
                lookback_days=self.lookback_months * 30,
                max_records=5000  # Limit for performance
            )
        
        print(f"[DEBUG] Historical Validation: {len(snapshots)} snapshots loaded")
        
        if len(snapshots) < self.min_samples:
            print(f"[DEBUG] Not enough snapshots ({len(snapshots)} < {self.min_samples})")
            return None
        
        # Convert to DataFrame
        df_hist = pd.DataFrame(snapshots)
        
        print(f"[DEBUG] DataFrame columns: {list(df_hist.columns)}")
        
        # Features to use
        feature_cols = [c for c in self.WHALE_FEATURE_WEIGHTS.keys() if c in df_hist.columns]
        
        print(f"[DEBUG] Feature columns found: {feature_cols}")
        
        if len(feature_cols) < 4:
            print(f"[DEBUG] Not enough feature columns ({len(feature_cols)} < 4)")
            return None
        
        # Build feature matrix
        X = df_hist[feature_cols].fillna(50).values
        
        # Apply weights
        weights = np.array([self.WHALE_FEATURE_WEIGHTS.get(c, 1.0) for c in feature_cols])
        X_weighted = X * weights
        
        # Scale
        cache_key = f"{symbol}_{timeframe}_whale"
        if cache_key not in self.scalers:
            self.scalers[cache_key] = StandardScaler()
            self.scalers[cache_key].fit(X_weighted)
        
        X_scaled = self.scalers[cache_key].transform(X_weighted)
        
        # Fit KNN
        k = min(self.k_neighbors, len(X))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_scaled)
        
        # Current conditions vector
        current_vector = np.array([[current_conditions.get(c, 50) for c in feature_cols]])
        current_weighted = current_vector * weights
        current_scaled = self.scalers[cache_key].transform(current_weighted)
        
        # Find neighbors
        distances, indices = knn.kneighbors(current_scaled)
        
        print(f"[DEBUG] KNN found {len(indices[0])} neighbors, distances: min={distances[0].min():.2f}, max={distances[0].max():.2f}")
        
        # Build matches - USE ALL TOP K NEIGHBORS (similarity threshold was too strict)
        # KNN already finds the most similar patterns, so just use them all
        matches = []
        max_dist = distances[0].max() if len(distances[0]) > 0 else 1
        min_dist = distances[0].min() if len(distances[0]) > 0 else 0
        
        for dist, idx in zip(distances[0], indices[0]):
            # Normalize similarity to 0-1 range based on distance spread
            if max_dist > min_dist:
                similarity = 1 - ((dist - min_dist) / (max_dist - min_dist))
            else:
                similarity = 1.0
            
            row = df_hist.iloc[idx]
            
            # Handle None AND NaN values explicitly
            hit_tp1 = row.get('hit_tp1')
            hit_sl = row.get('hit_sl')
            candles = row.get('candles_to_result')
            max_fav = row.get('max_favorable_pct')
            max_adv = row.get('max_adverse_pct')
            
            # Safe conversion functions
            def safe_bool(val):
                if val is None:
                    return False
                if isinstance(val, float) and np.isnan(val):
                    return False
                return bool(val)
            
            def safe_int_val(val):
                if val is None:
                    return 0
                try:
                    if isinstance(val, float) and np.isnan(val):
                        return 0
                    return int(val)
                except:
                    return 0
            
            def safe_float_val(val):
                if val is None:
                    return 0.0
                try:
                    if isinstance(val, float) and np.isnan(val):
                        return 0.0
                    return float(val)
                except:
                    return 0.0
            
            matches.append({
                'timestamp': row.get('timestamp', ''),
                'similarity': max(0.5, similarity),  # Floor at 50% since KNN found them
                'hit_tp1': safe_bool(hit_tp1),
                'hit_sl': safe_bool(hit_sl),
                'candles_to_result': safe_int_val(candles),
                'max_favorable_pct': safe_float_val(max_fav),
                'max_adverse_pct': safe_float_val(max_adv),
            })
        
        print(f"[DEBUG] Using all {len(matches)} KNN neighbors as matches")
        
        if len(matches) < self.min_samples:
            print(f"[DEBUG] Not enough matches ({len(matches)} < {self.min_samples})")
            return None
        
        return self._compile_validation(
            matches, our_score, timeframe, "whale_db", len(snapshots)
        )
    
    def _validate_from_technical(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_conditions: Dict,
        our_score: int,
        tp_distance_pct: float,
        sl_distance_pct: float,
        df: pd.DataFrame,
    ) -> Optional[HistoricalValidation]:
        """Validate using technical indicators from raw OHLCV data"""
        
        # Calculate technical conditions for all historical points
        records = []
        
        for i in range(100, len(df) - 50):
            try:
                conditions = self._calc_technical_conditions(df, i)
                if conditions is None:
                    continue
                
                outcome = self._simulate_outcome(
                    df, i, direction, tp_distance_pct, sl_distance_pct
                )
                if outcome is None:
                    continue
                
                records.append({
                    'timestamp': df.index[i] if hasattr(df.index[i], 'isoformat') else str(df.index[i]),
                    **conditions,
                    **outcome
                })
            except:
                continue
        
        if len(records) < self.min_samples:
            return None
        
        df_hist = pd.DataFrame(records)
        
        # Features
        feature_cols = [c for c in self.TECHNICAL_FEATURE_WEIGHTS.keys() if c in df_hist.columns]
        
        if len(feature_cols) < 4:
            return None
        
        X = df_hist[feature_cols].fillna(50).values
        weights = np.array([self.TECHNICAL_FEATURE_WEIGHTS.get(c, 1.0) for c in feature_cols])
        X_weighted = X * weights
        
        # Scale
        cache_key = f"{symbol}_{timeframe}_tech"
        if cache_key not in self.scalers:
            self.scalers[cache_key] = StandardScaler()
            self.scalers[cache_key].fit(X_weighted)
        
        X_scaled = self.scalers[cache_key].transform(X_weighted)
        
        # KNN
        k = min(self.k_neighbors, len(X))
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        knn.fit(X_scaled)
        
        # Current vector (technical only)
        current_vector = np.array([[current_conditions.get(c, 50) for c in feature_cols]])
        current_weighted = current_vector * weights
        current_scaled = self.scalers[cache_key].transform(current_weighted)
        
        distances, indices = knn.kneighbors(current_scaled)
        
        matches = []
        max_dist = distances[0].max() if len(distances[0]) > 0 else 1
        
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - (dist / max_dist) if max_dist > 0 else 1
            
            if similarity > 0.4:
                row = df_hist.iloc[idx]
                matches.append({
                    'timestamp': row.get('timestamp', ''),
                    'similarity': similarity,
                    'hit_tp1': row.get('hit_tp1', False),
                    'hit_sl': row.get('hit_sl', False),
                    'candles_to_result': row.get('candles_to_result', 0),
                    'max_favorable_pct': row.get('max_favorable_pct', 0),
                    'max_adverse_pct': row.get('max_adverse_pct', 0),
                })
        
        if len(matches) < self.min_samples:
            return None
        
        return self._compile_validation(
            matches, our_score, timeframe, "technical_only", len(records)
        )
    
    def _calc_technical_conditions(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Calculate technical conditions at a specific index"""
        try:
            window = df.iloc[max(0, idx-50):idx+1]
            if len(window) < 50:
                return None
            
            close = window['Close']
            high = window['High']
            low = window['Low']
            volume = window['Volume']
            
            current_price = close.iloc[-1]
            
            # EMAs
            ema_9 = close.ewm(span=9).mean().iloc[-1]
            ema_20 = close.ewm(span=20).mean().iloc[-1]
            ema_50 = close.ewm(span=50).mean().iloc[-1]
            
            # EMA alignment (1 = bullish stack, -1 = bearish, 0 = mixed)
            if ema_9 > ema_20 > ema_50:
                ema_alignment = 1
            elif ema_9 < ema_20 < ema_50:
                ema_alignment = -1
            else:
                ema_alignment = 0
            
            # Position in range
            swing_high = high.rolling(20).max().iloc[-1]
            swing_low = low.rolling(20).min().iloc[-1]
            
            if swing_high > swing_low:
                position_in_range = ((current_price - swing_low) / (swing_high - swing_low)) * 100
            else:
                position_in_range = 50
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / loss_val.iloc[-1] if loss_val.iloc[-1] != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
            
            # MFI
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi = 100 - (100 / (1 + positive_flow.iloc[-1] / negative_flow.iloc[-1])) if negative_flow.iloc[-1] != 0 else 50
            
            # CMF
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
            mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
            mf_volume = mf_multiplier * volume
            vol_sum = volume.rolling(20).sum().iloc[-1]
            cmf = mf_volume.rolling(20).sum().iloc[-1] / vol_sum if vol_sum != 0 else 0
            
            # Volume ratio
            vol_avg = volume.rolling(20).mean().iloc[-1]
            volume_ratio = volume.iloc[-1] / vol_avg if vol_avg != 0 else 1
            
            # ATR %
            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = (atr / current_price) * 100 if current_price > 0 else 2
            
            # Price vs EMA20 %
            price_vs_ema20 = ((current_price - ema_20) / ema_20) * 100 if ema_20 > 0 else 0
            
            return {
                'position_in_range': position_in_range,
                'mfi': mfi,
                'cmf': cmf * 100,  # Scale to similar range as others
                'rsi': rsi,
                'volume_ratio': volume_ratio * 50,  # Scale
                'atr_pct': atr_pct * 10,  # Scale
                'price_vs_ema20': price_vs_ema20 + 50,  # Center around 50
                'ema_alignment': (ema_alignment + 1) * 50,  # 0, 50, or 100
            }
            
        except Exception as e:
            return None
    
    def _simulate_outcome(
        self,
        df: pd.DataFrame,
        start_idx: int,
        direction: str,
        tp_pct: float,
        sl_pct: float,
        max_candles: int = 100
    ) -> Optional[Dict]:
        """Simulate outcome from a historical point"""
        try:
            entry_price = df.iloc[start_idx]['Close']
            
            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
            
            max_favorable = 0
            max_adverse = 0
            
            for i in range(1, min(max_candles, len(df) - start_idx)):
                candle = df.iloc[start_idx + i]
                
                if direction == 'LONG':
                    favorable = (candle['High'] - entry_price) / entry_price * 100
                    adverse = (entry_price - candle['Low']) / entry_price * 100
                    
                    max_favorable = max(max_favorable, favorable)
                    max_adverse = max(max_adverse, adverse)
                    
                    if candle['High'] >= tp_price:
                        return {
                            'hit_tp1': True, 'hit_sl': False,
                            'candles_to_result': i,
                            'max_favorable_pct': max_favorable,
                            'max_adverse_pct': max_adverse
                        }
                    if candle['Low'] <= sl_price:
                        return {
                            'hit_tp1': False, 'hit_sl': True,
                            'candles_to_result': i,
                            'max_favorable_pct': max_favorable,
                            'max_adverse_pct': max_adverse
                        }
                else:
                    favorable = (entry_price - candle['Low']) / entry_price * 100
                    adverse = (candle['High'] - entry_price) / entry_price * 100
                    
                    max_favorable = max(max_favorable, favorable)
                    max_adverse = max(max_adverse, adverse)
                    
                    if candle['Low'] <= tp_price:
                        return {
                            'hit_tp1': True, 'hit_sl': False,
                            'candles_to_result': i,
                            'max_favorable_pct': max_favorable,
                            'max_adverse_pct': max_adverse
                        }
                    if candle['High'] >= sl_price:
                        return {
                            'hit_tp1': False, 'hit_sl': True,
                            'candles_to_result': i,
                            'max_favorable_pct': max_favorable,
                            'max_adverse_pct': max_adverse
                        }
            
            return {
                'hit_tp1': False, 'hit_sl': False,
                'candles_to_result': max_candles,
                'max_favorable_pct': max_favorable,
                'max_adverse_pct': max_adverse
            }
            
        except:
            return None
    
    def _compile_validation(
        self,
        matches: List[Dict],
        our_score: int,
        timeframe: str,
        data_source: str,
        total_samples: int
    ) -> HistoricalValidation:
        """Compile validation results from matches"""
        
        n = len(matches)
        tf_minutes = self._tf_to_minutes(timeframe)
        
        # Win/Loss stats
        wins = sum(1 for m in matches if m.get('hit_tp1'))
        losses = sum(1 for m in matches if m.get('hit_sl'))
        timeouts = n - wins - losses
        
        print(f"[DEBUG] Compile: {n} matches, wins={wins}, losses={losses}, timeouts={timeouts}")
        
        # If no outcomes tracked, still show validation with 50% default
        if wins == 0 and losses == 0:
            print("[DEBUG] No outcomes tracked! Run: python backtest_outcomes.py")
            # Use 50% as default when no outcomes
            win_rate = 50.0
            loss_rate = 25.0
            timeout_rate = 25.0
            message = f"{n} similar patterns found. Run backtest_outcomes.py for win/loss data."
        else:
            win_rate = (wins / n * 100) if n > 0 else 0
            loss_rate = (losses / n * 100) if n > 0 else 0
            timeout_rate = (timeouts / n * 100) if n > 0 else 0
            message = f"Based on {n} similar whale data patterns" if data_source == "whale_db" else f"Based on {n} similar technical patterns"
        
        # Timing
        winners = [m for m in matches if m.get('hit_tp1')]
        losers = [m for m in matches if m.get('hit_sl')]
        
        avg_candles_tp1 = float(np.mean([m['candles_to_result'] for m in winners])) if winners else 0.0
        avg_candles_sl = float(np.mean([m['candles_to_result'] for m in losers])) if losers else 0.0
        
        # Handle NaN
        if np.isnan(avg_candles_tp1):
            avg_candles_tp1 = 0.0
        if np.isnan(avg_candles_sl):
            avg_candles_sl = 0.0
        
        avg_time_tp1 = self._candles_to_time(avg_candles_tp1, tf_minutes)
        avg_time_sl = self._candles_to_time(avg_candles_sl, tf_minutes)
        
        # Risk metrics - handle None values and NaN
        favorable_values = [m.get('max_favorable_pct', 0) for m in matches if m.get('max_favorable_pct') is not None]
        adverse_values = [m.get('max_adverse_pct', 0) for m in matches if m.get('max_adverse_pct') is not None]
        
        avg_max_favorable = float(np.mean(favorable_values)) if favorable_values else 0.0
        avg_max_adverse = float(np.mean(adverse_values)) if adverse_values else 0.0
        
        # Handle NaN
        if np.isnan(avg_max_favorable):
            avg_max_favorable = 0.0
        if np.isnan(avg_max_adverse):
            avg_max_adverse = 0.0
        
        # Similarity
        similarity_values = [m.get('similarity', 0.5) for m in matches if m.get('similarity') is not None]
        avg_similarity = float(np.mean(similarity_values) * 100) if similarity_values else 50.0
        
        # Handle NaN
        if np.isnan(avg_similarity):
            avg_similarity = 50.0
        
        # Confidence
        if n >= 50:
            sample_confidence = "HIGH"
        elif n >= 30:
            sample_confidence = "MEDIUM"
        else:
            sample_confidence = "LOW"
        
        # Grade
        if win_rate >= 75:
            history_grade = "A+"
        elif win_rate >= 65:
            history_grade = "A"
        elif win_rate >= 55:
            history_grade = "B"
        elif win_rate >= 45:
            history_grade = "C"
        elif win_rate >= 35:
            history_grade = "D"
        else:
            history_grade = "F"
        
        # Handle NaN before int conversion
        if np.isnan(win_rate):
            win_rate = 50.0
        historical_score = int(win_rate)
        
        # Handle NaN in our_score
        if our_score is None or (isinstance(our_score, float) and np.isnan(our_score)):
            our_score = 50
        
        # Alignment
        score_diff = our_score - historical_score
        if abs(score_diff) <= 15:
            alignment = "ALIGNED"
        elif score_diff > 15:
            alignment = "OVER_CONFIDENT"
        else:
            alignment = "UNDER_CONFIDENT"
        
        # Top matches
        sorted_matches = sorted(matches, key=lambda x: x.get('similarity', 0), reverse=True)[:5]
        top_matches = [
            HistoricalMatch(
                date=str(m.get('timestamp', ''))[:10],
                similarity=float(m.get('similarity', 0) or 0) * 100,
                result='WIN' if m.get('hit_tp1') else ('LOSS' if m.get('hit_sl') else 'PENDING'),
                candles=int(m.get('candles_to_result', 0) or 0),
                time_str=self._candles_to_time(m.get('candles_to_result', 0) or 0, tf_minutes),
            )
            for m in sorted_matches
        ]
        
        # Message already set above based on outcomes tracking
        
        # Helper to safely convert to int
        def safe_int(val, default=0):
            if val is None:
                return default
            try:
                if isinstance(val, float) and np.isnan(val):
                    return default
                return int(val)
            except (ValueError, TypeError):
                return default
        
        # Helper to safely convert to float
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                if isinstance(val, float) and np.isnan(val):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # Ensure all values are proper types (not None or NaN)
        return HistoricalValidation(
            matches_found=safe_int(n, 0),
            lookback_period=f"{self.lookback_months} months",
            avg_similarity=safe_float(avg_similarity, 50.0),
            data_source=str(data_source),
            win_rate=safe_float(win_rate, 50.0),
            loss_rate=safe_float(loss_rate, 25.0),
            timeout_rate=safe_float(timeout_rate, 25.0),
            avg_candles_to_tp1=safe_float(avg_candles_tp1, 0.0),
            avg_candles_to_sl=safe_float(avg_candles_sl, 0.0),
            avg_time_to_tp1=str(avg_time_tp1) if avg_time_tp1 else "N/A",
            avg_time_to_sl=str(avg_time_sl) if avg_time_sl else "N/A",
            avg_max_favorable=safe_float(avg_max_favorable, 0.0),
            avg_max_adverse=safe_float(avg_max_adverse, 0.0),
            sample_confidence=str(sample_confidence),
            history_grade=str(history_grade),
            alignment=str(alignment),
            our_score=safe_int(our_score, 50),
            historical_score=safe_int(historical_score, 50),
            top_matches=top_matches if top_matches else [],
            message=str(message),
        )
    
    def _tf_to_minutes(self, tf: str) -> int:
        """Convert timeframe string to minutes"""
        mapping = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360,
            '12h': 720, '1d': 1440, '1w': 10080
        }
        return mapping.get(tf, 15)
    
    def _candles_to_time(self, candles: float, tf_minutes: int) -> str:
        """Convert candles to human readable time"""
        if candles == 0:
            return "N/A"
        
        total_minutes = candles * tf_minutes
        
        if total_minutes < 60:
            return f"~{total_minutes:.0f} min"
        elif total_minutes < 1440:
            return f"~{total_minutes/60:.1f} hours"
        else:
            return f"~{total_minutes/1440:.1f} days"


# Global instance
_validator: Optional[HistoricalValidator] = None


def get_validator() -> HistoricalValidator:
    """Get or create global validator instance"""
    global _validator
    if _validator is None:
        _validator = HistoricalValidator()
    return _validator


def validate_signal_quick(
    symbol: str,
    timeframe: str,
    direction: str,
    current_conditions: Dict,
    our_score: int,
    tp_distance_pct: float,
    sl_distance_pct: float,
    df: pd.DataFrame = None,
    market_type: str = 'crypto',  # 'crypto' or 'stock'
) -> Optional[HistoricalValidation]:
    """
    Quick validation function for use in app.py
    
    Args:
        market_type: 'crypto' uses whale data, 'stock' uses Quiver institutional data
    """
    validator = get_validator()
    return validator.validate_signal(
        symbol, timeframe, direction,
        current_conditions, our_score,
        tp_distance_pct, sl_distance_pct, df,
        market_type=market_type
    )
