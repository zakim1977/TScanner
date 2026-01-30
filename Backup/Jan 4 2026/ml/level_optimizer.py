"""
Level Optimizer ML Module
=========================
Trains ML models to predict optimal SL/TP levels based on historical outcomes.

STAGE 2 of the trading pipeline:
- Stage 1 (existing): Direction, Score, Setup detection
- Stage 2 (this): Optimal SL/TP/R:R based on conditions

Uses historical whale data + market data to learn:
1. Optimal SL distance (avoid stop hunts)
2. TP1/TP2/TP3 probability (realistic targets)
3. Expected R:R for given conditions

Training Data Source:
- whale_data_store.py historical snapshots
- Binance historical candles for outcome measurement
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ ML packages not available. Install xgboost, sklearn.")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SetupFeatures:
    """Features extracted from a trading setup moment"""
    # Whale/Institutional
    whale_pct: float
    retail_pct: float
    whale_retail_diff: float
    oi_change_24h: float
    funding_rate: float
    
    # Technical
    rsi: float
    mfi: float
    position_in_range: float  # 0-100, where in the range
    atr_pct: float  # ATR as % of price
    volume_ratio: float
    
    # Price context
    price_change_24h: float
    
    # Structure (simplified)
    is_uptrend: int  # 1 = uptrend, 0 = range, -1 = downtrend
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML"""
        return np.array([
            self.whale_pct,
            self.retail_pct,
            self.whale_retail_diff,
            self.oi_change_24h,
            self.funding_rate,
            self.rsi,
            self.mfi,
            self.position_in_range,
            self.atr_pct,
            self.volume_ratio,
            self.price_change_24h,
            self.is_uptrend
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        return [
            'whale_pct', 'retail_pct', 'whale_retail_diff',
            'oi_change_24h', 'funding_rate',
            'rsi', 'mfi', 'position_in_range', 'atr_pct', 'volume_ratio',
            'price_change_24h', 'is_uptrend'
        ]


@dataclass
class SetupOutcome:
    """What happened after a setup was detected"""
    # Max excursions (in %)
    max_favorable_pct: float  # Best profit achieved
    max_adverse_pct: float  # Worst drawdown before profit
    
    # TP hit detection (based on common targets)
    hit_1pct: bool  # Hit +1%
    hit_2pct: bool  # Hit +2%
    hit_3pct: bool  # Hit +3%
    hit_5pct: bool  # Hit +5%
    
    # SL survival (could you survive with this SL?)
    survived_1pct_sl: bool  # Never dropped below -1%
    survived_1_5pct_sl: bool
    survived_2pct_sl: bool
    survived_3pct_sl: bool
    
    # Timing
    candles_to_max: int  # Candles to reach max favorable
    
    # Optimal levels (calculated)
    optimal_sl_pct: float  # SL that would survive
    optimal_tp1_pct: float  # TP that would hit with 80%+ probability
    achieved_rr: float  # max_favorable / max_adverse (if adverse > 0)


@dataclass 
class TrainingSample:
    """Complete training sample with features and outcomes"""
    symbol: str
    timestamp: str
    direction: str  # LONG or SHORT
    features: SetupFeatures
    outcome: SetupOutcome


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

class HistoricalSetupScanner:
    """
    Scans historical whale data to find high-conviction setups
    and measures what happened after each one.
    """
    
    def __init__(self, whale_db_path: str = "data/whale_history.db"):
        self.whale_db_path = whale_db_path
        self.training_samples: List[TrainingSample] = []
    
    def scan_for_setups(
        self,
        min_whale_pct: float = 65,
        min_whale_retail_diff: float = 15,
        lookback_days: int = 180
    ) -> List[Dict]:
        """
        Find historical moments where setup conditions were met.
        
        Args:
            min_whale_pct: Minimum whale long % for bullish setup
            min_whale_retail_diff: Minimum whale-retail gap
            lookback_days: How far back to look
        
        Returns:
            List of setup snapshots
        """
        setups = []
        
        try:
            conn = sqlite3.connect(self.whale_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            
            # Find BULLISH setups (whale long >= threshold)
            cursor.execute("""
                SELECT * FROM whale_snapshots
                WHERE timestamp >= ?
                AND whale_long_pct >= ?
                AND (whale_long_pct - retail_long_pct) >= ?
                ORDER BY timestamp ASC
            """, (cutoff, min_whale_pct, min_whale_retail_diff))
            
            bullish_setups = cursor.fetchall()
            for row in bullish_setups:
                setup = dict(row)
                setup['setup_direction'] = 'LONG'
                setups.append(setup)
            
            # Find BEARISH setups (whale short >= threshold)
            cursor.execute("""
                SELECT * FROM whale_snapshots
                WHERE timestamp >= ?
                AND (100 - whale_long_pct) >= ?
                AND (retail_long_pct - whale_long_pct) >= ?
                ORDER BY timestamp ASC
            """, (cutoff, min_whale_pct, min_whale_retail_diff))
            
            bearish_setups = cursor.fetchall()
            for row in bearish_setups:
                setup = dict(row)
                setup['setup_direction'] = 'SHORT'
                setups.append(setup)
            
            conn.close()
            
            print(f"[LevelOptimizer] Found {len(setups)} historical setups ({len(bullish_setups)} LONG, {len(bearish_setups)} SHORT)")
            return setups
            
        except Exception as e:
            print(f"[LevelOptimizer] Error scanning setups: {e}")
            return []
    
    def measure_outcome(
        self,
        setup: Dict,
        candles_df: pd.DataFrame,
        lookforward_candles: int = 100
    ) -> Optional[SetupOutcome]:
        """
        Measure what happened after a setup was detected.
        
        Args:
            setup: Setup snapshot from whale_data
            candles_df: DataFrame with OHLCV data
            lookforward_candles: How many candles to look forward
        
        Returns:
            SetupOutcome with measured results
        """
        try:
            setup_time = datetime.fromisoformat(setup['timestamp'].replace('Z', '+00:00'))
            setup_price = setup['price']
            direction = setup.get('setup_direction', 'LONG')
            
            # Find candles after setup time
            if 'DateTime' in candles_df.columns:
                candles_df['DateTime'] = pd.to_datetime(candles_df['DateTime'])
                future_candles = candles_df[candles_df['DateTime'] >= setup_time].head(lookforward_candles)
            else:
                # Assume index is datetime
                future_candles = candles_df[candles_df.index >= setup_time].head(lookforward_candles)
            
            if len(future_candles) < 10:
                return None  # Not enough data
            
            # Calculate price changes from setup price
            if direction == 'LONG':
                # For LONG: favorable = price going UP, adverse = price going DOWN
                highs = future_candles['High'].values
                lows = future_candles['Low'].values
                
                max_favorable_pct = ((highs.max() - setup_price) / setup_price) * 100
                max_adverse_pct = ((setup_price - lows.min()) / setup_price) * 100
                
                # Find candle where max was reached
                candles_to_max = np.argmax(highs) + 1
                
            else:  # SHORT
                # For SHORT: favorable = price going DOWN, adverse = price going UP
                highs = future_candles['High'].values
                lows = future_candles['Low'].values
                
                max_favorable_pct = ((setup_price - lows.min()) / setup_price) * 100
                max_adverse_pct = ((highs.max() - setup_price) / setup_price) * 100
                
                candles_to_max = np.argmin(lows) + 1
            
            # TP hit detection (from setup price)
            hit_1pct = max_favorable_pct >= 1.0
            hit_2pct = max_favorable_pct >= 2.0
            hit_3pct = max_favorable_pct >= 3.0
            hit_5pct = max_favorable_pct >= 5.0
            
            # SL survival detection
            survived_1pct_sl = max_adverse_pct < 1.0
            survived_1_5pct_sl = max_adverse_pct < 1.5
            survived_2pct_sl = max_adverse_pct < 2.0
            survived_3pct_sl = max_adverse_pct < 3.0
            
            # Calculate optimal levels
            # Optimal SL = max_adverse + small buffer (wouldn't get hit)
            optimal_sl_pct = max_adverse_pct + 0.3  # Add 0.3% buffer
            
            # Optimal TP1 = conservative target that would hit
            if max_favorable_pct >= 2.0:
                optimal_tp1_pct = min(max_favorable_pct * 0.6, 3.0)  # 60% of max, capped at 3%
            else:
                optimal_tp1_pct = max_favorable_pct * 0.7  # 70% of max for smaller moves
            
            # Achieved R:R
            achieved_rr = max_favorable_pct / max_adverse_pct if max_adverse_pct > 0.1 else max_favorable_pct / 0.1
            
            return SetupOutcome(
                max_favorable_pct=max_favorable_pct,
                max_adverse_pct=max_adverse_pct,
                hit_1pct=hit_1pct,
                hit_2pct=hit_2pct,
                hit_3pct=hit_3pct,
                hit_5pct=hit_5pct,
                survived_1pct_sl=survived_1pct_sl,
                survived_1_5pct_sl=survived_1_5pct_sl,
                survived_2pct_sl=survived_2pct_sl,
                survived_3pct_sl=survived_3pct_sl,
                candles_to_max=candles_to_max,
                optimal_sl_pct=optimal_sl_pct,
                optimal_tp1_pct=optimal_tp1_pct,
                achieved_rr=achieved_rr
            )
            
        except Exception as e:
            print(f"[LevelOptimizer] Error measuring outcome: {e}")
            return None
    
    def extract_features(self, setup: Dict) -> SetupFeatures:
        """Extract ML features from a setup snapshot"""
        
        whale_pct = setup.get('whale_long_pct', 50)
        retail_pct = setup.get('retail_long_pct', 50)
        
        # Determine trend from position in range
        position = setup.get('position_in_range', 50)
        if position < 30:
            is_uptrend = -1  # Near bottom = likely downtrend
        elif position > 70:
            is_uptrend = 1  # Near top = likely uptrend
        else:
            is_uptrend = 0  # Middle = range
        
        return SetupFeatures(
            whale_pct=whale_pct,
            retail_pct=retail_pct,
            whale_retail_diff=whale_pct - retail_pct,
            oi_change_24h=setup.get('oi_change_24h', 0) or 0,
            funding_rate=setup.get('funding_rate', 0) or 0,
            rsi=setup.get('rsi', 50) or 50,
            mfi=setup.get('mfi', 50) or 50,
            position_in_range=position,
            atr_pct=setup.get('atr_pct', 2) or 2,
            volume_ratio=setup.get('volume_ratio', 1) or 1,
            price_change_24h=setup.get('price_change_24h', 0) or 0,
            is_uptrend=is_uptrend
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingDataGenerator:
    """
    Generates ML training data from historical setups and candles.
    """
    
    def __init__(self):
        self.scanner = HistoricalSetupScanner()
        self.samples: List[TrainingSample] = []
    
    def generate_from_whale_history(
        self,
        symbols: List[str] = None,
        min_whale_pct: float = 65,
        lookback_days: int = 90,
        fetch_candles_fn=None  # Function to fetch historical candles
    ) -> pd.DataFrame:
        """
        Generate training data from whale history database.
        
        Args:
            symbols: List of symbols to process (None = all)
            min_whale_pct: Minimum whale % for setup detection
            lookback_days: Days of history to scan
            fetch_candles_fn: Function(symbol, timeframe, start_date, end_date) -> DataFrame
        
        Returns:
            DataFrame with features and outcomes
        """
        # Find all setups
        setups = self.scanner.scan_for_setups(
            min_whale_pct=min_whale_pct,
            lookback_days=lookback_days
        )
        
        if not setups:
            print("[LevelOptimizer] No setups found in history")
            return pd.DataFrame()
        
        # Filter by symbols if specified
        if symbols:
            setups = [s for s in setups if s['symbol'] in symbols]
        
        print(f"[LevelOptimizer] Processing {len(setups)} setups...")
        
        training_data = []
        
        for i, setup in enumerate(setups):
            try:
                symbol = setup['symbol']
                timeframe = setup.get('timeframe', '15m')
                
                # Fetch candles if function provided
                if fetch_candles_fn:
                    setup_time = datetime.fromisoformat(setup['timestamp'].replace('Z', '+00:00'))
                    candles_df = fetch_candles_fn(
                        symbol, 
                        timeframe,
                        setup_time,
                        setup_time + timedelta(days=7)  # Get 7 days after setup
                    )
                else:
                    # Skip if no candle fetcher
                    continue
                
                if candles_df is None or len(candles_df) < 20:
                    continue
                
                # Measure outcome
                outcome = self.scanner.measure_outcome(setup, candles_df)
                if outcome is None:
                    continue
                
                # Extract features
                features = self.scanner.extract_features(setup)
                
                # Create training row
                row = {
                    'symbol': symbol,
                    'timestamp': setup['timestamp'],
                    'direction': setup.get('setup_direction', 'LONG'),
                    
                    # Features
                    **asdict(features),
                    
                    # Outcomes (targets)
                    'max_favorable_pct': outcome.max_favorable_pct,
                    'max_adverse_pct': outcome.max_adverse_pct,
                    'optimal_sl_pct': outcome.optimal_sl_pct,
                    'optimal_tp1_pct': outcome.optimal_tp1_pct,
                    'achieved_rr': outcome.achieved_rr,
                    'hit_2pct': int(outcome.hit_2pct),
                    'hit_3pct': int(outcome.hit_3pct),
                    'survived_1_5pct_sl': int(outcome.survived_1_5pct_sl),
                    'survived_2pct_sl': int(outcome.survived_2pct_sl),
                    'candles_to_max': outcome.candles_to_max,
                }
                
                training_data.append(row)
                
                if (i + 1) % 50 == 0:
                    print(f"[LevelOptimizer] Processed {i + 1}/{len(setups)} setups")
                    
            except Exception as e:
                print(f"[LevelOptimizer] Error processing setup: {e}")
                continue
        
        df = pd.DataFrame(training_data)
        print(f"[LevelOptimizer] Generated {len(df)} training samples")
        
        return df
    
    def generate_from_candles_only(
        self,
        candles_df: pd.DataFrame,
        whale_data_fn=None,  # Function to simulate/get whale data
        lookforward_candles: int = 50
    ) -> pd.DataFrame:
        """
        Alternative: Generate training data from candles when whale history is limited.
        Simulates setups based on technical conditions.
        
        For each candle, we:
        1. Calculate technical features
        2. Look forward to measure what happened
        3. Create training sample
        
        Args:
            candles_df: Historical OHLCV data
            whale_data_fn: Optional function to get/simulate whale data
            lookforward_candles: Candles to look forward for outcome
        
        Returns:
            DataFrame with features and outcomes
        """
        if len(candles_df) < lookforward_candles + 50:
            print("[LevelOptimizer] Not enough candles for training")
            return pd.DataFrame()
        
        training_data = []
        
        # Calculate indicators once
        candles_df = self._add_indicators(candles_df)
        
        # Scan through candles (leave room for lookforward)
        for i in range(50, len(candles_df) - lookforward_candles):
            try:
                row = candles_df.iloc[i]
                future = candles_df.iloc[i:i + lookforward_candles]
                
                setup_price = row['Close']
                
                # Measure outcomes
                max_up = ((future['High'].max() - setup_price) / setup_price) * 100
                max_down = ((setup_price - future['Low'].min()) / setup_price) * 100
                
                # Skip flat periods (no movement)
                if max_up < 0.5 and max_down < 0.5:
                    continue
                
                # Determine direction based on which move was bigger
                if max_up > max_down:
                    direction = 'LONG'
                    max_favorable = max_up
                    max_adverse = max_down
                else:
                    direction = 'SHORT'
                    max_favorable = max_down
                    max_adverse = max_up
                
                # Extract features
                sample = {
                    'timestamp': row.get('DateTime', row.name),
                    'direction': direction,
                    
                    # Technical features
                    'whale_pct': 50 + (row.get('RSI', 50) - 50) * 0.3,  # Simulated
                    'retail_pct': 50,  # Simulated
                    'whale_retail_diff': (row.get('RSI', 50) - 50) * 0.3,
                    'oi_change_24h': 0,  # Unknown
                    'funding_rate': 0,  # Unknown
                    'rsi': row.get('RSI', 50),
                    'mfi': row.get('MFI', 50),
                    'position_in_range': row.get('Position', 50),
                    'atr_pct': row.get('ATR_pct', 2),
                    'volume_ratio': row.get('Volume_Ratio', 1),
                    'price_change_24h': 0,
                    'is_uptrend': 1 if row.get('RSI', 50) > 50 else -1,
                    
                    # Outcomes
                    'max_favorable_pct': max_favorable,
                    'max_adverse_pct': max_adverse,
                    'optimal_sl_pct': max_adverse + 0.3,
                    'optimal_tp1_pct': max_favorable * 0.6,
                    'achieved_rr': max_favorable / max(max_adverse, 0.1),
                    'hit_2pct': int(max_favorable >= 2.0),
                    'hit_3pct': int(max_favorable >= 3.0),
                    'survived_1_5pct_sl': int(max_adverse < 1.5),
                    'survived_2pct_sl': int(max_adverse < 2.0),
                    'candles_to_max': 25,  # Approximation
                }
                
                training_data.append(sample)
                
            except Exception as e:
                continue
        
        df = pd.DataFrame(training_data)
        print(f"[LevelOptimizer] Generated {len(df)} samples from candles")
        
        return df
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame"""
        df = df.copy()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MFI (simplified)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))
        
        # ATR %
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low'] - df['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
        
        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Position in range
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        df['Position'] = ((df['Close'] - low_20) / (high_20 - low_20)) * 100
        
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class LevelOptimizerModels:
    """
    XGBoost models for predicting optimal SL/TP levels.
    """
    
    def __init__(self, models_dir: str = "ml/models/level_optimizer"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.sl_model = None  # Predicts optimal SL %
        self.tp1_model = None  # Predicts optimal TP1 %
        self.tp1_prob_model = None  # Predicts probability of hitting 2%+
        self.rr_model = None  # Predicts expected R:R
        
        self.scaler = None
        self.feature_names = SetupFeatures.feature_names()
        
        self.is_trained = False
    
    def train(self, training_df: pd.DataFrame) -> Dict:
        """
        Train all models on the training data.
        
        Args:
            training_df: DataFrame from TrainingDataGenerator
        
        Returns:
            Dict with training metrics
        """
        if not HAS_ML:
            return {'error': 'ML packages not available'}
        
        if len(training_df) < 50:
            return {'error': f'Need at least 50 samples, got {len(training_df)}'}
        
        print(f"[LevelOptimizer] Training on {len(training_df)} samples...")
        
        # Prepare features
        X = training_df[self.feature_names].values
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Targets
        y_sl = training_df['optimal_sl_pct'].values
        y_tp1 = training_df['optimal_tp1_pct'].values
        y_tp1_prob = training_df['hit_2pct'].values  # Binary: did it hit 2%?
        y_rr = training_df['achieved_rr'].values
        
        # Clip extreme values
        y_sl = np.clip(y_sl, 0.5, 5.0)
        y_tp1 = np.clip(y_tp1, 0.5, 10.0)
        y_rr = np.clip(y_rr, 0.1, 10.0)
        
        # Split data
        X_train, X_test, y_sl_train, y_sl_test = train_test_split(
            X_scaled, y_sl, test_size=0.2, random_state=42
        )
        _, _, y_tp1_train, y_tp1_test = train_test_split(
            X_scaled, y_tp1, test_size=0.2, random_state=42
        )
        _, _, y_prob_train, y_prob_test = train_test_split(
            X_scaled, y_tp1_prob, test_size=0.2, random_state=42
        )
        _, _, y_rr_train, y_rr_test = train_test_split(
            X_scaled, y_rr, test_size=0.2, random_state=42
        )
        
        metrics = {}
        
        # Train SL model
        print("[LevelOptimizer] Training SL model...")
        self.sl_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.sl_model.fit(X_train, y_sl_train)
        sl_pred = self.sl_model.predict(X_test)
        metrics['sl_mae'] = mean_absolute_error(y_sl_test, sl_pred)
        metrics['sl_r2'] = r2_score(y_sl_test, sl_pred)
        print(f"  SL MAE: {metrics['sl_mae']:.3f}%, R²: {metrics['sl_r2']:.3f}")
        
        # Train TP1 model
        print("[LevelOptimizer] Training TP1 model...")
        self.tp1_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.tp1_model.fit(X_train, y_tp1_train)
        tp1_pred = self.tp1_model.predict(X_test)
        metrics['tp1_mae'] = mean_absolute_error(y_tp1_test, tp1_pred)
        metrics['tp1_r2'] = r2_score(y_tp1_test, tp1_pred)
        print(f"  TP1 MAE: {metrics['tp1_mae']:.3f}%, R²: {metrics['tp1_r2']:.3f}")
        
        # Train TP1 probability model
        print("[LevelOptimizer] Training TP probability model...")
        self.tp1_prob_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.tp1_prob_model.fit(X_train, y_prob_train)
        prob_pred = self.tp1_prob_model.predict(X_test)
        metrics['tp_prob_accuracy'] = accuracy_score(y_prob_test, prob_pred)
        print(f"  TP Prob Accuracy: {metrics['tp_prob_accuracy']:.3f}")
        
        # Train R:R model
        print("[LevelOptimizer] Training R:R model...")
        self.rr_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.rr_model.fit(X_train, y_rr_train)
        rr_pred = self.rr_model.predict(X_test)
        metrics['rr_mae'] = mean_absolute_error(y_rr_test, rr_pred)
        metrics['rr_r2'] = r2_score(y_rr_test, rr_pred)
        print(f"  R:R MAE: {metrics['rr_mae']:.3f}, R²: {metrics['rr_r2']:.3f}")
        
        # Feature importance
        metrics['feature_importance'] = dict(zip(
            self.feature_names,
            self.sl_model.feature_importances_
        ))
        
        self.is_trained = True
        metrics['samples_trained'] = len(training_df)
        metrics['trained_at'] = datetime.now().isoformat()
        
        # Save models
        self.save()
        
        return metrics
    
    def predict(self, features: SetupFeatures) -> Dict:
        """
        Predict optimal levels for a setup.
        
        Args:
            features: SetupFeatures object
        
        Returns:
            Dict with predictions
        """
        if not self.is_trained:
            return self._default_predictions()
        
        try:
            X = features.to_array().reshape(1, -1)
            X = np.nan_to_num(X, nan=0, posinf=100, neginf=-100)
            X_scaled = self.scaler.transform(X)
            
            return {
                'optimal_sl_pct': float(self.sl_model.predict(X_scaled)[0]),
                'optimal_tp1_pct': float(self.tp1_model.predict(X_scaled)[0]),
                'tp1_probability': float(self.tp1_prob_model.predict_proba(X_scaled)[0][1]),
                'expected_rr': float(self.rr_model.predict(X_scaled)[0]),
                'source': 'ml'
            }
            
        except Exception as e:
            print(f"[LevelOptimizer] Prediction error: {e}")
            return self._default_predictions()
    
    def _default_predictions(self) -> Dict:
        """Default predictions when ML is not available"""
        return {
            'optimal_sl_pct': 1.5,
            'optimal_tp1_pct': 2.5,
            'tp1_probability': 0.65,
            'expected_rr': 1.67,
            'source': 'default'
        }
    
    def save(self):
        """Save all models to disk"""
        try:
            if self.sl_model:
                self.sl_model.save_model(os.path.join(self.models_dir, 'sl_model.json'))
            if self.tp1_model:
                self.tp1_model.save_model(os.path.join(self.models_dir, 'tp1_model.json'))
            if self.tp1_prob_model:
                self.tp1_prob_model.save_model(os.path.join(self.models_dir, 'tp1_prob_model.json'))
            if self.rr_model:
                self.rr_model.save_model(os.path.join(self.models_dir, 'rr_model.json'))
            if self.scaler:
                with open(os.path.join(self.models_dir, 'scaler.pkl'), 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata = {
                'trained': self.is_trained,
                'feature_names': self.feature_names,
                'saved_at': datetime.now().isoformat()
            }
            with open(os.path.join(self.models_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            print(f"[LevelOptimizer] Models saved to {self.models_dir}")
            
        except Exception as e:
            print(f"[LevelOptimizer] Error saving models: {e}")
    
    def load(self) -> bool:
        """Load models from disk"""
        try:
            sl_path = os.path.join(self.models_dir, 'sl_model.json')
            if not os.path.exists(sl_path):
                return False
            
            self.sl_model = xgb.XGBRegressor()
            self.sl_model.load_model(sl_path)
            
            self.tp1_model = xgb.XGBRegressor()
            self.tp1_model.load_model(os.path.join(self.models_dir, 'tp1_model.json'))
            
            self.tp1_prob_model = xgb.XGBClassifier()
            self.tp1_prob_model.load_model(os.path.join(self.models_dir, 'tp1_prob_model.json'))
            
            self.rr_model = xgb.XGBRegressor()
            self.rr_model.load_model(os.path.join(self.models_dir, 'rr_model.json'))
            
            with open(os.path.join(self.models_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            print("[LevelOptimizer] Models loaded successfully")
            return True
            
        except Exception as e:
            print(f"[LevelOptimizer] Error loading models: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class LevelOptimizer:
    """
    Main interface for level optimization.
    Combines training data generation and ML prediction.
    """
    
    def __init__(self):
        self.data_generator = TrainingDataGenerator()
        self.models = LevelOptimizerModels()
        
        # Try to load existing models
        self.models.load()
    
    def train_from_candles(
        self,
        candles_df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Dict:
        """
        Quick training from candle data (when whale history is limited).
        
        Args:
            candles_df: DataFrame with OHLCV columns
            symbol: Symbol name for logging
        
        Returns:
            Training metrics
        """
        print(f"[LevelOptimizer] Generating training data from {symbol} candles...")
        
        training_df = self.data_generator.generate_from_candles_only(
            candles_df,
            lookforward_candles=50
        )
        
        if len(training_df) < 50:
            return {'error': 'Not enough training samples generated'}
        
        return self.models.train(training_df)
    
    def get_optimal_levels(
        self,
        entry_price: float,
        direction: str,
        features: Dict  # Raw features dict
    ) -> Dict:
        """
        Get ML-optimized levels for a trade.
        
        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            features: Dict with whale_pct, rsi, mfi, etc.
        
        Returns:
            Dict with entry, sl, tp1, tp2, tp3, probabilities
        """
        # Convert to SetupFeatures
        setup_features = SetupFeatures(
            whale_pct=features.get('whale_pct', 50),
            retail_pct=features.get('retail_pct', 50),
            whale_retail_diff=features.get('whale_pct', 50) - features.get('retail_pct', 50),
            oi_change_24h=features.get('oi_change', 0),
            funding_rate=features.get('funding_rate', 0),
            rsi=features.get('rsi', 50),
            mfi=features.get('mfi', 50),
            position_in_range=features.get('position_in_range', 50),
            atr_pct=features.get('atr_pct', 2),
            volume_ratio=features.get('volume_ratio', 1),
            price_change_24h=features.get('price_change_24h', 0),
            is_uptrend=features.get('is_uptrend', 0)
        )
        
        # Get ML predictions
        predictions = self.models.predict(setup_features)
        
        sl_pct = predictions['optimal_sl_pct']
        tp1_pct = predictions['optimal_tp1_pct']
        tp1_prob = predictions['tp1_probability']
        expected_rr = predictions['expected_rr']
        
        # Calculate actual price levels
        if direction == 'LONG':
            sl = entry_price * (1 - sl_pct / 100)
            tp1 = entry_price * (1 + tp1_pct / 100)
            tp2 = entry_price * (1 + tp1_pct * 1.5 / 100)
            tp3 = entry_price * (1 + tp1_pct * 2.5 / 100)
        else:  # SHORT
            sl = entry_price * (1 + sl_pct / 100)
            tp1 = entry_price * (1 - tp1_pct / 100)
            tp2 = entry_price * (1 - tp1_pct * 1.5 / 100)
            tp3 = entry_price * (1 - tp1_pct * 2.5 / 100)
        
        return {
            'entry': entry_price,
            'stop_loss': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'sl_pct': sl_pct,
            'tp1_pct': tp1_pct,
            'tp1_probability': tp1_prob,
            'tp2_probability': tp1_prob * 0.7,  # Estimate
            'tp3_probability': tp1_prob * 0.4,  # Estimate
            'expected_rr': expected_rr,
            'source': predictions['source']
        }
    
    @property
    def is_trained(self) -> bool:
        return self.models.is_trained


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Level Optimizer ML Module - Test")
    print("=" * 60)
    
    optimizer = LevelOptimizer()
    
    # Test prediction with dummy features
    test_features = {
        'whale_pct': 72,
        'retail_pct': 45,
        'oi_change': 5.2,
        'funding_rate': 0.01,
        'rsi': 45,
        'mfi': 55,
        'position_in_range': 35,
        'atr_pct': 2.1,
        'volume_ratio': 1.3,
        'price_change_24h': -2.5,
        'is_uptrend': 0
    }
    
    result = optimizer.get_optimal_levels(
        entry_price=100.0,
        direction='LONG',
        features=test_features
    )
    
    print("\nTest Prediction:")
    print(f"  Entry: ${result['entry']:.2f}")
    print(f"  SL: ${result['stop_loss']:.2f} (-{result['sl_pct']:.1f}%)")
    print(f"  TP1: ${result['tp1']:.2f} (+{result['tp1_pct']:.1f}%) - {result['tp1_probability']*100:.0f}% prob")
    print(f"  TP2: ${result['tp2']:.2f}")
    print(f"  TP3: ${result['tp3']:.2f}")
    print(f"  Expected R:R: {result['expected_rr']:.2f}")
    print(f"  Source: {result['source']}")
