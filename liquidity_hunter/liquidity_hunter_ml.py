"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    UNIFIED LIQUIDITY HUNTER ML                                 ║
║                                                                                ║
║  ONE MODEL TO RULE THEM ALL - No more scalp/day/swing/investment modes!        ║
║                                                                                ║
║  Predicts:                                                                     ║
║  • P(Sweep Long Target) - Probability of sweeping lows and going up           ║
║  • P(Sweep Short Target) - Probability of sweeping highs and going down       ║
║  • P(Reach TP) - Probability of reaching each target level                    ║
║                                                                                ║
║  Features are ATR-normalized so they work across ANY timeframe!               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# ML imports
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[UNIFIED_LH] sklearn/imblearn not available - install with: pip install scikit-learn imbalanced-learn")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "unified_lh_model.joblib")
DB_PATH = "data/unified_lh_training.db"

# Training parameters
DEFAULT_TRAINING_DAYS = 365
LOOKBACK_CANDLES = 50  # How far back to look for swing levels
FORWARD_CANDLES = 30   # How far forward to check outcomes
MIN_SAMPLES = 500      # Minimum samples to train


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def init_training_db():
    """Initialize the training database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Training samples table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            -- Context
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            timeframe TEXT,
            
            -- Level info (NORMALIZED - works any timeframe)
            level_price REAL NOT NULL,
            level_type TEXT,
            level_strength TEXT,
            direction TEXT NOT NULL,
            distance_atr REAL,
            
            -- Whale data
            whale_pct REAL,
            whale_delta REAL,
            
            -- Market context
            position_in_range REAL,
            volume_ratio REAL,
            momentum REAL,
            trend_aligned INTEGER,
            
            -- Target info
            target_price REAL,
            target_distance_atr REAL,
            
            -- LABELS (what we predict)
            sweep_occurred INTEGER,
            target_reached INTEGER,
            candles_to_sweep INTEGER,
            candles_to_target INTEGER,
            max_favorable_atr REAL,
            max_adverse_atr REAL
        )
    ''')
    
    # Model history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
            samples_count INTEGER,
            f1_score REAL,
            accuracy REAL,
            sweep_rate REAL,
            target_rate REAL,
            feature_importance TEXT,
            training_days INTEGER,
            symbols TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[UNIFIED_LH] Database initialized at {DB_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION (ATR-Normalized - Works Any Timeframe!)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR from DataFrame."""
    if len(df) < period + 1:
        return df['close'].iloc[-1] * 0.02  # Fallback: 2% of price
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    
    if pd.isna(atr) or atr <= 0:
        return df['close'].iloc[-1] * 0.02
    
    return atr


def extract_features(
    df: pd.DataFrame,
    level_price: float,
    level_type: str,
    level_strength: str,
    direction: str,
    whale_pct: float = 50,
    whale_delta: float = 0,
    target_price: float = None
) -> Dict:
    """
    Extract NORMALIZED features for unified model.
    
    Key: Everything is normalized by ATR or expressed as ratios.
    This means features work identically on 5m or 1d charts!
    """
    if df is None or len(df) < 20:
        return None
    
    current_price = float(df['close'].iloc[-1])
    atr = calculate_atr(df)
    
    features = {}
    
    # ═══════════════════════════════════════════════════════════════
    # 1. DISTANCE FEATURES (ATR-normalized)
    # ═══════════════════════════════════════════════════════════════
    distance = abs(current_price - level_price)
    features['distance_atr'] = distance / atr
    features['is_below'] = 1 if level_price < current_price else 0
    
    # Target distance
    if target_price and target_price > 0:
        target_distance = abs(target_price - current_price)
        features['target_distance_atr'] = target_distance / atr
    else:
        features['target_distance_atr'] = 2.0  # Default
    
    # ═══════════════════════════════════════════════════════════════
    # 2. LEVEL CHARACTERISTICS
    # ═══════════════════════════════════════════════════════════════
    
    # Level type encoding
    type_scores = {
        'EQUAL_LOW': 1.0, 'EQUAL_HIGH': 1.0,  # Strongest
        '100x': 0.9, 'LIQUIDATION_100x': 0.9,
        '50x': 0.7, 'LIQUIDATION_50x': 0.7,
        'LIQ_POOL_STRONG': 0.6, 'POOL_STRONG': 0.6,
        '25x': 0.5, 'LIQUIDATION_25x': 0.5,
        'LIQ_POOL_MODERATE': 0.4, 'POOL_MODERATE': 0.4,
        'SWING': 0.3
    }
    features['level_type_score'] = type_scores.get(level_type, 0.5)
    
    # Strength encoding
    strength_scores = {'MAJOR': 1.0, 'STRONG': 0.8, 'MODERATE': 0.5, 'WEAK': 0.2}
    features['level_strength_score'] = strength_scores.get(level_strength, 0.5)
    
    # Combined level quality
    features['level_quality'] = (features['level_type_score'] + features['level_strength_score']) / 2
    
    # ═══════════════════════════════════════════════════════════════
    # 3. WHALE FEATURES (The Edge!)
    # ═══════════════════════════════════════════════════════════════
    features['whale_pct'] = whale_pct / 100.0  # Normalize to 0-1
    features['whale_delta'] = whale_delta / 100.0 if whale_delta else 0
    
    # Whale positioning buckets
    features['whale_bullish'] = 1 if whale_pct >= 60 else 0
    features['whale_bearish'] = 1 if whale_pct <= 40 else 0
    features['whale_extreme'] = 1 if whale_pct >= 70 or whale_pct <= 30 else 0
    
    # Whale momentum
    features['whale_accumulating'] = 1 if whale_delta and whale_delta > 3 else 0
    features['whale_distributing'] = 1 if whale_delta and whale_delta < -3 else 0
    
    # CRITICAL: Whale-Direction Alignment
    # For LONG: bullish whales = aligned
    # For SHORT: bearish whales = aligned
    if direction == 'LONG':
        alignment = (whale_pct - 50) / 50  # -1 to +1, positive = bullish
    else:
        alignment = (50 - whale_pct) / 50  # -1 to +1, positive = bearish
    features['whale_aligned'] = alignment
    
    # ═══════════════════════════════════════════════════════════════
    # 4. PRICE CONTEXT (Normalized)
    # ═══════════════════════════════════════════════════════════════
    
    # Position in recent range
    high_20 = df['high'].tail(20).max()
    low_20 = df['low'].tail(20).min()
    range_size = high_20 - low_20
    if range_size > 0:
        features['position_in_range'] = (current_price - low_20) / range_size
    else:
        features['position_in_range'] = 0.5
    
    # At extremes
    features['at_range_low'] = 1 if features['position_in_range'] < 0.2 else 0
    features['at_range_high'] = 1 if features['position_in_range'] > 0.8 else 0
    
    # ═══════════════════════════════════════════════════════════════
    # 5. MOMENTUM & VOLUME (Normalized)
    # ═══════════════════════════════════════════════════════════════
    
    # RSI-like momentum (0-1)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    features['momentum'] = rsi.iloc[-1] / 100.0 if not pd.isna(rsi.iloc[-1]) else 0.5
    
    # Volume ratio
    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        current_vol = df['volume'].iloc[-1]
        features['volume_ratio'] = min(current_vol / vol_ma, 3.0) if vol_ma > 0 else 1.0
    else:
        features['volume_ratio'] = 1.0
    
    # ═══════════════════════════════════════════════════════════════
    # 6. TREND CONTEXT
    # ═══════════════════════════════════════════════════════════════
    
    # EMA alignment
    ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
    ema_50 = df['close'].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else ema_20
    
    if current_price > ema_20 > ema_50:
        features['trend'] = 1  # Bullish
    elif current_price < ema_20 < ema_50:
        features['trend'] = -1  # Bearish
    else:
        features['trend'] = 0  # Neutral
    
    # Trend-direction alignment
    if direction == 'LONG':
        features['trend_aligned'] = 1 if features['trend'] >= 0 else 0
    else:
        features['trend_aligned'] = 1 if features['trend'] <= 0 else 0
    
    # ═══════════════════════════════════════════════════════════════
    # 7. REVERSAL CANDLE PATTERNS (NEW - Critical for filtering fakes!)
    # ═══════════════════════════════════════════════════════════════

    # Analyze the most recent candle for reversal patterns
    if len(df) >= 3:
        candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        o, h, l, c = float(candle['open']), float(candle['high']), float(candle['low']), float(candle['close'])
        body = abs(c - o)
        full_range = h - l if h > l else 0.0001

        if direction == 'LONG':
            # For LONG: want lower wick > body (rejection)
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)
            features['reversal_rejection_ratio'] = lower_wick / full_range
            features['reversal_has_rejection'] = 1 if lower_wick > body and lower_wick > upper_wick else 0
            features['reversal_favorable_close'] = 1 if c > o else 0  # Bullish close

            # Bullish engulfing
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            is_engulfing = (c > o and prev_candle['close'] < prev_candle['open'] and
                          body > prev_body and c > prev_candle['open'])
            features['reversal_engulfing'] = 1 if is_engulfing else 0
        else:
            # For SHORT: want upper wick > body (rejection)
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)
            features['reversal_rejection_ratio'] = upper_wick / full_range
            features['reversal_has_rejection'] = 1 if upper_wick > body and upper_wick > lower_wick else 0
            features['reversal_favorable_close'] = 1 if c < o else 0  # Bearish close

            # Bearish engulfing
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            is_engulfing = (c < o and prev_candle['close'] > prev_candle['open'] and
                          body > prev_body and c < prev_candle['open'])
            features['reversal_engulfing'] = 1 if is_engulfing else 0

        # Combined reversal score (0-1)
        features['reversal_score'] = (
            features['reversal_has_rejection'] * 0.4 +
            features['reversal_favorable_close'] * 0.3 +
            features['reversal_engulfing'] * 0.3
        )
    else:
        features['reversal_rejection_ratio'] = 0
        features['reversal_has_rejection'] = 0
        features['reversal_favorable_close'] = 0
        features['reversal_engulfing'] = 0
        features['reversal_score'] = 0

    # ═══════════════════════════════════════════════════════════════
    # 8. RSI DIVERGENCE (NEW - Powerful reversal signal)
    # ═══════════════════════════════════════════════════════════════

    # Check for divergence in last 10 candles
    features['rsi_divergence'] = 0
    features['rsi_extreme'] = 0

    if len(df) >= 15:
        rsi_values = rsi.tail(10).dropna()
        price_lows = df['low'].tail(10)
        price_highs = df['high'].tail(10)

        current_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50

        # RSI extreme
        if direction == 'LONG' and current_rsi < 35:
            features['rsi_extreme'] = 1
        elif direction == 'SHORT' and current_rsi > 65:
            features['rsi_extreme'] = 1

        # Simple divergence check
        if len(rsi_values) >= 5 and len(price_lows) >= 5:
            if direction == 'LONG':
                # Bullish divergence: price lower low, RSI higher low
                price_made_lower = price_lows.iloc[-1] < price_lows.iloc[:-1].min()
                rsi_made_higher = rsi_values.iloc[-1] > rsi_values.iloc[:-1].min()
                if price_made_lower and rsi_made_higher:
                    features['rsi_divergence'] = 1
            else:
                # Bearish divergence: price higher high, RSI lower high
                price_made_higher = price_highs.iloc[-1] > price_highs.iloc[:-1].max()
                rsi_made_lower = rsi_values.iloc[-1] < rsi_values.iloc[:-1].max()
                if price_made_higher and rsi_made_lower:
                    features['rsi_divergence'] = 1

    # ═══════════════════════════════════════════════════════════════
    # 9. TREND STRENGTH (Enhanced)
    # ═══════════════════════════════════════════════════════════════

    # EMA 200 for stronger trend signal
    if len(df) >= 200:
        ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
    elif len(df) >= 100:
        ema_200 = df['close'].ewm(span=100).mean().iloc[-1]
    else:
        ema_200 = ema_50

    # Price vs EMA200 (normalized)
    features['price_vs_ema200'] = (current_price - ema_200) / atr

    # Strong trend detection
    if current_price > ema_20 > ema_50 > ema_200:
        features['strong_uptrend'] = 1
        features['strong_downtrend'] = 0
    elif current_price < ema_20 < ema_50 < ema_200:
        features['strong_uptrend'] = 0
        features['strong_downtrend'] = 1
    else:
        features['strong_uptrend'] = 0
        features['strong_downtrend'] = 0

    # Trading against strong trend (danger signal!)
    features['against_strong_trend'] = 0
    if direction == 'LONG' and features['strong_downtrend'] == 1:
        features['against_strong_trend'] = 1
    elif direction == 'SHORT' and features['strong_uptrend'] == 1:
        features['against_strong_trend'] = 1

    # ═══════════════════════════════════════════════════════════════
    # 10. COMPOSITE SCORES (Updated with new features)
    # ═══════════════════════════════════════════════════════════════

    # Overall setup quality (enhanced)
    features['setup_quality'] = (
        features['level_quality'] * 0.2 +
        max(0, features['whale_aligned']) * 0.2 +
        features['trend_aligned'] * 0.15 +
        features['reversal_score'] * 0.2 +  # NEW
        features['rsi_extreme'] * 0.1 +  # NEW
        features['rsi_divergence'] * 0.1 +  # NEW
        (1 - min(features['distance_atr'] / 3, 1)) * 0.05
    )

    # Penalty for trading against strong trend
    if features['against_strong_trend'] == 1:
        features['setup_quality'] *= 0.5  # Halve the quality score

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def find_swing_levels(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[float], List[float]]:
    """Find swing highs and lows in price data."""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        # Swing high: higher than surrounding candles
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, lookback+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, lookback+1)):
            swing_highs.append((i, df['high'].iloc[i]))
        
        # Swing low: lower than surrounding candles
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, lookback+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, lookback+1)):
            swing_lows.append((i, df['low'].iloc[i]))
    
    return swing_lows, swing_highs


def find_liquidity_levels(df: pd.DataFrame, atr: float = None) -> Dict:
    """
    Find ALL liquidity levels - the HYBRID approach.
    
    Finds:
    1. EQUAL_LOW / EQUAL_HIGH - Multiple tests at same level (strongest!)
    2. DOUBLE_LOW / DOUBLE_HIGH - Two tests (strong)
    3. SWING_LOW / SWING_HIGH - Single swing (moderate)
    
    These approximate where liquidations cluster because:
    - Equal lows = many stops at same price = liquidation magnet
    - Major swings = obvious support/resistance = stop placement zone
    
    Returns:
        {
            'lows': [(idx, price, type, strength), ...],
            'highs': [(idx, price, type, strength), ...]
        }
    """
    if atr is None:
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
    
    tolerance = atr * 0.3  # Levels within 0.3 ATR are "equal"
    
    # Find all swing levels first
    swing_lows, swing_highs = find_swing_levels(df, lookback=3)
    
    # ═══════════════════════════════════════════════════════════════════
    # CLUSTER EQUAL LOWS (Liquidation Magnets!)
    # ═══════════════════════════════════════════════════════════════════
    lows_with_type = []
    used_low_indices = set()
    
    for i, (idx1, price1) in enumerate(swing_lows):
        if idx1 in used_low_indices:
            continue
        
        cluster = [(idx1, price1)]
        used_low_indices.add(idx1)
        
        for j, (idx2, price2) in enumerate(swing_lows[i+1:], i+1):
            if idx2 in used_low_indices:
                continue
            if abs(price2 - price1) <= tolerance:
                cluster.append((idx2, price2))
                used_low_indices.add(idx2)
        
        avg_price = sum(p for _, p in cluster) / len(cluster)
        last_idx = max(idx for idx, _ in cluster)
        
        if len(cluster) >= 3:
            # Triple+ test = EQUAL_LOW (like 100x liquidation zone!)
            lows_with_type.append((last_idx, avg_price, 'EQUAL_LOW', 1.0))
        elif len(cluster) == 2:
            # Double test = strong level
            lows_with_type.append((last_idx, avg_price, 'DOUBLE_LOW', 0.85))
        else:
            # Single swing = regular level
            lows_with_type.append((idx1, price1, 'SWING_LOW', 0.5))
    
    # ═══════════════════════════════════════════════════════════════════
    # CLUSTER EQUAL HIGHS (Liquidation Magnets!)
    # ═══════════════════════════════════════════════════════════════════
    highs_with_type = []
    used_high_indices = set()
    
    for i, (idx1, price1) in enumerate(swing_highs):
        if idx1 in used_high_indices:
            continue
        
        cluster = [(idx1, price1)]
        used_high_indices.add(idx1)
        
        for j, (idx2, price2) in enumerate(swing_highs[i+1:], i+1):
            if idx2 in used_high_indices:
                continue
            if abs(price2 - price1) <= tolerance:
                cluster.append((idx2, price2))
                used_high_indices.add(idx2)
        
        avg_price = sum(p for _, p in cluster) / len(cluster)
        last_idx = max(idx for idx, _ in cluster)
        
        if len(cluster) >= 3:
            highs_with_type.append((last_idx, avg_price, 'EQUAL_HIGH', 1.0))
        elif len(cluster) == 2:
            highs_with_type.append((last_idx, avg_price, 'DOUBLE_HIGH', 0.85))
        else:
            highs_with_type.append((idx1, price1, 'SWING_HIGH', 0.5))
    
    return {
        'lows': lows_with_type,
        'highs': highs_with_type
    }


# Level type to strength mapping (for features)
LEVEL_TYPE_STRENGTH = {
    # Real liquidation levels (for live prediction)
    '100X_LIQ': 1.0,
    '50X_LIQ': 0.9,
    '25X_LIQ': 0.75,
    # Historical proxy levels (for training)
    'EQUAL_LOW': 1.0,
    'EQUAL_HIGH': 1.0,
    'DOUBLE_LOW': 0.85,
    'DOUBLE_HIGH': 0.85,
    'SWING_LOW': 0.5,
    'SWING_HIGH': 0.5,
    # Fallbacks
    'MAJOR': 0.8,
    'MODERATE': 0.5,
    'WEAK': 0.3,
}


def label_outcome(
    df: pd.DataFrame,
    idx: int,
    level_price: float,
    direction: str,
    atr: float,
    forward_candles: int = 30,
    level_type: str = 'SWING_LOW',
    level_strength: float = 0.5
) -> Dict:
    """
    Label the outcome of a potential sweep setup.
    
    FIXED: Dynamic targets based on level strength!
    - EQUAL lows (strongest) → Bigger bounce expected → 1.2 ATR target
    - DOUBLE lows → 1.0 ATR target  
    - SWING lows → 0.75 ATR target (most common)
    
    This gives REALISTIC targets that actually get hit!
    """
    if idx + forward_candles >= len(df):
        return None
    
    current_price = df['close'].iloc[idx]
    
    sweep_occurred = False
    target_reached = False
    candles_to_sweep = None
    candles_to_target = None
    max_favorable = 0
    max_adverse = 0
    
    # DYNAMIC TARGETS based on level strength!
    # Stronger levels = more liquidity = bigger bounce
    if level_strength >= 0.9:  # EQUAL lows (3+ tests)
        TARGET_ATR = 1.2
        STOP_ATR = 0.6
    elif level_strength >= 0.75:  # DOUBLE lows (2 tests)
        TARGET_ATR = 1.0
        STOP_ATR = 0.5
    else:  # Regular swings
        TARGET_ATR = 0.75  # More achievable!
        STOP_ATR = 0.4
    
    if direction == 'LONG':
        target_price = current_price + (TARGET_ATR * atr)
        stop_price = level_price - (STOP_ATR * atr)
    else:
        target_price = current_price - (TARGET_ATR * atr)
        stop_price = level_price + (STOP_ATR * atr)
    
    entry_price = None
    
    for i in range(1, forward_candles):
        candle = df.iloc[idx + i]
        
        if direction == 'LONG':
            # Check for sweep (price goes below level then closes above)
            if not sweep_occurred:
                if candle['low'] < level_price and candle['close'] > level_price:
                    sweep_occurred = True
                    candles_to_sweep = i
                    entry_price = candle['close']
            
            # After sweep, check for target
            if sweep_occurred and entry_price:
                favorable = candle['high'] - entry_price
                adverse = entry_price - candle['low']
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)
                
                if candle['high'] >= target_price and not target_reached:
                    target_reached = True
                    candles_to_target = i - candles_to_sweep
                
                # Check if stopped out
                if candle['low'] <= stop_price:
                    break
        else:
            # SHORT setup
            if not sweep_occurred:
                if candle['high'] > level_price and candle['close'] < level_price:
                    sweep_occurred = True
                    candles_to_sweep = i
                    entry_price = candle['close']
            
            if sweep_occurred and entry_price:
                favorable = entry_price - candle['low']
                adverse = candle['high'] - entry_price
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)
                
                if candle['low'] <= target_price and not target_reached:
                    target_reached = True
                    candles_to_target = i - candles_to_sweep
                
                if candle['high'] >= stop_price:
                    break
    
    return {
        'sweep_occurred': 1 if sweep_occurred else 0,
        'target_reached': 1 if target_reached else 0,
        'candles_to_sweep': candles_to_sweep,
        'candles_to_target': candles_to_target,
        'max_favorable_atr': max_favorable / atr if atr > 0 else 0,
        'max_adverse_atr': max_adverse / atr if atr > 0 else 0,
        'target_price': target_price
    }


def generate_samples_from_df(
    df: pd.DataFrame,
    symbol: str,
    whale_history: List[Dict] = None,
    timeframe: str = '1h'
) -> List[Dict]:
    """
    Generate training samples from a price DataFrame.
    
    HYBRID APPROACH:
    - Uses find_liquidity_levels() to find EQUAL lows/highs (strongest)
    - These approximate where real liquidations cluster
    - Model learns patterns that transfer to real 25x/50x/100x levels
    """
    if df is None or len(df) < LOOKBACK_CANDLES + FORWARD_CANDLES + 50:
        return []
    
    # Normalize columns
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    samples = []
    
    # Pre-calculate ATR for efficiency
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    atr_current = df['atr'].iloc[-1]
    
    # ═══════════════════════════════════════════════════════════════════
    # HYBRID: Find liquidity levels (EQUAL, DOUBLE, SWING)
    # ═══════════════════════════════════════════════════════════════════
    liquidity_levels = find_liquidity_levels(df, atr_current)
    
    # Count level types for logging
    equal_lows = sum(1 for _, _, t, _ in liquidity_levels['lows'] if 'EQUAL' in t)
    double_lows = sum(1 for _, _, t, _ in liquidity_levels['lows'] if 'DOUBLE' in t)
    swing_lows = sum(1 for _, _, t, _ in liquidity_levels['lows'] if t == 'SWING_LOW')
    equal_highs = sum(1 for _, _, t, _ in liquidity_levels['highs'] if 'EQUAL' in t)
    double_highs = sum(1 for _, _, t, _ in liquidity_levels['highs'] if 'DOUBLE' in t)
    swing_highs = sum(1 for _, _, t, _ in liquidity_levels['highs'] if t == 'SWING_HIGH')
    
    print(f"[UNIFIED_LH] {symbol} {timeframe}: LOWS: {equal_lows} equal, {double_lows} double, {swing_lows} swing | HIGHS: {equal_highs} equal, {double_highs} double, {swing_highs} swing")
    
    # Process each candle as potential entry point
    for i in range(LOOKBACK_CANDLES + 20, len(df) - FORWARD_CANDLES):
        current_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]
        
        if pd.isna(atr) or atr <= 0:
            continue
        
        # Get whale data for this timestamp (if available)
        whale_pct = 50
        whale_delta = 0
        if whale_history:
            for wh in reversed(whale_history):
                whale_pct = wh.get('whale_pct', 50)
                whale_delta = wh.get('whale_delta', 0)
                break
        
        # Find recent lows below current price (LONG opportunities)
        recent_lows = [(idx, price, ltype, strength) 
                       for idx, price, ltype, strength in liquidity_levels['lows'] 
                       if idx < i and i - idx <= LOOKBACK_CANDLES and price < current_price]
        
        # Find recent highs above current price (SHORT opportunities)
        recent_highs = [(idx, price, ltype, strength) 
                        for idx, price, ltype, strength in liquidity_levels['highs'] 
                        if idx < i and i - idx <= LOOKBACK_CANDLES and price > current_price]
        
        # ═══════════════════════════════════════════════════════════════
        # Process LONG opportunities (sweep lows)
        # PRIORITIZE: EQUAL > DOUBLE > SWING
        # ═══════════════════════════════════════════════════════════════
        sorted_lows = sorted(recent_lows, key=lambda x: x[3], reverse=True)[:3]
        
        for level_idx, level_price, level_type, level_strength in sorted_lows:
            distance_atr = (current_price - level_price) / atr
            
            if distance_atr > 4:  # Too far
                continue
            
            # Label outcome - pass level strength for dynamic targets!
            outcome = label_outcome(df, i, level_price, 'LONG', atr, FORWARD_CANDLES, level_type, level_strength)
            if outcome is None:
                continue
            
            # Extract features - NOW WITH REAL LEVEL TYPE!
            hist_df = df.iloc[max(0, i-50):i+1]
            features = extract_features(
                hist_df, level_price, level_type, level_strength, 'LONG',
                whale_pct, whale_delta, outcome['target_price']
            )
            
            if features is None:
                continue
            
            # Create sample
            sample = {
                'symbol': symbol,
                'timestamp': df.index[i] if hasattr(df.index[i], 'isoformat') else str(i),
                'timeframe': timeframe,
                'level_price': level_price,
                'level_type': level_type,
                'level_strength': level_strength,
                'direction': 'LONG',
                'whale_pct': whale_pct,
                'whale_delta': whale_delta,
                'target_price': outcome['target_price'],
                **features,
                **outcome
            }
            samples.append(sample)
        
        # ═══════════════════════════════════════════════════════════════
        # Process SHORT opportunities (sweep highs)
        # ═══════════════════════════════════════════════════════════════
        sorted_highs = sorted(recent_highs, key=lambda x: x[3], reverse=True)[:3]
        
        for level_idx, level_price, level_type, level_strength in sorted_highs:
            distance_atr = (level_price - current_price) / atr
            
            if distance_atr > 4:
                continue
            
            outcome = label_outcome(df, i, level_price, 'SHORT', atr, FORWARD_CANDLES, level_type, level_strength)
            if outcome is None:
                continue
            
            hist_df = df.iloc[max(0, i-50):i+1]
            features = extract_features(
                hist_df, level_price, level_type, level_strength, 'SHORT',
                whale_pct, whale_delta, outcome['target_price']
            )
            
            if features is None:
                continue
            
            sample = {
                'symbol': symbol,
                'timestamp': df.index[i] if hasattr(df.index[i], 'isoformat') else str(i),
                'timeframe': timeframe,
                'level_price': level_price,
                'level_type': level_type,
                'level_strength': level_strength,
                'direction': 'SHORT',
                'whale_pct': whale_pct,
                'whale_delta': whale_delta,
                'target_price': outcome['target_price'],
                **features,
                **outcome
            }
            samples.append(sample)
    
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedLiquidityHunterML:
    """
    ONE model for all liquidity hunting.
    
    No more scalp/day/swing/investment modes!
    
    Predicts:
    - P(Sweep) - Will price sweep this level?
    - P(Target) - After sweep, will target be reached?
    """
    
    # Features used for prediction (all ATR-normalized)
    # ENHANCED with reversal candle, RSI divergence, and trend strength features
    FEATURE_COLS = [
        # Distance features
        'distance_atr',
        'target_distance_atr',
        'is_below',

        # Level characteristics
        'level_type_score',
        'level_strength_score',
        'level_quality',

        # Whale features (The Edge!)
        'whale_pct',
        'whale_delta',
        'whale_bullish',
        'whale_bearish',
        'whale_extreme',
        'whale_accumulating',
        'whale_distributing',
        'whale_aligned',

        # Price context
        'position_in_range',
        'at_range_low',
        'at_range_high',

        # Momentum & Volume
        'momentum',
        'volume_ratio',

        # Trend features (basic)
        'trend',
        'trend_aligned',

        # NEW: Reversal candle patterns (Critical for filtering fakes!)
        'reversal_rejection_ratio',
        'reversal_has_rejection',
        'reversal_favorable_close',
        'reversal_engulfing',
        'reversal_score',

        # NEW: RSI divergence (Powerful reversal signal)
        'rsi_extreme',
        'rsi_divergence',

        # NEW: Enhanced trend strength
        'price_vs_ema200',
        'strong_uptrend',
        'strong_downtrend',
        'against_strong_trend',

        # Composite (updated with new features)
        'setup_quality',
    ]
    
    def __init__(self):
        self.sweep_model = None
        self.target_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        
        # Try to load existing model
        self._load()
    
    def _load(self) -> bool:
        """Load model from disk."""
        if os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.sweep_model = data['sweep_model']
                self.target_model = data['target_model']
                self.scaler = data['scaler']
                self.metrics = data.get('metrics', {})
                self.is_trained = True
                print(f"[UNIFIED_LH] Loaded model (Sweep F1: {self.metrics.get('sweep_f1', 0):.1%}, Target F1: {self.metrics.get('target_f1', 0):.1%})")
                return True
            except Exception as e:
                print(f"[UNIFIED_LH] Error loading model: {e}")
        return False
    
    def _save(self):
        """Save model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        data = {
            'sweep_model': self.sweep_model,
            'target_model': self.target_model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'feature_cols': self.FEATURE_COLS
        }
        joblib.dump(data, MODEL_PATH)
        print(f"[UNIFIED_LH] Model saved to {MODEL_PATH}")
    
    def train(self, samples: List[Dict]) -> Dict:
        """
        Train the unified model.
        
        Args:
            samples: List of training samples with features and labels
        
        Returns:
            Training metrics
        """
        if not ML_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        if len(samples) < MIN_SAMPLES:
            return {'error': f'Need at least {MIN_SAMPLES} samples, got {len(samples)}'}
        
        print(f"[UNIFIED_LH] Training on {len(samples)} samples...")
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Check which features are available
        available_features = [f for f in self.FEATURE_COLS if f in df.columns]
        print(f"[UNIFIED_LH] Using {len(available_features)} features")
        
        # Prepare features
        X = df[available_features].fillna(0)
        y_sweep = df['sweep_occurred']
        y_target = df['target_reached']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_sweep_train, y_sweep_test, y_target_train, y_target_test = \
            train_test_split(X_scaled, y_sweep, y_target, test_size=0.2, random_state=42)
        
        # Balance classes with SMOTE if needed
        try:
            smote = SMOTE(random_state=42)
            X_train_sweep, y_sweep_train_balanced = smote.fit_resample(X_train, y_sweep_train)
            X_train_target, y_target_train_balanced = smote.fit_resample(X_train, y_target_train)
        except:
            X_train_sweep, y_sweep_train_balanced = X_train, y_sweep_train
            X_train_target, y_target_train_balanced = X_train, y_target_train
        
        # Train SWEEP model
        print("[UNIFIED_LH] Training sweep prediction model...")
        self.sweep_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=20,
            random_state=42
        )
        self.sweep_model.fit(X_train_sweep, y_sweep_train_balanced)
        
        # Train TARGET model
        print("[UNIFIED_LH] Training target prediction model...")
        self.target_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=20,
            random_state=42
        )
        self.target_model.fit(X_train_target, y_target_train_balanced)
        
        # Evaluate
        sweep_pred = self.sweep_model.predict(X_test)
        target_pred = self.target_model.predict(X_test)
        
        sweep_f1 = f1_score(y_sweep_test, sweep_pred)
        target_f1 = f1_score(y_target_test, target_pred)
        sweep_acc = accuracy_score(y_sweep_test, sweep_pred)
        target_acc = accuracy_score(y_target_test, target_pred)
        
        # Feature importance
        feature_importance = dict(zip(available_features, self.sweep_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # ═══════════════════════════════════════════════════════════════
        # TRADING METRICS CALCULATION
        # ═══════════════════════════════════════════════════════════════
        
        # Win rate = sweeps that reached target / total sweeps
        sweeps_occurred = df[df['sweep_occurred'] == 1]
        if len(sweeps_occurred) > 0:
            targets_reached = sweeps_occurred[sweeps_occurred['target_reached'] == 1]
            win_rate = len(targets_reached) / len(sweeps_occurred)
            total_trades = len(sweeps_occurred)
        else:
            win_rate = 0
            total_trades = 0
        
        # Calculate average R:R from max favorable/adverse movements
        avg_rr = 2.5  # Default
        if 'max_favorable_atr' in df.columns and 'max_adverse_atr' in df.columns:
            favorable = df['max_favorable_atr'].mean()
            adverse = df['max_adverse_atr'].mean()
            if adverse > 0:
                avg_rr = favorable / adverse
        
        # ROI calculation: Win% * Reward - Loss% * Risk
        # Assuming 1% risk per trade
        risk_per_trade = 1.0
        reward_per_trade = risk_per_trade * avg_rr
        roi_per_trade = (win_rate * reward_per_trade) - ((1 - win_rate) * risk_per_trade)
        
        # Monthly ROI estimate (assume 3 trades per week for sweeps)
        trades_per_month = 12  # Conservative estimate
        monthly_roi = roi_per_trade * trades_per_month
        
        self.metrics = {
            'sweep_f1': sweep_f1,
            'target_f1': target_f1,
            'sweep_accuracy': sweep_acc,
            'target_accuracy': target_acc,
            'samples': len(samples),
            'sweep_rate': y_sweep.mean(),
            'target_rate': y_target.mean(),
            'features_used': len(available_features),
            'top_features': top_features,
            'trained_at': datetime.now().isoformat(),
            # Trading metrics
            'win_rate': win_rate,
            'total_trades': total_trades,
            'avg_rr': avg_rr,
            'roi_per_trade': roi_per_trade,
            'monthly_roi': monthly_roi,
            'trades_per_month': trades_per_month
        }
        
        print(f"\n{'='*60}")
        print(f"[UNIFIED_LH] TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Samples: {len(samples)}")
        print(f"Sweep Rate: {y_sweep.mean():.1%} | Target Rate: {y_target.mean():.1%}")
        print(f"Sweep F1: {sweep_f1:.1%} | Accuracy: {sweep_acc:.1%}")
        print(f"Target F1: {target_f1:.1%} | Accuracy: {target_acc:.1%}")
        print(f"\nTop Features:")
        for feat, imp in top_features[:5]:
            print(f"  {feat}: {imp:.3f}")
        print(f"{'='*60}\n")
        
        self.is_trained = True
        self._save()
        
        return self.metrics
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict probabilities for a liquidity level.
        
        Returns:
            {
                'sweep_prob': float (0-100) - Probability price sweeps this level
                'target_prob': float (0-100) - Probability of reaching target after sweep
                'combined_prob': float (0-100) - Combined probability (sweep * target)
                'quality': 'HIGH' | 'MEDIUM' | 'LOW'
            }
        """
        if not self.is_trained:
            return self._rule_based_prediction(features)
        
        try:
            # Prepare features
            X = pd.DataFrame([features])[self.FEATURE_COLS].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            sweep_prob = self.sweep_model.predict_proba(X_scaled)[0]
            sweep_prob = sweep_prob[1] if len(sweep_prob) > 1 else sweep_prob[0]
            
            target_prob = self.target_model.predict_proba(X_scaled)[0]
            target_prob = target_prob[1] if len(target_prob) > 1 else target_prob[0]
            
            combined = sweep_prob * target_prob
            
            # Quality label
            if combined >= 0.50:
                quality = 'HIGH'
            elif combined >= 0.30:
                quality = 'MEDIUM'
            else:
                quality = 'LOW'
            
            return {
                'sweep_prob': int(sweep_prob * 100),
                'target_prob': int(target_prob * 100),
                'combined_prob': int(combined * 100),
                'quality': quality
            }
            
        except Exception as e:
            print(f"[UNIFIED_LH] Prediction error: {e}")
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: Dict) -> Dict:
        """
        Fallback rule-based prediction when model not trained.
        ENHANCED with reversal candle, RSI divergence, and trend filters.
        """

        # Base probability from distance
        distance = features.get('distance_atr', 2)
        if distance < 0.5:
            sweep_prob = 75
        elif distance < 1.0:
            sweep_prob = 65
        elif distance < 2.0:
            sweep_prob = 50
        else:
            sweep_prob = 35

        # Whale alignment bonus
        whale_aligned = features.get('whale_aligned', 0)
        sweep_prob += int(whale_aligned * 15)

        # Level quality bonus
        level_quality = features.get('level_quality', 0.5)
        sweep_prob += int((level_quality - 0.5) * 20)

        # ═══════════════════════════════════════════════════════════
        # NEW: Reversal candle bonus/penalty
        # ═══════════════════════════════════════════════════════════
        reversal_score = features.get('reversal_score', 0)
        if reversal_score >= 0.7:
            sweep_prob += 10  # Strong reversal candle
        elif reversal_score >= 0.4:
            sweep_prob += 5   # Decent reversal candle
        elif reversal_score < 0.2:
            sweep_prob -= 10  # Weak reversal candle - penalize

        # ═══════════════════════════════════════════════════════════
        # NEW: RSI divergence bonus
        # ═══════════════════════════════════════════════════════════
        if features.get('rsi_divergence', 0) == 1:
            sweep_prob += 10  # Divergence is a strong signal

        if features.get('rsi_extreme', 0) == 1:
            sweep_prob += 5   # RSI at extreme

        # ═══════════════════════════════════════════════════════════
        # NEW: Trend alignment (Critical!)
        # ═══════════════════════════════════════════════════════════
        if features.get('against_strong_trend', 0) == 1:
            sweep_prob -= 20  # Heavy penalty for counter-trend trades!

        trend_aligned = features.get('trend_aligned', 0)
        if trend_aligned == 1:
            sweep_prob += 5

        # Clamp
        sweep_prob = max(15, min(90, sweep_prob))

        # Target probability depends more on reversal confirmation
        target_multiplier = 0.7 + (reversal_score * 0.2)  # 0.7 to 0.9 based on reversal
        target_prob = int(sweep_prob * target_multiplier)

        combined = int(sweep_prob * target_prob / 100)

        quality = 'HIGH' if combined >= 50 else 'MEDIUM' if combined >= 30 else 'LOW'

        return {
            'sweep_prob': sweep_prob,
            'target_prob': target_prob,
            'combined_prob': combined,
            'quality': quality
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON & CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_model_instance = None

def get_model() -> UnifiedLiquidityHunterML:
    """Get singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = UnifiedLiquidityHunterML()
    return _model_instance


def predict_level(
    df: pd.DataFrame,
    level_price: float,
    level_type: str,
    level_strength: str,
    direction: str,
    whale_pct: float = 50,
    whale_delta: float = 0,
    target_price: float = None
) -> Dict:
    """
    Convenience function to predict probability for a liquidity level.
    
    Args:
        df: Recent price DataFrame
        level_price: The liquidity level price
        level_type: Type of level ('EQUAL_LOW', '50x', 'LIQ_POOL', etc.)
        level_strength: 'STRONG', 'MODERATE', 'WEAK'
        direction: 'LONG' or 'SHORT'
        whale_pct: Current whale long percentage
        whale_delta: Change in whale percentage
        target_price: Target price (optional, will estimate if not provided)
    
    Returns:
        {
            'sweep_prob': int (0-100),
            'target_prob': int (0-100),
            'combined_prob': int (0-100),
            'quality': 'HIGH' | 'MEDIUM' | 'LOW'
        }
    """
    model = get_model()
    
    # Extract features
    features = extract_features(
        df, level_price, level_type, level_strength, direction,
        whale_pct, whale_delta, target_price
    )
    
    if features is None:
        return {'sweep_prob': 50, 'target_prob': 50, 'combined_prob': 25, 'quality': 'UNKNOWN'}
    
    return model.predict(features)


def get_model_status() -> Dict:
    """Get current model status."""
    model = get_model()
    return {
        'is_trained': model.is_trained,
        'metrics': model.metrics,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════

def train_unified_model(
    symbols: List[str] = None,
    days: int = 365,
    progress_callback: callable = None
) -> Dict:
    """
    Main training function - fetches data and trains model.
    
    Args:
        symbols: List of symbols to train on (default: major cryptos)
        days: Days of historical data to use
        progress_callback: Optional callback(status_text, progress_pct)
    
    Returns:
        Training metrics dict
    """
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
        ]
    
    print(f"\n{'='*60}")
    print(f"[UNIFIED_LH] STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Days: {days}")
    print(f"{'='*60}\n")
    
    def update_progress(text, pct):
        if progress_callback:
            progress_callback(text, pct)
        print(f"[UNIFIED_LH] {text}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Fetch data using parallel fetcher from core.data_fetcher
    # ═══════════════════════════════════════════════════════════════════
    update_progress("⚡ Fetching data in parallel...", 0.1)
    
    try:
        from core.data_fetcher import fetch_klines_parallel
        
        def fetch_progress(completed, total, symbol):
            pct = 0.1 + (completed / total) * 0.4  # 10% to 50%
            update_progress(f"⚡ Fetching {symbol}... {completed}/{total}", pct)
        
        klines_data = fetch_klines_parallel(
            symbols=symbols,
            interval='4h',
            limit=min(days * 6, 1500),
            progress_callback=fetch_progress
        )
        
    except ImportError as e:
        print(f"[UNIFIED_LH] core.data_fetcher not available: {e}")
        return {'error': 'data_fetcher not available. Make sure core/data_fetcher.py exists.'}
    
    if not klines_data:
        return {'error': 'No data fetched. Check internet connection.'}
    
    print(f"[UNIFIED_LH] Fetched {len(klines_data)} symbols")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Load whale history
    # ═══════════════════════════════════════════════════════════════════
    update_progress("📊 Loading whale history...", 0.5)
    
    whale_history = []
    try:
        import sqlite3
        whale_db = "data/whale_history.db"
        if os.path.exists(whale_db):
            conn = sqlite3.connect(whale_db)
            df_whale = pd.read_sql("SELECT * FROM whale_snapshots ORDER BY timestamp DESC LIMIT 10000", conn)
            whale_history = df_whale.to_dict('records')
            conn.close()
            print(f"[UNIFIED_LH] Loaded {len(whale_history)} whale history records")
    except Exception as e:
        print(f"[UNIFIED_LH] Could not load whale history: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Generate training samples
    # ═══════════════════════════════════════════════════════════════════
    update_progress("🔄 Processing data and generating samples...", 0.6)
    
    all_samples = []
    processed = 0
    total = len(klines_data)
    
    for symbol, df in klines_data.items():
        processed += 1
        pct = 0.6 + (processed / total) * 0.2  # 60% to 80%
        update_progress(f"🔄 Processing {symbol}... {processed}/{total}", pct)
        
        if df is not None and len(df) >= 200:
            try:
                samples = generate_samples_from_df(df, symbol, whale_history, '4h')
                all_samples.extend(samples)
            except Exception as e:
                print(f"[UNIFIED_LH] Error processing {symbol}: {e}")
    
    print(f"\n[UNIFIED_LH] Total samples: {len(all_samples)}")
    
    if len(all_samples) < MIN_SAMPLES:
        return {'error': f'Insufficient samples: {len(all_samples)}. Need at least {MIN_SAMPLES}.'}
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Train model
    # ═══════════════════════════════════════════════════════════════════
    update_progress("🤖 Training model...", 0.85)
    
    model = get_model()
    metrics = model.train(all_samples)
    
    update_progress("✅ Training complete!", 1.0)
    
    return metrics


# Initialize database on import
init_training_db()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNIFIED LIQUIDITY HUNTER ML - TRAINING")
    print("="*60 + "\n")
    
    # Train with default settings
    metrics = train_unified_model(days=365)
    
    if 'error' not in metrics:
        print("\n✅ Training successful!")
        print(f"Sweep F1: {metrics.get('sweep_f1', 0):.1%}")
        print(f"Target F1: {metrics.get('target_f1', 0):.1%}")
    else:
        print(f"\n❌ Training failed: {metrics.get('error')}")
