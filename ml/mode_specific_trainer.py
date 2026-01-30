"""
Mode-Specific Trainer
=====================

Provides feature extraction for mode-specific ML models.
This module bridges the gap between raw data and model input.

Required by ml_engine.py for mode-specific predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE NAMES FOR MODE-SPECIFIC MODELS
# These match what the trained models expect
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Whale/Positioning (6 features)
    'whale_pct',
    'retail_pct',
    'divergence',
    'position_pct',
    'volume_ratio',
    'RSI',
    
    # Trend & Structure (3 features)
    'trend',
    'ATR_pct',
    'BB_Squeeze_Pct',
    
    # Technical (4 features)
    'BB_Width',
    'buy_ratio',
    'price_change_5',
    'price_change_20',
    
    # Price Action (4 features)
    'price_change_50',
    'near_support',
    'near_resistance',
    'Close',
    
    # Additional features for some models (14 more = 31 total for stocks)
    'oi_change_24h',
    'funding_rate',
    'ta_score',
    'money_flow_encoded',
    'at_bullish_ob',
    'at_bearish_ob',
    'btc_correlation',
    'btc_trend_encoded',
    'market_fear_greed',
    'historical_win_rate',
    'explosion_score',
    'bb_width_pct',
    'energy_loaded',
    'compression_duration',
    
    # Crypto-specific (2 more = 33 total for crypto)
    'whale_dominance',
    'market_trend',
]

NUM_FEATURES = len(FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FROM ROW
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_row(row: pd.Series) -> np.ndarray:
    """
    Extract features from a pandas Series (row).
    
    This is used by ml_engine.py for mode-specific model predictions.
    The row should contain the raw data fields that we extract features from.
    
    Args:
        row: pandas Series with raw market data
        
    Returns:
        numpy array of features matching FEATURE_NAMES order
    """
    features = []
    
    # Whale/Positioning (6 features)
    features.append(float(row.get('whale_pct', 50)))
    features.append(float(row.get('retail_pct', 50)))
    features.append(float(row.get('divergence', row.get('whale_pct', 50) - row.get('retail_pct', 50))))
    features.append(float(row.get('position_pct', 50)))
    features.append(float(row.get('volume_ratio', 1.0)))
    features.append(float(row.get('RSI', row.get('rsi', 50))))
    
    # Trend & Structure (3 features)
    trend_val = row.get('trend', 0)
    if isinstance(trend_val, str):
        trend_map = {'BEARISH': -1, 'NEUTRAL': 0, 'BULLISH': 1}
        trend_val = trend_map.get(trend_val.upper(), 0)
    features.append(float(trend_val))
    features.append(float(row.get('ATR_pct', row.get('atr_pct', 2.0))))
    features.append(float(row.get('BB_Squeeze_Pct', row.get('bb_squeeze_pct', 50))))
    
    # Technical (4 features)
    features.append(float(row.get('BB_Width', row.get('bb_width', row.get('bb_width_pct', 3.0)))))
    features.append(float(row.get('buy_ratio', 0.5)))
    features.append(float(row.get('price_change_5', row.get('price_change_1h', 0))))
    features.append(float(row.get('price_change_20', row.get('price_change_24h', 0))))
    
    # Price Action (4 features)
    features.append(float(row.get('price_change_50', row.get('price_change_24h', 0) * 2)))
    features.append(float(1 if row.get('near_support', False) else 0))
    features.append(float(1 if row.get('near_resistance', False) else 0))
    features.append(float(row.get('Close', row.get('current_price', 100))))
    
    # Additional features (14 more)
    features.append(float(row.get('oi_change_24h', row.get('oi_change', 0))))
    features.append(float(row.get('funding_rate', 0)))
    features.append(float(row.get('ta_score', 50)))
    features.append(float(row.get('money_flow_encoded', 0)))
    features.append(float(1 if row.get('at_bullish_ob', False) else 0))
    features.append(float(1 if row.get('at_bearish_ob', False) else 0))
    features.append(float(row.get('btc_correlation', 0.5)))
    features.append(float(row.get('btc_trend_encoded', 0)))
    features.append(float(row.get('market_fear_greed', row.get('fear_greed', 50))))
    features.append(float(row.get('historical_win_rate', 50)))
    features.append(float(row.get('explosion_score', 0)))
    features.append(float(row.get('bb_width_pct', row.get('BB_Width', 3.0))))
    features.append(float(1 if row.get('energy_loaded', False) else 0))
    features.append(float(row.get('compression_duration', 0)))
    
    # Crypto-specific (2 more)
    whale_pct = float(row.get('whale_pct', 50))
    features.append(float((whale_pct - 50) / 50))  # whale_dominance normalized
    features.append(float(row.get('market_trend', row.get('trend', 0))))
    
    return np.array(features, dtype=np.float32)


def extract_features_from_dict(data: Dict) -> np.ndarray:
    """
    Extract features from a dictionary.
    
    Convenience wrapper around extract_features_from_row.
    
    Args:
        data: Dictionary with raw market data
        
    Returns:
        numpy array of features
    """
    row = pd.Series(data)
    return extract_features_from_row(row)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE SCALING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_ranges() -> Dict[str, tuple]:
    """
    Get expected ranges for each feature (for validation/normalization).
    
    Returns:
        Dict mapping feature name to (min, max) tuple
    """
    return {
        'whale_pct': (0, 100),
        'retail_pct': (0, 100),
        'divergence': (-100, 100),
        'position_pct': (0, 100),
        'volume_ratio': (0, 10),
        'RSI': (0, 100),
        'trend': (-1, 1),
        'ATR_pct': (0, 20),
        'BB_Squeeze_Pct': (0, 100),
        'BB_Width': (0, 20),
        'buy_ratio': (0, 1),
        'price_change_5': (-50, 50),
        'price_change_20': (-100, 100),
        'price_change_50': (-200, 200),
        'near_support': (0, 1),
        'near_resistance': (0, 1),
        'Close': (0, float('inf')),
        'oi_change_24h': (-100, 100),
        'funding_rate': (-0.5, 0.5),
        'ta_score': (0, 100),
        'money_flow_encoded': (-2, 2),
        'at_bullish_ob': (0, 1),
        'at_bearish_ob': (0, 1),
        'btc_correlation': (-1, 1),
        'btc_trend_encoded': (-1, 1),
        'market_fear_greed': (0, 100),
        'historical_win_rate': (0, 100),
        'explosion_score': (0, 100),
        'bb_width_pct': (0, 20),
        'energy_loaded': (0, 1),
        'compression_duration': (0, 50),
        'whale_dominance': (-1, 1),
        'market_trend': (-1, 1),
    }


def validate_features(features: np.ndarray) -> bool:
    """
    Validate that features are within expected ranges.
    
    Args:
        features: numpy array of features
        
    Returns:
        True if all features are valid
    """
    ranges = get_feature_ranges()
    
    for i, (name, (min_val, max_val)) in enumerate(zip(FEATURE_NAMES, ranges.values())):
        if i >= len(features):
            break
        val = features[i]
        if np.isnan(val) or np.isinf(val):
            return False
        # Don't validate 'Close' upper bound (can be any price)
        if name != 'Close' and max_val != float('inf'):
            if val < min_val - 10 or val > max_val + 10:  # Allow some slack
                return False
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# Alias for legacy code
MODE_FEATURE_NAMES = FEATURE_NAMES
