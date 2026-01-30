"""
Feature Extractor for ML Models
================================

Extracts standardized features from market data for ML predictions.
Uses the same data available to MASTER_RULES for consistency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Whale/Institutional (5 features)
    'whale_pct',                # Whale long % (0-100)
    'retail_pct',               # Retail long % (0-100)
    'whale_retail_divergence',  # whale_pct - retail_pct
    'whale_dominance',          # How much whales dominate (normalized)
    'funding_rate',             # Funding rate (for crypto)
    
    # Open Interest (4 features)
    'oi_change_24h',            # OI change % in 24h
    'oi_signal_encoded',        # OI signal: -2=STRONG_BEARISH, -1=BEARISH, 0=NEUTRAL, 1=BULLISH, 2=STRONG_BULLISH
    'price_change_24h',         # Price change % in 24h
    'price_change_1h',          # Price change % in 1h
    
    # Position/Range (3 features)
    'position_in_range',        # Position % (0-100)
    'position_label_encoded',   # EARLY=0, MIDDLE=1, LATE=2
    'range_size_pct',           # Size of range as % of price
    
    # Technical (5 features)
    'ta_score',                 # TA score (0-100)
    'rsi',                      # RSI value
    'rsi_zone_encoded',         # Oversold=-1, Neutral=0, Overbought=1
    'trend_encoded',            # Bearish=-1, Neutral=0, Bullish=1
    'volatility_pct',           # ATR as % of price
    
    # Money Flow (2 features)
    'money_flow_encoded',       # MARKDOWN=-2, DISTRIBUTION=-1, NEUTRAL=0, ACCUMULATION=1, MARKUP=2
    'volume_ratio',             # Current volume vs average
    
    # SMC Structure (4 features)
    'at_bullish_ob',            # 1 if at bullish OB, 0 otherwise
    'at_bearish_ob',            # 1 if at bearish OB, 0 otherwise
    'near_support',             # 1 if near support, 0 otherwise
    'near_resistance',          # 1 if near resistance, 0 otherwise
    
    # Market Context (4 features)
    'btc_correlation',          # Correlation with BTC (-1 to 1)
    'btc_trend_encoded',        # BTC trend: -1=Bearish, 0=Neutral, 1=Bullish
    'market_fear_greed',        # Fear & Greed index (0-100)
    'is_weekend',               # 1 if weekend, 0 otherwise
    
    # Historical Performance (3 features)
    'historical_win_rate',      # Win rate from similar setups (0-100)
    'similar_setup_count',      # Number of similar historical setups
    'avg_historical_return',    # Average return from similar setups
]

# Number of features
NUM_FEATURES = len(FEATURE_NAMES)


# ═══════════════════════════════════════════════════════════════════════════════
# ENCODING MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

OI_SIGNAL_MAP = {
    'STRONG_BEARISH': -2,
    'BEARISH': -1,
    'SHORT_COVERING': -1,  # Treat as slightly bearish
    'NEUTRAL': 0,
    'BULLISH': 1,
    'STRONG_BULLISH': 2,
}

POSITION_LABEL_MAP = {
    'EARLY': 0,
    'MIDDLE': 1,
    'LATE': 2,
}

MONEY_FLOW_MAP = {
    'MARKDOWN': -2,
    'DISTRIBUTION': -1,
    'PROFIT_TAKING': -1,
    'PROFIT TAKING': -1,
    'NEUTRAL': 0,
    'UNKNOWN': 0,
    'ACCUMULATION': 1,
    'MARKUP': 2,
}

TREND_MAP = {
    'BEARISH': -1,
    'LEAN_BEARISH': -1,
    'NEUTRAL': 0,
    'MIXED': 0,
    'UNKNOWN': 0,
    'LEAN_BULLISH': 1,
    'BULLISH': 1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureSet:
    """Container for extracted features"""
    features: np.ndarray
    feature_names: List[str]
    raw_data: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display"""
        return {name: float(val) for name, val in zip(self.feature_names, self.features)}
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for model input"""
        return pd.DataFrame([self.features], columns=self.feature_names)


def extract_features(
    # Whale/Institutional data
    whale_pct: float = 50,
    retail_pct: float = 50,
    funding_rate: float = 0,
    
    # OI/Price data
    oi_change: float = 0,
    oi_signal: str = 'NEUTRAL',
    price_change_24h: float = 0,
    price_change_1h: float = 0,
    
    # Position data
    position_pct: float = 50,
    swing_high: float = 0,
    swing_low: float = 0,
    current_price: float = 0,
    
    # Technical data
    ta_score: float = 50,
    rsi: float = 50,
    trend: str = 'NEUTRAL',
    atr: float = 0,
    
    # Money flow data
    money_flow_phase: str = 'NEUTRAL',
    volume_ratio: float = 1.0,
    
    # SMC data
    at_bullish_ob: bool = False,
    at_bearish_ob: bool = False,
    near_support: bool = False,
    near_resistance: bool = False,
    
    # Market context
    btc_correlation: float = 0,
    btc_trend: str = 'NEUTRAL',
    fear_greed: int = 50,
    is_weekend: bool = False,
    
    # Historical data
    historical_win_rate: float = None,
    similar_setup_count: int = 0,
    avg_historical_return: float = 0,
) -> FeatureSet:
    """
    Extract standardized features for ML model.
    
    All inputs match the data available in MASTER_RULES for consistency.
    
    Returns:
        FeatureSet with normalized features
    """
    
    # Calculate derived features
    whale_retail_divergence = whale_pct - retail_pct
    whale_dominance = (whale_pct - 50) / 50  # Normalized -1 to 1
    
    # Position label
    if position_pct <= 35:
        position_label = 'EARLY'
    elif position_pct >= 65:
        position_label = 'LATE'
    else:
        position_label = 'MIDDLE'
    
    # Range size
    range_size_pct = 0
    if current_price > 0 and swing_high > swing_low:
        range_size_pct = ((swing_high - swing_low) / current_price) * 100
    
    # RSI zone
    if rsi <= 30:
        rsi_zone = -1  # Oversold
    elif rsi >= 70:
        rsi_zone = 1   # Overbought
    else:
        rsi_zone = 0   # Neutral
    
    # Volatility as % of price
    volatility_pct = (atr / current_price * 100) if current_price > 0 else 0
    
    # Handle None values
    if historical_win_rate is None:
        historical_win_rate = 50  # Assume neutral
    
    # Build feature array in exact order
    features = np.array([
        # Whale/Institutional (5)
        whale_pct,
        retail_pct,
        whale_retail_divergence,
        whale_dominance,
        funding_rate * 100,  # Convert to percentage points
        
        # Open Interest (4)
        oi_change,
        OI_SIGNAL_MAP.get(oi_signal.upper(), 0),
        price_change_24h,
        price_change_1h,
        
        # Position/Range (3)
        position_pct,
        POSITION_LABEL_MAP.get(position_label, 1),
        range_size_pct,
        
        # Technical (5)
        ta_score,
        rsi,
        rsi_zone,
        TREND_MAP.get(trend.upper() if trend else 'NEUTRAL', 0),
        volatility_pct,
        
        # Money Flow (2)
        MONEY_FLOW_MAP.get(money_flow_phase.upper() if money_flow_phase else 'NEUTRAL', 0),
        volume_ratio,
        
        # SMC Structure (4)
        1.0 if at_bullish_ob else 0.0,
        1.0 if at_bearish_ob else 0.0,
        1.0 if near_support else 0.0,
        1.0 if near_resistance else 0.0,
        
        # Market Context (4)
        btc_correlation,
        TREND_MAP.get(btc_trend.upper() if btc_trend else 'NEUTRAL', 0),
        fear_greed,
        1.0 if is_weekend else 0.0,
        
        # Historical (3)
        historical_win_rate,
        min(similar_setup_count, 100),  # Cap at 100
        avg_historical_return,
    ], dtype=np.float32)
    
    # Store raw data for reference
    raw_data = {
        'whale_pct': whale_pct,
        'retail_pct': retail_pct,
        'oi_change': oi_change,
        'price_change_24h': price_change_24h,
        'position_pct': position_pct,
        'ta_score': ta_score,
        'money_flow_phase': money_flow_phase,
        'historical_win_rate': historical_win_rate,
    }
    
    return FeatureSet(
        features=features,
        feature_names=FEATURE_NAMES,
        raw_data=raw_data
    )


def extract_features_from_dict(data: Dict) -> FeatureSet:
    """
    Extract features from a dictionary (e.g., from API response).
    
    Handles various key naming conventions.
    """
    
    # Map common key variations
    whale_pct = data.get('whale_pct') or data.get('top_trader_long_pct') or 50
    retail_pct = data.get('retail_pct') or data.get('retail_long_pct') or 50
    
    return extract_features(
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        funding_rate=data.get('funding_rate', 0),
        oi_change=data.get('oi_change') or data.get('oi_change_24h', 0),
        oi_signal=data.get('oi_signal', 'NEUTRAL'),
        price_change_24h=data.get('price_change_24h', 0),
        price_change_1h=data.get('price_change_1h', 0),
        position_pct=data.get('position_pct') or data.get('position_in_range', 50),
        swing_high=data.get('swing_high', 0),
        swing_low=data.get('swing_low', 0),
        current_price=data.get('current_price') or data.get('price', 0),
        ta_score=data.get('ta_score', 50),
        rsi=data.get('rsi', 50),
        trend=data.get('trend') or data.get('structure_type', 'NEUTRAL'),
        atr=data.get('atr', 0),
        money_flow_phase=data.get('money_flow_phase') or data.get('flow_status', 'NEUTRAL'),
        volume_ratio=data.get('volume_ratio', 1.0),
        at_bullish_ob=data.get('at_bullish_ob', False),
        at_bearish_ob=data.get('at_bearish_ob', False),
        near_support=data.get('near_support') or data.get('at_support', False),
        near_resistance=data.get('near_resistance') or data.get('at_resistance', False),
        btc_correlation=data.get('btc_correlation', 0),
        btc_trend=data.get('btc_trend', 'NEUTRAL'),
        fear_greed=data.get('fear_greed') or data.get('market_fear_greed', 50),
        is_weekend=data.get('is_weekend', False),
        historical_win_rate=data.get('historical_win_rate'),
        similar_setup_count=data.get('similar_setup_count') or data.get('sample_size', 0),
        avg_historical_return=data.get('avg_historical_return', 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def get_top_features(
    feature_set: FeatureSet,
    feature_importances: np.ndarray,
    top_n: int = 5
) -> List[Tuple[str, float, float]]:
    """
    Get top contributing features for a prediction.
    
    Returns:
        List of (feature_name, feature_value, importance_score) tuples
    """
    # Combine feature values with importances
    combined = list(zip(
        feature_set.feature_names,
        feature_set.features,
        feature_importances
    ))
    
    # Sort by absolute importance
    combined.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return combined[:top_n]


def format_feature_explanation(
    feature_name: str,
    feature_value: float,
    importance: float,
    prediction_direction: str
) -> str:
    """
    Generate human-readable explanation for a feature's contribution.
    """
    
    # Feature-specific explanations
    explanations = {
        'whale_pct': lambda v, d: f"Whales {v:.0f}% long {'supports' if d == 'LONG' else 'contradicts'} {d}",
        'whale_retail_divergence': lambda v, d: f"Whale/Retail gap of {v:+.0f}% {'favors' if (v > 0 and d == 'LONG') or (v < 0 and d == 'SHORT') else 'against'} {d}",
        'position_in_range': lambda v, d: f"Position {v:.0f}% - {'EARLY' if v < 35 else 'LATE' if v > 65 else 'MIDDLE'} entry",
        'oi_change_24h': lambda v, d: f"OI {v:+.1f}% {'supporting' if (v > 0 and d == 'LONG') or (v < 0 and d == 'SHORT') else 'warning sign'}",
        'historical_win_rate': lambda v, d: f"Similar setups won {v:.0f}% historically",
        'ta_score': lambda v, d: f"Technical score {v:.0f}/100",
        'at_bullish_ob': lambda v, d: f"{'At' if v else 'Not at'} bullish Order Block",
        'at_bearish_ob': lambda v, d: f"{'At' if v else 'Not at'} bearish Order Block",
    }
    
    if feature_name in explanations:
        return explanations[feature_name](feature_value, prediction_direction)
    else:
        return f"{feature_name}: {feature_value:.2f}"
