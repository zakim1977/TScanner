"""
Money Flow Analysis Module
OBV, MFI, CMF, Whale Detection, Pre-breakout Detection
"""

import pandas as pd
import numpy as np
from typing import Dict

from .indicators import calculate_obv, calculate_mfi, calculate_cmf, calculate_ema

# ═══════════════════════════════════════════════════════════════════════════════
# MONEY FLOW CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_money_flow(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive money flow indicators
    
    Returns:
        Dict with OBV, MFI, CMF, and accumulation/distribution status
    """
    if df is None or len(df) < 20:
        return {
            'obv': 0,
            'obv_ema': 0,
            'mfi': 50,
            'cmf': 0,
            'is_accumulating': False,
            'is_distributing': False,
            'flow_status': 'Neutral'
        }
    
    try:
        # Calculate indicators
        obv = calculate_obv(df['Close'], df['Volume'])
        obv_ema = calculate_ema(obv, 20)
        mfi = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
        cmf = calculate_cmf(df['High'], df['Low'], df['Close'], df['Volume'], 20)
        
        # Get current values
        current_obv = obv.iloc[-1]
        current_obv_ema = obv_ema.iloc[-1]
        current_mfi = mfi.iloc[-1]
        current_cmf = cmf.iloc[-1]
        
        # Determine flow status
        obv_rising = current_obv > current_obv_ema
        mfi_bullish = current_mfi > 50
        cmf_bullish = current_cmf > 0
        
        bullish_count = sum([obv_rising, mfi_bullish, cmf_bullish])
        
        if bullish_count >= 2:
            is_accumulating = True
            is_distributing = False
            flow_status = 'Inflow'
        elif bullish_count <= 0:
            is_accumulating = False
            is_distributing = True
            flow_status = 'Outflow'
        else:
            is_accumulating = False
            is_distributing = False
            flow_status = 'Neutral'
        
        return {
            'obv': current_obv,
            'obv_ema': current_obv_ema,
            'mfi': current_mfi,
            'cmf': current_cmf,
            'is_accumulating': is_accumulating,
            'is_distributing': is_distributing,
            'flow_status': flow_status,
            'obv_rising': obv_rising,
            'mfi_bullish': mfi_bullish,
            'cmf_bullish': cmf_bullish
        }
        
    except Exception as e:
        return {
            'obv': 0, 'obv_ema': 0, 'mfi': 50, 'cmf': 0,
            'is_accumulating': False, 'is_distributing': False,
            'flow_status': 'Neutral'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WHALE ACTIVITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_whale_activity(df: pd.DataFrame, volume_threshold: float = 3.0) -> Dict:
    """
    Detect unusual volume activity that might indicate whale/institutional activity
    
    Args:
        df: OHLCV DataFrame
        volume_threshold: Multiplier of average volume to consider as "whale"
        
    Returns:
        Dict with whale detection results
    """
    if df is None or len(df) < 20:
        return {
            'whale_detected': False,
            'volume_spike': False,
            'volume_ratio': 1.0,
            'direction': 'Neutral'
        }
    
    try:
        # Calculate volume metrics
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Check for volume spike
        volume_spike = volume_ratio >= volume_threshold
        
        # Determine direction of whale activity
        if volume_spike:
            price_change = df['Close'].iloc[-1] - df['Open'].iloc[-1]
            if price_change > 0:
                direction = 'Bullish'
            elif price_change < 0:
                direction = 'Bearish'
            else:
                direction = 'Neutral'
        else:
            direction = 'Neutral'
        
        return {
            'whale_detected': volume_spike and volume_ratio >= volume_threshold,
            'volume_spike': volume_spike,
            'volume_ratio': volume_ratio,
            'direction': direction,
            'avg_volume': avg_volume,
            'current_volume': current_volume
        }
        
    except Exception as e:
        return {
            'whale_detected': False,
            'volume_spike': False,
            'volume_ratio': 1.0,
            'direction': 'Neutral'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-BREAKOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_pre_breakout(df: pd.DataFrame) -> Dict:
    """
    Detect pre-breakout conditions:
    - Decreasing volatility (compression)
    - Increasing volume
    - Price near key levels
    
    Returns:
        Dict with pre-breakout analysis
    """
    if df is None or len(df) < 30:
        return {
            'probability': 0,
            'compression': False,
            'volume_building': False,
            'direction_bias': 'Neutral'
        }
    
    try:
        # Calculate volatility compression
        atr_short = (df['High'] - df['Low']).rolling(5).mean().iloc[-1]
        atr_long = (df['High'] - df['Low']).rolling(20).mean().iloc[-1]
        volatility_ratio = atr_short / atr_long if atr_long > 0 else 1
        
        compression = volatility_ratio < 0.8  # Short-term volatility < Long-term
        
        # Check volume building
        vol_short = df['Volume'].rolling(5).mean().iloc[-1]
        vol_long = df['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = vol_short / vol_long if vol_long > 0 else 1
        
        volume_building = volume_ratio > 1.2  # Volume increasing
        
        # Check price position
        ema_20 = calculate_ema(df['Close'], 20).iloc[-1]
        ema_50 = calculate_ema(df['Close'], 50).iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # Determine direction bias
        if current_price > ema_20 > ema_50:
            direction_bias = 'Bullish'
        elif current_price < ema_20 < ema_50:
            direction_bias = 'Bearish'
        else:
            direction_bias = 'Neutral'
        
        # Calculate probability
        probability = 0
        if compression:
            probability += 30
        if volume_building:
            probability += 30
        if direction_bias != 'Neutral':
            probability += 20
        
        # Additional factors
        recent_range = df['High'].iloc[-10:].max() - df['Low'].iloc[-10:].min()
        prev_range = df['High'].iloc[-20:-10].max() - df['Low'].iloc[-20:-10].min()
        
        if recent_range < prev_range * 0.7:  # Range tightening
            probability += 20
        
        return {
            'probability': min(probability, 100),
            'compression': compression,
            'volume_building': volume_building,
            'direction_bias': direction_bias,
            'volatility_ratio': volatility_ratio,
            'volume_ratio': volume_ratio
        }
        
    except Exception as e:
        return {
            'probability': 0,
            'compression': False,
            'volume_building': False,
            'direction_bias': 'Neutral'
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ACCUMULATION/DISTRIBUTION ZONES
# ═══════════════════════════════════════════════════════════════════════════════

def detect_accumulation_zone(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect accumulation zones - areas where smart money is buying
    
    Characteristics:
    - Flat/ranging price action
    - Increasing OBV
    - Higher lows
    """
    if df is None or len(df) < lookback:
        return {'in_accumulation': False, 'zone_bottom': 0, 'zone_top': 0}
    
    try:
        # Check for ranging price
        price_range = df['High'].iloc[-lookback:].max() - df['Low'].iloc[-lookback:].min()
        avg_price = df['Close'].iloc[-lookback:].mean()
        range_pct = (price_range / avg_price) * 100
        
        is_ranging = range_pct < 10  # Less than 10% range
        
        # Check OBV trend
        obv = calculate_obv(df['Close'], df['Volume'])
        obv_start = obv.iloc[-lookback]
        obv_end = obv.iloc[-1]
        obv_rising = obv_end > obv_start
        
        # Check for higher lows
        lows = df['Low'].iloc[-lookback:]
        higher_lows = lows.iloc[-5:].min() > lows.iloc[:5].min()
        
        in_accumulation = is_ranging and obv_rising and higher_lows
        
        return {
            'in_accumulation': in_accumulation,
            'zone_bottom': df['Low'].iloc[-lookback:].min(),
            'zone_top': df['High'].iloc[-lookback:].max(),
            'is_ranging': is_ranging,
            'obv_rising': obv_rising,
            'higher_lows': higher_lows
        }
        
    except Exception as e:
        return {'in_accumulation': False, 'zone_bottom': 0, 'zone_top': 0}


def detect_distribution_zone(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect distribution zones - areas where smart money is selling
    
    Characteristics:
    - Flat/ranging price action
    - Decreasing OBV
    - Lower highs
    """
    if df is None or len(df) < lookback:
        return {'in_distribution': False, 'zone_bottom': 0, 'zone_top': 0}
    
    try:
        # Check for ranging price
        price_range = df['High'].iloc[-lookback:].max() - df['Low'].iloc[-lookback:].min()
        avg_price = df['Close'].iloc[-lookback:].mean()
        range_pct = (price_range / avg_price) * 100
        
        is_ranging = range_pct < 10
        
        # Check OBV trend
        obv = calculate_obv(df['Close'], df['Volume'])
        obv_start = obv.iloc[-lookback]
        obv_end = obv.iloc[-1]
        obv_falling = obv_end < obv_start
        
        # Check for lower highs
        highs = df['High'].iloc[-lookback:]
        lower_highs = highs.iloc[-5:].max() < highs.iloc[:5].max()
        
        in_distribution = is_ranging and obv_falling and lower_highs
        
        return {
            'in_distribution': in_distribution,
            'zone_bottom': df['Low'].iloc[-lookback:].min(),
            'zone_top': df['High'].iloc[-lookback:].max(),
            'is_ranging': is_ranging,
            'obv_falling': obv_falling,
            'lower_highs': lower_highs
        }
        
    except Exception as e:
        return {'in_distribution': False, 'zone_bottom': 0, 'zone_top': 0}
