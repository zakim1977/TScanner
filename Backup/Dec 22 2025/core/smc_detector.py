"""
Smart Money Concepts (SMC) Detection Module
Order Blocks, Fair Value Gaps, Liquidity Sweeps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_order_blocks(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Detect Order Blocks - last opposing candle before a strong move
    
    Bullish OB: Last bearish candle before strong bullish move
    Bearish OB: Last bullish candle before strong bearish move
    """
    if df is None or len(df) < lookback:
        return {}
    
    result = {
        'bullish_ob': False,
        'bearish_ob': False,
        'bullish_ob_top': 0,
        'bullish_ob_bottom': 0,
        'bearish_ob_top': 0,
        'bearish_ob_bottom': 0,
        'at_bullish_ob': False,
        'at_bearish_ob': False,
        'near_bullish_ob': False,
        'near_bearish_ob': False
    }
    
    try:
        current_price = df['Close'].iloc[-1]
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        
        # Look for bullish order blocks
        for i in range(len(df) - lookback, len(df) - 3):
            # Bearish candle followed by strong bullish move
            if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Bearish candle
                # Check for strong bullish move after
                move_size = df['Close'].iloc[i+1:i+4].max() - df['Close'].iloc[i]
                if move_size > atr * 2:  # Strong move
                    result['bullish_ob'] = True
                    result['bullish_ob_top'] = df['Open'].iloc[i]
                    result['bullish_ob_bottom'] = df['Close'].iloc[i]
                    
                    # Check if price is at OB
                    if result['bullish_ob_bottom'] <= current_price <= result['bullish_ob_top']:
                        result['at_bullish_ob'] = True
                    elif current_price <= result['bullish_ob_top'] * 1.02:
                        result['near_bullish_ob'] = True
        
        # Look for bearish order blocks
        for i in range(len(df) - lookback, len(df) - 3):
            # Bullish candle followed by strong bearish move
            if df['Close'].iloc[i] > df['Open'].iloc[i]:  # Bullish candle
                # Check for strong bearish move after
                move_size = df['Close'].iloc[i] - df['Close'].iloc[i+1:i+4].min()
                if move_size > atr * 2:  # Strong move
                    result['bearish_ob'] = True
                    result['bearish_ob_top'] = df['Close'].iloc[i]
                    result['bearish_ob_bottom'] = df['Open'].iloc[i]
                    
                    # Check if price is at OB
                    if result['bearish_ob_bottom'] <= current_price <= result['bearish_ob_top']:
                        result['at_bearish_ob'] = True
                    elif current_price >= result['bearish_ob_bottom'] * 0.98:
                        result['near_bearish_ob'] = True
        
        return result
        
    except Exception as e:
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, lookback: int = 30) -> Dict:
    """
    Detect Fair Value Gaps - imbalance zones where price moved too fast
    
    Bullish FVG: Gap between candle 1's high and candle 3's low (in uptrend)
    Bearish FVG: Gap between candle 1's low and candle 3's high (in downtrend)
    """
    if df is None or len(df) < lookback:
        return {}
    
    result = {
        'bullish_fvg': False,
        'bearish_fvg': False,
        'bullish_fvg_high': 0,
        'bullish_fvg_low': 0,
        'bearish_fvg_high': 0,
        'bearish_fvg_low': 0,
        'at_bullish_fvg': False,
        'at_bearish_fvg': False
    }
    
    try:
        current_price = df['Close'].iloc[-1]
        
        # Look for FVGs in recent candles
        for i in range(len(df) - lookback, len(df) - 2):
            # Bullish FVG: Low of candle 3 > High of candle 1
            if df['Low'].iloc[i+2] > df['High'].iloc[i]:
                result['bullish_fvg'] = True
                result['bullish_fvg_high'] = df['Low'].iloc[i+2]
                result['bullish_fvg_low'] = df['High'].iloc[i]
                
                # Check if price is at FVG
                if result['bullish_fvg_low'] <= current_price <= result['bullish_fvg_high']:
                    result['at_bullish_fvg'] = True
            
            # Bearish FVG: High of candle 3 < Low of candle 1
            if df['High'].iloc[i+2] < df['Low'].iloc[i]:
                result['bearish_fvg'] = True
                result['bearish_fvg_high'] = df['Low'].iloc[i]
                result['bearish_fvg_low'] = df['High'].iloc[i+2]
                
                # Check if price is at FVG
                if result['bearish_fvg_low'] <= current_price <= result['bearish_fvg_high']:
                    result['at_bearish_fvg'] = True
        
        return result
        
    except Exception as e:
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY SWEEPS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect Liquidity Sweeps - stop hunts below lows / above highs
    
    Bullish sweep: Price breaks below recent low then closes above
    Bearish sweep: Price breaks above recent high then closes below
    """
    if df is None or len(df) < lookback:
        return {}
    
    result = {
        'bullish_sweep': False,
        'bearish_sweep': False,
        'sweep_level': 0,
        'sweep_type': None
    }
    
    try:
        recent_high = df['High'].iloc[-lookback:-1].max()
        recent_low = df['Low'].iloc[-lookback:-1].min()
        
        last_high = df['High'].iloc[-1]
        last_low = df['Low'].iloc[-1]
        last_close = df['Close'].iloc[-1]
        last_open = df['Open'].iloc[-1]
        
        # Bullish sweep: Went below recent low but closed above it
        if last_low < recent_low and last_close > recent_low:
            result['bullish_sweep'] = True
            result['sweep_level'] = recent_low
            result['sweep_type'] = 'bullish'
        
        # Bearish sweep: Went above recent high but closed below it
        if last_high > recent_high and last_close < recent_high:
            result['bearish_sweep'] = True
            result['sweep_level'] = recent_high
            result['sweep_type'] = 'bearish'
        
        return result
        
    except Exception as e:
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SMC DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_smc(df: pd.DataFrame) -> Dict:
    """
    Main function to detect all SMC concepts
    
    Returns:
        Dict with 'order_blocks', 'fvg', 'liquidity_sweep' keys
    """
    if df is None or len(df) < 50:
        return {
            'order_blocks': {},
            'fvg': {},
            'liquidity_sweep': {}
        }
    
    return {
        'order_blocks': detect_order_blocks(df),
        'fvg': detect_fvg(df),
        'liquidity_sweep': detect_liquidity_sweep(df)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_market_structure(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Analyze market structure - Higher Highs, Higher Lows, etc.
    """
    if df is None or len(df) < lookback:
        return {'structure': 'Unknown', 'bias': 'Neutral'}
    
    try:
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df) - 5):
            # Swing high
            if df['High'].iloc[i] == df['High'].iloc[i-5:i+6].max():
                swing_highs.append(df['High'].iloc[i])
            # Swing low
            if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+6].min():
                swing_lows.append(df['Low'].iloc[i])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {'structure': 'Unknown', 'bias': 'Neutral'}
        
        # Check structure
        hh = swing_highs[-1] > swing_highs[-2]  # Higher High
        hl = swing_lows[-1] > swing_lows[-2]    # Higher Low
        lh = swing_highs[-1] < swing_highs[-2]  # Lower High
        ll = swing_lows[-1] < swing_lows[-2]    # Lower Low
        
        if hh and hl:
            structure = 'Bullish'
            bias = 'Long'
        elif lh and ll:
            structure = 'Bearish'
            bias = 'Short'
        elif hh and ll:
            structure = 'Ranging'
            bias = 'Neutral'
        elif lh and hl:
            structure = 'Consolidating'
            bias = 'Neutral'
        else:
            structure = 'Mixed'
            bias = 'Neutral'
        
        return {
            'structure': structure,
            'bias': bias,
            'last_swing_high': swing_highs[-1] if swing_highs else 0,
            'last_swing_low': swing_lows[-1] if swing_lows else 0
        }
        
    except Exception as e:
        return {'structure': 'Unknown', 'bias': 'Neutral'}
