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
        Dict with 'order_blocks', 'fvg', 'liquidity_sweep', 'structure' keys
    """
    if df is None or len(df) < 50:
        return {
            'order_blocks': {},
            'fvg': {},
            'liquidity_sweep': {},
            'structure': {'structure': 'Unknown', 'bias': 'Neutral', 'last_swing_high': 0, 'last_swing_low': 0}
        }
    
    return {
        'order_blocks': detect_order_blocks(df),
        'fvg': detect_fvg(df),
        'liquidity_sweep': detect_liquidity_sweep(df),
        'structure': analyze_market_structure(df)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_market_structure(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Analyze market structure - Higher Highs, Higher Lows, etc.
    
    IMPROVED: Now uses flexible swing detection and always returns swing levels
    """
    default_result = {
        'structure': 'Unknown', 
        'bias': 'Neutral',
        'last_swing_high': 0,
        'last_swing_low': 0
    }
    
    if df is None or len(df) < 20:
        return default_result
    
    try:
        # Limit to lookback period
        df_recent = df.tail(lookback) if len(df) > lookback else df
        
        # Always calculate simple swing range as fallback
        simple_high = df_recent['High'].max()
        simple_low = df_recent['Low'].min()
        
        # Find swing points with FLEXIBLE detection (3-bar pattern is more common)
        swing_highs = []
        swing_lows = []
        swing_high_prices = []
        swing_low_prices = []
        
        # Try with 3-bar pivot first (more sensitive)
        pivot_len = 3
        for i in range(pivot_len, len(df_recent) - pivot_len):
            # Swing high: Higher than pivot_len bars on each side
            if df_recent['High'].iloc[i] == df_recent['High'].iloc[i-pivot_len:i+pivot_len+1].max():
                swing_highs.append(i)
                swing_high_prices.append(df_recent['High'].iloc[i])
            # Swing low: Lower than pivot_len bars on each side
            if df_recent['Low'].iloc[i] == df_recent['Low'].iloc[i-pivot_len:i+pivot_len+1].min():
                swing_lows.append(i)
                swing_low_prices.append(df_recent['Low'].iloc[i])
        
        # If not enough swings found, try 2-bar pivot
        if len(swing_high_prices) < 2 or len(swing_low_prices) < 2:
            swing_highs = []
            swing_lows = []
            swing_high_prices = []
            swing_low_prices = []
            pivot_len = 2
            for i in range(pivot_len, len(df_recent) - pivot_len):
                if df_recent['High'].iloc[i] == df_recent['High'].iloc[i-pivot_len:i+pivot_len+1].max():
                    swing_highs.append(i)
                    swing_high_prices.append(df_recent['High'].iloc[i])
                if df_recent['Low'].iloc[i] == df_recent['Low'].iloc[i-pivot_len:i+pivot_len+1].min():
                    swing_lows.append(i)
                    swing_low_prices.append(df_recent['Low'].iloc[i])
        
        # Get the swing range (not just last swing!)
        # Use MAX of swing highs and MIN of swing lows for proper range
        if swing_high_prices:
            last_swing_high = max(swing_high_prices)  # Highest swing high
        else:
            last_swing_high = simple_high
            
        if swing_low_prices:
            last_swing_low = min(swing_low_prices)  # Lowest swing low
        else:
            last_swing_low = simple_low
        
        # Ensure we have a valid range (high > low)
        if last_swing_high <= last_swing_low:
            # Swings are inverted or same - use simple range
            last_swing_high = simple_high
            last_swing_low = simple_low
        
        # Determine structure if we have enough data
        if len(swing_high_prices) >= 2 and len(swing_low_prices) >= 2:
            # Compare last two swings
            hh = swing_high_prices[-1] > swing_high_prices[-2]  # Higher High
            hl = swing_low_prices[-1] > swing_low_prices[-2]    # Higher Low
            lh = swing_high_prices[-1] < swing_high_prices[-2]  # Lower High
            ll = swing_low_prices[-1] < swing_low_prices[-2]    # Lower Low
            
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
        else:
            # Not enough swings - use simple trend detection
            ema_fast = df_recent['Close'].ewm(span=10).mean().iloc[-1]
            ema_slow = df_recent['Close'].ewm(span=20).mean().iloc[-1]
            current = df_recent['Close'].iloc[-1]
            
            if current > ema_fast > ema_slow:
                structure = 'Bullish'
                bias = 'Long'
            elif current < ema_fast < ema_slow:
                structure = 'Bearish'
                bias = 'Short'
            else:
                structure = 'Mixed'
                bias = 'Neutral'
        
        return {
            'structure': structure,
            'bias': bias,
            'last_swing_high': last_swing_high,
            'last_swing_low': last_swing_low,
            'swing_count': len(swing_high_prices) + len(swing_low_prices)
        }
        
    except Exception as e:
        # Fallback: at least return the simple high/low
        try:
            return {
                'structure': 'Unknown', 
                'bias': 'Neutral',
                'last_swing_high': df['High'].tail(lookback).max() if df is not None else 0,
                'last_swing_low': df['Low'].tail(lookback).min() if df is not None else 0
            }
        except:
            return default_result
