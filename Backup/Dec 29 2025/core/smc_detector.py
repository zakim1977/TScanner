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
        'near_bearish_ob': False,
        'at_support': False,      # NEW: Price near swing low
        'at_resistance': False,   # NEW: Price near swing high
    }
    
    try:
        current_price = df['Close'].iloc[-1]
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        
        # Calculate swing levels for support/resistance detection
        df_recent = df.tail(lookback) if len(df) > lookback else df
        swing_high = df_recent['High'].max()
        swing_low = df_recent['Low'].min()
        swing_range = swing_high - swing_low if swing_high > swing_low else 1
        
        # Check if price is near support (swing low) - within 10% of range from bottom
        position_pct = ((current_price - swing_low) / swing_range) * 100 if swing_range > 0 else 50
        if position_pct <= 15:  # Within 15% from bottom = near support
            result['at_support'] = True
        elif position_pct >= 85:  # Within 15% from top = near resistance
            result['at_resistance'] = True
        
        # Look for bullish order blocks (find the MOST RECENT valid OB)
        best_bullish_ob = None
        for i in range(len(df) - 3, len(df) - lookback, -1):  # Search backwards for most recent
            # Bearish candle followed by strong bullish move
            if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Bearish candle
                # Check for strong bullish move after
                move_size = df['Close'].iloc[i+1:min(i+4, len(df))].max() - df['Close'].iloc[i]
                if move_size > atr * 1.0:  # Reduced to 1.0x for better sensitivity
                    ob_top = df['Open'].iloc[i]
                    ob_bottom = df['Close'].iloc[i]
                    ob_height = ob_top - ob_bottom
                    
                    # Store OB data
                    result['bullish_ob'] = True
                    result['bullish_ob_top'] = ob_top
                    result['bullish_ob_bottom'] = ob_bottom
                    
                    # Check if price is AT or NEAR this OB
                    # AT: Price is inside the OB zone
                    if ob_bottom <= current_price <= ob_top:
                        result['at_bullish_ob'] = True
                        break  # Found best one
                    # NEAR (above): Price is within 2% above OB top (just left the zone)
                    # This means OB is acting as SUPPORT below
                    elif ob_top < current_price <= ob_top * 1.03:
                        result['near_bullish_ob'] = True
                        break
                    # NEAR (approaching): Price is within 2% below OB bottom
                    elif ob_bottom * 0.98 <= current_price < ob_bottom:
                        result['near_bullish_ob'] = True
                        break
        
        # Look for bearish order blocks (find the MOST RECENT valid OB)
        for i in range(len(df) - 3, len(df) - lookback, -1):  # Search backwards for most recent
            # Bullish candle followed by strong bearish move
            if df['Close'].iloc[i] > df['Open'].iloc[i]:  # Bullish candle
                # Check for strong bearish move after
                move_size = df['Close'].iloc[i] - df['Close'].iloc[i+1:min(i+4, len(df))].min()
                if move_size > atr * 1.0:  # Reduced to 1.0x
                    ob_top = df['Close'].iloc[i]
                    ob_bottom = df['Open'].iloc[i]
                    ob_height = ob_top - ob_bottom
                    
                    # Store OB data
                    result['bearish_ob'] = True
                    result['bearish_ob_top'] = ob_top
                    result['bearish_ob_bottom'] = ob_bottom
                    
                    # Check if price is AT or NEAR this OB
                    # AT: Price is inside the OB zone
                    if ob_bottom <= current_price <= ob_top:
                        result['at_bearish_ob'] = True
                        break
                    # NEAR (below): Price is within 2% below OB bottom (just left zone)
                    # OB is acting as RESISTANCE above
                    elif ob_bottom * 0.97 <= current_price < ob_bottom:
                        result['near_bearish_ob'] = True
                        break
                    # NEAR (approaching): Price is within 2% above OB top
                    elif ob_top < current_price <= ob_top * 1.02:
                        result['near_bearish_ob'] = True
                        break
        
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

def analyze_market_structure(df: pd.DataFrame, lookback: int = 100) -> Dict:
    """
    Analyze market structure - Higher Highs, Higher Lows, etc.
    
    IMPROVED: Uses EXTENDED lookback for range calculation to capture major moves
    like crashes or rallies, while using shorter lookback for structure analysis.
    
    Args:
        df: OHLCV DataFrame
        lookback: Base lookback for structure (default 100, extended to 4x for range)
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
        # ═══════════════════════════════════════════════════════════════════════
        # EXTENDED LOOKBACK for swing range calculation
        # Use 4x the normal lookback to capture significant moves (crashes/rallies)
        # On 15m chart: 200 candles = ~50 hours = ~2 days
        # ═══════════════════════════════════════════════════════════════════════
        extended_lookback = min(len(df), lookback * 4)  # 4x lookback for range
        df_extended = df.tail(extended_lookback)
        
        # Calculate extended range (captures big moves)
        extended_high = df_extended['High'].max()
        extended_low = df_extended['Low'].min()
        extended_range = extended_high - extended_low
        
        # Also calculate recent range (for structure analysis)
        df_recent = df.tail(lookback) if len(df) > lookback else df
        simple_high = df_recent['High'].max()
        simple_low = df_recent['Low'].min()
        recent_range = simple_high - simple_low
        
        # ═══════════════════════════════════════════════════════════════════════
        # SMART RANGE SELECTION:
        # If extended range is significantly larger (>50% bigger), use it!
        # This catches coins that crashed/rallied but are now consolidating
        # ═══════════════════════════════════════════════════════════════════════
        use_extended = extended_range > recent_range * 1.5
        
        # DEBUG: Print range comparison
        print(f"🔍 SMC DEBUG: Extended={extended_high:.6f}-{extended_low:.6f} ({extended_range:.6f}) | Recent={simple_high:.6f}-{simple_low:.6f} ({recent_range:.6f}) | USE_EXTENDED={use_extended}")
        
        if use_extended:
            # Use extended range - captures major moves
            range_high = extended_high
            range_low = extended_low
        else:
            # Use recent range - normal consolidation
            range_high = simple_high
            range_low = simple_low
        
        # Find swing points with FLEXIBLE detection (3-bar pattern is more common)
        swing_highs = []
        swing_lows = []
        swing_high_prices = []
        swing_low_prices = []
        
        # Use extended data for swing detection to capture major swings
        df_for_swings = df_extended if use_extended else df_recent
        
        # Try with 3-bar pivot first (more sensitive)
        pivot_len = 3
        for i in range(pivot_len, len(df_for_swings) - pivot_len):
            # Swing high: Higher than pivot_len bars on each side
            if df_for_swings['High'].iloc[i] == df_for_swings['High'].iloc[i-pivot_len:i+pivot_len+1].max():
                swing_highs.append(i)
                swing_high_prices.append(df_for_swings['High'].iloc[i])
            # Swing low: Lower than pivot_len bars on each side
            if df_for_swings['Low'].iloc[i] == df_for_swings['Low'].iloc[i-pivot_len:i+pivot_len+1].min():
                swing_lows.append(i)
                swing_low_prices.append(df_for_swings['Low'].iloc[i])
        
        # If not enough swings found, try 2-bar pivot
        if len(swing_high_prices) < 2 or len(swing_low_prices) < 2:
            swing_highs = []
            swing_lows = []
            swing_high_prices = []
            swing_low_prices = []
            pivot_len = 2
            for i in range(pivot_len, len(df_for_swings) - pivot_len):
                if df_for_swings['High'].iloc[i] == df_for_swings['High'].iloc[i-pivot_len:i+pivot_len+1].max():
                    swing_highs.append(i)
                    swing_high_prices.append(df_for_swings['High'].iloc[i])
                if df_for_swings['Low'].iloc[i] == df_for_swings['Low'].iloc[i-pivot_len:i+pivot_len+1].min():
                    swing_lows.append(i)
                    swing_low_prices.append(df_for_swings['Low'].iloc[i])
        
        # Get the swing range (not just last swing!)
        # Use MAX of swing highs and MIN of swing lows for proper range
        if swing_high_prices:
            last_swing_high = max(swing_high_prices)  # Highest swing high
        else:
            last_swing_high = range_high
            
        if swing_low_prices:
            last_swing_low = min(swing_low_prices)  # Lowest swing low
        else:
            last_swing_low = range_low
        
        # ═══════════════════════════════════════════════════════════════════════
        # FINAL VALIDATION: Ensure swings capture the TRUE range
        # If calculated swings miss the extended highs/lows, use extended values
        # ═══════════════════════════════════════════════════════════════════════
        if use_extended:
            # Make sure we include the extreme points from extended range
            if last_swing_high < extended_high * 0.98:  # Missing significant high
                last_swing_high = extended_high
            if last_swing_low > extended_low * 1.02:  # Missing significant low
                last_swing_low = extended_low
        
        # Ensure we have a valid range (high > low)
        if last_swing_high <= last_swing_low:
            # Swings are inverted or same - use simple range
            last_swing_high = range_high
            last_swing_low = range_low
        
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