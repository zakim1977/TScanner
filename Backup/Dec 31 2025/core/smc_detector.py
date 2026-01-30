"""
Smart Money Concepts (SMC) Detection Module
Order Blocks, Fair Value Gaps, Liquidity Sweeps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def get_htf_for_entry(timeframe: str, trade_mode: str = 'DayTrade') -> str:
    """
    Get the Higher Timeframe (HTF) for direction and TP targets
    based on entry timeframe and trade mode.
    
    Professional SMC approach:
    - Entry TF: Precision entry at LTF Order Blocks
    - HTF: Direction bias + TP targets at HTF Order Blocks
    
    Args:
        timeframe: The entry/analysis timeframe (e.g., '15m', '1h')
        trade_mode: Trading mode ('Scalp', 'DayTrade', 'Swing', 'Investment')
    
    Returns:
        HTF string for fetching higher timeframe data
    """
    # Timeframe hierarchy mapping
    htf_map = {
        # Scalp mode
        ('1m', 'Scalp'): '15m',
        ('5m', 'Scalp'): '1h',
        ('15m', 'Scalp'): '1h',
        
        # Day Trade mode
        ('1m', 'DayTrade'): '15m',
        ('5m', 'DayTrade'): '1h',
        ('15m', 'DayTrade'): '4h',
        ('1h', 'DayTrade'): '4h',
        
        # Swing mode
        ('15m', 'Swing'): '4h',
        ('1h', 'Swing'): '4h',
        ('4h', 'Swing'): '1d',
        ('1d', 'Swing'): '1w',
        
        # Investment mode
        ('1d', 'Investment'): '1w',
        ('1w', 'Investment'): '1M',
    }
    
    # Normalize timeframe
    tf_lower = timeframe.lower()
    
    # Try exact match first
    key = (tf_lower, trade_mode)
    if key in htf_map:
        return htf_map[key]
    
    # Fallback: use general mapping based on timeframe only
    general_htf = {
        '1m': '15m',
        '5m': '1h',
        '15m': '4h',
        '1h': '4h',
        '4h': '1d',
        '1d': '1w',
        '1w': '1M',
    }
    
    return general_htf.get(tf_lower, '4h')  # Default to 4h


def get_htf_candle_count(htf: str) -> int:
    """Get appropriate candle count for HTF analysis"""
    counts = {
        '15m': 200,
        '1h': 150,
        '4h': 100,
        '1d': 60,
        '1w': 30,
        '1M': 24,
    }
    return counts.get(htf, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_order_blocks(df: pd.DataFrame, lookback: int = 100) -> Dict:
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
        
        # ═══════════════════════════════════════════════════════════════════
        # ORDER BLOCK DETECTION - EXACT LuxAlgo Implementation
        # ═══════════════════════════════════════════════════════════════════
        
        length = 10  # Swing lookback (same as LuxAlgo default)
        
        # Step 1: Find swing highs and swing lows
        swing_highs = []
        swing_lows = []
        
        for i in range(length, len(df) - length):
            # Swing High: high[i] > highest of previous bars
            upper = df['High'].iloc[i-length:i].max()
            if df['High'].iloc[i] > upper:
                swing_highs.append({'price': df['High'].iloc[i], 'index': i, 'crossed': False})
            
            # Swing Low: low[i] < lowest of previous bars
            lower = df['Low'].iloc[i-length:i].min()
            if df['Low'].iloc[i] < lower:
                swing_lows.append({'price': df['Low'].iloc[i], 'index': i, 'crossed': False})
        
        # Step 2: Find Bullish OBs (when price breaks above swing high)
        bullish_obs = []
        
        for sh in swing_highs:
            if sh['crossed']:
                continue
            
            # Check if any candle after swing high closes above it
            for i in range(sh['index'] + 1, len(df)):
                if df['Close'].iloc[i] > sh['price']:
                    sh['crossed'] = True
                    
                    # Find the LOWEST candle between swing high and breakout
                    min_low = float('inf')
                    ob_idx = sh['index']
                    
                    for j in range(sh['index'], i):
                        if df['Low'].iloc[j] < min_low:
                            min_low = df['Low'].iloc[j]
                            ob_idx = j
                    
                    # OB zone is the candle with lowest low
                    ob_top = df['High'].iloc[ob_idx]
                    ob_bottom = df['Low'].iloc[ob_idx]
                    
                    # Use candle body if bearish candle
                    if df['Close'].iloc[ob_idx] < df['Open'].iloc[ob_idx]:
                        ob_top = df['Open'].iloc[ob_idx]
                        ob_bottom = df['Close'].iloc[ob_idx]
                    
                    bullish_obs.append({
                        'top': ob_top,
                        'bottom': ob_bottom,
                        'index': ob_idx
                    })
                    break
        
        # Step 3: Find Bearish OBs (when price breaks below swing low)
        bearish_obs = []
        
        for sl in swing_lows:
            if sl['crossed']:
                continue
            
            # Check if any candle after swing low closes below it
            for i in range(sl['index'] + 1, len(df)):
                if df['Close'].iloc[i] < sl['price']:
                    sl['crossed'] = True
                    
                    # Find the HIGHEST candle between swing low and breakdown
                    max_high = 0
                    ob_idx = sl['index']
                    
                    for j in range(sl['index'], i):
                        if df['High'].iloc[j] > max_high:
                            max_high = df['High'].iloc[j]
                            ob_idx = j
                    
                    # OB zone is the candle with highest high
                    ob_top = df['High'].iloc[ob_idx]
                    ob_bottom = df['Low'].iloc[ob_idx]
                    
                    # Use candle body if bullish candle
                    if df['Close'].iloc[ob_idx] > df['Open'].iloc[ob_idx]:
                        ob_top = df['Close'].iloc[ob_idx]
                        ob_bottom = df['Open'].iloc[ob_idx]
                    
                    bearish_obs.append({
                        'top': ob_top,
                        'bottom': ob_bottom,
                        'index': ob_idx
                    })
                    break
        
        # Step 4: Pick nearest UNBROKEN Bullish OB below current price
        for ob in reversed(bullish_obs):
            if ob['top'] < current_price:
                # Check if broken (price closed below OB bottom)
                is_broken = False
                for i in range(ob['index'] + 1, len(df)):
                    if df['Close'].iloc[i] < ob['bottom']:
                        is_broken = True
                        break
                
                if not is_broken:
                    result['bullish_ob'] = True
                    result['bullish_ob_top'] = ob['top']
                    result['bullish_ob_bottom'] = ob['bottom']
                    if ob['top'] * 1.03 > current_price:
                        result['near_bullish_ob'] = True
                    break
        
        # Step 5: Pick nearest UNBROKEN Bearish OB above current price
        for ob in reversed(bearish_obs):
            if ob['bottom'] > current_price:
                # Check if broken (price closed above OB top)
                is_broken = False
                for i in range(ob['index'] + 1, len(df)):
                    if df['Close'].iloc[i] > ob['top']:
                        is_broken = True
                        break
                
                if not is_broken:
                    result['bearish_ob'] = True
                    result['bearish_ob_top'] = ob['top']
                    result['bearish_ob_bottom'] = ob['bottom']
                    if ob['bottom'] * 0.97 < current_price:
                        result['near_bearish_ob'] = True
                    break
        
        # ═══════════════════════════════════════════════════════════════════
        # OVERLAP CHECK: If both OBs exist and overlap OR are too close, keep only one
        # ═══════════════════════════════════════════════════════════════════
        if result['bullish_ob'] and result['bearish_ob']:
            bull_top = result['bullish_ob_top']
            bull_bottom = result['bullish_ob_bottom']
            bear_top = result['bearish_ob_top']
            bear_bottom = result['bearish_ob_bottom']
            
            # Check for direct overlap (zones intersect)
            zones_overlap = not (bull_top < bear_bottom or bear_top < bull_bottom)
            
            # Check for close proximity using ATR (dynamic, not hardcoded!)
            # If gap between zones is less than 1x ATR, they're too close
            gap_between = abs(bear_bottom - bull_top) if bear_bottom > bull_top else abs(bull_bottom - bear_top)
            zones_too_close = gap_between < atr  # Dynamic: within 1 ATR = too close
            
            if zones_overlap or zones_too_close:
                # Keep only the one relevant to current price position
                bull_mid = (bull_top + bull_bottom) / 2
                bear_mid = (bear_top + bear_bottom) / 2
                
                dist_to_bull = abs(current_price - bull_mid)
                dist_to_bear = abs(current_price - bear_mid)
                
                # Decision logic: keep the OB that makes sense for trading
                # If price is ABOVE both → keep bullish OB (support below)
                # If price is BELOW both → keep bearish OB (resistance above)
                # If price is BETWEEN → keep the closer one
                
                if current_price > max(bull_top, bear_top):
                    # Price above both - keep bullish (support below us)
                    result['bearish_ob'] = False
                    result['bearish_ob_top'] = 0
                    result['bearish_ob_bottom'] = 0
                    result['at_bearish_ob'] = False
                    result['near_bearish_ob'] = False
                elif current_price < min(bull_bottom, bear_bottom):
                    # Price below both - keep bearish (resistance above us)
                    result['bullish_ob'] = False
                    result['bullish_ob_top'] = 0
                    result['bullish_ob_bottom'] = 0
                    result['at_bullish_ob'] = False
                    result['near_bullish_ob'] = False
                else:
                    # Price is near/between zones - keep the closer one
                    if dist_to_bull < dist_to_bear:
                        result['bearish_ob'] = False
                        result['bearish_ob_top'] = 0
                        result['bearish_ob_bottom'] = 0
                        result['at_bearish_ob'] = False
                        result['near_bearish_ob'] = False
                    else:
                        result['bullish_ob'] = False
                        result['bullish_ob_top'] = 0
                        result['bullish_ob_bottom'] = 0
                        result['at_bullish_ob'] = False
                        result['near_bullish_ob'] = False
        
        return result
        
    except Exception as e:
        return result


def detect_all_order_blocks(df: pd.DataFrame, current_price: float, lookback: int = 50, max_obs: int = 5) -> Dict:
    """
    Detect ALL Order Blocks for TP targeting (HTF analysis)
    
    Returns lists of bullish and bearish OBs sorted by distance from current price.
    Used for structure-based TP calculation.
    
    Args:
        df: OHLCV DataFrame (HTF data)
        current_price: Current price to measure distance from
        lookback: How many candles to scan
        max_obs: Maximum OBs to return per direction
        
    Returns:
        Dict with 'bullish_obs' and 'bearish_obs' lists, each containing:
        {'top': float, 'bottom': float, 'mid': float, 'distance_pct': float}
    """
    result = {
        'bullish_obs': [],  # Support zones below price (for SHORT TPs)
        'bearish_obs': [],  # Resistance zones above price (for LONG TPs)
        'htf_swing_high': 0,
        'htf_swing_low': 0,
    }
    
    if df is None or len(df) < 10:
        return result
    
    try:
        # Calculate ATR for significance check
        df_copy = df.copy()
        df_copy['TR'] = pd.concat([
            df_copy['High'] - df_copy['Low'],
            abs(df_copy['High'] - df_copy['Close'].shift(1)),
            abs(df_copy['Low'] - df_copy['Close'].shift(1))
        ], axis=1).max(axis=1)
        atr = df_copy['TR'].rolling(14).mean().iloc[-1]
        
        if pd.isna(atr) or atr == 0:
            atr = (df['High'].iloc[-20:] - df['Low'].iloc[-20:]).mean()
        
        # HTF Swing High/Low
        result['htf_swing_high'] = df['High'].iloc[-lookback:].max()
        result['htf_swing_low'] = df['Low'].iloc[-lookback:].min()
        
        bullish_obs = []
        bearish_obs = []
        
        # Scan for ALL OBs with PROPER threshold (significant moves only)
        # HTF OBs must have strong moves after them to be significant
        for i in range(len(df) - 3, max(len(df) - lookback, 3), -1):
            try:
                # Bullish OB: Bearish candle followed by strong bullish move
                if df['Close'].iloc[i] < df['Open'].iloc[i]:
                    future_high = df['High'].iloc[i+1:min(i+4, len(df))].max()
                    move_size = future_high - df['Close'].iloc[i]
                    
                    # HTF OBs must have SIGNIFICANT moves (at least 1.5x ATR)
                    if move_size > atr * 1.5:
                        ob_top = df['Open'].iloc[i]
                        ob_bottom = df['Close'].iloc[i]
                        
                        if ob_top > ob_bottom:  # Valid OB
                            ob_mid = (ob_top + ob_bottom) / 2
                            
                            # Include if below OR containing current price (support zone)
                            if ob_top < current_price:
                                # OB is fully below price
                                distance_pct = ((current_price - ob_mid) / current_price) * 100
                                bullish_obs.append({
                                    'top': ob_top,
                                    'bottom': ob_bottom,
                                    'mid': ob_mid,
                                    'distance_pct': distance_pct,
                                    'contains_price': False
                                })
                            elif ob_bottom <= current_price <= ob_top:
                                # Price is INSIDE this OB!
                                bullish_obs.append({
                                    'top': ob_top,
                                    'bottom': ob_bottom,
                                    'mid': ob_mid,
                                    'distance_pct': 0,  # Inside = 0 distance
                                    'contains_price': True
                                })
                
                # Bearish OB: Bullish candle followed by strong bearish move
                if df['Close'].iloc[i] > df['Open'].iloc[i]:
                    future_low = df['Low'].iloc[i+1:min(i+4, len(df))].min()
                    move_size = df['Close'].iloc[i] - future_low
                    
                    # HTF OBs must have SIGNIFICANT moves (at least 1.5x ATR)
                    if move_size > atr * 1.5:
                        ob_top = df['Close'].iloc[i]
                        ob_bottom = df['Open'].iloc[i]
                        
                        if ob_top > ob_bottom:  # Valid OB
                            ob_mid = (ob_top + ob_bottom) / 2
                            
                            # Include if above OR containing current price (resistance zone)
                            if ob_bottom > current_price:
                                # OB is fully above price
                                distance_pct = ((ob_mid - current_price) / current_price) * 100
                                bearish_obs.append({
                                    'top': ob_top,
                                    'bottom': ob_bottom,
                                    'mid': ob_mid,
                                    'distance_pct': distance_pct,
                                    'contains_price': False
                                })
                            elif ob_bottom <= current_price <= ob_top:
                                # Price is INSIDE this OB!
                                bearish_obs.append({
                                    'top': ob_top,
                                    'bottom': ob_bottom,
                                    'mid': ob_mid,
                                    'distance_pct': 0,  # Inside = 0 distance
                                    'contains_price': True
                                })
            except:
                continue
        
        # Sort by distance and limit
        result['bullish_obs'] = sorted(bullish_obs, key=lambda x: x['distance_pct'])[:max_obs]
        result['bearish_obs'] = sorted(bearish_obs, key=lambda x: x['distance_pct'])[:max_obs]
        
        # ═══════════════════════════════════════════════════════════════════
        # HTF CONFLICT DETECTION - Is price INSIDE an HTF OB?
        # ═══════════════════════════════════════════════════════════════════
        result['inside_htf_bearish_ob'] = False
        result['inside_htf_bullish_ob'] = False
        result['htf_conflict_warning'] = None
        
        # Check bearish OBs for containing price
        for ob in bearish_obs:
            if ob.get('contains_price', False):
                result['inside_htf_bearish_ob'] = True
                result['htf_conflict_warning'] = f"Price INSIDE HTF Bearish OB (${ob['bottom']:.4f}-${ob['top']:.4f}) - RESISTANCE OVERHEAD for LONGS"
                break
        
        # Check bullish OBs for containing price
        if not result['htf_conflict_warning']:
            for ob in bullish_obs:
                if ob.get('contains_price', False):
                    result['inside_htf_bullish_ob'] = True
                    result['htf_conflict_warning'] = f"Price INSIDE HTF Bullish OB (${ob['bottom']:.4f}-${ob['top']:.4f}) - SUPPORT BELOW for SHORTS"
                    break
        
        return result
        
    except Exception as e:
        return result

# ═══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE GAPS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Detect Fair Value Gaps - LuxAlgo Implementation
    
    Bullish FVG: low > high[2] AND close[1] > high[2]
    Bearish FVG: high < low[2] AND close[1] < low[2]
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
        
        # Collect all FVGs
        bullish_fvgs = []
        bearish_fvgs = []
        
        for i in range(2, len(df) - 1):
            # Bullish FVG: low[i] > high[i-2] AND close[i-1] > high[i-2]
            if df['Low'].iloc[i] > df['High'].iloc[i-2] and df['Close'].iloc[i-1] > df['High'].iloc[i-2]:
                fvg_top = df['Low'].iloc[i]
                fvg_bottom = df['High'].iloc[i-2]
                bullish_fvgs.append({'top': fvg_top, 'bottom': fvg_bottom, 'index': i})
            
            # Bearish FVG: high[i] < low[i-2] AND close[i-1] < low[i-2]
            if df['High'].iloc[i] < df['Low'].iloc[i-2] and df['Close'].iloc[i-1] < df['Low'].iloc[i-2]:
                fvg_top = df['Low'].iloc[i-2]
                fvg_bottom = df['High'].iloc[i]
                bearish_fvgs.append({'top': fvg_top, 'bottom': fvg_bottom, 'index': i})
        
        # Find nearest UNMITIGATED Bullish FVG below current price
        for fvg in reversed(bullish_fvgs):
            if fvg['top'] < current_price:
                # Check if mitigated
                is_mitigated = False
                for j in range(fvg['index'] + 1, len(df)):
                    if df['Close'].iloc[j] < fvg['bottom']:
                        is_mitigated = True
                        break
                
                if not is_mitigated:
                    result['bullish_fvg'] = True
                    result['bullish_fvg_high'] = fvg['top']
                    result['bullish_fvg_low'] = fvg['bottom']
                    if fvg['bottom'] <= current_price <= fvg['top']:
                        result['at_bullish_fvg'] = True
                    break
        
        # Find nearest UNMITIGATED Bearish FVG above current price
        for fvg in reversed(bearish_fvgs):
            if fvg['bottom'] > current_price:
                # Check if mitigated
                is_mitigated = False
                for j in range(fvg['index'] + 1, len(df)):
                    if df['Close'].iloc[j] > fvg['top']:
                        is_mitigated = True
                        break
                
                if not is_mitigated:
                    result['bearish_fvg'] = True
                    result['bearish_fvg_high'] = fvg['top']
                    result['bearish_fvg_low'] = fvg['bottom']
                    if fvg['bottom'] <= current_price <= fvg['top']:
                        result['at_bearish_fvg'] = True
                    break
        
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