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
# ORDER BLOCKS - ICT/SMC Package Style (More Sensitive)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_order_blocks_ict(df: pd.DataFrame, swing_length: int = 10, trading_mode: str = 'Day Trade', check_mitigation: bool = True) -> Dict:
    """
    Detect Order Blocks - ICT/SMC Package Implementation
    
    More sensitive than LuxAlgo - detects OBs based on:
    1. Swing highs/lows formation
    2. Strong move after the swing (doesn't require full BOS)
    3. Volume confirmation (optional)
    
    Args:
        df: OHLCV DataFrame
        swing_length: Swing detection lookback
        trading_mode: Trading mode for threshold adjustment
        check_mitigation: If False, skip mitigation check (useful for limit entry detection)
    
    Returns ALL unmitigated OBs, not just nearest.
    """
    # Mode-aware swing length - ALIGNED WITH LUXALGO (uses 10)
    # Shorter = more OBs detected, catches recent structure
    mode_swing_length = {
        'Scalp': 5,
        'Day Trade': 8,
        'DayTrade': 8,
        'Swing': 10,      # Was 15 - too long, missing OBs
        'Investment': 12  # Was 20 - too long
    }.get(trading_mode, swing_length)
    
    # Mode-aware ATR multiplier - MUCH LOWER to find more OBs
    # LuxAlgo doesn't use ATR threshold - it uses swing structure only
    atr_multiplier = {
        'Scalp': 0.3,
        'Day Trade': 0.4,
        'DayTrade': 0.4,
        'Swing': 0.5,      # Was 1.0 - way too strict
        'Investment': 0.6  # Was 1.2 - too strict
    }.get(trading_mode, 0.4)
    
    result = {
        'bullish_obs': [],   # Support zones (for entry or SHORT TPs)
        'bearish_obs': [],   # Resistance zones (for LONG TPs)
        'bullish_ob': False,
        'bearish_ob': False,
        'bullish_ob_top': 0,
        'bullish_ob_bottom': 0,
        'bearish_ob_top': 0,
        'bearish_ob_bottom': 0,
        'at_bullish_ob': False,
        'at_bearish_ob': False,
        'debug_stats': {}  # Debug info
    }
    
    if df is None or len(df) < mode_swing_length * 2:
        return result
    
    try:
        current_price = df['Close'].iloc[-1]
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr == 0:
            atr = (df['High'].iloc[-20:] - df['Low'].iloc[-20:]).mean()
        
        min_move = atr * atr_multiplier  # Mode-aware threshold
        
        # Debug counters
        swing_low_count = 0
        bullish_ob_candidates = 0
        bullish_ob_mitigated = 0
        bullish_ob_valid = 0
        
        # Step 1: Detect swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(mode_swing_length, len(df) - mode_swing_length):
            # Swing High: highest in window
            if df['High'].iloc[i] == df['High'].iloc[i-mode_swing_length:i+mode_swing_length+1].max():
                swing_highs.append(i)
            # Swing Low: lowest in window
            if df['Low'].iloc[i] == df['Low'].iloc[i-mode_swing_length:i+mode_swing_length+1].min():
                swing_lows.append(i)
        
        swing_low_count = len(swing_lows)
        
        bullish_obs = []
        bearish_obs = []
        
        # Step 2: Bullish OB - last down candle before up move at swing low
        for sl_idx in swing_lows:
            if sl_idx >= len(df) - 3:
                continue
            
            # Check for strong up move after swing low
            future_high = df['High'].iloc[sl_idx+1:min(sl_idx+6, len(df))].max()
            move_up = future_high - df['Low'].iloc[sl_idx]
            
            if move_up >= min_move:  # Mode-aware threshold
                bullish_ob_candidates += 1
                
                # Find the down candle at or before swing low
                for j in range(sl_idx, max(sl_idx - 3, 0), -1):
                    if df['Close'].iloc[j] < df['Open'].iloc[j]:  # Down candle
                        ob_top = df['Open'].iloc[j]
                        ob_bottom = df['Close'].iloc[j]
                        ob_mid = (ob_top + ob_bottom) / 2
                        
                        # Check if mitigated (LESS STRICT - only if candle body fully below OB)
                        is_mitigated = False
                        if check_mitigation:
                            mitigation_count = 0
                            for k in range(j + 1, len(df)):
                                # Only mitigated if candle BODY (not wick) closed significantly below
                                candle_body_low = min(df['Open'].iloc[k], df['Close'].iloc[k])
                                if candle_body_low < ob_bottom:
                                    mitigation_count += 1
                                    # Need 2+ candles with body below to confirm mitigation
                                    if mitigation_count >= 2:
                                        is_mitigated = True
                                        bullish_ob_mitigated += 1
                                        break
                        
                        if not is_mitigated and ob_top > ob_bottom:
                            bullish_ob_valid += 1
                            distance_pct = abs(current_price - ob_mid) / current_price * 100
                            bullish_obs.append({
                                'top': ob_top,
                                'bottom': ob_bottom,
                                'mid': ob_mid,
                                'index': j,
                                'distance_pct': distance_pct
                            })
                        break
        
        # Step 3: Bearish OB - last up candle before down move at swing high
        for sh_idx in swing_highs:
            if sh_idx >= len(df) - 3:
                continue
            
            # Check for strong down move after swing high
            future_low = df['Low'].iloc[sh_idx+1:min(sh_idx+6, len(df))].min()
            move_down = df['High'].iloc[sh_idx] - future_low
            
            if move_down >= min_move:  # Mode-aware threshold
                # Find the up candle at or before swing high
                for j in range(sh_idx, max(sh_idx - 3, 0), -1):
                    if df['Close'].iloc[j] > df['Open'].iloc[j]:  # Up candle
                        ob_top = df['Close'].iloc[j]
                        ob_bottom = df['Open'].iloc[j]
                        ob_mid = (ob_top + ob_bottom) / 2
                        
                        # Check if mitigated (LESS STRICT - only if multiple candle bodies above)
                        is_mitigated = False
                        if check_mitigation:
                            mitigation_count = 0
                            for k in range(j + 1, len(df)):
                                candle_body_high = max(df['Open'].iloc[k], df['Close'].iloc[k])
                                if candle_body_high > ob_top:
                                    mitigation_count += 1
                                    if mitigation_count >= 2:
                                        is_mitigated = True
                                        break
                        
                        if not is_mitigated and ob_top > ob_bottom:
                            distance_pct = abs(current_price - ob_mid) / current_price * 100
                            bearish_obs.append({
                                'top': ob_top,
                                'bottom': ob_bottom,
                                'mid': ob_mid,
                                'index': j,
                                'distance_pct': distance_pct
                            })
                        break
        
        # Sort by distance from current price
        bullish_obs.sort(key=lambda x: x['distance_pct'])
        bearish_obs.sort(key=lambda x: x['distance_pct'])
        
        # Store all OBs
        result['bullish_obs'] = bullish_obs
        result['bearish_obs'] = bearish_obs
        
        # Store debug stats for inspection (optional)
        result['debug_stats'] = {
            'swing_lows': swing_low_count,
            'swing_highs': len(swing_highs),
            'bullish_candidates': bullish_ob_candidates,
            'bullish_mitigated': bullish_ob_mitigated,
            'bullish_valid': bullish_ob_valid,
            'atr': atr,
            'current_price': current_price
        }
        
        # Set nearest OB for backward compatibility
        # Nearest Bullish OB BELOW current price (support)
        below_price = [ob for ob in bullish_obs if ob['top'] < current_price]
        if below_price:
            nearest = below_price[0]
            result['bullish_ob'] = True
            result['bullish_ob_top'] = nearest['top']
            result['bullish_ob_bottom'] = nearest['bottom']
            if nearest['bottom'] <= current_price <= nearest['top']:
                result['at_bullish_ob'] = True
        
        # Nearest Bearish OB ABOVE current price (resistance)
        above_price = [ob for ob in bearish_obs if ob['bottom'] > current_price]
        if above_price:
            nearest = above_price[0]
            result['bearish_ob'] = True
            result['bearish_ob_top'] = nearest['top']
            result['bearish_ob_bottom'] = nearest['bottom']
            if nearest['bottom'] <= current_price <= nearest['top']:
                result['at_bearish_ob'] = True
        
        return result
        
    except Exception as e:
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCKS - LuxAlgo Style (Original, Strict)
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


def detect_all_order_blocks(df: pd.DataFrame, current_price: float, lookback: int = 50, max_obs: int = 5, trading_mode: str = 'Day Trade') -> Dict:
    """
    Detect ALL Order Blocks for limit entry and TP targeting
    
    Returns lists of bullish and bearish OBs sorted by distance from current price.
    
    Args:
        df: OHLCV DataFrame
        current_price: Current price to measure distance from
        lookback: How many candles to scan
        max_obs: Maximum OBs to return per direction
        trading_mode: Trading mode for ATR threshold adjustment
        
    Returns:
        Dict with 'bullish_obs' and 'bearish_obs' lists
    """
    result = {
        'bullish_obs': [],  # Support zones below price (for LONG limit entry)
        'bearish_obs': [],  # Resistance zones above price (for SHORT limit entry)
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
        
        # ═══════════════════════════════════════════════════════════════
        # MODE-AWARE ATR THRESHOLD - LOWERED TO FIND MORE OBs
        # LuxAlgo doesn't use ATR threshold - we use very low values
        # Scalp: 0.2x - find micro-structure
        # Day Trade: 0.3x - find intraday structure  
        # Swing: 0.4x - find swing structure
        # Investment: 0.5x - find significant structure
        # ═══════════════════════════════════════════════════════════════
        atr_multiplier = {
            'Scalp': 0.2,
            'Day Trade': 0.3,
            'DayTrade': 0.3,
            'Swing': 0.4,
            'Investment': 0.5
        }.get(trading_mode, 0.3)
        
        min_move_threshold = atr * atr_multiplier
        
        # HTF Swing High/Low
        result['htf_swing_high'] = df['High'].iloc[-lookback:].max()
        result['htf_swing_low'] = df['Low'].iloc[-lookback:].min()
        
        bullish_obs = []
        bearish_obs = []
        bullish_candidates = 0  # For debugging
        bearish_candidates = 0  # For debugging
        
        for i in range(len(df) - 3, max(len(df) - lookback, 3), -1):
            try:
                # Bullish OB: Bearish candle followed by bullish move
                if df['Close'].iloc[i] < df['Open'].iloc[i]:
                    future_high = df['High'].iloc[i+1:min(i+4, len(df))].max()
                    move_size = future_high - df['Close'].iloc[i]
                    
                    # Lowered threshold for more OB detection
                    if move_size > min_move_threshold:
                        bullish_candidates += 1
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
                
                # Bearish OB: Bullish candle followed by bearish move
                if df['Close'].iloc[i] > df['Open'].iloc[i]:
                    future_low = df['Low'].iloc[i+1:min(i+4, len(df))].min()
                    move_size = df['Close'].iloc[i] - future_low
                    
                    # Lowered threshold (was 1.5x ATR, now 0.5x)
                    if move_size > min_move_threshold:
                        bearish_candidates += 1
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

def detect_smc(df: pd.DataFrame, use_ict: bool = True, trading_mode: str = 'Day Trade') -> Dict:
    """
    Main function to detect all SMC concepts
    
    Args:
        df: OHLCV DataFrame
        use_ict: If True, use ICT-style detection (more sensitive), else LuxAlgo (strict)
        trading_mode: Trading mode for threshold adjustments
    
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
    
    # Use ICT-style detection (more OBs detected) or LuxAlgo (strict BOS required)
    if use_ict:
        order_blocks = detect_order_blocks_ict(df, trading_mode=trading_mode)
    else:
        order_blocks = detect_order_blocks(df)
    
    return {
        'order_blocks': order_blocks,
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


def get_all_limit_entry_candidates(
    df: pd.DataFrame, 
    current_price: float, 
    direction: str,
    max_distance_pct: float = 3.0,
    smc_data: Dict = None
) -> List[Dict]:
    """
    Collect ALL OB and FVG candidates for limit entry optimization.
    
    This is the SINGLE SOURCE for finding limit entry candidates.
    Called by trade_optimizer to score and select the best one.
    
    Args:
        df: OHLCV DataFrame
        current_price: Current price
        direction: 'LONG' or 'SHORT'
        max_distance_pct: Maximum distance from current price (%)
        smc_data: Pre-computed SMC data (optional, will compute if None)
        
    Returns:
        List of candidates: [{'price', 'type', 'source', 'desc'}, ...]
    """
    candidates = []
    
    if df is None or len(df) < 20:
        return candidates
    
    try:
        # Calculate price boundaries
        if direction == 'LONG':
            min_price = current_price * (1 - max_distance_pct / 100)
            max_price = current_price  # Must be BELOW current
        else:  # SHORT
            min_price = current_price  # Must be ABOVE current
            max_price = current_price * (1 + max_distance_pct / 100)
        
        # ═══════════════════════════════════════════════════════════════
        # SOURCE 1: detect_all_order_blocks (finds multiple OBs)
        # ═══════════════════════════════════════════════════════════════
        try:
            all_obs = detect_all_order_blocks(df, current_price, lookback=50, max_obs=5)
            
            if direction == 'LONG':
                for ob in all_obs.get('bullish_obs', []):
                    ob_price = ob.get('top', 0)
                    if ob_price > 0 and ob_price < current_price and ob_price >= min_price:
                        candidates.append({
                            'price': ob_price,
                            'type': 'OB',
                            'source': 'all_obs',
                            'desc': 'Bullish OB',
                            'zone_bottom': ob.get('bottom', ob_price * 0.99)
                        })
            else:  # SHORT
                for ob in all_obs.get('bearish_obs', []):
                    ob_price = ob.get('bottom', 0)
                    if ob_price > 0 and ob_price > current_price and ob_price <= max_price:
                        candidates.append({
                            'price': ob_price,
                            'type': 'OB',
                            'source': 'all_obs',
                            'desc': 'Bearish OB',
                            'zone_top': ob.get('top', ob_price * 1.01)
                        })
        except Exception:
            pass
        
        # ═══════════════════════════════════════════════════════════════
        # SOURCE 2: Single OB detector (SMC data)
        # ═══════════════════════════════════════════════════════════════
        if smc_data is None:
            smc_data = detect_smc(df, use_ict=True)
        
        ob_data = smc_data.get('order_blocks', {})
        
        if direction == 'LONG':
            bullish_ob_top = ob_data.get('bullish_ob_top', 0)
            if bullish_ob_top > 0 and bullish_ob_top < current_price and bullish_ob_top >= min_price:
                # Check not duplicate
                if not any(abs(c['price'] - bullish_ob_top) / current_price < 0.002 for c in candidates):
                    candidates.append({
                        'price': bullish_ob_top,
                        'type': 'OB',
                        'source': 'smc_single',
                        'desc': 'Bullish OB (nearest)',
                        'zone_bottom': ob_data.get('bullish_ob_bottom', bullish_ob_top * 0.99)
                    })
        else:  # SHORT
            bearish_ob_bottom = ob_data.get('bearish_ob_bottom', 0)
            if bearish_ob_bottom > 0 and bearish_ob_bottom > current_price and bearish_ob_bottom <= max_price:
                if not any(abs(c['price'] - bearish_ob_bottom) / current_price < 0.002 for c in candidates):
                    candidates.append({
                        'price': bearish_ob_bottom,
                        'type': 'OB',
                        'source': 'smc_single',
                        'desc': 'Bearish OB (nearest)',
                        'zone_top': ob_data.get('bearish_ob_top', bearish_ob_bottom * 1.01)
                    })
        
        # ═══════════════════════════════════════════════════════════════
        # SOURCE 3: FVG detector
        # ═══════════════════════════════════════════════════════════════
        fvg_data = smc_data.get('fvg', {})
        
        if direction == 'LONG':
            bullish_fvg = fvg_data.get('bullish_fvg_high', 0)
            if bullish_fvg > 0 and bullish_fvg < current_price and bullish_fvg >= min_price:
                if not any(abs(c['price'] - bullish_fvg) / current_price < 0.002 for c in candidates):
                    candidates.append({
                        'price': bullish_fvg,
                        'type': 'FVG',
                        'source': 'smc_fvg',
                        'desc': 'Bullish FVG',
                        'zone_bottom': fvg_data.get('bullish_fvg_low', bullish_fvg * 0.99)
                    })
        else:  # SHORT
            bearish_fvg = fvg_data.get('bearish_fvg_low', 0)
            if bearish_fvg > 0 and bearish_fvg > current_price and bearish_fvg <= max_price:
                if not any(abs(c['price'] - bearish_fvg) / current_price < 0.002 for c in candidates):
                    candidates.append({
                        'price': bearish_fvg,
                        'type': 'FVG',
                        'source': 'smc_fvg',
                        'desc': 'Bearish FVG',
                        'zone_top': fvg_data.get('bearish_fvg_high', bearish_fvg * 1.01)
                    })
        
        # ═══════════════════════════════════════════════════════════════
        # SOURCE 4: Additional FVG scan (if detect_fvg finds more)
        # ═══════════════════════════════════════════════════════════════
        try:
            all_fvgs = detect_fvg(df, lookback=50)
            
            if direction == 'LONG':
                # Check for bullish FVGs not already found
                for key in ['bullish_fvg_high', 'nearest_bullish_fvg']:
                    fvg_price = all_fvgs.get(key, 0)
                    if fvg_price > 0 and fvg_price < current_price and fvg_price >= min_price:
                        if not any(abs(c['price'] - fvg_price) / current_price < 0.002 for c in candidates):
                            candidates.append({
                                'price': fvg_price,
                                'type': 'FVG',
                                'source': 'fvg_scan',
                                'desc': 'Bullish FVG Zone'
                            })
            else:  # SHORT
                for key in ['bearish_fvg_low', 'nearest_bearish_fvg']:
                    fvg_price = all_fvgs.get(key, 0)
                    if fvg_price > 0 and fvg_price > current_price and fvg_price <= max_price:
                        if not any(abs(c['price'] - fvg_price) / current_price < 0.002 for c in candidates):
                            candidates.append({
                                'price': fvg_price,
                                'type': 'FVG',
                                'source': 'fvg_scan',
                                'desc': 'Bearish FVG Zone'
                            })
        except Exception:
            pass
        
        # Calculate distance for each candidate
        for c in candidates:
            if direction == 'LONG':
                c['distance_pct'] = ((current_price - c['price']) / current_price) * 100
            else:
                c['distance_pct'] = ((c['price'] - current_price) / current_price) * 100
        
        # Sort by distance (nearest first)
        candidates.sort(key=lambda x: x['distance_pct'])
        
    except Exception as e:
        pass
    
    return candidates