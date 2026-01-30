"""
Explosion Detector - Detect coins ready to make big moves
==========================================================
Based on professional trading concepts:
1. Volatility Compression → Expansion (Squeeze Loading)
2. Liquidity Sweep Detection
3. Time-based ignition (session timing)

These signals help identify WHEN a move is about to happen.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from ta.volatility import BollingerBands, AverageTrueRange


# ═══════════════════════════════════════════════════════════════════════════════
# 1. VOLATILITY COMPRESSION DETECTOR (Squeeze Loading)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_squeeze_loading(
    df: pd.DataFrame,
    whale_data: Optional[Dict] = None,
    oi_change_pct: float = 0,
    lookback: int = 100
) -> Dict:
    """
    Detect volatility compression - price about to explode.
    
    Squeeze Loading = BB width in bottom 10-15% + OI increasing + Whales aligned
    
    Returns:
        Dict with squeeze status and direction
    """
    result = {
        'squeeze_loading': False,
        'squeeze_percentile': 0,
        'squeeze_direction': None,  # 'LONG' or 'SHORT'
        'squeeze_score': 0,
        'bb_width_pct': 0,
        'conditions_met': []
    }
    
    if df is None or len(df) < lookback:
        return result
    
    try:
        # Calculate Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        # Calculate BB width as percentage of mid
        bb_width = (bb_upper - bb_lower) / bb_mid
        current_width = bb_width.iloc[-1]
        
        # Calculate percentile of current width vs last N bars
        recent_widths = bb_width.iloc[-lookback:].dropna()
        if len(recent_widths) > 10:
            percentile = (recent_widths < current_width).sum() / len(recent_widths) * 100
        else:
            percentile = 50
        
        result['bb_width_pct'] = round(current_width * 100, 2)
        result['squeeze_percentile'] = round(percentile, 1)
        
        # ─────────────────────────────────────────────────────────────────────
        # SQUEEZE LOADING CONDITIONS
        # ─────────────────────────────────────────────────────────────────────
        conditions = []
        score = 0
        
        # Condition 1: BB width in bottom 15%
        if percentile <= 15:
            conditions.append("BB_COMPRESSED")
            score += 30
        elif percentile <= 25:
            conditions.append("BB_TIGHT")
            score += 15
        
        # Condition 2: OI increasing (building positions)
        if oi_change_pct > 2:
            conditions.append("OI_BUILDING")
            score += 25
        elif oi_change_pct > 0:
            conditions.append("OI_POSITIVE")
            score += 10
        
        # Condition 3: Whale positioning aligned
        whale_direction = None
        if whale_data:
            whale_long = whale_data.get('whale_long_pct', 50)
            whale_short = whale_data.get('whale_short_pct', 50)
            
            if whale_long >= 60:
                conditions.append("WHALES_LONG")
                whale_direction = 'LONG'
                score += 25
            elif whale_short >= 60:
                conditions.append("WHALES_SHORT")
                whale_direction = 'SHORT'
                score += 25
            elif whale_long >= 55:
                conditions.append("WHALES_LEAN_LONG")
                whale_direction = 'LONG'
                score += 15
            elif whale_short >= 55:
                conditions.append("WHALES_LEAN_SHORT")
                whale_direction = 'SHORT'
                score += 15
        
        # Condition 4: Price near BB mid (coiled)
        current_price = df['Close'].iloc[-1]
        current_mid = bb_mid.iloc[-1]
        distance_from_mid = abs(current_price - current_mid) / current_mid * 100
        
        if distance_from_mid < 1:
            conditions.append("PRICE_COILED")
            score += 20
        elif distance_from_mid < 2:
            conditions.append("PRICE_NEAR_MID")
            score += 10
        
        # ─────────────────────────────────────────────────────────────────────
        # DETERMINE SQUEEZE STATUS
        # ─────────────────────────────────────────────────────────────────────
        result['conditions_met'] = conditions
        result['squeeze_score'] = score
        
        # Squeeze Loading: compressed + at least one more condition
        if percentile <= 15 and score >= 50:
            result['squeeze_loading'] = True
            result['squeeze_direction'] = whale_direction
        elif percentile <= 25 and score >= 70:
            result['squeeze_loading'] = True
            result['squeeze_direction'] = whale_direction
        
        return result
        
    except Exception as e:
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LIQUIDITY SWEEP DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweep(
    df: pd.DataFrame,
    lookback: int = 20,
    oi_change_pct: float = 0
) -> Dict:
    """
    Detect liquidity sweeps - stop hunts that precede reversals.
    
    Sweep High = high > rolling_max(high, n=20) then reverses
    Sweep Low = low < rolling_min(low, n=20) then reverses
    
    Most instant moves happen right after:
    - Equal highs taken
    - Equal lows taken
    - Session sweep
    """
    result = {
        'sweep_detected': False,
        'sweep_type': None,  # 'SWEEP_LOW' or 'SWEEP_HIGH'
        'sweep_reclaimed': False,
        'entry_zone': None,  # 'IMMEDIATE' if reclaimed
        'sweep_level': 0,
        'conditions_met': []
    }
    
    if df is None or len(df) < lookback + 5:
        return result
    
    try:
        # Calculate rolling highs and lows (the liquidity levels)
        rolling_high = df['High'].rolling(window=lookback).max()
        rolling_low = df['Low'].rolling(window=lookback).min()
        
        # Check last 3 candles for sweep
        for i in range(-3, 0):
            idx = len(df) + i
            if idx < lookback:
                continue
            
            current_high = df['High'].iloc[idx]
            current_low = df['Low'].iloc[idx]
            current_close = df['Close'].iloc[idx]
            prev_rolling_high = rolling_high.iloc[idx - 1]
            prev_rolling_low = rolling_low.iloc[idx - 1]
            
            # ─────────────────────────────────────────────────────────────────
            # SWEEP LOW (Bullish signal - whales cleared stops below)
            # ─────────────────────────────────────────────────────────────────
            if current_low < prev_rolling_low:
                # Swept the lows
                result['sweep_detected'] = True
                result['sweep_type'] = 'SWEEP_LOW'
                result['sweep_level'] = prev_rolling_low
                result['conditions_met'].append('LOWS_TAKEN')
                
                # Check if reclaimed (closed back above the sweep level)
                if current_close > prev_rolling_low:
                    result['sweep_reclaimed'] = True
                    result['conditions_met'].append('RECLAIMED')
                    
                    # Check OI spike (positions being built after sweep)
                    if oi_change_pct > 3:
                        result['conditions_met'].append('OI_SPIKE')
                        result['entry_zone'] = 'IMMEDIATE'
                    elif oi_change_pct > 0:
                        result['conditions_met'].append('OI_POSITIVE')
                        result['entry_zone'] = 'WAIT_CONFIRM'
                break
            
            # ─────────────────────────────────────────────────────────────────
            # SWEEP HIGH (Bearish signal - whales cleared stops above)
            # ─────────────────────────────────────────────────────────────────
            if current_high > prev_rolling_high:
                # Swept the highs
                result['sweep_detected'] = True
                result['sweep_type'] = 'SWEEP_HIGH'
                result['sweep_level'] = prev_rolling_high
                result['conditions_met'].append('HIGHS_TAKEN')
                
                # Check if reclaimed (closed back below the sweep level)
                if current_close < prev_rolling_high:
                    result['sweep_reclaimed'] = True
                    result['conditions_met'].append('RECLAIMED')
                    
                    # Check OI spike
                    if oi_change_pct > 3:
                        result['conditions_met'].append('OI_SPIKE')
                        result['entry_zone'] = 'IMMEDIATE'
                    elif oi_change_pct > 0:
                        result['conditions_met'].append('OI_POSITIVE')
                        result['entry_zone'] = 'WAIT_CONFIRM'
                break
        
        return result
        
    except Exception as e:
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TIME-BASED IGNITION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def detect_session_ignition(current_time: Optional[datetime] = None) -> Dict:
    """
    Detect if we're in a high-probability session window.
    
    Most crypto expansions start at:
    - London open (08:00-10:00 UTC)
    - NY open (13:00-15:00 UTC)
    - NY lunch fakeout (16:00-18:00 UTC)
    - Funding reset windows (00:00, 08:00, 16:00 UTC)
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    # Make sure we have UTC time
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    
    hour = current_time.hour
    minute = current_time.minute
    
    result = {
        'session': 'OFF_HOURS',
        'ignition_window': False,
        'score_boost': 0,
        'funding_window': False,
        'description': ''
    }
    
    # ─────────────────────────────────────────────────────────────────────────
    # FUNDING RESET WINDOWS (High volatility around these times)
    # ─────────────────────────────────────────────────────────────────────────
    funding_hours = [0, 8, 16]  # UTC
    for fh in funding_hours:
        if abs(hour - fh) == 0 and minute <= 30:
            result['funding_window'] = True
            result['score_boost'] += 5
            result['description'] = f"Funding reset window ({fh}:00 UTC)"
    
    # ─────────────────────────────────────────────────────────────────────────
    # SESSION WINDOWS
    # ─────────────────────────────────────────────────────────────────────────
    
    # London Open (08:00-10:00 UTC) - First major session
    if 8 <= hour < 10:
        result['session'] = 'LONDON_OPEN'
        result['ignition_window'] = True
        result['score_boost'] = 15
        result['description'] = 'London Open - High probability window'
    
    # London/NY Overlap (13:00-15:00 UTC) - Highest volume period
    elif 13 <= hour < 15:
        result['session'] = 'NY_OPEN'
        result['ignition_window'] = True
        result['score_boost'] = 20
        result['description'] = 'NY Open - Peak volume window'
    
    # NY Session (15:00-18:00 UTC)
    elif 15 <= hour < 18:
        result['session'] = 'NY_SESSION'
        result['ignition_window'] = True
        result['score_boost'] = 10
        result['description'] = 'NY Session - Active trading'
    
    # NY Lunch Fakeout (18:00-20:00 UTC) - Often reversals
    elif 18 <= hour < 20:
        result['session'] = 'NY_LUNCH'
        result['ignition_window'] = False
        result['score_boost'] = -5
        result['description'] = 'NY Lunch - Watch for fakeouts'
    
    # Asia Open (00:00-02:00 UTC)
    elif 0 <= hour < 2:
        result['session'] = 'ASIA_OPEN'
        result['ignition_window'] = True
        result['score_boost'] = 10
        result['description'] = 'Asia Open - New daily range'
    
    # Asia Session (02:00-08:00 UTC) - Usually consolidation
    elif 2 <= hour < 8:
        result['session'] = 'ASIA'
        result['ignition_window'] = False
        result['score_boost'] = 0
        result['description'] = 'Asia Session - Often ranges'
    
    # Late NY / Pre-Asia (20:00-00:00 UTC)
    else:
        result['session'] = 'TRANSITION'
        result['ignition_window'] = False
        result['score_boost'] = 0
        result['description'] = 'Transition period'
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMBINED EXPLOSION READINESS SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_explosion_readiness(
    df: pd.DataFrame,
    whale_data: Optional[Dict] = None,
    oi_change_pct: float = 0,
    current_time: Optional[datetime] = None
) -> Dict:
    """
    Calculate overall explosion readiness score.
    
    Combines:
    - Squeeze loading (compression)
    - Liquidity sweep
    - Session timing
    
    Returns score 0-100 and recommendation.
    """
    result = {
        'explosion_score': 0,
        'explosion_ready': False,
        'direction': None,
        'signals': [],
        'squeeze': {},
        'sweep': {},
        'session': {},
        'recommendation': ''
    }
    
    try:
        # Get individual signals
        squeeze = detect_squeeze_loading(df, whale_data, oi_change_pct)
        sweep = detect_liquidity_sweep(df, 20, oi_change_pct)
        session = detect_session_ignition(current_time)
        
        result['squeeze'] = squeeze
        result['sweep'] = sweep
        result['session'] = session
        
        # Build score
        score = 0
        signals = []
        direction = None
        
        # ─────────────────────────────────────────────────────────────────────
        # SQUEEZE LOADING (up to 40 points)
        # ─────────────────────────────────────────────────────────────────────
        if squeeze['squeeze_loading']:
            score += 40
            signals.append(f"🔥 SQUEEZE LOADING ({squeeze['squeeze_percentile']:.0f}% BB)")
            direction = squeeze['squeeze_direction']
        elif squeeze['squeeze_score'] >= 30:
            score += 20
            signals.append(f"⚡ Compression building ({squeeze['squeeze_percentile']:.0f}% BB)")
        
        # ─────────────────────────────────────────────────────────────────────
        # LIQUIDITY SWEEP (up to 35 points)
        # ─────────────────────────────────────────────────────────────────────
        if sweep['sweep_detected'] and sweep['sweep_reclaimed']:
            if sweep['entry_zone'] == 'IMMEDIATE':
                score += 35
                sweep_dir = 'LONG' if sweep['sweep_type'] == 'SWEEP_LOW' else 'SHORT'
                signals.append(f"🎯 LIQUIDITY CLEARED + RECLAIMED → {sweep_dir}")
                if direction is None:
                    direction = sweep_dir
            else:
                score += 20
                signals.append(f"⚠️ Sweep detected, waiting confirm")
        elif sweep['sweep_detected']:
            score += 10
            signals.append(f"👀 {sweep['sweep_type']} in progress")
        
        # ─────────────────────────────────────────────────────────────────────
        # SESSION TIMING (up to 20 points)
        # ─────────────────────────────────────────────────────────────────────
        if session['ignition_window']:
            score += session['score_boost']
            signals.append(f"🕐 {session['session']}: {session['description']}")
        
        if session['funding_window']:
            signals.append(f"💰 Funding reset window")
        
        # ─────────────────────────────────────────────────────────────────────
        # FINAL ASSESSMENT
        # ─────────────────────────────────────────────────────────────────────
        result['explosion_score'] = min(score, 100)
        result['signals'] = signals
        result['direction'] = direction
        
        if score >= 70:
            result['explosion_ready'] = True
            result['recommendation'] = f"🚀 HIGH PROBABILITY MOVE INCOMING - {direction or 'Watch for direction'}"
        elif score >= 50:
            result['recommendation'] = f"⚡ Building energy - Monitor closely"
        elif score >= 30:
            result['recommendation'] = f"👀 Some compression - Not ready yet"
        else:
            result['recommendation'] = f"😴 No immediate catalyst"
        
        return result
        
    except Exception as e:
        result['recommendation'] = f"Error: {str(e)}"
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STATE MACHINE - Market Phase Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_market_state(
    df: pd.DataFrame,
    whale_data: Optional[Dict] = None,
    oi_change_pct: float = 0
) -> Dict:
    """
    Detect current market state (state machine approach).
    
    States:
    - ACCUMULATION: Whales building, low volatility
    - COMPRESSION: Tight range, building pressure
    - LIQUIDITY_CLEAR: Stops taken, reversal incoming
    - IGNITION: Breakout starting
    - EXPANSION: Trend in progress
    - DISTRIBUTION: Whales exiting
    
    You enter ONLY in: IGNITION, early EXPANSION
    """
    result = {
        'state': 'UNKNOWN',
        'entry_valid': False,
        'state_score': 0,
        'description': ''
    }
    
    if df is None or len(df) < 50:
        return result
    
    try:
        # Get supporting signals
        squeeze = detect_squeeze_loading(df, whale_data, oi_change_pct)
        sweep = detect_liquidity_sweep(df, 20, oi_change_pct)
        
        # Calculate recent volatility trend
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        current_width = bb_width.iloc[-1]
        prev_width = bb_width.iloc[-5:-1].mean()
        width_expanding = current_width > prev_width * 1.1
        
        # Price momentum
        price_change_5 = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
        
        # ─────────────────────────────────────────────────────────────────────
        # STATE DETECTION
        # ─────────────────────────────────────────────────────────────────────
        
        # LIQUIDITY_CLEAR: Just swept and reclaimed
        if sweep['sweep_detected'] and sweep['sweep_reclaimed']:
            result['state'] = 'LIQUIDITY_CLEAR'
            result['entry_valid'] = True
            result['state_score'] = 85
            result['description'] = 'Stops cleared, reversal zone - ENTRY VALID'
        
        # IGNITION: Compression releasing, volatility expanding
        elif squeeze['squeeze_loading'] and width_expanding:
            result['state'] = 'IGNITION'
            result['entry_valid'] = True
            result['state_score'] = 90
            result['description'] = 'Squeeze releasing - PRIME ENTRY'
        
        # EXPANSION: Strong trend in progress
        elif width_expanding and abs(price_change_5) > 3:
            result['state'] = 'EXPANSION'
            result['entry_valid'] = abs(price_change_5) < 6  # Early expansion only
            result['state_score'] = 70 if result['entry_valid'] else 40
            result['description'] = 'Trend in progress' + (' - Still early' if result['entry_valid'] else ' - May be late')
        
        # COMPRESSION: Building pressure
        elif squeeze['squeeze_loading']:
            result['state'] = 'COMPRESSION'
            result['entry_valid'] = False
            result['state_score'] = 60
            result['description'] = 'Pressure building - WAIT for ignition'
        
        # ACCUMULATION: Whales building quietly
        elif whale_data and whale_data.get('whale_long_pct', 50) >= 60 and squeeze['squeeze_percentile'] < 30:
            result['state'] = 'ACCUMULATION'
            result['entry_valid'] = False
            result['state_score'] = 50
            result['description'] = 'Whales accumulating - WAIT'
        
        # DISTRIBUTION: Whales exiting
        elif whale_data and whale_data.get('retail_pct', 50) >= 60 and width_expanding:
            result['state'] = 'DISTRIBUTION'
            result['entry_valid'] = False
            result['state_score'] = 20
            result['description'] = 'Distribution phase - AVOID'
        
        # RANGING: No clear state
        else:
            result['state'] = 'RANGING'
            result['entry_valid'] = False
            result['state_score'] = 30
            result['description'] = 'No clear catalyst - WAIT'
        
        return result
        
    except Exception as e:
        result['description'] = f"Error: {str(e)}"
        return result
