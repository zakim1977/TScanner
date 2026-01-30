"""
Liquidity Hunter ML Integration Module

Provides easy-to-use functions for integrating Liquidity Sweep ML predictions
into Scanner and Single Analysis views.

Uses STREAMLIT COMPONENTS ONLY - no raw HTML.

Usage:
    from liquidity_hunter.lh_ml_integration import (
        get_sweep_analysis,
        render_sweep_badge_st,
        render_sweep_section_st
    )
    
    # In Scanner - get quick text badge
    sweep_data = get_sweep_analysis(df, symbol, mode='day_trade', market='crypto')
    badge_text = render_sweep_badge_st(sweep_data)  # Returns string for st.write()
    
    # In Single Analysis - render full section
    render_sweep_section_st(sweep_data)  # Renders directly using st components
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

def get_sweep_analysis(
    df: pd.DataFrame,
    symbol: str,
    mode: str = 'day_trade',
    market: str = 'crypto',
    whale_pct: float = 50.0,
    whale_delta: float = 0.0,
    current_price: float = None
) -> Dict:
    """
    Analyze current price action for liquidity sweeps with ML prediction.
    
    Args:
        df: OHLCV DataFrame (needs at least 50 candles)
        symbol: Trading pair (e.g., 'BTCUSDT')
        mode: 'scalp', 'day_trade', 'swing', 'investment'
        market: 'crypto', 'stock', 'etf'
        whale_pct: Current whale long percentage (50 = neutral)
        whale_delta: Change in whale % over lookback period
        current_price: Current price (uses df close if not provided)
    
    Returns:
        Dict with sweep detection and ML prediction
    """
    result = {
        'sweep_detected': False,
        'sweep_type': None,  # 'LONG' or 'SHORT'
        'sweep_level': None,
        'sweep_depth_atr': 0,
        'ml_quality': 'N/A',
        'ml_probability': 0,
        'ml_confidence': 0,
        'swing_filters': {},
        'factors': {'positive': [], 'negative': []},
        'model_available': False,
        'recommendation': 'NO_SWEEP'
    }
    
    if df is None or len(df) < 50:
        return result
    
    try:
        # Normalize columns
        if 'High' in df.columns:
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        current_price = current_price or closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]
        
        # Calculate ATR
        tr = np.maximum(highs - lows, 
                        np.maximum(np.abs(highs - np.roll(closes, 1)),
                                  np.abs(lows - np.roll(closes, 1))))
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]
        
        # Mode-dependent lookback
        lookback_map = {
            'scalp': 20,
            'day_trade': 30,
            'swing': 50,
            'investment': 50
        }
        lookback = lookback_map.get(mode, 30)
        
        # Find recent swing levels
        window = min(lookback, len(df) - 10)
        recent_lows = lows[-window:-1]
        recent_highs = highs[-window:-1]
        
        # Find swing low (potential LONG sweep target)
        swing_low = None
        swing_low_idx = None
        for i in range(5, len(recent_lows) - 5):
            if recent_lows[i] == min(recent_lows[max(0,i-5):i+5]):
                if swing_low is None or recent_lows[i] < swing_low:
                    swing_low = recent_lows[i]
                    swing_low_idx = i
        
        # Find swing high (potential SHORT sweep target)
        swing_high = None
        swing_high_idx = None
        for i in range(5, len(recent_highs) - 5):
            if recent_highs[i] == max(recent_highs[max(0,i-5):i+5]):
                if swing_high is None or recent_highs[i] > swing_high:
                    swing_high = recent_highs[i]
                    swing_high_idx = i
        
        # Check for LONG sweep (price went below swing low, closed above)
        if swing_low and current_low < swing_low and closes[-1] > swing_low:
            sweep_depth = (swing_low - current_low) / atr if atr > 0 else 0
            if sweep_depth >= 0.2:  # Minimum quality threshold
                result['sweep_detected'] = True
                result['sweep_type'] = 'LONG'
                result['sweep_level'] = swing_low
                result['sweep_depth_atr'] = sweep_depth
        
        # Check for SHORT sweep (price went above swing high, closed below)
        elif swing_high and current_high > swing_high and closes[-1] < swing_high:
            sweep_depth = (current_high - swing_high) / atr if atr > 0 else 0
            if sweep_depth >= 0.2:
                result['sweep_detected'] = True
                result['sweep_type'] = 'SHORT'
                result['sweep_level'] = swing_high
                result['sweep_depth_atr'] = sweep_depth
        
        # If sweep detected, get ML prediction
        if result['sweep_detected']:
            try:
                from liquidity_hunter.liquidity_hunter_ml import SweepPredictor
                
                predictor = SweepPredictor(market=market, mode=mode)
                result['model_available'] = predictor.model is not None
                
                if result['model_available']:
                    # Calculate features for ML
                    ema_20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
                    ema_50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
                    ema_200 = pd.Series(closes).ewm(span=200, adjust=False).mean().iloc[-1] if len(closes) >= 200 else ema_50
                    
                    # RSI
                    delta = pd.Series(closes).diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
                    rs = gain / loss if loss != 0 else 0
                    rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
                    
                    # Position in range
                    range_high = max(highs[-50:])
                    range_low = min(lows[-50:])
                    position_in_range = (current_price - range_low) / (range_high - range_low) * 100 if range_high != range_low else 50
                    
                    # Trend
                    trend_ema = 1 if ema_20 > ema_50 else -1
                    long_trend = 1 if current_price > ema_200 else -1
                    
                    # Candle analysis
                    candle_body = abs(closes[-1] - df['open'].iloc[-1])
                    candle_range = highs[-1] - lows[-1]
                    body_ratio = candle_body / candle_range if candle_range > 0 else 0
                    is_bullish = 1 if closes[-1] > df['open'].iloc[-1] else 0
                    
                    # Rejection wick
                    if result['sweep_type'] == 'LONG':
                        lower_wick = min(closes[-1], df['open'].iloc[-1]) - lows[-1]
                        rejection_wick_ratio = lower_wick / candle_range if candle_range > 0 else 0
                    else:
                        upper_wick = highs[-1] - max(closes[-1], df['open'].iloc[-1])
                        rejection_wick_ratio = upper_wick / candle_range if candle_range > 0 else 0
                    
                    # Volume
                    vol_avg = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 1
                    current_vol = df['volume'].iloc[-1] if 'volume' in df.columns else 1
                    volume_spike = current_vol / vol_avg if vol_avg > 0 else 1
                    
                    # Whale confirms
                    if result['sweep_type'] == 'LONG':
                        whale_confirms = 1 if whale_delta > 3 else (-1 if whale_delta < -3 else 0)
                    else:
                        whale_confirms = 1 if whale_delta < -3 else (-1 if whale_delta > 3 else 0)
                    
                    features = {
                        'sweep_depth_atr': result['sweep_depth_atr'],
                        'sweep_depth_pct': result['sweep_depth_atr'] * atr / result['sweep_level'] * 100 if result['sweep_level'] else 0,
                        'candles_since_level_formed': swing_low_idx if result['sweep_type'] == 'LONG' else swing_high_idx,
                        'atr': atr,
                        'volume_spike_ratio': volume_spike,
                        'rsi': rsi,
                        'trend_ema': trend_ema,
                        'ema_distance': (current_price - ema_50) / atr if atr > 0 else 0,
                        'position_in_range': position_in_range,
                        'momentum': ((closes[-1] - closes[-10]) / closes[-10] * 100) if len(closes) >= 10 else 0,
                        'body_ratio': body_ratio,
                        'is_bullish_candle': is_bullish,
                        'prev_trend': trend_ema,
                        'trend_aligned': 1 if (result['sweep_type'] == 'LONG' and trend_ema > 0) or (result['sweep_type'] == 'SHORT' and trend_ema < 0) else 0,
                        'long_trend': long_trend,
                        'price_vs_200ema': (current_price - ema_200) / atr if atr > 0 else 0,
                        'weekly_position': position_in_range,
                        'long_momentum': ((closes[-1] - closes[-30]) / closes[-30] * 100) if len(closes) >= 30 else 0,
                        'rejection_wick_ratio': rejection_wick_ratio,
                        'level_tests': 1,  # Simplified
                        'adx': 25,  # Default
                        'bb_position': position_in_range / 100,
                        'absorption_score': volume_spike,
                        'obv_trend': trend_ema,
                        'exhaustion_streak': 0,
                        'counter_trend': 1 if (result['sweep_type'] == 'LONG' and trend_ema < 0) or (result['sweep_type'] == 'SHORT' and trend_ema > 0) else 0,
                        'whale_pct': whale_pct,
                        'retail_pct': 100 - whale_pct,
                        'whale_delta': whale_delta,
                        'retail_delta': -whale_delta,
                        'whale_retail_delta': whale_pct - (100 - whale_pct),
                        'whale_confirms': whale_confirms,
                        'whale_trend_consistency': 0,
                        'whale_momentum': 0,
                        'whale_stability': 0.5,
                        'fear_greed': 50,
                        'btc_correlation': 0,
                        'level_strength': 'MODERATE',
                        'direction': result['sweep_type'],
                        'htf_trend': 'BULLISH' if long_trend > 0 else 'BEARISH'
                    }
                    
                    # Get ML prediction
                    prediction = predictor.predict(features)
                    
                    result['ml_quality'] = prediction.get('prediction', 'N/A')
                    result['ml_probability'] = prediction.get('probability', 0)
                    result['ml_confidence'] = prediction.get('confidence', 0)
                    result['factors'] = prediction.get('factors', {'positive': [], 'negative': []})
                    
                    # Swing filters (for swing mode)
                    if mode in ['swing', 'investment'] and 'swing_filters' in prediction:
                        result['swing_filters'] = prediction['swing_filters']
                    
                    # Generate recommendation
                    if result['ml_quality'] == 'HIGH_QUALITY':
                        result['recommendation'] = f"STRONG_{result['sweep_type']}"
                    elif result['ml_quality'] == 'MEDIUM_QUALITY':
                        result['recommendation'] = f"MODERATE_{result['sweep_type']}"
                    else:
                        result['recommendation'] = 'WEAK_SETUP'
                        
            except ImportError as e:
                print(f"[LH_ML] Could not import SweepPredictor: {e}")
            except Exception as e:
                print(f"[LH_ML] Prediction error: {e}")
        
        return result
        
    except Exception as e:
        print(f"[LH_ML] Sweep analysis error: {e}")
        return result


def render_sweep_badge_st(sweep_data: Dict) -> str:
    """
    Return a text badge for Scanner results (no HTML).
    
    Returns string like "🔄 LONG 🎯 72%" for use with st.write() or st.caption()
    """
    if not sweep_data.get('sweep_detected'):
        return ""
    
    sweep_type = sweep_data.get('sweep_type', 'LONG')
    quality = sweep_data.get('ml_quality', 'N/A')
    probability = sweep_data.get('ml_probability', 0)
    
    # Emoji coding
    if quality == 'HIGH_QUALITY':
        emoji = '🎯'
    elif quality == 'MEDIUM_QUALITY':
        emoji = '⚡'
    else:
        emoji = '⚠️'
    
    direction_emoji = '📈' if sweep_type == 'LONG' else '📉'
    
    return f"🔄 {sweep_type} {emoji} {probability:.0%}"


def render_sweep_section_st(sweep_data: Dict, show_details: bool = True):
    """
    Render full sweep analysis section using Streamlit components.
    
    Call this function directly - it renders using st.* components.
    """
    if not sweep_data.get('sweep_detected'):
        st.info("🔍 **Liquidity Sweep:** No active sweep detected. Monitoring for liquidity grabs at swing levels...")
        return
    
    sweep_type = sweep_data.get('sweep_type', 'LONG')
    quality = sweep_data.get('ml_quality', 'N/A')
    probability = sweep_data.get('ml_probability', 0)
    confidence = sweep_data.get('ml_confidence', 0)
    sweep_level = sweep_data.get('sweep_level', 0)
    sweep_depth = sweep_data.get('sweep_depth_atr', 0)
    model_available = sweep_data.get('model_available', False)
    recommendation = sweep_data.get('recommendation', 'WEAK_SETUP')
    
    # Quality emoji and label
    if quality == 'HIGH_QUALITY':
        quality_emoji = '🎯'
        quality_label = 'HIGH QUALITY'
    elif quality == 'MEDIUM_QUALITY':
        quality_emoji = '⚡'
        quality_label = 'MODERATE'
    else:
        quality_emoji = '⚠️'
        quality_label = 'LOW QUALITY'
    
    direction_emoji = '📈' if sweep_type == 'LONG' else '📉'
    
    # Header
    st.markdown(f"#### 🔄 Liquidity Sweep Detected: {direction_emoji} {sweep_type}")
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="ML Quality",
            value=f"{quality_emoji} {quality_label}"
        )
    
    with col2:
        st.metric(
            label="Win Probability",
            value=f"{probability:.1%}"
        )
    
    with col3:
        st.metric(
            label="Sweep Depth",
            value=f"{sweep_depth:.2f} ATR"
        )
    
    # Sweep level
    st.caption(f"Swept Level: ${sweep_level:,.2f}")
    
    # Factors
    if show_details:
        factors = sweep_data.get('factors', {})
        positive = factors.get('positive', [])
        negative = factors.get('negative', [])
        
        if positive or negative:
            with st.expander("📋 Key Factors", expanded=False):
                if positive:
                    for factor in positive[:3]:
                        st.success(f"✅ {factor}")
                
                if negative:
                    for factor in negative[:3]:
                        st.error(f"❌ {factor}")
    
    # Swing filters (if available)
    swing_filters = sweep_data.get('swing_filters', {})
    if swing_filters and show_details:
        filter_details = swing_filters.get('details', [])
        filters_passed = swing_filters.get('filters_passed', 0)
        max_filters = swing_filters.get('max_filters', 5)
        
        with st.expander(f"🎚️ Swing Filters: {filters_passed}/{max_filters} passed", expanded=False):
            for detail in filter_details[:5]:
                st.write(detail)
    
    # Model status warning
    if not model_available:
        st.warning("⚠️ ML model not trained - using rule-based analysis")


def get_sweep_summary(sweep_data: Dict) -> Dict:
    """
    Get a simplified summary for Scanner results.
    
    Returns dict with key fields for Scanner display.
    """
    return {
        'has_sweep': sweep_data.get('sweep_detected', False),
        'sweep_type': sweep_data.get('sweep_type'),
        'quality': sweep_data.get('ml_quality', 'N/A'),
        'probability': sweep_data.get('ml_probability', 0),
        'recommendation': sweep_data.get('recommendation', 'NO_SWEEP'),
        'badge_text': render_sweep_badge_st(sweep_data)
    }
