"""
Quality-Based Liquidity Hunter ML
==================================
SIMPLE approach: Predict "Should I take this trade?" (YES/NO)

Instead of predicting sweep outcomes, we FILTER for high-quality setups.

Key insight from user:
- Whale 65% (was 45%) = ACCUMULATING = ✅ BULLISH
- Whale 65% (was 85%) = DISTRIBUTING = ❌ BEARISH TRAP

The CHANGE matters more than absolute value!
"""

import os
import pickle
import random
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM (optional)
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import TensorFlow/Keras for Neural Network (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    TENSORFLOW_AVAILABLE = True
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TENSORFLOW_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "models/quality_model.pkl"
MODEL_PATH_ETF = "models/quality_model_etf.pkl"
MODEL_PATH_STOCK = "models/quality_model_stock.pkl"
FORWARD_CANDLES_DAILY = 60  # 60 trading days ≈ 3 months for daily timeframe
LOOKBACK_CANDLES = 50
FORWARD_CANDLES = 40  # Give trades more time to play out (40 × 4h = 160h = ~7 days)

# Stop loss multiplier (ATR)
STOP_ATR_MULTIPLIER = 1.0  # 1 ATR stop loss

# Risk:Reward ratio - KEY SETTING!
# At 1:1 R:R: need >50% win rate → EV = 2W - 1
# At 2:1 R:R: need >33% win rate → EV = 3W - 1 (EASIER!)
# At 3:1 R:R: need >25% win rate → EV = 4W - 1
TARGET_RR_RATIO = 2.0  # 2:1 R:R - optimal balance of win rate and reward

# STRATEGY MODE - ML DECIDES!
# The model evaluates BOTH directions (LONG and SHORT) for each sweep
# and picks the one with higher probability based on context features.
# No hardcoded mode needed - ML learns when to use reversal vs continuation.

# Minimum R:R to consider a "winning" trade
MIN_RR_FOR_WIN = 2.0  # Must hit at least 2:1 to count as win

# ═══════════════════════════════════════════════════════════════════════════════
# WHALE ANALYSIS - THE KEY INSIGHT!
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_whale_behavior(whale_pct: float, whale_delta: float, direction: str) -> Dict:
    """
    Analyze whale behavior - CHANGE matters more than absolute!
    
    Args:
        whale_pct: Current whale long % (0-100)
        whale_delta: Change in whale % (positive = more long, negative = less long)
        direction: 'LONG' or 'SHORT'
    
    Returns:
        Dict with whale analysis
    """
    # Determine whale behavior
    if whale_delta > 5:
        behavior = 'ACCUMULATING'  # Whales buying
    elif whale_delta < -5:
        behavior = 'DISTRIBUTING'  # Whales selling
    else:
        behavior = 'NEUTRAL'

    # Is whale aligned with trade direction?
    if direction == 'LONG':
        # For LONG: Want whales bullish AND accumulating
        whale_bullish = whale_pct >= 55
        whale_aligned = whale_bullish and whale_delta >= 0
        whale_strong = whale_pct >= 65 and whale_delta > 3
        whale_trap = whale_pct >= 60 and whale_delta < -5  # High but dropping = TRAP!
    else:
        # For SHORT: Want whales bearish AND distributing
        whale_bearish = whale_pct <= 45
        whale_aligned = whale_bearish and whale_delta <= 0
        whale_strong = whale_pct <= 35 and whale_delta < -3
        whale_trap = whale_pct <= 40 and whale_delta > 5  # Low but rising = TRAP!
    
    # Quality score based on whale behavior
    quality = 0
    
    if direction == 'LONG':
        if whale_pct >= 70 and whale_delta > 0:
            quality = 1.0  # Perfect: High and rising
        elif whale_pct >= 60 and whale_delta >= 0:
            quality = 0.8  # Good: Decent and stable/rising
        elif whale_pct >= 55 and whale_delta > 5:
            quality = 0.7  # OK: Medium but accelerating
        elif whale_pct >= 50 and whale_delta > 0:
            quality = 0.5  # Neutral but rising
        elif whale_delta < -5:
            quality = 0.1  # Bad: Distributing (regardless of %)
        else:
            quality = 0.3
    else:  # SHORT
        if whale_pct <= 30 and whale_delta < 0:
            quality = 1.0  # Perfect: Low and dropping
        elif whale_pct <= 40 and whale_delta <= 0:
            quality = 0.8  # Good: Decent and stable/dropping
        elif whale_pct <= 45 and whale_delta < -5:
            quality = 0.7  # OK: Medium but accelerating down
        elif whale_pct <= 50 and whale_delta < 0:
            quality = 0.5  # Neutral but dropping
        elif whale_delta > 5:
            quality = 0.1  # Bad: Accumulating (regardless of %)
        else:
            quality = 0.3
    
    return {
        'whale_pct': whale_pct,
        'whale_delta': whale_delta,
        'behavior': behavior,
        'aligned': whale_aligned,
        'strong': whale_strong,
        'trap': whale_trap,
        'quality': quality
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL QUALITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_level_quality(level_type: str, level_strength: float, distance_atr: float) -> Dict:
    """
    Analyze if level is worth trading.
    
    Only trade STRONG levels (EQUAL, DOUBLE) that are CLOSE.
    """
    # Level type scoring
    type_scores = {
        'EQUAL_LOW': 1.0,
        'EQUAL_HIGH': 1.0,
        'DOUBLE_LOW': 0.8,
        'DOUBLE_HIGH': 0.8,
        '100X_LIQ': 1.0,
        '50X_LIQ': 0.85,
        '25X_LIQ': 0.7,
        'SWING_LOW': 0.4,  # Low - usually noise
        'SWING_HIGH': 0.4,
    }
    
    type_score = type_scores.get(level_type, 0.3)
    
    # Distance scoring - closer is better
    if distance_atr <= 0.5:
        distance_score = 1.0  # Very close - immediate opportunity
    elif distance_atr <= 1.0:
        distance_score = 0.9
    elif distance_atr <= 1.5:
        distance_score = 0.7
    elif distance_atr <= 2.0:
        distance_score = 0.5
    elif distance_atr <= 3.0:
        distance_score = 0.3
    else:
        distance_score = 0.1  # Too far
    
    # Combined quality
    quality = (type_score * 0.6) + (distance_score * 0.4)
    
    # Is this a tradeable level?
    tradeable = (
        type_score >= 0.7 and  # EQUAL or DOUBLE only
        distance_atr <= 2.5 and  # Within range
        quality >= 0.5
    )
    
    return {
        'level_type': level_type,
        'type_score': type_score,
        'distance_atr': distance_atr,
        'distance_score': distance_score,
        'quality': quality,
        'tradeable': tradeable
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SHOULD I TAKE THIS TRADE? (Main Decision Function)
# ═══════════════════════════════════════════════════════════════════════════════

def should_take_trade(
    level_type: str,
    level_strength: float,
    distance_atr: float,
    whale_pct: float,
    whale_delta: float,
    direction: str,
    momentum: float = 0,
    volume_ratio: float = 1.0
) -> Dict:
    """
    Simple decision: Should I take this trade?
    
    This is the CORE function - combines whale + level analysis.
    
    Returns:
        Dict with decision and reasoning
    """
    whale = analyze_whale_behavior(whale_pct, whale_delta, direction)
    level = analyze_level_quality(level_type, level_strength, distance_atr)
    
    # Start with base score
    score = 0
    reasons = []
    red_flags = []
    
    # ═══════════════════════════════════════════════════════════════════
    # WHALE ANALYSIS (50% weight)
    # ═══════════════════════════════════════════════════════════════════
    
    if whale['trap']:
        score -= 30
        red_flags.append(f"⚠️ WHALE TRAP: {whale_pct:.0f}% but delta {whale_delta:+.1f}%")
    
    if whale['strong']:
        score += 25
        reasons.append(f"✅ Strong whale alignment: {whale_pct:.0f}% delta {whale_delta:+.1f}%")
    elif whale['aligned']:
        score += 15
        reasons.append(f"✅ Whale aligned: {whale_pct:.0f}%")
    
    if whale['behavior'] == 'ACCUMULATING' and direction == 'LONG':
        score += 10
        reasons.append("📈 Whales accumulating")
    elif whale['behavior'] == 'DISTRIBUTING' and direction == 'SHORT':
        score += 10
        reasons.append("📉 Whales distributing")
    elif whale['behavior'] == 'ACCUMULATING' and direction == 'SHORT':
        score -= 15
        red_flags.append("⚠️ Shorting while whales accumulate!")
    elif whale['behavior'] == 'DISTRIBUTING' and direction == 'LONG':
        score -= 15
        red_flags.append("⚠️ Longing while whales distribute!")
    
    # Whale quality contribution
    score += whale['quality'] * 20
    
    # ═══════════════════════════════════════════════════════════════════
    # LEVEL ANALYSIS (30% weight)
    # ═══════════════════════════════════════════════════════════════════
    
    if not level['tradeable']:
        score -= 20
        red_flags.append(f"⚠️ Weak level: {level_type} @ {distance_atr:.1f} ATR")
    
    if level['type_score'] >= 0.9:
        score += 15
        reasons.append(f"✅ Strong level: {level_type}")
    elif level['type_score'] >= 0.7:
        score += 10
        reasons.append(f"✅ Good level: {level_type}")
    
    if level['distance_score'] >= 0.8:
        score += 10
        reasons.append(f"✅ Close to level: {distance_atr:.1f} ATR")
    elif level['distance_atr'] > 2.5:
        score -= 10
        red_flags.append(f"⚠️ Too far: {distance_atr:.1f} ATR")
    
    # ═══════════════════════════════════════════════════════════════════
    # MOMENTUM (10% weight)
    # ═══════════════════════════════════════════════════════════════════
    
    if direction == 'LONG' and momentum > 0.3:
        score += 5
        reasons.append("📈 Momentum supportive")
    elif direction == 'SHORT' and momentum < -0.3:
        score += 5
        reasons.append("📉 Momentum supportive")
    elif direction == 'LONG' and momentum < -0.5:
        score -= 5
        red_flags.append("⚠️ Fighting momentum")
    elif direction == 'SHORT' and momentum > 0.5:
        score -= 5
        red_flags.append("⚠️ Fighting momentum")
    
    # ═══════════════════════════════════════════════════════════════════
    # VOLUME (10% weight)
    # ═══════════════════════════════════════════════════════════════════
    
    if volume_ratio > 1.5:
        score += 5
        reasons.append("📊 High volume")
    elif volume_ratio < 0.5:
        score -= 5
        red_flags.append("⚠️ Low volume")
    
    # ═══════════════════════════════════════════════════════════════════
    # FINAL DECISION
    # ═══════════════════════════════════════════════════════════════════
    
    # Normalize score to 0-100
    score = max(0, min(100, score + 50))  # Base 50, range 0-100
    
    # Decision thresholds
    if score >= 70:
        decision = 'STRONG_YES'
        take_trade = True
    elif score >= 55:
        decision = 'YES'
        take_trade = True
    elif score >= 45:
        decision = 'MAYBE'
        take_trade = False  # Skip marginal setups
    else:
        decision = 'NO'
        take_trade = False
    
    return {
        'take_trade': take_trade,
        'decision': decision,
        'score': score,
        'whale_analysis': whale,
        'level_analysis': level,
        'reasons': reasons,
        'red_flags': red_flags
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP-SPECIFIC FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_sweep_features(
    df: pd.DataFrame,
    sweep_candle_idx: int = None,
    candles_since_sweep: int = None,
    level_price: float = None,
    direction: str = None,
    atr: float = None
) -> Dict:
    """
    Calculate sweep-specific features from the sweep candle.

    These are the RAW measurements of sweep quality:
    - How deep did price go past the level?
    - How strong was the rejection (wick vs body)?
    - How fresh is the sweep?
    - Was there volume confirmation?
    """
    # Default values (used when no sweep data available)
    features = {
        # Layer 1 - Raw sweep data
        'sweep_depth_atr': 0.0,
        'sweep_wick_ratio': 0.0,
        'sweep_body_ratio': 0.5,
        'candles_since_sweep': 100,  # High value = old/no sweep
        'sweep_volume_ratio': 1.0,
        'price_change_since_sweep': 0.0,  # % change since sweep (negative = against direction)
        # Layer 3 - Sweep quality rules
        'rule_deep_sweep': 0,
        'rule_strong_rejection': 0,
        'rule_fresh_sweep': 0,
        'rule_volume_confirmed': 0,
        'rule_price_aligned': 0,  # 1 if price moved in expected direction
        # NEW: Follow-through & Distance features (Jan 2026)
        'follow_through_per_candle': 0.0,  # % change per candle since sweep
        'distance_from_sweep_pct': 10.0,   # Default far from level
        'indecision_candle': 0,            # 1 if body ≈ wick (choppy)
        'rule_weak_follow_through': 0,     # 1 if weak follow-through
        'rule_close_to_sweep_level': 0,    # 1 if close to sweep level
        # PRE-SWEEP STRUCTURE
        'structure_lower_highs': 0,  # 1 if bearish structure before sweep
        'structure_higher_lows': 0,  # 1 if bullish structure before sweep
        'structure_failed_bos': 0,   # 1 if failed Break of Structure
        'structure_trend_aligned': 0,  # 1 if sweep aligns with trend
        # IMPULSE vs CORRECTION
        'is_impulse_move': 0,  # 1 if expanding candles (impulse)
        'is_correction': 0,    # 1 if contracting candles (correction)
        'swing_expansion': 1.0,  # Move size relative to ATR
    }

    # If no sweep candle index provided, return defaults
    if sweep_candle_idx is None or df is None or len(df) == 0:
        return features

    # Validate index
    if sweep_candle_idx < 0 or sweep_candle_idx >= len(df):
        return features

    try:
        # Get sweep candle data
        sweep_candle = df.iloc[sweep_candle_idx]

        # Candle components
        open_price = float(sweep_candle['open'])
        high_price = float(sweep_candle['high'])
        low_price = float(sweep_candle['low'])
        close_price = float(sweep_candle['close'])
        volume = float(sweep_candle['volume']) if 'volume' in df.columns else 0

        # Candle measurements
        candle_range = high_price - low_price
        body_size = abs(close_price - open_price)

        # Calculate ATR if not provided
        if atr is None or atr <= 0:
            high = df['high']
            low = df['low']
            close = df['close']
            tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

        if atr <= 0:
            atr = candle_range  # Fallback

        # Avoid division by zero
        if candle_range <= 0:
            candle_range = atr * 0.01  # Tiny value

        # ═══════════════════════════════════════════════════════════════
        # RAW SWEEP MEASUREMENTS
        # ═══════════════════════════════════════════════════════════════

        # 1. Sweep depth: How far past the level did price go?
        if level_price is not None and direction is not None:
            if direction == 'LONG':
                # Sweep of LOW - measure how far below level
                sweep_depth = max(0, level_price - low_price)
            else:
                # Sweep of HIGH - measure how far above level
                sweep_depth = max(0, high_price - level_price)
            sweep_depth_atr = sweep_depth / atr if atr > 0 else 0
        else:
            sweep_depth_atr = 0

        # 2. Wick ratio: How much of the candle is rejection wick?
        # For LONG (sweep of low): lower wick matters
        # For SHORT (sweep of high): upper wick matters
        if direction == 'LONG':
            rejection_wick = min(open_price, close_price) - low_price
        elif direction == 'SHORT':
            rejection_wick = high_price - max(open_price, close_price)
        else:
            rejection_wick = max(
                min(open_price, close_price) - low_price,
                high_price - max(open_price, close_price)
            )

        sweep_wick_ratio = rejection_wick / candle_range if candle_range > 0 else 0

        # 3. Body ratio: Body size vs candle range (small body = indecision/rejection)
        sweep_body_ratio = body_size / candle_range if candle_range > 0 else 0.5

        # 4. Freshness: How many candles ago?
        if candles_since_sweep is not None:
            freshness = candles_since_sweep
        else:
            freshness = len(df) - 1 - sweep_candle_idx

        # 5. Volume on sweep vs average
        if volume > 0 and len(df) >= 20:
            avg_volume = df['volume'].iloc[-20:].mean()
            sweep_volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        else:
            sweep_volume_ratio = 1.0

        # ═══════════════════════════════════════════════════════════════
        # RULE FLAGS (ML learns when these matter)
        # ═══════════════════════════════════════════════════════════════

        # Deep sweep: Went > 0.5 ATR past level (strong liquidity grab)
        rule_deep_sweep = 1 if sweep_depth_atr > 0.5 else 0

        # Strong rejection: Wick > 60% of candle (clear rejection pattern)
        rule_strong_rejection = 1 if sweep_wick_ratio > 0.6 else 0

        # INDECISION CANDLE: Body ≈ Wick = CHOPPY market, no clear signal
        # Jan 27 DOT example: body 47%, wick 49% → indecision → LOSS
        # Jan 29 DOT example: body 4%, wick 26% → clear rejection → WIN
        # Flag when body is within 20% of wick ratio (e.g., 0.4 body vs 0.5 wick)
        indecision_candle = 1 if abs(sweep_body_ratio - sweep_wick_ratio) < 0.2 and sweep_body_ratio > 0.3 else 0

        # Fresh sweep: Within 5 candles (opportunity still valid)
        rule_fresh_sweep = 1 if freshness <= 5 else 0

        # 6. Price movement since sweep: Did price go in expected direction?
        # This is CRITICAL - if price moved against direction, setup is failing
        price_change_since_sweep = 0.0
        price_change_aligned = 0
        follow_through_per_candle = 0.0
        distance_from_sweep_pct = 0.0

        if sweep_candle_idx is not None and sweep_candle_idx < len(df) - 1:
            sweep_close = float(df.iloc[sweep_candle_idx]['close'])
            current_close = float(df.iloc[-1]['close'])
            price_change_since_sweep = (current_close - sweep_close) / sweep_close * 100

            # Check if price moved in expected direction
            if direction == 'LONG':
                price_change_aligned = 1 if price_change_since_sweep > -1 else 0  # Allow small drawdown
            else:  # SHORT
                price_change_aligned = 1 if price_change_since_sweep < 1 else 0  # Allow small bounce

            # NEW: Follow-through strength = price change per candle since sweep
            # Jan 27: -0.91% / 6 candles = -0.15%/candle (WEAK)
            # Jan 29: -2.90% / 6 candles = -0.48%/candle (STRONG)
            if freshness > 0:
                follow_through_per_candle = abs(price_change_since_sweep) / freshness

            # NEW: Distance from sweep level (how far price has moved from the swept level)
            # Closer to sweep level = higher chance of retest/failure
            if level_price > 0:
                distance_from_sweep_pct = abs(current_close - level_price) / level_price * 100

        # NEW: Weak follow-through flag
        # If sweep was 5+ candles ago but price only moved <2%, follow-through is weak
        rule_weak_follow_through = 1 if freshness >= 5 and abs(price_change_since_sweep) < 2.0 else 0

        # NEW: Price close to sweep level = potential retest/failure
        # If price is within 3% of sweep level, setup may fail
        rule_close_to_sweep_level = 1 if distance_from_sweep_pct < 3.0 else 0

        # Volume confirmed: 1.5x+ average volume (institutional participation)
        rule_volume_confirmed = 1 if sweep_volume_ratio > 1.5 else 0

        # ═══════════════════════════════════════════════════════════════
        # 7. PRE-SWEEP STRUCTURE ANALYSIS (Key for predicting failure!)
        # ═══════════════════════════════════════════════════════════════
        # These features capture the market structure BEFORE the sweep
        # to predict if the sweep is a trap or a real opportunity

        structure_lower_highs = 0  # Bearish: highs getting lower
        structure_higher_lows = 0  # Bullish: lows getting higher
        structure_failed_bos = 0   # Failed Break of Structure
        structure_trend_aligned = 0  # Sweep aligns with pre-sweep trend

        if sweep_candle_idx is not None and sweep_candle_idx >= 10:
            # Look at structure in the 10 candles before sweep
            pre_sweep_df = df.iloc[max(0, sweep_candle_idx-10):sweep_candle_idx]

            if len(pre_sweep_df) >= 5:
                # Find swing highs and lows in pre-sweep period
                highs = pre_sweep_df['high'].values
                lows = pre_sweep_df['low'].values

                # Check for lower highs (bearish structure)
                # Compare first half highs to second half highs
                first_half_high = max(highs[:len(highs)//2]) if len(highs) > 1 else highs[0]
                second_half_high = max(highs[len(highs)//2:]) if len(highs) > 1 else highs[-1]
                if second_half_high < first_half_high * 0.998:  # Lower high with tolerance
                    structure_lower_highs = 1

                # Check for higher lows (bullish structure)
                first_half_low = min(lows[:len(lows)//2]) if len(lows) > 1 else lows[0]
                second_half_low = min(lows[len(lows)//2:]) if len(lows) > 1 else lows[-1]
                if second_half_low > first_half_low * 1.002:  # Higher low with tolerance
                    structure_higher_lows = 1

                # Check for failed BOS (Break of Structure)
                # For sweep of HIGH: if price made lower highs before sweep, it's a failed BOS attempt
                # For sweep of LOW: if price made higher lows before sweep, it's a failed BOS attempt
                if direction == 'LONG':
                    # Sweep of LOW - check if we had higher lows (bullish) or lower lows (bearish trap)
                    recent_low = min(lows[-3:]) if len(lows) >= 3 else lows[-1]
                    earlier_low = min(lows[:len(lows)//2]) if len(lows) > 1 else lows[0]
                    if recent_low < earlier_low:  # Lower low before sweep of low = bearish, failed BOS
                        structure_failed_bos = 1
                else:  # SHORT - sweep of HIGH
                    # Check if we had lower highs (bearish) before sweep of high = trap for longs
                    recent_high = max(highs[-3:]) if len(highs) >= 3 else highs[-1]
                    earlier_high = max(highs[:len(highs)//2]) if len(highs) > 1 else highs[0]
                    if recent_high < earlier_high:  # Lower high before sweep of high = bearish structure
                        structure_failed_bos = 1

                # Check if sweep direction aligns with pre-sweep trend
                # LONG setup with higher lows = aligned (bullish into bullish)
                # SHORT setup with lower highs = aligned (bearish into bearish)
                if direction == 'LONG' and structure_higher_lows:
                    structure_trend_aligned = 1
                elif direction == 'SHORT' and structure_lower_highs:
                    structure_trend_aligned = 1
                # Counter-trend setups (traps) get structure_trend_aligned = 0

        # ═══════════════════════════════════════════════════════════════
        # 8. IMPULSE vs CORRECTION (Key SMC concept!)
        # ═══════════════════════════════════════════════════════════════
        # Impulse moves: Large candles, directional, high volume
        # Corrections: Small candles, choppy, low volume
        # ML learns: Sweeps during impulse = continuation, during correction = reversal

        is_impulse_move = 0
        is_correction = 0
        swing_expansion = 1.0  # Current swing size vs average

        if sweep_candle_idx is not None and sweep_candle_idx >= 5:
            recent_candles = df.iloc[max(0, sweep_candle_idx-5):sweep_candle_idx+1]

            if len(recent_candles) >= 3:
                # Calculate average candle size (range)
                candle_ranges = (recent_candles['high'] - recent_candles['low']).values
                avg_range = candle_ranges.mean()

                # Current move size (from 5 candles ago to sweep)
                move_size = abs(recent_candles['close'].iloc[-1] - recent_candles['close'].iloc[0])

                # Impulse: Large directional move with expanding candles
                if avg_range > 0:
                    # Check if candles are expanding (impulse) or contracting (correction)
                    first_half_avg = candle_ranges[:len(candle_ranges)//2].mean()
                    second_half_avg = candle_ranges[len(candle_ranges)//2:].mean()

                    if second_half_avg > first_half_avg * 1.2:  # Expanding
                        is_impulse_move = 1
                    elif second_half_avg < first_half_avg * 0.8:  # Contracting
                        is_correction = 1

                    # Swing expansion: How big is this move vs ATR?
                    if atr and atr > 0:
                        swing_expansion = move_size / (atr * 3)  # Normalize to ~1.0 for average move

        # Update features
        features = {
            # Layer 1 - Raw sweep data
            'sweep_depth_atr': round(sweep_depth_atr, 4),
            'sweep_wick_ratio': round(sweep_wick_ratio, 4),
            'sweep_body_ratio': round(sweep_body_ratio, 4),
            'candles_since_sweep': freshness,
            'sweep_volume_ratio': round(sweep_volume_ratio, 4),
            'price_change_since_sweep': round(price_change_since_sweep, 4),  # NEW: % change since sweep
            # Layer 3 - Sweep quality rules
            'rule_deep_sweep': rule_deep_sweep,
            'rule_strong_rejection': rule_strong_rejection,
            'rule_fresh_sweep': rule_fresh_sweep,
            'rule_volume_confirmed': rule_volume_confirmed,
            'rule_price_aligned': price_change_aligned,  # 1 if price moved in expected direction
            # NEW: Follow-through & Distance features (Jan 2026)
            'follow_through_per_candle': round(follow_through_per_candle, 4),  # % change per candle
            'distance_from_sweep_pct': round(distance_from_sweep_pct, 4),  # Distance from sweep level
            'indecision_candle': indecision_candle,  # 1 if body ≈ wick (choppy)
            'rule_weak_follow_through': rule_weak_follow_through,  # 1 if weak follow-through
            'rule_close_to_sweep_level': rule_close_to_sweep_level,  # 1 if close to sweep level
            # PRE-SWEEP STRUCTURE (predicts traps!)
            'structure_lower_highs': structure_lower_highs,  # 1 if bearish structure before sweep
            'structure_higher_lows': structure_higher_lows,  # 1 if bullish structure before sweep
            'structure_failed_bos': structure_failed_bos,    # 1 if failed Break of Structure
            'structure_trend_aligned': structure_trend_aligned,  # 1 if sweep aligns with trend
            # IMPULSE vs CORRECTION (SMC key concept!)
            'is_impulse_move': is_impulse_move,  # 1 if sweep during impulse (expanding candles)
            'is_correction': is_correction,       # 1 if sweep during correction (contracting)
            'swing_expansion': round(swing_expansion, 4),  # Size of move relative to ATR
        }

    except Exception as e:
        print(f"[QUALITY_ML] Sweep feature extraction error: {e}")

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FOR ML
# ═══════════════════════════════════════════════════════════════════════════════

def extract_quality_features(
    df: pd.DataFrame,
    level_price: float,
    level_type: str,
    level_strength: float,
    direction: str,
    whale_pct: float,
    whale_delta: float,
    sweep_status: Dict = None,  # Contains price action data
    sweep_idx: int = None,      # Index of sweep candle for PA analysis
    # Whale acceleration features (4h, 24h, 7d)
    whale_delta_4h: float = None,   # Early signal (4 hours)
    whale_delta_24h: float = None,
    whale_delta_7d: float = None,
    whale_acceleration: str = None,
    whale_early_signal: str = None,  # EARLY_ACCUMULATION, EARLY_DISTRIBUTION, etc.
    is_fresh_accumulation: bool = False,
    # OI features
    oi_change_24h: float = None,    # OI change % in 24h
    price_change_24h: float = None, # Price change % in 24h
    # Additional Layer 1 raw features
    retail_pct: float = None,       # Retail long % (for divergence)
    funding_rate: float = None,     # Funding rate
    # Sweep-specific features (core of liquidity hunting!)
    sweep_candle_idx: int = None,   # Index of the sweep candle in df
    candles_since_sweep: int = None, # How many candles since sweep
    # ML direction features
    is_sweep_of_low: bool = None,    # True if sweep of LOW level (None = infer from direction)
    # ETF/Stock flow features (replaces whale data)
    market_type: str = 'crypto',
    etf_flow_data: Dict = None
) -> Dict:
    """
    Extract features for quality prediction model.
    Now includes Price Action features AND whale acceleration (4h/24h/7d).
    """
    if df is None or len(df) < 20:
        return None


    try:
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]

        current_price = float(df['close'].iloc[-1])

        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        if atr <= 0:
            return None

        # Distance to level
        distance = abs(current_price - level_price)
        distance_atr = distance / atr

        # Momentum (price change over 10 candles)
        if len(df) >= 10:
            momentum = (current_price - df['close'].iloc[-10]) / (atr * 10)
        else:
            momentum = 0

        # Volume ratio
        if 'volume' in df.columns and len(df) >= 20:
            avg_vol = df['volume'].iloc[-20:].mean()
            recent_vol = df['volume'].iloc[-3:].mean()
            volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
        else:
            volume_ratio = 1.0

        # Volatility (ATR ratio to price)
        volatility = atr / current_price

        # Whale features
        whale = analyze_whale_behavior(whale_pct, whale_delta, direction)

        # Level features
        level = analyze_level_quality(level_type, level_strength, distance_atr)

        # ═══════════════════════════════════════════════════════════════════
        # PRICE ACTION FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════════════
        pa_candle_score = 0
        pa_structure_score = 0
        pa_has_order_block = 0
        pa_has_fvg = 0
        pa_volume_score = 0
        pa_momentum_score = 0
        pa_total_score = 0

        # Try to get price action from sweep_status first
        if sweep_status and sweep_status.get('price_action'):
            pa_data = sweep_status['price_action']
            pred = pa_data.get('prediction', {})
            comp = pred.get('component_scores', {})

            pa_candle_score = comp.get('candle_pattern', 0)
            pa_structure_score = comp.get('structure', 0)
            pa_has_order_block = 1 if comp.get('order_blocks', 0) > 0 else 0
            pa_has_fvg = 1 if comp.get('fvg', 0) > 0 else 0
            pa_volume_score = comp.get('volume', 0)
            pa_momentum_score = comp.get('momentum', 0)
            pa_total_score = pred.get('score', 0)

        # If no PA in sweep_status, try to calculate it
        elif sweep_idx is not None and sweep_idx >= 0 and sweep_idx < len(df):
            try:
                from .price_action_analyzer import analyze_sweep_reaction
                pa_result = analyze_sweep_reaction(df, sweep_idx, direction, atr)
                if pa_result.get('valid'):
                    pred = pa_result.get('prediction', {})
                    comp = pred.get('component_scores', {})

                    pa_candle_score = comp.get('candle_pattern', 0)
                    pa_structure_score = comp.get('structure', 0)
                    pa_has_order_block = 1 if comp.get('order_blocks', 0) > 0 else 0
                    pa_has_fvg = 1 if comp.get('fvg', 0) > 0 else 0
                    pa_volume_score = comp.get('volume', 0)
                    pa_momentum_score = comp.get('momentum', 0)
                    pa_total_score = pred.get('score', 0)
            except ImportError:
                pass  # Price action analyzer not available

        # ═══════════════════════════════════════════════════════════════════
        # WHALE ACCELERATION FEATURES (24h vs 7d) - THE KEY TO TIMING!
        # ═══════════════════════════════════════════════════════════════════
        # Calculate acceleration metrics
        whale_24h = whale_delta_24h if whale_delta_24h is not None else whale_delta
        whale_7d = whale_delta_7d if whale_delta_7d is not None else whale_delta
        whale_4h = whale_delta_4h if whale_delta_4h is not None else 0  # NEW: 4h early signal

        # Daily average from 7d
        daily_avg_7d = whale_7d / 7 if whale_7d else 0

        # Acceleration ratio: how much faster is 24h vs 7d average?
        if abs(daily_avg_7d) > 0.5:  # Avoid division by tiny numbers
            acceleration_ratio = whale_24h / daily_avg_7d if daily_avg_7d != 0 else 0
        else:
            acceleration_ratio = 0

        # NEW: 4h vs 24h ratio (early signal strength)
        if abs(whale_24h) > 0.5:
            early_signal_ratio = whale_4h / whale_24h if whale_24h != 0 else 0
        else:
            early_signal_ratio = 0

        # Encode acceleration status
        accel_accelerating = 1 if whale_acceleration == 'ACCELERATING' else 0
        accel_decelerating = 1 if whale_acceleration == 'DECELERATING' else 0
        accel_reversing = 1 if whale_acceleration == 'REVERSING' else 0
        accel_steady = 1 if whale_acceleration == 'STEADY' else 0

        # NEW: Encode early signal status (4h vs 24h divergence)
        early_accumulation = 1 if whale_early_signal == 'EARLY_ACCUMULATION' else 0
        early_distribution = 1 if whale_early_signal == 'EARLY_DISTRIBUTION' else 0
        fresh_accumulation = 1 if whale_early_signal == 'FRESH_ACCUMULATION' else 0
        fresh_distribution = 1 if whale_early_signal == 'FRESH_DISTRIBUTION' else 0

        # Fresh vs late entry signal
        is_fresh = 1 if is_fresh_accumulation else 0

        # Late entry detection: big 7d move but slow 24h
        is_late_entry = 0
        if direction == 'LONG' and whale_7d > 5 and whale_24h < daily_avg_7d * 0.5:
            is_late_entry = 1
        elif direction == 'SHORT' and whale_7d < -5 and whale_24h > daily_avg_7d * 0.5:
            is_late_entry = 1

        # ═══════════════════════════════════════════════════════════════════
        # REGIME FEATURES (Critical for different market conditions)
        # ═══════════════════════════════════════════════════════════════════
        # Volatility regime: compare current ATR to historical
        atr_20 = tr.rolling(20).mean()
        atr_50 = tr.rolling(50).mean()
        if len(atr_50.dropna()) > 0 and atr_50.iloc[-1] > 0:
            regime_volatility = atr_20.iloc[-1] / atr_50.iloc[-1]  # >1 = high vol, <1 = low vol
        else:
            regime_volatility = 1.0

        # Trend strength: using simple price position relative to moving averages
        if len(df) >= 20:
            ma_20 = df['close'].rolling(20).mean().iloc[-1]
            ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20
            # Trend strength: distance from MA as % of ATR
            trend_strength = (current_price - ma_20) / atr if atr > 0 else 0
        else:
            trend_strength = 0

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 2: ENGINEERED - Whale-Retail Divergence (smart vs dumb money)
        # ═══════════════════════════════════════════════════════════════════
        retail = retail_pct if retail_pct is not None else 50
        whale_retail_divergence = whale_pct - retail  # Positive = whales more bullish than retail

        return {
            # ═══════════════════════════════════════════════════════════════
            # LAYER 1: RAW SIGNALS (Truth - what is happening?)
            # ═══════════════════════════════════════════════════════════════
            'whale_pct': whale_pct,
            'retail_pct': retail,
            'whale_delta_4h': whale_4h,
            'whale_delta_24h': whale_24h,
            'whale_delta_7d': whale_7d,
            'oi_change_24h': oi_change_24h if oi_change_24h is not None else 0,
            'price_change_24h': price_change_24h if price_change_24h is not None else 0,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'funding_rate': funding_rate if funding_rate is not None else 0,
            'momentum': momentum,

            # ═══════════════════════════════════════════════════════════════
            # LAYER 2: ENGINEERED FEATURES (Context - how meaningful?)
            # ═══════════════════════════════════════════════════════════════
            'distance_atr': distance_atr,
            'whale_retail_divergence': whale_retail_divergence,
            'whale_acceleration_ratio': acceleration_ratio,
            'whale_early_signal_ratio': early_signal_ratio,
            'whale_daily_avg_7d': daily_avg_7d,
            'level_type_score': level['type_score'],
            'level_distance_score': level['distance_score'],
            'level_quality': level['quality'],
            # Price Action context
            'pa_candle_score': pa_candle_score,
            'pa_structure_score': pa_structure_score,
            'pa_volume_score': pa_volume_score,
            'pa_momentum_score': pa_momentum_score,
            'pa_total_score': pa_total_score,

            # ═══════════════════════════════════════════════════════════════
            # LAYER 3: RULE FLAGS AS FEATURES (ML learns when rules work/fail)
            # ═══════════════════════════════════════════════════════════════
            # Whale rules
            'rule_whale_bullish': 1 if whale_pct > 60 else 0,
            'rule_whale_bearish': 1 if whale_pct < 40 else 0,
            'rule_whale_accumulating': 1 if whale_24h > 3 else 0,
            'rule_whale_distributing': 1 if whale_24h < -3 else 0,
            'rule_whale_aligned': 1 if whale['aligned'] else 0,
            'rule_whale_trap': 1 if whale['trap'] else 0,
            'rule_whale_strong': 1 if whale['strong'] else 0,
            # OI rules
            'rule_oi_increasing': 1 if (oi_change_24h or 0) > 2 else 0,
            'rule_oi_decreasing': 1 if (oi_change_24h or 0) < -2 else 0,
            'rule_oi_price_aligned': 1 if ((oi_change_24h or 0) > 0 and (price_change_24h or 0) > 0) or
                                          ((oi_change_24h or 0) < 0 and (price_change_24h or 0) < 0) else 0,
            # Timing rules
            'rule_is_fresh_entry': is_fresh,
            'rule_is_late_entry': is_late_entry,
            'rule_early_accumulation': early_accumulation,
            'rule_early_distribution': early_distribution,
            # Acceleration status (categorical)
            'whale_accel_accelerating': accel_accelerating,
            'whale_accel_decelerating': accel_decelerating,
            'whale_accel_reversing': accel_reversing,
            'whale_accel_steady': accel_steady,
            # Level rules
            'rule_level_tradeable': 1 if level['tradeable'] else 0,
            'pa_has_order_block': pa_has_order_block,
            'pa_has_fvg': pa_has_fvg,

            # ═══════════════════════════════════════════════════════════════
            # REGIME FEATURES (Critical!)
            # ═══════════════════════════════════════════════════════════════
            'regime_volatility': regime_volatility,
            'regime_trend_strength': trend_strength,

            # ═══════════════════════════════════════════════════════════════
            # SWEEP-SPECIFIC FEATURES (Core of liquidity hunting!)
            # ═══════════════════════════════════════════════════════════════
            **_calculate_sweep_features(
                df=df,
                sweep_candle_idx=sweep_candle_idx,
                candles_since_sweep=candles_since_sweep,
                level_price=level_price,
                direction=direction,
                atr=atr
            ),

            # Direction encoding (ML decides best direction!)
            'is_trade_long': 1 if direction == 'LONG' else 0,
            'is_sweep_of_low': 1 if (is_sweep_of_low if is_sweep_of_low is not None else (direction == 'LONG')) else 0,
        }

        # ═══════════════════════════════════════════════════════════════════
        # ETF/STOCK: Replace whale features with money flow features
        # ═══════════════════════════════════════════════════════════════════
        if market_type in ('etf', 'stock'):
            flow = etf_flow_data
            if flow is None:
                # Compute from df if not pre-computed
                try:
                    from core.etf_flow import calculate_etf_flow_score
                    flow = calculate_etf_flow_score(df, include_institutional=False)
                except Exception:
                    flow = {}

            # Map flow features into whale feature slots
            features['whale_pct'] = flow.get('flow_score', 0)
            features['whale_delta_4h'] = flow.get('mfi_value', 50)
            features['whale_delta_24h'] = flow.get('cmf_value', 0)
            features['whale_delta_7d'] = flow.get('price_extension_ema200', 0)
            features['retail_pct'] = flow.get('price_extension_ema50', 0)
            features['funding_rate'] = flow.get('institutional_score', 0)
            features['oi_change_24h'] = flow.get('options_sentiment_score', 0)

            # Map flow rules
            phase = flow.get('flow_phase', 'NEUTRAL')
            features['rule_whale_bullish'] = 1 if phase == 'ACCUMULATING' else 0
            features['rule_whale_bearish'] = 1 if phase == 'DISTRIBUTING' else 0
            features['rule_whale_accumulating'] = 1 if flow.get('in_accumulation_zone', False) else 0
            features['rule_whale_distributing'] = 1 if flow.get('in_distribution_zone', False) else 0

            # Flow aligned with trade direction
            flow_score = flow.get('flow_score', 0)
            flow_supports_long = flow_score > 20
            flow_supports_short = flow_score < -20
            features['rule_whale_aligned'] = 1 if (
                (direction == 'LONG' and flow_supports_long) or
                (direction == 'SHORT' and flow_supports_short)
            ) else 0
            features['rule_whale_trap'] = 1 if phase == 'EXTENDED' else 0
            features['rule_whale_strong'] = 1 if abs(flow_score) > 60 else 0

            # OBV trend replaces acceleration
            features['whale_acceleration_ratio'] = flow.get('obv_trend', 0)
            features['whale_early_signal_ratio'] = 0
            features['whale_daily_avg_7d'] = 0
            features['whale_retail_divergence'] = 0

            # Zero out crypto-only flags
            features['rule_oi_increasing'] = 0
            features['rule_oi_decreasing'] = 0
            features['rule_oi_price_aligned'] = 0
            features['rule_early_accumulation'] = 0
            features['rule_early_distribution'] = 0
            features['whale_accel_accelerating'] = 0
            features['whale_accel_decelerating'] = 0
            features['whale_accel_reversing'] = 0
            features['whale_accel_steady'] = 0

        return features

    except Exception as e:
        print(f"[QUALITY_ML] Feature extraction error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL: DID THIS TRADE WIN? (Simple 1:1 R:R)
# ═══════════════════════════════════════════════════════════════════════════════

def label_trade_outcome(
    df: pd.DataFrame,
    idx: int,
    level_price: float,
    direction: str,
    atr: float,
    trade_direction: str = None,
    forward_candles: int = None
) -> Optional[Dict]:
    """
    Label trade outcome: Did trade hit target R:R?

    Args:
        direction: Sweep direction (LONG = sweep of low, SHORT = sweep of high)
        trade_direction: Actual trade direction (LONG or SHORT). If None, defaults to direction.
    """
    if forward_candles is None:
        forward_candles = FORWARD_CANDLES
    if idx + forward_candles >= len(df):
        return None

    current_price = df['close'].iloc[idx]

    # ML-driven: caller specifies trade direction explicitly
    if trade_direction is None:
        trade_direction = direction

    # Define trade levels based on actual trade direction
    # Uses current_price (the price when we're evaluating the setup)
    if trade_direction == 'LONG':
        stop_price = current_price - (STOP_ATR_MULTIPLIER * atr)
        risk = current_price - stop_price
        target_price = current_price + (risk * TARGET_RR_RATIO)
    else:
        stop_price = current_price + (STOP_ATR_MULTIPLIER * atr)
        risk = stop_price - current_price
        target_price = current_price - (risk * TARGET_RR_RATIO)

    # Track outcome
    sweep_occurred = False
    sweep_candle_offset = 0
    entry_price = None
    hit_target = False
    hit_stop = False
    max_favorable = 0
    max_adverse = 0

    for i in range(1, forward_candles):
        candle = df.iloc[idx + i]

        # Check for sweep occurrence (based on original level direction, not trade direction)
        if direction == 'LONG':
            if not sweep_occurred:
                if candle['low'] <= level_price and candle['close'] > level_price:
                    sweep_occurred = True
                    sweep_candle_offset = i
                    entry_price = candle['close']
        else:
            if not sweep_occurred:
                if candle['high'] >= level_price and candle['close'] < level_price:
                    sweep_occurred = True
                    sweep_candle_offset = i
                    entry_price = candle['close']

        # After entry, track outcome based on TRADE direction
        if sweep_occurred and entry_price:
            if trade_direction == 'LONG':
                favorable = candle['high'] - entry_price
                adverse = entry_price - candle['low']
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)

                if candle['high'] >= target_price:
                    hit_target = True
                    break
                if candle['low'] <= stop_price:
                    hit_stop = True
                    break
            else:
                favorable = entry_price - candle['low']
                adverse = candle['high'] - entry_price
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)

                if candle['low'] <= target_price:
                    hit_target = True
                    break
                if candle['high'] >= stop_price:
                    hit_stop = True
                    break
    
    if not sweep_occurred:
        return None
    
    # Calculate actual R:R achieved
    actual_rr = max_favorable / max_adverse if max_adverse > 0 else 0
    
    return {
        'sweep_occurred': True,
        'sweep_candle_offset': sweep_candle_offset,
        'trade_direction': trade_direction,
        'hit_target': hit_target,
        'hit_stop': hit_stop,
        'won': hit_target and not hit_stop,
        'actual_rr': actual_rr,
        'max_favorable': max_favorable,
        'max_adverse': max_adverse,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY MODEL CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class QualityModel:
    """
    Quality prediction model with ML-driven direction selection.

    ML evaluates BOTH directions for each sweep and picks the best one:
    - Sweep of LOW → ML evaluates LONG (reversal) vs SHORT (continuation)
    - Sweep of HIGH → ML evaluates SHORT (reversal) vs LONG (continuation)

    Uses predict_best_direction() to compare both directions.
    """
    
    FEATURE_COLS = [
        # ═══════════════════════════════════════════════════════════════════
        # LAYER 1: RAW SIGNALS (Truth - what is happening?)
        # ═══════════════════════════════════════════════════════════════════
        'whale_pct',                   # Raw whale long % (0-100)
        'retail_pct',                  # Raw retail long % (0-100) - NEW
        'whale_delta_4h',              # Raw 4h change
        'whale_delta_24h',             # Raw 24h change
        'whale_delta_7d',              # Raw 7d change
        'oi_change_24h',               # Raw OI change %
        'price_change_24h',            # Raw price change %
        'volume_ratio',                # Raw volume vs average
        'volatility',                  # Raw ATR/price
        'funding_rate',                # Raw funding rate - NEW
        'momentum',                    # Raw momentum (price change / ATR)

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 2: ENGINEERED FEATURES (Context - how meaningful is this?)
        # ═══════════════════════════════════════════════════════════════════
        'distance_atr',                # Distance to level in ATR units
        'whale_retail_divergence',     # whale_pct - retail_pct (smart vs dumb money) - NEW
        'whale_acceleration_ratio',    # 24h vs 7d daily avg (is accumulation speeding up?)
        'whale_early_signal_ratio',    # 4h vs 24h ratio (early reversal detection)
        'whale_daily_avg_7d',          # Average daily change over 7d
        'level_type_score',            # Level strength (EQUAL > DOUBLE > SWING)
        'level_distance_score',        # How close to level
        'level_quality',               # Combined level quality
        # Price Action context
        'pa_candle_score',             # Candle pattern strength
        'pa_structure_score',          # Market structure score
        'pa_volume_score',             # Volume confirmation score
        'pa_momentum_score',           # Momentum confirmation
        'pa_total_score',              # Combined PA score

        # ═══════════════════════════════════════════════════════════════════
        # LAYER 3: RULE FLAGS AS FEATURES (ML learns when rules work/fail)
        # ═══════════════════════════════════════════════════════════════════
        # Whale rules (ML decides when to trust these)
        'rule_whale_bullish',          # 1 if whale_pct > 60 - NEW
        'rule_whale_bearish',          # 1 if whale_pct < 40 - NEW
        'rule_whale_accumulating',     # 1 if whale_delta_24h > 3
        'rule_whale_distributing',     # 1 if whale_delta_24h < -3
        'rule_whale_aligned',          # 1 if whale direction matches trade
        'rule_whale_trap',             # 1 if high % but negative delta (trap!)
        'rule_whale_strong',           # 1 if strong alignment
        # OI rules
        'rule_oi_increasing',          # 1 if OI going up (new positions)
        'rule_oi_decreasing',          # 1 if OI going down (closing positions)
        'rule_oi_price_aligned',       # 1 if OI and price moving together
        # Timing rules
        'rule_is_fresh_entry',         # 1 if catching early accumulation
        'rule_is_late_entry',          # 1 if already accumulated a lot
        'rule_early_accumulation',     # 1 if 4h bullish while 24h bearish (early signal!)
        'rule_early_distribution',     # 1 if 4h bearish while 24h bullish
        # Acceleration status (categorical - one-hot encoded)
        'whale_accel_accelerating',    # Accumulation speeding up
        'whale_accel_decelerating',    # Accumulation slowing down
        'whale_accel_reversing',       # Direction changing!
        'whale_accel_steady',          # Stable accumulation
        # Level rules
        'rule_level_tradeable',        # 1 if level meets quality threshold
        'pa_has_order_block',          # 1 if order block present
        'pa_has_fvg',                  # 1 if fair value gap present

        # ═══════════════════════════════════════════════════════════════════
        # REGIME FEATURES (different rules work in different regimes)
        # ═══════════════════════════════════════════════════════════════════
        'regime_volatility',           # High/Low volatility environment
        'regime_trend_strength',       # Trending vs ranging

        # ═══════════════════════════════════════════════════════════════════
        # SWEEP-SPECIFIC FEATURES (Core of liquidity hunting strategy!)
        # ═══════════════════════════════════════════════════════════════════
        # Layer 1 - Raw sweep data
        'sweep_depth_atr',             # How deep price went past level (in ATR)
        'sweep_wick_ratio',            # Rejection wick size vs candle range (0-1)
        'sweep_body_ratio',            # Body size vs candle range (0-1)
        'candles_since_sweep',         # How many candles ago (freshness)
        'sweep_volume_ratio',          # Volume on sweep candle vs average
        'price_change_since_sweep',    # % price change since sweep (negative = against direction)
        # Layer 3 - Sweep quality rules
        'rule_deep_sweep',             # 1 if sweep > 0.5 ATR past level
        'rule_strong_rejection',       # 1 if wick > 60% of candle
        'rule_fresh_sweep',            # 1 if within 5 candles
        'rule_volume_confirmed',       # 1 if sweep volume > 1.5x average
        'rule_price_aligned',          # 1 if price moved in expected direction since sweep

        # NEW: Follow-through & Distance features (Jan 2026 - identifies weak setups)
        'follow_through_per_candle',   # Price change % per candle since sweep (higher = stronger)
        'distance_from_sweep_pct',     # How far price is from sweep level (closer = retest risk)
        'indecision_candle',           # 1 if sweep candle body ≈ wick (choppy market)
        'rule_weak_follow_through',    # 1 if sweep 5+ candles ago but price moved <2%
        'rule_close_to_sweep_level',   # 1 if price within 3% of sweep level (retest risk)

        # PRE-SWEEP STRUCTURE (predicts traps vs real setups!)
        'structure_lower_highs',       # 1 if bearish structure (lower highs) before sweep
        'structure_higher_lows',       # 1 if bullish structure (higher lows) before sweep
        'structure_failed_bos',        # 1 if failed Break of Structure before sweep
        'structure_trend_aligned',     # 1 if sweep direction aligns with pre-sweep trend

        # IMPULSE vs CORRECTION (SMC key concept!)
        'is_impulse_move',             # 1 if sweep during impulse (expanding candles)
        'is_correction',               # 1 if sweep during correction (contracting candles)
        'swing_expansion',             # Size of current move relative to ATR (>1 = large move)

        # Direction encoding (ML decides best direction!)
        'is_trade_long',               # 1 if evaluating LONG trade, 0 if SHORT
        'is_sweep_of_low',             # 1 if sweep of LOW level, 0 if HIGH level
    ]
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.metrics = {}
        self.is_trained = False
        self.model_path = model_path or MODEL_PATH
    
    def train(self, samples: List[Dict], training_days: int = 180, n_symbols: int = 1) -> Dict:
        """
        Train the quality model.
        
        Args:
            samples: List of training samples
            training_days: Actual number of days used for training
        
        Label: 1 = Trade won (hit 1:1 target), 0 = Trade lost
        """
        if len(samples) < 100:
            return {'error': f'Need at least 100 samples, got {len(samples)}'}
        
        print(f"\n{'='*60}")
        print(f"[QUALITY_ML] Training on {len(samples)} samples")
        print(f"[QUALITY_ML] Strategy: ML-DRIVEN direction @ {TARGET_RR_RATIO}:1 R:R")
        print(f"[QUALITY_ML] → ML evaluates BOTH directions per sweep, picks best")
        print(f"{'='*60}")
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Filter to only samples with outcomes
        df = df[df['sweep_occurred'] == True].copy()
        print(f"[QUALITY_ML] Samples with sweeps: {len(df)}")

        # Show LONG vs SHORT distribution
        if 'direction' in df.columns:
            long_count = (df['direction'] == 'LONG').sum()
            short_count = (df['direction'] == 'SHORT').sum()
            long_win_rate = df[df['direction'] == 'LONG']['won'].mean() if long_count > 0 else 0
            short_win_rate = df[df['direction'] == 'SHORT']['won'].mean() if short_count > 0 else 0
            print(f"[QUALITY_ML] Trade direction split: {long_count} LONG ({long_win_rate:.1%} win), {short_count} SHORT ({short_win_rate:.1%} win)")

        # Show sweep direction distribution (what level was swept)
        if 'sweep_direction' in df.columns:
            sweep_low = (df['sweep_direction'] == 'LONG').sum()
            sweep_high = (df['sweep_direction'] == 'SHORT').sum()
            print(f"[QUALITY_ML] Sweep type split: {sweep_low} sweep-of-LOW, {sweep_high} sweep-of-HIGH")
            # Cross-reference: sweep LOW + trade LONG = reversal, sweep LOW + trade SHORT = continuation
            if 'direction' in df.columns:
                rev_low = ((df['sweep_direction'] == 'LONG') & (df['direction'] == 'LONG')).sum()
                cont_low = ((df['sweep_direction'] == 'LONG') & (df['direction'] == 'SHORT')).sum()
                rev_high = ((df['sweep_direction'] == 'SHORT') & (df['direction'] == 'SHORT')).sum()
                cont_high = ((df['sweep_direction'] == 'SHORT') & (df['direction'] == 'LONG')).sum()
                rev_low_wr = df[(df['sweep_direction'] == 'LONG') & (df['direction'] == 'LONG')]['won'].mean() if rev_low > 0 else 0
                cont_low_wr = df[(df['sweep_direction'] == 'LONG') & (df['direction'] == 'SHORT')]['won'].mean() if cont_low > 0 else 0
                rev_high_wr = df[(df['sweep_direction'] == 'SHORT') & (df['direction'] == 'SHORT')]['won'].mean() if rev_high > 0 else 0
                cont_high_wr = df[(df['sweep_direction'] == 'SHORT') & (df['direction'] == 'LONG')]['won'].mean() if cont_high > 0 else 0
                print(f"[QUALITY_ML]   Sweep LOW → LONG (reversal):     {rev_low} samples, {rev_low_wr:.1%} win")
                print(f"[QUALITY_ML]   Sweep LOW → SHORT (continuation): {cont_low} samples, {cont_low_wr:.1%} win")
                print(f"[QUALITY_ML]   Sweep HIGH → SHORT (reversal):    {rev_high} samples, {rev_high_wr:.1%} win")
                print(f"[QUALITY_ML]   Sweep HIGH → LONG (continuation): {cont_high} samples, {cont_high_wr:.1%} win")

        if len(df) < 50:
            return {'error': f'Need at least 50 sweep samples, got {len(df)}'}

        # Prepare features
        available_features = [f for f in self.FEATURE_COLS if f in df.columns]
        X = df[available_features].fillna(0)
        
        # Label: Did the trade WIN? (1:1 R:R or better)
        y = (df['won'] == True).astype(int)

        # SANITY CHECK: Raw win rate IS the strategy's base edge
        raw_win_rate = y.mean()
        print(f"\n🔍 SANITY CHECK - BASE STRATEGY PERFORMANCE:")
        print(f"   Raw win rate (no ML filtering): {raw_win_rate:.1%}")
        print(f"   At {TARGET_RR_RATIO}:1 R:R, break-even is {1/(TARGET_RR_RATIO+1):.1%}")
        if raw_win_rate > 1/(TARGET_RR_RATIO+1):
            print(f"   ✅ BASE STRATEGY IS PROFITABLE! ML will optimize further.")
        else:
            print(f"   ❌ Base strategy loses money. ML needs to find winning subset.")

        print(f"\n[QUALITY_ML] Win rate in data: {y.mean():.1%}")
        print(f"[QUALITY_ML] Features: {len(available_features)}")

        # ═══════════════════════════════════════════════════════════════════
        # TIME-BASED SPLIT (No data leakage!)
        # ═══════════════════════════════════════════════════════════════════
        # Sort by sample index (chronological order within each symbol)
        if 'sample_idx' in df.columns:
            df = df.sort_values('sample_idx').reset_index(drop=True)
            X = df[available_features].fillna(0)
            y = (df['won'] == True).astype(int)

        # Time-based split: Train on first 80%, Test on last 20% (future)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"[QUALITY_ML] ⏰ TIME-BASED SPLIT: Train on older {split_idx} samples, Test on newer {len(X)-split_idx} samples")
        
        # ═══════════════════════════════════════════════════════════════════
        # MULTI-MODEL COMPARISON
        # ═══════════════════════════════════════════════════════════════════

        # Define candidate models - INCREASED ITERATIONS for better learning
        candidate_models = {
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,  # Was 100 - more boosting rounds
                max_depth=5,       # Was 4 - slightly deeper trees
                min_samples_leaf=15,
                learning_rate=0.05,  # Slower learning, more iterations
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=300,  # Was 100 - more trees
                max_depth=8,       # Was 6 - deeper trees
                min_samples_leaf=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # Use all CPU cores
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=2000,     # Was 1000 - more iterations
                random_state=42,
                class_weight='balanced',
                C=0.5  # Regularization to prevent overfitting
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=150,
                learning_rate=0.5,
                # Use deeper trees to consider more features (not just decision stumps!)
                estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=10),
                random_state=42
            ),
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            candidate_models['XGBoost'] = XGBClassifier(
                n_estimators=200,  # Was 100
                max_depth=5,
                learning_rate=0.05,  # Slower learning
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1
            )

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            candidate_models['LightGBM'] = LGBMClassifier(
                n_estimators=200,  # Was 100
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )

        # Cross-validation for all models
        # USE TIME SERIES SPLIT - no shuffling, respects time order
        cv = TimeSeriesSplit(n_splits=5)  # Walk-forward validation
        model_results = {}

        print(f"\n{'─'*60}")
        print(f"MODEL COMPARISON (5-Fold Cross-Validation)")
        print(f"{'─'*60}")
        print(f"{'Model':<20} {'CV F1':>10} {'CV Acc':>10} {'Std':>8} {'Overfit':>10}")
        print(f"{'─'*60}")

        best_cv_f1 = -1
        best_model_name = None

        for model_name, model in candidate_models.items():
            try:
                # Cross-validation scores
                cv_f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
                cv_acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

                # Train on full train set to check overfitting
                model.fit(X_train, y_train)
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                overfit_gap = train_acc - test_acc

                cv_f1_mean = cv_f1_scores.mean()
                cv_f1_std = cv_f1_scores.std()
                cv_acc_mean = cv_acc_scores.mean()

                # Overfit indicator
                if overfit_gap > 0.15:
                    overfit_indicator = '🚨 HIGH'
                elif overfit_gap > 0.08:
                    overfit_indicator = '⚠️ MED'
                else:
                    overfit_indicator = '✅ LOW'

                model_results[model_name] = {
                    'model': model,
                    'cv_f1_mean': cv_f1_mean,
                    'cv_f1_std': cv_f1_std,
                    'cv_acc_mean': cv_acc_mean,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'overfit_gap': overfit_gap,
                }

                print(f"{model_name:<20} {cv_f1_mean:>9.1%} {cv_acc_mean:>9.1%} {cv_f1_std:>7.1%} {overfit_indicator:>10}")

                # Track best model (prioritize CV F1 with low overfitting penalty)
                adjusted_score = cv_f1_mean - (overfit_gap * 0.5)  # Penalize overfitting
                if adjusted_score > best_cv_f1:
                    best_cv_f1 = adjusted_score
                    best_model_name = model_name

            except Exception as e:
                print(f"{model_name:<20} {'ERROR':>10} - {str(e)[:30]}")

        print(f"{'─'*60}")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"{'─'*60}")

        # Use the best model
        best_result = model_results[best_model_name]
        self.model = best_result['model']
        self.model.fit(X_train, y_train)  # Retrain on train set

        # Store all model results for later analysis
        self.model_comparison = model_results
        self.best_model_name = best_model_name

        # ═══════════════════════════════════════════════════════════════════
        # OVERFITTING DETECTION (for best model)
        # ═══════════════════════════════════════════════════════════════════

        # 1. Train vs Test accuracy (key overfitting indicator)
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # Use test metrics as primary metrics
        accuracy = test_accuracy
        f1 = test_f1
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        # 2. Cross-validation for robust estimate (use cached results)
        cv_f1_mean = best_result['cv_f1_mean']
        cv_f1_std = best_result['cv_f1_std']
        cv_acc_mean = best_result['cv_acc_mean']
        cv_acc_std = cv_f1_std  # Approximate

        # 3. Overfitting score: how much worse is test vs train
        # >10% gap = overfitting warning, >20% = severe overfitting
        overfit_gap_accuracy = train_accuracy - test_accuracy
        overfit_gap_f1 = train_f1 - test_f1

        if overfit_gap_accuracy > 0.20 or overfit_gap_f1 > 0.20:
            overfit_status = 'SEVERE'
        elif overfit_gap_accuracy > 0.10 or overfit_gap_f1 > 0.10:
            overfit_status = 'WARNING'
        elif overfit_gap_accuracy > 0.05 or overfit_gap_f1 > 0.05:
            overfit_status = 'MILD'
        else:
            overfit_status = 'OK'
        
        # Feature importance (handle different model types)
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (RandomForest, GradientBoosting, XGBoost, LightGBM)
            feature_importance = dict(zip(available_features, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            # Linear models (LogisticRegression, SVM)
            # Use absolute value of coefficients as importance
            coef = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            feature_importance = dict(zip(available_features, coef))
        else:
            # Fallback - no feature importance available
            feature_importance = {f: 0 for f in available_features}

        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate expected performance
        # If we only take trades where model says "YES" (prob > 0.5)
        high_conf_mask = y_prob >= 0.5
        if high_conf_mask.sum() > 0:
            high_conf_win_rate = y_test[high_conf_mask].mean()
            high_conf_trades = high_conf_mask.sum()
        else:
            high_conf_win_rate = 0
            high_conf_trades = 0
        
        # Very high confidence (prob > 0.6)
        very_high_mask = y_prob >= 0.6
        if very_high_mask.sum() > 0:
            very_high_win_rate = y_test[very_high_mask].mean()
            very_high_trades = very_high_mask.sum()
        else:
            very_high_win_rate = 0
            very_high_trades = 0

        # ═══════════════════════════════════════════════════════════════
        # LONG vs SHORT SPLIT ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        # Get direction from test set
        df_test = df.iloc[split_idx:].copy()

        # Check if 'direction' column exists
        if 'direction' in df_test.columns:
            is_long_test = df_test['direction'] == 'LONG'
            is_short_test = df_test['direction'] == 'SHORT'

            # LONG trades analysis
            long_mask = is_long_test.values
            if long_mask.sum() > 0:
                long_base_win = y_test[long_mask].mean()
                long_total = long_mask.sum()

                # High conf LONG
                long_high_mask = long_mask & high_conf_mask
                if long_high_mask.sum() > 0:
                    long_high_win = y_test[long_high_mask].mean()
                    long_high_trades = long_high_mask.sum()
                else:
                    long_high_win = 0
                    long_high_trades = 0

                # Very high conf LONG
                long_very_high_mask = long_mask & very_high_mask
                if long_very_high_mask.sum() > 0:
                    long_very_high_win = y_test[long_very_high_mask].mean()
                    long_very_high_trades = long_very_high_mask.sum()
                else:
                    long_very_high_win = 0
                    long_very_high_trades = 0
            else:
                long_base_win = long_high_win = long_very_high_win = 0
                long_total = long_high_trades = long_very_high_trades = 0

            # SHORT trades analysis
            short_mask = is_short_test.values
            if short_mask.sum() > 0:
                short_base_win = y_test[short_mask].mean()
                short_total = short_mask.sum()

                # High conf SHORT
                short_high_mask = short_mask & high_conf_mask
                if short_high_mask.sum() > 0:
                    short_high_win = y_test[short_high_mask].mean()
                    short_high_trades = short_high_mask.sum()
                else:
                    short_high_win = 0
                    short_high_trades = 0

                # Very high conf SHORT
                short_very_high_mask = short_mask & very_high_mask
                if short_very_high_mask.sum() > 0:
                    short_very_high_win = y_test[short_very_high_mask].mean()
                    short_very_high_trades = short_very_high_mask.sum()
                else:
                    short_very_high_win = 0
                    short_very_high_trades = 0
            else:
                short_base_win = short_high_win = short_very_high_win = 0
                short_total = short_high_trades = short_very_high_trades = 0

            # Store for later printing
            direction_split = {
                'long': {
                    'total': long_total,
                    'base_win': long_base_win,
                    'high_win': long_high_win,
                    'high_trades': long_high_trades,
                    'very_high_win': long_very_high_win,
                    'very_high_trades': long_very_high_trades
                },
                'short': {
                    'total': short_total,
                    'base_win': short_base_win,
                    'high_win': short_high_win,
                    'high_trades': short_high_trades,
                    'very_high_win': short_very_high_win,
                    'very_high_trades': short_very_high_trades
                }
            }
        else:
            direction_split = None

        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: Trade frequency and duration analysis
        # ═══════════════════════════════════════════════════════════════
        
        # Get ALL predictions on full dataset (not just test)
        X_full = df[available_features].fillna(0)
        y_full_prob = self.model.predict_proba(X_full)[:, 1]
        
        # Count high/very high confidence trades in FULL dataset
        full_high_conf = (y_full_prob >= 0.5).sum()
        full_very_high = (y_full_prob >= 0.6).sum()
        
        # Use the actual training_days passed to this function
        # (training_days is a parameter of the train() method)
        
        # Estimate trades per month
        # CORRECT: Just divide total trades by number of months
        months_of_training = training_days / 30
        
        # Realistic cap: a trader can only take ~10 very-high-confidence trades/month
        MAX_TRADES_PER_MONTH = 10
        n_sym = max(n_symbols, 1)

        if months_of_training > 0:
            very_high_per_month_raw = full_very_high / months_of_training
            high_conf_per_month_raw = full_high_conf / months_of_training

            # Per-symbol rate (actual signal frequency per asset)
            very_high_per_symbol_month = very_high_per_month_raw / n_sym
            high_conf_per_symbol_month = high_conf_per_month_raw / n_sym

            # Capped at realistic trading capacity
            very_high_per_month = min(very_high_per_month_raw, MAX_TRADES_PER_MONTH)
            high_conf_per_month = min(high_conf_per_month_raw, MAX_TRADES_PER_MONTH * 2)
        else:
            very_high_per_month = 0
            high_conf_per_month = 0
            very_high_per_symbol_month = 0
            high_conf_per_symbol_month = 0
        
        # Monthly ROI = EV per trade × risk% × trades per month
        # EV per trade = Win% × R:R - Loss% × 1
        # With 1% risk per trade, multiply EV by 0.01
        RISK_PER_TRADE = 0.01  # 1% of account per trade

        if very_high_win_rate > 0:
            ev_per_trade = very_high_win_rate * TARGET_RR_RATIO - (1 - very_high_win_rate) * 1
            roi_per_trade_pct = ev_per_trade * RISK_PER_TRADE * 100  # % of account per trade
            monthly_roi_very_high = roi_per_trade_pct * very_high_per_month
        else:
            roi_per_trade_pct = 0
            monthly_roi_very_high = 0

        if high_conf_win_rate > 0:
            ev_per_trade_high = high_conf_win_rate * TARGET_RR_RATIO - (1 - high_conf_win_rate) * 1
            roi_per_trade_high = ev_per_trade_high * RISK_PER_TRADE * 100
            monthly_roi_high = roi_per_trade_high * high_conf_per_month
        else:
            roi_per_trade_high = 0
            monthly_roi_high = 0
        
        self.metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'total_samples': len(df),
            'sweep_samples': len(df),
            'base_win_rate': y.mean(),
            'high_conf_win_rate': high_conf_win_rate,
            'high_conf_trades': high_conf_trades,  # In test set
            'high_conf_trades_full': full_high_conf,  # In FULL dataset
            'very_high_win_rate': very_high_win_rate,
            'very_high_trades': very_high_trades,  # In test set
            'very_high_trades_full': full_very_high,  # In FULL dataset
            'features_used': len(available_features),
            'top_features': top_features,
            'trained_at': datetime.now().isoformat(),
            # Trade frequency (capped at realistic max)
            'training_days_est': training_days,
            'n_symbols_trained': n_symbols,
            'very_high_per_month': very_high_per_month,
            'high_conf_per_month': high_conf_per_month,
            'very_high_per_symbol_month': very_high_per_symbol_month,
            'high_conf_per_symbol_month': high_conf_per_symbol_month,
            # Monthly projections
            'monthly_roi_very_high': monthly_roi_very_high,
            'monthly_roi_high': monthly_roi_high,
            # OVERFITTING METRICS (NEW)
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'cv_f1_mean': cv_f1_mean,
            'cv_f1_std': cv_f1_std,
            'cv_accuracy_mean': cv_acc_mean,
            'cv_accuracy_std': cv_acc_std,
            'overfit_gap_accuracy': overfit_gap_accuracy,
            'overfit_gap_f1': overfit_gap_f1,
            'overfit_status': overfit_status,
            'test_set_size': len(y_test),
            'train_set_size': len(y_train),
            # MODEL COMPARISON (NEW)
            'best_model': best_model_name,
            'models_tested': list(model_results.keys()),
            'model_scores': {name: {'cv_f1': r['cv_f1_mean'], 'overfit': r['overfit_gap']}
                           for name, r in model_results.items()},
        }
        
        # Calculate expected ROI based on TARGET_RR_RATIO
        # EV = (Win% × R:R) - (Loss% × 1) = Win% × (R:R + 1) - 1
        # At 2:1 R:R: EV = 3W - 1, break even at W = 33.3%
        rr_multiplier = TARGET_RR_RATIO + 1  # e.g., 2:1 R:R → multiplier = 3

        base_roi = rr_multiplier * y.mean() - 1
        high_conf_roi = rr_multiplier * high_conf_win_rate - 1 if high_conf_win_rate > 0 else 0
        very_high_roi = rr_multiplier * very_high_win_rate - 1 if very_high_win_rate > 0 else 0

        self.metrics['base_roi_per_trade'] = base_roi * 100
        self.metrics['high_conf_roi_per_trade'] = high_conf_roi * 100
        self.metrics['very_high_roi_per_trade'] = very_high_roi * 100
        self.metrics['target_rr_ratio'] = TARGET_RR_RATIO
        self.metrics['break_even_win_rate'] = 1 / rr_multiplier  # e.g., 33.3% for 2:1
        
        self.is_trained = True
        
        # Save model
        self._save()
        
        print(f"\n{'='*60}")
        print(f"[QUALITY_ML] TRAINING COMPLETE!")
        print(f"{'='*60}")

        # ═══════════════════════════════════════════════════════════════════
        # OVERFITTING CHECK (CRITICAL!)
        # ═══════════════════════════════════════════════════════════════════
        if overfit_status == 'SEVERE':
            overfit_icon = '🚨'
        elif overfit_status == 'WARNING':
            overfit_icon = '⚠️'
        elif overfit_status == 'MILD':
            overfit_icon = '⚡'
        else:
            overfit_icon = '✅'

        print(f"\n{overfit_icon} OVERFITTING CHECK: {overfit_status}")
        print(f"{'─'*60}")
        print(f"  Train Accuracy: {train_accuracy:.1%}  |  Test Accuracy: {test_accuracy:.1%}  |  Gap: {overfit_gap_accuracy:+.1%}")
        print(f"  Train F1:       {train_f1:.1%}  |  Test F1:       {test_f1:.1%}  |  Gap: {overfit_gap_f1:+.1%}")
        print(f"  5-Fold CV F1:   {cv_f1_mean:.1%} ± {cv_f1_std:.1%}")
        print(f"  5-Fold CV Acc:  {cv_acc_mean:.1%} ± {cv_acc_std:.1%}")
        print(f"  Train/Test Split: {len(y_train)}/{len(y_test)} samples")

        if overfit_status == 'SEVERE':
            print(f"\n  🚨 SEVERE OVERFITTING DETECTED!")
            print(f"     Model is memorizing training data, not learning patterns.")
            print(f"     Recommendations: More data, simpler model, or regularization.")
        elif overfit_status == 'WARNING':
            print(f"\n  ⚠️ OVERFITTING WARNING!")
            print(f"     Train >> Test accuracy. Results may not generalize well.")
            print(f"     Consider: More training data or cross-validate with more folds.")
        elif overfit_status == 'MILD':
            print(f"\n  ⚡ Mild overfitting - acceptable but monitor on live data.")

        print(f"\n{'─'*60}")
        print(f"Test Set Performance @ {TARGET_RR_RATIO}:1 R:R (Break-even: {1/(TARGET_RR_RATIO+1):.1%})")
        print(f"{'─'*60}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"F1 Score: {f1:.1%}")
        print(f"Precision: {precision:.1%}")
        print(f"Recall: {recall:.1%}")
        print(f"")
        print(f"{'─'*60}")
        print(f"WIN RATES & ROI @ {TARGET_RR_RATIO}:1 R:R")
        print(f"{'─'*60}")
        print(f"Base Win Rate: {y.mean():.1%} → ROI: {base_roi*100:+.1f}% {'✅ PROFIT' if base_roi > 0 else '❌ LOSS'}")
        print(f"High Conf (>50%): {high_conf_win_rate:.1%} ({high_conf_trades} test) → ROI: {high_conf_roi*100:+.1f}% {'✅ PROFIT' if high_conf_roi > 0 else '❌ LOSS'}")
        print(f"Very High (>60%): {very_high_win_rate:.1%} ({very_high_trades} test) → ROI: {very_high_roi*100:+.1f}% {'✅ PROFIT' if very_high_roi > 0 else '❌ LOSS'}")

        # Print LONG vs SHORT split if available
        if direction_split:
            rr = TARGET_RR_RATIO + 1
            print(f"")
            print(f"{'─'*60}")
            print(f"📊 LONG vs SHORT BREAKDOWN (Test Set)")
            print(f"{'─'*60}")

            ls = direction_split['long']
            ss = direction_split['short']

            # Calculate ROIs
            long_base_roi = rr * ls['base_win'] - 1 if ls['base_win'] > 0 else -1
            long_high_roi = rr * ls['high_win'] - 1 if ls['high_win'] > 0 else -1
            long_vh_roi = rr * ls['very_high_win'] - 1 if ls['very_high_win'] > 0 else -1

            short_base_roi = rr * ss['base_win'] - 1 if ss['base_win'] > 0 else -1
            short_high_roi = rr * ss['high_win'] - 1 if ss['high_win'] > 0 else -1
            short_vh_roi = rr * ss['very_high_win'] - 1 if ss['very_high_win'] > 0 else -1

            print(f"")
            print(f"🟢 LONG TRADES ({ls['total']} samples):")
            print(f"   Base Win Rate:    {ls['base_win']:.1%} → ROI: {long_base_roi*100:+.1f}%")
            print(f"   High Conf (>50%): {ls['high_win']:.1%} ({ls['high_trades']} trades) → ROI: {long_high_roi*100:+.1f}%")
            print(f"   Very High (>60%): {ls['very_high_win']:.1%} ({ls['very_high_trades']} trades) → ROI: {long_vh_roi*100:+.1f}%")

            print(f"")
            print(f"🔴 SHORT TRADES ({ss['total']} samples):")
            print(f"   Base Win Rate:    {ss['base_win']:.1%} → ROI: {short_base_roi*100:+.1f}%")
            print(f"   High Conf (>50%): {ss['high_win']:.1%} ({ss['high_trades']} trades) → ROI: {short_high_roi*100:+.1f}%")
            print(f"   Very High (>60%): {ss['very_high_win']:.1%} ({ss['very_high_trades']} trades) → ROI: {short_vh_roi*100:+.1f}%")

            # Comparison
            print(f"")
            print(f"⚖️  COMPARISON (at >60% confidence):")
            if ls['very_high_win'] > 0 and ss['very_high_win'] > 0:
                if ls['very_high_win'] > ss['very_high_win']:
                    diff = ls['very_high_win'] - ss['very_high_win']
                    print(f"   LONG outperforms SHORT by {diff:.1%} win rate")
                else:
                    diff = ss['very_high_win'] - ls['very_high_win']
                    print(f"   SHORT outperforms LONG by {diff:.1%} win rate")
            elif ls['very_high_win'] > 0:
                print(f"   Only LONG has >60% confidence trades")
            elif ss['very_high_win'] > 0:
                print(f"   Only SHORT has >60% confidence trades")

        print(f"")
        print(f"Trade Frequency ({n_symbols} symbols, capped at {MAX_TRADES_PER_MONTH} trades/month):")
        print(f"  Training period: ~{training_days:.0f} days")
        print(f"  Very High per symbol/month: ~{very_high_per_symbol_month:.2f}")
        print(f"  Very High trades/month (capped): ~{very_high_per_month:.1f}")
        print(f"  High Conf trades/month (capped): ~{high_conf_per_month:.1f}")
        print(f"")
        print(f"Monthly ROI Projection (at {TARGET_RR_RATIO}:1 R:R, 1% risk per trade):")
        print(f"  EV per Very High trade: {roi_per_trade_pct:.2f}% of account")
        print(f"  Very High only: {monthly_roi_very_high:+.1f}%")
        print(f"  High Conf only: {monthly_roi_high:+.1f}%")
        print(f"")
        print(f"Top 15 Features:")
        for feat, imp in top_features[:15]:
            print(f"  {feat}: {imp:.3f}")

        # Show whale acceleration features specifically
        whale_accel_features = ['whale_delta_24h', 'whale_delta_7d', 'whale_daily_avg_7d',
                                'whale_acceleration_ratio', 'whale_accel_accelerating',
                                'whale_accel_decelerating', 'whale_accel_reversing',
                                'whale_accel_steady', 'rule_is_fresh_entry', 'rule_is_late_entry']
        print(f"\nWhale Acceleration Features:")
        for feat in whale_accel_features:
            imp = feature_importance.get(feat, 0)
            bar = "█" * int(imp * 20) if imp > 0 else ""
            print(f"  {feat}: {imp:.4f} {bar}")

        # Show flow feature stats (ETF/Stock — values mapped into whale slots)
        flow_mapped = ['whale_pct', 'whale_delta_4h', 'whale_delta_24h', 'whale_delta_7d',
                        'retail_pct', 'funding_rate', 'oi_change_24h']
        flow_labels = ['flow_score', 'mfi_value', 'cmf_value', 'ext_ema200',
                        'ext_ema50', 'inst_score', 'options_score']
        print(f"\nFlow/Whale Feature Stats (ETF=flow values, Crypto=whale values):")
        for feat, label in zip(flow_mapped, flow_labels):
            if feat in df.columns:
                mean_val = df[feat].mean()
                std_val = df[feat].std()
                min_val = df[feat].min()
                max_val = df[feat].max()
                print(f"  {feat} ({label}): mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")

        # Show sweep-specific features
        sweep_features = ['sweep_depth_atr', 'sweep_wick_ratio', 'sweep_body_ratio',
                          'candles_since_sweep', 'sweep_volume_ratio',
                          'rule_deep_sweep', 'rule_strong_rejection', 'rule_fresh_sweep', 'rule_volume_confirmed']
        print(f"\nSweep-Specific Features (NEW):")
        for feat in sweep_features:
            imp = feature_importance.get(feat, 0)
            bar = "█" * int(imp * 20) if imp > 0 else ""
            print(f"  {feat}: {imp:.4f} {bar}")

        # Check if sweep features have variance (if they're all 0 or constant, they won't help)
        print(f"\nSweep Feature Stats (checking for variance):")
        for feat in sweep_features:
            if feat in df.columns:
                mean_val = df[feat].mean()
                std_val = df[feat].std()
                min_val = df[feat].min()
                max_val = df[feat].max()
                print(f"  {feat}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
            else:
                print(f"  {feat}: NOT IN DATA ⚠️")
        
        return self.metrics
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict if this is a quality setup.
        """
        if not self.is_trained:
            self._load()
        
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Prepare features
        X = pd.DataFrame([features])[self.FEATURE_COLS].fillna(0)
        
        prob = self.model.predict_proba(X)[0][1]
        
        if prob >= 0.6:
            decision = 'STRONG_YES'
            take_trade = True
        elif prob >= 0.5:
            decision = 'YES'
            take_trade = True
        elif prob >= 0.4:
            decision = 'MAYBE'
            take_trade = False
        else:
            decision = 'NO'
            take_trade = False
        
        return {
            'probability': prob,
            'decision': decision,
            'take_trade': take_trade,
            'expected_win_rate': prob,
            'expected_roi': (2 * prob - 1) * 100  # At 1:1 R:R
        }
    
    def predict_best_direction(self, features: Dict) -> Dict:
        """
        Predict BOTH directions and pick the best one.

        Runs the model twice:
        1. With is_trade_long=1 (LONG trade)
        2. With is_trade_long=0 (SHORT trade)

        Returns the direction with higher probability + both probabilities.
        """
        if not self.is_trained:
            self._load()

        if self.model is None:
            return {'error': 'Model not trained', 'best_direction': 'LONG', 'long_probability': 0.5, 'short_probability': 0.5}

        # Predict LONG
        features_long = {**features, 'is_trade_long': 1}
        X_long = pd.DataFrame([features_long])[self.FEATURE_COLS].fillna(0)
        prob_long = self.model.predict_proba(X_long)[0][1]

        # Predict SHORT
        features_short = {**features, 'is_trade_long': 0}
        X_short = pd.DataFrame([features_short])[self.FEATURE_COLS].fillna(0)
        prob_short = self.model.predict_proba(X_short)[0][1]

        # Pick best direction
        if prob_long >= prob_short:
            best_dir = 'LONG'
            best_prob = prob_long
        else:
            best_dir = 'SHORT'
            best_prob = prob_short

        # Decision based on best probability
        if best_prob >= 0.6:
            decision = 'STRONG_YES'
            take_trade = True
        elif best_prob >= 0.5:
            decision = 'YES'
            take_trade = True
        elif best_prob >= 0.4:
            decision = 'MAYBE'
            take_trade = False
        else:
            decision = 'NO'
            take_trade = False

        return {
            'best_direction': best_dir,
            'long_probability': prob_long,
            'short_probability': prob_short,
            'probability': best_prob,
            'decision': decision,
            'take_trade': take_trade,
            'expected_win_rate': best_prob,
            'expected_roi': (2 * best_prob - 1) * 100
        }

    def _save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metrics': self.metrics,
                'is_trained': self.is_trained
            }, f)
        print(f"[QUALITY_ML] Model saved to {self.model_path}")

    def _load(self):
        """Load model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.metrics = data['metrics']
                self.is_trained = data['is_trained']
            print(f"[QUALITY_ML] Model loaded from {self.model_path}")
        else:
            print(f"[QUALITY_ML] No saved model found at {self.model_path}")

    def get_feature_importance(self) -> Dict:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            self._load()

        if self.model is None:
            return {'error': 'Model not trained'}

        result = {
            'feature_cols': self.FEATURE_COLS,
            'num_features': len(self.FEATURE_COLS),
            'importances': {},
            'whale_accel_features': {}
        }

        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, feat in enumerate(self.FEATURE_COLS):
                if i < len(importances):
                    result['importances'][feat] = float(importances[i])
        elif hasattr(self.model, 'coef_'):
            coef = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            for i, feat in enumerate(self.FEATURE_COLS):
                if i < len(coef):
                    result['importances'][feat] = float(coef[i])

        # Extract whale acceleration features specifically
        whale_accel = ['whale_delta_24h', 'whale_delta_7d', 'whale_daily_avg_7d',
                       'whale_acceleration_ratio', 'whale_accel_accelerating',
                       'whale_accel_decelerating', 'whale_accel_reversing',
                       'whale_accel_steady', 'whale_is_fresh', 'whale_is_late_entry']

        for feat in whale_accel:
            result['whale_accel_features'][feat] = result['importances'].get(feat, 'NOT_FOUND')

        return result


# Global instance
_quality_model = None

def get_quality_model() -> QualityModel:
    global _quality_model
    if _quality_model is None:
        _quality_model = QualityModel()
        _quality_model._load()
    return _quality_model


def get_quality_model_status() -> Dict:
    """Get current model status."""
    model = get_quality_model()
    return {
        'is_trained': model.is_trained,
        'metrics': model.metrics
    }


# ─── ETF Model ────────────────────────────────────────────────────────────────
_quality_model_etf = None

def get_quality_model_etf() -> QualityModel:
    global _quality_model_etf
    if _quality_model_etf is None:
        _quality_model_etf = QualityModel(model_path=MODEL_PATH_ETF)
        _quality_model_etf._load()
    return _quality_model_etf

def get_quality_model_etf_status() -> Dict:
    model = get_quality_model_etf()
    return {'is_trained': model.is_trained, 'metrics': model.metrics}


# ─── Stock Model ──────────────────────────────────────────────────────────────
_quality_model_stock = None

def get_quality_model_stock() -> QualityModel:
    global _quality_model_stock
    if _quality_model_stock is None:
        _quality_model_stock = QualityModel(model_path=MODEL_PATH_STOCK)
        _quality_model_stock._load()
    return _quality_model_stock

def get_quality_model_stock_status() -> Dict:
    model = get_quality_model_stock()
    return {'is_trained': model.is_trained, 'metrics': model.metrics}


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_quality_samples(
    df: pd.DataFrame,
    symbol: str,
    whale_history: List[Dict] = None,
    forward_candles: int = None,
    market_type: str = 'crypto'
) -> List[Dict]:
    """
    Generate training samples for quality model.
    
    Only includes STRONG levels (EQUAL, DOUBLE).
    """
    _fwd = forward_candles or FORWARD_CANDLES
    if df is None or len(df) < LOOKBACK_CANDLES + _fwd + 50:
        print(f"[QUALITY_ML] {symbol}: Not enough data ({len(df) if df is not None else 0} candles)")
        return []
    
    # Normalize
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    atr_current = df['atr'].iloc[-1]
    if pd.isna(atr_current) or atr_current <= 0:
        atr_current = df['close'].iloc[-1] * 0.02
    
    # Find liquidity levels using BUILT-IN function
    liq_levels = _find_liquidity_levels_simple(df, atr_current)
    
    # Count by type
    all_levels = liq_levels['lows'] + liq_levels['highs']
    equal_count = sum(1 for _, _, t, _ in all_levels if 'EQUAL' in t)
    double_count = sum(1 for _, _, t, _ in all_levels if 'DOUBLE' in t)
    swing_count = sum(1 for _, _, t, _ in all_levels if 'SWING' in t)
    
    print(f"[QUALITY_ML] {symbol}: {equal_count} EQUAL, {double_count} DOUBLE, {swing_count} SWING levels")
    
    if equal_count + double_count == 0:
        print(f"[QUALITY_ML] {symbol}: No EQUAL/DOUBLE levels found, including SWING levels for training")
    
    samples = []

    # Get whale data with 4h, 24h AND 7d delta calculation
    # For 4h candles: 4h = 1 reading, 24h = 6 candles, 7d = 42 candles
    def get_whale_at_time(idx, direction='LONG'):
        """
        Calculate whale metrics at a specific candle index.
        Returns: whale_pct, whale_delta, whale_delta_4h, whale_delta_24h, whale_delta_7d, whale_acceleration, whale_early_signal, is_fresh, oi_change_24h, retail_pct, funding_rate
        """
        whale_pct = 50
        whale_delta = 0
        whale_delta_4h = 0   # NEW: Early signal
        whale_delta_24h = 0
        whale_delta_7d = 0
        whale_acceleration = 'UNKNOWN'
        whale_early_signal = None  # NEW: EARLY_ACCUMULATION, EARLY_DISTRIBUTION, etc.
        is_fresh_accumulation = False
        oi_change_24h = 0  # OI change from whale_history
        retail_pct = 50    # NEW: Retail long %
        funding_rate = 0   # NEW: Funding rate

        if whale_history and len(whale_history) > 0:
            # Use index-based matching (simple approach for 4h candles)
            # whale_history is ordered by timestamp
            wh_len = len(whale_history)

            # Map candle index to approximate whale history index
            # Assume roughly 1 whale reading per 4-6 hours
            wh_idx = min(idx // 2, wh_len - 1)  # Rough mapping

            if wh_idx >= 0 and wh_idx < wh_len:
                whale_pct = whale_history[wh_idx].get('whale_pct', whale_history[wh_idx].get('whale_long_pct', 50))
                # Get additional raw features from whale_history
                oi_change_24h = whale_history[wh_idx].get('oi_change_24h', 0) or 0
                retail_pct = whale_history[wh_idx].get('retail_pct', whale_history[wh_idx].get('retail_long_pct', 50)) or 50
                funding_rate = whale_history[wh_idx].get('funding_rate', 0) or 0

                # Calculate 4h delta (1 reading back) - EARLY SIGNAL
                wh_idx_4h = max(0, wh_idx - 1)
                if wh_idx_4h < wh_len and wh_idx_4h != wh_idx:
                    whale_4h_ago = whale_history[wh_idx_4h].get('whale_pct', whale_history[wh_idx_4h].get('whale_long_pct', 50))
                    whale_delta_4h = whale_pct - whale_4h_ago

                # Calculate 24h delta (6 candles back = ~3 whale readings)
                wh_idx_24h = max(0, wh_idx - 3)
                if wh_idx_24h < wh_len:
                    whale_24h_ago = whale_history[wh_idx_24h].get('whale_pct', whale_history[wh_idx_24h].get('whale_long_pct', 50))
                    whale_delta_24h = whale_pct - whale_24h_ago

                # Calculate 7d delta (42 candles back = ~21 whale readings)
                wh_idx_7d = max(0, wh_idx - 21)
                if wh_idx_7d < wh_len:
                    whale_7d_ago = whale_history[wh_idx_7d].get('whale_pct', whale_history[wh_idx_7d].get('whale_long_pct', 50))
                    whale_delta_7d = whale_pct - whale_7d_ago

                # Use 24h delta as primary delta
                whale_delta = whale_delta_24h

                # Calculate acceleration
                daily_avg_7d = whale_delta_7d / 7 if whale_delta_7d else 0

                if abs(whale_delta_24h) > abs(daily_avg_7d) * 1.5:
                    if (whale_delta_24h > 0 and whale_delta_7d > 0) or (whale_delta_24h < 0 and whale_delta_7d < 0):
                        whale_acceleration = 'ACCELERATING'
                    else:
                        whale_acceleration = 'REVERSING'
                elif abs(whale_delta_24h) < abs(daily_avg_7d) * 0.5:
                    whale_acceleration = 'DECELERATING'
                else:
                    whale_acceleration = 'STEADY'

                # NEW: Detect EARLY reversal signals (4h vs 24h)
                if whale_delta_24h < -2 and whale_delta_4h > 1:
                    whale_early_signal = 'EARLY_ACCUMULATION'
                elif whale_delta_24h > 2 and whale_delta_4h < -1:
                    whale_early_signal = 'EARLY_DISTRIBUTION'
                elif whale_delta_4h > 2 and whale_delta_24h > 0:
                    whale_early_signal = 'FRESH_ACCUMULATION'
                elif whale_delta_4h < -2 and whale_delta_24h < 0:
                    whale_early_signal = 'FRESH_DISTRIBUTION'

                # Fresh accumulation check
                if direction == 'LONG' and whale_delta_24h > daily_avg_7d and whale_delta_24h > 2:
                    is_fresh_accumulation = True
                elif direction == 'SHORT' and whale_delta_24h < daily_avg_7d and whale_delta_24h < -2:
                    is_fresh_accumulation = True

        return whale_pct, whale_delta, whale_delta_4h, whale_delta_24h, whale_delta_7d, whale_acceleration, whale_early_signal, is_fresh_accumulation, oi_change_24h, retail_pct, funding_rate
    
    # Process each candle
    for i in range(LOOKBACK_CANDLES + 20, len(df) - _fwd):
        current_price = df['close'].iloc[i]
        atr = df['atr'].iloc[i]

        if pd.isna(atr) or atr <= 0:
            continue

        # Find nearby levels - ONLY EQUAL and DOUBLE (no SWING - too much noise)
        # This filters for high-quality liquidity levels only
        nearby_lows = [
            (idx, price, ltype, strength)
            for idx, price, ltype, strength in liq_levels['lows']
            if idx < i and i - idx <= LOOKBACK_CANDLES and price < current_price
            and ('EQUAL' in ltype or 'DOUBLE' in ltype)  # Only strong levels
        ]

        nearby_highs = [
            (idx, price, ltype, strength)
            for idx, price, ltype, strength in liq_levels['highs']
            if idx < i and i - idx <= LOOKBACK_CANDLES and price > current_price
            and ('EQUAL' in ltype or 'DOUBLE' in ltype)  # Only strong levels
        ]

        # ═══════════════════════════════════════════════════════════════════
        # CONTINUATION + REVERSAL TRAINING
        # ML learns BOTH patterns and decides which context favors which:
        #
        # CONTINUATION (trade opposite to sweep):
        #   Sweep of LOW → SHORT (price continues down after grabbing liquidity)
        #   Sweep of HIGH → LONG (price continues up after grabbing liquidity)
        #
        # REVERSAL (trade same as sweep):
        #   Sweep of LOW → LONG (price bounces after sweeping lows - trap for shorts)
        #   Sweep of HIGH → SHORT (price drops after sweeping highs - trap for longs)
        #
        # Features like structure_lower_highs, structure_failed_bos help ML decide
        # ═══════════════════════════════════════════════════════════════════

        def generate_continuation_sample(level_idx, level_price, level_type, level_strength, sweep_direction, is_low):
            """Generate CONTINUATION sample: trade opposite to sweep direction."""
            if sweep_direction == 'LONG':
                distance_atr = (current_price - level_price) / atr
            else:
                distance_atr = (level_price - current_price) / atr

            if distance_atr > 3:
                return

            # Continuation: trade OPPOSITE to sweep direction
            trade_dir = 'SHORT' if sweep_direction == 'LONG' else 'LONG'

            # Get whale data
            whale_pct, whale_delta, whale_delta_4h, whale_delta_24h, whale_delta_7d, whale_acceleration, whale_early_signal, is_fresh, oi_change_24h, retail_pct, funding_rate = get_whale_at_time(i, trade_dir)

            # Calculate price change 24h
            price_change_24h = 0
            if i >= 6:
                price_24h_ago = df['close'].iloc[i - 6]
                if price_24h_ago > 0:
                    price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100

            outcome = label_trade_outcome(df, i, level_price, sweep_direction, atr, trade_direction=trade_dir, forward_candles=_fwd)
            if outcome is None:
                return

            sweep_offset = outcome.get('sweep_candle_offset', 1)
            sweep_candle_abs_idx = i + sweep_offset

            detection_delay = random.randint(1, 10)
            eval_point = min(sweep_candle_abs_idx + detection_delay, len(df) - 1)

            hist_df = df.iloc[max(0, eval_point-50):eval_point+1]
            sweep_idx_in_hist = len(hist_df) - 1 - detection_delay
            sweep_idx_in_hist = max(0, min(sweep_idx_in_hist, len(hist_df) - 1))
            candles_since = detection_delay

            # Compute ETF flow features during training (from df only, no API)
            # Use larger window (200+ candles) for reliable EMA/OBV/MFI/CMF
            _etf_flow = None
            if market_type in ('etf', 'stock'):
                try:
                    from core.etf_flow import calculate_etf_flow_score
                    flow_df = df.iloc[max(0, eval_point-250):eval_point+1]
                    _etf_flow = calculate_etf_flow_score(flow_df, include_institutional=False)
                except Exception:
                    pass

            features = extract_quality_features(
                hist_df, level_price, level_type, level_strength,
                trade_dir, whale_pct, whale_delta,
                sweep_idx=sweep_idx_in_hist,
                whale_delta_4h=whale_delta_4h,
                whale_delta_24h=whale_delta_24h,
                whale_delta_7d=whale_delta_7d,
                whale_acceleration=whale_acceleration,
                whale_early_signal=whale_early_signal,
                is_fresh_accumulation=is_fresh,
                oi_change_24h=oi_change_24h,
                price_change_24h=price_change_24h,
                retail_pct=retail_pct,
                funding_rate=funding_rate,
                sweep_candle_idx=sweep_idx_in_hist,
                candles_since_sweep=candles_since,
                is_sweep_of_low=is_low,
                market_type=market_type,
                etf_flow_data=_etf_flow
            )

            if features is None:
                return

            # FORCE flow override here (bypass extract_quality_features caching issue)
            if market_type in ('etf', 'stock') and _etf_flow:
                flow = _etf_flow
                features['whale_pct'] = flow.get('flow_score', 0)
                features['whale_delta_4h'] = flow.get('mfi_value', 50)
                features['whale_delta_24h'] = flow.get('cmf_value', 0)
                features['whale_delta_7d'] = flow.get('price_extension_ema200', 0)
                features['retail_pct'] = flow.get('price_extension_ema50', 0)
                features['funding_rate'] = flow.get('institutional_score', 0)
                features['oi_change_24h'] = flow.get('options_sentiment_score', 0)
                phase = flow.get('flow_phase', 'NEUTRAL')
                features['rule_whale_bullish'] = 1 if phase == 'ACCUMULATING' else 0
                features['rule_whale_bearish'] = 1 if phase == 'DISTRIBUTING' else 0
                features['rule_whale_accumulating'] = 1 if flow.get('in_accumulation_zone', False) else 0
                features['rule_whale_distributing'] = 1 if flow.get('in_distribution_zone', False) else 0
                flow_score = flow.get('flow_score', 0)
                features['rule_whale_aligned'] = 1 if (
                    (trade_dir == 'LONG' and flow_score > 20) or
                    (trade_dir == 'SHORT' and flow_score < -20)
                ) else 0
                features['rule_whale_trap'] = 1 if phase == 'EXTENDED' else 0
                features['rule_whale_strong'] = 1 if abs(flow_score) > 60 else 0
                features['whale_acceleration_ratio'] = flow.get('obv_trend', 0)

            sample = {
                'symbol': symbol,
                'sample_idx': i,
                'direction': trade_dir,
                'sweep_direction': sweep_direction,
                'level_type': level_type,
                'level_strength': level_strength,
                'is_reversal': 0,  # Continuation sample
                **features,
                **outcome
            }
            samples.append(sample)

        def generate_reversal_sample(level_idx, level_price, level_type, level_strength, sweep_direction, is_low):
            """Generate REVERSAL sample: trade SAME as sweep direction (bounce/rejection)."""
            if sweep_direction == 'LONG':
                distance_atr = (current_price - level_price) / atr
            else:
                distance_atr = (level_price - current_price) / atr

            if distance_atr > 3:
                return

            # Reversal: trade SAME as sweep direction (price reverses after sweep)
            # Sweep of LOW (sweep_direction='LONG') → Trade LONG (bounce up)
            # Sweep of HIGH (sweep_direction='SHORT') → Trade SHORT (drop down)
            trade_dir = sweep_direction  # SAME direction = reversal

            # Get whale data
            whale_pct, whale_delta, whale_delta_4h, whale_delta_24h, whale_delta_7d, whale_acceleration, whale_early_signal, is_fresh, oi_change_24h, retail_pct, funding_rate = get_whale_at_time(i, trade_dir)

            # Calculate price change 24h
            price_change_24h = 0
            if i >= 6:
                price_24h_ago = df['close'].iloc[i - 6]
                if price_24h_ago > 0:
                    price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100

            outcome = label_trade_outcome(df, i, level_price, sweep_direction, atr, trade_direction=trade_dir, forward_candles=_fwd)
            if outcome is None:
                return

            sweep_offset = outcome.get('sweep_candle_offset', 1)
            sweep_candle_abs_idx = i + sweep_offset

            detection_delay = random.randint(1, 10)
            eval_point = min(sweep_candle_abs_idx + detection_delay, len(df) - 1)

            hist_df = df.iloc[max(0, eval_point-50):eval_point+1]
            sweep_idx_in_hist = len(hist_df) - 1 - detection_delay
            sweep_idx_in_hist = max(0, min(sweep_idx_in_hist, len(hist_df) - 1))
            candles_since = detection_delay

            # Compute ETF flow features during training (from df only, no API)
            _etf_flow = None
            if market_type in ('etf', 'stock'):
                try:
                    from core.etf_flow import calculate_etf_flow_score
                    flow_df = df.iloc[max(0, eval_point-250):eval_point+1]
                    _etf_flow = calculate_etf_flow_score(flow_df, include_institutional=False)
                except Exception:
                    pass

            features = extract_quality_features(
                hist_df, level_price, level_type, level_strength,
                trade_dir, whale_pct, whale_delta,
                sweep_idx=sweep_idx_in_hist,
                whale_delta_4h=whale_delta_4h,
                whale_delta_24h=whale_delta_24h,
                whale_delta_7d=whale_delta_7d,
                whale_acceleration=whale_acceleration,
                whale_early_signal=whale_early_signal,
                is_fresh_accumulation=is_fresh,
                oi_change_24h=oi_change_24h,
                price_change_24h=price_change_24h,
                retail_pct=retail_pct,
                funding_rate=funding_rate,
                sweep_candle_idx=sweep_idx_in_hist,
                candles_since_sweep=candles_since,
                is_sweep_of_low=is_low,
                market_type=market_type,
                etf_flow_data=_etf_flow
            )

            if features is None:
                return

            # FORCE flow override here (bypass extract_quality_features caching issue)
            if market_type in ('etf', 'stock') and _etf_flow:
                flow = _etf_flow
                features['whale_pct'] = flow.get('flow_score', 0)
                features['whale_delta_4h'] = flow.get('mfi_value', 50)
                features['whale_delta_24h'] = flow.get('cmf_value', 0)
                features['whale_delta_7d'] = flow.get('price_extension_ema200', 0)
                features['retail_pct'] = flow.get('price_extension_ema50', 0)
                features['funding_rate'] = flow.get('institutional_score', 0)
                features['oi_change_24h'] = flow.get('options_sentiment_score', 0)
                phase = flow.get('flow_phase', 'NEUTRAL')
                features['rule_whale_bullish'] = 1 if phase == 'ACCUMULATING' else 0
                features['rule_whale_bearish'] = 1 if phase == 'DISTRIBUTING' else 0
                features['rule_whale_accumulating'] = 1 if flow.get('in_accumulation_zone', False) else 0
                features['rule_whale_distributing'] = 1 if flow.get('in_distribution_zone', False) else 0
                flow_score = flow.get('flow_score', 0)
                features['rule_whale_aligned'] = 1 if (
                    (trade_dir == 'LONG' and flow_score > 20) or
                    (trade_dir == 'SHORT' and flow_score < -20)
                ) else 0
                features['rule_whale_trap'] = 1 if phase == 'EXTENDED' else 0
                features['rule_whale_strong'] = 1 if abs(flow_score) > 60 else 0
                features['whale_acceleration_ratio'] = flow.get('obv_trend', 0)

            sample = {
                'symbol': symbol,
                'sample_idx': i,
                'direction': trade_dir,
                'sweep_direction': sweep_direction,
                'level_type': level_type,
                'level_strength': level_strength,
                'is_reversal': 1,  # Flag to identify reversal samples
                **features,
                **outcome
            }
            samples.append(sample)

        # ═══════════════════════════════════════════════════════════════════
        # BOTH CONTINUATION + REVERSAL TRAINING (88% win rate at >60% confidence)
        # ML learns to distinguish good trades from bad, filters out reversals
        # ═══════════════════════════════════════════════════════════════════

        # Process LOW sweeps
        for level_idx, level_price, level_type, level_strength in nearby_lows[-3:]:
            # Continuation: Sweep LOW → SHORT (price continues down) - 74% win rate
            generate_continuation_sample(level_idx, level_price, level_type, level_strength, 'LONG', is_low=True)
            # Reversal: Sweep LOW → LONG (price reverses up) - 6% win rate
            # ML learns to give this LOW probability, filtered out at >60% confidence
            generate_reversal_sample(level_idx, level_price, level_type, level_strength, 'LONG', is_low=True)

        # Process HIGH sweeps
        for level_idx, level_price, level_type, level_strength in nearby_highs[-3:]:
            # Continuation: Sweep HIGH → LONG (price continues up) - 70% win rate
            generate_continuation_sample(level_idx, level_price, level_type, level_strength, 'SHORT', is_low=False)
            # Reversal: Sweep HIGH → SHORT (price reverses down) - 10% win rate
            # ML learns to give this LOW probability, filtered out at >60% confidence
            generate_reversal_sample(level_idx, level_price, level_type, level_strength, 'SHORT', is_low=False)
    
    print(f"[QUALITY_ML] {symbol}: Generated {len(samples)} samples")
    return samples


def _find_liquidity_levels_simple(df: pd.DataFrame, atr: float) -> Dict:
    """
    Find liquidity levels (EQUAL, DOUBLE, SWING).
    
    Self-contained - doesn't depend on other modules.
    """
    lows = []
    highs = []
    
    # Find swing points
    swing_window = 5
    
    for i in range(swing_window, len(df) - swing_window):
        # Swing low
        is_swing_low = all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, swing_window+1))
        is_swing_low = is_swing_low and all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, swing_window+1))
        
        if is_swing_low:
            lows.append((i, df['low'].iloc[i]))
        
        # Swing high
        is_swing_high = all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, swing_window+1))
        is_swing_high = is_swing_high and all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, swing_window+1))
        
        if is_swing_high:
            highs.append((i, df['high'].iloc[i]))
    
    # Cluster nearby levels
    tolerance = atr * 0.3  # Levels within 0.3 ATR are considered "equal"
    
    def cluster_levels(levels):
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x[1])
        
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for idx, price in sorted_levels[1:]:
            # Check if this level is close to current cluster
            cluster_price = np.mean([p for _, p in current_cluster])
            
            if abs(price - cluster_price) <= tolerance:
                current_cluster.append((idx, price))
            else:
                # Finalize current cluster
                test_count = len(current_cluster)
                avg_price = np.mean([p for _, p in current_cluster])
                latest_idx = max(idx for idx, _ in current_cluster)
                
                if test_count >= 3:
                    level_type = 'EQUAL'
                    strength = 1.0
                elif test_count == 2:
                    level_type = 'DOUBLE'
                    strength = 0.85
                else:
                    level_type = 'SWING'
                    strength = 0.5
                
                clustered.append((latest_idx, avg_price, level_type, strength))
                
                # Start new cluster
                current_cluster = [(idx, price)]
        
        # Don't forget last cluster
        if current_cluster:
            test_count = len(current_cluster)
            avg_price = np.mean([p for _, p in current_cluster])
            latest_idx = max(idx for idx, _ in current_cluster)
            
            if test_count >= 3:
                level_type = 'EQUAL'
                strength = 1.0
            elif test_count == 2:
                level_type = 'DOUBLE'
                strength = 0.85
            else:
                level_type = 'SWING'
                strength = 0.5
            
            clustered.append((latest_idx, avg_price, level_type, strength))
        
        return clustered
    
    return {
        'lows': cluster_levels(lows),
        'highs': cluster_levels(highs)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_quality_model(
    symbols: List[str] = None,
    days: int = 365,
    progress_callback: callable = None
) -> Dict:
    """
    Train the quality model on historical data.
    """
    if symbols is None:
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'POLUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT'  # MATIC→POL
        ]
    
    def update_progress(text, pct):
        if progress_callback:
            progress_callback(text, pct)
        print(f"[QUALITY_ML] {text}")
    
    update_progress("⚡ Fetching price data...", 0.1)
    
    # Fetch data
    try:
        from core.data_fetcher import fetch_klines_parallel
        
        def fetch_progress(completed, total, symbol):
            pct = 0.1 + (completed / total) * 0.3
            update_progress(f"⚡ Fetching {symbol}... {completed}/{total}", pct)
        
        klines_data = fetch_klines_parallel(
            symbols=symbols,
            interval='4h',
            limit=min(days * 6, 1500),
            progress_callback=fetch_progress
        )
    except ImportError:
        return {'error': 'data_fetcher not available'}
    
    if not klines_data:
        return {'error': 'No data fetched'}
    
    update_progress("📊 Loading whale history...", 0.45)
    
    # Load whale history
    whale_history = {}
    try:
        whale_db = "data/whale_history.db"
        if os.path.exists(whale_db):
            import sqlite3
            conn = sqlite3.connect(whale_db)
            df_whale = pd.read_sql(
                "SELECT symbol, timestamp, whale_long_pct FROM whale_snapshots ORDER BY timestamp",
                conn
            )
            conn.close()
            
            for symbol in df_whale['symbol'].unique():
                sym_data = df_whale[df_whale['symbol'] == symbol]
                whale_history[symbol] = sym_data.to_dict('records')
            
            update_progress(f"📊 Loaded whale data for {len(whale_history)} symbols", 0.5)
    except Exception as e:
        update_progress(f"⚠️ Could not load whale history: {e}", 0.5)
    
    update_progress("🔄 Generating quality samples...", 0.55)
    
    # Generate samples
    all_samples = []
    total_symbols = len(klines_data)
    
    for idx, (symbol, df) in enumerate(klines_data.items()):
        pct = 0.55 + (idx / total_symbols) * 0.35
        update_progress(f"🔄 Processing {symbol}... {idx+1}/{total_symbols}", pct)
        
        if df is None or len(df) < 100:
            continue
        
        sym_whale = whale_history.get(symbol, [])
        samples = generate_quality_samples(df, symbol, sym_whale)
        
        all_samples.extend(samples)
        print(f"[QUALITY_ML] {symbol}: {len(samples)} samples, total: {len(all_samples)}")
    
    if len(all_samples) < 100:
        return {'error': f'Not enough samples: {len(all_samples)}'}
    
    update_progress(f"🧠 Training model on {len(all_samples)} samples...", 0.92)
    
    # Train - pass actual training days for accurate frequency calculation
    model = get_quality_model()
    metrics = model.train(all_samples, training_days=days, n_symbols=total_symbols)
    
    update_progress("✅ Training complete!", 1.0)
    
    return metrics


def train_quality_model_etf(
    symbols: List[str] = None,
    days: int = 500,
    progress_callback: callable = None
) -> Dict:
    """Train quality model on ETF daily data."""
    if symbols is None:
        try:
            from core.data_fetcher import get_popular_etfs
            symbols = get_popular_etfs()
        except ImportError:
            symbols = [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'ARKK',
                'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB',
                'GLD', 'SLV', 'TLT', 'HYG', 'LQD', 'VNQ', 'IEMG', 'EFA',
                'SMH', 'SOXX', 'XBI', 'IBB', 'KWEB', 'FXI'
            ]

    return _train_quality_model_market(
        symbols=symbols,
        days=days,
        progress_callback=progress_callback,
        market_label='ETF',
        model_getter=get_quality_model_etf,
        interval='1d',
        forward_candles=FORWARD_CANDLES_DAILY
    )


def train_quality_model_stock(
    symbols: List[str] = None,
    days: int = 500,
    progress_callback: callable = None
) -> Dict:
    """Train quality model on Stock daily data."""
    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'UNH', 'DIS', 'BAC',
            'ADBE', 'CRM', 'NFLX', 'CSCO', 'PEP', 'TMO', 'COST', 'AVGO',
            'AMD', 'INTC', 'QCOM', 'TXN', 'LOW', 'SBUX', 'GS', 'BLK',
            'CAT', 'DE', 'BA', 'GE', 'MMM', 'HON',
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',
            'LLY', 'ABBV', 'MRK', 'PFE', 'BMY'
        ]

    return _train_quality_model_market(
        symbols=symbols,
        days=days,
        progress_callback=progress_callback,
        market_label='Stock',
        model_getter=get_quality_model_stock,
        interval='1d',
        forward_candles=FORWARD_CANDLES_DAILY
    )


def _train_quality_model_market(
    symbols: List[str],
    days: int,
    progress_callback: callable,
    market_label: str,
    model_getter: callable,
    interval: str = '1d',
    forward_candles: int = None
) -> Dict:
    """Shared training logic for ETF/Stock quality models."""

    def update_progress(text, pct):
        if progress_callback:
            progress_callback(text, pct)
        print(f"[QUALITY_ML_{market_label.upper()}] {text}")

    update_progress(f"Fetching {market_label} price data...", 0.1)

    try:
        from core.data_fetcher import fetch_stock_data
    except ImportError:
        return {'error': 'data_fetcher not available'}

    klines_data = {}
    total = len(symbols)
    for idx_s, symbol in enumerate(symbols):
        pct = 0.1 + (idx_s / total) * 0.35
        update_progress(f"Fetching {symbol}... {idx_s+1}/{total}", pct)
        try:
            df = fetch_stock_data(symbol, interval=interval, limit=days)
            if df is not None and len(df) >= 100:
                klines_data[symbol] = df
        except Exception as e:
            print(f"[QUALITY_ML_{market_label.upper()}] Failed to fetch {symbol}: {e}")

    if not klines_data:
        return {'error': f'No {market_label} data fetched'}

    update_progress(f"Generating quality samples ({len(klines_data)} symbols)...", 0.5)

    all_samples = []
    total_symbols = len(klines_data)
    for idx_s, (symbol, df) in enumerate(klines_data.items()):
        pct = 0.55 + (idx_s / total_symbols) * 0.35
        update_progress(f"Processing {symbol}... {idx_s+1}/{total_symbols}", pct)

        if df is None or len(df) < 100:
            continue

        # No whale data for ETFs/Stocks — use flow features instead
        _mtype = market_label.lower() if market_label.lower() in ('etf', 'stock') else 'stock'
        samples = generate_quality_samples(df, symbol, whale_history=[], forward_candles=forward_candles, market_type=_mtype)
        all_samples.extend(samples)
        print(f"[QUALITY_ML_{market_label.upper()}] {symbol}: {len(samples)} samples, total: {len(all_samples)}")

    if len(all_samples) < 100:
        return {'error': f'Not enough samples: {len(all_samples)}'}

    update_progress(f"Training {market_label} model on {len(all_samples)} samples...", 0.92)

    model = model_getter()
    metrics = model.train(all_samples, training_days=days, n_symbols=total_symbols)

    update_progress(f"{market_label} training complete!", 1.0)
    return metrics


if __name__ == "__main__":
    # Test whale analysis
    print("\n=== Whale Analysis Tests ===")

    # Test 1: High and rising (bullish)
    result = analyze_whale_behavior(65, 10, 'LONG')


# ═══════════════════════════════════════════════════════════════════════════════
# EASY INTEGRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_quality_prediction(
    symbol: str,
    direction: str,
    level_type: str,
    level_price: float,
    current_price: float,
    atr: float,
    whale_pct: float = 50,
    whale_delta: float = 0,
    momentum: float = 0,
    volume_ratio: float = 1.0,
    volatility: float = None,
    # Price Action features (from sweep analysis)
    pa_candle_score: float = 0,
    pa_structure_score: float = 0,
    pa_has_order_block: int = 0,
    pa_has_fvg: int = 0,
    pa_volume_score: float = 0,
    pa_momentum_score: float = 0,
    pa_total_score: float = 0,
    # Whale acceleration features (4h, 24h, 7d)
    whale_delta_4h: float = None,    # Early signal (4 hours)
    whale_delta_24h: float = None,
    whale_delta_7d: float = None,
    whale_acceleration: str = None,
    whale_early_signal: str = None,  # EARLY_ACCUMULATION, etc.
    is_fresh_accumulation: bool = False,
    # OI features
    oi_change_24h: float = None,     # OI change % in 24h
    price_change_24h: float = None,  # Price change % in 24h
    # NEW: Additional Layer 1 raw features
    retail_pct: float = None,        # Retail long %
    funding_rate: float = None,      # Funding rate
    # Regime features
    regime_volatility: float = None,
    regime_trend_strength: float = None,
    # Sweep-specific features (core of liquidity hunting!)
    sweep_depth_atr: float = None,
    sweep_wick_ratio: float = None,
    sweep_body_ratio: float = None,
    candles_since_sweep: int = None,
    sweep_volume_ratio: float = None,
    _model_override: 'QualityModel' = None,
    # ETF/Stock flow features
    market_type: str = 'crypto',
    etf_flow_data: Dict = None,
    # NEW: Price change since sweep
    price_change_since_sweep: float = 0.0,
    rule_price_aligned: int = 0,
    # NEW: Pre-sweep structure features
    structure_lower_highs: int = 0,
    structure_higher_lows: int = 0,
    structure_failed_bos: int = 0,
    structure_trend_aligned: int = 0,
    # NEW: Impulse vs Correction
    is_impulse_move: int = 0,
    is_correction: int = 0,
    swing_expansion: float = 1.0,
    # NEW (Jan 2026): Follow-through & Distance features
    follow_through_per_candle: float = 0.0,
    distance_from_sweep_pct: float = 10.0,
    indecision_candle: int = 0,
    rule_weak_follow_through: int = 0,
    rule_close_to_sweep_level: int = 0
) -> Dict:
    """
    Easy integration function for Scanner and Single Analysis.

    Args:
        symbol: e.g., 'BTCUSDT'
        direction: 'LONG' or 'SHORT'
        level_type: 'EQUAL_LOW', 'DOUBLE_LOW', 'SWING_LOW', etc.
        level_price: Price of the liquidity level
        current_price: Current market price
        atr: Average True Range
        whale_pct: Current whale positioning (0-100)
        whale_delta: Change in whale % from previous reading
        momentum: RSI-normalized momentum (-1 to 1)
        volume_ratio: Current volume / average volume
    
    Returns:
        {
            'probability': 0.0-1.0,
            'decision': 'STRONG_YES' | 'YES' | 'MAYBE' | 'NO',
            'take_trade': True/False,
            'expected_win_rate': 0.0-1.0,
            'expected_roi': % at 1:1 R:R,
            'reasons': [...],
            'warnings': [...]
        }
    """
    model = _model_override or get_quality_model()

    if not model.is_trained:
        return {
            'probability': 0.5,
            'decision': 'UNKNOWN',
            'take_trade': False,
            'expected_win_rate': 0.5,
            'expected_roi': 0,
            'reasons': ['Model not trained'],
            'warnings': ['Train the Quality Model first!']
        }
    
    # Calculate distance in ATR
    if direction == 'LONG':
        distance_atr = (current_price - level_price) / atr if atr > 0 else 1.0
    else:
        distance_atr = (level_price - current_price) / atr if atr > 0 else 1.0
    
    # Determine level type score
    level_type_upper = level_type.upper()
    if 'EQUAL' in level_type_upper:
        level_type_score = 1.0
        level_strength = 1.0
    elif 'DOUBLE' in level_type_upper:
        level_type_score = 0.85
        level_strength = 0.85
    else:  # SWING
        level_type_score = 0.5
        level_strength = 0.5
    
    # Analyze whale behavior
    whale_analysis = analyze_whale_behavior(whale_pct, whale_delta, direction)
    
    # Analyze level quality
    level_analysis = analyze_level_quality(level_type, level_strength, distance_atr)
    
    # ═══════════════════════════════════════════════════════════════════
    # WHALE ACCELERATION FEATURES (4h, 24h, 7d) - THE KEY TO TIMING!
    # ═══════════════════════════════════════════════════════════════════
    whale_4h = whale_delta_4h if whale_delta_4h is not None else 0
    whale_24h = whale_delta_24h if whale_delta_24h is not None else whale_delta
    whale_7d = whale_delta_7d if whale_delta_7d is not None else whale_delta
    daily_avg_7d = whale_7d / 7 if whale_7d else 0

    # Acceleration ratio (24h vs 7d avg)
    if abs(daily_avg_7d) > 0.5:
        acceleration_ratio = whale_24h / daily_avg_7d if daily_avg_7d != 0 else 0
    else:
        acceleration_ratio = 0

    # NEW: Early signal ratio (4h vs 24h)
    if abs(whale_24h) > 0.5:
        early_signal_ratio = whale_4h / whale_24h if whale_24h != 0 else 0
    else:
        early_signal_ratio = 0

    # Encode acceleration status
    accel_accelerating = 1 if whale_acceleration == 'ACCELERATING' else 0
    accel_decelerating = 1 if whale_acceleration == 'DECELERATING' else 0
    accel_reversing = 1 if whale_acceleration == 'REVERSING' else 0
    accel_steady = 1 if whale_acceleration == 'STEADY' else 0

    # NEW: Encode early signal status (4h vs 24h divergence)
    early_accumulation = 1 if whale_early_signal == 'EARLY_ACCUMULATION' else 0
    early_distribution = 1 if whale_early_signal == 'EARLY_DISTRIBUTION' else 0
    fresh_accumulation = 1 if whale_early_signal == 'FRESH_ACCUMULATION' else 0
    fresh_distribution = 1 if whale_early_signal == 'FRESH_DISTRIBUTION' else 0

    # Fresh vs late entry detection
    is_fresh = 1 if is_fresh_accumulation else 0
    is_late_entry = 0
    if direction == 'LONG' and whale_7d > 5 and whale_24h < daily_avg_7d * 0.5:
        is_late_entry = 1
    elif direction == 'SHORT' and whale_7d < -5 and whale_24h > daily_avg_7d * 0.5:
        is_late_entry = 1

    # Whale-Retail divergence
    retail = retail_pct if retail_pct is not None else 50
    whale_retail_divergence = whale_pct - retail

    # Default volatility if not provided
    vol = volatility if volatility is not None else (atr / current_price if current_price > 0 else 0.02)

    # Default regime features if not provided
    reg_vol = regime_volatility if regime_volatility is not None else 1.0
    reg_trend = regime_trend_strength if regime_trend_strength is not None else 0

    # Build features for model (3-LAYER APPROACH)
    features = {
        # ═══════════════════════════════════════════════════════════════
        # LAYER 1: RAW SIGNALS (Truth)
        # ═══════════════════════════════════════════════════════════════
        'whale_pct': whale_pct,
        'retail_pct': retail,
        'whale_delta_4h': whale_4h,
        'whale_delta_24h': whale_24h,
        'whale_delta_7d': whale_7d,
        'oi_change_24h': oi_change_24h if oi_change_24h is not None else 0,
        'price_change_24h': price_change_24h if price_change_24h is not None else 0,
        'volume_ratio': volume_ratio,
        'volatility': vol,
        'funding_rate': funding_rate if funding_rate is not None else 0,
        'momentum': momentum,

        # ═══════════════════════════════════════════════════════════════
        # LAYER 2: ENGINEERED FEATURES (Context)
        # ═══════════════════════════════════════════════════════════════
        'distance_atr': distance_atr,
        'whale_retail_divergence': whale_retail_divergence,
        'whale_acceleration_ratio': acceleration_ratio,
        'whale_early_signal_ratio': early_signal_ratio,
        'whale_daily_avg_7d': daily_avg_7d,
        'level_type_score': level_type_score,
        'level_distance_score': level_analysis['distance_score'],
        'level_quality': level_analysis['quality'],
        'pa_candle_score': pa_candle_score,
        'pa_structure_score': pa_structure_score,
        'pa_volume_score': pa_volume_score,
        'pa_momentum_score': pa_momentum_score,
        'pa_total_score': pa_total_score,

        # ═══════════════════════════════════════════════════════════════
        # LAYER 3: RULE FLAGS AS FEATURES (ML learns when rules work/fail)
        # ═══════════════════════════════════════════════════════════════
        'rule_whale_bullish': 1 if whale_pct > 60 else 0,
        'rule_whale_bearish': 1 if whale_pct < 40 else 0,
        'rule_whale_accumulating': 1 if whale_24h > 3 else 0,
        'rule_whale_distributing': 1 if whale_24h < -3 else 0,
        'rule_whale_aligned': 1 if whale_analysis['aligned'] else 0,
        'rule_whale_trap': 1 if whale_analysis['trap'] else 0,
        'rule_whale_strong': 1 if whale_analysis['strong'] else 0,
        'rule_oi_increasing': 1 if (oi_change_24h or 0) > 2 else 0,
        'rule_oi_decreasing': 1 if (oi_change_24h or 0) < -2 else 0,
        'rule_oi_price_aligned': 1 if ((oi_change_24h or 0) > 0 and (price_change_24h or 0) > 0) or
                                      ((oi_change_24h or 0) < 0 and (price_change_24h or 0) < 0) else 0,
        'rule_is_fresh_entry': is_fresh,
        'rule_is_late_entry': is_late_entry,
        'rule_early_accumulation': early_accumulation,
        'rule_early_distribution': early_distribution,
        'whale_accel_accelerating': accel_accelerating,
        'whale_accel_decelerating': accel_decelerating,
        'whale_accel_reversing': accel_reversing,
        'whale_accel_steady': accel_steady,
        'rule_level_tradeable': 1 if level_analysis['tradeable'] else 0,
        'pa_has_order_block': pa_has_order_block,
        'pa_has_fvg': pa_has_fvg,

        # ═══════════════════════════════════════════════════════════════
        # REGIME FEATURES (Critical!)
        # ═══════════════════════════════════════════════════════════════
        'regime_volatility': reg_vol,
        'regime_trend_strength': reg_trend,

        # ═══════════════════════════════════════════════════════════════
        # SWEEP-SPECIFIC FEATURES (Core of liquidity hunting!)
        # ═══════════════════════════════════════════════════════════════
        # Layer 1 - Raw sweep data
        'sweep_depth_atr': sweep_depth_atr if sweep_depth_atr is not None else 0,
        'sweep_wick_ratio': sweep_wick_ratio if sweep_wick_ratio is not None else 0,
        'sweep_body_ratio': sweep_body_ratio if sweep_body_ratio is not None else 0.5,
        'candles_since_sweep': candles_since_sweep if candles_since_sweep is not None else 100,
        'sweep_volume_ratio': sweep_volume_ratio if sweep_volume_ratio is not None else 1.0,
        # Layer 3 - Sweep quality rules
        'rule_deep_sweep': 1 if (sweep_depth_atr or 0) > 0.5 else 0,
        'rule_strong_rejection': 1 if (sweep_wick_ratio or 0) > 0.6 else 0,
        'rule_fresh_sweep': 1 if (candles_since_sweep or 100) <= 5 else 0,
        'rule_volume_confirmed': 1 if (sweep_volume_ratio or 1.0) > 1.5 else 0,

        # Direction encoding (ML decides best direction!)
        # is_trade_long will be set by predict_best_direction() for both directions
        'is_trade_long': 1 if direction == 'LONG' else 0,
        'is_sweep_of_low': 1 if ('LOW' in level_type_upper or direction == 'LONG') else 0,

        # ═══════════════════════════════════════════════════════════════════
        # NEW FEATURES (Added Jan 2026) - Required for trained model
        # ═══════════════════════════════════════════════════════════════════
        # Price change since sweep
        'price_change_since_sweep': price_change_since_sweep,
        'rule_price_aligned': rule_price_aligned,

        # Pre-sweep structure features
        'structure_lower_highs': structure_lower_highs,
        'structure_higher_lows': structure_higher_lows,
        'structure_failed_bos': structure_failed_bos,
        'structure_trend_aligned': structure_trend_aligned,

        # Impulse vs Correction features
        'is_impulse_move': is_impulse_move,
        'is_correction': is_correction,
        'swing_expansion': swing_expansion,

        # NEW (Jan 2026): Follow-through & Distance features
        'follow_through_per_candle': follow_through_per_candle,
        'distance_from_sweep_pct': distance_from_sweep_pct,
        'indecision_candle': indecision_candle,
        'rule_weak_follow_through': rule_weak_follow_through,
        'rule_close_to_sweep_level': rule_close_to_sweep_level,
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # ETF/STOCK: Override whale features with flow features
    # ═══════════════════════════════════════════════════════════════════
    if market_type in ('etf', 'stock') and etf_flow_data:
        flow = etf_flow_data
        features['whale_pct'] = flow.get('flow_score', 0)
        features['whale_delta_4h'] = flow.get('mfi_value', 50)
        features['whale_delta_24h'] = flow.get('cmf_value', 0)
        features['whale_delta_7d'] = flow.get('price_extension_ema200', 0)
        features['retail_pct'] = flow.get('price_extension_ema50', 0)
        features['funding_rate'] = flow.get('institutional_score', 0)
        features['oi_change_24h'] = flow.get('options_sentiment_score', 0)

        phase = flow.get('flow_phase', 'NEUTRAL')
        flow_score = flow.get('flow_score', 0)
        features['rule_whale_bullish'] = 1 if phase == 'ACCUMULATING' else 0
        features['rule_whale_bearish'] = 1 if phase == 'DISTRIBUTING' else 0
        features['rule_whale_accumulating'] = 1 if flow.get('in_accumulation_zone', False) else 0
        features['rule_whale_distributing'] = 1 if flow.get('in_distribution_zone', False) else 0
        features['rule_whale_aligned'] = 1 if (
            (direction == 'LONG' and flow_score > 20) or
            (direction == 'SHORT' and flow_score < -20)
        ) else 0
        features['rule_whale_trap'] = 1 if phase == 'EXTENDED' else 0
        features['rule_whale_strong'] = 1 if abs(flow_score) > 60 else 0
        features['whale_acceleration_ratio'] = flow.get('obv_trend', 0)
        features['whale_early_signal_ratio'] = 0
        features['whale_daily_avg_7d'] = 0
        features['whale_retail_divergence'] = 0
        features['rule_oi_increasing'] = 0
        features['rule_oi_decreasing'] = 0
        features['rule_oi_price_aligned'] = 0
        features['rule_early_accumulation'] = 0
        features['rule_early_distribution'] = 0
        features['whale_accel_accelerating'] = 0
        features['whale_accel_decelerating'] = 0
        features['whale_accel_reversing'] = 0
        features['whale_accel_steady'] = 0

    # ═══════════════════════════════════════════════════════════════════
    # DEBUG: Log all features being passed to ML (3-Layer Approach)
    # ═══════════════════════════════════════════════════════════════════
    debug_symbol = symbol if symbol else "UNKNOWN"
    print(f"\n{'='*60}")
    print(f"[ML_DEBUG] {debug_symbol} - Quality Prediction (3-Layer)")
    print(f"{'='*60}")
    print(f"  Direction: {direction}, Level: {level_type}")
    print(f"  --- LAYER 1: RAW SIGNALS ---")
    print(f"  Whale %: {whale_pct:.1f}%, Retail %: {retail:.1f}%")
    print(f"  Whale 4h/24h/7d: {whale_4h:+.1f}% / {whale_24h:+.1f}% / {whale_7d:+.1f}%")
    print(f"  OI Change: {(oi_change_24h or 0):+.1f}%, Price Change: {(price_change_24h or 0):+.1f}%")
    print(f"  Funding: {(funding_rate or 0):.4f}, Vol Ratio: {volume_ratio:.2f}")
    print(f"  --- LAYER 2: ENGINEERED ---")
    print(f"  Whale-Retail Divergence: {whale_retail_divergence:+.1f}%")
    print(f"  Acceleration Ratio: {acceleration_ratio:.2f}")
    print(f"  Distance ATR: {distance_atr:.2f}")
    print(f"  --- LAYER 3: RULE FLAGS ---")
    print(f"  rule_whale_bullish={features['rule_whale_bullish']}, rule_whale_trap={features['rule_whale_trap']}")
    print(f"  rule_oi_increasing={features['rule_oi_increasing']}, rule_oi_price_aligned={features['rule_oi_price_aligned']}")
    print(f"  --- REGIME ---")
    print(f"  Volatility: {reg_vol:.2f}, Trend: {reg_trend:.2f}")
    print(f"  OI Change 24h: {oi_change_24h:+.2f}%" if oi_change_24h else "  OI Change 24h: N/A")
    print(f"  Price Change 24h: {price_change_24h:+.2f}%" if price_change_24h else "  Price Change 24h: N/A")
    print(f"  --- SWEEP FEATURES ---")
    print(f"  Depth: {(sweep_depth_atr or 0):.2f} ATR, Wick: {(sweep_wick_ratio or 0):.1%}, Body: {(sweep_body_ratio or 0.5):.1%}")
    print(f"  Candles Ago: {candles_since_sweep or 'N/A'}, Vol Ratio: {(sweep_volume_ratio or 1.0):.2f}x")
    print(f"  rule_deep_sweep={features['rule_deep_sweep']}, rule_strong_rejection={features['rule_strong_rejection']}")
    print(f"  rule_fresh_sweep={features['rule_fresh_sweep']}, rule_volume_confirmed={features['rule_volume_confirmed']}")
    print(f"  Price Change Since Sweep: {features.get('price_change_since_sweep', 0):+.2f}%")
    print(f"  rule_price_aligned={features.get('rule_price_aligned', 0)}")
    print(f"  --- PRE-SWEEP STRUCTURE ---")
    print(f"  lower_highs={features.get('structure_lower_highs', 0)}, higher_lows={features.get('structure_higher_lows', 0)}")
    print(f"  failed_bos={features.get('structure_failed_bos', 0)}, trend_aligned={features.get('structure_trend_aligned', 0)}")
    print(f"  --- IMPULSE vs CORRECTION ---")
    print(f"  is_impulse={features.get('is_impulse_move', 0)}, is_correction={features.get('is_correction', 0)}")
    print(f"  swing_expansion={features.get('swing_expansion', 1.0):.2f}x ATR")
    print(f"{'='*60}")

    # Get prediction - ML evaluates BOTH directions and picks best
    prediction = model.predict_best_direction(features)

    print(f"[ML_DEBUG] {debug_symbol} - BEST DIRECTION: {prediction['best_direction']}")
    print(f"[ML_DEBUG] {debug_symbol} - LONG: {prediction['long_probability']:.1%} | SHORT: {prediction['short_probability']:.1%}")
    print(f"[ML_DEBUG] {debug_symbol} - PREDICTION: {prediction['probability']:.1%} ({prediction['decision']})")
    print(f"{'='*60}\n")

    # Add reasons and warnings
    reasons = []
    warnings = []
    
    # Whale reasons
    if whale_analysis['strong']:
        reasons.append(f"🐋 Strong whale alignment ({whale_pct:.0f}%)")
    if whale_analysis['behavior'] == 'ACCUMULATING' and direction == 'LONG':
        reasons.append(f"📈 Whales accumulating (+{whale_delta:.1f}%)")
    if whale_analysis['behavior'] == 'DISTRIBUTING' and direction == 'SHORT':
        reasons.append(f"📉 Whales distributing ({whale_delta:.1f}%)")
    
    # Level reasons
    if 'EQUAL' in level_type_upper:
        reasons.append(f"⭐ Strong EQUAL level (3+ tests)")
    elif 'DOUBLE' in level_type_upper:
        reasons.append(f"✓ DOUBLE level (2 tests)")
    
    if distance_atr <= 1.0:
        reasons.append(f"🎯 Close to level ({distance_atr:.1f} ATR)")
    
    # Warnings
    if whale_analysis['trap']:
        warnings.append(f"⚠️ WHALE TRAP: {whale_pct:.0f}% but dropping {whale_delta:.1f}%!")
    if whale_analysis['behavior'] == 'ACCUMULATING' and direction == 'SHORT':
        warnings.append(f"⚠️ Shorting while whales accumulate!")
    if whale_analysis['behavior'] == 'DISTRIBUTING' and direction == 'LONG':
        warnings.append(f"⚠️ Longing while whales distribute!")
    if 'SWING' in level_type_upper:
        warnings.append(f"⚠️ Weak SWING level (single test)")
    if distance_atr > 2.0:
        warnings.append(f"⚠️ Far from level ({distance_atr:.1f} ATR)")

    # NEW: Whale acceleration warnings/reasons
    if is_late_entry:
        warnings.append(f"⏰ LATE ENTRY: 7d move {whale_7d:+.1f}% but 24h slowing ({whale_24h:+.1f}%)")
    if whale_acceleration == 'REVERSING':
        warnings.append(f"🚨 WHALE REVERSAL: 7d={whale_7d:+.1f}% but 24h={whale_24h:+.1f}%!")
    if whale_acceleration == 'DECELERATING' and abs(whale_7d) > 5:
        warnings.append(f"🐢 Whale momentum slowing: 7d={whale_7d:+.1f}% but 24h={whale_24h:+.1f}%")

    # Fresh accumulation is a positive reason
    if is_fresh_accumulation:
        if direction == 'LONG' and whale_24h > 2:
            reasons.append(f"🚀 FRESH accumulation! 24h: {whale_24h:+.1f}%")
        elif direction == 'SHORT' and whale_24h < -2:
            reasons.append(f"🚀 FRESH distribution! 24h: {whale_24h:+.1f}%")
    if whale_acceleration == 'ACCELERATING':
        reasons.append(f"📈 Whale momentum accelerating")

    # ETF/Stock flow-specific reasons/warnings (replace whale messages)
    if market_type in ('etf', 'stock') and etf_flow_data:
        flow = etf_flow_data
        phase = flow.get('flow_phase', 'NEUTRAL')
        flow_score = flow.get('flow_score', 0)

        # Clear whale-specific reasons/warnings (not relevant for ETF)
        reasons = [r for r in reasons if 'whale' not in r.lower() and 'Whale' not in r]
        warnings = [w for w in warnings if 'whale' not in w.lower() and 'WHALE' not in w]

        if phase == 'ACCUMULATING':
            reasons.append(f"💰 Money flowing IN (flow score: {flow_score:+.0f})")
            if direction == 'LONG':
                reasons.append(f"📈 Flow aligned — ACCUMULATE on dip")
        elif phase == 'DISTRIBUTING':
            warnings.append(f"💸 Money flowing OUT (flow score: {flow_score:+.0f})")
            if direction == 'LONG':
                warnings.append(f"⚠️ Buying while institutions distribute")
        elif phase == 'EXTENDED':
            ext = flow.get('price_extension_ema200', 0)
            warnings.append(f"🔴 Price EXTENDED +{ext:.1f}% above EMA200 — TRIM territory")
            warnings.append(f"Action: {flow.get('action', 'HOLD')}")

        if flow.get('in_accumulation_zone'):
            reasons.append(f"📊 In accumulation zone (OBV rising + higher lows)")
        if flow.get('in_distribution_zone'):
            warnings.append(f"📊 In distribution zone (OBV falling + lower highs)")

    prediction['reasons'] = reasons
    prediction['warnings'] = warnings
    prediction['whale_analysis'] = whale_analysis
    prediction['level_analysis'] = level_analysis
    if market_type in ('etf', 'stock') and etf_flow_data:
        prediction['etf_flow'] = etf_flow_data

    # Include ALL features used by ML for UI debug display
    prediction['ml_features'] = {
        'direction': direction,
        'level_type': level_type,
        'level_price': level_price,
        'current_price': current_price,
        'atr': atr,
        **features  # All 32 features
    }

    return prediction


def format_quality_badge(prediction: Dict) -> str:
    """
    Format quality prediction as a colored badge for UI.
    
    Returns HTML string for st.markdown.
    """
    decision = prediction.get('decision', 'UNKNOWN')
    prob = prediction.get('probability', 0.5)
    
    if decision == 'STRONG_YES':
        color = '#00ff88'  # Bright green
        icon = '🔥'
    elif decision == 'YES':
        color = '#88ff00'  # Yellow-green
        icon = '✅'
    elif decision == 'MAYBE':
        color = '#ffaa00'  # Orange
        icon = '⚠️'
    else:  # NO
        color = '#ff4444'  # Red
        icon = '❌'
    
    return f"{icon} **{decision}** ({prob:.0%})"


def get_quality_prediction_etf(symbol: str, df: pd.DataFrame = None, **kwargs) -> Dict:
    """Get quality prediction using the ETF model with money flow features."""
    kwargs.setdefault('whale_pct', 50)
    kwargs.setdefault('whale_delta', 0)

    # Compute live flow data (with institutional API if df available)
    etf_flow_data = kwargs.pop('etf_flow_data', None)
    if etf_flow_data is None and df is not None:
        try:
            from core.etf_flow import calculate_etf_flow_score
            etf_flow_data = calculate_etf_flow_score(df, ticker=symbol, include_institutional=True)
        except Exception:
            pass

    return get_quality_prediction(
        symbol=symbol,
        _model_override=get_quality_model_etf(),
        market_type='etf',
        etf_flow_data=etf_flow_data,
        **kwargs
    )


def get_quality_prediction_stock(symbol: str, df: pd.DataFrame = None, **kwargs) -> Dict:
    """Get quality prediction using the Stock model with money flow features."""
    kwargs.setdefault('whale_pct', 50)
    kwargs.setdefault('whale_delta', 0)

    etf_flow_data = kwargs.pop('etf_flow_data', None)
    if etf_flow_data is None and df is not None:
        try:
            from core.etf_flow import calculate_etf_flow_score
            etf_flow_data = calculate_etf_flow_score(df, ticker=symbol, include_institutional=True)
        except Exception:
            pass

    return get_quality_prediction(
        symbol=symbol,
        _model_override=get_quality_model_stock(),
        market_type='stock',
        etf_flow_data=etf_flow_data,
        **kwargs
    )


# Legacy test code
def _run_tests():
    print(f"65% whale, +10% delta, LONG: {result['behavior']}, quality={result['quality']:.1f}")
    
    # Test 2: High but dropping (TRAP!)
    result = analyze_whale_behavior(65, -10, 'LONG')
    print(f"65% whale, -10% delta, LONG: {result['behavior']}, quality={result['quality']:.1f}, TRAP={result['trap']}")
    
    # Test 3: Low but rising (bad for short)
    result = analyze_whale_behavior(40, 8, 'SHORT')
    print(f"40% whale, +8% delta, SHORT: {result['behavior']}, quality={result['quality']:.1f}")
    
    print("\n=== Should Take Trade Tests ===")
    
    # Good LONG setup
    result = should_take_trade('EQUAL_LOW', 1.0, 1.2, 68, 5, 'LONG')
    print(f"EQUAL_LOW, whale 68% +5%: {result['decision']} (score={result['score']})")
    
    # Bad LONG setup (whale trap)
    result = should_take_trade('EQUAL_LOW', 1.0, 1.2, 65, -8, 'LONG')
    print(f"EQUAL_LOW, whale 65% -8%: {result['decision']} (score={result['score']})")
    print(f"  Red flags: {result['red_flags']}")
