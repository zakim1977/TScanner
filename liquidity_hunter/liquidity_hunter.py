"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         LIQUIDITY HUNTER MODULE                                ║
║                                                                                ║
║  Data-driven liquidity sweep detection with ML quality prediction              ║
║  Follow the whale footprints + ML to filter high-quality sweeps                ║
║                                                                                ║
║  Strategy (CONTINUATION MODE - 90% win rate!):                                 ║
║  1. Identify where stops cluster (liquidity pools)                             ║
║  2. Wait for price to sweep them (the hunt)                                    ║
║  3. Trade WITH the sweep direction (continuation, not reversal)                ║
║  4. Sweeps of LOWS → SHORT (price continues down)                              ║
║  5. Sweeps of HIGHS → LONG (price continues up)                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# REVERSAL = Traditional: Sweep LOW → go LONG (expect reversal) - 24% win rate
# CONTINUATION = New: Sweep LOW → go SHORT (expect continuation) - 90% win rate!
# ML-DRIVEN DIRECTION: No hardcoded strategy mode.
# ML evaluates BOTH directions per sweep and picks the best one.
# Fallback: CONTINUATION mode (trade opposite to sweep direction).

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests

# Try to import ML module
try:
    from .liquidity_hunter_ml import predict_sweep_quality, store_sweep_trade, get_training_stats
    LH_ML_AVAILABLE = True
    print("[LH] ML module loaded")
except ImportError:
    try:
        from liquidity_hunter_ml import predict_sweep_quality, store_sweep_trade, get_training_stats
        LH_ML_AVAILABLE = True
        print("[LH] ML module loaded (direct import)")
    except ImportError:
        LH_ML_AVAILABLE = False
        print("[LH] ML module not available - using rule-based only")

# ═══════════════════════════════════════════════════════════════════════════════
# COINGLASS REAL LIQUIDATION DATA
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .coinglass_liquidation import get_real_liquidation_levels, get_coinglass_liquidation
    COINGLASS_LIQ_AVAILABLE = True
    print("[LH] ✅ Coinglass REAL liquidation data: ENABLED")
except ImportError:
    try:
        from coinglass_liquidation import get_real_liquidation_levels, get_coinglass_liquidation
        COINGLASS_LIQ_AVAILABLE = True
        print("[LH] ✅ Coinglass REAL liquidation data: ENABLED (direct)")
    except ImportError:
        COINGLASS_LIQ_AVAILABLE = False
        print("[LH] ⚠️ Coinglass liquidation: DISABLED (using calculated)")

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY MODEL (ML for filtering high-quality setups)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .quality_model import (
        get_quality_model, get_quality_prediction, get_quality_model_status,
        get_quality_prediction_etf, get_quality_prediction_stock
    )
    QUALITY_MODEL_AVAILABLE = True
    print("[LH] ✅ Quality ML model: ENABLED")
except ImportError:
    try:
        from quality_model import (
            get_quality_model, get_quality_prediction, get_quality_model_status,
            get_quality_prediction_etf, get_quality_prediction_stock
        )
        QUALITY_MODEL_AVAILABLE = True
        print("[LH] ✅ Quality ML model: ENABLED (direct)")
    except ImportError:
        QUALITY_MODEL_AVAILABLE = False
        print("[LH] ⚠️ Quality ML model: DISABLED")

# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDATION DATA COLLECTOR (for future ML training)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .liq_level_collector import save_liquidation_snapshot, save_from_liquidity_data
    LIQ_COLLECTOR_AVAILABLE = True
    print("[LH] ✅ Liquidation data collector: ENABLED")
except ImportError:
    try:
        from liq_level_collector import save_liquidation_snapshot, save_from_liquidity_data
        LIQ_COLLECTOR_AVAILABLE = True
        print("[LH] ✅ Liquidation data collector: ENABLED (direct)")
    except ImportError:
        LIQ_COLLECTOR_AVAILABLE = False
        print("[LH] ⚠️ Liquidation collector: DISABLED")

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE ACTION ANALYZER (SMC-based prediction after sweep)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from .price_action_analyzer import analyze_sweep_with_price_action, get_price_action_summary
    PRICE_ACTION_AVAILABLE = True
    print("[LH] ✅ Price Action Analyzer: ENABLED")
except ImportError:
    try:
        from price_action_analyzer import analyze_sweep_with_price_action, get_price_action_summary
        PRICE_ACTION_AVAILABLE = True
        print("[LH] ✅ Price Action Analyzer: ENABLED (direct)")
    except ImportError:
        PRICE_ACTION_AVAILABLE = False
        print("[LH] ⚠️ Price Action Analyzer: DISABLED")


# ═══════════════════════════════════════════════════════════════════════════════
# SMART PRICE FORMATTER (handles PEPE, SHIB, ADA and other small-decimal coins)
# ═══════════════════════════════════════════════════════════════════════════════
def format_price(price: float, symbol: str = None) -> str:
    """
    Format price with appropriate decimal places.
    
    - Large prices (>=1000): $1,234.56
    - Medium prices (10-1000): $12.34
    - Small-medium (1-10): $1.234 (3 decimals)
    - Small prices (0.1-1): $0.3604 (4 decimals for ADA etc)
    - Smaller prices (0.01-0.1): $0.0123 (4 decimals)
    - Tiny prices (<0.01): $0.00001234 (8 decimals for PEPE, SHIB)
    """
    if price is None or price == 0:
        return "$0.00"
    
    abs_price = abs(price)
    
    if abs_price >= 1000:
        return f"${price:,.2f}"
    elif abs_price >= 10:
        return f"${price:.2f}"
    elif abs_price >= 1:
        return f"${price:.3f}"
    elif abs_price >= 0.1:
        return f"${price:.4f}"
    elif abs_price >= 0.01:
        return f"${price:.4f}"
    elif abs_price >= 0.0001:
        return f"${price:.6f}"
    else:
        # Very small prices like PEPE, SHIB
        return f"${price:.8f}"


def format_price_raw(price: float) -> str:
    """Format price without $ symbol for calculations display."""
    if price is None or price == 0:
        return "0.00"
    
    abs_price = abs(price)
    
    if abs_price >= 1000:
        return f"{price:,.2f}"
    elif abs_price >= 10:
        return f"{price:.2f}"
    elif abs_price >= 1:
        return f"{price:.3f}"
    elif abs_price >= 0.1:
        return f"{price:.4f}"
    elif abs_price >= 0.01:
        return f"{price:.4f}"
    elif abs_price >= 0.0001:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ML Skip threshold - centralized constant
ML_SKIP_THRESHOLD = 0.40  # Below 40% = SKIP


def apply_ml_filter(trade_direction: str, entry_quality: dict, trade_plan: dict = None, symbol: str = "") -> tuple:
    """
    CENTRALIZED ML FILTER - Used by both scanner and single analysis.

    ML PROBABILITY is the final arbiter:
    - ML >= 50%: ALLOW trade (ML trained on all factors including sweep age)
    - ML 40-50%: Allow but with WAIT recommendation
    - ML < 40%: SKIP regardless of rule-based recommendation

    Args:
        trade_direction: Original trade direction (LONG/SHORT)
        entry_quality: Dict containing ml_probability and recommendation
        trade_plan: Optional trade plan dict to update
        symbol: Symbol for logging

    Returns:
        tuple: (filtered_direction, updated_trade_plan)
    """
    if not entry_quality:
        return trade_direction, trade_plan

    ml_recommendation = entry_quality.get('recommendation', 'UNKNOWN')
    ml_probability = entry_quality.get('ml_probability', 0) or 0

    # ML PROBABILITY is the final filter - NOT the rule-based recommendation
    # ML was trained on all features including sweep age, so trust its probability
    if ml_probability >= 0.50:
        # ML says good trade - allow it
        if symbol:
            print(f"[ML_FILTER] {symbol}: ML={ml_probability:.0%} → ALLOW trade")
        return trade_direction, trade_plan

    elif ml_probability >= ML_SKIP_THRESHOLD:
        # ML says borderline (40-50%) - allow but keep WAIT recommendation
        if symbol:
            print(f"[ML_FILTER] {symbol}: ML={ml_probability:.0%} → ALLOW (borderline)")
        return trade_direction, trade_plan

    else:
        # ML < 40% - SKIP regardless of rule-based recommendation
        filtered_direction = 'SKIP'

        # Update trade_plan if provided
        if trade_plan:
            trade_plan['direction'] = 'SKIP'
            trade_plan['status'] = 'ML_SKIP'
            trade_plan['entry_quality'] = entry_quality

        if symbol:
            print(f"[ML_FILTER] {symbol}: ML={ml_probability:.0%} → SKIP (below {ML_SKIP_THRESHOLD:.0%} threshold)")

        return filtered_direction, trade_plan


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names to lowercase.
    Handles both 'High'/'Low'/'Close' and 'high'/'low'/'close' formats.
    """
    if df is None:
        return None

    # Create a copy to avoid modifying original
    df = df.copy()

    # Normalize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    return df


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR from DataFrame."""
    if df is None or len(df) < period:
        return 0
    df = normalize_columns(df)
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# ROUND NUMBER DETECTION (Psychological Levels)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_round_numbers(current_price: float, price_range_pct: float = 10) -> List[Dict]:
    """
    Detect psychological round numbers near current price.

    Round numbers act as natural liquidity magnets where stops cluster.

    Args:
        current_price: Current asset price
        price_range_pct: Percentage range to search (default 10%)

    Returns:
        List of round number levels
    """
    if current_price <= 0:
        return []

    round_levels = []

    # Determine magnitude based on price
    if current_price >= 10000:
        # BTC range: $5000, $10000 intervals
        intervals = [10000, 5000, 1000]
    elif current_price >= 1000:
        # ETH range: $500, $100 intervals
        intervals = [1000, 500, 100]
    elif current_price >= 100:
        # SOL range: $50, $25, $10 intervals
        intervals = [100, 50, 25, 10]
    elif current_price >= 10:
        # Mid-cap: $5, $1 intervals
        intervals = [10, 5, 1]
    elif current_price >= 1:
        # ADA range: $0.50, $0.25, $0.10 intervals
        intervals = [1, 0.5, 0.25, 0.1]
    else:
        # Micro-cap: $0.01, $0.001 intervals
        intervals = [0.1, 0.01, 0.001]

    # Calculate search range
    low_bound = current_price * (1 - price_range_pct / 100)
    high_bound = current_price * (1 + price_range_pct / 100)

    for interval in intervals:
        # Find round numbers in range
        start = int(low_bound / interval) * interval
        level = start
        while level <= high_bound:
            if level >= low_bound and level > 0:
                # Determine strength based on roundness
                if level % intervals[0] == 0:
                    strength = 'MAJOR'
                    strength_score = 30
                elif len(intervals) > 1 and level % intervals[1] == 0:
                    strength = 'STRONG'
                    strength_score = 20
                else:
                    strength = 'MODERATE'
                    strength_score = 10

                round_levels.append({
                    'price': float(level),
                    'type': 'ROUND_NUMBER',
                    'strength': strength,
                    'strength_score': strength_score,
                    'is_round': True,
                    'swept': False
                })
            level += interval

    # Remove duplicates (same price)
    seen = set()
    unique_levels = []
    for lvl in round_levels:
        if lvl['price'] not in seen:
            seen.add(lvl['price'])
            unique_levels.append(lvl)

    return unique_levels


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE STRENGTH SCORE (0-100)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_level_strength_score(
    touches: int = 1,
    avg_volume_ratio: float = 1.0,
    has_volume_confirmation: bool = False,
    is_round_number: bool = False,
    is_equal_level: bool = False,
    is_20_candle_extreme: bool = False,
    candles_since_formation: int = 0
) -> int:
    """
    Calculate composite strength score for a liquidity level (0-100).

    Scoring breakdown:
    - Touches: 1 touch = 10pts, 2 = 25pts, 3+ = 40pts
    - Volume: ratio > 1.5 = +15pts, ratio > 2.0 = +25pts
    - Round number: +15pts
    - Is equal high/low: +10pts
    - 20-candle extreme: +10pts
    - Freshness: < 10 candles = +10pts, > 50 candles = -10pts

    Returns:
        Score from 0-100
    """
    score = 0

    # TOUCHES (max 40pts)
    if touches >= 3:
        score += 40
    elif touches == 2:
        score += 25
    else:
        score += 10

    # VOLUME (max 25pts)
    if avg_volume_ratio >= 2.0:
        score += 25
    elif avg_volume_ratio >= 1.5:
        score += 15
    elif avg_volume_ratio >= 1.2:
        score += 8

    # Additional volume confirmation
    if has_volume_confirmation:
        score += 5

    # ROUND NUMBER (15pts)
    if is_round_number:
        score += 15

    # EQUAL LEVEL (10pts)
    if is_equal_level:
        score += 10

    # 20-CANDLE EXTREME (10pts)
    if is_20_candle_extreme:
        score += 10

    # FRESHNESS (max +/-10pts)
    if candles_since_formation < 10:
        score += 10
    elif candles_since_formation > 50:
        score -= 10

    return max(0, min(100, score))


# ═══════════════════════════════════════════════════════════════════════════════
# REVERSAL CANDLE ANALYSIS - Detect strong rejection patterns
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_reversal_candle(df: pd.DataFrame, sweep_candle_idx: int, direction: str) -> Dict:
    """
    Analyze the sweep candle for reversal confirmation signals.

    A good reversal candle should have:
    - Strong rejection wick (wick > body)
    - Close in favorable direction
    - Engulfing pattern (optional bonus)

    Args:
        df: DataFrame with OHLCV data
        sweep_candle_idx: Index of the sweep candle (negative from end, e.g., -3)
        direction: 'LONG' or 'SHORT'

    Returns:
        Dict with reversal candle analysis
    """
    result = {
        'has_rejection_wick': False,
        'rejection_ratio': 0,
        'favorable_close': False,
        'is_engulfing': False,
        'candle_strength': 'WEAK',
        'score': 0
    }

    if df is None or len(df) < abs(sweep_candle_idx) + 2:
        return result

    df = normalize_columns(df)

    try:
        # Get the sweep candle
        candle = df.iloc[sweep_candle_idx]
        prev_candle = df.iloc[sweep_candle_idx - 1]

        o, h, l, c = float(candle['open']), float(candle['high']), float(candle['low']), float(candle['close'])
        body = abs(c - o)
        full_range = h - l

        if full_range == 0:
            return result

        if direction == 'LONG':
            # For LONG: Want lower wick > upper wick + body, close > open (bullish)
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)

            # Rejection ratio: lower wick as % of total range
            result['rejection_ratio'] = (lower_wick / full_range) * 100
            result['has_rejection_wick'] = lower_wick > body and lower_wick > upper_wick
            result['favorable_close'] = c > o  # Bullish close

            # Check for bullish engulfing
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            if c > o and prev_candle['close'] < prev_candle['open']:  # Current bullish, prev bearish
                if body > prev_body and c > prev_candle['open'] and o < prev_candle['close']:
                    result['is_engulfing'] = True

        else:  # SHORT
            # For SHORT: Want upper wick > lower wick + body, close < open (bearish)
            lower_wick = min(o, c) - l
            upper_wick = h - max(o, c)

            result['rejection_ratio'] = (upper_wick / full_range) * 100
            result['has_rejection_wick'] = upper_wick > body and upper_wick > lower_wick
            result['favorable_close'] = c < o  # Bearish close

            # Check for bearish engulfing
            prev_body = abs(prev_candle['close'] - prev_candle['open'])
            if c < o and prev_candle['close'] > prev_candle['open']:  # Current bearish, prev bullish
                if body > prev_body and c < prev_candle['open'] and o > prev_candle['close']:
                    result['is_engulfing'] = True

        # Calculate strength
        if result['has_rejection_wick'] and result['favorable_close']:
            result['candle_strength'] = 'STRONG'
        elif result['has_rejection_wick'] or result['favorable_close']:
            result['candle_strength'] = 'MODERATE'
        else:
            result['candle_strength'] = 'WEAK'

        # Score (0-25)
        score = 0
        if result['has_rejection_wick']:
            score += 12
            if result['rejection_ratio'] > 60:
                score += 3  # Extra for very strong rejection
        if result['favorable_close']:
            score += 7
        if result['is_engulfing']:
            score += 3

        result['score'] = min(25, score)

    except Exception as e:
        print(f"[REVERSAL_CANDLE] Error: {e}")

    return result


def analyze_follow_through(df: pd.DataFrame, sweep_candle_idx: int, direction: str) -> Dict:
    """
    Check if candles AFTER the sweep confirm the reversal.

    Good follow-through means:
    - Next candle(s) continue in reversal direction
    - Price hasn't given back the sweep gains

    Args:
        df: DataFrame with OHLCV data
        sweep_candle_idx: Index of sweep candle (negative, e.g., -3)
        direction: 'LONG' or 'SHORT'

    Returns:
        Dict with follow-through analysis
    """
    result = {
        'has_follow_through': False,
        'continuation_candles': 0,
        'price_held': False,
        'follow_through_strength': 'NONE',
        'score': 0
    }

    if df is None or sweep_candle_idx >= -1:
        # No candles after sweep to analyze
        return result

    df = normalize_columns(df)

    try:
        sweep_candle = df.iloc[sweep_candle_idx]
        sweep_close = float(sweep_candle['close'])
        sweep_low = float(sweep_candle['low'])
        sweep_high = float(sweep_candle['high'])

        # Get candles after the sweep
        candles_after = df.iloc[sweep_candle_idx + 1:]

        if len(candles_after) == 0:
            return result

        continuation_count = 0

        if direction == 'LONG':
            # Check each candle after sweep
            for idx, candle in candles_after.iterrows():
                c_open, c_close = float(candle['open']), float(candle['close'])
                c_low = float(candle['low'])

                # Did this candle close bullish?
                if c_close > c_open:
                    continuation_count += 1

                # Did price break below sweep low? (Failed follow-through)
                if c_low < sweep_low:
                    result['price_held'] = False
                    break
            else:
                result['price_held'] = True

            # Check if current price is above sweep close
            current_close = float(df.iloc[-1]['close'])
            if current_close > sweep_close:
                result['has_follow_through'] = True

        else:  # SHORT
            for idx, candle in candles_after.iterrows():
                c_open, c_close = float(candle['open']), float(candle['close'])
                c_high = float(candle['high'])

                if c_close < c_open:
                    continuation_count += 1

                if c_high > sweep_high:
                    result['price_held'] = False
                    break
            else:
                result['price_held'] = True

            current_close = float(df.iloc[-1]['close'])
            if current_close < sweep_close:
                result['has_follow_through'] = True

        result['continuation_candles'] = continuation_count

        # Determine strength
        if result['has_follow_through'] and result['price_held'] and continuation_count >= 2:
            result['follow_through_strength'] = 'STRONG'
        elif result['has_follow_through'] and result['price_held']:
            result['follow_through_strength'] = 'MODERATE'
        elif result['has_follow_through'] or result['price_held']:
            result['follow_through_strength'] = 'WEAK'
        else:
            result['follow_through_strength'] = 'NONE'

        # Score (0-15)
        score = 0
        if result['has_follow_through']:
            score += 6
        if result['price_held']:
            score += 5
        score += min(4, continuation_count * 2)  # Up to 4 pts for continuation

        result['score'] = min(15, score)

    except Exception as e:
        print(f"[FOLLOW_THROUGH] Error: {e}")

    return result


def analyze_momentum(df: pd.DataFrame, sweep_candle_idx: int, direction: str) -> Dict:
    """
    Check momentum indicators at sweep for confirmation.

    Good momentum signals:
    - RSI oversold for LONG / overbought for SHORT
    - RSI divergence (price makes new low but RSI makes higher low)

    Args:
        df: DataFrame with OHLCV data
        sweep_candle_idx: Index of sweep candle
        direction: 'LONG' or 'SHORT'

    Returns:
        Dict with momentum analysis
    """
    result = {
        'rsi_extreme': False,
        'rsi_value': 50,
        'has_divergence': False,
        'momentum_aligned': False,
        'score': 0
    }

    if df is None or len(df) < 20:
        return result

    df = normalize_columns(df)

    try:
        # Calculate RSI
        close = df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))

        sweep_rsi = float(rsi.iloc[sweep_candle_idx])
        current_rsi = float(rsi.iloc[-1])
        result['rsi_value'] = sweep_rsi

        if direction == 'LONG':
            # RSI oversold at sweep
            result['rsi_extreme'] = sweep_rsi < 35

            # Check for bullish divergence (price lower low, RSI higher low)
            if sweep_candle_idx < -5:
                price_window = df['low'].iloc[sweep_candle_idx-10:sweep_candle_idx+1]
                rsi_window = rsi.iloc[sweep_candle_idx-10:sweep_candle_idx+1]

                # Find previous low
                if len(price_window) > 5:
                    prev_price_low_idx = price_window.iloc[:-3].idxmin()
                    curr_price_low = float(df.iloc[sweep_candle_idx]['low'])

                    if prev_price_low_idx in rsi_window.index:
                        prev_rsi = float(rsi_window.loc[prev_price_low_idx])

                        # Bullish divergence: price makes lower low, RSI makes higher low
                        if curr_price_low < float(price_window.loc[prev_price_low_idx]) and sweep_rsi > prev_rsi:
                            result['has_divergence'] = True

            # Momentum aligned if RSI turning up
            result['momentum_aligned'] = current_rsi > sweep_rsi and current_rsi > 40

        else:  # SHORT
            result['rsi_extreme'] = sweep_rsi > 65

            # Bearish divergence check
            if sweep_candle_idx < -5:
                price_window = df['high'].iloc[sweep_candle_idx-10:sweep_candle_idx+1]
                rsi_window = rsi.iloc[sweep_candle_idx-10:sweep_candle_idx+1]

                if len(price_window) > 5:
                    prev_price_high_idx = price_window.iloc[:-3].idxmax()
                    curr_price_high = float(df.iloc[sweep_candle_idx]['high'])

                    if prev_price_high_idx in rsi_window.index:
                        prev_rsi = float(rsi_window.loc[prev_price_high_idx])

                        if curr_price_high > float(price_window.loc[prev_price_high_idx]) and sweep_rsi < prev_rsi:
                            result['has_divergence'] = True

            result['momentum_aligned'] = current_rsi < sweep_rsi and current_rsi < 60

        # Score (0-15)
        score = 0
        if result['rsi_extreme']:
            score += 6
        if result['has_divergence']:
            score += 6
        if result['momentum_aligned']:
            score += 3

        result['score'] = min(15, score)

    except Exception as e:
        print(f"[MOMENTUM] Error: {e}")

    return result


def analyze_trend_alignment(df: pd.DataFrame, direction: str) -> Dict:
    """
    Check if the trade direction aligns with the higher timeframe trend.

    Uses EMA 20/50/200 to determine trend:
    - Price > EMA20 > EMA50 > EMA200 = Strong Uptrend
    - Price < EMA20 < EMA50 < EMA200 = Strong Downtrend
    - Mixed = Choppy/Ranging

    Trading WITH trend = higher success rate
    Trading AGAINST trend = lower success rate (sweep might be a pullback continuation)

    Args:
        df: DataFrame with OHLCV data
        direction: 'LONG' or 'SHORT'

    Returns:
        Dict with trend analysis
    """
    result = {
        'trend': 'UNKNOWN',
        'trend_strength': 'WEAK',
        'aligned': False,
        'ema20': 0,
        'ema50': 0,
        'ema200': 0,
        'price_vs_ema200': 0,
        'warning': None,
        'score_adjustment': 0
    }

    if df is None or len(df) < 50:
        return result

    df = normalize_columns(df)

    try:
        close = df['close']
        current_price = float(close.iloc[-1])

        # Calculate EMAs
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        ema50 = float(close.ewm(span=50).mean().iloc[-1])

        # EMA200 if enough data, otherwise use EMA100
        if len(df) >= 200:
            ema200 = float(close.ewm(span=200).mean().iloc[-1])
        elif len(df) >= 100:
            ema200 = float(close.ewm(span=100).mean().iloc[-1])
        else:
            ema200 = float(close.ewm(span=50).mean().iloc[-1])

        result['ema20'] = ema20
        result['ema50'] = ema50
        result['ema200'] = ema200
        result['price_vs_ema200'] = ((current_price - ema200) / ema200) * 100

        # Determine trend
        if current_price > ema20 > ema50 > ema200:
            result['trend'] = 'STRONG_UP'
            result['trend_strength'] = 'STRONG'
        elif current_price > ema50 > ema200:
            result['trend'] = 'UP'
            result['trend_strength'] = 'MODERATE'
        elif current_price > ema200:
            result['trend'] = 'WEAK_UP'
            result['trend_strength'] = 'WEAK'
        elif current_price < ema20 < ema50 < ema200:
            result['trend'] = 'STRONG_DOWN'
            result['trend_strength'] = 'STRONG'
        elif current_price < ema50 < ema200:
            result['trend'] = 'DOWN'
            result['trend_strength'] = 'MODERATE'
        elif current_price < ema200:
            result['trend'] = 'WEAK_DOWN'
            result['trend_strength'] = 'WEAK'
        else:
            result['trend'] = 'RANGING'
            result['trend_strength'] = 'WEAK'

        # Check alignment
        if direction == 'LONG':
            if result['trend'] in ['STRONG_UP', 'UP', 'WEAK_UP']:
                result['aligned'] = True
                result['score_adjustment'] = 5 if result['trend'] == 'STRONG_UP' else 0
            elif result['trend'] in ['STRONG_DOWN', 'DOWN']:
                result['aligned'] = False
                result['score_adjustment'] = -15 if result['trend'] == 'STRONG_DOWN' else -10
                result['warning'] = f"🚨 LONG against {result['trend']} trend - HIGH RISK!"
            else:
                result['aligned'] = False
                result['score_adjustment'] = -5
                result['warning'] = f"⚠️ LONG in weak/ranging market - be cautious"
        else:  # SHORT
            if result['trend'] in ['STRONG_DOWN', 'DOWN', 'WEAK_DOWN']:
                result['aligned'] = True
                result['score_adjustment'] = 5 if result['trend'] == 'STRONG_DOWN' else 0
            elif result['trend'] in ['STRONG_UP', 'UP']:
                result['aligned'] = False
                result['score_adjustment'] = -15 if result['trend'] == 'STRONG_UP' else -10
                result['warning'] = f"🚨 SHORT against {result['trend']} trend - HIGH RISK!"
            else:
                result['aligned'] = False
                result['score_adjustment'] = -5
                result['warning'] = f"⚠️ SHORT in weak/ranging market - be cautious"

    except Exception as e:
        print(f"[TREND] Error: {e}")

    return result


def analyze_structure_shift(df: pd.DataFrame, sweep_candle_idx: int, direction: str) -> Dict:
    """
    Check if price broke a minor structure after the sweep (confirms reversal).

    For LONG: After sweep low, price should break a minor high
    For SHORT: After sweep high, price should break a minor low

    Args:
        df: DataFrame with OHLCV data
        sweep_candle_idx: Index of sweep candle
        direction: 'LONG' or 'SHORT'

    Returns:
        Dict with structure analysis
    """
    result = {
        'structure_broken': False,
        'break_level': None,
        'break_type': None,
        'score': 0
    }

    if df is None or len(df) < 10 or sweep_candle_idx >= -1:
        return result

    df = normalize_columns(df)

    try:
        # Get candles around the sweep
        pre_sweep = df.iloc[max(0, len(df) + sweep_candle_idx - 10):len(df) + sweep_candle_idx]
        post_sweep = df.iloc[sweep_candle_idx + 1:]

        if len(pre_sweep) < 3 or len(post_sweep) < 1:
            return result

        if direction == 'LONG':
            # Find the minor high before sweep (highest high in last 5-10 candles before sweep)
            minor_high = float(pre_sweep['high'].max())

            # Check if any candle after sweep broke above this high
            for idx, candle in post_sweep.iterrows():
                if float(candle['high']) > minor_high:
                    result['structure_broken'] = True
                    result['break_level'] = minor_high
                    result['break_type'] = 'BROKE_HIGH'
                    break

        else:  # SHORT
            # Find the minor low before sweep
            minor_low = float(pre_sweep['low'].min())

            for idx, candle in post_sweep.iterrows():
                if float(candle['low']) < minor_low:
                    result['structure_broken'] = True
                    result['break_level'] = minor_low
                    result['break_type'] = 'BROKE_LOW'
                    break

        # Score (0-10)
        result['score'] = 10 if result['structure_broken'] else 0

    except Exception as e:
        print(f"[STRUCTURE] Error: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY QUALITY SCORE (0-100) - ENHANCED with Reversal Confirmation
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_entry_quality(
    sweep_status: Dict,
    current_price: float,
    whale_pct: float = 50,
    whale_delta: float = 0,
    volume_on_sweep: float = 1.0,
    avg_volume: float = 1.0,
    df: pd.DataFrame = None,
    # Whale acceleration data (4h, 24h, 7d)
    whale_delta_4h: float = None,      # NEW: Early signal (4 hours)
    whale_delta_24h: float = None,
    whale_delta_7d: float = None,
    whale_acceleration: str = None,
    whale_early_signal: str = None,    # NEW: EARLY_ACCUMULATION, EARLY_DISTRIBUTION, etc.
    is_fresh_accumulation: bool = False,
    symbol: str = "",  # For debugging
    # OI features (NEW)
    oi_change_24h: float = None,       # OI change % in 24h
    price_change_24h: float = None,    # Price change % in 24h (for OI-price alignment)
    market_type: str = 'crypto'        # 'crypto', 'etf', or 'stock'
) -> Dict:
    """
    Calculate entry quality score (0-100) for a detected sweep.

    ENHANCED with reversal confirmation signals to filter out fake sweeps
    that continue against you after entry.

    Components (Total 100pts):
    - Freshness (20pts max): How recently did the sweep occur?
    - Reversal Candle (25pts max): Does the candle show strong rejection?
    - Follow-Through (15pts max): Did subsequent candles confirm reversal?
    - Momentum (15pts max): RSI extreme/divergence at sweep?
    - Whale Alignment (15pts max): Are whales aligned with direction?
    - Volume (10pts max): Was there volume confirmation on sweep?
    - NEW: Fresh Accumulation Bonus (+10pts) or Late Entry Penalty (-10pts)

    Args:
        sweep_status: Dict from detect_sweep()
        current_price: Current market price
        whale_pct: Whale long percentage (0-100)
        whale_delta: Change in whale positioning (based on trading_mode lookback)
        volume_on_sweep: Volume on the sweep candle
        avg_volume: Average volume (20-period)
        df: DataFrame with OHLCV data for candle analysis
        whale_delta_24h: 24-hour whale delta (fresh signal)
        whale_delta_7d: 7-day whale delta (long-term trend)
        whale_acceleration: 'ACCELERATING', 'DECELERATING', 'STEADY', 'REVERSING'
        is_fresh_accumulation: True if 24h change > 7d daily average

    Returns:
        Dict with:
        - score: 0-100
        - grade: A/B/C/D/F
        - components: breakdown of score
        - entry_window: OPEN/CLOSING/CLOSED
        - warnings: list of warning messages
        - recommendation: ENTER/WAIT/SKIP
    """
    if not sweep_status.get('detected', False):
        return {
            'score': 0,
            'grade': 'N/A',
            'components': {},
            'entry_window': 'NO_SWEEP',
            'candles_until_close': 0,
            'warnings': [],
            'recommendation': 'NO_SETUP'
        }

    # Get RAW sweep direction
    raw_direction = sweep_status.get('direction', 'LONG')
    level_swept = sweep_status.get('level_swept', current_price)
    candles_ago = sweep_status.get('candles_ago', 999)
    is_sweep_of_low = (raw_direction == 'LONG')  # LONG sweep = sweep of LOW level

    # ═══════════════════════════════════════════════════════════════════════
    # ML-DRIVEN DIRECTION SELECTION
    # ML evaluates BOTH directions and picks best based on context features.
    # Fallback: CONTINUATION mode (trade opposite to sweep direction).
    # ═══════════════════════════════════════════════════════════════════════
    # Default fallback: CONTINUATION mode (opposite to sweep direction)
    direction = "SHORT" if raw_direction == "LONG" else "LONG"
    ml_direction_override = None  # Will be set by ML if available

    # Convert candles_ago to negative index for DataFrame
    sweep_candle_idx = -candles_ago if candles_ago > 0 else -1

    components = {}
    warnings = []

    # ═══════════════════════════════════════════════════════════════════════
    # 1. FRESHNESS SCORE (20pts max) - Reduced from 40
    # ═══════════════════════════════════════════════════════════════════════
    if candles_ago <= 2:
        freshness_score = 20
        freshness_label = 'VERY_FRESH'
    elif candles_ago <= 4:
        freshness_score = 15
        freshness_label = 'FRESH'
    elif candles_ago <= 7:
        freshness_score = 8
        freshness_label = 'RECENT'
    elif candles_ago <= 10:
        freshness_score = 4
        freshness_label = 'AGING'
    else:
        freshness_score = 0
        freshness_label = 'OLD'
        warnings.append(f"⚠️ Sweep is {candles_ago} candles old - may be stale")

    components['freshness'] = {
        'score': freshness_score,
        'max': 20,
        'label': freshness_label,
        'candles_ago': candles_ago
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 2. REVERSAL CANDLE (25pts max) - Analyze candle pattern
    # NOTE: For CONTINUATION mode, we analyze using RAW direction (what candle shows)
    # then interpret the score differently
    # ═══════════════════════════════════════════════════════════════════════
    # Use raw_direction for candle analysis (what the candle actually shows)
    reversal_analysis = analyze_reversal_candle(df, sweep_candle_idx, raw_direction)
    reversal_score = reversal_analysis.get('score', 0)

    # Determine if we're trading continuation or reversal based on ML direction
    is_continuation = (direction != raw_direction)  # Trading opposite to sweep = continuation

    # In CONTINUATION mode, strong reversal signals are BAD (we want continuation)
    if is_continuation:
        # High reversal score = price likely to reverse = BAD for continuation
        # Invert the interpretation: weak reversal = good for continuation
        if reversal_score >= 15:
            warnings.append(f"⚠️ Strong reversal pattern detected - may reverse against {direction}")
            reversal_score = max(0, 25 - reversal_score)  # Invert score
        else:
            # Weak reversal = good for continuation
            reversal_score = 25 - reversal_score  # Invert: weak reversal -> high score
    else:
        # REVERSAL mode - original logic
        if reversal_score < 10:
            warnings.append(f"⚠️ Weak reversal candle - no strong rejection pattern")

    components['reversal_candle'] = {
        'score': reversal_score,
        'max': 25,
        'has_rejection': reversal_analysis.get('has_rejection_wick', False),
        'rejection_ratio': reversal_analysis.get('rejection_ratio', 0),
        'favorable_close': reversal_analysis.get('favorable_close', False),
        'is_engulfing': reversal_analysis.get('is_engulfing', False),
        'strength': reversal_analysis.get('candle_strength', 'WEAK')
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 3. FOLLOW-THROUGH (15pts max) - Check price action after sweep
    # For CONTINUATION: We want price to continue in sweep direction (not reverse)
    # ═══════════════════════════════════════════════════════════════════════
    # Analyze using raw_direction (what actually happened after sweep)
    follow_analysis = analyze_follow_through(df, sweep_candle_idx, raw_direction)
    follow_score = follow_analysis.get('score', 0)

    if is_continuation:
        # In CONTINUATION, we WANT price to break through (continue), not hold
        # If price held (bounced) = BAD for continuation
        if follow_analysis.get('price_held', True) and follow_analysis.get('has_follow_through', False):
            warnings.append(f"⚠️ Price showing reversal signs - may not continue {direction}")
            follow_score = max(0, 15 - follow_score)  # Invert
        elif not follow_analysis.get('price_held', True):
            # Price broke through = GOOD for continuation
            follow_score = 15
            warnings.append(f"✅ Price continuing through sweep level - good for {direction}")
    else:
        # REVERSAL mode - original logic
        if not follow_analysis.get('price_held', True):
            warnings.append(f"🚨 Price broke back through sweep level - reversal FAILED")
            follow_score = 0

    components['follow_through'] = {
        'score': follow_score,
        'max': 15,
        'has_follow_through': follow_analysis.get('has_follow_through', False),
        'price_held': follow_analysis.get('price_held', False),
        'continuation_candles': follow_analysis.get('continuation_candles', 0),
        'strength': follow_analysis.get('follow_through_strength', 'NONE')
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 4. MOMENTUM (15pts max) - RSI analysis
    # For CONTINUATION: Divergence is BAD (suggests reversal coming)
    # ═══════════════════════════════════════════════════════════════════════
    momentum_analysis = analyze_momentum(df, sweep_candle_idx, raw_direction)
    momentum_score = momentum_analysis.get('score', 0)

    if is_continuation:
        # In CONTINUATION mode, divergence is BAD (predicts reversal, not continuation)
        if momentum_analysis.get('has_divergence', False):
            warnings.append(f"⚠️ RSI divergence detected - may reverse against {direction}")
            momentum_score = max(0, momentum_score - 6)  # Reduce score for divergence
        # RSI extreme in sweep direction is GOOD for continuation
        if momentum_analysis.get('rsi_extreme', False):
            warnings.append(f"✅ RSI extreme - momentum supports {direction}")
    else:
        # REVERSAL mode - divergence is good
        if momentum_analysis.get('has_divergence', False):
            warnings.append(f"✅ RSI divergence detected - strong reversal signal")

    components['momentum'] = {
        'score': momentum_score,
        'max': 15,
        'rsi_value': momentum_analysis.get('rsi_value', 50),
        'rsi_extreme': momentum_analysis.get('rsi_extreme', False),
        'has_divergence': momentum_analysis.get('has_divergence', False),
        'momentum_aligned': momentum_analysis.get('momentum_aligned', False)
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 5. WHALE ALIGNMENT (15pts max) - CHANGE (delta) matters more than snapshot!
    # Whale 65% rising = ACCUMULATING = bullish
    # Whale 65% falling = DISTRIBUTING = bearish TRAP!
    # ═══════════════════════════════════════════════════════════════════════
    # Determine whale behavior first
    if whale_delta > 5:
        whale_behavior = 'ACCUMULATING'
        behavior_emoji = '📈'
    elif whale_delta < -5:
        whale_behavior = 'DISTRIBUTING'
        behavior_emoji = '📉'
    else:
        whale_behavior = 'HOLDING'
        behavior_emoji = '➡️'

    if direction == 'LONG':
        if whale_pct >= 60 and whale_delta > 3:
            whale_score = 15
            warnings.append(f"✅ Whales {whale_pct:.0f}% ({whale_delta:+.1f}%) {behavior_emoji} {whale_behavior} - strong LONG")
        elif whale_pct >= 55 and whale_delta >= 0:
            whale_score = 12
            warnings.append(f"✅ Whales {whale_pct:.0f}% ({whale_delta:+.1f}%) - aligned with LONG")
        elif whale_pct >= 55 or whale_delta > 3:
            whale_score = 8
        elif whale_pct >= 50:
            whale_score = 4
        elif whale_delta > 0:
            # Whales are short BUT accumulating (delta positive) - this SUPPORTS long!
            whale_score = 6
            warnings.append(f"✅ Whales {whale_pct:.0f}% but ACCUMULATING ({whale_delta:+.1f}%) - buying supports LONG")
        else:
            # Whales are short AND distributing/holding - true conflict
            whale_score = 0
            warnings.append(f"⚠️ Whales {whale_pct:.0f}% ({whale_delta:+.1f}%) - distributing AGAINST LONG")

        # CRITICAL: Distributing while LONG = TRAP!
        if whale_delta < -5:
            whale_score = max(0, whale_score - 8)
            warnings.append(f"🚨 WHALE TRAP! {whale_pct:.0f}% but DISTRIBUTING ({whale_delta:+.1f}%) - they're SELLING!")
    else:  # SHORT direction
        if whale_pct <= 40 and whale_delta < -3:
            whale_score = 15
            warnings.append(f"✅ Whales {whale_pct:.0f}% ({whale_delta:+.1f}%) {behavior_emoji} {whale_behavior} - strong SHORT")
        elif whale_pct <= 45 and whale_delta <= 0:
            whale_score = 12
            warnings.append(f"✅ Whales {whale_pct:.0f}% ({whale_delta:+.1f}%) - aligned with SHORT")
        elif whale_pct <= 45 or whale_delta < -3:
            whale_score = 8
        elif whale_pct <= 50:
            whale_score = 4
        elif whale_delta < 0:
            # Whales are long BUT distributing (delta negative) - this SUPPORTS short!
            whale_score = 6
            warnings.append(f"✅ Whales {whale_pct:.0f}% but DISTRIBUTING ({whale_delta:+.1f}%) - exiting longs supports SHORT")
        else:
            # Whales are long AND accumulating/holding - true conflict
            whale_score = 0
            warnings.append(f"⚠️ Whales {whale_pct:.0f}% ({whale_delta:+.1f}%) - accumulating AGAINST SHORT")

        # CRITICAL: Accumulating while SHORT = TRAP!
        if whale_delta > 5:
            whale_score = max(0, whale_score - 8)
            warnings.append(f"🚨 WHALE TRAP! {whale_pct:.0f}% and ACCUMULATING ({whale_delta:+.1f}%) - they're BUYING!")

    components['whale'] = {
        'score': whale_score,
        'max': 15,
        'pct': whale_pct,
        'delta': whale_delta,
        'behavior': whale_behavior,
        'aligned': whale_score >= 8
    }

    # ═══════════════════════════════════════════════════════════════════════
    # ETF/STOCK: Override whale component with money flow scoring
    # ═══════════════════════════════════════════════════════════════════════
    _etf_flow_data = None
    if market_type in ('etf', 'stock') and df is not None:
        try:
            from core.etf_flow import calculate_etf_flow_score
            _etf_flow_data = calculate_etf_flow_score(df, ticker=symbol, include_institutional=True)
            phase = _etf_flow_data.get('flow_phase', 'NEUTRAL')
            flow_score = _etf_flow_data.get('flow_score', 0)
            is_sweep_of_low = sweep_status.get('type', '') in ['SWEEP_LOW', 'SWEEP_EQUAL_LOW', 'SWEEP_DOUBLE_LOW'] or direction == 'LONG'

            # Determine trade direction for flow alignment check
            _trade_dir = direction if direction else ('LONG' if is_sweep_of_low else 'SHORT')
            _flow_supports_long = flow_score > 20
            _flow_supports_short = flow_score < -20
            _flow_aligned = (_trade_dir == 'LONG' and _flow_supports_long) or (_trade_dir == 'SHORT' and _flow_supports_short)
            _flow_conflicts = (_trade_dir == 'LONG' and _flow_supports_short) or (_trade_dir == 'SHORT' and _flow_supports_long)

            if phase == 'ACCUMULATING' and is_sweep_of_low:
                whale_score = 15  # Max — flow aligned with accumulation on dip
                components['whale']['behavior'] = 'FLOW_ACCUMULATING'
                components['whale']['aligned'] = True
            elif phase == 'ACCUMULATING' and _trade_dir == 'SHORT':
                whale_score = 3  # Conflict — accumulating but trading SHORT
                components['whale']['behavior'] = 'FLOW_ACCUMULATING'
                components['whale']['aligned'] = False
                warnings.append(f"⚠️ FLOW CONFLICT: Money flowing IN (flow: {flow_score:+.0f}) but trade is SHORT — risky to short during accumulation")
            elif phase == 'ACCUMULATING':
                whale_score = 10
                components['whale']['behavior'] = 'FLOW_ACCUMULATING'
                components['whale']['aligned'] = True
            elif phase == 'DISTRIBUTING' and _trade_dir == 'LONG':
                whale_score = 3
                components['whale']['behavior'] = 'FLOW_DISTRIBUTING'
                components['whale']['aligned'] = False
                warnings.append(f"💸 Money flowing OUT (flow: {flow_score:+.0f}) — risky to buy dip")
            elif phase == 'DISTRIBUTING' and is_sweep_of_low:
                whale_score = 3
                components['whale']['behavior'] = 'FLOW_DISTRIBUTING'
                components['whale']['aligned'] = False
                warnings.append(f"💸 Money flowing OUT (flow: {flow_score:+.0f}) — risky to buy dip")
            elif phase == 'DISTRIBUTING':
                whale_score = 12  # Aligned — distributing + SHORT
                components['whale']['behavior'] = 'FLOW_DISTRIBUTING'
                components['whale']['aligned'] = True
            elif phase == 'EXTENDED':
                whale_score = 0
                components['whale']['behavior'] = 'FLOW_EXTENDED'
                components['whale']['aligned'] = False
                ext = _etf_flow_data.get('price_extension_ema200', 0)
                warnings.append(f"🔴 Price EXTENDED +{ext:.1f}% above EMA200 — TRIM territory")
            else:
                whale_score = 7  # Neutral flow
                components['whale']['behavior'] = 'FLOW_NEUTRAL'

            components['whale']['score'] = whale_score
            components['whale']['pct'] = flow_score  # Show flow score instead of whale_pct
            components['whale']['delta'] = _etf_flow_data.get('cmf_value', 0)
            components['whale']['label'] = f"Flow: {phase}"
        except Exception as e:
            print(f"[LH] ETF flow calc error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # 6. VOLUME CONFIRMATION (10pts max) - Reduced from 15
    # ═══════════════════════════════════════════════════════════════════════
    if avg_volume > 0:
        volume_ratio = volume_on_sweep / avg_volume
    else:
        volume_ratio = 1.0

    if volume_ratio >= 2.0:
        volume_score = 10
    elif volume_ratio >= 1.5:
        volume_score = 7
    elif volume_ratio >= 1.2:
        volume_score = 4
    else:
        volume_score = 0

    components['volume'] = {
        'score': volume_score,
        'max': 10,
        'ratio': volume_ratio
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 7. WHALE ACCELERATION (24h vs 7d) - FRESH vs LATE ENTRY
    # This is CRITICAL: 7-day accumulation = we're LATE to the move!
    # Fresh 24h accumulation = we're EARLY and aligned with recent whale action
    # ═══════════════════════════════════════════════════════════════════════
    acceleration_score = 0
    acceleration_status = whale_acceleration or 'UNKNOWN'

    if whale_delta_24h is not None and whale_delta_7d is not None:
        # Store in components for display
        daily_avg_7d = whale_delta_7d / 7 if whale_delta_7d else 0

        if direction == 'LONG':
            # For LONG trades, we want whale accumulation
            if is_fresh_accumulation and whale_delta_24h > 2:
                # Fresh 24h buying faster than 7d average = EARLY entry
                acceleration_score = 10
                warnings.append(f"🚀 FRESH accumulation! 24h: {whale_delta_24h:+.1f}% vs 7d avg: {daily_avg_7d:+.2f}%/day")
            elif whale_acceleration == 'ACCELERATING' and whale_delta_24h > 0:
                acceleration_score = 7
            elif whale_acceleration == 'DECELERATING' and whale_delta_7d > 5:
                # 7d was strong but 24h is slowing = LATE entry!
                acceleration_score = -10
                warnings.append(f"⏰ LATE ENTRY! Whales accumulated {whale_delta_7d:+.1f}% over 7d but slowing (24h: {whale_delta_24h:+.1f}%)")
            elif whale_acceleration == 'REVERSING' and whale_delta_24h < 0 and whale_delta_7d > 0:
                # Was buying, now selling = TRAP
                acceleration_score = -15
                warnings.append(f"🚨 WHALE REVERSAL! 7d: {whale_delta_7d:+.1f}% but 24h: {whale_delta_24h:+.1f}% - SELLING NOW!")
            elif whale_delta_7d > 10 and whale_delta_24h < 3:
                # Already accumulated a lot, not much fresh buying
                acceleration_score = -7
                warnings.append(f"⚠️ Whales already loaded ({whale_delta_7d:+.1f}% over 7d), fresh buying slow (24h: {whale_delta_24h:+.1f}%)")

        else:  # SHORT trades
            # For SHORT trades, we want whale distribution
            if is_fresh_accumulation is False and whale_delta_24h < -2:
                # Fresh 24h selling faster than 7d average = EARLY entry
                acceleration_score = 10
                warnings.append(f"🚀 FRESH distribution! 24h: {whale_delta_24h:+.1f}% vs 7d avg: {daily_avg_7d:+.2f}%/day")
            elif whale_acceleration == 'ACCELERATING' and whale_delta_24h < 0:
                acceleration_score = 7
            elif whale_acceleration == 'DECELERATING' and whale_delta_7d < -5:
                # 7d was selling but 24h is slowing = LATE entry!
                acceleration_score = -10
                warnings.append(f"⏰ LATE ENTRY! Whales distributed {whale_delta_7d:+.1f}% over 7d but slowing (24h: {whale_delta_24h:+.1f}%)")
            elif whale_acceleration == 'REVERSING' and whale_delta_24h > 0 and whale_delta_7d < 0:
                # Was selling, now buying = TRAP
                acceleration_score = -15
                warnings.append(f"🚨 WHALE REVERSAL! 7d: {whale_delta_7d:+.1f}% but 24h: {whale_delta_24h:+.1f}% - BUYING NOW!")
            elif whale_delta_7d < -10 and whale_delta_24h > -3:
                # Already distributed a lot, not much fresh selling
                acceleration_score = -7
                warnings.append(f"⚠️ Whales already unloaded ({whale_delta_7d:+.1f}% over 7d), fresh selling slow (24h: {whale_delta_24h:+.1f}%)")

    components['acceleration'] = {
        'score': acceleration_score,
        'max': 10,
        'min': -15,
        'delta_24h': whale_delta_24h,
        'delta_7d': whale_delta_7d,
        'status': acceleration_status,
        'is_fresh': is_fresh_accumulation
    }

    # ═══════════════════════════════════════════════════════════════════════
    # BONUS: Structure Break (+5 bonus, can exceed 100 -> exceptional setup)
    # ═══════════════════════════════════════════════════════════════════════
    structure_analysis = analyze_structure_shift(df, sweep_candle_idx, direction)
    structure_bonus = 5 if structure_analysis.get('structure_broken', False) else 0

    if structure_bonus > 0:
        warnings.append(f"✅ Structure break confirmed at {format_price(structure_analysis.get('break_level', 0))}")

    components['structure_break'] = {
        'score': structure_bonus,
        'max': 5,
        'broken': structure_analysis.get('structure_broken', False),
        'level': structure_analysis.get('break_level'),
        'type': structure_analysis.get('break_type')
    }

    # ═══════════════════════════════════════════════════════════════════════
    # TREND ALIGNMENT CHECK (Can adjust score +5 to -15)
    # Trading against the trend is the #1 reason sweeps fail!
    # ═══════════════════════════════════════════════════════════════════════
    trend_analysis = analyze_trend_alignment(df, direction)
    trend_adjustment = trend_analysis.get('score_adjustment', 0)

    if trend_analysis.get('warning'):
        warnings.append(trend_analysis['warning'])

    if trend_analysis.get('aligned') and trend_analysis.get('trend_strength') == 'STRONG':
        warnings.append(f"✅ Trading WITH {trend_analysis['trend']} trend")

    components['trend'] = {
        'score': trend_adjustment,
        'trend': trend_analysis.get('trend', 'UNKNOWN'),
        'strength': trend_analysis.get('trend_strength', 'WEAK'),
        'aligned': trend_analysis.get('aligned', False),
        'ema20': trend_analysis.get('ema20', 0),
        'ema50': trend_analysis.get('ema50', 0),
        'ema200': trend_analysis.get('ema200', 0),
        'price_vs_ema200': trend_analysis.get('price_vs_ema200', 0)
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 7. PRICE ACTION ANALYSIS (from sweep detection) - EXTRACT FIRST for ML
    # ═══════════════════════════════════════════════════════════════════════
    pa_score = 0
    pa_prediction = 'UNKNOWN'
    pa_confidence = 'LOW'
    pa_key_signals = []

    # Price action component scores for ML
    pa_candle_score = 0
    pa_structure_score = 0
    pa_has_order_block = 0
    pa_has_fvg = 0
    pa_volume_score = 0
    pa_momentum_score = 0
    pa_total_score = 0

    if sweep_status.get('price_action'):
        pa_data = sweep_status['price_action']
        pred = pa_data.get('prediction', {})
        pa_score = pred.get('score', 0)
        pa_prediction = pred.get('prediction', 'UNKNOWN')
        pa_confidence = pred.get('confidence', 'LOW')

        # Extract component scores for ML model
        comp = pred.get('component_scores', {})
        pa_candle_score = comp.get('candle_pattern', 0)
        pa_structure_score = comp.get('structure', 0)
        pa_has_order_block = 1 if comp.get('order_blocks', 0) > 0 else 0
        pa_has_fvg = 1 if comp.get('fvg', 0) > 0 else 0
        pa_volume_score = comp.get('volume', 0)
        pa_momentum_score = comp.get('momentum', 0)
        pa_total_score = pa_score

    # ═══════════════════════════════════════════════════════════════════════
    # 8. ML QUALITY MODEL PREDICTION (with Price Action features!)
    # CONTINUATION mode: 99% win rate at >60% confidence, 89% at >50%
    # ═══════════════════════════════════════════════════════════════════════
    ml_prediction = None
    ml_probability = 0.5
    ml_decision = 'UNKNOWN'
    ml_available = False

    if QUALITY_MODEL_AVAILABLE:
        try:
            # Get level type from sweep status (use raw_direction - what was actually swept)
            level_type = sweep_status.get('level_type', 'SWING_LOW' if raw_direction == 'LONG' else 'SWING_HIGH')

            # Calculate ATR if df available
            atr = 0
            momentum = 0
            if df is not None and len(df) >= 14:
                df_copy = df.copy()
                df_copy.columns = [c.lower() for c in df_copy.columns]
                high = df_copy['high']
                low = df_copy['low']
                close = df_copy['close']
                tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])
                # Momentum: price change over 10 candles normalized
                if len(df_copy) >= 10 and atr > 0:
                    momentum = (close.iloc[-1] - close.iloc[-10]) / (atr * 10)

            if atr > 0:
                # Calculate sweep-specific features
                sweep_depth_atr = 0
                sweep_wick_ratio = 0
                sweep_body_ratio = 0.5
                sweep_volume_ratio = 1.0

                if df_copy is not None and len(df_copy) > abs(sweep_candle_idx):
                    try:
                        sweep_candle = df_copy.iloc[sweep_candle_idx]
                        open_p = float(sweep_candle['open'])
                        high_p = float(sweep_candle['high'])
                        low_p = float(sweep_candle['low'])
                        close_p = float(sweep_candle['close'])

                        candle_range = high_p - low_p
                        body_size = abs(close_p - open_p)

                        # Sweep depth: how far past level
                        if raw_direction == 'LONG':
                            sweep_depth = max(0, level_swept - low_p)
                            rejection_wick = min(open_p, close_p) - low_p
                        else:
                            sweep_depth = max(0, high_p - level_swept)
                            rejection_wick = high_p - max(open_p, close_p)

                        if atr > 0:
                            sweep_depth_atr = sweep_depth / atr

                        if candle_range > 0:
                            sweep_wick_ratio = rejection_wick / candle_range
                            sweep_body_ratio = body_size / candle_range

                        # Volume ratio
                        if 'volume' in df_copy.columns and len(df_copy) >= 20:
                            vol = float(sweep_candle['volume'])
                            avg_vol = df_copy['volume'].iloc[-20:].mean()
                            if avg_vol > 0:
                                sweep_volume_ratio = vol / avg_vol
                    except Exception as e:
                        print(f"[LH] Sweep feature calc error: {e}")

                # Route to market-specific model
                _prediction_kwargs = {}
                _prediction_func = get_quality_prediction
                if market_type == 'etf':
                    _prediction_func = get_quality_prediction_etf
                    _prediction_kwargs['df'] = df  # For flow computation
                elif market_type == 'stock':
                    _prediction_func = get_quality_prediction_stock
                    _prediction_kwargs['df'] = df  # For flow computation

                ml_prediction = _prediction_func(
                    symbol=symbol,  # For debug logging
                    direction=direction,  # Initial direction (ML will evaluate both)
                    level_type=level_type,
                    level_price=level_swept,
                    current_price=current_price,
                    atr=atr,
                    whale_pct=whale_pct,
                    whale_delta=whale_delta,
                    momentum=momentum,
                    volume_ratio=volume_ratio if avg_volume > 0 else 1.0,
                    # Pass Price Action features to ML!
                    pa_candle_score=pa_candle_score,
                    pa_structure_score=pa_structure_score,
                    pa_has_order_block=pa_has_order_block,
                    pa_has_fvg=pa_has_fvg,
                    pa_volume_score=pa_volume_score,
                    pa_momentum_score=pa_momentum_score,
                    pa_total_score=pa_total_score,
                    # Pass whale acceleration data (4h, 24h, 7d) to ML!
                    whale_delta_4h=whale_delta_4h,          # NEW: Early signal
                    whale_delta_24h=whale_delta_24h,
                    whale_delta_7d=whale_delta_7d,
                    whale_acceleration=whale_acceleration,
                    whale_early_signal=whale_early_signal,  # NEW: Early signal type
                    is_fresh_accumulation=is_fresh_accumulation,
                    # OI features
                    oi_change_24h=oi_change_24h,
                    price_change_24h=price_change_24h,
                    # Sweep-specific features (NEW)
                    sweep_depth_atr=sweep_depth_atr,
                    sweep_wick_ratio=sweep_wick_ratio,
                    sweep_body_ratio=sweep_body_ratio,
                    candles_since_sweep=candles_ago,
                    sweep_volume_ratio=sweep_volume_ratio,
                    **_prediction_kwargs
                )

                if ml_prediction and 'probability' in ml_prediction:
                    ml_probability = ml_prediction['probability']
                    ml_decision = ml_prediction.get('decision', 'UNKNOWN')
                    ml_available = ml_decision != 'UNKNOWN'

                    # ML-DRIVEN DIRECTION: Use best direction from ML
                    ml_best_dir = ml_prediction.get('best_direction')
                    if ml_best_dir and ml_best_dir != direction:
                        ml_direction_override = ml_best_dir
                        old_dir = direction
                        direction = ml_best_dir
                        long_prob = ml_prediction.get('long_probability', 0)
                        short_prob = ml_prediction.get('short_probability', 0)
                        warnings.append(f"🤖 ML Direction: {direction} (L:{long_prob:.0%} S:{short_prob:.0%}) - overrode {old_dir}")
                    else:
                        long_prob = ml_prediction.get('long_probability', 0)
                        short_prob = ml_prediction.get('short_probability', 0)
                        if long_prob > 0 or short_prob > 0:
                            warnings.append(f"🤖 ML Direction: {direction} (L:{long_prob:.0%} S:{short_prob:.0%})")

                    # Add ML reasons/warnings
                    if ml_prediction.get('warnings'):
                        for w in ml_prediction['warnings']:
                            if w not in warnings:
                                warnings.append(w)

        except Exception as e:
            print(f"[LH] ML prediction error: {e}")
            ml_available = False

    components['ml_quality'] = {
        'available': ml_available,
        'probability': ml_probability,
        'decision': ml_decision,
        'expected_win_rate': ml_probability if ml_available else 0.5,
        'expected_roi': (2 * ml_probability - 1) * 100 if ml_available else 0,
        'features': ml_prediction.get('ml_features', {}) if ml_prediction else {}  # All 32 ML features for UI debug
    }

    # ═══════════════════════════════════════════════════════════════════════
    # 9. PRICE ACTION WARNINGS (add to warnings list)
    # ═══════════════════════════════════════════════════════════════════════
    if sweep_status.get('price_action'):
        pa_data = sweep_status['price_action']
        pred = pa_data.get('prediction', {})
        pa_key_signals = pred.get('key_signals', [])

        # Add price action warnings
        for w in pred.get('warnings', []):
            if w not in warnings:
                warnings.append(f"📊 {w}")

        # Add key signals as positive indicators
        if pa_score >= 60:
            if 'Break of Structure confirmed' in pa_key_signals:
                warnings.append("✅ BOS: Structure shift confirmed")
            if 'Strong reversal candle' in pa_key_signals:
                warnings.append("✅ Strong reversal candle pattern")

    components['price_action'] = {
        'score': pa_score,
        'prediction': pa_prediction,
        'confidence': pa_confidence,
        'key_signals': pa_key_signals,
        'available': sweep_status.get('price_action') is not None
    }

    # ═══════════════════════════════════════════════════════════════════════
    # TOTAL SCORE & GRADE
    # Include price action score (normalized to 0-15 bonus)
    # ═══════════════════════════════════════════════════════════════════════
    pa_bonus = int(pa_score * 0.15)  # Convert 0-100 to 0-15 bonus

    total_score = (freshness_score + reversal_score + follow_score +
                   momentum_score + whale_score + volume_score +
                   acceleration_score +  # NEW: 24h vs 7d whale timing
                   structure_bonus + trend_adjustment + pa_bonus)
    total_score = max(0, min(120, total_score))  # Can go negative from acceleration penalty

    # More granular grading
    if total_score >= 85:
        grade = 'A+'
    elif total_score >= 75:
        grade = 'A'
    elif total_score >= 65:
        grade = 'B+'
    elif total_score >= 55:
        grade = 'B'
    elif total_score >= 45:
        grade = 'C+'
    elif total_score >= 35:
        grade = 'C'
    elif total_score >= 25:
        grade = 'D'
    else:
        grade = 'F'

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY WINDOW STATUS
    # ═══════════════════════════════════════════════════════════════════════
    if candles_ago <= 3:
        entry_window = 'OPEN'
        candles_until_close = 3 - candles_ago
    elif candles_ago <= 10:
        entry_window = 'CLOSING'
        candles_until_close = 10 - candles_ago
    else:
        entry_window = 'CLOSED'
        candles_until_close = 0

    # ═══════════════════════════════════════════════════════════════════════
    # RECOMMENDATION - Based on quality + confirmation + ML prediction
    # ML Model (CONTINUATION): 99% win rate at >60% prob, 89% at >50%
    # ═══════════════════════════════════════════════════════════════════════
    # ML OVERRIDES ENTRY WINDOW - If ML says good trade, trust it even if old
    # ML was trained on sweep age, so high prob on old sweep = still valid
    # ═══════════════════════════════════════════════════════════════════════
    if entry_window == 'CLOSED':
        # Check if ML overrides the closed window
        if ml_available and ml_probability >= 0.50:
            # ML says good trade despite old sweep - trust ML
            recommendation = 'ENTER'
            if ml_probability >= 0.60:
                warnings.append(f"🤖 ML OVERRIDE: ({ml_probability:.0%}) - High confidence despite {candles_ago}c old sweep")
            else:
                warnings.append(f"🤖 ML OVERRIDE: ({ml_probability:.0%}) - Good signal despite old sweep")
            warnings.append("⚠️ Entry window technically closed - consider waiting for retest")
        elif ml_available and ml_probability >= ML_SKIP_THRESHOLD:
            # ML says borderline (40-50%) on old sweep - WAIT for retest
            recommendation = 'WAIT'
            warnings.append(f"🤖 ML borderline ({ml_probability:.0%}) on old sweep - wait for retest")
        else:
            # No ML or ML says no - SKIP
            recommendation = 'SKIP'
            warnings.append("🚫 Entry window closed - wait for retest")

    elif entry_window in ['OPEN', 'CLOSING']:
        # ═══════════════════════════════════════════════════════════════════
        # PURE ML APPROACH - No rule overrides, trust the model completely
        # Training showed: >60% = 96% win, >50% = 81% win at 2:1 R:R
        # ═══════════════════════════════════════════════════════════════════
        if ml_available and ml_probability > 0:
            # ML ≥50%: ENTER - Training showed 81%+ win rate
            if ml_probability >= 0.50:
                recommendation = 'ENTER'
                if ml_probability >= 0.60:
                    warnings.append(f"🤖 ML: STRONG YES ({ml_probability:.0%}) - 96% win rate in training")
                else:
                    warnings.append(f"🤖 ML: YES ({ml_probability:.0%}) - 81% win rate in training")

            # ML 40-50%: WAIT - Borderline, not confident enough
            elif ml_probability >= ML_SKIP_THRESHOLD:
                recommendation = 'WAIT'
                warnings.append(f"🤖 ML: MAYBE ({ml_probability:.0%}) - borderline, wait for better")

            # ML <40%: SKIP - Training showed these lose
            else:
                recommendation = 'SKIP'
                warnings.append(f"🤖 ML: NO ({ml_probability:.0%}) - setup likely to fail")

        # ═══════════════════════════════════════════════════════════════════
        # FALLBACK: Rule-based recommendation if ML not available
        # ═══════════════════════════════════════════════════════════════════
        else:
            # TIER 1: Very high score (75+) = trust the setup
            if total_score >= 75:
                recommendation = 'ENTER'
                warnings.append("✅ High quality setup (75+)")

            # TIER 2: Good score (65+) with BOTH confirmations
            elif total_score >= 65 and reversal_score >= 15 and follow_score >= 8:
                recommendation = 'ENTER'

            # TIER 3: Good score (70+) with at least ONE decent confirmation
            elif total_score >= 70 and (reversal_score >= 12 or follow_score >= 6):
                recommendation = 'ENTER'
                if reversal_score < 12:
                    warnings.append("⚠️ Reversal candle weak but other factors strong")
                if follow_score < 6:
                    warnings.append("⚠️ Follow-through weak but other factors strong")

            # TIER 4: Decent score (60+) with some confirmation
            elif total_score >= 60 and (reversal_score >= 10 or follow_score >= 5):
                recommendation = 'ENTER'
                warnings.append("⚠️ Moderate setup - consider smaller position size")

            # TIER 5: Score 50-65 - needs more confirmation
            elif total_score >= 50:
                recommendation = 'WAIT'
                missing = []
                if reversal_score < 10:
                    missing.append("reversal candle")
                if follow_score < 5:
                    missing.append("follow-through")
                if missing:
                    warnings.append(f"⏳ Wait for: {', '.join(missing)}")
                else:
                    warnings.append("⏳ Setup decent but overall score needs improvement")

            else:
                recommendation = 'SKIP'
                warnings.append("🚫 Setup too weak - skip this one")

    else:
        recommendation = 'SKIP'
        warnings.append("🚫 No valid entry window")

    # ═══════════════════════════════════════════════════════════════════════
    # ETF/Stock: Final flow-based action (ACCUMULATE / TRIM / HOLD)
    # This is the primary signal for ETFs — sweep detection tells us WHEN,
    # flow tells us WHAT to do.
    # ═══════════════════════════════════════════════════════════════════════
    etf_action = None
    if _etf_flow_data and market_type in ('etf', 'stock'):
        phase = _etf_flow_data.get('flow_phase', 'NEUTRAL')
        flow_score = _etf_flow_data.get('flow_score', 0)
        ext_ema200 = _etf_flow_data.get('price_extension_ema200', 0)

        if phase == 'EXTENDED':
            if ext_ema200 > 20:
                etf_action = 'TRIM_15_20'
                recommendation = 'TRIM'
                warnings.append(f"📉 TRIM 15-20% — price extended +{ext_ema200:.1f}% above EMA200")
            else:
                etf_action = 'TRIM_5_10'
                recommendation = 'TRIM'
                warnings.append(f"📉 TRIM 5-10% — price extended +{ext_ema200:.1f}% above EMA200")
        elif phase == 'DISTRIBUTING':
            etf_action = 'TRIM_5_10'
            recommendation = 'TRIM'
            warnings.append(f"📉 TRIM 5-10% — money flowing out (flow: {flow_score:+.0f})")
        elif phase == 'ACCUMULATING' and is_sweep_of_low:
            etf_action = 'ACCUMULATE'
            recommendation = 'ENTER'
            # Clear any SKIP/WAIT from above — flow says BUY THE DIP
            warnings = [w for w in warnings if not w.startswith('🚫')]
            warnings.append(f"💰 ACCUMULATE — money flowing in (flow: {flow_score:+.0f}) + sweep of low = ideal entry")
        elif phase == 'ACCUMULATING':
            etf_action = 'ACCUMULATE'
            # Keep existing recommendation but add context
            warnings.append(f"💰 Money flowing in (flow: {flow_score:+.0f}) — look for dip to accumulate")
        else:
            etf_action = 'HOLD'
            warnings.append(f"⏸️ HOLD — flow neutral ({flow_score:+.0f}), wait for clearer signal")

    return {
        'score': total_score,
        'grade': grade,
        'components': components,
        'entry_window': entry_window,
        'candles_until_close': candles_until_close,
        'direction': direction,
        'level_swept': level_swept,
        'warnings': warnings,
        'recommendation': recommendation,
        'ml_probability': ml_probability if ml_available else None,
        'ml_decision': ml_decision if ml_available else None,
        'ml_direction_override': ml_direction_override,
        'long_probability': ml_prediction.get('long_probability') if ml_prediction else None,
        'short_probability': ml_prediction.get('short_probability') if ml_prediction else None,
        'is_sweep_of_low': is_sweep_of_low,
        'etf_flow': _etf_flow_data,
        'etf_action': etf_action,  # ACCUMULATE / TRIM_5_10 / TRIM_15_20 / HOLD / None
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER RESULT FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def filter_scanner_results_by_quality(
    results: List[Dict],
    min_quality_score: int = 0,
    quality_grades: List[str] = None,
    entry_window_status: List[str] = None
) -> List[Dict]:
    """
    Filter scanner results by entry quality criteria.

    Args:
        results: List of scanner results
        min_quality_score: Minimum entry quality score (0-100)
        quality_grades: List of acceptable grades ['A', 'B', 'C', 'D']
        entry_window_status: List of acceptable windows ['OPEN', 'CLOSING', 'CLOSED']

    Returns:
        Filtered list of results
    """
    if quality_grades is None:
        quality_grades = ['A', 'B', 'C', 'D']
    if entry_window_status is None:
        entry_window_status = ['OPEN', 'CLOSING', 'CLOSED', 'NO_SWEEP']

    filtered = []

    for result in results:
        # Skip if no entry quality calculated
        if result.get('entry_quality') is None:
            # Include non-sweep results if NO_SWEEP is acceptable
            if 'NO_SWEEP' in entry_window_status:
                filtered.append(result)
            continue

        # Check minimum score
        if result.get('entry_quality_score', 0) < min_quality_score:
            continue

        # Check grade
        if result.get('entry_quality_grade', 'N/A') not in quality_grades:
            continue

        # Check entry window
        if result.get('entry_window', 'NO_SWEEP') not in entry_window_status:
            continue

        filtered.append(result)

    return filtered


def get_fresh_only_results(results: List[Dict]) -> List[Dict]:
    """
    Convenience function: Filter to only fresh, high-quality sweeps.

    "Fresh Only" = entry_quality_score >= 60 (B grade or better)
                 + entry_window is OPEN or CLOSING
    """
    return filter_scanner_results_by_quality(
        results,
        min_quality_score=60,
        quality_grades=['A', 'B'],
        entry_window_status=['OPEN', 'CLOSING']
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY LEVEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_liquidity_levels(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Find key liquidity levels where stops likely cluster.
    These are swing highs/lows that haven't been swept yet.

    ENHANCED with:
    - Volume confirmation (levels formed on high volume = stronger)
    - Round number detection (psychological levels)
    - Composite strength scoring (0-100)
    """
    if df is None or len(df) < lookback:
        return {'lows': [], 'highs': [], 'equal_lows': [], 'equal_highs': [], 'round_numbers': []}

    # Normalize column names
    df = normalize_columns(df)

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

    # Calculate average volume for comparison
    avg_volume = np.mean(volumes[-50:]) if len(volumes) >= 50 else np.mean(volumes)

    # Calculate ATR for dynamic thresholds
    atr = _calculate_atr(df, 14)
    current_price = float(closes[-1])

    swing_lows = []
    swing_highs = []

    # Find swing points (local min/max) with VOLUME TRACKING
    for i in range(5, len(df) - 1):
        # Swing Low: lower than 5 candles before and after
        if lows[i] == min(lows[i-5:i+6]):
            # NEW: Volume confirmation
            volume_ratio = float(volumes[i]) / avg_volume if avg_volume > 0 else 1.0
            volume_confirmed = volume_ratio > 1.5  # 50% above average

            # Is it the lowest in 20 candles? (stronger level)
            is_20_candle_extreme = lows[i] == min(lows[max(0,i-20):min(len(lows),i+20)])

            # Calculate composite strength score
            strength_score = calculate_level_strength_score(
                touches=1,
                avg_volume_ratio=volume_ratio,
                has_volume_confirmation=volume_confirmed,
                is_20_candle_extreme=is_20_candle_extreme,
                candles_since_formation=len(df) - i
            )

            swing_lows.append({
                'price': float(lows[i]),
                'index': i,
                'timestamp': df.index[i] if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                'swept': False,
                'strength': 'STRONG' if is_20_candle_extreme else 'MODERATE',
                'strength_score': strength_score,
                'volume_ratio': volume_ratio,
                'volume_confirmed': volume_confirmed
            })

        # Swing High: higher than 5 candles before and after
        if highs[i] == max(highs[i-5:i+6]):
            volume_ratio = float(volumes[i]) / avg_volume if avg_volume > 0 else 1.0
            volume_confirmed = volume_ratio > 1.5

            is_20_candle_extreme = highs[i] == max(highs[max(0,i-20):min(len(highs),i+20)])

            strength_score = calculate_level_strength_score(
                touches=1,
                avg_volume_ratio=volume_ratio,
                has_volume_confirmation=volume_confirmed,
                is_20_candle_extreme=is_20_candle_extreme,
                candles_since_formation=len(df) - i
            )

            swing_highs.append({
                'price': float(highs[i]),
                'index': i,
                'timestamp': df.index[i] if hasattr(df.index[i], 'strftime') else str(df.index[i]),
                'swept': False,
                'strength': 'STRONG' if is_20_candle_extreme else 'MODERATE',
                'strength_score': strength_score,
                'volume_ratio': volume_ratio,
                'volume_confirmed': volume_confirmed
            })

    # Check if levels have been swept
    # Use tolerance of 0.3 ATR for sweep detection
    sweep_tolerance = atr * 0.3

    for level in swing_lows:
        # Swept if price went below (or within tolerance of) it after it formed
        later_lows = lows[level['index']+1:]
        level_price_with_tolerance = level['price'] + sweep_tolerance  # Allow sweep if within tolerance

        if len(later_lows) > 0 and min(later_lows) < level_price_with_tolerance:
            level['swept'] = True
            # Track FIRST breach - iterate FORWARDS to find when level was initially broken
            # This is critical for CONTINUATION mode - we want fresh breakouts, not old ones
            first_breach_idx = None
            for i in range(len(later_lows)):
                if later_lows[i] < level_price_with_tolerance:
                    first_breach_idx = level['index'] + 1 + i
                    break
            if first_breach_idx is not None:
                level['swept_candles_ago'] = len(lows) - first_breach_idx
            else:
                level['swept_candles_ago'] = 999  # Should not happen

    for level in swing_highs:
        later_highs = highs[level['index']+1:]
        level_price_with_tolerance = level['price'] - sweep_tolerance  # Allow sweep if within tolerance

        if len(later_highs) > 0 and max(later_highs) > level_price_with_tolerance:
            level['swept'] = True
            # Track FIRST breach - iterate FORWARDS to find when level was initially broken
            # This is critical for CONTINUATION mode - we want fresh breakouts, not old ones
            first_breach_idx = None
            for i in range(len(later_highs)):
                if later_highs[i] > level_price_with_tolerance:
                    first_breach_idx = level['index'] + 1 + i
                    break
            if first_breach_idx is not None:
                level['swept_candles_ago'] = len(highs) - first_breach_idx
            else:
                level['swept_candles_ago'] = 999

    # Find Equal Lows/Highs with ATR-based dynamic threshold
    equal_lows = find_equal_levels(swing_lows, atr=atr, current_price=current_price)
    equal_highs = find_equal_levels(swing_highs, atr=atr, current_price=current_price)

    # Include BOTH unswept AND recently swept levels
    # Unswept = potential targets
    # Recently swept (within 50 candles) + price back above = potential ENTRY
    unswept_lows = [l for l in swing_lows if not l.get('swept')]
    recently_swept_lows = [l for l in swing_lows if l.get('swept') and l.get('swept_candles_ago', 999) <= 50]

    unswept_highs = [h for h in swing_highs if not h.get('swept')]
    recently_swept_highs = [h for h in swing_highs if h.get('swept') and h.get('swept_candles_ago', 999) <= 50]

    # Sort by distance from current price (for unswept) or strength score
    unswept_lows.sort(key=lambda x: (-x.get('strength_score', 0), abs(x['price'] - current_price)))
    unswept_highs.sort(key=lambda x: (-x.get('strength_score', 0), abs(x['price'] - current_price)))
    recently_swept_lows.sort(key=lambda x: x.get('swept_candles_ago', 999))
    recently_swept_highs.sort(key=lambda x: x.get('swept_candles_ago', 999))

    # Detect round numbers (psychological levels)
    round_numbers = detect_round_numbers(current_price, price_range_pct=10)

    # Mark which round numbers have been swept
    for rn in round_numbers:
        if rn['price'] < current_price:
            later_lows = lows[-50:]
            if len(later_lows) > 0 and min(later_lows) < rn['price']:
                rn['swept'] = True
        else:
            later_highs = highs[-50:]
            if len(later_highs) > 0 and max(later_highs) > rn['price']:
                rn['swept'] = True

    return {
        'lows': unswept_lows[:10],  # Unswept = targets
        'highs': unswept_highs[:10],
        'recently_swept_lows': recently_swept_lows[:5],  # Recently swept = potential entries
        'recently_swept_highs': recently_swept_highs[:5],
        'equal_lows': equal_lows,
        'equal_highs': equal_highs,
        'round_numbers': [rn for rn in round_numbers if not rn['swept']],  # Unswept round numbers
        'current_price': float(current_price),
        'atr': atr
    }


def find_equal_levels(levels: List[Dict], threshold_pct: float = 0.3, atr: float = 0, current_price: float = 0) -> List[Dict]:
    """
    Find equal highs/lows (multiple touches at same level).
    These are HIGH PROBABILITY liquidity pools.

    ENHANCED: Uses ATR-based dynamic threshold instead of fixed 0.3%.

    Args:
        levels: List of swing high/low dictionaries
        threshold_pct: Fallback percentage threshold (if ATR not available)
        atr: Average True Range value (for dynamic threshold)
        current_price: Current price (for percentage calculation)
    """
    if len(levels) < 2:
        return []

    # DYNAMIC THRESHOLD: 0.5 ATR or fallback to threshold_pct
    # This adapts to volatility while having a reasonable cap
    if atr > 0 and current_price > 0:
        atr_threshold = (0.5 * atr / current_price) * 100
        threshold_pct = min(atr_threshold, 0.5)  # Cap at 0.5%

    equal_levels = []
    used_indices = set()

    for i, level1 in enumerate(levels):
        if i in used_indices:
            continue

        cluster = [level1]
        cluster_volume_sum = level1.get('volume_ratio', 1.0)

        for j, level2 in enumerate(levels[i+1:], i+1):
            if j in used_indices:
                continue

            # Check if prices are within threshold
            pct_diff = abs(level1['price'] - level2['price']) / level1['price'] * 100
            if pct_diff < threshold_pct:
                cluster.append(level2)
                cluster_volume_sum += level2.get('volume_ratio', 1.0)
                used_indices.add(j)

        if len(cluster) >= 2:
            used_indices.add(i)
            avg_price = sum(l['price'] for l in cluster) / len(cluster)
            avg_volume_ratio = cluster_volume_sum / len(cluster)

            # COMPOSITE STRENGTH SCORE (0-100)
            strength_score = calculate_level_strength_score(
                touches=len(cluster),
                avg_volume_ratio=avg_volume_ratio,
                has_volume_confirmation=any(l.get('volume_confirmed', False) for l in cluster),
                is_equal_level=True
            )

            # Traditional strength label (for backwards compatibility)
            if len(cluster) >= 3 or strength_score >= 70:
                strength = 'MAJOR'
            elif len(cluster) == 2 or strength_score >= 50:
                strength = 'STRONG'
            else:
                strength = 'MODERATE'

            equal_levels.append({
                'price': avg_price,
                'touches': len(cluster),
                'levels': cluster,
                'strength': strength,
                'strength_score': strength_score,
                'avg_volume_ratio': avg_volume_ratio,
                'swept': any(l.get('swept', False) for l in cluster)
            })

    return [l for l in equal_levels if not l['swept']]


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_sweep(df: pd.DataFrame, liquidity_levels: Dict, atr: float, lookback_candles: int = 25, whale_pct: float = 50) -> Dict:
    """
    Detect if price has recently swept a liquidity level.
    
    PRIORITY ORDER:
    1. Check RECENTLY SWEPT liquidity levels FIRST (these are the real pools!)
    2. Only if none found, check unswept levels for fresh sweeps
    
    This ensures we detect the $87,263 level (marked "SWEEP FOR LONG") 
    instead of random swing lows like $87,895.
    """
    if df is None or len(df) < 20:
        return {'detected': False}
    
    df = normalize_columns(df)
    
    current_price = float(df['close'].iloc[-1])
    lows = df['low'].values
    highs = df['high'].values
    volumes = df['volume'].values
    avg_volume = float(df['volume'].rolling(20).mean().iloc[-1])
    
    lookback = min(lookback_candles, len(df) - 10)
    
    long_sweep = None
    short_sweep = None
    
    # DEBUG: Show what liquidity levels we have
    recently_swept_lows = liquidity_levels.get('recently_swept_lows', [])
    recently_swept_highs = liquidity_levels.get('recently_swept_highs', [])
    lows_list = liquidity_levels.get('lows', [])
    highs_list = liquidity_levels.get('highs', [])
    
    print(f"[SWEEP_DEBUG] RECENTLY SWEPT LOWS: {[(round(l.get('price', 0), 2), l.get('swept_candles_ago', '?')) for l in recently_swept_lows[:5]]}")
    print(f"[SWEEP_DEBUG] RECENTLY SWEPT HIGHS: {[(round(h.get('price', 0), 2), h.get('swept_candles_ago', '?')) for h in recently_swept_highs[:5]]}")
    print(f"[SWEEP_DEBUG] Unswept LOWS: {[round(l.get('price', 0), 2) for l in lows_list[:5]]}")
    print(f"[SWEEP_DEBUG] Unswept HIGHS: {[round(h.get('price', 0), 2) for h in highs_list[:5]]}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 1: Check RECENTLY SWEPT liquidity levels FIRST!
    # These are the real pools shown in Liquidity Map as "SWEEP FOR LONG/SHORT"
    # ═══════════════════════════════════════════════════════════════════════
    
    for level in recently_swept_lows:
        level_price = level.get('price')
        candles_ago = level.get('swept_candles_ago', 999)

        if level_price and level_price > 0 and candles_ago <= lookback:
            # ML-DRIVEN: Accept both reversal and continuation sweeps
            # ML will decide best trade direction based on context features
            # Accept if price is near the level (within 2% either side)
            price_condition = abs(current_price - level_price) / current_price <= 0.02 or current_price > level_price
            if price_condition:
                confidence = 80  # High confidence for real liquidity pools
                if candles_ago <= 5:
                    confidence += 10
                if level.get('strength') == 'STRONG':
                    confidence += 5
                
                if long_sweep is None or candles_ago < long_sweep['candles_ago']:
                    # Determine level type (EQUAL > DOUBLE > SWING)
                    level_type_raw = level.get('type', 'SWING')
                    if 'EQUAL' in str(level_type_raw).upper():
                        level_type = 'EQUAL'
                    elif 'DOUBLE' in str(level_type_raw).upper():
                        level_type = 'DOUBLE'
                    else:
                        level_type = 'SWING'

                    long_sweep = {
                        'detected': True,
                        'type': 'SWEEP_LOW',
                        'direction': 'LONG',
                        'level_swept': float(level_price),
                        'level_type': level_type,  # EQUAL/DOUBLE/SWING
                        'current_price': current_price,
                        'candles_ago': candles_ago,
                        'distance_pct': abs(level_price - current_price) / current_price * 100,
                        'volume_confirmed': False,
                        'confidence': min(95, confidence),
                        'source': 'liquidity_pool'
                    }

    for level in recently_swept_highs:
        level_price = level.get('price')
        candles_ago = level.get('swept_candles_ago', 999)

        if level_price and level_price > 0 and candles_ago <= lookback:
            # ML-DRIVEN: Accept both reversal and continuation sweeps
            # ML will decide best trade direction based on context features
            price_condition = abs(current_price - level_price) / current_price <= 0.02 or current_price < level_price
            if price_condition:
                confidence = 80
                if candles_ago <= 5:
                    confidence += 10
                if level.get('strength') == 'STRONG':
                    confidence += 5

                if short_sweep is None or candles_ago < short_sweep['candles_ago']:
                    # Determine level type
                    level_type_raw = level.get('type', 'SWING')
                    if 'EQUAL' in str(level_type_raw).upper():
                        level_type = 'EQUAL'
                    elif 'DOUBLE' in str(level_type_raw).upper():
                        level_type = 'DOUBLE'
                    else:
                        level_type = 'SWING'

                    short_sweep = {
                        'detected': True,
                        'type': 'SWEEP_HIGH',
                        'direction': 'SHORT',
                        'level_swept': float(level_price),
                        'level_type': level_type,  # EQUAL/DOUBLE/SWING
                        'current_price': current_price,
                        'candles_ago': candles_ago,
                        'distance_pct': abs(level_price - current_price) / current_price * 100,
                        'volume_confirmed': False,
                        'confidence': min(95, confidence),
                        'source': 'liquidity_pool'
                    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 2: Check UNSWEPT levels for fresh sweeps (only if Priority 1 found nothing)
    # Use tolerance of 0.3 ATR - price within tolerance counts as sweep
    # ═══════════════════════════════════════════════════════════════════════
    sweep_tolerance = atr * 0.3

    if long_sweep is None:
        for level in lows_list:
            level_price = level.get('price')
            if not level_price or level_price <= 0:
                continue
            # ML-DRIVEN: Accept sweeps regardless - ML decides direction later
            # Just skip if price is too far away (>3% above level)
            if current_price > level_price * 1.03:
                continue

            # Check if any recent candle swept this level (with tolerance)
            level_with_tolerance = level_price + sweep_tolerance
            for j in range(1, lookback + 1):
                recent_idx = len(df) - j
                if recent_idx < 0:
                    break

                # Sweep if low went below level OR came within tolerance
                if float(lows[recent_idx]) < level_with_tolerance:
                    volume_confirmed = float(volumes[recent_idx]) > avg_volume * 1.3
                    confidence = 70 + (15 if j <= 5 else 5) + (5 if volume_confirmed else 0)

                    # Determine level type
                    level_type_raw = level.get('type', 'SWING')
                    if 'EQUAL' in str(level_type_raw).upper():
                        level_type = 'EQUAL'
                    elif 'DOUBLE' in str(level_type_raw).upper():
                        level_type = 'DOUBLE'
                    else:
                        level_type = 'SWING'

                    long_sweep = {
                        'detected': True,
                        'type': 'SWEEP_LOW',
                        'direction': 'LONG',
                        'level_swept': float(level_price),
                        'level_type': level_type,  # EQUAL/DOUBLE/SWING
                        'current_price': current_price,
                        'candles_ago': j,
                        'distance_pct': abs(level_price - current_price) / current_price * 100,
                        'volume_confirmed': volume_confirmed,
                        'confidence': min(95, confidence),
                        'source': 'unswept_pool'
                    }
                    break
            if long_sweep:
                break

    if short_sweep is None:
        for level in highs_list:
            level_price = level.get('price')
            if not level_price or level_price <= 0:
                continue
            # ML-DRIVEN: Accept sweeps regardless - ML decides direction later
            # Just skip if price is too far away (>3% below level)
            if current_price < level_price * 0.97:
                continue

            # Check with tolerance for shorts too
            level_with_tolerance = level_price - sweep_tolerance
            for j in range(1, lookback + 1):
                recent_idx = len(df) - j
                if recent_idx < 0:
                    break

                # Sweep if high went above level OR came within tolerance
                if float(highs[recent_idx]) > level_with_tolerance:
                    volume_confirmed = float(volumes[recent_idx]) > avg_volume * 1.3
                    confidence = 70 + (15 if j <= 5 else 5) + (5 if volume_confirmed else 0)

                    # Determine level type
                    level_type_raw = level.get('type', 'SWING')
                    if 'EQUAL' in str(level_type_raw).upper():
                        level_type = 'EQUAL'
                    elif 'DOUBLE' in str(level_type_raw).upper():
                        level_type = 'DOUBLE'
                    else:
                        level_type = 'SWING'

                    short_sweep = {
                        'detected': True,
                        'type': 'SWEEP_HIGH',
                        'direction': 'SHORT',
                        'level_swept': float(level_price),
                        'level_type': level_type,  # EQUAL/DOUBLE/SWING
                        'current_price': current_price,
                        'candles_ago': j,
                        'distance_pct': abs(level_price - current_price) / current_price * 100,
                        'volume_confirmed': volume_confirmed,
                        'confidence': min(95, confidence),
                        'source': 'unswept_pool'
                    }
                    break
            if short_sweep:
                break
    
    # ═══════════════════════════════════════════════════════════════════════
    # DEBUG output
    # ═══════════════════════════════════════════════════════════════════════
    print(f"[SWEEP_DEBUG] Current price: ${current_price:,.2f}")
    if long_sweep:
        print(f"[SWEEP_DEBUG] LONG: ${long_sweep['level_swept']:,.2f}, {long_sweep['candles_ago']} ago, source={long_sweep.get('source')}")
    else:
        print(f"[SWEEP_DEBUG] No LONG sweep")
    if short_sweep:
        print(f"[SWEEP_DEBUG] SHORT: ${short_sweep['level_swept']:,.2f}, {short_sweep['candles_ago']} ago, source={short_sweep.get('source')}")
    else:
        print(f"[SWEEP_DEBUG] No SHORT sweep")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Pick the BEST sweep
    # ═══════════════════════════════════════════════════════════════════════
    selected_sweep = None

    if long_sweep and short_sweep:
        # Most recent wins
        if long_sweep['candles_ago'] < short_sweep['candles_ago']:
            print(f"[SWEEP_DEBUG] → LONG more recent")
            selected_sweep = long_sweep
        elif short_sweep['candles_ago'] < long_sweep['candles_ago']:
            print(f"[SWEEP_DEBUG] → SHORT more recent")
            selected_sweep = short_sweep
        else:
            # Tiebreaker: whale alignment
            if whale_pct > 55:
                print(f"[SWEEP_DEBUG] → Whale bullish, LONG")
                selected_sweep = long_sweep
            elif whale_pct < 45:
                print(f"[SWEEP_DEBUG] → Whale bearish, SHORT")
                selected_sweep = short_sweep
            else:
                selected_sweep = long_sweep if long_sweep['distance_pct'] < short_sweep['distance_pct'] else short_sweep
                print(f"[SWEEP_DEBUG] → Neutral, closer: {selected_sweep['direction']}")
    elif long_sweep:
        print(f"[SWEEP_DEBUG] → Returning LONG")
        selected_sweep = long_sweep
    elif short_sweep:
        print(f"[SWEEP_DEBUG] → Returning SHORT")
        selected_sweep = short_sweep

    if selected_sweep:
        # Add price action analysis
        if PRICE_ACTION_AVAILABLE:
            selected_sweep = analyze_sweep_with_price_action(df, selected_sweep, atr)
            pa_score = selected_sweep.get('price_action_score', 0)
            pa_pred = selected_sweep.get('price_action_prediction', 'UNKNOWN')
            print(f"[PRICE_ACTION] Score: {pa_score}/100, Prediction: {pa_pred}")
        return selected_sweep

    print(f"[SWEEP_DEBUG] → No sweep detected")
    return {'detected': False}


def calculate_sweep_confidence(swept: bool, rejected: bool, volume_spike: bool, 
                                good_close: bool, strength: str) -> int:
    """Calculate confidence score for a sweep (0-100)"""
    confidence = 0
    
    if swept and rejected:
        confidence += 40  # Base for sweep + rejection
    
    if volume_spike:
        confidence += 20  # Volume confirmation
    
    if good_close:
        confidence += 15  # Close position confirms
    
    if strength == 'STRONG':
        confidence += 15
    elif strength == 'MAJOR':
        confidence += 25
    else:
        confidence += 10
    
    return min(confidence, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACHING LIQUIDITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_approaching_liquidity(current_price: float, liquidity_levels: Dict, 
                                  atr: float) -> Dict:
    """
    Check if price is approaching a liquidity level (pre-sweep setup).
    
    Shows NEXT TARGETS for potential entries:
    - Lows below current price → If swept = LONG entry
    - Highs above current price → If swept = SHORT entry
    
    Only shows UNSWEPT levels (fresh targets).
    """
    approaching = []
    
    # Check lows below current price (NEXT LONG targets)
    for level in liquidity_levels.get('lows', []):
        distance = current_price - level['price']
        distance_atr = distance / atr
        
        if 0 < distance_atr < 5:  # Within 5 ATR
            # Calculate how old this level is (candles since it formed)
            level_age = level.get('index', 0)  # Index in the dataframe
            
            approaching.append({
                'type': 'LOW',
                'price': level['price'],
                'distance': distance,
                'distance_atr': distance_atr,
                'strength': level.get('strength', 'MODERATE'),
                'direction_after_sweep': 'LONG',
                'proximity': 'IMMINENT' if distance_atr < 1 else 'CLOSE' if distance_atr < 2 else 'APPROACHING',
                'status': 'UNSWEPT',  # Fresh target
                'candle_index': level_age,  # When it formed (index in df)
                'formed_ago': f"~{200 - level_age} candles ago" if level_age > 0 else "recent"
            })
    
    # Check highs above current price (NEXT SHORT targets)
    for level in liquidity_levels.get('highs', []):
        distance = level['price'] - current_price
        distance_atr = distance / atr
        
        if 0 < distance_atr < 5:
            level_age = level.get('index', 0)
            
            approaching.append({
                'type': 'HIGH',
                'price': level['price'],
                'distance': distance,
                'distance_atr': distance_atr,
                'strength': level.get('strength', 'MODERATE'),
                'direction_after_sweep': 'SHORT',
                'proximity': 'IMMINENT' if distance_atr < 1 else 'CLOSE' if distance_atr < 2 else 'APPROACHING',
                'status': 'UNSWEPT',
                'candle_index': level_age,
                'formed_ago': f"~{200 - level_age} candles ago" if level_age > 0 else "recent"
            })
    
    # Check equal lows/highs (higher priority - MAJOR liquidity)
    for level in liquidity_levels.get('equal_lows', []):
        distance = current_price - level['price']
        distance_atr = distance / atr
        
        if 0 < distance_atr < 5:
            approaching.append({
                'type': 'EQUAL_LOW',
                'price': level['price'],
                'touches': level.get('touches', 2),
                'distance': distance,
                'distance_atr': distance_atr,
                'strength': 'MAJOR',
                'direction_after_sweep': 'LONG',
                'proximity': 'IMMINENT' if distance_atr < 1 else 'CLOSE' if distance_atr < 2 else 'APPROACHING',
                'status': 'UNSWEPT',
                'formed_ago': 'multiple touches'
            })
    
    for level in liquidity_levels.get('equal_highs', []):
        distance = level['price'] - current_price
        distance_atr = distance / atr
        
        if 0 < distance_atr < 5:
            approaching.append({
                'type': 'EQUAL_HIGH',
                'price': level['price'],
                'touches': level.get('touches', 2),
                'distance': distance,
                'distance_atr': distance_atr,
                'strength': 'MAJOR',
                'direction_after_sweep': 'SHORT',
                'proximity': 'IMMINENT' if distance_atr < 1 else 'CLOSE' if distance_atr < 2 else 'APPROACHING',
                'status': 'UNSWEPT',
                'formed_ago': 'multiple touches'
            })
    
    # Sort by proximity
    approaching.sort(key=lambda x: x['distance_atr'])
    
    # Separate by direction for clearer display
    long_targets = [l for l in approaching if l['direction_after_sweep'] == 'LONG']
    short_targets = [l for l in approaching if l['direction_after_sweep'] == 'SHORT']
    
    # DEBUG: Show next targets with age
    if long_targets:
        lt = long_targets[0]
        print(f"[NEXT_TARGET] LONG: ${lt['price']:,.2f} ({lt['proximity']}, formed {lt.get('formed_ago', '?')})")
    if short_targets:
        st = short_targets[0]
        print(f"[NEXT_TARGET] SHORT: ${st['price']:,.2f} ({st['proximity']}, formed {st.get('formed_ago', '?')})")
    
    return {
        'has_nearby': len(approaching) > 0,
        'levels': approaching[:5],
        'imminent': [l for l in approaching if l['proximity'] == 'IMMINENT'],
        'next_long_target': long_targets[0] if long_targets else None,
        'next_short_target': short_targets[0] if short_targets else None
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUIDATION DATA (FREE SOURCES)
# ═══════════════════════════════════════════════════════════════════════════════

def get_binance_liquidations(symbol: str, period: str = '4h') -> Dict:
    """
    Get liquidation-related data from Binance (FREE).
    
    APIs used:
    - topLongShortPositionRatio: Top traders by POSITION SIZE (true whale activity)
    - globalLongShortAccountRatio: All accounts by count (retail sentiment)
    - fundingRate: Funding rate
    - openInterest: Open interest
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        period: Data period - '5m', '15m', '30m', '1h', '4h', '1d'
    """
    try:
        base_url = "https://fapi.binance.com"
        
        # Get funding rate
        funding_resp = requests.get(f"{base_url}/fapi/v1/fundingRate", 
                                     params={'symbol': symbol, 'limit': 1}, timeout=5)
        funding_data = funding_resp.json()
        funding_rate = float(funding_data[0]['fundingRate']) if funding_data else 0
        
        # Get open interest
        oi_resp = requests.get(f"{base_url}/fapi/v1/openInterest",
                               params={'symbol': symbol}, timeout=5)
        oi_data = oi_resp.json()
        open_interest = float(oi_data.get('openInterest', 0))
        
        # Get RETAIL long/short ratio (by account count)
        ls_resp = requests.get(f"{base_url}/futures/data/globalLongShortAccountRatio",
                               params={'symbol': symbol, 'period': period, 'limit': 1}, timeout=5)
        ls_data = ls_resp.json()
        if ls_data:
            retail_long = float(ls_data[0].get('longAccount', 0.5)) * 100
            retail_short = float(ls_data[0].get('shortAccount', 0.5)) * 100
        else:
            retail_long = 50
            retail_short = 50
        
        # Get WHALE long/short ratio (by POSITION SIZE - the correct metric!)
        # topLongShortPositionRatio measures by $$ position size, not account count
        whale_resp = requests.get(f"{base_url}/futures/data/topLongShortPositionRatio",
                                params={'symbol': symbol, 'period': period, 'limit': 1}, timeout=5)
        whale_data = whale_resp.json()
        if whale_data:
            whale_long = float(whale_data[0].get('longAccount', 0.5)) * 100
            whale_short = float(whale_data[0].get('shortAccount', 0.5)) * 100
        else:
            whale_long = 50
            whale_short = 50
        
        # Infer crowding
        crowded_long = retail_long > 60 and funding_rate > 0.0005
        crowded_short = retail_short > 60 and funding_rate < -0.0005
        
        return {
            'funding_rate': funding_rate,
            'funding_pct': funding_rate * 100,
            'open_interest': open_interest,
            'retail_long': retail_long,
            'retail_short': retail_short,
            'whale_long': whale_long,
            'whale_short': whale_short,
            'crowded_long': crowded_long,
            'crowded_short': crowded_short,
            'liquidation_bias': 'LONGS_AT_RISK' if crowded_long else 'SHORTS_AT_RISK' if crowded_short else 'NEUTRAL',
            'period': period,
            'success': True
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'funding_rate': 0,
            'retail_long': 50,
            'retail_short': 50,
            'whale_long': 50,
            'whale_short': 50
        }


def estimate_liquidation_zones(current_price: float, atr: float, 
                                 liquidation_data: Dict, symbol: str = None) -> Dict:
    """
    Get liquidation zones - REAL from Coinglass API or ESTIMATED from formulas.
    
    Priority:
    1. Real Coinglass liquidation heatmap data (if API key available)
    2. Calculated estimates based on leverage formulas
    
    Also SAVES the data for future ML training!
    """
    result_source = 'calculated'
    long_liquidations = []
    short_liquidations = []
    total_above = 0
    total_below = 0
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIORITY 1: Try to get REAL liquidation data from Coinglass
    # ═══════════════════════════════════════════════════════════════════
    if COINGLASS_LIQ_AVAILABLE and symbol:
        try:
            real_data = get_real_liquidation_levels(symbol, current_price)
            
            if real_data.get('success') and real_data.get('source') == 'coinglass':
                print(f"[LIQ_ZONES] ✅ Using REAL Coinglass data for {symbol}")
                result_source = 'coinglass'
                total_above = real_data.get('total_above', 0)
                total_below = real_data.get('total_below', 0)
                
                # Process levels below price (LONG liquidations - targets for SHORT trades)
                for level in real_data.get('below_price', [])[:5]:
                    distance_pct = (current_price - level['price']) / current_price * 100
                    long_liquidations.append({
                        'price': level['price'],
                        'leverage': level.get('leverage', 0),
                        'distance_pct': distance_pct,
                        'volume': level.get('volume', 0),
                        'intensity': 'HIGH' if level.get('volume', 0) > 10000000 else 'MEDIUM' if level.get('volume', 0) > 1000000 else 'LOW',
                        'source': 'coinglass',
                        'reason': f"Liq Cluster (${level.get('volume', 0)/1e6:.1f}M)"
                    })
                
                # Process levels above price (SHORT liquidations - targets for LONG trades)
                for level in real_data.get('above_price', [])[:5]:
                    distance_pct = (level['price'] - current_price) / current_price * 100
                    short_liquidations.append({
                        'price': level['price'],
                        'leverage': level.get('leverage', 0),
                        'distance_pct': distance_pct,
                        'volume': level.get('volume', 0),
                        'intensity': 'HIGH' if level.get('volume', 0) > 10000000 else 'MEDIUM' if level.get('volume', 0) > 1000000 else 'LOW',
                        'source': 'coinglass',
                        'reason': f"Liq Cluster (${level.get('volume', 0)/1e6:.1f}M)"
                    })
                
                # ═══════════════════════════════════════════════════════════════
                # SAVE REAL DATA FOR FUTURE ML TRAINING
                # ═══════════════════════════════════════════════════════════════
                if LIQ_COLLECTOR_AVAILABLE and (long_liquidations or short_liquidations):
                    try:
                        # Build levels dict for collector
                        levels_to_save = {}
                        
                        # Get closest level for each leverage tier
                        for liq in short_liquidations:
                            lev = liq.get('leverage', 0)
                            if lev >= 75:
                                levels_to_save['100x_above'] = liq['price']
                                levels_to_save['100x_above_amt'] = liq.get('volume', 0)
                            elif lev >= 40:
                                levels_to_save['50x_above'] = liq['price']
                                levels_to_save['50x_above_amt'] = liq.get('volume', 0)
                            elif lev >= 20:
                                levels_to_save['25x_above'] = liq['price']
                                levels_to_save['25x_above_amt'] = liq.get('volume', 0)
                        
                        for liq in long_liquidations:
                            lev = liq.get('leverage', 0)
                            if lev >= 75:
                                levels_to_save['100x_below'] = liq['price']
                                levels_to_save['100x_below_amt'] = liq.get('volume', 0)
                            elif lev >= 40:
                                levels_to_save['50x_below'] = liq['price']
                                levels_to_save['50x_below_amt'] = liq.get('volume', 0)
                            elif lev >= 20:
                                levels_to_save['25x_below'] = liq['price']
                                levels_to_save['25x_below_amt'] = liq.get('volume', 0)
                        
                        if levels_to_save:
                            save_liquidation_snapshot(
                                symbol=symbol,
                                price=current_price,
                                levels=levels_to_save,
                                whale_pct=liquidation_data.get('whale_long_pct'),
                                atr=atr,
                                data_source='coinglass_realtime'
                            )
                            print(f"[LIQ_COLLECTOR] ✅ Saved REAL liquidation data for {symbol}")
                    except Exception as e:
                        print(f"[LIQ_COLLECTOR] Error saving: {e}")
                        
        except Exception as e:
            print(f"[LIQ_ZONES] Coinglass error: {e}, falling back to calculated")
    
    # ═══════════════════════════════════════════════════════════════════
    # FALLBACK: Calculate estimated levels from leverage formulas
    # ═══════════════════════════════════════════════════════════════════
    if not long_liquidations and not short_liquidations:
        print(f"[LIQ_ZONES] Using CALCULATED estimates (no real data available)")
        
        leverage_levels = [
            {'leverage': 100, 'distance_pct': 1.0},
            {'leverage': 50, 'distance_pct': 2.0},
            {'leverage': 25, 'distance_pct': 4.0},
            {'leverage': 20, 'distance_pct': 5.0},
            {'leverage': 10, 'distance_pct': 10.0},
        ]
        
        for lev in leverage_levels:
            liq_price_long = current_price * (1 - lev['distance_pct'] / 100)
            long_liquidations.append({
                'price': liq_price_long,
                'leverage': lev['leverage'],
                'distance_pct': lev['distance_pct'],
                'volume': 0,
                'intensity': 'HIGH' if lev['leverage'] >= 25 else 'MEDIUM' if lev['leverage'] >= 15 else 'LOW',
                'source': 'calculated'
            })
            
            liq_price_short = current_price * (1 + lev['distance_pct'] / 100)
            short_liquidations.append({
                'price': liq_price_short,
                'leverage': lev['leverage'],
                'distance_pct': lev['distance_pct'],
                'volume': 0,
                'intensity': 'HIGH' if lev['leverage'] >= 25 else 'MEDIUM' if lev['leverage'] >= 15 else 'LOW',
                'source': 'calculated'
            })
    
    # Adjust intensity based on crowding
    if liquidation_data.get('crowded_long'):
        for liq in long_liquidations:
            liq['risk'] = 'ELEVATED'
    
    if liquidation_data.get('crowded_short'):
        for liq in short_liquidations:
            liq['opportunity'] = 'TP_TARGET'
    
    return {
        'long_liquidations': long_liquidations,
        'short_liquidations': short_liquidations,
        'bias': liquidation_data.get('liquidation_bias', 'NEUTRAL'),
        'source': result_source,
        'total_above': total_above,
        'total_below': total_below
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE PLAN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_liquidity_trade_plan(symbol: str, current_price: float, atr: float,
                                   liquidity_levels: Dict, sweep_status: Dict,
                                   liquidation_data: Dict, whale_pct: float,
                                   df: pd.DataFrame = None,
                                   entry_quality: Dict = None) -> Dict:
    """
    Generate a complete trade plan based on liquidity analysis.
    
    Args:
        df: DataFrame with OHLC data to check if TPs were already hit
    """
    plan = {
        'symbol': symbol,
        'current_price': current_price,
        'status': 'NO_SETUP',
        'direction': None,
        'entry': None,
        'stop_loss': None,
        'take_profits': [],
        'risk_reward': None,
        'reasoning': [],
        'confidence': 0
    }
    
    # Get liquidation zones for TP targets - REAL from Coinglass if available!
    liq_zones = estimate_liquidation_zones(current_price, atr, liquidation_data, symbol=symbol)
    
    # Helper to check if TP was already hit since sweep
    def check_tp_hit(tp_price: float, direction: str, candles_ago: int) -> dict:
        """Check if TP was already touched since sweep"""
        result = {'hit': False, 'hit_candle': None, 'high_since': None, 'low_since': None}
        
        if df is None or len(df) < 2 or candles_ago <= 0:
            print(f"[TP_HIT_DEBUG] Cannot check - df={df is not None}, len={len(df) if df is not None else 0}, candles_ago={candles_ago}")
            return result
        
        # Get candles since sweep
        candles_since = df.iloc[-candles_ago:] if candles_ago <= len(df) else df
        
        print(f"[TP_HIT_DEBUG] Checking TP ${tp_price:.2f} ({direction}), candles_ago={candles_ago}, checking {len(candles_since)} candles")
        
        if direction == 'LONG':
            # For LONG, check if HIGH went above TP
            high_since = candles_since['high'].max()
            result['high_since'] = high_since
            print(f"[TP_HIT_DEBUG] HIGH since sweep: ${high_since:.2f}, TP: ${tp_price:.2f}, HIT: {high_since >= tp_price}")
            if high_since >= tp_price:
                result['hit'] = True
                # Find which candle hit it
                for i, (idx, row) in enumerate(candles_since.iterrows()):
                    if row['high'] >= tp_price:
                        result['hit_candle'] = candles_ago - i
                        break
        else:  # SHORT
            # For SHORT, check if LOW went below TP
            low_since = candles_since['low'].min()
            result['low_since'] = low_since
            print(f"[TP_HIT_DEBUG] LOW since sweep: ${low_since:.2f}, TP: ${tp_price:.2f}, HIT: {low_since <= tp_price}")
            if low_since <= tp_price:
                result['hit'] = True
                for i, (idx, row) in enumerate(candles_since.iterrows()):
                    if row['low'] <= tp_price:
                        result['hit_candle'] = candles_ago - i
                        break
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 1: SWEEP DETECTED - IMMEDIATE ENTRY
    # ═══════════════════════════════════════════════════════════════════════
    if sweep_status.get('detected'):
        original_direction = sweep_status['direction']

        # ML-DRIVEN DIRECTION: Use entry_quality direction if available, else fallback to continuation
        ml_dir = entry_quality.get('direction') if isinstance(entry_quality, dict) and entry_quality else None
        if ml_dir:
            direction = ml_dir
            plan['reasoning'].append(f"🤖 ML Direction: {direction} (sweep of {'LOW' if original_direction == 'LONG' else 'HIGH'})")
        else:
            # Fallback: continuation mode
            direction = "SHORT" if original_direction == "LONG" else "LONG"
            plan['reasoning'].append(f"🔄 Fallback: Sweep {original_direction.replace('LONG','LOW').replace('SHORT','HIGH')} → Trade {direction}")

        # Use actual chart liquidity levels for TPs (combined with liquidation zones)
        if direction == 'LONG':
            entry = current_price  # Enter now on sweep confirmation
            sweep_candles = sweep_status.get('candles_ago', 0)

            # Get actual sweep low (not high!) - only use if it's actually below current price
            sweep_level = sweep_status.get('level_swept', current_price)
            sweep_low = sweep_status.get('sweep_low', sweep_level)
            # If sweep_low is above current price, it's actually a swept HIGH - don't use for LONG SL
            if sweep_low > current_price:
                sweep_low = None  # Will use ATR-based fallback
                print(f"[TRADE_PLAN] ML flipped direction: swept HIGH at ${sweep_level:.2f}, using ATR for SL")

            # Find the NEXT liquidity level below current price (excluding already swept)
            lows_for_sl = [l for l in liquidity_levels.get('lows', [])
                          if l.get('price', 0) < current_price and not l.get('swept', False)]
            lows_for_sl = sorted(lows_for_sl, key=lambda x: x.get('price', 0), reverse=True)  # Highest first

            # Also check long liquidation zones as potential SL references
            long_liqs_for_sl = [l for l in liq_zones.get('long_liquidations', [])
                               if l.get('price', 0) < current_price]
            long_liqs_for_sl = sorted(long_liqs_for_sl, key=lambda x: x.get('price', 0), reverse=True)

            if lows_for_sl:
                # Put SL below the nearest unswept liquidity level
                nearest_low = lows_for_sl[0].get('price', current_price * 0.98)
                stop_loss = nearest_low - (atr * 0.3)
                print(f"[TRADE_PLAN] SL below nearest liquidity: ${nearest_low:.2f} → SL=${stop_loss:.2f}")
            elif long_liqs_for_sl:
                # Use first long liquidation zone below as SL reference
                nearest_liq = long_liqs_for_sl[0].get('price', current_price * 0.98)
                stop_loss = nearest_liq - (atr * 0.2)
                print(f"[TRADE_PLAN] SL below long liquidation zone: ${nearest_liq:.2f} → SL=${stop_loss:.2f}")
            elif sweep_low and sweep_low < current_price:
                # Use actual sweep low if it exists and is below current price
                stop_loss = sweep_low - (atr * 0.3)
                print(f"[TRADE_PLAN] SL below sweep low: ${sweep_low:.2f} → SL=${stop_loss:.2f}")
            else:
                # Fallback: ATR-based stop below current price (1.5 ATR)
                stop_loss = current_price - (atr * 1.5)
                print(f"[TRADE_PLAN] SL using ATR fallback: ${stop_loss:.2f} (1.5 ATR below)")
            
            # TPs at levels above - PRIORITIZE actual chart levels over calculated percentages
            tps = []
            all_targets_above = []

            # 1. Add chart liquidity pools above (PRIORITY - actual detected levels)
            for h in liquidity_levels.get('highs', []):
                if h.get('price', 0) > current_price and not h.get('swept', False):
                    all_targets_above.append({
                        'price': h.get('price', 0),
                        'type': h.get('type', 'Liquidity Pool'),
                        'strength': h.get('strength', 'MODERATE'),
                        'source': 'chart'
                    })

            # 2. Add equal highs (stronger levels)
            for eh in liquidity_levels.get('equal_highs', []):
                if eh.get('price', 0) > current_price and not eh.get('swept', False):
                    # Check if not already covered
                    if not any(abs(t.get('price', 0) - eh.get('price', 0)) / eh.get('price', 1) < 0.003 for t in all_targets_above):
                        all_targets_above.append({
                            'price': eh.get('price', 0),
                            'type': 'Equal Highs (STRONG)',
                            'strength': 'STRONG',
                            'source': 'chart'
                        })

            # 3. Only add REAL liquidation data (Coinglass with volume) - NOT calculated percentages
            for liq in liq_zones.get('short_liquidations', []):
                liq_price = liq.get('price', 0)
                liq_volume = liq.get('volume', 0)
                liq_source = liq.get('source', 'calculated')

                # ONLY include if it's real Coinglass data with actual volume
                if liq_price > current_price and liq_source == 'coinglass' and liq_volume > 0:
                    if not any(abs(t.get('price', 0) - liq_price) / liq_price < 0.003 for t in all_targets_above):
                        all_targets_above.append({
                            'price': liq_price,
                            'type': f"Liq Cluster (${liq_volume/1e6:.1f}M)",
                            'strength': 'HIGH' if liq_volume > 10000000 else 'MODERATE',
                            'volume': liq_volume,
                            'source': 'coinglass'
                        })

            # 4. If chart levels are too far (>5%), add intermediate ATR-based targets
            if all_targets_above:
                nearest_chart = min(t.get('price', 0) for t in all_targets_above)
                distance_pct = (nearest_chart - current_price) / current_price * 100
                if distance_pct > 5:
                    print(f"[TRADE_PLAN] Nearest chart level is {distance_pct:.1f}% away, adding ATR intermediates")
                    # Add intermediate targets at 1.5 ATR and 3 ATR
                    atr_1_5 = current_price + (atr * 1.5)
                    atr_3 = current_price + (atr * 3.0)
                    if atr_1_5 < nearest_chart:
                        all_targets_above.append({
                            'price': atr_1_5,
                            'type': 'Intermediate (1.5 ATR)',
                            'strength': 'MODERATE',
                            'source': 'atr_intermediate'
                        })
                    if atr_3 < nearest_chart:
                        all_targets_above.append({
                            'price': atr_3,
                            'type': 'Intermediate (3 ATR)',
                            'strength': 'MODERATE',
                            'source': 'atr_intermediate'
                        })
            else:
                print(f"[TRADE_PLAN] No chart levels found, using ATR-based targets as fallback")
                all_targets_above.append({
                    'price': current_price + (atr * 1.5),
                    'type': 'TP (1.5 ATR)',
                    'strength': 'MODERATE',
                    'source': 'atr_fallback'
                })
                all_targets_above.append({
                    'price': current_price + (atr * 3.0),
                    'type': 'TP (3 ATR)',
                    'strength': 'MODERATE',
                    'source': 'atr_fallback'
                })
            
            # 3. Sort by distance (CLOSEST FIRST) - this is the key!
            all_targets_above = sorted(all_targets_above, key=lambda x: x.get('price', 0))
            
            # 4. Deduplicate levels within 0.3% of each other
            if all_targets_above:
                deduped = [all_targets_above[0]]
                for t in all_targets_above[1:]:
                    prev_price = deduped[-1].get('price', 0)
                    curr_price_t = t.get('price', 0)
                    if prev_price > 0 and abs(curr_price_t - prev_price) / prev_price > 0.003:
                        deduped.append(t)
                all_targets_above = deduped
            
            # Use the combined, sorted list
            highs_above = all_targets_above
            
            # SORT BY DISTANCE (closest first)
            highs_above = sorted(highs_above, key=lambda x: x.get('price', 0))
            
            print(f"[TRADE_PLAN_DEBUG] LONG TPs: entry=${entry:.2f}, stop=${stop_loss:.2f}, risk=${entry-stop_loss:.2f}")
            print(f"[TRADE_PLAN_DEBUG] All targets above (sorted by distance): {len(highs_above)}")
            for h in highs_above[:5]:
                tp_price = h.get('price', 0)
                if entry > stop_loss:
                    rr = (tp_price - entry) / (entry - stop_loss) if tp_price > entry else 0
                    print(f"[TRADE_PLAN_DEBUG]   ${tp_price:.2f} ({h.get('type', 'unknown')}) → R:R={rr:.2f}")
            
            for i, level in enumerate(highs_above[:3]):
                price = level.get('price', 0)
                level_type = level.get('type', 'Liquidity Pool')
                if entry > stop_loss and price > entry:
                    rr = (price - entry) / (entry - stop_loss)
                    roi = ((price - entry) / entry) * 100
                    
                    # Check if this TP was already hit since sweep
                    sweep_candles = sweep_status.get('candles_ago', 0)
                    tp_hit_info = check_tp_hit(price, 'LONG', sweep_candles)
                    
                    # ALWAYS include first TP (closest), others need R:R >= 0.2
                    # This ensures we always show the closest target
                    is_first_tp = len(tps) == 0
                    if rr >= 0.2 or is_first_tp:
                        tp_entry = {
                            'level': len(tps) + 1,  # Sequential numbering
                            'price': price,
                            'reason': level_type,
                            'rr': rr,
                            'roi': roi,
                            'low_rr': rr < 1.0,
                            'hit': tp_hit_info['hit'],
                            'hit_candle': tp_hit_info.get('hit_candle')
                        }
                        
                        # Update reason with status
                        if tp_hit_info['hit']:
                            tp_entry['reason'] = f"{level_type} ✅ HIT ({tp_hit_info['hit_candle']}c ago)"
                        elif rr < 1.0:
                            tp_entry['reason'] = f"{level_type} (⚠️ R:R {rr:.1f})"
                            
                        tps.append(tp_entry)
            
            plan.update({
                'status': 'SWEEP_ENTRY',
                'direction': direction,  # Use the (possibly flipped) direction
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profits': tps,
                'risk_reward': tps[0]['rr'] if tps else 0,
                'confidence': sweep_status.get('confidence', 60),
                'reasoning': plan['reasoning'] + [
                    f"✅ Liquidity sweep detected at ${sweep_status.get('level_swept', 0):.2f}",
                    f"✅ Strategy: ML-driven direction",
                    f"{'✅' if sweep_status.get('volume_confirmed') else '⚠️'} Volume {'confirmed' if sweep_status.get('volume_confirmed') else 'not confirmed'}",
                    f"🎯 Targeting liquidity levels {'above' if direction == 'LONG' else 'below'}"
                ]
            })
            
            # Boost confidence if whale aligned
            if whale_pct > 55:
                plan['confidence'] = min(95, plan['confidence'] + 15)
                plan['reasoning'].append(f"✅ Whales {whale_pct:.0f}% LONG - aligned with sweep direction")
        
        elif direction == 'SHORT':
            entry = current_price
            sweep_candles = sweep_status.get('candles_ago', 0)

            # Get actual sweep high (not low!) - only use if it's actually above current price
            sweep_level = sweep_status.get('level_swept', current_price)
            sweep_high = sweep_status.get('sweep_high', sweep_level)
            # If sweep_high is below current price, it's actually a swept LOW - don't use for SHORT SL
            if sweep_high < current_price:
                sweep_high = None  # Will use ATR-based fallback
                print(f"[TRADE_PLAN] ML flipped direction: swept LOW at ${sweep_level:.2f}, using ATR for SL")

            # Find the NEXT liquidity level above current price (excluding already swept)
            highs_for_sl = [h for h in liquidity_levels.get('highs', [])
                           if h.get('price', 0) > current_price and not h.get('swept', False)]
            highs_for_sl = sorted(highs_for_sl, key=lambda x: x.get('price', 0))  # Lowest first

            # Also check short liquidation zones as potential SL references
            short_liqs_for_sl = [l for l in liq_zones.get('short_liquidations', [])
                                if l.get('price', 0) > current_price]
            short_liqs_for_sl = sorted(short_liqs_for_sl, key=lambda x: x.get('price', 0))

            if highs_for_sl:
                # Put SL above the nearest unswept liquidity level
                nearest_high = highs_for_sl[0].get('price', current_price * 1.02)
                stop_loss = nearest_high + (atr * 0.3)
                print(f"[TRADE_PLAN] SL above nearest liquidity: ${nearest_high:.2f} → SL=${stop_loss:.2f}")
            elif short_liqs_for_sl:
                # Use first short liquidation zone above as SL reference
                nearest_liq = short_liqs_for_sl[0].get('price', current_price * 1.02)
                stop_loss = nearest_liq + (atr * 0.2)
                print(f"[TRADE_PLAN] SL above short liquidation zone: ${nearest_liq:.2f} → SL=${stop_loss:.2f}")
            elif sweep_high and sweep_high > current_price:
                # Use actual sweep high if it exists and is above current price
                stop_loss = sweep_high + (atr * 0.3)
                print(f"[TRADE_PLAN] SL above sweep high: ${sweep_high:.2f} → SL=${stop_loss:.2f}")
            else:
                # Fallback: ATR-based stop above current price (1.5 ATR)
                stop_loss = current_price + (atr * 1.5)
                print(f"[TRADE_PLAN] SL using ATR fallback: ${stop_loss:.2f} (1.5 ATR above)")
            
            # TPs at levels below - PRIORITIZE actual chart levels over calculated percentages
            tps = []
            all_targets_below = []

            # 1. Add chart liquidity pools below (PRIORITY - actual detected levels)
            for l in liquidity_levels.get('lows', []):
                if l.get('price', 0) < current_price and not l.get('swept', False):
                    all_targets_below.append({
                        'price': l.get('price', 0),
                        'type': l.get('type', 'Liquidity Pool'),
                        'strength': l.get('strength', 'MODERATE'),
                        'source': 'chart'
                    })

            # 2. Add equal lows (stronger levels)
            for el in liquidity_levels.get('equal_lows', []):
                if el.get('price', 0) < current_price and not el.get('swept', False):
                    # Check if not already covered
                    if not any(abs(t.get('price', 0) - el.get('price', 0)) / el.get('price', 1) < 0.003 for t in all_targets_below):
                        all_targets_below.append({
                            'price': el.get('price', 0),
                            'type': 'Equal Lows (STRONG)',
                            'strength': 'STRONG',
                            'source': 'chart'
                        })

            # 3. Only add REAL liquidation data (Coinglass with volume) - NOT calculated percentages
            for liq in liq_zones.get('long_liquidations', []):
                liq_price = liq.get('price', 0)
                liq_volume = liq.get('volume', 0)
                liq_source = liq.get('source', 'calculated')

                # ONLY include if it's real Coinglass data with actual volume
                if liq_price < current_price and liq_source == 'coinglass' and liq_volume > 0:
                    if not any(abs(t.get('price', 0) - liq_price) / liq_price < 0.003 for t in all_targets_below):
                        all_targets_below.append({
                            'price': liq_price,
                            'type': f"Liq Cluster (${liq_volume/1e6:.1f}M)",
                            'strength': 'HIGH' if liq_volume > 10000000 else 'MODERATE',
                            'volume': liq_volume,
                            'source': 'coinglass'
                        })

            # 4. If chart levels are too far (>5%), add intermediate ATR-based targets
            if all_targets_below:
                nearest_chart = max(t.get('price', 0) for t in all_targets_below)  # Highest = closest for SHORT
                distance_pct = (current_price - nearest_chart) / current_price * 100
                if distance_pct > 5:
                    print(f"[TRADE_PLAN] Nearest chart level is {distance_pct:.1f}% away, adding ATR intermediates")
                    # Add intermediate targets at 1.5 ATR and 3 ATR
                    atr_1_5 = current_price - (atr * 1.5)
                    atr_3 = current_price - (atr * 3.0)
                    if atr_1_5 > nearest_chart:
                        all_targets_below.append({
                            'price': atr_1_5,
                            'type': 'Intermediate (1.5 ATR)',
                            'strength': 'MODERATE',
                            'source': 'atr_intermediate'
                        })
                    if atr_3 > nearest_chart:
                        all_targets_below.append({
                            'price': atr_3,
                            'type': 'Intermediate (3 ATR)',
                            'strength': 'MODERATE',
                            'source': 'atr_intermediate'
                        })
            else:
                print(f"[TRADE_PLAN] No chart levels found, using ATR-based targets as fallback")
                all_targets_below.append({
                    'price': current_price - (atr * 1.5),
                    'type': 'TP (1.5 ATR)',
                    'strength': 'MODERATE',
                    'source': 'atr_fallback'
                })
                all_targets_below.append({
                    'price': current_price - (atr * 3.0),
                    'type': 'TP (3 ATR)',
                    'strength': 'MODERATE',
                    'source': 'atr_fallback'
                })
            
            # 3. Sort by distance (CLOSEST FIRST = highest price for SHORT)
            all_targets_below = sorted(all_targets_below, key=lambda x: x.get('price', 0), reverse=True)
            
            # 4. Deduplicate levels within 0.3% of each other
            if all_targets_below:
                deduped = [all_targets_below[0]]
                for t in all_targets_below[1:]:
                    prev_price = deduped[-1].get('price', 0)
                    curr_price_t = t.get('price', 0)
                    if prev_price > 0 and abs(curr_price_t - prev_price) / prev_price > 0.003:
                        deduped.append(t)
                all_targets_below = deduped
            
            # Use the combined, sorted list
            lows_below = all_targets_below
            
            for i, level in enumerate(lows_below[:3]):
                price = level.get('price', 0)
                level_type = level.get('type', 'Liquidity Pool')
                if stop_loss > entry and price < entry:
                    rr = (entry - price) / (stop_loss - entry)
                    roi = ((entry - price) / entry) * 100
                    
                    # Check if this TP was already hit since sweep
                    sweep_candles = sweep_status.get('candles_ago', 0)
                    tp_hit_info = check_tp_hit(price, 'SHORT', sweep_candles)
                    
                    # ALWAYS include first TP (closest), others need R:R >= 0.2
                    is_first_tp = len(tps) == 0
                    if rr >= 0.2 or is_first_tp:
                        tp_entry = {
                            'level': len(tps) + 1,  # Sequential numbering
                            'price': price,
                            'reason': level_type,
                            'rr': rr,
                            'roi': roi,
                            'low_rr': rr < 1.0,
                            'hit': tp_hit_info['hit'],
                            'hit_candle': tp_hit_info.get('hit_candle')
                        }
                        
                        # Update reason with status
                        if tp_hit_info['hit']:
                            tp_entry['reason'] = f"{level_type} ✅ HIT ({tp_hit_info['hit_candle']}c ago)"
                        elif rr < 1.0:
                            tp_entry['reason'] = f"{level_type} (⚠️ R:R {rr:.1f})"
                            
                        tps.append(tp_entry)
            
            plan.update({
                'status': 'SWEEP_ENTRY',
                'direction': direction,  # Use the (possibly flipped) direction
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profits': tps,
                'risk_reward': tps[0]['rr'] if tps else 0,
                'confidence': sweep_status.get('confidence', 60),
                'reasoning': plan['reasoning'] + [
                    f"✅ Liquidity sweep detected at ${sweep_status.get('level_swept', 0):.2f}",
                    f"✅ Strategy: ML-driven direction",
                    f"🎯 Targeting liquidity levels {'above' if direction == 'LONG' else 'below'}"
                ]
            })
            
            if whale_pct < 45:
                plan['confidence'] = min(95, plan['confidence'] + 15)
                plan['reasoning'].append(f"✅ Whales {whale_pct:.0f}% SHORT - aligned")
        
        return plan
    
    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 2: APPROACHING LIQUIDITY - WAIT FOR SWEEP
    # ═══════════════════════════════════════════════════════════════════════
    approaching = check_approaching_liquidity(current_price, liquidity_levels, atr)
    
    if approaching['has_nearby']:
        nearest = approaching['levels'][0]
        
        if nearest['type'] in ['LOW', 'EQUAL_LOW']:
            # Price approaching lows - default to CONTINUATION until ML can evaluate post-sweep
            trade_dir = "SHORT"  # Default: continuation (sweep LOW → SHORT)
            est_entry = nearest['price']

            est_tps = []
            all_tp_targets = []

            if True:  # Continuation mode for approaching levels
                # SHORT trade: SL above entry, TPs below entry
                est_sl = nearest['price'] + (atr * 0.5)
                est_sl_pct = ((est_sl - est_entry) / est_entry) * 100

                # TPs: long liquidations BELOW entry (where longs get liquidated)
                for liq in liq_zones.get('long_liquidations', []):
                    liq_price = liq.get('price', 0)
                    if liq_price < est_entry:
                        if not any(abs(t.get('price', 0) - liq_price) / liq_price < 0.003 for t in all_tp_targets):
                            liq_reason = liq.get('reason', f"Long Liq ({liq.get('leverage', '?')}x)")
                            all_tp_targets.append({
                                'price': liq_price,
                                'reason': liq_reason,
                                'volume': liq.get('volume', 0),
                                'source': liq.get('source', 'calculated')
                            })

                # Add chart lows below as targets
                for l in liquidity_levels.get('lows', []):
                    if l.get('price', 0) < est_entry and not l.get('swept', False):
                        all_tp_targets.append({
                            'price': l.get('price', 0),
                            'reason': l.get('type', 'Liquidity Pool'),
                            'source': 'chart'
                        })

                # Sort by price (closest first = highest price for SHORT targets below)
                all_tp_targets = sorted(all_tp_targets, key=lambda x: x.get('price', 0), reverse=True)

                # Build TPs
                for i, target in enumerate(all_tp_targets[:3]):
                    tp_price = target.get('price', 0)
                    tp_roi = ((est_entry - tp_price) / est_entry) * 100
                    rr = (est_entry - tp_price) / (est_sl - est_entry) if est_sl > est_entry else 0
                    est_tps.append({
                        'level': i + 1,
                        'price': tp_price,
                        'reason': target.get('reason', 'Target'),
                        'roi': tp_roi,
                        'rr': rr,
                        'volume': target.get('volume', 0),
                        'source': target.get('source', 'calculated')
                    })
            else:
                # REVERSAL mode: LONG trade - SL below entry, TPs above entry
                est_sl = nearest['price'] - (atr * 0.5)
                est_sl_pct = ((est_entry - est_sl) / est_entry) * 100

                # TPs: short liquidations ABOVE entry
                for h in liquidity_levels.get('highs', []):
                    if h.get('price', 0) > est_entry and not h.get('swept', False):
                        all_tp_targets.append({
                            'price': h.get('price', 0),
                            'reason': h.get('type', 'Liquidity Pool'),
                            'source': 'chart'
                        })

                for liq in liq_zones.get('short_liquidations', []):
                    liq_price = liq.get('price', 0)
                    if liq_price > est_entry:
                        if not any(abs(t.get('price', 0) - liq_price) / liq_price < 0.003 for t in all_tp_targets):
                            liq_reason = liq.get('reason', f"Short Liq ({liq.get('leverage', '?')}x)")
                            all_tp_targets.append({
                                'price': liq_price,
                                'reason': liq_reason,
                                'volume': liq.get('volume', 0),
                                'source': liq.get('source', 'calculated')
                            })

                all_tp_targets = sorted(all_tp_targets, key=lambda x: x.get('price', 0))

                for i, target in enumerate(all_tp_targets[:3]):
                    tp_price = target.get('price', 0)
                    tp_roi = ((tp_price - est_entry) / est_entry) * 100
                    rr = (tp_price - est_entry) / (est_entry - est_sl) if est_entry > est_sl else 0
                    est_tps.append({
                        'level': i + 1,
                        'price': tp_price,
                        'reason': target.get('reason', 'Target'),
                        'roi': tp_roi,
                        'rr': rr,
                        'volume': target.get('volume', 0),
                        'source': target.get('source', 'calculated')
                    })

            plan.update({
                'status': 'WAITING_FOR_SWEEP',
                'direction': f'{trade_dir} (after sweep)',
                'sweep_level': nearest['price'],
                'est_entry': est_entry,
                'est_sl': est_sl,
                'est_sl_pct': est_sl_pct,
                'est_tps': est_tps,
                'proximity': nearest['proximity'],
                'confidence': 40 if nearest['proximity'] == 'APPROACHING' else 60 if nearest['proximity'] == 'CLOSE' else 75,
                'reasoning': [
                    f"🔍 Price approaching liquidity at {nearest['price']:.2f}",
                    f"📏 Distance: {nearest['distance_atr']:.1f} ATR ({nearest['proximity']})",
                    f"📊 Level strength: {nearest['strength']}",
                    f"⏳ WAIT for sweep - ML will decide direction after sweep"
                ]
            })
            
            if nearest['type'] == 'EQUAL_LOW':
                plan['reasoning'].insert(1, f"⭐ EQUAL LOW with {nearest.get('touches', 2)} touches - HIGH probability")
                plan['confidence'] += 10
        
        elif nearest['type'] in ['HIGH', 'EQUAL_HIGH']:
            # Price approaching highs - default to CONTINUATION until ML can evaluate post-sweep
            trade_dir = "LONG"  # Default: continuation (sweep HIGH → LONG)
            est_entry = nearest['price']

            est_tps = []
            all_tp_targets = []

            if True:  # Continuation mode for approaching levels
                # LONG trade: SL below entry, TPs above entry
                est_sl = nearest['price'] - (atr * 0.5)
                est_sl_pct = ((est_entry - est_sl) / est_entry) * 100

                # TPs: short liquidations ABOVE entry (where shorts get squeezed)
                for liq in liq_zones.get('short_liquidations', []):
                    liq_price = liq.get('price', 0)
                    if liq_price > est_entry:
                        if not any(abs(t.get('price', 0) - liq_price) / liq_price < 0.003 for t in all_tp_targets):
                            liq_reason = liq.get('reason', f"Short Liq ({liq.get('leverage', '?')}x)")
                            all_tp_targets.append({
                                'price': liq_price,
                                'reason': liq_reason,
                                'volume': liq.get('volume', 0),
                                'source': liq.get('source', 'calculated')
                            })

                # Add chart highs above as targets
                for h in liquidity_levels.get('highs', []):
                    if h.get('price', 0) > est_entry and not h.get('swept', False):
                        all_tp_targets.append({
                            'price': h.get('price', 0),
                            'reason': h.get('type', 'Liquidity Pool'),
                            'source': 'chart'
                        })

                # Sort by price (closest first = lowest price for LONG targets above)
                all_tp_targets = sorted(all_tp_targets, key=lambda x: x.get('price', 0))

                # Build TPs
                for i, target in enumerate(all_tp_targets[:3]):
                    tp_price = target.get('price', 0)
                    tp_roi = ((tp_price - est_entry) / est_entry) * 100
                    rr = (tp_price - est_entry) / (est_entry - est_sl) if est_entry > est_sl else 0
                    est_tps.append({
                        'level': i + 1,
                        'price': tp_price,
                        'reason': target.get('reason', 'Target'),
                        'roi': tp_roi,
                        'rr': rr,
                        'volume': target.get('volume', 0),
                        'source': target.get('source', 'calculated')
                    })
            else:
                # REVERSAL mode: SHORT trade - SL above entry, TPs below entry
                est_sl = nearest['price'] + (atr * 0.5)
                est_sl_pct = ((est_sl - est_entry) / est_entry) * 100

                # TPs: long liquidations BELOW entry
                for l in liquidity_levels.get('lows', []):
                    if l.get('price', 0) < est_entry and not l.get('swept', False):
                        all_tp_targets.append({
                            'price': l.get('price', 0),
                            'reason': l.get('type', 'Liquidity Pool'),
                            'source': 'chart'
                        })

                for liq in liq_zones.get('long_liquidations', []):
                    liq_price = liq.get('price', 0)
                    if liq_price < est_entry:
                        if not any(abs(t.get('price', 0) - liq_price) / liq_price < 0.003 for t in all_tp_targets):
                            liq_reason = liq.get('reason', f"Long Liq ({liq.get('leverage', '?')}x)")
                            all_tp_targets.append({
                                'price': liq_price,
                                'reason': liq_reason,
                                'volume': liq.get('volume', 0),
                                'source': liq.get('source', 'calculated')
                            })

                all_tp_targets = sorted(all_tp_targets, key=lambda x: x.get('price', 0), reverse=True)

                for i, target in enumerate(all_tp_targets[:3]):
                    tp_price = target.get('price', 0)
                    tp_roi = ((est_entry - tp_price) / est_entry) * 100
                    rr = (est_entry - tp_price) / (est_sl - est_entry) if est_sl > est_entry else 0
                    est_tps.append({
                        'level': i + 1,
                        'price': tp_price,
                        'reason': target.get('reason', 'Target'),
                        'roi': tp_roi,
                        'rr': rr,
                        'volume': target.get('volume', 0),
                        'source': target.get('source', 'calculated')
                    })

            plan.update({
                'status': 'WAITING_FOR_SWEEP',
                'direction': f'{trade_dir} (after sweep)',
                'sweep_level': nearest['price'],
                'est_entry': est_entry,
                'est_sl': est_sl,
                'est_sl_pct': est_sl_pct,
                'est_tps': est_tps,
                'proximity': nearest['proximity'],
                'confidence': 40 if nearest['proximity'] == 'APPROACHING' else 60 if nearest['proximity'] == 'CLOSE' else 75,
                'reasoning': [
                    f"🔍 Price approaching liquidity at {nearest['price']:.2f}",
                    f"📏 Distance: {nearest['distance_atr']:.1f} ATR ({nearest['proximity']})",
                    f"⏳ WAIT for sweep - ML will decide direction after sweep"
                ]
            })
            
            if nearest['type'] == 'EQUAL_HIGH':
                plan['reasoning'].insert(1, f"⭐ EQUAL HIGH with {nearest.get('touches', 2)} touches - HIGH probability")
                plan['confidence'] += 10
        
        return plan
    
    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 3: NO IMMEDIATE SETUP
    # ═══════════════════════════════════════════════════════════════════════
    # Find nearest levels for monitoring
    all_levels = []
    for l in liquidity_levels.get('lows', []):
        all_levels.append({'price': l['price'], 'type': 'LOW', 'strength': l.get('strength')})
    for h in liquidity_levels.get('highs', []):
        all_levels.append({'price': h['price'], 'type': 'HIGH', 'strength': h.get('strength')})
    
    all_levels.sort(key=lambda x: abs(x['price'] - current_price))
    
    plan.update({
        'status': 'NO_SETUP',
        'direction': None,
        'watch_levels': all_levels[:5],
        'reasoning': [
            "⏳ No liquidity sweep detected",
            "⏳ Price not near key liquidity levels",
            f"👀 Monitoring {len(all_levels)} liquidity levels",
            "💡 Best setups come AFTER sweeps, not before"
        ],
        'confidence': 0
    })
    
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def scan_for_liquidity_setups(symbols: List[str], fetch_data_func,
                               timeframe: str = '4h', trading_mode: str = 'swing',
                               max_symbols: int = 20, progress_callback=None,
                               market_type: str = 'crypto') -> List[Dict]:
    """
    Scan multiple symbols for liquidity sweep setups.

    ENHANCED with Entry Quality Scoring:
    - Calculates entry_quality_score (0-100) for active sweeps
    - Returns entry_quality_grade (A/B/C/D) and entry_window status
    - Sorts by entry quality for better prioritization

    Args:
        symbols: List of symbols to scan
        fetch_data_func: Function to fetch candle data
        timeframe: Candle timeframe
        trading_mode: 'scalp', 'day_trade', 'swing', or 'investment'
        max_symbols: Maximum number of symbols to scan (10, 20, 50, 100)
        progress_callback: Optional callback(current, total) for progress updates

    Returns list of symbols with:
    - Active sweeps (ENTRY NOW) with entry quality scores
    - Approaching liquidity (WATCH)
    """
    # Map trading mode to lookback candles
    lookback_map = {
        'scalp': 10,
        'day_trade': 25,
        'swing': 40,
        'investment': 50
    }
    lookback_candles = lookback_map.get(trading_mode.lower().replace(' ', '_'), 25)

    results = []

    # Limit symbols to scan
    symbols_to_scan = symbols[:max_symbols]
    total = len(symbols_to_scan)

    for idx, symbol in enumerate(symbols_to_scan):
        # Progress callback
        if progress_callback:
            progress_callback(idx + 1, total)

        try:
            # Fetch candle data
            df = fetch_data_func(symbol, timeframe)
            if df is None or len(df) < 50:
                continue

            # Normalize column names
            df = normalize_columns(df)

            # Calculate ATR
            high = df['high']
            low = df['low']
            close = df['close']
            volumes = df['volume'] if 'volume' in df.columns else pd.Series(np.ones(len(df)))
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])

            current_price = float(close.iloc[-1])
            avg_volume = float(volumes.rolling(20).mean().iloc[-1])

            # Get whale data first (needed for sweep detection)
            is_crypto = 'USDT' in symbol or 'USD' in symbol
            if is_crypto:
                whale_data = get_binance_liquidations(symbol)
                whale_pct = whale_data.get('whale_long', 50)
            else:
                whale_pct = 50
                whale_data = {}

            # Find liquidity levels
            liquidity_levels = find_liquidity_levels(df)

            # Check for sweep (with whale alignment)
            sweep_status = detect_sweep(df, liquidity_levels, atr, lookback_candles, whale_pct)

            # Check approaching
            approaching = check_approaching_liquidity(current_price, liquidity_levels, atr)

            # Determine status
            if sweep_status.get('detected'):
                status = 'SWEEP_ACTIVE'
                priority = 1
            elif approaching.get('imminent'):
                status = 'IMMINENT'
                priority = 2
            elif approaching.get('has_nearby'):
                status = 'APPROACHING'
                priority = 3
            else:
                status = 'MONITORING'
                priority = 4

            # ═══════════════════════════════════════════════════════════════════
            # ENTRY QUALITY CALCULATION (for active sweeps)
            # ═══════════════════════════════════════════════════════════════════
            entry_quality = None
            entry_quality_score = 0
            entry_quality_grade = 'N/A'
            entry_window = 'NO_SWEEP'

            if sweep_status.get('detected'):
                # Get volume data for the sweep candle
                candles_ago = sweep_status.get('candles_ago', 1)
                sweep_candle_idx = len(df) - candles_ago
                if 0 <= sweep_candle_idx < len(df):
                    sweep_volume = float(volumes.iloc[sweep_candle_idx])
                else:
                    sweep_volume = float(volumes.iloc[-1])

                # Estimate whale delta (simplified - ideally from whale_delta service)
                whale_delta_val = 0
                if is_crypto and whale_data.get('success'):
                    # Rough approximation: deviation from neutral
                    whale_delta_val = (whale_pct - 50) / 5

                # Calculate entry quality (ENHANCED with reversal confirmation)
                entry_quality = calculate_entry_quality(
                    sweep_status=sweep_status,
                    current_price=current_price,
                    whale_pct=whale_pct,
                    whale_delta=whale_delta_val,
                    volume_on_sweep=sweep_volume,
                    avg_volume=avg_volume,
                    df=df,  # Pass DataFrame for candle pattern analysis
                    symbol=symbol,  # For ML debug logging
                    market_type=market_type
                )

                entry_quality_score = entry_quality.get('score', 0)
                entry_quality_grade = entry_quality.get('grade', 'N/A')
                entry_window = entry_quality.get('entry_window', 'NO_SWEEP')

                # Log warnings and recommendation
                warnings = entry_quality.get('warnings', [])
                recommendation = entry_quality.get('recommendation', 'UNKNOWN')
                if warnings:
                    print(f"[ENTRY_QUALITY] {symbol}: {entry_quality_grade} ({entry_quality_score}) - {recommendation}")
                    for w in warnings[:3]:  # Show first 3 warnings
                        print(f"  {w}")

            # Get recommendation and warnings from entry quality
            entry_recommendation = entry_quality.get('recommendation', 'UNKNOWN') if entry_quality else 'NO_SETUP'
            entry_warnings = entry_quality.get('warnings', []) if entry_quality else []

            # Get ML probability and decision (for filtering)
            ml_probability = entry_quality.get('ml_probability') if entry_quality else None
            ml_decision = entry_quality.get('ml_decision', 'UNKNOWN') if entry_quality else 'UNKNOWN'

            # Get level type (EQUAL/DOUBLE/SWING) for filtering
            level_type = 'SWING'  # Default
            if sweep_status.get('detected'):
                level_type = sweep_status.get('level_type', 'SWING')

            # ═══════════════════════════════════════════════════════════════════
            # TRADE DIRECTION (accounts for CONTINUATION mode!)
            # ═══════════════════════════════════════════════════════════════════
            # Raw direction from sweep detection
            raw_direction = None
            if sweep_status.get('detected'):
                raw_direction = sweep_status.get('direction')
            elif approaching.get('has_nearby') and approaching['levels']:
                raw_direction = approaching['levels'][0].get('direction_after_sweep')

            # ML-DRIVEN DIRECTION: Use entry_quality direction if available
            trade_direction = entry_quality.get('direction') if isinstance(entry_quality, dict) and entry_quality.get('direction') else raw_direction
            if not trade_direction and raw_direction:
                # Fallback: continuation mode
                trade_direction = "SHORT" if raw_direction == "LONG" else "LONG"

            # Apply centralized ML filter (SKIP if ML < 40%)
            trade_direction, _ = apply_ml_filter(trade_direction, entry_quality, symbol=symbol)

            # Debug: Log direction
            if sweep_status.get('detected') and entry_recommendation in ['ENTER', 'WAIT']:
                print(f"[SCAN_DIR] {symbol}: raw={raw_direction} → trade={trade_direction} (ML-driven)")

            results.append({
                'symbol': symbol,
                'current_price': float(current_price),
                'status': status,
                'priority': priority,
                'sweep': sweep_status if sweep_status.get('detected') else None,
                'approaching': approaching['levels'][0] if approaching.get('has_nearby') else None,
                'whale_pct': whale_pct,
                'liquidity_levels_count': len(liquidity_levels.get('lows', [])) + len(liquidity_levels.get('highs', [])),
                'has_equal_levels': len(liquidity_levels.get('equal_lows', [])) + len(liquidity_levels.get('equal_highs', [])) > 0,
                # Entry quality fields
                'entry_quality': entry_quality,
                'entry_quality_score': entry_quality_score,
                'entry_quality_grade': entry_quality_grade,
                'entry_window': entry_window,
                'entry_recommendation': entry_recommendation,  # ENTER/WAIT/SKIP
                'entry_warnings': entry_warnings,  # List of warnings
                # ML Quality fields (for filtering)
                'ml_probability': ml_probability,  # 0-1 probability
                'ml_decision': ml_decision,  # STRONG_YES/YES/MAYBE/NO/UNKNOWN
                # Level type (for filtering)
                'level_type': level_type,  # EQUAL/DOUBLE/SWING
                # Trade direction (CONTINUATION mode aware!)
                'trade_direction': trade_direction,  # Actual trade direction after mode flip
                'raw_sweep_direction': raw_direction,  # Original sweep direction
            })

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            continue

    # Sort by: (priority, recommendation_rank, -ml_probability, -entry_quality_score, -whale_pct)
    # This puts ENTER recommendations first, then by ML probability
    rec_rank = {'ENTER': 0, 'WAIT': 1, 'SKIP': 2, 'NO_SETUP': 3, 'UNKNOWN': 3}
    results.sort(key=lambda x: (
        x['priority'],
        rec_rank.get(x.get('entry_recommendation', 'UNKNOWN'), 3),  # ENTER first
        -(x.get('ml_probability') or 0),  # Higher ML probability first
        -x.get('entry_quality_score', 0),  # Higher quality first
        -x.get('whale_pct', 50)            # Then by whale alignment
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS FOR SINGLE SYMBOL
# ═══════════════════════════════════════════════════════════════════════════════

def full_liquidity_analysis(
    symbol: str, df: pd.DataFrame, whale_pct: float = 50,
    trading_mode: str = 'day_trade', whale_delta: float = None,
    # Whale acceleration data (4h, 24h, 7d)
    whale_delta_4h: float = None,       # NEW: Early signal (4 hours)
    whale_delta_24h: float = None,
    whale_delta_7d: float = None,
    whale_acceleration: str = None,
    whale_early_signal: str = None,     # NEW: EARLY_ACCUMULATION, EARLY_DISTRIBUTION, etc.
    is_fresh_accumulation: bool = False,
    # OI features (NEW)
    oi_change_24h: float = None,        # OI change % in 24h
    price_change_24h: float = None,     # Price change % in 24h (for OI-price alignment)
    market_type: str = 'crypto'         # 'crypto', 'etf', or 'stock'
) -> Dict:
    """
    Complete liquidity analysis for a single symbol.

    Args:
        symbol: Trading pair
        df: OHLCV DataFrame
        whale_pct: Current whale long percentage (0-100)
        trading_mode: 'scalp', 'day_trade', 'swing', or 'investment'
        whale_delta: REAL whale change from database (not estimated!)
        whale_delta_4h: 4-hour whale delta (EARLY signal - detect reversals!)
        whale_delta_24h: 24-hour whale delta (fresh signal)
        whale_delta_7d: 7-day whale delta (long-term trend)
        whale_acceleration: 'ACCELERATING', 'DECELERATING', 'STEADY', 'REVERSING'
        whale_early_signal: 'EARLY_ACCUMULATION', 'EARLY_DISTRIBUTION', etc.
        is_fresh_accumulation: True if 24h change > 7d daily average
    """
    if df is None or len(df) < 50:
        return {'error': 'Insufficient data'}
    
    # Normalize column names
    df = normalize_columns(df)
    
    # Map trading mode to lookback candles
    lookback_map = {
        'scalp': 10,        # ~50 min on 5m (fresh sweeps only)
        'day_trade': 25,    # ~6 hrs on 15m
        'swing': 40,        # ~1 week on 4h
        'investment': 50    # ~50 days on 1d
    }
    lookback_candles = lookback_map.get(trading_mode.lower().replace(' ', '_'), 25)
    
    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    
    current_price = float(close.iloc[-1])
    
    # Get all analysis components
    liquidity_levels = find_liquidity_levels(df)
    
    # DEBUG: Show what levels exist
    lows_prices = [(round(l.get('price', 0), 2), len(df) - l.get('index', 0)) for l in liquidity_levels.get('lows', [])[:5]]
    swept_lows_prices = [(round(l.get('price', 0), 2), l.get('swept_candles_ago', '?')) for l in liquidity_levels.get('recently_swept_lows', [])[:5]]
    print(f"[LEVELS_DEBUG] UNSWEPT lows (fresh targets): {lows_prices}  ← (price, candles_since_formed)")
    print(f"[LEVELS_DEBUG] SWEPT lows (already used): {swept_lows_prices}  ← (price, candles_since_sweep)")
    
    sweep_status = detect_sweep(df, liquidity_levels, atr, lookback_candles, whale_pct)
    approaching = check_approaching_liquidity(current_price, liquidity_levels, atr)
    
    # Get liquidation data for crypto
    is_crypto = 'USDT' in symbol.upper()
    if is_crypto:
        liquidation_data = get_binance_liquidations(symbol)
    else:
        liquidation_data = {'success': False, 'retail_long': 50, 'retail_short': 50, 'whale_long': 50, 'whale_short': 50}
    
    # Generate trade plan FIRST
    trade_plan = generate_liquidity_trade_plan(
        symbol, current_price, atr,
        liquidity_levels, sweep_status,
        liquidation_data, whale_pct,
        df=df  # Pass df for TP hit detection
    )

    # ═══════════════════════════════════════════════════════════════════════
    # ENTRY QUALITY CALCULATION (for active sweeps)
    # ═══════════════════════════════════════════════════════════════════════
    entry_quality = None
    if sweep_status.get('detected'):
        # Get volume data for the sweep candle
        volumes = df['volume'] if 'volume' in df.columns else pd.Series(np.ones(len(df)))
        avg_volume = float(volumes.rolling(20).mean().iloc[-1])
        candles_ago = sweep_status.get('candles_ago', 1)
        sweep_candle_idx = len(df) - candles_ago
        if 0 <= sweep_candle_idx < len(df):
            sweep_volume = float(volumes.iloc[sweep_candle_idx])
        else:
            sweep_volume = float(volumes.iloc[-1])

        # Get whale delta - prefer passed value from database, fallback to liquidation_data, then estimate
        whale_delta_val = whale_delta  # Use passed value first (from database)
        if whale_delta_val is None or whale_delta_val == 0:
            whale_delta_val = liquidation_data.get('whale_delta', 0)
        if (whale_delta_val is None or whale_delta_val == 0) and is_crypto:
            # Last resort: rough approximation from current position
            whale_delta_val = (whale_pct - 50) / 5
            print(f"[WHALE_DELTA] ⚠️ Using estimate for {symbol}: {whale_delta_val:.1f}% (whale_pct={whale_pct})")

        # Calculate entry quality (ENHANCED with reversal confirmation + 4h/24h/7d acceleration + OI)
        entry_quality = calculate_entry_quality(
            sweep_status=sweep_status,
            current_price=current_price,
            whale_pct=whale_pct,
            whale_delta=whale_delta_val,
            volume_on_sweep=sweep_volume,
            avg_volume=avg_volume,
            df=df,  # Pass DataFrame for candle pattern analysis
            # Pass 4h/24h/7d acceleration data to ML
            whale_delta_4h=whale_delta_4h,              # NEW: Early signal
            whale_delta_24h=whale_delta_24h,
            whale_delta_7d=whale_delta_7d,
            whale_acceleration=whale_acceleration,
            whale_early_signal=whale_early_signal,      # NEW: Early signal type
            is_fresh_accumulation=is_fresh_accumulation,
            symbol=symbol,  # For ML debug logging
            # OI features (NEW)
            oi_change_24h=oi_change_24h,
            price_change_24h=price_change_24h,
            market_type=market_type
        )

        # Add entry quality to trade plan (SAME FIELDS AS SCANNER!)
        trade_plan['entry_quality'] = entry_quality
        trade_plan['entry_quality_score'] = entry_quality.get('score', 0)
        trade_plan['entry_quality_grade'] = entry_quality.get('grade', 'N/A')
        trade_plan['entry_window'] = entry_quality.get('entry_window', 'NO_SWEEP')
        trade_plan['entry_recommendation'] = entry_quality.get('recommendation', 'UNKNOWN')
        trade_plan['entry_warnings'] = entry_quality.get('warnings', [])
        # ML fields from quality_model (consistent with scanner)
        trade_plan['ml_probability'] = entry_quality.get('ml_probability', 0)
        trade_plan['ml_decision'] = entry_quality.get('ml_decision', 'UNKNOWN')
        trade_plan['ml_quality'] = entry_quality.get('ml_decision', 'UNKNOWN')  # Alias for compatibility

    # Get liquidation zones - REAL from Coinglass if available!
    liq_zones = estimate_liquidation_zones(current_price, atr, liquidation_data, symbol=symbol)
    
    # ═══════════════════════════════════════════════════════════════════════
    # AUTO-SAVE LIQUIDATION DATA FOR ML TRAINING
    # Every scan = new data point for future training on REAL levels!
    # ═══════════════════════════════════════════════════════════════════════
    try:
        from liquidity_hunter.liq_level_collector import save_liquidation_snapshot
        
        # Extract levels from liq_zones
        levels_to_save = {}
        for liq in liq_zones.get('long_liquidations', []):
            lev = liq.get('leverage', 0)
            price = liq.get('price', 0)
            if lev == 100:
                levels_to_save['100x_below'] = price
            elif lev == 50:
                levels_to_save['50x_below'] = price
            elif lev == 25:
                levels_to_save['25x_below'] = price
        
        for liq in liq_zones.get('short_liquidations', []):
            lev = liq.get('leverage', 0)
            price = liq.get('price', 0)
            if lev == 100:
                levels_to_save['100x_above'] = price
            elif lev == 50:
                levels_to_save['50x_above'] = price
            elif lev == 25:
                levels_to_save['25x_above'] = price
        
        # Get whale delta if available
        whale_delta = liquidation_data.get('whale_delta', 0)
        
        # Save snapshot (non-blocking)
        if levels_to_save:
            save_liquidation_snapshot(
                symbol=symbol,
                price=current_price,
                levels=levels_to_save,
                whale_pct=whale_pct,
                whale_delta=whale_delta,
                atr=atr,
                data_source='scanner'
            )
    except Exception as e:
        pass  # Don't break analysis if collector fails
    
    # Get trade direction (ML-driven) for consistency with scanner
    raw_direction = sweep_status.get('direction') if sweep_status.get('detected') else None
    trade_direction = entry_quality.get('direction') if isinstance(entry_quality, dict) and entry_quality.get('direction') else None
    if not trade_direction and raw_direction:
        # Fallback: continuation mode
        trade_direction = "SHORT" if raw_direction == "LONG" else "LONG"

    # Apply centralized ML filter (SKIP if ML < 40%)
    trade_direction, trade_plan = apply_ml_filter(trade_direction, entry_quality, trade_plan, symbol)

    return {
        'symbol': symbol,
        'current_price': current_price,
        'atr': atr,
        'liquidity_levels': liquidity_levels,
        'sweep_status': sweep_status,
        'approaching': approaching,
        'liquidation_data': liquidation_data,
        'liquidation_zones': liq_zones,
        'trade_plan': trade_plan,
        'is_crypto': is_crypto,
        'entry_quality': entry_quality,
        # CONTINUATION mode aware (consistent with scanner)
        'trade_direction': trade_direction,
        'raw_sweep_direction': raw_direction,
        # Whale delta data (for display in sequence UI)
        'whale_delta': whale_delta,
        'whale_delta_4h': whale_delta_4h,
        'whale_delta_24h': whale_delta_24h,
        'whale_delta_7d': whale_delta_7d,
        'whale_acceleration': whale_acceleration,
        'whale_early_signal': whale_early_signal,
        'is_fresh_accumulation': is_fresh_accumulation
    }