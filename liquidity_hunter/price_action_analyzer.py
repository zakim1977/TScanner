"""
Price Action Analyzer for Liquidity Sweeps

Analyzes price behavior after a liquidity sweep to predict the next move.
Uses Smart Money Concepts (SMC) techniques:
- Candle pattern analysis (rejection, engulfing, pin bars)
- Structure analysis (BOS, CHoCH, HH/HL/LH/LL)
- Order Block detection
- Fair Value Gap (FVG) detection
- Volume profile analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def analyze_sweep_reaction(df: pd.DataFrame, sweep_candle_idx: int,
                           sweep_direction: str, atr: float) -> Dict:
    """
    Analyze how price reacted after a liquidity sweep.

    Args:
        df: OHLCV DataFrame
        sweep_candle_idx: Index of the candle that swept liquidity
        sweep_direction: 'LONG' (swept low) or 'SHORT' (swept high)
        atr: Average True Range for normalization

    Returns:
        Dict with reaction analysis and prediction
    """
    # Validate inputs - need at least the sweep candle to exist
    if df is None or len(df) < 20:
        return {'valid': False, 'reason': 'Insufficient data'}

    if sweep_candle_idx < 0 or sweep_candle_idx >= len(df):
        return {'valid': False, 'reason': f'Invalid sweep index: {sweep_candle_idx}'}

    # Get candles
    sweep_candle = df.iloc[sweep_candle_idx]
    candles_after = df.iloc[sweep_candle_idx + 1:] if sweep_candle_idx + 1 < len(df) else pd.DataFrame()
    candles_before = df.iloc[max(0, sweep_candle_idx - 10):sweep_candle_idx]

    # Analyze components
    candle_pattern = analyze_candle_pattern(sweep_candle, candles_after, sweep_direction, atr)
    structure = analyze_structure(df, sweep_candle_idx, sweep_direction, atr)
    order_blocks = detect_order_blocks(df, sweep_candle_idx, sweep_direction, atr)
    fvg = detect_fair_value_gaps(df, sweep_candle_idx, sweep_direction, atr)
    volume_analysis = analyze_volume_profile(df, sweep_candle_idx, sweep_direction)
    momentum = analyze_momentum(df, sweep_candle_idx, sweep_direction)

    # Calculate overall prediction score
    prediction = calculate_prediction(
        candle_pattern, structure, order_blocks, fvg, volume_analysis, momentum, sweep_direction
    )

    return {
        'valid': True,
        'sweep_direction': sweep_direction,
        'candle_pattern': candle_pattern,
        'structure': structure,
        'order_blocks': order_blocks,
        'fair_value_gaps': fvg,
        'volume': volume_analysis,
        'momentum': momentum,
        'prediction': prediction
    }


def analyze_candle_pattern(sweep_candle: pd.Series, candles_after: pd.DataFrame,
                           direction: str, atr: float) -> Dict:
    """
    Analyze the sweep candle and following candles for reversal patterns.
    """
    o, h, l, c = sweep_candle['open'], sweep_candle['high'], sweep_candle['low'], sweep_candle['close']
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    total_range = h - l

    patterns = []
    score = 0

    # === SWEEP CANDLE ANALYSIS ===

    # 1. Rejection Wick (Pin Bar)
    if direction == 'LONG':
        # For LONG: Want long lower wick (rejection of low prices)
        if total_range > 0:
            rejection_ratio = lower_wick / total_range
            if rejection_ratio > 0.6:
                patterns.append('STRONG_REJECTION_WICK')
                score += 25
            elif rejection_ratio > 0.4:
                patterns.append('REJECTION_WICK')
                score += 15

        # Bullish close (close > open)
        if c > o:
            patterns.append('BULLISH_CLOSE')
            score += 10

    else:  # SHORT
        # For SHORT: Want long upper wick (rejection of high prices)
        if total_range > 0:
            rejection_ratio = upper_wick / total_range
            if rejection_ratio > 0.6:
                patterns.append('STRONG_REJECTION_WICK')
                score += 25
            elif rejection_ratio > 0.4:
                patterns.append('REJECTION_WICK')
                score += 15

        # Bearish close (close < open)
        if c < o:
            patterns.append('BEARISH_CLOSE')
            score += 10

    # 2. Doji (indecision)
    if body < total_range * 0.1 and total_range > atr * 0.5:
        patterns.append('DOJI')
        score += 5  # Neutral - could go either way

    # 3. Large body (strong conviction)
    if body > total_range * 0.7:
        if (direction == 'LONG' and c > o) or (direction == 'SHORT' and c < o):
            patterns.append('STRONG_BODY_REVERSAL')
            score += 15
        else:
            patterns.append('STRONG_BODY_CONTINUATION')
            score -= 15  # Against reversal

    # === FOLLOW-THROUGH CANDLES ===

    follow_through_score = 0
    follow_through_candles = 0

    if len(candles_after) > 0:
        for i, (idx, candle) in enumerate(candles_after.iterrows()):
            if i >= 5:  # Only check first 5 candles
                break

            candle_c, candle_o = candle['close'], candle['open']

            if direction == 'LONG':
                # Want bullish candles (close > open) and higher closes
                if candle_c > candle_o:
                    follow_through_candles += 1
                    follow_through_score += 5 * (5 - i)  # More weight to earlier candles
                elif candle_c < candle_o:
                    follow_through_score -= 3 * (5 - i)
            else:
                # Want bearish candles for SHORT
                if candle_c < candle_o:
                    follow_through_candles += 1
                    follow_through_score += 5 * (5 - i)
                elif candle_c > candle_o:
                    follow_through_score -= 3 * (5 - i)

        if follow_through_candles >= 3:
            patterns.append('STRONG_FOLLOW_THROUGH')
            score += min(follow_through_score, 25)
        elif follow_through_candles >= 2:
            patterns.append('FOLLOW_THROUGH')
            score += min(follow_through_score, 15)

    # === ENGULFING PATTERN ===
    if len(candles_after) >= 1:
        next_candle = candles_after.iloc[0]
        nc_o, nc_c = next_candle['open'], next_candle['close']
        nc_body = abs(nc_c - nc_o)

        if direction == 'LONG' and nc_c > nc_o:
            # Bullish engulfing: next candle body engulfs sweep candle body
            if nc_c > max(o, c) and nc_o < min(o, c) and nc_body > body:
                patterns.append('BULLISH_ENGULFING')
                score += 20
        elif direction == 'SHORT' and nc_c < nc_o:
            # Bearish engulfing
            if nc_o > max(o, c) and nc_c < min(o, c) and nc_body > body:
                patterns.append('BEARISH_ENGULFING')
                score += 20

    # Determine strength
    if score >= 50:
        strength = 'STRONG'
    elif score >= 25:
        strength = 'MODERATE'
    elif score >= 0:
        strength = 'WEAK'
    else:
        strength = 'BEARISH' if direction == 'LONG' else 'BULLISH'

    return {
        'patterns': patterns,
        'score': score,
        'strength': strength,
        'rejection_ratio': lower_wick / total_range if direction == 'LONG' and total_range > 0 else upper_wick / total_range if total_range > 0 else 0,
        'body_ratio': body / total_range if total_range > 0 else 0,
        'follow_through_candles': follow_through_candles
    }


def analyze_structure(df: pd.DataFrame, sweep_idx: int, direction: str, atr: float) -> Dict:
    """
    Analyze market structure after the sweep.
    Detects: BOS (Break of Structure), CHoCH (Change of Character), HH/HL/LH/LL
    """
    if len(df) < sweep_idx + 3:
        return {'valid': False, 'structure_type': 'UNKNOWN'}

    # Get price data after sweep
    post_sweep = df.iloc[sweep_idx:]
    highs = post_sweep['high'].values
    lows = post_sweep['low'].values
    closes = post_sweep['close'].values

    # Find swing points after sweep
    swing_highs = []
    swing_lows = []

    for i in range(2, min(len(post_sweep) - 2, 15)):
        # Swing high: higher than 2 candles before and after
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({'idx': i, 'price': highs[i]})
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({'idx': i, 'price': lows[i]})

    structure_type = 'UNKNOWN'
    bos_detected = False
    choch_detected = False

    # Get pre-sweep structure for comparison
    pre_sweep = df.iloc[max(0, sweep_idx - 20):sweep_idx]

    if direction == 'LONG':
        # After sweeping a low, look for bullish structure shift
        # CHoCH: First higher high after the sweep
        # BOS: Break above recent swing high

        if len(pre_sweep) > 0:
            recent_high = pre_sweep['high'].max()

            # Check if any candle after sweep broke above recent high
            for i, h in enumerate(highs[1:], 1):
                if h > recent_high:
                    bos_detected = True
                    structure_type = 'BULLISH_BOS'
                    break

        # Check for higher lows (bullish structure)
        if len(swing_lows) >= 2:
            if swing_lows[-1]['price'] > swing_lows[0]['price']:
                structure_type = 'HIGHER_LOWS'
                choch_detected = True

    else:  # SHORT
        # After sweeping a high, look for bearish structure shift
        if len(pre_sweep) > 0:
            recent_low = pre_sweep['low'].min()

            for i, l in enumerate(lows[1:], 1):
                if l < recent_low:
                    bos_detected = True
                    structure_type = 'BEARISH_BOS'
                    break

        # Check for lower highs (bearish structure)
        if len(swing_highs) >= 2:
            if swing_highs[-1]['price'] < swing_highs[0]['price']:
                structure_type = 'LOWER_HIGHS'
                choch_detected = True

    # Score the structure
    score = 0
    if bos_detected:
        score += 30
    if choch_detected:
        score += 20
    if structure_type in ['BULLISH_BOS', 'BEARISH_BOS']:
        score += 10
    elif structure_type in ['HIGHER_LOWS', 'LOWER_HIGHS']:
        score += 15

    return {
        'valid': True,
        'structure_type': structure_type,
        'bos_detected': bos_detected,
        'choch_detected': choch_detected,
        'swing_highs_count': len(swing_highs),
        'swing_lows_count': len(swing_lows),
        'score': score
    }


def detect_order_blocks(df: pd.DataFrame, sweep_idx: int, direction: str, atr: float) -> Dict:
    """
    Detect Order Blocks (OB) near the sweep area.

    Bullish OB: Last bearish candle before a strong bullish move
    Bearish OB: Last bullish candle before a strong bearish move
    """
    order_blocks = []

    # Look for OBs in the 10 candles before sweep
    start_idx = max(0, sweep_idx - 10)
    lookback = df.iloc[start_idx:sweep_idx + 1]

    if len(lookback) < 3:
        return {'order_blocks': [], 'has_ob': False, 'score': 0}

    for i in range(len(lookback) - 2):
        candle = lookback.iloc[i]
        next_candle = lookback.iloc[i + 1]

        o, c = candle['open'], candle['close']
        h, l = candle['high'], candle['low']
        body = abs(c - o)

        # Next candle info
        nc_o, nc_c = next_candle['open'], next_candle['close']
        nc_body = abs(nc_c - nc_o)

        if direction == 'LONG':
            # Bullish OB: Bearish candle followed by strong bullish move
            if c < o:  # Bearish candle
                # Check if next candle is strongly bullish
                if nc_c > nc_o and nc_body > body * 1.5:
                    order_blocks.append({
                        'type': 'BULLISH_OB',
                        'high': h,
                        'low': l,
                        'midpoint': (h + l) / 2,
                        'strength': 'STRONG' if nc_body > body * 2 else 'MODERATE'
                    })
        else:
            # Bearish OB: Bullish candle followed by strong bearish move
            if c > o:  # Bullish candle
                if nc_c < nc_o and nc_body > body * 1.5:
                    order_blocks.append({
                        'type': 'BEARISH_OB',
                        'high': h,
                        'low': l,
                        'midpoint': (h + l) / 2,
                        'strength': 'STRONG' if nc_body > body * 2 else 'MODERATE'
                    })

    score = 0
    for ob in order_blocks:
        if ob['strength'] == 'STRONG':
            score += 15
        else:
            score += 10

    return {
        'order_blocks': order_blocks[-3:],  # Keep last 3
        'has_ob': len(order_blocks) > 0,
        'count': len(order_blocks),
        'score': min(score, 25)  # Cap at 25
    }


def detect_fair_value_gaps(df: pd.DataFrame, sweep_idx: int, direction: str, atr: float) -> Dict:
    """
    Detect Fair Value Gaps (FVG) / Imbalances near the sweep.

    FVG is a 3-candle pattern where middle candle's range doesn't overlap
    with the gap between candle 1's high and candle 3's low (or vice versa).
    """
    fvgs = []

    # Look for FVGs in candles around the sweep
    start_idx = max(0, sweep_idx - 5)
    end_idx = min(len(df), sweep_idx + 10)

    for i in range(start_idx, end_idx - 2):
        c1 = df.iloc[i]
        c2 = df.iloc[i + 1]
        c3 = df.iloc[i + 2]

        # Bullish FVG: Gap between candle 1's high and candle 3's low
        if c3['low'] > c1['high']:
            gap_size = c3['low'] - c1['high']
            if gap_size > atr * 0.1:  # Minimum gap size
                fvgs.append({
                    'type': 'BULLISH_FVG',
                    'top': c3['low'],
                    'bottom': c1['high'],
                    'size': gap_size,
                    'size_atr': gap_size / atr,
                    'filled': False  # Would need to check if price returned
                })

        # Bearish FVG: Gap between candle 3's high and candle 1's low
        if c3['high'] < c1['low']:
            gap_size = c1['low'] - c3['high']
            if gap_size > atr * 0.1:
                fvgs.append({
                    'type': 'BEARISH_FVG',
                    'top': c1['low'],
                    'bottom': c3['high'],
                    'size': gap_size,
                    'size_atr': gap_size / atr,
                    'filled': False
                })

    # Score based on FVGs that support the direction
    score = 0
    supporting_fvgs = []

    for fvg in fvgs:
        if (direction == 'LONG' and fvg['type'] == 'BULLISH_FVG') or \
           (direction == 'SHORT' and fvg['type'] == 'BEARISH_FVG'):
            supporting_fvgs.append(fvg)
            score += min(fvg['size_atr'] * 10, 15)

    return {
        'all_fvgs': fvgs[-5:],  # Keep last 5
        'supporting_fvgs': supporting_fvgs,
        'count': len(fvgs),
        'supporting_count': len(supporting_fvgs),
        'score': min(int(score), 20)
    }


def analyze_volume_profile(df: pd.DataFrame, sweep_idx: int, direction: str) -> Dict:
    """
    Analyze volume on sweep and reaction candles.
    """
    if 'volume' not in df.columns:
        return {'valid': False, 'score': 0}

    # Get volume data
    volumes = df['volume'].values
    sweep_volume = volumes[sweep_idx]
    avg_volume = np.mean(volumes[max(0, sweep_idx - 20):sweep_idx]) if sweep_idx > 0 else sweep_volume

    volume_ratio = sweep_volume / avg_volume if avg_volume > 0 else 1.0

    # Analyze post-sweep volume
    post_sweep_volumes = volumes[sweep_idx + 1:min(len(volumes), sweep_idx + 6)]
    post_sweep_avg = np.mean(post_sweep_volumes) if len(post_sweep_volumes) > 0 else 0

    patterns = []
    score = 0

    # High volume on sweep = strong manipulation
    if volume_ratio > 2.0:
        patterns.append('HIGH_VOLUME_SWEEP')
        score += 20
    elif volume_ratio > 1.5:
        patterns.append('ELEVATED_VOLUME_SWEEP')
        score += 10

    # Declining volume after = exhaustion
    if post_sweep_avg < sweep_volume * 0.7:
        patterns.append('VOLUME_EXHAUSTION')
        score += 10

    # Increasing volume in direction = confirmation
    if post_sweep_avg > avg_volume:
        patterns.append('VOLUME_CONFIRMATION')
        score += 10

    return {
        'valid': True,
        'sweep_volume': sweep_volume,
        'avg_volume': avg_volume,
        'volume_ratio': volume_ratio,
        'post_sweep_avg': post_sweep_avg,
        'patterns': patterns,
        'score': min(score, 25)
    }


def analyze_momentum(df: pd.DataFrame, sweep_idx: int, direction: str) -> Dict:
    """
    Analyze momentum indicators around the sweep.
    """
    closes = df['close'].values

    # Calculate RSI
    def calc_rsi(prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    rsi_at_sweep = calc_rsi(closes[:sweep_idx + 1]) if sweep_idx >= 14 else 50
    rsi_current = calc_rsi(closes) if len(closes) >= 14 else 50

    score = 0
    signals = []

    if direction == 'LONG':
        # RSI oversold at sweep = good
        if rsi_at_sweep < 30:
            signals.append('RSI_OVERSOLD_AT_SWEEP')
            score += 15
        elif rsi_at_sweep < 40:
            signals.append('RSI_LOW_AT_SWEEP')
            score += 10

        # RSI rising after = confirmation
        if rsi_current > rsi_at_sweep + 10:
            signals.append('RSI_RISING')
            score += 10

        # Bullish divergence: price made lower low but RSI made higher low
        if sweep_idx > 20:
            prev_low_idx = np.argmin(closes[sweep_idx - 20:sweep_idx]) + (sweep_idx - 20)
            if closes[sweep_idx] < closes[prev_low_idx]:
                rsi_at_prev = calc_rsi(closes[:prev_low_idx + 1])
                if rsi_at_sweep > rsi_at_prev:
                    signals.append('BULLISH_DIVERGENCE')
                    score += 20
    else:
        # RSI overbought at sweep = good for short
        if rsi_at_sweep > 70:
            signals.append('RSI_OVERBOUGHT_AT_SWEEP')
            score += 15
        elif rsi_at_sweep > 60:
            signals.append('RSI_HIGH_AT_SWEEP')
            score += 10

        if rsi_current < rsi_at_sweep - 10:
            signals.append('RSI_FALLING')
            score += 10

        # Bearish divergence
        if sweep_idx > 20:
            prev_high_idx = np.argmax(closes[sweep_idx - 20:sweep_idx]) + (sweep_idx - 20)
            if closes[sweep_idx] > closes[prev_high_idx]:
                rsi_at_prev = calc_rsi(closes[:prev_high_idx + 1])
                if rsi_at_sweep < rsi_at_prev:
                    signals.append('BEARISH_DIVERGENCE')
                    score += 20

    return {
        'rsi_at_sweep': rsi_at_sweep,
        'rsi_current': rsi_current,
        'rsi_change': rsi_current - rsi_at_sweep,
        'signals': signals,
        'score': min(score, 30)
    }


def calculate_prediction(candle_pattern: Dict, structure: Dict, order_blocks: Dict,
                         fvg: Dict, volume: Dict, momentum: Dict, direction: str) -> Dict:
    """
    Calculate overall prediction based on all price action components.
    """
    # Aggregate scores
    total_score = (
        candle_pattern.get('score', 0) +
        structure.get('score', 0) +
        order_blocks.get('score', 0) +
        fvg.get('score', 0) +
        volume.get('score', 0) +
        momentum.get('score', 0)
    )

    max_score = 25 + 30 + 25 + 20 + 25 + 30  # 155 max

    # Normalize to 0-100
    normalized_score = min(100, int(total_score / max_score * 100))

    # Determine prediction
    if normalized_score >= 70:
        prediction = f"STRONG_{direction}"
        confidence = "HIGH"
        action = "ENTER" if direction == 'LONG' else "ENTER SHORT"
    elif normalized_score >= 50:
        prediction = f"MODERATE_{direction}"
        confidence = "MEDIUM"
        action = "ENTER with caution"
    elif normalized_score >= 30:
        prediction = "NEUTRAL"
        confidence = "LOW"
        action = "WAIT for confirmation"
    else:
        opposite = "SHORT" if direction == 'LONG' else "LONG"
        prediction = f"WEAK_{direction}"
        confidence = "LOW"
        action = f"SKIP - may reverse to {opposite}"

    # Key signals summary
    key_signals = []
    if candle_pattern.get('strength') == 'STRONG':
        key_signals.append("Strong reversal candle")
    if structure.get('bos_detected'):
        key_signals.append("Break of Structure confirmed")
    if structure.get('choch_detected'):
        key_signals.append("Change of Character detected")
    if order_blocks.get('has_ob'):
        key_signals.append(f"{order_blocks['count']} Order Block(s) supporting")
    if fvg.get('supporting_count', 0) > 0:
        key_signals.append(f"{fvg['supporting_count']} Fair Value Gap(s)")
    if 'BULLISH_DIVERGENCE' in momentum.get('signals', []) or 'BEARISH_DIVERGENCE' in momentum.get('signals', []):
        key_signals.append("RSI Divergence detected")
    if volume.get('volume_ratio', 1) > 1.5:
        key_signals.append("High volume sweep")

    # Warning signals
    warnings = []
    if candle_pattern.get('strength') == 'WEAK':
        warnings.append("Weak reversal candle pattern")
    if candle_pattern.get('score', 0) < 0:
        warnings.append("Price action suggests continuation, not reversal")
    if not structure.get('bos_detected') and not structure.get('choch_detected'):
        warnings.append("No structure shift confirmed yet")
    if volume.get('volume_ratio', 1) < 0.8:
        warnings.append("Low volume sweep - weak manipulation")

    return {
        'direction': direction,
        'prediction': prediction,
        'confidence': confidence,
        'action': action,
        'score': normalized_score,
        'component_scores': {
            'candle_pattern': candle_pattern.get('score', 0),
            'structure': structure.get('score', 0),
            'order_blocks': order_blocks.get('score', 0),
            'fvg': fvg.get('score', 0),
            'volume': volume.get('score', 0),
            'momentum': momentum.get('score', 0)
        },
        'key_signals': key_signals,
        'warnings': warnings
    }


def get_price_action_summary(analysis: Dict) -> str:
    """
    Generate a human-readable summary of the price action analysis.
    """
    if not analysis.get('valid'):
        return "Insufficient data for price action analysis"

    pred = analysis.get('prediction', {})

    summary_lines = [
        f"📊 **Price Action Analysis**",
        f"",
        f"**Prediction:** {pred.get('prediction', 'UNKNOWN')}",
        f"**Confidence:** {pred.get('confidence', 'UNKNOWN')} ({pred.get('score', 0)}/100)",
        f"**Action:** {pred.get('action', 'WAIT')}",
        f"",
        f"**Key Signals:**"
    ]

    for signal in pred.get('key_signals', []):
        summary_lines.append(f"  ✅ {signal}")

    if pred.get('warnings'):
        summary_lines.append(f"")
        summary_lines.append(f"**Warnings:**")
        for warning in pred.get('warnings', []):
            summary_lines.append(f"  ⚠️ {warning}")

    return "\n".join(summary_lines)


# === INTEGRATION HELPER ===

def analyze_sweep_with_price_action(df: pd.DataFrame, sweep_status: Dict, atr: float) -> Dict:
    """
    Integration function to analyze a detected sweep with full price action.
    Call this from liquidity_hunter.py after detecting a sweep.

    Args:
        df: OHLCV DataFrame
        sweep_status: Dict from detect_sweep()
        atr: Average True Range

    Returns:
        Dict with price action analysis added to sweep_status
    """
    if not sweep_status.get('detected'):
        return sweep_status

    candles_ago = sweep_status.get('candles_ago', 1)
    # candles_ago = len(df) - breach_idx, so breach_idx = len(df) - candles_ago
    sweep_idx = len(df) - candles_ago
    direction = sweep_status.get('direction', 'LONG')

    # Validate index
    if sweep_idx < 0 or sweep_idx >= len(df):
        print(f"[PA_DEBUG] Invalid sweep_idx: {sweep_idx}, len(df): {len(df)}, candles_ago: {candles_ago}")
        return sweep_status

    # Run full price action analysis
    pa_analysis = analyze_sweep_reaction(df, sweep_idx, direction, atr)

    # Add to sweep status
    sweep_status['price_action'] = pa_analysis
    sweep_status['price_action_score'] = pa_analysis.get('prediction', {}).get('score', 0)
    sweep_status['price_action_prediction'] = pa_analysis.get('prediction', {}).get('prediction', 'UNKNOWN')
    sweep_status['price_action_confidence'] = pa_analysis.get('prediction', {}).get('confidence', 'LOW')
    sweep_status['price_action_action'] = pa_analysis.get('prediction', {}).get('action', 'WAIT')

    return sweep_status
