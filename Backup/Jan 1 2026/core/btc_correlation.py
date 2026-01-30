"""
BTC Correlation Module
======================
Most altcoins are highly correlated with BTC.
This module provides BTC market context to help filter/adjust alt signals.

Key Insight:
- When BTC dumps, alts dump harder (usually)
- Going long alts when BTC is bearish = counter-trend risk
- Aligning alt trades with BTC trend = higher probability
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

# Cache for BTC data to avoid repeated fetches
_btc_cache: Dict[str, dict] = {}
_btc_cache_lock = threading.Lock()
_btc_cache_ttl = 60  # seconds


@dataclass
class BTCContext:
    """BTC market context for altcoin analysis"""
    
    # Current state
    price: float
    change_1h: float
    change_24h: float
    
    # Trend
    trend: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    trend_strength: str  # 'STRONG', 'MODERATE', 'WEAK'
    
    # Structure
    structure: str  # 'UPTREND', 'DOWNTREND', 'RANGING'
    position_in_range: float  # 0-100%
    
    # For alt signal alignment
    alt_long_safe: bool  # Is it safe to long alts?
    alt_short_safe: bool  # Is it safe to short alts?
    
    # Recommendation
    recommendation: str  # What to do with alt signals
    confidence: str  # HIGH, MEDIUM, LOW
    
    # Timestamp
    timestamp: str


def get_btc_context(timeframe: str = '15m') -> Optional[BTCContext]:
    """
    Get current BTC market context.
    
    Args:
        timeframe: Timeframe to analyze BTC on (should match alt timeframe)
        
    Returns:
        BTCContext object or None if fetch fails
    """
    cache_key = f"btc_{timeframe}"
    
    # Check cache
    with _btc_cache_lock:
        if cache_key in _btc_cache:
            cached = _btc_cache[cache_key]
            if (datetime.now() - cached['time']).seconds < _btc_cache_ttl:
                return cached['data']
    
    try:
        from .data_fetcher import fetch_binance_klines
        
        # Fetch BTC data
        df = fetch_binance_klines('BTCUSDT', timeframe, 100)
        
        if df is None or len(df) < 50:
            return None
        
        # Calculate context
        context = _analyze_btc(df, timeframe)
        
        # Cache it
        with _btc_cache_lock:
            _btc_cache[cache_key] = {
                'data': context,
                'time': datetime.now()
            }
        
        return context
        
    except Exception as e:
        print(f"Error fetching BTC context: {e}")
        return None


def _analyze_btc(df: pd.DataFrame, timeframe: str) -> BTCContext:
    """Analyze BTC data and return context"""
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    current_price = close.iloc[-1]
    
    # Price changes
    change_1h = 0
    change_24h = 0
    
    # Calculate based on timeframe
    tf_minutes = _tf_to_minutes(timeframe)
    candles_1h = max(1, int(60 / tf_minutes))
    candles_24h = max(1, int(1440 / tf_minutes))
    
    if len(df) > candles_1h:
        price_1h_ago = close.iloc[-candles_1h - 1]
        change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100
    
    if len(df) > candles_24h:
        price_24h_ago = close.iloc[-candles_24h - 1]
        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    
    # EMAs for trend
    ema_9 = close.ewm(span=9).mean().iloc[-1]
    ema_20 = close.ewm(span=20).mean().iloc[-1]
    ema_50 = close.ewm(span=50).mean().iloc[-1]
    
    # Trend determination
    if ema_9 > ema_20 > ema_50:
        trend = 'BULLISH'
        ema_spread = ((ema_9 - ema_50) / ema_50) * 100
        trend_strength = 'STRONG' if ema_spread > 2 else 'MODERATE' if ema_spread > 1 else 'WEAK'
    elif ema_9 < ema_20 < ema_50:
        trend = 'BEARISH'
        ema_spread = ((ema_50 - ema_9) / ema_50) * 100
        trend_strength = 'STRONG' if ema_spread > 2 else 'MODERATE' if ema_spread > 1 else 'WEAK'
    else:
        trend = 'NEUTRAL'
        trend_strength = 'WEAK'
    
    # Structure (recent swing analysis)
    lookback = 20
    recent_high = high.iloc[-lookback:].max()
    recent_low = low.iloc[-lookback:].min()
    
    # Find swing highs/lows
    highs = []
    lows = []
    for i in range(2, len(df) - 2):
        if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
           high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
            highs.append(high.iloc[i])
        if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
           low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
            lows.append(low.iloc[i])
    
    # Structure based on recent swings
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            structure = 'UPTREND'
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            structure = 'DOWNTREND'
        else:
            structure = 'RANGING'
    else:
        structure = 'RANGING'
    
    # Position in range
    if recent_high > recent_low:
        position_in_range = ((current_price - recent_low) / (recent_high - recent_low)) * 100
    else:
        position_in_range = 50
    
    # Alt signal safety
    # Long alts is safe when BTC is bullish or neutral-to-bullish
    alt_long_safe = trend in ['BULLISH', 'NEUTRAL'] and change_1h > -2
    
    # Short alts is safe when BTC is bearish or neutral-to-bearish
    alt_short_safe = trend in ['BEARISH', 'NEUTRAL'] and change_1h < 2
    
    # Recommendation
    if trend == 'BULLISH' and structure == 'UPTREND':
        recommendation = "BTC bullish - Alt longs aligned"
        confidence = 'HIGH'
    elif trend == 'BEARISH' and structure == 'DOWNTREND':
        recommendation = "BTC bearish - Alt shorts aligned, avoid longs"
        confidence = 'HIGH'
    elif trend == 'BULLISH' and change_1h < -1:
        recommendation = "BTC dipping in uptrend - Good alt long entry?"
        confidence = 'MEDIUM'
    elif trend == 'BEARISH' and change_1h > 1:
        recommendation = "BTC bouncing in downtrend - Risky for alt longs"
        confidence = 'MEDIUM'
    else:
        recommendation = "BTC mixed - Be selective with alts"
        confidence = 'LOW'
    
    return BTCContext(
        price=current_price,
        change_1h=change_1h,
        change_24h=change_24h,
        trend=trend,
        trend_strength=trend_strength,
        structure=structure,
        position_in_range=position_in_range,
        alt_long_safe=alt_long_safe,
        alt_short_safe=alt_short_safe,
        recommendation=recommendation,
        confidence=confidence,
        timestamp=datetime.now().strftime('%H:%M:%S')
    )


def calculate_btc_correlation(alt_symbol: str, timeframe: str = '15m', periods: int = 50) -> Dict:
    """
    Calculate actual correlation coefficient (r) between an altcoin and BTC.
    
    Args:
        alt_symbol: Altcoin symbol (e.g., 'ETHUSDT', 'ETH', 'eth')
        timeframe: Timeframe for correlation calculation
        periods: Number of periods to calculate correlation over
        
    Returns:
        {
            'correlation': float (-1 to 1),
            'strength': str ('VERY HIGH', 'HIGH', 'MODERATE', 'LOW', 'INDEPENDENT'),
            'interpretation': str,
            'btc_relevance': str ('CRITICAL', 'IMPORTANT', 'MODERATE', 'LOW')
        }
    """
    # Normalize symbol - ensure it has USDT suffix
    alt_symbol = alt_symbol.upper().strip()
    if not alt_symbol.endswith('USDT'):
        alt_symbol = alt_symbol + 'USDT'
    
    # Skip if it's BTC itself
    if alt_symbol == 'BTCUSDT':
        return {
            'correlation': 1.0,
            'strength': 'PERFECT',
            'interpretation': 'This is BTC',
            'btc_relevance': 'N/A',
            'periods': 0,
        }
    
    cache_key = f"corr_{alt_symbol}_{timeframe}"
    
    # Check cache
    with _btc_cache_lock:
        if cache_key in _btc_cache:
            cached = _btc_cache[cache_key]
            if (datetime.now() - cached['time']).seconds < _btc_cache_ttl * 5:  # 5 min cache for correlation
                return cached['data']
    
    try:
        from .data_fetcher import fetch_binance_klines
        
        # Fetch both BTC and alt data
        btc_df = fetch_binance_klines('BTCUSDT', timeframe, periods + 10)
        alt_df = fetch_binance_klines(alt_symbol, timeframe, periods + 10)
        
        # Debug: Check what we got
        btc_len = len(btc_df) if btc_df is not None else 0
        alt_len = len(alt_df) if alt_df is not None else 0
        
        if btc_df is None or alt_df is None:
            return _default_correlation_with_reason(f"Data fetch failed: BTC={btc_len}, {alt_symbol}={alt_len}")
        
        if len(btc_df) < periods or len(alt_df) < periods:
            return _default_correlation_with_reason(f"Insufficient data: BTC={btc_len}, {alt_symbol}={alt_len} (need {periods})")
        
        # Calculate returns (percentage change)
        btc_returns = btc_df['Close'].pct_change().dropna().tail(periods)
        alt_returns = alt_df['Close'].pct_change().dropna().tail(periods)
        
        # Align lengths
        min_len = min(len(btc_returns), len(alt_returns))
        btc_returns = btc_returns.tail(min_len).values
        alt_returns = alt_returns.tail(min_len).values
        
        # Calculate Pearson correlation
        if len(btc_returns) < 20:
            return _default_correlation_with_reason(f"Not enough returns: {len(btc_returns)} (need 20)")
        
        correlation = np.corrcoef(btc_returns, alt_returns)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            return _default_correlation_with_reason("Correlation calculation returned NaN")
        
        # Interpret correlation
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.85:
            strength = 'VERY HIGH'
            btc_relevance = 'CRITICAL'
            if correlation > 0:
                interpretation = f"Moves almost exactly with BTC - BTC direction is critical"
            else:
                interpretation = f"Moves opposite to BTC - Rare, use as hedge"
        elif abs_corr >= 0.7:
            strength = 'HIGH'
            btc_relevance = 'IMPORTANT'
            interpretation = f"Strong BTC follower - Check BTC before trading"
        elif abs_corr >= 0.5:
            strength = 'MODERATE'
            btc_relevance = 'MODERATE'
            interpretation = f"Moderate BTC influence - Has some independence"
        elif abs_corr >= 0.3:
            strength = 'LOW'
            btc_relevance = 'LOW'
            interpretation = f"Weak BTC correlation - More independent moves"
        else:
            strength = 'INDEPENDENT'
            btc_relevance = 'MINIMAL'
            interpretation = f"Moves independently of BTC - Focus on coin-specific factors"
        
        result = {
            'correlation': round(correlation, 2),
            'strength': strength,
            'interpretation': interpretation,
            'btc_relevance': btc_relevance,
            'periods': min_len,
        }
        
        # Cache result
        with _btc_cache_lock:
            _btc_cache[cache_key] = {'data': result, 'time': datetime.now()}
        
        return result
        
    except Exception as e:
        return _default_correlation_with_reason(f"Error: {str(e)[:50]}")


def _default_correlation() -> Dict:
    """Return default correlation when calculation fails"""
    return {
        'correlation': 0.75,  # Assume moderate-high by default (most alts)
        'strength': 'ESTIMATED',
        'interpretation': 'Using typical alt correlation (most alts ~0.7-0.8 with BTC)',
        'btc_relevance': 'IMPORTANT',
        'periods': 0,
    }


def _default_correlation_with_reason(reason: str) -> Dict:
    """Return default correlation with debug reason"""
    return {
        'correlation': 0.75,  # Assume moderate-high by default (most alts)
        'strength': 'ESTIMATED',
        'interpretation': f'Est. ~0.75 ({reason})',
        'btc_relevance': 'IMPORTANT',
        'periods': 0,
    }


def check_btc_alignment(
    alt_direction: str,  # 'LONG' or 'SHORT'
    btc_context: Optional[BTCContext] = None,
    timeframe: str = '15m'
) -> Dict:
    """
    Check if an alt signal aligns with BTC trend.
    
    Returns:
        {
            'aligned': bool,
            'warning': str or None,
            'adjustment': str,  # 'FULL_SIZE', 'REDUCE_SIZE', 'SKIP'
            'reason': str
        }
    """
    if btc_context is None:
        btc_context = get_btc_context(timeframe)
    
    if btc_context is None:
        return {
            'aligned': True,  # Can't check, assume OK
            'warning': None,
            'adjustment': 'FULL_SIZE',
            'reason': 'BTC data unavailable'
        }
    
    alt_direction = alt_direction.upper()
    
    if alt_direction == 'LONG':
        if btc_context.alt_long_safe:
            if btc_context.trend == 'BULLISH':
                return {
                    'aligned': True,
                    'warning': None,
                    'adjustment': 'FULL_SIZE',
                    'reason': f'BTC {btc_context.trend} ({btc_context.change_1h:+.1f}% 1h) - Long aligned'
                }
            else:
                return {
                    'aligned': True,
                    'warning': 'BTC neutral - monitor closely',
                    'adjustment': 'FULL_SIZE',
                    'reason': f'BTC {btc_context.trend} - Acceptable for longs'
                }
        else:
            if btc_context.change_1h < -3:
                return {
                    'aligned': False,
                    'warning': f'⚠️ BTC dumping ({btc_context.change_1h:+.1f}% 1h) - Alt long risky!',
                    'adjustment': 'SKIP',
                    'reason': 'BTC in active selloff - avoid alt longs'
                }
            else:
                return {
                    'aligned': False,
                    'warning': f'⚠️ BTC {btc_context.trend} - Alt long counter-trend',
                    'adjustment': 'REDUCE_SIZE',
                    'reason': f'BTC bearish ({btc_context.change_24h:+.1f}% 24h) - reduce size'
                }
    
    elif alt_direction == 'SHORT':
        if btc_context.alt_short_safe:
            if btc_context.trend == 'BEARISH':
                return {
                    'aligned': True,
                    'warning': None,
                    'adjustment': 'FULL_SIZE',
                    'reason': f'BTC {btc_context.trend} ({btc_context.change_1h:+.1f}% 1h) - Short aligned'
                }
            else:
                return {
                    'aligned': True,
                    'warning': 'BTC neutral - monitor closely',
                    'adjustment': 'FULL_SIZE',
                    'reason': f'BTC {btc_context.trend} - Acceptable for shorts'
                }
        else:
            if btc_context.change_1h > 3:
                return {
                    'aligned': False,
                    'warning': f'⚠️ BTC pumping ({btc_context.change_1h:+.1f}% 1h) - Alt short risky!',
                    'adjustment': 'SKIP',
                    'reason': 'BTC in active rally - avoid alt shorts'
                }
            else:
                return {
                    'aligned': False,
                    'warning': f'⚠️ BTC {btc_context.trend} - Alt short counter-trend',
                    'adjustment': 'REDUCE_SIZE',
                    'reason': f'BTC bullish ({btc_context.change_24h:+.1f}% 24h) - reduce size'
                }
    
    return {
        'aligned': True,
        'warning': None,
        'adjustment': 'FULL_SIZE',
        'reason': 'No directional signal to check'
    }


def render_btc_context_html(btc_context: Optional[BTCContext], correlation_data: Dict = None) -> str:
    """Render BTC context as HTML for display"""
    
    if btc_context is None:
        return "<div style='background: #1a1a2e; border-radius: 8px; padding: 10px; border: 1px dashed #444;'><span style='color: #888;'>₿ BTC Context: Unavailable</span></div>"
    
    # Colors
    trend_color = "#00ff88" if btc_context.trend == 'BULLISH' else "#ff6b6b" if btc_context.trend == 'BEARISH' else "#ffcc00"
    change_1h_color = "#00ff88" if btc_context.change_1h > 0 else "#ff6b6b" if btc_context.change_1h < 0 else "#888"
    change_24h_color = "#00ff88" if btc_context.change_24h > 0 else "#ff6b6b" if btc_context.change_24h < 0 else "#888"
    
    # Structure emoji
    struct_emoji = "📈" if btc_context.structure == 'UPTREND' else "📉" if btc_context.structure == 'DOWNTREND' else "↔️"
    
    # Safety indicators
    long_safe = "✅" if btc_context.alt_long_safe else "⚠️"
    short_safe = "✅" if btc_context.alt_short_safe else "⚠️"
    
    # Correlation section (if provided)
    corr_html = ""
    if correlation_data and correlation_data.get('correlation') is not None:
        r = correlation_data['correlation']
        strength = correlation_data.get('strength', 'UNKNOWN')
        
        # Color based on correlation strength
        if abs(r) >= 0.85:
            corr_color = "#ff6b6b"  # Very high - BTC critical
            corr_bg = "#2a1a1a"
        elif abs(r) >= 0.7:
            corr_color = "#ffaa00"  # High
            corr_bg = "#2a2a1a"
        elif abs(r) >= 0.5:
            corr_color = "#ffcc00"  # Moderate
            corr_bg = "#252520"
        else:
            corr_color = "#00ff88"  # Low - more independent
            corr_bg = "#1a2a1a"
        
        # Sign indicator
        sign = "+" if r > 0 else ""
        
        # Build correlation HTML without leading whitespace (prevents markdown code block interpretation)
        corr_html = f"<div style='display: flex; gap: 15px; margin-top: 8px; padding: 8px; background: {corr_bg}; border-radius: 6px;'><div style='text-align: center;'><div style='color: #666; font-size: 0.75em;'>Correlation (r)</div><div style='color: {corr_color}; font-weight: bold; font-size: 1.2em;'>{sign}{r:.2f}</div></div><div style='flex: 1;'><div style='color: {corr_color}; font-weight: bold; font-size: 0.9em;'>{strength}</div><div style='color: #888; font-size: 0.8em;'>{correlation_data.get('interpretation', '')}</div></div></div>"
    
    # Build main HTML without indentation to prevent Streamlit markdown issues
    return f"<div style='background: linear-gradient(135deg, #1a1a2e 0%, #0d0d1a 100%); border-radius: 10px; padding: 12px; border-left: 3px solid #f7931a;'><div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'><span style='color: #f7931a; font-weight: bold;'>₿ BTC Market Context</span><span style='color: #888; font-size: 0.8em;'>{btc_context.timestamp}</span></div><div style='display: flex; gap: 15px; flex-wrap: wrap;'><div style='text-align: center;'><div style='color: #666; font-size: 0.75em;'>Price</div><div style='color: #f7931a; font-weight: bold;'>${btc_context.price:,.0f}</div></div><div style='text-align: center;'><div style='color: #666; font-size: 0.75em;'>1h</div><div style='color: {change_1h_color}; font-weight: bold;'>{btc_context.change_1h:+.1f}%</div></div><div style='text-align: center;'><div style='color: #666; font-size: 0.75em;'>24h</div><div style='color: {change_24h_color}; font-weight: bold;'>{btc_context.change_24h:+.1f}%</div></div><div style='text-align: center;'><div style='color: #666; font-size: 0.75em;'>Trend</div><div style='color: {trend_color}; font-weight: bold;'>{btc_context.trend}</div></div><div style='text-align: center;'><div style='color: #666; font-size: 0.75em;'>Structure</div><div style='color: #ccc;'>{struct_emoji} {btc_context.structure}</div></div></div>{corr_html}<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #333;'><span style='color: #888; font-size: 0.85em;'>{btc_context.recommendation}</span><span style='margin-left: 10px; font-size: 0.8em;'>Alt Longs: {long_safe} | Alt Shorts: {short_safe}</span></div></div>"


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes"""
    mapping = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360,
        '12h': 720, '1d': 1440, '1w': 10080
    }
    return mapping.get(tf.lower(), 15)
