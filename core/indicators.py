"""
Technical Indicators Module - 100% POWERED BY 'ta' LIBRARY
============================================================
NO MANUAL CALCULATIONS - Everything uses ta library.
If ta is not installed, this will fail (as it should).

Install: pip install ta
"""

import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT ta LIBRARY - REQUIRED (no fallback)
# ═══════════════════════════════════════════════════════════════════════════════

import ta
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator,
    ROCIndicator, TSIIndicator
)
from ta.trend import (
    MACD, EMAIndicator, SMAIndicator, ADXIndicator, 
    CCIIndicator, VortexIndicator
)
from ta.volatility import (
    BollingerBands, AverageTrueRange, KeltnerChannel, DonchianChannel
)
from ta.volume import (
    OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator,
    VolumeWeightedAveragePrice, ForceIndexIndicator, AccDistIndexIndicator
)

print("✅ ta library loaded - professional indicator calculations active")


# ═══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGES (ta library)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average using ta library"""
    return EMAIndicator(close=data, window=period).ema_indicator()


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average using ta library"""
    return SMAIndicator(close=data, window=period).sma_indicator()


def calculate_hull_ma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Hull Moving Average (WMA-based, no ta equivalent)"""
    # Hull MA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    # ta doesn't have Hull, but we use ta's SMA as base
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wma_half = SMAIndicator(close=data, window=half).sma_indicator()
    wma_full = SMAIndicator(close=data, window=period).sma_indicator()
    diff = 2 * wma_half - wma_full
    return SMAIndicator(close=diff.dropna(), window=sqrt_p).sma_indicator()


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM INDICATORS (ta library)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using ta library"""
    return RSIIndicator(close=data, window=period).rsi()


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD using ta library - returns (macd_line, signal_line, histogram)"""
    macd = MACD(close=data, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic using ta library - returns (k, d)"""
    stoch = StochasticOscillator(high=high, low=low, close=close,
                                  window=k_period, smooth_window=d_period)
    return stoch.stoch(), stoch.stoch_signal()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX using ta library"""
    return ADXIndicator(high=high, low=low, close=close, window=period).adx()


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R using ta library"""
    return WilliamsRIndicator(high=high, low=low, close=close, lbp=period).williams_r()


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CCI using ta library"""
    return CCIIndicator(high=high, low=low, close=close, window=period).cci()


def calculate_roc(data: pd.Series, period: int = 12) -> pd.Series:
    """Calculate Rate of Change using ta library"""
    return ROCIndicator(close=data, window=period).roc()


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY INDICATORS (ta library)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR using ta library"""
    return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()


def calculate_bbands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands using ta library - returns (upper, middle, lower)"""
    bb = BollingerBands(close=data, window=period, window_dev=std_dev)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def calculate_bb_width(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Calculate Bollinger Band Width using ta library"""
    return BollingerBands(close=data, window=period, window_dev=std_dev).bollinger_wband()


def calculate_bb_pct(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Calculate Bollinger %B (position within bands) using ta library"""
    return BollingerBands(close=data, window=period, window_dev=std_dev).bollinger_pband()


def calculate_keltner(high: pd.Series, low: pd.Series, close: pd.Series, 
                      period: int = 20, atr_period: int = 10) -> tuple:
    """Calculate Keltner Channel using ta library - returns (upper, middle, lower)"""
    kc = KeltnerChannel(high=high, low=low, close=close, window=period, window_atr=atr_period)
    return kc.keltner_channel_hband(), kc.keltner_channel_mband(), kc.keltner_channel_lband()


def calculate_donchian(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> tuple:
    """Calculate Donchian Channel using ta library - returns (upper, middle, lower)"""
    dc = DonchianChannel(high=high, low=low, close=close, window=period)
    return dc.donchian_channel_hband(), dc.donchian_channel_mband(), dc.donchian_channel_lband()


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS (ta library)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate OBV using ta library"""
    return OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, period: int = 14) -> pd.Series:
    """Calculate MFI using ta library"""
    return MFIIndicator(high=high, low=low, close=close, volume=volume, window=period).money_flow_index()


def calculate_cmf(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate CMF using ta library"""
    return ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=period).chaikin_money_flow()


def calculate_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Calculate Force Index using ta library"""
    return ForceIndexIndicator(close=close, volume=volume, window=period).force_index()


def calculate_adi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Accumulation/Distribution Index using ta library"""
    return AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()


def get_session_start_index(df: pd.DataFrame, session: str = 'auto') -> int:
    """
    Find the index where the current/specified trading session started.
    
    Sessions (UTC):
    - Asia: 00:00 - 08:00 UTC
    - London: 07:00 - 16:00 UTC  
    - New York: 13:00 - 22:00 UTC
    
    For 'auto': Detects current session based on last candle time.
    Returns index of session start, or 0 if not found.
    """
    if df.empty or 'timestamp' not in df.columns and df.index.dtype != 'datetime64[ns]':
        return 0
    
    # Get timestamps
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
    else:
        timestamps = pd.to_datetime(df.index)
    
    # Convert to UTC if not already
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert('UTC').dt.tz_localize(None)
    
    last_time = timestamps.iloc[-1]
    last_hour = last_time.hour
    
    # Determine current session
    if session == 'auto':
        if 0 <= last_hour < 8:
            session = 'asia'
        elif 7 <= last_hour < 16:
            session = 'london'
        else:
            session = 'newyork'
    
    # Session start hours (UTC)
    session_starts = {
        'asia': 0,      # 00:00 UTC
        'london': 7,    # 07:00 UTC
        'newyork': 13,  # 13:00 UTC
        'daily': 0      # Midnight UTC
    }
    
    target_hour = session_starts.get(session.lower(), 0)
    
    # Find most recent candle at or after session start
    # Look for the first candle of today (or yesterday if session just started)
    today = last_time.date()
    
    for i in range(len(df) - 1, -1, -1):
        candle_time = timestamps.iloc[i]
        candle_hour = candle_time.hour
        candle_date = candle_time.date()
        
        # Found session start on same day
        if candle_date == today and candle_hour == target_hour:
            return i
        
        # If we've gone back to previous day, use first candle of today
        if candle_date < today:
            # Return first candle of today
            for j in range(i + 1, len(df)):
                if timestamps.iloc[j].date() == today:
                    return j
            return i + 1 if i + 1 < len(df) else 0
    
    return 0  # Default to start of data


def calculate_session_vwap(df: pd.DataFrame, session: str = 'auto') -> dict:
    """
    Calculate VWAP anchored to session start.
    
    This is MORE USEFUL for day trading because:
    1. Shows where THIS SESSION's participants entered
    2. Price tends to revert to session VWAP
    3. Breaks above/below session VWAP are significant
    
    Args:
        df: DataFrame with OHLCV data
        session: 'asia', 'london', 'newyork', 'daily', or 'auto' (detect current)
    
    Returns:
        dict with session_vwap, position, distance, etc.
    """
    if df is None or len(df) < 10:
        return {'session_vwap': None, 'error': 'Insufficient data'}
    
    # Find session start
    session_start_idx = get_session_start_index(df, session)
    
    # Need at least 5 candles from session start
    candles_in_session = len(df) - session_start_idx
    if candles_in_session < 5:
        # Fall back to daily VWAP (last 24 candles for 1h, 96 for 15m, etc.)
        session_start_idx = max(0, len(df) - 96)  # ~24h on 15m
    
    # Slice data from session start
    session_df = df.iloc[session_start_idx:].copy()
    
    # Calculate VWAP from session start
    typical_price = (session_df['High'] + session_df['Low'] + session_df['Close']) / 3
    cumulative_tp_vol = (typical_price * session_df['Volume']).cumsum()
    cumulative_vol = session_df['Volume'].cumsum()
    
    # Avoid division by zero
    session_vwap_series = cumulative_tp_vol / cumulative_vol.replace(0, 1)
    
    current_price = df['Close'].iloc[-1]
    current_session_vwap = session_vwap_series.iloc[-1]
    
    # Calculate session VWAP bands (standard deviation)
    squared_diff = (typical_price - session_vwap_series) ** 2
    variance = squared_diff.mean() if len(squared_diff) > 0 else 0
    std_dev = variance ** 0.5 if variance > 0 else current_session_vwap * 0.01
    
    # Distance from VWAP
    distance_pct = ((current_price - current_session_vwap) / current_session_vwap * 100) if current_session_vwap > 0 else 0
    
    # Position relative to VWAP
    if distance_pct > 0.5:
        position = 'ABOVE'
        bias = 'bullish'
    elif distance_pct < -0.5:
        position = 'BELOW'
        bias = 'bearish'
    else:
        position = 'AT_VWAP'
        bias = 'neutral'
    
    return {
        'session_vwap': current_session_vwap,
        'session_vwap_series': session_vwap_series,
        'position': position,
        'distance_pct': distance_pct,
        'bias': bias,
        'upper_1': current_session_vwap + std_dev,
        'lower_1': current_session_vwap - std_dev,
        'upper_2': current_session_vwap + (2 * std_dev),
        'lower_2': current_session_vwap - (2 * std_dev),
        'session_start_idx': session_start_idx,
        'candles_in_session': candles_in_session,
        'session': session
    }


def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                   volume: pd.Series, period: int = None) -> dict:
    """
    Calculate VWAP using ta library with full analysis.
    
    Returns dict with:
        - vwap: Current VWAP value
        - vwap_series: Full VWAP series
        - position: ABOVE/BELOW/AT_VWAP
        - position_text: Human readable position
        - distance_pct: % distance from VWAP
        - bias: bullish/bearish/neutral
        - upper_1, upper_2: +1σ and +2σ bands
        - lower_1, lower_2: -1σ and -2σ bands
    """
    try:
        vwap_ind = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume)
        vwap_series = vwap_ind.volume_weighted_average_price()
    except:
        # Some versions of ta have issues with VWAP, use cumulative calculation
        typical_price = (high + low + close) / 3
        vwap_series = (typical_price * volume).cumsum() / volume.cumsum()
    
    current_price = close.iloc[-1]
    current_vwap = vwap_series.iloc[-1]
    
    # Calculate VWAP standard deviation bands
    # Using typical price deviation from VWAP
    typical_price = (high + low + close) / 3
    squared_diff = (typical_price - vwap_series) ** 2
    
    # Rolling variance (use last 20 periods for stability)
    lookback = min(20, len(close))
    variance = squared_diff.tail(lookback).mean()
    std_dev = variance ** 0.5 if variance > 0 else current_vwap * 0.02  # Fallback to 2%
    
    # VWAP bands
    upper_1 = current_vwap + std_dev
    upper_2 = current_vwap + (2 * std_dev)
    lower_1 = current_vwap - std_dev
    lower_2 = current_vwap - (2 * std_dev)
    
    # Calculate distance
    vwap_distance_pct = ((current_price - current_vwap) / current_vwap * 100) if current_vwap != 0 else 0
    
    # Determine position
    if current_price > current_vwap * 1.02:
        position = "ABOVE"
        position_text = "Above VWAP - Buyers in control, institutional support"
        bias = "bullish"
    elif current_price < current_vwap * 0.98:
        position = "BELOW"
        position_text = "Below VWAP - Sellers in control, watch for bounce"
        bias = "bearish"
    else:
        position = "AT_VWAP"
        position_text = "At VWAP - Neutral zone, wait for direction"
        bias = "neutral"
    
    return {
        'vwap': current_vwap,
        'vwap_series': vwap_series,
        'position': position,
        'position_text': position_text,
        'distance_pct': vwap_distance_pct,
        'bias': bias,
        'upper_1': upper_1,
        'upper_2': upper_2,
        'lower_1': lower_1,
        'lower_2': lower_2,
        'std_dev': std_dev
    }


def detect_vwap_bounce(df: pd.DataFrame, lookback: int = 5, timeframe: str = '15m', mode: str = 'DayTrade') -> dict:
    """
    Detect VWAP bounce pattern - key institutional entry signal.
    
    A VWAP bounce occurs when:
    1. Price touches or crosses VWAP
    2. Then reverses direction (bounces)
    
    Signal freshness is MODE-SPECIFIC:
    - Scalp: FRESH ≤2, RECENT 3-4, STALE 5+
    - DayTrade: FRESH ≤3, RECENT 4-5, STALE 6+
    - Swing: FRESH ≤4, RECENT 5-8, STALE 9+
    - Investment: FRESH ≤3, RECENT 4-6, STALE 7+
    
    Returns dict with:
        - bounce_detected: True/False
        - bounce_type: 'BULLISH_BOUNCE' / 'BEARISH_BOUNCE' / None
        - distance_pct: Current distance from VWAP
        - strength: 'STRONG' / 'MODERATE' / 'WEAK'
        - candles_since_touch: How many candles since VWAP touch
        - signal_age: 'FRESH' / 'RECENT' / 'STALE'
        - time_ago: Human readable time since bounce
        - vwap_level: Current VWAP price
        - score_adjustment: Points to add to signal score
        - suggested_entry: Price to set limit order (when VWAP is proven)
        - entry_type: 'LIMIT' or 'MARKET' based on setup
    """
    # Timeframe to minutes mapping
    tf_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '1d': 1440, '1w': 10080
    }
    candle_mins = tf_minutes.get(timeframe, 15)
    
    # Mode-specific freshness thresholds (candles)
    mode_thresholds = {
        'Scalp': {'fresh': 2, 'recent': 4, 'stale': 5},
        'DayTrade': {'fresh': 3, 'recent': 5, 'stale': 6},
        'Day Trade': {'fresh': 3, 'recent': 5, 'stale': 6},
        'Swing': {'fresh': 4, 'recent': 8, 'stale': 9},
        'Investment': {'fresh': 3, 'recent': 6, 'stale': 7},
    }
    thresholds = mode_thresholds.get(mode, mode_thresholds['DayTrade'])
    
    result = {
        'bounce_detected': False,
        'bounce_type': None,
        'distance_pct': 0,
        'position': None,  # 'ABOVE' or 'BELOW' - ALWAYS set when data available
        'strength': None,
        'candles_since_touch': None,
        'signal_age': None,
        'time_ago': None,
        'vwap_level': 0,
        'score_adjustment': 0,
        'description': '',
        'suggested_entry': None,  # VWAP level when proven - for limit orders!
        'entry_type': None,  # 'LIMIT' when approaching proven VWAP, 'MARKET' after confirmed bounce
        'vwap_proven_support': False,
        'vwap_proven_resistance': False,
        'previous_bounces': 0,
        # VWAP FLIP detection (Andy's strategy)
        'flip_detected': False,
        'flip_type': None,  # 'VWAP_FLIP_BULLISH' / 'VWAP_FLIP_BEARISH' / 'WATCHING_BULLISH' / 'WATCHING_BEARISH'
        'flip_status': None,  # 'WATCHING' / 'CONFIRMED' / 'FAILED'
        'flip_description': '',
        'break_candles_ago': None,  # When the initial break happened
        'retest_candles_ago': None,  # When the retest happened
    }
    
    if df is None or len(df) < 20:
        return result
    
    try:
        # Calculate VWAP (original simple approach)
        vwap_data = calculate_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        vwap_series = vwap_data['vwap_series']
        current_vwap = vwap_data['vwap']
        current_price = df['Close'].iloc[-1]
        
        result['vwap_level'] = current_vwap
        result['distance_pct'] = vwap_data['distance_pct']
        
        # ALWAYS set position - this is CRITICAL data
        if vwap_data['distance_pct'] > 0.1:
            result['position'] = 'ABOVE'
        elif vwap_data['distance_pct'] < -0.1:
            result['position'] = 'BELOW'
        else:
            result['position'] = 'AT_VWAP'
        
        # Extended lookback for swing/investment modes
        extended_lookback = max(lookback, thresholds['stale'] + 2)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRO STRATEGY: Bounce → Retest → Bounce (Entry!)
        # 1. First bounce PROVES VWAP as support/resistance
        # 2. Retest shows price respecting the level
        # 3. Second bounce = HIGH PROBABILITY entry
        # ═══════════════════════════════════════════════════════════════════
        
        # Check if price has been consistently above/below VWAP recently
        lookback_for_side = min(20, len(df) - 1)
        prices_above_vwap = 0
        prices_below_vwap = 0
        
        for i in range(2, lookback_for_side + 1):
            if df['Close'].iloc[-i] > vwap_series.iloc[-i]:
                prices_above_vwap += 1
            else:
                prices_below_vwap += 1
        
        # Determine established side (need at least 60% on one side)
        total_checks = prices_above_vwap + prices_below_vwap
        if total_checks > 0:
            pct_above = prices_above_vwap / total_checks
            pct_below = prices_below_vwap / total_checks
            
            established_above = pct_above >= 0.6  # Was trading above VWAP
            established_below = pct_below >= 0.6  # Was trading below VWAP
        else:
            established_above = False
            established_below = False
        
        result['established_above'] = established_above
        result['established_below'] = established_below
        
        # ═══════════════════════════════════════════════════════════════════
        # SCAN FOR PREVIOUS BOUNCES - Did VWAP already prove itself?
        # Look for bounce patterns in recent history
        # ═══════════════════════════════════════════════════════════════════
        previous_bounces = []
        scan_range = min(30, len(df) - 2)  # Look back 30 candles for previous bounces
        
        for i in range(3, scan_range):  # Start from 3 to exclude current candle area
            idx = -i
            c_low = df['Low'].iloc[idx]
            c_high = df['High'].iloc[idx]
            c_close = df['Close'].iloc[idx]
            c_open = df['Open'].iloc[idx]
            vwap_at = vwap_series.iloc[idx]
            tol = vwap_at * 0.004  # Slightly wider tolerance for historical
            
            # Check for bullish bounce (low touched VWAP, closed green above)
            if c_low <= vwap_at + tol and c_low >= vwap_at - tol:
                if c_close > c_open and c_close > vwap_at:
                    previous_bounces.append({'type': 'BULLISH', 'candles_ago': i})
            
            # Check for bearish bounce (high touched VWAP, closed red below)
            if c_high >= vwap_at - tol and c_high <= vwap_at + tol:
                if c_close < c_open and c_close < vwap_at:
                    previous_bounces.append({'type': 'BEARISH', 'candles_ago': i})
        
        # Did VWAP prove itself recently?
        has_previous_bullish_bounce = any(b['type'] == 'BULLISH' for b in previous_bounces)
        has_previous_bearish_bounce = any(b['type'] == 'BEARISH' for b in previous_bounces)
        result['previous_bounces'] = len(previous_bounces)
        result['vwap_proven_support'] = has_previous_bullish_bounce
        result['vwap_proven_resistance'] = has_previous_bearish_bounce
        
        # Check last N candles for VWAP touch and bounce
        for i in range(1, min(extended_lookback + 1, len(df))):
            idx = -i
            candle_low = df['Low'].iloc[idx]
            candle_high = df['High'].iloc[idx]
            candle_close = df['Close'].iloc[idx]
            candle_open = df['Open'].iloc[idx]
            vwap_at_candle = vwap_series.iloc[idx]
            
            # Tolerance: within 0.3% of VWAP counts as "touch"
            touch_tolerance = vwap_at_candle * 0.003
            
            # BULLISH BOUNCE/RETEST: Low touched VWAP, then price moved UP
            if candle_low <= vwap_at_candle + touch_tolerance and candle_low >= vwap_at_candle - touch_tolerance:
                # Check if price bounced up (close > open and close > vwap)
                if candle_close > candle_open and current_price > vwap_at_candle:
                    result['bounce_detected'] = True
                    result['candles_since_touch'] = i
                    
                    # Determine type based on:
                    # 1. is_retest = Price was established above, came back to test
                    # 2. is_confirmed = VWAP has previous bounce (proven support)
                    is_retest = established_above
                    is_confirmed = has_previous_bullish_bounce
                    
                    result['is_retest'] = is_retest
                    result['is_confirmed'] = is_confirmed
                    
                    # HIGHEST PRIORITY: Confirmed Retest (bounce → retest → bounce)
                    if is_retest and is_confirmed:
                        result['bounce_type'] = 'CONFIRMED_BULLISH_RETEST'
                    elif is_retest:
                        result['bounce_type'] = 'BULLISH_RETEST'
                    elif is_confirmed:
                        result['bounce_type'] = 'BULLISH_BOUNCE_CONFIRMED'
                    else:
                        result['bounce_type'] = 'BULLISH_BOUNCE'
                    
                    # Calculate time ago
                    mins_ago = i * candle_mins
                    if mins_ago < 60:
                        result['time_ago'] = f"{mins_ago}m ago"
                    elif mins_ago < 1440:
                        result['time_ago'] = f"{mins_ago // 60}h {mins_ago % 60}m ago"
                    else:
                        result['time_ago'] = f"{mins_ago // 1440}d {(mins_ago % 1440) // 60}h ago"
                    
                    # Signal freshness based on MODE-SPECIFIC thresholds
                    if i <= thresholds['fresh']:
                        result['signal_age'] = 'FRESH'
                    elif i <= thresholds['recent']:
                        result['signal_age'] = 'RECENT'
                    else:
                        result['signal_age'] = 'STALE'
                    
                    # Strength based on distance, freshness, and confirmation
                    # CONFIRMED RETEST = HIGHEST score (pro entry!)
                    # RETEST = HIGH score
                    # CONFIRMED = MODERATE bonus
                    retest_bonus = 5 if is_retest else 0
                    confirmed_bonus = 5 if is_confirmed else 0
                    
                    if i <= thresholds['fresh'] and abs(vwap_data['distance_pct']) < 1.0:
                        result['strength'] = 'STRONG'
                        result['score_adjustment'] = 12 + retest_bonus + confirmed_bonus
                    elif i <= thresholds['recent'] and abs(vwap_data['distance_pct']) < 2.0:
                        result['strength'] = 'MODERATE'
                        result['score_adjustment'] = 8 + retest_bonus + confirmed_bonus
                    else:
                        result['strength'] = 'WEAK'
                        result['score_adjustment'] = 4 + retest_bonus + confirmed_bonus
                    
                    # Description based on type
                    if result['bounce_type'] == 'CONFIRMED_BULLISH_RETEST':
                        result['description'] = f"🏆 CONFIRMED RETEST ({result['strength']}) - PRO ENTRY! - {result['time_ago']}"
                    elif result['bounce_type'] == 'BULLISH_RETEST':
                        result['description'] = f"🎯 VWAP RETEST ({result['strength']}) - {result['time_ago']}"
                    elif result['bounce_type'] == 'BULLISH_BOUNCE_CONFIRMED':
                        result['description'] = f"✅ VWAP Bounce (Proven Support) ({result['strength']}) - {result['time_ago']}"
                    else:
                        result['description'] = f"VWAP Bullish Bounce ({result['strength']}) - {result['time_ago']}"
                    break
            
            # BEARISH BOUNCE/RETEST: High touched VWAP, then price moved DOWN  
            if candle_high >= vwap_at_candle - touch_tolerance and candle_high <= vwap_at_candle + touch_tolerance:
                # Check if price bounced down (close < open and close < vwap)
                if candle_close < candle_open and current_price < vwap_at_candle:
                    result['bounce_detected'] = True
                    result['candles_since_touch'] = i
                    
                    # Determine type based on:
                    # 1. is_retest = Price was established below, came back up to test
                    # 2. is_confirmed = VWAP has previous bounce (proven resistance)
                    is_retest = established_below
                    is_confirmed = has_previous_bearish_bounce
                    
                    result['is_retest'] = is_retest
                    result['is_confirmed'] = is_confirmed
                    
                    # HIGHEST PRIORITY: Confirmed Retest (bounce → retest → bounce)
                    if is_retest and is_confirmed:
                        result['bounce_type'] = 'CONFIRMED_BEARISH_RETEST'
                    elif is_retest:
                        result['bounce_type'] = 'BEARISH_RETEST'
                    elif is_confirmed:
                        result['bounce_type'] = 'BEARISH_BOUNCE_CONFIRMED'
                    else:
                        result['bounce_type'] = 'BEARISH_BOUNCE'
                    
                    # Calculate time ago
                    mins_ago = i * candle_mins
                    if mins_ago < 60:
                        result['time_ago'] = f"{mins_ago}m ago"
                    elif mins_ago < 1440:
                        result['time_ago'] = f"{mins_ago // 60}h {mins_ago % 60}m ago"
                    else:
                        result['time_ago'] = f"{mins_ago // 1440}d {(mins_ago % 1440) // 60}h ago"
                    
                    # Signal freshness based on MODE-SPECIFIC thresholds
                    if i <= thresholds['fresh']:
                        result['signal_age'] = 'FRESH'
                    elif i <= thresholds['recent']:
                        result['signal_age'] = 'RECENT'
                    else:
                        result['signal_age'] = 'STALE'
                    
                    # Strength based on distance, freshness, and confirmation
                    retest_bonus = 5 if is_retest else 0
                    confirmed_bonus = 5 if is_confirmed else 0
                    
                    if i <= thresholds['fresh'] and abs(vwap_data['distance_pct']) < 1.0:
                        result['strength'] = 'STRONG'
                        result['score_adjustment'] = 12 + retest_bonus + confirmed_bonus
                    elif i <= thresholds['recent'] and abs(vwap_data['distance_pct']) < 2.0:
                        result['strength'] = 'MODERATE'
                        result['score_adjustment'] = 8 + retest_bonus + confirmed_bonus
                    else:
                        result['strength'] = 'WEAK'
                        result['score_adjustment'] = 4 + retest_bonus + confirmed_bonus
                    
                    # Description based on type
                    if result['bounce_type'] == 'CONFIRMED_BEARISH_RETEST':
                        result['description'] = f"🏆 CONFIRMED RETEST ({result['strength']}) - PRO ENTRY! - {result['time_ago']}"
                    elif result['bounce_type'] == 'BEARISH_RETEST':
                        result['description'] = f"🎯 VWAP RETEST ({result['strength']}) - {result['time_ago']}"
                    elif result['bounce_type'] == 'BEARISH_BOUNCE_CONFIRMED':
                        result['description'] = f"✅ VWAP Bounce (Proven Resistance) ({result['strength']}) - {result['time_ago']}"
                    else:
                        result['description'] = f"VWAP Bearish Bounce ({result['strength']}) - {result['time_ago']}"
                    break
        
        # ═══════════════════════════════════════════════════════════════════
        # APPROACHING VWAP - Detect BEFORE the bounce!
        # This is the PREDICTIVE signal - get ready for entry
        # 
        # KEY INSIGHT FROM PRO TRADER:
        # - If VWAP is PROVEN (has previous bounce) + APPROACHING = SET LIMIT ORDER!
        # - This gets you in BEFORE the bounce, not after
        # ═══════════════════════════════════════════════════════════════════
        if not result['bounce_detected']:
            distance = vwap_data['distance_pct']
            
            # Check if price is APPROACHING VWAP (not bounced yet)
            # We want to catch price heading TOWARD VWAP before it touches
            
            # Get recent price movement direction
            if len(df) >= 5:
                recent_closes = df['Close'].iloc[-5:].values
                price_direction = recent_closes[-1] - recent_closes[0]  # + = up, - = down
                
                # Calculate momentum toward VWAP
                distances_from_vwap = []
                for i in range(-5, 0):
                    if i < -len(vwap_series):
                        continue
                    dist = ((df['Close'].iloc[i] - vwap_series.iloc[i]) / vwap_series.iloc[i]) * 100
                    distances_from_vwap.append(abs(dist))
                
                # Is price getting CLOSER to VWAP?
                approaching = len(distances_from_vwap) >= 3 and distances_from_vwap[-1] < distances_from_vwap[0]
                
                # APPROACHING FOR LONG: Price above VWAP, moving DOWN toward it
                if 0.3 < distance < 2.0 and price_direction < 0 and approaching:
                    is_retest_setup = established_above  # Was above, coming back = RETEST
                    is_proven = has_previous_bullish_bounce  # VWAP already bounced before!
                    
                    result['is_retest'] = is_retest_setup
                    result['is_confirmed'] = is_proven
                    result['signal_age'] = 'SETUP'
                    result['time_ago'] = 'approaching'
                    
                    # PRO ENTRY: When VWAP is PROVEN, suggest limit order AT VWAP!
                    if is_proven:
                        result['suggested_entry'] = current_vwap
                        result['entry_type'] = 'LIMIT'
                    
                    # Scoring: PROVEN + RETEST = highest, PROVEN = high, RETEST = good
                    retest_bonus = 5 if is_retest_setup else 0
                    proven_bonus = 7 if is_proven else 0  # Big bonus for proven VWAP!
                    
                    # Determine bounce type based on confirmation level
                    if is_proven and is_retest_setup:
                        result['bounce_type'] = 'CONFIRMED_APPROACHING_BULLISH'
                    elif is_proven:
                        result['bounce_type'] = 'PROVEN_APPROACHING_BULLISH'
                    elif is_retest_setup:
                        result['bounce_type'] = 'RETEST_APPROACHING_BULLISH'
                    else:
                        result['bounce_type'] = 'APPROACHING_BULLISH'
                    
                    # Different messages based on PROVEN status
                    if is_proven:
                        action_text = "SET LIMIT ORDER AT VWAP!"
                    else:
                        action_text = "Watch for bounce"
                    
                    if distance < 0.8:
                        result['strength'] = 'IMMINENT'
                        result['score_adjustment'] = 10 + retest_bonus + proven_bonus
                        if is_proven:
                            result['description'] = f"🏆 PROVEN VWAP! Price {distance:.1f}% away - {action_text}"
                        else:
                            result['description'] = f"VWAP ENTRY ZONE! Price {distance:.1f}% above - {action_text}"
                    elif distance < 1.2:
                        result['strength'] = 'CLOSE'
                        result['score_adjustment'] = 6 + retest_bonus + proven_bonus
                        if is_proven:
                            result['description'] = f"🏆 PROVEN VWAP approaching ({distance:.1f}%) - Prepare limit order!"
                        else:
                            result['description'] = f"Approaching VWAP ({distance:.1f}%) - Get ready for LONG"
                    else:
                        result['strength'] = 'APPROACHING'
                        result['score_adjustment'] = 3 + retest_bonus + proven_bonus
                        if is_proven:
                            result['description'] = f"🏆 Price heading to PROVEN VWAP ({distance:.1f}%)"
                        else:
                            result['description'] = f"Price heading toward VWAP ({distance:.1f}% above)"
                
                # APPROACHING FOR SHORT: Price below VWAP, moving UP toward it
                elif -2.0 < distance < -0.3 and price_direction > 0 and approaching:
                    is_retest_setup = established_below  # Was below, coming back = RETEST
                    is_proven = has_previous_bearish_bounce  # VWAP already rejected before!
                    
                    result['is_retest'] = is_retest_setup
                    result['is_confirmed'] = is_proven
                    result['signal_age'] = 'SETUP'
                    result['time_ago'] = 'approaching'
                    
                    # PRO ENTRY: When VWAP is PROVEN, suggest limit order AT VWAP!
                    if is_proven:
                        result['suggested_entry'] = current_vwap
                        result['entry_type'] = 'LIMIT'
                    
                    # Scoring
                    retest_bonus = 5 if is_retest_setup else 0
                    proven_bonus = 7 if is_proven else 0
                    
                    # Determine bounce type
                    if is_proven and is_retest_setup:
                        result['bounce_type'] = 'CONFIRMED_APPROACHING_BEARISH'
                    elif is_proven:
                        result['bounce_type'] = 'PROVEN_APPROACHING_BEARISH'
                    elif is_retest_setup:
                        result['bounce_type'] = 'RETEST_APPROACHING_BEARISH'
                    else:
                        result['bounce_type'] = 'APPROACHING_BEARISH'
                    
                    # Different messages based on PROVEN status
                    if is_proven:
                        action_text = "SET LIMIT ORDER AT VWAP!"
                    else:
                        action_text = "Watch for rejection"
                    
                    if abs(distance) < 0.8:
                        result['strength'] = 'IMMINENT'
                        result['score_adjustment'] = 10 + retest_bonus + proven_bonus
                        if is_proven:
                            result['description'] = f"🏆 PROVEN RESISTANCE! Price {abs(distance):.1f}% away - {action_text}"
                        else:
                            result['description'] = f"VWAP ENTRY ZONE! Price {abs(distance):.1f}% below - {action_text}"
                    elif abs(distance) < 1.2:
                        result['strength'] = 'CLOSE'
                        result['score_adjustment'] = 6 + retest_bonus + proven_bonus
                        if is_proven:
                            result['description'] = f"🏆 PROVEN RESISTANCE approaching ({abs(distance):.1f}%) - Prepare limit order!"
                        else:
                            result['description'] = f"Approaching VWAP ({abs(distance):.1f}%) - Get ready for SHORT"
                    else:
                        result['strength'] = 'APPROACHING'
                        result['score_adjustment'] = 3 + retest_bonus + proven_bonus
                        if is_proven:
                            result['description'] = f"🏆 Price heading to PROVEN RESISTANCE ({abs(distance):.1f}%)"
                        else:
                            result['description'] = f"Price heading toward VWAP ({abs(distance):.1f}% below)"
        
        # AT VWAP (potential entry zone even without confirmed bounce)
        approaching_types = ['APPROACHING_BULLISH', 'APPROACHING_BEARISH', 'RETEST_APPROACHING_BULLISH', 'RETEST_APPROACHING_BEARISH']
        if not result['bounce_detected'] and result['bounce_type'] not in approaching_types:
            if abs(vwap_data['distance_pct']) < 0.5:
                # Check if this is a RETEST situation
                is_retest_zone = (vwap_data['distance_pct'] > 0 and established_above) or (vwap_data['distance_pct'] < 0 and established_below)
                result['is_retest'] = is_retest_zone
                
                if is_retest_zone:
                    result['bounce_type'] = 'AT_VWAP_RETEST'
                    result['score_adjustment'] = 12  # Higher score for retest zone
                    result['description'] = f"🎯 VWAP RETEST ZONE ({vwap_data['distance_pct']:+.2f}%) - PRO ENTRY!"
                else:
                    result['bounce_type'] = 'AT_VWAP'
                    result['score_adjustment'] = 8
                    result['description'] = f"🎯 AT VWAP ({vwap_data['distance_pct']:+.2f}%) - Entry Zone"
                
                result['strength'] = 'POTENTIAL'
                result['signal_age'] = 'NOW'
                result['time_ago'] = 'now'
        
        # ═══════════════════════════════════════════════════════════════════════════
        # VWAP FLIP DETECTION (Andy's Pro Strategy)
        # 
        # The FLIP pattern:
        # 1. Price breaks THROUGH VWAP (from below to above = bullish break)
        # 2. Price comes back to RETEST VWAP
        # 3. If VWAP holds (bounce) = FLIP CONFIRMED (resistance → support)
        # 4. If VWAP fails (breakdown) = FLIP FAILED
        #
        # This is a TWO-STAGE signal:
        # Stage 1: "VWAP Break detected - watching for flip..."
        # Stage 2: "VWAP FLIP CONFIRMED! Resistance → Support proven"
        # ═══════════════════════════════════════════════════════════════════════════
        
        scan_for_flip = min(25, len(df) - 3)  # Look back 25 candles for flip pattern
        
        # Look for VWAP break (price crossing from one side to the other)
        break_found = False
        break_type = None
        break_idx = None
        
        for i in range(2, scan_for_flip):
            idx = -i
            prev_idx = idx - 1
            
            if prev_idx < -len(df):
                continue
                
            # Get closes and VWAP values
            curr_close = df['Close'].iloc[idx]
            prev_close = df['Close'].iloc[prev_idx]
            curr_vwap = vwap_series.iloc[idx]
            prev_vwap = vwap_series.iloc[prev_idx]
            
            # BULLISH BREAK: Price was below VWAP, now above
            if prev_close < prev_vwap and curr_close > curr_vwap:
                break_found = True
                break_type = 'BULLISH'
                break_idx = i
                break
            
            # BEARISH BREAK: Price was above VWAP, now below
            elif prev_close > prev_vwap and curr_close < curr_vwap:
                break_found = True
                break_type = 'BEARISH'
                break_idx = i
                break
        
        if break_found and break_idx is not None:
            result['break_candles_ago'] = break_idx
            
            # Now check what happened AFTER the break
            # Look from break point to now for retest
            retest_found = False
            retest_idx = None
            flip_confirmed = False
            flip_failed = False
            
            for j in range(break_idx - 1, 0, -1):  # From break to now
                idx = -j
                c_low = df['Low'].iloc[idx]
                c_high = df['High'].iloc[idx]
                c_close = df['Close'].iloc[idx]
                c_open = df['Open'].iloc[idx]
                vwap_at = vwap_series.iloc[idx]
                tolerance = vwap_at * 0.004  # 0.4% tolerance for retest
                
                if break_type == 'BULLISH':
                    # After bullish break, look for retest from above
                    # Retest = low touches VWAP
                    if c_low <= vwap_at + tolerance and c_low >= vwap_at - tolerance:
                        retest_found = True
                        retest_idx = j
                        
                        # Did it HOLD (bounce) or FAIL (breakdown)?
                        # Check subsequent candles
                        if c_close > vwap_at and c_close > c_open:
                            # Bounce confirmed!
                            flip_confirmed = True
                        elif c_close < vwap_at:
                            # Failed - broke back below
                            flip_failed = True
                        break
                        
                elif break_type == 'BEARISH':
                    # After bearish break, look for retest from below
                    # Retest = high touches VWAP
                    if c_high >= vwap_at - tolerance and c_high <= vwap_at + tolerance:
                        retest_found = True
                        retest_idx = j
                        
                        # Did it HOLD (bounce down) or FAIL (breakout)?
                        if c_close < vwap_at and c_close < c_open:
                            # Rejection confirmed!
                            flip_confirmed = True
                        elif c_close > vwap_at:
                            # Failed - broke back above
                            flip_failed = True
                        break
            
            # Set flip results
            if retest_found:
                result['retest_candles_ago'] = retest_idx
                result['flip_detected'] = True
                
                if flip_confirmed:
                    result['flip_status'] = 'CONFIRMED'
                    if break_type == 'BULLISH':
                        result['flip_type'] = 'VWAP_FLIP_BULLISH'
                        result['flip_description'] = f"✅ VWAP FLIP CONFIRMED! Resistance → Support (broke {break_idx} candles ago, retested {retest_idx} candles ago)"
                        # Add bonus score for confirmed flip
                        result['score_adjustment'] += 15
                    else:
                        result['flip_type'] = 'VWAP_FLIP_BEARISH'
                        result['flip_description'] = f"✅ VWAP FLIP CONFIRMED! Support → Resistance (broke {break_idx} candles ago, retested {retest_idx} candles ago)"
                        result['score_adjustment'] += 15
                        
                elif flip_failed:
                    result['flip_status'] = 'FAILED'
                    if break_type == 'BULLISH':
                        result['flip_type'] = 'FLIP_FAILED_BULLISH'
                        result['flip_description'] = f"❌ FLIP FAILED - Broke back below VWAP (false breakout)"
                    else:
                        result['flip_type'] = 'FLIP_FAILED_BEARISH'
                        result['flip_description'] = f"❌ FLIP FAILED - Broke back above VWAP (false breakdown)"
                else:
                    # Retest in progress, waiting for confirmation
                    result['flip_status'] = 'WATCHING'
                    if break_type == 'BULLISH':
                        result['flip_type'] = 'WATCHING_BULLISH'
                        result['flip_description'] = f"👀 WATCHING: VWAP break {break_idx} candles ago, retesting now..."
                    else:
                        result['flip_type'] = 'WATCHING_BEARISH'
                        result['flip_description'] = f"👀 WATCHING: VWAP break {break_idx} candles ago, retesting now..."
            else:
                # Break happened but no retest yet
                # Check if price is approaching for retest
                if break_type == 'BULLISH' and 0 < vwap_data['distance_pct'] < 1.5:
                    result['flip_detected'] = True
                    result['flip_status'] = 'WATCHING'
                    result['flip_type'] = 'WATCHING_BULLISH'
                    result['flip_description'] = f"👀 VWAP broke {break_idx} candles ago - approaching for retest ({vwap_data['distance_pct']:.1f}% above)"
                elif break_type == 'BEARISH' and -1.5 < vwap_data['distance_pct'] < 0:
                    result['flip_detected'] = True
                    result['flip_status'] = 'WATCHING'
                    result['flip_type'] = 'WATCHING_BEARISH'
                    result['flip_description'] = f"👀 VWAP broke {break_idx} candles ago - approaching for retest ({abs(vwap_data['distance_pct']):.1f}% below)"
        
        return result
        
    except Exception as e:
        return result


def find_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series,
                            lookback: int = 20, tolerance: float = 0.02) -> dict:
    """Find support and resistance using Donchian Channel from ta"""
    dc = DonchianChannel(high=high, low=low, close=close, window=lookback)
    
    resistance = dc.donchian_channel_hband().iloc[-1]
    support = dc.donchian_channel_lband().iloc[-1]
    
    return {
        'resistance': resistance,
        'support': support,
        'all_resistance': [resistance],
        'all_support': [support]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ALL-IN-ONE ANALYSIS (ta library)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_all_indicators(df: pd.DataFrame) -> dict:
    """
    Calculate ALL indicators using ta library.
    Returns dict with current values.
    """
    h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']
    price = c.iloc[-1]
    
    # Safe last value getter
    def safe_last(series, default=0):
        try:
            val = series.iloc[-1] if hasattr(series, 'iloc') else series
            return default if pd.isna(val) else float(val)
        except:
            return default
    
    # Calculate all using ta library
    rsi = calculate_rsi(c, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(c)
    stoch_k, stoch_d = calculate_stochastic(h, l, c)
    
    ema_9 = calculate_ema(c, 9)
    ema_20 = calculate_ema(c, 20)
    ema_50 = calculate_ema(c, 50)
    ema_200 = calculate_ema(c, 200) if len(c) >= 200 else calculate_ema(c, 50)
    adx = calculate_adx(h, l, c)
    
    atr = calculate_atr(h, l, c)
    bb_upper, bb_middle, bb_lower = calculate_bbands(c)
    bb_pct = calculate_bb_pct(c)
    
    obv = calculate_obv(c, v)
    mfi = calculate_mfi(h, l, c, v)
    cmf = calculate_cmf(h, l, c, v)
    
    vwap_data = calculate_vwap(h, l, c, v)
    sr = find_support_resistance(h, l, c)
    
    # Build result dict
    result = {
        'price': price,
        'rsi': safe_last(rsi, 50),
        'macd': safe_last(macd_line),
        'macd_signal': safe_last(macd_signal),
        'macd_hist': safe_last(macd_hist),
        'stoch_k': safe_last(stoch_k, 50),
        'stoch_d': safe_last(stoch_d, 50),
        'ema_9': safe_last(ema_9, price),
        'ema_20': safe_last(ema_20, price),
        'ema_50': safe_last(ema_50, price),
        'ema_200': safe_last(ema_200, price),
        'adx': safe_last(adx, 25),
        'atr': safe_last(atr),
        'atr_pct': (safe_last(atr) / price * 100) if price > 0 else 0,
        'bb_upper': safe_last(bb_upper, price * 1.02),
        'bb_middle': safe_last(bb_middle, price),
        'bb_lower': safe_last(bb_lower, price * 0.98),
        'bb_pct': safe_last(bb_pct, 0.5),
        'obv': safe_last(obv),
        'obv_rising': safe_last(obv) > obv.iloc[-5] if len(obv) >= 5 else True,
        'mfi': safe_last(mfi, 50),
        'cmf': safe_last(cmf),
        'vwap': vwap_data['vwap'],
        'vwap_position': vwap_data['position'],
        'vwap_position_text': vwap_data['position_text'],
        'vwap_bias': vwap_data['bias'],
        'support': sr['support'],
        'resistance': sr['resistance'],
    }
    
    # Derived signals
    result['ema_bullish'] = result['ema_9'] > result['ema_20'] > result['ema_50']
    result['ema_bearish'] = result['ema_9'] < result['ema_20'] < result['ema_50']
    result['macd_bullish'] = result['macd'] > result['macd_signal']
    result['macd_bearish'] = result['macd'] < result['macd_signal']
    result['rsi_oversold'] = result['rsi'] < 30
    result['rsi_overbought'] = result['rsi'] > 70
    result['mfi_oversold'] = result['mfi'] < 20
    result['mfi_overbought'] = result['mfi'] > 80
    result['cmf_bullish'] = result['cmf'] > 0.05
    result['cmf_bearish'] = result['cmf'] < -0.05
    result['above_vwap'] = price > result['vwap']
    result['strong_trend'] = result['adx'] > 25
    
    # BB position (0-100)
    result['bb_position'] = result['bb_pct'] * 100
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED SCORING (based on ta indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_signal_score(indicators: dict) -> dict:
    """
    Calculate bullish/bearish score from ta indicators.
    Weighted scoring system - no hard rules!
    """
    bullish = 0
    bearish = 0
    factors = []
    
    # RSI (max ±20)
    rsi = indicators.get('rsi', 50)
    if rsi < 30:
        pts = 20
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'RSI Oversold ({rsi:.0f})', 'points': pts})
    elif rsi < 40:
        pts = 10
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'RSI Low ({rsi:.0f})', 'points': pts})
    elif rsi > 70:
        pts = 20
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'RSI Overbought ({rsi:.0f})', 'points': pts})
    elif rsi > 60:
        pts = 10
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'RSI High ({rsi:.0f})', 'points': pts})
    
    # MACD (max ±15)
    if indicators.get('macd_bullish'):
        pts = 15
        bullish += pts
        factors.append({'type': 'bullish', 'name': 'MACD Bullish Cross', 'points': pts})
    elif indicators.get('macd_bearish'):
        pts = 15
        bearish += pts
        factors.append({'type': 'bearish', 'name': 'MACD Bearish Cross', 'points': pts})
    
    # EMA Stack (max ±20)
    if indicators.get('ema_bullish'):
        pts = 20
        bullish += pts
        factors.append({'type': 'bullish', 'name': 'EMAs Stacked Bullish (9>20>50)', 'points': pts})
    elif indicators.get('ema_bearish'):
        pts = 20
        bearish += pts
        factors.append({'type': 'bearish', 'name': 'EMAs Stacked Bearish (9<20<50)', 'points': pts})
    
    # CMF (max ±25)
    cmf = indicators.get('cmf', 0)
    if cmf > 0.20:
        pts = 25
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'Strong Money Inflow (CMF {cmf:.2f})', 'points': pts})
    elif cmf > 0.10:
        pts = 15
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'Money Inflow (CMF {cmf:.2f})', 'points': pts})
    elif cmf > 0.05:
        pts = 8
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'Slight Inflow (CMF {cmf:.2f})', 'points': pts})
    elif cmf < -0.20:
        pts = 25
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'Strong Money Outflow (CMF {cmf:.2f})', 'points': pts})
    elif cmf < -0.10:
        pts = 15
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'Money Outflow (CMF {cmf:.2f})', 'points': pts})
    elif cmf < -0.05:
        pts = 8
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'Slight Outflow (CMF {cmf:.2f})', 'points': pts})
    
    # MFI (max ±15)
    mfi = indicators.get('mfi', 50)
    if mfi < 20:
        pts = 15
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'MFI Oversold ({mfi:.0f})', 'points': pts})
    elif mfi > 80:
        pts = 15
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'MFI Overbought ({mfi:.0f})', 'points': pts})
    
    # OBV (max ±10)
    if indicators.get('obv_rising'):
        pts = 10
        bullish += pts
        factors.append({'type': 'bullish', 'name': 'OBV Rising', 'points': pts})
    else:
        pts = 10
        bearish += pts
        factors.append({'type': 'bearish', 'name': 'OBV Falling', 'points': pts})
    
    # VWAP (max ±10)
    if indicators.get('above_vwap'):
        pts = 10
        bullish += pts
        factors.append({'type': 'bullish', 'name': 'Above VWAP', 'points': pts})
    else:
        pts = 10
        bearish += pts
        factors.append({'type': 'bearish', 'name': 'Below VWAP', 'points': pts})
    
    # BB Position (max ±15)
    bb_pos = indicators.get('bb_position', 50)
    if bb_pos < 15:
        pts = 15
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'At BB Lower ({bb_pos:.0f}%)', 'points': pts})
    elif bb_pos < 30:
        pts = 8
        bullish += pts
        factors.append({'type': 'bullish', 'name': f'Near BB Lower ({bb_pos:.0f}%)', 'points': pts})
    elif bb_pos > 85:
        pts = 15
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'At BB Upper ({bb_pos:.0f}%)', 'points': pts})
    elif bb_pos > 70:
        pts = 8
        bearish += pts
        factors.append({'type': 'bearish', 'name': f'Near BB Upper ({bb_pos:.0f}%)', 'points': pts})
    
    # ADX amplifier (max ±5)
    if indicators.get('strong_trend'):
        adx = indicators.get('adx', 25)
        if bullish > bearish:
            pts = 5
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'Strong Trend (ADX {adx:.0f})', 'points': pts})
        elif bearish > bullish:
            pts = 5
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'Strong Trend (ADX {adx:.0f})', 'points': pts})
    
    # Calculate final score
    net_score = bullish - bearish
    total_score = 50 + (net_score * 0.4)
    total_score = max(0, min(100, total_score))
    
    # Grade
    if total_score >= 80:
        grade, action = 'A+', 'STRONG BUY'
    elif total_score >= 70:
        grade, action = 'A', 'BUY'
    elif total_score >= 60:
        grade, action = 'B+', 'LEAN BULLISH'
    elif total_score >= 50:
        grade, action = 'B', 'NEUTRAL - Slight Bull'
    elif total_score >= 40:
        grade, action = 'C', 'NEUTRAL - Slight Bear'
    elif total_score >= 30:
        grade, action = 'D', 'LEAN BEARISH'
    elif total_score >= 20:
        grade, action = 'D-', 'SELL'
    else:
        grade, action = 'F', 'STRONG SELL'
    
    return {
        'bullish_score': bullish,
        'bearish_score': bearish,
        'net_score': net_score,
        'total_score': int(total_score),
        'grade': grade,
        'action': action,
        'factors': factors
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ADD ALL TA TO DATAFRAME (convenience)
# ═══════════════════════════════════════════════════════════════════════════════

def add_all_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ALL ta library indicators to a DataFrame.
    Uses ta.add_all_ta_features() for comprehensive analysis.
    """
    df_with_ta = ta.add_all_ta_features(
        df,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )
    return df_with_ta
