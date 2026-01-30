"""
Mode-Specific Historical ML Trainer
====================================

Trains SEPARATE ML models for each trading mode because:
- Scalp patterns ≠ Swing patterns
- A BB squeeze on 5m means something different than on 4h
- Hold times and target expectations vary by mode

Models Created:
- model_scalp.pkl      → 1m/5m timeframes
- model_daytrade.pkl   → 15m/1h timeframes
- model_swing.pkl      → 4h/1d timeframes
- model_investment.pkl → 1d/1w timeframes

Each model learns:
- Direction (LONG/SHORT/WAIT)
- Confidence (0-100%)
- ETA to TP1 (estimated candles)
- Optimal SL distance (ATR multiplier)

Usage:
    python ml/mode_specific_trainer.py                    # Train all modes
    python ml/mode_specific_trainer.py --mode daytrade    # Train specific mode
    python ml/mode_specific_trainer.py --days 180         # More history
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import time
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import ML libraries
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Mode definitions with their timeframes
MODE_CONFIG = {
    'scalp': {
        'timeframes': ['1m', '5m'],
        'lookahead_candles': 12,      # Look 12 candles ahead
        'profit_threshold': 0.5,       # 0.5% profit target
        'loss_threshold': 0.3,         # 0.3% stop loss
        'hold_time_minutes': 30,       # Typical hold
    },
    'daytrade': {
        'timeframes': ['15m', '1h'],
        'lookahead_candles': 16,       # ~4 hours on 15m
        'profit_threshold': 1.5,       # 1.5% profit target
        'loss_threshold': 1.0,         # 1% stop loss
        'hold_time_minutes': 240,      # 4 hours typical
    },
    'swing': {
        'timeframes': ['4h', '1d'],
        'lookahead_candles': 20,       # ~3-4 days on 4h
        'profit_threshold': 3.0,       # 3% profit target
        'loss_threshold': 2.0,         # 2% stop loss
        'hold_time_minutes': 2880,     # 2 days typical
    },
    'investment': {
        'timeframes': ['1d'],          # Only daily - weekly has too few candles
        'lookahead_candles': 14,       # 2 weeks ahead
        'profit_threshold': 8.0,       # 8% profit target
        'loss_threshold': 5.0,         # 5% stop loss
        'hold_time_minutes': 20160,    # 2 weeks typical
    }
}

# Symbols to train on
DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT'
]

EXTENDED_SYMBOLS = DEFAULT_SYMBOLS + [
    'DOTUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT',
    'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'FILUSDT', 'INJUSDT'
]


def fetch_top_futures_symbols(n: int = 50) -> List[str]:
    """
    Dynamically fetch top N futures pairs by volume.
    This ensures we train on whatever is currently most active/liquid.
    
    Benefits:
    - Always trains on the most relevant coins
    - Captures new trending coins automatically
    - Diverse mix: majors, alts, memecoins, new listings
    """
    try:
        # Fetch 24h ticker data from Binance Futures
        url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Filter USDT pairs and sort by volume
        usdt_pairs = [
            item for item in data 
            if item['symbol'].endswith('USDT') 
            and float(item['quoteVolume']) > 0
        ]
        
        # Sort by quote volume (descending)
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        # Get top N symbols
        symbols = [item['symbol'] for item in usdt_pairs[:n]]
        
        print(f"  📊 Fetched top {len(symbols)} futures pairs by volume", flush=True)
        return symbols
        
    except Exception as e:
        print(f"  ⚠️ Could not fetch dynamic symbols: {e}", flush=True)
        print(f"  📋 Using extended default list", flush=True)
        return EXTENDED_SYMBOLS


def get_diverse_training_symbols(n: int = 30) -> List[str]:
    """
    Get a diverse mix of symbols for robust training.
    
    Ensures we have:
    - Top volume coins (liquid, reliable data)
    - Mix of market cap tiers (large, mid, small)
    - Different sectors (L1s, DeFi, memecoins, AI, gaming)
    """
    try:
        all_symbols = fetch_top_futures_symbols(100)  # Get top 100
        
        if len(all_symbols) < 20:
            return EXTENDED_SYMBOLS
        
        # Always include BTC and ETH
        selected = ['BTCUSDT', 'ETHUSDT']
        
        # Add top 10 by volume (excluding BTC/ETH)
        for sym in all_symbols:
            if sym not in selected and len(selected) < 12:
                selected.append(sym)
        
        # Add some from middle of the list (mid-cap diversity)
        mid_range = all_symbols[20:50]
        for sym in mid_range[:8]:
            if sym not in selected:
                selected.append(sym)
        
        # Add some lower volume but still liquid (small-cap diversity)
        lower_range = all_symbols[50:80]
        for sym in lower_range[:5]:
            if sym not in selected:
                selected.append(sym)
        
        # Ensure we have at least n symbols
        for sym in all_symbols:
            if len(selected) >= n:
                break
            if sym not in selected:
                selected.append(sym)
        
        print(f"  🎯 Selected {len(selected)} diverse symbols for training", flush=True)
        return selected[:n]
        
    except Exception as e:
        print(f"  ⚠️ Error selecting diverse symbols: {e}", flush=True)
        return EXTENDED_SYMBOLS

TIMEFRAME_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
    '1d': 1440, '1w': 10080
}

# Feature names for the model
FEATURE_NAMES = [
    # Whale/Retail positioning
    'whale_pct', 'retail_pct', 'whale_retail_divergence',
    'position_in_range', 'position_early', 'position_late',
    
    # Open Interest
    'oi_change', 'oi_signal', 'oi_bullish', 'oi_bearish',
    
    # Technical indicators
    'rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
    'trend', 'trend_bullish', 'trend_bearish',
    'atr_pct',
    
    # Bollinger/Explosion
    'bb_squeeze_pct', 'bb_width_pct', 'energy_loaded', 'explosion_ready',
    
    # Price action
    'price_change_short', 'price_change_medium', 'price_change_long',
    'volume_ratio', 'buy_pressure',
    
    # SMC levels
    'near_support', 'near_resistance', 'at_demand_zone', 'at_supply_zone',
    
    # Market context
    'btc_correlation', 'market_trend',
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str, limit: int = 1000, end_time: int = None) -> pd.DataFrame:
    """Fetch klines from Binance API."""
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if end_time:
        params['endTime'] = end_time
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'taker_buy_base']:
            df[col] = df[col].astype(float)
        
        df['buy_ratio'] = df['taker_buy_base'] / df['Volume'].replace(0, 1)
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'buy_ratio']]
        df.set_index('timestamp', inplace=True)
        return df
        
    except Exception as e:
        return pd.DataFrame()


def fetch_historical_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch extended historical data."""
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    minutes_per_candle = TIMEFRAME_MINUTES.get(interval, 15)
    candles_needed = int((days * 24 * 60) / minutes_per_candle)
    
    while candles_needed > 0:
        limit = min(1000, candles_needed)
        df = fetch_binance_klines(symbol, interval, limit, end_time=end_time)
        if df.empty:
            break
        all_data.append(df)
        candles_needed -= len(df)
        if len(df) > 0:
            end_time = int(df.index[0].timestamp() * 1000) - 1
        time.sleep(0.1)
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators."""
    if df.empty or len(df) < 50:
        return df
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['ATR_pct'] = df['ATR'] / df['Close'] * 100
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] * 100
    
    # BB Squeeze (inverted - high = tight)
    df['BB_Squeeze_Pct'] = 100 - df['BB_Width'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    )
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Trend
    df['trend'] = np.where(df['Close'] > df['SMA_50'], 1, np.where(df['Close'] < df['SMA_50'], -1, 0))
    
    # Price changes (different windows for different contexts)
    df['price_change_5'] = df['Close'].pct_change(5) * 100
    df['price_change_20'] = df['Close'].pct_change(20) * 100
    df['price_change_50'] = df['Close'].pct_change(50) * 100
    
    # Volume
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-10)
    
    # Swing High/Low for position in range
    df['swing_high'] = df['High'].rolling(20, center=True).max()
    df['swing_low'] = df['Low'].rolling(20, center=True).min()
    range_size = df['swing_high'] - df['swing_low']
    df['position_pct'] = ((df['Close'] - df['swing_low']) / (range_size + 1e-10)) * 100
    df['position_pct'] = df['position_pct'].clip(0, 100)
    
    # Support/Resistance proximity
    df['near_support'] = (df['position_pct'] < 20).astype(int)
    df['near_resistance'] = (df['position_pct'] > 80).astype(int)
    
    return df


def load_real_whale_data(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Load REAL whale data from whale_history.db and merge with candle data.
    Falls back to simulation ONLY if no real data exists.
    """
    import sqlite3
    
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'whale_history.db')
    
    if not os.path.exists(db_path):
        print(f"    ⚠️ No whale_history.db - using simulated data")
        return simulate_whale_positioning(df)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Load real whale data for this symbol
        query = """
        SELECT timestamp, whale_long_pct, retail_long_pct, oi_change_24h, 
               funding_rate, position_in_range, mfi, rsi as whale_rsi,
               hit_tp1, hit_sl, max_favorable_pct, max_adverse_pct
        FROM whale_snapshots 
        WHERE symbol = ? 
          AND whale_long_pct IS NOT NULL
          AND whale_long_pct > 0
        ORDER BY timestamp
        """
        
        whale_df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        if whale_df.empty or len(whale_df) < 50:
            print(f"    ⚠️ Insufficient real data for {symbol} ({len(whale_df)} records) - using simulated")
            return simulate_whale_positioning(df)
        
        print(f"    ✅ Using REAL whale data: {len(whale_df)} records for {symbol}")
        
        # Parse timestamps (handle ISO8601 format from database)
        whale_df['timestamp'] = pd.to_datetime(whale_df['timestamp'], format='ISO8601')
        whale_df = whale_df.sort_values('timestamp')
        
        # Merge by closest timestamp
        df = df.copy()
        
        # Handle timestamp - could be index, column, or both
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(names='timestamp')
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df.index)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Use merge_asof to match closest timestamps
        df = df.sort_values('timestamp')
        whale_df = whale_df.sort_values('timestamp')
        
        merged = pd.merge_asof(
            df, 
            whale_df[['timestamp', 'whale_long_pct', 'retail_long_pct', 'oi_change_24h', 
                     'funding_rate', 'position_in_range', 'mfi', 'whale_rsi',
                     'hit_tp1', 'hit_sl', 'max_favorable_pct', 'max_adverse_pct']],
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('4h')  # Match within 4 hours
        )
        
        # Rename to expected columns
        merged['whale_pct'] = merged['whale_long_pct'].fillna(50)
        merged['retail_pct'] = merged['retail_long_pct'].fillna(50)
        merged['divergence'] = merged['whale_pct'] - merged['retail_pct']
        
        # Fill any remaining NaN from unmatched rows
        merged['whale_pct'] = merged['whale_pct'].ffill().fillna(50)
        merged['retail_pct'] = merged['retail_pct'].ffill().fillna(50)
        merged['divergence'] = merged['whale_pct'] - merged['retail_pct']
        
        # Use real outcome data if available for labels
        if 'hit_tp1' in merged.columns:
            merged['real_outcome_available'] = merged['hit_tp1'].notna()
        
        return merged
        
    except Exception as e:
        print(f"    ⚠️ Error loading real data: {e} - using simulated")
        return simulate_whale_positioning(df)


def simulate_whale_positioning(df: pd.DataFrame) -> pd.DataFrame:
    """
    FALLBACK: Simulate whale/retail positioning from price action.
    Only used when no real data exists in whale_history.db
    """
    if df.empty:
        return df
    
    # Smooth buy ratio
    buy_ratio_smooth = df['buy_ratio'].rolling(10).mean().fillna(0.5)
    
    # Accumulation detection (low volatility periods)
    vol_percentile = df['BB_Width'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    
    # RSI-based positioning
    rsi = df['RSI'].fillna(50)
    rsi_whale_score = np.where(rsi < 30, 70, np.where(rsi > 70, 30, 50 + (50 - rsi) * 0.4))
    
    # Trend component
    trend_score = np.where(df['trend'] == 1, 60, np.where(df['trend'] == -1, 40, 50))
    
    # Combine
    df['whale_pct'] = (
        buy_ratio_smooth * 100 * 0.3 +
        (1 - vol_percentile) * 30 * 0.2 +
        rsi_whale_score * 0.3 +
        trend_score * 0.2
    ).clip(20, 80)
    df['whale_pct'] = df['whale_pct'].rolling(5).mean().fillna(50)
    
    # Retail often opposite
    df['retail_pct'] = (100 - df['whale_pct'] + np.random.normal(0, 3, len(df))).clip(20, 80)
    df['divergence'] = df['whale_pct'] - df['retail_pct']
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL GENERATION (Mode-Specific)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_labels_for_mode(
    df: pd.DataFrame, 
    mode: str,
    lookahead: int,
    profit_threshold: float,
    loss_threshold: float
) -> pd.DataFrame:
    """
    Generate labels based on what ACTUALLY happened.
    
    Labels:
    - direction: 2=LONG, 1=WAIT, 0=SHORT
    - candles_to_tp: How many candles until profit target hit (for ETA)
    - max_adverse: Maximum drawdown before profit (for SL calibration)
    """
    if df.empty:
        return df
    
    directions = []
    candles_to_tp = []
    max_adverse_moves = []
    
    for i in range(len(df) - lookahead):
        current_price = df['Close'].iloc[i]
        
        # Look at future price action
        future_highs = df['High'].iloc[i+1:i+lookahead+1].values
        future_lows = df['Low'].iloc[i+1:i+lookahead+1].values
        future_closes = df['Close'].iloc[i+1:i+lookahead+1].values
        
        # Calculate moves
        max_up_pct = ((future_highs.max() - current_price) / current_price) * 100
        max_down_pct = ((current_price - future_lows.min()) / current_price) * 100
        
        # Find when TP was hit (if ever)
        tp_candle_long = None
        tp_candle_short = None
        
        for j, (high, low) in enumerate(zip(future_highs, future_lows)):
            up_pct = ((high - current_price) / current_price) * 100
            down_pct = ((current_price - low) / current_price) * 100
            
            if tp_candle_long is None and up_pct >= profit_threshold:
                tp_candle_long = j + 1
            if tp_candle_short is None and down_pct >= profit_threshold:
                tp_candle_short = j + 1
        
        # Determine direction based on which target hit first and cleanly
        if max_up_pct >= profit_threshold and max_up_pct > max_down_pct * 1.3:
            direction = 2  # LONG
            candles = tp_candle_long if tp_candle_long else lookahead
            adverse = max_down_pct
        elif max_down_pct >= profit_threshold and max_down_pct > max_up_pct * 1.3:
            direction = 0  # SHORT
            candles = tp_candle_short if tp_candle_short else lookahead
            adverse = max_up_pct
        else:
            direction = 1  # WAIT (choppy)
            candles = lookahead
            adverse = max(max_up_pct, max_down_pct)
        
        directions.append(direction)
        candles_to_tp.append(candles)
        max_adverse_moves.append(adverse)
    
    # Pad end
    directions.extend([1] * lookahead)
    candles_to_tp.extend([lookahead] * lookahead)
    max_adverse_moves.extend([0] * lookahead)
    
    df['direction'] = directions
    df['candles_to_tp'] = candles_to_tp
    df['max_adverse'] = max_adverse_moves
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_row(row: pd.Series) -> np.ndarray:
    """Extract feature vector from a single row."""
    features = np.zeros(len(FEATURE_NAMES))
    
    try:
        # Whale/Retail
        features[0] = row.get('whale_pct', 50)
        features[1] = row.get('retail_pct', 50)
        features[2] = row.get('divergence', 0)
        features[3] = row.get('position_pct', 50)
        features[4] = 1 if features[3] < 30 else 0  # Early
        features[5] = 1 if features[3] > 70 else 0  # Late
        
        # OI (simulated from volume)
        vol_ratio = row.get('volume_ratio', 1)
        features[6] = vol_ratio - 1  # oi_change proxy
        features[7] = 1 if vol_ratio > 1.2 else (-1 if vol_ratio < 0.8 else 0)
        features[8] = 1 if vol_ratio > 1.3 else 0  # OI bullish
        features[9] = 1 if vol_ratio < 0.7 else 0  # OI bearish
        
        # Technical
        rsi = row.get('RSI', 50)
        features[10] = rsi
        features[11] = 1 if rsi < 30 else 0
        features[12] = 1 if rsi > 70 else 0
        features[13] = 1 if 40 <= rsi <= 60 else 0
        features[14] = row.get('trend', 0)
        features[15] = 1 if features[14] == 1 else 0
        features[16] = 1 if features[14] == -1 else 0
        features[17] = row.get('ATR_pct', 2)
        
        # Bollinger/Explosion
        bb_squeeze = row.get('BB_Squeeze_Pct', 50)
        features[18] = bb_squeeze
        features[19] = row.get('BB_Width', 3)
        features[20] = 1 if bb_squeeze > 75 else 0  # Energy loaded
        features[21] = 1 if bb_squeeze > 85 else 0  # Explosion ready
        
        # Price action
        features[22] = row.get('price_change_5', 0)
        features[23] = row.get('price_change_20', 0)
        features[24] = row.get('price_change_50', 0)
        features[25] = row.get('volume_ratio', 1)
        features[26] = row.get('buy_ratio', 0.5)
        
        # SMC
        features[27] = row.get('near_support', 0)
        features[28] = row.get('near_resistance', 0)
        features[29] = 1 if features[3] < 15 else 0  # At demand
        features[30] = 1 if features[3] > 85 else 0  # At supply
        
        # Market context (placeholder)
        features[31] = 0.7  # BTC correlation
        features[32] = features[14]  # Market trend = trend
        
    except Exception:
        pass
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataset_for_mode(
    symbols: List[str],
    mode: str,
    days: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset for a specific trading mode.
    
    Returns:
        X: Features
        y_direction: Direction labels (0=SHORT, 1=WAIT, 2=LONG)
        y_eta: Candles to TP
        y_sl: Max adverse move (for SL calibration)
    """
    config = MODE_CONFIG[mode]
    timeframes = config['timeframes']
    lookahead = config['lookahead_candles']
    profit_thresh = config['profit_threshold']
    loss_thresh = config['loss_threshold']
    
    all_X = []
    all_y_dir = []
    all_y_eta = []
    all_y_sl = []
    
    print(f"\n{'─' * 50}", flush=True)
    print(f"  Mode: {mode.upper()}", flush=True)
    print(f"  Timeframes: {timeframes}", flush=True)
    print(f"  Targets: TP={profit_thresh}%, SL={loss_thresh}%", flush=True)
    print(f"{'─' * 50}", flush=True)
    
    total_symbols = len(symbols) * len(timeframes)
    processed = 0
    
    for timeframe in timeframes:
        print(f"\n  📊 Fetching {timeframe} data:", flush=True)
        
        for symbol in symbols:
            processed += 1
            pct = int((processed / total_symbols) * 100)
            print(f"    [{pct:3d}%] {symbol}...", end=" ", flush=True)
            
            # Fetch data
            df = fetch_historical_data(symbol, timeframe, days)
            if df.empty or len(df) < 200:
                print("❌ No data", flush=True)
                continue
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Load REAL whale data from whale_history.db (falls back to simulation if unavailable)
            df = load_real_whale_data(symbol, df)
            
            # Generate labels
            df = generate_labels_for_mode(df, mode, lookahead, profit_thresh, loss_thresh)
            
            # Drop NaN
            df = df.dropna()
            
            if len(df) < 100:
                print("❌ Insufficient", flush=True)
                continue
            
            # Extract features
            count = 0
            for i in range(50, len(df) - lookahead):
                row = df.iloc[i]
                features = extract_features_from_row(row)
                
                all_X.append(features)
                all_y_dir.append(int(row['direction']))
                all_y_eta.append(int(row['candles_to_tp']))
                all_y_sl.append(float(row['max_adverse']))
                count += 1
            
            print(f"✅ {count:,} samples", flush=True)
    
    if not all_X:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return np.array(all_X), np.array(all_y_dir), np.array(all_y_eta), np.array(all_y_sl)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_mode_model(
    X: np.ndarray,
    y_direction: np.ndarray,
    y_eta: np.ndarray,
    y_sl: np.ndarray,
    mode: str
) -> Dict:
    """
    Train models for a specific trading mode.
    
    Creates:
    - Direction classifier (LONG/SHORT/WAIT)
    - ETA regressor (candles to TP)
    - SL regressor (optimal stop loss distance)
    """
    if not HAS_SKLEARN:
        return {'success': False, 'error': 'sklearn not installed'}
    
    print(f"\n{'─' * 50}", flush=True)
    print(f"  🤖 Training {mode.upper()} models...", flush=True)
    print(f"  📊 Samples: {len(X):,}", flush=True)
    print(f"  📈 Distribution: SHORT={sum(y_direction==0):,}, WAIT={sum(y_direction==1):,}, LONG={sum(y_direction==2):,}", flush=True)
    print(f"{'─' * 50}", flush=True)
    
    # Split data
    X_train, X_test, y_dir_train, y_dir_test = train_test_split(
        X, y_direction, test_size=0.2, random_state=42, stratify=y_direction
    )
    _, _, y_eta_train, y_eta_test = train_test_split(
        X, y_eta, test_size=0.2, random_state=42
    )
    _, _, y_sl_train, y_sl_test = train_test_split(
        X, y_sl, test_size=0.2, random_state=42
    )
    
    # Scale features
    print(f"  [1/4] Scaling features...", end=" ", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✅", flush=True)
    
    # Train Direction Classifier
    print(f"  [2/4] Training direction classifier...", end=" ", flush=True)
    if HAS_LGBM:
        dir_model = LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            verbose=-1,
            force_col_wise=True
        )
    else:
        dir_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    
    dir_model.fit(X_train_scaled, y_dir_train)
    dir_preds = dir_model.predict(X_test_scaled)
    dir_f1 = f1_score(y_dir_test, dir_preds, average='weighted')
    dir_acc = accuracy_score(y_dir_test, dir_preds)
    print(f"✅ F1={dir_f1:.1%}, Acc={dir_acc:.1%}", flush=True)
    
    # Train ETA Regressor
    print(f"  [3/4] Training ETA regressor...", end=" ", flush=True)
    if HAS_LGBM:
        eta_model = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1)
    else:
        eta_model = RandomForestRegressor(n_estimators=100, max_depth=6)
    
    eta_model.fit(X_train_scaled, y_eta_train)
    eta_preds = eta_model.predict(X_test_scaled)
    eta_mae = np.mean(np.abs(eta_preds - y_eta_test))
    print(f"✅ MAE={eta_mae:.1f} candles", flush=True)
    
    # Train SL Regressor
    print(f"  [4/4] Training SL regressor...", end=" ", flush=True)
    if HAS_LGBM:
        sl_model = LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1)
    else:
        sl_model = RandomForestRegressor(n_estimators=100, max_depth=6)
    
    sl_model.fit(X_train_scaled, y_sl_train)
    sl_preds = sl_model.predict(X_test_scaled)
    sl_mae = np.mean(np.abs(sl_preds - y_sl_test))
    print(f"✅ MAE={sl_mae:.2f}%", flush=True)
    
    print(f"\n  🎉 {mode.upper()} training complete!", flush=True)
    
    return {
        'success': True,
        'mode': mode,
        'direction_model': dir_model,
        'eta_model': eta_model,
        'sl_model': sl_model,
        'scaler': scaler,
        'metrics': {
            'direction_f1': dir_f1,
            'direction_accuracy': dir_acc,
            'eta_mae': eta_mae,
            'sl_mae': sl_mae,
            'n_samples': len(X),
            'label_distribution': {
                'SHORT': int(sum(y_direction == 0)),
                'WAIT': int(sum(y_direction == 1)),
                'LONG': int(sum(y_direction == 2))
            }
        }
    }


def save_mode_model(result: Dict, mode: str):
    """Save trained models for a specific mode."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save models
    with open(os.path.join(MODEL_DIR, f'direction_model_{mode}.pkl'), 'wb') as f:
        pickle.dump(result['direction_model'], f)
    
    with open(os.path.join(MODEL_DIR, f'eta_model_{mode}.pkl'), 'wb') as f:
        pickle.dump(result['eta_model'], f)
    
    with open(os.path.join(MODEL_DIR, f'sl_model_{mode}.pkl'), 'wb') as f:
        pickle.dump(result['sl_model'], f)
    
    with open(os.path.join(MODEL_DIR, f'scaler_{mode}.pkl'), 'wb') as f:
        pickle.dump(result['scaler'], f)
    
    # Save metadata
    metadata = {
        'mode': mode,
        'trained_at': datetime.now().isoformat(),
        'is_real_ml': True,
        'model_type': 'mode_specific',
        'feature_names': FEATURE_NAMES,
        'n_features': len(FEATURE_NAMES),
        'metrics': result['metrics'],
        'config': MODE_CONFIG[mode]
    }
    
    with open(os.path.join(MODEL_DIR, f'metadata_{mode}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved to: {MODEL_DIR}/[direction|eta|sl]_model_{mode}.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_modes(symbols: List[str] = None, days: int = 90, modes: List[str] = None):
    """Train models for all (or specified) trading modes."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    if modes is None:
        modes = list(MODE_CONFIG.keys())
    
    print("\n" + "═" * 60)
    print("INVESTORIQ MODE-SPECIFIC ML TRAINER")
    print("═" * 60)
    print(f"Symbols: {len(symbols)}")
    print(f"Days: {days}")
    print(f"Modes: {modes}")
    print("═" * 60)
    
    results = {}
    
    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"TRAINING MODE: {mode.upper()}")
        print(f"{'─' * 60}")
        
        # Create dataset
        X, y_dir, y_eta, y_sl = create_dataset_for_mode(symbols, mode, days)
        
        if len(X) < 100:
            print(f"\n  ❌ Not enough data for {mode}")
            results[mode] = {'success': False, 'error': 'Insufficient data'}
            continue
        
        # Train
        result = train_mode_model(X, y_dir, y_eta, y_sl, mode)
        
        if result['success']:
            save_mode_model(result, mode)
            results[mode] = result['metrics']
            print(f"\n  ✅ {mode.upper()} training complete!")
        else:
            results[mode] = {'success': False, 'error': result.get('error', 'Unknown')}
    
    # Summary
    print("\n" + "═" * 60)
    print("TRAINING SUMMARY")
    print("═" * 60)
    
    for mode, metrics in results.items():
        if isinstance(metrics, dict) and 'direction_f1' in metrics:
            print(f"  {mode.upper():12} F1={metrics['direction_f1']:.1%}  ETA±{metrics['eta_mae']:.1f}  SL±{metrics['sl_mae']:.2f}%  ({metrics['n_samples']} samples)")
        else:
            print(f"  {mode.upper():12} ❌ Failed")
    
    print("═" * 60)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train mode-specific ML models')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--days', '-d', type=int, default=90, help='Days of history')
    parser.add_argument('--mode', '-m', choices=['scalp', 'daytrade', 'swing', 'investment'], 
                       help='Train only specific mode')
    args = parser.parse_args()
    
    symbols = args.symbols or DEFAULT_SYMBOLS
    modes = [args.mode] if args.mode else None
    
    train_all_modes(symbols, args.days, modes)