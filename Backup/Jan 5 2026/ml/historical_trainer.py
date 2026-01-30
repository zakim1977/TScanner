"""
Historical Market Data Trainer
================================

Trains ML models on ACTUAL historical market data from Binance.
No need to wait for trade outcomes - we know what happened historically!

Process:
1. Fetch historical OHLCV data for multiple symbols
2. Calculate technical indicators at each candle
3. Simulate whale/retail positioning from price action
4. Create labels based on actual price movement
5. Train models on thousands of samples

Usage:
    python ml/historical_trainer.py
    python ml/historical_trainer.py --symbols BTCUSDT ETHUSDT --timeframe 15m --days 90
    python ml/historical_trainer.py --all-symbols --days 180
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.multi_model_trainer import train_multi_model, MODEL_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT'
]

ALL_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT',
    'DOTUSDT', 'ATOMUSDT', 'LTCUSDT', 'UNIUSDT', 'APTUSDT',
    'ARBUSDT', 'OPUSDT', 'NEARUSDT', 'FILUSDT', 'INJUSDT'
]

TIMEFRAME_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
    '1d': 1440, '1w': 10080
}

PROFIT_THRESHOLD_PCT = 1.5
LOSS_THRESHOLD_PCT = 1.0
LOOKAHEAD_CANDLES = 20


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str = '15m', limit: int = 1000, end_time: int = None) -> pd.DataFrame:
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    if end_time:
        params['endTime'] = end_time
    
    try:
        response = requests.get(url, params=params, timeout=10)
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
        print(f"  ⚠️ Error: {e}")
        return pd.DataFrame()


def fetch_historical_data(symbol: str, interval: str = '15m', days: int = 90) -> pd.DataFrame:
    print(f"  📊 {symbol} {interval} ({days}d)...", end=" ", flush=True)
    
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
        print("❌")
        return pd.DataFrame()
    
    combined = pd.concat(all_data).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    print(f"✅ {len(combined)}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # BB Squeeze Percentile
    df['BB_Squeeze_Pct'] = 100 - df['BB_Width'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    )
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Price changes
    df['price_change_1'] = df['Close'].pct_change(1) * 100
    df['price_change_5'] = df['Close'].pct_change(5) * 100
    df['price_change_20'] = df['Close'].pct_change(20) * 100
    
    # Volume
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-10)
    
    # Swing High/Low
    df['swing_high'] = df['High'].rolling(10, center=True).max()
    df['swing_low'] = df['Low'].rolling(10, center=True).min()
    
    # Position in range
    range_size = df['swing_high'] - df['swing_low']
    df['position_pct'] = ((df['Close'] - df['swing_low']) / (range_size + 1e-10)) * 100
    df['position_pct'] = df['position_pct'].clip(0, 100)
    
    # Trend
    df['trend'] = np.where(df['Close'] > df['SMA_50'], 1, np.where(df['Close'] < df['SMA_50'], -1, 0))
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED WHALE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_whale_positioning(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    buy_ratio_smooth = df['buy_ratio'].rolling(10).mean().fillna(0.5)
    whale_from_buyratio = buy_ratio_smooth * 100
    
    vol_percentile = df['BB_Width'].rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    accumulation_score = (1 - vol_percentile) * 30
    
    rsi = df['RSI'].fillna(50)
    rsi_whale_score = np.where(rsi < 30, 70, np.where(rsi > 70, 30, 50 + (50 - rsi) * 0.4))
    
    trend_score = np.where(df['trend'] == 1, 55, np.where(df['trend'] == -1, 45, 50))
    
    df['whale_pct'] = (
        whale_from_buyratio * 0.3 + accumulation_score * 0.2 +
        rsi_whale_score * 0.3 + trend_score * 0.2
    ).clip(20, 80)
    df['whale_pct'] = df['whale_pct'].rolling(5).mean().fillna(50)
    
    df['retail_pct'] = (100 - df['whale_pct'] + np.random.normal(0, 5, len(df))).clip(20, 80)
    df['divergence'] = df['whale_pct'] - df['retail_pct']
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_labels(df: pd.DataFrame, lookahead: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    
    labels = []
    for i in range(len(df) - lookahead):
        current_price = df['Close'].iloc[i]
        future_highs = df['High'].iloc[i+1:i+lookahead+1]
        future_lows = df['Low'].iloc[i+1:i+lookahead+1]
        
        max_up = (future_highs.max() - current_price) / current_price * 100
        max_down = (current_price - future_lows.min()) / current_price * 100
        
        if max_up >= PROFIT_THRESHOLD_PCT and max_up > max_down * 1.5:
            label = 2  # LONG
        elif max_down >= PROFIT_THRESHOLD_PCT and max_down > max_up * 1.5:
            label = 0  # SHORT
        else:
            label = 1  # WAIT
        
        labels.append(label)
    
    labels.extend([1] * lookahead)
    df['label'] = labels
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'whale_pct', 'retail_pct', 'whale_retail_divergence',
    'funding_rate', 'oi_change', 'oi_signal_encoded',
    'price_change_24h', 'price_change_1h', 'position_in_range',
    'distance_to_high', 'distance_to_low', 'range_size_pct',
    'ta_score', 'rsi', 'rsi_oversold', 'rsi_overbought',
    'trend_encoded', 'atr_pct', 'money_flow_encoded', 'volume_ratio',
    'at_bullish_ob', 'at_bearish_ob', 'near_support', 'near_resistance',
    'btc_correlation', 'btc_trend_encoded', 'fear_greed_normalized', 'is_weekend',
    'historical_win_rate', 'similar_setup_count', 'avg_historical_return',
    'explosion_score', 'explosion_ready', 'bb_squeeze_pct',
    'bb_width_pct', 'compression_duration', 'energy_loaded'
]


def extract_features_from_row(row: pd.Series) -> np.ndarray:
    features = np.zeros(len(FEATURE_NAMES))
    try:
        features[0] = row.get('whale_pct', 50)
        features[1] = row.get('retail_pct', 50)
        features[2] = row.get('divergence', 0)
        features[3] = 0
        features[4] = row.get('volume_ratio', 1) - 1
        features[5] = 1 if features[4] > 0.2 else (-1 if features[4] < -0.2 else 0)
        features[6] = row.get('price_change_20', 0)
        features[7] = row.get('price_change_5', 0)
        features[8] = row.get('position_pct', 50)
        
        swing_high = row.get('swing_high', row['Close'])
        swing_low = row.get('swing_low', row['Close'])
        features[9] = (swing_high - row['Close']) / row['Close'] * 100
        features[10] = (row['Close'] - swing_low) / row['Close'] * 100
        features[11] = (swing_high - swing_low) / row['Close'] * 100
        
        rsi = row.get('RSI', 50)
        features[12] = 50 + (rsi - 50) * 0.5
        features[13] = rsi
        features[14] = 1 if rsi < 30 else 0
        features[15] = 1 if rsi > 70 else 0
        features[16] = row.get('trend', 0)
        features[17] = row.get('ATR_pct', 2)
        
        buy_ratio = row.get('buy_ratio', 0.5)
        features[18] = 1 if buy_ratio > 0.55 else (-1 if buy_ratio < 0.45 else 0)
        features[19] = row.get('volume_ratio', 1)
        features[20] = 1 if features[8] < 20 else 0
        features[21] = 1 if features[8] > 80 else 0
        features[22] = 1 if features[8] < 30 else 0
        features[23] = 1 if features[8] > 70 else 0
        features[24] = 0.7
        features[25] = row.get('trend', 0)
        features[26] = 0.5
        features[27] = 0
        features[28] = 55
        features[29] = 10
        features[30] = 1.5
        
        bb_squeeze = row.get('BB_Squeeze_Pct', 50)
        features[31] = bb_squeeze * 0.8
        features[32] = 1 if bb_squeeze > 80 else 0
        features[33] = bb_squeeze
        features[34] = row.get('BB_Width', 3)
        features[35] = 0
        features[36] = 1 if bb_squeeze > 75 else 0
    except:
        pass
    return features


def create_training_dataset(symbols: List[str], interval: str, days: int) -> Tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    
    print(f"\n{'═' * 60}")
    print(f"FETCHING HISTORICAL DATA")
    print(f"{'═' * 60}")
    print(f"Symbols: {len(symbols)} | Timeframe: {interval} | Days: {days}")
    
    for symbol in symbols:
        df = fetch_historical_data(symbol, interval, days)
        if df.empty or len(df) < 200:
            continue
        
        df = calculate_indicators(df)
        df = simulate_whale_positioning(df)
        df = generate_labels(df, LOOKAHEAD_CANDLES)
        df = df.dropna()
        
        if len(df) < 100:
            continue
        
        for i in range(50, len(df) - LOOKAHEAD_CANDLES):
            row = df.iloc[i]
            features = extract_features_from_row(row)
            all_X.append(features)
            all_y.append(int(row['label']))
    
    if not all_X:
        return np.array([]), np.array([])
    
    X, y = np.array(all_X), np.array(all_y)
    print(f"\n{'─' * 60}")
    print(f"Dataset: {len(X)} samples")
    print(f"Labels: SHORT={sum(y==0)}, WAIT={sum(y==1)}, LONG={sum(y==2)}")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_on_historical_data(symbols: List[str] = None, interval: str = '15m', days: int = 90, save: bool = True) -> Dict:
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    
    print("\n" + "═" * 60)
    print("INVESTORIQ HISTORICAL ML TRAINER")
    print("═" * 60)
    print(f"Symbols: {len(symbols)} | Timeframe: {interval} | Days: {days}")
    
    X, y = create_training_dataset(symbols, interval, days)
    
    if len(X) < 100:
        print("\n❌ Not enough data. Try more symbols or days.")
        return {'success': False, 'error': 'Insufficient data'}
    
    print(f"\n{'═' * 60}")
    print("TRAINING ML MODELS")
    print(f"{'═' * 60}")
    
    result = train_multi_model(X, y, FEATURE_NAMES, save=save, save_all=save)
    
    if result['success']:
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'symbols': symbols,
            'interval': interval,
            'days': days,
            'n_samples': len(X),
            'label_distribution': {'SHORT': int(sum(y==0)), 'WAIT': int(sum(y==1)), 'LONG': int(sum(y==2))},
            'best_model': result['best_model'],
            'best_f1': result['best_f1'],
            'is_real_ml': True
        }
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'═' * 60}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'═' * 60}")
        print(f"Best Model: {result['best_model']}")
        print(f"F1 Score: {result['best_f1']:.1%}")
        print(f"Samples: {len(X)}")
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML on historical market data')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--all-symbols', action='store_true', help='Use all 20 symbols')
    parser.add_argument('--timeframe', '-t', default='15m', help='Timeframe (default: 15m)')
    parser.add_argument('--days', '-d', type=int, default=90, help='Days of history (default: 90)')
    parser.add_argument('--no-save', action='store_true', help='Do not save models')
    args = parser.parse_args()
    
    symbols = ALL_SYMBOLS if args.all_symbols else (args.symbols or DEFAULT_SYMBOLS)
    result = train_on_historical_data(symbols, args.timeframe, args.days, not args.no_save)
    
    if not result.get('success'):
        sys.exit(1)
