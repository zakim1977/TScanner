"""
Stock & ETF ML Trainer
=======================

Trains ML models on stock and ETF historical data using yfinance.
Stocks/ETFs have YEARS of history - perfect for Investment mode!

Usage:
    python ml/stock_trainer.py                      # Default stocks
    python ml/stock_trainer.py --etfs              # Train on ETFs
    python ml/stock_trainer.py --symbols AAPL MSFT GOOGL  # Custom
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️ yfinance not installed. Run: pip install yfinance")

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

# Default stock symbols (mix of sectors)
DEFAULT_STOCKS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    # Finance
    'JPM', 'BAC', 'GS',
    # Healthcare
    'JNJ', 'PFE', 'UNH',
    # Consumer
    'WMT', 'KO', 'MCD',
    # Energy
    'XOM', 'CVX',
    # Industrial
    'CAT', 'BA'
]

# Default ETFs
DEFAULT_ETFS = [
    'SPY', 'QQQ', 'IWM', 'DIA',      # Index ETFs
    'XLF', 'XLK', 'XLE', 'XLV',      # Sector ETFs
    'GLD', 'SLV',                     # Commodities
    'TLT', 'HYG',                     # Bonds
    'VTI', 'VOO', 'VEA', 'VWO',      # Vanguard
    'ARKK', 'ARKG',                   # ARK
]

# Mode config for stocks (different from crypto)
STOCK_MODE_CONFIG = {
    'daytrade': {
        'interval': '15m',
        'period': '60d',              # yfinance limit for intraday
        'lookahead_candles': 16,
        'profit_threshold': 1.0,      # Stocks move less than crypto
        'loss_threshold': 0.7,
    },
    'swing': {
        'interval': '1d',
        'period': '2y',
        'lookahead_candles': 10,      # 2 weeks
        'profit_threshold': 3.0,
        'loss_threshold': 2.0,
    },
    'investment': {
        'interval': '1d',
        'period': '5y',               # 5 years of daily data!
        'lookahead_candles': 20,      # 1 month
        'profit_threshold': 10.0,
        'loss_threshold': 6.0,
    }
}

# Feature names (same structure as crypto, plus institutional)
FEATURE_NAMES = [
    'rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
    'trend', 'trend_bullish', 'trend_bearish',
    'atr_pct', 'bb_squeeze_pct', 'bb_width_pct',
    'price_change_short', 'price_change_medium', 'price_change_long',
    'volume_ratio', 'volume_trend',
    'above_sma20', 'above_sma50', 'above_sma200',
    'macd_signal', 'macd_histogram',
    'position_pct', 'near_support', 'near_resistance',
    'momentum_5', 'momentum_20',
    'volatility_rank', 'volume_rank',
    # NEW: Institutional features from stock_history
    'congress_score', 'insider_score', 'short_interest', 'institutional_diff',
]


# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL DATA LOADING (REAL DATA from stock_history/*.json)
# ═══════════════════════════════════════════════════════════════════════════════

def load_institutional_data(symbol: str) -> Dict:
    """
    Load REAL institutional data from stock_history JSON files.
    Returns dict with congress_score, insider_score, short_interest_pct, etc.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stock_history')
    json_path = os.path.join(data_dir, f'{symbol}_history.json')
    
    if not os.path.exists(json_path):
        return {
            'congress_score': 50,
            'insider_score': 50,
            'short_interest_pct': 0,
            'combined_score': 50,
            'has_real_data': False
        }
    
    try:
        with open(json_path, 'r') as f:
            records = json.load(f)
        
        if not records:
            return {
                'congress_score': 50,
                'insider_score': 50,
                'short_interest_pct': 0,
                'combined_score': 50,
                'has_real_data': False
            }
        
        # Get most recent record
        latest = records[-1]
        
        return {
            'congress_score': latest.get('congress_score', 50),
            'insider_score': latest.get('insider_score', 50),
            'short_interest_pct': latest.get('short_interest_pct', 0),
            'combined_score': latest.get('combined_score', 50),
            'has_real_data': True,
            'records': records  # All historical records for time-series merge
        }
    except Exception as e:
        return {
            'congress_score': 50,
            'insider_score': 50,
            'short_interest_pct': 0,
            'combined_score': 50,
            'has_real_data': False
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING - Use the SAME function as Scanner/Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_stock_data_for_training(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch historical data using the SAME method as Scanner/Single Analysis.
    This avoids SSL issues by using direct API calls instead of yfinance library.
    """
    try:
        # Import the working fetch function from data_fetcher
        from core.data_fetcher import fetch_stock_data as fetch_from_core
        
        # Convert period to limit (number of candles)
        # For intraday (15m), need more candles to cover the period
        period_to_limit = {
            '60d': 4000,    # 60 days × 26 candles/day (6.5h × 4) = 1560, add buffer
            '90d': 6000,    # 90 days worth
            '180d': 180,    # daily
            '1y': 365,      # daily
            '2y': 730,      # daily
            '5y': 1825,     # daily
        }
        limit = period_to_limit.get(period, 365)
        
        # Use the working fetch function
        df = fetch_from_core(symbol, interval, limit)
        
        if df is None or df.empty:
            print(f"Empty response from fetch", flush=True)
            return pd.DataFrame()
        
        # Handle column names (might be lowercase or uppercase)
        df.columns = [col.capitalize() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Missing columns: {missing}. Available: {list(df.columns)}", flush=True)
            return pd.DataFrame()
        
        # Reset index if DateTime is the index
        if 'DateTime' in df.columns:
            df = df.set_index('DateTime')
        
        result = df[required_cols].copy()
        print(f"✅ {len(result)} samples", flush=True)
        return result
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


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
    
    # BB Squeeze
    df['BB_Squeeze_Pct'] = 100 - df['BB_Width'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    )
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Trend
    df['trend'] = np.where(df['Close'] > df['SMA_50'], 1, 
                          np.where(df['Close'] < df['SMA_50'], -1, 0))
    
    # Price changes
    df['price_change_5'] = df['Close'].pct_change(5) * 100
    df['price_change_20'] = df['Close'].pct_change(20) * 100
    df['price_change_50'] = df['Close'].pct_change(50) * 100
    
    # Volume
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-10)
    df['volume_trend'] = df['Volume'].rolling(10).mean() / (df['Volume'].rolling(30).mean() + 1e-10)
    
    # Position in range
    df['swing_high'] = df['High'].rolling(20, center=True).max()
    df['swing_low'] = df['Low'].rolling(20, center=True).min()
    range_size = df['swing_high'] - df['swing_low']
    df['position_pct'] = ((df['Close'] - df['swing_low']) / (range_size + 1e-10)) * 100
    df['position_pct'] = df['position_pct'].clip(0, 100)
    
    # Momentum
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Volatility rank
    df['volatility_rank'] = df['ATR_pct'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    
    # Volume rank
    df['volume_rank'] = df['volume_ratio'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_row(row: pd.Series, institutional: Dict = None) -> np.ndarray:
    """Extract feature vector from a single row, including institutional data."""
    features = np.zeros(len(FEATURE_NAMES))
    
    try:
        # RSI features
        rsi = row.get('RSI', 50)
        features[0] = rsi
        features[1] = 1 if rsi < 30 else 0
        features[2] = 1 if rsi > 70 else 0
        features[3] = 1 if 40 <= rsi <= 60 else 0
        
        # Trend features
        trend = row.get('trend', 0)
        features[4] = trend
        features[5] = 1 if trend == 1 else 0
        features[6] = 1 if trend == -1 else 0
        
        # Volatility
        features[7] = row.get('ATR_pct', 2)
        features[8] = row.get('BB_Squeeze_Pct', 50)
        features[9] = row.get('BB_Width', 3)
        
        # Price changes
        features[10] = row.get('price_change_5', 0)
        features[11] = row.get('price_change_20', 0)
        features[12] = row.get('price_change_50', 0)
        
        # Volume
        features[13] = row.get('volume_ratio', 1)
        features[14] = row.get('volume_trend', 1)
        
        # MA position
        close = row.get('Close', 0)
        features[15] = 1 if close > row.get('SMA_20', close) else 0
        features[16] = 1 if close > row.get('SMA_50', close) else 0
        features[17] = 1 if close > row.get('SMA_200', close) else 0
        
        # MACD
        features[18] = 1 if row.get('MACD', 0) > row.get('MACD_Signal', 0) else -1
        features[19] = row.get('MACD_Hist', 0)
        
        # Position
        features[20] = row.get('position_pct', 50)
        features[21] = 1 if features[20] < 20 else 0  # Near support
        features[22] = 1 if features[20] > 80 else 0  # Near resistance
        
        # Momentum
        features[23] = row.get('momentum_5', 0) * 100
        features[24] = row.get('momentum_20', 0) * 100
        
        # Ranks
        features[25] = row.get('volatility_rank', 0.5)
        features[26] = row.get('volume_rank', 0.5)
        
        # NEW: Institutional features (REAL DATA from stock_history)
        if institutional:
            features[27] = institutional.get('congress_score', 50)
            features[28] = institutional.get('insider_score', 50)
            features[29] = institutional.get('short_interest_pct', 0)
            features[30] = features[27] - features[28]  # institutional_diff
        else:
            features[27] = 50  # congress_score
            features[28] = 50  # insider_score
            features[29] = 0   # short_interest
            features[30] = 0   # institutional_diff
        
    except Exception:
        pass
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_labels(
    df: pd.DataFrame,
    lookahead: int,
    profit_threshold: float,
    loss_threshold: float
) -> pd.DataFrame:
    """Generate labels based on actual outcomes."""
    if df.empty:
        return df
    
    directions = []
    candles_to_tp = []
    max_adverse_moves = []
    
    for i in range(len(df) - lookahead):
        current_price = df['Close'].iloc[i]
        
        future_highs = df['High'].iloc[i+1:i+lookahead+1].values
        future_lows = df['Low'].iloc[i+1:i+lookahead+1].values
        
        max_up_pct = ((future_highs.max() - current_price) / current_price) * 100
        max_down_pct = ((current_price - future_lows.min()) / current_price) * 100
        
        # Find TP candle
        tp_candle_long = None
        tp_candle_short = None
        
        for j, (high, low) in enumerate(zip(future_highs, future_lows)):
            up_pct = ((high - current_price) / current_price) * 100
            down_pct = ((current_price - low) / current_price) * 100
            
            if tp_candle_long is None and up_pct >= profit_threshold:
                tp_candle_long = j + 1
            if tp_candle_short is None and down_pct >= profit_threshold:
                tp_candle_short = j + 1
        
        # Determine direction
        if max_up_pct >= profit_threshold and max_up_pct > max_down_pct * 1.3:
            direction = 2  # LONG
            candles = tp_candle_long if tp_candle_long else lookahead
            adverse = max_down_pct
        elif max_down_pct >= profit_threshold and max_down_pct > max_up_pct * 1.3:
            direction = 0  # SHORT
            candles = tp_candle_short if tp_candle_short else lookahead
            adverse = max_up_pct
        else:
            direction = 1  # WAIT
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
# DATASET CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_dataset_for_stocks(
    symbols: List[str],
    mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create training dataset for stocks/ETFs."""
    config = STOCK_MODE_CONFIG[mode]
    interval = config['interval']
    period = config['period']
    lookahead = config['lookahead_candles']
    profit_thresh = config['profit_threshold']
    loss_thresh = config['loss_threshold']
    
    all_X = []
    all_y_dir = []
    all_y_eta = []
    all_y_sl = []
    
    print(f"\n{'─' * 50}", flush=True)
    print(f"  Mode: {mode.upper()} (Stocks/ETFs)", flush=True)
    print(f"  Interval: {interval}, Period: {period}", flush=True)
    print(f"  Targets: TP={profit_thresh}%, SL={loss_thresh}%", flush=True)
    print(f"{'─' * 50}", flush=True)
    
    total_symbols = len(symbols)
    
    for idx, symbol in enumerate(symbols):
        pct = int(((idx + 1) / total_symbols) * 100)
        print(f"  [{pct:3d}%] {symbol}...", end=" ", flush=True)
        
        # Fetch data using SAME method as Scanner/Analysis
        df = fetch_stock_data_for_training(symbol, interval, period)
        if df is None or df.empty:
            print("❌ Fetch failed", flush=True)
            continue
        if len(df) < 100:
            print(f"❌ Only {len(df)} rows (need 100+)", flush=True)
            continue
        
        # Load REAL institutional data from stock_history
        institutional = load_institutional_data(symbol)
        if institutional.get('has_real_data'):
            print(f"📊", end=" ", flush=True)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Generate labels
        df = generate_labels(df, lookahead, profit_thresh, loss_thresh)
        
        # Drop NaN
        df = df.dropna()
        
        if len(df) < 50:
            print("❌ Insufficient", flush=True)
            continue
        
        # Extract features (with institutional data)
        count = 0
        for i in range(30, len(df) - lookahead):
            row = df.iloc[i]
            features = extract_features_from_row(row, institutional)
            
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

def train_stock_model(
    X: np.ndarray,
    y_direction: np.ndarray,
    y_eta: np.ndarray,
    y_sl: np.ndarray,
    mode: str,
    asset_type: str = 'stock'
) -> Dict:
    """Train models for stocks/ETFs."""
    if not HAS_SKLEARN:
        return {'success': False, 'error': 'sklearn not installed'}
    
    print(f"\n{'─' * 50}", flush=True)
    print(f"  🤖 Training {mode.upper()} model ({asset_type})...", flush=True)
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
            n_estimators=200, max_depth=8, learning_rate=0.1,
            num_leaves=31, verbose=-1, force_col_wise=True
        )
    else:
        dir_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1
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
    
    print(f"\n  🎉 {mode.upper()} ({asset_type}) training complete!", flush=True)
    
    return {
        'success': True,
        'mode': mode,
        'asset_type': asset_type,
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


def save_stock_model(result: Dict, mode: str, asset_type: str = 'stock'):
    """Save trained models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    suffix = f"{mode}_{asset_type}"
    
    with open(os.path.join(MODEL_DIR, f'direction_model_{suffix}.pkl'), 'wb') as f:
        pickle.dump(result['direction_model'], f)
    
    with open(os.path.join(MODEL_DIR, f'eta_model_{suffix}.pkl'), 'wb') as f:
        pickle.dump(result['eta_model'], f)
    
    with open(os.path.join(MODEL_DIR, f'sl_model_{suffix}.pkl'), 'wb') as f:
        pickle.dump(result['sl_model'], f)
    
    with open(os.path.join(MODEL_DIR, f'scaler_{suffix}.pkl'), 'wb') as f:
        pickle.dump(result['scaler'], f)
    
    metadata = {
        'mode': mode,
        'asset_type': asset_type,
        'trained_at': datetime.now().isoformat(),
        'is_real_ml': True,
        'model_type': f'{asset_type}_specific',
        'feature_names': FEATURE_NAMES,
        'n_features': len(FEATURE_NAMES),
        'metrics': result['metrics'],
        'config': STOCK_MODE_CONFIG[mode]
    }
    
    with open(os.path.join(MODEL_DIR, f'metadata_{suffix}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  💾 Saved to: {MODEL_DIR}/*_{suffix}.pkl", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train_stocks(symbols: List[str] = None, modes: List[str] = None, asset_type: str = 'stock'):
    """Train models for stocks/ETFs."""
    if not HAS_YFINANCE:
        print("❌ yfinance not installed!")
        return {}
    
    if symbols is None:
        symbols = DEFAULT_STOCKS if asset_type == 'stock' else DEFAULT_ETFS
    if modes is None:
        modes = ['swing', 'investment']  # Most relevant for stocks
    
    print("\n" + "═" * 60)
    print(f"INVESTORIQ STOCK/ETF ML TRAINER")
    print("═" * 60)
    print(f"Asset Type: {asset_type.upper()}")
    print(f"Symbols: {len(symbols)}")
    print(f"Modes: {modes}")
    print("═" * 60)
    
    results = {}
    
    for mode in modes:
        if mode not in STOCK_MODE_CONFIG:
            print(f"⚠️ Mode {mode} not supported for stocks")
            continue
        
        # Create dataset
        X, y_dir, y_eta, y_sl = create_dataset_for_stocks(symbols, mode)
        
        if len(X) < 100:
            print(f"\n  ❌ Not enough data for {mode}")
            results[mode] = {'success': False, 'error': 'Insufficient data'}
            continue
        
        # Train
        result = train_stock_model(X, y_dir, y_eta, y_sl, mode, asset_type)
        
        if result['success']:
            save_stock_model(result, mode, asset_type)
            results[mode] = result['metrics']
        else:
            results[mode] = {'success': False, 'error': result.get('error', 'Unknown')}
    
    # Summary
    print("\n" + "═" * 60)
    print("TRAINING SUMMARY")
    print("═" * 60)
    
    for mode, metrics in results.items():
        if isinstance(metrics, dict) and 'direction_f1' in metrics:
            print(f"  {mode.upper():12} F1={metrics['direction_f1']:.1%}  ETA±{metrics['eta_mae']:.1f}  SL±{metrics['sl_mae']:.2f}%  ({metrics['n_samples']:,} samples)")
        else:
            print(f"  {mode.upper():12} ❌ Failed")
    
    print("═" * 60)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train stock/ETF ML models')
    parser.add_argument('--symbols', nargs='+', help='Symbols to train on')
    parser.add_argument('--etfs', action='store_true', help='Train on ETFs instead of stocks')
    parser.add_argument('--mode', '-m', choices=['daytrade', 'swing', 'investment'], 
                       help='Train only specific mode')
    parser.add_argument('--all-modes', action='store_true', help='Train all modes')
    args = parser.parse_args()
    
    asset_type = 'etf' if args.etfs else 'stock'
    symbols = args.symbols
    
    if args.all_modes:
        modes = ['daytrade', 'swing', 'investment']
    elif args.mode:
        modes = [args.mode]
    else:
        modes = ['swing', 'investment']  # Default for stocks
    
    train_stocks(symbols, modes, asset_type)
