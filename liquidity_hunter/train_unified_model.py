"""
Unified ML Model Training with Price Action Features

Trains on 365 days of historical data across all major coins to create
a universal model that works for any coin.

Features:
- Traditional: direction, level_type, distance, whale_pct, volume_ratio
- Price Action: candle_pattern, structure, order_blocks, fvg, momentum
- Outcome: Did TP1 hit before SL? (binary classification)

Usage:
    python -m liquidity_hunter.train_unified_model --days 365 --coins 50
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Training database path
TRAINING_DB_PATH = 'data/unified_pa_training.db'
MODEL_PATH = 'models/unified_quality_model.pkl'


def init_training_db():
    """Initialize the training database with price action features."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            timeframe TEXT,

            -- Sweep info
            direction TEXT,
            level_type TEXT,
            level_price REAL,
            sweep_price REAL,
            candles_ago INTEGER,

            -- Traditional features
            distance_pct REAL,
            whale_pct REAL,
            whale_delta REAL,
            volume_ratio REAL,
            rsi_value REAL,

            -- Price Action features (NEW)
            pa_candle_score INTEGER,
            pa_structure_score INTEGER,
            pa_has_order_block INTEGER,
            pa_has_fvg INTEGER,
            pa_volume_score INTEGER,
            pa_momentum_score INTEGER,
            pa_total_score INTEGER,

            -- Entry quality
            entry_quality_score INTEGER,
            entry_quality_grade TEXT,

            -- Outcome (what we're predicting)
            outcome TEXT,  -- 'WIN', 'LOSS', 'PENDING'
            tp1_hit INTEGER,  -- 1 if TP1 was hit first, 0 if SL hit first
            max_profit_pct REAL,
            max_drawdown_pct REAL,
            bars_to_outcome INTEGER,

            -- Metadata
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(symbol, timestamp, direction)
        )
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_symbol ON training_samples(symbol)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_outcome ON training_samples(outcome)
    ''')

    conn.commit()
    conn.close()
    print(f"[TRAIN] Database initialized at {TRAINING_DB_PATH}")


def fetch_historical_data(symbol: str, days: int = 365, timeframe: str = '4h') -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data for a symbol."""
    try:
        # Import data fetcher
        import sys
        sys.path.insert(0, '.')
        from core.data_fetcher import fetch_binance_klines

        # Calculate limit based on days and timeframe
        candles_per_day = {'1h': 24, '4h': 6, '1d': 1}.get(timeframe, 6)
        limit = min(days * candles_per_day, 1000)  # Binance limit

        df = fetch_binance_klines(symbol, interval=timeframe, limit=limit)
        return df

    except Exception as e:
        print(f"[TRAIN] Error fetching {symbol}: {e}")
        return None


def simulate_sweep_outcome(df: pd.DataFrame, sweep_idx: int, direction: str,
                           entry_price: float, atr: float) -> Dict:
    """
    Simulate what would have happened if we entered at the sweep.

    Returns outcome (WIN/LOSS), max profit, max drawdown, bars to outcome.
    """
    # Define TP and SL based on direction
    if direction == 'LONG':
        tp1 = entry_price + atr * 1.5  # 1.5 ATR target
        sl = entry_price - atr * 1.0   # 1 ATR stop
    else:
        tp1 = entry_price - atr * 1.5
        sl = entry_price + atr * 1.0

    # Look at candles after sweep (up to 50 candles)
    max_bars = min(50, len(df) - sweep_idx - 1)
    if max_bars < 1:
        return {'outcome': 'PENDING', 'tp1_hit': 0, 'max_profit_pct': 0,
                'max_drawdown_pct': 0, 'bars_to_outcome': 0}

    post_sweep = df.iloc[sweep_idx + 1:sweep_idx + 1 + max_bars]

    max_profit_pct = 0
    max_drawdown_pct = 0
    outcome = 'PENDING'
    tp1_hit = 0
    bars_to_outcome = 0

    for i, (idx, candle) in enumerate(post_sweep.iterrows()):
        high = candle['high']
        low = candle['low']

        if direction == 'LONG':
            # Calculate profit/drawdown
            profit_pct = (high - entry_price) / entry_price * 100
            drawdown_pct = (entry_price - low) / entry_price * 100

            max_profit_pct = max(max_profit_pct, profit_pct)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

            # Check TP1 and SL
            if high >= tp1:
                outcome = 'WIN'
                tp1_hit = 1
                bars_to_outcome = i + 1
                break
            if low <= sl:
                outcome = 'LOSS'
                tp1_hit = 0
                bars_to_outcome = i + 1
                break
        else:
            # SHORT
            profit_pct = (entry_price - low) / entry_price * 100
            drawdown_pct = (high - entry_price) / entry_price * 100

            max_profit_pct = max(max_profit_pct, profit_pct)
            max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

            if low <= tp1:
                outcome = 'WIN'
                tp1_hit = 1
                bars_to_outcome = i + 1
                break
            if high >= sl:
                outcome = 'LOSS'
                tp1_hit = 0
                bars_to_outcome = i + 1
                break

    return {
        'outcome': outcome,
        'tp1_hit': tp1_hit,
        'max_profit_pct': max_profit_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'bars_to_outcome': bars_to_outcome
    }


def generate_training_sample(df: pd.DataFrame, sweep_idx: int, symbol: str,
                              timeframe: str) -> Optional[Dict]:
    """
    Generate a training sample from a detected sweep.

    Runs full analysis including price action and simulates outcome.
    """
    try:
        # Import required modules
        from liquidity_hunter.liquidity_hunter import (
            normalize_columns, find_liquidity_levels, detect_sweep,
            calculate_entry_quality
        )
        from liquidity_hunter.price_action_analyzer import analyze_sweep_reaction

        # Get data up to sweep point (simulate real-time)
        historical_df = df.iloc[:sweep_idx + 1].copy()
        if len(historical_df) < 50:
            return None

        historical_df = normalize_columns(historical_df)

        # Calculate ATR
        high = historical_df['high']
        low = historical_df['low']
        close = historical_df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        if atr <= 0:
            return None

        current_price = float(close.iloc[-1])

        # Find liquidity levels
        liquidity_levels = find_liquidity_levels(historical_df)

        # Detect sweep (simplified - we know there's activity at this point)
        sweep_status = detect_sweep(historical_df, liquidity_levels, atr, lookback_candles=25, whale_pct=50)

        if not sweep_status.get('detected'):
            return None

        direction = sweep_status.get('direction', 'LONG')
        level_price = sweep_status.get('level_swept', current_price)
        level_type = sweep_status.get('level_type', 'SWING')
        candles_ago = sweep_status.get('candles_ago', 1)

        # Run price action analysis
        pa_analysis = analyze_sweep_reaction(historical_df, sweep_idx - candles_ago, direction, atr)

        if not pa_analysis.get('valid'):
            pa_analysis = {'prediction': {'component_scores': {}}}

        pa_scores = pa_analysis.get('prediction', {}).get('component_scores', {})

        # Calculate entry quality
        volumes = historical_df['volume']
        avg_volume = float(volumes.rolling(20).mean().iloc[-1])
        sweep_volume = float(volumes.iloc[-candles_ago]) if candles_ago < len(volumes) else avg_volume
        volume_ratio = sweep_volume / avg_volume if avg_volume > 0 else 1.0

        entry_quality = calculate_entry_quality(
            sweep_status=sweep_status,
            current_price=current_price,
            whale_pct=50,  # Neutral for historical
            whale_delta=0,
            volume_on_sweep=sweep_volume,
            avg_volume=avg_volume,
            df=historical_df
        )

        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_value = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50

        # Simulate outcome using FULL df (future data)
        entry_price = current_price
        outcome_data = simulate_sweep_outcome(df, sweep_idx, direction, entry_price, atr)

        # Build training sample
        sample = {
            'timestamp': str(df.index[sweep_idx]) if hasattr(df.index[sweep_idx], 'strftime') else str(sweep_idx),
            'symbol': symbol,
            'timeframe': timeframe,

            # Sweep info
            'direction': direction,
            'level_type': level_type,
            'level_price': level_price,
            'sweep_price': current_price,
            'candles_ago': candles_ago,

            # Traditional features
            'distance_pct': abs(level_price - current_price) / current_price * 100,
            'whale_pct': 50,  # Neutral for historical
            'whale_delta': 0,
            'volume_ratio': volume_ratio,
            'rsi_value': rsi_value,

            # Price Action features
            'pa_candle_score': pa_scores.get('candle_pattern', 0),
            'pa_structure_score': pa_scores.get('structure', 0),
            'pa_has_order_block': 1 if pa_scores.get('order_blocks', 0) > 0 else 0,
            'pa_has_fvg': 1 if pa_scores.get('fvg', 0) > 0 else 0,
            'pa_volume_score': pa_scores.get('volume', 0),
            'pa_momentum_score': pa_scores.get('momentum', 0),
            'pa_total_score': pa_analysis.get('prediction', {}).get('score', 0),

            # Entry quality
            'entry_quality_score': entry_quality.get('score', 0),
            'entry_quality_grade': entry_quality.get('grade', 'N/A'),

            # Outcome
            'outcome': outcome_data['outcome'],
            'tp1_hit': outcome_data['tp1_hit'],
            'max_profit_pct': outcome_data['max_profit_pct'],
            'max_drawdown_pct': outcome_data['max_drawdown_pct'],
            'bars_to_outcome': outcome_data['bars_to_outcome']
        }

        return sample

    except Exception as e:
        print(f"[TRAIN] Error generating sample: {e}")
        return None


def save_training_sample(sample: Dict):
    """Save a training sample to the database."""
    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT OR REPLACE INTO training_samples (
                timestamp, symbol, timeframe, direction, level_type, level_price,
                sweep_price, candles_ago, distance_pct, whale_pct, whale_delta,
                volume_ratio, rsi_value, pa_candle_score, pa_structure_score,
                pa_has_order_block, pa_has_fvg, pa_volume_score, pa_momentum_score,
                pa_total_score, entry_quality_score, entry_quality_grade,
                outcome, tp1_hit, max_profit_pct, max_drawdown_pct, bars_to_outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample['timestamp'], sample['symbol'], sample['timeframe'],
            sample['direction'], sample['level_type'], sample['level_price'],
            sample['sweep_price'], sample['candles_ago'], sample['distance_pct'],
            sample['whale_pct'], sample['whale_delta'], sample['volume_ratio'],
            sample['rsi_value'], sample['pa_candle_score'], sample['pa_structure_score'],
            sample['pa_has_order_block'], sample['pa_has_fvg'], sample['pa_volume_score'],
            sample['pa_momentum_score'], sample['pa_total_score'],
            sample['entry_quality_score'], sample['entry_quality_grade'],
            sample['outcome'], sample['tp1_hit'], sample['max_profit_pct'],
            sample['max_drawdown_pct'], sample['bars_to_outcome']
        ))
        conn.commit()
    except Exception as e:
        print(f"[TRAIN] Error saving sample: {e}")
    finally:
        conn.close()


def generate_training_data(symbols: List[str], days: int = 365,
                           timeframe: str = '4h', progress_callback=None) -> int:
    """
    Generate training data for all symbols.

    Scans historical data for sweeps and generates training samples.
    """
    init_training_db()

    total_samples = 0
    total_symbols = len(symbols)

    for sym_idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(sym_idx + 1, total_symbols, symbol)

        print(f"\n[TRAIN] Processing {symbol} ({sym_idx + 1}/{total_symbols})...")

        # Fetch historical data
        df = fetch_historical_data(symbol, days=days, timeframe=timeframe)
        if df is None or len(df) < 100:
            print(f"[TRAIN] Skipping {symbol} - insufficient data")
            continue

        # Normalize
        try:
            from liquidity_hunter.liquidity_hunter import normalize_columns
            df = normalize_columns(df)
        except:
            continue

        # Scan for potential sweeps (every 10 candles to speed up)
        samples_for_symbol = 0
        step = 5  # Check every 5 candles

        for i in range(100, len(df) - 50, step):
            # Check if this looks like a sweep point (significant low or high)
            window = df.iloc[i-10:i+1]
            current_low = float(df.iloc[i]['low'])
            current_high = float(df.iloc[i]['high'])
            window_min = float(window['low'].min())
            window_max = float(window['high'].max())

            # Is this a potential sweep? (made new low or high in window)
            is_potential_sweep = (current_low <= window_min * 1.001) or (current_high >= window_max * 0.999)

            if not is_potential_sweep:
                continue

            # Generate training sample
            sample = generate_training_sample(df, i, symbol, timeframe)

            if sample and sample['outcome'] in ['WIN', 'LOSS']:
                save_training_sample(sample)
                samples_for_symbol += 1
                total_samples += 1

        print(f"[TRAIN] {symbol}: {samples_for_symbol} samples generated")

    print(f"\n[TRAIN] Total samples generated: {total_samples}")
    return total_samples


def train_model() -> Dict:
    """
    Train the unified ML model on collected data.

    Returns training metrics.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import LabelEncoder

    # Load training data
    conn = sqlite3.connect(TRAINING_DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM training_samples
        WHERE outcome IN ('WIN', 'LOSS')
    ''', conn)
    conn.close()

    if len(df) < 100:
        print(f"[TRAIN] Insufficient data: {len(df)} samples (need 100+)")
        return {'error': 'Insufficient data', 'samples': len(df)}

    print(f"[TRAIN] Training on {len(df)} samples...")
    print(f"[TRAIN] Win rate in data: {(df['tp1_hit'] == 1).mean():.1%}")

    # Prepare features
    feature_cols = [
        # Traditional
        'distance_pct', 'volume_ratio', 'rsi_value',
        # Price Action (NEW)
        'pa_candle_score', 'pa_structure_score', 'pa_has_order_block',
        'pa_has_fvg', 'pa_volume_score', 'pa_momentum_score', 'pa_total_score',
        # Entry quality
        'entry_quality_score'
    ]

    # Encode categorical
    le_direction = LabelEncoder()
    df['direction_enc'] = le_direction.fit_transform(df['direction'])

    le_level = LabelEncoder()
    df['level_type_enc'] = le_level.fit_transform(df['level_type'].fillna('SWING'))

    feature_cols.extend(['direction_enc', 'level_type_enc'])

    # Handle missing values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    X = df[feature_cols].values
    y = df['tp1_hit'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Calculate win rate at different probability thresholds
    thresholds = {}
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        mask = y_prob >= thresh
        if mask.sum() > 0:
            win_rate = y_test[mask].mean()
            count = mask.sum()
            thresholds[f'prob_{int(thresh*100)}'] = {
                'win_rate': float(win_rate),
                'count': int(count),
                'pct_of_total': float(count / len(y_test))
            }

    # Save model
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'le_direction': le_direction,
        'le_level': le_level,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'thresholds': thresholds,
        'importance': importance_sorted,
        'trained_at': datetime.now().isoformat(),
        'samples': len(df),
        'win_rate_base': float((df['tp1_hit'] == 1).mean())
    }

    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n[TRAIN] ═══════════════════════════════════════")
    print(f"[TRAIN] MODEL TRAINING COMPLETE")
    print(f"[TRAIN] ═══════════════════════════════════════")
    print(f"[TRAIN] Samples: {len(df)}")
    print(f"[TRAIN] Base win rate: {(df['tp1_hit'] == 1).mean():.1%}")
    print(f"[TRAIN] Accuracy: {accuracy:.1%}")
    print(f"[TRAIN] Precision: {precision:.1%}")
    print(f"[TRAIN] F1 Score: {f1:.1%}")
    print(f"[TRAIN] CV Score: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
    print(f"\n[TRAIN] Win rates by probability threshold:")
    for k, v in thresholds.items():
        print(f"[TRAIN]   {k}: {v['win_rate']:.1%} ({v['count']} samples, {v['pct_of_total']:.1%} of total)")
    print(f"\n[TRAIN] Top 5 features:")
    for feat, imp in importance_sorted[:5]:
        print(f"[TRAIN]   {feat}: {imp:.3f}")
    print(f"\n[TRAIN] Model saved to {MODEL_PATH}")

    return model_data


def get_unified_prediction(features: Dict) -> Dict:
    """
    Get prediction from the unified model.

    Args:
        features: Dict with all feature values

    Returns:
        Dict with probability, decision, confidence
    """
    if not os.path.exists(MODEL_PATH):
        return {'probability': 0.5, 'decision': 'UNKNOWN', 'error': 'Model not trained'}

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['model']
        feature_cols = model_data['feature_cols']
        le_direction = model_data['le_direction']
        le_level = model_data['le_level']

        # Prepare features
        X = []
        for col in feature_cols:
            if col == 'direction_enc':
                direction = features.get('direction', 'LONG')
                try:
                    val = le_direction.transform([direction])[0]
                except:
                    val = 0
            elif col == 'level_type_enc':
                level_type = features.get('level_type', 'SWING')
                try:
                    val = le_level.transform([level_type])[0]
                except:
                    val = 0
            else:
                val = features.get(col, 0)
            X.append(val)

        X = np.array([X])

        # Predict
        prob = model.predict_proba(X)[0][1]

        # Determine decision
        if prob >= 0.6:
            decision = 'STRONG_YES'
        elif prob >= 0.5:
            decision = 'YES'
        elif prob >= 0.4:
            decision = 'MAYBE'
        else:
            decision = 'NO'

        # Get expected win rate from thresholds
        thresholds = model_data.get('thresholds', {})
        if prob >= 0.6 and 'prob_60' in thresholds:
            expected_win_rate = thresholds['prob_60']['win_rate']
        elif prob >= 0.5 and 'prob_50' in thresholds:
            expected_win_rate = thresholds['prob_50']['win_rate']
        else:
            expected_win_rate = prob

        return {
            'probability': float(prob),
            'decision': decision,
            'expected_win_rate': expected_win_rate,
            'take_trade': prob >= 0.5,
            'model_accuracy': model_data['metrics']['accuracy']
        }

    except Exception as e:
        return {'probability': 0.5, 'decision': 'ERROR', 'error': str(e)}


def get_default_crypto_symbols(limit: int = 50) -> List[str]:
    """Get top crypto symbols for training."""
    try:
        from core.data_fetcher import get_default_futures_pairs
        pairs = get_default_futures_pairs()
        return pairs[:limit]
    except:
        # Fallback list
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT',
            'MATICUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'UNIUSDT',
            'ETCUSDT', 'XLMUSDT', 'APTUSDT', 'FILUSDT', 'ARBUSDT',
            'OPUSDT', 'NEARUSDT', 'AAVEUSDT', 'MKRUSDT', 'GRTUSDT',
            'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FTMUSDT', 'SANDUSDT',
            'MANAUSDT', 'AXSUSDT', 'THETAUSDT', 'EGLDUSDT', 'EOSUSDT',
            'XTZUSDT', 'FLOWUSDT', 'CHZUSDT', 'LRCUSDT', 'ENJUSDT',
            'CRVUSDT', 'SNXUSDT', 'COMPUSDT', 'YFIUSDT', 'SUSHIUSDT',
            '1INCHUSDT', 'BATUSDT', 'ZRXUSDT', 'RENUSDT', 'KAVAUSDT'
        ][:limit]


def run_full_training(days: int = 365, num_coins: int = 50, timeframe: str = '4h'):
    """
    Run the full training pipeline.

    1. Generate training data from historical sweeps
    2. Train the unified model
    3. Save model and report metrics
    """
    print("=" * 70)
    print("UNIFIED ML MODEL TRAINING WITH PRICE ACTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Days of history: {days}")
    print(f"  Number of coins: {num_coins}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Training DB: {TRAINING_DB_PATH}")
    print(f"  Model path: {MODEL_PATH}")

    # Get symbols
    symbols = get_default_crypto_symbols(num_coins)
    print(f"\nSymbols to process: {len(symbols)}")

    # Generate training data
    print("\n" + "=" * 70)
    print("PHASE 1: GENERATING TRAINING DATA")
    print("=" * 70)

    def progress(current, total, symbol):
        pct = current / total * 100
        print(f"[{pct:5.1f}%] Processing {symbol}...")

    total_samples = generate_training_data(symbols, days=days, timeframe=timeframe,
                                            progress_callback=progress)

    if total_samples < 100:
        print(f"\n[ERROR] Not enough samples generated ({total_samples}). Need at least 100.")
        return None

    # Train model
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING ML MODEL")
    print("=" * 70)

    model_data = train_model()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    return model_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train unified ML model with price action')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--coins', type=int, default=50, help='Number of coins to train on')
    parser.add_argument('--timeframe', type=str, default='4h', help='Candle timeframe')
    parser.add_argument('--train-only', action='store_true', help='Only train (skip data generation)')

    args = parser.parse_args()

    if args.train_only:
        print("Training model on existing data...")
        train_model()
    else:
        run_full_training(days=args.days, num_coins=args.coins, timeframe=args.timeframe)
