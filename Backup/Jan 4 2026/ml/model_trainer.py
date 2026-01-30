"""
Model Trainer - Train ML Models from Historical Data
======================================================

Extracts training data from whale_history.db and trade_history.json,
trains classification and regression models for direction and TP/SL.
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .feature_extractor import FEATURE_NAMES, extract_features


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Minimum samples needed for training
MIN_SAMPLES = 100

# Train/test split
TEST_SIZE = 0.2

# Target definitions
# Direction: 0=SHORT, 1=WAIT, 2=LONG
DIRECTION_MAP = {'SHORT': 0, 'WAIT': 1, 'LONG': 2}
DIRECTION_REVERSE = {0: 'SHORT', 1: 'WAIT', 2: 'LONG'}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingRecord:
    """Single training record with features and target"""
    features: np.ndarray
    direction: str          # LONG, SHORT, WAIT
    outcome: str            # WIN, LOSS, PENDING
    tp1_hit: bool
    tp2_hit: bool
    sl_hit: bool
    final_pnl: float
    r_multiple: float
    tp1_pct: float          # TP1 distance as %
    sl_pct: float           # SL distance as %
    symbol: str
    timeframe: str
    created_at: str


def extract_from_whale_db(db_path: str, limit: int = None) -> List[TrainingRecord]:
    """
    Extract training data from whale_history.db
    
    The whale database stores snapshots with known outcomes.
    """
    records = []
    
    if not os.path.exists(db_path):
        print(f"⚠️ Database not found: {db_path}")
        return records
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get snapshots with known outcomes
        query = """
            SELECT 
                symbol, timeframe, created_at,
                whale_long_pct, retail_long_pct,
                oi_change_24h, price_change_24h,
                funding_rate, position_pct,
                ta_score, rsi, trend,
                money_flow_phase,
                hit_tp1, hit_sl, candles_to_outcome,
                swing_high, swing_low, current_price
            FROM whale_snapshots
            WHERE hit_tp1 IS NOT NULL OR hit_sl IS NOT NULL
            ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for row in rows:
            try:
                # Determine direction based on outcome
                # If TP1 was hit, it was a winning trade in that direction
                hit_tp1 = row['hit_tp1'] == 1 if row['hit_tp1'] is not None else False
                hit_sl = row['hit_sl'] == 1 if row['hit_sl'] is not None else False
                
                # Skip unclear outcomes
                if not hit_tp1 and not hit_sl:
                    continue
                
                # Determine outcome
                if hit_tp1:
                    outcome = 'WIN'
                    direction = 'LONG'  # Assuming stored as long trades
                else:
                    outcome = 'LOSS'
                    direction = 'LONG'
                
                # Extract features
                whale_pct = row['whale_long_pct'] or 50
                retail_pct = row['retail_long_pct'] or 50
                position_pct = row['position_pct'] or 50
                
                features = extract_features(
                    whale_pct=whale_pct,
                    retail_pct=retail_pct,
                    funding_rate=row['funding_rate'] or 0,
                    oi_change=row['oi_change_24h'] or 0,
                    price_change_24h=row['price_change_24h'] or 0,
                    position_pct=position_pct,
                    swing_high=row['swing_high'] or 0,
                    swing_low=row['swing_low'] or 0,
                    current_price=row['current_price'] or 0,
                    ta_score=row['ta_score'] or 50,
                    rsi=row['rsi'] or 50,
                    trend=row['trend'] or 'NEUTRAL',
                    money_flow_phase=row['money_flow_phase'] or 'NEUTRAL',
                ).features
                
                # Calculate TP/SL percentages (use defaults if not stored)
                tp1_pct = 2.5  # Default
                sl_pct = 1.5   # Default
                
                record = TrainingRecord(
                    features=features,
                    direction=direction,
                    outcome=outcome,
                    tp1_hit=hit_tp1,
                    tp2_hit=False,  # Not tracked in DB
                    sl_hit=hit_sl,
                    final_pnl=tp1_pct if hit_tp1 else -sl_pct,
                    r_multiple=1.0 if hit_tp1 else -1.0,
                    tp1_pct=tp1_pct,
                    sl_pct=sl_pct,
                    symbol=row['symbol'],
                    timeframe=row['timeframe'],
                    created_at=row['created_at'],
                )
                records.append(record)
                
            except Exception as e:
                continue
        
        conn.close()
        print(f"✅ Extracted {len(records)} records from whale_history.db")
        
    except Exception as e:
        print(f"⚠️ Error extracting from whale DB: {e}")
    
    return records


def extract_from_trade_history(json_path: str) -> List[TrainingRecord]:
    """
    Extract training data from trade_history.json
    
    Trade history has detailed outcome data.
    """
    records = []
    
    if not os.path.exists(json_path):
        print(f"⚠️ Trade history not found: {json_path}")
        return records
    
    try:
        with open(json_path, 'r') as f:
            trades = json.load(f)
        
        for trade in trades:
            try:
                # Skip active trades
                if trade.get('status') == 'active':
                    continue
                
                # Get outcome
                tp1_hit = trade.get('tp1_hit', False)
                tp2_hit = trade.get('tp2_hit', False)
                sl_hit = trade.get('close_reason') == 'SL_HIT'
                
                if trade.get('status') == 'WIN' or tp1_hit:
                    outcome = 'WIN'
                elif trade.get('status') == 'LOSS' or sl_hit:
                    outcome = 'LOSS'
                else:
                    continue  # Skip unclear
                
                direction = trade.get('direction', 'LONG')
                
                # Extract what features we have
                whale_pct = trade.get('whale_pct', 50)
                position_pct = trade.get('position_pct', 50)
                
                # Calculate TP/SL percentages
                entry = trade.get('entry', 0)
                tp1 = trade.get('tp1', 0)
                sl = trade.get('stop_loss', 0)
                
                if entry > 0:
                    if direction == 'LONG':
                        tp1_pct = ((tp1 - entry) / entry) * 100 if tp1 > entry else 2.5
                        sl_pct = ((entry - sl) / entry) * 100 if sl < entry else 1.5
                    else:
                        tp1_pct = ((entry - tp1) / entry) * 100 if tp1 < entry else 2.5
                        sl_pct = ((sl - entry) / entry) * 100 if sl > entry else 1.5
                else:
                    tp1_pct = 2.5
                    sl_pct = 1.5
                
                features = extract_features(
                    whale_pct=whale_pct,
                    position_pct=position_pct,
                    ta_score=trade.get('score', 50),
                ).features
                
                record = TrainingRecord(
                    features=features,
                    direction=direction,
                    outcome=outcome,
                    tp1_hit=tp1_hit,
                    tp2_hit=tp2_hit,
                    sl_hit=sl_hit,
                    final_pnl=trade.get('final_pnl', 0),
                    r_multiple=trade.get('r_multiple', 0),
                    tp1_pct=tp1_pct,
                    sl_pct=sl_pct,
                    symbol=trade.get('symbol', 'UNKNOWN'),
                    timeframe=trade.get('timeframe', '15m'),
                    created_at=trade.get('created_at', ''),
                )
                records.append(record)
                
            except Exception as e:
                continue
        
        print(f"✅ Extracted {len(records)} records from trade_history.json")
        
    except Exception as e:
        print(f"⚠️ Error extracting from trade history: {e}")
    
    return records


def prepare_training_data(records: List[TrainingRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training data from records.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y_direction: Direction labels (n_samples,)
        y_tp_sl: TP/SL targets (n_samples, 3) - [tp1_pct, tp2_pct, sl_pct]
    """
    
    X = np.array([r.features for r in records])
    
    # Direction target: For winning trades, use their direction
    # For losing trades, opposite would have been better (but we'll use WAIT)
    y_direction = []
    for r in records:
        if r.outcome == 'WIN':
            y_direction.append(r.direction)
        else:
            # Loss - the signal was wrong, should have waited
            y_direction.append('WAIT')
    y_direction = np.array([DIRECTION_MAP.get(d, 1) for d in y_direction])
    
    # TP/SL targets (for winning trades, use actual; for losses, adjust)
    y_tp_sl = []
    for r in records:
        if r.outcome == 'WIN':
            # Use actual values from winning trades
            y_tp_sl.append([r.tp1_pct, r.tp1_pct * 1.5, r.sl_pct])
        else:
            # For losses, TP was too far or SL too tight - adjust
            y_tp_sl.append([r.tp1_pct * 0.8, r.tp1_pct * 1.2, r.sl_pct * 1.2])
    y_tp_sl = np.array(y_tp_sl)
    
    return X, y_direction, y_tp_sl


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(
    whale_db_path: str = None,
    trade_history_path: str = None,
    save_models: bool = True
) -> Dict:
    """
    Train ML models from historical data.
    
    Args:
        whale_db_path: Path to whale_history.db
        trade_history_path: Path to trade_history.json
        save_models: Whether to save trained models to disk
    
    Returns:
        Dict with training results and metrics
    """
    
    # Default paths
    if whale_db_path is None:
        whale_db_path = os.path.join(DATA_DIR, 'whale_history.db')
    if trade_history_path is None:
        trade_history_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'trade_history.json'
        )
    
    print("=" * 60)
    print("ML MODEL TRAINING")
    print("=" * 60)
    
    # Extract data
    records = []
    records.extend(extract_from_whale_db(whale_db_path))
    records.extend(extract_from_trade_history(trade_history_path))
    
    if len(records) < MIN_SAMPLES:
        print(f"⚠️ Not enough data for training. Need {MIN_SAMPLES}, have {len(records)}")
        print("Will use heuristic predictions until more data is collected.")
        return {
            'success': False,
            'error': f'Insufficient data: {len(records)} < {MIN_SAMPLES}',
            'records_found': len(records),
        }
    
    print(f"\n📊 Total training records: {len(records)}")
    
    # Prepare data
    X, y_direction, y_tp_sl = prepare_training_data(records)
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_dir_train, y_dir_test, y_tp_train, y_tp_test = train_test_split(
        X, y_direction, y_tp_sl, test_size=TEST_SIZE, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train direction model (LightGBM or fallback to RandomForest)
    try:
        import lightgbm as lgb
        
        direction_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        print("\n🌳 Training direction model (LightGBM)...")
        
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        
        direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        print("\n🌳 Training direction model (RandomForest)...")
    
    direction_model.fit(X_train_scaled, y_dir_train)
    
    # Evaluate direction model
    from sklearn.metrics import accuracy_score, classification_report
    y_dir_pred = direction_model.predict(X_test_scaled)
    dir_accuracy = accuracy_score(y_dir_test, y_dir_pred)
    print(f"Direction accuracy: {dir_accuracy:.1%}")
    
    # Train TP/SL model (regression)
    try:
        import lightgbm as lgb
        tp_sl_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor
        tp_sl_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
    
    print("\n📈 Training TP/SL model...")
    # Train on TP1 only for simplicity (can expand later)
    tp_sl_model.fit(X_train_scaled, y_tp_train)
    
    # Evaluate TP/SL model
    from sklearn.metrics import mean_absolute_error
    y_tp_pred = tp_sl_model.predict(X_test_scaled)
    tp_mae = mean_absolute_error(y_tp_test, y_tp_pred)
    print(f"TP/SL MAE: {tp_mae:.2f}%")
    
    # Feature importances
    if hasattr(direction_model, 'feature_importances_'):
        importances = direction_model.feature_importances_
        top_features = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        print("\n📊 Top 10 Important Features:")
        for name, imp in top_features:
            print(f"  {name}: {imp:.3f}")
    
    # Save models
    if save_models:
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Update class labels to strings
        direction_model.classes_ = np.array(['SHORT', 'WAIT', 'LONG'])
        
        with open(os.path.join(MODEL_DIR, 'direction_model.pkl'), 'wb') as f:
            pickle.dump(direction_model, f)
        
        with open(os.path.join(MODEL_DIR, 'tp_sl_model.pkl'), 'wb') as f:
            pickle.dump(tp_sl_model, f)
        
        with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        metadata = {
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(records),
            'n_features': len(FEATURE_NAMES),
            'direction_accuracy': float(dir_accuracy),
            'tp_sl_mae': float(tp_mae),
            'feature_names': FEATURE_NAMES,
        }
        with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Models saved to {MODEL_DIR}")
    
    # Calculate class distribution
    unique, counts = np.unique(y_direction, return_counts=True)
    class_dist = {DIRECTION_REVERSE.get(u, str(u)): int(c) for u, c in zip(unique, counts)}
    
    return {
        'success': True,
        'records_used': len(records),
        'direction_accuracy': float(dir_accuracy),
        'tp_sl_mae': float(tp_mae),
        'class_distribution': class_dist,
        'top_features': top_features if 'top_features' in dir() else [],
        'model_version': metadata['version'] if save_models else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models')
    parser.add_argument('--whale-db', type=str, help='Path to whale_history.db')
    parser.add_argument('--trade-history', type=str, help='Path to trade_history.json')
    parser.add_argument('--no-save', action='store_true', help='Do not save models')
    
    args = parser.parse_args()
    
    result = train_models(
        whale_db_path=args.whale_db,
        trade_history_path=args.trade_history,
        save_models=not args.no_save
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2))
