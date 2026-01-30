#!/usr/bin/env python3
"""
InvestorIQ ML Model Training Script
====================================

Run this script to train ML models from your whale_history.db

Usage:
    python3 train_ml.py                    # Auto-detect whale_history.db
    python3 train_ml.py path/to/whale_history.db   # Specify path
"""

import os
import sys
import json
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🤖 INVESTORIQ ML MODEL TRAINING")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# FIND DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def find_whale_db():
    """Find whale_history.db in common locations"""
    
    # Check command line argument first
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            return path
        else:
            print(f"❌ File not found: {path}")
            sys.exit(1)
    
    # Common locations to check
    locations = [
        "whale_history.db",
        "data/whale_history.db",
        "../whale_history.db",
        os.path.expanduser("~/whale_history.db"),
        os.path.expanduser("~/TScanner/whale_history.db"),
        os.path.expanduser("~/TScanner/data/whale_history.db"),
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            return loc
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'whale_pct', 'retail_pct', 'whale_retail_divergence', 
    'whale_dominance', 'funding_rate',
    'oi_change_24h', 'oi_signal_encoded', 'price_change_24h', 'price_change_1h',
    'position_in_range', 'position_label_encoded', 'range_size_pct',
    'ta_score', 'rsi', 'rsi_zone_encoded', 'trend_encoded', 'volatility_pct',
    'money_flow_encoded', 'volume_ratio',
    'at_bullish_ob', 'at_bearish_ob', 'near_support', 'near_resistance',
    'btc_correlation', 'btc_trend_encoded', 'market_fear_greed', 'is_weekend',
    'historical_win_rate', 'similar_setup_count', 'avg_historical_return',
]

MONEY_FLOW_MAP = {
    'MARKDOWN': -2, 'DISTRIBUTION': -1, 'PROFIT_TAKING': -1, 'PROFIT TAKING': -1,
    'NEUTRAL': 0, 'UNKNOWN': 0, 'ACCUMULATION': 1, 'MARKUP': 2,
}

TREND_MAP = {
    'BEARISH': -1, 'LEAN_BEARISH': -1, 'NEUTRAL': 0, 'MIXED': 0,
    'UNKNOWN': 0, 'LEAN_BULLISH': 1, 'BULLISH': 1,
}


def extract_features(row):
    """Extract features from a database row"""
    
    # Handle different possible column names
    whale_pct = (row.get('whale_long_pct') or row.get('top_trader_long_pct') or 
                 row.get('whale_pct') or row.get('top_long_pct') or 50)
    retail_pct = (row.get('retail_long_pct') or row.get('retail_pct') or 
                  row.get('long_short_ratio') or 50)
    position_pct = (row.get('position_pct') or row.get('position_in_range') or 
                    row.get('range_position') or 50)
    
    # Position label
    if position_pct <= 35:
        position_label = 0  # EARLY
    elif position_pct >= 65:
        position_label = 2  # LATE
    else:
        position_label = 1  # MIDDLE
    
    # RSI zone
    rsi = row.get('rsi') or row.get('rsi_14') or 50
    if rsi <= 30:
        rsi_zone = -1
    elif rsi >= 70:
        rsi_zone = 1
    else:
        rsi_zone = 0
    
    # Money flow
    mf = row.get('money_flow_phase') or row.get('flow_phase') or row.get('phase') or 'NEUTRAL'
    money_flow_encoded = MONEY_FLOW_MAP.get(str(mf).upper() if mf else 'NEUTRAL', 0)
    
    # Trend
    trend = row.get('trend') or row.get('structure') or row.get('market_structure') or 'NEUTRAL'
    trend_encoded = TREND_MAP.get(str(trend).upper() if trend else 'NEUTRAL', 0)
    
    # OI change
    oi_change = (row.get('oi_change_24h') or row.get('oi_change') or 
                 row.get('open_interest_change') or 0)
    
    # Price change
    price_change = (row.get('price_change_24h') or row.get('price_change') or 
                    row.get('change_24h') or 0)
    
    features = np.array([
        whale_pct,
        retail_pct,
        whale_pct - retail_pct,  # divergence
        (whale_pct - 50) / 50,   # whale_dominance
        (row.get('funding_rate') or row.get('funding') or 0) * 100,
        oi_change,
        0,  # oi_signal_encoded (derived)
        price_change,
        0,  # price_change_1h
        position_pct,
        position_label,
        0,  # range_size_pct
        row.get('ta_score') or row.get('score') or 50,
        rsi,
        rsi_zone,
        trend_encoded,
        0,  # volatility_pct
        money_flow_encoded,
        1.0,  # volume_ratio
        0, 0, 0, 0,  # SMC features
        row.get('btc_correlation') or 0,  # btc_correlation
        0,  # btc_trend_encoded
        50, # fear_greed
        0,  # is_weekend
        row.get('historical_win_rate') or 50,
        row.get('sample_size') or 0,
        0,  # avg_historical_return
    ], dtype=np.float32)
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_training_data(db_path):
    """Extract training data from whale_history.db"""
    
    records = []
    
    print(f"\n📂 Loading data from: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        print(f"   Tables found: {tables}")
        
        # Check column names
        cursor.execute("PRAGMA table_info(whale_snapshots)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"   Columns: {columns[:10]}...")  # Show first 10
        
        # Find the timestamp column
        timestamp_col = None
        for col in ['created_at', 'timestamp', 'date', 'time', 'snapshot_time']:
            if col in columns:
                timestamp_col = col
                break
        
        # Try to get data with outcomes
        if timestamp_col:
            query = f"""
                SELECT * FROM whale_snapshots
                WHERE hit_tp1 IS NOT NULL OR hit_sl IS NOT NULL
                ORDER BY {timestamp_col} DESC
            """
        else:
            query = """
                SELECT * FROM whale_snapshots
                WHERE hit_tp1 IS NOT NULL OR hit_sl IS NOT NULL
            """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"   Records with outcomes: {len(rows)}")
        
        for row in rows:
            row_dict = dict(row)
            
            hit_tp1 = row_dict.get('hit_tp1') == 1
            hit_sl = row_dict.get('hit_sl') == 1
            
            if not hit_tp1 and not hit_sl:
                continue
            
            features = extract_features(row_dict)
            outcome = 'WIN' if hit_tp1 else 'LOSS'
            
            records.append({
                'features': features,
                'outcome': outcome,
                'direction': 'LONG',  # Assuming long trades
                'symbol': row_dict.get('symbol'),
                'timestamp': row_dict.get(timestamp_col) if timestamp_col else None,
                'market': 'crypto',
            })
        
        # Also check total records
        cursor.execute("SELECT COUNT(*) FROM whale_snapshots")
        total = cursor.fetchone()[0]
        print(f"   Total records in DB: {total}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return records


def extract_stock_training_data(stock_history_dir):
    """Extract training data from stock_history JSON files"""
    
    records = []
    
    if not os.path.exists(stock_history_dir):
        print(f"\n📂 Stock history dir not found: {stock_history_dir}")
        return records
    
    print(f"\n📂 Loading stock data from: {stock_history_dir}")
    
    json_files = [f for f in os.listdir(stock_history_dir) if f.endswith('.json') or f.endswith('_history')]
    print(f"   Found {len(json_files)} stock files")
    
    wins = 0
    losses = 0
    
    for filename in json_files:
        filepath = os.path.join(stock_history_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            snapshots = []
            if isinstance(data, list):
                snapshots = data
            elif isinstance(data, dict):
                snapshots = data.get('snapshots', data.get('history', [data]))
            
            for snapshot in snapshots:
                if not isinstance(snapshot, dict):
                    continue
                
                # Check for outcome
                hit_tp1 = snapshot.get('hit_tp1') == 1 or snapshot.get('hit_tp1') == True
                hit_sl = snapshot.get('hit_sl') == 1 or snapshot.get('hit_sl') == True
                
                if not hit_tp1 and not hit_sl:
                    continue
                
                features = extract_stock_features(snapshot)
                outcome = 'WIN' if hit_tp1 else 'LOSS'
                
                if outcome == 'WIN':
                    wins += 1
                else:
                    losses += 1
                
                symbol = snapshot.get('symbol') or filename.replace('_history', '').replace('.json', '')
                
                records.append({
                    'features': features,
                    'outcome': outcome,
                    'direction': 'LONG',
                    'symbol': symbol,
                    'timestamp': snapshot.get('timestamp') or snapshot.get('date'),
                    'market': 'stock',
                })
                
        except Exception as e:
            continue  # Skip problematic files
    
    print(f"   Stock records with outcomes: {len(records)}")
    if len(records) > 0:
        print(f"   Wins: {wins} | Losses: {losses} ({wins/(wins+losses)*100:.1f}% win rate)")
    
    return records


def extract_stock_features(row):
    """Extract features from a stock data record"""
    
    # Stock-specific fields (from Quiver data)
    congress_score = row.get('congress_score') or row.get('congress_sentiment') or 50
    insider_score = row.get('insider_score') or row.get('insider_sentiment') or 50
    short_interest = row.get('short_interest') or row.get('short_pct') or 0
    
    # Map to similar structure as crypto
    # Use institutional scores as proxy for "whale" activity
    inst_score = (congress_score + insider_score) / 2
    
    position_pct = (row.get('position_pct') or row.get('position_in_range') or 
                    row.get('range_position') or 50)
    
    # Position label
    if position_pct <= 35:
        position_label = 0  # EARLY
    elif position_pct >= 65:
        position_label = 2  # LATE
    else:
        position_label = 1  # MIDDLE
    
    # RSI zone
    rsi = row.get('rsi') or row.get('rsi_14') or 50
    if rsi <= 30:
        rsi_zone = -1
    elif rsi >= 70:
        rsi_zone = 1
    else:
        rsi_zone = 0
    
    # Trend
    trend = row.get('trend') or row.get('structure') or 'NEUTRAL'
    trend_encoded = TREND_MAP.get(str(trend).upper() if trend else 'NEUTRAL', 0)
    
    # Price change
    price_change = (row.get('price_change_24h') or row.get('price_change') or 
                    row.get('change_pct') or 0)
    
    features = np.array([
        inst_score,           # Use institutional as "whale"
        50,                   # No retail equivalent for stocks
        inst_score - 50,      # Divergence proxy
        (inst_score - 50) / 50,
        0,                    # No funding rate for stocks
        0,                    # No OI for stocks
        0,
        price_change,
        0,
        position_pct,
        position_label,
        0,
        row.get('ta_score') or row.get('score') or 50,
        rsi,
        rsi_zone,
        trend_encoded,
        0,
        0,                    # Money flow
        1.0,
        0, 0, 0, 0,           # SMC features
        0,                    # No BTC correlation for stocks
        0,
        50,
        0,
        row.get('historical_win_rate') or 50,
        row.get('sample_size') or 0,
        0,
    ], dtype=np.float32)
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(records):
    """Train ML models from extracted records"""
    
    print(f"\n🎯 Training with {len(records)} records...")
    
    # Prepare data
    X = np.array([r['features'] for r in records])
    
    # Target: 2=LONG (win), 1=WAIT (loss as we should have waited), 0=SHORT
    y = np.array([2 if r['outcome'] == 'WIN' else 1 for r in records])
    
    print(f"   Features shape: {X.shape}")
    print(f"   Wins: {sum(y == 2)}, Should-Wait: {sum(y == 1)}")
    
    # Check for sklearn
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        print("\n❌ scikit-learn not installed!")
        print("   Run: pip install scikit-learn")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try LightGBM first, fallback to RandomForest
    try:
        import lightgbm as lgb
        print("\n🌳 Training with LightGBM...")
        
        direction_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
        tp_sl_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        
    except ImportError:
        print("\n🌳 Training with RandomForest (install lightgbm for better results)...")
        
        direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        tp_sl_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
    
    # Train direction model
    direction_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = direction_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Direction Model Accuracy: {accuracy:.1%}")
    
    # Train TP/SL model (train on TP1 only - 1D target)
    # We'll use fixed ratios for TP2 and SL relative to TP1
    y_tp1 = np.array([2.5 for _ in records])  # Default TP1 percentage
    y_tp1_train, y_tp1_test = train_test_split(y_tp1, test_size=0.2, random_state=42)
    tp_sl_model.fit(X_train_scaled, y_tp1_train)
    
    # Feature importances
    if hasattr(direction_model, 'feature_importances_'):
        importances = direction_model.feature_importances_
        top_features = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("\n📈 Top 10 Important Features:")
        for name, imp in top_features:
            bar = "█" * int(imp * 50)
            print(f"   {name:30} {bar} {imp:.3f}")
    
    # Note: LightGBM uses numeric classes internally (0, 1, 2)
    # We'll map them back to strings when making predictions
    # 0=SHORT, 1=WAIT, 2=LONG
    
    return {
        'direction_model': direction_model,
        'tp_sl_model': tp_sl_model,
        'scaler': scaler,
        'accuracy': accuracy,
        'n_samples': len(records),
        'class_mapping': {0: 'SHORT', 1: 'WAIT', 2: 'LONG'},
    }
    


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(models):
    """Save trained models to ml/models/"""
    
    # Create directory
    model_dir = os.path.join(os.path.dirname(__file__), 'ml', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n💾 Saving models to: {model_dir}")
    
    # Save direction model
    with open(os.path.join(model_dir, 'direction_model.pkl'), 'wb') as f:
        pickle.dump(models['direction_model'], f)
    print("   ✅ direction_model.pkl")
    
    # Save TP/SL model
    with open(os.path.join(model_dir, 'tp_sl_model.pkl'), 'wb') as f:
        pickle.dump(models['tp_sl_model'], f)
    print("   ✅ tp_sl_model.pkl")
    
    # Save scaler
    with open(os.path.join(model_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(models['scaler'], f)
    print("   ✅ feature_scaler.pkl")
    
    # Save metadata
    metadata = {
        'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'trained_at': datetime.now().isoformat(),
        'n_samples': models['n_samples'],
        'direction_accuracy': float(models['accuracy']),
        'feature_names': FEATURE_NAMES,
        'class_mapping': {'0': 'SHORT', '1': 'WAIT', '2': 'LONG'},
    }
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   ✅ model_metadata.json")
    
    return model_dir


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Find database
    db_path = find_whale_db()
    
    if db_path is None:
        print("\n❌ whale_history.db not found!")
        print("\nUsage:")
        print("   python train_ml.py                         # Auto-detect")
        print("   python train_ml.py /path/to/whale_history.db  # Specify path")
        print("\nCommon locations checked:")
        print("   - ./whale_history.db")
        print("   - ./data/whale_history.db")
        print("   - ~/TScanner/whale_history.db")
        sys.exit(1)
    
    print(f"\n✅ Found database: {db_path}")
    
    # Extract crypto data
    records = extract_training_data(db_path)
    crypto_count = len(records)
    
    # Extract stock data
    stock_history_dir = os.path.join(os.path.dirname(db_path), 'stock_history')
    if os.path.exists(stock_history_dir):
        stock_records = extract_stock_training_data(stock_history_dir)
        records.extend(stock_records)
        stock_count = len(stock_records)
    else:
        stock_count = 0
        print(f"\n📂 No stock_history folder found at: {stock_history_dir}")
    
    print(f"\n📊 Total training data:")
    print(f"   Crypto records: {crypto_count}")
    print(f"   Stock records:  {stock_count}")
    print(f"   Combined:       {len(records)}")
    
    if len(records) < 50:
        print(f"\n⚠️ Only {len(records)} records with outcomes found.")
        print("   Need at least 50 for meaningful training (100+ recommended).")
        print("\n📝 To add outcomes to your data, run:")
        print("   python backtest_outcomes.py")
        
        if len(records) < 10:
            print("\n❌ Not enough data to train. Exiting.")
            sys.exit(1)
        
        print(f"\n⚠️ Training anyway with {len(records)} records (results may be poor)...")
    
    # Train models
    models = train_models(records)
    
    if models is None:
        sys.exit(1)
    
    # Save models
    model_dir = save_models(models)
    
    print("\n" + "=" * 60)
    print("✅ ML TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Results:")
    print(f"   Crypto records: {crypto_count}")
    print(f"   Stock records:  {stock_count}")
    print(f"   Total used:     {models['n_samples']}")
    print(f"   Accuracy:       {models['accuracy']:.1%}")
    print(f"\n🎛️ You can now use ML mode in the app!")
    print("   Select '🤖 ML Model' or '⚡ Hybrid' in the sidebar.")


if __name__ == '__main__':
    main()