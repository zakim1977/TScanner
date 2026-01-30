#!/usr/bin/env python3
"""
InvestorIQ Multi-Model ML Training
===================================

Trains multiple ML models and automatically selects the best one!

Models trained:
- LightGBM (if installed)
- XGBoost (if installed)
- Random Forest
- Gradient Boosting
- Extra Trees
- Logistic Regression (baseline)

Usage:
    python3 train_ml_multi.py                    # Auto-detect whale_history.db
    python3 train_ml_multi.py path/to/whale_history.db   # Specify path
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

print("=" * 70)
print("🤖 INVESTORIQ MULTI-MODEL ML TRAINING")
print("=" * 70)
print("   Training multiple models and selecting the BEST one!")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# FIND DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def find_whale_db():
    """Find whale_history.db in common locations"""
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            return path
        else:
            print(f"❌ File not found: {path}")
            sys.exit(1)
    
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
# FEATURE EXTRACTION (same as train_ml.py)
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
    'FOMO / DIST RISK': -1, 'EXHAUSTION': -1,
    'NEUTRAL': 0, 'TRANSITION': 0, 'RANGING': 0,
    'ACCUMULATION': 1, 'EARLY_ACCUMULATION': 1, 'MARKUP': 2, 'EXPANSION': 2
}

TREND_MAP = {
    'Bearish': -1, 'BEARISH': -1, 'bearish': -1,
    'Unknown': 0, 'Mixed': 0, 'Consolidating': 0, 'Ranging': 0, 'NEUTRAL': 0,
    'Bullish': 1, 'BULLISH': 1, 'bullish': 1
}


def extract_features_for_prediction(row):
    """
    Extract features for PREDICTION - excludes price_change to prevent data leakage!
    
    This is used when we're creating labels from future price changes.
    We can't use current price_change as a feature if we're predicting future price.
    """
    try:
        # Handle different column naming conventions
        whale_pct = row.get('whale_long_pct') or row.get('whale_pct') or row.get('top_trader_long_pct') or 50
        retail_pct = row.get('retail_long_pct') or row.get('retail_pct') or 50
        oi_change = row.get('oi_change_24h') or row.get('oi_change') or 0
        funding = row.get('funding_rate') or row.get('funding') or 0
        position_pct = row.get('position_in_range') or row.get('position_pct') or 50
        ta_score = row.get('ta_score') or 50
        trend = row.get('trend') or row.get('structure') or 'Unknown'
        money_flow = row.get('money_flow_phase') or row.get('money_flow') or 'NEUTRAL'
        
        # Ensure numeric values
        whale_pct = float(whale_pct) if whale_pct else 50
        retail_pct = float(retail_pct) if retail_pct else 50
        oi_change = float(oi_change) if oi_change else 0
        funding = float(funding) if funding else 0
        position_pct = float(position_pct) if position_pct else 50
        ta_score = float(ta_score) if ta_score else 50
        
        # Derived features (NO price_change - that would be leakage!)
        divergence = whale_pct - retail_pct
        whale_dominance = whale_pct / max(retail_pct, 1)
        
        # Encode categoricals
        oi_signal = 1 if oi_change > 2 else (-1 if oi_change < -2 else 0)
        position_label = 0 if position_pct < 35 else (2 if position_pct > 65 else 1)
        range_size = abs(oi_change)  # Removed price_change from here too
        
        rsi = 50
        rsi_zone = 0 if rsi < 30 else (2 if rsi > 70 else 1)
        
        trend_encoded = TREND_MAP.get(str(trend), 0)
        money_flow_encoded = MONEY_FLOW_MAP.get(str(money_flow), 0)
        volume_ratio = 1.0
        
        at_bullish_ob = 0
        at_bearish_ob = 0
        near_support = 1 if position_pct < 30 else 0
        near_resistance = 1 if position_pct > 70 else 0
        
        btc_correlation = 0.5
        btc_trend = 0
        fear_greed = 50
        is_weekend = 0
        
        historical_win_rate = row.get('historical_win_rate') or row.get('win_rate') or 0.5
        similar_count = row.get('similar_setup_count') or row.get('similar_count') or 0
        avg_return = row.get('avg_historical_return') or row.get('avg_return') or 0
        
        historical_win_rate = float(historical_win_rate) if historical_win_rate else 0.5
        similar_count = float(similar_count) if similar_count else 0
        avg_return = float(avg_return) if avg_return else 0
        
        # Feature vector - note: price_change features are set to 0 to prevent leakage
        # but we keep the same feature count for compatibility
        features = [
            whale_pct, retail_pct, divergence, whale_dominance, funding,
            oi_change, oi_signal, 0, 0,  # price_change_24h, price_change_1h = 0 (excluded!)
            position_pct, position_label, range_size,
            ta_score, rsi, rsi_zone, trend_encoded, 0,  # volatility = 0 (excluded!)
            money_flow_encoded, volume_ratio,
            at_bullish_ob, at_bearish_ob, near_support, near_resistance,
            btc_correlation, btc_trend, fear_greed, is_weekend,
            historical_win_rate, similar_count, avg_return
        ]
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        return None


def extract_features_from_row(row):
    """Extract feature vector from database row"""
    try:
        # Handle different column naming conventions
        whale_pct = row.get('whale_long_pct') or row.get('whale_pct') or row.get('top_trader_long_pct') or 50
        retail_pct = row.get('retail_long_pct') or row.get('retail_pct') or 50
        oi_change = row.get('oi_change_24h') or row.get('oi_change') or 0
        price_change_24h = row.get('price_change_24h') or row.get('price_change') or 0
        price_change_1h = row.get('price_change_1h') or 0
        funding = row.get('funding_rate') or row.get('funding') or 0
        position_pct = row.get('position_in_range') or row.get('position_pct') or 50
        ta_score = row.get('ta_score') or 50
        trend = row.get('trend') or row.get('structure') or 'Unknown'
        money_flow = row.get('money_flow_phase') or row.get('money_flow') or 'NEUTRAL'
        
        # Ensure numeric values
        whale_pct = float(whale_pct) if whale_pct else 50
        retail_pct = float(retail_pct) if retail_pct else 50
        oi_change = float(oi_change) if oi_change else 0
        price_change_24h = float(price_change_24h) if price_change_24h else 0
        price_change_1h = float(price_change_1h) if price_change_1h else 0
        funding = float(funding) if funding else 0
        position_pct = float(position_pct) if position_pct else 50
        ta_score = float(ta_score) if ta_score else 50
        
        # Derived features
        divergence = whale_pct - retail_pct
        whale_dominance = whale_pct / max(retail_pct, 1)
        
        # Encode categoricals
        oi_signal = 1 if oi_change > 2 else (-1 if oi_change < -2 else 0)
        position_label = 0 if position_pct < 35 else (2 if position_pct > 65 else 1)
        range_size = abs(oi_change) + abs(price_change_24h)
        
        rsi = 50  # Default if not available
        rsi_zone = 0 if rsi < 30 else (2 if rsi > 70 else 1)
        
        trend_encoded = TREND_MAP.get(str(trend), 0)
        volatility = abs(price_change_24h) + abs(price_change_1h)
        money_flow_encoded = MONEY_FLOW_MAP.get(str(money_flow), 0)
        volume_ratio = 1.0
        
        at_bullish_ob = 0
        at_bearish_ob = 0
        near_support = 1 if position_pct < 30 else 0
        near_resistance = 1 if position_pct > 70 else 0
        
        btc_correlation = 0.5
        btc_trend = 0
        fear_greed = 50
        is_weekend = 0
        
        historical_win_rate = row.get('historical_win_rate') or row.get('win_rate') or 0.5
        similar_count = row.get('similar_setup_count') or row.get('similar_count') or 0
        avg_return = row.get('avg_historical_return') or row.get('avg_return') or 0
        
        # Ensure numeric
        historical_win_rate = float(historical_win_rate) if historical_win_rate else 0.5
        similar_count = float(similar_count) if similar_count else 0
        avg_return = float(avg_return) if avg_return else 0
        
        features = [
            whale_pct, retail_pct, divergence, whale_dominance, funding,
            oi_change, oi_signal, price_change_24h, price_change_1h,
            position_pct, position_label, range_size,
            ta_score, rsi, rsi_zone, trend_encoded, volatility,
            money_flow_encoded, volume_ratio,
            at_bullish_ob, at_bearish_ob, near_support, near_resistance,
            btc_correlation, btc_trend, fear_greed, is_weekend,
            historical_win_rate, similar_count, avg_return
        ]
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        return None


def load_training_data(db_path):
    """Load and prepare training data from database"""
    
    print(f"\n📂 Loading data from: {db_path}")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"   Tables found: {tables}")
    
    # Show schema for whale_snapshots
    if 'whale_snapshots' in tables:
        cursor.execute("PRAGMA table_info(whale_snapshots)")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"   whale_snapshots columns: {columns[:10]}..." if len(columns) > 10 else f"   whale_snapshots columns: {columns}")
    
    records = []
    
    # Method 1: Try trade_outcomes table (if exists)
    if 'trade_outcomes' in tables:
        print("   Using trade_outcomes table...")
        cursor.execute("""
            SELECT o.*, s.* 
            FROM trade_outcomes o
            JOIN whale_snapshots s ON o.snapshot_id = s.id
            WHERE o.outcome IN ('WIN', 'LOSS', 'PARTIAL')
        """)
        
        outcome_rows = cursor.fetchall()
        print(f"   Found {len(outcome_rows)} outcome records")
        
        for row in outcome_rows:
            row_dict = dict(row)
            features = extract_features_from_row(row_dict)
            if features is not None:
                records.append({
                    'features': features,
                    'outcome': row_dict.get('outcome', 'LOSS'),
                    'direction': row_dict.get('direction', 'LONG'),
                    'pnl_pct': row_dict.get('pnl_pct', 0) or 0
                })
    
    # Method 2: Generate labels from price changes in whale_snapshots
    if len(records) < 50 and 'whale_snapshots' in tables:
        print("   Generating labels from whale_snapshots data...")
        print("   ⚠️ Using LAGGED approach to prevent data leakage...")
        
        # Get snapshots ordered by time
        cursor.execute("""
            SELECT * FROM whale_snapshots 
            WHERE whale_long_pct IS NOT NULL 
            ORDER BY symbol, timestamp ASC
        """)
        
        snapshot_rows = cursor.fetchall()
        print(f"   Found {len(snapshot_rows)} snapshots")
        
        # Group by symbol to find sequential snapshots
        from collections import defaultdict
        symbol_snapshots = defaultdict(list)
        for row in snapshot_rows:
            row_dict = dict(row)
            symbol_snapshots[row_dict.get('symbol', 'UNKNOWN')].append(row_dict)
        
        print(f"   Grouped into {len(symbol_snapshots)} symbols")
        
        # For each symbol, use CURRENT snapshot features to predict NEXT snapshot outcome
        for symbol, snapshots in symbol_snapshots.items():
            if len(snapshots) < 2:
                continue
                
            for i in range(len(snapshots) - 1):
                current = snapshots[i]
                future = snapshots[i + 1]
                
                # Features from CURRENT snapshot (excluding price_change which would leak)
                features = extract_features_for_prediction(current)
                
                if features is None:
                    continue
                
                # Label from FUTURE price movement
                future_price_change = future.get('price_change_24h') or future.get('price_change') or 0
                current_whale_pct = current.get('whale_long_pct') or current.get('whale_pct') or 50
                
                try:
                    future_price_change = float(future_price_change)
                    current_whale_pct = float(current_whale_pct)
                except:
                    continue
                
                # Outcome: Did the FUTURE price move in the direction whales predicted?
                if current_whale_pct > 55 and future_price_change > 0.5:
                    outcome = 'WIN'
                elif current_whale_pct < 45 and future_price_change < -0.5:
                    outcome = 'WIN'
                elif current_whale_pct > 60 and future_price_change > 1.5:
                    outcome = 'WIN'
                elif current_whale_pct < 40 and future_price_change < -1.5:
                    outcome = 'WIN'
                elif abs(future_price_change) < 0.3:
                    continue  # Skip tiny moves
                else:
                    outcome = 'LOSS'
                
                records.append({
                    'features': features,
                    'outcome': outcome,
                    'direction': 'LONG' if current_whale_pct > 50 else 'SHORT',
                    'pnl_pct': future_price_change
                })
    
    conn.close()
    
    print(f"   Extracted {len(records)} valid training records")
    
    # Show outcome distribution
    if len(records) > 0:
        wins = sum(1 for r in records if r['outcome'] == 'WIN')
        losses = len(records) - wins
        print(f"   Distribution: {wins} wins ({wins/len(records)*100:.1f}%), {losses} losses ({losses/len(records)*100:.1f}%)")
    
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_models():
    """Get all available models to train"""
    
    models = {}
    
    # Check for sklearn (required)
    try:
        from sklearn.ensemble import (
            RandomForestClassifier, 
            GradientBoostingClassifier,
            ExtraTreesClassifier,
            AdaBoostClassifier
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=50,  # Reduced from 150 for speed
            max_depth=4,      # Reduced from 6
            learning_rate=0.15,
            random_state=42
        )
        
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        models['AdaBoost'] = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        )
        
        models['LogisticRegression'] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        models['KNN'] = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            n_jobs=-1
        )
        
    except ImportError:
        print("❌ scikit-learn not installed! Run: pip install scikit-learn")
        return None
    
    # Check for LightGBM
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        print("   ✅ LightGBM available")
    except ImportError:
        print("   ⚠️ LightGBM not installed (pip install lightgbm)")
    
    # Check for XGBoost
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        print("   ✅ XGBoost available")
    except ImportError:
        print("   ⚠️ XGBoost not installed (pip install xgboost)")
    
    # Check for CatBoost
    try:
        from catboost import CatBoostClassifier
        models['CatBoost'] = CatBoostClassifier(
            iterations=100,  # Reduced from 200
            depth=5,         # Reduced from 6
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        print("   ✅ CatBoost available")
    except ImportError:
        print("   ⚠️ CatBoost not installed (pip install catboost)")
    
    return models


def train_and_evaluate_models(records):
    """Train all models and return results sorted by performance"""
    
    print(f"\n🎯 Training with {len(records)} records...")
    
    # Prepare data
    X = np.array([r['features'] for r in records])
    y = np.array([1 if r['outcome'] == 'WIN' else 0 for r in records])
    
    print(f"   Features shape: {X.shape}")
    print(f"   Class distribution: WIN={sum(y==1)}, LOSS={sum(y==0)}")
    
    # Import sklearn
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get available models
    print("\n🔍 Checking available models...")
    models = get_available_models()
    
    if models is None:
        return None, None, None
    
    print(f"\n📊 Training {len(models)} models with 5-fold cross-validation...")
    print("-" * 70)
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        try:
            print(f"\n   🔄 Training {name}...", end=" ")
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Test set evaluation
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
            
            results.append({
                'name': name,
                'model': model,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'score': (cv_mean * 0.4 + accuracy * 0.3 + f1 * 0.2 + auc * 0.1)  # Combined score
            })
            
            print(f"CV: {cv_mean:.1%}±{cv_std:.1%} | Test: {accuracy:.1%} | F1: {f1:.2f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Sort by combined score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results, scaler, (X_test_scaled, y_test)


def display_results(results):
    """Display model comparison results"""
    
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON RESULTS (sorted by combined score)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Model':<20} {'CV Score':<12} {'Test Acc':<10} {'F1':<8} {'AUC':<8} {'Combined':<10}")
    print("-" * 70)
    
    for i, r in enumerate(results, 1):
        marker = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{marker}{i:<3} {r['name']:<20} {r['cv_mean']:.1%}±{r['cv_std']:.1%}  {r['accuracy']:.1%}       {r['f1']:.2f}     {r['auc']:.2f}     {r['score']:.3f}")
    
    print("-" * 70)
    
    best = results[0]
    print(f"\n🏆 BEST MODEL: {best['name']}")
    print(f"   Cross-Validation: {best['cv_mean']:.1%} (±{best['cv_std']:.1%})")
    print(f"   Test Accuracy:    {best['accuracy']:.1%}")
    print(f"   Precision:        {best['precision']:.1%}")
    print(f"   Recall:           {best['recall']:.1%}")
    print(f"   F1 Score:         {best['f1']:.2f}")
    print(f"   ROC AUC:          {best['auc']:.2f}")


def save_best_model(results, scaler):
    """Save the best model to ml/models/"""
    
    best = results[0]
    model = best['model']
    
    # Create output directory
    output_dir = Path(__file__).parent / "ml" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "direction_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Saved direction model: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / "feature_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Saved feature scaler: {scaler_path}")
    
    # Create dummy TP/SL model (for compatibility)
    from sklearn.ensemble import RandomForestRegressor
    tp_sl_model = RandomForestRegressor(n_estimators=10, random_state=42)
    tp_sl_model.fit([[0]*30], [2.5])  # Dummy fit
    
    tp_sl_path = output_dir / "tp_sl_model.pkl"
    with open(tp_sl_path, 'wb') as f:
        pickle.dump(tp_sl_model, f)
    print(f"💾 Saved TP/SL model: {tp_sl_path}")
    
    # Save metadata
    metadata = {
        'version': datetime.now().strftime('v%Y%m%d_%H%M%S'),
        'best_model': best['name'],
        'cv_accuracy': float(best['cv_mean']),
        'test_accuracy': float(best['accuracy']),
        'f1_score': float(best['f1']),
        'auc_score': float(best['auc']),
        'combined_score': float(best['score']),
        'feature_names': FEATURE_NAMES,
        'all_models_tested': [
            {
                'name': r['name'],
                'cv_mean': float(r['cv_mean']),
                'accuracy': float(r['accuracy']),
                'f1': float(r['f1']),
                'score': float(r['score'])
            }
            for r in results
        ],
        'trained_at': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")
    
    # Show feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print(f"\n📈 Top 10 Important Features ({best['name']}):")
        for name, imp in top_features:
            bar = "█" * int(imp * 50)
            print(f"   {name:30} {bar} {imp:.3f}")
    
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main training function"""
    
    # Find database
    db_path = find_whale_db()
    
    if db_path is None:
        print("\n❌ Could not find whale_history.db!")
        print("\nLocations checked:")
        print("   - whale_history.db")
        print("   - data/whale_history.db")
        print("   - ~/whale_history.db")
        print("   - ~/TScanner/whale_history.db")
        print("\nUsage: python3 train_ml_multi.py /path/to/whale_history.db")
        sys.exit(1)
    
    # Load data
    records = load_training_data(db_path)
    
    if len(records) == 0:
        print("\n❌ No training records found!")
        print("   Make sure your whale_history.db has whale_snapshots data.")
        print("   Run Scanner with 'Store whale data' enabled to collect data.")
        sys.exit(1)
    
    if len(records) < 50:
        print(f"\n⚠️ Only {len(records)} records found.")
        print("   Need at least 50 records for reliable training.")
        print("   Keep running Scanner with 'Store whale data' enabled!")
        
        if len(records) < 20:
            print("\n❌ Too few records to train. Exiting.")
            sys.exit(1)
    
    # Train all models
    results, scaler, test_data = train_and_evaluate_models(records)
    
    if results is None:
        print("\n❌ Training failed!")
        sys.exit(1)
    
    # Display results
    display_results(results)
    
    # Save best model
    best = save_best_model(results, scaler)
    
    print("\n" + "=" * 70)
    print("✅ MULTI-MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n🏆 Best Model: {best['name']} ({best['accuracy']:.1%} accuracy)")
    print("\n   Model saved to: ml/models/")
    print("   Restart Streamlit to use the new model!")
    print("\n💡 Tips:")
    print("   - Install lightgbm & xgboost for more options: pip install lightgbm xgboost")
    print("   - More training data = better models!")
    print("   - Re-run this script periodically as you collect more data")


if __name__ == "__main__":
    main()