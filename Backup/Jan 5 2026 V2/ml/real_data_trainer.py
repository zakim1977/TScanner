"""
Real Data ML Trainer
====================
Trains ML models on REAL whale/institutional data instead of simulated data.

Data Sources:
- Crypto: whale_history.db (183K+ records with real whale%, retail%, OI, outcomes)
- Stock: data/stock_history/*.json (institutional congress/insider scores, outcomes)

This replaces the old trainer that simulated whale data from volume/RSI.

Models Produced:
- crypto_direction_model.pkl   → Predicts LONG/SHORT/WAIT for crypto
- crypto_tp_model.pkl          → Predicts TP probability
- stock_direction_model.pkl    → Predicts LONG/SHORT/WAIT for stocks
- stock_tp_model.pkl           → Predicts TP probability

Usage:
    from ml.real_data_trainer import RealDataTrainer
    
    trainer = RealDataTrainer()
    trainer.train_crypto_models()
    trainer.train_stock_models()
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn not found. Install with: pip install scikit-learn")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'real_data')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Crypto features - REAL data from whale_history.db
CRYPTO_FEATURES = [
    'whale_long_pct',       # Real whale positioning (0-100)
    'retail_long_pct',      # Real retail positioning (0-100)
    'whale_retail_diff',    # Derived: whale - retail (divergence)
    'oi_change_24h',        # Real OI change
    'funding_rate',         # Real funding rate
    'position_in_range',    # Position % (0-100)
    'position_early',       # 1 if position < 35
    'position_late',        # 1 if position > 65
    'mfi',                  # Money Flow Index
    'rsi',                  # RSI
    'rsi_oversold',         # 1 if RSI < 30
    'rsi_overbought',       # 1 if RSI > 70
    'volume_ratio',         # Volume vs average
    'atr_pct',              # ATR as % of price
    'price_change_24h',     # 24h price change
    'whale_dominant',       # 1 if whale > 65
    'retail_dominant',      # 1 if retail > whale + 10
    'squeeze_potential',    # whale > 60 AND retail < 50
]

# Stock features - REAL data from JSON files
STOCK_FEATURES = [
    'congress_score',       # Congress trading activity (0-100)
    'insider_score',        # Insider trading score (0-100)
    'short_interest_pct',   # Short interest %
    'combined_score',       # Combined institutional score
    'institutional_diff',   # Derived: congress - insider
    'position_pct',         # Position in range (0-100)
    'position_early',       # 1 if position < 35
    'position_late',        # 1 if position > 65
    'ta_score',             # Technical analysis score
    'rsi',                  # RSI
    'rsi_oversold',         # 1 if RSI < 30
    'rsi_overbought',       # 1 if RSI > 70
    'price_change_24h',     # 24h price change
    'congress_bullish',     # 1 if congress > 60
    'insider_bullish',      # 1 if insider > 60
    'short_squeeze_risk',   # 1 if short_interest > 15
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

class CryptoDataLoader:
    """Load REAL crypto data from whale_history.db"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(DATA_DIR, 'whale_history.db')
    
    def load_training_data(self, min_samples: int = 1000) -> pd.DataFrame:
        """
        Load training data from whale_history.db
        Only loads records WITH outcome data (hit_tp1 IS NOT NULL)
        """
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return pd.DataFrame()
        
        print(f"📂 Loading from {self.db_path}...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load records with outcomes
        query = """
        SELECT 
            symbol, timestamp, 
            whale_long_pct, retail_long_pct, 
            oi_change_24h, funding_rate,
            position_in_range, mfi, rsi, 
            volume_ratio, atr_pct, price_change_24h,
            signal_direction, predictive_score,
            hit_tp1, hit_sl, 
            candles_to_result, max_favorable_pct, max_adverse_pct
        FROM whale_snapshots 
        WHERE hit_tp1 IS NOT NULL 
          AND whale_long_pct IS NOT NULL
          AND whale_long_pct > 0
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df):,} records with outcomes")
        
        if len(df) < min_samples:
            print(f"⚠️ Need at least {min_samples} samples for training")
            return pd.DataFrame()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and labels from raw data.
        
        Returns:
            X: Feature matrix
            y: Labels (0=SHORT, 1=WAIT, 2=LONG)
            feature_names: List of feature names
        """
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Create derived features
        df = df.copy()
        
        # Whale-Retail divergence (KEY FEATURE!)
        df['whale_retail_diff'] = df['whale_long_pct'] - df['retail_long_pct']
        
        # Position flags
        df['position_early'] = (df['position_in_range'] < 35).astype(int)
        df['position_late'] = (df['position_in_range'] > 65).astype(int)
        
        # RSI flags
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # Whale/Retail dominance flags
        df['whale_dominant'] = (df['whale_long_pct'] > 65).astype(int)
        df['retail_dominant'] = ((df['retail_long_pct'] - df['whale_long_pct']) > 10).astype(int)
        
        # Squeeze potential (whales bullish, retail not)
        df['squeeze_potential'] = ((df['whale_long_pct'] > 60) & (df['retail_long_pct'] < 50)).astype(int)
        
        # Fill NaN values
        for col in CRYPTO_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 50)
        
        # Build feature matrix
        feature_cols = [c for c in CRYPTO_FEATURES if c in df.columns]
        X = df[feature_cols].values
        
        # Build labels from outcomes
        # LONG: hit_tp1=1 AND max_favorable > max_adverse (profitable long)
        # SHORT: hit_sl=1 AND price went down significantly
        # WAIT: neither clear
        
        labels = []
        for _, row in df.iterrows():
            hit_tp1 = row['hit_tp1']
            hit_sl = row['hit_sl']
            max_fav = row['max_favorable_pct'] or 0
            max_adv = abs(row['max_adverse_pct'] or 0)
            
            if hit_tp1 == 1 and max_fav > max_adv:
                labels.append(2)  # LONG
            elif hit_sl == 1 or max_adv > max_fav * 1.5:
                labels.append(0)  # SHORT (or avoid long)
            else:
                labels.append(1)  # WAIT
        
        y = np.array(labels)
        
        print(f"📊 Features: {len(feature_cols)}, Samples: {len(X)}")
        print(f"   Labels: LONG={sum(y==2)}, WAIT={sum(y==1)}, SHORT={sum(y==0)}")
        
        return X, y, feature_cols


class StockDataLoader:
    """Load REAL stock data from JSON files"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(DATA_DIR, 'stock_history')
    
    def load_training_data(self, min_samples: int = 100) -> pd.DataFrame:
        """Load training data from all stock JSON files"""
        if not os.path.exists(self.data_dir):
            print(f"❌ Stock data directory not found: {self.data_dir}")
            return pd.DataFrame()
        
        print(f"📂 Loading from {self.data_dir}...")
        
        all_records = []
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        for filename in json_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_records.extend(data)
            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
        
        if not all_records:
            print("❌ No stock data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        
        # Filter to records with outcomes
        df = df[df['outcome_tracked'] == True].copy()
        
        print(f"✅ Loaded {len(df):,} records with outcomes from {len(json_files)} stocks")
        
        if len(df) < min_samples:
            print(f"⚠️ Need at least {min_samples} samples for training")
            return pd.DataFrame()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and labels from stock data"""
        if df.empty:
            return np.array([]), np.array([]), []
        
        df = df.copy()
        
        # Create derived features
        df['institutional_diff'] = df['congress_score'] - df['insider_score']
        
        # Position flags
        df['position_early'] = (df['position_pct'] < 35).astype(int)
        df['position_late'] = (df['position_pct'] > 65).astype(int)
        
        # RSI flags  
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # Institutional flags
        df['congress_bullish'] = (df['congress_score'] > 60).astype(int)
        df['insider_bullish'] = (df['insider_score'] > 60).astype(int)
        
        # Short squeeze risk
        df['short_squeeze_risk'] = (df['short_interest_pct'] > 15).astype(int)
        
        # Fill NaN values
        for col in STOCK_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 50)
        
        # Build feature matrix
        feature_cols = [c for c in STOCK_FEATURES if c in df.columns]
        X = df[feature_cols].values
        
        # Build labels from outcomes
        labels = []
        for _, row in df.iterrows():
            hit_tp1 = row.get('hit_tp1', False)
            hit_sl = row.get('hit_sl', False)
            max_fav = row.get('max_favorable_pct', 0) or 0
            max_adv = abs(row.get('max_adverse_pct', 0) or 0)
            
            if hit_tp1 and max_fav > max_adv:
                labels.append(2)  # LONG
            elif hit_sl or max_adv > max_fav * 1.5:
                labels.append(0)  # SHORT/AVOID
            else:
                labels.append(1)  # WAIT
        
        y = np.array(labels)
        
        print(f"📊 Features: {len(feature_cols)}, Samples: {len(X)}")
        print(f"   Labels: LONG={sum(y==2)}, WAIT={sum(y==1)}, SHORT={sum(y==0)}")
        
        return X, y, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class RealDataTrainer:
    """Train ML models on REAL whale/institutional data"""
    
    def __init__(self):
        self.crypto_loader = CryptoDataLoader()
        self.stock_loader = StockDataLoader()
        
        # Create model directory
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def _get_best_classifier(self):
        """Get the best available classifier"""
        if HAS_XGB:
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif HAS_LGBM:
            return LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        elif HAS_SKLEARN:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ImportError("No ML library available!")
    
    def train_crypto_models(self) -> Dict:
        """Train models on REAL crypto whale data"""
        print("\n" + "=" * 60)
        print("🐋 TRAINING CRYPTO MODELS ON REAL WHALE DATA")
        print("=" * 60)
        
        # Load real data
        df = self.crypto_loader.load_training_data()
        if df.empty:
            return {'success': False, 'error': 'No data'}
        
        # Prepare features
        X, y, feature_names = self.crypto_loader.prepare_features(df)
        if len(X) == 0:
            return {'success': False, 'error': 'Feature extraction failed'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train direction model
        print("\n🎯 Training direction model...")
        direction_model = self._get_best_classifier()
        direction_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = direction_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📈 Results:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   F1 Score: {f1:.1%}")
        print("\n" + classification_report(y_test, y_pred, 
              target_names=['SHORT', 'WAIT', 'LONG']))
        
        # Feature importance
        if hasattr(direction_model, 'feature_importances_'):
            importances = direction_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n🔑 Top 10 Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save models
        model_path = os.path.join(MODEL_DIR, 'crypto_direction_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'crypto_scaler.pkl')
        meta_path = os.path.join(MODEL_DIR, 'crypto_metadata.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(direction_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'samples': len(X),
            'features': feature_names,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'label_map': {0: 'SHORT', 1: 'WAIT', 2: 'LONG'},
            'data_source': 'whale_history.db (REAL DATA)'
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Models saved to {MODEL_DIR}/")
        
        return {
            'success': True,
            'accuracy': accuracy,
            'f1_score': f1,
            'samples': len(X),
            'features': feature_names,
            'model_path': model_path
        }
    
    def train_stock_models(self) -> Dict:
        """Train models on REAL stock institutional data"""
        print("\n" + "=" * 60)
        print("📊 TRAINING STOCK MODELS ON REAL INSTITUTIONAL DATA")
        print("=" * 60)
        
        # Load real data
        df = self.stock_loader.load_training_data(min_samples=50)
        if df.empty:
            return {'success': False, 'error': 'No data'}
        
        # Prepare features
        X, y, feature_names = self.stock_loader.prepare_features(df)
        if len(X) == 0:
            return {'success': False, 'error': 'Feature extraction failed'}
        
        # Split data (smaller test size for limited data)
        test_size = 0.2 if len(X) > 500 else 0.15
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\n📊 Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train direction model
        print("\n🎯 Training direction model...")
        direction_model = self._get_best_classifier()
        direction_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = direction_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n📈 Results:")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   F1 Score: {f1:.1%}")
        
        # Feature importance
        if hasattr(direction_model, 'feature_importances_'):
            importances = direction_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n🔑 Top Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save models
        model_path = os.path.join(MODEL_DIR, 'stock_direction_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'stock_scaler.pkl')
        meta_path = os.path.join(MODEL_DIR, 'stock_metadata.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(direction_model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'samples': len(X),
            'features': feature_names,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'label_map': {0: 'SHORT', 1: 'WAIT', 2: 'LONG'},
            'data_source': 'stock_history/*.json (REAL DATA)'
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Models saved to {MODEL_DIR}/")
        
        return {
            'success': True,
            'accuracy': accuracy,
            'f1_score': f1,
            'samples': len(X),
            'features': feature_names,
            'model_path': model_path
        }
    
    def train_all(self) -> Dict:
        """Train both crypto and stock models"""
        print("\n" + "=" * 70)
        print("🚀 REAL DATA ML TRAINER - Using ACTUAL Whale/Institutional Data!")
        print("=" * 70)
        
        results = {}
        
        # Train crypto
        results['crypto'] = self.train_crypto_models()
        
        # Train stock
        results['stock'] = self.train_stock_models()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 TRAINING SUMMARY")
        print("=" * 70)
        
        if results['crypto'].get('success'):
            print(f"✅ Crypto: {results['crypto']['accuracy']:.1%} accuracy on {results['crypto']['samples']:,} samples")
        else:
            print(f"❌ Crypto: {results['crypto'].get('error', 'Failed')}")
        
        if results['stock'].get('success'):
            print(f"✅ Stock: {results['stock']['accuracy']:.1%} accuracy on {results['stock']['samples']:,} samples")
        else:
            print(f"❌ Stock: {results['stock'].get('error', 'Failed')}")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE (Uses trained models)
# ═══════════════════════════════════════════════════════════════════════════════

class RealDataPredictor:
    """Make predictions using models trained on REAL data"""
    
    def __init__(self):
        self.crypto_model = None
        self.crypto_scaler = None
        self.crypto_features = None
        
        self.stock_model = None
        self.stock_scaler = None
        self.stock_features = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained models"""
        # Crypto model
        crypto_model_path = os.path.join(MODEL_DIR, 'crypto_direction_model.pkl')
        crypto_scaler_path = os.path.join(MODEL_DIR, 'crypto_scaler.pkl')
        crypto_meta_path = os.path.join(MODEL_DIR, 'crypto_metadata.json')
        
        if os.path.exists(crypto_model_path):
            with open(crypto_model_path, 'rb') as f:
                self.crypto_model = pickle.load(f)
            with open(crypto_scaler_path, 'rb') as f:
                self.crypto_scaler = pickle.load(f)
            with open(crypto_meta_path, 'r') as f:
                meta = json.load(f)
                self.crypto_features = meta['features']
            print(f"✅ Crypto model loaded ({meta['accuracy']:.1%} accuracy)")
        
        # Stock model
        stock_model_path = os.path.join(MODEL_DIR, 'stock_direction_model.pkl')
        stock_scaler_path = os.path.join(MODEL_DIR, 'stock_scaler.pkl')
        stock_meta_path = os.path.join(MODEL_DIR, 'stock_metadata.json')
        
        if os.path.exists(stock_model_path):
            with open(stock_model_path, 'rb') as f:
                self.stock_model = pickle.load(f)
            with open(stock_scaler_path, 'rb') as f:
                self.stock_scaler = pickle.load(f)
            with open(stock_meta_path, 'r') as f:
                meta = json.load(f)
                self.stock_features = meta['features']
            print(f"✅ Stock model loaded ({meta['accuracy']:.1%} accuracy)")
    
    def predict_crypto(self, features: Dict) -> Dict:
        """
        Predict direction for crypto using REAL-data trained model.
        
        Args:
            features: Dict with keys matching CRYPTO_FEATURES
                - whale_long_pct: Real whale % (0-100)
                - retail_long_pct: Real retail % (0-100)
                - oi_change_24h: OI change
                - position_in_range: Position % (0-100)
                - rsi: RSI value
                - etc.
        
        Returns:
            Dict with direction, confidence, reasoning
        """
        if self.crypto_model is None:
            return {'direction': 'WAIT', 'confidence': 50, 'error': 'Model not loaded'}
        
        # Build feature vector
        feature_vector = self._build_crypto_features(features)
        
        # Scale
        X = np.array([feature_vector])
        X_scaled = self.crypto_scaler.transform(X)
        
        # Predict
        pred = self.crypto_model.predict(X_scaled)[0]
        proba = self.crypto_model.predict_proba(X_scaled)[0]
        
        direction_map = {0: 'SHORT', 1: 'WAIT', 2: 'LONG'}
        direction = direction_map[pred]
        confidence = float(max(proba) * 100)
        
        # Generate reasoning
        whale_pct = features.get('whale_long_pct', 50)
        retail_pct = features.get('retail_long_pct', 50)
        diff = whale_pct - retail_pct
        
        reasoning = f"Whale {whale_pct:.0f}% vs Retail {retail_pct:.0f}% (gap: {diff:+.0f}%)"
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': {
                'SHORT': float(proba[0] * 100),
                'WAIT': float(proba[1] * 100),
                'LONG': float(proba[2] * 100)
            },
            'reasoning': reasoning,
            'model_type': 'REAL_DATA'
        }
    
    def predict_stock(self, features: Dict) -> Dict:
        """Predict direction for stock using REAL-data trained model"""
        if self.stock_model is None:
            return {'direction': 'WAIT', 'confidence': 50, 'error': 'Model not loaded'}
        
        # Build feature vector
        feature_vector = self._build_stock_features(features)
        
        # Scale
        X = np.array([feature_vector])
        X_scaled = self.stock_scaler.transform(X)
        
        # Predict
        pred = self.stock_model.predict(X_scaled)[0]
        proba = self.stock_model.predict_proba(X_scaled)[0]
        
        direction_map = {0: 'SHORT', 1: 'WAIT', 2: 'LONG'}
        direction = direction_map[pred]
        confidence = float(max(proba) * 100)
        
        # Generate reasoning
        congress = features.get('congress_score', 50)
        insider = features.get('insider_score', 50)
        
        reasoning = f"Congress {congress:.0f} | Insider {insider:.0f}"
        
        return {
            'direction': direction,
            'confidence': confidence,
            'probabilities': {
                'SHORT': float(proba[0] * 100),
                'WAIT': float(proba[1] * 100),
                'LONG': float(proba[2] * 100)
            },
            'reasoning': reasoning,
            'model_type': 'REAL_DATA'
        }
    
    def _build_crypto_features(self, features: Dict) -> List[float]:
        """Build feature vector for crypto prediction"""
        # Extract base features
        whale_pct = features.get('whale_long_pct', 50)
        retail_pct = features.get('retail_long_pct', 50)
        position = features.get('position_in_range', 50)
        rsi = features.get('rsi', 50)
        
        return [
            whale_pct,
            retail_pct,
            whale_pct - retail_pct,  # whale_retail_diff
            features.get('oi_change_24h', 0),
            features.get('funding_rate', 0),
            position,
            1 if position < 35 else 0,  # position_early
            1 if position > 65 else 0,  # position_late
            features.get('mfi', 50),
            rsi,
            1 if rsi < 30 else 0,  # rsi_oversold
            1 if rsi > 70 else 0,  # rsi_overbought
            features.get('volume_ratio', 1.0),
            features.get('atr_pct', 2.0),
            features.get('price_change_24h', 0),
            1 if whale_pct > 65 else 0,  # whale_dominant
            1 if retail_pct > whale_pct + 10 else 0,  # retail_dominant
            1 if whale_pct > 60 and retail_pct < 50 else 0,  # squeeze_potential
        ]
    
    def _build_stock_features(self, features: Dict) -> List[float]:
        """Build feature vector for stock prediction"""
        congress = features.get('congress_score', 50)
        insider = features.get('insider_score', 50)
        position = features.get('position_pct', 50)
        rsi = features.get('rsi', 50)
        short_int = features.get('short_interest_pct', 0)
        
        return [
            congress,
            insider,
            short_int,
            features.get('combined_score', 50),
            congress - insider,  # institutional_diff
            position,
            1 if position < 35 else 0,  # position_early
            1 if position > 65 else 0,  # position_late
            features.get('ta_score', 50),
            rsi,
            1 if rsi < 30 else 0,  # rsi_oversold
            1 if rsi > 70 else 0,  # rsi_overbought
            features.get('price_change_24h', 0),
            1 if congress > 60 else 0,  # congress_bullish
            1 if insider > 60 else 0,  # insider_bullish
            1 if short_int > 15 else 0,  # short_squeeze_risk
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML models on REAL data')
    parser.add_argument('--crypto', action='store_true', help='Train crypto model only')
    parser.add_argument('--stock', action='store_true', help='Train stock model only')
    parser.add_argument('--all', action='store_true', help='Train all models (default)')
    
    args = parser.parse_args()
    
    trainer = RealDataTrainer()
    
    if args.crypto:
        trainer.train_crypto_models()
    elif args.stock:
        trainer.train_stock_models()
    else:
        trainer.train_all()
