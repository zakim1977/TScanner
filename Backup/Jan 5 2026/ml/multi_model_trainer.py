"""
Multi-Model ML Trainer - Compare Multiple Algorithms
======================================================

Trains multiple ML models and selects the best performing one:
- LightGBM (Gradient Boosting)
- XGBoost (Extreme Gradient Boosting)
- Random Forest
- Ridge Classifier
- Logistic Regression
- Support Vector Machine
- Gradient Boosting Classifier

Each model is evaluated and the best one is automatically selected.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MIN_SAMPLES = 50  # Minimum for training
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Direction mapping
DIRECTION_MAP = {'SHORT': 0, 'WAIT': 1, 'LONG': 2}
DIRECTION_REVERSE = {0: 'SHORT', 1: 'WAIT', 2: 'LONG'}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """Results from training a single model"""
    name: str
    model: Any
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    feature_importances: Optional[np.ndarray] = None
    cross_val_scores: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""


def get_model_configs() -> Dict[str, Dict]:
    """
    Returns configuration for all available models.
    Each model is configured with optimized hyperparameters.
    """
    configs = {}
    
    # 1. LightGBM - Fast gradient boosting
    try:
        import lightgbm as lgb
        configs['LightGBM'] = {
            'class': lgb.LGBMClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': RANDOM_STATE,
                'verbose': -1,
                'n_jobs': -1
            },
            'needs_scaling': False,
            'has_feature_importance': True
        }
    except ImportError:
        print("⚠️ LightGBM not available")
    
    # 2. XGBoost - Extreme gradient boosting
    try:
        import xgboost as xgb
        configs['XGBoost'] = {
            'class': xgb.XGBClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': RANDOM_STATE,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss',
                'n_jobs': -1
            },
            'needs_scaling': False,
            'has_feature_importance': True
        }
    except ImportError:
        print("⚠️ XGBoost not available")
    
    # 3. Random Forest - Ensemble of decision trees
    from sklearn.ensemble import RandomForestClassifier
    configs['RandomForest'] = {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'needs_scaling': False,
        'has_feature_importance': True
    }
    
    # 4. Gradient Boosting - Sklearn's gradient boosting
    from sklearn.ensemble import GradientBoostingClassifier
    configs['GradientBoosting'] = {
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'random_state': RANDOM_STATE
        },
        'needs_scaling': False,
        'has_feature_importance': True
    }
    
    # 5. Ridge Classifier - Linear model with L2 regularization
    from sklearn.linear_model import RidgeClassifier
    configs['RidgeClassifier'] = {
        'class': RidgeClassifier,
        'params': {
            'alpha': 1.0,
            'random_state': RANDOM_STATE
        },
        'needs_scaling': True,
        'has_feature_importance': True  # Uses coef_
    }
    
    # 6. Logistic Regression - Probabilistic linear classifier
    from sklearn.linear_model import LogisticRegression
    configs['LogisticRegression'] = {
        'class': LogisticRegression,
        'params': {
            'C': 1.0,
            'max_iter': 1000,
            'multi_class': 'multinomial',
            'solver': 'lbfgs',
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'needs_scaling': True,
        'has_feature_importance': True  # Uses coef_
    }
    
    # 7. Support Vector Machine - Kernel-based classifier
    from sklearn.svm import SVC
    configs['SVM'] = {
        'class': SVC,
        'params': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': RANDOM_STATE
        },
        'needs_scaling': True,
        'has_feature_importance': False
    }
    
    # 8. Extra Trees - Extremely randomized trees
    from sklearn.ensemble import ExtraTreesClassifier
    configs['ExtraTrees'] = {
        'class': ExtraTreesClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE,
            'n_jobs': -1
        },
        'needs_scaling': False,
        'has_feature_importance': True
    }
    
    # 9. AdaBoost - Adaptive boosting
    from sklearn.ensemble import AdaBoostClassifier
    configs['AdaBoost'] = {
        'class': AdaBoostClassifier,
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE
        },
        'needs_scaling': False,
        'has_feature_importance': True
    }
    
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_model(
    model_name: str,
    config: Dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    feature_names: List[str]
) -> ModelResult:
    """
    Train a single model and evaluate its performance.
    """
    import time
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, classification_report
    )
    from sklearn.model_selection import cross_val_score
    
    print(f"\n{'─' * 50}")
    print(f"Training: {model_name}")
    print(f"{'─' * 50}")
    
    start_time = time.time()
    
    # Select appropriate data (scaled or unscaled)
    if config['needs_scaling']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Create and train model
    try:
        model = config['class'](**config['params'])
        model.fit(X_tr, y_train)
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return ModelResult(
            name=model_name,
            model=None,
            accuracy=0,
            precision=0,
            recall=0,
            f1_score=0,
            training_time=0
        )
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_te)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['SHORT', 'WAIT', 'LONG'], zero_division=0)
    
    # Cross-validation (if fast enough)
    cv_scores = None
    if training_time < 30:  # Only if training was fast
        try:
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')
        except:
            pass
    
    # Feature importances
    feat_imp = None
    if config['has_feature_importance']:
        if hasattr(model, 'feature_importances_'):
            feat_imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute mean of coefficients
            feat_imp = np.abs(model.coef_).mean(axis=0)
    
    # Print results
    print(f"  ✅ Accuracy: {acc:.1%}")
    print(f"  📊 Precision: {prec:.1%}")
    print(f"  📊 Recall: {rec:.1%}")
    print(f"  📊 F1 Score: {f1:.1%}")
    print(f"  ⏱️ Training time: {training_time:.2f}s")
    if cv_scores is not None:
        print(f"  🔄 Cross-val: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
    
    return ModelResult(
        name=model_name,
        model=model,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_score=f1,
        training_time=training_time,
        feature_importances=feat_imp,
        cross_val_scores=cv_scores,
        confusion_matrix=conf_matrix,
        classification_report=class_report
    )


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> Tuple[List[ModelResult], ModelResult]:
    """
    Train all available models and return results.
    
    Returns:
        Tuple of (all_results, best_result)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "═" * 60)
    print("MULTI-MODEL TRAINING COMPETITION")
    print("═" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\n📊 Dataset: {len(X)} samples")
    print(f"   Training: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Features: {len(feature_names)}")
    
    # Scale data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get model configurations
    configs = get_model_configs()
    print(f"\n🤖 Models to train: {len(configs)}")
    
    # Train each model
    results = []
    for name, config in configs.items():
        result = train_single_model(
            name, config,
            X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            feature_names
        )
        if result.model is not None:
            results.append(result)
    
    # Find best model
    if not results:
        print("\n❌ No models trained successfully!")
        return [], None
    
    # Sort by F1 score (best overall metric for imbalanced classes)
    results.sort(key=lambda x: x.f1_score, reverse=True)
    best = results[0]
    
    # Print comparison
    print("\n" + "═" * 60)
    print("MODEL COMPARISON (sorted by F1 Score)")
    print("═" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("─" * 60)
    for r in results:
        marker = "🏆" if r.name == best.name else "  "
        print(f"{marker}{r.name:<18} {r.accuracy:>10.1%} {r.precision:>10.1%} {r.recall:>10.1%} {r.f1_score:>10.1%}")
    
    print("\n" + "═" * 60)
    print(f"🏆 BEST MODEL: {best.name}")
    print(f"   F1 Score: {best.f1_score:.1%}")
    print(f"   Accuracy: {best.accuracy:.1%}")
    print("═" * 60)
    
    # Print feature importances for best model
    if best.feature_importances is not None:
        print(f"\n📊 Top 10 Important Features ({best.name}):")
        importances = list(zip(feature_names, best.feature_importances))
        importances.sort(key=lambda x: x[1], reverse=True)
        for i, (name, imp) in enumerate(importances[:10], 1):
            print(f"   {i:2}. {name:<30} {imp:.4f}")
    
    # Print confusion matrix
    print(f"\n📋 Confusion Matrix ({best.name}):")
    print("              Predicted")
    print("            SHORT  WAIT  LONG")
    for i, label in enumerate(['SHORT', 'WAIT', 'LONG']):
        row = best.confusion_matrix[i] if best.confusion_matrix is not None else [0, 0, 0]
        print(f"Actual {label:<5} {row[0]:5} {row[1]:5} {row[2]:5}")
    
    return results, best, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE/LOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save_best_model(
    result: ModelResult,
    scaler,
    feature_names: List[str],
    model_dir: str = MODEL_DIR
) -> str:
    """
    Save the best model and associated metadata.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, 'direction_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(result.model, f)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metadata
    metadata = {
        'model_name': result.name,
        'accuracy': result.accuracy,
        'precision': result.precision,
        'recall': result.recall,
        'f1_score': result.f1_score,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'trained_at': datetime.now().isoformat(),
        'cross_val_mean': float(result.cross_val_scores.mean()) if result.cross_val_scores is not None else None,
        'cross_val_std': float(result.cross_val_scores.std()) if result.cross_val_scores is not None else None,
    }
    
    meta_path = os.path.join(model_dir, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature importances
    if result.feature_importances is not None:
        imp_data = {name: float(imp) for name, imp in zip(feature_names, result.feature_importances)}
        imp_path = os.path.join(model_dir, 'feature_importances.json')
        with open(imp_path, 'w') as f:
            json.dump(imp_data, f, indent=2, sort_keys=True)
    
    print(f"\n✅ Model saved to: {model_dir}")
    print(f"   - direction_model.pkl ({result.name})")
    print(f"   - scaler.pkl")
    print(f"   - model_metadata.json")
    
    return model_path


def save_all_models(
    results: List[ModelResult],
    scaler,
    feature_names: List[str],
    model_dir: str = MODEL_DIR
) -> None:
    """
    Save all trained models for potential ensemble use.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    all_models_dir = os.path.join(model_dir, 'all_models')
    os.makedirs(all_models_dir, exist_ok=True)
    
    for result in results:
        if result.model is not None:
            model_path = os.path.join(all_models_dir, f'{result.name.lower()}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(result.model, f)
    
    print(f"\n📦 All {len(results)} models saved to: {all_models_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """
    Combines multiple models for more robust predictions.
    
    Supports:
    - Voting (majority vote)
    - Averaging (average probabilities)
    - Weighted (weight by F1 score)
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_names = []
        self._load_models(model_dir)
    
    def _load_models(self, model_dir: str):
        """Load all available models."""
        all_models_dir = os.path.join(model_dir, 'all_models')
        
        if not os.path.exists(all_models_dir):
            print("⚠️ No ensemble models found")
            return
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load metadata for weights
        meta_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.feature_names = meta.get('feature_names', [])
        
        # Load all models
        for filename in os.listdir(all_models_dir):
            if filename.endswith('_model.pkl'):
                model_name = filename.replace('_model.pkl', '')
                model_path = os.path.join(all_models_dir, filename)
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    self.weights[model_name] = 1.0  # Default weight
                except:
                    pass
        
        print(f"📦 Loaded {len(self.models)} models for ensemble")
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> Tuple[str, float]:
        """
        Make ensemble prediction.
        
        Args:
            X: Feature vector (single sample)
            method: 'voting', 'averaging', or 'weighted'
        
        Returns:
            Tuple of (direction, confidence)
        """
        if not self.models:
            return 'WAIT', 50.0
        
        X = np.array(X).reshape(1, -1)
        
        # Scale if needed
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            try:
                # Some models need scaled, some don't
                needs_scaling = name.lower() in ['ridgeclassifier', 'logisticregression', 'svm']
                X_use = X_scaled if needs_scaling else X
                
                pred = model.predict(X_use)[0]
                predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_use)[0]
                    probabilities.append(prob)
            except:
                pass
        
        if not predictions:
            return 'WAIT', 50.0
        
        if method == 'voting':
            # Majority vote
            from collections import Counter
            vote_counts = Counter(predictions)
            winner = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[winner] / len(predictions) * 100
            direction = DIRECTION_REVERSE.get(winner, 'WAIT')
            
        elif method == 'averaging' and probabilities:
            # Average probabilities
            avg_probs = np.mean(probabilities, axis=0)
            winner = np.argmax(avg_probs)
            confidence = avg_probs[winner] * 100
            direction = DIRECTION_REVERSE.get(winner, 'WAIT')
            
        else:
            # Fallback to voting
            from collections import Counter
            vote_counts = Counter(predictions)
            winner = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[winner] / len(predictions) * 100
            direction = DIRECTION_REVERSE.get(winner, 'WAIT')
        
        return direction, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_multi_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save: bool = True,
    save_all: bool = True
) -> Dict:
    """
    Main function to train multiple models and select the best.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        feature_names: List of feature names
        save: Save best model
        save_all: Save all models for ensemble
    
    Returns:
        Dict with training results
    """
    
    if len(X) < MIN_SAMPLES:
        print(f"⚠️ Not enough samples: {len(X)} < {MIN_SAMPLES}")
        return {
            'success': False,
            'error': f'Insufficient data: {len(X)} samples',
            'min_required': MIN_SAMPLES
        }
    
    # Train all models
    results, best, scaler = train_all_models(X, y, feature_names)
    
    if best is None:
        return {
            'success': False,
            'error': 'No models trained successfully'
        }
    
    # Save models
    if save:
        save_best_model(best, scaler, feature_names)
    
    if save_all:
        save_all_models(results, scaler, feature_names)
    
    return {
        'success': True,
        'best_model': best.name,
        'best_accuracy': best.accuracy,
        'best_f1': best.f1_score,
        'all_results': [
            {
                'name': r.name,
                'accuracy': r.accuracy,
                'f1_score': r.f1_score
            }
            for r in results
        ],
        'n_models_trained': len(results),
        'feature_names': feature_names
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multiple ML models')
    parser.add_argument('--data', type=str, help='Path to training data (CSV)')
    parser.add_argument('--no-save', action='store_true', help='Do not save models')
    parser.add_argument('--demo', action='store_true', help='Run with demo data')
    args = parser.parse_args()
    
    if args.demo:
        # Generate demo data for testing
        print("\n🎮 Running with DEMO data...")
        np.random.seed(42)
        n_samples = 200
        n_features = 36
        
        # Create synthetic features
        X = np.random.randn(n_samples, n_features)
        
        # Create labels with some pattern
        scores = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + np.random.randn(n_samples) * 0.5
        y = np.where(scores > 0.5, 2, np.where(scores < -0.5, 0, 1))  # LONG, WAIT, SHORT
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        result = train_multi_model(X, y, feature_names, save=not args.no_save)
        print(f"\n✅ Demo complete: {result}")
    
    elif args.data:
        # Load from CSV
        print(f"\n📂 Loading data from: {args.data}")
        df = pd.read_csv(args.data)
        
        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = list(df.columns[:-1])
        
        result = train_multi_model(X, y, feature_names, save=not args.no_save)
        print(f"\n✅ Training complete: {result}")
    
    else:
        print("Usage: python multi_model_trainer.py --demo")
        print("       python multi_model_trainer.py --data training_data.csv")
