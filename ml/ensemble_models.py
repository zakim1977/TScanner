"""
Multi-Model Ensemble System
===========================
Trains multiple ML models and selects the best for each label/mode.

Models included:
1. GradientBoosting (sklearn)
2. HistGradientBoosting (sklearn fast)
3. RandomForest
4. ExtraTrees
5. AdaBoost
6. XGBoost (if available)
7. LightGBM (if available)
8. CatBoost (if available)
9. MLPClassifier (Neural Network)
10. LogisticRegression (baseline)
11. SVC (Support Vector)
12. KNeighbors

For each label, we train ALL models and pick the best based on F1 score.
Optionally creates a voting ensemble of top 3 models.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Try importing optional libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def get_all_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Returns dictionary of all available models with tuned hyperparameters.
    
    Args:
        scale_pos_weight: Weight for positive class (for imbalanced data)
    """
    models = {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRADIENT BOOSTING FAMILY
    # ═══════════════════════════════════════════════════════════════════════════
    
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    
    models['GradientBoosting_Deep'] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        min_samples_leaf=15,
        subsample=0.85,
        random_state=42
    )
    
    models['HistGradientBoosting'] = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=20,
        random_state=42
    )
    
    models['HistGradientBoosting_Fast'] = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=30,
        random_state=42
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RANDOM FOREST FAMILY
    # ═══════════════════════════════════════════════════════════════════════════
    
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    models['RandomForest_Deep'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    models['RandomForest_Shallow'] = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=20,
        min_samples_split=30,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    models['ExtraTrees_Deep'] = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BOOSTING VARIANTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    models['AdaBoost'] = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    
    models['AdaBoost_Strong'] = AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # XGBOOST (if available)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        models['XGBoost_Deep'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        models['XGBoost_Fast'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIGHTGBM (if available)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        
        models['LightGBM_Deep'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
        
        models['LightGBM_Fast'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.15,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CATBOOST (if available)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=False
        )
        
        models['CatBoost_Deep'] = CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=False
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEURAL NETWORKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    models['MLP_Small'] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    models['MLP_Medium'] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    models['MLP_Large'] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    models['MLP_Wide'] = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LINEAR MODELS
    # ═══════════════════════════════════════════════════════════════════════════
    
    models['LogisticRegression'] = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    models['LogisticRegression_L1'] = LogisticRegression(
        C=0.5,
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    models['Ridge'] = RidgeClassifier(
        alpha=1.0,
        class_weight='balanced',
        random_state=42
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SVM - DISABLED (too slow with large datasets, O(n²) complexity)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # NOTE: SVC removed because it takes 30+ minutes with 9000+ records
    # Boosting models (XGBoost, LightGBM) perform better anyway
    
    # models['SVC_RBF'] = SVC(
    #     kernel='rbf',
    #     C=1.0,
    #     gamma='scale',
    #     class_weight='balanced',
    #     probability=True,
    #     random_state=42
    # )
    # 
    # models['SVC_Linear'] = SVC(
    #     kernel='linear',
    #     C=1.0,
    #     class_weight='balanced',
    #     probability=True,
    #     random_state=42
    # )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OTHER
    # ═══════════════════════════════════════════════════════════════════════════
    
    models['KNeighbors'] = KNeighborsClassifier(
        n_neighbors=15,
        weights='distance',
        n_jobs=-1
    )
    
    models['KNeighbors_5'] = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        n_jobs=-1
    )
    
    models['GaussianNB'] = GaussianNB()
    
    models['LDA'] = LinearDiscriminantAnalysis()
    
    models['DecisionTree'] = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    
    models['Bagging_DT'] = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    
    return models


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_name: str,
    progress_callback=None
) -> dict:
    """
    Train all available models on the data and return performance metrics.
    
    Returns:
        dict with model results sorted by F1 score
    """
    # Calculate class weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = min(neg_count / pos_count, 10.0) if pos_count > 0 else 1.0
    
    # Get all models
    all_models = get_all_models(scale_pos_weight)
    
    results = {}
    total_models = len(all_models)
    
    for i, (name, model) in enumerate(all_models.items()):
        try:
            # Show progress: which model out of total
            if progress_callback:
                progress_callback(
                    i / total_models,
                    f"Model {i+1}/{total_models}: {name}..."
                )
            
            # Create sample weights
            sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
            
            # Check if model supports sample_weight
            if hasattr(model, 'fit'):
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                except TypeError:
                    # Model doesn't support sample_weight
                    model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
            else:
                y_proba = None
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'has_proba': y_proba is not None
            }
            
            # Show completion with F1 score
            if progress_callback:
                progress_callback(
                    (i + 1) / total_models,
                    f"Model {i+1}/{total_models}: {name} ✓ F1={f1:.1%}"
                )
                
        except Exception as e:
            # Skip models that fail
            print(f"  {name} failed: {e}")
            continue
    
    # Sort by F1 score
    results = dict(sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True))
    
    return results


def create_voting_ensemble(top_models: list, voting: str = 'soft') -> VotingClassifier:
    """
    Create a voting ensemble from top performing models.
    
    Args:
        top_models: List of (name, model) tuples
        voting: 'soft' for probability voting, 'hard' for majority voting
    """
    # Filter to models with predict_proba for soft voting
    if voting == 'soft':
        estimators = [(name, model) for name, model in top_models if hasattr(model, 'predict_proba')]
    else:
        estimators = top_models
    
    if len(estimators) < 2:
        return None
    
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )


def create_stacking_ensemble(
    base_models: list,
    final_estimator=None
) -> StackingClassifier:
    """
    Create a stacking ensemble with a meta-learner.
    
    Args:
        base_models: List of (name, model) tuples for base layer
        final_estimator: Meta-learner model (default: LogisticRegression)
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    
    return StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )


def get_best_models_per_label(all_results: dict, top_n: int = 3) -> dict:
    """
    Get the best N models for each label.
    
    Args:
        all_results: Dict of {label: {model_name: metrics}}
        top_n: Number of top models to return per label
    
    Returns:
        Dict of {label: [(model_name, model, metrics), ...]}
    """
    best_models = {}
    
    for label, results in all_results.items():
        # Sort by F1
        sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        # Get top N
        best_models[label] = [
            (name, data['model'], data)
            for name, data in sorted_results[:top_n]
        ]
    
    return best_models


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE: Available Models
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_DESCRIPTIONS = {
    'GradientBoosting': 'Classic gradient boosting - good baseline',
    'GradientBoosting_Deep': 'Deeper GB - may capture complex patterns',
    'HistGradientBoosting': 'Fast histogram-based GB - handles large data',
    'RandomForest': 'Ensemble of decision trees - robust',
    'RandomForest_Deep': 'Deeper RF - may overfit',
    'ExtraTrees': 'Extra randomized trees - faster, more random',
    'AdaBoost': 'Adaptive boosting - focuses on hard examples',
    'XGBoost': 'Extreme gradient boosting - often best performance',
    'XGBoost_Deep': 'Deeper XGB - complex patterns',
    'LightGBM': 'Fast gradient boosting - handles categorical',
    'LightGBM_Deep': 'Deeper LGB - complex patterns',
    'CatBoost': 'Categorical boosting - handles categorical well',
    'MLP_Small': 'Small neural network - fast',
    'MLP_Medium': 'Medium neural network - balanced',
    'MLP_Large': 'Large neural network - may capture complex patterns',
    'LogisticRegression': 'Linear baseline - interpretable',
    'SVC_RBF': 'SVM with RBF kernel - good for non-linear',
    'KNeighbors': 'K-nearest neighbors - simple but effective',
    'GaussianNB': 'Naive Bayes - fast baseline',
    'LDA': 'Linear Discriminant Analysis - dimensionality reduction',
}


def print_available_models():
    """Print all available models and their status."""
    print("=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)
    
    models = get_all_models()
    
    for name in sorted(models.keys()):
        desc = MODEL_DESCRIPTIONS.get(name.split('_')[0], 'Custom variant')
        print(f"  {name}: {desc}")
    
    print(f"\nTotal: {len(models)} models")
    print(f"XGBoost: {'✅ Available' if HAS_XGBOOST else '❌ Not installed'}")
    print(f"LightGBM: {'✅ Available' if HAS_LIGHTGBM else '❌ Not installed'}")
    print(f"CatBoost: {'✅ Available' if HAS_CATBOOST else '❌ Not installed'}")


if __name__ == '__main__':
    print_available_models()
