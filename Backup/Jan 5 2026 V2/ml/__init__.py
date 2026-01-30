"""
InvestorIQ Machine Learning Module
===================================

Provides ML-based predictions alongside rule-based MASTER_RULES.

IMPORTANT - HONEST LABELING:
- Use is_real_trained_ml() to check if ACTUAL ML is running
- Use get_prediction_source() to get honest label for display
- Heuristic = rules-based scoring (not real ML!)
- Trained ML = model trained on historical data

Modules:
    - feature_extractor: Extract features from market data
    - multi_model_trainer: Train multiple models, select best
    - mode_specific_trainer: Train mode-specific models
    - stock_trainer: Train stock/ETF models
    - ml_engine: Main interface for predictions
    - training_ui: Streamlit UI for model training

Available Models (via multi_model_trainer):
    - LightGBM (Gradient Boosting)
    - XGBoost (Extreme Gradient Boosting)
    - RandomForest (Ensemble Trees)
    - GradientBoosting (Sklearn)
    - ExtraTrees (Randomized Trees)
    - AdaBoost (Adaptive Boosting)
    - RidgeClassifier (Linear + L2)
    - LogisticRegression (Probabilistic)
    - SVM (Support Vector Machine)
"""

from .ml_engine import (
    MLEngine, 
    get_ml_prediction, 
    is_ml_available, 
    is_model_loaded,
    is_model_trained,
    is_real_trained_ml,  # NEW: Check if ACTUAL ML
    get_prediction_source,  # NEW: 'Trained ML' or 'Heuristic'
    get_model_label,  # NEW: Full descriptive label
    get_model_info,  # NEW: Complete model info dict
    reload_engine,  # Force reload after training
)
from .feature_extractor import extract_features, FEATURE_NAMES

__all__ = [
    'MLEngine',
    'get_ml_prediction', 
    'extract_features',
    'FEATURE_NAMES',
    'is_ml_available',
    'is_model_loaded',
    'is_model_trained',
    # NEW - Honest labeling functions
    'is_real_trained_ml',
    'get_prediction_source',
    'get_model_label',
    'get_model_info',
    'reload_engine',
]