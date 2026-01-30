"""
InvestorIQ Machine Learning Module
===================================

Provides ML-based predictions alongside rule-based MASTER_RULES.

Modules:
    - feature_extractor: Extract features from market data
    - model_trainer: Train ML models
    - model_predictor: Make predictions
    - tp_sl_optimizer: Optimize TP/SL levels
    - ml_engine: Main interface for predictions
"""

from .ml_engine import MLEngine, get_ml_prediction, is_ml_available, is_model_loaded
from .feature_extractor import extract_features, FEATURE_NAMES

__all__ = [
    'MLEngine',
    'get_ml_prediction', 
    'extract_features',
    'FEATURE_NAMES',
    'is_ml_available',
    'is_model_loaded'
]
