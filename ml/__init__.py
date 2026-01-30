"""
InvestorIQ Probabilistic ML Module
===================================

PROBABILISTIC ML SYSTEM
========================
This ML system predicts PROBABILITIES of different outcomes, then RULES interpret
these probabilities to decide LONG/SHORT/WAIT.

Why Probabilistic?
- Transparency: See WHY (P_continuation=64%, P_fakeout=19%)
- Calibration: Verify 60% predictions actually hit 60%
- Mode-specific: Different outcomes per trading style

MODE-SPECIFIC PREDICTIONS:
--------------------------
Scalp/DayTrade:
    P(continuation)   - Will price continue in current direction?
    P(fakeout)        - Is this a trap/reversal setup?
    P(vol_expansion)  - Will volatility explode?

Swing:
    P(trend_holds)    - Will existing trend continue?
    P(reversal)       - Is a reversal likely?
    P(drawdown)       - Is significant adverse move expected?

Investment:
    P(accumulation)   - Is smart money accumulating?
    P(distribution)   - Is smart money distributing?
    P(large_drawdown) - Is major correction possible?

USAGE:
------
    from ml import get_unified_ml_prediction
    
    pred = get_unified_ml_prediction(df, 'daytrade', whale_data)
    
    # Access probabilities
    print(pred.probabilities)  # {'continuation': 0.64, 'fakeout': 0.19, ...}
    
    # Access derived direction (rules interpreted)
    print(pred.direction)      # 'LONG'
    print(pred.confidence)     # 72.0

TRAINING:
---------
    from ml import train_probabilistic_model
    
    metrics = train_probabilistic_model(df, mode='daytrade')
    print(metrics['metrics']['continuation']['accuracy'])  # 0.68

KEY PRINCIPLE:
--------------
    ML informs. Rules decide.
"""

# Core probabilistic ML system
from .probabilistic_ml import (
    ProbabilisticMLTrainer,
    ProbabilisticPrediction,
    get_probabilistic_prediction,
    ENHANCED_FEATURES,
    NUM_ENHANCED_FEATURES,
    MODE_LABELS,
    extract_enhanced_features,
    generate_probabilistic_labels,
    verify_probability_calibration,
    print_calibration_report,
)

# Integration layer (unified interface)
from .probabilistic_integration import (
    UnifiedMLEngine,
    get_unified_ml_prediction,
    get_probability_breakdown,
    train_probabilistic_model,
)

# Legacy support (backward compatibility)
from .ml_engine import (
    MLEngine,
    MLPrediction,
    get_ml_prediction,
    is_ml_available,
    is_model_loaded,
    is_model_trained,
)

# Feature extraction (used by both systems)
from .feature_extractor import (
    extract_features,
    extract_features_from_dict,
    FEATURE_NAMES,
    FeatureSet,
)


__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # RECOMMENDED: Probabilistic ML (new system)
    # ═══════════════════════════════════════════════════════════════════════════
    'ProbabilisticMLTrainer',
    'ProbabilisticPrediction',
    'get_probabilistic_prediction',
    'get_unified_ml_prediction',      # RECOMMENDED: Auto-selects best model
    'get_probability_breakdown',       # For UI display
    'train_probabilistic_model',       # For training
    'UnifiedMLEngine',
    'get_all_model_metrics',           # NEW: Load all model metrics
    'get_model_status',                # NEW: Formatted status string
    
    # Feature definitions
    'ENHANCED_FEATURES',
    'NUM_ENHANCED_FEATURES',
    'MODE_LABELS',
    'MODE_LABELS_CRYPTO',              # NEW: Crypto-specific thresholds
    'MODE_LABELS_STOCK',               # NEW: Stock-specific thresholds
    'get_mode_labels',                 # NEW: Get labels by market type
    'extract_enhanced_features',
    'generate_probabilistic_labels',
    
    # Verification
    'verify_probability_calibration',
    'print_calibration_report',
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGACY: Old ML system (backward compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    'MLEngine',
    'MLPrediction',
    'get_ml_prediction',
    'is_ml_available',
    'is_model_loaded',
    'is_model_trained',
    
    # Legacy feature extraction
    'extract_features',
    'extract_features_from_dict',
    'FEATURE_NAMES',
    'FeatureSet',
]


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
"""
TRAINING MODES:
---------------
┌─────────────┬────────────┬─────────────────────────────────────────────────┐
│ Mode        │ Timeframes │ Labels                                          │
├─────────────┼────────────┼─────────────────────────────────────────────────┤
│ scalp       │ 1m, 5m     │ continuation, fakeout, vol_expansion            │
│ daytrade    │ 15m, 1h    │ continuation, fakeout, vol_expansion            │
│ swing       │ 4h, 1d     │ trend_holds, reversal, drawdown                 │
│ investment  │ 1d, 1w     │ accumulation, distribution, large_drawdown      │
└─────────────┴────────────┴─────────────────────────────────────────────────┘

INPUT FEATURES (45):
--------------------
- Price & Structure (15): returns, range, BoS, compression, ATR
- Positioning & Flow (12): OI, funding, whale/retail, divergence
- SMC / Context (12): liquidity, OBs, FVGs, accumulation, sessions
- Market Context (6): BTC correlation, fear/greed, HTF alignment

MODEL FILES:
------------
After training, models saved to: ml/models/probabilistic/
- prob_model_scalp_crypto.pkl     / prob_model_scalp_stock.pkl
- prob_model_daytrade_crypto.pkl  / prob_model_daytrade_stock.pkl
- prob_model_swing_crypto.pkl     / prob_model_swing_stock.pkl
- prob_model_investment_crypto.pkl / prob_model_investment_stock.pkl

Metrics JSON (for quick display):
- metrics_swing_crypto.json  → Shows F1 scores without loading model
"""