# InvestorIQ ML Integration Guide

## Overview

The ML module provides machine learning predictions alongside the existing Rule-Based (MASTER_RULES) system. Users can switch between three analysis modes:

1. **📊 Rule-Based** - Current MASTER_RULES scoring system (unchanged)
2. **🤖 ML Model** - Machine learning predictions with optimal TP/SL
3. **⚡ Hybrid** - Combines both for highest accuracy when they agree

## File Structure

```
TScanner/
├── ml/                          # ML Module
│   ├── __init__.py              # Package init
│   ├── feature_extractor.py     # Extract 30 features for ML
│   ├── ml_engine.py             # Main prediction engine
│   ├── hybrid_engine.py         # Combines Rule + ML
│   ├── model_trainer.py         # Train models from data
│   ├── ui_components.py         # UI elements for ML
│   └── models/                  # Trained model files
│       ├── direction_model.pkl  # Predicts LONG/SHORT/WAIT
│       ├── tp_sl_model.pkl      # Optimizes TP/SL levels
│       ├── feature_scaler.pkl   # Feature normalization
│       └── model_metadata.json  # Training info
│
├── core/
│   └── analysis_router.py       # Routes to selected engine
│
└── app.py                       # Toggle in sidebar
```

## How It Works

### Mode Toggle (Sidebar)

Both Scanner and Single Analysis have an "Analysis Engine" dropdown:

```
Analysis Engine:  [📊 Rule-Based ▾]
                  [📊 Rule-Based   ]
                  [🤖 ML Model     ]
                  [⚡ Hybrid       ]
```

### Features Used (30 total)

```python
# Whale/Institutional (5)
'whale_pct', 'retail_pct', 'whale_retail_divergence', 
'whale_dominance', 'funding_rate'

# Open Interest (4)
'oi_change_24h', 'oi_signal_encoded', 
'price_change_24h', 'price_change_1h'

# Position/Range (3)
'position_in_range', 'position_label_encoded', 'range_size_pct'

# Technical (5)
'ta_score', 'rsi', 'rsi_zone_encoded', 'trend_encoded', 'volatility_pct'

# Money Flow (2)
'money_flow_encoded', 'volume_ratio'

# SMC Structure (4)
'at_bullish_ob', 'at_bearish_ob', 'near_support', 'near_resistance'

# Market Context (4)
'btc_correlation', 'btc_trend_encoded', 'market_fear_greed', 'is_weekend'

# Historical (3)
'historical_win_rate', 'similar_setup_count', 'avg_historical_return'
```

## Training the Model

### Option 1: Command Line

```bash
cd TScanner
python3 -m ml.model_trainer \
    --whale-db data/whale_history.db \
    --trade-history trade_history.json
```

### Option 2: Python Script

```python
from ml.model_trainer import train_models

result = train_models(
    whale_db_path='data/whale_history.db',
    trade_history_path='trade_history.json',
    save_models=True
)

print(f"Accuracy: {result['direction_accuracy']:.1%}")
```

### Minimum Requirements

- **100+ records** with known outcomes (WIN/LOSS)
- Data from `whale_history.db` with `hit_tp1` / `hit_sl` filled
- OR completed trades in `trade_history.json`

## Using the ML Predictions

### In Code

```python
from ml.ml_engine import get_ml_prediction

prediction = get_ml_prediction(
    whale_pct=84,
    retail_pct=57,
    oi_change=-0.7,
    position_pct=28,
    current_price=97450
)

print(f"Direction: {prediction.direction}")      # LONG
print(f"Confidence: {prediction.confidence}%")    # 84%
print(f"TP1: +{prediction.optimal_tp1_pct}%")    # +2.5%
print(f"SL: -{prediction.optimal_sl_pct}%")      # -1.2%
```

### Hybrid Mode

```python
from ml.hybrid_engine import get_hybrid_prediction
from core.MASTER_RULES import get_trade_decision

# Get rule-based decision
rule_decision = get_trade_decision(...)

# Get hybrid prediction
hybrid = get_hybrid_prediction(
    rule_decision=rule_decision,
    whale_pct=84,
    # ... other params
)

if hybrid.engines_agree:
    print("✅ Both engines agree - HIGH confidence")
else:
    print("⚠️ Engines disagree - review manually")
```

## ML Output Format

```python
@dataclass
class MLPrediction:
    direction: str              # 'LONG', 'SHORT', 'WAIT'
    confidence: float           # 0-100%
    predicted_move: float       # Expected % move
    
    optimal_tp1_pct: float      # ML-optimized TP1 distance
    optimal_tp2_pct: float      # ML-optimized TP2 distance
    optimal_sl_pct: float       # ML-optimized SL distance
    
    expected_rr: float          # Risk/Reward ratio
    win_probability: float      # Probability of hitting TP1
    
    top_features: List          # Top 5 contributing factors
    reasoning: str              # Human-readable explanation
    
    similar_trades_count: int   # Historical matches
    similar_trades_win_rate: float
```

## Fallback Behavior

- If ML models not trained → Uses **heuristic predictions** (similar to rules)
- Model version shows as `heuristic_v1` when not trained
- UI shows warning: "⚠️ ML not trained - using Rules"

## Next Steps (Phase 4-5)

Future phases (not implemented yet):
- Auto-scan every 15 minutes
- Telegram/Discord alerts
- Binance API integration
- Auto-execute trades with TP/SL
- Continuous model retraining

## Notes

- ML models are saved in `ml/models/` directory
- Training requires `scikit-learn` and optionally `lightgbm`
- Position size fixed at $2,500 per trade (no leverage)
- All data used is from your whale_history.db and trade outcomes
