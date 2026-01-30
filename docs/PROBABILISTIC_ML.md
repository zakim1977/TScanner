# Probabilistic ML System - Technical Documentation

## Overview

The new probabilistic ML system predicts **probabilities of different outcomes** rather than directly predicting LONG/SHORT/WAIT. This gives you:

1. **Transparency** - See WHY the model suggests a direction
2. **Calibration** - Verify 60% predictions actually hit 60% of the time
3. **Mode-specific outcomes** - Different predictions for day trade vs swing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES (45)                          │
├─────────────────────────────────────────────────────────────────┤
│ Price & Structure (15):                                         │
│   - Returns (1,3,5,10 bars)                                     │
│   - Range metrics (high-low, close position)                    │
│   - Break of structure (direction)                              │
│   - Compression (BB squeeze, candles tight)                     │
│   - ATR & volatility change                                     │
├─────────────────────────────────────────────────────────────────┤
│ Positioning & Flow (12):                                        │
│   - OI change (1h, 24h)                                         │
│   - Price vs OI divergence (key signal!)                        │
│   - Funding rate (level + delta)                                │
│   - Long/short ratio                                            │
│   - Whale/Retail net positions + divergence                     │
│   - Institutional buy/sell imbalance                            │
│   - Smart money flow (CMF)                                      │
├─────────────────────────────────────────────────────────────────┤
│ SMC / Context (12):                                             │
│   - Liquidity sweep flags (bull/bear)                           │
│   - Order block proximity (at/near)                             │
│   - FVG proximity (bull/bear)                                   │
│   - Accumulation/Distribution scores                            │
│   - Session timing (Asian/London/NY/overlap)                    │
├─────────────────────────────────────────────────────────────────┤
│ Market Context (6):                                             │
│   - BTC correlation + trend                                     │
│   - HTF trend alignment                                         │
│   - Fear/Greed index                                            │
│   - Weekend flag                                                │
│   - Days since major news                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PROBABILISTIC PREDICTION (per mode)                │
├─────────────────────────────────────────────────────────────────┤
│ DAY TRADE / SCALP:                                              │
│   P(continuation) = 0.64  ← Will price continue?                │
│   P(fakeout) = 0.19       ← Is this a trap?                     │
│   P(vol_expansion) = 0.72 ← Will volatility explode?            │
├─────────────────────────────────────────────────────────────────┤
│ SWING:                                                          │
│   P(trend_holds) = 0.71   ← Will trend continue?                │
│   P(reversal) = 0.15      ← Is reversal likely?                 │
│   P(drawdown) = 0.32      ← Significant adverse move?           │
├─────────────────────────────────────────────────────────────────┤
│ INVESTMENT:                                                     │
│   P(accumulation) = 0.58  ← Smart money accumulating?           │
│   P(distribution) = 0.22  ← Smart money distributing?           │
│   P(large_drawdown) = 0.18← Major correction possible?          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RULES INTERPRETATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ IF P(continuation) > 60% AND P(fakeout) < 30%:                  │
│    IF whale_divergence > 5%:  → LONG                            │
│    IF whale_divergence < -5%: → SHORT                           │
│    ELSE: → WAIT (no whale edge)                                 │
│                                                                 │
│ IF P(fakeout) > 50%:                                            │
│    → WAIT (high trap risk!)                                     │
│                                                                 │
│ IF P(vol_expansion) > 70% AND P(continuation) > 50%:            │
│    → Direction based on whale divergence                        │
│                                                                 │
│ ELSE:                                                           │
│    → WAIT (no clear signal)                                     │
│                                                                 │
│ ML informs. Rules decide.                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUT                               │
├─────────────────────────────────────────────────────────────────┤
│ Direction: LONG                                                 │
│ Confidence: 72%                                                 │
│ Reasoning: P_cont=64% | P_fake=19% | Whales +15% → LONG         │
│                                                                 │
│ Top Features:                                                   │
│   1. whale_retail_divergence: +15.3 (importance: 0.12)          │
│   2. bb_squeeze_pct: 78.5 (importance: 0.09)                    │
│   3. price_vs_oi_divergence: +2.1 (importance: 0.08)            │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences from Old System

| Aspect | Old System | New Probabilistic System |
|--------|------------|--------------------------|
| Output | LONG/SHORT/WAIT directly | P(continuation), P(fakeout), etc. |
| Transparency | "ML says LONG" | "64% chance of continuation, 19% fakeout risk" |
| Calibration | Hard to verify | 60% predictions should hit 60% of time |
| Flexibility | Black box | You can adjust threshold (e.g., require P > 70%) |
| Mode-specific | Same labels | Different outcomes per mode |

## How to Train

```python
from ml import train_probabilistic_model
import pandas as pd

# Load your historical data
df = pd.read_csv('historical_data.csv')

# Must have these columns:
# - Open, High, Low, Close, Volume
# - ATR (average true range)
# - BB_upper, BB_lower (Bollinger Bands)
# Optional: CMF, hour, etc.

# Train for day trading
metrics = train_probabilistic_model(
    df=df,
    mode='daytrade',
    whale_data={'whale_long_pct': 55, 'retail_long_pct': 48},
    progress_callback=lambda pct, msg: print(f"{pct:.0%}: {msg}")
)

print(f"Continuation accuracy: {metrics['metrics']['continuation']['accuracy']:.1%}")
print(f"Fakeout accuracy: {metrics['metrics']['fakeout']['accuracy']:.1%}")
```

## How to Use

```python
from ml import get_unified_ml_prediction

# Get prediction
pred = get_unified_ml_prediction(
    df=df,
    mode='daytrade',
    whale_data={
        'whale_long_pct': 63,
        'retail_long_pct': 48,
        'oi_change_24h': 5.2,
        'funding_rate': 0.01,
    }
)

# Access probabilities
print(f"P(continuation) = {pred.probabilities['continuation']:.2f}")
print(f"P(fakeout) = {pred.probabilities['fakeout']:.2f}")
print(f"P(vol_expansion) = {pred.probabilities['vol_expansion']:.2f}")

# Access derived direction
print(f"Direction: {pred.direction} ({pred.confidence:.0f}%)")
print(f"Reasoning: {pred.reasoning}")

# Top features driving this prediction
for name, value, importance in pred.top_features:
    print(f"  {name}: {value:.2f} (importance: {importance:.3f})")
```

## How to Verify Results

### 1. Probability Calibration

A well-calibrated model has actual outcomes match predicted probabilities:

| Predicted P | Expected Hits | Acceptable Range |
|-------------|---------------|------------------|
| 50-60% | ~55% | 50-65% |
| 60-70% | ~65% | 58-72% |
| 70-80% | ~75% | 68-82% |
| 80-90% | ~85% | 78-92% |

```python
from ml.probabilistic_ml import verify_probability_calibration, print_calibration_report

# Collect predictions over time
predictions = [
    {'probabilities': pred.probabilities, 'actual_outcomes': {'continuation': 1}},
    # ... more predictions with actual outcomes
]

buckets = verify_probability_calibration(predictions, 'continuation')
print_calibration_report(buckets, 'continuation')
```

### 2. Return vs Probability

Higher P(continuation) should correlate with higher returns:

```
P(continuation)  |  Avg Return
    50-60%       |    +0.5%
    60-70%       |    +1.2%
    70-80%       |    +2.1%
    80-90%       |    +3.5%
```

### 3. Drawdown vs Probability

Higher P(fakeout) should correlate with larger drawdowns:

```
P(fakeout)  |  Avg Drawdown
   < 20%    |    -0.8%
  20-40%    |    -1.5%
  40-60%    |    -2.8%
   > 60%    |    -4.2%
```

## Integration with MASTER_RULES

The probabilistic ML system **complements** MASTER_RULES:

1. **MASTER_RULES**: Analyzes whale/retail, OI, position → Suggests direction
2. **Probabilistic ML**: Predicts outcome probabilities → Validates or warns

When both agree with high confidence, the signal is strong.
When they conflict, investigate further.

```python
# Example integration
from core.MASTER_RULES import get_trade_decision
from ml import get_unified_ml_prediction

# Get MASTER_RULES decision
rules_decision = get_trade_decision(whale_pct=63, retail_pct=48, ...)

# Get ML probabilities
ml_pred = get_unified_ml_prediction(df, 'daytrade', whale_data)

# Check alignment
if rules_decision.trade_direction == ml_pred.direction:
    if ml_pred.probabilities['fakeout'] < 0.25:
        print("✅ Strong signal - Rules + ML agree, low fakeout risk")
    else:
        print("⚠️ Agree but watch for fakeout")
else:
    print(f"⚠️ Conflict: Rules={rules_decision.trade_direction}, ML={ml_pred.direction}")
```

## Files Added

```
ml/
├── probabilistic_ml.py        # Core probabilistic ML system
├── probabilistic_integration.py  # Integration with existing system
└── __init__.py                # Updated exports
```

## Next Steps

1. **Train models**: Use `train_probabilistic_model()` with historical data
2. **Backtest**: Verify probability calibration on out-of-sample data
3. **Integrate**: Replace or augment ML predictions in app.py
4. **Monitor**: Track actual vs predicted probabilities over time
