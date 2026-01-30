# InvestorIQ - Clean Package

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Directory Structure
```
InvestorIQ/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── styles.py              # UI styling
├── requirements.txt       # Dependencies
├── user_settings.json     # Your saved settings
│
├── core/                  # Core analysis modules
│   ├── trade_optimizer.py    # NEW: Constraint-based SL/TP optimizer
│   ├── MASTER_RULES.py       # Unified decision matrix (27 scenarios)
│   ├── signal_generator.py   # Trading signals
│   ├── smc_detector.py       # Smart Money Concepts detection
│   ├── trade_manager.py      # SMC package integration
│   ├── money_flow_context.py # Market phase detection
│   ├── whale_institutional.py# Whale/institutional data
│   ├── btc_correlation.py    # BTC correlation for altcoins
│   ├── data_fetcher.py       # Price data fetching
│   └── ... (other modules)
│
├── ml/                    # Machine Learning
│   ├── ml_engine.py          # ML prediction engine
│   ├── hybrid_engine.py      # ML + Rules hybrid
│   ├── feature_extractor.py  # Feature engineering
│   └── models/               # Trained models
│
└── utils/                 # Utilities
    ├── charts.py             # Chart generation
    ├── formatters.py         # Price/number formatting
    └── trade_storage.py      # Trade history storage
```

## Key Features

### Trade Optimizer (NEW)
The `trade_optimizer.py` module provides constraint-based level optimization:

**Level Collection:**
- Current TF: Order Blocks, FVGs, Swing H/L
- Higher TF: HTF OBs, Structure
- Historical: Previous day H/L/C
- (Future: POC, VWAP)

**Constraints by Mode:**
| Mode | Timeframe | Max SL | Min R:R |
|------|-----------|--------|---------|
| Scalp | 1m, 5m | 1% | 1:1 |
| Day Trade | 15m, 1h | 3% | 1:1 |
| Swing | 4h, 1d | 5% | 1.5:1 |
| Investment | 1w | 10% | 2:1 |

**Stop Loss Strategy:**
- Structure Break: Below key level (invalidates trade)
- Anti-Hunt: Below structure + ATR buffer + ugly number

**If no valid solution → Returns WAIT (don't trade)**

### MASTER_RULES
27 scenario decision matrix combining:
- Market Structure (Bullish/Bearish/Ranging)
- Money Flow Phase (Accumulation/Distribution/etc.)
- Whale Positioning (% Long/Short)

### SMC Integration
Uses `smartmoneyconcepts` PyPI package for:
- Order Block detection
- Fair Value Gap detection
- Swing High/Low identification
- BOS/CHoCH detection

## Files Removed (Garbage)
The following were removed from the original package:
- Debug scripts (Check_data.py, Debug_Coinglass.py, etc.)
- Training scripts (train_ml.py, calibrate_thresholds.py, etc.)
- Old documentation (.md files)
- Cache directories (__pycache__, catboost_info)
- Duplicate/unused modules

## Support
For issues, check:
1. All dependencies installed: `pip install -r requirements.txt`
2. SMC package installed: `pip install smartmoneyconcepts`
3. API keys configured in Settings
