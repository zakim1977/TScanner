# TScanner - InvestorIQ Smart Money Analysis Platform

## Project Overview

A comprehensive trading analysis platform built with Streamlit that combines:
- Smart Money Concepts (SMC) technical analysis
- Whale/institutional tracking
- Machine learning predictions
- Liquidity sweep detection

**Markets Supported:** Crypto, Stocks, ETFs

---

## Key Files

### Main Application
- `app.py` - Main Streamlit application (~21,800+ lines)
  - Page config, imports, session state
  - Scanner functionality
  - Single asset analysis
  - Trade management
  - ML Training page (Liquidity Hunter ML + Legacy ML tabs)
  - Liquidity Hunter page (with market type dropdown: Crypto/Stock/ETF)

### Configuration
- `config.py` - App settings, trading modes, thresholds
- `styles.py` - Custom CSS styling

---

## Core Modules (`/core`)

| File | Purpose |
|------|---------|
| `data_fetcher.py` | Fetch OHLCV data from Binance (crypto) and Yahoo Finance (stocks/ETFs) |
| `signal_generator.py` | Generate trade signals with scoring |
| `smc_detector.py` | Smart Money Concepts detection (order blocks, structure) |
| `money_flow.py` | Calculate money flow, whale activity |
| `level_calculator.py` | Calculate support/resistance, entry/exit levels |
| `indicators.py` | RSI, EMA, ATR, BBANDS, VWAP calculations |
| `strategy_selector.py` | Select best entry strategy based on conditions |
| `trade_optimizer.py` | Optimize trade setups |
| `liquidity.py` | Liquidity score calculations |
| `narrative_engine.py` | Generate human-readable analysis |
| `unified_scoring.py` | Single source of truth for all scores |
| `analysis_router.py` | Route between Rule-Based, ML, Hybrid modes |
| `etf_flow.py` | **ETF/Stock money flow scoring** — OBV, MFI, CMF, price extension, institutional |
| `whale_institutional.py` | Whale/institutional data analysis |
| `institutional_engine.py` | Unified institutional analysis |
| `alert_system.py` | Alert and prediction system |
| `education.py` | Educational content |
| `error_tracker.py` | Error tracking system |

### Key data_fetcher functions
```python
fetch_binance_klines(symbol, interval, limit)  # Crypto via Binance
fetch_stock_data(symbol, interval='1d', limit=500)  # Stocks/ETFs via Yahoo Finance
get_popular_etfs()  # Returns ~65 popular ETF symbols
get_default_futures_pairs()  # Returns crypto futures pairs
```

---

## Liquidity Hunter Module (`/liquidity_hunter`)

**Purpose:** Data-driven liquidity sweep detection with ML quality prediction

### Key Files

| File | Purpose |
|------|---------|
| `liquidity_hunter.py` | **Core logic** - Level detection, sweep detection, trade plans |
| `liquidity_hunter_ui.py` | Streamlit UI components |
| `liquidity_hunter_ml.py` | Unified ML model for sweep probability prediction (crypto) |
| `liquidity_sequence.py` | Sequence visualization with ML probabilities |
| `coinglass_liquidation.py` | Real liquidation data from Coinglass API |
| `liq_level_collector.py` | Data collection for ML training |
| `quality_model.py` | **Multi-market** trade quality prediction (YES/NO) — Crypto, ETF, Stock |
| `price_action_analyzer.py` | SMC-based price action analysis for sweep reactions |
| `__init__.py` | Module exports |

### Core Functions (`liquidity_hunter.py`)

```python
# Level Detection
find_liquidity_levels(df, lookback=50)  # Find swing highs/lows with volume tracking
detect_round_numbers(current_price, price_range_pct=10)  # Psychological levels
calculate_level_strength_score(...)  # Composite 0-100 strength

# Sweep Detection
detect_sweep(df, liquidity_levels, atr, lookback_candles, whale_pct)
check_approaching_liquidity(current_price, liquidity_levels, atr)

# Entry Quality — market_type routes to correct ML model
calculate_entry_quality(sweep_status, current_price, whale_pct, whale_delta, volume_on_sweep, avg_volume, market_type='crypto')
# Returns: {score: 0-100, grade: A/B/C/D, entry_window: OPEN/CLOSING/CLOSED, components: {...}}

# Scanner — market_type determines symbol list + ML model
scan_for_liquidity_setups(symbols, fetch_data_func, timeframe, trading_mode, max_symbols, market_type='crypto')
filter_scanner_results_by_quality(results, min_quality_score, quality_grades, entry_window_status)
get_fresh_only_results(results)  # B+ grade, OPEN/CLOSING window

# Full Analysis
full_liquidity_analysis(symbol, df, whale_pct, trading_mode, market_type='crypto')
generate_liquidity_trade_plan(symbol, current_price, atr, liquidity_levels, sweep_status, liquidation_data, whale_pct, df)
```

### Entry Quality Score System (ENHANCED - Jan 2025)

Scores 0-105 based on 6 components + bonus + ML:

| Component | Max Points | Criteria |
|-----------|------------|----------|
| Freshness | 20 | Sweep age (0-2c=20, 3-4c=15, 5-7c=8, 8-10c=4, 11+=0) |
| Reversal Candle | 25 | Rejection wick (+12), favorable close (+7), engulfing (+3) |
| Follow-Through | 15 | Price held (+5), continuation (+6), cont. candles (+4) |
| Momentum | 15 | RSI extreme (+6), RSI divergence (+6), aligned (+3) |
| Whale | 15 | Whale % alignment + delta direction |
| Volume | 10 | Volume spike on sweep (2x=10, 1.5x=7, 1.2x=4) |
| Structure Break | +5 | BONUS if price broke minor structure confirming reversal |
| Trend | -15 to +5 | Trading with/against EMA 20/50/200 trend |
| **ML Quality** | 0-100% | Trained model prediction (market-specific) |

**Grades:** A+ (85+), A (75+), B+ (65+), B (55+), C+ (45+), C (35+), D (25+), F (<25)
**Entry Window:** OPEN (0-3c), CLOSING (4-10c), CLOSED (11+c)
**Recommendation:** ENTER (ML>50% + rule confirmation), WAIT (needs more signals), SKIP (weak/failed)

### ML Quality Model (`quality_model.py`) — Multi-Market

**Three separate models, one per market:**

| Model | File | Data Source | Whale Data | Forward Candles |
|-------|------|-------------|------------|-----------------|
| Crypto | `models/quality_model.pkl` | Binance 4h klines | Yes (whale_pct, whale_delta) | 40 (4h candles) |
| ETF | `models/quality_model_etf.pkl` | Yahoo Finance daily | No (defaults: 50%, delta=0) | 60 (trading days) |
| Stock | `models/quality_model_stock.pkl` | Yahoo Finance daily | No (defaults: 50%, delta=0) | 60 (trading days) |

**Decision thresholds (all markets):**
- **STRONG_YES (>60%):** HIGH CONFIDENCE ENTER
- **YES (50-60%):** ENTER with rule confirmation
- **MAYBE (40-50%):** Borderline - Only ENTER if rule-based very strong
- **NO (<40%):** SKIP - Setup likely to fail

**ROI metrics are capped** at MAX_TRADES_PER_MONTH=10 to reflect realistic trading capacity (prevents inflated projections from counting all symbols simultaneously).

Key functions:
```python
from liquidity_hunter.quality_model import (
    # Crypto
    get_quality_prediction, train_quality_model, get_quality_model_status,
    # ETF
    get_quality_prediction_etf, train_quality_model_etf, get_quality_model_etf_status,
    # Stock
    get_quality_prediction_stock, train_quality_model_stock, get_quality_model_stock_status,
)

# Crypto prediction
prediction = get_quality_prediction(
    symbol='BTCUSDT', direction='LONG', level_type='EQUAL_LOW',
    level_price=90000, current_price=91000, atr=500,
    whale_pct=65, whale_delta=5
)
# Returns: {probability: 0.67, decision: 'YES', take_trade: True, ...}

# ETF prediction (uses money flow instead of whale data)
prediction = get_quality_prediction_etf(
    symbol='SPY', df=df, direction='LONG', level_type='SWING_LOW',
    level_price=580, current_price=582, atr=3
)
# Automatically computes flow score from df + institutional API

# Stock prediction
prediction = get_quality_prediction_stock(
    symbol='AAPL', direction='SHORT', level_type='EQUAL_HIGH',
    level_price=230, current_price=228, atr=2
)

# Train models
train_quality_model(symbols=['BTCUSDT', 'ETHUSDT'], days=365)
train_quality_model_etf(symbols=['SPY', 'QQQ'], days=500)
train_quality_model_stock(symbols=['AAPL', 'MSFT'], days=500)
```

### Market Type Routing

The `market_type` parameter (`'crypto'`, `'etf'`, `'stock'`) flows through:
1. **app.py** Liquidity Hunter dropdown → derives `market_type`
2. → `full_liquidity_analysis(market_type=...)` or `scan_for_liquidity_setups(market_type=...)`
3. → `calculate_entry_quality(market_type=...)`
4. → Routes to `get_quality_prediction()` / `get_quality_prediction_etf()` / `get_quality_prediction_stock()`

### Reversal Confirmation

The enhanced system checks for reversal confirmation:
- **Rejection Wick:** Sweep candle should have strong wick vs body (manipulation trap)
- **Favorable Close:** Bullish close for LONG, bearish for SHORT
- **Follow-Through:** Subsequent candles continue reversal direction
- **Price Held:** Price hasn't given back sweep gains
- **RSI Divergence:** Price makes new low but RSI makes higher low (bullish divergence)
- **Structure Break:** Price broke minor high/low confirming direction change

### UI Functions (`liquidity_hunter_ui.py`)

```python
render_liquidity_header(analysis)
render_liquidity_map(liquidity_levels, current_price)
render_sweep_status(sweep_status)
render_approaching_levels(approaching)
render_trade_plan(trade_plan)  # Shows entry quality badge + recommendation + ETF flow badge
render_whale_positioning(liquidation_data, whale_delta)
render_scanner_results(results, show_quality_badge=True)  # Shows quality grades + flow phase for ETF
render_etf_flow_badge(etf_flow)  # Color-coded phase badge (ACCUMULATING/DISTRIBUTING/EXTENDED/NEUTRAL)
render_etf_flow_panel(etf_flow)  # Full flow panel with OBV/MFI/CMF, extension, action recommendation
render_entry_quality_badge(entry_quality)  # HTML badge with ENTER/WAIT/SKIP
render_entry_quality_breakdown(entry_quality)  # 6-component display + warnings
render_scanner_quality_summary(results)  # Shows grade counts + window breakdown
render_scanner_controls()  # Returns filter dict with quality filters
filter_scanner_results(results, filters)  # Applies all filters including quality
```

### Scanner Filter Controls

`render_scanner_controls()` returns:
```python
{
    'scan_count': int,           # 10, 20, 50, 100
    'direction_filter': str,     # ALL, LONG ONLY, SHORT ONLY
    'status_filter': list,       # [SWEEP_ACTIVE, IMMINENT, APPROACHING, MONITORING]
    'min_whale_pct': int,        # 40-80
    # Entry quality filters:
    'fresh_only': bool,          # If True, only B+ grade with OPEN/CLOSING window
    'max_candles_ago': int,      # 3-50, max sweep age
    'quality_grades': list,      # [A, B, C, D]
    'entry_windows': list        # [OPEN, CLOSING, CLOSED, NO_SWEEP]
}
```

---

## ML Training Page (`app.py`)

The ML Training page has **two main tabs**:

### Tab 1: Liquidity Hunter ML (default)
- **Quality Model Training** with 3 sub-tabs:
  - **Crypto** — trains on Binance 4h data, includes whale features, unified sweep model status + liquidation data collection (crypto-only sections inside this tab)
  - **ETF** — trains on Yahoo Finance daily data, no whale data, 60 forward candles
  - **Stock** — trains on Yahoo Finance daily data, no whale data, 60 forward candles
- Each sub-tab shows: model status metrics, training controls, symbol selection
- **Deep ML Pattern Detection** — placeholder section for future advanced models (chart patterns, regime classification, multi-timeframe confluence)

### Tab 2: Legacy ML
- Old ML training UI (`ml/training_ui.py` → `render_training_page()`)
- Probabilistic models per mode/market

---

## ML Module (`/ml`)

- `ml/models/probabilistic/` - Trained ML models per mode/market
- Model naming: `prob_model_{mode}_{market}.pkl`
- Modes: scalp, daytrade, swing, investment
- Markets: crypto, stock, etf

---

## Utils (`/utils`)

| File | Purpose |
|------|---------|
| `formatters.py` | Price formatting, ROI calculation |
| `trade_storage.py` | Trade history, settings persistence |
| `charts.py` | Chart creation |
| `institutional_ui.py` | Institutional analysis UI components |

---

## Data Flow

1. **Scanner Flow:**
   ```
   market_type -> symbols -> fetch_data -> find_liquidity_levels -> detect_sweep -> calculate_entry_quality(market_type) -> sort by quality -> display
   ```

2. **Single Analysis Flow:**
   ```
   market_type -> symbol -> fetch_data -> full_liquidity_analysis(market_type) -> trade_plan with entry_quality -> render UI
   ETF/Stock: also computes etf_flow_score -> flow panel + flow badge in trade plan
   ```

3. **ML Training Flow:**
   ```
   select market tab -> choose symbols + days -> fetch historical data -> generate_quality_samples(forward_candles) -> model.train(n_symbols) -> save model.pkl
   ```

---

## Key Concepts

### Liquidity Levels
- **Swing Highs/Lows:** Local price extremes where stops cluster
- **Equal Highs/Lows:** Multiple touches = MAJOR liquidity
- **Round Numbers:** Psychological levels ($100K, $50K)
- **Strength Score:** 0-100 based on touches, volume, formation age

### Sweep Detection
- Price breaks through liquidity level and reverses
- Priority 1: Recently swept levels (real pools)
- Priority 2: Fresh sweeps of unswept levels
- Tracked: candles_ago, direction, confidence

### Whale Alignment (Crypto only)
- `whale_pct`: Whale long percentage (from Binance top traders)
- `whale_delta`: Change in positioning (accumulating/distributing)
- Conflict warning: Sweep direction vs whale flow mismatch

### ETF/Stock Money Flow (`core/etf_flow.py`)

Replaces whale data for ETF/Stock markets. Scores -100 to +100 using volume-based indicators.

**Components:**

| Component | Points | Source |
|-----------|--------|--------|
| OBV trend (rising vs EMA) | ±20 | On-Balance Volume |
| MFI (mapped 0-100 → ±20) | ±20 | Money Flow Index |
| CMF (mapped -1..+1 → ±20) | ±20 | Chaikin Money Flow |
| Accumulation zone detected | +15 | Low price + rising OBV + MFI<30 |
| Distribution zone detected | -15 | High price + falling OBV + MFI>70 |
| Institutional score (live only) | ±25 | `get_stock_institutional_analysis()` |

**Flow Phases:**
- `ACCUMULATING` — flow_score >= 40, price not extended → Action: `ACCUMULATE_MORE`
- `NEUTRAL` — neither accumulating nor distributing → Action: `HOLD`
- `DISTRIBUTING` — flow_score <= -40 → Action: `TRIM_5_10`
- `EXTENDED` — price >10% above EMA50 or >20% above EMA200 → Action: `TRIM_15_20`

**Integration with Entry Quality:**
- ACCUMULATING + sweep of low → whale component = 15/15 (max), label "FLOW ALIGNED"
- DISTRIBUTING + sweep of low → whale component = 3/15 (risky), warning added
- EXTENDED → whale component = 0/15, recommendation overridden to "TRIM"
- Flow data stored in `entry_quality['etf_flow']`

**Integration with ML Quality Model:**
- Flow features map into whale feature slots (same vector size, no model architecture change):
  - `whale_pct` ← `flow_score`, `whale_delta_4h` ← `mfi_value`, `whale_delta_24h` ← `cmf_value`
  - `rule_whale_bullish` ← flow ACCUMULATING, `rule_whale_bearish` ← flow DISTRIBUTING
- Training uses OHLCV-derived flow only (no API calls)
- Live prediction adds institutional data as bonus features

```python
from core.etf_flow import calculate_etf_flow_score

flow = calculate_etf_flow_score(df, ticker='SPY', include_institutional=True)
# Returns: {flow_score: 55, flow_phase: 'ACCUMULATING', action: 'ACCUMULATE_MORE',
#           obv_trend: 1, mfi_value: 35, cmf_value: 0.12, price_extension_ema50: 3.2,
#           price_extension_ema200: 8.1, in_accumulation_zone: True, ...}
```

---

## Recent Enhancements (Jan 2025 - Jan 2026)

1. **ETF Money Flow Scoring Layer (Jan 2026):**
   - `core/etf_flow.py` — OBV, MFI, CMF, price extension, accumulation/distribution zone detection
   - Flow score (-100 to +100) with phase detection: ACCUMULATING / DISTRIBUTING / EXTENDED / NEUTRAL
   - Action recommendations: ACCUMULATE_MORE / HOLD / TRIM_5_10 / TRIM_15_20
   - Integrated into entry quality (replaces whale component for ETF/Stock)
   - Integrated into ML quality model (flow features mapped to whale feature slots)
   - UI: flow panel with component breakdown, flow badge in trade plan, phase indicator in scanner

2. **Multi-Market ML Quality Models (Jan 2026):**
   - Separate trained models for Crypto, ETF, and Stock
   - `market_type` parameter routes through scanner + analysis to correct model
   - ETF/Stock use daily candles (Yahoo Finance), 60 forward candles + money flow features
   - Crypto uses 4h candles (Binance), 40 forward candles + whale features
   - ROI metrics capped at 10 trades/month (realistic projections)

2. **ML Training Page Restructured (Jan 2026):**
   - Two main tabs: Liquidity Hunter ML (default) + Legacy ML
   - Quality Model has 3 sub-tabs: Crypto, ETF, Stock
   - Unified sweep model + liquidation data moved inside Crypto tab only
   - Deep ML Pattern Detection placeholder added

3. **Liquidity Hunter Market Dropdown (Jan 2026):**
   - Dropdown selects Crypto/Stock/ETF
   - Scanner uses correct symbol lists per market
   - Data fetching routes to Binance (crypto) or Yahoo (stock/ETF)
   - ML prediction uses market-specific trained model

4. **Enhanced Liquidity Detection (Jan 2025):**
   - Volume confirmation for levels
   - Round number detection
   - ATR-based dynamic clustering
   - Composite strength scores

5. **ENHANCED Entry Quality Score System (Jan 2025):**
   - 0-105 scoring with A+/A/B+/B/C+/C/D/F grades
   - Reversal Candle Analysis, Follow-Through Check, Momentum Confirmation
   - Structure Break Bonus, ENTER/WAIT/SKIP Recommendation, Warnings System

6. **Scanner Improvements:**
   - Sort by ENTER recommendations first, then WAIT, then SKIP
   - Quality badges with recommendation indicator
   - Market-specific symbol lists and ML model routing

---

## API Keys (Optional)

- `coinglass_api_key` - Real liquidation heatmap data
- `quiver_api_key` - Stock institutional data (Congress, insider trades)

---

## Running the App

```bash
streamlit run app.py
```

---

## File Locations Quick Reference

```
TScanner/
├── app.py                          # Main application (~21,800+ lines)
├── config.py                       # Settings
├── styles.py                       # CSS
├── CLAUDE.md                       # This file
├── models/                         # Trained ML model files
│   ├── quality_model.pkl           # Crypto quality model
│   ├── quality_model_etf.pkl       # ETF quality model
│   └── quality_model_stock.pkl     # Stock quality model
├── core/                           # Core analysis modules
│   ├── data_fetcher.py             # Binance + Yahoo Finance data
│   ├── etf_flow.py                 # ETF/Stock money flow scoring
│   ├── signal_generator.py
│   ├── smc_detector.py
│   └── ...
├── liquidity_hunter/               # Liquidity sweep module
│   ├── __init__.py
│   ├── liquidity_hunter.py         # Core logic (market_type aware)
│   ├── liquidity_hunter_ui.py      # UI components
│   ├── liquidity_hunter_ml.py      # Unified sweep ML (crypto)
│   ├── liquidity_sequence.py       # Sequence visualization
│   ├── coinglass_liquidation.py    # Real liquidation data
│   ├── liq_level_collector.py      # Data collection
│   ├── quality_model.py            # Multi-market quality prediction
│   └── price_action_analyzer.py    # SMC price action analysis
├── ml/                             # Legacy ML models
│   └── models/probabilistic/
├── utils/                          # Utilities
│   ├── formatters.py
│   ├── trade_storage.py
│   ├── charts.py
│   └── institutional_ui.py
└── coin_risk_detector.py           # Coin risk detection
```
