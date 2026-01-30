# 📊 InvestorIQ Data Strategy: Historical Import + Daily Collection

## Overview

InvestorIQ uses a **hybrid data strategy** combining:
1. **One-time historical import** from paid APIs (instant backtesting)
2. **Ongoing daily collection** from free APIs (growing dataset)

This approach gives you immediate backtesting capability while building a comprehensive proprietary dataset over time.

---

## 🎯 Supported Markets

| Market | Data Source | Cost | Historical Import |
|--------|-------------|------|-------------------|
| **Crypto** | Coinglass | $35/mo | ✅ Ready |
| **Stocks** | Quiver Quant | Free tier / $30/mo | ✅ Ready |

---

## 🐋 CRYPTO: Whale Data Strategy

### The Problem

| Challenge | Impact |
|-----------|--------|
| New users have 0 historical data | Can't validate signals |
| Building data takes months | Delayed value |
| Free APIs have limited history | Can't backtest |
| No outcome tracking | Can't measure win rate |

### The Solution: Hybrid Approach

#### Phase 1: Historical Import (One-Time)
```
[Coinglass Pro API] → [Historical Data Fetcher] → [Local SQLite DB]
```
- Import 6-12 months of historical OI, funding, L/S ratios
- Cost: $35/month (Hobbyist tier)

#### Phase 2: Daily Collection (Ongoing)
```
[Daily Scans] → [Whale Data Store] → [Local SQLite DB]
```
- Every scan stores current whale data automatically
- Cost: FREE (Binance public API)

### Crypto Import Command
```bash
python -m core.historical_data_fetcher \
    --api-key YOUR_COINGLASS_KEY \
    --days 180 \
    --interval 4h
```

---

## 📈 STOCKS: Institutional Data Strategy

### Data Sources from Quiver

| Data Type | What It Shows | Predictive Value |
|-----------|---------------|------------------|
| **Congress Trading** | Nancy Pelosi, etc. buying/selling | HIGH |
| **Insider Trading** | CEO/CFO buys (they know the company) | VERY HIGH |
| **Short Interest** | Squeeze potential | MEDIUM |

### Stock Import Command
```bash
python -m core.stock_historical_fetcher \
    --api-key YOUR_QUIVER_KEY \
    --symbols AAPL MSFT NVDA TSLA \
    --days 180
```

### Quiver API Tiers

| Tier | Cost | Calls/Day | Good For |
|------|------|-----------|----------|
| **Free** | $0 | 100 | Testing, small portfolios |
| **Pro** | $30/mo | 1000 | Full import + daily use |

**Tip:** Free tier is often enough for personal use!

---

## 🔧 Implementation

### Files Created/Modified

| File | Purpose |
|------|---------|
| `core/historical_data_fetcher.py` | NEW: Imports historical data from Coinglass |
| `core/whale_data_store.py` | UPDATED: Tracks data_source (live vs historical) |

### Database Schema
```sql
whale_snapshots (
    id, symbol, timeframe, timestamp,
    
    -- Whale Data
    whale_long_pct, retail_long_pct,
    oi_value, oi_change_24h, funding_rate,
    
    -- Price
    price, price_change_24h,
    
    -- Technical
    position_in_range, mfi, cmf, rsi, volume_ratio, atr_pct,
    
    -- Signal
    signal_direction, predictive_score,
    
    -- Outcomes (filled later)
    hit_tp1, hit_sl, candles_to_result,
    max_favorable_pct, max_adverse_pct,
    
    -- NEW: Data source tracking
    data_source TEXT DEFAULT 'live'  -- 'live' or 'historical'
)
```

---

## 📥 How to Import Historical Data

### Option 1: Streamlit UI (Recommended)
```python
# Add to app.py sidebar
from core.historical_data_fetcher import create_import_ui
create_import_ui()
```

### Option 2: Command Line
```bash
cd TScanner
python -m core.historical_data_fetcher \
    --api-key YOUR_COINGLASS_KEY \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --days 180 \
    --interval 4h
```

### Option 3: Python Script
```python
from core.historical_data_fetcher import HistoricalDataImporter

importer = HistoricalDataImporter(api_key="your_key")

stats = importer.import_historical_data(
    symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
    lookback_days=180,
    interval='4h'
)

print(f"Imported: {stats['records_imported']} records")
```

---

## 💰 Cost Analysis

### Coinglass Pro ($30/month)

| Data Point | Included | Value |
|------------|----------|-------|
| Historical OI | ✅ 1+ year | HIGH |
| Historical Funding | ✅ 1+ year | HIGH |
| Historical L/S Ratio | ✅ 1+ year | HIGH |
| Liquidation Heatmap | ✅ Real-time | VERY HIGH |
| Real-time Alerts | ✅ | MEDIUM |

**ROI Calculation:**
- If avoiding 1 bad trade/month saves $100+ → $30 is worth it
- Historical validation prevents "flying blind"
- Liquidation heatmap prevents stop hunts

### Alternative: CryptoQuant ($29/month)
Similar historical data, different focus (more on-chain).

---

## 📈 Workflow After Setup

### Daily Flow
```
1. Morning scan runs
2. Whale data automatically stored (data_source='live')
3. Historical validation checks against DB
4. Win rate calculated from historical outcomes
```

### Weekly/Monthly
```
1. Background process checks old signals
2. Determines if TP/SL was hit
3. Updates outcome fields
4. Win rate accuracy improves
```

---

## 🗄️ Database Growth Projections

| Period | Live Records | Historical Records | Total |
|--------|--------------|-------------------|-------|
| Day 1 (import) | 0 | ~10,000 | 10,000 |
| Week 1 | ~500 | ~10,000 | 10,500 |
| Month 1 | ~2,000 | ~10,000 | 12,000 |
| Month 6 | ~12,000 | ~10,000 | 22,000 |
| Year 1 | ~24,000 | ~10,000 | 34,000 |

**Storage:** ~50-100MB per year (very manageable)

---

## 🔄 Data Quality

### Historical Data (Imported)
- ✅ Consistent intervals (4h)
- ✅ Complete coverage
- ⚠️ No technical indicators (calculated on demand)
- ⚠️ No signals attached (pure whale data)

### Live Data (Daily Collection)
- ✅ Has technical indicators
- ✅ Has signals and scores
- ✅ Better for validation matching
- ⚠️ Gaps possible (if not scanning)

### Best of Both Worlds
The historical validator uses BOTH:
1. Match current conditions against historical whale patterns
2. Enhance with technical data when available
3. Calculate outcomes from price data

---

## 🚀 Getting Started

### Step 1: Get Coinglass API Key
1. Sign up at [coinglass.com](https://www.coinglass.com/)
2. Subscribe to Pro ($30/month)
3. Go to API section and generate key

### Step 2: Run Import
```bash
# Import top 20 coins, 6 months history
python -m core.historical_data_fetcher \
    --api-key YOUR_KEY \
    --days 180
```

### Step 3: Verify
```python
from core.whale_data_store import get_whale_store

store = get_whale_store()
stats = store.get_statistics()

print(f"Total records: {stats['total_snapshots']}")
print(f"Historical: {stats['historical_records']}")
print(f"Live: {stats['live_records']}")
```

### Step 4: Start Scanning
Normal scans will now:
- Store live data
- Validate against historical patterns
- Show historical win rates

---

## 📊 Example: What Validation Looks Like

**Before Historical Data:**
```
Historical Validation: INSUFFICIENT DATA
"Need 20+ similar patterns to validate"
```

**After Historical Import:**
```
Historical Validation: STRONG MATCH
├── Similar patterns found: 47
├── Win rate: 68%
├── Avg time to TP1: 12 hours
├── History grade: A
└── Confidence: HIGH

Top Matches:
• 2024-11-15: BTCUSDT, 67% whale long, OI +5% → WIN (8 candles)
• 2024-10-22: BTCUSDT, 70% whale long, OI +3% → WIN (14 candles)
• 2024-09-08: BTCUSDT, 65% whale long, OI +7% → LOSS (6 candles)
```

---

## ❓ FAQ

**Q: Do I need the paid API forever?**
A: No! One month of import + ongoing free collection is fine. Only renew if you want updated historical data or liquidation heatmaps.

**Q: Will this slow down my scans?**
A: No. Data is stored locally in SQLite. Queries are fast (<100ms).

**Q: Can I export my data?**
A: Yes! The SQLite database can be exported, backed up, or analyzed externally.

**Q: What if I don't want to pay?**
A: The system works with free data too - it just takes longer to build up validation data. Technical-only validation is still available as fallback.

---

## 📝 Summary

| Approach | Cost | Time to Value | Quality |
|----------|------|---------------|---------|
| Free only | $0 | 3-6 months | Good |
| Hybrid (recommended) | $30 once | Immediate | Excellent |
| Full paid | $30/mo | Immediate | Excellent+ |

**Recommendation:** Subscribe to Coinglass for ONE month, import historical data, then continue with free daily collection. Re-subscribe only if you need liquidation heatmaps or fresh historical data.
