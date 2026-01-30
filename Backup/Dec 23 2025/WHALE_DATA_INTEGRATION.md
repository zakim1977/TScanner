# 🐋 InvestorIQ - Whale & Institutional Data Integration

## What's New

### Whale Data Now Shows In ALL THREE Places:

1. **📊 Scanner Results** - After "Why This Trade" section
2. **📈 Trade Monitor** - Inside each trade's expander
3. **🔬 Single Analysis** - After "Action Plan" section

All three show the SAME data format for consistency!

Fetches **FREE** data from Binance Futures API:
- **Open Interest (OI)** - Total futures positions
- **Funding Rate** - Who's overleveraged
- **Long/Short Ratio (Retail)** - What retail traders are doing
- **Long/Short Ratio (Top Traders)** - What WHALES are doing
- **Taker Buy/Sell Volume** - Aggressive buying/selling

### 2. 📈 Enhanced Trade Monitor

Now shows for each trade:
- **Trading Mode Badge** (Scalp/Day Trade/Swing/Position/Investment)
- **Timeframe Badge** (5m/15m/1h/4h/1d)
- **Grade Badge** (A+/A/B/C/D)
- **Entry Date**
- **Direction** (LONG/SHORT)

Plus **Whale Data** for each crypto trade:
- OI Change 24h
- Funding Rate
- Whales Long %
- Retail Long %
- OI + Price Interpretation (what it means!)

### 3. 📊 Mode-Based Trade Grouping

Trade Monitor now shows breakdown by trading mode:
```
┌──────────┬────────────┬─────────┬──────────┐
│  Scalp   │ Day Trade  │  Swing  │ Position │
│    3     │     2      │    1    │    0     │
└──────────┴────────────┴─────────┴──────────┘
```

---

## 📊 Understanding OI + Price (The KEY to Whale Detection)

### Open Interest + Price Relationship

| OI Change | Price Change | What It Means | Action |
|-----------|--------------|---------------|--------|
| 📈 OI UP | 📈 Price UP | **New LONGS entering** | ✅ Bullish continuation |
| 📈 OI UP | 📉 Price DOWN | **New SHORTS entering** | ❌ Bearish continuation |
| 📉 OI DOWN | 📈 Price UP | **Short covering** (weak) | ⚠️ May reverse |
| 📉 OI DOWN | 📉 Price DOWN | **Long liquidation** (weak) | ⚠️ May reverse |

### Funding Rate (Contrarian!)

| Funding | Meaning | Action |
|---------|---------|--------|
| > 0.1% | Longs overleveraged | 🔴 Dump coming |
| < -0.1% | Shorts overleveraged | 🟢 Pump coming |

### Retail vs Top Traders

| Situation | Action |
|-----------|--------|
| Retail 70% LONG + Whales SHORT | 🔴 FADE RETAIL |
| Retail 70% SHORT + Whales LONG | 🟢 FADE RETAIL |

---

## Files Changed

1. **`core/whale_institutional.py`** (NEW) - Whale data fetcher
2. **`core/__init__.py`** - Added whale module export
3. **`app.py`** - Enhanced Trade Monitor with:
   - Whale data import
   - Mode/Timeframe badges
   - Whale data display
   - OI interpretation guide
   - Mode-based grouping

---

## API Endpoints Used (FREE from Binance)

| Endpoint | Data |
|----------|------|
| `/fapi/v1/openInterest` | Current OI |
| `/futures/data/openInterestHist` | OI history |
| `/fapi/v1/fundingRate` | Funding rate |
| `/futures/data/globalLongShortAccountRatio` | Retail L/S |
| `/futures/data/topLongShortPositionRatio` | Top Traders L/S |
| `/futures/data/takerlongshortRatio` | Taker volume |

**Note:** These endpoints require Binance Futures API access. If blocked (like in some corporate networks), the whale data section will gracefully degrade and not show.

---

## Usage

### In Trade Monitor:
1. Go to **📈 Trade Monitor**
2. Each trade now shows mode, timeframe, and grade badges
3. For crypto trades (USDT pairs), whale data is fetched automatically
4. Click **"📊 Understanding Whale Data"** expander for interpretation guide

### Best Setups to Look For:
1. **Top traders LONG + Retail SHORT + Funding negative** = STRONG LONG
2. **Top traders SHORT + Retail LONG + Funding positive** = STRONG SHORT
3. **OI rising + Price consolidating** = Breakout coming

---

## Troubleshooting

### Whale data not showing?
- Only works for crypto (USDT pairs)
- Binance Futures API must be accessible
- Some networks/regions may block fapi.binance.com

### Mode not showing correctly?
- Make sure to use Scanner or Single Analysis to add trades
- Older trades may not have mode_name saved
