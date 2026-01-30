"""
Configuration and Constants for Crypto Scanner Pro
"""

# ═══════════════════════════════════════════════════════════════════════════════
# API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BINANCE_BASE_URL = "https://api.binance.com/api/v3"

# Timeframe mappings
TIMEFRAME_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MODES
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_MODES = {
    "scalp": {
        "name": "Scalp",
        "timeframes": ["1m", "5m"],
        "default_tf": "5m",
        "hold_time": "Minutes to 1 hour",
        "icon": "⚡",
        "description": "Quick in-and-out trades targeting small moves",
        # Actual trading parameters
        "max_sl_pct": 3.0,      # Tight stops for scalping
        "target_tp1_pct": 1.0,  # Quick small wins
        "target_rr": 1.5,       # Lower R:R acceptable for high win rate
        "min_rr_tp1": 0.75      # Accept lower TP1 R:R for speed
    },
    "day_trade": {
        "name": "Day Trade",
        "timeframes": ["5m", "15m", "1h"],
        "default_tf": "15m",
        "hold_time": "1-8 hours",
        "icon": "📊",
        "description": "Intraday positions closed same day",
        # Actual trading parameters
        "max_sl_pct": 5.0,      # Medium stops
        "target_tp1_pct": 3.0,  # Decent first target
        "target_rr": 2.0,       # Standard R:R
        "min_rr_tp1": 1.0       # Need at least 1:1 at TP1
    },
    "swing_trade": {
        "name": "Swing Trade",
        "timeframes": ["1h", "4h", "1d"],
        "default_tf": "4h",
        "hold_time": "2-14 days",
        "icon": "📈",
        "description": "Capture medium-term price swings",
        # Actual trading parameters
        "max_sl_pct": 8.0,      # Wider stops for swing
        "target_tp1_pct": 5.0,  # Larger first target
        "target_rr": 2.5,       # Better R:R for longer holds
        "min_rr_tp1": 1.0       # Standard TP1 requirement
    },
    "investment": {
        "name": "Long-Term Investment",
        "timeframes": ["1d", "1w"],
        "default_tf": "1w",
        "hold_time": "Months to Years",
        "icon": "💎",
        "description": "Accumulation strategy for wealth building",
        # Actual trading parameters
        "max_sl_pct": 15.0,     # Wide stops for volatility
        "target_tp1_pct": 10.0, # Large first target
        "target_rr": 3.0,       # High R:R for patience
        "min_rr_tp1": 1.0       # Standard TP1 requirement
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Top crypto pairs to scan
TOP_CRYPTO_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    "SHIBUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT",
    "UNIUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "APTUSDT",
    "NEARUSDT", "FILUSDT", "LDOUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT", "JUPUSDT",
    "WIFUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "ORDIUSDT",
    "FETUSDT", "RENDERUSDT", "TAOUSDT", "KASUSDT", "RUNEUSDT",
    "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "COMPUSDT",
    "ICPUSDT", "HBARUSDT", "VETUSDT", "ALGOUSDT", "FTMUSDT"
]

# Popular ETFs for scanning
TOP_ETFS = [
    # Broad Market
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VT",
    # Sector
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
    # Tech/Growth  
    "ARKK", "VGT", "IGV", "SOXX", "SMH",
    # Bonds
    "TLT", "IEF", "BND", "AGG", "LQD", "HYG",
    # International
    "EFA", "EEM", "VEA", "VWO",
    # Commodities
    "GLD", "SLV", "USO", "UNG",
    # Dividend
    "VYM", "SCHD", "DVY",
    # Real Estate
    "VNQ", "IYR",
]

# Popular stocks for scanning
TOP_STOCKS = [
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Tech
    "AMD", "INTC", "CRM", "ADBE", "ORCL", "CSCO", "QCOM", "AVGO",
    # Finance
    "JPM", "BAC", "WFC", "GS", "V", "MA", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",
    # Consumer
    "WMT", "HD", "NKE", "MCD", "SBUX", "COST", "PG", "KO",
    # Industrial
    "CAT", "BA", "GE", "HON", "UPS",
    # Energy
    "XOM", "CVX", "COP",
    # Communication
    "DIS", "NFLX", "CMCSA",
]

# Signal quality thresholds
SIGNAL_THRESHOLDS = {
    "high_confidence": 70,
    "medium_confidence": 50,
    "low_confidence": 35,
    "minimum": 25
}

# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RISK_PERCENT = 2.0  # % of account per trade
DEFAULT_RR_RATIO = 2.0  # Minimum Risk:Reward ratio
MAX_RISK_PERCENT = 10.0

# Take Profit multipliers (based on ATR)
TP_MULTIPLIERS = {
    "tp1": 1.5,
    "tp2": 2.5,
    "tp3": 4.0
}

# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════════

TRADES_FILE = "trade_history.json"

# ═══════════════════════════════════════════════════════════════════════════════
# UI SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

APP_TITLE = "InvestorIQ"
APP_SUBTITLE = "Smart Money Analysis for Crypto • Stocks • ETFs"
APP_ICON = "🧠"

# Auto-refresh options (seconds)
REFRESH_OPTIONS = [0, 30, 60, 120, 300]

# ═══════════════════════════════════════════════════════════════════════════════
# TIMEFRAME CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

def get_timeframe_context(timeframe: str) -> dict:
    """Get trading context based on timeframe"""
    contexts = {
        '5m': {'style': 'Scalp', 'hold': '15min-1h', 'icon': '⚡'},
        '15m': {'style': 'Day Trade', 'hold': '1-4h', 'icon': '📊'},
        '1h': {'style': 'Day/Swing', 'hold': '4-24h', 'icon': '📈'},
        '4h': {'style': 'Swing', 'hold': '1-5 days', 'icon': '📈'},
        '1d': {'style': 'Position', 'hold': '1-4 weeks', 'icon': '🏦'},
        '1w': {'style': 'Investment', 'hold': '1-3 months', 'icon': '💎'}
    }
    return contexts.get(timeframe, {'style': 'Swing', 'hold': '1-7 days', 'icon': '📈'})

# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY EXPLANATIONS (for education tab)
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_EXPLANATIONS = {
    "overview": """
## 🎯 Trading Strategy Overview

This scanner uses **Smart Money Concepts (SMC)** combined with **Volume Analysis** to detect 
potential moves BEFORE they happen.

### Core Philosophy
> "Trade where institutions trade - at key levels with volume confirmation"

### What We Detect:
1. **Accumulation Zones** - Where smart money is quietly buying
2. **Compression/Squeeze** - Price coiling before explosive move
3. **Support/Resistance Zones** - Key price levels
4. **Volume Anomalies** - Unusual buying/selling activity
5. **Money Flow** - OBV, MFI, CMF indicators
""",
    
    "volume_analysis": """
### 📊 Volume Analysis (What We Use)

**OBV (On-Balance Volume)**
- Cumulative volume indicator
- Rising OBV + flat price = accumulation (bullish)
- Falling OBV + flat price = distribution (bearish)

**MFI (Money Flow Index)**
- RSI but with volume
- < 20 = oversold (potential buy)
- > 80 = overbought (potential sell)

**CMF (Chaikin Money Flow)**
- Measures buying/selling pressure
- > 0 = buying pressure
- < 0 = selling pressure
""",

    "smc_concepts": """
### 🏦 Smart Money Concepts

**Order Blocks (OB)**
- Last opposing candle before a strong move
- Represents institutional entry zones
- Price often returns to test these levels

**Fair Value Gaps (FVG)**
- Imbalance zones where price moved too fast
- Acts as magnet - price tends to fill these
- Good entry points when price returns

**Liquidity Sweeps**
- Stop hunts below lows / above highs
- Creates fake breakouts
- Smart money grabs liquidity before real move
""",

    "signal_grades": """
### 📊 Signal Grading System

| Grade | Score | Meaning |
|-------|-------|---------|
| A+ | 80+ | Exceptional - Multiple confirmations |
| A | 60-79 | Strong - Good setup with confirmations |
| B | 40-59 | Moderate - Decent but needs monitoring |
| C | 20-39 | Weak - Watch only, don't trade |
| D | <20 | Poor - Skip entirely |

**What adds to score:**
- At Order Block (+25)
- At FVG (+20)
- Money Inflow (+20)
- Volume Spike (+15)
- Pre-breakout Pattern (+15)
"""
}
