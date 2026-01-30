"""
Market Conditions Data Fetcher
==============================
Fetches market condition data to pass into MASTER_RULES.get_trade_decision()

This module provides:
- Fear & Greed Index from alternative.me API
- BTC trend from Binance
- Options expiry calendar check
- Holiday/liquidity check

Usage:
    from core.market_conditions import get_market_data
    
    market = get_market_data()
    decision = get_trade_decision(
        whale_pct=70,
        retail_pct=55,
        ...
        fear_greed=market['fear_greed'],
        btc_change_24h=market['btc_change_24h'],
        is_options_expiry=market['is_options_expiry'],
        is_holiday=market['is_holiday']
    )
"""

import requests
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


# Major options expiry dates (last Friday of each month, quarterly dates are bigger)
MAJOR_EXPIRY_DATES_2025 = [
    "2025-01-31", "2025-02-28", "2025-03-28", "2025-04-25",
    "2025-05-30", "2025-06-27", "2025-07-25", "2025-08-29",
    "2025-09-26", "2025-10-31", "2025-11-28", "2025-12-26",
]

QUARTERLY_EXPIRY_DATES_2025 = [
    "2025-03-28", "2025-06-27", "2025-09-26", "2025-12-26"
]

# Major holidays (thin liquidity)
HOLIDAYS = [
    (12, 24), (12, 25), (12, 26), (12, 31),  # Christmas/NY
    (1, 1),   # New Year
    (7, 4),   # July 4th
    (11, 28), (11, 29),  # Thanksgiving
]

# Cache for API calls
_cache = {}
_cache_duration = 300  # 5 minutes


def get_fear_greed() -> int:
    """
    Fetch Fear & Greed Index from alternative.me API
    Returns: int 0-100 (0=Extreme Fear, 100=Extreme Greed)
    """
    try:
        cache_key = 'fear_greed'
        if cache_key in _cache:
            cached = _cache[cache_key]
            if (datetime.now() - cached['time']).seconds < _cache_duration:
                return cached['value']
        
        response = requests.get(
            "https://api.alternative.me/fng/",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                value = int(data['data'][0].get('value', 50))
                _cache[cache_key] = {'value': value, 'time': datetime.now()}
                return value
        
        return 50  # Default neutral
        
    except Exception as e:
        logger.warning(f"Failed to fetch Fear & Greed: {e}")
        return 50


def get_btc_change_24h() -> float:
    """
    Get BTC 24h price change from Binance
    Returns: float percentage change
    """
    try:
        cache_key = 'btc_change'
        if cache_key in _cache:
            cached = _cache[cache_key]
            if (datetime.now() - cached['time']).seconds < _cache_duration:
                return cached['value']
        
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={'symbol': 'BTCUSDT'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            value = float(data.get('priceChangePercent', 0))
            _cache[cache_key] = {'value': value, 'time': datetime.now()}
            return value
        
        return 0
        
    except Exception as e:
        logger.warning(f"Failed to fetch BTC change: {e}")
        return 0


def is_options_expiry(date: datetime = None) -> bool:
    """Check if today is a major options expiry day"""
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    return date_str in QUARTERLY_EXPIRY_DATES_2025 or date_str in MAJOR_EXPIRY_DATES_2025


def is_quarterly_expiry(date: datetime = None) -> bool:
    """Check if today is a QUARTERLY options expiry (bigger impact)"""
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    return date_str in QUARTERLY_EXPIRY_DATES_2025


def is_holiday_or_weekend(date: datetime = None) -> bool:
    """Check if current time has thin liquidity (weekends, holidays)"""
    if date is None:
        date = datetime.now()
    
    # Weekend
    if date.weekday() >= 5:
        return True
    
    # Holiday
    if (date.month, date.day) in HOLIDAYS:
        return True
    
    return False


def get_market_data() -> Dict:
    """
    Get all market condition data in one call.
    Use this to get parameters for MASTER_RULES.get_trade_decision()
    
    Returns dict with:
        - fear_greed: int (0-100)
        - fear_greed_label: str (Extreme Fear, Fear, Neutral, Greed, Extreme Greed)
        - btc_change_24h: float
        - btc_trend: str (STRONG_DOWN, DOWN, NEUTRAL, UP, STRONG_UP)
        - is_options_expiry: bool
        - is_quarterly_expiry: bool
        - is_holiday: bool
        - liquidity: str (HIGH, NORMAL, THIN, VERY_THIN)
        - warnings: list of warning strings
    """
    fear_greed = get_fear_greed()
    btc_change = get_btc_change_24h()
    options_expiry = is_options_expiry()
    quarterly = is_quarterly_expiry()
    holiday = is_holiday_or_weekend()
    
    # Fear/Greed label
    if fear_greed <= 25:
        fg_label = "Extreme Fear"
    elif fear_greed <= 45:
        fg_label = "Fear"
    elif fear_greed <= 55:
        fg_label = "Neutral"
    elif fear_greed <= 75:
        fg_label = "Greed"
    else:
        fg_label = "Extreme Greed"
    
    # BTC trend
    if btc_change <= -5:
        btc_trend = "STRONG_DOWN"
    elif btc_change <= -2:
        btc_trend = "DOWN"
    elif btc_change >= 5:
        btc_trend = "STRONG_UP"
    elif btc_change >= 2:
        btc_trend = "UP"
    else:
        btc_trend = "NEUTRAL"
    
    # Liquidity
    if holiday:
        liquidity = "VERY_THIN"
    elif datetime.now().weekday() >= 5:
        liquidity = "THIN"
    else:
        liquidity = "NORMAL"
    
    # Warnings
    warnings = []
    if quarterly:
        warnings.append("⚠️ QUARTERLY OPTIONS EXPIRY - Expect HIGH volatility!")
    elif options_expiry:
        warnings.append("⚠️ Monthly options expiry - Expect volatility")
    
    if holiday:
        warnings.append("⚠️ HOLIDAY - Very thin liquidity!")
    elif datetime.now().weekday() >= 5:
        warnings.append("⚠️ Weekend - Thin liquidity")
    
    if btc_change <= -5:
        warnings.append(f"🔴 BTC DUMP: {btc_change:+.1f}%")
    elif btc_change <= -3:
        warnings.append(f"⚠️ BTC weak: {btc_change:+.1f}%")
    
    return {
        'fear_greed': fear_greed,
        'fear_greed_label': fg_label,
        'btc_change_24h': btc_change,
        'btc_trend': btc_trend,
        'is_options_expiry': options_expiry,
        'is_quarterly_expiry': quarterly,
        'is_holiday': holiday,
        'liquidity': liquidity,
        'warnings': warnings
    }


def get_market_summary() -> str:
    """Get a one-line market summary for display"""
    data = get_market_data()
    
    parts = []
    
    # Fear & Greed emoji
    fg = data['fear_greed']
    if fg <= 25:
        emoji = '😱'
    elif fg <= 45:
        emoji = '😰'
    elif fg <= 55:
        emoji = '😐'
    elif fg <= 75:
        emoji = '😊'
    else:
        emoji = '🤑'
    parts.append(f"{emoji} F&G: {fg}")
    
    # BTC
    btc = data['btc_change_24h']
    btc_emoji = '🟢' if btc > 0 else '🔴' if btc < 0 else '⚪'
    parts.append(f"{btc_emoji} BTC: {btc:+.1f}%")
    
    # Liquidity
    if data['is_holiday']:
        parts.append("💧 HOLIDAY")
    elif data['is_options_expiry']:
        parts.append("📅 EXPIRY")
    
    return " | ".join(parts)


if __name__ == "__main__":
    print("=" * 60)
    print("MARKET CONDITIONS CHECK")
    print("=" * 60)
    
    data = get_market_data()
    
    print(f"\n📊 Fear & Greed: {data['fear_greed']} ({data['fear_greed_label']})")
    print(f"₿ BTC 24h: {data['btc_change_24h']:+.1f}% ({data['btc_trend']})")
    print(f"📅 Options Expiry: {data['is_options_expiry']} (Quarterly: {data['is_quarterly_expiry']})")
    print(f"💧 Liquidity: {data['liquidity']} (Holiday: {data['is_holiday']})")
    
    if data['warnings']:
        print("\n⚠️ WARNINGS:")
        for w in data['warnings']:
            print(f"   {w}")
    
    print(f"\nSummary: {get_market_summary()}")
