"""
Formatting Utilities
Price formatting (Professional style), percentages, numbers
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FORMATTING (Professional Style)
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_price(price: float) -> str:
    """
    Format price professionally:
    - >= $1: 2 decimals  → $51.01, $3,245.50
    - < $1: 4 decimals   → $0.0993, $0.0001
    """
    if price is None or price == 0:
        return "$0.00"
    
    if price >= 1:
        # 2 decimals for $1+
        if price >= 1000:
            return f"${price:,.2f}"
        else:
            return f"${price:.2f}"
    else:
        # 4 decimals for < $1
        return f"${price:.4f}"


def fmt_price_simple(price: float) -> str:
    """Format price without $ symbol"""
    if price is None or price == 0:
        return "0.00"
    
    if price >= 1:
        if price >= 1000:
            return f"{price:,.2f}"
        else:
            return f"{price:.2f}"
    else:
        return f"{price:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTAGE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format a percentage value"""
    if value is None:
        return "0.00%"
    
    if include_sign and value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"


def calc_roi(target: float, entry: float) -> float:
    """Calculate ROI percentage between entry and target"""
    if entry == 0:
        return 0
    return ((target - entry) / entry) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBER FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator"""
    if value is None:
        return "0"
    
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:,.{decimals}f}"


def format_volume(volume: float) -> str:
    """Format volume with appropriate suffix"""
    if volume is None or volume == 0:
        return "0"
    
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# TIME FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_timeframe(timeframe: str) -> str:
    """Format timeframe string for display"""
    mapping = {
        '1m': '1 Minute',
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '1h': '1 Hour',
        '4h': '4 Hours',
        '1d': '1 Day',
        '1w': '1 Week'
    }
    return mapping.get(timeframe, timeframe)


def estimate_time_to_target(entry_price: float, target_price: float, 
                            atr: float, timeframe: str) -> dict:
    """
    Estimate time to reach target based on ATR and timeframe
    
    Args:
        entry_price: Entry price
        target_price: Target price (TP or SL)
        atr: Average True Range
        timeframe: Timeframe string
        
    Returns:
        dict with 'candles', 'time_str', 'confidence'
    """
    # Distance to target in price
    distance = abs(target_price - entry_price)
    
    # How many ATRs away is the target?
    if atr > 0:
        atr_multiple = distance / atr
    else:
        atr_multiple = 5
    
    # Base estimate: 1 ATR = ~3-5 candles typically
    estimated_candles = int(atr_multiple * 4)
    
    # Convert to time based on timeframe
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
    }
    
    minutes_per_candle = timeframe_minutes.get(timeframe, 60)
    total_minutes = estimated_candles * minutes_per_candle
    
    # Format time string
    if total_minutes < 60:
        time_str = f"~{total_minutes}min"
    elif total_minutes < 1440:
        hours = total_minutes / 60
        time_str = f"~{hours:.1f}h"
    else:
        days = total_minutes / 1440
        time_str = f"~{days:.1f}d"
    
    # Confidence based on ATR multiple
    if atr_multiple < 2:
        confidence = 'High'
    elif atr_multiple < 4:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    return {
        'candles': estimated_candles,
        'time_str': time_str,
        'confidence': confidence,
        'minutes': total_minutes
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GRADE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_grade_emoji(score: int) -> str:
    """Get emoji for signal grade"""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟢"
    elif score >= 40:
        return "🟡"
    elif score >= 20:
        return "🟠"
    else:
        return "🔴"


def get_grade_letter(score: int) -> str:
    """Get letter grade for score"""
    if score >= 80:
        return "A+"
    elif score >= 60:
        return "A"
    elif score >= 40:
        return "B"
    elif score >= 20:
        return "C"
    else:
        return "D"


def get_quality_badge(score: int) -> str:
    """Get quality badge for score"""
    if score >= 70:
        return "🔥 HIGH QUALITY"
    elif score >= 50:
        return "✅ GOOD"
    elif score >= 30:
        return "👀 WATCH"
    else:
        return "⚠️ WEAK"
