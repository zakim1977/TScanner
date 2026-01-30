"""
Formatting Utilities
Price formatting (Professional style), percentages, numbers
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FORMATTING (Professional Style)
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_price(price: float) -> str:
    """
    Format price like Binance - dynamic precision based on price level.
    Shows enough decimals to capture meaningful price differences.
    
    Binance precision rules (approximate):
    - $10000+: 2 decimals (BTC: $96,123.45)
    - $100-9999: 2 decimals (ETH: $3,456.78)
    - $10-99: 2-3 decimals ($45.67 or $45.678)
    - $1-9.99: 3 decimals ($1.489)
    - $0.1-0.99: 4 decimals ($0.5678)
    - $0.01-0.099: 5 decimals ($0.05678)
    - Below: 6-8 decimals
    """
    # Convert to float if string
    try:
        price = float(price) if price else 0
    except (ValueError, TypeError):
        return "$0.00"
    
    if price is None or price == 0:
        return "$0.00"
    
    abs_price = abs(price)
    
    # Determine decimal places based on price magnitude (Binance-style)
    if abs_price >= 10000:
        # BTC level: 2 decimals with comma
        return f"${price:,.2f}"
    elif abs_price >= 1000:
        # $1000-$9999: 2 decimals with comma
        return f"${price:,.2f}"
    elif abs_price >= 100:
        # $100-$999: 2 decimals
        return f"${price:.2f}"
    elif abs_price >= 10:
        # $10-$99: 2-3 decimals
        return f"${price:.3f}"
    elif abs_price >= 1:
        # $1-$9.99: 3 decimals (ASRUSDT at $1.489)
        return f"${price:.3f}"
    elif abs_price >= 0.1:
        # $0.10-$0.99: 4 decimals
        return f"${price:.4f}"
    elif abs_price >= 0.01:
        # $0.01-$0.099: 5 decimals
        return f"${price:.5f}"
    elif abs_price >= 0.001:
        # $0.001-$0.0099: 6 decimals
        return f"${price:.6f}"
    elif abs_price >= 0.0001:
        # $0.0001-$0.00099: 7 decimals
        return f"${price:.7f}"
    else:
        # Micro prices: 8 decimals
        return f"${price:.8f}"


def fmt_price_simple(price: float) -> str:
    """Format price without $ symbol - same precision as fmt_price"""
    # Convert to float if string
    try:
        price = float(price) if price else 0
    except (ValueError, TypeError):
        return "0.00"
    
    if price is None or price == 0:
        return "0.00"
    
    abs_price = abs(price)
    
    if abs_price >= 10000:
        return f"{price:,.2f}"
    elif abs_price >= 1000:
        return f"{price:,.2f}"
    elif abs_price >= 100:
        return f"{price:.2f}"
    elif abs_price >= 10:
        return f"{price:.3f}"
    elif abs_price >= 1:
        return f"{price:.3f}"
    elif abs_price >= 0.1:
        return f"{price:.4f}"
    elif abs_price >= 0.01:
        return f"{price:.5f}"
    elif abs_price >= 0.001:
        return f"{price:.6f}"
    elif abs_price >= 0.0001:
        return f"{price:.7f}"
    else:
        return f"{price:.8f}"


# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTAGE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format a percentage value"""
    # Convert to float if string
    try:
        value = float(value) if value else 0
    except (ValueError, TypeError):
        return "0.00%"
    
    if value is None:
        return "0.00%"
    
    if include_sign and value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"


def calc_roi(target: float, entry: float) -> float:
    """Calculate ROI percentage between entry and target"""
    try:
        target = float(target) if target else 0
        entry = float(entry) if entry else 0
    except (ValueError, TypeError):
        return 0
    if entry == 0:
        return 0
    return ((target - entry) / entry) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBER FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator"""
    # Convert to float if string
    try:
        value = float(value) if value else 0
    except (ValueError, TypeError):
        return "0"
    
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
    # Convert to float if string
    try:
        volume = float(volume) if volume else 0
    except (ValueError, TypeError):
        return "0"
    
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
    # Convert to float
    try:
        entry_price = float(entry_price) if entry_price else 0
        target_price = float(target_price) if target_price else 0
        atr = float(atr) if atr else 0
    except (ValueError, TypeError):
        return {'candles': 0, 'time_str': 'N/A', 'confidence': 0, 'probability': 0}
    
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


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SIGNAL HEADER COMPONENT
# Used by Scanner and Single Analysis for consistent display
# ═══════════════════════════════════════════════════════════════════════════════

def get_setup_info(direction: str, position: str = None, position_pct: float = None, setup_type: str = None) -> tuple:
    """
    Get setup text and color based on direction, position, AND setup type.
    
    Args:
        direction: 'LONG', 'SHORT', 'WAIT', etc.
        position: 'EARLY', 'MIDDLE', 'LATE' (optional)
        position_pct: Position percentage 0-100 (optional, used if position not provided)
        setup_type: 'IDEAL_LONG', 'IDEAL_SHORT', 'WHALE_EXIT_TRAP', 'STRONG_ML_DISAGREE' (optional)
    
    Returns: (setup_text, setup_color, conclusion_color, conclusion_bg)
    
    Priority:
        1. IDEAL_LONG/IDEAL_SHORT = Best setups (gold star)
        2. WHALE_EXIT_TRAP = Avoid! (red warning)
        3. Position warnings (LATE don't chase, EARLY risky short)
        4. Normal setup badge
    """
    # Determine position label if only pct provided
    if position is None and position_pct is not None:
        if position_pct >= 65:
            position = 'LATE'
        elif position_pct <= 35:
            position = 'EARLY'
        else:
            position = 'MIDDLE'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 PRIORITY 1: IDEAL SETUPS (best opportunities!)
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'IDEAL_LONG':
        return (
            "🎯 IDEAL LONG",
            "#ffd700",  # Gold
            "#00ff88",
            "rgba(255, 215, 0, 0.15)"
        )
    
    if setup_type == 'IDEAL_SHORT':
        return (
            "🎯 IDEAL SHORT",
            "#ffd700",  # Gold
            "#ff6b6b",
            "rgba(255, 215, 0, 0.15)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟢 PRIORITY 1.5: STRONG SETUPS (Phase + Whale aligned)
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'STRONG_LONG':
        return (
            "🟢 STRONG LONG",
            "#00ff88",
            "#00ff88",
            "rgba(0, 255, 136, 0.15)"
        )
    
    if setup_type == 'STRONG_SHORT':
        return (
            "🔴 STRONG SHORT",
            "#ff6b6b",
            "#ff6b6b",
            "rgba(255, 107, 107, 0.15)"
        )
    
    if setup_type == 'CONFIDENT_LONG':
        return (
            "🟢 LONG",
            "#00ff88",
            "#00ff88",
            "rgba(0, 255, 136, 0.12)"
        )
    
    if setup_type == 'CONFIDENT_SHORT':
        return (
            "🔴 SHORT",
            "#ff6b6b",
            "#ff6b6b",
            "rgba(255, 107, 107, 0.12)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚨 PRIORITY 2: WHALE EXIT TRAP (avoid!)
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'WHALE_EXIT_TRAP':
        return (
            "🚨 WHALE EXIT TRAP",
            "#ff3333",  # Red warning
            "#ff3333",
            "rgba(255, 51, 51, 0.15)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⚠️ PRIORITY 3: STRONG ML DISAGREE (warning)
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'STRONG_ML_DISAGREE':
        return (
            "⚠️ ML CAUTION",
            "#ff9500",  # Orange warning
            "#ff9500",
            "rgba(255, 149, 0, 0.1)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⏳ WAIT type
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'WAIT' or setup_type == 'NO_ML':
        return (
            "⏳ WAIT",
            "#ffcc00",
            "#ffcc00",
            "rgba(255, 204, 0, 0.1)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEAK setups (direction but not confirmed)
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'WEAK_LONG':
        return (
            "🟢 LONG",
            "#00d4aa",  # Slightly muted green
            "#00d4aa",
            "rgba(0, 212, 170, 0.1)"
        )
    
    if setup_type == 'WEAK_SHORT':
        return (
            "🔴 SHORT",
            "#ff8888",  # Slightly muted red
            "#ff8888",
            "rgba(255, 136, 136, 0.1)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 4: POSITION WARNINGS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # LONG but LATE = Don't chase!
    if direction == 'LONG' and position == 'LATE':
        return (
            "⚠️ LATE - DON'T CHASE",
            "#ff9500",  # Orange warning
            "#ff9500",
            "rgba(255, 149, 0, 0.1)"
        )
    
    # SHORT but EARLY = Risky (price near lows)
    if direction == 'SHORT' and position == 'EARLY':
        return (
            "⚠️ EARLY - RISKY SHORT",
            "#ff9500",  # Orange warning
            "#ff9500",
            "rgba(255, 149, 0, 0.1)"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIORITY 5: NORMAL SETUP BADGES
    # ═══════════════════════════════════════════════════════════════════════════
    if direction == 'LONG':
        return (
            "🟢 LONG SETUP",
            "#00ff88",
            "#00ff88",
            "rgba(0, 255, 136, 0.1)"
        )
    elif direction == 'SHORT':
        return (
            "🔴 SHORT SETUP",
            "#ff6b6b",
            "#ff6b6b",
            "rgba(255, 107, 107, 0.1)"
        )
    elif direction == 'WAIT':
        return (
            "⏳ WAIT",
            "#ffcc00",
            "#ffcc00",
            "rgba(255, 204, 0, 0.1)"
        )
    else:
        return (
            "⚠️ CAUTION",
            "#ff9500",
            "#ff9500",
            "rgba(255, 165, 0, 0.1)"
        )


def get_header_color(action_word: str) -> tuple:
    """
    Get header color and emoji based on action word.
    Returns: (header_color, header_emoji)
    """
    action_upper = action_word.upper() if action_word else ""
    
    if "BUY" in action_upper or "LONG" in action_upper:
        return ("#00ff88", "🟢")
    elif "SELL" in action_upper or "SHORT" in action_upper:
        return ("#ff6b6b", "🔴")
    elif "WAIT" in action_upper or "CONFLICT" in action_upper:
        return ("#ffcc00", "🟡")
    else:
        return ("#888", "⚪")


def render_signal_header_html(
    symbol: str,
    action_word: str,
    score: int,
    summary: str,
    direction_score: int,
    squeeze_score: int,
    timing_score: int,
    setup_text: str,
    setup_color: str,
    bg_color: str = "#1a1a2e",
    explosion_score: int = 0,  # Explosion readiness score
    # NEW: ML vs Rules info for professional display
    ml_direction: str = None,
    ml_confidence: float = None,
    rules_direction: str = None,
    is_conflict: bool = False,
    final_verdict: str = None,  # The actual final decision
    # NEW: VWAP bounce info
    vwap_bounce: dict = None,
    # NEW: Data quality tracking
    data_quality: dict = None,
) -> str:
    """
    Generate the signal header HTML box.
    Used by Scanner and Single Analysis for consistent display.
    
    NEW: Shows ML vs Rules in header when there's a conflict (professional approach)
    NEW: Shows data quality warning when using defaults
    """
    header_color, header_emoji = get_header_color(action_word)
    
    # Build DATA QUALITY warning badge if data unreliable
    data_quality_badge = ""
    if data_quality:
        reliability = data_quality.get('reliability_pct', 100)
        warnings = data_quality.get('warnings', [])
        if reliability < 75 or warnings:
            if reliability < 50:
                dq_color = "#ff4444"
                dq_text = f"⚠️ {reliability}% Data"
            else:
                dq_color = "#ffcc00"
                dq_text = f"⚠️ {reliability}% Data"
            
            # Build tooltip with warnings
            tooltip = " | ".join(warnings) if warnings else "Some data using defaults"
            data_quality_badge = f"<span style='background: {dq_color}20; color: {dq_color}; padding: 4px 10px; border-radius: 15px; font-size: 0.7em; font-weight: bold; margin-left: 12px; cursor: help;' title='{tooltip}'>{dq_text}</span>"
    
    # Build explosion display if score > 0
    explosion_display = ""
    if explosion_score > 0:
        exp_color = "#00ff88" if explosion_score >= 70 else "#ffcc00" if explosion_score >= 50 else "#888"
        explosion_display = f" | <span style='color: {exp_color};'>Explosion: {explosion_score}/100</span>"
    
    # Build VWAP display
    vwap_display = ""
    if vwap_bounce:
        bounce_type = vwap_bounce.get('bounce_type', '')
        strength = vwap_bounce.get('strength', '')
        signal_age = vwap_bounce.get('signal_age', '')
        time_ago = vwap_bounce.get('time_ago', '')
        distance_pct = vwap_bounce.get('distance_pct', 0)
        is_retest = vwap_bounce.get('is_retest', False)
        is_confirmed = vwap_bounce.get('is_confirmed', False)
        
        # Determine display based on bounce type - CONFIRMED gets gold/trophy!
        if bounce_type == 'CONFIRMED_BULLISH_RETEST':
            vwap_color = "#ffd700"  # Gold for confirmed!
            vwap_text = "VWAP: 🏆🟢 CONFIRMED"
        elif bounce_type == 'BULLISH_RETEST':
            vwap_color = "#00ff88"
            vwap_text = "VWAP: 🎯🟢 RETEST"
        elif bounce_type == 'BULLISH_BOUNCE_CONFIRMED':
            vwap_color = "#00ff88"
            vwap_text = "VWAP: ✅🟢"
        elif bounce_type == 'BULLISH_BOUNCE':
            vwap_color = "#00ff88"
            vwap_text = "VWAP: 🟢"
        elif bounce_type == 'CONFIRMED_BEARISH_RETEST':
            vwap_color = "#ffd700"  # Gold for confirmed!
            vwap_text = "VWAP: 🏆🔴 CONFIRMED"
        elif bounce_type == 'BEARISH_RETEST':
            vwap_color = "#ff4444"
            vwap_text = "VWAP: 🎯🔴 RETEST"
        elif bounce_type == 'BEARISH_BOUNCE_CONFIRMED':
            vwap_color = "#ff4444"
            vwap_text = "VWAP: ✅🔴"
        elif bounce_type == 'BEARISH_BOUNCE':
            vwap_color = "#ff4444"
            vwap_text = "VWAP: 🔴"
        # APPROACHING WITH PROVEN VWAP - THE PRO SIGNALS!
        elif bounce_type == 'CONFIRMED_APPROACHING_BULLISH':
            vwap_color = "#ffd700"  # Gold!
            vwap_text = "VWAP: 🏆⬇️ LIMIT!"
        elif bounce_type == 'PROVEN_APPROACHING_BULLISH':
            vwap_color = "#ffd700"
            vwap_text = "VWAP: 🏆⬇️ LIMIT!"
        elif bounce_type == 'RETEST_APPROACHING_BULLISH':
            vwap_color = "#00d4ff"
            vwap_text = "VWAP: 🎯⬇️ RETEST"
        elif bounce_type == 'APPROACHING_BULLISH':
            vwap_color = "#00d4ff"
            vwap_text = "VWAP: ⬇️🎯"
        elif bounce_type == 'CONFIRMED_APPROACHING_BEARISH':
            vwap_color = "#ffd700"  # Gold!
            vwap_text = "VWAP: 🏆⬆️ LIMIT!"
        elif bounce_type == 'PROVEN_APPROACHING_BEARISH':
            vwap_color = "#ffd700"
            vwap_text = "VWAP: 🏆⬆️ LIMIT!"
        elif bounce_type == 'RETEST_APPROACHING_BEARISH':
            vwap_color = "#ff9800"
            vwap_text = "VWAP: 🎯⬆️ RETEST"
        elif bounce_type == 'APPROACHING_BEARISH':
            vwap_color = "#ff9800"
            vwap_text = "VWAP: ⬆️🎯"
        elif bounce_type == 'AT_VWAP_RETEST':
            vwap_color = "#00ff88"
            vwap_text = "VWAP: 🎯 RETEST"
        elif bounce_type == 'AT_VWAP':
            vwap_color = "#ffcc00"
            vwap_text = "VWAP: 🎯 AT"
        else:
            # No active bounce signal - show position relative to VWAP (ALWAYS)
            position = vwap_bounce.get('position', '')
            distance_pct_val = abs(distance_pct) if distance_pct else 0
            
            if position == 'ABOVE':
                vwap_color = "#00d4aa"
                # Simple display with distance
                vwap_display = f" | <span style='color: {vwap_color};'>VWAP: 📈 +{distance_pct_val:.1f}%</span>"
            elif position == 'BELOW':
                vwap_color = "#ff9800"
                vwap_display = f" | <span style='color: {vwap_color};'>VWAP: 📉 -{distance_pct_val:.1f}%</span>"
            elif position == 'AT_VWAP':
                vwap_color = "#ffcc00"
                vwap_display = f" | <span style='color: {vwap_color};'>VWAP: 🎯 AT</span>"
            elif distance_pct > 0.1:
                # Fallback: infer from distance_pct
                vwap_color = "#00d4aa"
                vwap_display = f" | <span style='color: {vwap_color};'>VWAP: 📈 +{distance_pct_val:.1f}%</span>"
            elif distance_pct < -0.1:
                vwap_color = "#ff9800"
                vwap_display = f" | <span style='color: {vwap_color};'>VWAP: 📉 -{distance_pct_val:.1f}%</span>"
            else:
                vwap_color = "#ffcc00"
                vwap_display = f" | <span style='color: {vwap_color};'>VWAP: 🎯 AT</span>"
            
            # Skip the rest of vwap_text processing - we already built vwap_display
            vwap_text = ""
        
        if vwap_text:
            # Show age/status with color coding
            if signal_age == 'FRESH':
                age_color = "#00ff88"
                age_badge = "🆕"
            elif signal_age == 'RECENT':
                age_color = "#ffcc00"
                age_badge = ""
            elif signal_age == 'STALE':
                age_color = "#ff6b6b"
                age_badge = "⏰"
            elif signal_age == 'SETUP':
                age_color = "#00d4ff"
                age_badge = "⚡"
            elif signal_age == 'NOW':
                age_color = "#ffcc00"
                age_badge = "🎯"
            else:
                age_color = "#888"
                age_badge = ""
            
            # Strength indicator for approaching signals
            if strength == 'IMMINENT':
                strength_text = "ENTRY!"
                strength_color = "#00ff88"
            elif strength == 'CLOSE':
                strength_text = "READY"
                strength_color = "#00d4ff"
            elif strength == 'APPROACHING':
                strength_text = "WATCH"
                strength_color = "#ffcc00"
            else:
                strength_text = signal_age if signal_age else ""
                strength_color = age_color
            
            # Distance indicator for entry quality
            if abs(distance_pct) < 1:
                dist_color = "#00ff88"
                dist_icon = "🎯"
            elif abs(distance_pct) < 2:
                dist_color = "#ffcc00"
                dist_icon = ""
            else:
                dist_color = "#ff6b6b"
                dist_icon = "🏃"
            
            # Compact display based on type
            approaching_types = [
                'APPROACHING_BULLISH', 'APPROACHING_BEARISH', 
                'RETEST_APPROACHING_BULLISH', 'RETEST_APPROACHING_BEARISH',
                'PROVEN_APPROACHING_BULLISH', 'PROVEN_APPROACHING_BEARISH',
                'CONFIRMED_APPROACHING_BULLISH', 'CONFIRMED_APPROACHING_BEARISH'
            ]
            if bounce_type in approaching_types:
                # Show strength for approaching - and entry level if proven!
                suggested_entry = vwap_bounce.get('suggested_entry')
                entry_type = vwap_bounce.get('entry_type')
                
                # Show actual signed distance for consistency with VWAP Analysis section
                dist_str = f" {dist_icon}{distance_pct:+.1f}%"
                
                # If PROVEN, show entry level
                if entry_type == 'LIMIT' and suggested_entry:
                    entry_str = f" | <span style='color: #ffd700; font-weight: bold;'>LIMIT @ {fmt_price(suggested_entry)}</span>"
                    vwap_display = f" | <span style='color: {vwap_color};'>{vwap_text}</span> <span style='color: {strength_color}; font-weight: bold;'>{strength_text}</span>{entry_str}"
                else:
                    vwap_display = f" | <span style='color: {vwap_color};'>{vwap_text}</span> <span style='color: {strength_color}; font-weight: bold;'>{strength_text}</span><span style='color: {dist_color};'>{dist_str}</span>"
            else:
                # Show age for bounces - use signed distance for consistency
                dist_str = f" {dist_icon}{distance_pct:+.1f}%"
                vwap_display = f" | <span style='color: {vwap_color};'>{vwap_text}</span> <span style='color: {age_color};'>{age_badge}{signal_age}</span><span style='color: {dist_color};'>{dist_str}</span>"
        
        # ═══════════════════════════════════════════════════════════════════
        # VWAP FLIP Display (Andy's Strategy - Two Stage Signal)
        # ═══════════════════════════════════════════════════════════════════
        flip_detected = vwap_bounce.get('flip_detected', False)
        flip_type = vwap_bounce.get('flip_type', '')
        flip_status = vwap_bounce.get('flip_status', '')
        
        if flip_detected and flip_type:
            if flip_status == 'CONFIRMED':
                if 'BULLISH' in flip_type:
                    flip_display = " | <span style='background: #00ff88; color: #000; padding: 2px 8px; border-radius: 10px; font-weight: bold;'>✅ FLIP CONFIRMED (R→S)</span>"
                else:
                    flip_display = " | <span style='background: #ff4444; color: #fff; padding: 2px 8px; border-radius: 10px; font-weight: bold;'>✅ FLIP CONFIRMED (S→R)</span>"
                vwap_display += flip_display
            elif flip_status == 'WATCHING':
                if 'BULLISH' in flip_type:
                    flip_display = " | <span style='background: #ffcc00; color: #000; padding: 2px 8px; border-radius: 10px;'>👀 WATCHING FLIP</span>"
                else:
                    flip_display = " | <span style='background: #ff9800; color: #000; padding: 2px 8px; border-radius: 10px;'>👀 WATCHING FLIP</span>"
                vwap_display += flip_display
            elif flip_status == 'FAILED':
                flip_display = " | <span style='color: #ff6b6b;'>❌ Flip Failed</span>"
                vwap_display += flip_display
    
    # Build ALIGNMENT badge - show when ML + Rules agree
    alignment_badge = ""
    if ml_direction and ml_confidence and rules_direction and not is_conflict:
        # They're aligned - show badge with confidence
        if ml_confidence >= 90:
            alignment_badge = "<span style='background: #00ff88; color: #000; padding: 4px 10px; border-radius: 15px; font-size: 0.7em; font-weight: bold; margin-left: 12px; vertical-align: middle;'>✅ FULL ALIGNMENT</span>"
        elif ml_confidence >= 75:
            alignment_badge = "<span style='background: #00d4aa; color: #000; padding: 4px 10px; border-radius: 15px; font-size: 0.7em; font-weight: bold; margin-left: 12px; vertical-align: middle;'>✅ ALIGNED</span>"
    elif is_conflict:
        alignment_badge = "<span style='background: #ff9800; color: #000; padding: 4px 10px; border-radius: 15px; font-size: 0.7em; font-weight: bold; margin-left: 12px; vertical-align: middle;'>⚠️ CONFLICT</span>"
    
    # Build ML vs Rules display for header (only show detailed when conflict)
    engines_display = ""
    if ml_direction and ml_confidence and rules_direction:
        ml_color = "#a855f7"  # Purple for ML
        rules_color = "#3b82f6"  # Blue for Rules
        
        # Simplified display in header - SINGLE LINE to avoid rendering issues
        if is_conflict:
            final_text = final_verdict or action_word.split()[0]
            engines_display = f"<div style='margin-top: 8px; padding: 8px 12px; background: #0d0d1a; border-radius: 6px; display: inline-block;'><span style='color: {ml_color};'>🤖 ML: {ml_direction} ({ml_confidence:.0f}%)</span><span style='color: #444; margin: 0 10px;'>vs</span><span style='color: {rules_color};'>📊 Rules: {rules_direction}</span><span style='color: #888; margin-left: 15px;'>→</span><span style='color: {header_color}; font-weight: bold; margin-left: 5px;'>Final: {final_text}</span></div>"
    
    # Build return HTML - keep structure but ensure engines_display is clean
    html_parts = []
    html_parts.append(f"<div style='background: linear-gradient(135deg, {bg_color} 0%, #16213e 100%); padding: 25px; border-radius: 12px; margin-bottom: 20px; border-left: 5px solid {header_color};'>")
    html_parts.append(f"<div style='display: flex; justify-content: space-between; align-items: center;'>")
    html_parts.append(f"<div style='flex: 1;'>")
    html_parts.append(f"<h2 style='margin: 0; color: {header_color}; font-size: 1.8em;'>{header_emoji} {symbol}: {action_word} ({int(score)}% predictive score){alignment_badge}{data_quality_badge}</h2>")
    html_parts.append(f"<p style='color: #ccc; margin: 10px 0 0 0; font-size: 1.1em;'>{summary}</p>")
    html_parts.append(f"<p style='color: #888; margin: 10px 0 0 0;'>Direction: {direction_score}/40 | Squeeze: {squeeze_score}/30 | Entry: {timing_score}/30{explosion_display}{vwap_display}</p>")
    html_parts.append(engines_display)
    html_parts.append(f"</div>")
    html_parts.append(f"<div style='text-align: right; padding-left: 20px;'>")
    html_parts.append(f"<div style='color: {setup_color}; font-size: 2.2em; font-weight: bold; white-space: nowrap;'>{setup_text}</div>")
    html_parts.append(f"</div></div></div>")
    
    return "".join(html_parts)


def render_combined_learning_html(
    conclusion: str,
    conclusion_action: str,
    conclusion_color: str,
    conclusion_bg: str,
    stories: list,
    is_squeeze: bool = False,
    squeeze_type: str = None,
    direction: str = None,
    has_conflict: bool = False,
    conflicts: list = None,
    is_capitulation_long: bool = False,
    whale_pct: float = 50,
    unified_story: str = None,  # NEW: Single coherent synthesis
    explosion_state: str = None,  # NEW: Market state
    explosion_score: int = 0,  # NEW: Explosion readiness score
    # NEW: Educational metrics
    compression_pct: float = 50,  # BB tightness (0-100, higher = more compressed)
    squeeze_score: int = 0,  # Squeeze score (0-30)
    retail_pct: float = 50,  # Retail positioning
    position_pct: float = 50,  # Position in range
    ml_direction: str = None,  # ML prediction
    rules_direction: str = None,  # Rules prediction
) -> dict:
    """
    Generate Combined Learning section content.
    Returns dict with 'conclusion_html' and 'stories_html' for use in expander.
    """
    # Conclusion box HTML
    conclusion_html = f"""
    <div style='background: {conclusion_bg}; border: 2px solid {conclusion_color}; border-radius: 10px; padding: 15px; margin-bottom: 15px;'>
        <div style='color: {conclusion_color}; font-size: 1.3em; font-weight: bold;'>🎯 FINAL VERDICT</div>
        <div style='color: #fff; font-size: 1.1em; margin-top: 8px;'>{conclusion}</div>
        <div style='color: {conclusion_color}; font-size: 1em; margin-top: 8px;'>Action: <strong>{conclusion_action}</strong></div>
    </div>
    """
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SETUP METRICS - Simple format that renders properly in Streamlit
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Compression status (BB tightness)
    # squeeze_pct uses HIGH = compressed convention
    if compression_pct >= 80:
        comp_status, comp_color, comp_emoji = "LOADED", "#00ff88", "✅"
    elif compression_pct >= 60:
        comp_status, comp_color, comp_emoji = "BUILDING", "#ffcc00", "⚡"
    elif compression_pct <= 30:
        comp_status, comp_color, comp_emoji = "RELAXED", "#888", "➖"
    else:
        comp_status, comp_color, comp_emoji = "NORMAL", "#aaa", "📊"
    
    # Squeeze status (Whale vs Retail)
    divergence = whale_pct - retail_pct
    if divergence >= 15:
        sqz_status, sqz_color, sqz_emoji = "HIGH LONG", "#00ff88", "🐋"
    elif divergence >= 5:
        sqz_status, sqz_color, sqz_emoji = "LEAN LONG", "#00d4aa", "📈"
    elif divergence <= -15:
        sqz_status, sqz_color, sqz_emoji = "HIGH SHORT", "#ff6b6b", "🐻"
    elif divergence <= -5:
        sqz_status, sqz_color, sqz_emoji = "LEAN SHORT", "#ff9500", "📉"
    else:
        sqz_status, sqz_color, sqz_emoji = "NEUTRAL", "#888", "↔️"
    
    # Entry timing
    if position_pct <= 30:
        timing_status, timing_color, timing_emoji = "EARLY", "#00ff88", "✅"
    elif position_pct <= 50:
        timing_status, timing_color, timing_emoji = "MIDDLE", "#ffcc00", "👀"
    elif position_pct <= 70:
        timing_status, timing_color, timing_emoji = "LATE", "#ff9500", "⚠️"
    else:
        timing_status, timing_color, timing_emoji = "CHASING", "#ff6b6b", "❌"
    
    # ML + Rules agreement - SINGLE SOURCE OF TRUTH for alignment
    # ALIGNED: Both bullish, both bearish, OR both neutral/wait
    # CONFLICT: Directions don't match
    if ml_direction and rules_direction:
        ml_bull = ml_direction in ['LONG']
        ml_bear = ml_direction in ['SHORT']
        ml_neutral = ml_direction in ['WAIT', 'NEUTRAL']
        rules_bull = rules_direction in ['BULLISH', 'LEAN_BULLISH']
        rules_bear = rules_direction in ['BEARISH', 'LEAN_BEARISH']
        rules_neutral = rules_direction in ['NEUTRAL', 'WAIT', 'MIXED']
        
        # Check alignment
        if (ml_bull and rules_bull) or (ml_bear and rules_bear) or (ml_neutral and rules_neutral):
            agree_status, agree_color, agree_emoji = "ALIGNED", "#00ff88", "✅"
        else:
            agree_status, agree_color, agree_emoji = "CONFLICT", "#ff6b6b", "⚠️"
    else:
        agree_status, agree_color, agree_emoji = "N/A", "#888", "➖"
    
    # No separate checklist - all metrics integrated into unified_html
    checklist_html = ""  # Removed - was breaking in Streamlit
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ENHANCED UNIFIED ANALYSIS - Includes all educational metrics inline
    # ═══════════════════════════════════════════════════════════════════════════════
    import html as html_module
    
    # Build enhanced unified story with metrics
    unified_html = ""
    
    # Start with the original story if provided
    story_text = html_module.escape(unified_story) if unified_story else ""
    
    # If no story provided, generate a basic one from metrics
    if not story_text:
        story_parts = []
        # Whale direction
        if whale_pct >= 65:
            story_parts.append(f"🐋 Whales strongly bullish ({whale_pct:.0f}%)")
        elif whale_pct >= 55:
            story_parts.append(f"📈 Whales lean bullish ({whale_pct:.0f}%)")
        elif whale_pct <= 35:
            story_parts.append(f"🐻 Whales strongly bearish ({whale_pct:.0f}%)")
        elif whale_pct <= 45:
            story_parts.append(f"📉 Whales lean bearish ({whale_pct:.0f}%)")
        else:
            story_parts.append(f"↔️ Whales neutral ({whale_pct:.0f}%)")
        
        # Compression
        if compression_pct >= 80:
            story_parts.append("⚡ BB very tight - explosion imminent")
        elif compression_pct >= 60:
            story_parts.append("⚡ Compression building")
        
        # Timing
        if position_pct <= 30:
            story_parts.append("✅ Early entry - great timing")
        elif position_pct >= 70:
            story_parts.append("⚠️ Late entry - reduced R:R")
        
        story_text = " | ".join(story_parts)
    
    # Build metrics line
    metrics_parts = []
    metrics_parts.append(f"<span style='color: {comp_color};'>{comp_emoji} Compression: {compression_pct:.0f}% ({comp_status})</span>")
    metrics_parts.append(f"<span style='color: {sqz_color};'>{sqz_emoji} Squeeze: {sqz_status}</span>")
    metrics_parts.append(f"<span style='color: {timing_color};'>{timing_emoji} Timing: {timing_status} ({position_pct:.0f}%)</span>")
    if ml_direction and rules_direction:
        metrics_parts.append(f"<span style='color: {agree_color};'>{agree_emoji} ML+Rules: {agree_status}</span>")
    
    metrics_line = " | ".join(metrics_parts)
    
    unified_html = f"<div style='background: linear-gradient(135deg, #1a2a3a 0%, #1a1a2e 100%); border: 1px solid #00d4ff33; border-radius: 10px; padding: 15px; margin-bottom: 15px;'><div style='color: #00d4ff; font-size: 1em; font-weight: bold; margin-bottom: 8px;'>📊 UNIFIED ANALYSIS</div><div style='color: #ddd; font-size: 0.95em; line-height: 1.6;'>{story_text}</div><div style='margin-top: 12px; padding-top: 10px; border-top: 1px solid #333; font-size: 0.85em;'>{metrics_line}</div></div>"
    
    # NEW: Explosion State indicator
    explosion_html = ""
    if explosion_state and explosion_state not in ['UNKNOWN', 'RANGING'] and explosion_score >= 30:
        exp_colors = {
            'IGNITION': ('#00ff88', '🚀'),
            'LIQUIDITY_CLEAR': ('#00ff88', '🎯'),
            'COMPRESSION': ('#ffcc00', '⚡'),
            'EXPANSION': ('#00d4ff', '📈'),
            'ACCUMULATION': ('#888', '📦'),
            'DISTRIBUTION': ('#ff6b6b', '🔴'),
        }
        exp_color, exp_emoji = exp_colors.get(explosion_state, ('#888', '📊'))
        explosion_html = f"""
        <div style='background: rgba(0, 212, 255, 0.1); border: 1px solid {exp_color}; border-radius: 8px; padding: 10px 15px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: {exp_color}; font-weight: bold;'>{exp_emoji} Market State: {explosion_state}</span>
            <span style='color: {exp_color};'>Explosion Score: {explosion_score}/100</span>
        </div>
        """
    
    # Stories HTML - build without extra whitespace
    import re
    import html
    stories_parts = []
    for title, content in stories:
        # First escape any HTML entities in content to prevent breaking
        safe_content = html.escape(str(content))
        # Then convert newlines to <br> 
        safe_content = safe_content.replace('\n\n', '<br><br>').replace('\n', '<br>')
        # Convert markdown bold to HTML bold (after escaping so ** aren't escaped)
        safe_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', safe_content)
        
        # Escape title too
        safe_title = html.escape(str(title))
        
        # Build each story div without indentation
        story_div = f"<div style='background: #1a1a2e; border-radius: 8px; padding: 12px; margin-bottom: 10px; border-left: 3px solid #444;'><div style='color: #888; font-size: 0.85em; margin-bottom: 5px;'>{safe_title}</div><div style='color: #ddd; font-size: 0.95em;'>{safe_content}</div></div>"
        stories_parts.append(story_div)
    
    stories_html = "".join(stories_parts)
    
    # Squeeze warning HTML
    squeeze_html = ""
    if is_squeeze and squeeze_type:
        # Determine squeeze direction from trade direction
        # LONG trade = shorts getting squeezed (price UP)
        # SHORT trade = longs getting squeezed (price DOWN)
        is_short_squeeze = direction == 'LONG'
        squeezed_side = "SHORTS" if is_short_squeeze else "LONGS"
        price_direction = "UP" if is_short_squeeze else "DOWN"
        squeeze_warning_color = "#00ff88" if is_short_squeeze else "#ff6b6b"
        
        # Format squeeze intensity
        squeeze_intensity = "HIGH" if "HIGH" in str(squeeze_type) else "MODERATE" if "MODERATE" in str(squeeze_type) else "POTENTIAL"
        
        squeeze_html = f"""
        <div style='background: rgba(255, 0, 255, 0.1); border: 2px solid #ff00ff; border-radius: 10px; padding: 15px; margin-top: 15px;'>
            <div style='color: #ff00ff; font-size: 1.1em; font-weight: bold;'>⚡ {squeeze_intensity} SQUEEZE ALERT</div>
            <div style='color: #fff; margin-top: 8px;'>
                <strong>{squeezed_side} SQUEEZE</strong> detected!<br>
                This means <strong>{squeezed_side} are getting liquidated</strong>.<br>
                Price will move <strong>{price_direction}</strong> as they're forced to close.<br>
                Your trade: <strong style='color: {squeeze_warning_color};'>{direction}</strong>
            </div>
        </div>
        """
    
    # Conflict warning HTML
    conflict_html = ""
    if has_conflict and conflicts:
        conflict_html = f"""
        <div style='background: rgba(255, 165, 0, 0.15); border: 2px solid #ff9500; border-radius: 10px; padding: 15px; margin-top: 15px;'>
            <div style='color: #ff9500; font-size: 1.1em; font-weight: bold;'>⚠️ INDICATOR CONFLICT DETECTED</div>
            <div style='color: #fff; margin-top: 8px;'>
                {'<br>'.join(conflicts)}
            </div>
            <div style='color: #ff9500; margin-top: 10px; font-style: italic;'>
                When indicators conflict, reduce position size or wait for alignment.
            </div>
        </div>
        """
    
    # Capitulation HTML
    capitulation_html = ""
    if is_capitulation_long:
        capitulation_html = f"""
        <div style='background: rgba(0, 255, 136, 0.15); border: 2px solid #00ff88; border-radius: 10px; padding: 15px; margin-top: 15px;'>
            <div style='color: #00ff88; font-size: 1.1em; font-weight: bold;'>🔥 CAPITULATION + WHALE ACCUMULATION</div>
            <div style='color: #fff; margin-top: 8px;'>
                This is a <strong>classic accumulation setup</strong>!<br>
                • Flow shows CAPITULATION (retail panic selling)<br>
                • Whales are {whale_pct:.0f}% LONG (buying the panic)<br>
                • This is how smart money accumulates at bottoms!
            </div>
        </div>
        """
    
    return {
        'conclusion_html': conclusion_html,
        'checklist_html': checklist_html,  # NEW: Educational Setup Checklist
        'unified_html': unified_html,  # NEW: Unified synthesis
        'explosion_html': explosion_html,  # NEW: Market state indicator
        'stories_html': stories_html,
        'squeeze_html': squeeze_html,
        'conflict_html': conflict_html,
        'capitulation_html': capitulation_html
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL VALIDATION RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_historical_validation_html(
    hist_validation,
    total_snapshots: int = 0,
    symbols_tracked: int = 0,
    show_details: bool = True
) -> dict:
    """
    Render historical validation results as HTML.
    
    Args:
        hist_validation: HistoricalValidation object or None
        total_snapshots: Total snapshots in database
        symbols_tracked: Number of symbols tracked
        show_details: Whether to show detailed breakdown
        
    Returns:
        dict with 'has_data', 'title', 'content_html', 'summary_html'
    """
    
    if hist_validation and hist_validation.matches_found >= 20:
        # FULL VALIDATION - Have enough data
        
        # Color based on win rate
        if hist_validation.win_rate >= 65:
            hist_color = "#00ff88"
            hist_bg = "#0a2a1a"
            grade_emoji = "🟢"
        elif hist_validation.win_rate >= 50:
            hist_color = "#ffcc00"
            hist_bg = "#2a2a0a"
            grade_emoji = "🟡"
        else:
            hist_color = "#ff6b6b"
            hist_bg = "#2a1a1a"
            grade_emoji = "🔴"
        
        # Alignment
        if hist_validation.alignment == "ALIGNED":
            align_icon = "✅"
            align_text = "Aligned with prediction"
            align_color = "#00ff88"
        elif hist_validation.alignment == "OVER_CONFIDENT":
            align_icon = "⚠️"
            align_text = f"History lower ({hist_validation.historical_score}%)"
            align_color = "#ffcc00"
        else:
            align_icon = "💡"
            align_text = f"History better ({hist_validation.historical_score}%)"
            align_color = "#ffcc00"
        
        title = f"📊 HISTORICAL VALIDATION - {hist_validation.history_grade} ({hist_validation.matches_found} patterns)"
        
        # Summary line for compact display (single line to avoid rendering issues)
        summary_html = f"<div style='display: flex; gap: 15px; align-items: center; flex-wrap: wrap;'><div style='background: {hist_bg}; padding: 5px 10px; border-radius: 4px;'><span style='color: #888; font-size: 0.75em;'>Win Rate:</span><span style='color: {hist_color}; font-weight: bold;'> {hist_validation.win_rate:.0f}%</span></div><div style='background: #1a1a2e; padding: 5px 10px; border-radius: 4px;'><span style='color: #888; font-size: 0.75em;'>ETA TP1:</span><span style='color: #00ff88;'> ~{hist_validation.avg_time_to_tp1}</span></div><div style='background: #1a1a2e; padding: 5px 10px; border-radius: 4px;'><span style='color: #888; font-size: 0.75em;'>ETA SL:</span><span style='color: #ff6b6b;'> ~{hist_validation.avg_time_to_sl}</span></div><span style='color: {align_color}; font-size: 0.85em;'>{align_icon} {align_text}</span></div>"
        
        # Full content
        content_html = f"""
        <div style='background: {hist_bg}; border-radius: 10px; padding: 15px; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='color: #888; font-size: 0.85em;'>Historical Win Rate</span><br>
                    <span style='color: {hist_color}; font-size: 2em; font-weight: bold;'>{hist_validation.win_rate:.0f}%</span>
                    <span style='color: #888;'> ({hist_validation.matches_found} samples)</span>
                </div>
                <div style='text-align: right;'>
                    <span style='color: #888; font-size: 0.85em;'>Our Score vs History</span><br>
                    <span style='color: #ccc;'>{hist_validation.our_score}</span>
                    <span style='color: #888;'> vs </span>
                    <span style='color: {hist_color};'>{hist_validation.historical_score}</span>
                </div>
            </div>
            <div style='color: {"#00ff88" if hist_validation.alignment == "ALIGNED" else "#ffcc00"}; margin-top: 10px;'>
                {align_icon} {align_text}
            </div>
        </div>
        
        <div style='display: flex; gap: 10px; margin-bottom: 10px;'>
            <div style='flex: 1; background: #1a2a1a; border-radius: 8px; padding: 12px; text-align: center;'>
                <div style='color: #888; font-size: 0.8em;'>⏱️ ETA to TP1 (if win)</div>
                <div style='color: #00ff88; font-size: 1.3em; font-weight: bold;'>{hist_validation.avg_time_to_tp1}</div>
            </div>
            <div style='flex: 1; background: #2a1a1a; border-radius: 8px; padding: 12px; text-align: center;'>
                <div style='color: #888; font-size: 0.8em;'>⏱️ ETA to SL (if loss)</div>
                <div style='color: #ff6b6b; font-size: 1.3em; font-weight: bold;'>{hist_validation.avg_time_to_sl}</div>
            </div>
        </div>
        
        <div style='background: #1a1a2e; border-radius: 8px; padding: 12px;'>
            <div style='color: #888; font-size: 0.85em; margin-bottom: 8px;'>📈 Typical Move Ranges</div>
            <div style='display: flex; justify-content: space-around;'>
                <div style='text-align: center;'>
                    <div style='color: #00ff88;'>Best Case (Avg)</div>
                    <div style='color: #00ff88; font-weight: bold;'>+{hist_validation.avg_max_favorable:.1f}%</div>
                </div>
                <div style='text-align: center;'>
                    <div style='color: #ff6b6b;'>Worst Case (Avg)</div>
                    <div style='color: #ff6b6b; font-weight: bold;'>-{hist_validation.avg_max_adverse:.1f}%</div>
                </div>
            </div>
        </div>
        
        <div style='color: #555; font-size: 0.75em; margin-top: 10px;'>
            💾 {hist_validation.message} • Confidence: {hist_validation.sample_confidence}
        </div>
        """
        
        return {
            'has_data': True,
            'title': title,
            'summary_html': summary_html,
            'content_html': content_html,
            'win_rate': hist_validation.win_rate,
            'eta_tp1': hist_validation.avg_time_to_tp1,
            'eta_sl': hist_validation.avg_time_to_sl,
            'grade': hist_validation.history_grade,
        }
    
    else:
        # NOT ENOUGH DATA - Show building status
        matches_found = hist_validation.matches_found if hist_validation else 0
        
        title = f"📊 HISTORICAL VALIDATION - Building ({matches_found}/20)"
        
        summary_html = f"""
        <div style='color: #666; font-size: 0.85em;'>
            📈 Building history... ({matches_found}/20 patterns needed)
        </div>
        """
        
        content_html = f"""
        <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border: 1px dashed #444;'>
            <div style='color: #888; font-size: 1.1em; margin-bottom: 10px;'>
                📈 Building Historical Database...
            </div>
            <div style='color: #666; font-size: 0.9em; line-height: 1.6;'>
                Need <b>20+ similar patterns</b> for reliable validation.<br>
                Currently have: <span style='color: #ffcc00;'>{matches_found}</span> patterns for this setup.
            </div>
            <div style='margin-top: 15px; padding: 10px; background: #0d0d1a; border-radius: 6px;'>
                <div style='color: #00d4ff; font-size: 0.85em;'>💡 How to build faster:</div>
                <div style='color: #888; font-size: 0.8em; margin-top: 5px;'>
                    1. Run <b>Market Pulse</b> with "Store whale data" checked<br>
                    2. Scan 200-300 coins daily<br>
                    3. After ~1-2 weeks, validation will be available
                </div>
            </div>
            <div style='margin-top: 10px; color: #555; font-size: 0.75em;'>
                Database: {total_snapshots} total snapshots • {symbols_tracked} symbols tracked
            </div>
        </div>
        """
        
        return {
            'has_data': False,
            'title': title,
            'summary_html': summary_html,
            'content_html': content_html,
            'win_rate': None,
            'eta_tp1': None,
            'eta_sl': None,
            'grade': None,
        }