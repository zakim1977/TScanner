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


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SIGNAL HEADER COMPONENT
# Used by Scanner and Single Analysis for consistent display
# ═══════════════════════════════════════════════════════════════════════════════

def get_setup_info(direction: str) -> tuple:
    """
    Get setup text and color based on direction.
    Returns: (setup_text, setup_color, conclusion_color, conclusion_bg)
    """
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
    bg_color: str = "#1a1a2e"
) -> str:
    """
    Generate the signal header HTML box.
    Used by Scanner and Single Analysis for consistent display.
    """
    header_color, header_emoji = get_header_color(action_word)
    
    return f"""
    <div style='background: linear-gradient(135deg, {bg_color} 0%, #16213e 100%); 
                padding: 25px; border-radius: 12px; margin-bottom: 20px;
                border-left: 5px solid {header_color};'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='flex: 1;'>
                <h2 style='margin: 0; color: {header_color}; font-size: 1.8em;'>
                    {header_emoji} {symbol}: {action_word} ({score}% predictive score)
                </h2>
                <p style='color: #ccc; margin: 10px 0 0 0; font-size: 1.1em;'>
                    {summary}
                </p>
                <p style='color: #888; margin: 10px 0 0 0;'>
                    Direction: {direction_score}/40 | Squeeze: {squeeze_score}/30 | Entry: {timing_score}/30
                </p>
            </div>
            <div style='text-align: right; padding-left: 20px;'>
                <div style='color: {setup_color}; font-size: 2.2em; font-weight: bold; white-space: nowrap;'>
                    {setup_text}
                </div>
            </div>
        </div>
    </div>
    """


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
    whale_pct: float = 50
) -> dict:
    """
    Generate Combined Learning section content.
    Returns dict with 'conclusion_html' and 'stories_html' for use in expander.
    """
    # Conclusion box HTML
    conclusion_html = f"""
    <div style='background: {conclusion_bg}; border: 2px solid {conclusion_color}; border-radius: 10px; padding: 15px; margin-bottom: 15px;'>
        <div style='color: {conclusion_color}; font-size: 1.3em; font-weight: bold;'>🎯 CONCLUSION</div>
        <div style='color: #fff; font-size: 1.1em; margin-top: 8px;'>{conclusion}</div>
        <div style='color: {conclusion_color}; font-size: 1em; margin-top: 8px;'>Action: <strong>{conclusion_action}</strong></div>
    </div>
    """
    
    # Stories HTML
    stories_html = ""
    for title, content in stories:
        stories_html += f"""
        <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; margin-bottom: 10px; border-left: 3px solid #444;'>
            <div style='color: #888; font-size: 0.85em; margin-bottom: 5px;'>{title}</div>
            <div style='color: #ddd; font-size: 0.95em;'>{content}</div>
        </div>
        """
    
    # Squeeze warning HTML
    squeeze_html = ""
    if is_squeeze and squeeze_type:
        squeeze_warning_color = "#00ff88" if squeeze_type == 'SHORT' else "#ff6b6b"
        squeeze_html = f"""
        <div style='background: rgba(255, 0, 255, 0.1); border: 2px solid #ff00ff; border-radius: 10px; padding: 15px; margin-top: 15px;'>
            <div style='color: #ff00ff; font-size: 1.1em; font-weight: bold;'>⚡ SQUEEZE ALERT</div>
            <div style='color: #fff; margin-top: 8px;'>
                <strong>{squeeze_type} SQUEEZE</strong> detected!<br>
                This means <strong>{squeeze_type}S are getting liquidated</strong>.<br>
                Price will move <strong>{'UP' if squeeze_type == 'SHORT' else 'DOWN'}</strong> as they're forced to close.<br>
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
        'stories_html': stories_html,
        'squeeze_html': squeeze_html,
        'conflict_html': conflict_html,
        'capitulation_html': capitulation_html
    }
