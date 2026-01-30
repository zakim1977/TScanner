"""
═══════════════════════════════════════════════════════════════════════════════
MONEY FLOW CONTEXT MODULE
═══════════════════════════════════════════════════════════════════════════════

This module contextualizes Money Flow based on POSITION in range.

The Problem:
- Raw OBV/MFI/CMF just measure "buying" or "selling" pressure
- BUT buying at 90% of range is NOT accumulation - it's distribution risk!
- Smart money accumulates LOW and distributes HIGH

The Solution:
- Combine Money Flow direction with Position in Range
- Give accurate labels that reflect true Wyckoff phases

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# EDUCATION CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

MONEY_FLOW_EDUCATION = {
    "ACCUMULATION": {
        "title": "Accumulation Phase",
        "emoji": "🟢",
        "short": "Smart money buying at lows",
        "description": """
**What it means:** Institutional investors are quietly building positions at low prices while retail traders are fearful or uninterested.

**How we detect it:** Volume-weighted buying (OBV/MFI/CMF showing inflow) while price is in the LOWER 30% of its recent range.

**Why it matters:** This is the BEST time to enter. You're buying alongside smart money before the markup phase begins.

**What to do:** 
- Scale into positions
- Use wider stops (accumulation can take time)
- Be patient - the move may not happen immediately
""",
        "wyckoff": "Wyckoff Phase A/B - Selling climax → Automatic rally → Secondary test"
    },
    
    "MARKUP": {
        "title": "Markup Phase", 
        "emoji": "📈",
        "short": "Trend continuation - price rising with volume",
        "description": """
**What it means:** The accumulation phase is complete and price is now trending upward. Volume confirms the move.

**How we detect it:** Volume-weighted buying while price is in the MIDDLE of its range (30-70%).

**Why it matters:** The trend is established. This is a good time for trend-following entries on pullbacks.

**What to do:**
- Buy dips within the trend
- Trail stops as price moves up
- Watch for signs of distribution at highs
""",
        "wyckoff": "Wyckoff Phase C/D - Sign of strength → Last point of support → Breakout"
    },
    
    "FOMO / DIST RISK": {
        "title": "FOMO / Distribution Risk",
        "emoji": "⚠️", 
        "short": "Retail FOMO - smart money may be selling",
        "description": """
**What it means:** Price is HIGH in its range but volume is still showing buying. This is typically RETAIL FOMO - late buyers chasing the move.

**How we detect it:** Volume-weighted buying while price is in the UPPER 30% of its range (70-100%).

**Why it matters:** Smart money often SELLS into this retail buying. What looks like "accumulation" at highs is actually distribution in disguise.

**What to do:**
- DO NOT chase here
- If already long, consider taking profits
- Wait for a pullback to better levels
- Watch Layer 1 whale data for confirmation
""",
        "wyckoff": "Wyckoff Phase D - Upthrust after distribution (UTAD) trap"
    },
    
    "DISTRIBUTION": {
        "title": "Distribution Phase",
        "emoji": "🔴",
        "short": "Smart money selling at highs",
        "description": """
**What it means:** Institutional investors are selling their positions to retail traders who are buying at high prices.

**How we detect it:** Volume-weighted selling (OBV/MFI/CMF showing outflow) while price is in the UPPER range.

**Why it matters:** This often precedes a significant price drop. Smart money is exiting.

**What to do:**
- Take profits on longs
- DO NOT buy here
- Consider short positions if Layer 1 confirms
- Wait for markdown to complete before buying
""",
        "wyckoff": "Wyckoff Distribution - Preliminary supply → Buying climax → Secondary test"
    },
    
    "PROFIT TAKING": {
        "title": "Profit Taking",
        "emoji": "🟡",
        "short": "Some selling mid-trend - normal pullback",
        "description": """
**What it means:** Within an uptrend, some participants are taking profits. This is normal and healthy.

**How we detect it:** Volume-weighted selling while price is in the MIDDLE of its range.

**Why it matters:** This can create buying opportunities within a larger uptrend. Not every sell signal means the trend is over.

**What to do:**
- If bullish trend intact, look for support to buy
- Don't panic sell - assess the bigger picture
- Check Layer 1 whale positioning for trend confirmation
""",
        "wyckoff": "Normal reaccumulation within markup phase"
    },
    
    "CAPITULATION": {
        "title": "Capitulation",
        "emoji": "🔶",
        "short": "Panic selling at lows - potential bottom",
        "description": """
**What it means:** Weak hands are panic selling at low prices. This often marks the END of a downtrend.

**How we detect it:** Volume-weighted selling while price is in the LOWER 30% of its range.

**Why it matters:** Capitulation often creates the best buying opportunities. "Blood in the streets" moments.

**What to do:**
- Watch for reversal signals
- Scale in carefully (catching falling knives is risky)
- Use Layer 1 to confirm if whales are buying the panic
""",
        "wyckoff": "Wyckoff Selling Climax - Maximum fear, smart money starts buying"
    },
    
    "CONSOLIDATION": {
        "title": "Consolidation",
        "emoji": "↔️",
        "short": "Pause in trend - no clear direction",
        "description": """
**What it means:** Price is ranging without clear buying or selling pressure. The market is undecided.

**How we detect it:** Neither strong inflow nor outflow, typically in the middle of the range.

**Why it matters:** Consolidation precedes breakouts. The longer the consolidation, the bigger the eventual move.

**What to do:**
- Wait for breakout direction
- Trade the range (buy low, sell high within range)
- Reduce position size until direction is clear
- Watch Layer 1 for early directional clues
""",
        "wyckoff": "Trading range - Market seeking equilibrium"
    },
    
    "RE-ACCUMULATION": {
        "title": "Re-accumulation",
        "emoji": "🔵",
        "short": "Building base at support",
        "description": """
**What it means:** After a move up, price has pulled back to support and is building a new base.

**How we detect it:** Neutral flow while price is at the LOWER end of its range.

**Why it matters:** This is often a pause before the next leg up in a larger uptrend.

**What to do:**
- Look for signs of accumulation starting
- Good area to add to existing positions
- Use the support level for stop placement
""",
        "wyckoff": "Reaccumulation - Pause within larger markup phase"
    },
    
    "EXHAUSTION": {
        "title": "Exhaustion",
        "emoji": "😴",
        "short": "Trend losing momentum at highs",
        "description": """
**What it means:** Price is high but volume/momentum is fading. The move is running out of steam.

**How we detect it:** Neutral flow while price is at the UPPER end of its range.

**Why it matters:** Exhaustion often precedes reversals or significant pullbacks.

**What to do:**
- Tighten stops on existing positions
- Take partial profits
- Don't add to positions here
- Wait for pullback or trend reversal confirmation
""",
        "wyckoff": "Upthrust - Failed test of highs, potential reversal"
    }
}


def get_money_flow_education(phase: str) -> dict:
    """Get education content for a specific phase"""
    return MONEY_FLOW_EDUCATION.get(phase, {
        "title": phase,
        "emoji": "❓",
        "short": "Unknown phase",
        "description": "No education available for this phase.",
        "wyckoff": ""
    })


@dataclass
class MoneyFlowContext:
    """Contextualized money flow based on position in range"""
    # Raw data
    is_inflow: bool           # OBV/MFI/CMF show buying
    is_outflow: bool          # OBV/MFI/CMF show selling
    position_pct: float       # 0-100% position in swing range
    position_label: str       # EARLY, MIDDLE, LATE
    
    # Contextualized interpretation
    phase: str                # ACCUMULATION, DISTRIBUTION, MARKUP, MARKDOWN, etc.
    phase_color: str          # Color for UI
    phase_emoji: str          # Emoji for UI
    phase_detail: str         # Explanation
    
    # Scoring impact
    bullish_bias: int         # -30 to +30 points
    confidence: str           # HIGH, MEDIUM, LOW
    warning: Optional[str]    # Warning message if any


def get_money_flow_context(
    is_accumulating: bool,
    is_distributing: bool,
    position_pct: float,
    structure_type: str = 'Unknown',
    oi_change: float = 0,
    price_change: float = 0
) -> MoneyFlowContext:
    """
    ═══════════════════════════════════════════════════════════════════════════
    WYCKOFF-INSPIRED PHASE DETECTION
    ═══════════════════════════════════════════════════════════════════════════
    
    Combines Money Flow (buying/selling pressure) with Position in Range
    to determine the TRUE market phase.
    
    WYCKOFF PHASES:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN → ACCUMULATION...     │
    │       (bottom)    (rise)     (top)        (fall)      (bottom)         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    LOGIC MATRIX:
    ┌──────────────┬─────────────────┬─────────────────┬─────────────────────┐
    │   Position   │     INFLOW      │     OUTFLOW     │      NEUTRAL        │
    ├──────────────┼─────────────────┼─────────────────┼─────────────────────┤
    │ EARLY (0-30%)│ ✅ ACCUMULATION │ CAPITULATION    │ RE-ACCUMULATION     │
    │              │ Smart $ buying  │ Panic selling   │ Building base       │
    │              │ +25 pts         │ +10 pts (bounce)│ +5 pts              │
    ├──────────────┼─────────────────┼─────────────────┼─────────────────────┤
    │ MIDDLE(30-70)│ MARKUP          │ PROFIT TAKING   │ CONSOLIDATION       │
    │              │ Trend continues │ Some selling    │ Pause in trend      │
    │              │ +15 pts         │ -5 pts          │ 0 pts               │
    ├──────────────┼─────────────────┼─────────────────┼─────────────────────┤
    │ LATE (70-100)│ ⚠️ FOMO/DISTRIB │ DISTRIBUTION    │ EXHAUSTION          │
    │              │ Retail buying!  │ Smart $ selling │ Trend tired         │
    │              │ -15 pts         │ -25 pts         │ -10 pts             │
    └──────────────┴─────────────────┴─────────────────┴─────────────────────┘
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    # Determine position label
    if position_pct <= 30:
        position_label = "EARLY"
    elif position_pct <= 70:
        position_label = "MIDDLE"
    else:
        position_label = "LATE"
    
    # Default values
    phase = "NEUTRAL"
    phase_color = "#888"
    phase_emoji = "⚪"
    phase_detail = "No clear phase"
    bullish_bias = 0
    confidence = "LOW"
    warning = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # EARLY POSITION (0-30%) - Bottom of range
    # ═══════════════════════════════════════════════════════════════════════
    if position_label == "EARLY":
        if is_accumulating:
            # ✅ BEST CASE: Smart money buying at lows
            phase = "ACCUMULATION"
            phase_color = "#00ff88"
            phase_emoji = "🟢"
            phase_detail = "Smart money buying at lows"
            bullish_bias = 25
            confidence = "HIGH"
            
        elif is_distributing:
            # Selling at lows = capitulation
            # BUT we need to check OI AND whale positioning!
            phase = "CAPITULATION"
            phase_emoji = "🔶"
            
            # ═══════════════════════════════════════════════════════════════
            # OI Direction + Whale Positioning = TRUE signal
            # ═══════════════════════════════════════════════════════════════
            # OI↑ + Whales HIGH = Retail shorting, whales holding = BULLISH
            # OI↑ + Whales LOW = Smart shorts adding = BEARISH
            # OI↓ = Shorts covering = Potential reversal
            # ═══════════════════════════════════════════════════════════════
            
            if oi_change > 1.0:
                # OI RISING - need to check whale positioning (done in MASTER_RULES)
                phase_color = "#ffcc00"  # Yellow - context dependent
                phase_detail = "OI rising - check whale positioning"
                bullish_bias = 0  # Neutral here - MASTER_RULES decides based on whales
                confidence = "MEDIUM"
                warning = "OI↑ + Price↓: If Whales HIGH = squeeze fuel, If Whales LOW = bearish trap"
            elif oi_change < -1.0:
                # OI FALLING = Shorts covering = Potential reversal!
                phase_color = "#ff9500"  # Orange - watch for reversal
                phase_detail = "Shorts covering - watch for reversal"
                bullish_bias = 15  # Potential bounce
                confidence = "MEDIUM"
                warning = "✅ OI falling - shorts may be covering. Watch for reversal signal."
            else:
                # OI neutral
                phase_color = "#ff9500"
                phase_detail = "Panic selling - check whale positioning"
                bullish_bias = 5
                confidence = "LOW"
                warning = "Capitulation detected - whale positioning determines next move"
            
        else:
            # Neutral at lows = building a base
            phase = "RE-ACCUMULATION"
            phase_color = "#00d4ff"
            phase_emoji = "🔵"
            phase_detail = "Building base at support"
            bullish_bias = 5
            confidence = "LOW"
    
    # ═══════════════════════════════════════════════════════════════════════
    # MIDDLE POSITION (30-70%) - Mid-range
    # ═══════════════════════════════════════════════════════════════════════
    elif position_label == "MIDDLE":
        if is_accumulating:
            # Buying in middle = trend continuation
            phase = "MARKUP"
            phase_color = "#00ff88"
            phase_emoji = "📈"
            phase_detail = "Trend continuation"
            bullish_bias = 15
            confidence = "MEDIUM"
            
        elif is_distributing:
            # Some profit taking mid-trend
            phase = "PROFIT TAKING"
            phase_color = "#ffcc00"
            phase_emoji = "🟡"
            phase_detail = "Some profit taking"
            bullish_bias = -5
            confidence = "MEDIUM"
            warning = "Watch for trend weakness"
            
        else:
            # Neutral = consolidation
            phase = "CONSOLIDATION"
            phase_color = "#888"
            phase_emoji = "↔️"
            phase_detail = "Pause in trend"
            bullish_bias = 0
            confidence = "LOW"
    
    # ═══════════════════════════════════════════════════════════════════════
    # LATE POSITION (70-100%) - Top of range
    # ═══════════════════════════════════════════════════════════════════════
    elif position_label == "LATE":
        if is_accumulating:
            # ⚠️ DANGER: "Buying" at highs is often retail FOMO
            # Smart money SELLS into this buying
            phase = "FOMO / DIST RISK"
            phase_color = "#ff9500"
            phase_emoji = "⚠️"
            phase_detail = "Retail FOMO - distribution risk!"
            bullish_bias = -15
            confidence = "HIGH"
            warning = "Smart money often sells into retail buying at highs"
            
        elif is_distributing:
            # Classic distribution at highs
            phase = "DISTRIBUTION"
            phase_color = "#ff6b6b"
            phase_emoji = "🔴"
            phase_detail = "Smart money selling at highs"
            bullish_bias = -25
            confidence = "HIGH"
            
        else:
            # Neutral at highs = exhaustion
            phase = "EXHAUSTION"
            phase_color = "#ff6b6b"
            phase_emoji = "😴"
            phase_detail = "Trend losing momentum"
            bullish_bias = -10
            confidence = "MEDIUM"
            warning = "Late in move - higher risk entry"
    
    # ═══════════════════════════════════════════════════════════════════════
    # SPECIAL CASES: Use OI + Price for additional context
    # ═══════════════════════════════════════════════════════════════════════
    
    # New longs at early position = very bullish
    if position_label == "EARLY" and oi_change > 1.0 and price_change > 0:
        bullish_bias = min(bullish_bias + 10, 30)
        confidence = "HIGH"
        phase_detail += " + New institutional longs"
    
    # New shorts at late position = very bearish
    if position_label == "LATE" and oi_change > 1.0 and price_change < 0:
        bullish_bias = max(bullish_bias - 10, -30)
        confidence = "HIGH"
        phase_detail += " + New institutional shorts"
    
    return MoneyFlowContext(
        is_inflow=is_accumulating,
        is_outflow=is_distributing,
        position_pct=position_pct,
        position_label=position_label,
        phase=phase,
        phase_color=phase_color,
        phase_emoji=phase_emoji,
        phase_detail=phase_detail,
        bullish_bias=bullish_bias,
        confidence=confidence,
        warning=warning
    )


def get_smart_action(
    direction: str,
    direction_confidence: str,
    position_label: str,
    position_pct: float,
    ta_score: int,
    flow_context: MoneyFlowContext
) -> tuple:
    """
    ═══════════════════════════════════════════════════════════════════════════
    PREDICTIVE ACTION LOGIC
    ═══════════════════════════════════════════════════════════════════════════
    
    This is the KEY change from confirmatory to predictive:
    
    OLD (Confirmatory):
        Strong Direction + Early + Poor TA = "WAIT" (misses the move)
    
    NEW (Predictive):
        Strong Direction + Early = "ACCUMULATE" (catch the move early)
        Strong Direction + Late = "WAIT FOR PULLBACK" (don't chase)
    
    ═══════════════════════════════════════════════════════════════════════════
    """
    
    action = "WAIT"
    action_detail = ""
    score_modifier = 0
    
    # ═══════════════════════════════════════════════════════════════════════
    # BULLISH DIRECTION
    # ═══════════════════════════════════════════════════════════════════════
    if direction in ["BULLISH", "WEAK_BULLISH"]:
        
        if position_label == "EARLY":
            # EARLY + Bullish = Best setup - accumulate regardless of TA
            if direction_confidence == "HIGH":
                action = "ACCUMULATE"
                action_detail = "Strong institutional buying at lows"
                score_modifier = 15  # Boost score
            else:
                action = "EARLY ENTRY"
                action_detail = "Catching move early - scale in"
                score_modifier = 10
                
        elif position_label == "MIDDLE":
            # MIDDLE = OK but need decent TA
            if ta_score >= 50:
                action = "BUY"
                action_detail = "Trend continuation entry"
                score_modifier = 0
            else:
                action = "CAUTIOUS BUY"
                action_detail = "Mid-trend - use tight stop"
                score_modifier = -5
                
        elif position_label == "LATE":
            # LATE + Bullish = Don't chase!
            if flow_context.phase in ["FOMO / DIST RISK", "DISTRIBUTION"]:
                action = "AVOID"
                action_detail = "Distribution at highs - don't chase"
                score_modifier = -20
            else:
                action = "WAIT FOR PULLBACK"
                action_detail = "Late in move - wait for dip"
                score_modifier = -10
    
    # ═══════════════════════════════════════════════════════════════════════
    # BEARISH DIRECTION
    # ═══════════════════════════════════════════════════════════════════════
    elif direction == "BEARISH":
        
        if position_label == "LATE":  # High in range = early for shorts
            if direction_confidence == "HIGH":
                action = "SHORT"
                action_detail = "Strong institutional selling at highs"
                score_modifier = 15
            else:
                action = "EARLY SHORT"
                action_detail = "Catching downmove early"
                score_modifier = 10
                
        elif position_label == "MIDDLE":
            if ta_score >= 50:
                action = "SHORT"
                action_detail = "Downtrend continuation"
                score_modifier = 0
            else:
                action = "CAUTIOUS SHORT"
                action_detail = "Mid-trend - use tight stop"
                score_modifier = -5
                
        elif position_label == "EARLY":  # Low in range = late for shorts
            if flow_context.phase == "CAPITULATION":
                action = "AVOID SHORT"
                action_detail = "Capitulation at lows - potential bounce"
                score_modifier = -20
            else:
                action = "WAIT FOR RALLY"
                action_detail = "Late in downmove - wait for bounce to short"
                score_modifier = -10
    
    # ═══════════════════════════════════════════════════════════════════════
    # UNCERTAIN / NEUTRAL
    # ═══════════════════════════════════════════════════════════════════════
    else:
        if ta_score >= 65:
            action = "TA TRADE"
            action_detail = "No directional edge - trade TA only"
            score_modifier = 0
        else:
            action = "WAIT"
            action_detail = "No clear edge"
            score_modifier = -10
    
    return action, action_detail, score_modifier


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_position_quality(position_pct: float, direction: str) -> tuple:
    """
    Returns (quality_label, quality_color, quality_score_impact)
    
    For LONGS: Lower position = better
    For SHORTS: Higher position = better
    """
    if direction in ["BULLISH", "WEAK_BULLISH", "LONG"]:
        if position_pct <= 20:
            return ("OPTIMAL", "#00ff88", 10)
        elif position_pct <= 35:
            return ("GOOD", "#00d4aa", 5)
        elif position_pct <= 50:
            return ("OK", "#ffcc00", 0)
        elif position_pct <= 70:
            return ("LATE", "#ff9500", -10)
        else:
            return ("CHASING", "#ff6b6b", -20)
            
    elif direction in ["BEARISH", "SHORT"]:
        if position_pct >= 80:
            return ("OPTIMAL", "#00ff88", 10)
        elif position_pct >= 65:
            return ("GOOD", "#00d4aa", 5)
        elif position_pct >= 50:
            return ("OK", "#ffcc00", 0)
        elif position_pct >= 30:
            return ("LATE", "#ff9500", -10)
        else:
            return ("CHASING", "#ff6b6b", -20)
    
    return ("NEUTRAL", "#888", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYSIS SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
# 
# RULE: Higher TF = Direction, Lower TF = Timing
#
# Standard Practice: 4-6x ratio between timeframes
# Source: Professional trading education (Real Trading, Trade with Pros)
#
# Timeframe Hierarchy (4-6x ratio):
# ┌─────────────┬──────────────┬─────────────────┬───────┐
# │ Mode        │ Entry TF     │ Higher TF       │ Ratio │
# ├─────────────┼──────────────┼─────────────────┼───────┤
# │ Scalp       │ 1m           │ 5m              │ 5x    │
# │ Scalp       │ 5m           │ 15m             │ 3x    │
# │ DayTrade    │ 15m          │ 1h              │ 4x    │
# │ DayTrade    │ 1h           │ 4h              │ 4x    │
# │ Swing       │ 4h           │ 1d              │ 6x    │
# │ Swing       │ 1d           │ 1w              │ 7x    │
# │ Investment  │ 1w           │ 1M              │ 4x    │
# └─────────────┴──────────────┴─────────────────┴───────┘
# ═══════════════════════════════════════════════════════════════════════════════

# Timeframe hierarchy mapping (4-6x ratio - professional standard)
TIMEFRAME_HIERARCHY = {
    # Scalp mode
    '1m': '5m',      # 5x ratio
    '5m': '15m',     # 3x ratio
    # DayTrade mode
    '15m': '1h',     # 4x ratio
    '1h': '4h',      # 4x ratio
    # Swing mode
    '4h': '1d',      # 6x ratio
    '1d': '1w',      # 7x ratio
    # Investment mode
    '1w': '1M',      # ~4x ratio
    '1M': '1M',      # No higher for monthly
}

# Timeframe display names for tips
TIMEFRAME_NAMES = {
    '1m': '1-minute',
    '5m': '5-minute',
    '15m': '15-minute',
    '1h': '1-hour',
    '4h': '4-hour',
    '1d': 'Daily',
    '1w': 'Weekly',
    '1M': 'Monthly',
}


def get_higher_timeframe(current_tf: str) -> str:
    """Get the higher timeframe for context (4-6x ratio)"""
    return TIMEFRAME_HIERARCHY.get(current_tf, '4h')


def get_timeframe_name(tf: str) -> str:
    """Get human readable timeframe name"""
    return TIMEFRAME_NAMES.get(tf, tf)


@dataclass
class MultiTimeframeTip:
    """Multi-timeframe trading tip"""
    has_tip: bool
    tip_type: str           # 'PULLBACK_LONG', 'PULLBACK_SHORT', 'ALIGNED', 'CONFLICTING'
    tip_emoji: str
    tip_title: str
    tip_detail: str
    tip_color: str
    action: str
    higher_tf: str = ''     # The higher timeframe used
    higher_phase: str = ''  # The higher TF phase


def get_multi_timeframe_tip(
    current_tf: str,
    current_phase: str,
    higher_tf_phase: str = None,
    predictive_signal: str = None
) -> MultiTimeframeTip:
    """
    Generate trading tip when timeframes show different signals.
    
    RULE: Higher TF = Direction, Lower TF = Timing
    
    Examples:
    - 1H MARKUP + 15m CONSOLIDATION = Pullback opportunity (BUY THE DIP)
    - 1H DISTRIBUTION + 15m MARKUP = Dead cat bounce (DON'T BUY)
    - 1H MARKUP + 15m MARKUP = Aligned but maybe chasing
    """
    
    # Define bullish and bearish phases
    bullish_phases = ['ACCUMULATION', 'MARKUP', 'RE-ACCUMULATION']
    bearish_phases = ['DISTRIBUTION', 'FOMO / DIST RISK', 'EXHAUSTION', 'PROFIT TAKING']
    neutral_phases = ['CONSOLIDATION', 'CAPITULATION']
    
    # Get higher TF
    higher_tf = get_higher_timeframe(current_tf)
    higher_tf_name = get_timeframe_name(higher_tf)
    current_tf_name = get_timeframe_name(current_tf)
    
    # No higher TF data - check if predictive signal diverges from current phase
    if not higher_tf_phase:
        # Check if predictive signal is bullish but current phase is neutral/bearish
        if predictive_signal:
            pred_bullish = any(x in predictive_signal.upper() for x in ['LONG', 'BUY', 'ACCUMULATION'])
            pred_bearish = any(x in predictive_signal.upper() for x in ['SHORT', 'SELL', 'EXIT'])
            
            if pred_bullish and current_phase in neutral_phases:
                return MultiTimeframeTip(
                    has_tip=True,
                    tip_type='DIVERGENCE_BULLISH',
                    tip_emoji='🎯',
                    tip_title='Accumulation Detected',
                    tip_detail=f'{current_tf_name} shows {current_phase} but whales are accumulating. Check {higher_tf_name} for trend!',
                    tip_color='#00ff88',
                    action=f'If {higher_tf_name} bullish → BUY THE DIP. If {higher_tf_name} bearish → Wait for confirmation.',
                    higher_tf=higher_tf,
                    higher_phase=''
                )
            elif pred_bullish and current_phase in bearish_phases:
                return MultiTimeframeTip(
                    has_tip=True,
                    tip_type='REVERSAL_BULLISH',
                    tip_emoji='🔄',
                    tip_title='Potential Reversal',
                    tip_detail=f'{current_tf_name} shows {current_phase} (bearish) but whales accumulating. Reversal forming?',
                    tip_color='#00d4aa',
                    action=f'Check {higher_tf_name}. Small position OK with tight stop. Add when confirmed.',
                    higher_tf=higher_tf,
                    higher_phase=''
                )
            elif pred_bearish and current_phase in bullish_phases:
                return MultiTimeframeTip(
                    has_tip=True,
                    tip_type='DIVERGENCE_BEARISH',
                    tip_emoji='⚠️',
                    tip_title='Distribution Warning',
                    tip_detail=f'{current_tf_name} shows {current_phase} but smart money distributing. Top forming!',
                    tip_color='#ff9500',
                    action=f'Check {higher_tf_name}. If also distributing → EXIT LONGS. Trail stops tight.',
                    higher_tf=higher_tf,
                    higher_phase=''
                )
            elif pred_bearish and current_phase in neutral_phases:
                return MultiTimeframeTip(
                    has_tip=True,
                    tip_type='BREAKDOWN_WARNING',
                    tip_emoji='📉',
                    tip_title='Breakdown Risk',
                    tip_detail=f'{current_tf_name} {current_phase} but predictive bearish. Breakdown possible!',
                    tip_color='#ff6b6b',
                    action=f'Avoid longs. Check {higher_tf_name} - if bearish → SHORT on support break.',
                    higher_tf=higher_tf,
                    higher_phase=''
                )
        
        return MultiTimeframeTip(
            has_tip=False,
            tip_type='NONE',
            tip_emoji='',
            tip_title='',
            tip_detail='',
            tip_color='#888',
            action='',
            higher_tf=higher_tf,
            higher_phase=''
        )
    
    # Have higher TF data - check for pullback opportunities
    higher_bullish = higher_tf_phase in bullish_phases
    higher_bearish = higher_tf_phase in bearish_phases
    higher_neutral = higher_tf_phase in neutral_phases
    current_bullish = current_phase in bullish_phases
    current_bearish = current_phase in bearish_phases
    current_neutral = current_phase in neutral_phases
    
    higher_tf_name = get_timeframe_name(higher_tf) if higher_tf else 'Higher TF'
    current_tf_name = get_timeframe_name(current_tf)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PULLBACK OPPORTUNITIES (Best setups!)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Higher TF bullish + Lower TF pausing/correcting = BUY THE DIP
    if higher_bullish and (current_neutral or current_bearish):
        return MultiTimeframeTip(
            has_tip=True,
            tip_type='PULLBACK_LONG',
            tip_emoji='🛒',
            tip_title=f'🎯 Pullback in Uptrend',
            tip_detail=f'{higher_tf_name} shows {higher_tf_phase} (bullish). {current_tf_name} shows {current_phase} (pause). This is the DIP!',
            tip_color='#00ff88',
            action=f'BUY THE DIP! Enter when {current_tf_name} shows accumulation, or scale in now. Stop below recent low.',
            higher_tf=higher_tf,
            higher_phase=higher_tf_phase
        )
    
    # Higher TF bearish + Lower TF rallying = SELL THE RALLY
    if higher_bearish and (current_neutral or current_bullish):
        return MultiTimeframeTip(
            has_tip=True,
            tip_type='PULLBACK_SHORT',
            tip_emoji='📉',
            tip_title=f'⚠️ Rally in Downtrend (Trap!)',
            tip_detail=f'{higher_tf_name} shows {higher_tf_phase} (bearish). {current_tf_name} shows {current_phase} (dead cat bounce).',
            tip_color='#ff6b6b',
            action=f'SELL THE RALLY! Short when {current_tf_name} shows distribution. Stop above recent high.',
            higher_tf=higher_tf,
            higher_phase=higher_tf_phase
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ALIGNED SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if higher_bullish and current_bullish:
        return MultiTimeframeTip(
            has_tip=True,
            tip_type='ALIGNED_BULLISH',
            tip_emoji='🚀',
            tip_title='All Timeframes Bullish',
            tip_detail=f'{higher_tf_name} ({higher_tf_phase}) + {current_tf_name} ({current_phase}) = Strong uptrend!',
            tip_color='#00d4aa',
            action=f'Trend confirmed but might be chasing. Wait for {current_tf_name} pause for better entry.',
            higher_tf=higher_tf,
            higher_phase=higher_tf_phase
        )
    
    if higher_bearish and current_bearish:
        return MultiTimeframeTip(
            has_tip=True,
            tip_type='ALIGNED_BEARISH',
            tip_emoji='🔻',
            tip_title='All Timeframes Bearish',
            tip_detail=f'{higher_tf_name} ({higher_tf_phase}) + {current_tf_name} ({current_phase}) = Strong downtrend!',
            tip_color='#ff6b6b',
            action=f'Downtrend confirmed. Wait for {current_tf_name} rally to short, or stay out.',
            higher_tf=higher_tf,
            higher_phase=higher_tf_phase
        )
    
    # Higher TF neutral
    if higher_neutral:
        return MultiTimeframeTip(
            has_tip=True,
            tip_type='HIGHER_NEUTRAL',
            tip_emoji='↔️',
            tip_title=f'{higher_tf_name} is Ranging',
            tip_detail=f'{higher_tf_name} shows {higher_tf_phase} (no trend). Range-bound market.',
            tip_color='#ffcc00',
            action=f'Trade the range: Buy near lows, sell near highs. Or wait for breakout.',
            higher_tf=higher_tf,
            higher_phase=higher_tf_phase
        )
    
    # Default - no clear tip
    return MultiTimeframeTip(
        has_tip=False,
        tip_type='NEUTRAL',
        tip_emoji='',
        tip_title='',
        tip_detail='',
        tip_color='#888',
        action='',
        higher_tf=higher_tf if higher_tf else '',
        higher_phase=higher_tf_phase if higher_tf_phase else ''
    )
