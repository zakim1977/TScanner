"""
═══════════════════════════════════════════════════════════════════════════════
                    INVESTORIQ MASTER TRADING RULES
                    ================================
                    
    THIS IS THE SINGLE SOURCE OF TRUTH FOR ALL TRADING DECISIONS
    
    NO OTHER FILE should make trading decisions. Everything calls this.
    
    ONE FUNCTION: get_phase_position_signal() → The Matrix
    ONE DECISION: Phase (ML) + Position (%) = LONG / SHORT / WAIT
    ONE ANSWER: No conflicting logic, no whale overrides
    
═══════════════════════════════════════════════════════════════════════════════

CORE PHILOSOPHY (Updated):
--------------------------
1. PHASE (from ML) tells us WHERE in the market cycle we are
2. POSITION (from structure) tells us WHERE in the current move we are
3. Phase + Position matrix = FINAL DECISION (LONG/SHORT/WAIT)
4. Whale data is DISPLAY ONLY - no decision override
   - Why? Whale % is a snapshot with no entry time information
   - 63% long could be fresh (bullish) or about to exit (distribution)
   - The PHASE already tells us what's happening

THE MATRIX:
-----------
| Phase          | EARLY (0-35%) | MIDDLE (35-65%) | LATE (65-100%) |
|----------------|---------------|-----------------|----------------|
| ACCUMULATION   | 🟢 LONG       | 🟢 LONG         | ⏳ WAIT        |
| MARKUP         | 🟢 LONG       | 🟢 LONG         | ⏳ WAIT        |
| DISTRIBUTION   | ⏳ WAIT       | 🔴 SHORT        | 🔴 SHORT       |
| MARKDOWN       | ⏳ WAIT       | 🔴 SHORT        | 🔴 SHORT       |

GOLDEN RULES (Simplified):
--------------------------
Rule 1: Trust the PHASE - ML learned from 25,000+ samples what each phase looks like
Rule 2: Trust the POSITION - Price structure doesn't lie
Rule 3: Don't override with whale % - we don't know when they entered
Rule 4: EARLY in LONG phase = Best long entry
Rule 5: LATE in SHORT phase = Best short entry
Rule 6: LATE in LONG phase = WAIT (don't chase)
Rule 7: EARLY in SHORT phase = WAIT (don't catch falling knife)

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS - Clear naming for all states
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    LEAN_LONG = "LEAN_LONG"
    NEUTRAL = "NEUTRAL"
    LEAN_SHORT = "LEAN_SHORT"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG_SHORT"


class Action(Enum):
    STRONG_LONG = "STRONG_LONG"       # High conviction long
    LONG_SETUP = "LONG_SETUP"         # Good long setup
    BUILDING_LONG = "BUILDING_LONG"   # Long forming
    MONITOR_LONG = "MONITOR_LONG"     # Watch for long
    CAUTION_LONG = "CAUTION_LONG"     # Long risky - retail trap
    WAIT = "WAIT"                     # No trade
    CAUTION_SHORT = "CAUTION_SHORT"   # Short risky
    MONITOR_SHORT = "MONITOR_SHORT"   # Watch for short
    BUILDING_SHORT = "BUILDING_SHORT" # Short forming
    SHORT_SETUP = "SHORT_SETUP"       # Good short setup
    STRONG_SHORT = "STRONG_SHORT"     # High conviction short


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Position(Enum):
    EARLY = "EARLY"     # 0-35% - Near lows (good for longs)
    MIDDLE = "MIDDLE"   # 35-65% - Mid range (wait)
    LATE = "LATE"       # 65-100% - Near highs (good for shorts)


class OISignal(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"   # OI↑ Price↑ = New longs
    BULLISH = "BULLISH"                  # OI↑ Price↓ = Accumulation
    BEARISH = "BEARISH"                  # OI↓ Price↑ = Shorts closing
    STRONG_BEARISH = "STRONG_BEARISH"   # OI↓ Price↓ = Longs exiting
    NEUTRAL = "NEUTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS - All numbers in ONE place
# ═══════════════════════════════════════════════════════════════════════════════

class T:
    """All thresholds - NEVER use magic numbers elsewhere"""
    
    # Whale/Retail Positioning
    EXTREME_BULLISH = 75
    STRONG_BULLISH = 70
    BULLISH = 65
    LEAN_BULLISH = 60
    NEUTRAL = 50
    LEAN_BEARISH = 40
    BEARISH = 35
    STRONG_BEARISH = 30
    EXTREME_BEARISH = 25
    
    # Position in Range
    EARLY_MAX = 35      # 0-35% = EARLY
    LATE_MIN = 65       # 65-100% = LATE
    
    # OI Thresholds
    OI_STRONG = 3.0     # Strong OI change
    OI_MODERATE = 1.0   # Moderate OI change
    
    # Price Thresholds
    PRICE_SIGNIFICANT = 1.0  # Significant price move


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE + POSITION DECISION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is the CORE DECISION LOGIC that combines:
# - ML Phase (where in market cycle)
# - Move Position (where in price range)
# To give a clear LONG / SHORT / WAIT recommendation
#
# ═══════════════════════════════════════════════════════════════════════════════

def get_phase_position_signal(phase: str, position_pct: float) -> dict:
    """
    Combines ML Phase + Position to give clear trading signal.
    
    Args:
        phase: Market phase (CAPITULATION, ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN, etc.)
        position_pct: Position in range (0-100, where 0=lows, 100=highs)
    
    Returns:
        dict with: action, color, emoji, reason, long_ok, short_ok
    """
    phase_upper = phase.upper().replace(' ', '_') if phase else 'UNKNOWN'
    
    # Determine position bucket
    if position_pct <= 35:
        pos = 'EARLY'  # At lows
    elif position_pct >= 65:
        pos = 'LATE'   # At highs
    else:
        pos = 'MIDDLE'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # THE DECISION MATRIX
    # ═══════════════════════════════════════════════════════════════════════════
    # Format: (action, color, emoji, reason, long_ok, short_ok)
    
    matrix = {
        # CAPITULATION: Panic phase - LATE is best for longs (panic ending)
        ('CAPITULATION', 'EARLY'): ('WAIT', '#ffcc00', '🟡', 'Panic starting - wait for exhaustion', False, False),
        ('CAPITULATION', 'MIDDLE'): ('LONG', '#00ff88', '🟢', 'Panic ongoing - watch for reversal', True, False),
        ('CAPITULATION', 'LATE'): ('LONG', '#00ff88', '🟢', 'Panic ending - potential bottom!', True, False),
        
        # ACCUMULATION: Smart money buying - EARLY/MID best for longs
        ('ACCUMULATION', 'EARLY'): ('LONG', '#00ff88', '🟢', 'Smart money buying at lows!', True, False),
        ('ACCUMULATION', 'MIDDLE'): ('LONG', '#00ff88', '🟢', 'Accumulation ongoing', True, False),
        ('ACCUMULATION', 'LATE'): ('WAIT', '#ffcc00', '🟡', 'Accumulation ending - breakout soon', True, False),
        
        # RE_ACCUMULATION: Pullback in uptrend
        ('RE_ACCUMULATION', 'EARLY'): ('LONG', '#00ff88', '🟢', 'Pullback buy opportunity!', True, False),
        ('RE_ACCUMULATION', 'MIDDLE'): ('WAIT', '#ffcc00', '🟡', 'Mid-pullback - wait', False, False),
        ('RE_ACCUMULATION', 'LATE'): ('WAIT', '#ff6b6b', '🔴', 'Pullback over - may have missed', False, False),
        
        # MARKUP: Uptrend phase - ride the trend!
        # In a confirmed uptrend, only LATE entries are risky
        ('MARKUP', 'EARLY'): ('LONG', '#00ff88', '🟢', 'Early in uptrend - best entry!', True, False),
        ('MARKUP', 'MIDDLE'): ('LONG', '#00ff88', '🟢', 'Trend confirmed - ride it!', True, False),
        ('MARKUP', 'LATE'): ('WAIT', '#ffcc00', '🟡', 'Late in trend - wait for pullback', False, False),
        
        # DISTRIBUTION: Smart money selling - BAD for longs, GOOD for shorts
        ('DISTRIBUTION', 'EARLY'): ('WAIT', '#ff6b6b', '🔴', 'Distribution at lows - false bottom', False, False),
        ('DISTRIBUTION', 'MIDDLE'): ('SHORT', '#ff6b6b', '🔴', 'Smart money selling', False, True),
        ('DISTRIBUTION', 'LATE'): ('SHORT', '#ff6b6b', '🔴', 'Top forming - short setup!', False, True),
        
        # PROFIT_TAKING: Taking profits - cautious
        ('PROFIT_TAKING', 'EARLY'): ('WAIT', '#ff6b6b', '🔴', 'Profit taking - weakness', False, False),
        ('PROFIT_TAKING', 'MIDDLE'): ('WAIT', '#ff6b6b', '🔴', 'Selling pressure', False, True),
        ('PROFIT_TAKING', 'LATE'): ('SHORT', '#ff6b6b', '🔴', 'Profit taking at highs', False, True),
        
        # MARKDOWN: Downtrend phase - trend is your friend
        # In a confirmed downtrend, only EARLY (catching falling knife) is risky
        ('MARKDOWN', 'EARLY'): ('WAIT', '#ff6b6b', '🔴', 'Falling knife - do NOT catch!', False, False),
        ('MARKDOWN', 'MIDDLE'): ('SHORT', '#ff6b6b', '🔴', 'Downtrend confirmed - short it!', False, True),
        ('MARKDOWN', 'LATE'): ('SHORT', '#ff6b6b', '🔴', 'Bear rally to resistance - best short!', False, True),
        
        # FOMO / DIST RISK: Late stage distribution, retail FOMO
        ('FOMO_/_DIST_RISK', 'EARLY'): ('WAIT', '#ff9500', '⚠️', 'FOMO zone - risky for longs', False, False),
        ('FOMO_/_DIST_RISK', 'MIDDLE'): ('WAIT', '#ff6b6b', '🔴', 'Distribution risk - smart money exiting', False, True),
        ('FOMO_/_DIST_RISK', 'LATE'): ('SHORT', '#ff6b6b', '🔴', 'Exit liquidity trap - SHORT setup!', False, True),
        
        # EXHAUSTION: Trend exhaustion
        ('EXHAUSTION', 'EARLY'): ('WAIT', '#ffcc00', '🟡', 'Trend exhaustion starting', False, False),
        ('EXHAUSTION', 'MIDDLE'): ('WAIT', '#ff9500', '⚠️', 'Exhaustion - reversal possible', False, False),
        ('EXHAUSTION', 'LATE'): ('WAIT', '#ff6b6b', '🔴', 'Trend exhausted - expect reversal', False, False),
        
        # CONSOLIDATION: Sideways, waiting for breakout
        ('CONSOLIDATION', 'EARLY'): ('WAIT', '#888', '⚪', 'Consolidating near lows', False, False),
        ('CONSOLIDATION', 'MIDDLE'): ('WAIT', '#888', '⚪', 'Range-bound - wait for breakout', False, False),
        ('CONSOLIDATION', 'LATE'): ('WAIT', '#888', '⚪', 'Consolidating near highs', False, False),
        
        # NEUTRAL: No clear phase
        ('NEUTRAL', 'EARLY'): ('WAIT', '#888', '⚪', 'No clear trend', False, False),
        ('NEUTRAL', 'MIDDLE'): ('WAIT', '#888', '⚪', 'Market uncertain', False, False),
        ('NEUTRAL', 'LATE'): ('WAIT', '#888', '⚪', 'No clear direction', False, False),
    }
    
    # Get signal from matrix or return default
    key = (phase_upper, pos)
    if key in matrix:
        action, color, emoji, reason, long_ok, short_ok = matrix[key]
    else:
        # Default for unknown phases
        action, color, emoji, reason, long_ok, short_ok = ('WAIT', '#888', '⚪', f'Unknown phase: {phase}', False, False)
    
    return {
        'action': action,           # LONG, SHORT, or WAIT
        'color': color,             # For display
        'emoji': emoji,             # 🟢🟡🔴
        'reason': reason,           # Explanation
        'tip': reason,              # Alias for reason (for compatibility)
        'long_ok': long_ok,         # Is long valid?
        'short_ok': short_ok,       # Is short valid?
        'phase': phase_upper,       # Normalized phase
        'position': pos,            # EARLY/MIDDLE/LATE
        'position_pct': position_pct
    }


def get_final_action(phase: str, position_pct: float, whale_pct: float = 50, retail_pct: float = 50) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════
    THE SINGLE SOURCE OF TRUTH FOR ALL TRADING DECISIONS
    ═══════════════════════════════════════════════════════════════════════════
    
    Decision is based ONLY on:
    - Phase (from ML): WHERE in the market cycle
    - Position %: WHERE in the current move (0-100)
    
    Whale data is for DISPLAY ONLY - not a decision override.
    Why? Whale % is a snapshot with no entry time. 63% long could mean:
    - Fresh entry (bullish) OR
    - Legacy position about to exit (bearish during distribution)
    
    The PHASE tells us what's actually happening. Trust the phase.
    
    Returns:
        dict with action, reason, color, emoji, phase_signal, and display info
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE + POSITION = THE ONLY DECISION
    # ═══════════════════════════════════════════════════════════════════════════
    phase_signal = get_phase_position_signal(phase, position_pct)
    
    # The phase+position matrix gives us the action directly
    final_action = phase_signal['action']  # LONG, SHORT, or WAIT
    final_reason = phase_signal['reason']
    final_color = phase_signal['color']
    final_emoji = phase_signal['emoji']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WHALE DATA = DISPLAY INFO ONLY (not decision override!)
    # ═══════════════════════════════════════════════════════════════════════════
    whale_confirms_long = whale_pct >= 55
    whale_confirms_short = whale_pct <= 45
    retail_trap = retail_pct > whale_pct + 10 and retail_pct > 55
    
    # Add whale context to reason (informational only)
    whale_note = ""
    if final_action == 'LONG':
        if whale_confirms_long:
            whale_note = f" (Whales: {whale_pct:.0f}% ✓)"
        else:
            whale_note = f" (Whales: {whale_pct:.0f}% - monitor)"
    elif final_action == 'SHORT':
        if whale_confirms_short:
            whale_note = f" (Whales: {whale_pct:.0f}% ✓)"
        else:
            whale_note = f" (Whales: {whale_pct:.0f}% - monitor)"
    
    return {
        'action': final_action,
        'reason': final_reason + whale_note,
        'color': final_color,
        'emoji': final_emoji,
        'confirmed': True,  # Phase+Position is always the confirmed decision
        'phase_signal': phase_signal,
        # Display info (not used for decisions)
        'whale_confirms': whale_confirms_long if final_action == 'LONG' else whale_confirms_short if final_action == 'SHORT' else False,
        'retail_trap': retail_trap,
        'whale_pct': whale_pct,
        'retail_pct': retail_pct
    }


def get_display_signal(
    phase: str,
    position_pct: float,
    whale_pct: float = 50,
    retail_pct: float = 50,
    ml_direction: str = None,
    ml_confidence: float = None
) -> dict:
    """
    🎯 UNIFIED DISPLAY SIGNAL - SINGLE SOURCE OF TRUTH
    
    Call this from ALL views (Single Analysis, Scanner, Market Pulse, Trade Monitor)
    to get consistent signal display information.
    
    Args:
        phase: Market phase from ML (MARKUP, ACCUMULATION, etc.)
        position_pct: Position in range (0-100)
        whale_pct: Whale positioning percentage
        retail_pct: Retail positioning percentage
        ml_direction: Raw ML direction (LONG/SHORT/WAIT) - for display only
        ml_confidence: ML confidence percentage - for display only
    
    Returns:
        dict with all display info:
        - action: STRONG_LONG, LONG, WAIT, SHORT, STRONG_SHORT
        - direction: LONG, SHORT, WAIT (simplified)
        - reason: Explanation text
        - tip: Short tip for display
        - color: Hex color for display
        - emoji: Display emoji
        - confirmed: Whether whale confirms
        - position_label: EARLY, MIDDLE, LATE
        - setup_type: IDEAL_LONG, CONFIDENT_LONG, etc.
        - setup_text: Display text for big circle
        - setup_color: Color for big circle
        - ml_conflict: Whether ML disagrees with Phase+Position
        - ml_info: ML display info (if available)
    """
    # Get Phase+Position signal (THE source of truth)
    final_signal = get_final_action(phase, position_pct, whale_pct, retail_pct)
    phase_signal = final_signal['phase_signal']
    
    action = final_signal['action']
    confirmed = final_signal['confirmed']
    reason = final_signal['reason']
    color = final_signal['color']
    emoji = final_signal['emoji']
    
    # Determine simplified direction for filtering
    if action in ['STRONG_LONG', 'LONG', 'CAUTION_LONG']:
        direction = 'LONG'
    elif action in ['STRONG_SHORT', 'SHORT']:
        direction = 'SHORT'
    else:
        direction = 'WAIT'
    
    # Position label
    if position_pct <= 35:
        position_label = 'EARLY'
    elif position_pct >= 65:
        position_label = 'LATE'
    else:
        position_label = 'MIDDLE'
    
    # Determine setup_type and setup_text based on action + confirmation
    if action == 'STRONG_LONG':
        if position_label == 'EARLY':
            setup_type = 'IDEAL_LONG'
            setup_text = '🎯 IDEAL LONG'
            setup_color = '#ffd700'  # Gold
        else:
            setup_type = 'CONFIDENT_LONG'
            setup_text = '🟢 STRONG LONG'
            setup_color = '#00ff88'
    elif action == 'STRONG_SHORT':
        if position_label == 'LATE':
            setup_type = 'IDEAL_SHORT'
            setup_text = '🎯 IDEAL SHORT'
            setup_color = '#ffd700'  # Gold
        else:
            setup_type = 'CONFIDENT_SHORT'
            setup_text = '🔴 STRONG SHORT'
            setup_color = '#ff6b6b'
    elif action == 'CAUTION_LONG':
        setup_type = 'CAUTION'
        setup_text = '⚠️ CAUTION LONG'
        setup_color = '#ffcc00'
    elif action == 'LONG':
        setup_type = 'WEAK_LONG'
        setup_text = '🟢 LONG'
        setup_color = '#00ff88'
    elif action == 'SHORT':
        setup_type = 'WEAK_SHORT'
        setup_text = '🔴 SHORT'
        setup_color = '#ff6b6b'
    else:  # WAIT
        setup_type = 'WAIT'
        setup_text = '⏳ WAIT'
        setup_color = '#ffcc00'
    
    # Check for ML conflict
    ml_conflict = False
    ml_info = None
    if ml_direction and ml_confidence:
        ml_info = {
            'direction': ml_direction,
            'confidence': ml_confidence,
            'color': '#00ff88' if ml_direction == 'LONG' else '#ff6b6b' if ml_direction == 'SHORT' else '#ffcc00'
        }
        
        # Check if ML disagrees with Phase+Position
        if ml_direction != direction:
            ml_conflict = True
            # If strong ML disagrees, add warning but DON'T override
            if ml_confidence >= 70:
                setup_text = f'⚠️ {setup_text}'  # Add warning to existing text
    
    return {
        # Core signal
        'action': action,
        'direction': direction,
        'reason': reason,
        'tip': phase_signal.get('tip', reason),
        'color': color,
        'emoji': emoji,
        'confirmed': confirmed,
        
        # Position info
        'position_label': position_label,
        'position_pct': position_pct,
        
        # Display info for big circle
        'setup_type': setup_type,
        'setup_text': setup_text,
        'setup_color': setup_color,
        
        # ML info (for conflict display)
        'ml_conflict': ml_conflict,
        'ml_info': ml_info,
        
        # Raw phase signal for reference
        'phase_signal': phase_signal,
        'final_signal': final_signal
    }


def generate_full_story(
    # ML Signals
    ml_signals: dict = None,  # {continue_up: %, continue_down: %, fakeout_up: %, fakeout_down: %, vol_expansion: %}
    f1_scores: dict = None,   # {continue_up: 0.70, continue_down: 0.65, ...}
    ml_direction: str = None,
    ml_confidence: float = None,
    
    # Phase + Position
    phase: str = None,
    position_pct: float = 50,
    
    # Whale Data
    whale_pct: float = 50,
    retail_pct: float = 50,
    oi_change: float = 0,
    price_change: float = 0,
    
    # NEW: Whale Delta (historical change)
    whale_delta: float = None,
    retail_delta: float = None,
    delta_lookback: str = '24h',
    whale_confirms: bool = None,  # True = confirms final_action, False = conflicts
    
    # TA
    ta_score: float = 50,
    structure: str = None,
    
    # Final Decision
    final_action: str = None,
    final_reason: str = None,
    
    # NEW: Trading Mode + Market Type (for correct label display)
    trading_mode: str = 'daytrade',  # scalp, daytrade, swing, investment
    market_type: str = 'crypto',      # crypto, stock, etf
) -> dict:
    """
    🎯 GENERATE FULL EDUCATIONAL STORY
    
    This is the heart of InvestorIQ - explaining WHY we recommend what we recommend.
    Returns a comprehensive narrative that walks through:
    1. ML predictions and their reliability (F1 scores)
    2. Market phase analysis
    3. Position timing analysis
    4. Whale/institutional behavior
    5. How everything combines for the final decision
    
    Returns:
        dict with:
        - story_html: Full HTML story
        - sections: Individual story sections
        - summary: One-line summary
        - confidence: Overall confidence in the recommendation
    """
    
    sections = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📊 SECTION 1: ML PREDICTIONS STORY (MODE-AWARE!)
    # ═══════════════════════════════════════════════════════════════════════════
    ml_story = ""
    ml_summary = ""
    
    # Normalize mode
    mode_lower = (trading_mode or 'daytrade').lower().replace(' ', '_').replace('day_trade', 'daytrade')
    is_stock_swing = mode_lower == 'swing' and market_type in ['stock', 'etf']
    
    def f1_reliability(score):
        if score >= 0.75: return ("🟢 HIGHLY RELIABLE", "#00ff88")
        elif score >= 0.65: return ("🟡 RELIABLE", "#ffcc00")
        elif score >= 0.55: return ("🟠 MODERATE", "#ff9500")
        else: return ("🔴 LOW RELIABILITY", "#ff6b6b")
    
    if ml_signals:
        f1 = f1_scores or {}
        
        # ═══════════════════════════════════════════════════════════════════════
        # STOCK SWING MODE - Quality + Mean Reversion Labels
        # ═══════════════════════════════════════════════════════════════════════
        if is_stock_swing:
            setup_high = ml_signals.get('setup_quality_high', 0) or 0
            setup_low = ml_signals.get('setup_quality_low', 0) or 0
            mr_long = ml_signals.get('mean_reversion_long', 0) or 0
            mr_short = ml_signals.get('mean_reversion_short', 0) or 0
            breakout = ml_signals.get('breakout_valid', 0) or 0
            vol_exp = ml_signals.get('volatility_expansion', 0) or 0
            
            f1_setup_high = f1.get('setup_quality_high', 0.5)
            f1_setup_low = f1.get('setup_quality_low', 0.5)
            f1_mr_long = f1.get('mean_reversion_long', 0.5)
            f1_mr_short = f1.get('mean_reversion_short', 0.5)
            f1_breakout = f1.get('breakout_valid', 0.5)
            f1_vol = f1.get('volatility_expansion', 0.5)
            
            ml_story = (
                "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #a855f7;'>"
                "<div style='color: #a855f7; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🤖 STEP 1: What Does ML See? (Stock Swing)</div>"
                "<div style='color: #ccc; line-height: 1.8; margin-bottom: 15px;'>The ML model analyzes setup quality and mean reversion opportunities:</div>"
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                # Setup Quality High
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if setup_high > 30 else '#333'};'>"
                f"<div style='color: #00ff88; font-weight: bold;'>✅ Setup Quality HIGH</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{setup_high:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_setup_high:.0%} {f1_reliability(f1_setup_high)[0]}</div></div>"
                # Setup Quality Low  
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if setup_low > 30 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>❌ Setup Quality LOW</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{setup_low:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_setup_low:.0%} {f1_reliability(f1_setup_low)[0]}</div></div>"
                # Mean Reversion Long
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if mr_long > 20 else '#333'};'>"
                f"<div style='color: #00ff88; font-weight: bold;'>📈 Mean Rev LONG</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{mr_long:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_mr_long:.0%} {f1_reliability(f1_mr_long)[0]}</div></div>"
                # Mean Reversion Short
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if mr_short > 20 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>📉 Mean Rev SHORT</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{mr_short:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_mr_short:.0%} {f1_reliability(f1_mr_short)[0]}</div></div>"
                "</div>"
            )
            
            # Second row with Breakout and Vol Expansion
            ml_story += (
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                f"<div style='flex: 1; min-width: 140px; background: #1a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ffcc00' if breakout > 20 else '#333'};'>"
                f"<div style='color: #ffcc00; font-weight: bold;'>🚀 Breakout Valid</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{breakout:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_breakout:.0%} {f1_reliability(f1_breakout)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #0a0a1a; padding: 10px; border-radius: 8px; border: 1px solid {'#3b82f6' if vol_exp > 30 else '#333'};'>"
                f"<div style='color: #3b82f6; font-weight: bold;'>💥 Vol Expansion</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{vol_exp:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_vol:.0%} {f1_reliability(f1_vol)[0]}</div></div>"
                "</div>"
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔗 CONNECTING NARRATIVE - How these signals relate (Stock Swing)
            # ═══════════════════════════════════════════════════════════════════
            narrative_parts = []
            
            # 1. Setup Quality comparison - this is critical for R:R
            quality_diff = setup_high - setup_low
            if quality_diff > 15:
                narrative_parts.append(f"<span style='color: #00ff88;'>Setup Quality HIGH ({setup_high:.1f}%) beats LOW ({setup_low:.1f}%) by {quality_diff:.0f} points</span> — this means trades taken now have good odds of hitting their profit target before the stop loss.")
            elif quality_diff < -15:
                narrative_parts.append(f"<span style='color: #ff6b6b;'>Setup Quality LOW ({setup_low:.1f}%) beats HIGH ({setup_high:.1f}%) by {abs(quality_diff):.0f} points</span> — this means trades taken now will likely hit the stop loss before the profit target. Bad risk/reward.")
            elif abs(quality_diff) < 5:
                narrative_parts.append(f"Setup Quality HIGH ({setup_high:.1f}%) and LOW ({setup_low:.1f}%) are nearly equal — the model can't predict whether your trade will hit TP or SL first. It's a coin flip.")
            else:
                better = "HIGH" if quality_diff > 0 else "LOW"
                better_color = "#00ff88" if quality_diff > 0 else "#ff6b6b"
                narrative_parts.append(f"Setup Quality <span style='color: {better_color};'>{better}</span> has a slight edge, but the difference is small. Be selective with entries.")
            
            # 2. Mean Reversion - directional signal
            # Check for meaningful signal OR significant imbalance between LONG and SHORT
            mr_max = max(mr_long, mr_short)
            mr_min = max(min(mr_long, mr_short), 0.01)  # Avoid division by zero
            mr_ratio = mr_max / mr_min if mr_min > 0 else 999
            
            if mr_long > 10 or mr_short > 10 or (mr_ratio > 10 and mr_max > 5):
                # One side is significantly dominant
                if mr_short > mr_long and (mr_short > 10 or mr_ratio > 10):
                    if mr_short > 25:
                        narrative_parts.append(f"<span style='color: #ff6b6b;'>Mean Reversion SHORT ({mr_short:.1f}%) is elevated</span> — the stock appears OVERBOUGHT. When prices rise too far too fast, they often pull back. Consider taking profits or waiting for a dip.")
                    else:
                        narrative_parts.append(f"<span style='color: #ff6b6b;'>Mean Reversion leans SHORT ({mr_short:.1f}% vs LONG {mr_long:.1f}%)</span> — there's a modest overbought signal. Not strong enough for a high-conviction SHORT, but the bias is bearish.")
                elif mr_long > mr_short and (mr_long > 10 or mr_ratio > 10):
                    if mr_long > 25:
                        narrative_parts.append(f"<span style='color: #00ff88;'>Mean Reversion LONG ({mr_long:.1f}%) is elevated</span> — the stock appears OVERSOLD. When prices drop too far too fast, they often bounce back. This could be a buy-the-dip opportunity.")
                    else:
                        narrative_parts.append(f"<span style='color: #00ff88;'>Mean Reversion leans LONG ({mr_long:.1f}% vs SHORT {mr_short:.1f}%)</span> — there's a modest oversold signal. Not strong enough for a high-conviction LONG, but the bias is bullish.")
                elif mr_long > 8 and mr_short > 8:
                    narrative_parts.append(f"Both Mean Reversion signals are active (LONG: {mr_long:.1f}%, SHORT: {mr_short:.1f}%) — the stock is volatile. Big moves in either direction are possible.")
            else:
                narrative_parts.append(f"Mean Reversion signals are quiet (LONG: {mr_long:.1f}%, SHORT: {mr_short:.1f}%) — the stock isn't at an extreme. No oversold bounce or overbought fade expected.")
            
            # 3. Breakout and Volatility context
            if breakout > 25:
                narrative_parts.append(f"<span style='color: #ffcc00;'>Breakout Valid ({breakout:.1f}%) is elevated</span> — if price is near a key level, a breakout through that level is likely to follow through rather than fail.")
            elif vol_exp > 35:
                narrative_parts.append(f"<span style='color: #3b82f6;'>Volatility Expansion ({vol_exp:.1f}%) is high</span> — expect bigger than normal price swings. Adjust your position size down to manage risk.")
            elif vol_exp < 10 and breakout < 10:
                narrative_parts.append(f"Breakout ({breakout:.1f}%) and Volatility ({vol_exp:.1f}%) are both low — expect range-bound, choppy action. Trend-following strategies may struggle.")
            
            # 4. Final synthesis
            max_signal = max(setup_high, setup_low, mr_long, mr_short)
            if max_signal < 10:
                synthesis = "🔍 <b>The Bottom Line:</b> All signals are very weak (<10%). The model doesn't see a clear opportunity. When there's no edge, it's best to <span style='color: #ffcc00;'>WAIT for better setups</span>."
                ml_summary = "ML: No clear edge - WAIT"
            elif mr_long > 25:
                if setup_high > setup_low:
                    synthesis = f"🎯 <b>The Bottom Line:</b> Strong LONG opportunity! Mean Reversion LONG at {mr_long:.1f}% says the stock is oversold, AND setup quality favors trades hitting TP. <span style='color: #00ff88;'>LONG looks good</span>."
                    ml_summary = f"ML: LONG ({mr_long:.0f}% MR)"
                else:
                    synthesis = f"🔍 <b>The Bottom Line:</b> Mean Reversion LONG ({mr_long:.1f}%) suggests a bounce, but setup quality is poor. The bounce may not reach your profit target. <span style='color: #ffcc00;'>Risky LONG - reduce size</span>."
                    ml_summary = f"ML: Risky LONG ({mr_long:.0f}%)"
            elif mr_short > 25:
                if setup_high > setup_low:
                    synthesis = f"🎯 <b>The Bottom Line:</b> Strong SHORT opportunity! Mean Reversion SHORT at {mr_short:.1f}% says the stock is overbought, AND setup quality favors trades hitting TP. <span style='color: #ff6b6b;'>SHORT looks good</span>."
                    ml_summary = f"ML: SHORT ({mr_short:.0f}% MR)"
                else:
                    synthesis = f"🔍 <b>The Bottom Line:</b> Mean Reversion SHORT ({mr_short:.1f}%) suggests a fade, but setup quality is poor. The pullback may not reach your target. <span style='color: #ffcc00;'>Risky SHORT - reduce size</span>."
                    ml_summary = f"ML: Risky SHORT ({mr_short:.0f}%)"
            elif mr_short > 10 and mr_short > mr_long * 5:
                # Moderate SHORT signal with clear dominance over LONG
                synthesis = f"🔍 <b>The Bottom Line:</b> Mean Reversion leans SHORT ({mr_short:.1f}%) — there's a modest overbought signal. Not a screaming SHORT, but <span style='color: #ffcc00;'>be cautious with LONGs</span> and consider waiting for a pullback."
                ml_summary = f"ML: Slight bearish lean ({mr_short:.0f}%)"
            elif mr_long > 10 and mr_long > mr_short * 5:
                # Moderate LONG signal with clear dominance over SHORT
                synthesis = f"🔍 <b>The Bottom Line:</b> Mean Reversion leans LONG ({mr_long:.1f}%) — there's a modest oversold signal. Not a screaming LONG, but <span style='color: #00d4aa;'>dips could be bought</span> with appropriate risk management."
                ml_summary = f"ML: Slight bullish lean ({mr_long:.0f}%)"
            elif setup_low > setup_high + 15:
                synthesis = f"⚠️ <b>The Bottom Line:</b> Setup Quality LOW ({setup_low:.1f}%) dominates — most trades taken now will hit stop loss first. <span style='color: #ff6b6b;'>AVOID trading</span> until setup quality improves."
                ml_summary = f"ML: Bad R:R - AVOID"
            elif setup_high > setup_low + 15:
                synthesis = f"🔍 <b>The Bottom Line:</b> Setup Quality HIGH ({setup_high:.1f}%) is good, but no strong directional signal. Get direction from price action, then <span style='color: #00d4aa;'>take the trade with confidence</span>."
                ml_summary = f"ML: Good R:R ({setup_high:.0f}%)"
            else:
                synthesis = "🔍 <b>The Bottom Line:</b> Signals are mixed with no clear direction. <span style='color: #ffcc00;'>Be patient</span> and wait for a clearer setup."
                ml_summary = "ML: Mixed - WAIT"
            
            # Build the narrative HTML
            narrative_html = "<div style='background: #0d1a2e; border: 2px solid #3b82f6; border-radius: 10px; padding: 15px; margin: 15px 0;'>"
            narrative_html += "<div style='color: #3b82f6; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🔗 How These Signals Connect:</div>"
            narrative_html += "<div style='color: #ddd; line-height: 1.8;'>"
            for i, part in enumerate(narrative_parts):
                narrative_html += f"<div style='margin-bottom: 10px;'><span style='color: #888;'>{i+1}.</span> {part}</div>"
            narrative_html += f"<div style='margin-top: 15px; padding-top: 12px; border-top: 1px solid #444;'>{synthesis}</div>"
            narrative_html += "</div></div>"
            
            ml_story += narrative_html
            
            # Quick reference
            ml_story += (
                "<div style='background: #1a1a2e; border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 10px 0;'>"
                "<div style='color: #a855f7; font-weight: bold; margin-bottom: 8px;'>📚 Quick Reference - What Each Signal Means:</div>"
                "<div style='color: #ccc; line-height: 1.6; font-size: 0.9em;'>"
                "<div style='margin-bottom: 6px;'>✅ <b>Setup Quality HIGH:</b> Your trade will likely hit profit target → Good entry</div>"
                "<div style='margin-bottom: 6px;'>❌ <b>Setup Quality LOW:</b> Your trade will likely hit stop loss → Bad entry</div>"
                "<div style='margin-bottom: 6px;'>📈 <b>Mean Rev LONG:</b> Stock is oversold → Expecting bounce UP</div>"
                "<div style='margin-bottom: 6px;'>📉 <b>Mean Rev SHORT:</b> Stock is overbought → Expecting pullback DOWN</div>"
                "<div style='margin-bottom: 6px;'>🚀 <b>Breakout Valid:</b> If price breaks a level, it will likely follow through</div>"
                "<div>💥 <b>Vol Expansion:</b> Expect bigger price swings → Reduce position size</div>"
                "</div></div>"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # CRYPTO SWING MODE - Trend + Reversal Labels
        # ═══════════════════════════════════════════════════════════════════════
        elif mode_lower == 'swing':
            trend_bull = ml_signals.get('trend_holds_bull', 0) or 0
            trend_bear = ml_signals.get('trend_holds_bear', 0) or 0
            rev_bull = ml_signals.get('reversal_to_bull', 0) or 0
            rev_bear = ml_signals.get('reversal_to_bear', 0) or 0
            drawdown = ml_signals.get('drawdown', 0) or 0
            
            f1_trend_bull = f1.get('trend_holds_bull', 0.5)
            f1_trend_bear = f1.get('trend_holds_bear', 0.5)
            f1_rev_bull = f1.get('reversal_to_bull', 0.5)
            f1_rev_bear = f1.get('reversal_to_bear', 0.5)
            f1_drawdown = f1.get('drawdown', 0.5)
            
            ml_story = (
                "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #a855f7;'>"
                "<div style='color: #a855f7; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🤖 STEP 1: What Does ML See? (Swing)</div>"
                "<div style='color: #ccc; line-height: 1.8; margin-bottom: 15px;'>The ML model predicts trend continuation and reversals:</div>"
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if trend_bull > 30 else '#333'};'>"
                f"<div style='color: #00ff88; font-weight: bold;'>📈 Trend BULL</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{trend_bull:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_trend_bull:.0%} {f1_reliability(f1_trend_bull)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if trend_bear > 30 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>📉 Trend BEAR</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{trend_bear:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_trend_bear:.0%} {f1_reliability(f1_trend_bear)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if rev_bull > 30 else '#333'};'>"
                f"<div style='color: #00d4aa; font-weight: bold;'>🔄 Reversal UP</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{rev_bull:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_rev_bull:.0%} {f1_reliability(f1_rev_bull)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if rev_bear > 30 else '#333'};'>"
                f"<div style='color: #ff9500; font-weight: bold;'>🔄 Reversal DOWN</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{rev_bear:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_rev_bear:.0%} {f1_reliability(f1_rev_bear)[0]}</div></div>"
                "</div>"
            )
            
            # Drawdown row
            ml_story += (
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                f"<div style='flex: 1; min-width: 200px; background: #1a0a1a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if drawdown > 50 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>🛡️ Drawdown Risk</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{drawdown:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_drawdown:.0%} {f1_reliability(f1_drawdown)[0]}</div></div>"
                "</div>"
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔗 CONNECTING NARRATIVE - How these signals relate (Crypto Swing)
            # ═══════════════════════════════════════════════════════════════════
            narrative_parts = []
            
            # 1. Trend signals - is there a dominant trend?
            trend_diff = trend_bull - trend_bear
            if trend_diff > 20:
                narrative_parts.append(f"<span style='color: #00ff88;'>Trend BULL ({trend_bull:.1f}%) is much stronger than BEAR ({trend_bear:.1f}%)</span> — the uptrend appears healthy and likely to continue. Going LONG aligns with the trend.")
            elif trend_diff < -20:
                narrative_parts.append(f"<span style='color: #ff6b6b;'>Trend BEAR ({trend_bear:.1f}%) is much stronger than BULL ({trend_bull:.1f}%)</span> — the downtrend appears healthy and likely to continue. Going SHORT aligns with the trend.")
            elif abs(trend_diff) < 5:
                narrative_parts.append(f"Trend BULL ({trend_bull:.1f}%) and BEAR ({trend_bear:.1f}%) are nearly equal — there's no clear trend direction. The market is choppy or consolidating.")
            else:
                stronger = "BULL" if trend_diff > 0 else "BEAR"
                stronger_color = "#00ff88" if trend_diff > 0 else "#ff6b6b"
                narrative_parts.append(f"Trend <span style='color: {stronger_color};'>{stronger}</span> has a slight edge, but it's not dominant. Trend may be weakening or transitioning.")
            
            # 2. Reversal signals - is a turn coming?
            if rev_bull > 25 or rev_bear > 25:
                if rev_bull > rev_bear and rev_bull > 25:
                    if trend_bear > trend_bull:
                        narrative_parts.append(f"<span style='color: #00ff88;'>Reversal UP ({rev_bull:.1f}%) is significant while we're in a downtrend</span> — this could be a bottom forming! Watch for confirmation before going LONG.")
                    else:
                        narrative_parts.append(f"Reversal UP ({rev_bull:.1f}%) is elevated, but we're already in an uptrend — this might just be continuation. Less meaningful for new longs.")
                elif rev_bear > rev_bull and rev_bear > 25:
                    if trend_bull > trend_bear:
                        narrative_parts.append(f"<span style='color: #ff6b6b;'>Reversal DOWN ({rev_bear:.1f}%) is significant while we're in an uptrend</span> — this could be a top forming! Consider taking profits or going SHORT.")
                    else:
                        narrative_parts.append(f"Reversal DOWN ({rev_bear:.1f}%) is elevated, but we're already in a downtrend — this might be capitulation. Could be a bounce coming.")
            else:
                narrative_parts.append(f"Reversal signals are quiet (UP: {rev_bull:.1f}%, DOWN: {rev_bear:.1f}%) — no major trend change expected. The current trend should continue.")
            
            # 3. Drawdown context
            if drawdown > 50:
                narrative_parts.append(f"<span style='color: #ff6b6b;'>⚠️ Drawdown Risk ({drawdown:.1f}%) is HIGH</span> — expect significant volatility! Even if your direction is right, you may face painful dips. Reduce position size.")
            elif drawdown > 30:
                narrative_parts.append(f"Drawdown Risk ({drawdown:.1f}%) is moderate — some volatility expected. Use appropriate stop losses.")
            else:
                narrative_parts.append(f"<span style='color: #00ff88;'>Drawdown Risk ({drawdown:.1f}%) is LOW</span> — relatively calm conditions. Your trades should run smoothly if the direction is right.")
            
            # 4. Final synthesis
            max_signal = max(trend_bull, trend_bear, rev_bull, rev_bear)
            if max_signal < 15:
                synthesis = "🔍 <b>The Bottom Line:</b> All signals are weak (<15%). The model isn't confident about the trend direction or reversals. <span style='color: #ffcc00;'>WAIT for clearer signals</span>."
                ml_summary = "ML: No clear direction - WAIT"
            elif rev_bull > 35 and trend_bear > trend_bull:
                synthesis = f"🎯 <b>The Bottom Line:</b> Potential trend reversal! Reversal UP ({rev_bull:.1f}%) is strong while we're in a downtrend. This looks like a <span style='color: #00ff88;'>LONG opportunity</span> if you can handle the risk."
                ml_summary = f"ML: Reversal LONG ({rev_bull:.0f}%)"
            elif rev_bear > 35 and trend_bull > trend_bear:
                synthesis = f"🎯 <b>The Bottom Line:</b> Potential trend reversal! Reversal DOWN ({rev_bear:.1f}%) is strong while we're in an uptrend. This looks like a <span style='color: #ff6b6b;'>SHORT opportunity</span> or time to take profits."
                ml_summary = f"ML: Reversal SHORT ({rev_bear:.0f}%)"
            elif trend_bull > trend_bear + 15:
                if drawdown > 50:
                    synthesis = f"🔍 <b>The Bottom Line:</b> Trend is bullish ({trend_bull:.1f}%), but drawdown risk is high ({drawdown:.1f}%). <span style='color: #ffcc00;'>LONG with reduced size</span> — direction looks right but volatility is elevated."
                    ml_summary = f"ML: Cautious LONG ({trend_bull:.0f}%)"
                else:
                    synthesis = f"🎯 <b>The Bottom Line:</b> Clear bullish trend ({trend_bull:.1f}% vs {trend_bear:.1f}%) with manageable risk. <span style='color: #00ff88;'>LONG looks good</span>. Ride the trend!"
                    ml_summary = f"ML: LONG ({trend_bull:.0f}%)"
            elif trend_bear > trend_bull + 15:
                if drawdown > 50:
                    synthesis = f"🔍 <b>The Bottom Line:</b> Trend is bearish ({trend_bear:.1f}%), but drawdown risk is high ({drawdown:.1f}%). <span style='color: #ffcc00;'>SHORT with reduced size</span> — direction looks right but expect bounces."
                    ml_summary = f"ML: Cautious SHORT ({trend_bear:.0f}%)"
                else:
                    synthesis = f"🎯 <b>The Bottom Line:</b> Clear bearish trend ({trend_bear:.1f}% vs {trend_bull:.1f}%) with manageable risk. <span style='color: #ff6b6b;'>SHORT looks good</span>. Ride the trend!"
                    ml_summary = f"ML: SHORT ({trend_bear:.0f}%)"
            else:
                synthesis = "🔍 <b>The Bottom Line:</b> Signals are mixed — no dominant trend or clear reversal. <span style='color: #ffcc00;'>Be patient</span> and wait for a cleaner setup."
                ml_summary = "ML: Mixed - WAIT"
            
            # Build the narrative HTML
            narrative_html = "<div style='background: #0d1a2e; border: 2px solid #3b82f6; border-radius: 10px; padding: 15px; margin: 15px 0;'>"
            narrative_html += "<div style='color: #3b82f6; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🔗 How These Signals Connect:</div>"
            narrative_html += "<div style='color: #ddd; line-height: 1.8;'>"
            for i, part in enumerate(narrative_parts):
                narrative_html += f"<div style='margin-bottom: 10px;'><span style='color: #888;'>{i+1}.</span> {part}</div>"
            narrative_html += f"<div style='margin-top: 15px; padding-top: 12px; border-top: 1px solid #444;'>{synthesis}</div>"
            narrative_html += "</div></div>"
            
            ml_story += narrative_html
            
            # Quick reference
            ml_story += (
                "<div style='background: #1a1a2e; border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 10px 0;'>"
                "<div style='color: #a855f7; font-weight: bold; margin-bottom: 8px;'>📚 Quick Reference - What Each Signal Means:</div>"
                "<div style='color: #ccc; line-height: 1.6; font-size: 0.9em;'>"
                "<div style='margin-bottom: 6px;'>📈 <b>Trend BULL:</b> Uptrend expected to continue (2%+ upside)</div>"
                "<div style='margin-bottom: 6px;'>📉 <b>Trend BEAR:</b> Downtrend expected to continue (2%+ downside)</div>"
                "<div style='margin-bottom: 6px;'>🔄 <b>Reversal UP:</b> Bottom forming → Good time to go LONG</div>"
                "<div style='margin-bottom: 6px;'>🔄 <b>Reversal DOWN:</b> Top forming → Good time to go SHORT or take profits</div>"
                "<div>🛡️ <b>Drawdown Risk:</b> How much volatility to expect → Size your position accordingly</div>"
                "</div></div>"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # INVESTMENT MODE - Accumulation/Distribution Labels
        # ═══════════════════════════════════════════════════════════════════════
        elif mode_lower == 'investment':
            accum = ml_signals.get('accumulation', 0) or 0
            distrib = ml_signals.get('distribution', 0) or 0
            rev_bull = ml_signals.get('reversal_to_bull', 0) or 0
            rev_bear = ml_signals.get('reversal_to_bear', 0) or 0
            large_dd = ml_signals.get('large_drawdown', 0) or 0
            
            f1_accum = f1.get('accumulation', 0.5)
            f1_distrib = f1.get('distribution', 0.5)
            f1_rev_bull = f1.get('reversal_to_bull', 0.5)
            f1_rev_bear = f1.get('reversal_to_bear', 0.5)
            f1_dd = f1.get('large_drawdown', 0.5)
            
            ml_story = (
                "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #a855f7;'>"
                "<div style='color: #a855f7; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🤖 STEP 1: What Does ML See? (Investment)</div>"
                "<div style='color: #ccc; line-height: 1.8; margin-bottom: 15px;'>The ML model analyzes institutional accumulation/distribution:</div>"
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if accum > 30 else '#333'};'>"
                f"<div style='color: #00ff88; font-weight: bold;'>💰 Accumulation</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{accum:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_accum:.0%} {f1_reliability(f1_accum)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if distrib > 30 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>📤 Distribution</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{distrib:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_distrib:.0%} {f1_reliability(f1_distrib)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if rev_bull > 30 else '#333'};'>"
                f"<div style='color: #00d4aa; font-weight: bold;'>🔄 Major Reversal UP</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{rev_bull:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_rev_bull:.0%} {f1_reliability(f1_rev_bull)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if rev_bear > 30 else '#333'};'>"
                f"<div style='color: #ff9500; font-weight: bold;'>🔄 Major Reversal DOWN</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{rev_bear:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_rev_bear:.0%} {f1_reliability(f1_rev_bear)[0]}</div></div>"
                "</div>"
            )
            
            # Large drawdown row
            ml_story += (
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                f"<div style='flex: 1; min-width: 200px; background: #1a0a1a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if large_dd > 50 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>⚠️ Large Drawdown Risk (7%+)</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{large_dd:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_dd:.0%} {f1_reliability(f1_dd)[0]}</div></div>"
                "</div>"
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔗 CONNECTING NARRATIVE - How these signals relate to each other
            # ═══════════════════════════════════════════════════════════════════
            narrative_parts = []
            
            # 1. Compare Accumulation vs Distribution
            if accum > 5 or distrib > 5:
                if distrib > accum * 2 and distrib > 5:
                    narrative_parts.append(f"<span style='color: #ff6b6b;'>Distribution ({distrib:.1f}%) is significantly higher than Accumulation ({accum:.1f}%)</span> — this suggests smart money may be quietly selling their positions to retail buyers.")
                elif accum > distrib * 2 and accum > 5:
                    narrative_parts.append(f"<span style='color: #00ff88;'>Accumulation ({accum:.1f}%) is significantly higher than Distribution ({distrib:.1f}%)</span> — this suggests smart money is quietly building positions while others are fearful.")
                elif abs(accum - distrib) < 3:
                    narrative_parts.append(f"Accumulation ({accum:.1f}%) and Distribution ({distrib:.1f}%) are roughly balanced — no clear institutional bias detected.")
                else:
                    winner = "Accumulation" if accum > distrib else "Distribution"
                    winner_val = max(accum, distrib)
                    loser_val = min(accum, distrib)
                    narrative_parts.append(f"{winner} ({winner_val:.1f}%) has a slight edge over the other ({loser_val:.1f}%), but the difference is small.")
            else:
                narrative_parts.append(f"Both Accumulation ({accum:.1f}%) and Distribution ({distrib:.1f}%) are very low — the model doesn't see clear institutional activity in either direction.")
            
            # 2. Reversal signals context
            if rev_bull > 10 or rev_bear > 10:
                if rev_bull > rev_bear and rev_bull > 10:
                    if rev_bull > 30:
                        narrative_parts.append(f"<span style='color: #00ff88;'>Major Reversal UP ({rev_bull:.1f}%) is elevated</span> — the model sees potential for a significant bottom forming. If you're waiting to buy, this could be the opportunity.")
                    else:
                        narrative_parts.append(f"Major Reversal UP ({rev_bull:.1f}%) shows some activity — there's a hint of a potential bottom, but conviction is not strong yet.")
                elif rev_bear > rev_bull and rev_bear > 10:
                    if rev_bear > 30:
                        narrative_parts.append(f"<span style='color: #ff6b6b;'>Major Reversal DOWN ({rev_bear:.1f}%) is elevated</span> — the model sees potential for a significant top forming. Consider taking some profits if you're holding.")
                    else:
                        narrative_parts.append(f"Major Reversal DOWN ({rev_bear:.1f}%) shows some activity — there's a hint of a potential top, but conviction is not strong yet.")
            else:
                narrative_parts.append(f"Both reversal signals are quiet (UP: {rev_bull:.1f}%, DOWN: {rev_bear:.1f}%) — no major trend change expected in the near term.")
            
            # 3. Risk assessment
            if large_dd > 30:
                narrative_parts.append(f"<span style='color: #ff6b6b;'>⚠️ Large Drawdown Risk ({large_dd:.1f}%) is concerning</span> — the model sees elevated risk of a 7%+ decline. Consider reducing position size or waiting.")
            elif large_dd > 15:
                narrative_parts.append(f"Drawdown Risk ({large_dd:.1f}%) is moderate — some caution is warranted, but not alarming.")
            else:
                narrative_parts.append(f"<span style='color: #00ff88;'>Drawdown Risk ({large_dd:.1f}%) is low</span> — the model doesn't see major downside risk at this time.")
            
            # 4. Final synthesis
            max_signal = max(accum, distrib, rev_bull, rev_bear)
            if max_signal < 10:
                synthesis = "🔍 <b>The Bottom Line:</b> All signals are weak (under 10%). The model isn't confident about any scenario. When there's no clear edge, the smart move is to <span style='color: #ffcc00;'>HOLD your current position</span> and wait for clearer signals."
                ml_summary = "ML: No clear signal - HOLD"
            elif accum > 30:
                synthesis = f"🎯 <b>The Bottom Line:</b> Accumulation dominates at {accum:.1f}%. Smart money appears to be buying. This is typically a <span style='color: #00ff88;'>good time to ACCUMULATE</span> if you believe in the long-term thesis."
                ml_summary = f"ML: Accumulate ({accum:.0f}%)"
            elif distrib > 30:
                synthesis = f"🎯 <b>The Bottom Line:</b> Distribution dominates at {distrib:.1f}%. Smart money appears to be selling. Consider <span style='color: #ff6b6b;'>TRIMMING positions</span> or being very selective with new entries."
                ml_summary = f"ML: Distribute ({distrib:.0f}%)"
            elif rev_bull > 25:
                synthesis = f"🎯 <b>The Bottom Line:</b> Reversal UP signal at {rev_bull:.1f}% suggests a bottom may be forming. If you're looking to <span style='color: #00ff88;'>start a position</span>, this could be an opportunity."
                ml_summary = f"ML: Potential bottom ({rev_bull:.0f}%)"
            elif rev_bear > 25:
                synthesis = f"🎯 <b>The Bottom Line:</b> Reversal DOWN signal at {rev_bear:.1f}% suggests a top may be forming. Consider <span style='color: #ff6b6b;'>taking profits</span> or waiting for better prices."
                ml_summary = f"ML: Potential top ({rev_bear:.0f}%)"
            else:
                # Mixed but with some lean
                if distrib > accum:
                    synthesis = f"🔍 <b>The Bottom Line:</b> Signals are mixed but lean slightly bearish (Distribution {distrib:.1f}% > Accumulation {accum:.1f}%). <span style='color: #ffcc00;'>Be cautious</span> with new longs. HOLD existing positions but don't add aggressively."
                    ml_summary = "ML: Slight bearish lean - HOLD"
                elif accum > distrib:
                    synthesis = f"🔍 <b>The Bottom Line:</b> Signals are mixed but lean slightly bullish (Accumulation {accum:.1f}% > Distribution {distrib:.1f}%). Existing positions are <span style='color: #00d4aa;'>okay to hold</span>. Small additions could work."
                    ml_summary = "ML: Slight bullish lean - HOLD"
                else:
                    synthesis = "🔍 <b>The Bottom Line:</b> Signals are balanced with no clear direction. <span style='color: #ffcc00;'>HOLD current positions</span> and wait for a clearer picture."
                    ml_summary = "ML: Balanced - HOLD"
            
            # Build the narrative HTML
            narrative_html = "<div style='background: #0d1a2e; border: 2px solid #3b82f6; border-radius: 10px; padding: 15px; margin: 15px 0;'>"
            narrative_html += "<div style='color: #3b82f6; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🔗 How These Signals Connect:</div>"
            narrative_html += "<div style='color: #ddd; line-height: 1.8;'>"
            for i, part in enumerate(narrative_parts):
                narrative_html += f"<div style='margin-bottom: 10px;'><span style='color: #888;'>{i+1}.</span> {part}</div>"
            narrative_html += f"<div style='margin-top: 15px; padding-top: 12px; border-top: 1px solid #444;'>{synthesis}</div>"
            narrative_html += "</div></div>"
            
            ml_story += narrative_html
            
            # Simplified interpretation (replaces the old static one)
            ml_story += (
                "<div style='background: #1a1a2e; border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 10px 0;'>"
                "<div style='color: #a855f7; font-weight: bold; margin-bottom: 8px;'>📚 Quick Reference - What Each Signal Means:</div>"
                "<div style='color: #ccc; line-height: 1.6; font-size: 0.9em;'>"
                "<div style='margin-bottom: 6px;'>💰 <b>Accumulation:</b> Big players buying quietly → Price likely to rise later</div>"
                "<div style='margin-bottom: 6px;'>📤 <b>Distribution:</b> Big players selling quietly → Price may fall later</div>"
                "<div style='margin-bottom: 6px;'>🔄 <b>Reversal UP:</b> Downtrend ending → Good time to buy</div>"
                "<div style='margin-bottom: 6px;'>🔄 <b>Reversal DOWN:</b> Uptrend ending → Good time to sell/trim</div>"
                "<div>⚠️ <b>Large Drawdown:</b> Risk of 7%+ drop → Size your position carefully</div>"
                "</div></div>"
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # SCALP/DAYTRADE MODE - Continue + Fakeout Labels
        # ═══════════════════════════════════════════════════════════════════════
        else:
            cont_up = ml_signals.get('continue_up', 0) or 0
            cont_down = ml_signals.get('continue_down', 0) or 0
            fake_up = ml_signals.get('fakeout_up', 0) or 0
            fake_down = ml_signals.get('fakeout_down', 0) or 0
            vol_exp = ml_signals.get('vol_expansion', 0) or 0
            
            # Get F1 scores with defaults
            f1_cont_up = f1.get('continue_up', 0.5)
            f1_cont_down = f1.get('continue_down', 0.5)
            f1_fake_up = f1.get('fakeout_up', 0.5)
            f1_fake_down = f1.get('fakeout_down', 0.5)
            
            # Build ML story with clean HTML
            ml_story = (
                "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #a855f7;'>"
                "<div style='color: #a855f7; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🤖 STEP 1: What Does ML See?</div>"
                "<div style='color: #ccc; line-height: 1.8; margin-bottom: 15px;'>The ML model analyzes technical features and predicts probabilities:</div>"
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                f"<div style='flex: 1; min-width: 140px; background: #0a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#00ff88' if cont_up > 30 else '#333'};'>"
                f"<div style='color: #00ff88; font-weight: bold;'>📈 Continue UP</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{cont_up:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_cont_up:.0%} {f1_reliability(f1_cont_up)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a0a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ff6b6b' if cont_down > 30 else '#333'};'>"
                f"<div style='color: #ff6b6b; font-weight: bold;'>📉 Continue DOWN</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{cont_down:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_cont_down:.0%} {f1_reliability(f1_cont_down)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ffcc00' if fake_up > 20 else '#333'};'>"
                f"<div style='color: #ffcc00; font-weight: bold;'>🎭 Fakeout UP</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{fake_up:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_fake_up:.0%} {f1_reliability(f1_fake_up)[0]}</div></div>"
                f"<div style='flex: 1; min-width: 140px; background: #1a1a0a; padding: 10px; border-radius: 8px; border: 1px solid {'#ffcc00' if fake_down > 20 else '#333'};'>"
                f"<div style='color: #ffcc00; font-weight: bold;'>🎭 Fakeout DOWN</div>"
                f"<div style='color: #fff; font-size: 1.5em; font-weight: bold;'>{fake_down:.2f}%</div>"
                f"<div style='color: #888; font-size: 0.8em;'>F1: {f1_fake_down:.0%} {f1_reliability(f1_fake_down)[0]}</div></div>"
                "</div>"
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 🔗 CONNECTING NARRATIVE - How these signals relate (Scalp/DayTrade)
            # ═══════════════════════════════════════════════════════════════════
            narrative_parts = []
            
            # 1. Direction comparison - which way is the market leaning?
            direction_diff = cont_up - cont_down
            if direction_diff > 20:
                narrative_parts.append(f"<span style='color: #00ff88;'>Continue UP ({cont_up:.1f}%) is much stronger than DOWN ({cont_down:.1f}%)</span> — the model sees the current move continuing upward. LONG trades align with the trend.")
            elif direction_diff < -20:
                narrative_parts.append(f"<span style='color: #ff6b6b;'>Continue DOWN ({cont_down:.1f}%) is much stronger than UP ({cont_up:.1f}%)</span> — the model sees the current move continuing downward. SHORT trades align with the trend.")
            elif abs(direction_diff) < 5:
                narrative_parts.append(f"Continue UP ({cont_up:.1f}%) and DOWN ({cont_down:.1f}%) are nearly equal — the model can't decide which way the market will go. This is a coin flip situation.")
            else:
                stronger = "UP" if direction_diff > 0 else "DOWN"
                stronger_color = "#00ff88" if direction_diff > 0 else "#ff6b6b"
                narrative_parts.append(f"Continue <span style='color: {stronger_color};'>{stronger}</span> has a slight edge, but the difference is small. Be selective with entries.")
            
            # 2. Trap detection - is the obvious trade dangerous?
            if fake_up > 15 or fake_down > 15:
                if fake_down > fake_up and fake_down > 15:
                    if cont_up > cont_down:
                        narrative_parts.append(f"<span style='color: #ff6b6b;'>⚠️ TRAP WARNING: Fakeout DOWN ({fake_down:.1f}%) is elevated while direction leans bullish!</span> This means the obvious LONG could be a TRAP. Price may look bullish but then reverse DOWN to stop out longs.")
                    else:
                        narrative_parts.append(f"Fakeout DOWN ({fake_down:.1f}%) is elevated — there's some risk of a bull trap, but direction already leans bearish so this is less concerning.")
                elif fake_up > fake_down and fake_up > 15:
                    if cont_down > cont_up:
                        narrative_parts.append(f"<span style='color: #00ff88;'>⚠️ TRAP WARNING: Fakeout UP ({fake_up:.1f}%) is elevated while direction leans bearish!</span> This means the obvious SHORT could be a TRAP. Price may look bearish but then reverse UP to stop out shorts.")
                    else:
                        narrative_parts.append(f"Fakeout UP ({fake_up:.1f}%) is elevated — there's some risk of a bear trap, but direction already leans bullish so this is less concerning.")
                elif fake_up > 10 and fake_down > 10:
                    narrative_parts.append(f"Both Fakeout signals are active (UP: {fake_up:.1f}%, DOWN: {fake_down:.1f}%) — the market is choppy and prone to trapping traders in both directions. This is a tough environment for directional trades.")
            else:
                narrative_parts.append(f"Fakeout signals are quiet (UP: {fake_up:.1f}%, DOWN: {fake_down:.1f}%) — no major trap risk detected. If you get the direction right, the trade should run without nasty reversals.")
            
            # 3. Overall confidence
            max_signal = max(cont_up, cont_down, fake_up, fake_down)
            if max_signal < 10:
                narrative_parts.append("<span style='color: #ffcc00;'>All signals are very weak (&lt;10%)</span> — the model is essentially saying 'I have no idea what's going to happen.' This is the worst time to trade.")
            elif max_signal < 25:
                narrative_parts.append(f"The strongest signal is only {max_signal:.1f}% — moderate confidence. The model sees something but isn't highly convicted. Trade smaller if you trade at all.")
            else:
                narrative_parts.append(f"The model shows meaningful conviction with signals up to {max_signal:.1f}% — there's something to work with here. Trust the direction if it aligns with your other analysis.")
            
            # 4. Final synthesis
            if max_signal < 10:
                synthesis = "🔍 <b>The Bottom Line:</b> All signals are too weak to act on. The model has no edge here. <span style='color: #ffcc00;'>WAIT for a better setup</span> — trading now is gambling."
                ml_summary = "ML: No edge - WAIT"
            elif fake_down > cont_up and fake_down > 20 and cont_up > cont_down:
                synthesis = f"⚠️ <b>The Bottom Line:</b> BULL TRAP likely! Direction leans UP ({cont_up:.1f}%), but Fakeout DOWN ({fake_down:.1f}%) is higher — the obvious LONG is dangerous. Either <span style='color: #ffcc00;'>WAIT</span> or go <span style='color: #ff6b6b;'>SHORT</span> (contrarian)."
                ml_summary = f"ML: Bull Trap Risk - WAIT/SHORT"
            elif fake_up > cont_down and fake_up > 20 and cont_down > cont_up:
                synthesis = f"⚠️ <b>The Bottom Line:</b> BEAR TRAP likely! Direction leans DOWN ({cont_down:.1f}%), but Fakeout UP ({fake_up:.1f}%) is higher — the obvious SHORT is dangerous. Either <span style='color: #ffcc00;'>WAIT</span> or go <span style='color: #00ff88;'>LONG</span> (contrarian)."
                ml_summary = f"ML: Bear Trap Risk - WAIT/LONG"
            elif cont_up > cont_down + 15 and fake_down < 15:
                synthesis = f"🎯 <b>The Bottom Line:</b> Clear bullish bias! Continue UP ({cont_up:.1f}%) beats DOWN ({cont_down:.1f}%) with low trap risk. <span style='color: #00ff88;'>LONG looks good</span>."
                ml_summary = f"ML: LONG ({cont_up:.0f}%)"
            elif cont_down > cont_up + 15 and fake_up < 15:
                synthesis = f"🎯 <b>The Bottom Line:</b> Clear bearish bias! Continue DOWN ({cont_down:.1f}%) beats UP ({cont_up:.1f}%) with low trap risk. <span style='color: #ff6b6b;'>SHORT looks good</span>."
                ml_summary = f"ML: SHORT ({cont_down:.0f}%)"
            elif vol_exp > 50:
                synthesis = f"💥 <b>The Bottom Line:</b> Volatility expansion ({vol_exp:.1f}%) suggests a big move is coming, but direction is unclear. <span style='color: #ffcc00;'>Wait for the breakout direction</span> before trading."
                ml_summary = f"ML: Big move coming - WAIT"
            else:
                synthesis = "🔍 <b>The Bottom Line:</b> Signals are mixed without a clear edge. <span style='color: #ffcc00;'>Be patient</span> — better setups will come."
                ml_summary = "ML: Mixed - WAIT"
            
            # Build the narrative HTML
            narrative_html = "<div style='background: #0d1a2e; border: 2px solid #3b82f6; border-radius: 10px; padding: 15px; margin: 15px 0;'>"
            narrative_html += "<div style='color: #3b82f6; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🔗 How These Signals Connect:</div>"
            narrative_html += "<div style='color: #ddd; line-height: 1.8;'>"
            for i, part in enumerate(narrative_parts):
                narrative_html += f"<div style='margin-bottom: 10px;'><span style='color: #888;'>{i+1}.</span> {part}</div>"
            narrative_html += f"<div style='margin-top: 15px; padding-top: 12px; border-top: 1px solid #444;'>{synthesis}</div>"
            narrative_html += "</div></div>"
            
            ml_story += narrative_html
            
            # Quick reference
            ml_story += (
                "<div style='background: #1a1a2e; border: 1px solid #444; border-radius: 8px; padding: 12px; margin: 10px 0;'>"
                "<div style='color: #a855f7; font-weight: bold; margin-bottom: 8px;'>📚 Quick Reference - What Each Signal Means:</div>"
                "<div style='color: #ccc; line-height: 1.6; font-size: 0.9em;'>"
                "<div style='margin-bottom: 6px;'>📈 <b>Continue UP:</b> Current move likely continues upward → Favor LONG</div>"
                "<div style='margin-bottom: 6px;'>📉 <b>Continue DOWN:</b> Current move likely continues downward → Favor SHORT</div>"
                "<div style='margin-bottom: 6px;'>🎭 <b>Fakeout UP:</b> Looks bearish but will reverse UP → Bear trap, careful with SHORT</div>"
                "<div style='margin-bottom: 6px;'>🎭 <b>Fakeout DOWN:</b> Looks bullish but will reverse DOWN → Bull trap, careful with LONG</div>"
                "<div>💥 <b>Vol Expansion:</b> Big move expected → Wait for direction, then trade with momentum</div>"
                "</div></div>"
            )
        
        sections.append({'title': 'ML Predictions', 'html': ml_story, 'summary': ml_summary})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🌊 SECTION 2: MARKET PHASE STORY (Wyckoff Cycle)
    # ═══════════════════════════════════════════════════════════════════════════
    phase_story = ""
    phase_summary = ""
    
    if phase:
        phase_upper = phase.upper().replace(' ', '_')
        
        # Phase descriptions
        phase_info = {
            'ACCUMULATION': {
                'emoji': '🟢', 'color': '#00ff88',
                'description': 'Smart money is quietly buying while retail is fearful.',
                'action': 'IDEAL for LONGS - buying with the institutions!',
                'visual': '📉 → 📊 → 📈'
            },
            'MARKUP': {
                'emoji': '🚀', 'color': '#00ff88',
                'description': 'Uptrend in progress. Smart money already positioned, now riding the wave.',
                'action': 'LONGS valid if not too late. Trend is your friend!',
                'visual': '📈📈📈'
            },
            'DISTRIBUTION': {
                'emoji': '🔴', 'color': '#ff6b6b',
                'description': 'Smart money is quietly selling to eager retail buyers.',
                'action': 'AVOID LONGS - you may become exit liquidity!',
                'visual': '📈 → 📊 → 📉'
            },
            'MARKDOWN': {
                'emoji': '💀', 'color': '#ff6b6b',
                'description': 'Downtrend in progress. Smart money already exited.',
                'action': 'SHORTS valid. Do NOT catch falling knives!',
                'visual': '📉📉📉'
            },
            'CAPITULATION': {
                'emoji': '😱', 'color': '#00ff88',
                'description': 'Panic selling! Retail dumping at the worst time.',
                'action': 'CONTRARIAN LONG opportunity if whales accumulating!',
                'visual': '📉📉📉 → 🔄'
            },
            'RE_ACCUMULATION': {
                'emoji': '🔄', 'color': '#00d4aa',
                'description': 'Pullback in uptrend. Smart money adding to positions.',
                'action': 'LONG on pullback - the dip to buy!',
                'visual': '📈 → 📉 → 📈'
            },
        }
        
        info = phase_info.get(phase_upper, {
            'emoji': '❓', 'color': '#888',
            'description': f'Phase: {phase}',
            'action': 'Exercise caution',
            'visual': '📊'
        })
        
        # Position interpretation
        if position_pct <= 35:
            pos_label = "EARLY"
            pos_color = "#00ff88"
            pos_desc = "Near the lows of the range - potential opportunity!"
        elif position_pct >= 65:
            pos_label = "LATE"
            pos_color = "#ff6b6b"
            pos_desc = "Near the highs of the range - don't chase!"
        else:
            pos_label = "MIDDLE"
            pos_color = "#ffcc00"
            pos_desc = "Mid-range - watch for direction confirmation"
        
        # Phase + Position combination story
        if phase_upper == 'MARKUP' and position_pct >= 65:
            combo_verdict = "⚠️ MARKUP but LATE = Don't chase! Wait for a pullback to enter."
            combo_color = "#ffcc00"
        elif phase_upper == 'MARKUP' and position_pct <= 35:
            combo_verdict = "🎯 MARKUP + EARLY = Ideal long entry! Trend confirmed, good R:R."
            combo_color = "#00ff88"
        elif phase_upper == 'ACCUMULATION':
            combo_verdict = "🎯 ACCUMULATION = Best phase for longs! Smart money buying."
            combo_color = "#00ff88"
        elif phase_upper == 'DISTRIBUTION':
            combo_verdict = "🚨 DISTRIBUTION = Avoid longs! Smart money exiting."
            combo_color = "#ff6b6b"
        elif phase_upper == 'MARKDOWN' and position_pct >= 65:
            combo_verdict = "🎯 MARKDOWN + LATE (high in range) = Good short entry on bounce."
            combo_color = "#ff6b6b"
        elif phase_upper == 'CAPITULATION' and position_pct >= 65:
            combo_verdict = "🔥 CAPITULATION + LATE = Potential bottom! Watch for reversal."
            combo_color = "#00ff88"
        else:
            combo_verdict = f"Phase: {phase_upper} + Position: {pos_label} = Monitor for better setup."
            combo_color = "#888"
        
        phase_story = (
            "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #00d4ff;'>"
            "<div style='color: #00d4ff; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🌊 STEP 2: Where Are We in the Market Cycle?</div>"
            "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;'>"
            f"<div style='flex: 1; min-width: 200px; background: #0a1a2a; padding: 12px; border-radius: 8px; border: 2px solid {info['color']};'>"
            f"<div style='color: {info['color']}; font-size: 1.3em; font-weight: bold;'>{info['emoji']} {phase_upper}</div>"
            f"<div style='color: #ccc; margin-top: 8px;'>{info['description']}</div>"
            f"<div style='color: {info['color']}; margin-top: 8px; font-weight: bold;'>{info['action']}</div>"
            f"<div style='color: #666; margin-top: 5px; font-size: 1.2em;'>{info['visual']}</div></div>"
            f"<div style='flex: 1; min-width: 200px; background: #0a1a2a; padding: 12px; border-radius: 8px; border: 2px solid {pos_color};'>"
            f"<div style='color: {pos_color}; font-size: 1.3em; font-weight: bold;'>📍 Position: {pos_label} ({position_pct:.0f}%)</div>"
            f"<div style='color: #ccc; margin-top: 8px;'>{pos_desc}</div>"
            f"<div style='margin-top: 10px; height: 8px; background: #333; border-radius: 4px; overflow: hidden;'>"
            f"<div style='width: {position_pct}%; height: 100%; background: linear-gradient(90deg, #00ff88, #ffcc00, #ff6b6b);'></div></div>"
            "<div style='display: flex; justify-content: space-between; color: #666; font-size: 0.8em; margin-top: 3px;'>"
            "<span>Lows</span><span>Mid</span><span>Highs</span></div></div></div>"
            f"<div style='background: #0d0d1a; padding: 12px; border-radius: 8px; border: 1px solid {combo_color};'>"
            f"<div style='color: {combo_color}; font-weight: bold; font-size: 1.05em;'>{combo_verdict}</div></div></div>"
        )
        
        phase_summary = f"Phase: {phase_upper} + {pos_label}"
        sections.append({'title': 'Market Phase', 'html': phase_story, 'summary': phase_summary})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🐋 SECTION 3: WHALE BEHAVIOR STORY
    # ═══════════════════════════════════════════════════════════════════════════
    whale_story = ""
    whale_summary = ""
    
    divergence = whale_pct - retail_pct
    
    # Whale interpretation
    if whale_pct >= 70:
        whale_stance = "STRONGLY BULLISH"
        whale_color = "#00ff88"
        whale_emoji = "🐋📈"
    elif whale_pct >= 55:
        whale_stance = "BULLISH"
        whale_color = "#00d4aa"
        whale_emoji = "🐋📊"
    elif whale_pct <= 30:
        whale_stance = "STRONGLY BEARISH"
        whale_color = "#ff6b6b"
        whale_emoji = "🐋📉"
    elif whale_pct <= 45:
        whale_stance = "BEARISH"
        whale_color = "#ff9500"
        whale_emoji = "🐋📊"
    else:
        whale_stance = "NEUTRAL"
        whale_color = "#888"
        whale_emoji = "🐋⚖️"
    
    # Divergence interpretation
    if divergence > 15:
        div_text = f"🎯 Smart money MORE bullish than retail by {divergence:.0f}% - FOLLOW THE WHALES!"
        div_color = "#00ff88"
    elif divergence < -15:
        div_text = f"⚠️ Retail MORE bullish than whales by {abs(divergence):.0f}% - POTENTIAL TRAP!"
        div_color = "#ff6b6b"
    else:
        div_text = f"📊 Whale/Retail roughly aligned ({divergence:+.0f}% divergence)"
        div_color = "#888"
    
    # OI + Price interpretation
    if oi_change > 2 and price_change > 1:
        oi_price_text = "📈 OI UP + Price UP = New LONG positions opening (bullish!)"
        oi_price_color = "#00ff88"
    elif oi_change > 2 and price_change < -1:
        oi_price_text = "📉 OI UP + Price DOWN = New SHORT positions opening (bearish!)"
        oi_price_color = "#ff6b6b"
    elif oi_change < -2 and price_change > 1:
        oi_price_text = "🔄 OI DOWN + Price UP = Shorts closing (short squeeze!)"
        oi_price_color = "#00d4ff"
    elif oi_change < -2 and price_change < -1:
        oi_price_text = "🔄 OI DOWN + Price DOWN = Longs closing (long squeeze!)"
        oi_price_color = "#ff9500"
    else:
        oi_price_text = f"📊 OI: {oi_change:+.1f}%, Price: {price_change:+.1f}% - No strong signal"
        oi_price_color = "#888"
    
    whale_story = (
        "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #00d4aa;'>"
        "<div style='color: #00d4aa; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>🐋 STEP 3: What Are the Whales Doing?</div>"
        "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;'>"
        f"<div style='flex: 1; min-width: 150px; background: #0a2a1a; padding: 12px; border-radius: 8px;'>"
        f"<div style='color: #888; font-size: 0.85em;'>Whale Position</div>"
        f"<div style='color: {whale_color}; font-size: 1.8em; font-weight: bold;'>{whale_pct:.0f}% LONG</div>"
        f"<div style='color: {whale_color}; font-weight: bold;'>{whale_emoji} {whale_stance}</div></div>"
        f"<div style='flex: 1; min-width: 150px; background: #2a1a0a; padding: 12px; border-radius: 8px;'>"
        f"<div style='color: #888; font-size: 0.85em;'>Retail Position</div>"
        f"<div style='color: #ff9500; font-size: 1.8em; font-weight: bold;'>{retail_pct:.0f}% LONG</div>"
        f"<div style='color: #888;'>{'🐑 Following whales' if abs(divergence) < 10 else '🎭 Different view'}</div></div></div>"
        f"<div style='background: #0d0d1a; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>"
        f"<div style='color: {div_color}; font-weight: bold;'>{div_text}</div></div>"
        f"<div style='background: #0d0d1a; padding: 10px; border-radius: 8px;'>"
        f"<div style='color: {oi_price_color};'>{oi_price_text}</div></div></div>"
    )
    
    whale_summary = f"Whales: {whale_pct:.0f}% ({whale_stance})"
    sections.append({'title': 'Whale Behavior', 'html': whale_story, 'summary': whale_summary})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📈 SECTION 4: WHALE MOMENTUM (Delta Change)
    # ═══════════════════════════════════════════════════════════════════════════
    delta_story = ""
    delta_summary = ""
    
    if whale_delta is not None:
        # Determine whale momentum
        if whale_delta > 5:
            delta_emoji = "🚀"
            delta_label = "STRONGLY ACCUMULATING"
            delta_color = "#00ff88"
            delta_desc = f"Whales increased LONG positions by {whale_delta:+.1f}% over {delta_lookback}!"
        elif whale_delta > 2:
            delta_emoji = "📈"
            delta_label = "ACCUMULATING"
            delta_color = "#00d4aa"
            delta_desc = f"Whales steadily adding to LONG positions ({whale_delta:+.1f}% over {delta_lookback})."
        elif whale_delta < -5:
            delta_emoji = "🔻"
            delta_label = "STRONGLY DISTRIBUTING"
            delta_color = "#ff6b6b"
            delta_desc = f"Whales reduced LONG positions by {whale_delta:+.1f}% over {delta_lookback}!"
        elif whale_delta < -2:
            delta_emoji = "📉"
            delta_label = "DISTRIBUTING"
            delta_color = "#ff9500"
            delta_desc = f"Whales reducing LONG positions ({whale_delta:+.1f}% over {delta_lookback})."
        else:
            delta_emoji = "➡️"
            delta_label = "STABLE"
            delta_color = "#888"
            delta_desc = f"Whale positions relatively unchanged ({whale_delta:+.1f}% over {delta_lookback})."
        
        # Retail delta description
        retail_delta_val = retail_delta if retail_delta is not None else -whale_delta
        if retail_delta_val > 2:
            retail_desc = f"Retail INCREASED longs by {retail_delta_val:+.1f}%"
            retail_emoji = "📈"
        elif retail_delta_val < -2:
            retail_desc = f"Retail DECREASED longs by {retail_delta_val:+.1f}%"
            retail_emoji = "📉"
        else:
            retail_desc = f"Retail stable ({retail_delta_val:+.1f}%)"
            retail_emoji = "➡️"
        
        # Confirmation analysis
        if whale_confirms is True:
            confirm_emoji = "🟢"
            confirm_color = "#00ff88"
            confirm_text = f"✅ CONFIRMS {final_action}! Whale momentum aligns with trade direction."
        elif whale_confirms is False:
            confirm_emoji = "🔴"
            confirm_color = "#ff6b6b"
            confirm_text = f"⚠️ CONFLICTS with {final_action}! Whale momentum moving opposite to trade."
        else:
            confirm_emoji = "🟡"
            confirm_color = "#ffcc00"
            confirm_text = "Whale momentum neutral - no strong confirmation or conflict."
        
        # Phase alignment
        phase_upper = (phase or '').upper().replace(' ', '_')
        phase_align_text = ""
        if phase_upper in ['ACCUMULATION', 'RE_ACCUMULATION'] and whale_delta > 0:
            phase_align_text = f"✓ Phase ({phase_upper}) + Whale accumulating = ALIGNED"
            phase_align_color = "#00ff88"
        elif phase_upper in ['DISTRIBUTION', 'PROFIT_TAKING'] and whale_delta < 0:
            phase_align_text = f"✓ Phase ({phase_upper}) + Whale distributing = ALIGNED"
            phase_align_color = "#00ff88"
        elif phase_upper in ['ACCUMULATION', 'RE_ACCUMULATION'] and whale_delta < -2:
            phase_align_text = f"⚠️ Phase says {phase_upper} but whales are selling!"
            phase_align_color = "#ff6b6b"
        elif phase_upper in ['DISTRIBUTION', 'PROFIT_TAKING'] and whale_delta > 2:
            phase_align_text = f"⚠️ Phase says {phase_upper} but whales are buying!"
            phase_align_color = "#ffcc00"
        elif phase_upper:
            phase_align_text = f"Phase: {phase_upper} | Whale momentum: {delta_label}"
            phase_align_color = "#888"
        
        delta_story = (
            "<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #f59e0b;'>"
            "<div style='color: #f59e0b; font-weight: bold; font-size: 1.1em; margin-bottom: 12px;'>📊 STEP 4: Are Whales Buying or Selling Recently?</div>"
            "<div style='color: #ccc; line-height: 1.8; margin-bottom: 15px;'>"
            f"Current positioning is one thing, but <b style='color: #f59e0b;'>momentum tells the real story</b>. "
            f"Here's how whale positions changed over the last <b>{delta_lookback}</b>:</div>"
            "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 15px;'>"
            f"<div style='flex: 1; min-width: 180px; background: #0a2a1a; padding: 12px; border-radius: 8px; border: 2px solid {delta_color};'>"
            f"<div style='color: #888; font-size: 0.85em;'>🐋 Whale Change ({delta_lookback})</div>"
            f"<div style='color: {delta_color}; font-size: 1.8em; font-weight: bold;'>{whale_delta:+.1f}%</div>"
            f"<div style='color: {delta_color}; font-weight: bold;'>{delta_emoji} {delta_label}</div>"
            f"<div style='color: #aaa; font-size: 0.85em; margin-top: 5px;'>{delta_desc}</div></div>"
            f"<div style='flex: 1; min-width: 180px; background: #2a1a0a; padding: 12px; border-radius: 8px;'>"
            f"<div style='color: #888; font-size: 0.85em;'>🐑 Retail Change ({delta_lookback})</div>"
            f"<div style='color: #ff9500; font-size: 1.8em; font-weight: bold;'>{retail_delta_val:+.1f}%</div>"
            f"<div style='color: #888;'>{retail_emoji} {retail_desc}</div></div></div>"
            f"<div style='background: #0d0d1a; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid {confirm_color};'>"
            f"<div style='color: {confirm_color}; font-weight: bold; font-size: 1.05em;'>{confirm_emoji} {confirm_text}</div></div>"
        )
        
        if phase_align_text:
            delta_story += (
                f"<div style='background: #0d0d1a; padding: 10px; border-radius: 8px;'>"
                f"<div style='color: {phase_align_color};'>{phase_align_text}</div></div>"
            )
        
        delta_story += "</div>"
        
        delta_summary = f"Δ{whale_delta:+.0f}% ({delta_label})"
        sections.append({'title': 'Whale Momentum', 'html': delta_story, 'summary': delta_summary})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINE ALL SECTIONS - CONCLUSION AT TOP!
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Build conclusion HTML first (this goes at TOP)
    if final_action and 'LONG' in final_action.upper():
        final_color = "#00ff88"
        final_emoji = "🟢"
        final_bg = "#0a2a1a"
    elif final_action and 'SHORT' in final_action.upper():
        final_color = "#ff6b6b"
        final_emoji = "🔴"
        final_bg = "#2a0a0a"
    else:
        final_color = "#ffcc00"
        final_emoji = "⏳"
        final_bg = "#2a2a0a"
    
    # CONCLUSION AT TOP - Clean single-line HTML
    # Build whale badge with delta
    whale_delta_arrow = ""
    whale_delta_color = "#00d4aa"
    if whale_delta is not None:
        if whale_delta > 2:
            whale_delta_arrow = f" <span style='color:#00ff88;'>↑{whale_delta:+.0f}%</span>"
        elif whale_delta < -2:
            whale_delta_arrow = f" <span style='color:#ff6b6b;'>↓{whale_delta:+.0f}%</span>"
        else:
            whale_delta_arrow = f" <span style='color:#888;'>→{whale_delta:+.0f}%</span>"
    
    # Build retail badge with delta
    retail_delta_val = retail_delta if retail_delta is not None else (-whale_delta if whale_delta is not None else None)
    retail_delta_arrow = ""
    if retail_delta_val is not None:
        if retail_delta_val > 2:
            retail_delta_arrow = f" <span style='color:#00ff88;'>↑{retail_delta_val:+.0f}%</span>"
        elif retail_delta_val < -2:
            retail_delta_arrow = f" <span style='color:#ff6b6b;'>↓{retail_delta_val:+.0f}%</span>"
        else:
            retail_delta_arrow = f" <span style='color:#888;'>→{retail_delta_val:+.0f}%</span>"
    
    # Confirmation emoji for the badges
    confirm_emoji = ""
    if whale_confirms is True:
        confirm_emoji = " 🟢"
    elif whale_confirms is False:
        confirm_emoji = " 🔴"
    
    conclusion_html = (
        f"<div style='background: linear-gradient(135deg, {final_bg} 0%, #1a1a2e 100%); border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 2px solid {final_color};'>"
        f"<div style='display: flex; align-items: center; gap: 20px;'>"
        f"<div style='background: {final_color}20; border: 2px solid {final_color}; border-radius: 50%; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center;'>"
        f"<span style='font-size: 2.5em;'>{final_emoji}</span></div>"
        f"<div><div style='color: {final_color}; font-size: 2em; font-weight: bold;'>{final_action or 'WAIT'}</div>"
        f"<div style='color: #ccc; margin-top: 5px; font-size: 1.1em;'>{final_reason or 'Waiting for better setup'}</div></div></div>"
        f"<div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #333; display: flex; flex-wrap: wrap; gap: 10px;'>"
        f"<div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px;'><span style='color: #a855f7;'>🤖 ML:</span> <span style='color: #ccc;'>{ml_summary or 'N/A'}</span></div>"
        f"<div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px;'><span style='color: #00d4ff;'>🌊 Phase:</span> <span style='color: #ccc;'>{phase_summary or 'N/A'}</span></div>"
        f"<div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px;'><span style='color: #00d4aa;'>🐋 Whales:</span> <span style='color: #ccc;'>{whale_pct:.0f}%{whale_delta_arrow}{confirm_emoji}</span></div>"
        f"<div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px;'><span style='color: #ff9500;'>🐑 Retail:</span> <span style='color: #ccc;'>{retail_pct:.0f}%{retail_delta_arrow}</span></div>"
        f"<div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px;'><span style='color: #888;'>📊 TA:</span> <span style='color: #ccc;'>{ta_score:.0f}/100</span></div>"
        f"</div></div>"
        f"<div style='color: #888; font-size: 0.9em; margin-bottom: 15px; padding: 10px; background: #0d0d1a; border-radius: 8px;'>"
        f"📖 <b style='color: #fff;'>How did we get here?</b> Scroll down to see the step-by-step analysis...</div>"
    )
    
    # Combine: Conclusion FIRST, then educational sections
    full_html = conclusion_html + "".join([s['html'] for s in sections])
    
    # One-line summary (include delta if available)
    delta_part = f" | Δ{whale_delta:+.0f}%" if whale_delta is not None else ""
    summary = f"{ml_summary} | {phase_summary} | {whale_summary}{delta_part} → {final_action or 'WAIT'}"
    
    return {
        'story_html': full_html,
        'sections': sections,
        'summary': summary,
        'ml_summary': ml_summary,
        'phase_summary': phase_summary,
        'whale_summary': whale_summary,
        'delta_summary': delta_summary,
        'final_action': final_action,
        'final_reason': final_reason
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MEME COIN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM: Meme coins have different risk profiles:
# - Whales can BE the manipulators (pump & dump)
# - High volatility = stop losses easily triggered
# - No real utility = pure speculation
# - "Whale bullish" might mean "whale setting up dump"
#
# SOLUTION: Detect meme coins and add warnings/reduce trust
#
# ═══════════════════════════════════════════════════════════════════════════════

# Known meme coin name patterns (add more as needed)
MEME_COIN_PATTERNS = [
    # Classic memes
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'BABYDOGE',
    # Animal memes
    'CAT', 'DOG', 'NEKO', 'INU', 'SNEK', 'FROG', 'APE', 'MONKEY',
    # Fun/Joke patterns
    'MOON', 'SAFE', 'ELON', 'WOJAK', 'CHAD', 'BASED', 'COPE', 'HOPPY',
    'TURBO', 'LADYS', 'BITCOIN', 'ANDY', 'BRETT', 'PONKE', 'POPCAT',
    # Giggle-type names (silly/funny)
    'GIGGLE', 'LOL', 'LMAO', 'ROFL', 'KEKW', 'HAHA', 'JOKE', 'FUN',
    # Food memes
    'PIZZA', 'BURGER', 'SUSHI', 'TACO',
    # Other common patterns
    'TRUMP', 'BIDEN', 'MAGA', 'CUMMIES', 'CUM', 'ASS', 'POOP', 'SHIT',
    # AI meme coins
    'GOAT', 'ACT', 'FARTCOIN', 'ZEREBRO',
]

# Patterns that suggest NOT a meme (real projects)
LEGIT_PROJECT_PATTERNS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'LINK',
    'MATIC', 'UNI', 'AAVE', 'MKR', 'SNX', 'COMP', 'CRV', 'LDO',
    'ARB', 'OP', 'ATOM', 'NEAR', 'APT', 'SUI', 'SEI', 'INJ', 'TIA',
    'FET', 'RNDR', 'AR', 'FIL', 'OCEAN', 'GRT', 'API3',
]


def is_meme_coin(symbol: str) -> dict:
    """
    Detect if a symbol is likely a meme coin.
    
    Returns:
        dict with:
        - is_meme: bool
        - confidence: HIGH/MEDIUM/LOW
        - reason: why we think it's a meme
        - risk_level: EXTREME/HIGH/MODERATE
    """
    if not symbol:
        return {'is_meme': False, 'confidence': 'LOW', 'reason': '', 'risk_level': 'MODERATE'}
    
    # Clean symbol
    symbol_upper = symbol.upper().replace('USDT', '').replace('USD', '').replace('PERP', '')
    
    # Check if it's a known legit project
    for legit in LEGIT_PROJECT_PATTERNS:
        if symbol_upper == legit:
            return {
                'is_meme': False, 
                'confidence': 'HIGH', 
                'reason': 'Known legitimate project',
                'risk_level': 'MODERATE'
            }
    
    # Check against meme patterns
    for meme_pattern in MEME_COIN_PATTERNS:
        if meme_pattern in symbol_upper:
            return {
                'is_meme': True,
                'confidence': 'HIGH',
                'reason': f"Name contains '{meme_pattern}' - known meme pattern",
                'risk_level': 'EXTREME'
            }
    
    # Heuristics for unknown coins
    # Short silly names are often memes
    if len(symbol_upper) <= 4 and symbol_upper not in LEGIT_PROJECT_PATTERNS:
        # Could be meme, but not sure
        return {
            'is_meme': False,
            'confidence': 'LOW',
            'reason': 'Unknown short name - use caution',
            'risk_level': 'HIGH'
        }
    
    return {
        'is_meme': False,
        'confidence': 'MEDIUM',
        'reason': '',
        'risk_level': 'MODERATE'
    }


def get_meme_coin_warnings(symbol: str, whale_pct: float, price_change: float) -> list:
    """
    Get specific warnings for meme coins.
    """
    warnings = []
    meme_check = is_meme_coin(symbol)
    
    if meme_check['is_meme']:
        warnings.append(f"🎰 MEME COIN DETECTED: {meme_check['reason']}")
        warnings.append("⚠️ MEME RISK: Whales may BE the manipulators (pump & dump)")
        warnings.append("⚠️ MEME RISK: High volatility = stops easily hunted")
        warnings.append("💡 MEME TIP: Use smaller position size, expect manipulation")
        
        # Additional warnings based on conditions
        if whale_pct >= 70:
            warnings.append("🚨 DANGER: High whale % on meme = potential dump setup!")
        
        if abs(price_change) > 10:
            warnings.append(f"🚨 EXTREME VOLATILITY: {price_change:+.1f}% move - manipulation likely!")
    
    elif meme_check['risk_level'] == 'HIGH':
        warnings.append(f"⚠️ UNKNOWN COIN: {meme_check['reason']} - trade with caution")
    
    return warnings

@dataclass
class TradeDecision:
    """Complete trading decision with all context - THE ONLY OUTPUT NEEDED"""
    
    # Primary outputs
    action: str              # What to do: STRONG_LONG, WAIT, etc.
    trade_direction: str     # LONG, SHORT, or WAIT
    confidence: str          # HIGH, MEDIUM, LOW
    
    # Scores (0-100 scale)
    direction_score: int     # 0-40 points (whale conviction)
    squeeze_score: int       # 0-30 points (divergence/trap)
    entry_score: int         # 0-30 points (position + TA)
    total_score: int         # 0-100 points (THE score)
    
    # Labels for display
    direction_label: str     # BULLISH, LEAN_BULLISH, etc.
    squeeze_label: str       # SQUEEZE, SLIGHT_EDGE, TRAP, etc.
    position_label: str      # EARLY, MIDDLE, LATE
    oi_signal: str           # What OI+Price tells us
    
    # Explanations
    main_reason: str         # Primary reason for decision
    warnings: List[str]      # Any warnings/cautions
    story: str               # Full narrative explanation
    
    # Validity flags
    is_valid_long: bool      # Can we go long?
    is_valid_short: bool     # Can we go short?
    invalidation_reason: str # Why invalid (if applicable)
    
    # Trap detection
    is_trap: bool = False           # Is a trap detected?
    trap_type: str = ""             # LONG_TRAP or SHORT_TRAP
    is_short_trap: bool = False     # Specifically SHORT_TRAP (retail shorts trapped)
    
    # Stories for Combined Learning
    oi_story: str = ""           # What OI is telling us
    whale_story: str = ""        # What whales are doing  
    position_story: str = ""     # Entry timing analysis
    market_story: str = ""       # Market conditions (Fear/Greed, BTC, Liquidity)
    historical_story: str = ""   # Historical performance of similar setups
    conclusion: str = ""         # Final conclusion text
    conclusion_action: str = ""  # Action text (LONG NOW, WAIT, etc.)
    
    # Market Conditions
    fear_greed: int = 50         # Fear & Greed index used
    market_modifier: int = 0     # Score adjustment from market conditions
    
    # Historical Validation
    historical_win_rate: float = None   # Win rate of similar setups
    historical_sample_size: int = 0     # Number of similar setups found
    historical_modifier: int = 0        # Score adjustment from history
    
    # Pattern Detection (Deep Learning - Optional Guidance)
    pattern_detected: str = None        # Pattern name if detected (e.g., "Bull Flag")
    pattern_confidence: float = 0.0     # Pattern detection confidence (0-100)
    pattern_aligns: bool = False        # True if pattern direction matches trade direction
    pattern_story: str = ""             # Pattern explanation


# ═══════════════════════════════════════════════════════════════════════════════
# OI + PRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_oi_price(oi_change: float, price_change: float) -> Tuple[str, str, int]:
    """
    Analyze OI and Price combination.
    
    Returns: (signal, explanation, score_modifier)
    
    ╔═══════════════╦═══════════════╦════════════════════════════════════════╗
    ║ OI Change     ║ Price Change  ║ Meaning                                ║
    ╠═══════════════╬═══════════════╬════════════════════════════════════════╣
    ║ ↑ Rising      ║ ↑ Rising      ║ New LONGS entering (BULLISH)           ║
    ║ ↑ Rising      ║ ↓ Falling     ║ New SHORTS entering (BEARISH)          ║
    ║ ↓ Falling     ║ ↑ Rising      ║ SHORTS closing - DON'T CHASE!          ║
    ║ ↓ Falling     ║ ↓ Falling     ║ LONGS exiting/liquidating (BEARISH)    ║
    ╚═══════════════╩═══════════════╩════════════════════════════════════════╝
    
    CRITICAL INSIGHT:
    SHORT COVERING (OI↓ + Price↑) is the #1 trap for retail traders!
    They see price going up and FOMO in, becoming exit liquidity.
    """
    
    oi_rising = oi_change > T.OI_MODERATE
    oi_falling = oi_change < -T.OI_MODERATE
    price_rising = price_change > T.PRICE_SIGNIFICANT
    price_falling = price_change < -T.PRICE_SIGNIFICANT
    
    # SHORT COVERING detection thresholds
    # These are CRITICAL patterns that override other signals
    oi_significant_drop = oi_change < -2.0    # Significant outflow
    oi_any_decrease = oi_change < -0.5        # Any meaningful outflow
    price_any_increase = price_change > 0.3   # Any meaningful up move
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 1: SHORT COVERING - Most important pattern to detect!
    # ═══════════════════════════════════════════════════════════════════════
    
    if oi_significant_drop and price_any_increase:
        # STRONG SHORT COVERING - OI dropping significantly while price up
        # This is NOT real demand - shorts are being forced to close
        return "SHORT_COVERING", f"🚨 Shorts covering (OI {oi_change:+.1f}%) - NOT new buyers! Don't chase!", -10
    
    if oi_any_decrease and price_rising:
        # MODERATE SHORT COVERING - OI down while price clearly rising
        return "SHORT_COVERING", f"⚠️ Short covering rally (OI {oi_change:+.1f}%) - Rally may fade", -6
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRIORITY 2: Standard OI + Price patterns
    # ═══════════════════════════════════════════════════════════════════════
    
    if oi_rising and price_rising:
        return "STRONG_BULLISH", "New longs entering - Money flowing IN with price UP", 4
    
    elif oi_rising and price_falling:
        return "BEARISH", "New shorts entering - Betting against price", -2
    
    elif oi_falling and price_falling:
        return "STRONG_BEARISH", "Longs exiting - Money flowing OUT with price DOWN", -4
    
    else:
        return "NEUTRAL", "No significant OI/Price signal", 0


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION IN RANGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_position(position_pct: float) -> Tuple[str, int]:
    """
    Determine position in range and score modifier.
    
    EARLY (0-35%): Good for longs, bad for shorts
    MIDDLE (35-65%): Wait for better entry
    LATE (65-100%): Good for shorts, bad for longs
    """
    
    if position_pct <= T.EARLY_MAX:
        return "EARLY", 8  # Bonus for good entry
    elif position_pct >= T.LATE_MIN:
        return "LATE", -4  # Penalty for late entry on longs
    else:
        return "MIDDLE", 0  # No bonus/penalty


# ═══════════════════════════════════════════════════════════════════════════════
# WHALE vs RETAIL ANALYSIS (THE CORE LOGIC)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_whale_retail(whale_pct: float, retail_pct: float) -> dict:
    """
    THE MOST IMPORTANT FUNCTION
    
    Analyzes whale vs retail positioning to determine:
    1. Direction (bullish/bearish)
    2. Conviction (high/medium/low)
    3. Squeeze potential
    4. Trap risk
    """
    
    divergence = whale_pct - retail_pct
    
    result = {
        'whale_pct': whale_pct,
        'retail_pct': retail_pct,
        'divergence': divergence,
        'direction': 'NEUTRAL',
        'confidence': 'LOW',
        'direction_score': 15,
        'squeeze_type': 'NONE',
        'squeeze_score': 0,
        'is_trap': False,
        'trap_type': None,
        'is_short_trap': False,  # NEW: Specifically for SHORT_TRAP
        'warnings': [],
        'reason': ''
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 1: CHECK FOR RETAIL TRAP (Retail > Whale)
    # This is checked FIRST because it invalidates long setups
    # ANY amount of retail > whale is a warning!
    # ═══════════════════════════════════════════════════════════════════════
    
    if divergence < 0:  # Retail more bullish than whales by ANY amount
        result['is_trap'] = True
        result['trap_type'] = 'LONG_TRAP'
        result['warnings'].append(f"⚠️ RETAIL TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%")
        
        if divergence <= -15:
            result['squeeze_type'] = 'EXTREME_TRAP'
            result['squeeze_score'] = -15
            result['reason'] = f"🚨 EXTREME TRAP: Retail overleveraged by {abs(divergence):.0f}%"
        elif divergence <= -10:
            result['squeeze_type'] = 'HIGH_TRAP'
            result['squeeze_score'] = -10
            result['reason'] = f"⚠️ HIGH TRAP RISK: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        elif divergence <= -5:
            result['squeeze_type'] = 'MODERATE_TRAP'
            result['squeeze_score'] = -6
            result['reason'] = f"⚠️ TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        else:
            result['squeeze_type'] = 'SLIGHT_TRAP'
            result['squeeze_score'] = -3
            result['reason'] = f"⚠️ Caution: Retail ({retail_pct:.0f}%) slightly > Whales ({whale_pct:.0f}%)"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 2: CHECK FOR SHORT_TRAP (Whale >> Retail when whales bullish)
    # When whales are significantly MORE bullish than retail:
    #   - Retail is SHORT (bearish positioning)
    #   - Whales are LONG (bullish positioning) 
    #   - Retail shorts WILL BE SQUEEZED = SHORT_TRAP
    #
    # This is the OPPOSITE of LONG_TRAP and should trigger a LONG signal!
    # ═══════════════════════════════════════════════════════════════════════
    
    elif divergence >= 15 and whale_pct >= 65:
        # HIGH SHORT_TRAP: Whales very bullish, retail very bearish
        result['is_trap'] = True
        result['trap_type'] = 'SHORT_TRAP'
        result['is_short_trap'] = True  # NEW: Explicit flag
        result['squeeze_type'] = 'SHORT_TRAP_HIGH'
        result['squeeze_score'] = 25  # High score - this is bullish!
        result['reason'] = f"SHORT TRAP: Retail shorts ({retail_pct:.0f}%) trapped by whale longs ({whale_pct:.0f}%)!"
        result['warnings'].append(f"SHORT_TRAP: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% = Retail shorts will be SQUEEZED!")
        
    elif divergence >= 10 and whale_pct >= 60:
        # MODERATE SHORT_TRAP
        result['is_trap'] = True
        result['trap_type'] = 'SHORT_TRAP'
        result['is_short_trap'] = True  # NEW: Explicit flag
        result['squeeze_type'] = 'SHORT_TRAP_MODERATE'
        result['squeeze_score'] = 18
        result['reason'] = f"SHORT TRAP forming: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%"
        result['warnings'].append(f"SHORT_TRAP: Retail shorts may be squeezed!")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 3: CHECK FOR SQUEEZE (Whale > Retail, regular squeeze)
    # Whales positioned, retail not = squeeze incoming (but not SHORT_TRAP level)
    # ═══════════════════════════════════════════════════════════════════════
    
    elif divergence >= 15:
        result['squeeze_type'] = 'HIGH_SQUEEZE'
        result['squeeze_score'] = 20
        result['reason'] = f"SQUEEZE: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%"
        
    elif divergence >= 10:
        result['squeeze_type'] = 'MODERATE_SQUEEZE'
        result['squeeze_score'] = 14
        result['reason'] = f"Squeeze potential: {divergence:.0f}% divergence"
        
    elif divergence >= 5:
        result['squeeze_type'] = 'SLIGHT_EDGE'
        result['squeeze_score'] = 8
        result['reason'] = f"Slight edge: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%"
    
    else:
        result['squeeze_type'] = 'NO_EDGE'
        result['squeeze_score'] = 0
        result['reason'] = "No significant whale/retail divergence"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 3: WHALE CONVICTION (Direction)
    # ═══════════════════════════════════════════════════════════════════════
    
    if whale_pct >= T.EXTREME_BULLISH:
        result['direction'] = 'BULLISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 40
        
    elif whale_pct >= T.STRONG_BULLISH:
        result['direction'] = 'BULLISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 36
        
    elif whale_pct >= T.BULLISH:
        result['direction'] = 'BULLISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 32
        
    elif whale_pct >= T.LEAN_BULLISH:
        result['direction'] = 'LEAN_BULLISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 28
        
    elif whale_pct <= T.EXTREME_BEARISH:
        result['direction'] = 'BEARISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 40
        
    elif whale_pct <= T.STRONG_BEARISH:
        result['direction'] = 'BEARISH'
        result['confidence'] = 'HIGH'
        result['direction_score'] = 36
        
    elif whale_pct <= T.BEARISH:
        result['direction'] = 'BEARISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 32
        
    elif whale_pct <= T.LEAN_BEARISH:
        result['direction'] = 'LEAN_BEARISH'
        result['confidence'] = 'MEDIUM'
        result['direction_score'] = 28
        
    else:
        result['direction'] = 'NEUTRAL'
        result['confidence'] = 'LOW'
        result['direction_score'] = 15
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def check_special_scenarios(
    whale_pct: float, 
    retail_pct: float, 
    position: str,
    oi_signal: str
) -> Optional[dict]:
    """
    Check for special market conditions:
    - CAPITULATION: Everyone bearish at lows → Reversal long
    - EUPHORIA: Everyone bullish at highs → Reversal short
    """
    
    # CAPITULATION: Both whale and retail bearish at EARLY position
    # This is contrarian but at LOWS with everyone bearish = potential reversal
    if whale_pct <= T.BEARISH and retail_pct <= T.BEARISH and position == "EARLY":
        return {
            'scenario': 'CAPITULATION',
            'action': 'LONG NOW',
            'trade_direction': 'LONG',
            'confidence': 'HIGH',
            'score_bonus': 15,
            'reason': 'Everyone gave up at lows - reversal likely'
        }
    
    # EUPHORIA/DISTRIBUTION: Retail FOMO while whales are NOT as bullish
    # PREDICTIVE: Whales positioning to exit/short while retail buys the top
    # Key: Retail > Whale at LATE position = Retail trap!
    if retail_pct >= T.STRONG_BULLISH and whale_pct < retail_pct and position == "LATE":
        # Retail more bullish than whales at highs = TRAP
        return {
            'scenario': 'RETAIL_FOMO_TOP',
            'action': 'SHORT SETUP',
            'trade_direction': 'SHORT',
            'confidence': 'HIGH',
            'score_bonus': 12,
            'reason': f'Retail FOMO ({retail_pct:.0f}%) > Whales ({whale_pct:.0f}%) at highs - Distribution likely'
        }
    
    # EXTREME EUPHORIA: Both very bullish at highs - be cautious
    # This is less predictive, so lower confidence
    if whale_pct >= T.BULLISH and retail_pct >= T.BULLISH and position == "LATE":
        return {
            'scenario': 'EUPHORIA',
            'action': 'CAUTION',
            'trade_direction': 'WAIT',  # Not SHORT - whales still bullish!
            'confidence': 'MEDIUM',
            'score_bonus': 0,
            'reason': 'Everyone bullish at highs - wait for whale exit signal'
        }
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER DECISION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_trade_decision(
    whale_pct: float,
    retail_pct: float,
    oi_change: float = 0,
    price_change: float = 0,
    position_pct: float = 50,
    ta_score: float = 50,
    money_flow_phase: str = "UNKNOWN",
    # NEW: Symbol for meme coin detection
    symbol: str = "",                 # Symbol name (e.g., "BTCUSDT", "GIGGLEUSDT")
    # NEW: Market Conditions
    fear_greed: int = 50,           # Fear & Greed index (0-100)
    btc_change_24h: float = 0,      # BTC 24h change %
    is_options_expiry: bool = False, # Is today options expiry?
    is_holiday: bool = False,        # Holiday/weekend thin liquidity?
    # NEW: Historical Validation (input from database lookup)
    historical_win_rate: float = None,    # Win rate from similar setups (0-100)
    historical_sample_size: int = 0,      # Number of similar historical setups
    # NEW: SMC Level Proximity (from Pulse Flow / SMC detector)
    at_bullish_ob: bool = False,     # Price is AT a bullish Order Block
    at_bearish_ob: bool = False,     # Price is AT a bearish Order Block
    near_bullish_ob: bool = False,   # Price is NEAR a bullish Order Block (within 2%)
    near_bearish_ob: bool = False,   # Price is NEAR a bearish Order Block (within 2%)
    at_support: bool = False,        # Price is at key support level
    at_resistance: bool = False      # Price is at key resistance level
) -> TradeDecision:
    """
    ═══════════════════════════════════════════════════════════════════════════
    THE MASTER DECISION FUNCTION
    ═══════════════════════════════════════════════════════════════════════════
    
    This is the ONLY function that should be called for trading decisions.
    ALL other code paths should use this.
    
    Args:
        whale_pct: Whale long percentage (0-100)
        retail_pct: Retail long percentage (0-100)
        oi_change: Open Interest change % (24h)
        price_change: Price change % (24h)
        position_pct: Position in range (0=low, 100=high)
        ta_score: Technical analysis score (0-100)
        money_flow_phase: ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
        fear_greed: Fear & Greed index (0-100, lower = more fear)
        btc_change_24h: BTC 24h price change %
        is_options_expiry: True if major options expiry today
        is_holiday: True if holiday/weekend (thin liquidity)
        historical_win_rate: Win rate from similar historical setups (0-100)
        historical_sample_size: Number of similar setups in database
        at_bullish_ob: Price is AT a bullish Order Block (best long entry)
        at_bearish_ob: Price is AT a bearish Order Block (best short entry)
        near_bullish_ob: Price is NEAR bullish OB within 2% (good long entry)
        near_bearish_ob: Price is NEAR bearish OB within 2% (good short entry)
        at_support: Price is at key support level
        at_resistance: Price is at key resistance level
    
    Returns:
        TradeDecision with complete analysis
    
    MARKET CONDITIONS LOGIC:
    - EXTREME FEAR + Whales Buying = STRONG BUY (smart money accumulating)
    - EXTREME FEAR + Whales Selling = WAIT (capitulation not over)
    - EXTREME GREED + Whales Selling = STRONG SHORT (distribution)
    - EXTREME GREED + Whales Buying = CAUTION (late/FOMO trap)
    - Options Expiry = Reduce confidence (expect volatility)
    - Holiday/Weekend = Reduce confidence (thin liquidity)
    
    HISTORICAL VALIDATION LOGIC:
    - Win rate >= 70% with 15+ samples = +12 pts (proven setup!)
    - Win rate >= 60% with 15+ samples = +8 pts
    - Win rate <= 40% with 15+ samples = -15 pts (AVOID!)
    - Less than 5 samples = no adjustment (insufficient data)
    
    SMC LEVEL LOGIC (NEW!):
    - LATE position + at_bullish_ob = OVERRIDE late penalty for longs (good entry!)
    - EARLY position + at_bearish_ob = OVERRIDE early penalty for shorts (good entry!)
    - Being at OB/support/resistance trumps raw position percentage
    """
    
    warnings = []
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ABSOLUTE RULES - These override EVERYTHING else
    # Philosophy: Trade when you have EDGE. Wait when you don't.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    is_ranging_structure = money_flow_phase.upper() in ['PROFIT_TAKING', 'PROFIT TAKING', 'UNKNOWN', 'NEUTRAL']
    is_middle_position = 30 <= position_pct <= 70
    is_late_position = position_pct > 70
    is_retail_trap = retail_pct > whale_pct and retail_pct > 55
    
    # ABSOLUTE RULE 1: Never trade mid-range in unclear market
    if is_ranging_structure and is_middle_position:
        return TradeDecision(
            action="WAIT",
            trade_direction="WAIT",
            confidence="HIGH",
            direction_score=0,
            squeeze_score=0,
            entry_score=0,
            total_score=40,
            direction_label="NEUTRAL",
            squeeze_label="NO_EDGE",
            position_label="MIDDLE",
            oi_signal="NEUTRAL",
            main_reason=f"Mid-range ({position_pct:.0f}%) in unclear market - NO EDGE",
            warnings=["Never trade middle of range", "Wait for breakout or range edge"],
            story=f"Position is at {position_pct:.0f}% (middle) with {money_flow_phase} money flow. No clear edge.",
            is_valid_long=False,
            is_valid_short=False,
            invalidation_reason="Mid-range in unclear market",
            conclusion=f"WAIT - Mid-range ({position_pct:.0f}%) with {money_flow_phase}",
            conclusion_action="WAIT for breakout or edge of range",
        )
    
    # ABSOLUTE RULE 2: Never chase late positions (>70%)
    if is_late_position and not (at_bearish_ob or at_resistance):
        return TradeDecision(
            action="WAIT",
            trade_direction="WAIT",
            confidence="HIGH",
            direction_score=0,
            squeeze_score=0,
            entry_score=0,
            total_score=35,
            direction_label="NEUTRAL",
            squeeze_label="NO_EDGE",
            position_label="LATE",
            oi_signal="NEUTRAL",
            main_reason=f"Late position ({position_pct:.0f}%) - Don't chase!",
            warnings=["Position too late to enter", "Wait for pullback"],
            story=f"Position is at {position_pct:.0f}% of the move. Chasing late = poor R:R.",
            is_valid_long=False,
            is_valid_short=False,
            invalidation_reason="Late position - don't chase",
            conclusion=f"WAIT - Position {position_pct:.0f}% is too late",
            conclusion_action="WAIT for pullback",
        )
    
    # ABSOLUTE RULE 3: Never trade retail traps
    if is_retail_trap:
        return TradeDecision(
            action="WAIT",
            trade_direction="WAIT",
            confidence="HIGH",
            direction_score=0,
            squeeze_score=0,
            entry_score=0,
            total_score=30,
            direction_label="NEUTRAL",
            squeeze_label="TRAP",
            position_label="MIDDLE" if is_middle_position else ("LATE" if is_late_position else "EARLY"),
            oi_signal="RETAIL_TRAP",
            main_reason=f"RETAIL TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%",
            warnings=["Retail trap detected", "Smart money using retail as exit liquidity"],
            story=f"Retail positioning ({retail_pct:.0f}%) exceeds whale positioning ({whale_pct:.0f}%). Classic trap.",
            is_valid_long=False,
            is_valid_short=False,
            invalidation_reason="Retail trap",
            conclusion=f"RETAIL TRAP - Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%",
            conclusion_action="WAIT - Don't be exit liquidity",
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 0: MEME COIN DETECTION (CRITICAL RISK CHECK!)
    # ═══════════════════════════════════════════════════════════════════════
    # Meme coins have different risk profiles:
    # - Whales can BE the manipulators (pump & dump)
    # - High volatility = stop losses easily triggered
    # - "Whale bullish" might mean "whale setting up dump"
    # ═══════════════════════════════════════════════════════════════════════
    
    meme_check = is_meme_coin(symbol)
    is_meme = meme_check['is_meme']
    meme_risk_level = meme_check['risk_level']
    
    if is_meme:
        # Add all meme coin warnings
        meme_warnings = get_meme_coin_warnings(symbol, whale_pct, price_change)
        warnings.extend(meme_warnings)
    elif meme_risk_level == 'HIGH':
        warnings.append(f"⚠️ UNKNOWN COIN: Trade with caution - {meme_check['reason']}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Analyze Whale vs Retail
    # ═══════════════════════════════════════════════════════════════════════
    
    wr_analysis = analyze_whale_retail(whale_pct, retail_pct)
    warnings.extend(wr_analysis['warnings'])
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Analyze OI + Price
    # ═══════════════════════════════════════════════════════════════════════
    
    oi_signal, oi_explanation, oi_modifier = analyze_oi_price(oi_change, price_change)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Get Position in Range
    # ═══════════════════════════════════════════════════════════════════════
    
    position_label, position_modifier = get_position(position_pct)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3.5: Analyze Market Conditions (NEW!)
    # Fear/Greed + BTC trend + Options expiry + Liquidity
    # ═══════════════════════════════════════════════════════════════════════
    
    market_modifier = 0
    market_warnings = []
    market_story_parts = []
    
    # Determine whale direction for fear/greed logic
    whale_direction = 'BULLISH' if whale_pct >= 60 else ('BEARISH' if whale_pct <= 40 else 'NEUTRAL')
    
    # === FEAR & GREED ANALYSIS ===
    # This is the Warren Buffett logic: "Be greedy when others are fearful"
    
    if fear_greed <= 25:  # EXTREME FEAR
        if whale_direction == 'BULLISH':
            # Smart money buying while retail panics = GOLDEN OPPORTUNITY
            market_modifier += 10
            market_story_parts.append(f"🟢 EXTREME FEAR ({fear_greed}) + Whales BUYING = Smart money accumulation!")
        elif whale_direction == 'BEARISH':
            # Everyone selling including whales = capitulation not over
            market_modifier -= 10
            market_warnings.append(f"⚠️ EXTREME FEAR + Whales also selling = Capitulation")
            market_story_parts.append(f"🔴 EXTREME FEAR ({fear_greed}) + Whales SELLING = Wait for capitulation to end")
        else:
            market_story_parts.append(f"😱 EXTREME FEAR ({fear_greed}) - Watch for whale accumulation")
            
    elif fear_greed <= 45:  # FEAR
        if whale_direction == 'BULLISH':
            market_modifier += 5
            market_story_parts.append(f"🟢 FEAR ({fear_greed}) + Whales bullish = Accumulation zone")
        elif whale_direction == 'BEARISH':
            market_modifier -= 5
            market_story_parts.append(f"⚠️ FEAR ({fear_greed}) + Whales bearish = More downside likely")
            
    elif fear_greed >= 75:  # EXTREME GREED
        if whale_direction == 'BEARISH':
            # Smart money selling while retail is greedy = TOP SIGNAL
            market_modifier += 5  # Good for shorts
            market_story_parts.append(f"🔴 EXTREME GREED ({fear_greed}) + Whales SELLING = Distribution/Top signal!")
        elif whale_direction == 'BULLISH':
            # Even whales bullish at extreme greed = FOMO trap risk
            market_modifier -= 10
            market_warnings.append(f"⚠️ EXTREME GREED + Whales bullish = Late stage, FOMO trap risk")
            market_story_parts.append(f"⚠️ EXTREME GREED ({fear_greed}) - Even whales bullish could be FOMO trap")
        else:
            market_modifier -= 5
            market_story_parts.append(f"🤑 EXTREME GREED ({fear_greed}) - Market likely overextended")
            
    elif fear_greed >= 55:  # GREED
        if whale_direction == 'BEARISH':
            market_modifier += 3  # Slight edge for shorts
            market_story_parts.append(f"📊 GREED ({fear_greed}) + Whales bearish = Distribution phase")
        elif whale_direction == 'BULLISH':
            market_modifier -= 3
            market_warnings.append(f"⚠️ GREED ({fear_greed}) + Whales bullish = Late stage entry")
            market_story_parts.append(f"⚠️ GREED ({fear_greed}) - Be careful chasing longs")
    else:
        # NEUTRAL (45-55)
        market_story_parts.append(f"😐 Neutral sentiment ({fear_greed})")
    
    # === BTC TREND CHECK ===
    # Altcoins follow BTC - if BTC dumps, everything dumps
    
    if btc_change_24h <= -5:
        market_modifier -= 10
        market_warnings.append(f"🔴 BTC DUMP: {btc_change_24h:+.1f}% - Altcoins will follow!")
        market_story_parts.append(f"🔴 BTC crashing {btc_change_24h:+.1f}% - Risk off for all alts")
    elif btc_change_24h <= -3:
        market_modifier -= 5
        market_warnings.append(f"⚠️ BTC down {btc_change_24h:+.1f}%")
        market_story_parts.append(f"⚠️ BTC weak {btc_change_24h:+.1f}% - Alts may follow")
    elif btc_change_24h >= 5:
        market_modifier += 5
        market_story_parts.append(f"🟢 BTC pumping {btc_change_24h:+.1f}% - Risk on environment")
    elif btc_change_24h >= 3:
        market_modifier += 3
        market_story_parts.append(f"🟢 BTC strong {btc_change_24h:+.1f}%")
    
    # === OPTIONS EXPIRY CHECK ===
    if is_options_expiry:
        market_modifier -= 10
        market_warnings.append("⚠️ MAJOR OPTIONS EXPIRY - Expect HIGH volatility!")
        market_story_parts.append("📅 Options expiry day - Expect wild swings, consider waiting")
    
    # === LIQUIDITY CHECK ===
    if is_holiday:
        market_modifier -= 10
        market_warnings.append("⚠️ HOLIDAY/WEEKEND - Thin liquidity, wild swings possible!")
        market_story_parts.append("💧 Thin liquidity - Prices can move on low volume")
    
    # Build final market story
    market_story = " | ".join(market_story_parts) if market_story_parts else "Market conditions neutral"
    
    # Add market warnings to main warnings
    warnings.extend(market_warnings)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3.6: Historical Validation Adjustment (NEW!)
    # Uses actual historical performance of similar setups
    # ═══════════════════════════════════════════════════════════════════════
    
    historical_modifier = 0
    historical_story = ""
    
    if historical_win_rate is not None and historical_sample_size >= 5:
        # We have enough historical data to trust
        
        if historical_sample_size >= 15:
            # HIGH confidence in historical data
            if historical_win_rate >= 70:
                historical_modifier = +12
                historical_story = f"🟢 PROVEN SETUP: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples) - Historically excellent!"
            elif historical_win_rate >= 60:
                historical_modifier = +8
                historical_story = f"🟢 Good history: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples)"
            elif historical_win_rate >= 50:
                historical_modifier = +3
                historical_story = f"📊 Decent history: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples)"
            elif historical_win_rate >= 40:
                historical_modifier = -5
                historical_story = f"⚠️ Below average: {historical_win_rate:.0f}% win rate ({historical_sample_size} samples)"
            else:
                historical_modifier = -15
                historical_story = f"🔴 POOR HISTORY: Only {historical_win_rate:.0f}% win rate ({historical_sample_size} samples) - AVOID THIS SETUP!"
                warnings.append(f"🔴 Historical win rate only {historical_win_rate:.0f}% - consider skipping")
                
        elif historical_sample_size >= 10:
            # MEDIUM confidence
            if historical_win_rate >= 70:
                historical_modifier = +8
                historical_story = f"🟢 Good track record: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            elif historical_win_rate >= 55:
                historical_modifier = +4
                historical_story = f"📊 Positive history: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            elif historical_win_rate <= 40:
                historical_modifier = -10
                historical_story = f"🔴 Poor history: {historical_win_rate:.0f}% ({historical_sample_size} samples) - Caution!"
                warnings.append(f"⚠️ Similar setups only {historical_win_rate:.0f}% win rate")
            else:
                historical_story = f"📊 Mixed results: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
                
        else:  # 5-9 samples
            # LOW confidence - smaller adjustments
            if historical_win_rate >= 75:
                historical_modifier = +5
                historical_story = f"📊 Limited data positive: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            elif historical_win_rate <= 35:
                historical_modifier = -8
                historical_story = f"⚠️ Limited data negative: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
            else:
                historical_story = f"📊 Limited data: {historical_win_rate:.0f}% ({historical_sample_size} samples)"
    
    elif historical_sample_size > 0 and historical_sample_size < 5:
        historical_story = f"📊 Insufficient history: Only {historical_sample_size} similar setups found"
    else:
        historical_story = "📊 No historical data for this exact setup"
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Check Special Scenarios
    # ═══════════════════════════════════════════════════════════════════════
    
    special = check_special_scenarios(whale_pct, retail_pct, position_label, oi_signal)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Calculate Entry Score (SMC-AWARE!)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Determine if price is at a key SMC level
    at_bullish_level = at_bullish_ob or near_bullish_ob or at_support
    at_bearish_level = at_bearish_ob or near_bearish_ob or at_resistance
    
    # TA contributes up to 15 points
    ta_contribution = int((ta_score / 100) * 15)
    
    # Position contributes up to 15 points - BUT SMC levels can OVERRIDE!
    # 
    # KEY INSIGHT: Position % alone doesn't tell the full story.
    # Being LATE (86%) but AT a bullish Order Block is actually a GOOD entry!
    # The OB provides support that raw position % doesn't account for.
    #
    smc_override = False
    smc_override_reason = ""
    
    if position_label == "EARLY":
        position_contribution = 15
    elif position_label == "MIDDLE":
        position_contribution = 8
    else:  # LATE
        position_contribution = 3  # Default late penalty
        
        # SMC OVERRIDE: If LATE but at bullish OB/support → Good entry for longs!
        if at_bullish_level:
            position_contribution = 12  # Upgrade from 3 to 12
            smc_override = True
            if at_bullish_ob:
                smc_override_reason = "LATE but AT bullish OB - good entry!"
            elif near_bullish_ob:
                smc_override_reason = "LATE but NEAR bullish OB - decent entry"
            else:
                smc_override_reason = "LATE but AT support level"
    
    # For shorts: EARLY position at bearish OB is actually good
    # (We'll check direction later and apply this bonus)
    short_entry_bonus = 0
    if position_label == "EARLY" and at_bearish_level:
        short_entry_bonus = 8  # Bonus for shorts when at bearish OB at lows
        if at_bearish_ob:
            smc_override_reason = "EARLY but AT bearish OB - good short entry!"
        elif near_bearish_ob:
            smc_override_reason = "EARLY but NEAR bearish OB - decent short"
    
    entry_score = ta_contribution + position_contribution
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Calculate Total Score
    # ═══════════════════════════════════════════════════════════════════════
    
    direction_score = wr_analysis['direction_score']
    squeeze_score = max(0, wr_analysis['squeeze_score'])  # Don't go negative for total
    
    # ═══════════════════════════════════════════════════════════════════════
    # MEME COIN PENALTY
    # Reduce trust in scores for meme coins - higher manipulation risk
    # ═══════════════════════════════════════════════════════════════════════
    meme_modifier = 0
    if is_meme:
        meme_modifier = -15  # Significant penalty for meme coins
        warnings.append(f"📉 MEME PENALTY: -15 points applied (manipulation risk)")
    elif meme_risk_level == 'HIGH':
        meme_modifier = -8  # Smaller penalty for unknown coins
    
    # Include market_modifier AND meme_modifier in total score
    total_score = direction_score + squeeze_score + entry_score + oi_modifier + market_modifier + historical_modifier + meme_modifier
    total_score = max(0, min(100, total_score))
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 7: DETERMINE FINAL ACTION
    # This is where ALL the rules come together
    # ═══════════════════════════════════════════════════════════════════════
    
    direction = wr_analysis['direction']
    confidence = wr_analysis['confidence']
    is_trap = wr_analysis['is_trap']
    squeeze_type = wr_analysis['squeeze_type']
    
    # Default values
    action = "WAIT"
    trade_direction = "WAIT"
    is_valid_long = True
    is_valid_short = True
    invalidation_reason = ""
    main_reason = f"Analyzing: Whales {whale_pct:.0f}% | Position {position_pct:.0f}%"
    
    # --- SPECIAL SCENARIO OVERRIDE ---
    if special:
        action = special['action']
        trade_direction = special['trade_direction']
        total_score += special['score_bonus']
        main_reason = special['reason']
    
    # --- TRAP DETECTION (HIGHEST PRIORITY) ---
    # TWO types of traps:
    # 1. LONG_TRAP (Retail > Whale): Retail overleveraged long = DON'T GO LONG
    # 2. SHORT_TRAP (Whale >> Retail, whales bullish): Retail shorts trapped = GO LONG!
    elif is_trap:
        trap_type = wr_analysis.get('trap_type', '')
        is_short_trap = wr_analysis.get('is_short_trap', False) or trap_type == 'SHORT_TRAP'
        
        if is_short_trap:
            # SHORT_TRAP: Retail shorts are trapped by whale longs = GO LONG!
            # This is a BULLISH signal - opposite of LONG_TRAP
            is_valid_long = True
            is_valid_short = False  # Don't short - retail shorts are trapped!
            invalidation_reason = ""
            
            if position_label == "EARLY":
                # EARLY + SHORT_TRAP = EXCELLENT LONG SETUP!
                action = "STRONG_LONG"
                trade_direction = "LONG"
                main_reason = f"SHORT TRAP at EARLY: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% - Retail shorts will be squeezed!"
                # Bonus for excellent setup
                squeeze_score = min(30, squeeze_score + 5)
            elif position_label == "MIDDLE":
                action = "LONG_SETUP"
                trade_direction = "LONG"
                main_reason = f"SHORT TRAP: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% - Squeeze incoming!"
            else:  # LATE
                # SHORT_TRAP at LATE = still valid but wait for dip
                action = "BUILDING_LONG"
                trade_direction = "LONG"
                main_reason = f"SHORT TRAP at LATE: Valid but wait for pullback"
                warnings.append("SHORT_TRAP valid but LATE position - consider waiting for dip")
        else:
            # LONG_TRAP: Retail > Whale = NEVER go long
            is_valid_long = False
            invalidation_reason = f"Retail ({retail_pct:.0f}%) > Whale ({whale_pct:.0f}%) - trap risk"
            
            if direction in ['BULLISH', 'LEAN_BULLISH']:
                action = "CAUTION_LONG"
                trade_direction = "WAIT"
    # --- SHORT COVERING CHECK (OI↓ + Price↑) ---
    # This is a CRITICAL pattern: Price rising but money LEAVING
    # Shorts are covering, but NO NEW LONGS entering = temporary rally
    # 
    # RULE: If OI is significantly falling but price is up, this is NOT real demand!
    # This should block LONG even if position is EARLY
    #
    # Thresholds:
    #   - OI < -2.0% = SIGNIFICANT outflow
    #   - Price > 0.3% = Any meaningful up move
    #
    short_covering_detected = oi_change < -2.0 and price_change > 0.3
    
    if short_covering_detected and direction in ['BULLISH', 'LEAN_BULLISH']:
        warnings.append(f"🚨 SHORT COVERING: OI {oi_change:+.1f}% but Price {price_change:+.1f}%")
        action = "SHORT_COVERING"
        trade_direction = "WAIT"  # Don't chase short covering rallies!
        is_valid_long = False
        invalidation_reason = "Short covering rally - no new longs entering"
        main_reason = f"🚨 SHORT COVERING (OI↓{oi_change:.1f}% + Price↑{price_change:.1f}%) - Shorts closing, NOT new buyers!"
        # Significantly reduce score - this is not a sustainable rally
        direction_score = max(15, direction_score - 15)
        squeeze_score = max(5, squeeze_score - 10)
        # This overrides EARLY position bonus - short covering is short covering!
        entry_score = max(5, entry_score - 10)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PREDICTIVE PHASE MATRIX - What comes NEXT based on current conditions?
    # ═══════════════════════════════════════════════════════════════════════════
    #
    # WYCKOFF CYCLE: ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN → repeat
    #
    # The KEY question for each phase:
    # - Where are WHALES positioned? (leading indicator)
    # - Does OI confirm or contradict?
    # - What is the NEXT likely phase?
    #
    # ═══════════════════════════════════════════════════════════════════════════
    
    phase_upper = money_flow_phase.upper()
    
    # ───────────────────────────────────────────────────────────────────────────
    # FOMO / EXHAUSTION (at highs, retail euphoric)
    # NEXT PHASE: DISTRIBUTION → MARKDOWN
    # This is the #1 SHORT setup - retail trapped at top!
    # ───────────────────────────────────────────────────────────────────────────
    if phase_upper in ['FOMO', 'EXHAUSTION', 'FOMO / DIST RISK'] and position_label == "LATE":
        if retail_pct > whale_pct:
            # 🎯 CLASSIC TOP: Retail more bullish than whales at highs
            warnings.append(f"🔴 RETAIL FOMO TOP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}% at highs!")
            action = "PREDICTIVE_SHORT"
            trade_direction = "SHORT"
            is_valid_long = False
            main_reason = f"🎯 PREDICTIVE SHORT: Retail FOMO ({retail_pct:.0f}%) > Whales ({whale_pct:.0f}%) at LATE"
            direction_score = max(15, direction_score - 15)
        elif whale_pct <= 50:
            # Whales not bullish at highs = distribution
            warnings.append(f"🔴 EXHAUSTION: Whales only {whale_pct:.0f}% at highs - Distribution likely")
            action = "PREDICTIVE_SHORT" 
            trade_direction = "SHORT"
            is_valid_long = False
            main_reason = f"🎯 PREDICTIVE SHORT: EXHAUSTION + Whales {whale_pct:.0f}% at LATE"
    
    # ───────────────────────────────────────────────────────────────────────────
    # DISTRIBUTION (smart money exiting at highs)
    # NEXT PHASE: MARKDOWN
    # ───────────────────────────────────────────────────────────────────────────
    elif phase_upper in ['DISTRIBUTION'] and position_label == "LATE":
        if whale_pct <= 50:
            # ✅ CONFIRMED: Whales exiting = MARKDOWN coming
            warnings.append(f"🔴 DISTRIBUTION confirmed: Whales {whale_pct:.0f}% exiting at highs!")
            action = "PREDICTIVE_SHORT"
            trade_direction = "SHORT"
            is_valid_long = False
            main_reason = f"🎯 PREDICTIVE SHORT: DISTRIBUTION + Whales {whale_pct:.0f}% at LATE"
        elif whale_pct >= 65:
            # False signal - whales still bullish
            warnings.append(f"⚠️ Distribution signal but Whales {whale_pct:.0f}% still bullish")
    
    # ───────────────────────────────────────────────────────────────────────────
    # CAPITULATION (panic at lows) - Check whale positioning!
    # NEXT PHASE: ACCUMULATION → MARKUP (if whales buying)
    # ───────────────────────────────────────────────────────────────────────────
    elif phase_upper == 'CAPITULATION':
        if whale_pct >= 65:
            # 🎯 REVERSAL! Whales buying the panic
            if oi_change > 1.0:
                # OI rising = retail shorting into whale buying = SQUEEZE
                warnings.append(f"🔥 SQUEEZE SETUP: Whales {whale_pct:.0f}% + OI↑ = Retail shorts trapped!")
                squeeze_score = min(30, squeeze_score + 8)
                # Don't set LONG here - let normal bullish logic handle
            else:
                warnings.append(f"🟢 REVERSAL: Whales {whale_pct:.0f}% buying the capitulation!")
        elif whale_pct <= 45:
            # True capitulation - whales not buying yet
            if oi_change > 1.0:
                # Shorts still adding = more downside
                warnings.append(f"🚨 CAPITULATION continues: Whales {whale_pct:.0f}% + OI↑ = More downside")
                is_valid_long = False
                direction_score = max(10, direction_score - 15)
    
    # ───────────────────────────────────────────────────────────────────────────
    # ACCUMULATION (smart money buying at lows)
    # NEXT PHASE: MARKUP
    # ───────────────────────────────────────────────────────────────────────────
    elif phase_upper == 'ACCUMULATION':
        if whale_pct >= 65:
            # ✅ CONFIRMED: Whales accumulating = MARKUP coming
            warnings.append(f"🟢 ACCUMULATION confirmed: Whales {whale_pct:.0f}% = MARKUP coming!")
            squeeze_score = min(30, squeeze_score + 5)
        elif whale_pct <= 45:
            # ⚠️ FALSE SIGNAL: Volume shows buying but whales aren't
            warnings.append(f"⚠️ FAKE ACCUMULATION: Whales only {whale_pct:.0f}%!")
            direction_score = max(15, direction_score - 10)
    
    # ───────────────────────────────────────────────────────────────────────────
    # MARKDOWN (trending down)
    # NEXT PHASE: CAPITULATION → ACCUMULATION
    # ───────────────────────────────────────────────────────────────────────────
    elif phase_upper == 'MARKDOWN':
        if whale_pct >= 65 and position_label == "EARLY":
            # 🎯 REVERSAL SETUP: Whales accumulating during markdown
            warnings.append(f"🟢 REVERSAL: Whales {whale_pct:.0f}% accumulating in MARKDOWN!")
            squeeze_score = min(30, squeeze_score + 5)
        elif whale_pct <= 45:
            # Confirmed downtrend
            warnings.append(f"📉 MARKDOWN: Whales {whale_pct:.0f}% not buying yet")
            is_valid_long = False
            direction_score = max(15, direction_score - 10)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OI + PRICE PREDICTIVE CHECKS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # OI↑ + Price↓: Who is adding shorts?
    if oi_change > 2.0 and price_change < -2.0:
        if whale_pct >= 65:
            # Whales holding, retail shorting = SQUEEZE FUEL
            warnings.append(f"🔥 SQUEEZE FUEL: OI↑ + Price↓ but Whales {whale_pct:.0f}%!")
            squeeze_score = min(30, squeeze_score + 5)
        elif whale_pct <= 45:
            # Smart shorts adding
            warnings.append(f"📉 Smart shorts: OI↑ + Price↓ + Whales {whale_pct:.0f}%")
            is_valid_long = False
            direction_score = max(10, direction_score - 12)
    
    # OI↓ + Price↑: Short covering (NOT sustainable)
    elif oi_change < -2.0 and price_change > 2.0:
        if whale_pct < 55 and position_label == "LATE":
            warnings.append(f"⚠️ SHORT COVERING at highs - Rally will fade!")
            trade_direction = "WAIT"
            is_valid_long = False
            main_reason = f"🚨 SHORT COVERING at LATE - Don't chase!"
    
    # ───────────────────────────────────────────────────────────────────────────
    # Below is legacy code that will be replaced by above predictive logic
    # Keeping for backward compatibility but predictive takes priority
    # ───────────────────────────────────────────────────────────────────────────

    # --- MONEY FLOW PHASE CHECK ---
    # DISTRIBUTION / PROFIT TAKING = Smart money selling into strength
    # This is a WARNING even with bullish whale positioning!
    elif money_flow_phase.upper() in ['DISTRIBUTION', 'PROFIT_TAKING', 'PROFIT TAKING'] and direction in ['BULLISH', 'LEAN_BULLISH']:
        warnings.append(f"⚠️ PROFIT TAKING: Smart money may be selling into strength")
        # Don't change direction to WAIT, but reduce conviction
        action = "CAUTION_LONG"
        # Reduce scores - this is a late-stage signal
        direction_score = max(20, direction_score - 8)
        squeeze_score = max(10, squeeze_score - 5)
        entry_score = max(5, entry_score - 5)
        main_reason = f"Bullish but PROFIT TAKING - Smart money may be distributing"
        # If also in LATE position, this is very risky
        if position_label == "LATE":
            trade_direction = "WAIT"
            is_valid_long = False
            invalidation_reason = "Profit taking at highs - distribution likely"
            main_reason = f"⚠️ PROFIT TAKING at LATE position ({position_pct:.0f}%) - High risk!"
    
    # ═══════════════════════════════════════════════════════════════════════
    # --- OI RISING + PRICE FALLING: WHO IS ADDING? (PREDICTIVE ANALYSIS) ---
    # OI↑ + Price↓ = New positions opening while price drops
    # The KEY question: Are WHALES still bullish?
    # 
    # If Whales HIGH (>65%) + OI↑ + Price↓:
    #   → Whales HOLDING, Retail SHORTING = SQUEEZE FUEL! = BULLISH
    # If Whales LOW (<50%) + OI↑ + Price↓:
    #   → Whales EXITING, Smart shorts adding = BEARISH TRAP
    # ═══════════════════════════════════════════════════════════════════════
    oi_rising_price_falling = oi_change > 1.0 and price_change < -2.0
    
    if oi_rising_price_falling:
        if whale_pct >= 65:
            # BULLISH! Whales holding, retail shorting = squeeze setup
            warnings.append(f"🔥 SQUEEZE FUEL: Whales {whale_pct:.0f}% LONG while OI↑ + Price↓ = Retail shorts will get squeezed!")
            # Don't override to WAIT - this is actually bullish!
            # Let the normal bullish logic handle it
            squeeze_score = min(30, squeeze_score + 5)  # Boost squeeze potential
        elif whale_pct <= 50:
            # BEARISH TRAP! Whales not bullish, shorts are smart money
            warnings.append(f"🚨 BEARISH TRAP: Whales only {whale_pct:.0f}% + OI↑ + Price↓ = Smart money shorting!")
            action = "BEARISH_MOMENTUM"
            trade_direction = "WAIT"  # Don't long into this
            is_valid_long = False
            invalidation_reason = "Whales not bullish, smart shorts adding"
            main_reason = f"🚨 Whales only {whale_pct:.0f}% while shorts adding - Don't LONG!"
            direction_score = max(10, direction_score - 15)
        # else: whale_pct between 50-65 = unclear, let other logic decide
    
    # --- MONEY FLOW PHASE CHECK ---
    # MARKDOWN = Already trending down, be extra cautious on longs
    elif money_flow_phase.upper() == 'MARKDOWN' and direction in ['BULLISH', 'LEAN_BULLISH']:
        warnings.append(f"⚠️ MARKDOWN phase: Trend is down, longs are counter-trend")
        action = "CAUTION_LONG"
        direction_score = max(15, direction_score - 10)
        main_reason = f"Counter-trend long in MARKDOWN phase - high risk"
    
    # --- OI CONFLICT CHECK (OI↓ + Price↓) ---
    elif oi_signal == "STRONG_BEARISH" and direction in ['BULLISH', 'LEAN_BULLISH']:
        # OI falling + Price falling = Longs exiting
        # Even if whales are long, money is leaving
        #
        # EXCEPTION: If whales are VERY high (>70%) AND position is EARLY
        # This could be ACCUMULATION during a dip, not capitulation
        # Smart money buying the fear!
        #
        high_whale_early_entry = whale_pct >= 70 and position_label == "EARLY"
        
        if high_whale_early_entry:
            # Trust the whales - this looks like accumulation
            warnings.append(f"📊 OI↓ + Price↓ but Whales {whale_pct:.0f}% + EARLY - potential accumulation")
            action = "BUILDING_LONG"
            trade_direction = "LONG"  # Trust whales on dip
            main_reason = f"🐋 Accumulation dip: Whales {whale_pct:.0f}% buying at EARLY ({position_pct:.0f}%)"
            # Small score reduction for the OI conflict, but still bullish
            direction_score = max(25, direction_score - 5)
        else:
            warnings.append(f"⚠️ OI conflict: Longs exiting (OI {oi_change:+.1f}%, Price {price_change:+.1f}%)")
            action = "CAUTION_LONG"
            trade_direction = "WAIT"
            is_valid_long = False
            invalidation_reason = "Money leaving (OI↓ + Price↓)"
            main_reason = f"OI conflict - wait for OI to stabilize"
    
    # --- POSITION CHECK FOR LONGS ---
    # LATE position = DON'T CHASE! Wait for pullback.
    # EXCEPTION: If at bullish OB/support, LATE is actually a GOOD entry!
    elif direction in ['BULLISH', 'LEAN_BULLISH'] and position_label == "LATE" and not at_bullish_level:
        warnings.append(f"⚠️ LATE entry ({position_pct:.0f}%) - Wait for pullback!")
        
        # Even with squeeze, LATE is risky - wait for dip
        if confidence == 'HIGH' and squeeze_type in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']:
            action = "WAIT_FOR_DIP"
            trade_direction = "WAIT"  # NOT LONG! Wait for better entry
            main_reason = f"Squeeze potential but LATE ({position_pct:.0f}%) - wait for pullback"
            # Reduce score for late entry
            entry_score = max(0, entry_score - 10)
        else:
            action = "WAIT_FOR_DIP"
            trade_direction = "WAIT"  # NOT LONG! Don't chase
            main_reason = f"Bullish but LATE ({position_pct:.0f}%) - Don't chase! Wait for dip"
            entry_score = max(0, entry_score - 8)
    
    # --- LATE BUT AT BULLISH OB/SUPPORT (SMC OVERRIDE!) ---
    # Position is LATE but price is at a key support level - this IS a good entry!
    elif direction in ['BULLISH', 'LEAN_BULLISH'] and position_label == "LATE" and at_bullish_level:
        # Don't penalize - the OB/support provides a valid entry
        if at_bullish_ob:
            action = "LONG_SETUP"
            trade_direction = "LONG"
            main_reason = f"🎯 LATE but AT bullish OB - valid entry with defined risk!"
            warnings.append(f"✅ Position {position_pct:.0f}% but at bullish Order Block")
        elif near_bullish_ob:
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"LATE but NEAR bullish OB - decent entry"
            warnings.append(f"📍 Position {position_pct:.0f}% but near bullish Order Block")
        else:  # at_support
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"LATE but AT support level - entry with structure"
            warnings.append(f"📍 Position {position_pct:.0f}% but at key support")
    
    # --- HIGH CONFIDENCE BULLISH ---
    # Allow if: no trap, OR if SHORT_TRAP (retail shorts trapped = go long!)
    # is_long_trap excludes SHORT_TRAP cases where going long is correct
    elif direction == 'BULLISH' and confidence == 'HIGH' and not (is_trap and wr_analysis.get('trap_type', '') != 'SHORT_TRAP'):
        if squeeze_type in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']:
            action = "STRONG_LONG"
            trade_direction = "LONG"
            main_reason = f"🔥 High conviction + Squeeze: {wr_analysis['reason']}"
        elif position_label == "EARLY":
            action = "LONG_SETUP"
            trade_direction = "LONG"
            main_reason = f"Strong setup: Whale {whale_pct:.0f}% + EARLY position"
        else:
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"Building long: Whale conviction {whale_pct:.0f}%"
    
    # --- MEDIUM CONFIDENCE BULLISH ---
    # Allow if: no trap, OR if SHORT_TRAP (retail shorts trapped = go long!)
    elif direction in ['BULLISH', 'LEAN_BULLISH'] and not (is_trap and wr_analysis.get('trap_type', '') != 'SHORT_TRAP'):
        if position_label == "EARLY":
            action = "BUILDING_LONG"
            trade_direction = "LONG"
            main_reason = f"Lean bullish ({whale_pct:.0f}%) + good entry"
        elif position_label == "MIDDLE":
            action = "MONITOR_LONG"
            trade_direction = "LONG"
            main_reason = f"Lean bullish but MID range - wait for dip"
        else:
            action = "MONITOR_LONG"
            trade_direction = "LONG"
            main_reason = f"Lean bullish but LATE - wait for pullback"
    
    # --- HIGH CONFIDENCE BEARISH ---
    elif direction == 'BEARISH' and confidence == 'HIGH':
        if position_label == "LATE":
            action = "STRONG_SHORT"
            trade_direction = "SHORT"
            main_reason = f"Strong short: Whale {whale_pct:.0f}% bearish at highs"
        elif position_label == "MIDDLE":
            action = "SHORT_SETUP"
            trade_direction = "SHORT"
            main_reason = f"Short setup: Whale conviction"
        else:
            # EARLY position = near lows = BAD for shorts
            action = "WAIT_FOR_BOUNCE"
            trade_direction = "WAIT"  # NOT SHORT! Don't short the lows
            main_reason = f"Bearish but at LOWS ({position_pct:.0f}%) - Don't short here! Wait for bounce"
            warnings.append(f"⚠️ EARLY position ({position_pct:.0f}%) - bad entry for shorts")
    
    # --- MEDIUM CONFIDENCE BEARISH ---
    elif direction in ['BEARISH', 'LEAN_BEARISH']:
        if position_label == "LATE":
            action = "BUILDING_SHORT"
            trade_direction = "SHORT"
            main_reason = f"Lean bearish ({whale_pct:.0f}%) at highs"
        elif position_label == "MIDDLE":
            action = "MONITOR_SHORT"
            trade_direction = "SHORT"
            main_reason = f"Lean bearish ({whale_pct:.0f}%)"
        else:
            # EARLY = near lows = bad for shorts
            action = "WAIT_FOR_BOUNCE"
            trade_direction = "WAIT"
            main_reason = f"Bearish but near LOWS - don't short bottom"
            warnings.append(f"⚠️ Near lows ({position_pct:.0f}%) - wait for bounce")
    
    # --- NEUTRAL ---
    else:
        action = "WAIT"
        trade_direction = "WAIT"
        main_reason = f"No clear edge - Whale {whale_pct:.0f}%, Retail {retail_pct:.0f}%"
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECALCULATE TOTAL SCORE (after all adjustments)
    # Must include ALL modifiers: oi, market, historical
    # ═══════════════════════════════════════════════════════════════════════
    total_score = direction_score + squeeze_score + entry_score + oi_modifier + market_modifier + historical_modifier + meme_modifier
    total_score = max(0, min(100, total_score))
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 8: Build Stories for Combined Learning
    # ═══════════════════════════════════════════════════════════════════════
    
    story_parts = []
    
    # OI Story
    if oi_signal == "STRONG_BULLISH":
        oi_story = f"📈 OI up {oi_change:+.1f}% + Price up {price_change:+.1f}% → New LONGS entering. Fresh buying pressure!"
    elif oi_signal == "STRONG_BEARISH":
        oi_story = f"📉 OI down {oi_change:+.1f}% + Price down {price_change:+.1f}% → Longs EXITING. Selling pressure or liquidations."
    elif oi_signal == "SHORT_COVERING":
        oi_story = f"⚠️ SHORT COVERING: OI {oi_change:+.1f}% + Price {price_change:+.1f}% → Shorts closing but NO NEW LONGS! Rally may be temporary."
    elif oi_signal == "BULLISH":
        oi_story = f"📊 OI down {oi_change:+.1f}% + Price up {price_change:+.1f}% → Shorts CLOSING. Some squeeze potential."
    elif oi_signal == "BEARISH":
        oi_story = f"📊 OI up {oi_change:+.1f}% + Price down {price_change:+.1f}% → New SHORTS entering. Bearish pressure."
    else:
        # Neutral but still informative
        if oi_change < -2:
            oi_story = f"📉 OI falling {oi_change:+.1f}% (positions closing) | Price {price_change:+.1f}%"
        elif oi_change > 2:
            oi_story = f"📈 OI rising {oi_change:+.1f}% (new positions) | Price {price_change:+.1f}%"
        else:
            oi_story = f"📊 OI {oi_change:+.1f}% | Price {price_change:+.1f}% → No significant flow signal."
    
    # Whale Story
    divergence = whale_pct - retail_pct
    trap_type = wr_analysis.get('trap_type', '')
    is_short_trap_check = wr_analysis.get('is_short_trap', False) or trap_type == 'SHORT_TRAP'
    
    if is_trap and is_short_trap_check:
        # SHORT_TRAP = BULLISH! Retail shorts trapped
        whale_story = f"SHORT TRAP: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% - Retail shorts ({100-retail_pct:.0f}%) will be SQUEEZED! GO LONG!"
    elif is_trap:
        # LONG_TRAP = BEARISH! Retail longs trapped
        whale_story = f"LONG TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%! Retail overleveraged - whales may dump on them."
    elif divergence >= 15:
        whale_story = f"SQUEEZE SETUP: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% - {divergence:.0f}% divergence! Retail shorts will be liquidated."
    elif divergence >= 5:
        whale_story = f"Whales {whale_pct:.0f}% LONG vs Retail {retail_pct:.0f}% - Slight edge to bulls."
    elif divergence <= -10:
        whale_story = f"Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% - Retail more bullish. Caution on longs!"
    else:
        whale_story = f"Whales {whale_pct:.0f}% | Retail {retail_pct:.0f}% - No significant divergence."
    
    # Position Story - based on underlying direction, not trade_direction (which may be WAIT)
    # NOW WITH SMC CONTEXT!
    if position_label == "EARLY":
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            position_story = f"🎯 EARLY entry ({position_pct:.0f}%) → Near lows. Excellent R:R for longs!"
        elif direction in ['BEARISH', 'LEAN_BEARISH']:
            if at_bearish_level:
                position_story = f"🎯 EARLY ({position_pct:.0f}%) but AT bearish OB → Valid short entry with structure!"
            else:
                position_story = f"❌ EARLY ({position_pct:.0f}%) → Near LOWS! BAD for shorts - don't short the bottom!"
        else:
            position_story = f"📍 Position: {position_pct:.0f}% of range → Near lows."
    elif position_label == "LATE":
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            # SMC OVERRIDE for position story
            if at_bullish_ob:
                position_story = f"🎯 Position {position_pct:.0f}% but **AT BULLISH OB** → Valid entry! OB provides support."
            elif near_bullish_ob:
                position_story = f"📍 Position {position_pct:.0f}% but **NEAR BULLISH OB** → Decent entry with nearby support."
            elif at_support:
                position_story = f"📍 Position {position_pct:.0f}% but **AT SUPPORT** → Entry with structure."
            else:
                position_story = f"❌ LATE entry ({position_pct:.0f}%) → Near HIGHS! DON'T CHASE - wait for pullback!"
        elif direction in ['BEARISH', 'LEAN_BEARISH']:
            position_story = f"🎯 Good entry for shorts ({position_pct:.0f}%) → Near highs."
        else:
            position_story = f"📍 Position: {position_pct:.0f}% of range → Near highs."
    else:
        position_story = f"📍 MID-RANGE ({position_pct:.0f}%) → Wait for price to reach support/resistance."
    
    # Add money flow phase to position story if significant
    mf_upper = money_flow_phase.upper() if money_flow_phase else 'UNKNOWN'
    if mf_upper in ['DISTRIBUTION', 'PROFIT_TAKING', 'PROFIT TAKING']:
        position_story += f"\n\n⚠️ **MONEY FLOW: PROFIT TAKING** - Smart money may be selling into this rally!"
    elif mf_upper == 'MARKDOWN':
        position_story += f"\n\n⚠️ **MONEY FLOW: MARKDOWN** - Trend is down, be cautious with longs."
    elif mf_upper == 'ACCUMULATION':
        position_story += f"\n\n✅ **MONEY FLOW: ACCUMULATION** - Smart money building positions."
    elif mf_upper == 'MARKUP':
        position_story += f"\n\n✅ **MONEY FLOW: MARKUP** - Strong uptrend with volume."
    
    # Add MEME COIN warning to story
    if is_meme:
        position_story += f"\n\n🎰 **MEME COIN WARNING**: {symbol} is a meme coin!"
        position_story += f"\n- Whales on meme coins may BE the manipulators (pump & dump)"
        position_story += f"\n- Use smaller position size (25-50% of normal)"
        position_story += f"\n- Expect extreme volatility and stop hunting"
        position_story += f"\n- Score reduced by 15 points for manipulation risk"
    
    # Build summary story
    story_parts.append(f"🐋 Whales: {whale_pct:.0f}% long")
    story_parts.append(f"👥 Retail: {retail_pct:.0f}% long")
    
    if divergence > 10:
        story_parts.append(f"📊 Divergence: +{divergence:.0f}% (whales more bullish)")
    elif divergence < -5:
        story_parts.append(f"📊 Divergence: {divergence:.0f}% (⚠️ retail more bullish)")
    
    story_parts.append(f"📈 OI Signal: {oi_signal} ({oi_explanation})")
    story_parts.append(f"📍 Position: {position_label} ({position_pct:.0f}%)")
    
    story = " | ".join(story_parts)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 9: Build Conclusion (clean, no duplication)
    # ═══════════════════════════════════════════════════════════════════════
    
    # Create short reason for conclusion
    divergence = whale_pct - retail_pct
    
    if is_trap:
        short_reason = f"Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        conclusion = f"⚠️ RETAIL TRAP DETECTED - {short_reason}"
        conclusion_action = "WAIT - Don't fight the trap"
    elif special:
        conclusion = f"🎯 {special['scenario']} - {special['reason']}"
        conclusion_action = special['action']
    elif action == "WAIT_FOR_DIP":
        # Bullish but LATE - don't chase!
        conclusion = f"⏳ BULLISH BUT LATE ({position_pct:.0f}%) - Don't chase!"
        conclusion_action = "WAIT FOR DIP"
    elif action == "WAIT_FOR_BOUNCE":
        # Bearish but EARLY - don't short lows!
        conclusion = f"⏳ BEARISH BUT AT LOWS ({position_pct:.0f}%) - Don't short!"
        conclusion_action = "WAIT FOR BOUNCE"
    elif action == "SHORT_COVERING":
        # OI falling + Price rising = shorts covering, not new longs
        conclusion = f"⚠️ SHORT COVERING RALLY - OI {oi_change:+.1f}% + Price {price_change:+.1f}%"
        conclusion_action = "WAIT - Rally may be temporary"
    elif action == "CAUTION_LONG" and money_flow_phase.upper() in ['DISTRIBUTION', 'PROFIT_TAKING', 'PROFIT TAKING']:
        # Money flow showing profit taking - caution on longs
        mf_upper = money_flow_phase.upper()
        if position_label == "LATE":
            conclusion = f"⚠️ PROFIT TAKING at HIGHS ({position_pct:.0f}%) - High risk!"
            conclusion_action = "WAIT - Distribution likely"
        else:
            conclusion = f"⚠️ PROFIT TAKING detected - W:{whale_pct:.0f}% but smart money selling"
            conclusion_action = "CAUTION - Wait for confirmation"
    elif trade_direction == "LONG":
        if confidence == "HIGH":
            if squeeze_type in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']:
                conclusion = f"🟢 STRONG LONG (Squeeze) - W:{whale_pct:.0f}% vs R:{retail_pct:.0f}%"
            else:
                conclusion = f"🟢 STRONG LONG - Whales {whale_pct:.0f}% bullish"
            conclusion_action = "LONG NOW" if position_label == "EARLY" else "LONG (watch entry)"
        else:
            conclusion = f"🟢 LONG SETUP - Whales {whale_pct:.0f}% lean bullish"
            conclusion_action = "LONG on dip" if position_label != "EARLY" else "LONG"
    elif trade_direction == "SHORT":
        if confidence == "HIGH":
            conclusion = f"🔴 STRONG SHORT - Whales {whale_pct:.0f}% bearish"
            conclusion_action = "SHORT NOW" if position_label == "LATE" else "SHORT (watch entry)"
        else:
            conclusion = f"🔴 SHORT SETUP - Whales {whale_pct:.0f}% lean bearish"
            conclusion_action = "SHORT on rally" if position_label != "LATE" else "SHORT"
    else:
        conclusion = f"⏳ NO CLEAR EDGE - W:{whale_pct:.0f}% R:{retail_pct:.0f}%"
        conclusion_action = "WAIT for setup"
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 10: Return Complete Decision
    # ═══════════════════════════════════════════════════════════════════════
    
    return TradeDecision(
        action=action,
        trade_direction=trade_direction,
        confidence=confidence,
        
        direction_score=direction_score,
        squeeze_score=squeeze_score,
        entry_score=entry_score,
        total_score=total_score,
        
        direction_label=direction,
        squeeze_label=squeeze_type,
        position_label=position_label,
        oi_signal=oi_signal,
        
        main_reason=main_reason,
        warnings=warnings,
        story=story,
        
        is_valid_long=is_valid_long,
        is_valid_short=is_valid_short,
        invalidation_reason=invalidation_reason,
        
        # Trap detection
        is_trap=is_trap,
        trap_type=wr_analysis.get('trap_type', ''),
        is_short_trap=wr_analysis.get('is_short_trap', False) or wr_analysis.get('trap_type', '') == 'SHORT_TRAP',
        
        # Stories for Combined Learning
        oi_story=oi_story,
        whale_story=whale_story,
        position_story=position_story,
        market_story=market_story,
        historical_story=historical_story,  # NEW: Historical validation
        conclusion=conclusion,
        conclusion_action=conclusion_action,
        
        # Market Conditions
        fear_greed=fear_greed,
        market_modifier=market_modifier,
        
        # Historical Validation
        historical_win_rate=historical_win_rate,
        historical_sample_size=historical_sample_size,
        historical_modifier=historical_modifier
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DECISION FROM MARKET CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

def decide_from_context(ctx) -> TradeDecision:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    THE UNIFIED DECISION FUNCTION
    ═══════════════════════════════════════════════════════════════════════════════
    
    Takes a MarketContext object and returns the complete trading decision.
    This is the ONLY function that should be called for scoring.
    
    Args:
        ctx: MarketContext object with all gathered metrics
        
    Returns:
        TradeDecision with complete analysis, scores, and stories
    """
    
    return get_trade_decision(
        whale_pct=ctx.whale_pct,
        retail_pct=ctx.retail_pct,
        oi_change=ctx.oi_change_24h,
        price_change=ctx.price_change_24h,
        position_pct=ctx.position_in_range,
        ta_score=ctx.ta_score,
        money_flow_phase=ctx.money_flow_phase
    )


def get_unified_result(ctx, decision: TradeDecision) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    GET UNIFIED RESULT FOR DISPLAY
    ═══════════════════════════════════════════════════════════════════════════════
    
    Takes MarketContext and TradeDecision, returns everything needed for display.
    This replaces calculate_predictive_score output.
    
    Returns dict with:
        - All PredictiveScore-compatible fields
        - Combined learning data
        - Display-ready formatted strings
    """
    
    # Map direction label to display format
    direction_display = {
        'BULLISH': 'BULLISH',
        'LEAN_BULLISH': 'LEAN_BULLISH',
        'BEARISH': 'BEARISH',
        'LEAN_BEARISH': 'LEAN_BEARISH',
        'NEUTRAL': 'NEUTRAL',
    }.get(decision.direction_label, 'NEUTRAL')
    
    # Map squeeze label to display format
    squeeze_display = {
        'HIGH_SQUEEZE': 'HIGH',
        'MODERATE_SQUEEZE': 'MEDIUM',
        'SLIGHT_EDGE': 'LOW',
        'NO_EDGE': 'NONE',
        'EXTREME_TRAP': 'CONFLICT',
        'HIGH_TRAP': 'CONFLICT',
        'MODERATE_TRAP': 'CONFLICT',
        'SLIGHT_TRAP': 'CONFLICT',
    }.get(decision.squeeze_label, 'NONE')
    
    # Map entry timing
    if decision.trade_direction == 'WAIT':
        entry_timing = 'WAIT'
    elif decision.position_label == 'EARLY' and decision.trade_direction == 'LONG':
        entry_timing = 'NOW'
    elif decision.position_label == 'LATE' and decision.trade_direction == 'SHORT':
        entry_timing = 'NOW'
    elif decision.position_label == 'MIDDLE':
        entry_timing = 'SOON'
    else:
        entry_timing = 'WAIT'
    
    # Determine confidence for direction
    if decision.direction_score >= 36:
        direction_confidence = 'HIGH'
    elif decision.direction_score >= 28:
        direction_confidence = 'MEDIUM'
    else:
        direction_confidence = 'LOW' if decision.direction_score >= 15 else 'NONE'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD FINAL ACTION - ENTRY SCORE MUST BE RESPECTED!
    # If entry_score is low, we WAIT regardless of direction score
    # This prevents "BUY" when timing is bad
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Entry score thresholds for action override
    ENTRY_SCORE_BUY_MIN = 15      # Need at least 15/30 to say BUY
    ENTRY_SCORE_CAUTIOUS_MIN = 10  # Below 10 = always WAIT
    
    if decision.trade_direction == 'LONG':
        # Check if entry timing allows a BUY verdict
        if decision.entry_score < ENTRY_SCORE_CAUTIOUS_MIN:
            # Entry is too poor - WAIT regardless of total score
            final_action = '⏳ WAIT (Don\'t Chase)'
            entry_timing = 'WAIT'
        elif decision.entry_score < ENTRY_SCORE_BUY_MIN:
            # Entry is weak - downgrade to WATCHLIST/CAUTIOUS
            if decision.total_score >= 65:
                final_action = '👀 WATCHLIST'
            else:
                final_action = '⏳ WAIT FOR DIP'
            entry_timing = 'SOON' if decision.position_label == 'MIDDLE' else 'WAIT'
        else:
            # Entry is good enough - use total score
            if decision.total_score >= 80:
                final_action = '🚀 STRONG BUY'
            elif decision.total_score >= 65:
                final_action = '✅ BUY'
            elif decision.total_score >= 50:
                final_action = '📊 CAUTIOUS BUY'
            else:
                final_action = '⏳ WAIT FOR DIP'
    elif decision.trade_direction == 'SHORT':
        # Same logic for shorts
        if decision.entry_score < ENTRY_SCORE_CAUTIOUS_MIN:
            final_action = '⏳ WAIT (Don\'t Chase)'
            entry_timing = 'WAIT'
        elif decision.entry_score < ENTRY_SCORE_BUY_MIN:
            if decision.total_score >= 65:
                final_action = '👀 WATCHLIST'
            else:
                final_action = '⏳ WAIT FOR RALLY'
            entry_timing = 'SOON' if decision.position_label == 'MIDDLE' else 'WAIT'
        else:
            if decision.total_score >= 80:
                final_action = '🔴 STRONG SELL'
            elif decision.total_score >= 65:
                final_action = '🔴 SELL'
            elif decision.total_score >= 50:
                final_action = '📊 CAUTIOUS SELL'
            else:
                final_action = '⏳ WAIT FOR RALLY'
    else:
        if 'TRAP' in decision.action:
            final_action = f'⚠️ {decision.action}'
        elif 'WAIT_FOR' in decision.action:
            final_action = f'⏳ {decision.action.replace("_", " ")}'
        else:
            final_action = '⏳ WAIT'
    
    # Build summary
    if decision.trade_direction == 'WAIT':
        final_summary = decision.conclusion
    else:
        final_summary = f"{decision.conclusion}"
    
    # Reason strings
    direction_reason = f"Whales {ctx.whale_pct:.0f}% | OI: {ctx.oi_change_24h:+.1f}%"
    squeeze_reason = f"Divergence: {ctx.divergence:+.0f}%"
    timing_reason = f"TA: {ctx.ta_score} | Position: {decision.position_label} ({ctx.position_in_range:.0f}%)"
    
    return {
        # PredictiveScore-compatible fields
        'direction': direction_display,
        'direction_confidence': direction_confidence,
        'direction_score': decision.direction_score,
        'direction_reason': direction_reason,
        
        'squeeze_potential': squeeze_display,
        'divergence_pct': ctx.divergence,
        'squeeze_score': decision.squeeze_score,
        'squeeze_reason': squeeze_reason,
        
        'entry_timing': entry_timing,
        'ta_score': ctx.ta_score,
        'timing_score': decision.entry_score,
        'timing_reason': timing_reason,
        'move_position': decision.position_label,
        'move_position_pct': ctx.position_in_range,
        
        'final_score': decision.total_score,
        'final_action': final_action,
        'final_summary': final_summary,
        
        'trade_mode': ctx.trading_mode,
        'timeframe': ctx.timeframe,
        'trade_direction': decision.trade_direction,
        
        # Additional fields
        'action': decision.action,
        'confidence': decision.confidence,
        'is_valid_long': decision.is_valid_long,
        'is_valid_short': decision.is_valid_short,
        'warnings': decision.warnings,
        
        # Trap detection
        'is_trap': decision.is_trap,
        'trap_type': decision.trap_type,
        'is_short_trap': decision.is_short_trap,
        
        # Combined Learning stories
        'oi_story': decision.oi_story,
        'whale_story': decision.whale_story,
        'position_story': decision.position_story,
        'market_story': decision.market_story,
        'historical_story': decision.historical_story,  # NEW: Historical validation
        'conclusion': decision.conclusion,
        'conclusion_action': decision.conclusion_action,
        
        # Market Conditions data
        'fear_greed': decision.fear_greed,
        'market_modifier': decision.market_modifier,
        
        # Historical Validation data
        'historical_win_rate': decision.historical_win_rate,
        'historical_sample_size': decision.historical_sample_size,
        'historical_modifier': decision.historical_modifier,
        
        # Raw decision for reference
        'master_decision': decision,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CONFIRMATION (Deep Learning - Optional Guidance)
# ═══════════════════════════════════════════════════════════════════════════════

def add_pattern_confirmation(decision: TradeDecision, df, mode: str = 'daytrade', market: str = 'crypto') -> TradeDecision:
    """
    Enrich TradeDecision with pattern detection (optional guidance).
    
    Pattern detection is NOT primary signal - just confirmation.
    Only adds info if pattern aligns with ML direction.
    
    Args:
        decision: Existing TradeDecision from get_trade_decision()
        df: DataFrame with OHLCV data for pattern detection
        mode: Trading mode ('scalp', 'daytrade', 'swing', 'investment')
        market: Market type ('crypto', 'stocks', 'etfs')
    
    Returns:
        TradeDecision with pattern fields populated
    """
    if df is None or len(df) < 50:
        return decision
    
    try:
        from ml.ml_engine import MLEngine
        
        engine = MLEngine()
        if not engine.has_pattern_detection():
            return decision
        
        # Get pattern confirmation
        pattern_info = engine.get_pattern_confirmation(df, mode, market)
        
        if pattern_info.get('pattern_detected'):
            pattern_name = pattern_info['pattern_detected']
            pattern_conf = pattern_info['pattern_confidence']
            pattern_dir = pattern_info['pattern_direction']
            
            # Check if pattern aligns with trade direction
            aligns = False
            if decision.trade_direction == 'LONG' and pattern_dir == 'BULLISH':
                aligns = True
            elif decision.trade_direction == 'SHORT' and pattern_dir == 'BEARISH':
                aligns = True
            
            # Update decision
            decision.pattern_detected = pattern_name
            decision.pattern_confidence = pattern_conf
            decision.pattern_aligns = aligns
            
            # Generate pattern story
            if aligns:
                decision.pattern_story = f"📊 Pattern Confirmation: {pattern_name} detected ({pattern_conf:.0f}% conf) - ALIGNS with {decision.trade_direction}"
            else:
                decision.pattern_story = f"📊 Pattern Note: {pattern_name} detected ({pattern_conf:.0f}% conf) - Does NOT align with signal"
    
    except Exception as e:
        # Pattern detection failed - continue without it
        pass
    
    return decision


def get_pattern_guidance(df, mode: str = 'daytrade', market: str = 'crypto') -> dict:
    """
    Get standalone pattern guidance (without trade decision).
    
    Use this when you just want to see what patterns are detected.
    
    Returns:
        dict with: pattern_detected, pattern_confidence, pattern_direction, pattern_story
    """
    result = {
        'pattern_detected': None,
        'pattern_confidence': 0.0,
        'pattern_direction': 'NEUTRAL',
        'pattern_story': 'No pattern detected'
    }
    
    if df is None or len(df) < 50:
        return result
    
    try:
        from ml.ml_engine import MLEngine
        
        engine = MLEngine()
        if not engine.has_pattern_detection():
            result['pattern_story'] = 'Pattern detection not available'
            return result
        
        pattern_info = engine.get_pattern_confirmation(df, mode, market)
        
        if pattern_info.get('pattern_detected'):
            result['pattern_detected'] = pattern_info['pattern_detected']
            result['pattern_confidence'] = pattern_info['pattern_confidence']
            result['pattern_direction'] = pattern_info['pattern_direction']
            
            dir_emoji = '🟢' if pattern_info['pattern_direction'] == 'BULLISH' else '🔴' if pattern_info['pattern_direction'] == 'BEARISH' else '⚪'
            result['pattern_story'] = f"{dir_emoji} {pattern_info['pattern_detected']} ({pattern_info['pattern_confidence']:.0f}% confidence)"
    
    except Exception as e:
        result['pattern_story'] = f'Pattern detection error: {e}'
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# END OF MASTER RULES
# ═══════════════════════════════════════════════════════════════════════════════