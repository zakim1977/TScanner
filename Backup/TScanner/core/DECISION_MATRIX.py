"""
═══════════════════════════════════════════════════════════════════════════════
INVESTORIQ COMPLETE DECISION MATRIX
═══════════════════════════════════════════════════════════════════════════════

This is the SINGLE SOURCE OF TRUTH for all trading decisions.
NO MORE scattered logic - everything is defined HERE.

Key Principle: WHALE POSITIONING IS PRIMARY (Leading Indicator)
- Whales move first, retail follows
- When Retail > Whale on longs → Whales preparing to dump (AVOID LONGS)
- When Whale > Retail on longs → Squeeze potential (FAVOR LONGS)
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS - Single place for all threshold definitions
# ═══════════════════════════════════════════════════════════════════════════════

class Thresholds:
    """All thresholds in one place - NO MAGIC NUMBERS elsewhere"""
    
    # Whale/Retail Positioning Levels
    EXTREME_BULLISH = 75    # Very strong conviction
    STRONG_BULLISH = 70     # Strong conviction
    BULLISH = 65            # Clear bullish bias
    LEAN_BULLISH = 60       # Slight bullish lean
    NEUTRAL_HIGH = 55       # Upper neutral
    NEUTRAL_LOW = 45        # Lower neutral
    LEAN_BEARISH = 40       # Slight bearish lean
    BEARISH = 35            # Clear bearish bias
    STRONG_BEARISH = 30     # Strong bearish conviction
    EXTREME_BEARISH = 25    # Very strong bearish
    
    # Position in Range
    EARLY_MAX = 35          # 0-35% = EARLY (near lows)
    MID_MAX = 65            # 35-65% = MID
    LATE_MIN = 65           # 65-100% = LATE (near highs)
    
    # Divergence Thresholds (Whale - Retail)
    EXTREME_DIVERGENCE = 20     # Extreme squeeze/trap
    HIGH_DIVERGENCE = 15        # High squeeze/trap potential
    MEDIUM_DIVERGENCE = 10      # Clear edge
    LOW_DIVERGENCE = 5          # Slight edge
    
    # OI Change Thresholds
    OI_STRONG_RISING = 5        # Strong money inflow
    OI_RISING = 2               # Money inflow
    OI_FALLING = -2             # Money outflow
    OI_STRONG_FALLING = -5      # Strong money outflow


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS - Every possible combination
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    """Result of scenario evaluation"""
    action: str             # STRONG_LONG, LONG_SETUP, BUILDING_LONG, CAUTION_LONG, WAIT, etc.
    direction: str          # LONG, SHORT, WAIT
    confidence: str         # HIGH, MEDIUM, LOW
    score_modifier: int     # Points to add/subtract
    reason: str             # Human-readable explanation
    warnings: List[str]     # Any warnings to display


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER DECISION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_scenario(
    whale_pct: float,
    retail_pct: float,
    position_pct: float,
    oi_change: float = 0,
    money_flow_phase: str = 'UNKNOWN'
) -> ScenarioResult:
    """
    MASTER DECISION FUNCTION - Evaluates ALL inputs and returns definitive action.
    
    This replaces all scattered logic throughout the codebase.
    """
    
    T = Thresholds
    warnings = []
    
    # Calculate key metrics
    divergence = whale_pct - retail_pct  # Positive = whales more bullish
    
    # Determine position label
    if position_pct <= T.EARLY_MAX:
        position = 'EARLY'
    elif position_pct <= T.MID_MAX:
        position = 'MID'
    else:
        position = 'LATE'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 1: RETAIL TRAP DETECTION (HIGHEST PRIORITY)
    # If retail is more bullish than whales → NEVER go long
    # ═══════════════════════════════════════════════════════════════════════════
    
    if divergence < 0:  # Retail more bullish than whales
        retail_excess = abs(divergence)
        
        if retail_excess >= T.EXTREME_DIVERGENCE:
            # EXTREME TRAP - Whales about to dump hard
            warnings.append(f"🚨 EXTREME TRAP: Retail {retail_pct:.0f}% >> Whales {whale_pct:.0f}%")
            return ScenarioResult(
                action="AVOID_LONG",
                direction="WAIT",
                confidence="HIGH",
                score_modifier=-20,
                reason=f"Retail severely overleveraged - high dump risk",
                warnings=warnings
            )
            
        elif retail_excess >= T.HIGH_DIVERGENCE:
            # HIGH TRAP RISK
            warnings.append(f"⚠️ TRAP RISK: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%")
            return ScenarioResult(
                action="CAUTION_LONG",
                direction="WAIT",
                confidence="LOW",
                score_modifier=-15,
                reason=f"Retail overleveraged - wait for reset",
                warnings=warnings
            )
            
        elif retail_excess >= T.LOW_DIVERGENCE:
            # MODERATE TRAP RISK - Still avoid longs
            warnings.append(f"⚠️ Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}% - caution on longs")
            # Continue evaluation but with penalty
            
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 2: SQUEEZE DETECTION (Whales > Retail)
    # Whales positioned, retail not → Squeeze incoming
    # ═══════════════════════════════════════════════════════════════════════════
    
    if divergence >= T.EXTREME_DIVERGENCE:
        # EXTREME SQUEEZE - Shorts about to get rekt
        if position == 'EARLY':
            return ScenarioResult(
                action="STRONG_LONG",
                direction="LONG",
                confidence="HIGH",
                score_modifier=25,
                reason=f"🔥 SQUEEZE: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}% + EARLY position",
                warnings=[]
            )
        elif position == 'MID':
            return ScenarioResult(
                action="LONG_SETUP",
                direction="LONG",
                confidence="HIGH",
                score_modifier=20,
                reason=f"🔥 SQUEEZE: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%",
                warnings=[]
            )
        else:  # LATE
            warnings.append("⚠️ Squeeze but LATE in move - reduced size")
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=12,
                reason=f"Squeeze potential but late entry",
                warnings=warnings
            )
    
    elif divergence >= T.HIGH_DIVERGENCE:
        # HIGH SQUEEZE POTENTIAL
        if position in ['EARLY', 'MID']:
            return ScenarioResult(
                action="LONG_SETUP",
                direction="LONG",
                confidence="HIGH",
                score_modifier=18,
                reason=f"Squeeze setup: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=12,
                reason=f"Squeeze potential but late",
                warnings=["Late entry - tighten stops"]
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 3: CAPITULATION DETECTION (Both low + EARLY position)
    # Everyone bearish at lows = reversal setup
    # ═══════════════════════════════════════════════════════════════════════════
    
    if whale_pct <= T.BEARISH and retail_pct <= T.BEARISH and position == 'EARLY':
        # CAPITULATION - Everyone gave up at lows
        if money_flow_phase == 'ACCUMULATION':
            return ScenarioResult(
                action="EARLY_LONG",
                direction="LONG",
                confidence="HIGH",
                score_modifier=22,
                reason=f"🎯 CAPITULATION + Accumulation at lows - reversal setup",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=15,
                reason=f"Capitulation at lows - potential reversal",
                warnings=["Wait for accumulation confirmation"]
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 4: EUPHORIA DETECTION (Both high + LATE position)
    # Everyone bullish at highs = top signal
    # ═══════════════════════════════════════════════════════════════════════════
    
    if whale_pct >= T.BULLISH and retail_pct >= T.BULLISH and position == 'LATE':
        # EUPHORIA - Everyone bullish at highs
        if money_flow_phase == 'DISTRIBUTION':
            return ScenarioResult(
                action="EARLY_SHORT",
                direction="SHORT",
                confidence="HIGH",
                score_modifier=22,
                reason=f"🎯 EUPHORIA + Distribution at highs - reversal setup",
                warnings=[]
            )
        else:
            warnings.append("⚠️ Euphoria at highs - avoid new longs")
            return ScenarioResult(
                action="CAUTION_LONG",
                direction="WAIT",
                confidence="LOW",
                score_modifier=-5,
                reason=f"Euphoria at highs - wait for pullback",
                warnings=warnings
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 5: WHALE CONVICTION (Primary indicator)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Check if we already have a retail warning
    has_retail_warning = divergence < -T.LOW_DIVERGENCE
    
    if whale_pct >= T.EXTREME_BULLISH and not has_retail_warning:
        # Extreme whale conviction
        if position == 'EARLY':
            return ScenarioResult(
                action="STRONG_LONG",
                direction="LONG",
                confidence="HIGH",
                score_modifier=22,
                reason=f"Extreme whale conviction ({whale_pct:.0f}%) at EARLY position",
                warnings=[]
            )
        elif position == 'MID':
            return ScenarioResult(
                action="LONG_SETUP",
                direction="LONG",
                confidence="HIGH",
                score_modifier=18,
                reason=f"Extreme whale conviction ({whale_pct:.0f}%)",
                warnings=[]
            )
        else:  # LATE
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=12,
                reason=f"Strong whales but late entry",
                warnings=["Tighten stops - late in move"]
            )
    
    elif whale_pct >= T.STRONG_BULLISH and not has_retail_warning:
        if position in ['EARLY', 'MID']:
            return ScenarioResult(
                action="LONG_SETUP",
                direction="LONG",
                confidence="HIGH",
                score_modifier=16,
                reason=f"Strong whale conviction ({whale_pct:.0f}%)",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=10,
                reason=f"Strong whales but late",
                warnings=["Late entry"]
            )
    
    elif whale_pct >= T.BULLISH and not has_retail_warning:
        if position == 'EARLY':
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=14,
                reason=f"Bullish whales ({whale_pct:.0f}%) + EARLY position",
                warnings=[]
            )
        elif position == 'MID':
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=10,
                reason=f"Bullish whales ({whale_pct:.0f}%)",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="MONITOR_LONG",
                direction="LONG",
                confidence="LOW",
                score_modifier=6,
                reason=f"Bullish whales but LATE",
                warnings=["Wait for pullback"]
            )
    
    elif whale_pct >= T.LEAN_BULLISH and not has_retail_warning:
        if position == 'EARLY':
            return ScenarioResult(
                action="BUILDING_LONG",
                direction="LONG",
                confidence="MEDIUM",
                score_modifier=12,
                reason=f"Lean bullish whales ({whale_pct:.0f}%) + good entry",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="MONITOR_LONG",
                direction="LONG",
                confidence="LOW",
                score_modifier=6,
                reason=f"Slight whale lean ({whale_pct:.0f}%)",
                warnings=["Wait for stronger signal"]
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 6: BEARISH WHALE CONVICTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    if whale_pct <= T.EXTREME_BEARISH:
        if position == 'LATE':
            return ScenarioResult(
                action="STRONG_SHORT",
                direction="SHORT",
                confidence="HIGH",
                score_modifier=22,
                reason=f"Extreme bearish whales ({whale_pct:.0f}%) at highs",
                warnings=[]
            )
        elif position == 'MID':
            return ScenarioResult(
                action="SHORT_SETUP",
                direction="SHORT",
                confidence="HIGH",
                score_modifier=16,
                reason=f"Extreme bearish whales ({whale_pct:.0f}%)",
                warnings=[]
            )
        else:  # EARLY - potential capitulation
            return ScenarioResult(
                action="MONITOR_SHORT",
                direction="SHORT",
                confidence="LOW",
                score_modifier=8,
                reason=f"Bearish whales but at lows - capitulation?",
                warnings=["Near lows - watch for reversal"]
            )
    
    elif whale_pct <= T.STRONG_BEARISH:
        if position in ['MID', 'LATE']:
            return ScenarioResult(
                action="SHORT_SETUP",
                direction="SHORT",
                confidence="MEDIUM",
                score_modifier=14,
                reason=f"Strong bearish whales ({whale_pct:.0f}%)",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="MONITOR_SHORT",
                direction="SHORT",
                confidence="LOW",
                score_modifier=6,
                reason=f"Bearish whales but near lows",
                warnings=["Potential capitulation zone"]
            )
    
    elif whale_pct <= T.BEARISH:
        if position == 'LATE':
            return ScenarioResult(
                action="BUILDING_SHORT",
                direction="SHORT",
                confidence="MEDIUM",
                score_modifier=12,
                reason=f"Bearish whales ({whale_pct:.0f}%) + LATE position",
                warnings=[]
            )
        else:
            return ScenarioResult(
                action="MONITOR_SHORT",
                direction="SHORT",
                confidence="LOW",
                score_modifier=6,
                reason=f"Bearish whales ({whale_pct:.0f}%)",
                warnings=[]
            )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RULE 7: NEUTRAL ZONE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # If we have retail warning but got here, return caution
    if has_retail_warning:
        return ScenarioResult(
            action="CAUTION_LONG",
            direction="WAIT",
            confidence="LOW",
            score_modifier=-5,
            reason=f"Retail ({retail_pct:.0f}%) > Whales ({whale_pct:.0f}%) - wait",
            warnings=[f"⚠️ Retail more bullish than whales"]
        )
    
    # True neutral
    return ScenarioResult(
        action="WAIT",
        direction="WAIT",
        confidence="LOW",
        score_modifier=0,
        reason=f"No clear edge - Whales {whale_pct:.0f}%, Retail {retail_pct:.0f}%",
        warnings=["Wait for clearer setup"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE TABLE (For documentation)
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_TABLE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    INVESTORIQ DECISION MATRIX                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ WHALE %  │ RETAIL % │ DIVERGENCE │ POSITION │ ACTION         │ DIRECTION    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          RETAIL TRAP (AVOID LONGS)                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ANY      │ W+20%+   │ -20%+      │ ANY      │ AVOID_LONG     │ WAIT         ║
║ ANY      │ W+15%+   │ -15%+      │ ANY      │ CAUTION_LONG   │ WAIT         ║
║ ANY      │ W+5%+    │ -5%+       │ ANY      │ CAUTION_LONG   │ WAIT         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          SQUEEZE (FAVOR LONGS)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ R+20%+   │ ANY      │ +20%+      │ EARLY    │ STRONG_LONG    │ LONG         ║
║ R+20%+   │ ANY      │ +20%+      │ MID      │ LONG_SETUP     │ LONG         ║
║ R+20%+   │ ANY      │ +20%+      │ LATE     │ BUILDING_LONG  │ LONG         ║
║ R+15%+   │ ANY      │ +15%+      │ EARLY/MID│ LONG_SETUP     │ LONG         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          CAPITULATION (REVERSAL LONG)                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ≤35%     │ ≤35%     │ ~0         │ EARLY    │ EARLY_LONG     │ LONG         ║
║ ≤35%     │ ≤35%     │ ~0         │ EARLY    │ +ACCUMULATION  │ HIGH CONF    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          EUPHORIA (REVERSAL SHORT)                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ≥65%     │ ≥65%     │ ~0         │ LATE     │ EARLY_SHORT    │ SHORT        ║
║ ≥65%     │ ≥65%     │ ~0         │ LATE     │ +DISTRIBUTION  │ HIGH CONF    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          WHALE CONVICTION (PRIMARY)                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ≥75%     │ <W       │ +          │ EARLY    │ STRONG_LONG    │ LONG         ║
║ ≥75%     │ <W       │ +          │ MID      │ LONG_SETUP     │ LONG         ║
║ ≥75%     │ <W       │ +          │ LATE     │ BUILDING_LONG  │ LONG         ║
║ ≥70%     │ <W       │ +          │ EARLY/MID│ LONG_SETUP     │ LONG         ║
║ ≥65%     │ <W       │ +          │ EARLY    │ BUILDING_LONG  │ LONG         ║
║ ≥65%     │ <W       │ +          │ MID      │ BUILDING_LONG  │ LONG         ║
║ ≥65%     │ <W       │ +          │ LATE     │ MONITOR_LONG   │ LONG         ║
║ ≥60%     │ <W       │ +          │ EARLY    │ BUILDING_LONG  │ LONG         ║
║ ≥60%     │ <W       │ +          │ MID/LATE │ MONITOR_LONG   │ LONG         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          BEARISH WHALE CONVICTION                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ≤25%     │ ANY      │ ANY        │ LATE     │ STRONG_SHORT   │ SHORT        ║
║ ≤25%     │ ANY      │ ANY        │ MID      │ SHORT_SETUP    │ SHORT        ║
║ ≤25%     │ ANY      │ ANY        │ EARLY    │ MONITOR_SHORT  │ SHORT        ║
║ ≤30%     │ ANY      │ ANY        │ MID/LATE │ SHORT_SETUP    │ SHORT        ║
║ ≤35%     │ ANY      │ ANY        │ LATE     │ BUILDING_SHORT │ SHORT        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                          NEUTRAL / NO EDGE                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 45-55%   │ 45-55%   │ ~0         │ ANY      │ WAIT           │ WAIT         ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY PRINCIPLES:
1. Retail > Whale on LONG signals = ALWAYS WAIT (trap risk)
2. Whale > Retail = Squeeze potential (favor direction)
3. Position matters: EARLY = better entry, LATE = tighten stops
4. Capitulation (both low) + EARLY = reversal long opportunity
5. Euphoria (both high) + LATE = reversal short opportunity
"""


# ═══════════════════════════════════════════════════════════════════════════════
# OI MODIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_oi_modifier(base_result: ScenarioResult, oi_change: float) -> ScenarioResult:
    """
    Apply OI (Open Interest) modifier to base scenario result.
    OI confirms or weakens the signal.
    """
    T = Thresholds
    
    if base_result.direction == 'LONG':
        if oi_change >= T.OI_STRONG_RISING:
            # Strong money inflow confirms long
            base_result.score_modifier += 4
            base_result.reason += f" | OI +{oi_change:.1f}% confirms"
        elif oi_change <= T.OI_STRONG_FALLING:
            # Money leaving - weakens long
            base_result.score_modifier -= 4
            base_result.warnings.append(f"⚠️ OI falling ({oi_change:.1f}%) - weak conviction")
            
    elif base_result.direction == 'SHORT':
        if oi_change <= T.OI_STRONG_FALLING:
            # Money leaving confirms short
            base_result.score_modifier += 4
            base_result.reason += f" | OI {oi_change:.1f}% confirms exit"
        elif oi_change >= T.OI_STRONG_RISING:
            # Money entering - might squeeze shorts
            base_result.score_modifier -= 4
            base_result.warnings.append(f"⚠️ OI rising ({oi_change:.1f}%) - squeeze risk")
    
    return base_result


# ═══════════════════════════════════════════════════════════════════════════════
# MONEY FLOW PHASE MODIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_phase_modifier(base_result: ScenarioResult, phase: str, position: str) -> ScenarioResult:
    """
    Apply Money Flow Phase modifier.
    Phase alignment increases confidence.
    """
    
    if base_result.direction == 'LONG':
        if phase == 'ACCUMULATION' and position == 'EARLY':
            base_result.score_modifier += 5
            base_result.confidence = 'HIGH' if base_result.confidence == 'MEDIUM' else base_result.confidence
            base_result.reason += " | Accumulation phase"
        elif phase == 'MARKUP':
            base_result.score_modifier += 3
            base_result.reason += " | Markup phase"
        elif phase == 'DISTRIBUTION':
            base_result.score_modifier -= 5
            base_result.warnings.append("⚠️ Distribution phase - exit longs")
        elif phase == 'MARKDOWN':
            base_result.score_modifier -= 8
            base_result.warnings.append("🚨 Markdown phase - avoid longs")
            
    elif base_result.direction == 'SHORT':
        if phase == 'DISTRIBUTION' and position == 'LATE':
            base_result.score_modifier += 5
            base_result.confidence = 'HIGH' if base_result.confidence == 'MEDIUM' else base_result.confidence
            base_result.reason += " | Distribution phase"
        elif phase == 'MARKDOWN':
            base_result.score_modifier += 3
            base_result.reason += " | Markdown phase"
        elif phase == 'ACCUMULATION':
            base_result.score_modifier -= 5
            base_result.warnings.append("⚠️ Accumulation phase - exit shorts")
        elif phase == 'MARKUP':
            base_result.score_modifier -= 8
            base_result.warnings.append("🚨 Markup phase - avoid shorts")
    
    return base_result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def get_trading_decision(
    whale_pct: float,
    retail_pct: float,
    position_pct: float,
    oi_change: float = 0,
    money_flow_phase: str = 'UNKNOWN'
) -> ScenarioResult:
    """
    SINGLE FUNCTION to get trading decision.
    
    This should be the ONLY function called from app.py for decisions.
    """
    # Get base scenario
    result = evaluate_scenario(
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        position_pct=position_pct,
        oi_change=oi_change,
        money_flow_phase=money_flow_phase
    )
    
    # Determine position label for modifiers
    T = Thresholds
    if position_pct <= T.EARLY_MAX:
        position = 'EARLY'
    elif position_pct <= T.MID_MAX:
        position = 'MID'
    else:
        position = 'LATE'
    
    # Apply modifiers
    result = apply_oi_modifier(result, oi_change)
    result = apply_phase_modifier(result, money_flow_phase, position)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(SCENARIO_TABLE)
    print("\n" + "="*80)
    print("TEST CASES")
    print("="*80)
    
    test_cases = [
        # (whale, retail, position, oi, phase, expected_direction)
        (63, 70, 8, -1.2, 'ACCUMULATION', "WAIT - Retail > Whale"),
        (75, 55, 20, 5, 'ACCUMULATION', "LONG - Squeeze + Early"),
        (30, 30, 10, -3, 'ACCUMULATION', "LONG - Capitulation"),
        (70, 70, 85, 2, 'DISTRIBUTION', "SHORT - Euphoria"),
        (68, 50, 45, 3, 'MARKUP', "LONG - Whale conviction"),
        (50, 50, 50, 0, 'UNKNOWN', "WAIT - Neutral"),
    ]
    
    for whale, retail, pos, oi, phase, expected in test_cases:
        result = get_trading_decision(whale, retail, pos, oi, phase)
        print(f"\nW:{whale}% R:{retail}% Pos:{pos}% OI:{oi} Phase:{phase}")
        print(f"  → {result.action} | {result.direction} | {result.confidence}")
        print(f"  → {result.reason}")
        if result.warnings:
            print(f"  → Warnings: {result.warnings}")
        print(f"  Expected: {expected}")
