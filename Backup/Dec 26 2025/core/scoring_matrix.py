"""
═══════════════════════════════════════════════════════════════════════════════
INVESTORIQ SCORING MATRIX - KNOWLEDGE BASE
═══════════════════════════════════════════════════════════════════════════════

This module defines ALL combinations of TA + Whale/Institutional signals
and their expected outcomes. Instead of hardcoded if/else logic scattered
throughout the code, all scoring rules are defined here in one place.

PHILOSOPHY:
- TA Score = Technical foundation (chart patterns, indicators, structure)
- Whale Score = Smart money confirmation (OI, funding, positioning)
- Combined = Final verdict based on alignment/conflict

RULE: When TA and Whale CONFLICT, the lower confidence wins (be cautious)
RULE: When TA and Whale ALIGN, confidence is boosted
RULE: Neutral whale data = trade on TA alone (no penalty, no boost)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS FOR CLEAR CATEGORIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TAStrength(Enum):
    """Technical Analysis strength categories"""
    EXCEPTIONAL = "exceptional"  # 85-100: Multiple confirmations, A+ setup
    STRONG = "strong"            # 70-84: Good setup, clear direction
    MODERATE = "moderate"        # 55-69: Decent setup, some concerns
    WEAK = "weak"                # 40-54: Mixed signals
    POOR = "poor"                # 0-39: No clear setup

class WhaleVerdict(Enum):
    """Whale/Institutional verdict from API data"""
    STRONG_BUY = "strong_buy"      # High confidence bullish
    BUY = "buy"                    # Moderate bullish
    LEAN_LONG = "lean_long"        # Slight bullish bias
    NEUTRAL = "neutral"            # No clear edge
    LEAN_SHORT = "lean_short"      # Slight bearish bias
    SELL = "sell"                  # Moderate bearish
    STRONG_SELL = "strong_sell"    # High confidence bearish
    WAIT = "wait"                  # Mixed signals, wait for clarity
    AVOID = "avoid"                # Conflicting data, skip

class WhaleConfidence(Enum):
    """Confidence level of whale verdict"""
    HIGH = "high"       # 70%+ positioning, clear signals
    MEDIUM = "medium"   # 55-70% positioning, some signals
    LOW = "low"         # Near 50% positioning, mixed signals

class FinalAction(Enum):
    """Final recommended action"""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    CAUTIOUS_BUY = "CAUTIOUS BUY"
    HOLD = "HOLD"
    WAIT = "WAIT"
    CAUTIOUS_SELL = "CAUTIOUS SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    AVOID = "AVOID"

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING MATRIX - ALL COMBINATIONS DEFINED HERE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoringRule:
    """A single scoring rule defining outcome for a combination"""
    ta_strength: TAStrength
    whale_verdict: WhaleVerdict
    whale_confidence: WhaleConfidence
    
    # Outputs
    combined_score_range: Tuple[int, int]  # (min, max) score
    final_action: FinalAction
    position_size: str  # "100%", "75%", "50%", "25%", "0%"
    reasoning: str
    warnings: List[str]

# The master scoring matrix
SCORING_MATRIX: List[ScoringRule] = [
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXCEPTIONAL TA (85-100)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Exceptional TA + Bullish Whale
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.STRONG_BUY, WhaleConfidence.HIGH,
        (95, 100), FinalAction.STRONG_BUY, "100%",
        "Perfect alignment: A+ setup with strong whale confirmation",
        []
    ),
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.BUY, WhaleConfidence.HIGH,
        (90, 95), FinalAction.STRONG_BUY, "100%",
        "Excellent setup with whale support",
        []
    ),
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.BUY, WhaleConfidence.MEDIUM,
        (85, 90), FinalAction.BUY, "100%",
        "Strong setup with moderate whale support",
        []
    ),
    
    # Exceptional TA + Neutral Whale (NO PENALTY - trade on TA!)
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.NEUTRAL, WhaleConfidence.LOW,
        (80, 88), FinalAction.BUY, "75%",
        "Excellent TA setup, whales neutral - trade the chart",
        ["Whale data neutral - use tight stops"]
    ),
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.WAIT, WhaleConfidence.LOW,
        (75, 85), FinalAction.CAUTIOUS_BUY, "75%",
        "Strong TA but whale says wait - proceed with caution",
        ["⚠️ Whale verdict: WAIT - reduce size, tight stops"]
    ),
    
    # Exceptional TA + Bearish Whale (CONFLICT!)
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.SELL, WhaleConfidence.HIGH,
        (55, 65), FinalAction.WAIT, "25%",
        "⚠️ CONFLICT: Great TA but whales are selling!",
        ["🚨 Smart money divergence - high risk", "Wait for alignment"]
    ),
    ScoringRule(
        TAStrength.EXCEPTIONAL, WhaleVerdict.AVOID, WhaleConfidence.LOW,
        (60, 70), FinalAction.CAUTIOUS_BUY, "50%",
        "Strong TA but mixed whale signals",
        ["Mixed institutional signals - reduce exposure"]
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRONG TA (70-84)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Strong TA + Bullish Whale
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.STRONG_BUY, WhaleConfidence.HIGH,
        (88, 95), FinalAction.STRONG_BUY, "100%",
        "Good setup with strong whale confirmation",
        []
    ),
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.BUY, WhaleConfidence.HIGH,
        (82, 88), FinalAction.BUY, "100%",
        "Solid setup with whale support",
        []
    ),
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.BUY, WhaleConfidence.MEDIUM,
        (75, 82), FinalAction.BUY, "75%",
        "Good setup with moderate whale support",
        []
    ),
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.LEAN_LONG, WhaleConfidence.MEDIUM,
        (72, 78), FinalAction.BUY, "75%",
        "Good setup with slight whale bias long",
        []
    ),
    
    # Strong TA + Neutral Whale
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.NEUTRAL, WhaleConfidence.LOW,
        (68, 75), FinalAction.BUY, "75%",
        "Good TA, neutral whales - trade the chart",
        ["No whale confirmation - standard position"]
    ),
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.WAIT, WhaleConfidence.LOW,
        (60, 70), FinalAction.CAUTIOUS_BUY, "50%",
        "Good TA but whale data says wait",
        ["⚠️ Whale verdict: WAIT - reduce size"]
    ),
    
    # Strong TA + Bearish Whale
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.SELL, WhaleConfidence.HIGH,
        (45, 55), FinalAction.WAIT, "0%",
        "⚠️ CONFLICT: Good TA but whales selling",
        ["🚨 Do NOT enter - wait for resolution"]
    ),
    ScoringRule(
        TAStrength.STRONG, WhaleVerdict.LEAN_SHORT, WhaleConfidence.MEDIUM,
        (55, 65), FinalAction.CAUTIOUS_BUY, "25%",
        "Good TA but whales slightly short",
        ["Reduced conviction - small size only"]
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MODERATE TA (55-69)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Moderate TA + Bullish Whale (whale can save a mediocre setup!)
    ScoringRule(
        TAStrength.MODERATE, WhaleVerdict.STRONG_BUY, WhaleConfidence.HIGH,
        (75, 82), FinalAction.BUY, "75%",
        "Mediocre TA but strong whale buying - follow smart money",
        ["Whale accumulation may precede move"]
    ),
    ScoringRule(
        TAStrength.MODERATE, WhaleVerdict.BUY, WhaleConfidence.HIGH,
        (68, 75), FinalAction.BUY, "75%",
        "Average setup boosted by whale buying",
        []
    ),
    ScoringRule(
        TAStrength.MODERATE, WhaleVerdict.BUY, WhaleConfidence.MEDIUM,
        (62, 68), FinalAction.CAUTIOUS_BUY, "50%",
        "Average setup with some whale support",
        []
    ),
    
    # Moderate TA + Neutral Whale
    ScoringRule(
        TAStrength.MODERATE, WhaleVerdict.NEUTRAL, WhaleConfidence.LOW,
        (55, 62), FinalAction.CAUTIOUS_BUY, "50%",
        "Average setup, no whale edge",
        ["Standard risk management required"]
    ),
    ScoringRule(
        TAStrength.MODERATE, WhaleVerdict.WAIT, WhaleConfidence.LOW,
        (45, 55), FinalAction.WAIT, "25%",
        "Mediocre TA + whale says wait = skip",
        ["Not enough edge - wait for better setup"]
    ),
    
    # Moderate TA + Bearish Whale
    ScoringRule(
        TAStrength.MODERATE, WhaleVerdict.SELL, WhaleConfidence.HIGH,
        (30, 40), FinalAction.AVOID, "0%",
        "Weak TA + bearish whales = AVOID",
        ["🚨 No edge - skip this trade"]
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEAK TA (40-54)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Weak TA + Strong Bullish Whale (whale divergence - interesting!)
    ScoringRule(
        TAStrength.WEAK, WhaleVerdict.STRONG_BUY, WhaleConfidence.HIGH,
        (60, 70), FinalAction.CAUTIOUS_BUY, "50%",
        "📊 DIVERGENCE: Weak chart but whales accumulating!",
        ["Smart money may see what chart doesn't show", "Small size, wide stops"]
    ),
    ScoringRule(
        TAStrength.WEAK, WhaleVerdict.BUY, WhaleConfidence.HIGH,
        (55, 62), FinalAction.CAUTIOUS_BUY, "25%",
        "Weak TA but whale buying - watch for breakout",
        ["Wait for TA confirmation before adding"]
    ),
    
    # Weak TA + Neutral/Wait Whale
    ScoringRule(
        TAStrength.WEAK, WhaleVerdict.NEUTRAL, WhaleConfidence.LOW,
        (40, 50), FinalAction.WAIT, "0%",
        "Weak setup + no whale edge = no trade",
        ["Wait for better opportunity"]
    ),
    ScoringRule(
        TAStrength.WEAK, WhaleVerdict.WAIT, WhaleConfidence.LOW,
        (35, 45), FinalAction.AVOID, "0%",
        "No edge from TA or whales",
        ["Skip - look elsewhere"]
    ),
    
    # Weak TA + Bearish Whale
    ScoringRule(
        TAStrength.WEAK, WhaleVerdict.SELL, WhaleConfidence.HIGH,
        (20, 30), FinalAction.STRONG_SELL, "100%",
        "Weak TA + whale selling = SHORT opportunity",
        ["Consider short position if allowed"]
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POOR TA (0-39)
    # ═══════════════════════════════════════════════════════════════════════════
    
    ScoringRule(
        TAStrength.POOR, WhaleVerdict.STRONG_BUY, WhaleConfidence.HIGH,
        (50, 60), FinalAction.WAIT, "0%",
        "Bad chart but whale accumulation - wait for structure",
        ["Don't chase - wait for TA to catch up"]
    ),
    ScoringRule(
        TAStrength.POOR, WhaleVerdict.NEUTRAL, WhaleConfidence.LOW,
        (25, 35), FinalAction.AVOID, "0%",
        "No setup, no edge",
        []
    ),
    ScoringRule(
        TAStrength.POOR, WhaleVerdict.SELL, WhaleConfidence.HIGH,
        (10, 20), FinalAction.STRONG_SELL, "100%",
        "Everything bearish - strong short signal",
        []
    ),
]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_ta_strength(ta_score: int) -> TAStrength:
    """Convert numeric TA score to category"""
    if ta_score >= 85:
        return TAStrength.EXCEPTIONAL
    elif ta_score >= 70:
        return TAStrength.STRONG
    elif ta_score >= 55:
        return TAStrength.MODERATE
    elif ta_score >= 40:
        return TAStrength.WEAK
    else:
        return TAStrength.POOR

def get_whale_verdict(unified_verdict: dict) -> WhaleVerdict:
    """Convert unified verdict dict to WhaleVerdict enum"""
    if not unified_verdict:
        return WhaleVerdict.NEUTRAL
    
    # 🐛 FIX: get_unified_verdict() uses 'unified_action', not 'action'!
    action = str(unified_verdict.get('unified_action', unified_verdict.get('action', 'WAIT'))).upper()
    confidence = str(unified_verdict.get('confidence', 'LOW')).upper()
    
    # Also check scenario for additional context
    scenario = unified_verdict.get('scenario', {})
    scenario_name = str(scenario.get('name', '')).upper() if scenario else ''
    
    # Map action strings to verdict - check LONG/SHORT as well as BUY/SELL
    if 'STRONG' in action and ('BUY' in action or 'LONG' in action):
        return WhaleVerdict.STRONG_BUY
    elif 'LONG' in action and confidence == 'HIGH':
        return WhaleVerdict.BUY  # HIGH confidence LONG = BUY
    elif 'LONG' in action:
        return WhaleVerdict.LEAN_LONG
    elif 'BUY' in action and confidence == 'HIGH':
        return WhaleVerdict.BUY
    elif 'BUY' in action:
        return WhaleVerdict.LEAN_LONG
    elif 'STRONG' in action and ('SELL' in action or 'SHORT' in action):
        return WhaleVerdict.STRONG_SELL
    elif 'SHORT' in action and confidence == 'HIGH':
        return WhaleVerdict.SELL  # HIGH confidence SHORT = SELL
    elif 'SHORT' in action:
        return WhaleVerdict.LEAN_SHORT
    elif 'SELL' in action and confidence == 'HIGH':
        return WhaleVerdict.SELL
    elif 'SELL' in action:
        return WhaleVerdict.LEAN_SHORT
    elif 'AVOID' in action:
        return WhaleVerdict.AVOID
    elif 'WAIT' in action or 'RANGE' in action:
        # Check if it's really neutral or if scenario says bullish/bearish
        if 'BULLISH' in scenario_name or 'LONG' in scenario_name:
            return WhaleVerdict.LEAN_LONG if confidence == 'HIGH' else WhaleVerdict.NEUTRAL
        elif 'BEARISH' in scenario_name or 'SHORT' in scenario_name:
            return WhaleVerdict.LEAN_SHORT if confidence == 'HIGH' else WhaleVerdict.NEUTRAL
        return WhaleVerdict.WAIT
    else:
        # Check institutional_bias as fallback
        inst_bias = str(unified_verdict.get('institutional_bias', '')).upper()
        if inst_bias == 'BULLISH' and confidence == 'HIGH':
            return WhaleVerdict.BUY
        elif inst_bias == 'BULLISH':
            return WhaleVerdict.LEAN_LONG
        elif inst_bias == 'BEARISH' and confidence == 'HIGH':
            return WhaleVerdict.SELL
        elif inst_bias == 'BEARISH':
            return WhaleVerdict.LEAN_SHORT
        return WhaleVerdict.NEUTRAL

def get_whale_confidence(unified_verdict: dict, whale_data: dict = None) -> WhaleConfidence:
    """Determine whale confidence level"""
    if not unified_verdict:
        return WhaleConfidence.LOW
    
    conf_str = str(unified_verdict.get('confidence', 'LOW')).upper()
    
    if conf_str == 'HIGH':
        return WhaleConfidence.HIGH
    elif conf_str == 'MEDIUM':
        return WhaleConfidence.MEDIUM
    else:
        return WhaleConfidence.LOW

def find_matching_rule(ta_score: int, unified_verdict: dict, whale_data: dict = None) -> Optional[ScoringRule]:
    """Find the best matching rule from the scoring matrix"""
    ta_strength = get_ta_strength(ta_score)
    whale_verdict = get_whale_verdict(unified_verdict)
    whale_confidence = get_whale_confidence(unified_verdict, whale_data)
    
    # Find exact match first
    for rule in SCORING_MATRIX:
        if (rule.ta_strength == ta_strength and 
            rule.whale_verdict == whale_verdict and
            rule.whale_confidence == whale_confidence):
            return rule
    
    # Find partial match (same TA + whale verdict, any confidence)
    for rule in SCORING_MATRIX:
        if (rule.ta_strength == ta_strength and 
            rule.whale_verdict == whale_verdict):
            return rule
    
    # Find match with similar whale verdict
    similar_verdicts = {
        WhaleVerdict.STRONG_BUY: [WhaleVerdict.BUY, WhaleVerdict.LEAN_LONG],
        WhaleVerdict.BUY: [WhaleVerdict.STRONG_BUY, WhaleVerdict.LEAN_LONG],
        WhaleVerdict.LEAN_LONG: [WhaleVerdict.BUY, WhaleVerdict.NEUTRAL],
        WhaleVerdict.NEUTRAL: [WhaleVerdict.WAIT, WhaleVerdict.LEAN_LONG, WhaleVerdict.LEAN_SHORT],
        WhaleVerdict.WAIT: [WhaleVerdict.NEUTRAL, WhaleVerdict.AVOID],
        WhaleVerdict.AVOID: [WhaleVerdict.WAIT, WhaleVerdict.NEUTRAL],
        WhaleVerdict.LEAN_SHORT: [WhaleVerdict.SELL, WhaleVerdict.NEUTRAL],
        WhaleVerdict.SELL: [WhaleVerdict.STRONG_SELL, WhaleVerdict.LEAN_SHORT],
        WhaleVerdict.STRONG_SELL: [WhaleVerdict.SELL, WhaleVerdict.LEAN_SHORT],
    }
    
    for similar in similar_verdicts.get(whale_verdict, []):
        for rule in SCORING_MATRIX:
            if rule.ta_strength == ta_strength and rule.whale_verdict == similar:
                return rule
    
    # Fallback: return a default neutral rule
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_combined_score(
    ta_score: int,
    unified_verdict: dict = None,
    whale_data: dict = None
) -> dict:
    """
    Calculate combined score using the knowledge base.
    
    Args:
        ta_score: Technical analysis score (0-100)
        unified_verdict: Dict from get_unified_verdict() with action, confidence, scenario
        whale_data: Raw whale data dict (optional, for additional context)
    
    Returns:
        dict with:
            - combined_score: Final score (0-100)
            - final_action: Recommended action string
            - position_size: Recommended position size
            - reasoning: Why this score
            - warnings: List of warning messages
            - ta_strength: TA category
            - whale_verdict: Whale category
            - whale_confidence: Confidence level
            - rule_matched: Whether we found a matching rule
    """
    
    # Get categories
    ta_strength = get_ta_strength(ta_score)
    whale_verdict = get_whale_verdict(unified_verdict)
    whale_confidence = get_whale_confidence(unified_verdict, whale_data)
    
    # Find matching rule
    rule = find_matching_rule(ta_score, unified_verdict, whale_data)
    
    if rule:
        # Use rule to calculate score
        min_score, max_score = rule.combined_score_range
        
        # Interpolate within range based on TA score
        ta_ranges = {
            TAStrength.EXCEPTIONAL: (85, 100),
            TAStrength.STRONG: (70, 84),
            TAStrength.MODERATE: (55, 69),
            TAStrength.WEAK: (40, 54),
            TAStrength.POOR: (0, 39),
        }
        ta_min, ta_max = ta_ranges[ta_strength]
        
        # Linear interpolation
        if ta_max > ta_min:
            ta_pct = (ta_score - ta_min) / (ta_max - ta_min)
        else:
            ta_pct = 0.5
        
        combined_score = int(min_score + (max_score - min_score) * ta_pct)
        combined_score = max(0, min(100, combined_score))
        
        return {
            'combined_score': combined_score,
            'final_action': rule.final_action.value,
            'position_size': rule.position_size,
            'reasoning': rule.reasoning,
            'warnings': rule.warnings,
            'ta_strength': ta_strength.value,
            'whale_verdict': whale_verdict.value,
            'whale_confidence': whale_confidence.value,
            'rule_matched': True,
        }
    
    else:
        # Fallback calculation when no rule matches
        # This ensures we always return something reasonable
        
        # Base: weighted average
        whale_score = 50  # Default neutral
        if whale_verdict in [WhaleVerdict.STRONG_BUY, WhaleVerdict.BUY]:
            whale_score = 75 if whale_confidence == WhaleConfidence.HIGH else 65
        elif whale_verdict in [WhaleVerdict.LEAN_LONG]:
            whale_score = 58
        elif whale_verdict in [WhaleVerdict.LEAN_SHORT]:
            whale_score = 42
        elif whale_verdict in [WhaleVerdict.SELL, WhaleVerdict.STRONG_SELL]:
            whale_score = 25 if whale_confidence == WhaleConfidence.HIGH else 35
        elif whale_verdict in [WhaleVerdict.WAIT, WhaleVerdict.AVOID]:
            whale_score = 45  # Slight penalty for uncertainty
        
        # Asymmetric weighting: TA 60%, Whale 40%
        combined_score = int(ta_score * 0.6 + whale_score * 0.4)
        combined_score = max(0, min(100, combined_score))
        
        # Determine action
        if combined_score >= 75:
            action = FinalAction.STRONG_BUY.value
            size = "100%"
        elif combined_score >= 65:
            action = FinalAction.BUY.value
            size = "75%"
        elif combined_score >= 55:
            action = FinalAction.CAUTIOUS_BUY.value
            size = "50%"
        elif combined_score >= 45:
            action = FinalAction.WAIT.value
            size = "25%"
        else:
            action = FinalAction.AVOID.value
            size = "0%"
        
        warnings = []
        if whale_verdict in [WhaleVerdict.WAIT, WhaleVerdict.AVOID]:
            warnings.append("⚠️ Whale data inconclusive - use caution")
        
        return {
            'combined_score': combined_score,
            'final_action': action,
            'position_size': size,
            'reasoning': f"TA {ta_strength.value} + Whale {whale_verdict.value} (fallback calculation)",
            'warnings': warnings,
            'ta_strength': ta_strength.value,
            'whale_verdict': whale_verdict.value,
            'whale_confidence': whale_confidence.value,
            'rule_matched': False,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: PRINT ALL RULES (for debugging/documentation)
# ═══════════════════════════════════════════════════════════════════════════════

def print_scoring_matrix():
    """Print the entire scoring matrix for review"""
    print("\n" + "="*80)
    print("INVESTORIQ SCORING MATRIX - ALL RULES")
    print("="*80 + "\n")
    
    current_ta = None
    for rule in SCORING_MATRIX:
        if rule.ta_strength != current_ta:
            current_ta = rule.ta_strength
            print(f"\n{'─'*40}")
            print(f"📊 {current_ta.value.upper()} TA")
            print(f"{'─'*40}")
        
        print(f"\n  + {rule.whale_verdict.value} ({rule.whale_confidence.value})")
        print(f"    Score: {rule.combined_score_range[0]}-{rule.combined_score_range[1]}")
        print(f"    Action: {rule.final_action.value} | Size: {rule.position_size}")
        print(f"    {rule.reasoning}")
        if rule.warnings:
            for w in rule.warnings:
                print(f"    ⚠️ {w}")


if __name__ == "__main__":
    # Print all rules when run directly
    print_scoring_matrix()
    
    # Test examples
    print("\n" + "="*80)
    print("TEST EXAMPLES")
    print("="*80)
    
    # Test 1: Your case - High TA (90) + WAIT verdict
    result = calculate_combined_score(
        ta_score=90,
        unified_verdict={'action': 'WAIT', 'confidence': 'LOW', 'scenario': {}}
    )
    print(f"\nTest 1: TA=90 + WAIT/LOW")
    print(f"  Combined: {result['combined_score']}")
    print(f"  Action: {result['final_action']}")
    print(f"  Reasoning: {result['reasoning']}")
    
    # Test 2: Perfect alignment
    result = calculate_combined_score(
        ta_score=90,
        unified_verdict={'action': 'STRONG BUY', 'confidence': 'HIGH', 'scenario': {}}
    )
    print(f"\nTest 2: TA=90 + STRONG BUY/HIGH")
    print(f"  Combined: {result['combined_score']}")
    print(f"  Action: {result['final_action']}")
    
    # Test 3: Conflict
    result = calculate_combined_score(
        ta_score=85,
        unified_verdict={'action': 'SELL', 'confidence': 'HIGH', 'scenario': {}}
    )
    print(f"\nTest 3: TA=85 + SELL/HIGH (CONFLICT)")
    print(f"  Combined: {result['combined_score']}")
    print(f"  Action: {result['final_action']}")
    print(f"  Warnings: {result['warnings']}")
