"""
📊 INSTITUTIONAL SCORING MODULE
================================
This module provides backward compatibility while using the new rules engine.

All analysis is now handled by the configuration-driven rules engine.
Edit thresholds in: core/institutional_config.json

Author: InvestorIQ
Version: 2.0.0 (Rules Engine Backend)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT FROM RULES ENGINE (The actual analysis happens there)
# ═══════════════════════════════════════════════════════════════════════════════

from .rules_engine import (
    # Main analysis function
    analyze_institutional_data,
    analyze as analyze_raw,
    
    # Engine access
    get_engine,
    InstitutionalRulesEngine,
    
    # Data classes
    MarketData,
    AnalysisResult,
    LegacyCompatResult,
    
    # Enums
    Direction,
    Status,
    Confidence,
    
    # Configuration
    DEFAULT_CONFIG,
)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

class TradeStatus(Enum):
    """Status of a potential trade setup"""
    READY = "READY"
    WAIT = "WAIT"
    AVOID = "AVOID"
    CONFLICTING = "CONFLICTING"


class ScenarioType(Enum):
    """Types of market scenarios"""
    NEW_LONGS_WHALE_BULLISH = "NEW_LONGS_WHALE_BULLISH"
    NEW_LONGS_NEUTRAL_WHALES = "NEW_LONGS_NEUTRAL_WHALES"
    SHORT_SQUEEZE = "SHORT_SQUEEZE"
    LONG_LIQUIDATION_WHALE_BULLISH = "LONG_LIQUIDATION_WHALE_BULLISH"
    NEW_SHORTS_WHALE_BEARISH = "NEW_SHORTS_WHALE_BEARISH"
    NEW_SHORTS_NEUTRAL_WHALES = "NEW_SHORTS_NEUTRAL_WHALES"
    LONG_SQUEEZE = "LONG_SQUEEZE"
    SHORT_COVERING_WHALE_BEARISH = "SHORT_COVERING_WHALE_BEARISH"
    SHORT_COVERING_NO_CONVICTION = "SHORT_COVERING_NO_CONVICTION"
    LONG_LIQUIDATION_NO_SUPPORT = "LONG_LIQUIDATION_NO_SUPPORT"
    CONFLICTING = "CONFLICTING_SIGNALS"
    NO_EDGE = "NO_EDGE"


# Type alias for backward compatibility
InstitutionalScore = LegacyCompatResult


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DISPLAY (Loads from config)
# ═══════════════════════════════════════════════════════════════════════════════

def get_scenario_display(scenario) -> Dict:
    """
    Get display information for a scenario.
    
    Args:
        scenario: ScenarioType enum, string ID, or AnalysisResult
    
    Returns:
        Dict with emoji, action, status, description, what_to_do
    """
    engine = get_engine()
    
    # Handle different input types
    if isinstance(scenario, AnalysisResult):
        scenario_id = scenario.scenario_id
    elif isinstance(scenario, ScenarioType):
        scenario_id = scenario.value
    elif hasattr(scenario, 'scenario'):
        # LegacyCompatResult
        scenario_id = scenario.scenario
    else:
        scenario_id = str(scenario).upper().replace(' ', '_')
    
    # Get from config
    scenario_def = engine.scenarios.get(scenario_id, engine.scenarios.get('NO_EDGE', {}))
    
    # Support both old keys (description/action) and new keys (interpretation/recommendation)
    description = scenario_def.get('description', '') or scenario_def.get('interpretation', '')
    what_to_do = scenario_def.get('action', '') or scenario_def.get('recommendation', '')
    
    return {
        'emoji': scenario_def.get('emoji', '⚪'),
        'action': scenario_def.get('direction', 'NEUTRAL'),
        'status': scenario_def.get('status', 'AVOID'),
        'description': description,
        'what_to_do': what_to_do,
        'education': scenario_def.get('education', '')
    }


# Build SCENARIO_DETAILS from config
def _build_scenario_details() -> Dict:
    """Build scenario details dict from config"""
    engine = get_engine()
    details = {}
    for scenario_id, scenario_def in engine.scenarios.items():
        # Support both old keys (description/action) and new keys (interpretation/recommendation)
        description = scenario_def.get('description', '') or scenario_def.get('interpretation', '')
        what_to_do = scenario_def.get('action', '') or scenario_def.get('recommendation', '')
        
        details[scenario_id] = {
            'emoji': scenario_def.get('emoji', '⚪'),
            'action': scenario_def.get('direction', 'NEUTRAL'),
            'status': scenario_def.get('status', 'AVOID'),
            'description': description,
            'what_to_do': what_to_do
        }
    return details

SCENARIO_DETAILS = _build_scenario_details()


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST FUNCTIONALITY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WatchlistItem:
    """Item in the watchlist for WAIT trades"""
    symbol: str
    timeframe: str
    direction: str  # LONG or SHORT
    scenario: str   # Scenario ID string
    
    # Entry conditions
    entry_conditions: List[str] = field(default_factory=list)
    conditions_met: List[bool] = field(default_factory=list)
    
    # Trigger prices
    trigger_above: float = 0.0
    trigger_below: float = 0.0
    invalidation_price: float = 0.0
    
    # Tracking
    added_at: datetime = field(default_factory=datetime.now)
    price_at_add: float = 0.0
    whale_pct_at_add: float = 50.0
    
    # Status
    active: bool = True
    triggered: bool = False
    invalidated: bool = False
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED SCORING (Tech + Institutional)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_combined_score(
    technical_score: int,
    institutional_score
) -> Tuple[int, str, Dict]:
    """
    Calculate combined ranking score for scanner.
    
    Args:
        technical_score: Score from technical analysis (0-100)
        institutional_score: Result from analyze_institutional_data
    
    Returns:
        Tuple of (combined_score, status, details_dict)
    """
    # Handle both new AnalysisResult and legacy LegacyCompatResult
    if hasattr(institutional_score, 'institutional_score'):
        inst_score_val = institutional_score.institutional_score
        confidence = institutional_score.confidence if isinstance(institutional_score.confidence, str) else institutional_score.confidence.value
        alignment = getattr(institutional_score, 'alignment_with_tech', 'NEUTRAL')
        direction = institutional_score.direction_bias if isinstance(institutional_score.direction_bias, str) else institutional_score.direction_bias.value
        status_val = institutional_score.status.value if hasattr(institutional_score.status, 'value') else str(institutional_score.status)
    else:
        inst_score_val = 50
        confidence = "LOW"
        alignment = "NEUTRAL"
        direction = "NEUTRAL"
        status_val = "AVOID"
    
    # Alignment multiplier
    if alignment == "ALIGNED":
        alignment_mult = 1.25
    elif alignment == "CONFLICTING":
        alignment_mult = 0.75
    else:
        alignment_mult = 1.0
    
    # Confidence multiplier
    conf_mult = {"HIGH": 1.15, "MEDIUM": 1.0, "LOW": 0.85}.get(confidence, 1.0)
    
    # Calculate combined score
    base_score = (technical_score * 0.6) + (inst_score_val * 0.4)
    combined = int(base_score * alignment_mult * conf_mult)
    combined = max(0, min(100, combined))
    
    # Determine status
    if combined >= 70 and status_val == "READY":
        status = "READY"
    elif combined >= 50:
        status = "WATCH" if status_val in ["WAIT", "CONFLICTING"] else "READY"
    else:
        status = "AVOID"
    
    details = {
        'technical_score': technical_score,
        'institutional_score': inst_score_val,
        'alignment': alignment,
        'confidence': confidence,
        'alignment_multiplier': alignment_mult,
        'confidence_multiplier': conf_mult,
        'direction': direction
    }
    
    return combined, status, details


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def create_watchlist_item(
    symbol: str,
    timeframe: str,
    institutional_score,
    current_price: float,
    recent_high: float = 0.0,
    recent_low: float = 0.0
) -> Optional[WatchlistItem]:
    """
    Create a watchlist item for a WAIT scenario.
    """
    # Get attributes from score
    if hasattr(institutional_score, 'status'):
        status = institutional_score.status
        if hasattr(status, 'value'):
            status = status.value
    else:
        return None
    
    if status != "WAIT":
        return None
    
    direction = getattr(institutional_score, 'direction_bias', 'NEUTRAL')
    if hasattr(direction, 'value'):
        direction = direction.value
    
    scenario = getattr(institutional_score, 'scenario', 'NO_EDGE')
    if hasattr(scenario, 'value'):
        scenario = scenario.value
    
    wait_conditions = getattr(institutional_score, 'wait_conditions', [])
    whale_pct = getattr(institutional_score, 'whale_pct', 50.0)
    
    # Calculate trigger and invalidation prices
    if direction == "LONG":
        trigger_above = current_price * 1.02
        trigger_below = 0
        invalidation = recent_low if recent_low > 0 else current_price * 0.95
    elif direction == "SHORT":
        trigger_above = 0
        trigger_below = current_price * 0.98
        invalidation = recent_high if recent_high > 0 else current_price * 1.05
    else:
        trigger_above = current_price * 1.02
        trigger_below = current_price * 0.98
        invalidation = 0
    
    return WatchlistItem(
        symbol=symbol,
        timeframe=timeframe,
        direction=direction,
        scenario=scenario,
        entry_conditions=wait_conditions,
        conditions_met=[False] * len(wait_conditions),
        trigger_above=trigger_above,
        trigger_below=trigger_below,
        invalidation_price=invalidation,
        price_at_add=current_price,
        whale_pct_at_add=whale_pct
    )


def check_watchlist_triggers(
    item: WatchlistItem,
    current_price: float,
    current_whale_pct: float = 50.0
) -> Tuple[str, str]:
    """
    Check if watchlist item has been triggered or invalidated.
    """
    if not item.active:
        return "INACTIVE", "Item no longer active"
    
    # Check invalidation first
    if item.invalidation_price > 0:
        if item.direction == "LONG" and current_price < item.invalidation_price:
            return "INVALIDATED", f"Price below invalidation ({item.invalidation_price:.4f})"
        elif item.direction == "SHORT" and current_price > item.invalidation_price:
            return "INVALIDATED", f"Price above invalidation ({item.invalidation_price:.4f})"
    
    # Check triggers
    if item.trigger_above > 0 and current_price > item.trigger_above:
        return "TRIGGERED", f"Price broke above trigger ({item.trigger_above:.4f})"
    
    if item.trigger_below > 0 and current_price < item.trigger_below:
        return "TRIGGERED", f"Price broke below trigger ({item.trigger_below:.4f})"
    
    # Check whale positioning change
    whale_change = current_whale_pct - item.whale_pct_at_add
    if item.direction == "LONG" and whale_change < -10:
        return "INVALIDATED", f"Whales turned bearish"
    elif item.direction == "SHORT" and whale_change > 10:
        return "INVALIDATED", f"Whales turned bullish"
    
    return "WAITING", f"Monitoring... Price: {current_price:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== INSTITUTIONAL SCORING (Rules Engine Backend) ===\n")
    
    result = analyze_institutional_data(1.6, 1.7, 55, 56, 0.00005, "BULLISH")
    
    print(f"Input: OI +1.6%, Price +1.7%, Whales 55%, Retail 56%")
    print(f"Scenario: {result.scenario_name}")
    print(f"Direction: {result.direction_bias}")
    print(f"Confidence: {result.confidence}")
    print(f"Status: {result.status}")
    print(f"Score: {result.institutional_score}")
    
    display = get_scenario_display(result)
    print(f"\nDisplay: {display['emoji']} {display['description']}")
    
    combined, status, details = calculate_combined_score(75, result)
    print(f"\nCombined Score: {combined} ({status})")
