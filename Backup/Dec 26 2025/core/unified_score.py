"""
Unified Scoring System
======================
Single scoring engine for ALL markets: Crypto, Stocks, ETFs

This replaces the scattered scoring logic with a clean, configurable system.

Key Features:
1. Works with Whale data (Crypto) OR Quiver data (Stocks/ETFs)
2. All thresholds from YAML config
3. Priority-based rule evaluation
4. Audit trail for debugging
5. Unified output format

Usage:
    from core.unified_score import calculate_score
    
    # For Crypto
    result = calculate_score(
        market_type='crypto',
        whale_pct=63,
        retail_pct=70,
        oi_change=-1.9,
        ...
    )
    
    # For Stocks
    result = calculate_score(
        market_type='stock',
        congress_score=72,
        insider_score=65,
        short_interest=15,
        ...
    )
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Import centralized decision matrix
try:
    from .MASTER_RULES import get_trade_decision as get_master_decision, is_valid_long
    MATRIX_AVAILABLE = True
except ImportError:
    MATRIX_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ═══════════════════════════════════════════════════════════════════════════════

_config: Dict = {}
_config_loaded = False

def _load_config():
    """Load configuration from YAML"""
    global _config, _config_loaded
    
    if _config_loaded:
        return _config
    
    config_paths = [
        'config/trading_config.yaml',
        '../config/trading_config.yaml',
        os.path.join(os.path.dirname(__file__), '..', 'config', 'trading_config.yaml'),
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                _config = yaml.safe_load(f)
                _config_loaded = True
                return _config
    
    # Fallback defaults
    _config = _get_default_config()
    _config_loaded = True
    return _config


def _get_default_config() -> Dict:
    """Default configuration if YAML not found"""
    return {
        'whale_thresholds': {
            'extreme_bullish': 75, 'strong_bullish': 70, 'bullish': 65,
            'lean_bullish': 60, 'neutral_high': 55, 'neutral_low': 45,
            'lean_bearish': 40, 'bearish': 35, 'strong_bearish': 30,
            'extreme_bearish': 25
        },
        'divergence_thresholds': {
            'extreme': 20, 'squeeze_setup': 15, 'clear_edge': 10,
            'slight_edge': 5, 'no_edge': 0
        },
        'position_thresholds': {
            'early_long': 35, 'late_long': 65,
            'early_short': 65, 'late_short': 35
        },
        'scoring': {
            'layer1_direction': {
                'max_points': 40, 'extreme_conviction': 40,
                'strong_conviction': 36, 'clear_direction': 32,
                'lean_direction': 28, 'slight_lean': 22, 'neutral': 15
            },
            'layer2_squeeze': {
                'max_points': 30, 'extreme_divergence': 22,
                'high_divergence': 18, 'medium_divergence': 14,
                'low_divergence': 10, 'no_divergence_base': 8
            },
            'layer3_entry': {
                'max_points': 30, 'ta_strong': 15, 'ta_good': 12,
                'ta_moderate': 9, 'ta_weak': 6, 'position_optimal': 18,
                'position_middle': 12, 'position_late': 5, 'position_chasing': 0
            }
        },
        'penalties': {
            'retail_overleveraged': {
                'extreme_threshold': 70, 'high_threshold': 65,
                'extreme_penalty': 12, 'high_penalty': 8
            }
        }
    }


def get_threshold(path: str, default: Any = None) -> Any:
    """Get config value using dot notation: 'whale_thresholds.strong_bullish'"""
    config = _load_config()
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoreResult:
    """Unified score result for any market type"""
    
    # Final outputs
    final_score: int                    # 0-100
    trade_direction: str                # 'LONG', 'SHORT', 'WAIT'
    confidence: str                     # 'HIGH', 'MEDIUM', 'LOW'
    final_action: str                   # 'STRONG_LONG', 'BUILDING_LONG', etc.
    
    # Layer breakdown
    layer1_direction_score: int         # 0-40
    layer1_direction: str               # 'BULLISH', 'LEAN_BULLISH', etc.
    layer1_reason: str
    
    layer2_squeeze_score: int           # 0-30
    layer2_potential: str               # 'EXTREME', 'HIGH', 'MEDIUM', etc.
    layer2_reason: str
    
    layer3_entry_score: int             # 0-30
    layer3_position: str                # 'EARLY', 'MIDDLE', 'LATE', 'CHASING'
    layer3_ta_label: str                # 'Strong', 'Good', 'Moderate', 'Weak'
    layer3_reason: str
    
    # Position info
    move_position: str                  # 'EARLY', 'MIDDLE', 'LATE', 'CHASING'
    move_position_pct: float            # 0-100
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    penalties_applied: int = 0
    
    # Audit trail
    fired_rules: List[str] = field(default_factory=list)
    
    # Market type
    market_type: str = 'crypto'
    
    # Raw data reference
    smart_money_pct: float = 50         # Whale% for crypto, Congress% for stocks
    retail_pct: float = 50              # Retail% for crypto, inverse for stocks
    divergence: float = 0


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1: DIRECTION SCORING
# Works for both Crypto (Whale%) and Stocks (Congress + Insider)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_layer1_direction(
    smart_money_pct: float,
    retail_pct: float,
    oi_change: float = 0,
    price_change: float = 0,
    market_type: str = 'crypto'
) -> Tuple[int, str, str, str, List[str]]:
    """
    Score Layer 1: Direction based on smart money positioning.
    
    For Crypto: smart_money_pct = whale long %
    For Stocks: smart_money_pct = (congress_score + insider_score) / 2
    
    Returns: (score, direction, confidence, reason, warnings)
    """
    wt = _load_config().get('whale_thresholds', {})
    sc = _load_config().get('scoring', {}).get('layer1_direction', {})
    penalties = _load_config().get('penalties', {}).get('retail_overleveraged', {})
    
    warnings = []
    
    # Determine direction from smart money
    if smart_money_pct >= wt.get('extreme_bullish', 75):
        direction = 'BULLISH'
        confidence = 'HIGH'
        score = sc.get('extreme_conviction', 40)
        reason = f"🔥 EXTREME smart money conviction ({smart_money_pct:.0f}%)"
        
    elif smart_money_pct >= wt.get('strong_bullish', 70):
        direction = 'BULLISH'
        confidence = 'HIGH'
        score = sc.get('strong_conviction', 36)
        reason = f"💪 Strong smart money LONG ({smart_money_pct:.0f}%)"
        
    elif smart_money_pct >= wt.get('bullish', 65):
        direction = 'BULLISH'
        confidence = 'HIGH'
        score = sc.get('clear_direction', 32)
        reason = f"✅ Clear smart money bullish ({smart_money_pct:.0f}%)"
        
    elif smart_money_pct >= wt.get('lean_bullish', 60):
        direction = 'LEAN_BULLISH'
        confidence = 'MEDIUM'
        score = sc.get('lean_direction', 28)
        reason = f"📈 Smart money leaning long ({smart_money_pct:.0f}%)"
        
    elif smart_money_pct >= wt.get('neutral_high', 55):
        direction = 'LEAN_BULLISH'
        confidence = 'LOW'
        score = sc.get('slight_lean', 22)
        reason = f"Slight smart money long bias ({smart_money_pct:.0f}%)"
        
    elif smart_money_pct <= wt.get('extreme_bearish', 25):
        direction = 'BEARISH'
        confidence = 'HIGH'
        score = sc.get('extreme_conviction', 40)
        reason = f"🔥 EXTREME smart money SHORT ({100-smart_money_pct:.0f}%S)"
        
    elif smart_money_pct <= wt.get('strong_bearish', 30):
        direction = 'BEARISH'
        confidence = 'HIGH'
        score = sc.get('strong_conviction', 36)
        reason = f"💪 Strong smart money SHORT ({100-smart_money_pct:.0f}%S)"
        
    elif smart_money_pct <= wt.get('bearish', 35):
        direction = 'BEARISH'
        confidence = 'HIGH'
        score = sc.get('clear_direction', 32)
        reason = f"✅ Clear smart money bearish ({100-smart_money_pct:.0f}%S)"
        
    elif smart_money_pct <= wt.get('lean_bearish', 40):
        direction = 'LEAN_BEARISH'
        confidence = 'MEDIUM'
        score = sc.get('lean_direction', 28)
        reason = f"📉 Smart money leaning short ({100-smart_money_pct:.0f}%S)"
        
    elif smart_money_pct <= wt.get('neutral_low', 45):
        direction = 'LEAN_BEARISH'
        confidence = 'LOW'
        score = sc.get('slight_lean', 22)
        reason = f"Slight smart money short bias ({100-smart_money_pct:.0f}%S)"
        
    else:
        direction = 'NEUTRAL'
        confidence = 'LOW'
        score = sc.get('neutral', 15)
        reason = f"⏸️ Smart money neutral ({smart_money_pct:.0f}%) - no clear edge"
    
    # OI confirmation bonus (crypto only)
    if market_type == 'crypto' and oi_change != 0:
        oi_up = oi_change > 0.5
        price_up = price_change > 0.5
        
        if direction in ['BULLISH', 'LEAN_BULLISH'] and oi_up and price_up:
            score = min(40, score + 5)
            reason += f" | OI confirms (+{oi_change:.1f}%)"
            confidence = 'HIGH'
        elif direction in ['BEARISH', 'LEAN_BEARISH'] and oi_up and not price_up:
            score = min(40, score + 5)
            reason += f" | OI confirms (+{oi_change:.1f}%)"
            confidence = 'HIGH'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RETAIL OVERLEVERAGED CHECK - Critical for avoiding traps!
    # ═══════════════════════════════════════════════════════════════════════════
    divergence = smart_money_pct - retail_pct
    
    if divergence < -5 and direction in ['BULLISH', 'LEAN_BULLISH']:
        # Retail more bullish than smart money - WARNING!
        if retail_pct >= penalties.get('extreme_threshold', 70):
            penalty = penalties.get('extreme_penalty', 12)
            score = max(15, score - penalty)
            confidence = 'LOW'
            warnings.append(f"⚠️ RETAIL TRAP: Retail ({retail_pct:.0f}%) > Smart Money ({smart_money_pct:.0f}%)!")
            reason = f"⚠️ Retail overleveraged ({retail_pct:.0f}% vs {smart_money_pct:.0f}%)"
            
            # If extreme mismatch, change direction to WAIT
            if retail_pct >= 70 and smart_money_pct < 65:
                direction = 'NEUTRAL'
                warnings.append("Direction changed to NEUTRAL - retail trap risk too high")
                
        elif retail_pct >= penalties.get('high_threshold', 65):
            penalty = penalties.get('high_penalty', 8)
            score = max(18, score - penalty)
            warnings.append(f"⚠️ Retail ({retail_pct:.0f}%) more bullish than Smart Money ({smart_money_pct:.0f}%)")
    
    elif divergence > 5 and direction in ['BEARISH', 'LEAN_BEARISH']:
        # Retail more bearish than smart money
        retail_short = 100 - retail_pct
        if retail_short >= penalties.get('extreme_threshold', 70):
            penalty = penalties.get('extreme_penalty', 12)
            score = max(15, score - penalty)
            confidence = 'LOW'
            warnings.append(f"⚠️ RETAIL TRAP: Retail short ({retail_short:.0f}%) > Smart Money short!")
    
    return score, direction, confidence, reason, warnings


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2: SQUEEZE/EDGE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _score_layer2_squeeze(
    smart_money_pct: float,
    retail_pct: float,
    direction: str,
    market_type: str = 'crypto',
    short_interest: float = 0  # For stocks
) -> Tuple[int, str, str]:
    """
    Score Layer 2: Squeeze potential / Trading edge.
    
    For Crypto: Based on whale vs retail divergence
    For Stocks: Based on institutional alignment + short interest
    
    Returns: (score, potential, reason)
    """
    dt = _load_config().get('divergence_thresholds', {})
    sc = _load_config().get('scoring', {}).get('layer2_squeeze', {})
    wt = _load_config().get('whale_thresholds', {})
    
    divergence = smart_money_pct - retail_pct
    abs_divergence = abs(divergence)
    
    # Smart money conviction bonus (even without divergence)
    conviction_bonus = 0
    if smart_money_pct >= wt.get('strong_bullish', 70) or smart_money_pct <= wt.get('strong_bearish', 30):
        conviction_bonus = 10
    elif smart_money_pct >= wt.get('bullish', 65) or smart_money_pct <= wt.get('bearish', 35):
        conviction_bonus = 7
    elif smart_money_pct >= wt.get('lean_bullish', 60) or smart_money_pct <= wt.get('lean_bearish', 40):
        conviction_bonus = 4
    
    # Negative divergence penalty (retail more positioned)
    negative_penalty = 0
    if divergence < -5 and retail_pct > smart_money_pct:
        if retail_pct >= 70:
            negative_penalty = -12
        elif retail_pct >= 65:
            negative_penalty = -8
        elif retail_pct >= 60:
            negative_penalty = -5
    
    # Score based on divergence direction and magnitude
    if divergence >= dt.get('extreme', 20):
        # EXTREME positive divergence - SHORT SQUEEZE
        potential = "EXTREME_SQUEEZE"
        score = sc.get('extreme_divergence', 22) + conviction_bonus
        reason = f"🔥 SQUEEZE: Smart money {smart_money_pct:.0f}% vs Retail {retail_pct:.0f}% - shorts TRAPPED"
        
    elif divergence <= -dt.get('extreme', 20):
        # EXTREME negative divergence - LONG SQUEEZE (retail trap)
        potential = "RETAIL_TRAP"
        score = sc.get('no_divergence_base', 8) + negative_penalty
        reason = f"⚠️ TRAP: Retail {retail_pct:.0f}% > Smart Money {smart_money_pct:.0f}% - longs at risk!"
        
    elif divergence >= dt.get('squeeze_setup', 15):
        potential = "HIGH_SQUEEZE"
        score = sc.get('high_divergence', 18) + conviction_bonus
        reason = f"🔥 Squeeze setup: SM:{smart_money_pct:.0f}% vs R:{retail_pct:.0f}%"
        
    elif divergence <= -dt.get('squeeze_setup', 15):
        potential = "RETAIL_WARNING"
        score = sc.get('low_divergence', 10) + negative_penalty
        reason = f"⚠️ Retail overleveraged: R:{retail_pct:.0f}% > SM:{smart_money_pct:.0f}%"
        
    elif abs_divergence >= dt.get('clear_edge', 10):
        if divergence > 0:
            potential = "MEDIUM_EDGE"
            score = sc.get('medium_divergence', 14) + conviction_bonus
            reason = f"Edge: SM:{smart_money_pct:.0f}% vs R:{retail_pct:.0f}%"
        else:
            potential = "LOW_EDGE"
            score = max(6, sc.get('medium_divergence', 14) + negative_penalty)
            reason = f"⚠️ Retail more bullish: R:{retail_pct:.0f}% vs SM:{smart_money_pct:.0f}%"
            
    elif abs_divergence >= dt.get('slight_edge', 5):
        if divergence > 0:
            potential = "SLIGHT_EDGE"
            score = sc.get('low_divergence', 10) + conviction_bonus
            reason = f"Slight edge: SM:{smart_money_pct:.0f}% vs R:{retail_pct:.0f}%"
        else:
            potential = "SLIGHT_WARNING"
            score = max(6, sc.get('low_divergence', 10) + negative_penalty)
            reason = f"Retail slightly more positioned"
            
    else:
        # No significant divergence
        if conviction_bonus >= 7:
            potential = "CONVICTION_ONLY"
            score = sc.get('no_divergence_base', 8) + conviction_bonus
            reason = f"No divergence but strong conviction ({smart_money_pct:.0f}%)"
        else:
            potential = "NO_EDGE"
            score = sc.get('no_divergence_base', 8)
            reason = f"Neutral: SM:{smart_money_pct:.0f}% ≈ R:{retail_pct:.0f}%"
    
    # For stocks: Add short interest bonus
    if market_type == 'stock' and short_interest > 0:
        if short_interest >= 20 and direction in ['BULLISH', 'LEAN_BULLISH']:
            score = min(30, score + 5)
            reason += f" | High short interest ({short_interest:.0f}%) = squeeze potential"
            potential = "SHORT_SQUEEZE_SETUP"
    
    score = max(0, min(30, score))
    return score, potential, reason


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3: ENTRY TIMING SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def _score_layer3_entry(
    position_pct: float,
    ta_score: float,
    direction: str
) -> Tuple[int, str, str, str, str]:
    """
    Score Layer 3: Entry timing based on position in range + TA.
    
    Returns: (score, position_label, ta_label, move_position, reason)
    """
    pt = _load_config().get('position_thresholds', {})
    sc = _load_config().get('scoring', {}).get('layer3_entry', {})
    
    # TA Component (15 pts max)
    if ta_score >= 70:
        ta_pts = sc.get('ta_strong', 15)
        ta_label = "Strong"
    elif ta_score >= 55:
        ta_pts = sc.get('ta_good', 12)
        ta_label = "Good"
    elif ta_score >= 40:
        ta_pts = sc.get('ta_moderate', 9)
        ta_label = "Moderate"
    else:
        ta_pts = sc.get('ta_weak', 6)
        ta_label = "Weak"
    
    # Position Component (18 pts max)
    whale_bullish = direction in ['BULLISH', 'LEAN_BULLISH']
    whale_bearish = direction in ['BEARISH', 'LEAN_BEARISH']
    
    if whale_bullish:
        # For LONGS: want to be NEAR LOWS
        if position_pct <= pt.get('early_long', 35):
            pos_pts = sc.get('position_optimal', 18)
            move_position = "EARLY"
        elif position_pct <= 50:
            pos_pts = sc.get('position_middle', 12)
            move_position = "MIDDLE"
        elif position_pct <= pt.get('late_long', 65):
            pos_pts = sc.get('position_late', 5)
            move_position = "LATE"
        else:
            pos_pts = sc.get('position_chasing', 0)
            move_position = "CHASING"
            
    elif whale_bearish:
        # For SHORTS: want to be NEAR HIGHS
        if position_pct >= pt.get('early_short', 65):
            pos_pts = sc.get('position_optimal', 18)
            move_position = "EARLY"
        elif position_pct >= 50:
            pos_pts = sc.get('position_middle', 12)
            move_position = "MIDDLE"
        elif position_pct >= pt.get('late_short', 35):
            pos_pts = sc.get('position_late', 5)
            move_position = "LATE"
        else:
            pos_pts = sc.get('position_chasing', 0)
            move_position = "CHASING"
    else:
        # NEUTRAL - position less important
        move_position = "MIDDLE"
        pos_pts = sc.get('position_middle', 12)
    
    total = ta_pts + pos_pts
    reason = f"TA: {ta_label} ({ta_pts}pts) | Position: {move_position} ({pos_pts}pts)"
    
    return total, move_position, ta_label, move_position, reason


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL ACTION DETERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

def _determine_action(
    direction: str,
    confidence: str,
    squeeze_potential: str,
    move_position: str,
    final_score: int,
    warnings: List[str]
) -> Tuple[str, str]:
    """
    Determine final action and trade direction.
    
    Returns: (action_name, trade_direction)
    """
    # Detect retail trap more broadly - ANY retail > whale divergence is a warning for longs
    has_retail_trap = any('TRAP' in w or 'overleveraged' in w.lower() or 'Retail' in w for w in warnings)
    
    # Also check squeeze potential for retail warning
    is_retail_warning = squeeze_potential in ['RETAIL_TRAP', 'RETAIL_WARNING', 'SLIGHT_WARNING', 'LOW_EDGE']
    
    # For LONG signals: if retail is more bullish than whales, be cautious
    retail_long_risk = is_retail_warning and direction in ['BULLISH', 'LEAN_BULLISH']
    
    # HIGH CONFIDENCE actions
    if confidence == 'HIGH' and not has_retail_trap and not retail_long_risk:
        if direction == 'BULLISH':
            if squeeze_potential in ['EXTREME_SQUEEZE', 'HIGH_SQUEEZE']:
                return "STRONG_LONG", "LONG"
            elif move_position == 'EARLY':
                return "LONG_SETUP", "LONG"
            else:
                return "BUILDING_LONG", "LONG"
                
        elif direction == 'BEARISH':
            if squeeze_potential in ['EXTREME_SQUEEZE', 'HIGH_SQUEEZE']:
                return "STRONG_SHORT", "SHORT"
            elif move_position == 'EARLY':
                return "SHORT_SETUP", "SHORT"
            else:
                return "BUILDING_SHORT", "SHORT"
    
    # MEDIUM CONFIDENCE actions
    if confidence == 'MEDIUM' and not has_retail_trap and not retail_long_risk:
        if direction == 'LEAN_BULLISH':
            if move_position in ['EARLY', 'MIDDLE']:
                return "BUILDING_LONG", "LONG"
            else:
                return "WAIT_LONG", "LONG"
                
        elif direction == 'LEAN_BEARISH':
            if move_position in ['EARLY', 'MIDDLE']:
                return "BUILDING_SHORT", "SHORT"
            else:
                return "WAIT_SHORT", "SHORT"
    
    # LOW CONFIDENCE or WARNINGS
    if has_retail_trap or retail_long_risk:
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            return "CAUTION_LONG", "WAIT"
        elif direction in ['BEARISH', 'LEAN_BEARISH']:
            return "CAUTION_SHORT", "WAIT"
    
    if direction == 'NEUTRAL':
        return "WAIT", "WAIT"
    
    # Default: cautious approach
    if direction in ['BULLISH', 'LEAN_BULLISH']:
        return "MONITOR_LONG", "LONG"
    elif direction in ['BEARISH', 'LEAN_BEARISH']:
        return "MONITOR_SHORT", "SHORT"
    
    return "WAIT", "WAIT"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_score(
    market_type: str = 'crypto',
    # Crypto inputs (Whale data)
    whale_pct: float = None,
    retail_pct: float = None,
    oi_change: float = 0,
    funding_rate: float = 0,
    # Stock inputs (Quiver data)
    congress_score: float = None,
    insider_score: float = None,
    short_interest: float = 0,
    # Common inputs
    price_change: float = 0,
    position_pct: float = 50,
    ta_score: float = 50,
    swing_high: float = 0,
    swing_low: float = 0,
    current_price: float = 0,
    money_flow_phase: str = "UNKNOWN",
    structure: str = "UNKNOWN",
) -> ScoreResult:
    """
    Calculate unified score for any market type.
    
    For Crypto:
        result = calculate_score(
            market_type='crypto',
            whale_pct=63,
            retail_pct=70,
            oi_change=-1.9,
            position_pct=17,
            ta_score=65
        )
    
    For Stocks/ETFs:
        result = calculate_score(
            market_type='stock',
            congress_score=72,
            insider_score=65,
            short_interest=15,
            position_pct=30,
            ta_score=60
        )
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NORMALIZE INPUTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if market_type in ['crypto', 'Crypto', 'cryptocurrency']:
        # Crypto: Use whale and retail data directly
        smart_money_pct = whale_pct if whale_pct is not None else 50
        retail = retail_pct if retail_pct is not None else 50
        market = 'crypto'
        
    else:
        # Stocks/ETFs: Combine congress and insider scores
        congress = congress_score if congress_score is not None else 50
        insider = insider_score if insider_score is not None else 50
        
        # Smart money = average of congress and insider
        # (Congress trades often have more predictive value, so weight it higher)
        smart_money_pct = (congress * 0.6) + (insider * 0.4)
        
        # For stocks, we don't have "retail" positioning like crypto
        # Use inverse of smart money as proxy, or neutral
        retail = 100 - smart_money_pct if smart_money_pct != 50 else 50
        market = 'stock'
    
    # Calculate position in range if not provided
    if position_pct == 50 and swing_high and swing_low and current_price:
        if swing_high > swing_low:
            position_pct = ((current_price - swing_low) / (swing_high - swing_low)) * 100
            position_pct = max(0, min(100, position_pct))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCORE EACH LAYER
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Layer 1: Direction
    l1_score, direction, confidence, l1_reason, warnings = _score_layer1_direction(
        smart_money_pct, retail, oi_change, price_change, market
    )
    
    # Layer 2: Squeeze/Edge
    l2_score, squeeze_potential, l2_reason = _score_layer2_squeeze(
        smart_money_pct, retail, direction, market, short_interest
    )
    
    # Layer 3: Entry Timing
    l3_score, position_label, ta_label, move_position, l3_reason = _score_layer3_entry(
        position_pct, ta_score, direction
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CALCULATE FINAL SCORE
    # ═══════════════════════════════════════════════════════════════════════════
    
    final_score = l1_score + l2_score + l3_score
    final_score = max(0, min(100, final_score))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # USE DECISION MATRIX FOR CRYPTO (if available)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if market == 'crypto' and MATRIX_AVAILABLE and whale_pct is not None and retail_pct is not None:
        # Use MASTER_RULES for final decision - SINGLE SOURCE OF TRUTH
        master_result = get_master_decision(
            whale_pct=smart_money_pct,
            retail_pct=retail,
            position_pct=position_pct if position_pct else 50,
            oi_change=oi_change if oi_change else 0,
            price_change=0,  # Price change not available here
            ta_score=ta_score,
            money_flow_phase=money_flow_phase
        )
        
        final_action = master_result.action
        trade_direction = master_result.trade_direction  # Note: trade_direction not direction
        
        # If MASTER_RULES says long is invalid, respect that
        if not master_result.is_valid_long and trade_direction == 'LONG':
            trade_direction = 'WAIT'
            final_action = 'CAUTION_LONG'
        
        # Add master warnings
        for w in master_result.warnings:
            if w not in warnings:
                warnings.append(w)
        
        # Override confidence if master has higher conviction
        if master_result.confidence == 'HIGH' and confidence != 'HIGH':
            confidence = master_result.confidence
        elif master_result.confidence == 'LOW':
            confidence = 'LOW'
    else:
        # Fallback to original logic for stocks or when MASTER_RULES unavailable
        final_action, trade_direction = _determine_action(
            direction, confidence, squeeze_potential, move_position, final_score, warnings
        )
    
    # Calculate total penalties
    penalties_applied = 0
    for w in warnings:
        if 'TRAP' in w:
            penalties_applied += 12
        elif 'overleveraged' in w.lower():
            penalties_applied += 8
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD RESULT
    # ═══════════════════════════════════════════════════════════════════════════
    
    return ScoreResult(
        final_score=final_score,
        trade_direction=trade_direction,
        confidence=confidence,
        final_action=final_action,
        
        layer1_direction_score=l1_score,
        layer1_direction=direction,
        layer1_reason=l1_reason,
        
        layer2_squeeze_score=l2_score,
        layer2_potential=squeeze_potential,
        layer2_reason=l2_reason,
        
        layer3_entry_score=l3_score,
        layer3_position=position_label,
        layer3_ta_label=ta_label,
        layer3_reason=l3_reason,
        
        move_position=move_position,
        move_position_pct=position_pct,
        
        warnings=warnings,
        penalties_applied=penalties_applied,
        
        market_type=market,
        smart_money_pct=smart_money_pct,
        retail_pct=retail,
        divergence=smart_money_pct - retail,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def score_crypto(
    whale_pct: float,
    retail_pct: float,
    oi_change: float = 0,
    position_pct: float = 50,
    ta_score: float = 50,
    **kwargs
) -> ScoreResult:
    """Convenience function for crypto scoring"""
    return calculate_score(
        market_type='crypto',
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        oi_change=oi_change,
        position_pct=position_pct,
        ta_score=ta_score,
        **kwargs
    )


def score_stock(
    congress_score: float,
    insider_score: float,
    short_interest: float = 0,
    position_pct: float = 50,
    ta_score: float = 50,
    **kwargs
) -> ScoreResult:
    """Convenience function for stock scoring"""
    return calculate_score(
        market_type='stock',
        congress_score=congress_score,
        insider_score=insider_score,
        short_interest=short_interest,
        position_pct=position_pct,
        ta_score=ta_score,
        **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("UNIFIED SCORING SYSTEM TEST")
    print("=" * 70)
    
    # Test 1: Crypto - PORTALUSDT scenario (retail trap)
    print("\n📊 TEST 1: PORTALUSDT (Crypto - Retail Trap)")
    print("-" * 50)
    result = score_crypto(
        whale_pct=63,
        retail_pct=70,
        oi_change=-1.9,
        position_pct=17,
        ta_score=65
    )
    print(f"Final Score: {result.final_score}")
    print(f"Direction: {result.layer1_direction} ({result.confidence})")
    print(f"Trade: {result.trade_direction}")
    print(f"Action: {result.final_action}")
    print(f"Warnings: {result.warnings}")
    print(f"Layer 1: {result.layer1_direction_score}/40 - {result.layer1_reason}")
    print(f"Layer 2: {result.layer2_squeeze_score}/30 - {result.layer2_reason}")
    print(f"Layer 3: {result.layer3_entry_score}/30 - {result.layer3_reason}")
    
    # Test 2: Crypto - Strong bullish setup
    print("\n📊 TEST 2: Strong Bullish (Crypto)")
    print("-" * 50)
    result = score_crypto(
        whale_pct=72,
        retail_pct=48,
        oi_change=3.5,
        position_pct=25,
        ta_score=68
    )
    print(f"Final Score: {result.final_score}")
    print(f"Direction: {result.layer1_direction} ({result.confidence})")
    print(f"Trade: {result.trade_direction}")
    print(f"Action: {result.final_action}")
    
    # Test 3: Stock - Congress buying
    print("\n📊 TEST 3: AAPL (Stock - Congress Buying)")
    print("-" * 50)
    result = score_stock(
        congress_score=75,
        insider_score=68,
        short_interest=8,
        position_pct=35,
        ta_score=62
    )
    print(f"Final Score: {result.final_score}")
    print(f"Smart Money: {result.smart_money_pct:.1f}%")
    print(f"Direction: {result.layer1_direction} ({result.confidence})")
    print(f"Trade: {result.trade_direction}")
    print(f"Action: {result.final_action}")
    
    # Test 4: Stock - High short interest
    print("\n📊 TEST 4: GME-style (Stock - High Short Interest)")
    print("-" * 50)
    result = score_stock(
        congress_score=65,
        insider_score=70,
        short_interest=25,
        position_pct=40,
        ta_score=55
    )
    print(f"Final Score: {result.final_score}")
    print(f"Direction: {result.layer1_direction}")
    print(f"Trade: {result.trade_direction}")
    print(f"Action: {result.final_action}")
    print(f"Layer 2: {result.layer2_reason}")
