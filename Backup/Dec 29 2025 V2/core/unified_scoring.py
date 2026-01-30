"""
═══════════════════════════════════════════════════════════════════════════════
INVESTORIQ UNIFIED SCORING ENGINE
═══════════════════════════════════════════════════════════════════════════════

This is the SINGLE SOURCE OF TRUTH for all scores in the application.

ARCHITECTURE:
────────────────────────────────────────────────────────────────────────────────

LAYER 1: RAW DATA
    ├── Chart Data (OHLCV, Volume)
    ├── Binance API (OI, Funding, Positioning) - Crypto
    └── Quiver API (Congress, Insider, Short) - Stocks

LAYER 2: PROCESSORS  
    ├── TA Processor → ta_score, ta_factors
    └── Whale Processor → whale_verdict, whale_confidence, scenario

LAYER 3: UNIFIED SCORING ENGINE (THIS FILE)
    Input:  ta_score + whale_verdict
    Output: ALL scores derived consistently
        • ta_score (passed through)
        • inst_score (DERIVED from whale verdict - NOT calculated separately!)
        • combined_score (from scoring rules)
        • final_action
        • position_size
        • alignment
        • warnings

LAYER 4: DISPLAY
    All UI components read from Layer 3 output - no independent calculations!

────────────────────────────────────────────────────────────────────────────────

RULE: inst_score is NOT calculated from chart patterns.
      It is DERIVED from the whale/institutional verdict.
      This ensures all scores tell the SAME story.

────────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Import MASTER_RULES for final validation
try:
    from .MASTER_RULES import get_trade_decision, is_valid_long
    MASTER_RULES_AVAILABLE = True
except ImportError:
    MASTER_RULES_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TAStrength(Enum):
    """Technical Analysis strength categories"""
    EXCEPTIONAL = "exceptional"  # 85-100
    STRONG = "strong"            # 70-84
    MODERATE = "moderate"        # 55-69
    WEAK = "weak"                # 40-54
    POOR = "poor"                # 0-39


class WhaleVerdict(Enum):
    """Whale/Institutional verdict"""
    STRONG_BULLISH = "strong_bullish"   # Very high conviction long
    BULLISH = "bullish"                  # High conviction long
    LEAN_BULLISH = "lean_bullish"        # Slight bullish bias
    NEUTRAL = "neutral"                  # No clear edge
    LEAN_BEARISH = "lean_bearish"        # Slight bearish bias
    BEARISH = "bearish"                  # High conviction short
    STRONG_BEARISH = "strong_bearish"   # Very high conviction short
    WAIT = "wait"                        # Mixed signals - wait
    AVOID = "avoid"                      # Conflicting - skip


class Confidence(Enum):
    """Confidence level"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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
# TRADE MODE CONFIGURATION - Timeframe-aware weighting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeModeConfig:
    """Configuration for each trade mode"""
    name: str
    ta_weight: float          # How much TA matters (0-1)
    whale_weight: float       # How much whale data matters (0-1)
    min_oi_threshold: float   # Minimum OI change to be "significant"
    min_whale_pct: float      # Minimum whale % to be "bullish"
    data_relevance: str       # Description of whale data relevance


# Timeframe → Trade Mode mapping
TIMEFRAME_TO_MODE = {
    '1m': 'Scalp',
    '5m': 'Scalp', 
    '15m': 'DayTrade',
    '1h': 'DayTrade',
    '4h': 'Swing',
    '1d': 'Swing',
    '1w': 'Investment',
}

# Trade Mode configurations
# WHY these weights?
# - 24h whale data covers 288 candles on 5m = mostly noise for scalpers
# - 24h whale data covers 24 candles on 1h = somewhat relevant for day traders
# - 24h whale data covers 6 candles on 4h = highly relevant for swing traders
# - 24h whale data is same timeframe for weekly = critical for investors
TRADE_MODE_CONFIG = {
    'Scalp': TradeModeConfig(
        name='Scalp',
        ta_weight=0.90,           # 90% TA - whale data almost irrelevant
        whale_weight=0.10,        # 10% Whale - 24h data too old
        min_oi_threshold=2.0,     # Need >2% OI change to matter for scalp
        min_whale_pct=60.0,       # Need strong whale signal (60%+)
        data_relevance='LOW - 24h data covers 288+ candles, mostly noise'
    ),
    'DayTrade': TradeModeConfig(
        name='DayTrade',
        ta_weight=0.75,           # 75% TA
        whale_weight=0.25,        # 25% Whale - some relevance
        min_oi_threshold=1.0,     # Need >1% OI change
        min_whale_pct=57.0,       # Need decent whale signal (57%+)
        data_relevance='MEDIUM - 24h data covers 24-96 candles'
    ),
    'Swing': TradeModeConfig(
        name='Swing',
        ta_weight=0.50,           # 50% TA
        whale_weight=0.50,        # 50% Whale - highly relevant!
        min_oi_threshold=0.5,     # 0.5% OI is meaningful
        min_whale_pct=55.0,       # Standard threshold
        data_relevance='HIGH - 24h data covers 6-24 candles, very relevant'
    ),
    'Investment': TradeModeConfig(
        name='Investment',
        ta_weight=0.40,           # 40% TA
        whale_weight=0.60,        # 60% Whale - institutional moves key!
        min_oi_threshold=0.3,     # Even small OI matters
        min_whale_pct=54.0,       # Slightly lower threshold
        data_relevance='CRITICAL - Institutional positioning drives multi-day moves'
    ),
}


def get_trade_mode(timeframe: str) -> str:
    """Get trade mode from timeframe"""
    return TIMEFRAME_TO_MODE.get(timeframe, 'DayTrade')


def normalize_trade_mode(mode: str) -> str:
    """
    Normalize trade mode names from various sources to our standard names.
    
    Handles various inputs:
    - 'scalp', 'Scalp' → 'Scalp'
    - 'day_trade', 'Day Trade', 'DayTrade' → 'DayTrade'
    - 'swing', 'Swing Trade', 'SwingTrade', 'Swing' → 'Swing'
    - 'investment', 'Investment', 'invest' → 'Investment'
    """
    if not mode:
        return 'DayTrade'
    
    mode_lower = str(mode).lower().replace(' ', '').replace('_', '')
    
    if mode_lower in ['scalp', 'scalping']:
        return 'Scalp'
    elif mode_lower in ['daytrade', 'daytrading', 'intraday']:
        return 'DayTrade'
    elif mode_lower in ['swing', 'swingtrade', 'swingtrading']:
        return 'Swing'
    elif mode_lower in ['investment', 'invest', 'investing', 'longterm', 'position']:
        return 'Investment'
    else:
        return 'DayTrade'  # Default


def get_mode_config(trade_mode: str = None, timeframe: str = None) -> TradeModeConfig:
    """Get configuration for a trade mode"""
    if trade_mode is None and timeframe:
        trade_mode = get_trade_mode(timeframe)
    if trade_mode is None:
        trade_mode = 'DayTrade'  # Default
    
    # Normalize the mode name
    trade_mode = normalize_trade_mode(trade_mode)
    
    return TRADE_MODE_CONFIG.get(trade_mode, TRADE_MODE_CONFIG['DayTrade'])


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 NEW 3-LAYER PREDICTIVE SCORING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
#
# LAYER 1: OI + PRICE (40 points max) - WHERE is price going?
#          This is the PRIMARY predictive signal
#
# LAYER 2: WHALE vs RETAIL (30 points max) - HOW explosive will the move be?
#          High divergence = squeeze potential = bigger move
#
# LAYER 3: TA (30 points max) - WHEN to enter?
#          Good TA = enter now, Bad TA = wait for pullback
#
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictiveScore:
    """Output of the 3-layer predictive scoring system"""
    # Layer 1: Direction
    direction: str              # 'BULLISH', 'BEARISH', 'NEUTRAL'
    direction_confidence: str   # 'HIGH', 'MEDIUM', 'LOW'
    direction_score: int        # 0-40
    direction_reason: str       # "New longs entering (OI↑ Price↑)"
    
    # Layer 2: Squeeze Potential
    squeeze_potential: str      # 'HIGH', 'MEDIUM', 'LOW', 'NONE'
    divergence_pct: float       # Whale% - Retail%
    squeeze_score: int          # 0-30
    squeeze_reason: str         # "Whales 70% vs Retail 40% = squeeze setup"
    
    # Layer 3: Entry Timing (TA + Move Position)
    entry_timing: str           # 'NOW', 'SOON', 'WAIT'
    ta_score: int               # Raw TA score 0-100
    timing_score: int           # 0-30 (TA 20 + Move 10)
    timing_reason: str          # "Strong TA + EARLY (20%)"
    move_position: str          # 'EARLY', 'MIDDLE', 'LATE', 'RANGE'
    move_position_pct: float    # Position in swing range 0-100%
    
    # Final
    final_score: int            # 0-100
    final_action: str           # 'STRONG BUY', 'BUY', 'WAIT FOR PULLBACK', 'AVOID'
    final_summary: str          # Human readable summary
    
    # Trade context
    trade_mode: str
    timeframe: str
    
    # 🎯 SINGLE SOURCE OF TRUTH for setup direction
    trade_direction: str = 'WAIT'  # 'LONG', 'SHORT', 'WAIT', 'CAUTION'
    
    # 🎯 THE predictive signal dict (source of truth for combined learning!)
    predictive_signal: dict = None  # Contains signal, subtitle, action, interpretation


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 PREDICTIVE SIGNAL SYSTEM - LEADING INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
# 
# KEY INSIGHT: Structure is LAGGING, Money Flow + Whales are LEADING
#
# The RACE CAR analogy:
# - Structure = Where the car WAS (lagging)
# - Money Flow = Engine revving (leading) 
# - Whales = Smart money positioned (leading)
# - WAIT for structure = Other cars already gone!
#
# PREDICTIVE DIVERGENCE = Leading says UP while Lagging says DOWN (or vice versa)
# This is the EDGE - we see the move BEFORE it happens!
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 COMPREHENSIVE KNOWLEDGE BASE - ALL COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# 
# This knowledge base covers ALL possible combinations of:
# - Money Flow Phase (9 phases)
# - Whale Positioning (3 states: Bullish >55%, Bearish <45%, Neutral 45-55%)
# - Market Structure (3 states: Bullish, Bearish, Neutral/Ranging)
# - Asset-specific indicators (OI, Funding for Crypto)
#
# Total: 9 phases × 3 whale states × 3 structures = 81 base combinations
# Plus asset-specific rules for Crypto (OI, Funding) and Stocks/ETFs (no whale data)
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# PHASE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
MONEY_FLOW_PHASES = {
    'ACCUMULATION': {
        'description': 'Smart money quietly buying from panicked/tired retail',
        'whale_behavior': 'Buying aggressively',
        'retail_behavior': 'Selling at perceived loss',
        'phase_type': 'BULLISH',
    },
    'MARKUP': {
        'description': 'Trend confirmed, price rising with volume',
        'whale_behavior': 'Holding and adding on dips',
        'retail_behavior': 'FOMO buying late',
        'phase_type': 'BULLISH',
    },
    'RE-ACCUMULATION': {
        'description': 'Pause in uptrend, consolidation before next leg',
        'whale_behavior': 'Adding to positions',
        'retail_behavior': 'Taking profits too early',
        'phase_type': 'BULLISH',
    },
    'DISTRIBUTION': {
        'description': 'Smart money selling into strength',
        'whale_behavior': 'Selling to retail buyers',
        'retail_behavior': 'Buying the "breakout"',
        'phase_type': 'BEARISH',
    },
    'FOMO / DIST RISK': {
        'description': 'Retail FOMO at highs, smart money exiting',
        'whale_behavior': 'Selling to FOMO buyers',
        'retail_behavior': 'Chasing higher',
        'phase_type': 'BEARISH',
    },
    'EXHAUSTION': {
        'description': 'Final push before reversal, smart money gone',
        'whale_behavior': 'Already exited',
        'retail_behavior': 'Last buyers in',
        'phase_type': 'BEARISH',
    },
    'CONSOLIDATION': {
        'description': 'Range-bound, waiting for catalyst',
        'whale_behavior': 'Neutral, watching',
        'retail_behavior': 'Bored, overtrading',
        'phase_type': 'NEUTRAL',
    },
    'PROFIT TAKING': {
        'description': 'Some selling after gains, not full distribution',
        'whale_behavior': 'Trimming positions',
        'retail_behavior': 'Panic on red candles',
        'phase_type': 'NEUTRAL',
    },
    'CAPITULATION': {
        'description': 'Panic selling, potential bottom forming',
        'whale_behavior': 'Starting to buy from panic',
        'retail_behavior': 'Panic selling at lows',
        'phase_type': 'CONTEXT_DEPENDENT',  # Depends on whale positioning
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# WHALE POSITIONING THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
WHALE_THRESHOLDS = {
    'strong_bullish': 65,    # Very confident long
    'bullish': 55,           # Moderately long
    'neutral_high': 54,      # Upper neutral
    'neutral_low': 46,       # Lower neutral
    'bearish': 45,           # Moderately short
    'strong_bearish': 35,    # Very confident short
}

# ─────────────────────────────────────────────────────────────────────────────
# 📊 MASTER DECISION MATRIX - 81 COMBINATIONS
# ─────────────────────────────────────────────────────────────────────────────
# Format: (flow_phase, whale_state, structure) → action
# whale_state: 'BULLISH' (>55%), 'BEARISH' (<45%), 'NEUTRAL' (45-55%)
# structure: 'BULLISH', 'BEARISH', 'NEUTRAL' (consolidating/mixed/ranging)
# ─────────────────────────────────────────────────────────────────────────────

DECISION_MATRIX = {
    # ═══════════════════════════════════════════════════════════════════════════
    # ACCUMULATION PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('ACCUMULATION', 'BULLISH', 'BULLISH'): {
        'action': '🟢 STRONG LONG',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Perfect alignment: Accumulation + Whales long + Bullish structure',
        'entry': 'Buy now or on any dip',
    },
    ('ACCUMULATION', 'BULLISH', 'BEARISH'): {
        'action': '🟢 EARLY LONG',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Bullish divergence! Whales accumulating before reversal',
        'entry': 'Scale in, stop below recent low',
    },
    ('ACCUMULATION', 'BULLISH', 'NEUTRAL'): {
        'action': '🟢 LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Accumulation + Whales long, waiting for breakout',
        'entry': 'Buy near range lows',
    },
    ('ACCUMULATION', 'BEARISH', 'BULLISH'): {
        'action': '⚠️ CONFLICT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Flow says accumulation but whales not buying - suspicious',
        'entry': 'Wait for whale confirmation >55%',
    },
    ('ACCUMULATION', 'BEARISH', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Flow says accumulation but whales still short - early',
        'entry': 'Wait for whale flip',
    },
    ('ACCUMULATION', 'BEARISH', 'NEUTRAL'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Mixed signals - accumulation starting but whales not in yet',
        'entry': 'Watch for whale positioning change',
    },
    ('ACCUMULATION', 'NEUTRAL', 'BULLISH'): {
        'action': '🟢 LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Accumulation + Bullish structure, whales building',
        'entry': 'Buy on pullbacks',
    },
    ('ACCUMULATION', 'NEUTRAL', 'BEARISH'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Early accumulation, wait for confirmation',
        'entry': 'Watch for structure change',
    },
    ('ACCUMULATION', 'NEUTRAL', 'NEUTRAL'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Accumulation in range, direction unclear',
        'entry': 'Buy at range lows with tight stop',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # MARKUP PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('MARKUP', 'BULLISH', 'BULLISH'): {
        'action': '🟢 CONTINUE LONG',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Perfect trend: All signals aligned bullish',
        'entry': 'Add on dips, trail stop',
    },
    ('MARKUP', 'BULLISH', 'BEARISH'): {
        'action': '🟢 EARLY LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Markup starting, structure will follow',
        'entry': 'Scale in on structure break',
    },
    ('MARKUP', 'BULLISH', 'NEUTRAL'): {
        'action': '🟢 LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Markup in range, breakout likely',
        'entry': 'Buy now, add on breakout',
    },
    ('MARKUP', 'BEARISH', 'BULLISH'): {
        'action': '⚠️ CAUTION',
        'direction': 'REDUCE',
        'confidence': 'LOW',
        'reason': 'Markup but whales not participating - distribution risk',
        'entry': 'Take profits, dont add',
    },
    ('MARKUP', 'BEARISH', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'False markup? Whales still short',
        'entry': 'Wait for clarity',
    },
    ('MARKUP', 'BEARISH', 'NEUTRAL'): {
        'action': '⚠️ CONFLICT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Markup signal but whales not buying',
        'entry': 'Wait for whale confirmation',
    },
    ('MARKUP', 'NEUTRAL', 'BULLISH'): {
        'action': '🟢 LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Markup confirmed, whales building',
        'entry': 'Buy pullbacks',
    },
    ('MARKUP', 'NEUTRAL', 'BEARISH'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Early markup, wait for structure',
        'entry': 'Wait for bullish structure',
    },
    ('MARKUP', 'NEUTRAL', 'NEUTRAL'): {
        'action': '🟢 CAUTIOUS LONG',
        'direction': 'LONG',
        'confidence': 'LOW',
        'reason': 'Markup in consolidation',
        'entry': 'Small position, add on breakout',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # RE-ACCUMULATION PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('RE-ACCUMULATION', 'BULLISH', 'BULLISH'): {
        'action': '🟢 ADD LONG',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Healthy pause in uptrend, whales reloading',
        'entry': 'Add to position on dips',
    },
    ('RE-ACCUMULATION', 'BULLISH', 'BEARISH'): {
        'action': '🟢 BUY DIP',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Pullback in larger uptrend, whales buying',
        'entry': 'Scale in on support',
    },
    ('RE-ACCUMULATION', 'BULLISH', 'NEUTRAL'): {
        'action': '🟢 LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Consolidation before next leg, whales positioned',
        'entry': 'Buy near range lows',
    },
    ('RE-ACCUMULATION', 'BEARISH', 'BULLISH'): {
        'action': '⚠️ DISTRIBUTION?',
        'direction': 'REDUCE',
        'confidence': 'LOW',
        'reason': 'Looks like re-accumulation but whales selling',
        'entry': 'Reduce position, watch closely',
    },
    ('RE-ACCUMULATION', 'BEARISH', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Not re-accumulation - whales still bearish',
        'entry': 'Wait for whale flip',
    },
    ('RE-ACCUMULATION', 'BEARISH', 'NEUTRAL'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Unclear - could be distribution',
        'entry': 'Wait for confirmation',
    },
    ('RE-ACCUMULATION', 'NEUTRAL', 'BULLISH'): {
        'action': '🟢 HOLD LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Consolidation in uptrend',
        'entry': 'Hold, add on confirmation',
    },
    ('RE-ACCUMULATION', 'NEUTRAL', 'BEARISH'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Potential re-accumulation forming',
        'entry': 'Watch for whale buying',
    },
    ('RE-ACCUMULATION', 'NEUTRAL', 'NEUTRAL'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Ranging, waiting for direction',
        'entry': 'Range trade or wait',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # DISTRIBUTION PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('DISTRIBUTION', 'BULLISH', 'BULLISH'): {
        'action': '⚠️ TAKE PROFIT',
        'direction': 'REDUCE',
        'confidence': 'MEDIUM',
        'reason': 'Distribution starting despite bullish look - reduce',
        'entry': 'Take profits on rallies',
    },
    ('DISTRIBUTION', 'BULLISH', 'BEARISH'): {
        'action': '⚠️ CONFLICT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Distribution flow but whales still long - watch',
        'entry': 'Wait for clarity',
    },
    ('DISTRIBUTION', 'BULLISH', 'NEUTRAL'): {
        'action': '⚠️ CAUTION',
        'direction': 'REDUCE',
        'confidence': 'LOW',
        'reason': 'Distribution in range, whales may be exiting',
        'entry': 'Reduce position',
    },
    ('DISTRIBUTION', 'BEARISH', 'BULLISH'): {
        'action': '⛔ AVOID LONG',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'Classic distribution! Whales selling into strength',
        'entry': 'Short on rallies, stop above high',
    },
    ('DISTRIBUTION', 'BEARISH', 'BEARISH'): {
        'action': '🔴 CONTINUE SHORT',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'Distribution confirmed, downtrend intact',
        'entry': 'Add shorts on rallies',
    },
    ('DISTRIBUTION', 'BEARISH', 'NEUTRAL'): {
        'action': '🔴 SHORT',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'Distribution in range, breakdown expected',
        'entry': 'Short near range highs',
    },
    ('DISTRIBUTION', 'NEUTRAL', 'BULLISH'): {
        'action': '⚠️ TAKE PROFIT',
        'direction': 'REDUCE',
        'confidence': 'MEDIUM',
        'reason': 'Distribution starting, reduce longs',
        'entry': 'Take profits, dont add',
    },
    ('DISTRIBUTION', 'NEUTRAL', 'BEARISH'): {
        'action': '🔴 SHORT',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'Distribution + Bearish structure',
        'entry': 'Short on rallies',
    },
    ('DISTRIBUTION', 'NEUTRAL', 'NEUTRAL'): {
        'action': '⛔ AVOID',
        'direction': 'WAIT',
        'confidence': 'MEDIUM',
        'reason': 'Distribution in range - stay out',
        'entry': 'No trade, wait for clarity',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # FOMO / DIST RISK PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('FOMO / DIST RISK', 'BULLISH', 'BULLISH'): {
        'action': '⚠️ EXIT LONGS',
        'direction': 'REDUCE',
        'confidence': 'HIGH',
        'reason': 'FOMO top forming! Take profits NOW',
        'entry': 'Exit longs, dont chase',
    },
    ('FOMO / DIST RISK', 'BULLISH', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Conflicting signals',
        'entry': 'Stay out',
    },
    ('FOMO / DIST RISK', 'BULLISH', 'NEUTRAL'): {
        'action': '⚠️ CAUTION',
        'direction': 'REDUCE',
        'confidence': 'MEDIUM',
        'reason': 'FOMO risk even with whales long',
        'entry': 'Reduce exposure',
    },
    ('FOMO / DIST RISK', 'BEARISH', 'BULLISH'): {
        'action': '⛔ DONT BUY',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'Classic top! Retail FOMO, whales selling',
        'entry': 'Short on rallies',
    },
    ('FOMO / DIST RISK', 'BEARISH', 'BEARISH'): {
        'action': '🔴 SHORT',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'FOMO failed, downtrend continues',
        'entry': 'Add shorts',
    },
    ('FOMO / DIST RISK', 'BEARISH', 'NEUTRAL'): {
        'action': '🔴 EARLY SHORT',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'FOMO at range highs, breakdown likely',
        'entry': 'Short near highs',
    },
    ('FOMO / DIST RISK', 'NEUTRAL', 'BULLISH'): {
        'action': '⚠️ TAKE PROFIT',
        'direction': 'REDUCE',
        'confidence': 'MEDIUM',
        'reason': 'FOMO risk - reduce longs',
        'entry': 'Scale out',
    },
    ('FOMO / DIST RISK', 'NEUTRAL', 'BEARISH'): {
        'action': '🔴 SHORT',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'FOMO bounce in downtrend',
        'entry': 'Short the bounce',
    },
    ('FOMO / DIST RISK', 'NEUTRAL', 'NEUTRAL'): {
        'action': '⛔ AVOID',
        'direction': 'WAIT',
        'confidence': 'HIGH',
        'reason': 'FOMO in range = trap. Stay out!',
        'entry': 'No trade',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # EXHAUSTION PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('EXHAUSTION', 'BULLISH', 'BULLISH'): {
        'action': '⚠️ EXIT NOW',
        'direction': 'EXIT',
        'confidence': 'HIGH',
        'reason': 'Exhaustion at top! Exit before reversal',
        'entry': 'Close longs immediately',
    },
    ('EXHAUSTION', 'BULLISH', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Exhaustion in downtrend with whales long - confused',
        'entry': 'Wait for clarity',
    },
    ('EXHAUSTION', 'BULLISH', 'NEUTRAL'): {
        'action': '⚠️ CAUTION',
        'direction': 'REDUCE',
        'confidence': 'MEDIUM',
        'reason': 'Exhaustion forming',
        'entry': 'Reduce position',
    },
    ('EXHAUSTION', 'BEARISH', 'BULLISH'): {
        'action': '🔴 SHORT NOW',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'Blow-off top! Whales already out',
        'entry': 'Short aggressively',
    },
    ('EXHAUSTION', 'BEARISH', 'BEARISH'): {
        'action': '🔴 CONTINUE SHORT',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'Exhaustion confirmed, add to shorts',
        'entry': 'Add shorts on bounces',
    },
    ('EXHAUSTION', 'BEARISH', 'NEUTRAL'): {
        'action': '🔴 SHORT',
        'direction': 'SHORT',
        'confidence': 'HIGH',
        'reason': 'Exhaustion at range high',
        'entry': 'Short at resistance',
    },
    ('EXHAUSTION', 'NEUTRAL', 'BULLISH'): {
        'action': '⚠️ EXIT',
        'direction': 'EXIT',
        'confidence': 'HIGH',
        'reason': 'Exhaustion - exit longs',
        'entry': 'Close positions',
    },
    ('EXHAUSTION', 'NEUTRAL', 'BEARISH'): {
        'action': '🔴 SHORT',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'Exhaustion bounce in downtrend',
        'entry': 'Short rallies',
    },
    ('EXHAUSTION', 'NEUTRAL', 'NEUTRAL'): {
        'action': '⛔ AVOID',
        'direction': 'WAIT',
        'confidence': 'HIGH',
        'reason': 'Exhaustion - stay out until dust settles',
        'entry': 'No trade',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CONSOLIDATION PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('CONSOLIDATION', 'BULLISH', 'BULLISH'): {
        'action': '🟢 HOLD LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Consolidation in uptrend, whales holding',
        'entry': 'Hold, add on breakout',
    },
    ('CONSOLIDATION', 'BULLISH', 'BEARISH'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Consolidation after drop, whales accumulating?',
        'entry': 'Watch for breakout direction',
    },
    ('CONSOLIDATION', 'BULLISH', 'NEUTRAL'): {
        'action': '🟢 BUY RANGE LOW',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Range trade with bullish bias',
        'entry': 'Buy at range lows',
    },
    ('CONSOLIDATION', 'BEARISH', 'BULLISH'): {
        'action': '⚠️ CAUTION',
        'direction': 'REDUCE',
        'confidence': 'LOW',
        'reason': 'Consolidation but whales bearish',
        'entry': 'Reduce longs',
    },
    ('CONSOLIDATION', 'BEARISH', 'BEARISH'): {
        'action': '🔴 SELL RANGE HIGH',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'Consolidation in downtrend',
        'entry': 'Short at range highs',
    },
    ('CONSOLIDATION', 'BEARISH', 'NEUTRAL'): {
        'action': '🔴 LEAN SHORT',
        'direction': 'SHORT',
        'confidence': 'LOW',
        'reason': 'Range with bearish bias',
        'entry': 'Short at range highs',
    },
    ('CONSOLIDATION', 'NEUTRAL', 'BULLISH'): {
        'action': '🟢 HOLD',
        'direction': 'LONG',
        'confidence': 'LOW',
        'reason': 'Pause in uptrend',
        'entry': 'Hold, wait for breakout',
    },
    ('CONSOLIDATION', 'NEUTRAL', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Consolidation in downtrend',
        'entry': 'Wait for direction',
    },
    ('CONSOLIDATION', 'NEUTRAL', 'NEUTRAL'): {
        'action': '↔️ RANGE TRADE',
        'direction': 'RANGE',
        'confidence': 'LOW',
        'reason': 'Pure range - buy low, sell high',
        'entry': 'Range trade with tight stops',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # PROFIT TAKING PHASE (9 combinations)
    # ═══════════════════════════════════════════════════════════════════════════
    ('PROFIT TAKING', 'BULLISH', 'BULLISH'): {
        'action': '🟢 BUY DIP',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Healthy pullback! Whales holding, dip = opportunity',
        'entry': 'Buy the dip aggressively',
    },
    ('PROFIT TAKING', 'BULLISH', 'BEARISH'): {
        'action': '🟢 EARLY LONG',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Profit taking after larger move, whales still long',
        'entry': 'Scale in on support',
    },
    ('PROFIT TAKING', 'BULLISH', 'NEUTRAL'): {
        'action': '🟢 BUY DIP',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Profit taking in range, whales bullish',
        'entry': 'Buy near range lows',
    },
    ('PROFIT TAKING', 'BEARISH', 'BULLISH'): {
        'action': '⚠️ DISTRIBUTION?',
        'direction': 'REDUCE',
        'confidence': 'MEDIUM',
        'reason': 'Profit taking with whales bearish = distribution risk',
        'entry': 'Reduce longs, watch closely',
    },
    ('PROFIT TAKING', 'BEARISH', 'BEARISH'): {
        'action': '🔴 SHORT RALLY',
        'direction': 'SHORT',
        'confidence': 'MEDIUM',
        'reason': 'Profit taking bounce in downtrend',
        'entry': 'Short the bounce',
    },
    ('PROFIT TAKING', 'BEARISH', 'NEUTRAL'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Mixed signals',
        'entry': 'Wait for clarity',
    },
    ('PROFIT TAKING', 'NEUTRAL', 'BULLISH'): {
        'action': '🟢 BUY DIP',
        'direction': 'LONG',
        'confidence': 'MEDIUM',
        'reason': 'Normal profit taking in uptrend',
        'entry': 'Buy on support',
    },
    ('PROFIT TAKING', 'NEUTRAL', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Profit taking in downtrend',
        'entry': 'Wait for direction',
    },
    ('PROFIT TAKING', 'NEUTRAL', 'NEUTRAL'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Profit taking in range',
        'entry': 'Wait for edge of range',
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # CAPITULATION PHASE (9 combinations) - CONTEXT DEPENDENT!
    # ═══════════════════════════════════════════════════════════════════════════
    ('CAPITULATION', 'BULLISH', 'BULLISH'): {
        'action': '🟢 STRONG BUY',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Whales buying panic! Classic accumulation',
        'entry': 'Buy aggressively',
    },
    ('CAPITULATION', 'BULLISH', 'BEARISH'): {
        'action': '🟢 EARLY LONG',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Whales accumulating capitulation! Reversal coming',
        'entry': 'Scale in, stop below capitulation low',
    },
    ('CAPITULATION', 'BULLISH', 'NEUTRAL'): {
        'action': '🟢 BUY',
        'direction': 'LONG',
        'confidence': 'HIGH',
        'reason': 'Whales buying the panic in range',
        'entry': 'Buy at range lows',
    },
    ('CAPITULATION', 'BEARISH', 'BULLISH'): {
        'action': '⚠️ MORE PAIN',
        'direction': 'WAIT',
        'confidence': 'MEDIUM',
        'reason': 'Capitulation but whales still short - more downside',
        'entry': 'Wait for whale flip',
    },
    ('CAPITULATION', 'BEARISH', 'BEARISH'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'MEDIUM',
        'reason': 'Capitulation not done - whales still selling',
        'entry': 'Wait for whale buying',
    },
    ('CAPITULATION', 'BEARISH', 'NEUTRAL'): {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'MEDIUM',
        'reason': 'Capitulation ongoing, not finished',
        'entry': 'Wait for whale accumulation',
    },
    ('CAPITULATION', 'NEUTRAL', 'BULLISH'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Capitulation starting, watch whales',
        'entry': 'Wait for whale positioning',
    },
    ('CAPITULATION', 'NEUTRAL', 'BEARISH'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Capitulation forming, bottom may be near',
        'entry': 'Watch for whale buying >55%',
    },
    ('CAPITULATION', 'NEUTRAL', 'NEUTRAL'): {
        'action': '👀 WATCH',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': 'Capitulation in range, direction unclear',
        'entry': 'Wait for whale signal',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 📈 CRYPTO-SPECIFIC RULES (OI + Funding)
# ─────────────────────────────────────────────────────────────────────────────
CRYPTO_OI_RULES = {
    # OI Change + Price Change combinations
    ('OI_UP', 'PRICE_UP'): {
        'interpretation': 'New LONGS entering',
        'bias': 'BULLISH',
        'confidence_boost': 10,
    },
    ('OI_UP', 'PRICE_DOWN'): {
        'interpretation': 'New SHORTS entering',
        'bias': 'BEARISH',
        'confidence_boost': 10,
    },
    ('OI_DOWN', 'PRICE_UP'): {
        'interpretation': 'SHORT SQUEEZE - shorts closing',
        'bias': 'BULLISH',
        'confidence_boost': 15,
    },
    ('OI_DOWN', 'PRICE_DOWN'): {
        'interpretation': 'LONG LIQUIDATIONS - longs closing',
        'bias': 'BULLISH_SOON',  # Fuel for reversal
        'confidence_boost': 5,
    },
    ('OI_FLAT', 'PRICE_UP'): {
        'interpretation': 'Spot buying, no leverage',
        'bias': 'BULLISH',
        'confidence_boost': 5,
    },
    ('OI_FLAT', 'PRICE_DOWN'): {
        'interpretation': 'Spot selling, no leverage',
        'bias': 'BEARISH',
        'confidence_boost': 5,
    },
}

CRYPTO_FUNDING_RULES = {
    'extreme_negative': {
        'threshold': -0.1,  # <-0.1%
        'interpretation': 'Shorts paying longs - squeeze potential!',
        'bias': 'BULLISH',
        'action': 'LONG - Shorts overleveraged',
    },
    'negative': {
        'threshold': -0.03,  # -0.1% to -0.03%
        'interpretation': 'Slight short bias',
        'bias': 'SLIGHT_BULLISH',
        'action': 'Lean LONG',
    },
    'neutral': {
        'threshold': 0.03,  # -0.03% to 0.03%
        'interpretation': 'Balanced market',
        'bias': 'NEUTRAL',
        'action': 'No funding edge',
    },
    'positive': {
        'threshold': 0.1,  # 0.03% to 0.1%
        'interpretation': 'Slight long bias',
        'bias': 'SLIGHT_BEARISH',
        'action': 'Lean SHORT',
    },
    'extreme_positive': {
        'threshold': float('inf'),  # >0.1%
        'interpretation': 'Longs paying shorts - dump potential!',
        'bias': 'BEARISH',
        'action': 'SHORT - Longs overleveraged',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 📊 STOCK/ETF RULES (No Whale Data - Use TA + Flow)
# ─────────────────────────────────────────────────────────────────────────────
STOCK_ETF_RULES = {
    # When no whale data available, combine Flow + RSI
    ('ACCUMULATION', 'RSI_OVERSOLD'): {
        'action': '🟢 STRONG BUY',
        'reason': 'Accumulation + Oversold = Perfect setup',
    },
    ('ACCUMULATION', 'RSI_NEUTRAL'): {
        'action': '🟢 BUY',
        'reason': 'Accumulation phase, good entry',
    },
    ('ACCUMULATION', 'RSI_OVERBOUGHT'): {
        'action': '⚠️ WAIT FOR PULLBACK',
        'reason': 'Accumulation but overbought short-term',
    },
    ('MARKUP', 'RSI_OVERSOLD'): {
        'action': '🟢 BUY DIP',
        'reason': 'Markup trend, oversold = dip opportunity',
    },
    ('MARKUP', 'RSI_NEUTRAL'): {
        'action': '🟢 LONG',
        'reason': 'Uptrend intact',
    },
    ('MARKUP', 'RSI_OVERBOUGHT'): {
        'action': '⚠️ HOLD/REDUCE',
        'reason': 'Trend up but extended',
    },
    ('DISTRIBUTION', 'RSI_OVERSOLD'): {
        'action': '⏳ DEAD CAT?',
        'reason': 'Distribution + Oversold = bounce then lower',
    },
    ('DISTRIBUTION', 'RSI_NEUTRAL'): {
        'action': '⛔ AVOID LONG',
        'reason': 'Distribution phase',
    },
    ('DISTRIBUTION', 'RSI_OVERBOUGHT'): {
        'action': '🔴 SHORT',
        'reason': 'Distribution + Overbought = top',
    },
    ('FOMO / DIST RISK', 'RSI_OVERSOLD'): {
        'action': '⏳ WAIT',
        'reason': 'FOMO failed, wait for clarity',
    },
    ('FOMO / DIST RISK', 'RSI_NEUTRAL'): {
        'action': '⛔ AVOID',
        'reason': 'FOMO risk, stay out',
    },
    ('FOMO / DIST RISK', 'RSI_OVERBOUGHT'): {
        'action': '🔴 SHORT',
        'reason': 'FOMO at top = classic short',
    },
    ('CAPITULATION', 'RSI_OVERSOLD'): {
        'action': '👀 WATCH FOR REVERSAL',
        'reason': 'Capitulation + Oversold = potential bottom',
    },
    ('CAPITULATION', 'RSI_NEUTRAL'): {
        'action': '⏳ WAIT',
        'reason': 'Capitulation ongoing',
    },
    ('CAPITULATION', 'RSI_OVERBOUGHT'): {
        'action': '⏳ WAIT',
        'reason': 'Bounce in capitulation, unreliable',
    },
}

# RSI thresholds for stocks/ETFs
RSI_THRESHOLDS = {
    'oversold': 30,
    'overbought': 70,
}

# ─────────────────────────────────────────────────────────────────────────────
# 🔥 SQUEEZE SIGNAL RULES
# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Squeeze signals are OPPOSITE of what the name suggests!
# SHORT SQUEEZE = Shorts get liquidated = Price goes UP = LONG trade
# LONG SQUEEZE = Longs get liquidated = Price goes DOWN = SHORT trade
# ─────────────────────────────────────────────────────────────────────────────
SQUEEZE_SIGNAL_RULES = {
    'SHORT SQUEEZE': {
        'actual_direction': 'LONG',  # NOT SHORT!
        'explanation': 'Shorts get liquidated → forced to buy → price explodes UP',
        'whale_action': '🟢 LONG',
        'action': 'LONG immediately. Target 5-10%. Explosive UP move incoming.',
        'position_interpretation': 'For LONG: want to be NEAR LOWS (early)',
        'confidence': 'HIGH',
    },
    'LONG SQUEEZE': {
        'actual_direction': 'SHORT',  # NOT LONG!
        'explanation': 'Longs get liquidated → forced to sell → price crashes DOWN',
        'whale_action': '🔴 SHORT',
        'action': 'SHORT immediately. Target 5-10%. Violent DOWN move incoming.',
        'position_interpretation': 'For SHORT: want to be NEAR HIGHS (early)',
        'confidence': 'HIGH',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Simplified phase categories for quick lookup
# ─────────────────────────────────────────────────────────────────────────────
MONEY_FLOW_PHASE_RULES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # LEADING INDICATOR KNOWLEDGE BASE - SINGLE SOURCE OF TRUTH
    # ═══════════════════════════════════════════════════════════════════════════
    # 
    # CORE PRINCIPLE: Whale positioning is the PRIMARY signal.
    # OI and Structure are CONFIRMING indicators, not gating.
    # We want to be BEFORE the move, not after.
    #
    # ═══════════════════════════════════════════════════════════════════════════
    
    # --- INDICATOR HIERARCHY (Most Leading → Most Lagging) ---
    'indicator_priority': [
        'whale_positioning',      # Tier 1: LEADING - Act on this
        'whale_retail_divergence', # Tier 1: LEADING - Strong edge
        'money_flow_phase',       # Tier 2: MIXED - Early warning
        'oi_confirmation',        # Tier 2: CONFIRMING - Adds confidence
        'funding_rate',           # Tier 2: CONFIRMING - Squeeze potential
        'structure',              # Tier 3: LAGGING - For stop placement
        'ta_score',               # Tier 3: LAGGING - Entry timing only
    ],
    
    # --- WHALE POSITIONING THRESHOLDS (PRIMARY SIGNAL) ---
    'whale_thresholds': {
        'extreme_bullish': 75,    # Very rare, very strong
        'strong_bullish': 70,     # High conviction LONG
        'bullish': 65,            # Clear bullish bias
        'lean_bullish': 60,       # Slight bullish edge
        'neutral_high': 55,       # Upper neutral boundary
        'neutral_low': 45,        # Lower neutral boundary
        'lean_bearish': 40,       # Slight bearish edge
        'bearish': 35,            # Clear bearish bias
        'strong_bearish': 30,     # High conviction SHORT
        'extreme_bearish': 25,    # Very rare, very strong
    },
    
    # --- DIVERGENCE THRESHOLDS ---
    'divergence_thresholds': {
        'squeeze_setup': 15,      # Strong divergence = squeeze potential
        'clear_edge': 10,         # Meaningful edge
        'slight_edge': 5,         # Small but notable
        'no_edge': 0,             # No divergence benefit
    },
    
    # --- POSITION IN RANGE (Entry Quality) ---
    'position_thresholds': {
        'early_long': 35,         # Below this = EARLY for longs
        'late_long': 65,          # Above this = LATE for longs
        'early_short': 65,        # Above this = EARLY for shorts
        'late_short': 35,         # Below this = LATE for shorts
    },
    
    # --- FLOW PHASE CATEGORIES ---
    # Distribution phases - Whales EXITING - Don't buy unless whales diverge
    'distribution_phases': ['DISTRIBUTION', 'FOMO / DIST RISK', 'EXHAUSTION'],
    'distribution_action': 'AVOID LONG unless whales >65% (they know something)',
    'distribution_reason': 'Smart money typically exiting. But if whales still long, they see something retail doesn\'t.',
    
    # Accumulation phases - Whales BUYING - Support LONG signals
    'accumulation_phases': ['ACCUMULATION', 'MARKUP', 'RE-ACCUMULATION'],
    'accumulation_action': 'LONG when whales confirm (>55%)',
    'accumulation_reason': 'Smart money buying. Align with whales.',
    
    # Neutral phases - No clear flow signal
    'neutral_phases': ['CONSOLIDATION', 'PROFIT TAKING'],
    'neutral_action': 'Follow whale positioning as tiebreaker',
    'neutral_reason': 'Flow unclear. Whale positioning is the deciding factor.',
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CAPITULATION - SPECIAL LEADING INDICATOR
    # ═══════════════════════════════════════════════════════════════════════════
    # Capitulation is panic selling. When WHALES are buying during panic,
    # this is THE most predictive bullish signal. Act BEFORE confirmation.
    'capitulation_phase': 'CAPITULATION',
    'capitulation_rules': {
        'strong_long_threshold': 65,   # Whales >= 65% = STRONG BUY signal
        'long_threshold': 55,          # Whales >= 55% = BUY signal
        'wait_threshold': 45,          # Whales 45-55% = WATCH for accumulation
        'more_pain_threshold': 45,     # Whales < 45% = More downside likely
    },
    'capitulation_bullish_action': 'LONG NOW - Whales accumulating panic. This is how bottoms form.',
    'capitulation_neutral_action': 'WATCH - Potential bottom. Wait for whale positioning >55%.',
    'capitulation_bearish_action': 'WAIT - Whales not buying yet. More pain possible.',
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HIGH-PRIORITY PREDICTIVE PATTERNS (Check FIRST)
    # ═══════════════════════════════════════════════════════════════════════════
    'predictive_patterns': {
        'capitulation_reversal': {
            'conditions': {'phase': 'CAPITULATION', 'whale_min': 65, 'position_max': 35},
            'signal': 'EARLY LONG',
            'confidence': 'HIGH',
            'reason': 'Classic bottom pattern - whales buying panic at lows',
        },
        'accumulation_breakout': {
            'conditions': {'phase': 'ACCUMULATION', 'whale_min': 65, 'position_max': 35},
            'signal': 'EARLY LONG', 
            'confidence': 'HIGH',
            'reason': 'Smart money accumulating at range lows',
        },
        'distribution_breakdown': {
            'conditions': {'phase': 'DISTRIBUTION', 'whale_max': 35, 'position_min': 65},
            'signal': 'EARLY SHORT',
            'confidence': 'HIGH',
            'reason': 'Smart money distributing at range highs',
        },
        'short_squeeze': {
            'conditions': {'whale_min': 60, 'retail_max': 45, 'divergence_min': 15},
            'signal': 'SHORT SQUEEZE LONG',
            'confidence': 'HIGH',
            'reason': 'Massive divergence - shorts about to be liquidated',
        },
        'long_squeeze': {
            'conditions': {'whale_max': 40, 'retail_min': 55, 'divergence_max': -15},
            'signal': 'LONG SQUEEZE SHORT',
            'confidence': 'HIGH',
            'reason': 'Massive divergence - longs about to be liquidated',
        },
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WHAT NOT TO DO (Anti-patterns)
    # ═══════════════════════════════════════════════════════════════════════════
    'anti_patterns': {
        'wait_for_oi': 'DON\'T wait for OI confirmation - whales position BEFORE OI moves',
        'wait_for_structure': 'DON\'T wait for structure to break - that\'s AFTER the move',
        'wait_for_all_green': 'DON\'T wait for all indicators - be early, not late',
        'ignore_whale_divergence': 'DON\'T ignore whales disagreeing with structure - follow whales',
    },
}

PREDICTIVE_SIGNALS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟢 BULLISH DIVERGENCE - Structure down, Leading up
    # ═══════════════════════════════════════════════════════════════════════════
    
    'early_long_accumulation': {
        'signal': '🚀 EARLY LONG',
        'subtitle': 'Accumulation before reversal',
        'color': '#00ff88',
        'conditions': {
            'structure': ['Bearish', 'Mixed', 'Consolidating', 'Ranging'],
            'money_flow': ['ACCUMULATION', 'MARKUP', 'CAPITULATION', 'RE-ACCUMULATION'],
            'whale_pct_min': 55,
        },
        'interpretation': 'Whales accumulating while price still falling. This is HOW reversals start.',
        'action': 'Scale in LONG. Stop below recent low. Best entry before the crowd.',
        'confidence': 'HIGH',
    },
    
    'early_long_markup': {
        'signal': '📈 BUILDING LONG',
        'subtitle': 'Trend starting - get positioned',
        'color': '#00d4aa',
        'conditions': {
            'structure': ['Mixed', 'Consolidating'],
            'money_flow': ['MARKUP'],
            'whale_pct_min': 52,
            'oi_positive': True,
        },
        'interpretation': 'Money flowing in with whales positioned. Structure will follow.',
        'action': 'Enter LONG on dips. Trend about to confirm.',
        'confidence': 'MEDIUM',
    },
    
    'continue_long': {
        'signal': '✅ CONTINUE LONG',
        'subtitle': 'Trend confirmed - ride it',
        'color': '#00ff88',
        'conditions': {
            'structure': ['Bullish'],
            'money_flow': ['MARKUP', 'ACCUMULATION'],
            'whale_pct_min': 55,
        },
        'interpretation': 'All signals aligned bullish. Trend has legs.',
        'action': 'Hold LONG. Add on pullbacks. Trail stop.',
        'confidence': 'HIGH',
    },
    
    'buy_the_dip': {
        'signal': '🛒 BUY THE DIP',
        'subtitle': 'Healthy pullback in uptrend',
        'color': '#00d4aa',
        'conditions': {
            'structure': ['Bullish'],
            'money_flow': ['PROFIT TAKING', 'CONSOLIDATION'],
            'whale_pct_min': 55,
        },
        'interpretation': 'Profit taking in uptrend but whales still bullish. Dip = opportunity.',
        'action': 'Add to LONG position on support. Whales not selling.',
        'confidence': 'MEDIUM',
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔴 BEARISH DIVERGENCE - Structure up, Leading down
    # ═══════════════════════════════════════════════════════════════════════════
    
    'early_short_distribution': {
        'signal': '⚠️ TAKE PROFIT',
        'subtitle': 'Distribution at highs',
        'color': '#ff9500',
        'conditions': {
            'structure': ['Bullish', 'Mixed'],
            'money_flow': ['DISTRIBUTION', 'FOMO / DIST RISK', 'EXHAUSTION'],
            'whale_pct_max': 45,
        },
        'interpretation': 'Whales distributing while retail FOMO buying. Top forming.',
        'action': 'Close LONGS. Do NOT buy here. Consider SHORT on rallies.',
        'confidence': 'HIGH',
    },
    
    'early_short_setup': {
        'signal': '🔴 EARLY SHORT',
        'subtitle': 'Distribution before drop',
        'color': '#ff6b6b',
        'conditions': {
            'structure': ['Bullish', 'Mixed'],
            'money_flow': ['DISTRIBUTION'],
            'whale_pct_max': 45,
            'oi_positive': True,  # New shorts entering
        },
        'interpretation': 'Whales shorting while price still high. Drop incoming.',
        'action': 'Scale in SHORT. Stop above recent high.',
        'confidence': 'HIGH',
    },
    
    'continue_short': {
        'signal': '🔻 CONTINUE SHORT',
        'subtitle': 'Downtrend confirmed',
        'color': '#ff6b6b',
        'conditions': {
            'structure': ['Bearish'],
            'money_flow': ['DISTRIBUTION', 'PROFIT TAKING'],
            'whale_pct_max': 45,
        },
        'interpretation': 'All signals aligned bearish. Downtrend has legs.',
        'action': 'Hold SHORT. Add on rallies. Trail stop.',
        'confidence': 'HIGH',
    },
    
    'sell_the_rally': {
        'signal': '📉 SELL THE RALLY',
        'subtitle': 'Weak bounce in downtrend',
        'color': '#ff9500',
        'conditions': {
            'structure': ['Bearish'],
            'money_flow': ['MARKUP', 'RE-ACCUMULATION'],  # False hope
            'whale_pct_max': 48,
        },
        'interpretation': 'Short covering rally but whales still bearish. Rally = trap.',
        'action': 'SHORT on resistance. This is exit liquidity.',
        'confidence': 'MEDIUM',
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⚪ NEUTRAL / WAIT SIGNALS
    # ═══════════════════════════════════════════════════════════════════════════
    
    'range_bound': {
        'signal': '↔️ RANGE TRADE',
        'subtitle': 'Buy low, sell high in range',
        'color': '#ffcc00',
        'conditions': {
            'structure': ['Consolidating', 'Mixed'],
            'money_flow': ['CONSOLIDATION'],
            'whale_pct_neutral': True,  # 48-52%
        },
        'interpretation': 'No clear direction. Market ranging.',
        'action': 'Trade the range. Buy near lows, sell near highs.',
        'confidence': 'LOW',
    },
    
    'wait_for_clarity': {
        'signal': '⏳ WAIT',
        'subtitle': 'Conflicting signals',
        'color': '#888888',
        'conditions': {
            'default': True,  # Fallback when nothing matches
        },
        'interpretation': 'Mixed signals. No clear edge.',
        'action': 'Wait for alignment. Patience is a position.',
        'confidence': 'NONE',
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔥 SQUEEZE SETUPS - Explosive moves
    # ═══════════════════════════════════════════════════════════════════════════
    
    'short_squeeze': {
        'signal': '🔥 SHORT SQUEEZE',
        'subtitle': 'Shorts about to get liquidated',
        'color': '#ff00ff',
        'conditions': {
            'whale_pct_min': 60,
            'retail_pct_max': 45,  # Retail short
            'divergence_min': 15,
        },
        'interpretation': 'Whales long, retail short. Any up move = cascading liquidations.',
        'action': 'LONG immediately. Target 5-10%. Explosive move incoming.',
        'confidence': 'HIGH',
    },
    
    'long_squeeze': {
        'signal': '💥 LONG SQUEEZE',
        'subtitle': 'Longs about to get liquidated',
        'color': '#ff00ff',
        'conditions': {
            'whale_pct_max': 40,
            'retail_pct_min': 55,  # Retail long
            'divergence_max': -15,
        },
        'interpretation': 'Whales short, retail long. Any down move = cascading liquidations.',
        'action': 'SHORT immediately. Target 5-10%. Violent drop incoming.',
        'confidence': 'HIGH',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 🔍 KNOWLEDGE BASE LOOKUP FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_whale_state(whale_pct: float) -> str:
    """Convert whale percentage to state: BULLISH, BEARISH, or NEUTRAL"""
    if whale_pct >= WHALE_THRESHOLDS['bullish']:
        return 'BULLISH'
    elif whale_pct <= WHALE_THRESHOLDS['bearish']:
        return 'BEARISH'
    else:
        return 'NEUTRAL'

def get_structure_state(structure_type: str) -> str:
    """Convert structure type to state: BULLISH, BEARISH, or NEUTRAL"""
    if structure_type in ['Bullish']:
        return 'BULLISH'
    elif structure_type in ['Bearish']:
        return 'BEARISH'
    else:
        return 'NEUTRAL'  # Consolidating, Mixed, Ranging, Unknown

def lookup_decision_matrix(
    money_flow_phase: str,
    whale_pct: float,
    structure_type: str,
    rsi: float = None,
    oi_change: float = None,
    price_change: float = None,
    funding_rate: float = None,
    asset_type: str = 'CRYPTO'  # 'CRYPTO', 'STOCK', 'ETF'
) -> dict:
    """
    Look up the appropriate action from the comprehensive knowledge base.
    
    Returns:
        dict with keys: action, direction, confidence, reason, entry
    """
    whale_state = get_whale_state(whale_pct)
    structure_state = get_structure_state(structure_type)
    
    # Primary lookup: Decision Matrix
    key = (money_flow_phase, whale_state, structure_state)
    
    if key in DECISION_MATRIX:
        result = DECISION_MATRIX[key].copy()
        
        # Add crypto-specific modifiers
        if asset_type == 'CRYPTO':
            result = apply_crypto_modifiers(result, oi_change, price_change, funding_rate, whale_pct)
        
        return result
    
    # Fallback for unknown phases
    return {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': f'Unknown combination: {money_flow_phase} + {whale_state} whale + {structure_state} structure',
        'entry': 'Wait for clearer signal',
    }

def apply_crypto_modifiers(
    base_result: dict,
    oi_change: float,
    price_change: float,
    funding_rate: float,
    whale_pct: float
) -> dict:
    """Apply crypto-specific OI and funding rate modifiers to base decision"""
    result = base_result.copy()
    
    # OI + Price combination
    if oi_change is not None and price_change is not None:
        if oi_change > 1:
            oi_state = 'OI_UP'
        elif oi_change < -1:
            oi_state = 'OI_DOWN'
        else:
            oi_state = 'OI_FLAT'
            
        if price_change > 0.5:
            price_state = 'PRICE_UP'
        elif price_change < -0.5:
            price_state = 'PRICE_DOWN'
        else:
            price_state = 'PRICE_FLAT'
        
        oi_key = (oi_state, price_state)
        if oi_key in CRYPTO_OI_RULES:
            oi_rule = CRYPTO_OI_RULES[oi_key]
            result['oi_interpretation'] = oi_rule['interpretation']
            
            # Boost or reduce confidence based on OI alignment
            if oi_rule['bias'] == 'BULLISH' and result['direction'] == 'LONG':
                result['confidence'] = 'HIGH' if result['confidence'] in ['MEDIUM', 'HIGH'] else 'MEDIUM'
                result['reason'] += f" | OI confirms: {oi_rule['interpretation']}"
            elif oi_rule['bias'] == 'BEARISH' and result['direction'] == 'SHORT':
                result['confidence'] = 'HIGH' if result['confidence'] in ['MEDIUM', 'HIGH'] else 'MEDIUM'
                result['reason'] += f" | OI confirms: {oi_rule['interpretation']}"
            elif oi_rule['bias'] == 'BULLISH_SOON' and result['direction'] in ['WAIT', 'LONG']:
                result['reason'] += f" | Watch: {oi_rule['interpretation']}"
    
    # Funding rate modifier
    if funding_rate is not None:
        if funding_rate < CRYPTO_FUNDING_RULES['extreme_negative']['threshold']:
            funding_rule = CRYPTO_FUNDING_RULES['extreme_negative']
        elif funding_rate < CRYPTO_FUNDING_RULES['negative']['threshold']:
            funding_rule = CRYPTO_FUNDING_RULES['negative']
        elif funding_rate < CRYPTO_FUNDING_RULES['positive']['threshold']:
            funding_rule = CRYPTO_FUNDING_RULES['neutral']
        elif funding_rate < CRYPTO_FUNDING_RULES['extreme_positive']['threshold']:
            funding_rule = CRYPTO_FUNDING_RULES['positive']
        else:
            funding_rule = CRYPTO_FUNDING_RULES['extreme_positive']
        
        result['funding_interpretation'] = funding_rule['interpretation']
        
        # Strong funding signals can override or boost confidence
        if funding_rule['bias'] == 'BULLISH' and result['direction'] == 'LONG':
            result['confidence'] = 'HIGH'
            result['reason'] += f" | Funding: {funding_rule['interpretation']}"
        elif funding_rule['bias'] == 'BEARISH' and result['direction'] == 'SHORT':
            result['confidence'] = 'HIGH'
            result['reason'] += f" | Funding: {funding_rule['interpretation']}"
        elif funding_rule['bias'] == 'BULLISH' and result['direction'] in ['SHORT', 'WAIT']:
            result['funding_warning'] = f"⚠️ Funding favors LONG: {funding_rule['interpretation']}"
        elif funding_rule['bias'] == 'BEARISH' and result['direction'] in ['LONG', 'WAIT']:
            result['funding_warning'] = f"⚠️ Funding favors SHORT: {funding_rule['interpretation']}"
    
    return result

def get_stock_etf_decision(
    money_flow_phase: str,
    rsi: float,
    structure_type: str
) -> dict:
    """Get decision for stocks/ETFs when no whale data available"""
    
    # Determine RSI state
    if rsi < RSI_THRESHOLDS['oversold']:
        rsi_state = 'RSI_OVERSOLD'
    elif rsi > RSI_THRESHOLDS['overbought']:
        rsi_state = 'RSI_OVERBOUGHT'
    else:
        rsi_state = 'RSI_NEUTRAL'
    
    # Lookup in stock/ETF rules
    key = (money_flow_phase, rsi_state)
    
    if key in STOCK_ETF_RULES:
        rule = STOCK_ETF_RULES[key]
        
        # Determine direction from action
        if '🟢' in rule['action'] or 'BUY' in rule['action']:
            direction = 'LONG'
        elif '🔴' in rule['action'] or 'SHORT' in rule['action']:
            direction = 'SHORT'
        else:
            direction = 'WAIT'
        
        return {
            'action': rule['action'],
            'direction': direction,
            'confidence': 'MEDIUM',  # Lower confidence without whale data
            'reason': f"{rule['reason']} (RSI: {rsi:.0f})",
            'entry': 'Based on TA + Flow (no institutional data)',
            'no_whale_data': True,
        }
    
    # Fallback
    return {
        'action': '⏳ WAIT',
        'direction': 'WAIT',
        'confidence': 'LOW',
        'reason': f'No clear signal for {money_flow_phase} + RSI {rsi:.0f}',
        'entry': 'Wait for better setup',
        'no_whale_data': True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 📖 COMBINED LEARNING GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
# Connects ALL data points into ONE coherent narrative
# Tells the FULL STORY of what's happening and WHY
# ─────────────────────────────────────────────────────────────────────────────


def generate_combined_learning(
    signal_name: str,
    direction: str,
    whale_pct: float,
    retail_pct: float,
    oi_change: float,
    price_change: float,
    money_flow_phase: str,
    structure_type: str,
    position_pct: float,
    ta_score: float = 50,
    funding_rate: float = None,
    trade_direction: str = None,
    predictive_signal: dict = None,
    # NEW: SMC Level Proximity
    at_bullish_ob: bool = False,
    at_bearish_ob: bool = False,
    near_bullish_ob: bool = False,
    near_bearish_ob: bool = False,
    at_support: bool = False,
    at_resistance: bool = False,
    # NEW: Final verdict from main analysis (Hybrid/ML/Rules)
    final_verdict: str = None,           # The actual action: LONG, SHORT, WAIT
    final_verdict_reason: str = None,    # Why (e.g., "LATE (100%) + ML not bullish")
    ml_prediction: str = None,           # ML says: LONG, SHORT, WAIT, NEUTRAL
    engine_mode: str = 'hybrid',         # 'hybrid', 'ml', 'rules'
) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    UNIFIED COMBINED LEARNING - Powered by MASTER_RULES
    ═══════════════════════════════════════════════════════════════════════════════
    
    This function now delegates to MASTER_RULES for all decision logic.
    It formats the output for display in the Combined Learning section.
    
    NEW: If final_verdict is provided (from Hybrid/ML analysis), use that
    instead of recalculating. This ensures consistency across the UI.
    
    Returns:
        dict with conclusion, stories, and display data
    """
    
    stories = []
    conflicts = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Get decision from MASTER_RULES (THE source of truth)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if MASTER_RULES_AVAILABLE and whale_pct is not None and retail_pct is not None:
        try:
            decision = get_trade_decision(
                whale_pct=whale_pct,
                retail_pct=retail_pct,
                oi_change=oi_change if oi_change else 0,
                price_change=price_change if price_change else 0,
                position_pct=position_pct if position_pct else 50,
                ta_score=ta_score,
                money_flow_phase=money_flow_phase,
                # SMC Level Proximity
                at_bullish_ob=at_bullish_ob,
                at_bearish_ob=at_bearish_ob,
                near_bullish_ob=near_bullish_ob,
                near_bearish_ob=near_bearish_ob,
                at_support=at_support,
                at_resistance=at_resistance,
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Override conclusion with final_verdict if provided
            # This ensures the Combined Learning matches the main analysis!
            # ═══════════════════════════════════════════════════════════════════
            
            if final_verdict and final_verdict in ['LONG', 'SHORT', 'WAIT']:
                # Use the final verdict from Hybrid/ML analysis
                true_direction = final_verdict
                
                # Build conclusion based on final verdict + reason
                if final_verdict == 'WAIT':
                    if final_verdict_reason:
                        conclusion = f"⏳ WAIT - {final_verdict_reason}"
                    else:
                        conclusion = f"⏳ WAIT - No clear edge or poor entry timing"
                    conclusion_action = "WAIT for better setup"
                    
                    # Only add as CONFLICT if ML says something different!
                    # ML WAIT + Rules WAIT = ALIGNED (not conflict)
                    # ML LONG + Rules WAIT = CONFLICT
                    if engine_mode == 'hybrid' and ml_prediction:
                        if ml_prediction == 'LONG':
                            # ML bullish but we're waiting - this IS a conflict
                            conflicts.append(f"🤖 ML predicts LONG but entry timing poor")
                        elif ml_prediction == 'SHORT':
                            # ML bearish but we're waiting - could be conflict depending on direction
                            # Only conflict if Rules were leaning bullish
                            if direction in ['BULLISH', 'LEAN_BULLISH']:
                                conflicts.append(f"🤖 ML predicts SHORT vs bullish whale positioning")
                        # ML WAIT + final WAIT = No conflict, they agree!
                            
                elif final_verdict == 'LONG':
                    conclusion = decision.conclusion if 'LONG' in decision.conclusion else f"🟢 LONG SETUP - Whales {whale_pct:.0f}% lean bullish"
                    conclusion_action = decision.conclusion_action if decision.trade_direction == 'LONG' else "LONG"
                    
                    # Check for ML conflict
                    if engine_mode == 'hybrid' and ml_prediction:
                        if ml_prediction == 'SHORT':
                            conflicts.append(f"⚠️ ML predicts SHORT - exercise caution on long")
                        elif ml_prediction == 'WAIT':
                            conflicts.append(f"🤖 ML predicts WAIT - consider smaller position")
                    
                elif final_verdict == 'SHORT':
                    conclusion = decision.conclusion if 'SHORT' in decision.conclusion else f"🔴 SHORT SETUP - Whales {whale_pct:.0f}% lean bearish"
                    conclusion_action = decision.conclusion_action if decision.trade_direction == 'SHORT' else "SHORT"
                    
                    # Check for ML conflict
                    if engine_mode == 'hybrid' and ml_prediction:
                        if ml_prediction == 'LONG':
                            conflicts.append(f"⚠️ ML predicts LONG - exercise caution on short")
                        elif ml_prediction == 'WAIT':
                            conflicts.append(f"🤖 ML predicts WAIT - consider smaller position")
            else:
                # Use MASTER_RULES output directly (fallback)
                conclusion = decision.conclusion
                conclusion_action = decision.conclusion_action
                true_direction = decision.trade_direction
            
            # Build stories from MASTER_RULES
            stories.append(('🐋 Whale Analysis', decision.whale_story))
            stories.append(('📈 Open Interest', decision.oi_story))
            stories.append(('📍 Entry Timing', decision.position_story))
            
            # Add ML story if in hybrid mode
            if engine_mode == 'hybrid' and ml_prediction:
                ml_emoji = "🟢" if ml_prediction == 'LONG' else "🔴" if ml_prediction == 'SHORT' else "⏳"
                stories.append(('🤖 ML Prediction', f"{ml_emoji} ML says: {ml_prediction}"))
            
            # Add warnings as conflicts
            if decision.warnings:
                conflicts.extend(decision.warnings)
            
            # Determine squeeze/trap status
            is_squeeze = decision.squeeze_label in ['HIGH_SQUEEZE', 'MODERATE_SQUEEZE']
            squeeze_type = decision.squeeze_label if is_squeeze else None
            
            # Is this a capitulation long?
            is_capitulation_long = 'CAPITULATION' in decision.action or 'CAPITULATION' in conclusion
            
            # Is this a late squeeze (risky)?
            is_late_squeeze = is_squeeze and decision.position_label == 'LATE'
            
            # Build full narrative
            full_narrative = f"""
**🎯 CONCLUSION**
{conclusion}

**🐋 Whale Analysis**
{decision.whale_story}

**📈 Open Interest Analysis**
{decision.oi_story}

**📍 Entry Timing**
{decision.position_story}
"""
            
            return {
                'conclusion': conclusion,
                'conclusion_action': conclusion_action,
                'direction': true_direction,
                'position_quality': decision.position_label,
                'is_squeeze': is_squeeze,
                'squeeze_type': squeeze_type,
                'stories': stories,
                'full_narrative': full_narrative,
                'oi_story': decision.oi_story,
                'whale_story': decision.whale_story,
                'flow_story': f"Phase: {money_flow_phase}",
                'position_story': decision.position_story,
                'has_conflict': len(conflicts) > 0,
                'conflicts': conflicts,
                'oi_bias': 'BULLISH' if oi_change > 0 else 'BEARISH' if oi_change < 0 else 'NEUTRAL',
                'whale_bias': 'BULLISH' if whale_pct > 55 else 'BEARISH' if whale_pct < 45 else 'NEUTRAL',
                'is_capitulation_long': is_capitulation_long,
                'is_late_squeeze': is_late_squeeze,
            }
            
        except Exception as e:
            pass  # Fall through to fallback
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FALLBACK: Basic logic if MASTER_RULES not available
    # ═══════════════════════════════════════════════════════════════════════════
    
    divergence = (whale_pct or 50) - (retail_pct or 50)
    
    # Basic conclusion
    if divergence < -3:
        conclusion = f"⚠️ CAUTION: Retail ({retail_pct:.0f}%) more bullish than Whales ({whale_pct:.0f}%)"
        conclusion_action = "WAIT"
        true_direction = "WAIT"
    elif whale_pct >= 65 and position_pct <= 35:
        conclusion = f"🟢 LONG SETUP: Whales {whale_pct:.0f}% bullish at good entry"
        conclusion_action = "LONG"
        true_direction = "LONG"
    elif whale_pct <= 35 and position_pct >= 65:
        conclusion = f"🔴 SHORT SETUP: Whales {whale_pct:.0f}% bearish at good entry"
        conclusion_action = "SHORT"
        true_direction = "SHORT"
    else:
        conclusion = f"⏳ WAIT: No clear edge (W:{whale_pct:.0f}% R:{retail_pct:.0f}%)"
        conclusion_action = "WAIT"
        true_direction = "WAIT"
    
    # Basic stories
    oi_story = f"OI: {oi_change:+.1f}% | Price: {price_change:+.1f}%"
    whale_story = f"Whales: {whale_pct:.0f}% | Retail: {retail_pct:.0f}% | Divergence: {divergence:+.0f}%"
    position_story = f"Position in range: {position_pct:.0f}%"
    
    stories.append(('📊 Market Data', f"{oi_story}\n{whale_story}"))
    stories.append(('📍 Position', position_story))
    
    return {
        'conclusion': conclusion,
        'conclusion_action': conclusion_action,
        'direction': true_direction,
        'position_quality': 'EARLY' if position_pct <= 35 else 'LATE' if position_pct >= 65 else 'MIDDLE',
        'is_squeeze': divergence > 15,
        'squeeze_type': 'SQUEEZE' if divergence > 15 else None,
        'stories': stories,
        'full_narrative': f"{conclusion}\n\n{oi_story}\n{whale_story}\n{position_story}",
        'oi_story': oi_story,
        'whale_story': whale_story,
        'flow_story': f"Phase: {money_flow_phase}",
        'position_story': position_story,
        'has_conflict': divergence < -3,
        'conflicts': conflicts,
        'oi_bias': 'BULLISH' if oi_change > 0 else 'BEARISH' if oi_change < 0 else 'NEUTRAL',
        'whale_bias': 'BULLISH' if whale_pct > 55 else 'BEARISH' if whale_pct < 45 else 'NEUTRAL',
        'is_capitulation_long': False,
        'is_late_squeeze': False,
    }

def get_predictive_signal(
    structure_type: str,
    money_flow_phase: str,
    whale_pct: float,
    retail_pct: float,
    oi_change: float = 0,
    price_change: float = 0,
    position_pct: float = 50,
) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    COMPLETE PREDICTIVE SIGNAL MATRIX - 27 Scenarios (3x3x3)
    ═══════════════════════════════════════════════════════════════════════════════
    
    INPUTS:
    - Structure (LAGGING): Bullish, Bearish, Neutral
    - Money Flow (MIXED): Bullish phases, Bearish phases, Neutral phases  
    - Whale Position (LEADING): Bullish (>55%), Bearish (<45%), Neutral (45-55%)
    - OI + Price (MOST PREDICTIVE): New money entering tells us direction!
    
    RULE: When LEADING (whales) diverges from LAGGING (structure), follow the LEADER!
    NEW RULE: OI + Price confirmation is the MOST predictive signal!
    
    ═══════════════════════════════════════════════════════════════════════════════
    """
    
    # Classify inputs into 3 states each
    divergence = whale_pct - retail_pct
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OI + PRICE CONFIRMATION - The Most Predictive Signal!
    # This is checked FIRST because it's the strongest edge we have
    # ═══════════════════════════════════════════════════════════════════════════
    
    oi_confirms_bullish = oi_change > 2 and price_change > 2  # New longs entering
    oi_confirms_bearish = oi_change > 2 and price_change < -2  # New shorts entering
    oi_strong_bullish = oi_change > 5 and price_change > 3  # STRONG new long positions
    oi_strong_bearish = oi_change > 5 and price_change < -3  # STRONG new short positions
    
    # Whale state (LEADING indicator)
    whale_bullish = whale_pct >= 55
    whale_bearish = whale_pct <= 45
    whale_neutral = not whale_bullish and not whale_bearish  # 46-54%
    
    # Key insight: Neutral whales (45-55%) are NOT opposing the trade!
    # Only penalize when whales are AGAINST the direction
    whale_not_opposing_bullish = whale_pct >= 45  # Not bearish = not opposing longs
    whale_not_opposing_bearish = whale_pct <= 55  # Not bullish = not opposing shorts
    
    # Structure state (LAGGING indicator)
    structure_bullish = structure_type in ['Bullish']
    structure_bearish = structure_type in ['Bearish']
    structure_neutral = not structure_bullish and not structure_bearish
    
    # Money Flow state (MIXED - can be early warning)
    # Use knowledge base constants for consistency
    bullish_flows = MONEY_FLOW_PHASE_RULES['accumulation_phases']
    bearish_flows = MONEY_FLOW_PHASE_RULES['distribution_phases']
    neutral_flows = MONEY_FLOW_PHASE_RULES['neutral_phases']
    
    flow_bullish = money_flow_phase in bullish_flows
    flow_bearish = money_flow_phase in bearish_flows
    flow_neutral = money_flow_phase in neutral_flows
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 OI CONFIRMATION OVERRIDE - Highest Priority after Squeeze!
    # When OI+Price clearly shows new money entering, that's the strongest signal
    # ═══════════════════════════════════════════════════════════════════════════
    
    # STRONG OI BULLISH: New longs entering AND whales not opposing
    if oi_strong_bullish and whale_not_opposing_bullish and flow_bullish:
        return {
            'signal': '🚀 STRONG BUY',
            'subtitle': 'New longs entering with momentum',
            'color': '#00ff88',
            'interpretation': f'OI +{oi_change:.1f}% + Price +{price_change:.1f}% = Fresh buying. Flow: {money_flow_phase}. Whales: {whale_pct:.0f}% (not opposing).',
            'action': 'LONG now. New institutional money entering. Strong momentum.',
            'confidence': 'HIGH',
        }
    
    # OI BULLISH: New longs entering with bullish flow, whales not opposing
    if oi_confirms_bullish and whale_not_opposing_bullish and flow_bullish:
        return {
            'signal': '✅ BUY - OI CONFIRMED',
            'subtitle': 'New longs entering',
            'color': '#00ff88',
            'interpretation': f'OI +{oi_change:.1f}% + Price +{price_change:.1f}% = New longs. Flow: {money_flow_phase}. Whales: {whale_pct:.0f}%.',
            'action': 'LONG. OI confirms new positions entering in bullish direction.',
            'confidence': 'HIGH',
        }
    
    # STRONG OI BEARISH: New shorts entering AND whales not opposing
    if oi_strong_bearish and whale_not_opposing_bearish and flow_bearish:
        return {
            'signal': '🔴 STRONG SELL',
            'subtitle': 'New shorts entering with momentum',
            'color': '#ff4444',
            'interpretation': f'OI +{oi_change:.1f}% + Price {price_change:.1f}% = Fresh selling. Flow: {money_flow_phase}. Whales: {whale_pct:.0f}%.',
            'action': 'SHORT now. New institutional shorts entering. Strong downward pressure.',
            'confidence': 'HIGH',
        }
    
    # OI BEARISH: New shorts entering with bearish flow, whales not opposing
    if oi_confirms_bearish and whale_not_opposing_bearish and flow_bearish:
        return {
            'signal': '🔴 SELL - OI CONFIRMED',
            'subtitle': 'New shorts entering',
            'color': '#ff4444',
            'interpretation': f'OI +{oi_change:.1f}% + Price {price_change:.1f}% = New shorts. Flow: {money_flow_phase}. Whales: {whale_pct:.0f}%.',
            'action': 'SHORT. OI confirms new positions entering in bearish direction.',
            'confidence': 'HIGH',
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL: CHECK SQUEEZE SETUPS (highest priority - explosive moves)
    # Now considers POSITION - don't chase squeezes at extreme prices!
    # ═══════════════════════════════════════════════════════════════════════════
    
    if whale_pct >= 60 and retail_pct <= 45 and divergence >= 15:
        # SHORT SQUEEZE detected - but check position for entry quality!
        if position_pct <= 50:
            # Good entry - early/middle of range
            return {
                'signal': '🔥 SHORT SQUEEZE',
                'subtitle': 'Shorts about to get liquidated',
                'color': '#ff00ff',
                'interpretation': f'Whales {whale_pct:.0f}% long vs Retail {retail_pct:.0f}%. Position {position_pct:.0f}% = GOOD ENTRY!',
                'action': 'LONG immediately. Target 5-10%. Explosive move incoming.',
                'confidence': 'HIGH',
                'trade_direction': 'LONG',
            }
        elif position_pct <= 70:
            # Late but acceptable
            return {
                'signal': '🔥 SHORT SQUEEZE',
                'subtitle': 'Squeeze active - late entry',
                'color': '#ffaa00',
                'interpretation': f'Whales {whale_pct:.0f}% long vs Retail {retail_pct:.0f}%. Position {position_pct:.0f}% = LATE. Reduce size.',
                'action': 'CAUTIOUS LONG. Squeeze valid but entry not ideal. Use smaller size.',
                'confidence': 'MEDIUM',
                'trade_direction': 'LONG',
            }
        else:
            # Chasing - too late
            return {
                'signal': '⚠️ SQUEEZE (Late)',
                'subtitle': 'Squeeze in progress - WAIT',
                'color': '#ff6600',
                'interpretation': f'Whales {whale_pct:.0f}% long vs Retail {retail_pct:.0f}%. BUT position {position_pct:.0f}% = CHASING!',
                'action': 'WAIT for pullback. Squeeze valid but entry is terrible at highs.',
                'confidence': 'LOW',
                'trade_direction': 'WAIT',  # Don't enter at highs!
            }
    
    if whale_pct <= 40 and retail_pct >= 55 and divergence <= -15:
        # LONG SQUEEZE detected - but check position for entry quality!
        if position_pct >= 50:
            # Good entry - high/middle of range (good for shorts)
            return {
                'signal': '💥 LONG SQUEEZE',
                'subtitle': 'Longs about to get liquidated',
                'color': '#ff00ff',
                'interpretation': f'Whales {whale_pct:.0f}% long vs Retail {retail_pct:.0f}%. Position {position_pct:.0f}% = GOOD ENTRY!',
                'action': 'SHORT immediately. Target 5-10%. Violent drop incoming.',
                'confidence': 'HIGH',
                'trade_direction': 'SHORT',
            }
        elif position_pct >= 30:
            # Late but acceptable
            return {
                'signal': '💥 LONG SQUEEZE',
                'subtitle': 'Squeeze active - late entry',
                'color': '#ffaa00',
                'interpretation': f'Whales {whale_pct:.0f}% long vs Retail {retail_pct:.0f}%. Position {position_pct:.0f}% = LATE. Reduce size.',
                'action': 'CAUTIOUS SHORT. Squeeze valid but entry not ideal. Use smaller size.',
                'confidence': 'MEDIUM',
                'trade_direction': 'SHORT',
            }
        else:
            # Chasing - too late
            return {
                'signal': '⚠️ SQUEEZE (Late)',
                'subtitle': 'Squeeze in progress - WAIT',
                'color': '#ff6600',
                'interpretation': f'Whales {whale_pct:.0f}% long vs Retail {retail_pct:.0f}%. BUT position {position_pct:.0f}% = CHASING!',
                'action': 'WAIT for bounce. Squeeze valid but entry is terrible at lows.',
                'confidence': 'LOW',
                'trade_direction': 'WAIT',  # Don't enter at lows!
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔥 CAPITULATION + WHALES LONG = LEADING INDICATOR BUY SIGNAL
    # This is PREDICTIVE - whales are accumulating during panic BEFORE reversal
    # Do NOT wait for OI confirmation - that comes AFTER the move starts!
    # ═══════════════════════════════════════════════════════════════════════════
    
    if money_flow_phase == 'CAPITULATION' and whale_pct >= 65:
        # STRONG: Whales heavily long during capitulation - they're buying the panic
        if position_pct <= 35:
            return {
                'signal': '🔥 EARLY LONG',
                'subtitle': 'Whales accumulating capitulation!',
                'color': '#00ff88',
                'interpretation': f'CAPITULATION + Whales {whale_pct:.0f}% LONG at {position_pct:.0f}% of range. Classic smart money accumulation!',
                'action': 'LONG NOW. Whales buying panic. Reversal likely. Stop below recent low.',
                'confidence': 'HIGH',
                'trade_direction': 'LONG',
            }
        else:
            return {
                'signal': '✅ LONG SETUP',
                'subtitle': 'Whales buying capitulation',
                'color': '#00d4aa',
                'interpretation': f'CAPITULATION + Whales {whale_pct:.0f}% LONG. Good but entry at {position_pct:.0f}% - wait for better price.',
                'action': 'WAIT for dip to lower range, then LONG. Reversal setting up.',
                'confidence': 'MEDIUM',
                'trade_direction': 'LONG',
            }
    
    if money_flow_phase == 'CAPITULATION' and whale_pct >= 55:
        # MODERATE: Whales starting to accumulate
        if position_pct <= 40:
            return {
                'signal': '✅ ACCUMULATION',
                'subtitle': 'Whales starting to buy panic',
                'color': '#00d4aa',
                'interpretation': f'CAPITULATION at lows + Whales {whale_pct:.0f}% LONG. Early accumulation stage.',
                'action': 'Scale in LONG. Stop below capitulation low. Whales positioning.',
                'confidence': 'MEDIUM',
                'trade_direction': 'LONG',
            }
        else:
            return {
                'signal': '👀 WATCH FOR ENTRY',
                'subtitle': 'Capitulation + whale interest',
                'color': '#ffcc00',
                'interpretation': f'CAPITULATION + Whales {whale_pct:.0f}% but price at {position_pct:.0f}%. Wait for better entry.',
                'action': 'WATCH. Wait for price to pull back to range lows before LONG.',
                'confidence': 'LOW',
                'trade_direction': 'LONG',
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔥 ACCUMULATION + WHALES LONG at LOWS = EARLY LONG
    # ═══════════════════════════════════════════════════════════════════════════
    
    if money_flow_phase == 'ACCUMULATION' and whale_pct >= 65 and position_pct <= 35:
        return {
            'signal': '🔥 EARLY LONG',
            'subtitle': 'Accumulation at range lows',
            'color': '#00ff88',
            'interpretation': f'ACCUMULATION phase + Whales {whale_pct:.0f}% LONG at {position_pct:.0f}% of range. Ideal setup!',
            'action': 'LONG NOW. Smart money accumulating at lows. Strong conviction.',
            'confidence': 'HIGH',
            'trade_direction': 'LONG',
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔥 DISTRIBUTION/EXHAUSTION + WHALES SHORT at HIGHS = EARLY SHORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    if money_flow_phase in ['DISTRIBUTION', 'EXHAUSTION', 'FOMO / DIST RISK'] and whale_pct <= 35 and position_pct >= 65:
        return {
            'signal': '🔥 EARLY SHORT',
            'subtitle': 'Distribution at range highs',
            'color': '#ff4444',
            'interpretation': f'{money_flow_phase} + Whales only {whale_pct:.0f}% LONG at {position_pct:.0f}% of range. Smart money exiting!',
            'action': 'SHORT NOW. Smart money distributing at highs. Stop above recent high.',
            'confidence': 'HIGH',
            'trade_direction': 'SHORT',
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # STRUCTURE: BULLISH (9 scenarios: 1-9)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if structure_bullish:
        
        # 1. Bullish + Bullish Flow + Bullish Whales = CONTINUE LONG (HIGH)
        if flow_bullish and whale_bullish:
            return {
                'signal': '✅ CONTINUE LONG',
                'subtitle': 'All signals aligned bullish',
                'color': '#00ff88',
                'interpretation': f'Structure UP + Flow {money_flow_phase} + Whales {whale_pct:.0f}% long. Perfect alignment!',
                'action': 'Hold LONG. Add on pullbacks. Trail stop to lock profits.',
                'confidence': 'HIGH',
            }
        
        # 2. Bullish + Bullish Flow + Neutral Whales = LONG (MEDIUM)
        # OI confirmation upgrades confidence!
        if flow_bullish and whale_neutral:
            if oi_confirms_bullish:
                return {
                    'signal': '✅ LONG - CONFIRMED',
                    'subtitle': 'OI confirms new longs entering',
                    'color': '#00ff88',
                    'interpretation': f'Structure UP + Flow {money_flow_phase} + OI +{oi_change:.1f}%. New longs entering! Whales {whale_pct:.0f}% not opposing.',
                    'action': 'LONG with confidence. OI confirms institutional buying.',
                    'confidence': 'HIGH',
                }
            else:
                return {
                    'signal': '✅ LONG',
                    'subtitle': 'Good setup, awaiting whale confirmation',
                    'color': '#00d4aa',
                    'interpretation': f'Structure UP + Flow {money_flow_phase}. Whales {whale_pct:.0f}% (neutral). Waiting for >55% or OI confirmation.',
                    'action': 'LONG with normal size. Add when whales confirm >55% or OI confirms.',
                    'confidence': 'MEDIUM',
                }
        
        # 3. Bullish + Bullish Flow + Bearish Whales = CONFLICT (LOW)
        if flow_bullish and whale_bearish:
            return {
                'signal': '⚠️ CONFLICT',
                'subtitle': 'Structure vs Whales disagree',
                'color': '#ffcc00',
                'interpretation': f'Structure bullish but whales only {whale_pct:.0f}% long. They\'re not buying this rally.',
                'action': 'Reduce size. Consider exiting on next rally. Don\'t add.',
                'confidence': 'LOW',
            }
        
        # 4. Bullish + Neutral Flow + Bullish Whales = BUY THE DIP (HIGH)
        if flow_neutral and whale_bullish:
            return {
                'signal': '🛒 BUY THE DIP',
                'subtitle': 'Healthy pullback in uptrend',
                'color': '#00ff88',
                'interpretation': f'Uptrend intact. Flow shows {money_flow_phase} (pause). Whales {whale_pct:.0f}% still bullish.',
                'action': 'Add to LONG on support. This is the dip you wait for!',
                'confidence': 'HIGH',
            }
        
        # 5. Bullish + Neutral Flow + Neutral Whales = HOLD (MEDIUM)
        if flow_neutral and whale_neutral:
            return {
                'signal': '↔️ HOLD',
                'subtitle': 'Trend intact but no conviction',
                'color': '#ffcc00',
                'interpretation': f'Uptrend but Flow {money_flow_phase} + Whales {whale_pct:.0f}% neutral. Momentum fading.',
                'action': 'Hold existing. Don\'t add. Tighten stops.',
                'confidence': 'MEDIUM',
            }
        
        # 6. Bullish + Neutral Flow + Bearish Whales = TAKE PROFIT (HIGH)
        if flow_neutral and whale_bearish:
            return {
                'signal': '⚠️ TAKE PROFIT',
                'subtitle': 'Whales exiting - top forming',
                'color': '#ff9500',
                'interpretation': f'Structure up but whales only {whale_pct:.0f}% long during {money_flow_phase}. Distribution!',
                'action': 'Close longs. Do NOT buy. Consider SHORT on next rally.',
                'confidence': 'HIGH',
            }
        
        # 7. Bullish + Bearish Flow + Bullish Whales = REDUCE (LOW)
        if flow_bearish and whale_bullish:
            return {
                'signal': '⚠️ REDUCE SIZE',
                'subtitle': 'Flow warning but whales holding',
                'color': '#ffcc00',
                'interpretation': f'Flow shows {money_flow_phase} (bearish) but whales {whale_pct:.0f}% still long. Mixed signals.',
                'action': 'Reduce position size. Tighten stops. Watch for whale exit.',
                'confidence': 'LOW',
            }
        
        # 8. Bullish + Bearish Flow + Neutral Whales = EXIT SOON (MEDIUM)
        if flow_bearish and whale_neutral:
            return {
                'signal': '⚠️ EXIT SOON',
                'subtitle': 'Distribution starting',
                'color': '#ff9500',
                'interpretation': f'Flow shows {money_flow_phase}. Whales {whale_pct:.0f}% (leaving). Top forming.',
                'action': 'Exit on next rally. Do NOT hold through breakdown.',
                'confidence': 'MEDIUM',
            }
        
        # 9. Bullish + Bearish Flow + Bearish Whales = EXIT/SHORT (HIGH)
        if flow_bearish and whale_bearish:
            return {
                'signal': '🚨 EXIT LONGS',
                'subtitle': 'Top confirmed - reversal imminent',
                'color': '#ff4444',
                'interpretation': f'Flow {money_flow_phase} + Whales {whale_pct:.0f}% short. Smart money exited. TOP!',
                'action': 'EXIT all longs NOW. Consider SHORT on rallies.',
                'confidence': 'HIGH',
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRUCTURE: BEARISH (9 scenarios: 10-18)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if structure_bearish:
        
        # 10. Bearish + Bullish Flow + Bullish Whales = EARLY LONG (HIGH) ⭐
        if flow_bullish and whale_bullish:
            return {
                'signal': '🚀 EARLY LONG',
                'subtitle': 'Accumulation before reversal',
                'color': '#00ff88',
                'interpretation': f'Structure bearish BUT Flow {money_flow_phase} + Whales {whale_pct:.0f}% accumulating!',
                'action': 'Scale in LONG. Stop below recent low. Best entry before the crowd!',
                'confidence': 'HIGH',
            }
        
        # 11. Bearish + Bullish Flow + Neutral Whales = WATCH LONG (MEDIUM)
        if flow_bullish and whale_neutral:
            return {
                'signal': '👀 WATCH FOR LONG',
                'subtitle': 'Flow bullish, whales not confirmed',
                'color': '#00d4aa',
                'interpretation': f'Structure bearish, Flow {money_flow_phase} (bullish) but whales {whale_pct:.0f}% neutral.',
                'action': 'Watch for whales to go >55%. Small position OK with tight stop.',
                'confidence': 'MEDIUM',
            }
        
        # 12. Bearish + Bullish Flow + Bearish Whales = CONFLICT (LOW)
        if flow_bullish and whale_bearish:
            return {
                'signal': '⚠️ DEAD CAT BOUNCE',
                'subtitle': 'Rally in downtrend - trap!',
                'color': '#ff9500',
                'interpretation': f'Flow shows {money_flow_phase} but whales only {whale_pct:.0f}% long. They\'re selling this rally.',
                'action': 'Do NOT buy. This rally is a trap. Wait or SHORT on resistance.',
                'confidence': 'LOW',
            }
        
        # 13. Bearish + Neutral Flow + Bullish Whales = EARLY LONG (HIGH) ⭐
        if flow_neutral and whale_bullish:
            return {
                'signal': '🎯 EARLY LONG',
                'subtitle': 'Whales accumulating quietly',
                'color': '#00ff88',
                'interpretation': f'Structure bearish, Flow {money_flow_phase} BUT whales {whale_pct:.0f}% positioned long!',
                'action': 'Scale in LONG near support. Whales are ready for reversal.',
                'confidence': 'HIGH',
            }
        
        # 14. Bearish + Neutral Flow + Neutral Whales = WAIT (LOW)
        if flow_neutral and whale_neutral:
            return {
                'signal': '⏳ WAIT',
                'subtitle': 'No clear edge',
                'color': '#888888',
                'interpretation': f'Bearish structure, {money_flow_phase}, whales {whale_pct:.0f}% neutral. No conviction.',
                'action': 'Wait for clarity. No trade here.',
                'confidence': 'LOW',
            }
        
        # 15. Bearish + Neutral Flow + Bearish Whales = SELL RALLY (MEDIUM)
        if flow_neutral and whale_bearish:
            return {
                'signal': '📉 SELL THE RALLY',
                'subtitle': 'Short covering bounce - sell it',
                'color': '#ff6b6b',
                'interpretation': f'Downtrend + {money_flow_phase} + whales {whale_pct:.0f}% short. Rally = exit liquidity.',
                'action': 'SHORT on resistance. Any rally is a gift to short.',
                'confidence': 'MEDIUM',
            }
        
        # 16. Bearish + Bearish Flow + Bullish Whales = CONFLICT (LOW)
        if flow_bearish and whale_bullish:
            return {
                'signal': '⚠️ CONFLICT',
                'subtitle': 'Whales vs Flow disagree',
                'color': '#ffcc00',
                'interpretation': f'Flow {money_flow_phase} (bearish) but whales {whale_pct:.0f}% long. Someone is wrong.',
                'action': 'Wait. Either whales are early OR they\'ll capitulate. Watch whale %.',
                'confidence': 'LOW',
            }
        
        # 17. Bearish + Bearish Flow + Neutral Whales = SHORT (MEDIUM)
        if flow_bearish and whale_neutral:
            return {
                'signal': '🔴 SHORT',
                'subtitle': 'Downtrend continuing',
                'color': '#ff6b6b',
                'interpretation': f'Bearish structure + {money_flow_phase}. Whales {whale_pct:.0f}% not fighting it.',
                'action': 'SHORT on rallies. Stop above recent high.',
                'confidence': 'MEDIUM',
            }
        
        # 18. Bearish + Bearish Flow + Bearish Whales = CONTINUE SHORT (HIGH)
        if flow_bearish and whale_bearish:
            return {
                'signal': '🔻 CONTINUE SHORT',
                'subtitle': 'All signals aligned bearish',
                'color': '#ff4444',
                'interpretation': f'Structure DOWN + Flow {money_flow_phase} + Whales {whale_pct:.0f}% short. Perfect alignment!',
                'action': 'Hold SHORT. Add on rallies. Trail stop to lock profits.',
                'confidence': 'HIGH',
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRUCTURE: NEUTRAL/MIXED (9 scenarios: 19-27)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if structure_neutral:
        
        # 19. Neutral + Bullish Flow + Bullish Whales = BUILDING LONG (HIGH)
        if flow_bullish and whale_bullish:
            return {
                'signal': '📈 BUILDING LONG',
                'subtitle': 'Breakout setup forming',
                'color': '#00ff88',
                'interpretation': f'Range but Flow {money_flow_phase} + Whales {whale_pct:.0f}% long. Accumulation before breakout!',
                'action': 'LONG now or on dip to range low. Stop below range. Breakout coming!',
                'confidence': 'HIGH',
            }
        
        # 20. Neutral + Bullish Flow + Neutral Whales = WATCH LONG (MEDIUM)
        # BUT: If OI confirms bullish, upgrade to BUY!
        if flow_bullish and whale_neutral:
            if oi_confirms_bullish:
                return {
                    'signal': '✅ BUY DIP',
                    'subtitle': 'OI confirms bullish despite neutral whales',
                    'color': '#00d4aa',
                    'interpretation': f'Range + {money_flow_phase} (bullish). OI +{oi_change:.1f}% confirms new longs. Whales {whale_pct:.0f}% neutral but not opposing.',
                    'action': 'BUY on dips. OI confirms direction even without strong whale positioning.',
                    'confidence': 'MEDIUM',
                }
            else:
                return {
                    'signal': '👀 WATCH FOR LONG',
                    'subtitle': 'Flow bullish, awaiting confirmation',
                    'color': '#00d4aa',
                    'interpretation': f'Range + {money_flow_phase} (bullish) but whales {whale_pct:.0f}% neutral. Not confirmed yet.',
                    'action': 'Buy near range lows. Wait for whales >55% OR OI confirmation to add size.',
                    'confidence': 'LOW',
                }
        
        # 21. Neutral + Bullish Flow + Bearish Whales = CONFLICT (LOW)
        if flow_bullish and whale_bearish:
            return {
                'signal': '⚠️ CONFLICT',
                'subtitle': 'Flow vs Whales disagree',
                'color': '#ffcc00',
                'interpretation': f'Flow {money_flow_phase} (bullish) but whales {whale_pct:.0f}% bearish. Contradiction!',
                'action': 'Avoid. Wait for alignment. Range likely to break DOWN.',
                'confidence': 'LOW',
            }
        
        # 22. Neutral + Neutral Flow + Bullish Whales = ACCUMULATING (MEDIUM)
        if flow_neutral and whale_bullish:
            return {
                'signal': '🎯 ACCUMULATING',
                'subtitle': 'Whales buying the range',
                'color': '#00d4aa',
                'interpretation': f'Range + {money_flow_phase} + whales {whale_pct:.0f}% positioned long. Stealth accumulation!',
                'action': 'Buy near range lows. Stop below range. Upside breakout likely.',
                'confidence': 'MEDIUM',
            }
        
        # 23. Neutral + Neutral Flow + Neutral Whales = RANGE TRADE (LOW)
        if flow_neutral and whale_neutral:
            return {
                'signal': '↔️ RANGE TRADE',
                'subtitle': 'No direction - trade the range',
                'color': '#888888',
                'interpretation': f'Everything neutral. No trend. Pure range-bound price action.',
                'action': 'Buy range lows, sell range highs. Or wait for breakout.',
                'confidence': 'LOW',
                'trade_direction': 'WAIT',
            }
        
        # 24. Neutral + Neutral Flow + Bearish Whales = DISTRIBUTING (MEDIUM)
        if flow_neutral and whale_bearish:
            return {
                'signal': '⚠️ DISTRIBUTING',
                'subtitle': 'Whales selling the range',
                'color': '#ff9500',
                'interpretation': f'Range + {money_flow_phase} + whales {whale_pct:.0f}% positioned short. Stealth distribution!',
                'action': 'Sell near range highs. Stop above range. Downside breakout likely.',
                'confidence': 'MEDIUM',
            }
        
        # 25. Neutral + Bearish Flow + Bullish Whales = CONFLICT (LOW)
        if flow_bearish and whale_bullish:
            return {
                'signal': '⚠️ CONFLICT',
                'subtitle': 'Flow vs Whales disagree',
                'color': '#ffcc00',
                'interpretation': f'Flow {money_flow_phase} (bearish) but whales {whale_pct:.0f}% bullish. Contradiction!',
                'action': 'Wait for clarity. Whales usually win but flow can be early warning.',
                'confidence': 'LOW',
            }
        
        # 26. Neutral + Bearish Flow + Neutral Whales = WATCH SHORT (MEDIUM)
        # BUT: If OI confirms bearish, upgrade to SELL!
        if flow_bearish and whale_neutral:
            if oi_confirms_bearish:
                return {
                    'signal': '🔴 SELL RALLY',
                    'subtitle': 'OI confirms bearish despite neutral whales',
                    'color': '#ff6b6b',
                    'interpretation': f'Range + {money_flow_phase} (bearish). OI +{oi_change:.1f}% confirms new shorts. Whales {whale_pct:.0f}% neutral but not opposing.',
                    'action': 'SELL on rallies. OI confirms direction even without strong whale positioning.',
                    'confidence': 'MEDIUM',
                }
            else:
                return {
                    'signal': '👀 WATCH FOR SHORT',
                    'subtitle': 'Flow bearish, awaiting confirmation',
                    'color': '#ff9500',
                    'interpretation': f'Range + {money_flow_phase} (bearish). Whales {whale_pct:.0f}% neutral. Breakdown possible.',
                    'action': 'Sell near range highs. Wait for whales <45% OR OI confirmation to add size.',
                    'confidence': 'LOW',
                }
        
        # 27. Neutral + Bearish Flow + Bearish Whales = BUILDING SHORT (HIGH)
        if flow_bearish and whale_bearish:
            return {
                'signal': '📉 BUILDING SHORT',
                'subtitle': 'Breakdown setup forming',
                'color': '#ff4444',
                'interpretation': f'Range but Flow {money_flow_phase} + Whales {whale_pct:.0f}% short. Distribution before breakdown!',
                'action': 'SHORT now or on rally to range high. Stop above range. Breakdown coming!',
                'confidence': 'HIGH',
            }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FALLBACK - Should never reach here with complete matrix!
    # ═══════════════════════════════════════════════════════════════════════════
    
    return {
        'signal': '⏳ ANALYZING',
        'subtitle': 'Processing signals',
        'color': '#888888',
        'interpretation': f'Structure: {structure_type}, Flow: {money_flow_phase}, Whales: {whale_pct:.0f}%',
        'action': 'Wait for clearer signal.',
        'confidence': 'LOW',
        'trade_direction': 'WAIT',
    }


def _add_trade_direction(result: dict) -> dict:
    """
    Helper to ensure trade_direction is set based on signal.
    This is the SINGLE SOURCE OF TRUTH for direction mapping.
    """
    if 'trade_direction' in result:
        return result
    
    signal = result.get('signal', '').upper()
    
    # Check SQUEEZE signals FIRST (most important!)
    if 'SHORT SQUEEZE' in signal:
        result['trade_direction'] = 'LONG'  # SHORT SQUEEZE = LONG trade
    elif 'LONG SQUEEZE' in signal:
        result['trade_direction'] = 'SHORT'  # LONG SQUEEZE = SHORT trade
    # Buy/Long signals
    elif any(x in signal for x in ['BUY', 'LONG', 'ACCUMUL', 'BUILDING LONG']):
        result['trade_direction'] = 'LONG'
    # Sell/Short signals
    elif any(x in signal for x in ['SELL', 'SHORT', 'BUILDING SHORT', 'EXIT LONGS', 'DISTRIBUT']):
        result['trade_direction'] = 'SHORT'
    # Caution signals
    elif any(x in signal for x in ['TAKE PROFIT', 'EXIT SOON', 'REDUCE', 'AVOID']):
        result['trade_direction'] = 'CAUTION'
    # Everything else is WAIT
    else:
        result['trade_direction'] = 'WAIT'
    
    return result


def get_predictive_signal_with_direction(
    structure_type: str,
    money_flow_phase: str,
    whale_pct: float,
    retail_pct: float,
    oi_change: float = 0,
    price_change: float = 0,
    position_pct: float = 50,
) -> dict:
    """
    Wrapper that calls get_predictive_signal and ensures trade_direction is set.
    USE THIS FUNCTION instead of get_predictive_signal directly!
    """
    result = get_predictive_signal(
        structure_type=structure_type,
        money_flow_phase=money_flow_phase,
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        oi_change=oi_change,
        price_change=price_change,
        position_pct=position_pct,
    )
    return _add_trade_direction(result)


def calculate_predictive_score(
    oi_change: float,           # OI change % (24h)
    price_change: float,        # Price change % (24h)
    whale_pct: float,           # Top trader long %
    retail_pct: float,          # Retail long %
    ta_score: int,              # Technical analysis score 0-100
    trade_mode: str = 'DayTrade',
    timeframe: str = '1h',
    funding_rate: float = 0,    # Optional: funding rate
    support_level: float = None, # Optional: nearest support for "wait for dip"
    # Move Position parameters
    current_price: float = None,
    swing_high: float = None,
    swing_low: float = None,
    structure_type: str = 'Unknown',  # 'Bullish', 'Bearish', 'Mixed'
    money_flow_phase: str = 'CONSOLIDATION',  # Money flow phase for predictive signal
    # SMC Level Proximity - CRITICAL for LATE override!
    at_bullish_ob: bool = False,
    at_bearish_ob: bool = False,
    near_bullish_ob: bool = False,
    near_bearish_ob: bool = False,
    at_support: bool = False,
    at_resistance: bool = False,
) -> PredictiveScore:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    UNIFIED PREDICTIVE SCORING - Now powered by MASTER_RULES
    ═══════════════════════════════════════════════════════════════════════════════
    
    This function now delegates ALL scoring logic to MASTER_RULES.get_trade_decision()
    ensuring a SINGLE SOURCE OF TRUTH for all trading decisions.
    
    The flow is:
    1. Calculate position in range
    2. Call MASTER_RULES.get_trade_decision() 
    3. Format result into PredictiveScore
    
    No more scattered logic - MASTER_RULES is THE decision engine.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Calculate position in range
    # ═══════════════════════════════════════════════════════════════════════════
    
    position_pct = 50  # Default middle
    
    if swing_high and swing_low and swing_high > swing_low and current_price:
        position_pct = ((current_price - swing_low) / (swing_high - swing_low)) * 100
        position_pct = max(0, min(100, position_pct))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: Get decision from MASTER_RULES (THE source of truth)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if MASTER_RULES_AVAILABLE:
        decision = get_trade_decision(
            whale_pct=whale_pct,
            retail_pct=retail_pct,
            oi_change=oi_change,
            price_change=price_change,
            position_pct=position_pct,
            ta_score=ta_score,
            money_flow_phase=money_flow_phase,
            # SMC Level Proximity - allows LATE override when at OB!
            at_bullish_ob=at_bullish_ob,
            at_bearish_ob=at_bearish_ob,
            near_bullish_ob=near_bullish_ob,
            near_bearish_ob=near_bearish_ob,
            at_support=at_support,
            at_resistance=at_resistance,
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Map MASTER_RULES output to PredictiveScore format
        # ═══════════════════════════════════════════════════════════════════════
        
        # Direction mapping
        direction = decision.direction_label
        if direction not in ['BULLISH', 'LEAN_BULLISH', 'BEARISH', 'LEAN_BEARISH', 'NEUTRAL']:
            direction = 'NEUTRAL'
        
        # Direction confidence
        if decision.direction_score >= 36:
            direction_confidence = 'HIGH'
        elif decision.direction_score >= 28:
            direction_confidence = 'MEDIUM'
        elif decision.direction_score >= 15:
            direction_confidence = 'LOW'
        else:
            direction_confidence = 'NONE'
        
        # Direction reason
        divergence = whale_pct - retail_pct
        direction_reason = f"Whales {whale_pct:.0f}% long | OI: {oi_change:+.1f}%"
        if divergence > 10:
            direction_reason = f"🐋 Whales {whale_pct:.0f}% bullish (vs Retail {retail_pct:.0f}%)"
        elif divergence < -5:
            direction_reason = f"⚠️ Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}% - Trap risk!"
        
        # Squeeze potential mapping
        squeeze_map = {
            'HIGH_SQUEEZE': 'HIGH',
            'MODERATE_SQUEEZE': 'MEDIUM', 
            'SLIGHT_EDGE': 'LOW',
            'NO_EDGE': 'NONE',
            'EXTREME_TRAP': 'CONFLICT',
            'HIGH_TRAP': 'CONFLICT',
            'MODERATE_TRAP': 'CONFLICT',
            'SLIGHT_TRAP': 'CONFLICT',
        }
        squeeze_potential = squeeze_map.get(decision.squeeze_label, 'NONE')
        
        # Squeeze reason
        if squeeze_potential == 'CONFLICT':
            squeeze_reason = f"⚠️ TRAP: Retail {retail_pct:.0f}% > Whales {whale_pct:.0f}%"
        elif squeeze_potential in ['HIGH', 'MEDIUM']:
            squeeze_reason = f"🔥 SQUEEZE: Whales {whale_pct:.0f}% vs Retail {retail_pct:.0f}%"
        elif divergence > 5:
            squeeze_reason = f"Whales lead by {divergence:.0f}%"
        else:
            squeeze_reason = f"Divergence: {divergence:+.0f}%"
        
        # Entry timing based on trade_direction and position
        if decision.trade_direction == 'WAIT':
            entry_timing = 'WAIT'
        elif decision.position_label == 'EARLY' and decision.trade_direction == 'LONG':
            entry_timing = 'NOW'
        elif decision.position_label == 'LATE' and decision.trade_direction == 'SHORT':
            entry_timing = 'NOW'
        elif decision.position_label == 'MIDDLE':
            entry_timing = 'SOON'
        elif ta_score >= 60:
            entry_timing = 'SOON'
        else:
            entry_timing = 'WAIT'
        
        # Move position label
        move_position = decision.position_label
        
        # Timing reason
        if entry_timing == 'NOW':
            timing_reason = f"Good entry: {move_position} ({position_pct:.0f}%) + TA {ta_score}"
        elif entry_timing == 'WAIT':
            timing_reason = f"Wait: {move_position} ({position_pct:.0f}%) - {decision.position_story[:50]}"
        else:
            timing_reason = f"TA: {ta_score} | Position: {move_position} ({position_pct:.0f}%)"
        
        # Final action string
        trade_direction = decision.trade_direction
        final_score = decision.total_score
        
        if trade_direction == 'LONG':
            if final_score >= 80:
                final_action = '🚀 STRONG BUY'
            elif final_score >= 65:
                final_action = '✅ BUY'
            elif final_score >= 50:
                final_action = '📊 CAUTIOUS BUY'
            else:
                final_action = '⏳ WAIT FOR DIP'
        elif trade_direction == 'SHORT':
            if final_score >= 80:
                final_action = '🔴 STRONG SELL'
            elif final_score >= 65:
                final_action = '🔴 SELL'
            elif final_score >= 50:
                final_action = '📊 CAUTIOUS SELL'
            else:
                final_action = '⏳ WAIT FOR RALLY'
        else:
            # WAIT - show the specific action
            if 'TRAP' in decision.action:
                final_action = f'⚠️ CAUTION: Retail Trap'
            elif decision.action == 'WAIT_FOR_DIP':
                final_action = '⏳ WAIT FOR DIP'
            elif decision.action == 'WAIT_FOR_BOUNCE':
                final_action = '⏳ WAIT FOR BOUNCE'
            else:
                final_action = '⏳ WAIT'
        
        # Final summary
        final_summary = decision.conclusion
        
        # Build predictive signal dict for combined learning
        predictive_signal = {
            'signal': decision.action,
            'trade_direction': trade_direction,
            'subtitle': decision.main_reason,
            'action': decision.conclusion_action,
            'interpretation': decision.whale_story,
            'confidence': direction_confidence,
            'oi_story': decision.oi_story,
            'whale_story': decision.whale_story,
            'position_story': decision.position_story,
            'conclusion': decision.conclusion,
            'warnings': decision.warnings,
        }
        
        return PredictiveScore(
            direction=direction,
            direction_confidence=direction_confidence,
            direction_score=decision.direction_score,
            direction_reason=direction_reason,
            squeeze_potential=squeeze_potential,
            divergence_pct=divergence,
            squeeze_score=decision.squeeze_score,
            squeeze_reason=squeeze_reason,
            entry_timing=entry_timing,
            ta_score=ta_score,
            timing_score=decision.entry_score,
            timing_reason=timing_reason,
            move_position=move_position,
            move_position_pct=position_pct,
            final_score=final_score,
            final_action=final_action,
            final_summary=final_summary,
            trade_mode=trade_mode,
            timeframe=timeframe,
            trade_direction=trade_direction,
            predictive_signal=predictive_signal,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FALLBACK: If MASTER_RULES not available, use basic scoring
    # (This should rarely happen in production)
    # ═══════════════════════════════════════════════════════════════════════════
    
    divergence = whale_pct - retail_pct
    
    # Basic direction based on whale %
    if whale_pct >= 65:
        direction = "BULLISH"
        direction_confidence = "HIGH"
        direction_score = 32
    elif whale_pct >= 55:
        direction = "LEAN_BULLISH"
        direction_confidence = "MEDIUM"
        direction_score = 24
    elif whale_pct <= 35:
        direction = "BEARISH"
        direction_confidence = "HIGH"
        direction_score = 32
    elif whale_pct <= 45:
        direction = "LEAN_BEARISH"
        direction_confidence = "MEDIUM"
        direction_score = 24
    else:
        direction = "NEUTRAL"
        direction_confidence = "LOW"
        direction_score = 15
    
    direction_reason = f"Whales {whale_pct:.0f}% long"
    
    # Basic squeeze score
    if divergence > 15:
        squeeze_potential = "HIGH"
        squeeze_score = 25
    elif divergence > 5:
        squeeze_potential = "MEDIUM"
        squeeze_score = 18
    elif divergence < -5:
        squeeze_potential = "CONFLICT"
        squeeze_score = 5
    else:
        squeeze_potential = "NONE"
        squeeze_score = 10
    
    squeeze_reason = f"Divergence: {divergence:+.0f}%"
    
    # Basic position
    if position_pct <= 35:
        move_position = "EARLY"
    elif position_pct >= 65:
        move_position = "LATE"
    else:
        move_position = "MIDDLE"
    
    # Basic timing score
    timing_score = min(20, ta_score // 5) + (10 if move_position == "EARLY" else 5 if move_position == "MIDDLE" else 0)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # PREDICTIVE OVERRIDE: Don't wait for TA when whales are heavily positioned EARLY
    # ═══════════════════════════════════════════════════════════════════════════════
    # The whole point of PREDICTIVE analysis is to enter BEFORE the move.
    # If whales are 70%+ AND we're EARLY in range, TA being weak is EXPECTED
    # because the move hasn't started yet. That's the IDEAL predictive setup!
    #
    # Waiting for TA > 60 means waiting for confirmation = NOT predictive anymore
    # We'd be entering AFTER the move started = becoming exit liquidity
    # ═══════════════════════════════════════════════════════════════════════════════
    
    predictive_override = False
    if move_position == "EARLY" and whale_pct >= 70:
        # This is the IDEAL predictive setup:
        # - Whales heavily positioned (leading indicator)
        # - Price at bottom of range (early entry)
        # - Weak TA is EXPECTED (move hasn't started)
        predictive_override = True
        timing_score = max(timing_score, 22)  # Boost timing score
        entry_timing = "NOW"
        timing_reason = f"🎯 PREDICTIVE: Whales {whale_pct:.0f}% + EARLY position - TA will follow!"
    elif move_position == "EARLY" and whale_pct >= 60:
        # Strong setup but not as extreme
        predictive_override = True
        timing_score = max(timing_score, 18)
        entry_timing = "NOW" if ta_score >= 30 else "SOON"
        timing_reason = f"🎯 PREDICTIVE: Whales {whale_pct:.0f}% + EARLY - Good setup"
    elif ta_score >= 60 and move_position in ["EARLY", "MIDDLE"]:
        entry_timing = "NOW"
        timing_reason = f"Good entry: {move_position} ({position_pct:.0f}%) + TA {ta_score}"
    elif ta_score >= 40:
        entry_timing = "SOON"
        timing_reason = f"TA: {ta_score} | Position: {move_position} ({position_pct:.0f}%)"
    else:
        entry_timing = "WAIT"
        timing_reason = f"TA: {ta_score} | Position: {move_position} ({position_pct:.0f}%)"
    
    # Final score and action
    final_score = direction_score + squeeze_score + timing_score
    
    if direction in ["BULLISH", "LEAN_BULLISH"] and divergence >= 0:
        trade_direction = "LONG"
        final_action = "✅ BUY" if final_score >= 60 else "⏳ WAIT"
    elif direction in ["BEARISH", "LEAN_BEARISH"]:
        trade_direction = "SHORT"
        final_action = "🔴 SELL" if final_score >= 60 else "⏳ WAIT"
    else:
        trade_direction = "WAIT"
        final_action = "⏳ WAIT"
    
    final_summary = f"{direction} ({direction_confidence}) - Score: {final_score}"
    
    return PredictiveScore(
        direction=direction,
        direction_confidence=direction_confidence,
        direction_score=direction_score,
        direction_reason=direction_reason,
        squeeze_potential=squeeze_potential,
        divergence_pct=divergence,
        squeeze_score=squeeze_score,
        squeeze_reason=squeeze_reason,
        entry_timing=entry_timing,
        ta_score=ta_score,
        timing_score=timing_score,
        timing_reason=timing_reason,
        move_position=move_position,
        move_position_pct=position_pct,
        final_score=final_score,
        final_action=final_action,
        final_summary=final_summary,
        trade_mode=trade_mode,
        timeframe=timeframe,
        trade_direction=trade_direction,
        predictive_signal=None,
    )
    
# ═══════════════════════════════════════════════════════════════════════════════
# SCORING RESULT (Output of this engine)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoringResult:
    """
    The COMPLETE output of the scoring engine.
    All scores are derived consistently from the same inputs.
    """
    # Scores (0-100)
    ta_score: int
    inst_score: int  # DERIVED from whale verdict, weighted by mode!
    combined_score: int
    
    # Classifications
    ta_strength: str
    whale_verdict: str
    whale_confidence: str
    
    # Trade Context (NEW!)
    trade_mode: str      # 'Scalp', 'DayTrade', 'Swing', 'Investment'
    timeframe: str       # '1m', '5m', '15m', '1h', '4h', '1d', '1w'
    ta_weight: float     # How much TA contributed (0-1)
    whale_weight: float  # How much whale data contributed (0-1)
    
    # Action
    final_action: str
    position_size: str  # "100%", "75%", "50%", "25%", "0%"
    
    # Alignment
    alignment: str  # "ALIGNED", "NEUTRAL", "CONFLICT"
    alignment_note: str
    
    # UI
    confidence_level: str  # "HIGH", "GOOD", "MODERATE", "LOW"
    confidence_color: str  # Hex color
    
    # Explanation
    reasoning: str
    warnings: List[str] = field(default_factory=list)
    factors: List[Tuple[str, int]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING RULES TABLE
# Format: (ta_strength, whale_verdict, whale_confidence) -> (combined_range, inst_score, action, size, reasoning)
# ═══════════════════════════════════════════════════════════════════════════════

SCORING_RULES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # EXCEPTIONAL TA (85-100)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Exceptional + Strong Bullish
    (TAStrength.EXCEPTIONAL, WhaleVerdict.STRONG_BULLISH, Confidence.HIGH): {
        'combined': (95, 100), 'inst': 90, 'action': FinalAction.STRONG_BUY, 'size': '100%',
        'reasoning': 'Perfect setup: A+ technicals with strong whale confirmation',
        'warnings': []
    },
    (TAStrength.EXCEPTIONAL, WhaleVerdict.BULLISH, Confidence.HIGH): {
        'combined': (90, 96), 'inst': 80, 'action': FinalAction.STRONG_BUY, 'size': '100%',
        'reasoning': 'Excellent setup with whale support',
        'warnings': []
    },
    (TAStrength.EXCEPTIONAL, WhaleVerdict.BULLISH, Confidence.MEDIUM): {
        'combined': (85, 92), 'inst': 70, 'action': FinalAction.BUY, 'size': '100%',
        'reasoning': 'Strong setup with moderate whale support',
        'warnings': []
    },
    (TAStrength.EXCEPTIONAL, WhaleVerdict.LEAN_BULLISH, Confidence.MEDIUM): {
        'combined': (82, 88), 'inst': 60, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Excellent TA with slight whale bias long',
        'warnings': []
    },
    
    # Exceptional + Neutral/Wait
    (TAStrength.EXCEPTIONAL, WhaleVerdict.NEUTRAL, Confidence.LOW): {
        'combined': (78, 85), 'inst': 50, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Excellent TA, whales neutral - trade the chart',
        'warnings': ['Whale data neutral - use tight stops']
    },
    (TAStrength.EXCEPTIONAL, WhaleVerdict.WAIT, Confidence.LOW): {
        'combined': (72, 80), 'inst': 50, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Strong TA but whale says wait - proceed with caution',
        'warnings': ['⚠️ Whale verdict: WAIT - reduce size, tight stops']
    },
    
    # Exceptional + Bearish (CONFLICT)
    (TAStrength.EXCEPTIONAL, WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM): {
        'combined': (60, 70), 'inst': 40, 'action': FinalAction.CAUTIOUS_BUY, 'size': '25%',
        'reasoning': '⚠️ Strong TA but whales leaning short',
        'warnings': ['Smart money slightly bearish - reduced conviction']
    },
    (TAStrength.EXCEPTIONAL, WhaleVerdict.BEARISH, Confidence.HIGH): {
        'combined': (50, 60), 'inst': 30, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': '🚨 CONFLICT: Great TA but whales selling!',
        'warnings': ['Smart money divergence - high risk', 'Wait for alignment']
    },
    (TAStrength.EXCEPTIONAL, WhaleVerdict.AVOID, Confidence.LOW): {
        'combined': (55, 65), 'inst': 45, 'action': FinalAction.CAUTIOUS_BUY, 'size': '25%',
        'reasoning': 'Strong TA but mixed whale signals',
        'warnings': ['Mixed institutional signals - reduce exposure']
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRONG TA (70-84)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Strong + Bullish
    (TAStrength.STRONG, WhaleVerdict.STRONG_BULLISH, Confidence.HIGH): {
        'combined': (88, 95), 'inst': 85, 'action': FinalAction.STRONG_BUY, 'size': '100%',
        'reasoning': 'Good setup with strong whale confirmation',
        'warnings': []
    },
    (TAStrength.STRONG, WhaleVerdict.BULLISH, Confidence.HIGH): {
        'combined': (82, 90), 'inst': 75, 'action': FinalAction.BUY, 'size': '100%',
        'reasoning': 'Solid setup with whale support',
        'warnings': []
    },
    (TAStrength.STRONG, WhaleVerdict.BULLISH, Confidence.MEDIUM): {
        'combined': (75, 84), 'inst': 65, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Good setup with moderate whale support',
        'warnings': []
    },
    (TAStrength.STRONG, WhaleVerdict.LEAN_BULLISH, Confidence.MEDIUM): {
        'combined': (72, 80), 'inst': 58, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Good setup with slight whale bias long',
        'warnings': []
    },
    (TAStrength.STRONG, WhaleVerdict.LEAN_BULLISH, Confidence.LOW): {
        'combined': (68, 76), 'inst': 55, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Good TA, weak bullish whale signal',
        'warnings': ['Low confidence whale signal']
    },
    
    # Strong + Neutral/Wait
    (TAStrength.STRONG, WhaleVerdict.NEUTRAL, Confidence.LOW): {
        'combined': (68, 76), 'inst': 50, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Good TA, neutral whales - trade the chart',
        'warnings': ['No whale confirmation - standard position']
    },
    (TAStrength.STRONG, WhaleVerdict.WAIT, Confidence.LOW): {
        'combined': (60, 70), 'inst': 50, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Good TA but whale data says wait',
        'warnings': ['⚠️ Whale verdict: WAIT - reduce size']
    },
    (TAStrength.STRONG, WhaleVerdict.WAIT, Confidence.MEDIUM): {
        'combined': (58, 68), 'inst': 48, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Good TA but whales uncertain',
        'warnings': ['⚠️ Mixed whale signals']
    },
    
    # Strong + Bearish
    (TAStrength.STRONG, WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM): {
        'combined': (55, 65), 'inst': 42, 'action': FinalAction.CAUTIOUS_BUY, 'size': '25%',
        'reasoning': 'Good TA but whales slightly short',
        'warnings': ['Reduced conviction - small size only']
    },
    (TAStrength.STRONG, WhaleVerdict.BEARISH, Confidence.HIGH): {
        'combined': (45, 55), 'inst': 30, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': '⚠️ CONFLICT: Good TA but whales selling',
        'warnings': ['🚨 Do NOT enter - wait for resolution']
    },
    (TAStrength.STRONG, WhaleVerdict.AVOID, Confidence.LOW): {
        'combined': (50, 60), 'inst': 45, 'action': FinalAction.WAIT, 'size': '25%',
        'reasoning': 'Good TA but avoid signal from whales',
        'warnings': ['Mixed signals - wait for clarity']
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MODERATE TA (55-69)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Moderate + Bullish (whale can save mediocre setup!)
    (TAStrength.MODERATE, WhaleVerdict.STRONG_BULLISH, Confidence.HIGH): {
        'combined': (75, 85), 'inst': 85, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Mediocre TA but strong whale buying - follow smart money',
        'warnings': ['Whale accumulation may precede move']
    },
    (TAStrength.MODERATE, WhaleVerdict.BULLISH, Confidence.HIGH): {
        'combined': (70, 78), 'inst': 75, 'action': FinalAction.BUY, 'size': '75%',
        'reasoning': 'Average setup boosted by whale buying',
        'warnings': []
    },
    (TAStrength.MODERATE, WhaleVerdict.BULLISH, Confidence.MEDIUM): {
        'combined': (62, 72), 'inst': 65, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Average setup with some whale support',
        'warnings': []
    },
    (TAStrength.MODERATE, WhaleVerdict.LEAN_BULLISH, Confidence.MEDIUM): {
        'combined': (58, 68), 'inst': 58, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Moderate TA with slight whale support',
        'warnings': []
    },
    
    # Moderate + Neutral/Wait
    (TAStrength.MODERATE, WhaleVerdict.NEUTRAL, Confidence.LOW): {
        'combined': (55, 65), 'inst': 50, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': 'Average setup, no whale edge',
        'warnings': ['Standard risk management required']
    },
    (TAStrength.MODERATE, WhaleVerdict.WAIT, Confidence.LOW): {
        'combined': (48, 58), 'inst': 50, 'action': FinalAction.WAIT, 'size': '25%',
        'reasoning': 'Mediocre TA + whale says wait = skip',
        'warnings': ['Not enough edge - wait for better setup']
    },
    
    # Moderate + Bearish
    (TAStrength.MODERATE, WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM): {
        'combined': (42, 52), 'inst': 40, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': 'Average TA + bearish whale = no trade',
        'warnings': ['Skip - no edge']
    },
    (TAStrength.MODERATE, WhaleVerdict.BEARISH, Confidence.HIGH): {
        'combined': (30, 42), 'inst': 25, 'action': FinalAction.AVOID, 'size': '0%',
        'reasoning': 'Weak TA + bearish whales = AVOID',
        'warnings': ['🚨 No edge - skip this trade']
    },
    (TAStrength.MODERATE, WhaleVerdict.AVOID, Confidence.LOW): {
        'combined': (40, 50), 'inst': 45, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': 'Moderate setup but avoid signal',
        'warnings': ['Wait for better opportunity']
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WEAK TA (40-54)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Weak + Strong Bullish (whale divergence - interesting!)
    (TAStrength.WEAK, WhaleVerdict.STRONG_BULLISH, Confidence.HIGH): {
        'combined': (62, 72), 'inst': 85, 'action': FinalAction.CAUTIOUS_BUY, 'size': '50%',
        'reasoning': '📊 DIVERGENCE: Weak chart but whales accumulating!',
        'warnings': ['Smart money may see what chart doesnt show', 'Small size, wide stops']
    },
    (TAStrength.WEAK, WhaleVerdict.BULLISH, Confidence.HIGH): {
        'combined': (55, 65), 'inst': 75, 'action': FinalAction.CAUTIOUS_BUY, 'size': '25%',
        'reasoning': 'Weak TA but whale buying - watch for breakout',
        'warnings': ['Wait for TA confirmation before adding']
    },
    (TAStrength.WEAK, WhaleVerdict.BULLISH, Confidence.MEDIUM): {
        'combined': (50, 58), 'inst': 60, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': 'Weak TA, moderate whale interest',
        'warnings': ['Wait for TA improvement']
    },
    
    # Weak + Neutral/Wait
    (TAStrength.WEAK, WhaleVerdict.NEUTRAL, Confidence.LOW): {
        'combined': (40, 50), 'inst': 50, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': 'Weak setup + no whale edge = no trade',
        'warnings': ['Wait for better opportunity']
    },
    (TAStrength.WEAK, WhaleVerdict.WAIT, Confidence.LOW): {
        'combined': (35, 45), 'inst': 50, 'action': FinalAction.AVOID, 'size': '0%',
        'reasoning': 'No edge from TA or whales',
        'warnings': ['Skip - look elsewhere']
    },
    
    # Weak + Bearish
    (TAStrength.WEAK, WhaleVerdict.BEARISH, Confidence.HIGH): {
        'combined': (20, 30), 'inst': 20, 'action': FinalAction.STRONG_SELL, 'size': '100%',
        'reasoning': 'Weak TA + whale selling = SHORT opportunity',
        'warnings': ['Consider short position if allowed']
    },
    (TAStrength.WEAK, WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM): {
        'combined': (30, 40), 'inst': 35, 'action': FinalAction.AVOID, 'size': '0%',
        'reasoning': 'Weak all around - skip',
        'warnings': []
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POOR TA (0-39)
    # ═══════════════════════════════════════════════════════════════════════════
    
    (TAStrength.POOR, WhaleVerdict.STRONG_BULLISH, Confidence.HIGH): {
        'combined': (50, 60), 'inst': 80, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': 'Bad chart but whale accumulation - wait for structure',
        'warnings': ['Dont chase - wait for TA to catch up']
    },
    (TAStrength.POOR, WhaleVerdict.BULLISH, Confidence.HIGH): {
        'combined': (45, 55), 'inst': 70, 'action': FinalAction.WAIT, 'size': '0%',
        'reasoning': 'Poor TA, whale buying - wait for confirmation',
        'warnings': ['Need technical confirmation']
    },
    (TAStrength.POOR, WhaleVerdict.NEUTRAL, Confidence.LOW): {
        'combined': (25, 35), 'inst': 50, 'action': FinalAction.AVOID, 'size': '0%',
        'reasoning': 'No setup, no edge',
        'warnings': []
    },
    (TAStrength.POOR, WhaleVerdict.WAIT, Confidence.LOW): {
        'combined': (20, 30), 'inst': 50, 'action': FinalAction.AVOID, 'size': '0%',
        'reasoning': 'Nothing here - skip',
        'warnings': []
    },
    (TAStrength.POOR, WhaleVerdict.BEARISH, Confidence.HIGH): {
        'combined': (10, 20), 'inst': 15, 'action': FinalAction.STRONG_SELL, 'size': '100%',
        'reasoning': 'Everything bearish - strong short signal',
        'warnings': []
    },
}


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


def parse_whale_verdict(unified_verdict: dict) -> Tuple[WhaleVerdict, Confidence]:
    """
    Parse the unified_verdict dict into WhaleVerdict and Confidence enums.
    
    EXPLICIT MAPPING - No string matching shortcuts!
    Each scenario is mapped to its correct verdict based on what it actually means.
    
    unified_verdict comes from get_unified_verdict() in education.py and contains:
    - unified_action: 'LONG', 'SHORT', 'WAIT', etc.
    - confidence: 'HIGH', 'MEDIUM', 'LOW'
    - scenario: dict with 'name', 'recommendation', etc.
    - scenario_key: the scenario identifier
    - institutional_bias: 'BULLISH', 'BEARISH', 'NEUTRAL'
    """
    if not unified_verdict:
        return WhaleVerdict.NEUTRAL, Confidence.LOW
    
    # Get confidence
    conf_str = str(unified_verdict.get('confidence', 'LOW')).upper()
    confidence = Confidence.HIGH if conf_str == 'HIGH' else Confidence.MEDIUM if conf_str == 'MEDIUM' else Confidence.LOW
    
    # Get scenario key - this is the authoritative source
    scenario_key = unified_verdict.get('scenario_key', '')
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # EXPLICIT SCENARIO MAPPING - ALL SCENARIOS FULLY DEFINED
    # ═══════════════════════════════════════════════════════════════════════════════
    
    SCENARIO_TO_VERDICT = {
        # ─────────────────────────────────────────────────────────────────────────
        # HIGH CONFIDENCE BULLISH SCENARIOS
        # ─────────────────────────────────────────────────────────────────────────
        
        'new_longs_whale_bullish': (WhaleVerdict.BULLISH, Confidence.HIGH),
        # New longs entering + whales bullish = STRONG BUY signal
        # inst_score: 75-85
        
        'short_squeeze_setup': (WhaleVerdict.STRONG_BULLISH, Confidence.HIGH),
        # Shorts vulnerable to squeeze = VERY BULLISH
        # inst_score: 85-90
        
        # ─────────────────────────────────────────────────────────────────────────
        # HIGH CONFIDENCE BEARISH SCENARIOS
        # ─────────────────────────────────────────────────────────────────────────
        
        'new_shorts_whale_bearish': (WhaleVerdict.BEARISH, Confidence.HIGH),
        # New shorts entering + whales bearish = STRONG SELL signal
        # inst_score: 15-25
        
        'long_squeeze_setup': (WhaleVerdict.STRONG_BEARISH, Confidence.HIGH),
        # Longs vulnerable to squeeze = VERY BEARISH
        # inst_score: 10-20
        
        # ─────────────────────────────────────────────────────────────────────────
        # MEDIUM CONFIDENCE SCENARIOS
        # ─────────────────────────────────────────────────────────────────────────
        
        'new_longs_neutral': (WhaleVerdict.LEAN_BULLISH, Confidence.MEDIUM),
        # New longs but whales neutral = cautious bullish
        # inst_score: 55-60
        
        'new_shorts_neutral': (WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM),
        # New shorts but whales neutral = cautious bearish
        # inst_score: 40-45
        
        'long_liquidation_whale_bullish': (WhaleVerdict.LEAN_BULLISH, Confidence.MEDIUM),
        # Longs liquidating BUT whales buying = potential opportunity
        # Whales see value, could be accumulation
        # inst_score: 55-62
        
        'short_covering_whale_bearish': (WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM),
        # Shorts covering BUT whales still bearish = temporary bounce
        # Don't trust the rally
        # inst_score: 38-45
        
        # ─────────────────────────────────────────────────────────────────────────
        # LOW CONFIDENCE / WAIT SCENARIOS
        # ─────────────────────────────────────────────────────────────────────────
        
        'short_covering_neutral': (WhaleVerdict.WAIT, Confidence.LOW),
        # "Short Covering Rally" - temporary bounce, no whale support
        # Rally may be temporary - don't chase
        # inst_score: 48-52
        
        'long_liquidation_neutral': (WhaleVerdict.LEAN_BEARISH, Confidence.LOW),
        # "Long Liquidation Cascade" - BEARISH! Longs getting liquidated
        # This is NOT bullish just because it has "Long" in the name!
        # Forced selling in progress
        # inst_score: 35-42
        
        # ─────────────────────────────────────────────────────────────────────────
        # WHALE BIAS WITH NEUTRAL OI (New scenarios!)
        # ─────────────────────────────────────────────────────────────────────────
        
        'whale_bullish_oi_neutral': (WhaleVerdict.LEAN_BULLISH, Confidence.MEDIUM),
        # Whales positioned bullish but OI not moving
        # Cautious bullish - accumulation may be happening
        # inst_score: 58-62
        
        'whale_bearish_oi_neutral': (WhaleVerdict.LEAN_BEARISH, Confidence.MEDIUM),
        # Whales positioned bearish but OI not moving  
        # Cautious bearish - distribution may be happening
        # inst_score: 38-42
        
        'no_edge': (WhaleVerdict.NEUTRAL, Confidence.LOW),
        # No clear edge either way
        # inst_score: 50
        
        'conflicting_signals': (WhaleVerdict.AVOID, Confidence.LOW),
        # Tech and institutional conflict - stay out
        # inst_score: 45-50
    }
    
    # Check if we have a direct scenario mapping
    if scenario_key in SCENARIO_TO_VERDICT:
        return SCENARIO_TO_VERDICT[scenario_key]
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # FALLBACK: Parse action if no scenario key
    # ═══════════════════════════════════════════════════════════════════════════════
    
    action = str(unified_verdict.get('unified_action', 
                 unified_verdict.get('action', 'WAIT'))).upper()
    inst_bias = str(unified_verdict.get('institutional_bias', '')).upper()
    
    # Direct action mapping
    if 'STRONG' in action and ('LONG' in action or 'BUY' in action):
        return WhaleVerdict.STRONG_BULLISH, confidence
    elif 'STRONG' in action and ('SHORT' in action or 'SELL' in action):
        return WhaleVerdict.STRONG_BEARISH, confidence
    elif 'LONG' in action or 'BUY' in action:
        if confidence == Confidence.HIGH:
            return WhaleVerdict.BULLISH, confidence
        return WhaleVerdict.LEAN_BULLISH, confidence
    elif 'SHORT' in action or 'SELL' in action:
        if confidence == Confidence.HIGH:
            return WhaleVerdict.BEARISH, confidence
        return WhaleVerdict.LEAN_BEARISH, confidence
    elif 'AVOID' in action:
        return WhaleVerdict.AVOID, confidence
    elif 'WAIT' in action:
        return WhaleVerdict.WAIT, confidence
    
    # Final fallback: use institutional bias
    if inst_bias == 'BULLISH':
        return WhaleVerdict.LEAN_BULLISH if confidence != Confidence.HIGH else WhaleVerdict.BULLISH, confidence
    elif inst_bias == 'BEARISH':
        return WhaleVerdict.LEAN_BEARISH if confidence != Confidence.HIGH else WhaleVerdict.BEARISH, confidence
    
    return WhaleVerdict.NEUTRAL, confidence


def interpolate_score(ta_score: int, score_range: Tuple[int, int], ta_strength: TAStrength) -> int:
    """Interpolate combined score within range based on where TA falls in its category"""
    ta_ranges = {
        TAStrength.EXCEPTIONAL: (85, 100),
        TAStrength.STRONG: (70, 84),
        TAStrength.MODERATE: (55, 69),
        TAStrength.WEAK: (40, 54),
        TAStrength.POOR: (0, 39),
    }
    ta_min, ta_max = ta_ranges[ta_strength]
    
    if ta_max > ta_min:
        ta_pct = (ta_score - ta_min) / (ta_max - ta_min)
    else:
        ta_pct = 0.5
    
    min_score, max_score = score_range
    return int(min_score + (max_score - min_score) * ta_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION - THE SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_unified_score(
    ta_score: int,
    unified_verdict: dict = None,
    trade_mode: str = 'DayTrade',
    timeframe: str = '1h',
    ta_factors: List[Tuple[str, int]] = None,
    oi_change: float = None,
    whale_pct: float = None
) -> ScoringResult:
    """
    THE SINGLE SOURCE OF TRUTH FOR ALL SCORES.
    
    ═══════════════════════════════════════════════════════════════════════════════
    NOW TIMEFRAME-AWARE! Different trade modes weight TA vs Whale differently.
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        ta_score: Technical analysis score (0-100)
        unified_verdict: Dict from get_unified_verdict() with action, confidence, scenario
        trade_mode: 'Scalp', 'DayTrade', 'Swing', 'Investment'
        timeframe: '1m', '5m', '15m', '1h', '4h', '1d', '1w'
        ta_factors: Optional list of (reason, points) explaining TA score
        oi_change: OI change % (24h) - used to check if whale data is significant
        whale_pct: Top trader long % - used to check against mode threshold
    
    Returns:
        ScoringResult with all scores, actions, and explanations
    
    ═══════════════════════════════════════════════════════════════════════════════
    WEIGHTING BY TRADE MODE:
    
    Mode        │ TA Weight │ Whale Weight │ Min OI │ Min Whale% │ Why
    ────────────┼───────────┼──────────────┼────────┼────────────┼─────────────────
    Scalp       │    90%    │     10%      │  2.0%  │    60%     │ 24h data = noise
    DayTrade    │    75%    │     25%      │  1.0%  │    57%     │ Some relevance
    Swing       │    50%    │     50%      │  0.5%  │    55%     │ Highly relevant
    Investment  │    40%    │     60%      │  0.3%  │    54%     │ Critical signal
    ═══════════════════════════════════════════════════════════════════════════════
    """
    ta_score = max(0, min(100, ta_score))
    ta_factors = ta_factors or []
    
    # Get mode configuration
    mode_config = get_mode_config(trade_mode)
    
    # Get classifications
    ta_strength = get_ta_strength(ta_score)
    whale_verdict, whale_confidence = parse_whale_verdict(unified_verdict)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIMEFRAME-AWARE WHALE DATA VALIDITY CHECK
    # Is the whale data significant enough for this trade mode?
    # ═══════════════════════════════════════════════════════════════════════════
    
    whale_data_valid = True
    whale_data_warning = None
    
    # Check if OI change meets threshold for this mode
    if oi_change is not None:
        if abs(oi_change) < mode_config.min_oi_threshold:
            whale_data_valid = False
            whale_data_warning = f"OI change ({oi_change:+.1f}%) too small for {trade_mode} (needs ≥{mode_config.min_oi_threshold}%)"
    
    # Check if whale positioning meets threshold for this mode
    if whale_pct is not None:
        if 50 - mode_config.min_whale_pct < whale_pct < mode_config.min_whale_pct:
            # Whale pct is in the neutral zone for this mode
            if whale_verdict in [WhaleVerdict.LEAN_BULLISH, WhaleVerdict.LEAN_BEARISH]:
                whale_data_valid = False
                whale_data_warning = f"Whale {whale_pct:.0f}% not strong enough for {trade_mode} (needs ≥{mode_config.min_whale_pct}%)"
    
    # If whale data not valid for this mode, downgrade verdict
    original_verdict = whale_verdict
    if not whale_data_valid:
        if whale_verdict in [WhaleVerdict.LEAN_BULLISH, WhaleVerdict.LEAN_BEARISH]:
            whale_verdict = WhaleVerdict.NEUTRAL
            whale_confidence = Confidence.LOW
        elif whale_verdict in [WhaleVerdict.BULLISH, WhaleVerdict.BEARISH]:
            whale_verdict = WhaleVerdict.LEAN_BULLISH if whale_verdict == WhaleVerdict.BULLISH else WhaleVerdict.LEAN_BEARISH
            whale_confidence = Confidence.LOW
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIND MATCHING RULE
    # ═══════════════════════════════════════════════════════════════════════════
    
    rule_key = (ta_strength, whale_verdict, whale_confidence)
    rule = SCORING_RULES.get(rule_key)
    
    # If no exact match, try with different confidence levels
    if not rule:
        for conf in [Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW]:
            rule = SCORING_RULES.get((ta_strength, whale_verdict, conf))
            if rule:
                break
    
    # If still no match, try similar verdicts
    if not rule:
        similar_verdicts = {
            WhaleVerdict.STRONG_BULLISH: [WhaleVerdict.BULLISH],
            WhaleVerdict.BULLISH: [WhaleVerdict.STRONG_BULLISH, WhaleVerdict.LEAN_BULLISH],
            WhaleVerdict.LEAN_BULLISH: [WhaleVerdict.BULLISH, WhaleVerdict.NEUTRAL],
            WhaleVerdict.NEUTRAL: [WhaleVerdict.WAIT, WhaleVerdict.LEAN_BULLISH],
            WhaleVerdict.WAIT: [WhaleVerdict.NEUTRAL, WhaleVerdict.AVOID],
            WhaleVerdict.AVOID: [WhaleVerdict.WAIT, WhaleVerdict.NEUTRAL],
            WhaleVerdict.LEAN_BEARISH: [WhaleVerdict.BEARISH, WhaleVerdict.NEUTRAL],
            WhaleVerdict.BEARISH: [WhaleVerdict.STRONG_BEARISH, WhaleVerdict.LEAN_BEARISH],
            WhaleVerdict.STRONG_BEARISH: [WhaleVerdict.BEARISH],
        }
        
        for similar in similar_verdicts.get(whale_verdict, []):
            for conf in [whale_confidence, Confidence.MEDIUM, Confidence.LOW]:
                rule = SCORING_RULES.get((ta_strength, similar, conf))
                if rule:
                    break
            if rule:
                break
    
    # Final fallback - calculate scores directly
    if not rule:
        # Default calculation
        inst_score = 50  # Neutral
        if whale_verdict in [WhaleVerdict.STRONG_BULLISH]:
            inst_score = 85
        elif whale_verdict in [WhaleVerdict.BULLISH]:
            inst_score = 75
        elif whale_verdict in [WhaleVerdict.LEAN_BULLISH]:
            inst_score = 58
        elif whale_verdict in [WhaleVerdict.LEAN_BEARISH]:
            inst_score = 42
        elif whale_verdict in [WhaleVerdict.BEARISH]:
            inst_score = 25
        elif whale_verdict in [WhaleVerdict.STRONG_BEARISH]:
            inst_score = 15
        elif whale_verdict in [WhaleVerdict.WAIT, WhaleVerdict.AVOID]:
            inst_score = 50
        
        # Calculate combined - USE MODE WEIGHTS!
        combined_score = int(ta_score * mode_config.ta_weight + inst_score * mode_config.whale_weight)
        
        # Determine action
        if combined_score >= 80:
            action = FinalAction.STRONG_BUY
            size = '100%'
        elif combined_score >= 70:
            action = FinalAction.BUY
            size = '75%'
        elif combined_score >= 60:
            action = FinalAction.CAUTIOUS_BUY
            size = '50%'
        elif combined_score >= 50:
            action = FinalAction.WAIT
            size = '25%'
        else:
            action = FinalAction.AVOID
            size = '0%'
        
        rule = {
            'combined': (combined_score, combined_score),
            'inst': inst_score,
            'action': action,
            'size': size,
            'reasoning': f'Fallback: {ta_strength.value} TA + {whale_verdict.value} whale',
            'warnings': ['No exact rule matched - using fallback calculation']
        }
    
    # Extract rule values
    base_combined = interpolate_score(ta_score, rule['combined'], ta_strength)
    inst_score = rule['inst']
    
    # ═══════════════════════════════════════════════════════════════════════════
    # APPLY TIMEFRAME-AWARE WEIGHTING TO COMBINED SCORE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Recalculate combined score using mode-specific weights
    # This ensures proper weighting regardless of what the rule says
    combined_score = int(ta_score * mode_config.ta_weight + inst_score * mode_config.whale_weight)
    combined_score = max(0, min(100, combined_score))
    
    final_action = rule['action'] if isinstance(rule['action'], FinalAction) else rule['action']
    position_size = rule['size']
    reasoning = rule['reasoning']
    warnings = list(rule.get('warnings', []))
    
    # Add whale data validity warning if applicable
    if whale_data_warning:
        warnings.insert(0, f"⚠️ {whale_data_warning}")
        # Also add mode context
        warnings.append(f"📊 {trade_mode} mode: {int(mode_config.ta_weight*100)}% TA, {int(mode_config.whale_weight*100)}% Whale weighting")
    
    # If whale data was downgraded, adjust action
    if not whale_data_valid and original_verdict != whale_verdict:
        warnings.append(f"Whale verdict downgraded: {original_verdict.value} → {whale_verdict.value}")
        # Recalculate action based on new combined score
        if combined_score >= 80:
            final_action = FinalAction.STRONG_BUY
            position_size = '100%'
        elif combined_score >= 70:
            final_action = FinalAction.BUY
            position_size = '75%'
        elif combined_score >= 60:
            final_action = FinalAction.CAUTIOUS_BUY
            position_size = '50%'
        elif combined_score >= 50:
            final_action = FinalAction.WAIT
            position_size = '25%'
        else:
            final_action = FinalAction.AVOID
            position_size = '0%'
    
    # Determine alignment
    ta_bullish = ta_score >= 60
    inst_bullish = inst_score >= 55
    inst_bearish = inst_score <= 45
    
    if ta_bullish and inst_bullish:
        alignment = "ALIGNED ✅"
        alignment_note = "Both TA and Smart Money agree"
    elif ta_bullish and inst_bearish:
        alignment = "CONFLICT ⚠️"
        alignment_note = "TA bullish but Smart Money bearish"
    elif not ta_bullish and inst_bullish:
        alignment = "DIVERGENCE 📊"
        alignment_note = "Smart Money bullish despite weak TA"
    elif whale_verdict in [WhaleVerdict.WAIT, WhaleVerdict.NEUTRAL]:
        alignment = "NEUTRAL ⚪"
        alignment_note = "Whale data neutral - trade on TA"
    else:
        alignment = "NEUTRAL ⚪"
        alignment_note = "No strong conviction either way"
    
    # Add mode info to alignment note
    alignment_note += f" | {trade_mode} ({timeframe})"
    
    # Determine confidence level and color for UI
    if combined_score >= 80:
        confidence_level = "HIGH"
        confidence_color = "#00ff88"
    elif combined_score >= 65:
        confidence_level = "GOOD"
        confidence_color = "#00d4aa"
    elif combined_score >= 50:
        confidence_level = "MODERATE"
        confidence_color = "#ffcc00"
    else:
        confidence_level = "LOW"
        confidence_color = "#ff9500"
    
    # Build factors list
    factors = list(ta_factors)
    factors.append((f"Whale: {whale_verdict.value} ({whale_confidence.value})", inst_score - 50))
    factors.append((f"Mode: {trade_mode} ({int(mode_config.ta_weight*100)}% TA / {int(mode_config.whale_weight*100)}% Whale)", 0))
    
    return ScoringResult(
        ta_score=ta_score,
        inst_score=inst_score,
        combined_score=combined_score,
        ta_strength=ta_strength.value,
        whale_verdict=whale_verdict.value,
        whale_confidence=whale_confidence.value,
        trade_mode=trade_mode,
        timeframe=timeframe,
        ta_weight=mode_config.ta_weight,
        whale_weight=mode_config.whale_weight,
        final_action=final_action.value if isinstance(final_action, FinalAction) else str(final_action),
        position_size=position_size,
        alignment=alignment,
        alignment_note=alignment_note,
        confidence_level=confidence_level,
        confidence_color=confidence_color,
        reasoning=reasoning,
        warnings=warnings,
        factors=factors
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION - For backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_scores(
    ta_score: int, 
    unified_verdict: dict = None,
    trade_mode: str = 'DayTrade',
    timeframe: str = '1h',
    oi_change: float = None,
    whale_pct: float = None
) -> dict:
    """
    Convenience function that returns a dict (for backward compatibility).
    
    NOW TIMEFRAME-AWARE! Pass trade_mode and timeframe for proper weighting.
    
    Args:
        ta_score: Technical analysis score (0-100)
        unified_verdict: Dict from get_unified_verdict()
        trade_mode: 'Scalp', 'DayTrade', 'Swing', 'Investment'
        timeframe: '1m', '5m', '15m', '1h', '4h', '1d', '1w'
        oi_change: OI change % (24h) - for validity check
        whale_pct: Top trader long % - for validity check
    
    Returns dict with:
        - ta_score, inst_score, combined_score
        - trade_mode, timeframe, ta_weight, whale_weight
        - confidence_level, confidence_color
        - final_action, position_size
        - alignment, alignment_note
        - reasoning, warnings
    """
    result = calculate_unified_score(
        ta_score, 
        unified_verdict,
        trade_mode=trade_mode,
        timeframe=timeframe,
        oi_change=oi_change,
        whale_pct=whale_pct
    )
    
    return {
        'ta_score': result.ta_score,
        'inst_score': result.inst_score,
        'combined_score': result.combined_score,
        'trade_mode': result.trade_mode,
        'timeframe': result.timeframe,
        'ta_weight': result.ta_weight,
        'whale_weight': result.whale_weight,
        'confidence_level': result.confidence_level,
        'confidence_color': result.confidence_color,
        'action_hint': f"{result.position_size} position",
        'alignment': result.alignment,
        'alignment_note': result.alignment_note,
        'final_action': result.final_action,
        'position_size': result.position_size,
        'reasoning': result.reasoning,
        'warnings': result.warnings,
        'ta_factors': result.factors,
        'inst_factors': [(f"Whale verdict: {result.whale_verdict}", 0)],
        'whale_verdict': result.whale_verdict,
        'whale_confidence': result.whale_confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("UNIFIED SCORING ENGINE - TIMEFRAME-AWARE TESTS")
    print("=" * 80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 1: TRADE MODE WEIGHTING
    # Same data, different modes = different scores
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 80)
    print("TEST 1: TRADE MODE WEIGHTING")
    print("Same data (TA=60, Whale Bullish) → Different modes = Different scores")
    print("═" * 80)
    
    verdict = {'scenario_key': 'whale_bullish_oi_neutral', 'confidence': 'MEDIUM'}
    
    for mode in ['Scalp', 'DayTrade', 'Swing', 'Investment']:
        config = get_mode_config(mode)
        result = calculate_unified_score(
            ta_score=60, 
            unified_verdict=verdict,
            trade_mode=mode,
            timeframe='15m'
        )
        print(f"\n{mode}:")
        print(f"  Weights: {int(config.ta_weight*100)}% TA / {int(config.whale_weight*100)}% Whale")
        print(f"  Combined: {result.combined_score}")
        print(f"  Action: {result.final_action}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 2: OI THRESHOLD BY MODE
    # Small OI (0.2%) - valid for Swing, invalid for Scalp
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 80)
    print("TEST 2: OI THRESHOLD BY MODE")
    print("OI = +0.2% (your case!) → Different modes treat differently")
    print("═" * 80)
    
    for mode in ['Scalp', 'DayTrade', 'Swing', 'Investment']:
        config = get_mode_config(mode)
        result = calculate_unified_score(
            ta_score=40, 
            unified_verdict=verdict,
            trade_mode=mode,
            timeframe='15m',
            oi_change=0.2,
            whale_pct=55
        )
        print(f"\n{mode} (min OI: {config.min_oi_threshold}%):")
        print(f"  Combined: {result.combined_score}")
        print(f"  Action: {result.final_action}")
        if result.warnings:
            print(f"  Warnings: {result.warnings[0][:60]}...")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST 3: YOUR EXACT CASE
    # TA=40, OI=+0.2%, Whale=55%, Mode=DayTrade (15m)
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "═" * 80)
    print("TEST 3: YOUR EXACT CASE")
    print("TA=40, OI=+0.2%, Whale=55%, 15m DayTrade")
    print("═" * 80)
    
    result = calculate_unified_score(
        ta_score=40,
        unified_verdict={'scenario_key': 'whale_bullish_oi_neutral', 'confidence': 'MEDIUM'},
        trade_mode='DayTrade',
        timeframe='15m',
        oi_change=0.2,
        whale_pct=55
    )
    
    print(f"\nResult:")
    print(f"  TA Score:       {result.ta_score}")
    print(f"  Inst Score:     {result.inst_score}")
    print(f"  Combined Score: {result.combined_score}")
    print(f"  Trade Mode:     {result.trade_mode}")
    print(f"  Timeframe:      {result.timeframe}")
    print(f"  Weights:        {int(result.ta_weight*100)}% TA / {int(result.whale_weight*100)}% Whale")
    print(f"  Final Action:   {result.final_action}")
    print(f"  Position Size:  {result.position_size}")
    print(f"  Alignment:      {result.alignment}")
    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    • {w}")
    
    print("\n" + "═" * 80)
    print("EXPECTED: OI too small for DayTrade → Whale data downgraded → AVOID")
    print("═" * 80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEW 3-LAYER PREDICTIVE SCORING TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n\n")
    print("█" * 80)
    print("█  NEW 3-LAYER PREDICTIVE SCORING SYSTEM")
    print("█" * 80)
    
    test_cases_3layer = [
        # (OI%, Price%, Whale%, Retail%, TA, Description)
        (2.5, 3.6, 59, 50, 70, "Your screenshot: New longs + slight whale edge + good TA"),
        (2.5, 3.6, 70, 40, 70, "PERFECT SETUP: New longs + HIGH squeeze potential + good TA"),
        (2.5, 3.6, 70, 40, 35, "Good prediction but bad TA: Wait for pullback"),
        (-2.0, -3.0, 70, 40, 35, "Long liquidation: Whales bullish but longs getting wrecked"),
        (2.5, -3.0, 35, 60, 45, "New shorts + squeeze setup: BEARISH"),
        (0.2, 0.5, 55, 50, 70, "Weak OI + No divergence: TA trade only"),
        (-1.5, 2.0, 60, 55, 65, "Short covering rally: Weak bullish, may fade"),
    ]
    
    for oi, price, whale, retail, ta, desc in test_cases_3layer:
        print(f"\n{'─' * 70}")
        print(f"📊 {desc}")
        print(f"   OI: {oi:+.1f}% | Price: {price:+.1f}% | Whale: {whale}% | Retail: {retail}% | TA: {ta}")
        print(f"{'─' * 70}")
        
        result = calculate_predictive_score(
            oi_change=oi,
            price_change=price,
            whale_pct=whale,
            retail_pct=retail,
            ta_score=ta
        )
        
        print(f"""
  LAYER 1 - Direction:     {result.direction} ({result.direction_confidence})
            Score:         {result.direction_score}/40
            Reason:        {result.direction_reason}
            
  LAYER 2 - Squeeze:       {result.squeeze_potential}
            Divergence:    {result.divergence_pct:+.0f}% (Whale - Retail)
            Score:         {result.squeeze_score}/30
            Reason:        {result.squeeze_reason}
            
  LAYER 3 - Entry:         {result.entry_timing}
            TA Score:      {result.ta_score}
            Score:         {result.timing_score}/30
            Reason:        {result.timing_reason}
            
  ══════════════════════════════════════════════════════════════════
  FINAL SCORE:   {result.final_score}/100
  FINAL ACTION:  {result.final_action}
  SUMMARY:       {result.final_summary}
  ══════════════════════════════════════════════════════════════════
        """)
    
    print("\n" + "█" * 80)
    print("█  END OF 3-LAYER TESTS")
    print("█" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 HYBRID MODE HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_action_from_score(score: int, direction: str) -> str:
    """
    Get the appropriate action word based on score and direction.
    Used for Hybrid mode to ensure action matches blended score.
    
    Args:
        score: Final blended score (0-100)
        direction: 'LONG', 'SHORT', or 'WAIT'
    
    Returns:
        Action string like '🚀 STRONG BUY', '✅ BUY', etc.
    """
    if direction == 'LONG':
        if score >= 80:
            return '🚀 STRONG BUY'
        elif score >= 65:
            return '✅ BUY'
        elif score >= 50:
            return '📊 LEAN LONG'
        else:
            return '⏳ WAIT'
    elif direction == 'SHORT':
        if score >= 80:
            return '🔴 STRONG SELL'
        elif score >= 65:
            return '🔴 SELL'
        elif score >= 50:
            return '📊 LEAN SHORT'
        else:
            return '⏳ WAIT'
    else:
        return '⏳ WAIT'


def blend_hybrid_prediction(
    rule_score: int,
    rule_direction: str,  # 'BULLISH', 'BEARISH', 'NEUTRAL', etc.
    ml_direction: str,    # 'LONG', 'SHORT', 'WAIT'
    ml_confidence: float,
    position_pct: float = 50,  # Position in range 0-100
    whale_pct: float = 50,     # Whale long % (0-100)
    oi_change: float = 0,      # OI change 24h %
    historical_win_rate: float = None,  # Historical win rate for similar setups
) -> dict:
    """
    Blend rule-based and ML predictions for Hybrid mode.
    
    Args:
        rule_score: Score from rules engine (0-100)
        rule_direction: Direction from rules ('BULLISH', 'BEARISH', etc.)
        ml_direction: Direction from ML ('LONG', 'SHORT', 'WAIT')
        ml_confidence: ML confidence percentage (0-100)
        position_pct: Position in range (0-100). 0=bottom, 100=top
        whale_pct: Whale long percentage (0-100)
        oi_change: Open Interest change 24h (%)
        historical_win_rate: Win rate from similar historical setups (0-100)
    
    Returns dict with:
        - final_score: Blended score
        - final_direction: 'LONG', 'SHORT', or 'WAIT'
        - final_action: Action word matching blended score
        - engines_agree: Boolean
        - conflict_note: Description if conflict exists
        - setup_type: 'IDEAL', 'WHALE_EXIT_TRAP', 'STRONG_ML_DISAGREE', or None
        - tp_multiplier: For ideal setups, extended TP multiplier
    
    Setup Detection:
        - IDEAL LONG: Whales >65% + OI rising >3% + ML agrees/neutral + EARLY/MIDDLE
        - IDEAL SHORT: Whales <35% + OI rising >3% + ML agrees/neutral + LATE/MIDDLE  
        - WHALE EXIT TRAP: Whales >65% + OI falling <-2% + ML SHORT = AVOID!
        - STRONG ML DISAGREE: ML confidence >70% opposite direction = CAUTION
    
    Position Logic:
        - LONG + LATE (>65%) + ML disagrees = Force WAIT (don't chase!)
        - SHORT + EARLY (<35%) + ML disagrees = Force WAIT (risky short!)
    """
    # Normalize rule direction to LONG/SHORT/WAIT
    rule_is_bullish = rule_direction in ['BULLISH', 'WEAK_BULLISH', 'TA_BULLISH', 'LEAN_BULLISH']
    rule_is_bearish = rule_direction in ['BEARISH', 'WEAK_BEARISH', 'TA_BEARISH', 'LEAN_BEARISH']
    
    if rule_is_bullish:
        rule_dir_normalized = 'LONG'
    elif rule_is_bearish:
        rule_dir_normalized = 'SHORT'
    else:
        rule_dir_normalized = 'WAIT'
    
    # Determine position label
    is_late = position_pct >= 65
    is_early = position_pct <= 35
    is_middle = not is_late and not is_early
    
    # Check agreement
    ml_is_bullish = ml_direction == 'LONG'
    ml_is_bearish = ml_direction == 'SHORT'
    ml_is_neutral = ml_direction == 'WAIT'
    
    engines_agree = (rule_is_bullish and ml_is_bullish) or (rule_is_bearish and ml_is_bearish) or (rule_dir_normalized == 'WAIT' and ml_direction == 'WAIT')
    
    ml_score = int(ml_confidence)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 SETUP TYPE DETECTION: Ideal setups vs Traps
    # ═══════════════════════════════════════════════════════════════════════════
    setup_type = None
    tp_multiplier = 1.0
    
    # Check for WHALE EXIT TRAP first (highest priority warning)
    # Whales positioned but OI declining = they're EXITING, not adding
    whales_bullish = whale_pct >= 65
    whales_bearish = whale_pct <= 35
    oi_rising = oi_change >= 3.0
    oi_falling = oi_change <= -2.0
    
    # Historical validation
    history_supports = historical_win_rate is not None and historical_win_rate >= 50
    history_against = historical_win_rate is not None and historical_win_rate <= 35
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚨 WHALE EXIT TRAP: Whales long + OI falling = EXIT LIQUIDITY!
    # ═══════════════════════════════════════════════════════════════════════════
    if whales_bullish and oi_falling and rule_dir_normalized == 'LONG':
        setup_type = 'WHALE_EXIT_TRAP'
        # This is FORMUSDT scenario - ML correctly says SHORT
        
    elif whales_bearish and oi_falling and rule_dir_normalized == 'SHORT':
        setup_type = 'WHALE_EXIT_TRAP'
        # Inverse - whales short but closing shorts
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 IDEAL LONG: Whales bullish + OI rising + ML agrees + Good position
    # ═══════════════════════════════════════════════════════════════════════════
    elif whales_bullish and oi_rising and (is_early or is_middle):
        # ML should agree or at least not strongly disagree
        if ml_is_bullish or ml_is_neutral:
            setup_type = 'IDEAL_LONG'
            tp_multiplier = 1.5 if ml_is_bullish else 1.3  # Bigger TPs for conviction
            if history_supports:
                tp_multiplier += 0.2  # Even bigger if history confirms
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 PREDICTIVE LONG: Whales heavily bullish + EARLY position
    # This is the IDEAL PREDICTIVE setup - don't wait for TA/ML confirmation!
    # 
    # The whole point of predictive analysis is to enter BEFORE the move.
    # If whales are 70%+ AND we're EARLY, weak TA/ML is EXPECTED because
    # the move hasn't started yet. That's when we SHOULD enter!
    #
    # Waiting for TA/ML confirmation = entering AFTER the move = NOT predictive
    # ═══════════════════════════════════════════════════════════════════════════
    elif whale_pct >= 70 and is_early and rule_dir_normalized == 'LONG':
        # This is PREDICTIVE territory - trust the whale data!
        if ml_is_neutral or ml_is_bullish:
            setup_type = 'PREDICTIVE_LONG'
            tp_multiplier = 1.4  # Extended TPs for early predictive entry
        elif ml_is_bearish and ml_confidence < 70:
            # ML mildly bearish but not strongly - still trust whales
            setup_type = 'PREDICTIVE_LONG'
            tp_multiplier = 1.2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 PREDICTIVE SHORT: Whales heavily bearish + LATE position (top of range)
    # ═══════════════════════════════════════════════════════════════════════════
    elif whale_pct <= 30 and is_late and rule_dir_normalized == 'SHORT':
        if ml_is_neutral or ml_is_bearish:
            setup_type = 'PREDICTIVE_SHORT'
            tp_multiplier = 1.4
        elif ml_is_bullish and ml_confidence < 70:
            setup_type = 'PREDICTIVE_SHORT'
            tp_multiplier = 1.2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 IDEAL SHORT: Whales bearish + OI rising + ML agrees + Good position
    # ═══════════════════════════════════════════════════════════════════════════
    elif whales_bearish and oi_rising and (is_late or is_middle):
        # ML should agree or at least not strongly disagree
        if ml_is_bearish or ml_is_neutral:
            setup_type = 'IDEAL_SHORT'
            tp_multiplier = 1.5 if ml_is_bearish else 1.3
            if history_supports:
                tp_multiplier += 0.2
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⚠️ STRONG ML DISAGREE: ML high confidence opposite direction
    # ═══════════════════════════════════════════════════════════════════════════
    elif ml_confidence >= 70:
        if rule_dir_normalized == 'LONG' and ml_is_bearish:
            setup_type = 'STRONG_ML_DISAGREE'
        elif rule_dir_normalized == 'SHORT' and ml_is_bullish:
            setup_type = 'STRONG_ML_DISAGREE'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 POSITION OVERRIDE: Don't chase LATE longs or EARLY shorts!
    # ═══════════════════════════════════════════════════════════════════════════
    position_override = False
    position_override_reason = None
    
    # LONG but LATE and ML doesn't agree = DON'T CHASE
    if rule_dir_normalized == 'LONG' and is_late and not ml_is_bullish:
        position_override = True
        position_override_reason = f"LATE ({position_pct:.0f}%) + ML not bullish"
    
    # SHORT but EARLY and ML doesn't agree = RISKY SHORT
    if rule_dir_normalized == 'SHORT' and is_early and not ml_is_bearish:
        position_override = True
        position_override_reason = f"EARLY ({position_pct:.0f}%) + ML not bearish"
    
    # WHALE EXIT TRAP always forces WAIT
    if setup_type == 'WHALE_EXIT_TRAP':
        position_override = True
        position_override_reason = f"Whale Exit Trap (OI {oi_change:+.1f}%)"
    
    # STRONG ML DISAGREE with high confidence forces WAIT
    if setup_type == 'STRONG_ML_DISAGREE' and ml_confidence >= 75:
        position_override = True
        position_override_reason = f"ML strongly disagrees ({ml_confidence:.0f}% {ml_direction})"
    
    if engines_agree:
        # Agreement - boost confidence
        base_score = int((rule_score + ml_score) / 2 + 10)
        
        # Extra boost for IDEAL setups
        if setup_type in ['IDEAL_LONG', 'IDEAL_SHORT']:
            base_score += 10  # Conviction bonus
            if history_supports:
                base_score += 5  # History confirms
        
        final_score = min(95, base_score)
        final_direction = rule_dir_normalized
        conflict_note = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 PREDICTIVE SETUP: Trust whale data, don't penalize for ML WAIT!
    # This is the key insight: Waiting for ML/TA confirmation defeats the purpose
    # of predictive analysis. If whales are positioned EARLY, enter NOW!
    # ═══════════════════════════════════════════════════════════════════════════
    elif setup_type in ['PREDICTIVE_LONG', 'PREDICTIVE_SHORT']:
        # Don't penalize! Trust the predictive whale signal
        base_score = int(rule_score * 0.85 + ml_score * 0.15)  # Weight rules heavily
        final_score = max(55, base_score)  # Floor at 55 for predictive setups
        final_direction = rule_dir_normalized
        conflict_note = f"🎯 PREDICTIVE: Whales {whale_pct:.0f}% + EARLY - TA will follow!"
        # Don't apply position_override for predictive setups
        position_override = False
    
    else:
        # Disagreement - pick direction from higher confidence source
        # Penalty applied to blended score
        final_score = max(30, int((rule_score + ml_score) / 2 - 10))
        
        # Extra penalty for WHALE EXIT TRAP or STRONG ML DISAGREE
        if setup_type == 'WHALE_EXIT_TRAP':
            final_score = max(25, final_score - 15)
        elif setup_type == 'STRONG_ML_DISAGREE':
            final_score = max(30, final_score - 10)
        
        # If position override, force WAIT regardless of scores
        if position_override:
            final_direction = 'WAIT'
            conflict_note = position_override_reason
        # Determine which source to trust for direction
        elif rule_score > ml_score:
            final_direction = rule_dir_normalized
            conflict_note = f"Rules ({rule_score}) > ML ({ml_score})"
        elif ml_score > rule_score:
            # ML wins - use ML direction
            final_direction = ml_direction
            conflict_note = f"ML ({ml_score}) > Rules ({rule_score})"
        else:
            # Tie - default to rules
            final_direction = rule_dir_normalized
            conflict_note = f"Tie - defaulting to Rules"
    
    # Get action word from blended score and final direction
    final_action = get_action_from_score(final_score, final_direction)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🏷️ ADD SETUP TYPE BADGE TO ACTION
    # ═══════════════════════════════════════════════════════════════════════════
    if setup_type == 'IDEAL_LONG':
        final_action = f"🎯 {final_action} (IDEAL SETUP)"
    elif setup_type == 'IDEAL_SHORT':
        final_action = f"🎯 {final_action} (IDEAL SETUP)"
    elif setup_type == 'PREDICTIVE_LONG':
        final_action = f"🚀 {final_action} (PREDICTIVE - Whales {whale_pct:.0f}% EARLY)"
    elif setup_type == 'PREDICTIVE_SHORT':
        final_action = f"🚀 {final_action} (PREDICTIVE - Whales {whale_pct:.0f}% LATE)"
    elif setup_type == 'WHALE_EXIT_TRAP':
        final_action = f"🚨 WAIT (WHALE EXIT TRAP - OI {oi_change:+.1f}%)"
    elif setup_type == 'STRONG_ML_DISAGREE':
        final_action = f"⚠️ {final_action} (ML: {ml_direction} {ml_confidence:.0f}%)"
    elif engines_agree:
        final_action = f"⚡ {final_action} (ML agrees)"
    elif position_override:
        final_action = f"⚠️ {final_action} ({position_override_reason})"
    else:
        ml_indicator = ml_direction if ml_direction != 'WAIT' else 'NEUTRAL'
        final_action = f"⚠️ {final_action} (ML: {ml_indicator})"
    
    return {
        'final_score': final_score,
        'final_direction': final_direction,
        'final_action': final_action,
        'engines_agree': engines_agree,
        'conflict_note': conflict_note,
        'rule_score': rule_score,
        'ml_score': ml_score,
        'position_override': position_override,
        'setup_type': setup_type,
        'tp_multiplier': tp_multiplier,
    }