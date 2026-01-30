"""
═══════════════════════════════════════════════════════════════════════════════
SIGNAL GENERATOR V2 - STRUCTURE-BASED
═══════════════════════════════════════════════════════════════════════════════

Philosophy:
- Use REAL structure levels from chart (OB, FVG, S/R, swing points)
- NO arbitrary multipliers for TP/SL
- If no valid structure exists → NO TRADE (better than garbage)
- Mode/Timeframe determines lookback and max acceptable SL

The TA package does the work:
- Finds Order Blocks
- Finds Fair Value Gaps  
- Finds Support/Resistance
- Finds Swing Highs/Lows

We just:
1. Collect all levels
2. Filter by relevance (distance, mode limits)
3. Pick the best ones for entry/SL/TP
4. Reject if structure is garbage

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from .indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_macd,
    calculate_bbands, calculate_obv, calculate_mfi, calculate_cmf
)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIGNAL CLASS (Module level for import compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    """Trade signal with all relevant information - for backward compatibility"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr_ratio: float  # To TP2 (legacy)
    risk_pct: float
    confidence: int  # 0-100
    timeframe: str
    timestamp: str
    
    # Individual R:R for each target
    rr_tp1: float = 0.0
    rr_tp2: float = 0.0
    rr_tp3: float = 0.0
    
    # Level reasoning
    entry_reason: str = "Current price"
    sl_reason: str = "Technical level"
    tp1_reason: str = "Target 1"
    tp2_reason: str = "Target 2"
    tp3_reason: str = "Target 3"
    sl_type: str = "structure"  # 'ob', 'fvg', 'support', 'atr'
    
    # LIMIT ENTRY - Better entry at OB/support (NEW!)
    limit_entry: float = None  # Better entry price at OB/support
    limit_entry_reason: str = ""  # Why this level (e.g., "Bullish OB at $95,420")
    limit_entry_rr_tp1: float = 0.0  # R:R to TP1 from limit entry
    has_limit_entry: bool = False  # Whether a better entry exists
    
    # STRUCTURE LEVELS - All detected OBs, FVGs, Swings for limit entry calculation
    structure_levels: dict = None  # {'support': [...], 'resistance': [...]}
    
    # EXPLOSION-BASED TARGETING (NEW!)
    explosion_color: str = 'GRAY'  # 'GREEN', 'YELLOW', or 'GRAY'
    trade_mode_label: str = 'SCALP'  # 'IMPULSE', 'REACTION', or 'SCALP'
    targets_allowed: int = 1  # 1, 2, or 3 based on explosion color
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'tp3': self.tp3,
            'rr_ratio': self.rr_ratio,
            'rr_tp1': self.rr_tp1,
            'rr_tp2': self.rr_tp2,
            'rr_tp3': self.rr_tp3,
            'risk_pct': self.risk_pct,
            'confidence': self.confidence,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp,
            'entry_reason': self.entry_reason,
            'sl_reason': self.sl_reason,
            'tp1_reason': self.tp1_reason,
            'tp2_reason': self.tp2_reason,
            'tp3_reason': self.tp3_reason,
            'sl_type': self.sl_type,
            # Limit entry fields
            'limit_entry': self.limit_entry,
            'limit_entry_reason': self.limit_entry_reason,
            'limit_entry_rr_tp1': self.limit_entry_rr_tp1,
            'has_limit_entry': self.has_limit_entry,
            # Explosion targeting fields
            'explosion_color': self.explosion_color,
            'trade_mode_label': self.trade_mode_label,
            'targets_allowed': self.targets_allowed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION BY MODE
# ═══════════════════════════════════════════════════════════════════════════════

MODE_CONFIG = {
    'Scalp': {
        'timeframes': ['1m', '5m'],
        'max_sl_pct': 2.5,       # Max 2.5% stop loss
        'min_sl_pct': 0.8,       # MINIMUM SL distance (anti-hunt protection)
        'min_rr': 1.5,           # Minimum R:R for TP1
        'lookback_candles': 20,  # How far back to look for structure
        'anti_hunt_pct': 1.0,    # Buffer below/above levels (was 0.3!)
        'atr_buffer_mult': 1.0,  # ATR multiplier for additional buffer
    },
    'DayTrade': {
        'timeframes': ['15m', '1h'],
        'max_sl_pct': 5.0,
        'min_sl_pct': 1.2,       # MINIMUM SL distance
        'min_rr': 1.2,
        'lookback_candles': 30,
        'anti_hunt_pct': 1.5,    # Was 0.5!
        'atr_buffer_mult': 1.2,
    },
    'Swing': {
        'timeframes': ['4h', '1d'],
        'max_sl_pct': 10.0,
        'min_sl_pct': 2.0,       # MINIMUM SL distance
        'min_rr': 1.0,
        'lookback_candles': 50,
        'anti_hunt_pct': 2.0,    # Was 0.8!
        'atr_buffer_mult': 1.5,
    },
    'Investment': {
        'timeframes': ['1w'],
        'max_sl_pct': 15.0,
        'min_sl_pct': 3.0,       # MINIMUM SL distance
        'min_rr': 0.8,
        'lookback_candles': 52,  # ~1 year of weekly candles
        'anti_hunt_pct': 3.0,    # Was 1.0!
        'atr_buffer_mult': 2.0,
    },
}


def get_mode_config(timeframe: str) -> dict:
    """Get configuration for timeframe"""
    tf = timeframe.lower() if timeframe else '1h'
    for mode, config in MODE_CONFIG.items():
        if tf in config['timeframes']:
            return {'mode': mode, **config}
    return {'mode': 'DayTrade', **MODE_CONFIG['DayTrade']}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLOSION-BASED TARGET ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Target philosophy by Explosion Color:
# 🟢 GREEN (≥70): Impulse/Trend mode - 3 TPs (Liquidity → HTF → Fib/ATR)
# 🟡 YELLOW (≥50): Reaction mode - 2 TPs (Liquidity → 1.5x ATR max)
# ⚪ GRAY (<50): Scalp/Skip - 1 TP (0.5x ATR)
#
# Key insight: Explosion color decides HOW FAR price is ALLOWED to travel,
# not where price MUST go.
#
# ATR multipliers are SCALED by trading mode:
# - Scalp: tighter targets (0.5x multipliers)
# - DayTrade: standard targets (1x multipliers)  
# - Swing: wider targets (1.5x multipliers)
# - Investment: widest targets (2x multipliers)
# ═══════════════════════════════════════════════════════════════════════════════

# Mode-specific ATR scaling factors
MODE_ATR_SCALE = {
    'Scalp': 0.5,       # Tighter targets for scalping
    'DayTrade': 1.0,    # Standard targets
    'Swing': 1.5,       # Wider targets for swing trades
    'Investment': 2.0,  # Widest targets for position trades
}

def get_explosion_color(explosion_score: int) -> str:
    """
    Determine explosion color from score.
    Returns: 'GREEN', 'YELLOW', or 'GRAY'
    """
    if explosion_score >= 70:
        return 'GREEN'
    elif explosion_score >= 50:
        return 'YELLOW'
    else:
        return 'GRAY'


def calculate_level_confluence(level, atr: float, current_price: float, 
                               htf_levels: list = None, fib_levels: list = None) -> int:
    """
    Calculate confluence score for a potential target level.
    
    +1 for each: Liquidity, HTF level nearby, FVG, Fib overlap, ATR alignment
    
    Returns score 0-5
    """
    score = 0
    tolerance_pct = 0.5  # 0.5% tolerance for "nearby" levels
    
    level_price = level.price if hasattr(level, 'price') else level
    
    # +1 for being a liquidity level (swing high/low, range boundary)
    if hasattr(level, 'level_type'):
        if level.level_type in ['swing_high', 'swing_low', 'resistance', 'support']:
            score += 1
    
    # +1 for HTF level nearby
    if htf_levels:
        for htf in htf_levels:
            htf_price = htf.price if hasattr(htf, 'price') else htf
            if abs(level_price - htf_price) / level_price < tolerance_pct / 100:
                score += 1
                break
    
    # +1 for OB or FVG
    if hasattr(level, 'level_type'):
        if 'ob' in level.level_type.lower() or 'fvg' in level.level_type.lower():
            score += 1
    
    # +1 for Fib overlap
    if fib_levels:
        for fib in fib_levels:
            if abs(level_price - fib) / level_price < tolerance_pct / 100:
                score += 1
                break
    
    # +1 for ATR alignment (level is near a round ATR multiple)
    if atr > 0:
        distance = abs(level_price - current_price)
        atr_multiple = distance / atr
        # Check if close to 1x, 1.5x, 2x, 2.5x, 3x ATR
        atr_targets = [1.0, 1.5, 2.0, 2.5, 3.0]
        for target in atr_targets:
            if abs(atr_multiple - target) < 0.2:  # Within 0.2 ATR of a target
                score += 1
                break
    
    return min(5, score)


def assign_explosion_targets(
    direction: str,
    entry: float,
    atr: float,
    structure_levels: list,
    explosion_score: int,
    trade_mode: str = 'DayTrade',
    htf_levels: list = None,
    risk: float = None,
    # ═══════════════════════════════════════════════════════════════════
    # NEW: TP3 Condition Parameters (for GREEN only)
    # TP3 only allowed if: BB expanding + OI rising + Price above VWAP
    # ═══════════════════════════════════════════════════════════════════
    bb_expanding: bool = False,      # Is Bollinger Band width expanding?
    oi_rising: bool = False,         # Is Open Interest holding or rising?
    price_above_vwap: bool = False,  # Is price above VWAP / range midpoint?
    # ═══════════════════════════════════════════════════════════════════
    # NEW: Downgrade Risk Parameters
    # Even in GREEN, downgrade if these are True
    # ═══════════════════════════════════════════════════════════════════
    oi_collapsing: bool = False,     # OI dropping significantly
    funding_extreme: bool = False,   # Funding rate at extreme levels
    delta_divergence: bool = False,  # Volume delta diverging from price
) -> dict:
    """
    Assign TP targets based on explosion score color AND trading mode.
    
    Philosophy: "Explosion color decides how far price is ALLOWED to travel,
    not where price MUST go."
    
    Args:
        direction: 'LONG' or 'SHORT'
        entry: Entry price
        atr: ATR value
        structure_levels: List of StructureLevel objects (sorted by distance)
        explosion_score: 0-100 explosion readiness score
        trade_mode: 'Scalp', 'DayTrade', 'Swing', or 'Investment'
        htf_levels: Optional list of higher timeframe levels
        risk: Stop loss distance for R:R based fallbacks
        
        # TP3 Conditions (GREEN only - all must be True for TP3):
        bb_expanding: Bollinger Band width is expanding (volatility increasing)
        oi_rising: Open Interest holding steady or rising (money staying in)
        price_above_vwap: Price above VWAP or range midpoint (strength)
        
        # Downgrade Triggers (cancels TP3 even in GREEN):
        oi_collapsing: OI dropping fast (smart money exiting)
        funding_extreme: Funding rate at dangerous levels
        delta_divergence: Volume delta not confirming price direction
    
    Returns dict with:
        - tp1, tp2, tp3: Target prices
        - tp1_reason, tp2_reason, tp3_reason: Explanations
        - trade_mode_label: 'IMPULSE', 'REACTION', or 'SCALP'
        - targets_allowed: Number of targets that should be used
        - tp3_allowed: Whether TP3 conditions are met (for GREEN)
        - downgrade_active: Whether downgrade logic triggered
        - downgrade_reasons: List of reasons for downgrade
    """
    explosion_color = get_explosion_color(explosion_score)
    
    # Get mode-specific ATR scaling factor
    # Scalp=0.5x, DayTrade=1x, Swing=1.5x, Investment=2x
    mode_scale = MODE_ATR_SCALE.get(trade_mode, 1.0)
    
    # Scale ATR by trading mode
    scaled_atr = atr * mode_scale
    
    # Default risk if not provided
    if risk is None or risk <= 0:
        risk = scaled_atr
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TP3 CONDITIONS CHECK (for GREEN only)
    # TP3 runner only if: BB expanding AND OI rising AND Price above VWAP
    # ═══════════════════════════════════════════════════════════════════════════
    tp3_conditions_met = bb_expanding and oi_rising and price_above_vwap
    tp3_condition_reasons = []
    if not bb_expanding:
        tp3_condition_reasons.append("BB not expanding")
    if not oi_rising:
        tp3_condition_reasons.append("OI not rising")
    if not price_above_vwap:
        tp3_condition_reasons.append("Price below VWAP")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DOWNGRADE LOGIC CHECK
    # Even in GREEN, downgrade targets if risk factors present
    # ═══════════════════════════════════════════════════════════════════════════
    downgrade_reasons = []
    if oi_collapsing:
        downgrade_reasons.append("OI collapsing")
    if funding_extreme:
        downgrade_reasons.append("Funding extreme")
    if delta_divergence:
        downgrade_reasons.append("Delta divergence")
    
    downgrade_active = len(downgrade_reasons) > 0
    
    # Calculate Fib extension levels from entry (using scaled ATR)
    if direction == 'LONG':
        fib_1272 = entry + (scaled_atr * 2.5)  # Approximates 1.272 extension
        fib_1618 = entry + (scaled_atr * 3.5)  # Approximates 1.618 extension
    else:
        fib_1272 = entry - (scaled_atr * 2.5)
        fib_1618 = entry - (scaled_atr * 3.5)
    
    fib_levels = [fib_1272, fib_1618]
    
    # Filter structure levels by direction
    if direction == 'LONG':
        valid_targets = [l for l in structure_levels if l.price > entry]
    else:
        valid_targets = [l for l in structure_levels if l.price < entry]
    
    # Sort by distance from entry
    valid_targets.sort(key=lambda x: abs(x.price - entry))
    
    # Score each target by confluence
    scored_targets = []
    for level in valid_targets:
        conf_score = calculate_level_confluence(level, scaled_atr, entry, htf_levels, fib_levels)
        scored_targets.append((level, conf_score))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟢 GREEN (≥70): IMPULSE/TREND MODE - 3 TPs
    # Goal: Catch impulse + continuation
    # ATR targets scaled by trade_mode (Scalp=0.5x, DayTrade=1x, Swing=1.5x, Investment=2x)
    # 
    # TP3 CONDITIONS: Only allow runner if ALL conditions met:
    #   - BB width expanding (volatility increasing)
    #   - OI holding or rising (money staying in)
    #   - Price above VWAP / range midpoint (strength)
    #
    # DOWNGRADE LOGIC: Cancel TP3 if any trigger:
    #   - OI collapsing (smart money exiting)
    #   - Funding extreme (crowded trade)
    #   - Delta divergence (volume not confirming)
    # ═══════════════════════════════════════════════════════════════════════════
    if explosion_color == 'GREEN':
        trade_mode_label = 'IMPULSE'
        min_confluence = 2  # Require confluence ≥ 2 for targets
        
        # ─────────────────────────────────────────────────────────────────────────
        # TP1: Safety / Liquidity
        # Nearest liquidity (swing high, range high, session high) OR 1× ATR
        # ─────────────────────────────────────────────────────────────────────────
        tp1 = entry + scaled_atr if direction == 'LONG' else entry - scaled_atr
        tp1_reason = f"ATR 1x ({trade_mode})"
        
        # Find first valid structure target with sufficient confluence
        for level, conf in scored_targets:
            if conf >= 1:  # Minimum for TP1
                if direction == 'LONG':
                    tp1 = level.price * 0.998  # Slightly below resistance
                else:
                    tp1 = level.price * 1.002  # Slightly above support
                tp1_reason = f"🎯 {level.description} (Liquidity)"
                break
        
        # ─────────────────────────────────────────────────────────────────────────
        # TP2: Primary Expansion
        # Nearest HTF level (1H/4H FVG mid→top, HTF resistance/supply) OR 2× ATR
        # ─────────────────────────────────────────────────────────────────────────
        tp2 = entry + (scaled_atr * 2.0) if direction == 'LONG' else entry - (scaled_atr * 2.0)
        tp2_reason = f"ATR 2x ({trade_mode})"
        
        # Find HTF level or high-confluence target
        for level, conf in scored_targets:
            if conf >= min_confluence and abs(level.price - entry) > abs(tp1 - entry):
                if direction == 'LONG':
                    tp2 = level.price * 0.998
                else:
                    tp2 = level.price * 1.002
                tp2_reason = f"🚀 {level.description} (HTF/Confluence)"
                break
        
        # ─────────────────────────────────────────────────────────────────────────
        # TP3: Runner / Trend (CONDITIONAL!)
        # Fib extension 1.272–1.618 OR 3× ATR
        # 
        # ONLY IF all conditions met:
        #   ✓ BB width expanding
        #   ✓ OI holding or rising
        #   ✓ Price above VWAP / range midpoint
        #
        # BLOCKED IF any downgrade trigger:
        #   ✗ OI collapsing → Cancel TP3
        #   ✗ Funding extreme → Cancel TP3
        #   ✗ Delta divergence → Cancel TP3
        # ─────────────────────────────────────────────────────────────────────────
        
        # Check if TP3 is allowed
        tp3_allowed = tp3_conditions_met and not downgrade_active
        
        if tp3_allowed:
            # Full TP3 - Runner target allowed
            targets_allowed = 3
            tp3 = entry + (scaled_atr * 3.0) if direction == 'LONG' else entry - (scaled_atr * 3.0)
            tp3_reason = f"ATR 3x ({trade_mode})"
            
            # Use Fib extension if aligns with structure
            for level, conf in scored_targets:
                if conf >= min_confluence and abs(level.price - entry) > abs(tp2 - entry):
                    # Check if near Fib level
                    for fib in fib_levels:
                        if abs(level.price - fib) / entry < 0.01:  # Within 1%
                            if direction == 'LONG':
                                tp3 = level.price * 0.998
                            else:
                                tp3 = level.price * 1.002
                            tp3_reason = f"🏆 {level.description} (Fib 1.618)"
                            break
                    break
            
            # Fallback to Fib extension if no structure found
            if f"ATR 3x ({trade_mode})" in tp3_reason:
                tp3 = fib_1618
                tp3_reason = f"📐 Fib 1.618 ({trade_mode})"
        else:
            # TP3 BLOCKED - conditions not met or downgrade active
            targets_allowed = 2  # Downgrade to 2 targets only
            tp3 = tp2  # Set TP3 = TP2 (no runner)
            
            if downgrade_active:
                tp3_reason = f"🚫 Runner BLOCKED: {', '.join(downgrade_reasons)}"
            else:
                tp3_reason = f"🚫 Runner BLOCKED: {', '.join(tp3_condition_reasons)}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟡 YELLOW (≥50): REACTION MODE - 2 TPs
    # Goal: Capture initial reaction, not full trend
    # ATR targets scaled by trade_mode
    # ═══════════════════════════════════════════════════════════════════════════
    elif explosion_color == 'YELLOW':
        trade_mode_label = 'REACTION'
        targets_allowed = 2
        min_confluence = 1  # Lower bar for reaction trades
        
        # TP1: Nearest liquidity OR 1x scaled_ATR
        tp1 = entry + scaled_atr if direction == 'LONG' else entry - scaled_atr
        tp1_reason = f"ATR 1x ({trade_mode})"
        
        for level, conf in scored_targets:
            if conf >= min_confluence:
                if direction == 'LONG':
                    tp1 = level.price * 0.998
                else:
                    tp1 = level.price * 1.002
                tp1_reason = f"🎯 {level.description} (Liquidity)"
                break
        
        # TP2: Close HTF level OR 1.5x scaled_ATR (MAX for yellow)
        tp2 = entry + (scaled_atr * 1.5) if direction == 'LONG' else entry - (scaled_atr * 1.5)
        tp2_reason = f"ATR 1.5x ({trade_mode})"
        
        for level, conf in scored_targets:
            level_distance = abs(level.price - entry)
            max_distance = scaled_atr * 1.5
            if conf >= min_confluence and level_distance > abs(tp1 - entry) and level_distance <= max_distance:
                if direction == 'LONG':
                    tp2 = level.price * 0.998
                else:
                    tp2 = level.price * 1.002
                tp2_reason = f"⚡ {level.description} (Reaction)"
                break
        
        # TP3 = TP2 for yellow (no expansion)
        tp3 = tp2
        tp3_reason = "🚫 No Runner (Reaction Mode)"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ⚪ GRAY (<50): SCALP MODE - 1 TP (or skip)
    # Goal: Preserve capital
    # ATR targets scaled by trade_mode
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        trade_mode_label = 'SCALP'
        targets_allowed = 1
        
        # TP1: VWAP, range mid, or 0.5x scaled_ATR (very conservative)
        tp1 = entry + (scaled_atr * 0.5) if direction == 'LONG' else entry - (scaled_atr * 0.5)
        tp1_reason = f"ATR 0.5x ({trade_mode})"
        
        # Find nearest structure within scalp range
        for level, conf in scored_targets:
            level_distance = abs(level.price - entry)
            if level_distance <= scaled_atr * 0.7:  # Max 0.7 scaled_ATR for scalp
                if direction == 'LONG':
                    tp1 = level.price * 0.998
                else:
                    tp1 = level.price * 1.002
                tp1_reason = f"⚡ {level.description} (Scalp)"
                break
        
        # TP2 and TP3 = TP1 (no expansion for gray)
        tp2 = tp1
        tp3 = tp1
        tp2_reason = "🚫 No Expansion (Low Energy)"
        tp3_reason = "🚫 No Runner (Low Energy)"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENSURE TP ORDER IS CORRECT
    # ═══════════════════════════════════════════════════════════════════════════
    if direction == 'LONG':
        # For LONG: tp1 < tp2 < tp3
        if tp2 < tp1:
            tp2 = tp1 + (scaled_atr * 0.5)
        if tp3 < tp2:
            tp3 = tp2 + (scaled_atr * 0.5)
    else:
        # For SHORT: tp1 > tp2 > tp3
        if tp2 > tp1:
            tp2 = tp1 - (scaled_atr * 0.5)
        if tp3 > tp2:
            tp3 = tp2 - (scaled_atr * 0.5)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SET DEFAULTS FOR TP3 FLAGS (for YELLOW and GRAY which don't have TP3)
    # ═══════════════════════════════════════════════════════════════════════════
    if explosion_color != 'GREEN':
        tp3_allowed = False  # YELLOW and GRAY never have TP3
    
    return {
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'tp1_reason': tp1_reason,
        'tp2_reason': tp2_reason,
        'tp3_reason': tp3_reason,
        'trade_mode_label': trade_mode_label,
        'targets_allowed': targets_allowed,
        'explosion_color': explosion_color,
        # NEW: TP3 condition tracking
        'tp3_allowed': tp3_allowed,
        'tp3_conditions_met': tp3_conditions_met,
        'tp3_condition_reasons': tp3_condition_reasons,
        # NEW: Downgrade tracking
        'downgrade_active': downgrade_active,
        'downgrade_reasons': downgrade_reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE LEVEL DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructureLevel:
    """A price level with context"""
    price: float
    level_type: str      # 'ob', 'fvg', 'support', 'resistance', 'swing_high', 'swing_low'
    strength: int        # 1-5 (how strong is this level)
    description: str     # Human readable
    distance_pct: float  # Distance from current price (%)


@dataclass
class TradeSetup:
    """Complete trade setup with all levels"""
    symbol: str
    direction: str       # 'LONG' or 'SHORT'
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    
    # Risk metrics
    sl_pct: float        # Stop loss distance %
    rr_tp1: float        # Risk:Reward to TP1
    rr_tp2: float        # Risk:Reward to TP2
    rr_tp3: float        # Risk:Reward to TP3
    
    # Level details
    entry_reason: str
    sl_reason: str
    sl_type: str
    tp1_reason: str
    tp2_reason: str
    tp3_reason: str
    
    # Meta
    timeframe: str
    trade_mode: str
    confidence: int      # 0-100
    is_valid: bool       # True if trade meets all criteria
    rejection_reason: str  # Why trade was rejected (if invalid)
    
    # Structure levels for limit entry calculation
    structure_levels: dict = None  # {'support': [...], 'resistance': [...]}
    
    # Explosion-based targeting (NEW!)
    explosion_color: str = 'GRAY'  # 'GREEN', 'YELLOW', or 'GRAY'
    trade_mode_label: str = 'SCALP'  # 'IMPULSE', 'REACTION', or 'SCALP'
    targets_allowed: int = 1  # 1, 2, or 3 based on explosion color
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry': self.entry,
            'stop_loss': self.stop_loss,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'tp3': self.tp3,
            'sl_pct': self.sl_pct,
            'rr_ratio': self.rr_tp2,  # Legacy compatibility
            'rr_tp1': self.rr_tp1,
            'rr_tp2': self.rr_tp2,
            'rr_tp3': self.rr_tp3,
            'risk_pct': self.sl_pct,
            'entry_reason': self.entry_reason,
            'sl_reason': self.sl_reason,
            'sl_type': self.sl_type,
            'tp1_reason': self.tp1_reason,
            'tp2_reason': self.tp2_reason,
            'tp3_reason': self.tp3_reason,
            'timeframe': self.timeframe,
            'confidence': self.confidence,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'is_valid': self.is_valid,
            'rejection_reason': self.rejection_reason,
            'explosion_color': self.explosion_color,
            'trade_mode_label': self.trade_mode_label,
            'targets_allowed': self.targets_allowed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_all_structure_levels(df: pd.DataFrame, current_price: float, lookback: int = 30) -> Dict[str, List[StructureLevel]]:
    """
    Find ALL structure levels and categorize them.
    
    Returns:
        {
            'support': [StructureLevel, ...],  # Levels BELOW current price
            'resistance': [StructureLevel, ...]  # Levels ABOVE current price
        }
    """
    support_levels = []
    resistance_levels = []
    
    if df is None or len(df) < 10:
        return {'support': [], 'resistance': []}
    
    # Limit lookback
    lookback = min(lookback, len(df) - 5)
    df_recent = df.tail(lookback + 5)
    
    atr = (df_recent['High'] - df_recent['Low']).rolling(14).mean().iloc[-1]
    if pd.isna(atr) or atr == 0:
        atr = current_price * 0.02  # Fallback: 2% of price
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. SWING HIGHS AND LOWS (Most reliable)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(3, len(df_recent) - 3):
        # Swing Low
        if df_recent['Low'].iloc[i] == df_recent['Low'].iloc[i-3:i+4].min():
            price = df_recent['Low'].iloc[i]
            dist_pct = ((current_price - price) / current_price) * 100
            
            if price < current_price:
                support_levels.append(StructureLevel(
                    price=price,
                    level_type='swing_low',
                    strength=3,
                    description=f"Swing Low",
                    distance_pct=abs(dist_pct)
                ))
            else:
                resistance_levels.append(StructureLevel(
                    price=price,
                    level_type='swing_low',
                    strength=2,
                    description=f"Swing Low (above)",
                    distance_pct=abs(dist_pct)
                ))
        
        # Swing High
        if df_recent['High'].iloc[i] == df_recent['High'].iloc[i-3:i+4].max():
            price = df_recent['High'].iloc[i]
            dist_pct = ((price - current_price) / current_price) * 100
            
            if price > current_price:
                resistance_levels.append(StructureLevel(
                    price=price,
                    level_type='swing_high',
                    strength=3,
                    description=f"Swing High",
                    distance_pct=abs(dist_pct)
                ))
            else:
                support_levels.append(StructureLevel(
                    price=price,
                    level_type='swing_high',
                    strength=2,
                    description=f"Swing High (below)",
                    distance_pct=abs(dist_pct)
                ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. ORDER BLOCKS (High priority - institutional footprints)
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(3, len(df_recent) - 3):  # Same range as swing detection
        try:
            # Bullish OB: Red candle followed by strong bullish move
            # Bullish OB is ALWAYS SUPPORT (where buyers stepped in)
            if df_recent['Close'].iloc[i] < df_recent['Open'].iloc[i]:
                next_move = df_recent['High'].iloc[i+1:i+4].max() - df_recent['Close'].iloc[i]
                if next_move > atr * 1.0:
                    ob_price = df_recent['Close'].iloc[i]  # Bottom of OB
                    dist_pct = ((current_price - ob_price) / current_price) * 100
                    
                    # Only add if below current price (valid for LONG limit entry)
                    if ob_price < current_price:
                        level = StructureLevel(
                            price=ob_price,
                            level_type='bullish_ob',
                            strength=4,
                            description=f"Bullish OB",
                            distance_pct=abs(dist_pct)
                        )
                        support_levels.append(level)
            
            # Bearish OB: Green candle followed by strong bearish move
            # Bearish OB is ALWAYS RESISTANCE (where sellers stepped in)
            if df_recent['Close'].iloc[i] > df_recent['Open'].iloc[i]:
                next_move = df_recent['Close'].iloc[i] - df_recent['Low'].iloc[i+1:i+4].min()
                if next_move > atr * 1.0:
                    ob_price = df_recent['Close'].iloc[i]  # Top of OB
                    dist_pct = ((ob_price - current_price) / current_price) * 100
                    
                    # Only add if above current price (valid for SHORT limit entry)
                    if ob_price > current_price:
                        level = StructureLevel(
                            price=ob_price,
                            level_type='bearish_ob',
                            strength=4,
                            description=f"Bearish OB",
                            distance_pct=abs(dist_pct)
                        )
                        resistance_levels.append(level)
        except:
            continue
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. FAIR VALUE GAPS
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(3, len(df_recent) - 2):  # Fixed: scan all candles
        try:
            # Bullish FVG: Gap up
            if df_recent['Low'].iloc[i+2] > df_recent['High'].iloc[i]:
                fvg_price = df_recent['High'].iloc[i]  # Bottom of gap
                dist_pct = ((current_price - fvg_price) / current_price) * 100
                
                level = StructureLevel(
                    price=fvg_price,
                    level_type='bullish_fvg',
                    strength=3,
                    description=f"Bullish FVG",
                    distance_pct=abs(dist_pct)
                )
                
                if fvg_price < current_price:
                    support_levels.append(level)
            
            # Bearish FVG: Gap down
            if df_recent['High'].iloc[i+2] < df_recent['Low'].iloc[i]:
                fvg_price = df_recent['Low'].iloc[i]  # Top of gap
                dist_pct = ((fvg_price - current_price) / current_price) * 100
                
                level = StructureLevel(
                    price=fvg_price,
                    level_type='bearish_fvg',
                    strength=3,
                    description=f"Bearish FVG",
                    distance_pct=abs(dist_pct)
                )
                
                if fvg_price > current_price:
                    resistance_levels.append(level)
        except:
            continue
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. RECENT HIGH/LOW (Always valid)
    # ─────────────────────────────────────────────────────────────────────────
    recent_high = df_recent['High'].max()
    recent_low = df_recent['Low'].min()
    
    if recent_high > current_price * 1.002:  # At least 0.2% above
        dist_pct = ((recent_high - current_price) / current_price) * 100
        resistance_levels.append(StructureLevel(
            price=recent_high,
            level_type='recent_high',
            strength=4,
            description=f"Recent High ({lookback} candles)",
            distance_pct=dist_pct
        ))
    
    if recent_low < current_price * 0.998:  # At least 0.2% below
        dist_pct = ((current_price - recent_low) / current_price) * 100
        support_levels.append(StructureLevel(
            price=recent_low,
            level_type='recent_low',
            strength=4,
            description=f"Recent Low ({lookback} candles)",
            distance_pct=dist_pct
        ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Sort by distance (nearest first) and remove duplicates
    # ─────────────────────────────────────────────────────────────────────────
    def dedupe_levels(levels: List[StructureLevel], tolerance_pct: float = 0.5) -> List[StructureLevel]:
        """Remove levels that are too close to each other"""
        if not levels:
            return []
        
        # Sort by strength (higher first), then distance (closer first)
        levels.sort(key=lambda x: (-x.strength, x.distance_pct))
        
        result = []
        for level in levels:
            # Check if any existing level is too close
            is_duplicate = False
            for existing in result:
                price_diff_pct = abs(level.price - existing.price) / existing.price * 100
                if price_diff_pct < tolerance_pct:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                result.append(level)
        
        # Re-sort by distance for final output
        result.sort(key=lambda x: x.distance_pct)
        return result
    
    return {
        'support': dedupe_levels(support_levels),
        'resistance': dedupe_levels(resistance_levels)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGeneratorV2:
    """
    Generate trade signals using ACTUAL structure levels.
    No arbitrary multipliers. Structure-based or no trade.
    """
    
    @staticmethod
    def generate_signal(df: pd.DataFrame, symbol: str, timeframe: str = '1h', 
                        force_direction: str = None, explosion_score: int = 0,
                        trade_mode: str = None, htf_levels: list = None,
                        tp3_conditions: dict = None) -> Optional[TradeSetup]:
        """
        Generate a trade signal based on actual chart structure.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Chart timeframe
            force_direction: If set to 'LONG' or 'SHORT', override technical direction
                           Used when predictive system indicates different direction
            explosion_score: 0-100 explosion readiness score for target assignment
                           🟢 ≥70: Impulse mode (3 TPs)
                           🟡 ≥50: Reaction mode (2 TPs)
                           ⚪ <50: Scalp mode (1 TP)
            trade_mode: UI-selected trading mode ('Scalp', 'DayTrade', 'Swing', 'Investment')
                       Used for ATR scaling of targets. If None, derived from timeframe.
            htf_levels: Optional list of higher timeframe structure levels
            tp3_conditions: Dict with TP3 condition flags (for GREEN explosion):
                {
                    'bb_expanding': bool,      # BB width expanding?
                    'oi_rising': bool,         # OI holding or rising?
                    'price_above_vwap': bool,  # Price above VWAP/range mid?
                    'oi_collapsing': bool,     # Downgrade: OI dropping fast?
                    'funding_extreme': bool,   # Downgrade: Funding at extreme?
                    'delta_divergence': bool,  # Downgrade: Delta not confirming?
                }
        
        Returns None if:
        - No valid structure for SL
        - No valid structure for TP
        - R:R doesn't meet minimum
        - SL exceeds max for mode
        """
        if df is None or len(df) < 20:
            return None
        
        # Default TP3 conditions if not provided
        if tp3_conditions is None:
            tp3_conditions = {}
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # SETUP
            # ═══════════════════════════════════════════════════════════════════
            current_price = df['Close'].iloc[-1]
            config = get_mode_config(timeframe)
            
            # Use UI-selected trade_mode if provided, else use timeframe-derived mode
            # Also normalize trade_mode to handle variations (day_trade -> DayTrade)
            if trade_mode:
                # Normalize: 'day_trade' -> 'DayTrade', 'scalp' -> 'Scalp', etc.
                mode_map = {
                    'scalp': 'Scalp', 'Scalp': 'Scalp',
                    'day_trade': 'DayTrade', 'daytrade': 'DayTrade', 'DayTrade': 'DayTrade',
                    'swing': 'Swing', 'Swing': 'Swing',
                    'investment': 'Investment', 'Investment': 'Investment',
                }
                effective_trade_mode = mode_map.get(trade_mode, 'DayTrade')
            else:
                effective_trade_mode = config['mode']  # Fallback to timeframe-derived
            
            MAX_SL_PCT = config['max_sl_pct']
            MIN_SL_PCT = config.get('min_sl_pct', 1.0)  # NEW: Minimum SL distance for anti-hunt
            MIN_RR = config['min_rr']
            LOOKBACK = config['lookback_candles']
            ANTI_HUNT_PCT = config['anti_hunt_pct']
            ATR_BUFFER_MULT = config.get('atr_buffer_mult', 1.0)  # NEW: ATR multiplier for volatile coins
            
            # ═══════════════════════════════════════════════════════════════════
            # FIND ALL STRUCTURE LEVELS
            # ═══════════════════════════════════════════════════════════════════
            levels = find_all_structure_levels(df, current_price, LOOKBACK)
            
            # Filter by max SL distance
            valid_support = [l for l in levels['support'] if l.distance_pct <= MAX_SL_PCT]
            valid_resistance = [l for l in levels['resistance']]  # No max for TPs
            
            # ═══════════════════════════════════════════════════════════════════
            # DETERMINE DIRECTION
            # ═══════════════════════════════════════════════════════════════════
            
            # Always calculate these for confidence scoring
            rsi = calculate_rsi(df['Close'], 14).iloc[-1]
            ema_20 = calculate_ema(df['Close'], 20).iloc[-1]
            ema_50 = calculate_ema(df['Close'], 50).iloc[-1]
            macd_line, signal_line, _ = calculate_macd(df['Close'])
            current_macd = macd_line.iloc[-1]
            current_signal_line = signal_line.iloc[-1]
            
            bullish_signals = 0
            bearish_signals = 0
            
            if current_price > ema_20 > ema_50:
                bullish_signals += 2
            elif current_price < ema_20 < ema_50:
                bearish_signals += 2
            
            if rsi < 35:
                bullish_signals += 1
            elif rsi > 65:
                bearish_signals += 1
            
            if current_macd > current_signal_line:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # If force_direction is set, use that instead of technical analysis
            if force_direction in ['LONG', 'SHORT']:
                direction = force_direction
            else:
                direction = 'LONG' if bullish_signals >= bearish_signals else 'SHORT'
            
            # ═══════════════════════════════════════════════════════════════════
            # BUILD TRADE SETUP
            # ═══════════════════════════════════════════════════════════════════
            entry = current_price
            entry_reason = "Current price"
            
            # Extract TP3 conditions once (used for both LONG and SHORT)
            bb_expanding = tp3_conditions.get('bb_expanding', False)
            oi_rising = tp3_conditions.get('oi_rising', False)
            price_above_vwap = tp3_conditions.get('price_above_vwap', False)
            oi_collapsing = tp3_conditions.get('oi_collapsing', False)
            funding_extreme = tp3_conditions.get('funding_extreme', False)
            delta_divergence = tp3_conditions.get('delta_divergence', False)
            
            if direction == 'LONG':
                # ─────────────────────────────────────────────────────────────────
                # LONG: SL below support, TP at resistance (EXPLOSION-BASED)
                # ═══════════════════════════════════════════════════════════════════
                # ANTI-HUNT STOP LOSS PROTECTION
                # Stops get hunted because they're placed at obvious levels.
                # Solution: Place SL BELOW the liquidity sweep zone.
                # ═══════════════════════════════════════════════════════════════════
                
                # Find best SL level (nearest valid support)
                if not valid_support:
                    return SignalGeneratorV2._create_invalid_setup(
                        symbol, direction, entry, timeframe, config['mode'],
                        f"No support within {MAX_SL_PCT}% for SL"
                    )
                
                sl_level = valid_support[0]  # Nearest support
                
                # Calculate ATR for volatility-based buffer
                atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
                atr_pct = (atr / entry) * 100  # ATR as percentage of price
                
                # ═══════════════════════════════════════════════════════════════════
                # ANTI-HUNT BUFFER CALCULATION
                # Use MAXIMUM of:
                #   1. Fixed percentage buffer (ANTI_HUNT_PCT)
                #   2. ATR-based buffer (ATR * multiplier)
                #   3. Minimum SL distance from entry (MIN_SL_PCT)
                # ═══════════════════════════════════════════════════════════════════
                
                # Calculate structure-based SL with percentage buffer
                sl_from_structure = sl_level.price * (1 - ANTI_HUNT_PCT / 100)
                
                # Calculate ATR-based buffer (adds extra room for volatility)
                atr_buffer = atr * ATR_BUFFER_MULT
                sl_from_atr = sl_level.price - atr_buffer
                
                # Use the LOWER (safer) of the two - gives more room
                stop_loss = min(sl_from_structure, sl_from_atr)
                
                # Calculate actual SL %
                sl_pct = ((entry - stop_loss) / entry) * 100
                
                # ═══════════════════════════════════════════════════════════════════
                # ENFORCE MINIMUM SL DISTANCE (Anti-Hunt Floor)
                # Never let SL be less than MIN_SL_PCT from entry!
                # ═══════════════════════════════════════════════════════════════════
                if sl_pct < MIN_SL_PCT:
                    stop_loss = entry * (1 - MIN_SL_PCT / 100)
                    sl_pct = MIN_SL_PCT
                    sl_reason = f"Anti-hunt minimum ({MIN_SL_PCT:.1f}%)"
                    sl_type = "anti_hunt_min"
                else:
                    if stop_loss == sl_from_atr:
                        sl_reason = f"{sl_level.description} (-{ATR_BUFFER_MULT:.1f}×ATR buffer)"
                    else:
                        sl_reason = f"{sl_level.description} (-{ANTI_HUNT_PCT}% anti-hunt)"
                    sl_type = sl_level.level_type
                
                # Double-check SL is within MAX limits
                if sl_pct > MAX_SL_PCT:
                    stop_loss = entry * (1 - MAX_SL_PCT / 100)
                    sl_pct = MAX_SL_PCT
                    sl_reason = f"Capped at {MAX_SL_PCT}% (mode limit)"
                    sl_type = "max_cap"
                
                # Risk amount
                risk = entry - stop_loss
                
                # ═══════════════════════════════════════════════════════════════════
                # EXPLOSION-BASED TP ASSIGNMENT
                # 🟢 Green (≥70): 3 TPs - Impulse/Trend mode
                # 🟡 Yellow (≥50): 2 TPs - Reaction mode  
                # ⚪ Gray (<50): 1 TP - Scalp mode
                # ATR scaled by UI-selected trade_mode: Scalp=0.5x, DayTrade=1x, Swing=1.5x, Investment=2x
                # ═══════════════════════════════════════════════════════════════════
                atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
                
                # Use explosion-based targeting with UI-selected trade_mode for ATR scaling
                explosion_targets = assign_explosion_targets(
                    direction='LONG',
                    entry=entry,
                    atr=atr,
                    structure_levels=valid_resistance,
                    explosion_score=explosion_score,
                    trade_mode=effective_trade_mode,
                    htf_levels=htf_levels,
                    risk=risk,
                    # TP3 conditions (for GREEN only)
                    bb_expanding=bb_expanding,
                    oi_rising=oi_rising,
                    price_above_vwap=price_above_vwap,
                    # Downgrade triggers
                    oi_collapsing=oi_collapsing,
                    funding_extreme=funding_extreme,
                    delta_divergence=delta_divergence,
                )
                
                tp1 = explosion_targets['tp1']
                tp2 = explosion_targets['tp2']
                tp3 = explosion_targets['tp3']
                tp1_reason = explosion_targets['tp1_reason']
                tp2_reason = explosion_targets['tp2_reason']
                tp3_reason = explosion_targets['tp3_reason']
                trade_mode_label = explosion_targets['trade_mode_label']
                targets_allowed = explosion_targets['targets_allowed']
                
            else:
                # ─────────────────────────────────────────────────────────────────
                # SHORT: SL above resistance, TP at support (EXPLOSION-BASED)
                # ═══════════════════════════════════════════════════════════════════
                # ANTI-HUNT STOP LOSS PROTECTION
                # For shorts, SL is above entry at resistance. Stops get hunted by
                # quick wicks above obvious resistance levels.
                # Solution: Place SL ABOVE the liquidity sweep zone.
                # ═══════════════════════════════════════════════════════════════════
                
                # For shorts, we need resistance above for SL
                valid_resistance_for_sl = [l for l in levels['resistance'] if l.distance_pct <= MAX_SL_PCT]
                
                if not valid_resistance_for_sl:
                    return SignalGeneratorV2._create_invalid_setup(
                        symbol, direction, entry, timeframe, config['mode'],
                        f"No resistance within {MAX_SL_PCT}% for SL"
                    )
                
                sl_level = valid_resistance_for_sl[0]
                
                # Calculate ATR for volatility-based buffer
                atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
                atr_pct = (atr / entry) * 100  # ATR as percentage of price
                
                # ═══════════════════════════════════════════════════════════════════
                # ANTI-HUNT BUFFER CALCULATION (SHORT)
                # Use MAXIMUM of:
                #   1. Fixed percentage buffer (ANTI_HUNT_PCT)
                #   2. ATR-based buffer (ATR * multiplier)
                #   3. Minimum SL distance from entry (MIN_SL_PCT)
                # ═══════════════════════════════════════════════════════════════════
                
                # Calculate structure-based SL with percentage buffer
                sl_from_structure = sl_level.price * (1 + ANTI_HUNT_PCT / 100)
                
                # Calculate ATR-based buffer (adds extra room for volatility)
                atr_buffer = atr * ATR_BUFFER_MULT
                sl_from_atr = sl_level.price + atr_buffer
                
                # Use the HIGHER (safer) of the two for shorts - gives more room
                stop_loss = max(sl_from_structure, sl_from_atr)
                
                # Calculate actual SL %
                sl_pct = ((stop_loss - entry) / entry) * 100
                
                # ═══════════════════════════════════════════════════════════════════
                # ENFORCE MINIMUM SL DISTANCE (Anti-Hunt Floor)
                # Never let SL be less than MIN_SL_PCT from entry!
                # ═══════════════════════════════════════════════════════════════════
                if sl_pct < MIN_SL_PCT:
                    stop_loss = entry * (1 + MIN_SL_PCT / 100)
                    sl_pct = MIN_SL_PCT
                    sl_reason = f"Anti-hunt minimum ({MIN_SL_PCT:.1f}%)"
                    sl_type = "anti_hunt_min"
                else:
                    if stop_loss == sl_from_atr:
                        sl_reason = f"{sl_level.description} (+{ATR_BUFFER_MULT:.1f}×ATR buffer)"
                    else:
                        sl_reason = f"{sl_level.description} (+{ANTI_HUNT_PCT}% anti-hunt)"
                    sl_type = sl_level.level_type
                
                # Double-check SL is within MAX limits
                if sl_pct > MAX_SL_PCT:
                    stop_loss = entry * (1 + MAX_SL_PCT / 100)
                    sl_pct = MAX_SL_PCT
                    sl_reason = f"Capped at {MAX_SL_PCT}% (mode limit)"
                    sl_type = "max_cap"
                
                risk = stop_loss - entry
                
                # ═══════════════════════════════════════════════════════════════════
                # EXPLOSION-BASED TP ASSIGNMENT
                # 🟢 Green (≥70): 3 TPs - Impulse/Trend mode
                # 🟡 Yellow (≥50): 2 TPs - Reaction mode  
                # ⚪ Gray (<50): 1 TP - Scalp mode
                # ATR scaled by UI-selected trade_mode: Scalp=0.5x, DayTrade=1x, Swing=1.5x, Investment=2x
                # ═══════════════════════════════════════════════════════════════════
                atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
                
                # Use explosion-based targeting with UI-selected trade_mode for ATR scaling
                explosion_targets = assign_explosion_targets(
                    direction='SHORT',
                    entry=entry,
                    atr=atr,
                    structure_levels=valid_support,  # Support levels for SHORT TPs
                    explosion_score=explosion_score,
                    trade_mode=effective_trade_mode,
                    htf_levels=htf_levels,
                    risk=risk,
                    # TP3 conditions (for GREEN only)
                    bb_expanding=bb_expanding,
                    oi_rising=oi_rising,
                    price_above_vwap=price_above_vwap,
                    # Downgrade triggers
                    oi_collapsing=oi_collapsing,
                    funding_extreme=funding_extreme,
                    delta_divergence=delta_divergence,
                )
                
                tp1 = explosion_targets['tp1']
                tp2 = explosion_targets['tp2']
                tp3 = explosion_targets['tp3']
                tp1_reason = explosion_targets['tp1_reason']
                tp2_reason = explosion_targets['tp2_reason']
                tp3_reason = explosion_targets['tp3_reason']
                trade_mode_label = explosion_targets['trade_mode_label']
                targets_allowed = explosion_targets['targets_allowed']
            
            # ═══════════════════════════════════════════════════════════════════
            # CALCULATE R:R
            # ═══════════════════════════════════════════════════════════════════
            if direction == 'LONG':
                rr_tp1 = (tp1 - entry) / risk if risk > 0 else 0
                rr_tp2 = (tp2 - entry) / risk if risk > 0 else 0
                rr_tp3 = (tp3 - entry) / risk if risk > 0 else 0
            else:
                rr_tp1 = (entry - tp1) / risk if risk > 0 else 0
                rr_tp2 = (entry - tp2) / risk if risk > 0 else 0
                rr_tp3 = (entry - tp3) / risk if risk > 0 else 0
            
            # ═══════════════════════════════════════════════════════════════════
            # VALIDATE TRADE
            # ═══════════════════════════════════════════════════════════════════
            is_valid = True
            rejection_reason = ""
            
            # Check minimum R:R
            if rr_tp1 < MIN_RR:
                is_valid = False
                rejection_reason = f"TP1 R:R ({rr_tp1:.2f}) below minimum ({MIN_RR})"
            
            # Check SL makes sense
            if sl_pct < 0.1:
                is_valid = False
                rejection_reason = "SL too tight (< 0.1%)"
            
            # ═══════════════════════════════════════════════════════════════════
            # CALCULATE CONFIDENCE
            # ═══════════════════════════════════════════════════════════════════
            confidence = SignalGeneratorV2._calculate_confidence(
                direction, sl_type, rr_tp1, rsi, 
                bullish_signals if direction == 'LONG' else bearish_signals,
                len(valid_support), len(valid_resistance)
            )
            
            # Get explosion color for display
            explosion_color = get_explosion_color(explosion_score)
            
            return TradeSetup(
                symbol=symbol,
                direction=direction,
                entry=entry,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                sl_pct=sl_pct,
                rr_tp1=rr_tp1,
                rr_tp2=rr_tp2,
                rr_tp3=rr_tp3,
                entry_reason=entry_reason,
                sl_reason=sl_reason,
                sl_type=sl_type,
                tp1_reason=tp1_reason,
                tp2_reason=tp2_reason,
                tp3_reason=tp3_reason,
                timeframe=timeframe,
                trade_mode=effective_trade_mode,  # Use UI-selected mode (or fallback)
                confidence=confidence,
                is_valid=is_valid,
                rejection_reason=rejection_reason,
                structure_levels=levels,  # Pass ALL detected levels for limit entry
                explosion_color=explosion_color,
                trade_mode_label=trade_mode_label,
                targets_allowed=targets_allowed
            )
            
        except Exception as e:
            return None
    
    @staticmethod
    def _create_invalid_setup(symbol, direction, entry, timeframe, mode, reason) -> TradeSetup:
        """Create an invalid trade setup with rejection reason"""
        return TradeSetup(
            symbol=symbol,
            direction=direction,
            entry=entry,
            stop_loss=0,
            tp1=0,
            tp2=0,
            tp3=0,
            sl_pct=0,
            rr_tp1=0,
            rr_tp2=0,
            rr_tp3=0,
            entry_reason="N/A",
            sl_reason="N/A",
            sl_type="none",
            tp1_reason="N/A",
            tp2_reason="N/A",
            tp3_reason="N/A",
            timeframe=timeframe,
            trade_mode=mode,
            confidence=0,
            is_valid=False,
            rejection_reason=reason
        )
    
    @staticmethod
    def _calculate_confidence(direction, sl_type, rr_tp1, rsi, signal_count, 
                             support_count, resistance_count) -> int:
        """Calculate confidence score 0-100"""
        score = 40  # Base score
        
        # SL quality
        if sl_type in ['bullish_ob', 'bearish_ob']:
            score += 20
        elif sl_type in ['bullish_fvg', 'bearish_fvg']:
            score += 15
        elif sl_type in ['swing_low', 'swing_high', 'recent_low', 'recent_high']:
            score += 10
        
        # R:R quality
        if rr_tp1 >= 2.0:
            score += 15
        elif rr_tp1 >= 1.5:
            score += 10
        elif rr_tp1 >= 1.0:
            score += 5
        
        # RSI context
        if direction == 'LONG' and rsi < 35:
            score += 10
        elif direction == 'SHORT' and rsi > 65:
            score += 10
        
        # Signal alignment
        score += min(signal_count * 3, 12)
        
        # Structure richness
        if support_count >= 3 and resistance_count >= 3:
            score += 5
        
        return min(score, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# WRAPPER FOR BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# Keep old class name working
class SignalGenerator:
    """Wrapper for backward compatibility - uses V2 logic internally"""
    
    @staticmethod
    def generate_signal(df: pd.DataFrame, symbol: str, timeframe: str = '1h', 
                       force_direction: str = None, explosion_score: int = 0,
                       trade_mode: str = None, htf_levels: list = None,
                       tp3_conditions: dict = None):
        """
        Generate signal using V2 logic with explosion-based targeting.
        
        Args:
            tp3_conditions: Dict with TP3 condition flags (for GREEN explosion):
                {
                    'bb_expanding': bool,      # BB width expanding?
                    'oi_rising': bool,         # OI holding or rising?
                    'price_above_vwap': bool,  # Price above VWAP/range mid?
                    'oi_collapsing': bool,     # Downgrade: OI dropping fast?
                    'funding_extreme': bool,   # Downgrade: Funding at extreme?
                    'delta_divergence': bool,  # Downgrade: Delta not confirming?
                }
        """
        setup = SignalGeneratorV2.generate_signal(
            df, symbol, timeframe, 
            force_direction=force_direction,
            explosion_score=explosion_score,
            trade_mode=trade_mode,
            htf_levels=htf_levels,
            tp3_conditions=tp3_conditions  # Pass TP3 conditions
        )
        
        if setup is None or not setup.is_valid:
            return None
        
        # Convert to TradeSignal format for compatibility
        signal = TradeSignal(
            symbol=setup.symbol,
            direction=setup.direction,
            entry=setup.entry,
            stop_loss=setup.stop_loss,
            tp1=setup.tp1,
            tp2=setup.tp2,
            tp3=setup.tp3,
            rr_ratio=setup.rr_tp2,
            risk_pct=setup.sl_pct,
            confidence=setup.confidence,
            timeframe=setup.timeframe,
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            rr_tp1=setup.rr_tp1,
            rr_tp2=setup.rr_tp2,
            rr_tp3=setup.rr_tp3,
            entry_reason=setup.entry_reason,
            sl_reason=setup.sl_reason,
            tp1_reason=setup.tp1_reason,
            tp2_reason=setup.tp2_reason,
            tp3_reason=setup.tp3_reason,
            sl_type=setup.sl_type,
            structure_levels=setup.structure_levels  # Pass through for limit entry
        )
        
        # Add explosion targeting info to signal (for display)
        signal.explosion_color = setup.explosion_color
        signal.trade_mode_label = setup.trade_mode_label
        signal.targets_allowed = setup.targets_allowed
        
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_trend(df: pd.DataFrame) -> dict:
    """Analyze trend direction and strength"""
    if df is None or len(df) < 50:
        return {'trend': 'Unknown', 'strength': 0}
    
    ema_20 = calculate_ema(df['Close'], 20).iloc[-1]
    ema_50 = calculate_ema(df['Close'], 50).iloc[-1]
    current = df['Close'].iloc[-1]
    
    if current > ema_20 > ema_50:
        trend = 'Bullish'
        strength = min(100, int((current / ema_50 - 1) * 500))
    elif current < ema_20 < ema_50:
        trend = 'Bearish'
        strength = min(100, int((1 - current / ema_50) * 500))
    else:
        trend = 'Neutral'
        strength = 50
    
    return {'trend': trend, 'strength': strength}


def analyze_momentum(df: pd.DataFrame) -> dict:
    """Analyze momentum indicators"""
    if df is None or len(df) < 20:
        return {'rsi': 50, 'macd': 'Neutral', 'momentum': 'Neutral'}
    
    rsi = calculate_rsi(df['Close'], 14).iloc[-1]
    macd_line, signal_line, _ = calculate_macd(df['Close'])
    
    macd_status = 'Bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'Bearish'
    
    if rsi > 70: momentum = 'Overbought'
    elif rsi < 30: momentum = 'Oversold'
    elif rsi > 50: momentum = 'Bullish'
    else: momentum = 'Bearish'
    
    return {'rsi': rsi, 'macd': macd_status, 'momentum': momentum}