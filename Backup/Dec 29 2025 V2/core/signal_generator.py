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
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION BY MODE
# ═══════════════════════════════════════════════════════════════════════════════

MODE_CONFIG = {
    'Scalp': {
        'timeframes': ['1m', '5m'],
        'max_sl_pct': 2.0,       # Max 2% stop loss
        'min_rr': 1.5,           # Minimum R:R for TP1
        'lookback_candles': 20,  # How far back to look for structure
        'anti_hunt_pct': 0.3,    # Buffer below/above levels
    },
    'DayTrade': {
        'timeframes': ['15m', '1h'],
        'max_sl_pct': 4.0,
        'min_rr': 1.2,
        'lookback_candles': 30,
        'anti_hunt_pct': 0.5,
    },
    'Swing': {
        'timeframes': ['4h', '1d'],
        'max_sl_pct': 8.0,
        'min_rr': 1.0,
        'lookback_candles': 50,
        'anti_hunt_pct': 0.8,
    },
    'Investment': {
        'timeframes': ['1w'],
        'max_sl_pct': 15.0,
        'min_rr': 0.8,
        'lookback_candles': 52,  # ~1 year of weekly candles
        'anti_hunt_pct': 1.0,
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
    # 2. ORDER BLOCKS
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(len(df_recent) - lookback, len(df_recent) - 3):
        if i < 0:
            continue
            
        try:
            # Bullish OB: Red candle followed by strong bullish move
            if df_recent['Close'].iloc[i] < df_recent['Open'].iloc[i]:
                next_move = df_recent['High'].iloc[i+1:i+4].max() - df_recent['Close'].iloc[i]
                if next_move > atr * 1.5:
                    ob_price = df_recent['Close'].iloc[i]  # Bottom of OB
                    dist_pct = ((current_price - ob_price) / current_price) * 100
                    
                    level = StructureLevel(
                        price=ob_price,
                        level_type='bullish_ob',
                        strength=4,
                        description=f"Bullish OB",
                        distance_pct=abs(dist_pct)
                    )
                    
                    if ob_price < current_price:
                        support_levels.append(level)
                    else:
                        resistance_levels.append(level)
            
            # Bearish OB: Green candle followed by strong bearish move
            if df_recent['Close'].iloc[i] > df_recent['Open'].iloc[i]:
                next_move = df_recent['Close'].iloc[i] - df_recent['Low'].iloc[i+1:i+4].min()
                if next_move > atr * 1.5:
                    ob_price = df_recent['Close'].iloc[i]  # Top of OB
                    dist_pct = ((ob_price - current_price) / current_price) * 100
                    
                    level = StructureLevel(
                        price=ob_price,
                        level_type='bearish_ob',
                        strength=4,
                        description=f"Bearish OB",
                        distance_pct=abs(dist_pct)
                    )
                    
                    if ob_price > current_price:
                        resistance_levels.append(level)
                    else:
                        support_levels.append(level)
        except:
            continue
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. FAIR VALUE GAPS
    # ─────────────────────────────────────────────────────────────────────────
    for i in range(len(df_recent) - lookback, len(df_recent) - 2):
        if i < 0:
            continue
            
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
    def generate_signal(df: pd.DataFrame, symbol: str, timeframe: str = '1h', force_direction: str = None) -> Optional[TradeSetup]:
        """
        Generate a trade signal based on actual chart structure.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Chart timeframe
            force_direction: If set to 'LONG' or 'SHORT', override technical direction
                           Used when predictive system indicates different direction
        
        Returns None if:
        - No valid structure for SL
        - No valid structure for TP
        - R:R doesn't meet minimum
        - SL exceeds max for mode
        """
        if df is None or len(df) < 20:
            return None
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # SETUP
            # ═══════════════════════════════════════════════════════════════════
            current_price = df['Close'].iloc[-1]
            config = get_mode_config(timeframe)
            
            MAX_SL_PCT = config['max_sl_pct']
            MIN_RR = config['min_rr']
            LOOKBACK = config['lookback_candles']
            ANTI_HUNT_PCT = config['anti_hunt_pct']
            
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
            
            # If force_direction is set, use that instead of technical analysis
            if force_direction in ['LONG', 'SHORT']:
                direction = force_direction
            else:
                # Original technical direction logic
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
                
                direction = 'LONG' if bullish_signals >= bearish_signals else 'SHORT'
            
            # ═══════════════════════════════════════════════════════════════════
            # BUILD TRADE SETUP
            # ═══════════════════════════════════════════════════════════════════
            entry = current_price
            entry_reason = "Current price"
            
            if direction == 'LONG':
                # ─────────────────────────────────────────────────────────────────
                # LONG: SL below support, TP at resistance
                # ─────────────────────────────────────────────────────────────────
                
                # Find best SL level (nearest valid support)
                if not valid_support:
                    return SignalGeneratorV2._create_invalid_setup(
                        symbol, direction, entry, timeframe, config['mode'],
                        f"No support within {MAX_SL_PCT}% for SL"
                    )
                
                sl_level = valid_support[0]  # Nearest support
                # Apply anti-hunt buffer
                stop_loss = sl_level.price * (1 - ANTI_HUNT_PCT / 100)
                sl_reason = f"{sl_level.description} (-{ANTI_HUNT_PCT}% buffer)"
                sl_type = sl_level.level_type
                
                # Calculate actual SL %
                sl_pct = ((entry - stop_loss) / entry) * 100
                
                # Double-check SL is within limits
                if sl_pct > MAX_SL_PCT:
                    stop_loss = entry * (1 - MAX_SL_PCT / 100)
                    sl_pct = MAX_SL_PCT
                    sl_reason = f"Capped at {MAX_SL_PCT}% (mode limit)"
                    sl_type = "max_cap"
                
                # Risk amount
                risk = entry - stop_loss
                
                # Find TP levels (resistance above)
                if not valid_resistance:
                    # Use ATR-based fallback for TPs only
                    atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
                    tp1 = entry + (atr * 1.5)
                    tp2 = entry + (atr * 2.5)
                    tp3 = entry + (atr * 4.0)
                    tp1_reason = "ATR 1.5x (no structure)"
                    tp2_reason = "ATR 2.5x (no structure)"
                    tp3_reason = "ATR 4.0x (no structure)"
                else:
                    # Use actual structure levels
                    tp1 = valid_resistance[0].price * 0.998 if len(valid_resistance) > 0 else entry + (risk * 1.5)
                    tp1_reason = valid_resistance[0].description if len(valid_resistance) > 0 else "R:R 1.5"
                    
                    tp2 = valid_resistance[1].price * 0.998 if len(valid_resistance) > 1 else entry + (risk * 2.5)
                    tp2_reason = valid_resistance[1].description if len(valid_resistance) > 1 else "Extended"
                    
                    tp3 = valid_resistance[2].price * 0.998 if len(valid_resistance) > 2 else entry + (risk * 4.0)
                    tp3_reason = valid_resistance[2].description if len(valid_resistance) > 2 else "Max target"
                
            else:
                # ─────────────────────────────────────────────────────────────────
                # SHORT: SL above resistance, TP at support
                # ─────────────────────────────────────────────────────────────────
                
                # For shorts, we need resistance above for SL
                valid_resistance_for_sl = [l for l in levels['resistance'] if l.distance_pct <= MAX_SL_PCT]
                
                if not valid_resistance_for_sl:
                    return SignalGeneratorV2._create_invalid_setup(
                        symbol, direction, entry, timeframe, config['mode'],
                        f"No resistance within {MAX_SL_PCT}% for SL"
                    )
                
                sl_level = valid_resistance_for_sl[0]
                stop_loss = sl_level.price * (1 + ANTI_HUNT_PCT / 100)
                sl_reason = f"{sl_level.description} (+{ANTI_HUNT_PCT}% buffer)"
                sl_type = sl_level.level_type
                
                sl_pct = ((stop_loss - entry) / entry) * 100
                
                if sl_pct > MAX_SL_PCT:
                    stop_loss = entry * (1 + MAX_SL_PCT / 100)
                    sl_pct = MAX_SL_PCT
                    sl_reason = f"Capped at {MAX_SL_PCT}% (mode limit)"
                    sl_type = "max_cap"
                
                risk = stop_loss - entry
                
                # TPs at support below
                if not valid_support:
                    atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]
                    tp1 = entry - (atr * 1.5)
                    tp2 = entry - (atr * 2.5)
                    tp3 = entry - (atr * 4.0)
                    tp1_reason = "ATR 1.5x (no structure)"
                    tp2_reason = "ATR 2.5x (no structure)"
                    tp3_reason = "ATR 4.0x (no structure)"
                else:
                    tp1 = valid_support[0].price * 1.002 if len(valid_support) > 0 else entry - (risk * 1.5)
                    tp1_reason = valid_support[0].description if len(valid_support) > 0 else "R:R 1.5"
                    
                    tp2 = valid_support[1].price * 1.002 if len(valid_support) > 1 else entry - (risk * 2.5)
                    tp2_reason = valid_support[1].description if len(valid_support) > 1 else "Extended"
                    
                    tp3 = valid_support[2].price * 1.002 if len(valid_support) > 2 else entry - (risk * 4.0)
                    tp3_reason = valid_support[2].description if len(valid_support) > 2 else "Max target"
            
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
                trade_mode=config['mode'],
                confidence=confidence,
                is_valid=is_valid,
                rejection_reason=rejection_reason
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
    def generate_signal(df: pd.DataFrame, symbol: str, timeframe: str = '1h', force_direction: str = None):
        """Generate signal using V2 logic"""
        setup = SignalGeneratorV2.generate_signal(df, symbol, timeframe, force_direction=force_direction)
        
        if setup is None or not setup.is_valid:
            return None
        
        # Convert to TradeSignal format for compatibility
        return TradeSignal(
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
            sl_type=setup.sl_type
        )


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