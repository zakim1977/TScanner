"""
Professional Level Calculator
SMC-Based Entry, SL, TP Calculation with Structure Analysis

This module calculates trade levels based on:
- Order Blocks (institutional entry zones)
- Fair Value Gaps (imbalance zones)
- Support/Resistance (swing highs/lows)
- Liquidity Pools (stop hunt levels)
- ATR (only as backup/buffer)

Each level includes reasoning for transparency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PriceLevel:
    """A price level with metadata"""
    price: float
    level_type: str  # 'ob', 'fvg', 'sr', 'liquidity', 'atr'
    strength: int  # 1-5 (5 = strongest)
    description: str
    zone_top: float = 0
    zone_bottom: float = 0


@dataclass 
class TradeLevels:
    """Complete trade setup with reasoning"""
    direction: str  # 'LONG' or 'SHORT'
    
    # Entry
    entry: float
    entry_reason: str
    entry_zone_top: float
    entry_zone_bottom: float
    
    # Stop Loss
    stop_loss: float
    sl_reason: str
    sl_type: str  # 'below_ob', 'below_fvg', 'below_support', 'atr_fallback'
    
    # Take Profits
    tp1: float
    tp1_reason: str
    tp1_type: str
    
    tp2: float
    tp2_reason: str
    tp2_type: str
    
    tp3: float
    tp3_reason: str
    tp3_type: str
    
    # Risk metrics
    risk_pct: float
    rr_tp1: float
    rr_tp2: float
    rr_tp3: float
    
    # Confidence
    confidence: int  # 0-100
    confidence_reasons: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[PriceLevel]]:
    """
    Find Order Blocks - last opposing candle before strong move
    
    Bullish OB: Last red candle before strong green move (demand)
    Bearish OB: Last green candle before strong red move (supply)
    """
    bullish_obs = []
    bearish_obs = []
    
    if df is None or len(df) < lookback:
        return {'bullish': bullish_obs, 'bearish': bearish_obs}
    
    atr = (df['High'] - df['Low']).rolling(14).mean()
    
    for i in range(len(df) - lookback, len(df) - 3):
        # Bullish OB: Bearish candle followed by strong bullish move
        if df['Close'].iloc[i] < df['Open'].iloc[i]:  # Red candle
            # Check for strong bullish move after (at least 2x ATR)
            next_move = df['High'].iloc[i+1:i+4].max() - df['Close'].iloc[i]
            if next_move > atr.iloc[i] * 2:
                strength = min(5, int(next_move / atr.iloc[i]))
                bullish_obs.append(PriceLevel(
                    price=(df['Open'].iloc[i] + df['Close'].iloc[i]) / 2,
                    level_type='ob_bullish',
                    strength=strength,
                    description=f"Bullish OB ({strength}/5) - Institutional demand zone",
                    zone_top=df['Open'].iloc[i],
                    zone_bottom=df['Close'].iloc[i]
                ))
        
        # Bearish OB: Bullish candle followed by strong bearish move
        if df['Close'].iloc[i] > df['Open'].iloc[i]:  # Green candle
            next_move = df['Close'].iloc[i] - df['Low'].iloc[i+1:i+4].min()
            if next_move > atr.iloc[i] * 2:
                strength = min(5, int(next_move / atr.iloc[i]))
                bearish_obs.append(PriceLevel(
                    price=(df['Open'].iloc[i] + df['Close'].iloc[i]) / 2,
                    level_type='ob_bearish',
                    strength=strength,
                    description=f"Bearish OB ({strength}/5) - Institutional supply zone",
                    zone_top=df['Close'].iloc[i],
                    zone_bottom=df['Open'].iloc[i]
                ))
    
    # Sort by recency (most recent first) and strength
    bullish_obs.sort(key=lambda x: x.strength, reverse=True)
    bearish_obs.sort(key=lambda x: x.strength, reverse=True)
    
    return {'bullish': bullish_obs[:5], 'bearish': bearish_obs[:5]}


def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 30) -> Dict[str, List[PriceLevel]]:
    """
    Find Fair Value Gaps - imbalance zones
    
    Bullish FVG: Low[i+2] > High[i] (gap up)
    Bearish FVG: High[i+2] < Low[i] (gap down)
    """
    bullish_fvgs = []
    bearish_fvgs = []
    
    if df is None or len(df) < lookback:
        return {'bullish': bullish_fvgs, 'bearish': bearish_fvgs}
    
    for i in range(len(df) - lookback, len(df) - 2):
        # Bullish FVG
        if df['Low'].iloc[i+2] > df['High'].iloc[i]:
            gap_size = df['Low'].iloc[i+2] - df['High'].iloc[i]
            avg_range = (df['High'] - df['Low']).iloc[i-10:i].mean()
            strength = min(5, int(gap_size / avg_range * 2) + 1)
            
            bullish_fvgs.append(PriceLevel(
                price=(df['Low'].iloc[i+2] + df['High'].iloc[i]) / 2,
                level_type='fvg_bullish',
                strength=strength,
                description=f"Bullish FVG - Price magnet (unfilled gap)",
                zone_top=df['Low'].iloc[i+2],
                zone_bottom=df['High'].iloc[i]
            ))
        
        # Bearish FVG
        if df['High'].iloc[i+2] < df['Low'].iloc[i]:
            gap_size = df['Low'].iloc[i] - df['High'].iloc[i+2]
            avg_range = (df['High'] - df['Low']).iloc[i-10:i].mean()
            strength = min(5, int(gap_size / avg_range * 2) + 1)
            
            bearish_fvgs.append(PriceLevel(
                price=(df['Low'].iloc[i] + df['High'].iloc[i+2]) / 2,
                level_type='fvg_bearish',
                strength=strength,
                description=f"Bearish FVG - Price magnet (unfilled gap)",
                zone_top=df['Low'].iloc[i],
                zone_bottom=df['High'].iloc[i+2]
            ))
    
    return {'bullish': bullish_fvgs[:5], 'bearish': bearish_fvgs[:5]}


def find_swing_levels(df: pd.DataFrame, lookback: int = 5) -> Dict[str, List[PriceLevel]]:
    """
    Find swing highs and lows (support/resistance)
    """
    supports = []
    resistances = []
    
    if df is None or len(df) < lookback * 3:
        return {'support': supports, 'resistance': resistances}
    
    for i in range(lookback, len(df) - lookback):
        # Swing high (resistance)
        if df['High'].iloc[i] == df['High'].iloc[i-lookback:i+lookback+1].max():
            # Count how many times this level was tested
            level = df['High'].iloc[i]
            tests = sum(1 for j in range(len(df)) if abs(df['High'].iloc[j] - level) / level < 0.005)
            strength = min(5, tests)
            
            resistances.append(PriceLevel(
                price=level,
                level_type='resistance',
                strength=strength,
                description=f"Swing High - Resistance (tested {tests}x)",
                zone_top=level * 1.002,
                zone_bottom=level * 0.998
            ))
        
        # Swing low (support)
        if df['Low'].iloc[i] == df['Low'].iloc[i-lookback:i+lookback+1].min():
            level = df['Low'].iloc[i]
            tests = sum(1 for j in range(len(df)) if abs(df['Low'].iloc[j] - level) / level < 0.005)
            strength = min(5, tests)
            
            supports.append(PriceLevel(
                price=level,
                level_type='support',
                strength=strength,
                description=f"Swing Low - Support (tested {tests}x)",
                zone_top=level * 1.002,
                zone_bottom=level * 0.998
            ))
    
    # Remove duplicates (levels within 0.5% of each other)
    supports = _dedupe_levels(supports)
    resistances = _dedupe_levels(resistances)
    
    return {'support': supports[:10], 'resistance': resistances[:10]}


def find_liquidity_pools(df: pd.DataFrame, lookback: int = 20) -> Dict[str, List[PriceLevel]]:
    """
    Find liquidity pools - areas where stops are likely clustered
    
    Below equal lows = buy-side liquidity (stops of shorts)
    Above equal highs = sell-side liquidity (stops of longs)
    """
    buy_liquidity = []  # Below price (long stop targets)
    sell_liquidity = []  # Above price (short stop targets)
    
    if df is None or len(df) < lookback:
        return {'buy': buy_liquidity, 'sell': sell_liquidity}
    
    # Find equal highs (sell liquidity above)
    for i in range(len(df) - lookback, len(df) - 3):
        for j in range(i + 2, min(i + 10, len(df))):
            if abs(df['High'].iloc[i] - df['High'].iloc[j]) / df['High'].iloc[i] < 0.003:
                sell_liquidity.append(PriceLevel(
                    price=max(df['High'].iloc[i], df['High'].iloc[j]),
                    level_type='liquidity_sell',
                    strength=3,
                    description="Equal Highs - Sell-side liquidity (stop hunt target)",
                    zone_top=max(df['High'].iloc[i], df['High'].iloc[j]) * 1.005,
                    zone_bottom=max(df['High'].iloc[i], df['High'].iloc[j])
                ))
                break
    
    # Find equal lows (buy liquidity below)
    for i in range(len(df) - lookback, len(df) - 3):
        for j in range(i + 2, min(i + 10, len(df))):
            if abs(df['Low'].iloc[i] - df['Low'].iloc[j]) / df['Low'].iloc[i] < 0.003:
                buy_liquidity.append(PriceLevel(
                    price=min(df['Low'].iloc[i], df['Low'].iloc[j]),
                    level_type='liquidity_buy',
                    strength=3,
                    description="Equal Lows - Buy-side liquidity (stop hunt target)",
                    zone_top=min(df['Low'].iloc[i], df['Low'].iloc[j]),
                    zone_bottom=min(df['Low'].iloc[i], df['Low'].iloc[j]) * 0.995
                ))
                break
    
    return {'buy': _dedupe_levels(buy_liquidity), 'sell': _dedupe_levels(sell_liquidity)}


def _dedupe_levels(levels: List[PriceLevel], tolerance: float = 0.005) -> List[PriceLevel]:
    """Remove duplicate levels within tolerance"""
    if not levels:
        return levels
    
    # Sort by price
    levels.sort(key=lambda x: x.price)
    
    deduped = [levels[0]]
    for level in levels[1:]:
        if abs(level.price - deduped[-1].price) / deduped[-1].price > tolerance:
            deduped.append(level)
        elif level.strength > deduped[-1].strength:
            deduped[-1] = level  # Keep stronger level
    
    return deduped


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LEVEL CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_smart_levels(df: pd.DataFrame, direction: str = 'LONG') -> TradeLevels:
    """
    Calculate professional trade levels based on market structure
    
    Args:
        df: OHLCV DataFrame
        direction: 'LONG' or 'SHORT'
        
    Returns:
        TradeLevels object with entry, SL, TPs and reasoning
    """
    if df is None or len(df) < 50:
        return _fallback_levels(df, direction)
    
    current_price = df['Close'].iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    
    # Gather all levels
    obs = find_order_blocks(df)
    fvgs = find_fair_value_gaps(df)
    swing = find_swing_levels(df)
    liquidity = find_liquidity_pools(df)
    
    confidence_reasons = []
    confidence = 50  # Base confidence
    
    if direction == 'LONG':
        # ═══════════════════════════════════════════════════════════════════
        # LONG SETUP
        # ═══════════════════════════════════════════════════════════════════
        
        # --- ENTRY ---
        entry = current_price
        entry_reason = "Current price"
        entry_zone_top = current_price
        entry_zone_bottom = current_price
        
        # Check if we're at a bullish OB
        for ob in obs['bullish']:
            if ob.zone_bottom <= current_price <= ob.zone_top * 1.01:
                entry = current_price
                entry_reason = f"At {ob.description}"
                entry_zone_top = ob.zone_top
                entry_zone_bottom = ob.zone_bottom
                confidence += 15
                confidence_reasons.append(f"✅ Entry at Bullish Order Block")
                break
        
        # Check if we're at a bullish FVG
        for fvg in fvgs['bullish']:
            if fvg.zone_bottom <= current_price <= fvg.zone_top * 1.01:
                if "Order Block" not in entry_reason:
                    entry_reason = f"At {fvg.description}"
                    entry_zone_top = fvg.zone_top
                    entry_zone_bottom = fvg.zone_bottom
                confidence += 10
                confidence_reasons.append(f"✅ Entry at Bullish FVG")
                break
        
        # Check if we're at support
        for sup in swing['support']:
            if sup.zone_bottom * 0.99 <= current_price <= sup.zone_top * 1.02:
                if "Order Block" not in entry_reason and "FVG" not in entry_reason:
                    entry_reason = f"At {sup.description}"
                confidence += 10
                confidence_reasons.append(f"✅ Entry at Support level")
                break
        
        # --- STOP LOSS ---
        sl_candidates = []
        
        # Priority 1: Below nearest bullish OB
        for ob in obs['bullish']:
            if ob.zone_bottom < current_price:
                sl_candidates.append({
                    'price': ob.zone_bottom * 0.995,  # Small buffer below OB
                    'reason': f"Below {ob.description}",
                    'type': 'below_ob',
                    'priority': 1
                })
        
        # Priority 2: Below bullish FVG
        for fvg in fvgs['bullish']:
            if fvg.zone_bottom < current_price:
                sl_candidates.append({
                    'price': fvg.zone_bottom * 0.995,
                    'reason': f"Below {fvg.description}",
                    'type': 'below_fvg',
                    'priority': 2
                })
        
        # Priority 3: Below support
        for sup in swing['support']:
            if sup.price < current_price:
                sl_candidates.append({
                    'price': sup.price * 0.995,
                    'reason': f"Below {sup.description}",
                    'type': 'below_support',
                    'priority': 3
                })
        
        # Priority 4: Below liquidity (let them sweep first)
        for liq in liquidity['buy']:
            if liq.price < current_price:
                sl_candidates.append({
                    'price': liq.zone_bottom * 0.995,
                    'reason': f"Below {liq.description}",
                    'type': 'below_liquidity',
                    'priority': 4
                })
        
        # Select best SL
        if sl_candidates:
            # Sort by priority, then by distance (closer is better for R:R)
            sl_candidates.sort(key=lambda x: (x['priority'], current_price - x['price']))
            best_sl = sl_candidates[0]
            stop_loss = best_sl['price']
            sl_reason = best_sl['reason']
            sl_type = best_sl['type']
            
            if sl_type == 'below_ob':
                confidence += 15
                confidence_reasons.append("✅ SL below Order Block (protected)")
            elif sl_type == 'below_fvg':
                confidence += 10
                confidence_reasons.append("✅ SL below FVG")
        else:
            # Fallback: ATR-based
            stop_loss = current_price - (atr * 2)
            sl_reason = "ATR-based (no structure found)"
            sl_type = 'atr_fallback'
            confidence -= 10
            confidence_reasons.append("⚠️ SL using ATR fallback (structure unclear)")
        
        # --- TAKE PROFITS ---
        tp_candidates = []
        
        # Bearish OBs overhead (supply zones)
        for ob in obs['bearish']:
            if ob.zone_bottom > current_price:
                tp_candidates.append({
                    'price': ob.zone_bottom,  # Bottom of supply zone
                    'reason': f"At {ob.description}",
                    'type': 'bearish_ob',
                    'strength': ob.strength
                })
        
        # Bearish FVGs overhead
        for fvg in fvgs['bearish']:
            if fvg.zone_bottom > current_price:
                tp_candidates.append({
                    'price': fvg.zone_bottom,
                    'reason': f"At {fvg.description}",
                    'type': 'bearish_fvg',
                    'strength': fvg.strength
                })
        
        # Resistance levels
        for res in swing['resistance']:
            if res.price > current_price:
                tp_candidates.append({
                    'price': res.price * 0.998,  # Just before resistance
                    'reason': f"At {res.description}",
                    'type': 'resistance',
                    'strength': res.strength
                })
        
        # Sell liquidity (equal highs)
        for liq in liquidity['sell']:
            if liq.price > current_price:
                tp_candidates.append({
                    'price': liq.zone_top,  # Top of liquidity zone
                    'reason': f"At {liq.description}",
                    'type': 'liquidity',
                    'strength': 3
                })
        
        # Sort by distance
        tp_candidates.sort(key=lambda x: x['price'])
        
        # Assign TPs
        risk = entry - stop_loss
        
        if len(tp_candidates) >= 1:
            tp1 = tp_candidates[0]['price']
            tp1_reason = tp_candidates[0]['reason']
            tp1_type = tp_candidates[0]['type']
            confidence += 5
            confidence_reasons.append(f"✅ TP1 at structure level")
        else:
            tp1 = entry + (risk * 1.5)
            tp1_reason = "R:R 1.5 (no structure found)"
            tp1_type = 'rr_fallback'
        
        if len(tp_candidates) >= 2:
            tp2 = tp_candidates[1]['price']
            tp2_reason = tp_candidates[1]['reason']
            tp2_type = tp_candidates[1]['type']
        else:
            tp2 = entry + (risk * 2.5)
            tp2_reason = "R:R 2.5 (extended target)"
            tp2_type = 'rr_fallback'
        
        if len(tp_candidates) >= 3:
            tp3 = tp_candidates[2]['price']
            tp3_reason = tp_candidates[2]['reason']
            tp3_type = tp_candidates[2]['type']
        else:
            tp3 = entry + (risk * 4.0)
            tp3_reason = "R:R 4.0 (runner target)"
            tp3_type = 'rr_fallback'
    
    else:
        # ═══════════════════════════════════════════════════════════════════
        # SHORT SETUP
        # ═══════════════════════════════════════════════════════════════════
        
        # --- ENTRY ---
        entry = current_price
        entry_reason = "Current price"
        entry_zone_top = current_price
        entry_zone_bottom = current_price
        
        # Check if we're at a bearish OB
        for ob in obs['bearish']:
            if ob.zone_bottom * 0.99 <= current_price <= ob.zone_top:
                entry = current_price
                entry_reason = f"At {ob.description}"
                entry_zone_top = ob.zone_top
                entry_zone_bottom = ob.zone_bottom
                confidence += 15
                confidence_reasons.append(f"✅ Entry at Bearish Order Block")
                break
        
        # Check if we're at a bearish FVG
        for fvg in fvgs['bearish']:
            if fvg.zone_bottom * 0.99 <= current_price <= fvg.zone_top:
                if "Order Block" not in entry_reason:
                    entry_reason = f"At {fvg.description}"
                    entry_zone_top = fvg.zone_top
                    entry_zone_bottom = fvg.zone_bottom
                confidence += 10
                confidence_reasons.append(f"✅ Entry at Bearish FVG")
                break
        
        # Check if we're at resistance
        for res in swing['resistance']:
            if res.zone_bottom * 0.98 <= current_price <= res.zone_top * 1.01:
                if "Order Block" not in entry_reason and "FVG" not in entry_reason:
                    entry_reason = f"At {res.description}"
                confidence += 10
                confidence_reasons.append(f"✅ Entry at Resistance level")
                break
        
        # --- STOP LOSS ---
        sl_candidates = []
        
        # Priority 1: Above nearest bearish OB
        for ob in obs['bearish']:
            if ob.zone_top > current_price:
                sl_candidates.append({
                    'price': ob.zone_top * 1.005,
                    'reason': f"Above {ob.description}",
                    'type': 'above_ob',
                    'priority': 1
                })
        
        # Priority 2: Above bearish FVG
        for fvg in fvgs['bearish']:
            if fvg.zone_top > current_price:
                sl_candidates.append({
                    'price': fvg.zone_top * 1.005,
                    'reason': f"Above {fvg.description}",
                    'type': 'above_fvg',
                    'priority': 2
                })
        
        # Priority 3: Above resistance
        for res in swing['resistance']:
            if res.price > current_price:
                sl_candidates.append({
                    'price': res.price * 1.005,
                    'reason': f"Above {res.description}",
                    'type': 'above_resistance',
                    'priority': 3
                })
        
        # Select best SL
        if sl_candidates:
            sl_candidates.sort(key=lambda x: (x['priority'], x['price'] - current_price))
            best_sl = sl_candidates[0]
            stop_loss = best_sl['price']
            sl_reason = best_sl['reason']
            sl_type = best_sl['type']
            
            if sl_type == 'above_ob':
                confidence += 15
                confidence_reasons.append("✅ SL above Order Block (protected)")
        else:
            stop_loss = current_price + (atr * 2)
            sl_reason = "ATR-based (no structure found)"
            sl_type = 'atr_fallback'
            confidence -= 10
            confidence_reasons.append("⚠️ SL using ATR fallback")
        
        # --- TAKE PROFITS ---
        tp_candidates = []
        
        # Bullish OBs below (demand zones)
        for ob in obs['bullish']:
            if ob.zone_top < current_price:
                tp_candidates.append({
                    'price': ob.zone_top,
                    'reason': f"At {ob.description}",
                    'type': 'bullish_ob',
                    'strength': ob.strength
                })
        
        # Bullish FVGs below
        for fvg in fvgs['bullish']:
            if fvg.zone_top < current_price:
                tp_candidates.append({
                    'price': fvg.zone_top,
                    'reason': f"At {fvg.description}",
                    'type': 'bullish_fvg',
                    'strength': fvg.strength
                })
        
        # Support levels
        for sup in swing['support']:
            if sup.price < current_price:
                tp_candidates.append({
                    'price': sup.price * 1.002,
                    'reason': f"At {sup.description}",
                    'type': 'support',
                    'strength': sup.strength
                })
        
        # Sort by distance (closest first)
        tp_candidates.sort(key=lambda x: x['price'], reverse=True)
        
        # Assign TPs
        risk = stop_loss - entry
        
        if len(tp_candidates) >= 1:
            tp1 = tp_candidates[0]['price']
            tp1_reason = tp_candidates[0]['reason']
            tp1_type = tp_candidates[0]['type']
            confidence += 5
        else:
            tp1 = entry - (risk * 1.5)
            tp1_reason = "R:R 1.5 (no structure found)"
            tp1_type = 'rr_fallback'
        
        if len(tp_candidates) >= 2:
            tp2 = tp_candidates[1]['price']
            tp2_reason = tp_candidates[1]['reason']
            tp2_type = tp_candidates[1]['type']
        else:
            tp2 = entry - (risk * 2.5)
            tp2_reason = "R:R 2.5 (extended target)"
            tp2_type = 'rr_fallback'
        
        if len(tp_candidates) >= 3:
            tp3 = tp_candidates[2]['price']
            tp3_reason = tp_candidates[2]['reason']
            tp3_type = tp_candidates[2]['type']
        else:
            tp3 = entry - (risk * 4.0)
            tp3_reason = "R:R 4.0 (runner target)"
            tp3_type = 'rr_fallback'
    
    # Calculate risk metrics
    risk_amount = abs(entry - stop_loss)
    risk_pct = (risk_amount / entry) * 100
    
    rr_tp1 = abs(tp1 - entry) / risk_amount if risk_amount > 0 else 0
    rr_tp2 = abs(tp2 - entry) / risk_amount if risk_amount > 0 else 0
    rr_tp3 = abs(tp3 - entry) / risk_amount if risk_amount > 0 else 0
    
    # Confidence adjustments
    if rr_tp1 < 1:
        confidence -= 10
        confidence_reasons.append("⚠️ TP1 less than 1:1 R:R")
    elif rr_tp1 >= 2:
        confidence += 5
        confidence_reasons.append("✅ Excellent R:R ratio")
    
    confidence = max(0, min(100, confidence))
    
    return TradeLevels(
        direction=direction,
        entry=entry,
        entry_reason=entry_reason,
        entry_zone_top=entry_zone_top,
        entry_zone_bottom=entry_zone_bottom,
        stop_loss=stop_loss,
        sl_reason=sl_reason,
        sl_type=sl_type,
        tp1=tp1,
        tp1_reason=tp1_reason,
        tp1_type=tp1_type,
        tp2=tp2,
        tp2_reason=tp2_reason,
        tp2_type=tp2_type,
        tp3=tp3,
        tp3_reason=tp3_reason,
        tp3_type=tp3_type,
        risk_pct=risk_pct,
        rr_tp1=rr_tp1,
        rr_tp2=rr_tp2,
        rr_tp3=rr_tp3,
        confidence=confidence,
        confidence_reasons=confidence_reasons
    )


def _fallback_levels(df: pd.DataFrame, direction: str) -> TradeLevels:
    """ATR-based fallback when not enough data"""
    if df is None or len(df) < 20:
        return None
    
    current = df['Close'].iloc[-1]
    atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
    
    if direction == 'LONG':
        sl = current - (atr * 2)
        risk = current - sl
        tp1 = current + (risk * 1.5)
        tp2 = current + (risk * 2.5)
        tp3 = current + (risk * 4.0)
    else:
        sl = current + (atr * 2)
        risk = sl - current
        tp1 = current - (risk * 1.5)
        tp2 = current - (risk * 2.5)
        tp3 = current - (risk * 4.0)
    
    return TradeLevels(
        direction=direction,
        entry=current,
        entry_reason="Current price (fallback)",
        entry_zone_top=current,
        entry_zone_bottom=current,
        stop_loss=sl,
        sl_reason="ATR x2 (fallback - no structure)",
        sl_type='atr_fallback',
        tp1=tp1,
        tp1_reason="R:R 1.5 (fallback)",
        tp1_type='rr_fallback',
        tp2=tp2,
        tp2_reason="R:R 2.5 (fallback)",
        tp2_type='rr_fallback',
        tp3=tp3,
        tp3_reason="R:R 4.0 (fallback)",
        tp3_type='rr_fallback',
        risk_pct=(abs(current - sl) / current) * 100,
        rr_tp1=1.5,
        rr_tp2=2.5,
        rr_tp3=4.0,
        confidence=30,
        confidence_reasons=["⚠️ Using ATR fallback - no clear structure"]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Get all levels for display
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_levels(df: pd.DataFrame) -> Dict:
    """Get all detected levels for charting"""
    return {
        'order_blocks': find_order_blocks(df),
        'fvg': find_fair_value_gaps(df),
        'swing': find_swing_levels(df),
        'liquidity': find_liquidity_pools(df)
    }
