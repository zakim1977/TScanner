"""
Trade Manager Module - Smart Trade Management
==============================================
Uses ta library to find STRUCTURAL levels for SL/TP
Integrates with existing InvestorIQ system

NO arbitrary ATR breakeven - only structure-based stops!
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# Import ta library indicators
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel
from ta.volume import VolumeWeightedAveragePrice


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmartLevel:
    """A price level with context"""
    price: float
    level_type: str  # 'order_block', 'fvg', 'support', 'resistance', 'ema', 'vwap', 'swing', 'atr'
    strength: int = 1  # 1-5, higher = stronger
    description: str = ""


@dataclass
class SmartTradeLevels:
    """Complete trade setup with smart levels (ta library based)"""
    # Entry
    entry: float = 0
    entry_type: str = ""
    entry_reason: str = ""
    
    # Stop Loss
    stop_loss: float = 0
    sl_type: str = ""
    sl_reason: str = ""
    risk_pct: float = 0
    
    # Take Profits
    tp1: float = 0
    tp1_type: str = ""
    tp1_reason: str = ""
    tp1_rr: float = 0
    
    tp2: float = 0
    tp2_type: str = ""
    tp2_reason: str = ""
    tp2_rr: float = 0
    
    tp3: float = 0
    tp3_type: str = ""
    tp3_reason: str = ""
    tp3_rr: float = 0
    
    # Direction
    direction: str = "LONG"
    
    # Quality
    confidence: int = 50
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class EnhancedTrade:
    """Trade with MAE/MFE tracking for future analysis"""
    # Identity
    id: str = ""
    symbol: str = ""
    timeframe: str = ""
    
    # Entry info
    entry_price: float = 0
    entry_time: str = ""
    direction: str = "LONG"
    grade: str = "A"
    confidence: int = 70
    
    # Levels
    stop_loss: float = 0
    sl_type: str = ""
    sl_reason: str = ""
    tp1: float = 0
    tp2: float = 0
    tp3: float = 0
    
    # Position
    position_size: float = 2500  # $2,500 default
    risk_amount: float = 0
    
    # MAE/MFE tracking (for future vectorbt optimization)
    highest_price: float = 0
    lowest_price: float = 0
    highest_profit_pct: float = 0  # MFE - Maximum Favorable Excursion
    worst_drawdown_pct: float = 0  # MAE - Maximum Adverse Excursion
    
    # Status tracking
    status: str = "ACTIVE"  # ACTIVE, TP1_HIT, TP2_HIT, PROFIT_PROTECTED, CLOSED
    tp1_hit: bool = False
    tp2_hit: bool = False
    
    # SL management (Option C - structural only)
    original_sl: float = 0
    current_sl: float = 0
    sl_moved: bool = False
    sl_move_reason: str = ""
    sl_move_history: str = "[]"  # JSON string
    
    # Exit info
    exit_price: float = 0
    exit_time: str = ""
    exit_reason: str = ""
    
    # P&L
    pnl_pct: float = 0
    pnl_dollars: float = 0
    
    # Confirmations (multiple signals on same asset)
    confirmation_count: int = 1
    last_confirmation: str = ""
    
    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EnhancedTrade':
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════════════
# SMART LEVEL FINDER - Uses ta library
# ═══════════════════════════════════════════════════════════════════════════════

class SmartLevelFinder:
    """
    Finds structural levels using ta library.
    Priority: OB > FVG > S/R > EMA > VWAP > ATR fallback
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.h = df['High']
        self.l = df['Low']
        self.c = df['Close']
        self.v = df['Volume']
        self.current_price = self.c.iloc[-1]
        
        # Pre-calculate indicators using ta library
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """Calculate all needed indicators using ta library"""
        # EMAs (ta library)
        self.ema_9 = EMAIndicator(close=self.c, window=9).ema_indicator()
        self.ema_20 = EMAIndicator(close=self.c, window=20).ema_indicator()
        self.ema_50 = EMAIndicator(close=self.c, window=50).ema_indicator()
        
        # ATR (ta library)
        atr_ind = AverageTrueRange(high=self.h, low=self.l, close=self.c, window=14)
        self.atr = atr_ind.average_true_range()
        self.current_atr = self.atr.iloc[-1] if len(self.atr) > 0 and not pd.isna(self.atr.iloc[-1]) else self.current_price * 0.02
        
        # Bollinger Bands (ta library)
        bb = BollingerBands(close=self.c, window=20, window_dev=2)
        self.bb_upper = bb.bollinger_hband()
        self.bb_lower = bb.bollinger_lband()
        
        # Donchian Channel for Support/Resistance (ta library)
        dc = DonchianChannel(high=self.h, low=self.l, close=self.c, window=20)
        self.dc_upper = dc.donchian_channel_hband()
        self.dc_lower = dc.donchian_channel_lband()
        
        # VWAP (ta library)
        try:
            vwap = VolumeWeightedAveragePrice(high=self.h, low=self.l, close=self.c, volume=self.v)
            self.vwap = vwap.volume_weighted_average_price()
        except:
            typical_price = (self.h + self.l + self.c) / 3
            cum_vol = self.v.cumsum()
            self.vwap = (typical_price * self.v).cumsum() / cum_vol.replace(0, 1)
    
    def find_swing_lows(self, lookback: int = 20, min_strength: int = 2) -> List[SmartLevel]:
        """Find swing lows (potential support)"""
        levels = []
        
        for i in range(lookback, len(self.df) - min_strength):
            is_swing_low = True
            low_price = self.l.iloc[i]
            
            for j in range(1, min_strength + 1):
                if i - j >= 0 and self.l.iloc[i - j] <= low_price:
                    is_swing_low = False
                    break
                if i + j < len(self.df) and self.l.iloc[i + j] <= low_price:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                levels.append(SmartLevel(
                    price=low_price,
                    level_type='swing_low',
                    strength=min_strength,
                    description=f"Swing low"
                ))
        
        return levels
    
    def find_swing_highs(self, lookback: int = 20, min_strength: int = 2) -> List[SmartLevel]:
        """Find swing highs (potential resistance)"""
        levels = []
        
        for i in range(lookback, len(self.df) - min_strength):
            is_swing_high = True
            high_price = self.h.iloc[i]
            
            for j in range(1, min_strength + 1):
                if i - j >= 0 and self.h.iloc[i - j] >= high_price:
                    is_swing_high = False
                    break
                if i + j < len(self.df) and self.h.iloc[i + j] >= high_price:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                levels.append(SmartLevel(
                    price=high_price,
                    level_type='swing_high',
                    strength=min_strength,
                    description=f"Swing high"
                ))
        
        return levels
    
    def find_order_blocks(self, lookback: int = 50) -> Dict[str, List[SmartLevel]]:
        """Find bullish and bearish order blocks"""
        bullish_obs = []
        bearish_obs = []
        
        for i in range(2, min(lookback, len(self.df) - 2)):
            idx = len(self.df) - 1 - i
            
            if idx < 1:
                continue
            
            curr_open = self.df['Open'].iloc[idx]
            curr_close = self.c.iloc[idx]
            curr_high = self.h.iloc[idx]
            curr_low = self.l.iloc[idx]
            
            next_open = self.df['Open'].iloc[idx + 1]
            next_close = self.c.iloc[idx + 1]
            
            # Bullish OB: Bearish candle followed by strong bullish move
            if curr_close < curr_open:  # Current is bearish
                if next_close > next_open and next_close > curr_high:  # Next is bullish breakout
                    bullish_obs.append(SmartLevel(
                        price=curr_low,
                        level_type='bullish_ob',
                        strength=4,
                        description=f"Bullish OB"
                    ))
            
            # Bearish OB: Bullish candle followed by strong bearish move
            if curr_close > curr_open:  # Current is bullish
                if next_close < next_open and next_close < curr_low:  # Next is bearish breakout
                    bearish_obs.append(SmartLevel(
                        price=curr_high,
                        level_type='bearish_ob',
                        strength=4,
                        description=f"Bearish OB"
                    ))
        
        return {'bullish': bullish_obs, 'bearish': bearish_obs}
    
    def find_fvg(self, lookback: int = 30) -> Dict[str, List[SmartLevel]]:
        """Find Fair Value Gaps (imbalances)"""
        bullish_fvg = []
        bearish_fvg = []
        
        for i in range(2, min(lookback, len(self.df) - 1)):
            idx = len(self.df) - 1 - i
            
            if idx < 2:
                continue
            
            prev_high = self.h.iloc[idx - 1]
            next_low = self.l.iloc[idx + 1]
            prev_low = self.l.iloc[idx - 1]
            next_high = self.h.iloc[idx + 1]
            
            # Bullish FVG
            if next_low > prev_high:
                gap_size = next_low - prev_high
                if gap_size > self.current_atr * 0.3:
                    bullish_fvg.append(SmartLevel(
                        price=(prev_high + next_low) / 2,
                        level_type='bullish_fvg',
                        strength=3,
                        description=f"Bullish FVG"
                    ))
            
            # Bearish FVG
            if prev_low > next_high:
                gap_size = prev_low - next_high
                if gap_size > self.current_atr * 0.3:
                    bearish_fvg.append(SmartLevel(
                        price=(next_high + prev_low) / 2,
                        level_type='bearish_fvg',
                        strength=3,
                        description=f"Bearish FVG"
                    ))
        
        return {'bullish': bullish_fvg, 'bearish': bearish_fvg}
    
    def get_ema_levels(self) -> List[SmartLevel]:
        """Get EMA levels from ta library"""
        levels = []
        if len(self.ema_9) > 0 and not pd.isna(self.ema_9.iloc[-1]):
            levels.append(SmartLevel(price=self.ema_9.iloc[-1], level_type='ema_9', strength=2, description="EMA 9"))
        if len(self.ema_20) > 0 and not pd.isna(self.ema_20.iloc[-1]):
            levels.append(SmartLevel(price=self.ema_20.iloc[-1], level_type='ema_20', strength=3, description="EMA 20"))
        if len(self.ema_50) > 0 and not pd.isna(self.ema_50.iloc[-1]):
            levels.append(SmartLevel(price=self.ema_50.iloc[-1], level_type='ema_50', strength=4, description="EMA 50"))
        return levels
    
    def get_vwap_level(self) -> Optional[SmartLevel]:
        """Get VWAP level from ta library"""
        if len(self.vwap) > 0 and not pd.isna(self.vwap.iloc[-1]):
            return SmartLevel(
                price=self.vwap.iloc[-1],
                level_type='vwap',
                strength=4,
                description="VWAP"
            )
        return None
    
    def get_bb_levels(self) -> Dict[str, Optional[SmartLevel]]:
        """Get Bollinger Band levels from ta library"""
        result = {'upper': None, 'lower': None}
        if len(self.bb_upper) > 0 and not pd.isna(self.bb_upper.iloc[-1]):
            result['upper'] = SmartLevel(price=self.bb_upper.iloc[-1], level_type='bb_upper', strength=3, description="BB Upper")
        if len(self.bb_lower) > 0 and not pd.isna(self.bb_lower.iloc[-1]):
            result['lower'] = SmartLevel(price=self.bb_lower.iloc[-1], level_type='bb_lower', strength=3, description="BB Lower")
        return result
    
    def get_support_resistance(self) -> Dict[str, Optional[SmartLevel]]:
        """Get Support/Resistance from Donchian Channel (ta library)"""
        result = {'resistance': None, 'support': None}
        if len(self.dc_upper) > 0 and not pd.isna(self.dc_upper.iloc[-1]):
            result['resistance'] = SmartLevel(price=self.dc_upper.iloc[-1], level_type='resistance', strength=4, description="Resistance")
        if len(self.dc_lower) > 0 and not pd.isna(self.dc_lower.iloc[-1]):
            result['support'] = SmartLevel(price=self.dc_lower.iloc[-1], level_type='support', strength=4, description="Support")
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SMART TRADE CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SmartTradeCalculator:
    """
    Calculates optimal SL and TP using structural levels from ta library.
    Priority: Structure > Arbitrary percentages
    """
    
    def __init__(self, df: pd.DataFrame, direction: str = "LONG"):
        self.df = df
        self.direction = direction
        self.finder = SmartLevelFinder(df)
        self.current_price = df['Close'].iloc[-1]
        self.atr = self.finder.current_atr
        
        # Gather all levels
        self._gather_levels()
    
    def _gather_levels(self):
        """Gather all structural levels"""
        self.order_blocks = self.finder.find_order_blocks()
        self.fvg = self.finder.find_fvg()
        self.swing_lows = self.finder.find_swing_lows()
        self.swing_highs = self.finder.find_swing_highs()
        self.emas = self.finder.get_ema_levels()
        self.vwap = self.finder.get_vwap_level()
        self.bb = self.finder.get_bb_levels()
        self.sr = self.finder.get_support_resistance()
    
    def calculate_trade_levels(self, max_risk_pct: float = 5.0) -> SmartTradeLevels:
        """Calculate optimal entry, SL, and TPs based on structure"""
        result = SmartTradeLevels(direction=self.direction)
        result.entry = self.current_price
        result.entry_type = "market"
        result.entry_reason = "Current price"
        
        if self.direction == "LONG":
            result = self._calculate_long_levels(result, max_risk_pct)
        else:
            result = self._calculate_short_levels(result, max_risk_pct)
        
        return result
    
    def _calculate_long_levels(self, result: SmartTradeLevels, max_risk_pct: float) -> SmartTradeLevels:
        """Calculate levels for LONG trade"""
        price = self.current_price
        
        sl_candidates = []
        
        # 1. Bullish Order Blocks below price
        for ob in self.order_blocks['bullish']:
            if ob.price < price:
                risk = (price - ob.price) / price * 100
                if risk <= max_risk_pct:
                    sl_candidates.append({
                        'price': ob.price * 0.998,
                        'type': 'below_ob',
                        'reason': f"Below Bullish Order Block",
                        'strength': 5,
                        'risk': risk
                    })
        
        # 2. Bullish FVG below price
        for fvg in self.fvg['bullish']:
            if fvg.price < price:
                risk = (price - fvg.price) / price * 100
                if risk <= max_risk_pct:
                    sl_candidates.append({
                        'price': fvg.price * 0.998,
                        'type': 'below_fvg',
                        'reason': f"Below Bullish FVG",
                        'strength': 4,
                        'risk': risk
                    })
        
        # 3. Recent swing lows
        for swing in sorted(self.swing_lows, key=lambda x: x.price, reverse=True):
            if swing.price < price:
                risk = (price - swing.price) / price * 100
                if risk <= max_risk_pct:
                    sl_candidates.append({
                        'price': swing.price * 0.998,
                        'type': 'below_swing',
                        'reason': f"Below Swing Low",
                        'strength': 3,
                        'risk': risk
                    })
                    break
        
        # 4. Support
        support = self.sr.get('support')
        if support and support.price < price:
            risk = (price - support.price) / price * 100
            if risk <= max_risk_pct:
                sl_candidates.append({
                    'price': support.price * 0.998,
                    'type': 'below_support',
                    'reason': f"Below Support",
                    'strength': 3,
                    'risk': risk
                })
        
        # 5. EMA levels
        for ema in sorted(self.emas, key=lambda x: x.price, reverse=True):
            if ema.price < price:
                risk = (price - ema.price) / price * 100
                if risk <= max_risk_pct:
                    sl_candidates.append({
                        'price': ema.price * 0.998,
                        'type': f'below_{ema.level_type}',
                        'reason': f"Below {ema.description}",
                        'strength': 2,
                        'risk': risk
                    })
                    break
        
        # 6. VWAP
        if self.vwap and self.vwap.price < price:
            risk = (price - self.vwap.price) / price * 100
            if risk <= max_risk_pct:
                sl_candidates.append({
                    'price': self.vwap.price * 0.998,
                    'type': 'below_vwap',
                    'reason': "Below VWAP",
                    'strength': 3,
                    'risk': risk
                })
        
        # 7. ATR fallback
        atr_sl = price - (self.atr * 2)
        atr_risk = (price - atr_sl) / price * 100
        sl_candidates.append({
            'price': atr_sl,
            'type': 'atr_fallback',
            'reason': "2x ATR below entry",
            'strength': 1,
            'risk': atr_risk
        })
        
        # Select best SL (highest strength within risk tolerance)
        valid_sl = [s for s in sl_candidates if s['risk'] <= max_risk_pct and s['risk'] >= 0.5]
        if not valid_sl:
            valid_sl = sl_candidates
        
        best_sl = max(valid_sl, key=lambda x: (x['strength'], -x['risk']))
        
        result.stop_loss = best_sl['price']
        result.sl_type = best_sl['type']
        result.sl_reason = best_sl['reason']
        result.risk_pct = best_sl['risk']
        
        # Calculate TPs
        risk_amount = price - result.stop_loss
        
        tp_candidates = []
        
        # Bearish Order Blocks above price
        for ob in self.order_blocks['bearish']:
            if ob.price > price:
                reward = ob.price - price
                rr = reward / risk_amount if risk_amount > 0 else 0
                tp_candidates.append({'price': ob.price * 0.998, 'type': 'bearish_ob', 'reason': "Before Bearish OB", 'rr': rr})
        
        # Resistance
        resistance = self.sr.get('resistance')
        if resistance and resistance.price > price:
            reward = resistance.price - price
            rr = reward / risk_amount if risk_amount > 0 else 0
            tp_candidates.append({'price': resistance.price * 0.998, 'type': 'resistance', 'reason': "Before Resistance", 'rr': rr})
        
        # Swing highs
        for swing in sorted(self.swing_highs, key=lambda x: x.price):
            if swing.price > price:
                reward = swing.price - price
                rr = reward / risk_amount if risk_amount > 0 else 0
                tp_candidates.append({'price': swing.price * 0.998, 'type': 'swing_high', 'reason': "Before Swing High", 'rr': rr})
        
        # BB Upper
        bb_upper = self.bb.get('upper')
        if bb_upper and bb_upper.price > price:
            reward = bb_upper.price - price
            rr = reward / risk_amount if risk_amount > 0 else 0
            tp_candidates.append({'price': bb_upper.price * 0.998, 'type': 'bb_upper', 'reason': "Before BB Upper", 'rr': rr})
        
        # ATR-based targets (fallback)
        for mult, name in [(1.5, 'TP1'), (2.5, 'TP2'), (4.0, 'TP3')]:
            atr_tp = price + (self.atr * mult)
            reward = atr_tp - price
            rr = reward / risk_amount if risk_amount > 0 else 0
            tp_candidates.append({'price': atr_tp, 'type': f'atr_{name.lower()}', 'reason': f"{mult}x ATR", 'rr': rr})
        
        # Sort by R:R
        tp_candidates = sorted([t for t in tp_candidates if t['rr'] > 0], key=lambda x: x['rr'])
        
        # Assign TPs
        tp1_list = [t for t in tp_candidates if t['rr'] >= 1.0]
        if tp1_list:
            tp1 = tp1_list[0]
            result.tp1, result.tp1_type, result.tp1_reason, result.tp1_rr = tp1['price'], tp1['type'], tp1['reason'], tp1['rr']
        else:
            result.tp1 = price + (risk_amount * 1.5)
            result.tp1_type, result.tp1_reason, result.tp1_rr = 'calculated', '1.5:1 R:R', 1.5
        
        tp2_list = [t for t in tp_candidates if t['rr'] >= 2.0 and t['price'] > result.tp1]
        if tp2_list:
            tp2 = tp2_list[0]
            result.tp2, result.tp2_type, result.tp2_reason, result.tp2_rr = tp2['price'], tp2['type'], tp2['reason'], tp2['rr']
        else:
            result.tp2 = price + (risk_amount * 2.5)
            result.tp2_type, result.tp2_reason, result.tp2_rr = 'calculated', '2.5:1 R:R', 2.5
        
        tp3_list = [t for t in tp_candidates if t['rr'] >= 3.0 and t['price'] > result.tp2]
        if tp3_list:
            tp3 = tp3_list[0]
            result.tp3, result.tp3_type, result.tp3_reason, result.tp3_rr = tp3['price'], tp3['type'], tp3['reason'], tp3['rr']
        else:
            result.tp3 = price + (risk_amount * 4.0)
            result.tp3_type, result.tp3_reason, result.tp3_rr = 'calculated', '4:1 R:R', 4.0
        
        result.confidence = self._calculate_confidence(result)
        return result
    
    def _calculate_short_levels(self, result: SmartTradeLevels, max_risk_pct: float) -> SmartTradeLevels:
        """Calculate levels for SHORT trade (inverted logic)"""
        price = self.current_price
        
        sl_candidates = []
        
        # Above price SL for shorts
        for ob in self.order_blocks['bearish']:
            if ob.price > price:
                risk = (ob.price - price) / price * 100
                if risk <= max_risk_pct:
                    sl_candidates.append({'price': ob.price * 1.002, 'type': 'above_ob', 'reason': "Above Bearish OB", 'strength': 5, 'risk': risk})
        
        resistance = self.sr.get('resistance')
        if resistance and resistance.price > price:
            risk = (resistance.price - price) / price * 100
            if risk <= max_risk_pct:
                sl_candidates.append({'price': resistance.price * 1.002, 'type': 'above_resistance', 'reason': "Above Resistance", 'strength': 4, 'risk': risk})
        
        # ATR fallback
        atr_sl = price + (self.atr * 2)
        sl_candidates.append({'price': atr_sl, 'type': 'atr_fallback', 'reason': "2x ATR above entry", 'strength': 1, 'risk': (atr_sl - price) / price * 100})
        
        valid_sl = [s for s in sl_candidates if s['risk'] <= max_risk_pct and s['risk'] >= 0.5]
        if not valid_sl:
            valid_sl = sl_candidates
        
        best_sl = max(valid_sl, key=lambda x: (x['strength'], -x['risk']))
        result.stop_loss = best_sl['price']
        result.sl_type = best_sl['type']
        result.sl_reason = best_sl['reason']
        result.risk_pct = best_sl['risk']
        
        # TPs for short
        risk_amount = result.stop_loss - price
        result.tp1 = price - (risk_amount * 1.5)
        result.tp1_type, result.tp1_reason, result.tp1_rr = 'calculated', '1.5:1 R:R', 1.5
        result.tp2 = price - (risk_amount * 2.5)
        result.tp2_type, result.tp2_reason, result.tp2_rr = 'calculated', '2.5:1 R:R', 2.5
        result.tp3 = price - (risk_amount * 4.0)
        result.tp3_type, result.tp3_reason, result.tp3_rr = 'calculated', '4:1 R:R', 4.0
        
        result.confidence = self._calculate_confidence(result)
        return result
    
    def _calculate_confidence(self, levels: SmartTradeLevels) -> int:
        """Calculate confidence score based on level quality"""
        confidence = 50
        
        sl_bonuses = {'below_ob': 20, 'above_ob': 20, 'below_fvg': 15, 'above_fvg': 15,
                      'below_swing': 12, 'above_swing': 12, 'below_support': 10, 'above_resistance': 10,
                      'below_vwap': 10, 'above_vwap': 10, 'atr_fallback': 0}
        
        for key, bonus in sl_bonuses.items():
            if key in levels.sl_type:
                confidence += bonus
                break
        
        if levels.tp1_rr >= 2.0:
            confidence += 10
        elif levels.tp1_rr >= 1.5:
            confidence += 5
        
        if levels.risk_pct > 4.0:
            confidence -= 10
        elif levels.risk_pct < 2.0:
            confidence += 5
        
        return min(100, max(0, confidence))


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_smart_levels(df: pd.DataFrame, direction: str = "LONG", max_risk_pct: float = 5.0) -> SmartTradeLevels:
    """Quick function to calculate smart trade levels using ta library"""
    calculator = SmartTradeCalculator(df, direction)
    return calculator.calculate_trade_levels(max_risk_pct=max_risk_pct)


def find_structural_sl_update(df: pd.DataFrame, current_sl: float, entry_price: float, direction: str = "LONG") -> Optional[Dict]:
    """Find new structural SL to trail to (Option C - structure only, NO breakeven)"""
    finder = SmartLevelFinder(df)
    current_price = df['Close'].iloc[-1]
    
    if direction == "LONG":
        swing_lows = finder.find_swing_lows()
        for swing in sorted(swing_lows, key=lambda x: x.price, reverse=True):
            # New SL must be: above current SL, below entry, below current price
            if swing.price > current_sl and swing.price < entry_price and swing.price < current_price:
                return {'new_sl': swing.price * 0.998, 'type': 'swing_low', 'reason': f"Trail to swing low"}
        
        emas = finder.get_ema_levels()
        for ema in sorted(emas, key=lambda x: x.price, reverse=True):
            if ema.price > current_sl and ema.price < entry_price and ema.price < current_price:
                return {'new_sl': ema.price * 0.998, 'type': ema.level_type, 'reason': f"Trail to {ema.description}"}
    else:
        # SHORT - mirror logic
        swing_highs = finder.find_swing_highs()
        for swing in sorted(swing_highs, key=lambda x: x.price):
            if swing.price < current_sl and swing.price > entry_price and swing.price > current_price:
                return {'new_sl': swing.price * 1.002, 'type': 'swing_high', 'reason': f"Trail to swing high"}
    
    return None


def create_enhanced_trade(symbol: str, timeframe: str, direction: str, levels: SmartTradeLevels, 
                         grade: str, confidence: int, position_size: float = 2500) -> EnhancedTrade:
    """Create a new EnhancedTrade from analysis results"""
    now = datetime.now().isoformat()
    trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    risk_amount = position_size * (levels.risk_pct / 100)
    
    return EnhancedTrade(
        id=trade_id,
        symbol=symbol,
        timeframe=timeframe,
        entry_price=levels.entry,
        entry_time=now,
        direction=direction,
        grade=grade,
        confidence=confidence,
        stop_loss=levels.stop_loss,
        sl_type=levels.sl_type,
        sl_reason=levels.sl_reason,
        tp1=levels.tp1,
        tp2=levels.tp2,
        tp3=levels.tp3,
        position_size=position_size,
        risk_amount=risk_amount,
        highest_price=levels.entry,
        lowest_price=levels.entry,
        original_sl=levels.stop_loss,
        current_sl=levels.stop_loss,
        created_at=now,
        updated_at=now
    )


def update_trade_mae_mfe(trade: EnhancedTrade, current_price: float) -> EnhancedTrade:
    """Update trade with current price and MAE/MFE tracking"""
    trade.updated_at = datetime.now().isoformat()
    
    # Update highest/lowest
    if current_price > trade.highest_price:
        trade.highest_price = current_price
    if current_price < trade.lowest_price or trade.lowest_price == 0:
        trade.lowest_price = current_price
    
    # Calculate MAE/MFE
    if trade.direction == "LONG":
        trade.highest_profit_pct = (trade.highest_price - trade.entry_price) / trade.entry_price * 100
        trade.worst_drawdown_pct = (trade.lowest_price - trade.entry_price) / trade.entry_price * 100
        trade.pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
    else:
        trade.highest_profit_pct = (trade.entry_price - trade.lowest_price) / trade.entry_price * 100
        trade.worst_drawdown_pct = (trade.entry_price - trade.highest_price) / trade.entry_price * 100
        trade.pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
    
    trade.pnl_dollars = trade.position_size * (trade.pnl_pct / 100)
    
    # Check TP hits
    if trade.direction == "LONG":
        if current_price >= trade.tp1 and not trade.tp1_hit:
            trade.tp1_hit = True
            trade.status = "TP1_HIT"
        if current_price >= trade.tp2 and not trade.tp2_hit:
            trade.tp2_hit = True
            trade.status = "TP2_HIT"
    else:
        if current_price <= trade.tp1 and not trade.tp1_hit:
            trade.tp1_hit = True
            trade.status = "TP1_HIT"
        if current_price <= trade.tp2 and not trade.tp2_hit:
            trade.tp2_hit = True
            trade.status = "TP2_HIT"
    
    return trade
