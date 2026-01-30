"""
Signal Generator Module - PRO VERSION
Generates trade signals with SMC-BASED Entry, SL, TP levels
NOT FIXED PERCENTAGES - Uses Order Blocks, FVG, Support/Resistance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict

from .indicators import (
    calculate_rsi, calculate_ema, calculate_atr, calculate_macd,
    calculate_bbands, calculate_obv, calculate_mfi, calculate_cmf,
    find_support_resistance
)

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIGNAL DATA CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeSignal:
    """Trade signal with all relevant information"""
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
            'sl_type': self.sl_type
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SMC LEVEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_order_blocks(df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[Dict]]:
    """
    Find Order Blocks - last opposing candle before strong move
    Returns price zones where institutions entered
    """
    bullish_obs = []
    bearish_obs = []
    
    if df is None or len(df) < lookback:
        return {'bullish': [], 'bearish': []}
    
    atr = (df['High'] - df['Low']).rolling(14).mean()
    
    for i in range(len(df) - lookback, len(df) - 3):
        try:
            current_atr = atr.iloc[i]
            if pd.isna(current_atr) or current_atr == 0:
                continue
                
            # Bullish OB: Red candle followed by strong bullish move
            if df['Close'].iloc[i] < df['Open'].iloc[i]:
                next_high = df['High'].iloc[i+1:i+4].max()
                next_move = next_high - df['Close'].iloc[i]
                
                if next_move > current_atr * 1.5:
                    strength = min(5, int(next_move / current_atr))
                    bullish_obs.append({
                        'top': max(df['Open'].iloc[i], df['Close'].iloc[i]),
                        'bottom': min(df['Open'].iloc[i], df['Close'].iloc[i]),
                        'mid': (df['Open'].iloc[i] + df['Close'].iloc[i]) / 2,
                        'strength': strength,
                        'index': i
                    })
            
            # Bearish OB: Green candle followed by strong bearish move
            if df['Close'].iloc[i] > df['Open'].iloc[i]:
                next_low = df['Low'].iloc[i+1:i+4].min()
                next_move = df['Close'].iloc[i] - next_low
                
                if next_move > current_atr * 1.5:
                    strength = min(5, int(next_move / current_atr))
                    bearish_obs.append({
                        'top': max(df['Open'].iloc[i], df['Close'].iloc[i]),
                        'bottom': min(df['Open'].iloc[i], df['Close'].iloc[i]),
                        'mid': (df['Open'].iloc[i] + df['Close'].iloc[i]) / 2,
                        'strength': strength,
                        'index': i
                    })
        except:
            continue
    
    bullish_obs.sort(key=lambda x: x['strength'], reverse=True)
    bearish_obs.sort(key=lambda x: x['strength'], reverse=True)
    
    return {'bullish': bullish_obs[:5], 'bearish': bearish_obs[:5]}


def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 30) -> Dict[str, List[Dict]]:
    """Find Fair Value Gaps - price imbalances"""
    bullish_fvgs = []
    bearish_fvgs = []
    
    if df is None or len(df) < lookback:
        return {'bullish': [], 'bearish': []}
    
    for i in range(len(df) - lookback, len(df) - 2):
        try:
            # Bullish FVG: Gap up
            if df['Low'].iloc[i+2] > df['High'].iloc[i]:
                gap_size = df['Low'].iloc[i+2] - df['High'].iloc[i]
                bullish_fvgs.append({
                    'top': df['Low'].iloc[i+2],
                    'bottom': df['High'].iloc[i],
                    'mid': (df['Low'].iloc[i+2] + df['High'].iloc[i]) / 2,
                    'size': gap_size,
                    'index': i
                })
            
            # Bearish FVG: Gap down
            if df['High'].iloc[i+2] < df['Low'].iloc[i]:
                gap_size = df['Low'].iloc[i] - df['High'].iloc[i+2]
                bearish_fvgs.append({
                    'top': df['Low'].iloc[i],
                    'bottom': df['High'].iloc[i+2],
                    'mid': (df['Low'].iloc[i] + df['High'].iloc[i+2]) / 2,
                    'size': gap_size,
                    'index': i
                })
        except:
            continue
    
    bullish_fvgs.sort(key=lambda x: x['size'], reverse=True)
    bearish_fvgs.sort(key=lambda x: x['size'], reverse=True)
    
    return {'bullish': bullish_fvgs[:5], 'bearish': bearish_fvgs[:5]}


def find_swing_levels(df: pd.DataFrame, lookback: int = 50) -> Dict[str, List[Dict]]:
    """Find swing highs and lows for support/resistance"""
    supports = []
    resistances = []
    
    if df is None or len(df) < lookback:
        return {'support': [], 'resistance': []}
    
    for i in range(len(df) - lookback + 2, len(df) - 2):
        try:
            # Swing Low (support)
            if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
                df['Low'].iloc[i] < df['Low'].iloc[i-2] and
                df['Low'].iloc[i] < df['Low'].iloc[i+1] and 
                df['Low'].iloc[i] < df['Low'].iloc[i+2]):
                supports.append({
                    'price': df['Low'].iloc[i],
                    'index': i,
                    'tested': 1
                })
            
            # Swing High (resistance)
            if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
                df['High'].iloc[i] > df['High'].iloc[i-2] and
                df['High'].iloc[i] > df['High'].iloc[i+1] and 
                df['High'].iloc[i] > df['High'].iloc[i+2]):
                resistances.append({
                    'price': df['High'].iloc[i],
                    'index': i,
                    'tested': 1
                })
        except:
            continue
    
    supports.sort(key=lambda x: x['price'], reverse=True)
    resistances.sort(key=lambda x: x['price'])
    
    return {'support': supports[:5], 'resistance': resistances[:5]}


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR - SMC BASED
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Generate trade signals with SMC-based levels"""
    
    @staticmethod
    def generate_signal(df: pd.DataFrame, symbol: str, timeframe: str = '1h') -> Optional[TradeSignal]:
        """
        Generate a trade signal with SMC-BASED TP/SL levels
        
        SL Priority: OB > FVG > Support > ATR
        TP Priority: OB > FVG > Resistance > R:R
        """
        if df is None or len(df) < 50:
            return None
        
        try:
            current_price = df['Close'].iloc[-1]
            
            # Calculate indicators
            rsi = calculate_rsi(df['Close'], 14)
            ema_20 = calculate_ema(df['Close'], 20)
            ema_50 = calculate_ema(df['Close'], 50)
            atr = calculate_atr(df['High'], df['Low'], df['Close'], 14)
            macd_line, signal_line, histogram = calculate_macd(df['Close'])
            
            current_rsi = rsi.iloc[-1]
            current_atr = atr.iloc[-1]
            current_ema_20 = ema_20.iloc[-1]
            current_ema_50 = ema_50.iloc[-1]
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # GET SMC LEVELS
            obs = find_order_blocks(df)
            fvgs = find_fair_value_gaps(df)
            swings = find_swing_levels(df)
            
            # DETERMINE DIRECTION
            bullish_signals = 0
            bearish_signals = 0
            
            if current_price > current_ema_20 > current_ema_50:
                bullish_signals += 2
            elif current_price < current_ema_20 < current_ema_50:
                bearish_signals += 2
            
            if current_rsi < 35:
                bullish_signals += 1
            elif current_rsi > 65:
                bearish_signals += 1
            
            if current_macd > current_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # At Order Block bonus
            for ob in obs['bullish']:
                if ob['bottom'] * 0.99 <= current_price <= ob['top'] * 1.01:
                    bullish_signals += 2
                    break
            
            for ob in obs['bearish']:
                if ob['bottom'] * 0.99 <= current_price <= ob['top'] * 1.01:
                    bearish_signals += 2
                    break
            
            direction = 'LONG' if bullish_signals > bearish_signals else 'SHORT'
            
            # ═══════════════════════════════════════════════════════════════════
            # SMC-BASED LEVELS
            # ═══════════════════════════════════════════════════════════════════
            
            entry = current_price
            entry_reason = "Current market price"
            
            if direction == 'LONG':
                # Entry reason
                for ob in obs['bullish']:
                    if ob['bottom'] * 0.99 <= current_price <= ob['top'] * 1.01:
                        entry_reason = f"At Bullish OB ({ob['bottom']:.6f} - {ob['top']:.6f})"
                        break
                
                for fvg in fvgs['bullish']:
                    if fvg['bottom'] * 0.99 <= current_price <= fvg['top'] * 1.01:
                        if "OB" not in entry_reason:
                            entry_reason = f"At Bullish FVG ({fvg['bottom']:.6f} - {fvg['top']:.6f})"
                        break
                
                # STOP LOSS - Priority: OB > FVG > Support > ATR
                # ⚠️ ANTI-STOP-HUNT: Place SL BELOW the obvious level with extra buffer
                # Market makers hunt the obvious levels (0.5% below support)
                # We go 1.5-2% below to survive the sweep
                stop_loss = None
                sl_reason = ""
                sl_type = "atr"
                
                # ═══════════════════════════════════════════════════════════════════════
                # 🛡️ ANTI-HUNT BUFFER - Mode-aware to survive Binance stop hunts
                # ═══════════════════════════════════════════════════════════════════════
                # Binance (and other exchanges) hunt stops aggressively
                # They know where retail puts stops (just below support, just above resistance)
                # Solution: Place stops FURTHER than they expect
                #
                # Buffer levels by timeframe (higher TF = wider buffer needed):
                # - 1m/5m (Scalp): 1.5% - Tight but quick trades
                # - 15m/1h (Day): 2.0% - Standard day trade protection
                # - 4h/1d (Swing): 2.5% - Wider for overnight holds
                # - 1w (Position): 3.0% - Very wide for long holds
                # ═══════════════════════════════════════════════════════════════════════
                
                # Determine buffer based on timeframe
                tf_lower = timeframe.lower() if timeframe else '1h'
                if tf_lower in ['1m', '5m']:
                    ANTI_HUNT_BUFFER = 0.985  # 1.5% below level (scalp - need tighter)
                    ATR_MULTIPLIER = 2.5
                elif tf_lower in ['15m', '1h']:
                    ANTI_HUNT_BUFFER = 0.980  # 2.0% below level (day trade)
                    ATR_MULTIPLIER = 3.0
                elif tf_lower in ['4h', '1d']:
                    ANTI_HUNT_BUFFER = 0.975  # 2.5% below level (swing)
                    ATR_MULTIPLIER = 3.5
                else:  # 1w and beyond
                    ANTI_HUNT_BUFFER = 0.970  # 3.0% below level (position)
                    ATR_MULTIPLIER = 4.0
                
                for ob in obs['bullish']:
                    if ob['bottom'] < current_price:
                        # Place SL below the OB with anti-hunt buffer
                        stop_loss = ob['bottom'] * ANTI_HUNT_BUFFER
                        sl_reason = f"Below Bullish OB ({(1-ANTI_HUNT_BUFFER)*100:.1f}% anti-hunt)"
                        sl_type = "ob"
                        break
                
                if stop_loss is None:
                    for fvg in fvgs['bullish']:
                        if fvg['bottom'] < current_price:
                            stop_loss = fvg['bottom'] * ANTI_HUNT_BUFFER
                            sl_reason = f"Below Bullish FVG ({(1-ANTI_HUNT_BUFFER)*100:.1f}% anti-hunt)"
                            sl_type = "fvg"
                            break
                
                if stop_loss is None:
                    for sup in swings['support']:
                        if sup['price'] < current_price:
                            stop_loss = sup['price'] * ANTI_HUNT_BUFFER
                            sl_reason = f"Below Support ({(1-ANTI_HUNT_BUFFER)*100:.1f}% anti-hunt)"
                            sl_type = "support"
                            break
                
                if stop_loss is None:
                    stop_loss = current_price - (current_atr * ATR_MULTIPLIER)
                    sl_reason = f"ATR-based ({ATR_MULTIPLIER}x ATR)"
                    sl_type = "atr"
                
                # TAKE PROFITS - Priority: Bearish OB > FVG > Resistance
                tp_candidates = []
                
                for ob in obs['bearish']:
                    if ob['bottom'] > current_price:
                        tp_candidates.append({
                            'price': ob['bottom'],
                            'reason': f"Bearish OB @ {ob['bottom']:.6f}",
                            'type': 'ob'
                        })
                
                for fvg in fvgs['bearish']:
                    if fvg['bottom'] > current_price:
                        tp_candidates.append({
                            'price': fvg['bottom'],
                            'reason': f"Bearish FVG @ {fvg['bottom']:.6f}",
                            'type': 'fvg'
                        })
                
                for res in swings['resistance']:
                    if res['price'] > current_price:
                        tp_candidates.append({
                            'price': res['price'] * 0.998,
                            'reason': f"Resistance @ {res['price']:.6f}",
                            'type': 'resistance'
                        })
                
                tp_candidates.sort(key=lambda x: x['price'])
                risk = entry - stop_loss
                
                if len(tp_candidates) >= 1:
                    tp1 = tp_candidates[0]['price']
                    tp1_reason = tp_candidates[0]['reason']
                else:
                    tp1 = entry + (risk * 1.5)
                    tp1_reason = f"R:R 1.5 (no structure)"
                
                if len(tp_candidates) >= 2:
                    tp2 = tp_candidates[1]['price']
                    tp2_reason = tp_candidates[1]['reason']
                else:
                    tp2 = entry + (risk * 2.5)
                    tp2_reason = f"R:R 2.5 (extended)"
                
                if len(tp_candidates) >= 3:
                    tp3 = tp_candidates[2]['price']
                    tp3_reason = tp_candidates[2]['reason']
                else:
                    tp3 = entry + (risk * 4.0)
                    tp3_reason = f"R:R 4.0 (max target)"
                
            else:  # SHORT
                for ob in obs['bearish']:
                    if ob['bottom'] * 0.99 <= current_price <= ob['top'] * 1.01:
                        entry_reason = f"At Bearish OB ({ob['bottom']:.6f} - {ob['top']:.6f})"
                        break
                
                # ═══════════════════════════════════════════════════════════════════════
                # 🛡️ ANTI-HUNT BUFFER FOR SHORTS - Place SL ABOVE the obvious level
                # ═══════════════════════════════════════════════════════════════════════
                stop_loss = None
                sl_reason = ""
                sl_type = "atr"
                
                # Same timeframe-aware buffers as LONG (but inverted - above level)
                tf_lower = timeframe.lower() if timeframe else '1h'
                if tf_lower in ['1m', '5m']:
                    ANTI_HUNT_BUFFER_SHORT = 1.015  # 1.5% above level (scalp)
                    ATR_MULTIPLIER = 2.5
                elif tf_lower in ['15m', '1h']:
                    ANTI_HUNT_BUFFER_SHORT = 1.020  # 2.0% above level (day trade)
                    ATR_MULTIPLIER = 3.0
                elif tf_lower in ['4h', '1d']:
                    ANTI_HUNT_BUFFER_SHORT = 1.025  # 2.5% above level (swing)
                    ATR_MULTIPLIER = 3.5
                else:  # 1w and beyond
                    ANTI_HUNT_BUFFER_SHORT = 1.030  # 3.0% above level (position)
                    ATR_MULTIPLIER = 4.0
                
                for ob in obs['bearish']:
                    if ob['top'] > current_price:
                        stop_loss = ob['top'] * ANTI_HUNT_BUFFER_SHORT
                        sl_reason = f"Above Bearish OB ({(ANTI_HUNT_BUFFER_SHORT-1)*100:.1f}% anti-hunt)"
                        sl_type = "ob"
                        break
                
                if stop_loss is None:
                    for fvg in fvgs['bearish']:
                        if fvg['top'] > current_price:
                            stop_loss = fvg['top'] * ANTI_HUNT_BUFFER_SHORT
                            sl_reason = f"Above Bearish FVG ({(ANTI_HUNT_BUFFER_SHORT-1)*100:.1f}% anti-hunt)"
                            sl_type = "fvg"
                            break
                
                if stop_loss is None:
                    for res in swings['resistance']:
                        if res['price'] > current_price:
                            stop_loss = res['price'] * ANTI_HUNT_BUFFER_SHORT
                            sl_reason = f"Above Resistance ({(ANTI_HUNT_BUFFER_SHORT-1)*100:.1f}% anti-hunt)"
                            sl_type = "resistance"
                            break
                
                if stop_loss is None:
                    stop_loss = current_price + (current_atr * ATR_MULTIPLIER)
                    sl_reason = f"ATR-based ({ATR_MULTIPLIER}x ATR)"
                    sl_type = "atr"
                
                tp_candidates = []
                
                for ob in obs['bullish']:
                    if ob['top'] < current_price:
                        tp_candidates.append({
                            'price': ob['top'],
                            'reason': f"Bullish OB @ {ob['top']:.6f}",
                            'type': 'ob'
                        })
                
                for fvg in fvgs['bullish']:
                    if fvg['top'] < current_price:
                        tp_candidates.append({
                            'price': fvg['top'],
                            'reason': f"Bullish FVG @ {fvg['top']:.6f}",
                            'type': 'fvg'
                        })
                
                for sup in swings['support']:
                    if sup['price'] < current_price:
                        tp_candidates.append({
                            'price': sup['price'] * 1.002,
                            'reason': f"Support @ {sup['price']:.6f}",
                            'type': 'support'
                        })
                
                tp_candidates.sort(key=lambda x: x['price'], reverse=True)
                risk = stop_loss - entry
                
                if len(tp_candidates) >= 1:
                    tp1 = tp_candidates[0]['price']
                    tp1_reason = tp_candidates[0]['reason']
                else:
                    tp1 = entry - (risk * 1.5)
                    tp1_reason = f"R:R 1.5"
                
                if len(tp_candidates) >= 2:
                    tp2 = tp_candidates[1]['price']
                    tp2_reason = tp_candidates[1]['reason']
                else:
                    tp2 = entry - (risk * 2.5)
                    tp2_reason = f"R:R 2.5"
                
                if len(tp_candidates) >= 3:
                    tp3 = tp_candidates[2]['price']
                    tp3_reason = tp_candidates[2]['reason']
                else:
                    tp3 = entry - (risk * 4.0)
                    tp3_reason = f"R:R 4.0"
            
            # METRICS
            risk_amount = abs(entry - stop_loss)
            
            # Calculate R:R for EACH target
            tp1_reward = abs(tp1 - entry)
            tp2_reward = abs(tp2 - entry)
            tp3_reward = abs(tp3 - entry)
            
            rr_tp1 = tp1_reward / risk_amount if risk_amount > 0 else 0
            rr_tp2 = tp2_reward / risk_amount if risk_amount > 0 else 0
            rr_tp3 = tp3_reward / risk_amount if risk_amount > 0 else 0
            
            rr_ratio = rr_tp2  # Legacy - keep for compatibility
            risk_pct = (risk_amount / entry) * 100
            
            # ⚠️ FILTER: Skip trades with terrible TP1 R:R (less than 0.5:1)
            # If you're risking more than 2x what you can make at TP1, skip it
            if rr_tp1 < 0.5:
                return None  # Bad trade - don't even show it
            
            confidence = SignalGenerator._calculate_confidence(
                bullish_signals if direction == 'LONG' else bearish_signals,
                volume_ratio, current_rsi, rr_ratio, direction, sl_type,
                rr_tp1  # Pass TP1 R:R for scoring adjustment
            )
            
            return TradeSignal(
                symbol=symbol,
                direction=direction,
                entry=entry,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                rr_ratio=rr_ratio,
                risk_pct=risk_pct,
                confidence=confidence,
                timeframe=timeframe,
                timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                rr_tp1=rr_tp1,
                rr_tp2=rr_tp2,
                rr_tp3=rr_tp3,
                entry_reason=entry_reason,
                sl_reason=sl_reason,
                tp1_reason=tp1_reason,
                tp2_reason=tp2_reason,
                tp3_reason=tp3_reason,
                sl_type=sl_type
            )
            
        except Exception as e:
            print(f"Signal generation error: {e}")
            return None
    
    @staticmethod
    def _calculate_confidence(signal_count: int, volume_ratio: float, 
                             rsi: float, rr_ratio: float, direction: str,
                             sl_type: str = 'atr', rr_tp1: float = 1.0) -> int:
        """Calculate signal confidence score (0-100)"""
        score = 0
        
        score += min(signal_count * 6, 30)
        
        if volume_ratio > 2:
            score += 20
        elif volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1:
            score += 10
        
        if direction == 'LONG':
            if rsi < 30: score += 20
            elif rsi < 40: score += 15
            elif rsi < 50: score += 10
        else:
            if rsi > 70: score += 20
            elif rsi > 60: score += 15
            elif rsi > 50: score += 10
        
        # R:R scoring based on TP2 (legacy)
        if rr_ratio >= 3: score += 20
        elif rr_ratio >= 2: score += 15
        elif rr_ratio >= 1.5: score += 10
        
        # ⚠️ TP1 R:R SCORING - This is what MATTERS for most traders
        # Penalize if TP1 has bad R:R, bonus if good
        if rr_tp1 >= 1.5:
            score += 15  # Excellent - TP1 gives 1.5x reward vs risk
        elif rr_tp1 >= 1.0:
            score += 10  # Good - at least 1:1 at TP1
        elif rr_tp1 >= 0.75:
            score += 0   # Acceptable but not great
        elif rr_tp1 >= 0.5:
            score -= 10  # Poor - penalize
        # Below 0.5 is filtered out entirely
        
        # SL Quality Bonus
        if sl_type == 'ob': score += 15
        elif sl_type == 'fvg': score += 10
        elif sl_type in ['support', 'resistance']: score += 5
        
        score += 10
        
        return min(score, 100)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_trend(df: pd.DataFrame) -> dict:
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
