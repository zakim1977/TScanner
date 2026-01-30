"""
InvestorIQ - Alert & Prediction System
======================================
Detects setups BEFORE they happen and generates alerts
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL STAGES
# ═══════════════════════════════════════════════════════════════════════════════

class SignalStage(Enum):
    """Stage of a trading signal"""
    APPROACHING = "approaching"    # 2-5% away from level
    AT_LEVEL = "at_level"          # Within 1-2% of level
    CONFIRMED = "confirmed"        # Bounced + momentum confirmed
    TRIGGERED = "triggered"        # Entry executed
    INVALIDATED = "invalidated"    # Level broken wrong way


class AlertType(Enum):
    """Types of alerts"""
    APPROACHING_ENTRY = "approaching_entry"
    AT_ENTRY_ZONE = "at_entry_zone"
    APPROACHING_TP1 = "approaching_tp1"
    APPROACHING_TP2 = "approaching_tp2"
    APPROACHING_TP3 = "approaching_tp3"
    HIT_TP1 = "hit_tp1"
    HIT_TP2 = "hit_tp2"
    HIT_TP3 = "hit_tp3"
    APPROACHING_SL = "approaching_sl"
    HIT_SL = "hit_sl"
    DCA_OPPORTUNITY = "dca_opportunity"
    SIGNAL_UPGRADED = "signal_upgraded"
    SIGNAL_DOWNGRADED = "signal_downgraded"
    WHALE_ACTIVITY = "whale_activity"
    MOMENTUM_SHIFT = "momentum_shift"


@dataclass
class Alert:
    """Single alert"""
    alert_type: AlertType
    symbol: str
    message: str
    price: float
    target_price: float
    distance_pct: float
    urgency: str  # 'high', 'medium', 'low'
    timestamp: datetime = field(default_factory=datetime.now)
    sound: bool = False  # Whether to play sound
    
    @property
    def emoji(self) -> str:
        emoji_map = {
            AlertType.APPROACHING_ENTRY: "🔮",
            AlertType.AT_ENTRY_ZONE: "🎯",
            AlertType.APPROACHING_TP1: "📈",
            AlertType.APPROACHING_TP2: "📈",
            AlertType.APPROACHING_TP3: "🎉",
            AlertType.HIT_TP1: "✅",
            AlertType.HIT_TP2: "✅",
            AlertType.HIT_TP3: "🏆",
            AlertType.APPROACHING_SL: "⚠️",
            AlertType.HIT_SL: "🛑",
            AlertType.DCA_OPPORTUNITY: "💰",
            AlertType.SIGNAL_UPGRADED: "⬆️",
            AlertType.SIGNAL_DOWNGRADED: "⬇️",
            AlertType.WHALE_ACTIVITY: "🐋",
            AlertType.MOMENTUM_SHIFT: "🔄",
        }
        return emoji_map.get(self.alert_type, "🔔")
    
    @property
    def color(self) -> str:
        # TP alerts = GREEN (profit incoming!)
        # SL alerts = RED (danger!)
        # Other alerts = based on urgency
        
        tp_types = [
            AlertType.APPROACHING_TP1, AlertType.APPROACHING_TP2, AlertType.APPROACHING_TP3,
            AlertType.HIT_TP1, AlertType.HIT_TP2, AlertType.HIT_TP3
        ]
        sl_types = [AlertType.APPROACHING_SL, AlertType.HIT_SL]
        
        if self.alert_type in tp_types:
            return "#00d4aa"  # Green - profit!
        elif self.alert_type in sl_types:
            return "#ff4444"  # Red - danger!
        elif self.urgency == "high":
            return "#ff4444"
        elif self.urgency == "medium":
            return "#ffcc00"
        return "#00d4aa"


@dataclass
class PredictiveSignal:
    """A predictive signal - setup approaching but not yet active"""
    symbol: str
    stage: SignalStage
    
    # Level info
    level_type: str  # 'order_block', 'fvg', 'support', 'demand_zone'
    level_price: float
    current_price: float
    distance_pct: float
    
    # Direction
    direction: str  # 'LONG', 'SHORT'
    
    # Quality
    confidence: int  # 0-100
    
    # Narrative
    title: str
    description: str
    action: str
    
    # Levels if signal activates
    suggested_entry: float = 0
    suggested_sl: float = 0
    suggested_tp1: float = 0
    
    @property
    def stage_emoji(self) -> str:
        return {
            SignalStage.APPROACHING: "🔮",
            SignalStage.AT_LEVEL: "🎯",
            SignalStage.CONFIRMED: "✅",
            SignalStage.TRIGGERED: "🚀",
            SignalStage.INVALIDATED: "❌"
        }.get(self.stage, "❓")
    
    @property
    def stage_color(self) -> str:
        return {
            SignalStage.APPROACHING: "#9d4edd",
            SignalStage.AT_LEVEL: "#00d4ff",
            SignalStage.CONFIRMED: "#00d4aa",
            SignalStage.TRIGGERED: "#00ff88",
            SignalStage.INVALIDATED: "#ff4444"
        }.get(self.stage, "#888888")


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIVE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveAnalyzer:
    """
    Analyzes price action to find APPROACHING setups
    Detects moves BEFORE they happen
    """
    
    def __init__(self):
        self.signals: List[PredictiveSignal] = []
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> List[PredictiveSignal]:
        """
        Find all predictive signals for a symbol
        
        Returns signals at different stages:
        - APPROACHING: Price moving toward key level (2-5% away)
        - AT_LEVEL: Price at key level (within 1-2%)
        - CONFIRMED: Bounced from level with momentum
        """
        self.signals = []
        
        if df is None or len(df) < 50:
            return []
        
        current_price = df['Close'].iloc[-1]
        
        # Find key levels
        levels = self._find_key_levels(df)
        
        # Analyze each level
        for level in levels:
            signal = self._analyze_level(df, symbol, current_price, level)
            if signal:
                self.signals.append(signal)
        
        # Sort by confidence and distance
        self.signals.sort(key=lambda x: (x.confidence, -x.distance_pct), reverse=True)
        
        return self.signals
    
    def _find_key_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Find Order Blocks, FVGs, S/R levels"""
        levels = []
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        current_price = close.iloc[-1]
        avg_volume = volume.rolling(20).mean()
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. ORDER BLOCKS (Institutional Levels)
        # ═══════════════════════════════════════════════════════════════════
        
        for i in range(3, len(df) - 1):
            # Bullish OB: Last red candle before big green move
            if close.iloc[i] < close.iloc[i-1]:  # Red candle
                # Check if followed by strong bullish move
                future_high = high.iloc[i+1:min(i+6, len(df))].max()
                move_pct = (future_high - close.iloc[i]) / close.iloc[i] * 100
                
                if move_pct > 2:  # At least 2% move after
                    ob_top = high.iloc[i]
                    ob_bottom = low.iloc[i]
                    
                    # Only if price is ABOVE this level (for long setup)
                    if current_price > ob_top:
                        distance = (current_price - ob_top) / current_price * 100
                        
                        if 0.5 < distance < 8:  # Within actionable range
                            levels.append({
                                'type': 'bullish_ob',
                                'direction': 'LONG',
                                'price': ob_top,
                                'zone_bottom': ob_bottom,
                                'zone_top': ob_top,
                                'distance': distance,
                                'strength': min(5, int(move_pct)),
                                'description': f"Bullish Order Block - Institutional demand zone"
                            })
            
            # Bearish OB: Last green candle before big red move
            if close.iloc[i] > close.iloc[i-1]:  # Green candle
                future_low = low.iloc[i+1:min(i+6, len(df))].min()
                move_pct = (close.iloc[i] - future_low) / close.iloc[i] * 100
                
                if move_pct > 2:
                    ob_top = high.iloc[i]
                    ob_bottom = low.iloc[i]
                    
                    if current_price < ob_bottom:
                        distance = (ob_bottom - current_price) / current_price * 100
                        
                        if 0.5 < distance < 8:
                            levels.append({
                                'type': 'bearish_ob',
                                'direction': 'SHORT',
                                'price': ob_bottom,
                                'zone_bottom': ob_bottom,
                                'zone_top': ob_top,
                                'distance': distance,
                                'strength': min(5, int(move_pct)),
                                'description': f"Bearish Order Block - Institutional supply zone"
                            })
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. FAIR VALUE GAPS (Imbalances)
        # ═══════════════════════════════════════════════════════════════════
        
        for i in range(2, len(df) - 1):
            # Bullish FVG: Gap up (candle 3 low > candle 1 high)
            if low.iloc[i] > high.iloc[i-2]:
                fvg_top = low.iloc[i]
                fvg_bottom = high.iloc[i-2]
                fvg_mid = (fvg_top + fvg_bottom) / 2
                
                if current_price > fvg_top:
                    distance = (current_price - fvg_mid) / current_price * 100
                    
                    if 0.5 < distance < 6:
                        levels.append({
                            'type': 'bullish_fvg',
                            'direction': 'LONG',
                            'price': fvg_mid,
                            'zone_bottom': fvg_bottom,
                            'zone_top': fvg_top,
                            'distance': distance,
                            'strength': 3,
                            'description': f"Bullish FVG - Price magnet, unfilled gap"
                        })
            
            # Bearish FVG
            if high.iloc[i] < low.iloc[i-2]:
                fvg_top = low.iloc[i-2]
                fvg_bottom = high.iloc[i]
                fvg_mid = (fvg_top + fvg_bottom) / 2
                
                if current_price < fvg_bottom:
                    distance = (fvg_mid - current_price) / current_price * 100
                    
                    if 0.5 < distance < 6:
                        levels.append({
                            'type': 'bearish_fvg',
                            'direction': 'SHORT',
                            'price': fvg_mid,
                            'zone_bottom': fvg_bottom,
                            'zone_top': fvg_top,
                            'distance': distance,
                            'strength': 3,
                            'description': f"Bearish FVG - Price magnet overhead"
                        })
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. SUPPORT/RESISTANCE (Swing Levels)
        # ═══════════════════════════════════════════════════════════════════
        
        # Find swing lows (support)
        for i in range(5, len(df) - 5):
            if low.iloc[i] == low.iloc[i-5:i+6].min():
                level_price = low.iloc[i]
                
                if current_price > level_price:
                    distance = (current_price - level_price) / current_price * 100
                    
                    # Count touches
                    touches = sum(1 for j in range(len(df)) 
                                 if abs(low.iloc[j] - level_price) / level_price < 0.01)
                    
                    if 1 < distance < 10 and touches >= 2:
                        levels.append({
                            'type': 'support',
                            'direction': 'LONG',
                            'price': level_price,
                            'zone_bottom': level_price * 0.995,
                            'zone_top': level_price * 1.005,
                            'distance': distance,
                            'strength': min(5, touches),
                            'description': f"Support level - Tested {touches}x"
                        })
        
        # Find swing highs (resistance)
        for i in range(5, len(df) - 5):
            if high.iloc[i] == high.iloc[i-5:i+6].max():
                level_price = high.iloc[i]
                
                if current_price < level_price:
                    distance = (level_price - current_price) / current_price * 100
                    
                    touches = sum(1 for j in range(len(df)) 
                                 if abs(high.iloc[j] - level_price) / level_price < 0.01)
                    
                    if 1 < distance < 10 and touches >= 2:
                        levels.append({
                            'type': 'resistance',
                            'direction': 'SHORT',
                            'price': level_price,
                            'zone_bottom': level_price * 0.995,
                            'zone_top': level_price * 1.005,
                            'distance': distance,
                            'strength': min(5, touches),
                            'description': f"Resistance level - Tested {touches}x"
                        })
        
        # Remove duplicates (levels within 1% of each other)
        unique_levels = []
        for level in sorted(levels, key=lambda x: x['strength'], reverse=True):
            is_duplicate = False
            for existing in unique_levels:
                if abs(level['price'] - existing['price']) / level['price'] < 0.01:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_levels.append(level)
        
        return unique_levels[:10]  # Top 10 levels
    
    def _analyze_level(self, df: pd.DataFrame, symbol: str, 
                       current_price: float, level: Dict) -> Optional[PredictiveSignal]:
        """Analyze a single level and create predictive signal if valid"""
        
        distance = level['distance']
        level_price = level['price']
        direction = level['direction']
        
        # Determine stage based on distance
        if distance < 1.5:
            stage = SignalStage.AT_LEVEL
            title = f"🎯 AT {level['type'].upper().replace('_', ' ')}"
            action = "Entry zone active - Consider entering NOW"
            urgency = "high"
        elif distance < 3:
            stage = SignalStage.APPROACHING
            title = f"🔮 APPROACHING {level['type'].upper().replace('_', ' ')}"
            action = f"Set limit order at {self._fmt(level_price)} - {distance:.1f}% away"
            urgency = "medium"
        elif distance < 5:
            stage = SignalStage.APPROACHING
            title = f"📡 WATCHING {level['type'].upper().replace('_', ' ')}"
            action = f"Set alert at {self._fmt(level_price)} - {distance:.1f}% away"
            urgency = "low"
        else:
            return None  # Too far
        
        # Calculate confidence
        confidence = 40  # Base
        confidence += level['strength'] * 8  # Up to +40 from strength
        if 'ob' in level['type']:
            confidence += 15  # OBs are higher quality
        if distance < 2:
            confidence += 10  # Closer = more actionable
        confidence = min(100, confidence)
        
        # Calculate suggested levels
        zone_size = abs(level.get('zone_top', level_price) - level.get('zone_bottom', level_price))
        
        if direction == 'LONG':
            suggested_entry = level_price * 1.002  # Slightly above level
            suggested_sl = level.get('zone_bottom', level_price) * 0.99
            risk = suggested_entry - suggested_sl
            suggested_tp1 = suggested_entry + (risk * 1.5)
        else:
            suggested_entry = level_price * 0.998
            suggested_sl = level.get('zone_top', level_price) * 1.01
            risk = suggested_sl - suggested_entry
            suggested_tp1 = suggested_entry - (risk * 1.5)
        
        return PredictiveSignal(
            symbol=symbol,
            stage=stage,
            level_type=level['type'],
            level_price=level_price,
            current_price=current_price,
            distance_pct=distance,
            direction=direction,
            confidence=confidence,
            title=title,
            description=level['description'],
            action=action,
            suggested_entry=suggested_entry,
            suggested_sl=suggested_sl,
            suggested_tp1=suggested_tp1
        )
    
    def _fmt(self, price: float) -> str:
        if price >= 1:
            return f"${price:,.2f}"
        return f"${price:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE MONITOR ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

class TradeAlertMonitor:
    """
    Monitors active trades and generates alerts
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
    
    def check_trade(self, trade: Dict, current_price: float, 
                    prev_price: Optional[float] = None) -> List[Alert]:
        """
        Check a trade for alert conditions
        
        Args:
            trade: Trade dict with entry, sl, tp1, tp2, tp3
            current_price: Current market price
            prev_price: Previous price (for direction detection)
        
        Returns:
            List of triggered alerts
        """
        self.alerts = []
        
        # Convert all values to float to handle string inputs
        try:
            entry = float(trade.get('entry', 0) or 0)
            sl = float(trade.get('stop_loss', 0) or 0)
            tp1 = float(trade.get('tp1', 0) or 0)
            tp2 = float(trade.get('tp2', 0) or 0)
            tp3 = float(trade.get('tp3', 0) or 0)
            current_price = float(current_price) if current_price else 0
        except (ValueError, TypeError):
            return []  # Can't process invalid data
        
        symbol = trade.get('symbol', 'UNKNOWN')
        direction = trade.get('direction', 'LONG').upper()
        
        # Check if TPs were already hit (don't alert for approaching if already hit)
        tp1_already_hit = trade.get('tp1_hit', False)
        tp2_already_hit = trade.get('tp2_hit', False)
        tp3_already_hit = trade.get('tp3_hit', False)
        
        if entry <= 0 or current_price <= 0:
            return []
        
        # Calculate distances
        if direction == 'LONG':
            # For LONG: price going UP is good
            dist_to_tp1 = ((tp1 - current_price) / current_price * 100) if tp1 > 0 else 999
            dist_to_tp2 = ((tp2 - current_price) / current_price * 100) if tp2 > 0 else 999
            dist_to_tp3 = ((tp3 - current_price) / current_price * 100) if tp3 > 0 else 999
            dist_to_sl = ((current_price - sl) / current_price * 100) if sl > 0 else 999
            
            # Check TP approaches - ONLY if not already hit
            if not tp1_already_hit and 0 < dist_to_tp1 <= 1:
                self.alerts.append(Alert(
                    alert_type=AlertType.APPROACHING_TP1,
                    symbol=symbol,
                    message=f"Price {dist_to_tp1:.1f}% from TP1! Consider partial profit.",
                    price=current_price,
                    target_price=tp1,
                    distance_pct=dist_to_tp1,
                    urgency="high",
                    sound=True
                ))
            elif not tp1_already_hit and 0 < dist_to_tp1 <= 2:
                self.alerts.append(Alert(
                    alert_type=AlertType.APPROACHING_TP1,
                    symbol=symbol,
                    message=f"Approaching TP1 ({dist_to_tp1:.1f}% away)",
                    price=current_price,
                    target_price=tp1,
                    distance_pct=dist_to_tp1,
                    urgency="medium"
                ))
            
            # Check if TP1 HIT (only alert if not already marked as hit)
            if not tp1_already_hit and dist_to_tp1 <= 0:
                self.alerts.append(Alert(
                    alert_type=AlertType.HIT_TP1,
                    symbol=symbol,
                    message=f"🎉 TP1 HIT! Take 33% profit at {self._fmt(tp1)}",
                    price=current_price,
                    target_price=tp1,
                    distance_pct=0,
                    urgency="high",
                    sound=True
                ))
            
            # Similar for TP2, TP3...
            if not tp2_already_hit and 0 < dist_to_tp2 <= 1:
                self.alerts.append(Alert(
                    alert_type=AlertType.APPROACHING_TP2,
                    symbol=symbol,
                    message=f"Approaching TP2! ({dist_to_tp2:.1f}% away)",
                    price=current_price,
                    target_price=tp2,
                    distance_pct=dist_to_tp2,
                    urgency="high",
                    sound=True
                ))
            
            if not tp2_already_hit and dist_to_tp2 <= 0:
                self.alerts.append(Alert(
                    alert_type=AlertType.HIT_TP2,
                    symbol=symbol,
                    message=f"🎉 TP2 HIT! Take another 33% profit",
                    price=current_price,
                    target_price=tp2,
                    distance_pct=0,
                    urgency="high",
                    sound=True
                ))
            
            if not tp3_already_hit and dist_to_tp3 <= 0:
                self.alerts.append(Alert(
                    alert_type=AlertType.HIT_TP3,
                    symbol=symbol,
                    message=f"🏆 TP3 HIT! Full target reached!",
                    price=current_price,
                    target_price=tp3,
                    distance_pct=0,
                    urgency="high",
                    sound=True
                ))
            
            # Check SL approach - but only warn if trade is actually in danger
            # Calculate P&L to determine if we're moving towards SL
            pnl_pct = (current_price - entry) / entry * 100 if direction == 'LONG' else (entry - current_price) / entry * 100
            
            # Only warn about SL if:
            # 1. Trade is in LOSS (pnl < 0), OR
            # 2. Trade is barely profitable but very close to SL
            is_in_danger = pnl_pct < 0 or (pnl_pct < 0.5 and dist_to_sl <= 1)
            
            if 0 < dist_to_sl <= 1 and is_in_danger:
                self.alerts.append(Alert(
                    alert_type=AlertType.APPROACHING_SL,
                    symbol=symbol,
                    message=f"⚠️ WARNING: {dist_to_sl:.1f}% from Stop Loss! (P&L: {pnl_pct:+.1f}%)",
                    price=current_price,
                    target_price=sl,
                    distance_pct=dist_to_sl,
                    urgency="high",
                    sound=True
                ))
            elif 0 < dist_to_sl <= 3 and pnl_pct < -1:  # Only medium alert if actually losing
                self.alerts.append(Alert(
                    alert_type=AlertType.APPROACHING_SL,
                    symbol=symbol,
                    message=f"Caution: {dist_to_sl:.1f}% from Stop Loss (P&L: {pnl_pct:+.1f}%)",
                    price=current_price,
                    target_price=sl,
                    distance_pct=dist_to_sl,
                    urgency="medium"
                ))
            
            if dist_to_sl <= 0:
                self.alerts.append(Alert(
                    alert_type=AlertType.HIT_SL,
                    symbol=symbol,
                    message=f"🛑 STOP LOSS HIT at {self._fmt(sl)}",
                    price=current_price,
                    target_price=sl,
                    distance_pct=0,
                    urgency="high",
                    sound=True
                ))
            
            # Check DCA opportunity (price dropped but not to SL)
            # pnl_pct already calculated above
            if -8 < pnl_pct < -3 and dist_to_sl > 3:
                self.alerts.append(Alert(
                    alert_type=AlertType.DCA_OPPORTUNITY,
                    symbol=symbol,
                    message=f"💰 DCA opportunity? Price down {abs(pnl_pct):.1f}% from entry",
                    price=current_price,
                    target_price=entry,
                    distance_pct=abs(pnl_pct),
                    urgency="low"
                ))
        
        else:
            # SHORT trade logic (inverted)
            # Similar logic but inverted directions
            pass
        
        return self.alerts
    
    def _fmt(self, price: float) -> str:
        if price >= 1:
            return f"${price:,.2f}"
        return f"${price:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def find_approaching_setups(df: pd.DataFrame, symbol: str) -> List[PredictiveSignal]:
    """Find all approaching setups for a symbol"""
    analyzer = PredictiveAnalyzer()
    return analyzer.analyze(df, symbol)


def check_trade_alerts(trade: Dict, current_price: float) -> List[Alert]:
    """Check a trade for alerts"""
    monitor = TradeAlertMonitor()
    return monitor.check_trade(trade, current_price)


def format_alert_html(alert: Alert) -> str:
    """Format alert as HTML for display"""
    return f"""
    <div style='background: {alert.color}22; border-left: 3px solid {alert.color}; 
                padding: 10px 12px; border-radius: 6px; margin: 5px 0;'>
        <div style='color: {alert.color}; font-weight: bold;'>
            {alert.emoji} {alert.symbol}: {alert.alert_type.value.replace('_', ' ').title()}
        </div>
        <div style='color: #ddd; margin-top: 4px;'>{alert.message}</div>
        <div style='color: #888; font-size: 0.8em; margin-top: 4px;'>
            Current: {alert.price:.2f} | Target: {alert.target_price:.2f}
        </div>
    </div>
    """


def format_predictive_signal_html(signal: PredictiveSignal) -> str:
    """Format predictive signal as HTML"""
    return f"""
    <div style='background: {signal.stage_color}15; border-left: 4px solid {signal.stage_color}; 
                padding: 12px 15px; border-radius: 8px; margin: 8px 0;'>
        <div style='color: {signal.stage_color}; font-weight: bold; font-size: 1.1em;'>
            {signal.stage_emoji} {signal.symbol} - {signal.title}
        </div>
        <div style='color: #ccc; margin-top: 6px;'>{signal.description}</div>
        <div style='color: #fff; margin-top: 8px; background: #1a1a2e; padding: 8px; border-radius: 4px;'>
            <strong>Level:</strong> ${signal.level_price:,.2f} ({signal.distance_pct:.1f}% away)<br>
            <strong>Confidence:</strong> {signal.confidence}%<br>
            <strong>Action:</strong> {signal.action}
        </div>
        <div style='color: #888; margin-top: 8px; font-size: 0.85em;'>
            If triggered → Entry: ${signal.suggested_entry:,.2f} | SL: ${signal.suggested_sl:,.2f} | TP1: ${signal.suggested_tp1:,.2f}
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL ACTIVITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stealth_accumulation(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect stealth accumulation: Volume increasing while price stays flat
    This often precedes big moves UP - institutions quietly building positions
    """
    if df is None or len(df) < lookback:
        return {'detected': False, 'score': 0, 'reasons': []}
    
    recent = df.tail(lookback)
    reasons = []
    score = 0
    
    # Price range (should be tight - consolidating)
    price_range = (recent['High'].max() - recent['Low'].min()) / recent['Close'].mean() * 100
    
    # Volume trend (should be increasing)
    vol_first_half = recent['Volume'].head(lookback // 2).mean()
    vol_second_half = recent['Volume'].tail(lookback // 2).mean()
    vol_increase_pct = (vol_second_half / vol_first_half - 1) * 100 if vol_first_half > 0 else 0
    vol_increasing = vol_increase_pct > 15
    
    # OBV trend (should be rising even if price flat)
    obv = (np.sign(recent['Close'].diff()) * recent['Volume']).fillna(0).cumsum()
    obv_start = obv.iloc[lookback // 4]
    obv_end = obv.iloc[-1]
    obv_rising = obv_end > obv_start * 1.1
    
    # Check for higher lows (accumulation pattern)
    lows = recent['Low'].values
    higher_lows = all(lows[i] >= lows[i-1] * 0.99 for i in range(len(lows)//2, len(lows)))
    
    # Score components
    if price_range < 5:
        score += 20
        reasons.append(f"Tight range ({price_range:.1f}%) - Consolidation")
    elif price_range < 8:
        score += 10
        reasons.append(f"Moderate range ({price_range:.1f}%)")
    
    if vol_increasing:
        score += 25
        reasons.append(f"Volume building (+{vol_increase_pct:.0f}%)")
    
    if obv_rising:
        score += 30
        reasons.append("OBV rising (buying pressure hidden)")
    
    if higher_lows:
        score += 15
        reasons.append("Higher lows pattern (demand increasing)")
    
    is_stealth = score >= 50
    
    return {
        'detected': is_stealth,
        'score': score,
        'price_range_pct': price_range,
        'volume_increase_pct': vol_increase_pct,
        'obv_rising': obv_rising,
        'higher_lows': higher_lows,
        'reasons': reasons,
        'interpretation': "🦈 STEALTH ACCUMULATION - Smart money quietly buying before breakout" if is_stealth else "Normal activity"
    }


def detect_stealth_distribution(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Detect stealth distribution: Volume increasing while price stays flat or slightly up
    This often precedes big moves DOWN - institutions quietly selling positions
    """
    if df is None or len(df) < lookback:
        return {'detected': False, 'score': 0, 'reasons': []}
    
    recent = df.tail(lookback)
    reasons = []
    score = 0
    
    # Price trend (flat or slightly up)
    price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0] * 100
    
    # Volume trend
    vol_first_half = recent['Volume'].head(lookback // 2).mean()
    vol_second_half = recent['Volume'].tail(lookback // 2).mean()
    vol_increase_pct = (vol_second_half / vol_first_half - 1) * 100 if vol_first_half > 0 else 0
    vol_increasing = vol_increase_pct > 15
    
    # OBV trend (should be falling even if price flat/up)
    obv = (np.sign(recent['Close'].diff()) * recent['Volume']).fillna(0).cumsum()
    obv_start = obv.iloc[lookback // 4]
    obv_end = obv.iloc[-1]
    obv_falling = obv_end < obv_start * 0.9
    
    # Check for lower highs (distribution pattern)
    highs = recent['High'].values
    lower_highs = all(highs[i] <= highs[i-1] * 1.01 for i in range(len(highs)//2, len(highs)))
    
    # Score components
    if price_change >= -2 and price_change <= 5:
        score += 15
        reasons.append(f"Price stable ({price_change:+.1f}%) while selling")
    
    if vol_increasing:
        score += 25
        reasons.append(f"Volume building (+{vol_increase_pct:.0f}%)")
    
    if obv_falling:
        score += 35
        reasons.append("OBV falling (selling pressure hidden)")
    
    if lower_highs:
        score += 15
        reasons.append("Lower highs pattern (supply increasing)")
    
    is_distribution = score >= 50
    
    return {
        'detected': is_distribution,
        'score': score,
        'price_change_pct': price_change,
        'volume_increase_pct': vol_increase_pct,
        'obv_falling': obv_falling,
        'lower_highs': lower_highs,
        'reasons': reasons,
        'interpretation': "🦈 STEALTH DISTRIBUTION - Smart money quietly selling before drop" if is_distribution else "Normal activity"
    }


def detect_institutional_activity(df: pd.DataFrame, mf: Dict = None) -> Dict:
    """
    Comprehensive institutional activity detection
    Combines multiple signals to detect smart money behavior
    """
    if df is None or len(df) < 50:
        return {
            'activity_type': 'neutral',
            'score': 0,
            'confidence': 0,
            'signals': [],
            'recommendation': ''
        }
    
    signals = []
    bullish_score = 0
    bearish_score = 0
    
    # 1. Stealth Accumulation
    stealth_accum = detect_stealth_accumulation(df)
    if stealth_accum['detected']:
        bullish_score += 30
        signals.append({
            'type': 'stealth_accumulation',
            'emoji': '🦈',
            'message': 'Stealth Accumulation Detected',
            'detail': 'Volume building while price flat - institutions buying quietly',
            'color': '#00d4aa',
            'reasons': stealth_accum['reasons']
        })
    
    # 2. Stealth Distribution
    stealth_dist = detect_stealth_distribution(df)
    if stealth_dist['detected']:
        bearish_score += 30
        signals.append({
            'type': 'stealth_distribution',
            'emoji': '🦈',
            'message': 'Stealth Distribution Detected',
            'detail': 'Volume building while price flat/up - institutions selling quietly',
            'color': '#ff4444',
            'reasons': stealth_dist['reasons']
        })
    
    # 3. Money Flow signals (if provided)
    if mf:
        if mf.get('is_accumulating'):
            bullish_score += 20
            signals.append({
                'type': 'money_inflow',
                'emoji': '💰',
                'message': 'Money Flowing IN',
                'detail': f"OBV rising + CMF positive ({mf.get('cmf', 0):.3f})",
                'color': '#00d4aa',
                'reasons': ['OBV trend rising', 'Positive Chaikin Money Flow']
            })
        elif mf.get('is_distributing'):
            bearish_score += 20
            signals.append({
                'type': 'money_outflow',
                'emoji': '💸',
                'message': 'Money Flowing OUT',
                'detail': f"OBV falling + CMF negative ({mf.get('cmf', 0):.3f})",
                'color': '#ff4444',
                'reasons': ['OBV trend falling', 'Negative Chaikin Money Flow']
            })
        
        # MFI extremes
        mfi = mf.get('mfi', 50)
        if mfi < 20:
            bullish_score += 25
            signals.append({
                'type': 'capitulation',
                'emoji': '🔥',
                'message': f'MFI Capitulation ({mfi:.0f})',
                'detail': 'Extreme selling exhaustion - smart money often buys here',
                'color': '#00d4ff',
                'reasons': [f'MFI at {mfi:.0f} (below 20)', 'Indicates seller exhaustion']
            })
        elif mfi > 80:
            bearish_score += 25
            signals.append({
                'type': 'euphoria',
                'emoji': '⚠️',
                'message': f'MFI Euphoria ({mfi:.0f})',
                'detail': 'Extreme buying - smart money often sells here',
                'color': '#ff9500',
                'reasons': [f'MFI at {mfi:.0f} (above 80)', 'Indicates buyer exhaustion']
            })
    
    # 4. Volume anomalies
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    recent_vol = df['Volume'].tail(3).mean()
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
    
    if vol_ratio > 2:
        # Determine direction
        recent_close = df['Close'].tail(3)
        if recent_close.iloc[-1] > recent_close.iloc[0]:
            bullish_score += 15
            signals.append({
                'type': 'volume_spike_bullish',
                'emoji': '📊',
                'message': f'Bullish Volume Spike ({vol_ratio:.1f}x)',
                'detail': 'Unusually high volume on up move',
                'color': '#00d4aa',
                'reasons': [f'Volume {vol_ratio:.1f}x average', 'Price moving up']
            })
        else:
            bearish_score += 15
            signals.append({
                'type': 'volume_spike_bearish',
                'emoji': '📊',
                'message': f'Bearish Volume Spike ({vol_ratio:.1f}x)',
                'detail': 'Unusually high volume on down move',
                'color': '#ff4444',
                'reasons': [f'Volume {vol_ratio:.1f}x average', 'Price moving down']
            })
    
    # Determine overall activity
    net_score = bullish_score - bearish_score
    
    if net_score >= 40:
        activity_type = 'strong_bullish'
        recommendation = '🟢 STRONG INSTITUTIONAL BUYING - Consider long positions'
        color = '#00ff88'
    elif net_score >= 20:
        activity_type = 'bullish'
        recommendation = '🟢 Institutional buying detected - Favor long setups'
        color = '#00d4aa'
    elif net_score <= -40:
        activity_type = 'strong_bearish'
        recommendation = '🔴 STRONG INSTITUTIONAL SELLING - Avoid longs, consider shorts'
        color = '#ff4444'
    elif net_score <= -20:
        activity_type = 'bearish'
        recommendation = '🔴 Institutional selling detected - Be cautious on longs'
        color = '#ff6666'
    else:
        activity_type = 'neutral'
        recommendation = '⚪ No clear institutional bias - Wait for clearer signals'
        color = '#888888'
    
    confidence = min(100, abs(net_score) + 30) if signals else 0
    
    return {
        'activity_type': activity_type,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'net_score': net_score,
        'confidence': confidence,
        'signals': signals,
        'recommendation': recommendation,
        'color': color,
        'stealth_accum': stealth_accum,
        'stealth_dist': stealth_dist
    }


def format_institutional_activity_html(activity: Dict) -> str:
    """Format institutional activity as HTML"""
    signals_html = ""
    for sig in activity.get('signals', []):
        reasons_html = "<br>".join([f"• {r}" for r in sig.get('reasons', [])])
        signals_html += f"""
        <div style='background: {sig['color']}15; border-left: 3px solid {sig['color']}; 
                    padding: 10px 12px; border-radius: 6px; margin: 8px 0;'>
            <div style='color: {sig['color']}; font-weight: bold;'>
                {sig['emoji']} {sig['message']}
            </div>
            <div style='color: #ccc; margin-top: 4px; font-size: 0.9em;'>{sig['detail']}</div>
            <div style='color: #888; margin-top: 6px; font-size: 0.85em;'>{reasons_html}</div>
        </div>
        """
    
    return f"""
    <div style='background: #1a1a2e; border-radius: 12px; padding: 15px; margin: 10px 0;'>
        <div style='color: {activity['color']}; font-size: 1.2em; font-weight: bold; margin-bottom: 10px;'>
            🦈 Institutional Activity Analysis
        </div>
        <div style='color: #fff; background: {activity['color']}22; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
            {activity['recommendation']}
        </div>
        <div style='color: #888; margin-bottom: 10px;'>
            Confidence: <strong style='color: #fff;'>{activity['confidence']}%</strong> | 
            Bullish: <span style='color: #00d4aa;'>{activity['bullish_score']}</span> | 
            Bearish: <span style='color: #ff4444;'>{activity['bearish_score']}</span>
        </div>
        {signals_html}
    </div>
    """

