"""
Hybrid Engine - Combines Rule-Based and ML Predictions
========================================================

Merges MASTER_RULES logic with ML predictions for best results.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .ml_engine import MLPrediction, MLEngine, get_ml_prediction


@dataclass
class HybridPrediction:
    """Combined prediction from both engines"""
    
    # Final recommendation
    direction: str              # LONG, SHORT, WAIT
    confidence: float           # 0-100
    action: str                 # STRONG_LONG, BUY, WAIT, etc.
    
    # Source agreement
    rule_direction: str         # What MASTER_RULES says
    rule_score: float           # MASTER_RULES score
    ml_direction: str           # What ML says
    ml_confidence: float        # ML confidence
    engines_agree: bool         # Do they agree?
    
    # Levels (from ML optimizer)
    entry: float
    tp1: float
    tp2: float
    stop_loss: float
    tp1_pct: float
    tp2_pct: float
    sl_pct: float
    
    # Risk/Reward
    rr_ratio: float
    win_probability: float
    
    # Stories (from MASTER_RULES)
    whale_story: str
    position_story: str
    oi_story: str
    
    # ML insights
    ml_reasoning: str
    top_features: list
    
    # Historical context
    similar_trades: int
    historical_win_rate: float
    
    def to_dict(self) -> Dict:
        return {
            'direction': self.direction,
            'confidence': self.confidence,
            'action': self.action,
            'rule_direction': self.rule_direction,
            'rule_score': self.rule_score,
            'ml_direction': self.ml_direction,
            'ml_confidence': self.ml_confidence,
            'engines_agree': self.engines_agree,
            'entry': self.entry,
            'tp1': self.tp1,
            'tp2': self.tp2,
            'stop_loss': self.stop_loss,
            'tp1_pct': self.tp1_pct,
            'tp2_pct': self.tp2_pct,
            'sl_pct': self.sl_pct,
            'rr_ratio': self.rr_ratio,
            'win_probability': self.win_probability,
            'whale_story': self.whale_story,
            'position_story': self.position_story,
            'oi_story': self.oi_story,
            'ml_reasoning': self.ml_reasoning,
            'top_features': self.top_features,
            'similar_trades': self.similar_trades,
            'historical_win_rate': self.historical_win_rate,
        }


class HybridEngine:
    """
    Combines MASTER_RULES and ML predictions.
    
    Strategy:
    - If both agree: HIGH confidence
    - If disagree: Use the one with higher confidence, but flag conflict
    - ML provides optimized TP/SL levels
    - Rules provide context/stories
    """
    
    def __init__(self):
        self.ml_engine = MLEngine()
    
    def predict(
        self,
        # Rule-based decision (from MASTER_RULES)
        rule_decision: object,  # TradeDecision from MASTER_RULES
        
        # Data for ML prediction
        whale_pct: float = 50,
        retail_pct: float = 50,
        oi_change: float = 0,
        price_change_24h: float = 0,
        position_pct: float = 50,
        current_price: float = 0,
        swing_high: float = 0,
        swing_low: float = 0,
        ta_score: float = 50,
        rsi: float = 50,
        trend: str = 'NEUTRAL',
        money_flow_phase: str = 'NEUTRAL',
        at_bullish_ob: bool = False,
        at_bearish_ob: bool = False,
        btc_correlation: float = 0,
        btc_trend: str = 'NEUTRAL',
        historical_win_rate: float = None,
        similar_setup_count: int = 0,
        **kwargs
    ) -> HybridPrediction:
        """
        Generate hybrid prediction combining rules and ML.
        """
        
        # Get ML prediction
        ml_pred = self.ml_engine.predict(
            whale_pct=whale_pct,
            retail_pct=retail_pct,
            oi_change=oi_change,
            price_change_24h=price_change_24h,
            position_pct=position_pct,
            swing_high=swing_high,
            swing_low=swing_low,
            current_price=current_price,
            ta_score=ta_score,
            rsi=rsi,
            trend=trend,
            money_flow_phase=money_flow_phase,
            at_bullish_ob=at_bullish_ob,
            at_bearish_ob=at_bearish_ob,
            btc_correlation=btc_correlation,
            btc_trend=btc_trend,
            historical_win_rate=historical_win_rate,
            similar_setup_count=similar_setup_count,
            **kwargs
        )
        
        # Extract rule-based results
        rule_direction = rule_decision.trade_direction if hasattr(rule_decision, 'trade_direction') else 'WAIT'
        rule_score = rule_decision.total_score if hasattr(rule_decision, 'total_score') else 50
        rule_action = rule_decision.action if hasattr(rule_decision, 'action') else 'WAIT'
        
        # Check agreement
        engines_agree = self._directions_agree(rule_direction, ml_pred.direction)
        
        # Determine final direction and confidence
        if engines_agree:
            # Both agree - high confidence
            direction = ml_pred.direction
            confidence = (ml_pred.confidence + rule_score) / 2 + 10  # Bonus for agreement
            confidence = min(95, confidence)
        else:
            # Disagreement - use higher confidence one, but reduce overall confidence
            if ml_pred.confidence > rule_score:
                direction = ml_pred.direction
                confidence = ml_pred.confidence - 10  # Penalty for conflict
            else:
                direction = rule_direction
                confidence = rule_score - 10
            confidence = max(30, confidence)
        
        # Determine action word
        if direction == 'LONG':
            if confidence >= 75:
                action = 'STRONG_LONG'
            elif confidence >= 60:
                action = 'BUY'
            else:
                action = 'LEAN_LONG'
        elif direction == 'SHORT':
            if confidence >= 75:
                action = 'STRONG_SHORT'
            elif confidence >= 60:
                action = 'SELL'
            else:
                action = 'LEAN_SHORT'
        else:
            action = 'WAIT'
        
        # Calculate actual levels from ML percentages
        if current_price > 0 and direction != 'WAIT':
            if direction == 'LONG':
                entry = current_price
                tp1 = current_price * (1 + ml_pred.optimal_tp1_pct / 100)
                tp2 = current_price * (1 + ml_pred.optimal_tp2_pct / 100)
                stop_loss = current_price * (1 - ml_pred.optimal_sl_pct / 100)
            else:  # SHORT
                entry = current_price
                tp1 = current_price * (1 - ml_pred.optimal_tp1_pct / 100)
                tp2 = current_price * (1 - ml_pred.optimal_tp2_pct / 100)
                stop_loss = current_price * (1 + ml_pred.optimal_sl_pct / 100)
        else:
            entry = current_price
            tp1 = 0
            tp2 = 0
            stop_loss = 0
        
        # Get stories from rule decision
        whale_story = rule_decision.whale_story if hasattr(rule_decision, 'whale_story') else ''
        position_story = rule_decision.position_story if hasattr(rule_decision, 'position_story') else ''
        oi_story = rule_decision.oi_story if hasattr(rule_decision, 'oi_story') else ''
        
        return HybridPrediction(
            direction=direction,
            confidence=confidence,
            action=action,
            rule_direction=rule_direction,
            rule_score=rule_score,
            ml_direction=ml_pred.direction,
            ml_confidence=ml_pred.confidence,
            engines_agree=engines_agree,
            entry=entry,
            tp1=tp1,
            tp2=tp2,
            stop_loss=stop_loss,
            tp1_pct=ml_pred.optimal_tp1_pct,
            tp2_pct=ml_pred.optimal_tp2_pct,
            sl_pct=ml_pred.optimal_sl_pct,
            rr_ratio=ml_pred.expected_rr,
            win_probability=ml_pred.win_probability,
            whale_story=whale_story,
            position_story=position_story,
            oi_story=oi_story,
            ml_reasoning=ml_pred.reasoning,
            top_features=ml_pred.top_features,
            similar_trades=ml_pred.similar_trades_count,
            historical_win_rate=ml_pred.similar_trades_win_rate,
        )
    
    def _directions_agree(self, dir1: str, dir2: str) -> bool:
        """Check if two directions agree"""
        # Exact match
        if dir1 == dir2:
            return True
        
        # Similar directions
        bullish = {'LONG', 'BUY', 'LEAN_LONG', 'LEAN_BULLISH', 'BULLISH'}
        bearish = {'SHORT', 'SELL', 'LEAN_SHORT', 'LEAN_BEARISH', 'BEARISH'}
        
        if dir1 in bullish and dir2 in bullish:
            return True
        if dir1 in bearish and dir2 in bearish:
            return True
        
        return False


# Global instance
_hybrid_engine = None

def get_hybrid_prediction(rule_decision, **kwargs) -> HybridPrediction:
    """Convenience function for hybrid prediction"""
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridEngine()
    return _hybrid_engine.predict(rule_decision, **kwargs)
