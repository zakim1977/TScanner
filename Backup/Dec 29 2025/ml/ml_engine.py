"""
ML Engine - Main Interface for Predictions
============================================

Provides ML-based predictions with the same interface as MASTER_RULES.
Falls back to rule-based if model not trained.
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .feature_extractor import extract_features, extract_features_from_dict, FeatureSet, FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLPrediction:
    """ML prediction result - mirrors MASTER_RULES TradeDecision structure"""
    
    # Core prediction
    direction: str              # 'LONG', 'SHORT', 'WAIT'
    confidence: float           # 0-100
    predicted_move: float       # Expected % move in timeframe
    
    # Optimal levels (learned from history)
    optimal_tp1_pct: float      # Optimal TP1 distance %
    optimal_tp2_pct: float      # Optimal TP2 distance %
    optimal_sl_pct: float       # Optimal SL distance %
    
    # Risk/Reward
    expected_rr: float          # Expected risk/reward ratio
    win_probability: float      # Probability of hitting TP1
    
    # Explanation
    top_features: List[Tuple[str, float, float]]  # (name, value, importance)
    reasoning: str              # Human-readable explanation
    
    # Historical context
    similar_trades_count: int
    similar_trades_win_rate: float
    similar_trades_avg_return: float
    
    # Meta
    model_version: str
    prediction_time: str
    
    def to_dict(self) -> Dict:
        return {
            'direction': self.direction,
            'confidence': self.confidence,
            'predicted_move': self.predicted_move,
            'optimal_tp1_pct': self.optimal_tp1_pct,
            'optimal_tp2_pct': self.optimal_tp2_pct,
            'optimal_sl_pct': self.optimal_sl_pct,
            'expected_rr': self.expected_rr,
            'win_probability': self.win_probability,
            'top_features': self.top_features,
            'reasoning': self.reasoning,
            'similar_trades_count': self.similar_trades_count,
            'similar_trades_win_rate': self.similar_trades_win_rate,
            'similar_trades_avg_return': self.similar_trades_avg_return,
            'model_version': self.model_version,
            'prediction_time': self.prediction_time,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ML ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MLEngine:
    """
    Main ML prediction engine.
    
    Loads trained models and provides predictions.
    Falls back to heuristic-based predictions if models not available.
    """
    
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    
    def __init__(self):
        self.direction_model = None
        self.tp_sl_model = None
        self.feature_scaler = None
        self.model_metadata = {}
        self.is_trained = False
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            direction_path = os.path.join(self.MODEL_DIR, 'direction_model.pkl')
            tp_sl_path = os.path.join(self.MODEL_DIR, 'tp_sl_model.pkl')
            scaler_path = os.path.join(self.MODEL_DIR, 'feature_scaler.pkl')
            metadata_path = os.path.join(self.MODEL_DIR, 'model_metadata.json')
            
            if os.path.exists(direction_path):
                with open(direction_path, 'rb') as f:
                    self.direction_model = pickle.load(f)
                    
            if os.path.exists(tp_sl_path):
                with open(tp_sl_path, 'rb') as f:
                    self.tp_sl_model = pickle.load(f)
                    
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                    
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            self.is_trained = self.direction_model is not None
            
            if self.is_trained:
                print(f"✅ ML models loaded successfully (v{self.model_metadata.get('version', 'unknown')})")
            else:
                print("⚠️ ML models not found - using heuristic predictions")
                
        except Exception as e:
            print(f"⚠️ Error loading ML models: {e}")
            self.is_trained = False
    
    def is_model_loaded(self) -> bool:
        """Check if ML models are loaded and ready"""
        return self.is_trained
    
    def predict(
        self,
        # Same parameters as extract_features
        whale_pct: float = 50,
        retail_pct: float = 50,
        funding_rate: float = 0,
        oi_change: float = 0,
        oi_signal: str = 'NEUTRAL',
        price_change_24h: float = 0,
        price_change_1h: float = 0,
        position_pct: float = 50,
        swing_high: float = 0,
        swing_low: float = 0,
        current_price: float = 0,
        ta_score: float = 50,
        rsi: float = 50,
        trend: str = 'NEUTRAL',
        atr: float = 0,
        money_flow_phase: str = 'NEUTRAL',
        volume_ratio: float = 1.0,
        at_bullish_ob: bool = False,
        at_bearish_ob: bool = False,
        near_support: bool = False,
        near_resistance: bool = False,
        btc_correlation: float = 0,
        btc_trend: str = 'NEUTRAL',
        fear_greed: int = 50,
        is_weekend: bool = False,
        historical_win_rate: float = None,
        similar_setup_count: int = 0,
        avg_historical_return: float = 0,
    ) -> MLPrediction:
        """
        Generate ML prediction.
        
        Uses trained model if available, otherwise falls back to heuristics.
        """
        
        # Extract features
        feature_set = extract_features(
            whale_pct=whale_pct,
            retail_pct=retail_pct,
            funding_rate=funding_rate,
            oi_change=oi_change,
            oi_signal=oi_signal,
            price_change_24h=price_change_24h,
            price_change_1h=price_change_1h,
            position_pct=position_pct,
            swing_high=swing_high,
            swing_low=swing_low,
            current_price=current_price,
            ta_score=ta_score,
            rsi=rsi,
            trend=trend,
            atr=atr,
            money_flow_phase=money_flow_phase,
            volume_ratio=volume_ratio,
            at_bullish_ob=at_bullish_ob,
            at_bearish_ob=at_bearish_ob,
            near_support=near_support,
            near_resistance=near_resistance,
            btc_correlation=btc_correlation,
            btc_trend=btc_trend,
            fear_greed=fear_greed,
            is_weekend=is_weekend,
            historical_win_rate=historical_win_rate,
            similar_setup_count=similar_setup_count,
            avg_historical_return=avg_historical_return,
        )
        
        if self.is_trained:
            return self._predict_with_model(feature_set)
        else:
            return self._predict_heuristic(feature_set)
    
    def _predict_with_model(self, feature_set: FeatureSet) -> MLPrediction:
        """Predict using trained ML model"""
        
        # Scale features if scaler available
        features = feature_set.features.reshape(1, -1)
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)
        
        # Direction prediction
        direction_proba = self.direction_model.predict_proba(features)[0]
        classes = self.direction_model.classes_
        
        # Get predicted class and confidence
        pred_idx = np.argmax(direction_proba)
        direction_raw = classes[pred_idx]
        
        # Map numeric classes to string labels if needed
        CLASS_MAP = {0: 'SHORT', 1: 'WAIT', 2: 'LONG', 
                     '0': 'SHORT', '1': 'WAIT', '2': 'LONG',
                     'SHORT': 'SHORT', 'WAIT': 'WAIT', 'LONG': 'LONG'}
        direction = CLASS_MAP.get(direction_raw, str(direction_raw))
        
        confidence = direction_proba[pred_idx] * 100
        
        # Get feature importances
        if hasattr(self.direction_model, 'feature_importances_'):
            importances = self.direction_model.feature_importances_
        else:
            importances = np.zeros(len(FEATURE_NAMES))
        
        top_features = self._get_top_features(feature_set, importances)
        
        # TP/SL prediction
        if self.tp_sl_model is not None:
            tp_sl_pred = self.tp_sl_model.predict(features)[0]
            # Handle single value (TP1 only) or array [tp1, tp2, sl]
            if isinstance(tp_sl_pred, (int, float, np.floating)):
                optimal_tp1_pct = float(tp_sl_pred)
                optimal_tp2_pct = optimal_tp1_pct * 1.6  # TP2 = 1.6x TP1
                optimal_sl_pct = optimal_tp1_pct * 0.6   # SL = 0.6x TP1
            elif len(tp_sl_pred) >= 3:
                optimal_tp1_pct = tp_sl_pred[0]
                optimal_tp2_pct = tp_sl_pred[1]
                optimal_sl_pct = tp_sl_pred[2]
            else:
                optimal_tp1_pct = float(tp_sl_pred[0]) if len(tp_sl_pred) > 0 else 2.5
                optimal_tp2_pct = optimal_tp1_pct * 1.6
                optimal_sl_pct = optimal_tp1_pct * 0.6
        else:
            # Default based on direction
            if direction == 'LONG':
                optimal_tp1_pct = 2.5
                optimal_tp2_pct = 4.0
                optimal_sl_pct = 1.5
            elif direction == 'SHORT':
                optimal_tp1_pct = 2.0
                optimal_tp2_pct = 3.5
                optimal_sl_pct = 1.5
            else:
                optimal_tp1_pct = 0
                optimal_tp2_pct = 0
                optimal_sl_pct = 0
        
        # Calculate expected R:R
        expected_rr = optimal_tp1_pct / optimal_sl_pct if optimal_sl_pct > 0 else 0
        
        # Generate reasoning
        reasoning = self._generate_reasoning(direction, confidence, top_features)
        
        return MLPrediction(
            direction=direction,
            confidence=confidence,
            predicted_move=optimal_tp1_pct if direction != 'WAIT' else 0,
            optimal_tp1_pct=optimal_tp1_pct,
            optimal_tp2_pct=optimal_tp2_pct,
            optimal_sl_pct=optimal_sl_pct,
            expected_rr=expected_rr,
            win_probability=confidence,
            top_features=top_features,
            reasoning=reasoning,
            similar_trades_count=int(feature_set.features[FEATURE_NAMES.index('similar_setup_count')]),
            similar_trades_win_rate=feature_set.features[FEATURE_NAMES.index('historical_win_rate')],
            similar_trades_avg_return=feature_set.features[FEATURE_NAMES.index('avg_historical_return')],
            model_version=self.model_metadata.get('version', 'unknown'),
            prediction_time=datetime.now().isoformat(),
        )
    
    def _predict_heuristic(self, feature_set: FeatureSet) -> MLPrediction:
        """
        Heuristic-based prediction when model not trained.
        
        Uses rule-based logic similar to MASTER_RULES but with ML output format.
        """
        
        features = feature_set.features
        raw = feature_set.raw_data
        
        # Extract key features
        whale_pct = features[FEATURE_NAMES.index('whale_pct')]
        retail_pct = features[FEATURE_NAMES.index('retail_pct')]
        divergence = features[FEATURE_NAMES.index('whale_retail_divergence')]
        position_pct = features[FEATURE_NAMES.index('position_in_range')]
        oi_signal = features[FEATURE_NAMES.index('oi_signal_encoded')]
        ta_score = features[FEATURE_NAMES.index('ta_score')]
        money_flow = features[FEATURE_NAMES.index('money_flow_encoded')]
        historical_wr = features[FEATURE_NAMES.index('historical_win_rate')]
        at_bullish_ob = features[FEATURE_NAMES.index('at_bullish_ob')]
        at_bearish_ob = features[FEATURE_NAMES.index('at_bearish_ob')]
        
        # Calculate direction score
        direction_score = 0
        reasoning_parts = []
        
        # Whale positioning (strongest signal)
        if whale_pct >= 60:
            direction_score += 30
            reasoning_parts.append(f"Whales {whale_pct:.0f}% long (bullish)")
        elif whale_pct <= 40:
            direction_score -= 30
            reasoning_parts.append(f"Whales {whale_pct:.0f}% long (bearish)")
        
        # Divergence
        if divergence >= 15:
            direction_score += 20
            reasoning_parts.append(f"Whale/Retail divergence +{divergence:.0f}% (smart money bullish)")
        elif divergence <= -15:
            direction_score -= 20
            reasoning_parts.append(f"Whale/Retail divergence {divergence:.0f}% (retail trap risk)")
        
        # Position in range
        if position_pct <= 30:
            direction_score += 15
            reasoning_parts.append(f"EARLY position ({position_pct:.0f}%) - good entry for longs")
        elif position_pct >= 70:
            direction_score -= 15
            reasoning_parts.append(f"LATE position ({position_pct:.0f}%) - risky for longs")
        
        # SMC Override
        if position_pct >= 70 and at_bullish_ob:
            direction_score += 10
            reasoning_parts.append("At bullish OB - overrides late position")
        
        # OI Signal
        direction_score += oi_signal * 10
        if oi_signal >= 1:
            reasoning_parts.append("OI confirms bullish flow")
        elif oi_signal <= -1:
            reasoning_parts.append("OI confirms bearish flow")
        
        # Money flow
        direction_score += money_flow * 8
        
        # Historical win rate
        if historical_wr >= 70:
            direction_score += 10
            reasoning_parts.append(f"Historical win rate {historical_wr:.0f}% (proven setup)")
        elif historical_wr <= 40:
            direction_score -= 10
            reasoning_parts.append(f"Historical win rate {historical_wr:.0f}% (avoid)")
        
        # Determine direction
        if direction_score >= 30:
            direction = 'LONG'
            confidence = min(90, 50 + direction_score)
        elif direction_score <= -30:
            direction = 'SHORT'
            confidence = min(90, 50 + abs(direction_score))
        else:
            direction = 'WAIT'
            confidence = 50 - abs(direction_score)
        
        # Default TP/SL
        if direction == 'LONG':
            optimal_tp1_pct = 2.5
            optimal_tp2_pct = 4.0
            optimal_sl_pct = 1.5
        elif direction == 'SHORT':
            optimal_tp1_pct = 2.0
            optimal_tp2_pct = 3.5
            optimal_sl_pct = 1.5
        else:
            optimal_tp1_pct = 0
            optimal_tp2_pct = 0
            optimal_sl_pct = 0
        
        # Generate pseudo feature importances based on heuristic weights
        importances = np.zeros(len(FEATURE_NAMES))
        importances[FEATURE_NAMES.index('whale_pct')] = 0.25
        importances[FEATURE_NAMES.index('whale_retail_divergence')] = 0.20
        importances[FEATURE_NAMES.index('position_in_range')] = 0.15
        importances[FEATURE_NAMES.index('oi_signal_encoded')] = 0.15
        importances[FEATURE_NAMES.index('historical_win_rate')] = 0.10
        importances[FEATURE_NAMES.index('money_flow_encoded')] = 0.08
        importances[FEATURE_NAMES.index('ta_score')] = 0.07
        
        top_features = self._get_top_features(feature_set, importances)
        reasoning = " | ".join(reasoning_parts[:3]) if reasoning_parts else "Insufficient signal"
        
        return MLPrediction(
            direction=direction,
            confidence=confidence,
            predicted_move=optimal_tp1_pct if direction != 'WAIT' else 0,
            optimal_tp1_pct=optimal_tp1_pct,
            optimal_tp2_pct=optimal_tp2_pct,
            optimal_sl_pct=optimal_sl_pct,
            expected_rr=optimal_tp1_pct / optimal_sl_pct if optimal_sl_pct > 0 else 0,
            win_probability=confidence,
            top_features=top_features,
            reasoning=reasoning,
            similar_trades_count=int(features[FEATURE_NAMES.index('similar_setup_count')]),
            similar_trades_win_rate=historical_wr,
            similar_trades_avg_return=features[FEATURE_NAMES.index('avg_historical_return')],
            model_version='heuristic_v1',
            prediction_time=datetime.now().isoformat(),
        )
    
    def _get_top_features(
        self, 
        feature_set: FeatureSet, 
        importances: np.ndarray,
        top_n: int = 5
    ) -> List[Tuple[str, float, float]]:
        """Get top N most important features"""
        combined = list(zip(FEATURE_NAMES, feature_set.features, importances))
        combined.sort(key=lambda x: abs(x[2]), reverse=True)
        return [(name, float(val), float(imp)) for name, val, imp in combined[:top_n]]
    
    def _generate_reasoning(
        self,
        direction: str,
        confidence: float,
        top_features: List[Tuple[str, float, float]]
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []
        
        for name, value, importance in top_features[:3]:
            if name == 'whale_pct':
                parts.append(f"Whales {value:.0f}% long")
            elif name == 'whale_retail_divergence':
                parts.append(f"W/R divergence {value:+.0f}%")
            elif name == 'position_in_range':
                pos = 'EARLY' if value < 35 else 'LATE' if value > 65 else 'MID'
                parts.append(f"{pos} position ({value:.0f}%)")
            elif name == 'historical_win_rate':
                parts.append(f"Historical WR {value:.0f}%")
            elif name == 'oi_signal_encoded':
                signal = {-2: 'Strong bearish', -1: 'Bearish', 0: 'Neutral', 1: 'Bullish', 2: 'Strong bullish'}
                parts.append(f"OI: {signal.get(int(value), 'Neutral')}")
        
        return " | ".join(parts) if parts else f"{direction} with {confidence:.0f}% confidence"


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

# Global engine instance
_engine = None

def get_ml_prediction(**kwargs) -> MLPrediction:
    """
    Convenience function to get ML prediction.
    
    Creates engine instance on first call.
    """
    global _engine
    if _engine is None:
        _engine = MLEngine()
    return _engine.predict(**kwargs)


def is_ml_available() -> bool:
    """Check if ML models are trained and available"""
    global _engine
    if _engine is None:
        _engine = MLEngine()
    return _engine.is_trained


def is_model_loaded() -> bool:
    """Alias for is_ml_available for backward compatibility"""
    return is_ml_available()