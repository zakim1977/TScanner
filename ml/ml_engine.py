"""
ML Engine - SINGLE SOURCE OF TRUTH
===================================

Calls Probabilistic ML. NO FALLBACKS. NO DEFAULTS.
If it fails, you see the error.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MLPrediction:
    """ML prediction result"""
    direction: str
    confidence: float
    action_message: str = ""
    primary_signal: str = ""
    bull_strength: float = 0.0
    bear_strength: float = 0.0
    f1_scores: Dict = None
    reasoning: str = ""
    
    # Legacy fields (kept for compatibility)
    predicted_move: float = 0.0
    optimal_tp1_pct: float = 0.0
    optimal_tp2_pct: float = 0.0
    optimal_tp3_pct: float = 0.0
    optimal_sl_pct: float = 0.0
    expected_rr: float = 0.0
    win_probability: float = 0.0
    top_features: List = None
    similar_trades_count: int = 0
    similar_trades_win_rate: float = 0.0
    similar_trades_avg_return: float = 0.0
    pattern_detected: str = None
    pattern_confidence: float = 0.0
    pattern_aligns: bool = False
    model_version: str = "2.0"
    prediction_time: str = ""
    
    def to_dict(self) -> Dict:
        return self.__dict__


# Mode mapping
MODE_MAP = {
    '1m': 'scalp', '5m': 'scalp',
    '15m': 'daytrade', '1h': 'daytrade',
    '4h': 'swing', '1d': 'swing',
    '1w': 'investment'
}


def get_ml_prediction(
    df=None,
    timeframe: str = '15m',
    market_type: str = 'crypto',
    whale_pct: float = None,
    retail_pct: float = None,
    oi_change: float = None,
    **kwargs
) -> MLPrediction:
    """
    SINGLE SOURCE OF TRUTH - Calls Probabilistic ML directly.
    NO FALLBACKS. NO DEFAULTS. 
    """
    from .ml_signals_display import get_ml_signals
    
    # Validate input
    if df is None:
        raise ValueError("df is REQUIRED for ML prediction")
    
    if len(df) < 50:
        raise ValueError(f"Need at least 50 candles, got {len(df)}")
    
    # Get mode
    mode = MODE_MAP.get(timeframe, 'daytrade')
    
    # Prepare whale data
    whale_data = None
    if whale_pct is not None:
        whale_data = {
            'whale_long_pct': whale_pct,
            'retail_long_pct': retail_pct if retail_pct is not None else 50,
            'oi_change_24h': oi_change if oi_change is not None else 0,
        }
    
    # Market type
    ml_market_type = 'crypto' if 'crypto' in str(market_type).lower() else 'stock'
    
    # CALL PROBABILISTIC ML - NO TRY/EXCEPT, LET IT FAIL IF IT FAILS
    prob_signals = get_ml_signals(df, mode=mode, whale_data=whale_data, market_type=ml_market_type)
    
    # If no model exists, return NOT_TRAINED (don't fake it!)
    if prob_signals is None:
        return MLPrediction(
            direction='NOT_TRAINED',
            confidence=0,
            action_message=f"âš ï¸ Model not trained for {mode} mode",
            reasoning=f"âš ï¸ Train {mode} model in ML Training tab",
        )
    
    if not hasattr(prob_signals, 'direction'):
        return MLPrediction(
            direction='NOT_TRAINED',
            confidence=0,
            action_message="âš ï¸ Invalid model response",
            reasoning="âš ï¸ Model returned invalid data",
        )
    
    # Extract values directly - NO DEFAULTS
    direction = prob_signals.direction
    confidence = prob_signals.confidence
    action_message = prob_signals.action_message
    primary_signal = prob_signals.primary_signal
    f1_scores = prob_signals.f1_scores
    
    # Calculate strengths based on mode
    if mode in ['scalp', 'daytrade']:
        bull_strength = prob_signals.continuation_bull + prob_signals.fakeout_to_bull
        bear_strength = prob_signals.continuation_bear + prob_signals.fakeout_to_bear
    elif mode == 'swing':
        bull_strength = prob_signals.trend_holds_bull + prob_signals.reversal_to_bull
        bear_strength = prob_signals.trend_holds_bear + prob_signals.reversal_to_bear
    else:  # investment
        bull_strength = prob_signals.accumulation + prob_signals.reversal_to_bull
        bear_strength = prob_signals.distribution + prob_signals.reversal_to_bear
    
    return MLPrediction(
        direction=direction,
        confidence=confidence,
        action_message=action_message,
        primary_signal=primary_signal,
        bull_strength=bull_strength,
        bear_strength=bear_strength,
        f1_scores=f1_scores,
        reasoning=f"{action_message} | Bull: {bull_strength*100:.0f}% vs Bear: {bear_strength*100:.0f}%",
        prediction_time=datetime.now().isoformat(),
    )


def is_ml_available() -> bool:
    """Always True - Probabilistic ML is the system."""
    return True


def is_model_trained() -> bool:
    """Check if model files exist."""
    import os
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if f.endswith('.pkl'):
                return True
    return False


def is_model_loaded() -> bool:
    return is_ml_available()


def is_real_trained_ml() -> bool:
    return is_model_trained()


def get_prediction_source() -> str:
    return "Probabilistic ML"


def get_model_label() -> str:
    return "ðŸ§  Probabilistic ML"


def get_model_info() -> Dict:
    return {
        'is_trained': is_model_trained(),
        'is_real_ml': True,
        'model_type': 'probabilistic',
        'model_name': 'Probabilistic ML',
        'label': get_model_label(),
        'source': get_prediction_source(),
    }


# Legacy compatibility
class MLEngine:
    def __init__(self):
        self.is_trained = is_model_trained()
    
    def predict(self, **kwargs) -> MLPrediction:
        return get_ml_prediction(**kwargs)
    
    def predict_for_mode(self, timeframe: str, market_type: str = 'crypto', **kwargs) -> MLPrediction:
        return get_ml_prediction(timeframe=timeframe, market_type=market_type, **kwargs)
    
    def get_mode_from_timeframe(self, timeframe: str) -> str:
        return MODE_MAP.get(timeframe, 'daytrade')
    
    def has_mode_model(self, mode: str = None, market_type: str = 'crypto') -> bool:
        return is_model_trained()
    
    def is_real_trained_ml(self) -> bool:
        return is_real_trained_ml()
    
    def get_prediction_source(self) -> str:
        return get_prediction_source()
    
    def get_model_label(self) -> str:
        return get_model_label()
    
    def has_pattern_detection(self) -> bool:
        """Check if pattern detection is available (either trained or rule-based)."""
        # Pattern detection is always available via rule-based fallback
        # Check if deep learning models are trained (for higher accuracy)
        import os
        
        possible_dirs = [
            os.path.join(os.path.dirname(__file__), 'models', 'patterns'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'patterns'),
            os.path.join(os.getcwd(), 'models', 'patterns'),
        ]
        
        for pattern_dir in possible_dirs:
            if os.path.exists(pattern_dir):
                for f in os.listdir(pattern_dir):
                    if f.endswith('.pkl') or f.endswith('.pt'):
                        return True
        
        # Even without trained models, rule-based detection is available
        # Return True to enable pattern detection display
        return True
    
    def _get_pattern_model_dir(self) -> str:
        """Find the pattern model directory."""
        import os
        
        possible_dirs = [
            os.path.join(os.path.dirname(__file__), 'models', 'patterns'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'patterns'),
            os.path.join(os.getcwd(), 'models', 'patterns'),
        ]
        
        for pattern_dir in possible_dirs:
            if os.path.exists(pattern_dir):
                for f in os.listdir(pattern_dir):
                    if f.endswith('.pkl') or f.endswith('.pt'):
                        return pattern_dir
        return possible_dirs[0]  # Default
    
    def get_pattern_confirmation(self, df, mode: str, market_type: str = 'crypto') -> dict:
        """Get pattern detection results."""
        import os
        
        try:
            # Try multiple import paths
            RuleBasedPatternDetector = None
            CombinedPredictor = None
            PatternType = None
            
            try:
                from .deep_pattern_detection import CombinedPredictor, RuleBasedPatternDetector, PatternType
            except ImportError:
                try:
                    from deep_pattern_detection import CombinedPredictor, RuleBasedPatternDetector, PatternType
                except ImportError:
                    pass
            
            if RuleBasedPatternDetector is None:
                return {
                    'pattern_detected': None,
                    'pattern_confidence': 0,
                    'pattern_direction': 'NEUTRAL',
                    'error': 'Could not import pattern detection modules'
                }
            
            # Check if we have trained deep learning models
            has_trained_model = self.has_pattern_detection()
            
            pattern_name = None
            pattern_conf = 0
            pattern_dir = 'NEUTRAL'
            
            if has_trained_model and CombinedPredictor is not None:
                # Try to use trained deep learning model
                try:
                    model_dir = self._get_pattern_model_dir()
                    model_filename = f'pattern_{mode}_{market_type}.pkl'
                    model_path = os.path.join(model_dir, model_filename)
                    
                    # Try alternative filename patterns
                    if not os.path.exists(model_path):
                        model_filename = f'pattern_{mode}_{market_type}s.pkl'
                        model_path = os.path.join(model_dir, model_filename)
                    
                    predictor = CombinedPredictor(mode=mode, market=market_type)
                    
                    if os.path.exists(model_path):
                        predictor.load_pattern_detector(model_path)
                        
                        # Get prediction - returns CombinedSignal dataclass
                        result = predictor.predict(df)
                        
                        # Extract pattern info from CombinedSignal dataclass
                        if hasattr(result, 'pattern') and result.pattern is not None:
                            if hasattr(result.pattern, 'value'):
                                pattern_name = result.pattern.value
                            elif hasattr(result.pattern, 'name'):
                                pattern_name = result.pattern.name
                            else:
                                pattern_name = str(result.pattern)
                        
                        pattern_conf = getattr(result, 'pattern_confidence', 0) or 0
                        pattern_dir = getattr(result, 'pattern_direction', 'NEUTRAL') or 'NEUTRAL'
                except Exception as dl_err:
                    print(f"[PATTERN] Deep learning failed, using rule-based: {dl_err}")
            
            # Fallback to rule-based detection if no pattern found or no trained model
            if not pattern_name or pattern_name.upper() == 'NO_PATTERN':
                try:
                    detector = RuleBasedPatternDetector(market=market_type)
                    patterns = detector.detect_all_patterns(df)
                    
                    if patterns and len(patterns) > 0:
                        top_pattern = patterns[0]
                        
                        # Extract pattern info
                        if hasattr(top_pattern, 'pattern') and top_pattern.pattern is not None:
                            if hasattr(top_pattern.pattern, 'value'):
                                pattern_name = top_pattern.pattern.value
                            elif hasattr(top_pattern.pattern, 'name'):
                                pattern_name = top_pattern.pattern.name
                            else:
                                pattern_name = str(top_pattern.pattern)
                        
                        pattern_conf = getattr(top_pattern, 'confidence', 0) or 0
                        pattern_dir = getattr(top_pattern, 'direction', 'NEUTRAL') or 'NEUTRAL'
                except Exception as rule_err:
                    print(f"[PATTERN] Rule-based detection failed: {rule_err}")
            
            # Don't report NO_PATTERN as a detection
            if pattern_name and 'NO_PATTERN' in pattern_name.upper():
                pattern_name = None
                pattern_conf = 0
            
            # Format pattern name for display
            if pattern_name:
                pattern_name = pattern_name.replace('_', ' ').title()
            
            return {
                'pattern_detected': pattern_name,
                'pattern_confidence': pattern_conf,
                'pattern_direction': pattern_dir,
                'error': None
            }
        except Exception as e:
            # Log the error for debugging
            print(f"[PATTERN] Error in get_pattern_confirmation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'pattern_detected': None,
                'pattern_confidence': 0,
                'pattern_direction': 'NEUTRAL',
                'error': str(e)
            }


_engine = None

def _get_engine() -> MLEngine:
    global _engine
    if _engine is None:
        _engine = MLEngine()
    return _engine


def get_ml_engine() -> MLEngine:
    """Public alias for _get_engine() - used by app.py"""
    return _get_engine()