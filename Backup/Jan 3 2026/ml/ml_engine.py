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
    
    Supports MODE-SPECIFIC models:
    - model_scalp.pkl for Scalp mode (1m, 5m)
    - model_daytrade.pkl for DayTrade mode (15m, 1h)
    - model_swing.pkl for Swing mode (4h, 1d)
    - model_investment.pkl for Investment mode (1d, 1w)
    """
    
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    
    # Timeframe to mode mapping
    TIMEFRAME_TO_MODE = {
        '1m': 'scalp', '5m': 'scalp',
        '15m': 'daytrade', '1h': 'daytrade',
        '4h': 'swing', '1d': 'swing',
        '1w': 'investment'
    }
    
    def __init__(self):
        self.direction_model = None
        
        # Mode-specific models
        self.mode_models = {}  # {mode: {'direction': model, 'eta': model, 'sl': model, 'scaler': scaler}}
        self.tp_sl_model = None
        self.feature_scaler = None
        self.model_metadata = {}
        self.training_metadata = {}  # NEW: Info about how model was trained
        self.is_trained = False
        self._is_real_ml = False  # NEW: True only if trained on actual data
        self._model_type = 'heuristic'  # NEW: 'heuristic', 'trained_ml', 'historical_ml'
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            direction_path = os.path.join(self.MODEL_DIR, 'direction_model.pkl')
            tp_sl_path = os.path.join(self.MODEL_DIR, 'tp_sl_model.pkl')
            scaler_path = os.path.join(self.MODEL_DIR, 'scaler.pkl')  # Updated path
            metadata_path = os.path.join(self.MODEL_DIR, 'model_metadata.json')
            training_meta_path = os.path.join(self.MODEL_DIR, 'training_metadata.json')
            
            # Try alternate scaler path
            if not os.path.exists(scaler_path):
                scaler_path = os.path.join(self.MODEL_DIR, 'feature_scaler.pkl')
            
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
            
            # NEW: Load training metadata to check if it's real ML
            if os.path.exists(training_meta_path):
                with open(training_meta_path, 'r') as f:
                    self.training_metadata = json.load(f)
                    
                # Check if this is actually trained ML
                if self.training_metadata.get('is_real_ml', False):
                    self._is_real_ml = True
                    self._model_type = 'historical_ml'
                    
            # Also check n_samples in model_metadata as backup
            if self.model_metadata.get('n_samples', 0) > 100:
                # Has substantial training data
                if not self._is_real_ml:
                    self._is_real_ml = True
                    self._model_type = 'trained_ml'
            
            # VALIDATE: Check if model was trained with same number of features
            if self.direction_model is not None:
                model_features = self.model_metadata.get('feature_names', [])
                model_n_features = len(model_features) if model_features else 0
                current_n_features = len(FEATURE_NAMES)
                
                if model_n_features > 0 and model_n_features != current_n_features:
                    print(f"⚠️ Feature mismatch: Model={model_n_features}, Code={current_n_features}")
                    print(f"   Using heuristic predictions (model needs retraining)")
                    self.is_trained = False
                    self._is_real_ml = False
                    self._model_type = 'heuristic'
                    self.direction_model = None
                else:
                    self.is_trained = True
                    if self._is_real_ml:
                        model_name = self.model_metadata.get('model_name', 'Unknown')
                        samples = self.training_metadata.get('total_samples', 0)
                        print(f"✅ Trained ML loaded: {model_name} ({samples} samples)")
                    else:
                        print(f"⚠️ Model loaded but no training metadata - treating as heuristic")
                        self._model_type = 'heuristic'
            else:
                # No model files - use heuristic silently
                self.is_trained = False
                self._is_real_ml = False
                self._model_type = 'heuristic'
                
        except Exception as e:
            print(f"⚠️ Error loading ML models: {e}")
            self.is_trained = False
            self._is_real_ml = False
            self._model_type = 'heuristic'
        
        # Also load mode-specific models
        self._load_mode_models()
    
    def _load_mode_models(self):
        """Load mode-specific trained models (scalp, daytrade, swing, investment)."""
        modes = ['scalp', 'daytrade', 'swing', 'investment']
        
        for mode in modes:
            try:
                dir_path = os.path.join(self.MODEL_DIR, f'direction_model_{mode}.pkl')
                eta_path = os.path.join(self.MODEL_DIR, f'eta_model_{mode}.pkl')
                sl_path = os.path.join(self.MODEL_DIR, f'sl_model_{mode}.pkl')
                scaler_path = os.path.join(self.MODEL_DIR, f'scaler_{mode}.pkl')
                meta_path = os.path.join(self.MODEL_DIR, f'metadata_{mode}.json')
                
                if os.path.exists(dir_path) and os.path.exists(scaler_path):
                    self.mode_models[mode] = {
                        'direction': pickle.load(open(dir_path, 'rb')),
                        'eta': pickle.load(open(eta_path, 'rb')) if os.path.exists(eta_path) else None,
                        'sl': pickle.load(open(sl_path, 'rb')) if os.path.exists(sl_path) else None,
                        'scaler': pickle.load(open(scaler_path, 'rb')),
                        'metadata': json.load(open(meta_path, 'r')) if os.path.exists(meta_path) else {}
                    }
                    
                    # If we have mode-specific models, we have real ML
                    self._is_real_ml = True
                    self._model_type = 'mode_specific_ml'
                    
            except Exception as e:
                # Silently continue - mode model not available
                pass
        
        if self.mode_models:
            modes_loaded = list(self.mode_models.keys())
            print(f"✅ Mode-specific ML loaded: {modes_loaded}")
    
    def has_mode_model(self, mode: str = None, timeframe: str = None) -> bool:
        """Check if a mode-specific model is available."""
        if timeframe and not mode:
            mode = self.TIMEFRAME_TO_MODE.get(timeframe)
        return mode in self.mode_models if mode else False
    
    def get_mode_from_timeframe(self, timeframe: str) -> str:
        """Get trading mode from timeframe."""
        return self.TIMEFRAME_TO_MODE.get(timeframe, 'daytrade')
    
    def is_model_loaded(self) -> bool:
        """Check if ML models are loaded and ready"""
        return self.is_trained
    
    def is_real_trained_ml(self) -> bool:
        """
        Check if this is REAL trained ML (not heuristic).
        Returns True only if model was trained on actual historical data.
        """
        return self._is_real_ml
    
    def get_model_type(self) -> str:
        """
        Get the type of prediction being used.
        Returns: 'heuristic', 'trained_ml', or 'historical_ml'
        """
        return self._model_type
    
    def get_model_label(self) -> str:
        """
        Get an HONEST label for what prediction method is being used.
        This should be shown to users!
        """
        if self._is_real_ml:
            model_name = self.model_metadata.get('model_name', 'ML')
            samples = self.training_metadata.get('total_samples', 0)
            return f"🤖 Trained ML ({model_name}, {samples} samples)"
        else:
            return "📊 Heuristic (rules-based scoring)"
    
    def get_prediction_source(self) -> str:
        """
        Short label for display in UI.
        """
        if self._is_real_ml:
            return "Trained ML"
        else:
            return "Heuristic"
    
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
        # NEW: Explosion & Energy parameters
        explosion_score: float = 0,
        explosion_ready: bool = False,
        bb_squeeze_pct: float = 50,
        bb_width_pct: float = 5,
        compression_duration: int = 0,
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
            # NEW: Explosion & Energy features
            explosion_score=explosion_score,
            explosion_ready=explosion_ready,
            bb_squeeze_pct=bb_squeeze_pct,
            bb_width_pct=bb_width_pct,
            compression_duration=compression_duration,
        )
        
        if self.is_trained:
            return self._predict_with_model(feature_set)
        else:
            return self._predict_heuristic(feature_set)
    
    def predict_for_mode(
        self,
        mode: str = None,
        timeframe: str = None,
        features_dict: Dict = None,
        **kwargs
    ) -> MLPrediction:
        """
        Predict using MODE-SPECIFIC model.
        
        Args:
            mode: Trading mode ('scalp', 'daytrade', 'swing', 'investment')
            timeframe: Timeframe (will determine mode if mode not specified)
            features_dict: Pre-computed features dictionary
            **kwargs: Feature values (same as predict())
        
        Returns:
            MLPrediction with direction, confidence, ETA, and optimal SL
        """
        # Determine mode from timeframe if not specified
        if not mode and timeframe:
            mode = self.get_mode_from_timeframe(timeframe)
        
        if not mode:
            mode = 'daytrade'  # Default
        
        # Check if we have a mode-specific model
        if mode in self.mode_models:
            return self._predict_with_mode_model(mode, features_dict, **kwargs)
        else:
            # Fall back to regular prediction
            return self.predict(**kwargs) if not features_dict else self.predict(**features_dict)
    
    def _predict_with_mode_model(self, mode: str, features_dict: Dict = None, **kwargs) -> MLPrediction:
        """Predict using a mode-specific trained model."""
        from .mode_specific_trainer import FEATURE_NAMES as MODE_FEATURE_NAMES, extract_features_from_row
        import pandas as pd
        
        mode_model = self.mode_models[mode]
        
        # Prepare features - either from dict or kwargs
        input_data = features_dict if features_dict else kwargs
        
        # Create a mock row for feature extraction
        row = pd.Series({
            'whale_pct': input_data.get('whale_pct', 50),
            'retail_pct': input_data.get('retail_pct', 50),
            'divergence': input_data.get('whale_pct', 50) - input_data.get('retail_pct', 50),
            'position_pct': input_data.get('position_pct', 50),
            'volume_ratio': input_data.get('volume_ratio', 1),
            'RSI': input_data.get('rsi', 50),
            'trend': 1 if input_data.get('trend', 'NEUTRAL') == 'BULLISH' else (-1 if input_data.get('trend') == 'BEARISH' else 0),
            'ATR_pct': input_data.get('atr', 0) / input_data.get('current_price', 1) * 100 if input_data.get('current_price', 0) > 0 else 2,
            'BB_Squeeze_Pct': input_data.get('bb_squeeze_pct', 50),
            'BB_Width': input_data.get('bb_width_pct', 3),
            'buy_ratio': 0.5 + (input_data.get('whale_pct', 50) - 50) / 100,
            'price_change_5': input_data.get('price_change_1h', 0),
            'price_change_20': input_data.get('price_change_24h', 0),
            'price_change_50': input_data.get('price_change_24h', 0) * 2,
            'near_support': 1 if input_data.get('near_support', False) else 0,
            'near_resistance': 1 if input_data.get('near_resistance', False) else 0,
            'Close': input_data.get('current_price', 100),
        })
        
        features = extract_features_from_row(row)
        features = features.reshape(1, -1)
        
        # Scale features
        scaler = mode_model['scaler']
        features_scaled = scaler.transform(features)
        
        # Direction prediction
        dir_model = mode_model['direction']
        direction_proba = dir_model.predict_proba(features_scaled)[0]
        classes = dir_model.classes_
        pred_idx = np.argmax(direction_proba)
        direction_raw = classes[pred_idx]
        
        CLASS_MAP = {0: 'SHORT', 1: 'WAIT', 2: 'LONG'}
        direction = CLASS_MAP.get(direction_raw, 'WAIT')
        confidence = direction_proba[pred_idx] * 100
        
        # ETA prediction (candles to TP)
        eta_candles = None
        if mode_model.get('eta') is not None:
            eta_candles = mode_model['eta'].predict(features_scaled)[0]
        
        # SL distance prediction
        optimal_sl = None
        if mode_model.get('sl') is not None:
            optimal_sl = mode_model['sl'].predict(features_scaled)[0]
        
        # Get metadata
        metadata = mode_model.get('metadata', {})
        metrics = metadata.get('metrics', {})
        config = metadata.get('config', {})
        
        # Default TP/SL based on mode config
        profit_thresh = config.get('profit_threshold', 1.5)
        loss_thresh = config.get('loss_threshold', 1.0)
        
        # Build reasoning
        reasoning_parts = [f"Mode-specific ML ({mode.upper()}) prediction"]
        if eta_candles:
            reasoning_parts.append(f"ETA to TP: ~{int(eta_candles)} candles")
        if optimal_sl:
            reasoning_parts.append(f"Optimal SL distance: {optimal_sl:.2f}%")
        
        # Feature importances
        if hasattr(dir_model, 'feature_importances_'):
            importances = dir_model.feature_importances_
            top_idx = np.argsort(importances)[-5:][::-1]
            top_features = [(MODE_FEATURE_NAMES[i], float(features[0][i]), float(importances[i])) 
                           for i in top_idx]
        else:
            top_features = []
        
        return MLPrediction(
            direction=direction,
            confidence=confidence,
            predicted_move=profit_thresh if direction != 'WAIT' else 0,
            optimal_tp1_pct=profit_thresh,
            optimal_tp2_pct=profit_thresh * 1.5,
            optimal_sl_pct=optimal_sl if optimal_sl else loss_thresh,
            expected_rr=profit_thresh / loss_thresh if loss_thresh > 0 else 1.5,
            win_probability=confidence / 100,
            top_features=top_features,
            reasoning=" | ".join(reasoning_parts),
            similar_trades_count=metrics.get('n_samples', 0),
            similar_trades_win_rate=metrics.get('direction_f1', 0) * 100,
            similar_trades_avg_return=profit_thresh,
            model_version=f"mode_specific_{mode}",
            prediction_time=datetime.now().isoformat(),
        )
    
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
        
        # NEW: Extract explosion & energy features
        explosion_score = features[FEATURE_NAMES.index('explosion_score')]
        explosion_ready = features[FEATURE_NAMES.index('explosion_ready')]
        bb_squeeze_pct = features[FEATURE_NAMES.index('bb_squeeze_pct')]
        energy_loaded = features[FEATURE_NAMES.index('energy_loaded')]
        
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
        
        # NEW: Explosion & Energy features boost score!
        if energy_loaded:
            # Energy is loaded - boost confidence in direction
            if direction_score > 0:
                direction_score += 15
                reasoning_parts.append(f"⚡ ENERGY LOADED - explosion imminent!")
            elif direction_score < 0:
                direction_score -= 15
                reasoning_parts.append(f"⚡ ENERGY LOADED - explosion imminent!")
        
        if explosion_score >= 60:
            direction_score += int(explosion_score / 10) if direction_score > 0 else -int(explosion_score / 10)
            reasoning_parts.append(f"Explosion score {explosion_score:.0f}/100")
        
        if bb_squeeze_pct >= 75:
            direction_score += 10 if direction_score > 0 else -10
            reasoning_parts.append(f"BB squeeze {bb_squeeze_pct:.0f}% - tight compression")
        
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
        importances[FEATURE_NAMES.index('whale_pct')] = 0.20
        importances[FEATURE_NAMES.index('whale_retail_divergence')] = 0.18
        importances[FEATURE_NAMES.index('position_in_range')] = 0.12
        importances[FEATURE_NAMES.index('oi_signal_encoded')] = 0.12
        importances[FEATURE_NAMES.index('historical_win_rate')] = 0.08
        importances[FEATURE_NAMES.index('money_flow_encoded')] = 0.06
        importances[FEATURE_NAMES.index('ta_score')] = 0.05
        # NEW: Explosion features
        importances[FEATURE_NAMES.index('explosion_score')] = 0.08
        importances[FEATURE_NAMES.index('energy_loaded')] = 0.06
        importances[FEATURE_NAMES.index('bb_squeeze_pct')] = 0.05
        
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
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Global engine instance
_engine = None

def _get_engine() -> MLEngine:
    """Get or create the global ML engine instance."""
    global _engine
    if _engine is None:
        _engine = MLEngine()
    return _engine


def get_ml_prediction(**kwargs) -> MLPrediction:
    """
    Convenience function to get ML prediction.
    
    Creates engine instance on first call.
    """
    return _get_engine().predict(**kwargs)


def is_ml_available() -> bool:
    """
    Check if ML prediction is available.
    
    Always returns True because heuristic fallback is always available.
    The system uses trained model if available, otherwise smart heuristic.
    """
    return True  # Heuristic is ALWAYS available


def is_model_trained() -> bool:
    """Check if trained ML model is loaded (vs using heuristic)"""
    return _get_engine().is_trained


def is_model_loaded() -> bool:
    """Alias for is_ml_available for backward compatibility"""
    return is_ml_available()


def is_real_trained_ml() -> bool:
    """
    Check if REAL trained ML is being used.
    
    Returns True ONLY if:
    - Model was trained on actual historical data
    - Training metadata confirms it's real ML
    
    Returns False if using heuristic (rules-based scoring).
    """
    return _get_engine().is_real_trained_ml()


def get_prediction_source() -> str:
    """
    Get the source of predictions: 'Trained ML' or 'Heuristic'.
    Use this to display honest labels to users!
    """
    return _get_engine().get_prediction_source()


def get_model_label() -> str:
    """
    Get full descriptive label for the prediction method.
    Example: '🤖 Trained ML (LightGBM, 5000 samples)' or '📊 Heuristic (rules-based)'
    """
    return _get_engine().get_model_label()


def get_model_info() -> Dict:
    """
    Get complete info about the current prediction method.
    """
    engine = _get_engine()
    return {
        'is_trained': engine.is_trained,
        'is_real_ml': engine._is_real_ml,
        'model_type': engine._model_type,
        'model_name': engine.model_metadata.get('model_name', 'Heuristic'),
        'accuracy': engine.model_metadata.get('accuracy', None),
        'f1_score': engine.model_metadata.get('f1_score', None),
        'training_samples': engine.training_metadata.get('total_samples', 0),
        'trained_on': engine.training_metadata.get('trained_at', None),
        'training_symbols': engine.training_metadata.get('symbols', []),
        'label': engine.get_model_label(),
        'source': engine.get_prediction_source(),
    }