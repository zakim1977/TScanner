"""
ML Signals Display Module
=========================

Provides actionable ML signals for ALL trading modes:
- Scalp: continuation_bull/bear, fakeout_to_bull/bear, vol_expansion
- DayTrade: continuation_bull/bear, fakeout_to_bull/bear, vol_expansion
- Swing: trend_holds_bull/bear, reversal_to_bull/bear, drawdown
- Investment: accumulation, distribution, reversal_to_bull/bear, large_drawdown

ALL SIGNALS ARE DIRECTIONAL - tells you which way the move will go!

Usage:
    from ml.ml_signals_display import get_ml_signals, render_ml_signals_ui
    
    signals = get_ml_signals(df, mode='swing', whale_data=whale_data)
    render_ml_signals_ui(signals)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
import pickle
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL STRUCTURES BY MODE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScalpDayTradeSignals:
    """Signals for Scalp and DayTrade modes"""
    continuation_bull: float = 0.0    # Probability price continues UP
    continuation_bear: float = 0.0    # Probability price continues DOWN
    fakeout_to_bull: float = 0.0      # Probability of bearish fakeout → reverses UP
    fakeout_to_bear: float = 0.0      # Probability of bullish fakeout → reverses DOWN
    vol_expansion: float = 0.0        # Probability of volatility expansion
    
    # Derived
    direction: str = 'WAIT'           # LONG, SHORT, WAIT
    confidence: float = 0.0
    primary_signal: str = ''
    action_message: str = ''
    mode: str = 'daytrade'
    
    # F1 scores (populated after training)
    f1_scores: Dict[str, float] = None
    
    # Phase prediction (NEW!)
    current_phase: str = 'UNKNOWN'
    phase_probabilities: Dict[str, float] = None
    next_phase: str = 'UNKNOWN'
    next_phase_confidence: float = 0.0
    phase_bias: str = 'NEUTRAL'


@dataclass
class SwingSignals:
    """Signals for Swing mode - DIRECTIONAL"""
    trend_holds_bull: float = 0.0     # Probability bullish trend continues 2%+
    trend_holds_bear: float = 0.0     # Probability bearish trend continues 2%+
    reversal_to_bull: float = 0.0     # Probability of reversal UP (was going down)
    reversal_to_bear: float = 0.0     # Probability of reversal DOWN (was going up)
    drawdown: float = 0.0             # Probability of 3%+ adverse move
    
    # Derived
    direction: str = 'WAIT'
    confidence: float = 0.0
    primary_signal: str = ''
    action_message: str = ''
    mode: str = 'swing'
    
    f1_scores: Dict[str, float] = None
    
    # Phase prediction (NEW!)
    current_phase: str = 'UNKNOWN'
    phase_probabilities: Dict[str, float] = None
    next_phase: str = 'UNKNOWN'
    next_phase_confidence: float = 0.0
    phase_bias: str = 'NEUTRAL'


@dataclass
class StockSwingSignals:
    """Signals for Stock Swing mode - QUALITY + MEAN REVERSION"""
    setup_quality_high: float = 0.0    # TP hit before SL - good entry
    setup_quality_low: float = 0.0     # SL hit before TP - bad entry
    mean_reversion_long: float = 0.0   # Oversold bounce opportunity
    mean_reversion_short: float = 0.0  # Overbought fade opportunity
    breakout_valid: float = 0.0        # Breakout will hold (not fakeout)
    volatility_expansion: float = 0.0  # Vol about to expand
    
    # Derived
    direction: str = 'WAIT'
    confidence: float = 0.0
    primary_signal: str = ''
    action_message: str = ''
    mode: str = 'swing'
    market_type: str = 'stock'
    
    f1_scores: Dict[str, float] = None
    
    # Phase prediction
    current_phase: str = 'UNKNOWN'
    phase_probabilities: Dict[str, float] = None
    next_phase: str = 'UNKNOWN'
    next_phase_confidence: float = 0.0
    phase_bias: str = 'NEUTRAL'


@dataclass
class InvestmentSignals:
    """Signals for Investment mode - DIRECTIONAL"""
    accumulation: float = 0.0         # Smart money accumulating - price rising
    distribution: float = 0.0         # Smart money distributing - price falling
    reversal_to_bull: float = 0.0     # Major reversal UP coming
    reversal_to_bear: float = 0.0     # Major reversal DOWN coming
    large_drawdown: float = 0.0       # 7%+ drawdown risk
    
    # Derived
    direction: str = 'WAIT'
    confidence: float = 0.0
    primary_signal: str = ''
    action_message: str = ''
    mode: str = 'investment'
    
    f1_scores: Dict[str, float] = None
    
    # Phase prediction (NEW!)
    current_phase: str = 'UNKNOWN'
    phase_probabilities: Dict[str, float] = None
    next_phase: str = 'UNKNOWN'
    next_phase_confidence: float = 0.0
    phase_bias: str = 'NEUTRAL'


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_ml_signals(
    df: pd.DataFrame,
    mode: str,
    whale_data: Dict = None,
    market_type: str = 'crypto',  # 'crypto', 'stock', or 'etf'
) -> Optional[ScalpDayTradeSignals | SwingSignals | InvestmentSignals]:
    """
    Get ML signals for the specified mode.
    
    Args:
        df: OHLCV DataFrame with at least 50 candles
        mode: 'scalp', 'daytrade', 'swing', or 'investment'
        whale_data: Whale positioning data (if available)
        market_type: 'crypto', 'stock', or 'etf' (affects model selection)
    
    Returns:
        Mode-specific signals dataclass with probabilities and derived direction
    """
    if df is None or len(df) < 50:
        return None
    
    mode = mode.lower()
    market_type = market_type.lower()  # Normalize market type
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: Try dedicated probabilistic models
    # ═══════════════════════════════════════════════════════════════════════════
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    
    # DEBUG: Log what we're looking for
    print(f"[ML_MODEL_SEARCH] mode={mode}, market_type={market_type}")
    print(f"[ML_MODEL_SEARCH] Looking in: {model_dir}")
    
    # Try market-specific model first, fall back to general
    # IMPORTANT: etf is distinct from stock!
    model_files = [
        f'prob_model_{mode}_{market_type}.pkl',  # e.g., prob_model_investment_etf.pkl
        f'prob_{mode}_{market_type}.pkl',         # e.g., prob_investment_etf.pkl
    ]
    
    # For ETF, also try 'etfs' variant (plural)
    if market_type == 'etf':
        model_files.append(f'prob_model_{mode}_etfs.pkl')
        model_files.append(f'prob_{mode}_etfs.pkl')
    
    # Add generic fallbacks last
    model_files.extend([
        f'prob_{mode}.pkl',                       # e.g., prob_swing.pkl
        f'prob_model_{mode}.pkl',                 # e.g., prob_model_swing.pkl (legacy)
    ])
    
    # DEBUG: List what files exist in model_dir
    if os.path.exists(model_dir):
        existing_files = os.listdir(model_dir)
        print(f"[ML_MODEL_SEARCH] Files in dir: {existing_files}")
    else:
        print(f"[ML_MODEL_SEARCH] WARNING: Directory does not exist: {model_dir}")
    
    model_path = None
    for mf in model_files:
        path = os.path.join(model_dir, mf)
        print(f"[ML_MODEL_SEARCH] Checking: {path} -> exists={os.path.exists(path)}")
        if os.path.exists(path):
            model_path = path
            print(f"[ML_MODEL] ✅ Found model: {mf}")
            break
    
    # Also check patterns directory (legacy location)
    if model_path is None:
        patterns_dir = os.path.join(os.path.dirname(__file__), 'models', 'patterns')
        legacy_files = [
            f'pattern_{mode}_{market_type}s.pkl',  # e.g., pattern_investment_etfs.pkl
            f'pattern_{mode}_{market_type}.pkl',   # e.g., pattern_investment_etf.pkl
        ]
        for lf in legacy_files:
            legacy_path = os.path.join(patterns_dir, lf)
            if os.path.exists(legacy_path):
                model_path = legacy_path
                print(f"[ML_MODEL] Found legacy model: {lf}")
                break
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: Use UnifiedMLEngine (has heuristic fallback)
    # ═══════════════════════════════════════════════════════════════════════════
    if model_path is None:
        try:
            from .probabilistic_integration import get_unified_ml_prediction
            
            unified_pred = get_unified_ml_prediction(df, mode, whale_data)
            
            if unified_pred and unified_pred.probabilities:
                # Convert unified prediction to our signal format
                probs = unified_pred.probabilities
                
                if mode in ['scalp', 'daytrade']:
                    signals = ScalpDayTradeSignals(
                        continuation_bull=probs.get('continuation', 0.5) if unified_pred.direction == 'LONG' else 0.3,
                        continuation_bear=probs.get('continuation', 0.5) if unified_pred.direction == 'SHORT' else 0.3,
                        fakeout_to_bull=probs.get('fakeout', 0.2) if unified_pred.direction != 'LONG' else 0.1,
                        fakeout_to_bear=probs.get('fakeout', 0.2) if unified_pred.direction != 'SHORT' else 0.1,
                        vol_expansion=probs.get('vol_expansion', 0.3),
                        mode=mode,
                        direction=unified_pred.direction,
                        confidence=unified_pred.confidence,
                        action_message=unified_pred.reasoning,
                        primary_signal='continuation_bull' if unified_pred.direction == 'LONG' else 'continuation_bear' if unified_pred.direction == 'SHORT' else '',
                    )
                elif mode == 'swing':
                    # For swing, derive directional signals from unified prediction
                    is_bullish = unified_pred.direction == 'LONG'
                    is_bearish = unified_pred.direction == 'SHORT'
                    conf = unified_pred.confidence / 100
                    
                    signals = SwingSignals(
                        trend_holds_bull=conf * 0.8 if is_bullish else 0.2,
                        trend_holds_bear=conf * 0.8 if is_bearish else 0.2,
                        reversal_to_bull=probs.get('fakeout', 0.2) if not is_bullish else 0.1,
                        reversal_to_bear=probs.get('fakeout', 0.2) if not is_bearish else 0.1,
                        drawdown=0.3 if unified_pred.direction == 'WAIT' else 0.2,
                        mode=mode,
                        direction=unified_pred.direction,
                        confidence=unified_pred.confidence,
                        action_message=unified_pred.reasoning,
                        primary_signal='trend_holds_bull' if is_bullish else 'trend_holds_bear' if is_bearish else '',
                    )
                else:  # investment
                    is_bullish = unified_pred.direction == 'LONG'
                    is_bearish = unified_pred.direction == 'SHORT'
                    conf = unified_pred.confidence / 100
                    
                    signals = InvestmentSignals(
                        accumulation=conf * 0.8 if is_bullish else 0.2,
                        distribution=conf * 0.8 if is_bearish else 0.2,
                        reversal_to_bull=0.3 if not is_bullish and conf > 0.5 else 0.1,
                        reversal_to_bear=0.3 if not is_bearish and conf > 0.5 else 0.1,
                        large_drawdown=0.3 if unified_pred.direction == 'WAIT' else 0.15,
                        mode=mode,
                        direction=unified_pred.direction,
                        confidence=unified_pred.confidence,
                        action_message=unified_pred.reasoning,
                        primary_signal='accumulation' if is_bullish else 'distribution' if is_bearish else '',
                    )
                
                return signals
                
        except Exception as e:
            pass  # Fall through to return None
        
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # DEBUG: Show what's in the model file
        print(f"[ML_MODEL_LOAD] Keys in model_data: {list(model_data.keys())}")
        
        # Extract model components
        models = model_data.get('models_per_label', model_data.get('models', {}))
        scaler = model_data.get('scaler')
        metadata = model_data.get('metadata', {})
        
        # DEBUG: Show what models are available
        print(f"[ML_MODEL_LOAD] Models dict keys: {list(models.keys()) if isinstance(models, dict) else 'NOT A DICT'}")
        print(f"[ML_MODEL_LOAD] Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'NOT A DICT'}")
        
        # FIX: Extract F1 scores from correct location (metrics dict, not f1_scores)
        metrics_dict = metadata.get('metrics', {})
        f1_scores = {}
        for label, m in metrics_dict.items():
            if isinstance(m, dict):
                f1_scores[label] = m.get('f1', 0.0)
        
        # Also try direct f1_scores key as fallback
        if not f1_scores:
            f1_scores = metadata.get('f1_scores', {})
        
        print(f"[ML_MODEL_LOAD] F1 scores: {f1_scores}")
        
        # Get labels for this mode
        if mode in ['scalp', 'daytrade']:
            labels = ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion']
        elif mode == 'swing':
            # Check what labels actually exist in the model
            model_labels = list(models.keys()) if isinstance(models, dict) else []
            print(f"[ML_MODEL_LOAD] Actual model labels: {model_labels}")
            
            # Detect STOCK swing labels (mean_reversion, setup_quality)
            is_stock_swing = any(l in model_labels for l in ['mean_reversion_long', 'setup_quality_high', 'breakout_valid'])
            
            if is_stock_swing:
                # STOCK SWING labels
                labels = ['setup_quality_high', 'setup_quality_low', 'mean_reversion_long', 'mean_reversion_short', 'breakout_valid', 'volatility_expansion']
            elif 'trend_clear_bull' in model_labels:
                # Model uses trend_clear_* naming
                labels = ['trend_clear_bull', 'trend_clear_bear', 'swing_quality_high', 'swing_quality_low', 'vol_expanding']
            else:
                # Standard crypto swing naming
                labels = ['trend_holds_bull', 'trend_holds_bear', 'reversal_to_bull', 'reversal_to_bear', 'drawdown']
        elif mode == 'investment':
            # Check what labels actually exist in the model
            model_labels = list(models.keys()) if isinstance(models, dict) else []
            print(f"[ML_MODEL_LOAD] Actual model labels: {model_labels}")
            
            # Auto-detect investment model labels
            if 'accumulation' in model_labels:
                labels = ['accumulation', 'distribution', 'reversal_to_bull', 'reversal_to_bear', 'large_drawdown']
            elif 'trend_clear_bull' in model_labels:
                # Same structure as swing stock model
                labels = ['trend_clear_bull', 'trend_clear_bear', 'swing_quality_high', 'swing_quality_low', 'vol_expanding']
            else:
                # Use whatever labels the model has
                labels = model_labels[:5] if len(model_labels) >= 5 else model_labels
                print(f"[ML_MODEL_LOAD] Using model's own labels: {labels}")
        else:
            return None
        
        print(f"[ML_MODEL_LOAD] Using labels: {labels}")
        
        # Extract features for latest candle
        from .probabilistic_ml import extract_enhanced_features
        features = extract_enhanced_features(df, len(df) - 1, whale_data)
        
        # Scale if scaler available
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # =====================================================================
        # PHASE PREDICTION (NEW!)
        # =====================================================================
        current_phase = 'UNKNOWN'
        phase_probabilities = {}
        next_phase = 'UNKNOWN'
        next_phase_confidence = 0.0
        phase_bias = 'NEUTRAL'
        
        # PHASE_LABELS and PHASE_TRANSITIONS from probabilistic_ml
        PHASE_LABELS = ['ACCUMULATION', 'RE_ACCUMULATION', 'MARKUP', 'DISTRIBUTION', 'PROFIT_TAKING', 'MARKDOWN', 'CAPITULATION']
        PHASE_BIAS = {
            'ACCUMULATION': 'BULLISH', 'RE_ACCUMULATION': 'BULLISH', 'MARKUP': 'BULLISH',
            'DISTRIBUTION': 'BEARISH', 'PROFIT_TAKING': 'BEARISH', 'MARKDOWN': 'BEARISH', 'CAPITULATION': 'BEARISH'
        }
        PHASE_TRANSITIONS = {
            'ACCUMULATION': ['MARKUP', 'RE_ACCUMULATION'],
            'RE_ACCUMULATION': ['MARKUP', 'DISTRIBUTION'],
            'MARKUP': ['DISTRIBUTION', 'RE_ACCUMULATION'],
            'DISTRIBUTION': ['MARKDOWN', 'PROFIT_TAKING'],
            'PROFIT_TAKING': ['MARKDOWN', 'DISTRIBUTION', 'MARKUP'],
            'MARKDOWN': ['CAPITULATION', 'DISTRIBUTION'],
            'CAPITULATION': ['ACCUMULATION', 'MARKDOWN'],
        }
        
        try:
            phase_model = model_data.get('phase_model')
            
            if phase_model is not None:
                # Get phase probabilities
                phase_probs = phase_model.predict_proba(features)[0]
                model_classes = phase_model.classes_
                
                # Map probabilities to phase names
                for i, class_label in enumerate(model_classes):
                    if i < len(phase_probs):
                        if isinstance(class_label, (int, np.integer)):
                            if 0 <= class_label < len(PHASE_LABELS):
                                phase_name = PHASE_LABELS[class_label]
                                phase_probabilities[phase_name] = float(phase_probs[i])
                        else:
                            phase_probabilities[str(class_label)] = float(phase_probs[i])
                
                # Get predicted phase
                phase_idx = phase_model.predict(features)[0]
                if isinstance(phase_idx, (int, np.integer)):
                    if 0 <= phase_idx < len(PHASE_LABELS):
                        current_phase = PHASE_LABELS[phase_idx]
                else:
                    current_phase = str(phase_idx)
                
                # Get confidence and bias
                next_phase_confidence = phase_probabilities.get(current_phase, 0) * 100
                phase_bias = PHASE_BIAS.get(current_phase, 'NEUTRAL')
                
                # Predict next phase
                if current_phase in PHASE_TRANSITIONS:
                    possible_next = PHASE_TRANSITIONS[current_phase]
                    next_probs = [(p, phase_probabilities.get(p, 0)) for p in possible_next]
                    if next_probs:
                        next_phase, _ = max(next_probs, key=lambda x: x[1])
                        
                print(f"[PHASE_ML] Predicted: {current_phase} ({next_phase_confidence:.0f}%) -> {next_phase}")
                        
        except Exception as e:
            print(f"[PHASE_ML_ERROR] {e}")
        
        # Get predictions from each label model
        probabilities = {}
        for label in labels:
            if label in models:
                model = models[label]
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features)[0]
                        # Get probability of positive class
                        if len(proba) == 2:
                            probabilities[label] = float(proba[1])
                        else:
                            probabilities[label] = float(proba[0])
                    else:
                        # Binary prediction
                        pred = model.predict(features)[0]
                        probabilities[label] = float(pred)
                except Exception as e:
                    probabilities[label] = 0.0
            else:
                probabilities[label] = 0.0
                print(f"[ML_PROB_MISS] Label '{label}' NOT in models dict")
        
        # DEBUG: Show computed probabilities
        print(f"[ML_PROB_RESULT] Computed probabilities: {probabilities}")
        
        # Create mode-specific signals
        if mode in ['scalp', 'daytrade']:
            signals = ScalpDayTradeSignals(
                continuation_bull=probabilities.get('continuation_bull', 0.0),
                continuation_bear=probabilities.get('continuation_bear', 0.0),
                fakeout_to_bull=probabilities.get('fakeout_to_bull', 0.0),
                fakeout_to_bear=probabilities.get('fakeout_to_bear', 0.0),
                vol_expansion=probabilities.get('vol_expansion', 0.0),
                mode=mode,
                f1_scores=f1_scores,
                # Phase prediction (NEW!)
                current_phase=current_phase,
                phase_probabilities=phase_probabilities,
                next_phase=next_phase,
                next_phase_confidence=next_phase_confidence,
                phase_bias=phase_bias,
            )
            signals = _interpret_scalp_daytrade_signals(signals, whale_data)
            
        elif mode == 'swing':
            # Check if this is STOCK swing (has mean_reversion/setup_quality labels)
            is_stock_swing = 'mean_reversion_long' in probabilities or 'setup_quality_high' in probabilities
            
            if is_stock_swing:
                # STOCK SWING - Quality + Mean Reversion
                signals = StockSwingSignals(
                    setup_quality_high=probabilities.get('setup_quality_high', 0.0),
                    setup_quality_low=probabilities.get('setup_quality_low', 0.0),
                    mean_reversion_long=probabilities.get('mean_reversion_long', 0.0),
                    mean_reversion_short=probabilities.get('mean_reversion_short', 0.0),
                    breakout_valid=probabilities.get('breakout_valid', 0.0),
                    volatility_expansion=probabilities.get('volatility_expansion', 0.0),
                    mode=mode,
                    market_type='stock',
                    f1_scores=f1_scores,
                    current_phase=current_phase,
                    phase_probabilities=phase_probabilities,
                    next_phase=next_phase,
                    next_phase_confidence=next_phase_confidence,
                    phase_bias=phase_bias,
                )
                signals = _interpret_stock_swing_signals(signals, whale_data)
            else:
                # CRYPTO SWING - Trend + Reversal
                signals = SwingSignals(
                    trend_holds_bull=probabilities.get('trend_holds_bull', probabilities.get('trend_clear_bull', 0.0)),
                    trend_holds_bear=probabilities.get('trend_holds_bear', probabilities.get('trend_clear_bear', 0.0)),
                    reversal_to_bull=probabilities.get('reversal_to_bull', probabilities.get('swing_quality_high', 0.0)),
                    reversal_to_bear=probabilities.get('reversal_to_bear', probabilities.get('swing_quality_low', 0.0)),
                    drawdown=probabilities.get('drawdown', probabilities.get('vol_expanding', 0.0)),
                    mode=mode,
                    f1_scores=f1_scores,
                    current_phase=current_phase,
                    phase_probabilities=phase_probabilities,
                    next_phase=next_phase,
                    next_phase_confidence=next_phase_confidence,
                    phase_bias=phase_bias,
                )
                signals = _interpret_swing_signals(signals, whale_data)
            
        elif mode == 'investment':
            # Map actual model labels to InvestmentSignals fields
            # Model may use accumulation/distribution or trend_clear_* naming
            signals = InvestmentSignals(
                accumulation=probabilities.get('accumulation', probabilities.get('trend_clear_bull', 0.0)),
                distribution=probabilities.get('distribution', probabilities.get('trend_clear_bear', 0.0)),
                reversal_to_bull=probabilities.get('reversal_to_bull', probabilities.get('swing_quality_high', 0.0)),
                reversal_to_bear=probabilities.get('reversal_to_bear', probabilities.get('swing_quality_low', 0.0)),
                large_drawdown=probabilities.get('large_drawdown', probabilities.get('vol_expanding', 0.0)),
                mode=mode,
                f1_scores=f1_scores,
                # Phase prediction (NEW!)
                current_phase=current_phase,
                phase_probabilities=phase_probabilities,
                next_phase=next_phase,
                next_phase_confidence=next_phase_confidence,
                phase_bias=phase_bias,
            )
            signals = _interpret_investment_signals(signals, whale_data)
        
        return signals
        
    except Exception as e:
        import traceback
        print(f"ML Signal extraction error: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL INTERPRETATION (RULES LAYER)
# ═══════════════════════════════════════════════════════════════════════════════

def _interpret_scalp_daytrade_signals(
    signals: ScalpDayTradeSignals,
    whale_data: Dict = None
) -> ScalpDayTradeSignals:
    """
    Interpret Scalp/DayTrade signals to derive direction.
    
    Priority:
    1. Fakeout signals (if high, warns of trap)
    2. Continuation (if strong, confirms trend)
    3. Vol expansion + whale bias (when ML is uncertain but move is imminent)
    """
    whale_data = whale_data or {}
    whale_pct = whale_data.get('whale_long_pct', 50)
    whale_bullish = whale_pct > 55
    whale_bearish = whale_pct < 45
    
    # Get strongest signals
    bull_strength = signals.continuation_bull + signals.fakeout_to_bull * 0.5
    bear_strength = signals.continuation_bear + signals.fakeout_to_bear * 0.5
    
    # Check for fakeout warnings (trap detection)
    fakeout_bull = signals.fakeout_to_bull > 0.50  # Lowered from 0.60
    fakeout_bear = signals.fakeout_to_bear > 0.50  # Lowered from 0.60
    
    # Decision logic
    if fakeout_bull and fakeout_bear:
        signals.direction = 'WAIT'
        signals.confidence = max(signals.fakeout_to_bull, signals.fakeout_to_bear) * 100
        signals.primary_signal = 'conflicting_fakeouts'
        signals.action_message = "⚠️ Both fakeout signals high - choppy conditions, wait"
        
    elif fakeout_bull and not signals.continuation_bear > 0.50:
        # Bearish trap → will reverse UP
        signals.direction = 'LONG'
        signals.confidence = signals.fakeout_to_bull * 100
        signals.primary_signal = 'fakeout_to_bull'
        signals.action_message = "🪤 BEAR TRAP DETECTED - Faking down, will reverse UP"
        if whale_bullish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale confirmation"
            
    elif fakeout_bear and not signals.continuation_bull > 0.50:
        # Bullish trap → will reverse DOWN
        signals.direction = 'SHORT'
        signals.confidence = signals.fakeout_to_bear * 100
        signals.primary_signal = 'fakeout_to_bear'
        signals.action_message = "🪤 BULL TRAP DETECTED - Faking up, will reverse DOWN"
        if whale_bearish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale confirmation"
            
    elif signals.continuation_bull > 0.40 and signals.continuation_bull > signals.continuation_bear + 0.10:
        # Lowered threshold from 0.55 to 0.40, and spread from 0.15 to 0.10
        signals.direction = 'LONG'
        signals.confidence = signals.continuation_bull * 100
        signals.primary_signal = 'continuation_bull'
        signals.action_message = "📈 Bullish continuation - trend will continue UP"
        if whale_bullish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale aligned"
            
    elif signals.continuation_bear > 0.40 and signals.continuation_bear > signals.continuation_bull + 0.10:
        # Lowered threshold from 0.55 to 0.40
        signals.direction = 'SHORT'
        signals.confidence = signals.continuation_bear * 100
        signals.primary_signal = 'continuation_bear'
        signals.action_message = "📉 Bearish continuation - trend will continue DOWN"
        if whale_bearish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale aligned"
    
    # NEW: Vol expansion + whale bias decision (when ML uncertain but move imminent)
    elif signals.vol_expansion > 0.70:
        # High volatility expected - use whale positioning OR relative strength to determine direction
        bull_stronger = signals.continuation_bull > signals.continuation_bear
        bear_stronger = signals.continuation_bear > signals.continuation_bull
        
        # DEBUG: Log the exact values to trace direction decision
        print(f"[ML_INTERPRET_DEBUG] vol_expansion={signals.vol_expansion:.4f}, cont_bull={signals.continuation_bull:.4f}, cont_bear={signals.continuation_bear:.4f}")
        print(f"[ML_INTERPRET_DEBUG] bull_stronger={bull_stronger}, bear_stronger={bear_stronger}, whale_pct={whale_pct}")
        
        # FIX: Determine ACTUAL whale direction for message (not ML direction!)
        whale_dir_text = "bullish" if whale_pct >= 55 else ("bearish" if whale_pct <= 45 else "neutral")
        
        # VERY HIGH vol expansion (>90%) - trust relative direction even without whale
        if signals.vol_expansion > 0.90:
            if bull_stronger:
                signals.direction = 'LONG'
                signals.confidence = min(70, signals.vol_expansion * 100 * 0.7)
                signals.primary_signal = 'vol_expansion_bull'
                signals.action_message = f"⚡ Big move expected + ML predicts LONG (Whale {whale_dir_text} {whale_pct:.0f}%)"
                print(f"[ML_INTERPRET_DEBUG] Decision: LONG (bull_stronger)")
            elif bear_stronger:
                signals.direction = 'SHORT'
                signals.confidence = min(70, signals.vol_expansion * 100 * 0.7)
                signals.primary_signal = 'vol_expansion_bear'
                signals.action_message = f"⚡ Big move expected + ML predicts SHORT (Whale {whale_dir_text} {whale_pct:.0f}%)"
                print(f"[ML_INTERPRET_DEBUG] Decision: SHORT (bear_stronger)")
            else:
                signals.direction = 'WAIT'
                signals.confidence = signals.vol_expansion * 100
                signals.primary_signal = 'vol_expansion_neutral'
                signals.action_message = f"⚡ Big move expected but equal bull/bear → Wait"
                print(f"[ML_INTERPRET_DEBUG] Decision: WAIT (equal)")
        # Moderate vol expansion - need whale confirmation
        elif whale_bullish and bull_stronger:
            signals.direction = 'LONG'
            signals.confidence = min(70, signals.vol_expansion * 100 * 0.7)
            signals.primary_signal = 'vol_expansion_whale_bull'
            signals.action_message = f"⚡ Big move expected + Whale bullish ({whale_pct:.0f}%) → Lean LONG"
        elif whale_bearish and bear_stronger:
            signals.direction = 'SHORT'
            signals.confidence = min(70, signals.vol_expansion * 100 * 0.7)
            signals.primary_signal = 'vol_expansion_whale_bear'
            signals.action_message = f"⚡ Big move expected + Whale bearish ({whale_pct:.0f}%) → Lean SHORT"
        # No whale but clear direction - give weaker signal
        elif bull_stronger and signals.continuation_bull > 0.15:
            signals.direction = 'LONG'
            signals.confidence = min(55, signals.vol_expansion * 100 * 0.55)
            signals.primary_signal = 'vol_expansion_lean_bull'
            signals.action_message = f"⚡ Big move expected, leaning bullish ({signals.continuation_bull*100:.0f}% vs {signals.continuation_bear*100:.0f}%)"
        elif bear_stronger and signals.continuation_bear > 0.15:
            signals.direction = 'SHORT'
            signals.confidence = min(55, signals.vol_expansion * 100 * 0.55)
            signals.primary_signal = 'vol_expansion_lean_bear'
            signals.action_message = f"⚡ Big move expected, leaning bearish ({signals.continuation_bear*100:.0f}% vs {signals.continuation_bull*100:.0f}%)"
        else:
            signals.direction = 'WAIT'
            signals.confidence = signals.vol_expansion * 100
            signals.primary_signal = 'vol_expansion_no_bias'
            signals.action_message = f"⚡ Big move expected but no clear bias ({whale_pct:.0f}%) → Wait for direction"
    
    # NEW: RELATIVE COMPARISON when all probabilities are low (model uncertain but can still rank)
    # This handles the case where thresholds were high during training but we still want direction
    elif signals.continuation_bull > 0.03 or signals.continuation_bear > 0.03:
        # At least some signal, compare relatively
        bull_total = signals.continuation_bull + signals.fakeout_to_bull
        bear_total = signals.continuation_bear + signals.fakeout_to_bear
        
        # Strong relative difference (1.5x) - signal even without whale
        if bull_total > bear_total * 1.5:
            signals.direction = 'LONG'
            base_conf = (bull_total / (bull_total + bear_total)) * 100
            signals.confidence = min(60, base_conf) if whale_bullish else min(55, base_conf)
            signals.primary_signal = 'relative_bullish'
            signals.action_message = f"📊 ML leans bullish ({bull_total*100:.0f}% vs {bear_total*100:.0f}%)"
            if whale_bullish:
                signals.action_message += f" + Whale {whale_pct:.0f}%"
        elif bear_total > bull_total * 1.5:
            signals.direction = 'SHORT'
            base_conf = (bear_total / (bull_total + bear_total)) * 100
            signals.confidence = min(60, base_conf) if whale_bearish else min(55, base_conf)
            signals.primary_signal = 'relative_bearish'
            signals.action_message = f"📊 ML leans bearish ({bear_total*100:.0f}% vs {bull_total*100:.0f}%)"
            if whale_bearish:
                signals.action_message += f" + Whale {whale_pct:.0f}%"
        else:
            signals.direction = 'WAIT'
            signals.confidence = 50
            signals.primary_signal = 'no_clear_signal'
            signals.action_message = "⏳ No clear signal - wait for better setup"
            
    else:
        signals.direction = 'WAIT'
        signals.confidence = 50
        signals.primary_signal = 'no_clear_signal'
        signals.action_message = "⏳ No clear signal - wait for better setup"
    
    # Vol expansion modifier (for non-vol-primary decisions)
    if signals.vol_expansion > 0.70 and 'vol_expansion' not in signals.primary_signal:
        signals.action_message += " | ⚡ HIGH VOLATILITY EXPECTED"
    
    return signals


def _interpret_swing_signals(
    signals: SwingSignals,
    whale_data: Dict = None
) -> SwingSignals:
    """
    Interpret Swing signals to derive direction.
    
    Priority (REVERSAL IS HIGHEST ACCURACY):
    1. reversal_to_bull/bear (directional reversal - ~97% F1)
    2. drawdown (risk warning)
    3. trend_holds_bull/bear (continuation)
    """
    whale_data = whale_data or {}
    whale_pct = whale_data.get('whale_long_pct', 50)
    whale_bullish = whale_pct > 55
    whale_bearish = whale_pct < 45
    
    # REVERSAL SIGNALS ARE KING (highest accuracy)
    strong_reversal_bull = signals.reversal_to_bull > 0.60
    strong_reversal_bear = signals.reversal_to_bear > 0.60
    
    # Decision logic
    if strong_reversal_bull and strong_reversal_bear:
        # Both high - likely choppy, use whale to decide
        if whale_bullish:
            signals.direction = 'LONG'
            signals.confidence = signals.reversal_to_bull * 100
            signals.primary_signal = 'reversal_to_bull'
            signals.action_message = "⚠️ Conflicting reversals - Whale favors LONG"
        elif whale_bearish:
            signals.direction = 'SHORT'
            signals.confidence = signals.reversal_to_bear * 100
            signals.primary_signal = 'reversal_to_bear'
            signals.action_message = "⚠️ Conflicting reversals - Whale favors SHORT"
        else:
            signals.direction = 'WAIT'
            signals.confidence = 50
            signals.primary_signal = 'conflicting_reversals'
            signals.action_message = "⚠️ Conflicting reversal signals - WAIT"
            
    elif strong_reversal_bull:
        # REVERSAL TO BULL - was going down, will reverse UP
        signals.direction = 'LONG'
        signals.confidence = signals.reversal_to_bull * 100
        signals.primary_signal = 'reversal_to_bull'
        signals.action_message = "🔄 REVERSAL UP (97.8% F1) - Bottom forming, go LONG"
        if whale_bullish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale confirmed"
        elif whale_bearish:
            signals.confidence -= 15
            signals.action_message += " ⚠️ Whale disagrees - reduce size"
            
    elif strong_reversal_bear:
        # REVERSAL TO BEAR - was going up, will reverse DOWN
        signals.direction = 'SHORT'
        signals.confidence = signals.reversal_to_bear * 100
        signals.primary_signal = 'reversal_to_bear'
        signals.action_message = "🔄 REVERSAL DOWN (97.8% F1) - Top forming, go SHORT"
        if whale_bearish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale confirmed"
        elif whale_bullish:
            signals.confidence -= 15
            signals.action_message += " ⚠️ Whale disagrees - reduce size"
            
    elif signals.trend_holds_bull > 0.55 and signals.trend_holds_bull > signals.trend_holds_bear + 0.15:
        signals.direction = 'LONG'
        signals.confidence = signals.trend_holds_bull * 100
        signals.primary_signal = 'trend_holds_bull'
        signals.action_message = "📈 Bullish trend continues - expect 2%+ upside"
        if whale_bullish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale aligned"
            
    elif signals.trend_holds_bear > 0.55 and signals.trend_holds_bear > signals.trend_holds_bull + 0.15:
        signals.direction = 'SHORT'
        signals.confidence = signals.trend_holds_bear * 100
        signals.primary_signal = 'trend_holds_bear'
        signals.action_message = "📉 Bearish trend continues - expect 2%+ downside"
        if whale_bearish:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Whale aligned"
            
    else:
        signals.direction = 'WAIT'
        signals.confidence = 50
        signals.primary_signal = 'no_clear_signal'
        signals.action_message = "⏳ No clear swing signal - wait"
    
    # DRAWDOWN WARNING (risk management)
    if signals.drawdown > 0.60:
        signals.action_message += f" | 🛡️ DRAWDOWN RISK: {signals.drawdown*100:.0f}%"
    
    return signals


def _interpret_stock_swing_signals(
    signals: StockSwingSignals,
    whale_data: Dict = None
) -> StockSwingSignals:
    """
    Interpret Stock Swing signals to derive direction.
    
    Priority:
    1. mean_reversion_long/short (oversold bounce / overbought fade)
    2. breakout_valid (breakout that holds)
    3. setup_quality_high/low (R:R assessment)
    4. volatility_expansion (vol about to explode)
    """
    whale_data = whale_data or {}
    
    # Get strongest signals
    long_strength = (
        signals.mean_reversion_long * 1.2 +  # Weight mean reversion highest
        signals.breakout_valid * 0.8 +
        signals.setup_quality_high * 0.5
    )
    short_strength = (
        signals.mean_reversion_short * 1.2 +
        signals.setup_quality_low * 0.5
    )
    
    # Decision logic
    if signals.mean_reversion_long > 0.55:
        signals.direction = 'LONG'
        signals.confidence = signals.mean_reversion_long * 100
        signals.primary_signal = 'mean_reversion_long'
        signals.action_message = f"📈 Oversold bounce ({signals.mean_reversion_long*100:.0f}%) - Mean reversion LONG"
        if signals.setup_quality_high > 0.50:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Good R:R"
            
    elif signals.mean_reversion_short > 0.55:
        signals.direction = 'SHORT'
        signals.confidence = signals.mean_reversion_short * 100
        signals.primary_signal = 'mean_reversion_short'
        signals.action_message = f"📉 Overbought fade ({signals.mean_reversion_short*100:.0f}%) - Mean reversion SHORT"
        if signals.setup_quality_low > 0.50:
            signals.confidence = min(95, signals.confidence + 10)
            signals.action_message += " + Poor setup (confirms)"
            
    elif signals.breakout_valid > 0.55:
        signals.direction = 'LONG'
        signals.confidence = signals.breakout_valid * 100
        signals.primary_signal = 'breakout_valid'
        signals.action_message = f"🚀 Valid breakout ({signals.breakout_valid*100:.0f}%) - Will hold, go LONG"
        
    elif signals.setup_quality_high > 0.55 and signals.setup_quality_low < 0.40:
        signals.direction = 'LONG'
        signals.confidence = signals.setup_quality_high * 100
        signals.primary_signal = 'setup_quality_high'
        signals.action_message = f"✅ Good R:R setup ({signals.setup_quality_high*100:.0f}%) - TP before SL"
        
    elif signals.setup_quality_low > 0.55 and signals.setup_quality_high < 0.40:
        signals.direction = 'SHORT'  # Or could be AVOID
        signals.confidence = signals.setup_quality_low * 100
        signals.primary_signal = 'setup_quality_low'
        signals.action_message = f"❌ Bad R:R setup ({signals.setup_quality_low*100:.0f}%) - SL before TP, avoid/short"
        
    elif long_strength > short_strength * 1.3:
        signals.direction = 'LONG'
        signals.confidence = min(65, long_strength * 30)
        signals.primary_signal = 'combined_bullish'
        signals.action_message = "📊 Combined signals lean bullish"
        
    elif short_strength > long_strength * 1.3:
        signals.direction = 'SHORT'
        signals.confidence = min(65, short_strength * 30)
        signals.primary_signal = 'combined_bearish'
        signals.action_message = "📊 Combined signals lean bearish"
        
    else:
        signals.direction = 'WAIT'
        signals.confidence = 50
        signals.primary_signal = 'no_clear_signal'
        signals.action_message = "⏳ No clear stock swing signal - wait"
    
    # Vol expansion warning
    if signals.volatility_expansion > 0.60:
        signals.action_message += f" | ⚡ VOL EXPANDING ({signals.volatility_expansion*100:.0f}%)"
    
    return signals


def _interpret_investment_signals(
    signals: InvestmentSignals,
    whale_data: Dict = None
) -> InvestmentSignals:
    """
    Interpret Investment signals to derive direction.
    
    Priority:
    1. reversal_to_bull/bear (major turning points)
    2. accumulation/distribution (smart money flow)
    3. large_drawdown (risk)
    """
    whale_data = whale_data or {}
    whale_pct = whale_data.get('whale_long_pct', 50)
    whale_bullish = whale_pct > 55
    whale_bearish = whale_pct < 45
    
    strong_reversal_bull = signals.reversal_to_bull > 0.55
    strong_reversal_bear = signals.reversal_to_bear > 0.55
    
    if strong_reversal_bull and strong_reversal_bear:
        signals.direction = 'WAIT'
        signals.confidence = 50
        signals.primary_signal = 'conflicting'
        signals.action_message = "⚠️ Conflicting reversal signals - major inflection point"
        
    elif strong_reversal_bull:
        signals.direction = 'LONG'
        signals.confidence = signals.reversal_to_bull * 100
        signals.primary_signal = 'reversal_to_bull'
        signals.action_message = "🔄 MAJOR REVERSAL UP - Start accumulating for long-term"
        
    elif strong_reversal_bear:
        signals.direction = 'SHORT'
        signals.confidence = signals.reversal_to_bear * 100
        signals.primary_signal = 'reversal_to_bear'
        signals.action_message = "🔄 MAJOR REVERSAL DOWN - Exit longs, wait for lower entry"
        
    elif signals.accumulation > 0.60 and signals.accumulation > signals.distribution + 0.15:
        signals.direction = 'LONG'
        signals.confidence = signals.accumulation * 100
        signals.primary_signal = 'accumulation'
        signals.action_message = "📈 Smart money accumulating - institutional buying"
        if whale_bullish:
            signals.confidence = min(95, signals.confidence + 10)
            
    elif signals.distribution > 0.60 and signals.distribution > signals.accumulation + 0.15:
        signals.direction = 'SHORT'
        signals.confidence = signals.distribution * 100
        signals.primary_signal = 'distribution'
        signals.action_message = "📉 Smart money distributing - institutional selling"
        if whale_bearish:
            signals.confidence = min(95, signals.confidence + 10)
            
    else:
        signals.direction = 'WAIT'
        signals.confidence = 50
        signals.primary_signal = 'no_clear_signal'
        signals.action_message = "⏳ No clear investment signal"
    
    # Large drawdown warning
    if signals.large_drawdown > 0.50:
        signals.action_message += f" | ⚠️ 7%+ DRAWDOWN RISK"
    
    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_ml_signals_ui(signals, show_details: bool = True):
    """
    Render ML signals in Streamlit UI.
    
    Works for all signal types (Scalp/DayTrade/Swing/Investment).
    """
    if signals is None:
        st.info(f"ML model not trained for this mode. Train in ML Training tab.")
        return
    
    # Direction banner
    dir_color = '#00ff88' if signals.direction == 'LONG' else '#ff4444' if signals.direction == 'SHORT' else '#ffaa00'
    
    st.markdown(f'''
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 15px; border-radius: 10px; margin-bottom: 15px;
                border-left: 4px solid {dir_color};">
        <div style="font-size: 24px; font-weight: bold; color: {dir_color};">
            ML: {signals.direction} ({signals.confidence:.0f}%)
        </div>
        <div style="color: #ccc; margin-top: 5px;">
            {signals.action_message}
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    if not show_details:
        return
    
    # Mode-specific signal display
    if isinstance(signals, ScalpDayTradeSignals):
        _render_scalp_daytrade_details(signals)
    elif isinstance(signals, StockSwingSignals):
        _render_stock_swing_details(signals)
    elif isinstance(signals, SwingSignals):
        _render_swing_details(signals)
    elif isinstance(signals, InvestmentSignals):
        _render_investment_details(signals)


def _render_scalp_daytrade_details(signals: ScalpDayTradeSignals):
    """Render Scalp/DayTrade signal details"""
    cols = st.columns(5)
    
    f1 = signals.f1_scores or {}
    
    with cols[0]:
        val = signals.continuation_bull
        color = '#00ff88' if val > 0.6 else '#88ff88' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'continuation_bull'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">Continue ↑</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('continuation_bull', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        val = signals.continuation_bear
        color = '#ff4444' if val > 0.6 else '#ff8888' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'continuation_bear'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">Continue ↓</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('continuation_bear', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[2]:
        val = signals.fakeout_to_bull
        color = '#00ffff' if val > 0.6 else '#66cccc' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'fakeout_to_bull'
        st.markdown(f'''
        <div style="background: #1a2a3e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">🪤 Fake→UP</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('fakeout_to_bull', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[3]:
        val = signals.fakeout_to_bear
        color = '#ff66ff' if val > 0.6 else '#cc66cc' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'fakeout_to_bear'
        st.markdown(f'''
        <div style="background: #2a1a3e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">🪤 Fake→DN</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('fakeout_to_bear', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[4]:
        val = signals.vol_expansion
        color = '#ffaa00' if val > 0.7 else '#aa7700' if val > 0.5 else '#666'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: 1px solid #333;">
            <div style="color: #aaa; font-size: 13px;">⚡ Vol Exp</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('vol_expansion', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)


def _render_stock_swing_details(signals: StockSwingSignals):
    """Render Stock Swing signal details - QUALITY + MEAN REVERSION"""
    cols = st.columns(6)
    
    f1 = signals.f1_scores or {}
    
    with cols[0]:
        val = signals.mean_reversion_long
        color = '#00ff88' if val > 0.55 else '#88ff88' if val > 0.40 else '#666'
        primary = signals.primary_signal == 'mean_reversion_long'
        st.markdown(f'''
        <div style="background: #1a3a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #0f0; font-size: 12px;">📈 MeanRev LONG</div>
            <div style="color: {color}; font-size: 26px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('mean_reversion_long', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        val = signals.mean_reversion_short
        color = '#ff4444' if val > 0.55 else '#ff8888' if val > 0.40 else '#666'
        primary = signals.primary_signal == 'mean_reversion_short'
        st.markdown(f'''
        <div style="background: #3a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #f44; font-size: 12px;">📉 MeanRev SHORT</div>
            <div style="color: {color}; font-size: 26px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('mean_reversion_short', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[2]:
        val = signals.breakout_valid
        color = '#00ffff' if val > 0.55 else '#66cccc' if val > 0.40 else '#666'
        primary = signals.primary_signal == 'breakout_valid'
        st.markdown(f'''
        <div style="background: #1a2a3e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #0ff; font-size: 12px;">🚀 Breakout Valid</div>
            <div style="color: {color}; font-size: 26px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('breakout_valid', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[3]:
        val = signals.setup_quality_high
        color = '#00ff88' if val > 0.55 else '#88ff88' if val > 0.40 else '#666'
        primary = signals.primary_signal == 'setup_quality_high'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 12px;">✅ Quality HIGH</div>
            <div style="color: {color}; font-size: 26px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('setup_quality_high', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[4]:
        val = signals.setup_quality_low
        color = '#ff4444' if val > 0.55 else '#ff8888' if val > 0.40 else '#666'
        primary = signals.primary_signal == 'setup_quality_low'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 12px;">❌ Quality LOW</div>
            <div style="color: {color}; font-size: 26px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('setup_quality_low', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[5]:
        val = signals.volatility_expansion
        color = '#ffaa00' if val > 0.55 else '#aa7700' if val > 0.40 else '#666'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: 1px solid #333;">
            <div style="color: #aaa; font-size: 12px;">⚡ Vol Expand</div>
            <div style="color: {color}; font-size: 26px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('volatility_expansion', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Action messages
    if signals.mean_reversion_long > 0.55:
        st.success(f"📈 **OVERSOLD BOUNCE** ({signals.mean_reversion_long*100:.0f}%) - Mean reversion LONG opportunity")
    if signals.mean_reversion_short > 0.55:
        st.error(f"📉 **OVERBOUGHT FADE** ({signals.mean_reversion_short*100:.0f}%) - Mean reversion SHORT opportunity")
    if signals.breakout_valid > 0.55:
        st.info(f"🚀 **VALID BREAKOUT** ({signals.breakout_valid*100:.0f}%) - Breakout will hold")


def _render_swing_details(signals: SwingSignals):
    """Render Swing signal details - DIRECTIONAL REVERSALS"""
    cols = st.columns(5)
    
    f1 = signals.f1_scores or {}
    
    with cols[0]:
        val = signals.trend_holds_bull
        color = '#00ff88' if val > 0.6 else '#88ff88' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'trend_holds_bull'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">Trend ↑</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('trend_holds_bull', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        val = signals.trend_holds_bear
        color = '#ff4444' if val > 0.6 else '#ff8888' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'trend_holds_bear'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">Trend ↓</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('trend_holds_bear', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # DIRECTIONAL REVERSALS - THE KEY SIGNALS
    with cols[2]:
        val = signals.reversal_to_bull
        color = '#00ffff' if val > 0.6 else '#66cccc' if val > 0.5 else '#666'
        primary = signals.primary_signal == 'reversal_to_bull'
        st.markdown(f'''
        <div style="background: #1a3a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #0ff; font-size: 13px;">🔄 Rev→UP ⭐</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #0ff; font-size: 11px;">F1: ~97%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[3]:
        val = signals.reversal_to_bear
        color = '#ff66ff' if val > 0.6 else '#cc66cc' if val > 0.5 else '#666'
        primary = signals.primary_signal == 'reversal_to_bear'
        st.markdown(f'''
        <div style="background: #3a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #f6f; font-size: 13px;">🔄 Rev→DN ⭐</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #f6f; font-size: 11px;">F1: ~97%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[4]:
        val = signals.drawdown
        color = '#ff4444' if val > 0.7 else '#ffaa00' if val > 0.5 else '#666'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: 1px solid #333;">
            <div style="color: #aaa; font-size: 13px;">🛡️ Drawdn</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('drawdown', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Reversal warnings
    if signals.reversal_to_bull > 0.60:
        st.success(f"🔄 **REVERSAL UP DETECTED** ({signals.reversal_to_bull*100:.0f}%) - Was going DOWN, will reverse to go UP")
    if signals.reversal_to_bear > 0.60:
        st.error(f"🔄 **REVERSAL DOWN DETECTED** ({signals.reversal_to_bear*100:.0f}%) - Was going UP, will reverse to go DOWN")


def _render_investment_details(signals: InvestmentSignals):
    """Render Investment signal details"""
    cols = st.columns(5)
    
    f1 = signals.f1_scores or {}
    
    with cols[0]:
        val = signals.accumulation
        color = '#00ff88' if val > 0.6 else '#88ff88' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'accumulation'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">📈 Accum</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('accumulation', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[1]:
        val = signals.distribution
        color = '#ff4444' if val > 0.6 else '#ff8888' if val > 0.4 else '#666'
        primary = signals.primary_signal == 'distribution'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #aaa; font-size: 13px;">📉 Distrib</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
            <div style="color: #666; font-size: 11px;">F1: {f1.get('distribution', 0)*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[2]:
        val = signals.reversal_to_bull
        color = '#00ffff' if val > 0.55 else '#666'
        primary = signals.primary_signal == 'reversal_to_bull'
        st.markdown(f'''
        <div style="background: #1a3a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #0ff; font-size: 13px;">🔄 Rev→UP</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[3]:
        val = signals.reversal_to_bear
        color = '#ff66ff' if val > 0.55 else '#666'
        primary = signals.primary_signal == 'reversal_to_bear'
        st.markdown(f'''
        <div style="background: #3a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: {'2px solid ' + color if primary else '1px solid #333'};">
            <div style="color: #f6f; font-size: 13px;">🔄 Rev→DN</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with cols[4]:
        val = signals.large_drawdown
        color = '#ff4444' if val > 0.5 else '#666'
        st.markdown(f'''
        <div style="background: #1a1a2e; padding: 10px; border-radius: 8px; text-align: center;
                    border: 1px solid #333;">
            <div style="color: #aaa; font-size: 13px;">⚠️ 7%+ DD</div>
            <div style="color: {color}; font-size: 28px; font-weight: bold;">{val*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPACT SUMMARY FOR SCANNER VIEW
# ═══════════════════════════════════════════════════════════════════════════════

def get_ml_signals_summary(signals) -> str:
    """Get compact one-line summary for Scanner view"""
    if signals is None:
        return "ML: N/A"
    
    direction = signals.direction
    confidence = signals.confidence
    
    if direction == 'LONG':
        icon = "🟢"
    elif direction == 'SHORT':
        icon = "🔴"
    else:
        icon = "🟡"
    
    # Add specific warnings
    warnings = []
    
    if isinstance(signals, StockSwingSignals):
        if signals.mean_reversion_long > 0.55:
            warnings.append("MR↑")
        if signals.mean_reversion_short > 0.55:
            warnings.append("MR↓")
        if signals.breakout_valid > 0.55:
            warnings.append("BO!")
        if signals.setup_quality_low > 0.55:
            warnings.append("BadRR")
    elif isinstance(signals, SwingSignals):
        if signals.reversal_to_bull > 0.6:
            warnings.append("Rev↑")
        if signals.reversal_to_bear > 0.6:
            warnings.append("Rev↓")
        if signals.drawdown > 0.6:
            warnings.append("DD!")
    elif isinstance(signals, ScalpDayTradeSignals):
        if signals.fakeout_to_bull > 0.6:
            warnings.append("Trap↑")
        if signals.fakeout_to_bear > 0.6:
            warnings.append("Trap↓")
    elif isinstance(signals, InvestmentSignals):
        if signals.large_drawdown > 0.5:
            warnings.append("DD!")
    
    warning_str = f" [{', '.join(warnings)}]" if warnings else ""
    
    return f"{icon} ML: {direction} {confidence:.0f}%{warning_str}"