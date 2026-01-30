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


@dataclass
class InvestmentSignals:
    """Signals for Investment mode - DIRECTIONAL"""
    accumulation: float = 0.0         # Smart money accumulating → price rising
    distribution: float = 0.0         # Smart money distributing → price falling
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


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_ml_signals(
    df: pd.DataFrame,
    mode: str,
    whale_data: Dict = None,
    market_type: str = 'crypto',  # 'crypto' or 'stock'
) -> Optional[ScalpDayTradeSignals | SwingSignals | InvestmentSignals]:
    """
    Get ML signals for the specified mode.
    
    Args:
        df: OHLCV DataFrame with at least 50 candles
        mode: 'scalp', 'daytrade', 'swing', or 'investment'
        whale_data: Whale positioning data (if available)
        market_type: 'crypto' or 'stock' (affects model selection)
    
    Returns:
        Mode-specific signals dataclass with probabilities and derived direction
    """
    if df is None or len(df) < 50:
        return None
    
    mode = mode.lower()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 1: Try dedicated probabilistic models
    # ═══════════════════════════════════════════════════════════════════════════
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    
    # Try market-specific model first, fall back to general
    model_files = [
        f'prob_model_{mode}_{market_type}.pkl',  # NEW: e.g., prob_model_swing_crypto.pkl
        f'prob_{mode}_{market_type}.pkl',         # e.g., prob_swing_crypto.pkl
        f'prob_{mode}.pkl',                       # e.g., prob_swing.pkl
        f'prob_model_{mode}.pkl',                 # e.g., prob_model_swing.pkl (legacy)
    ]
    
    model_path = None
    for mf in model_files:
        path = os.path.join(model_dir, mf)
        if os.path.exists(path):
            model_path = path
            break
    
    # Also check patterns directory (legacy location)
    if model_path is None:
        patterns_dir = os.path.join(os.path.dirname(__file__), 'models', 'patterns')
        legacy_files = [
            f'pattern_{mode}_{market_type}s.pkl',  # e.g., pattern_swing_stocks.pkl
            f'pattern_{mode}_{market_type}.pkl',   # e.g., pattern_swing_crypto.pkl
        ]
        for lf in legacy_files:
            legacy_path = os.path.join(patterns_dir, lf)
            if os.path.exists(legacy_path):
                model_path = legacy_path
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
        
        # Extract model components
        models = model_data.get('models_per_label', model_data.get('models', {}))
        scaler = model_data.get('scaler')
        metadata = model_data.get('metadata', {})
        
        # FIX: Extract F1 scores from correct location (metrics dict, not f1_scores)
        metrics_dict = metadata.get('metrics', {})
        f1_scores = {}
        for label, m in metrics_dict.items():
            if isinstance(m, dict):
                f1_scores[label] = m.get('f1', 0.0)
        
        # Also try direct f1_scores key as fallback
        if not f1_scores:
            f1_scores = metadata.get('f1_scores', {})
        
        # Get labels for this mode
        if mode in ['scalp', 'daytrade']:
            labels = ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion']
        elif mode == 'swing':
            labels = ['trend_holds_bull', 'trend_holds_bear', 'reversal_to_bull', 'reversal_to_bear', 'drawdown']
        elif mode == 'investment':
            labels = ['accumulation', 'distribution', 'reversal_to_bull', 'reversal_to_bear', 'large_drawdown']
        else:
            return None
        
        # Extract features for latest candle
        from .probabilistic_ml import extract_enhanced_features
        features = extract_enhanced_features(df, len(df) - 1, whale_data)
        
        # Scale if scaler available
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
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
            )
            signals = _interpret_scalp_daytrade_signals(signals, whale_data)
            
        elif mode == 'swing':
            signals = SwingSignals(
                trend_holds_bull=probabilities.get('trend_holds_bull', 0.0),
                trend_holds_bear=probabilities.get('trend_holds_bear', 0.0),
                reversal_to_bull=probabilities.get('reversal_to_bull', 0.0),
                reversal_to_bear=probabilities.get('reversal_to_bear', 0.0),
                drawdown=probabilities.get('drawdown', 0.0),
                mode=mode,
                f1_scores=f1_scores,
            )
            signals = _interpret_swing_signals(signals, whale_data)
            
        elif mode == 'investment':
            signals = InvestmentSignals(
                accumulation=probabilities.get('accumulation', 0.0),
                distribution=probabilities.get('distribution', 0.0),
                reversal_to_bull=probabilities.get('reversal_to_bull', 0.0),
                reversal_to_bear=probabilities.get('reversal_to_bear', 0.0),
                large_drawdown=probabilities.get('large_drawdown', 0.0),
                mode=mode,
                f1_scores=f1_scores,
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
        
        # VERY HIGH vol expansion (>90%) - trust relative direction even without whale
        if signals.vol_expansion > 0.90:
            if bull_stronger:
                signals.direction = 'LONG'
                signals.confidence = min(70, signals.vol_expansion * 100 * 0.7)
                signals.primary_signal = 'vol_expansion_bull'
                signals.action_message = f"⚡ Big move expected + Whale bullish ({whale_pct:.0f}%) → Lean LONG"
            elif bear_stronger:
                signals.direction = 'SHORT'
                signals.confidence = min(70, signals.vol_expansion * 100 * 0.7)
                signals.primary_signal = 'vol_expansion_bear'
                signals.action_message = f"⚡ Big move expected + Whale bearish ({whale_pct:.0f}%) → Lean SHORT"
            else:
                signals.direction = 'WAIT'
                signals.confidence = signals.vol_expansion * 100
                signals.primary_signal = 'vol_expansion_neutral'
                signals.action_message = f"⚡ Big move expected but equal bull/bear → Wait"
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
    
    if isinstance(signals, SwingSignals):
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