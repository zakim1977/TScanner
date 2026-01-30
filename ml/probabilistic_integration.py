"""
Probabilistic ML Integration
=============================

Integrates the new probabilistic ML system with InvestorIQ.

This module bridges:
1. Old system: direction_model → LONG/SHORT/WAIT
2. New system: prob_model → P_continuation, P_fakeout, P_vol_expansion → LONG/SHORT/WAIT

The key insight: ML outputs PROBABILITIES, Rules make DECISIONS.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

# Import both systems
from .probabilistic_ml import (
    ProbabilisticMLTrainer,
    ProbabilisticPrediction,
    get_probabilistic_prediction,
    ENHANCED_FEATURES,
    MODE_LABELS,
)
from .ml_engine import MLPrediction, MLEngine


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ML INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedMLEngine:
    """
    Unified ML interface that supports both:
    1. Legacy direction prediction (LONG/SHORT/WAIT)
    2. New probabilistic prediction (P_continuation, P_fakeout, etc.)
    
    Usage:
        engine = UnifiedMLEngine()
        
        # Legacy mode (backward compatible)
        pred = engine.predict(df, mode='daytrade', legacy=True)
        print(pred.direction, pred.confidence)
        
        # Probabilistic mode (new)
        pred = engine.predict(df, mode='daytrade', legacy=False)
        print(pred.probabilities)  # {'continuation': 0.64, 'fakeout': 0.19, ...}
        print(pred.direction, pred.confidence)  # Still get direction!
    """
    
    def __init__(self):
        self.legacy_engine = MLEngine()
        self.prob_trainer = ProbabilisticMLTrainer()
        self._use_probabilistic = self._check_prob_models()
    
    def _check_prob_models(self) -> bool:
        """Check if probabilistic models are trained"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
        if not os.path.exists(model_dir):
            return False
        
        for mode in ['scalp', 'daytrade', 'swing', 'investment']:
            model_path = os.path.join(model_dir, f'prob_model_{mode}.pkl')
            if os.path.exists(model_path):
                return True
        
        return False
    
    def predict(
        self,
        df: pd.DataFrame,
        mode: str = 'daytrade',
        whale_data: Dict = None,
        smc_data: Dict = None,
        market_context: Dict = None,
        legacy: bool = None,  # None = auto-detect
    ) -> ProbabilisticPrediction:
        """
        Get prediction using best available model.
        
        Args:
            df: OHLCV DataFrame
            mode: Trading mode (scalp, daytrade, swing, investment)
            whale_data: Whale/retail positioning
            smc_data: SMC structure (OBs, FVGs)
            market_context: BTC correlation, fear/greed
            legacy: Force legacy mode if True
        
        Returns:
            ProbabilisticPrediction (always includes direction!)
        """
        # Auto-detect: use probabilistic if available
        if legacy is None:
            legacy = not self._use_probabilistic
        
        if legacy:
            # Convert legacy prediction to probabilistic format
            return self._legacy_to_probabilistic(df, mode, whale_data, smc_data, market_context)
        else:
            # Use new probabilistic model
            return self.prob_trainer.predict(df, mode, -1, whale_data, smc_data, market_context)
    
    def _legacy_to_probabilistic(
        self,
        df: pd.DataFrame,
        mode: str,
        whale_data: Dict = None,
        smc_data: Dict = None,
        market_context: Dict = None,
    ) -> ProbabilisticPrediction:
        """Convert legacy prediction to probabilistic format"""
        
        # Get legacy prediction
        from .feature_extractor import extract_features_from_dict
        
        # Build feature dict from available data
        whale_data = whale_data or {}
        smc_data = smc_data or {}
        market_context = market_context or {}
        
        feature_dict = {
            'whale_pct': whale_data.get('whale_long_pct', 50),
            'retail_pct': whale_data.get('retail_long_pct', 50),
            'oi_change': whale_data.get('oi_change_24h', 0),
            'price_change_24h': whale_data.get('price_change_24h', 0),
            'funding_rate': whale_data.get('funding_rate', 0),
            'at_bullish_ob': smc_data.get('at_bullish_ob', False),
            'at_bearish_ob': smc_data.get('at_bearish_ob', False),
            'btc_correlation': market_context.get('btc_correlation', 0.5),
            'fear_greed': market_context.get('fear_greed_index', 50),
            # Add price data
            'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
            'ta_score': 50,  # Default
        }
        
        # Try legacy prediction
        try:
            timeframe = '15m' if mode == 'daytrade' else '5m' if mode == 'scalp' else '4h' if mode == 'swing' else '1d'
            legacy_pred = self.legacy_engine.predict(feature_dict, timeframe)
            
            # Convert to probabilities
            confidence = legacy_pred.confidence / 100
            
            if legacy_pred.direction == 'LONG':
                probabilities = {
                    'continuation': min(0.95, confidence + 0.1),
                    'fakeout': max(0.05, 1 - confidence - 0.1),
                    'vol_expansion': 0.5,
                }
            elif legacy_pred.direction == 'SHORT':
                probabilities = {
                    'continuation': min(0.95, confidence + 0.1),
                    'fakeout': max(0.05, 1 - confidence - 0.1),
                    'vol_expansion': 0.5,
                }
            else:  # WAIT
                probabilities = {
                    'continuation': 0.4,
                    'fakeout': 0.4,
                    'vol_expansion': 0.3,
                }
            
            return ProbabilisticPrediction(
                mode=mode,
                probabilities=probabilities,
                direction=legacy_pred.direction,
                confidence=legacy_pred.confidence,
                reasoning=f"[Legacy] {legacy_pred.reasoning}",
                top_features=legacy_pred.top_features[:5] if legacy_pred.top_features else [],
                model_version=f"legacy_{mode}",
                prediction_time=datetime.now().isoformat(),
            )
            
        except Exception as e:
            # NO HEURISTIC - Return NOT_TRAINED
            return ProbabilisticPrediction(
                mode=mode,
                probabilities={},
                direction='NOT_TRAINED',
                confidence=0,
                reasoning=f"⚠️ Model not trained for {mode}. Error: {str(e)}",
                top_features=[],
                model_version="not_trained",
                prediction_time=datetime.now().isoformat(),
            )
    
    def get_detailed_probabilities(
        self,
        df: pd.DataFrame,
        mode: str,
        whale_data: Dict = None,
        smc_data: Dict = None,
        market_context: Dict = None,
    ) -> Dict:
        """
        Get detailed probability breakdown for display.
        
        Returns dict with:
        - probabilities: {label: probability}
        - direction: LONG/SHORT/WAIT
        - confidence: 0-100
        - reasoning: Human explanation
        - top_features: Most important features
        - label_meanings: What each probability means
        """
        pred = self.predict(df, mode, whale_data, smc_data, market_context)
        
        # Add label meanings
        mode_config = MODE_LABELS.get(mode, MODE_LABELS['daytrade'])
        
        label_meanings = {}
        if mode in ['daytrade', 'scalp']:
            label_meanings = {
                'continuation': 'Price will continue in current direction',
                'fakeout': 'Initial move will reverse (trap)',
                'vol_expansion': 'Volatility will increase significantly',
            }
        elif mode == 'swing':
            label_meanings = {
                'trend_holds': 'Existing trend will continue',
                'reversal': 'Trend reversal is likely',
                'drawdown': 'Significant adverse move expected',
            }
        else:  # investment
            label_meanings = {
                'accumulation': 'Smart money is accumulating',
                'distribution': 'Smart money is distributing',
                'large_drawdown': 'Major correction possible',
            }
        
        return {
            'probabilities': pred.probabilities,
            'direction': pred.direction,
            'confidence': pred.confidence,
            'reasoning': pred.reasoning,
            'top_features': pred.top_features,
            'label_meanings': label_meanings,
            'mode': mode,
            'model_version': pred.model_version,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def train_probabilistic_model(
    df: pd.DataFrame,
    mode: str,
    whale_data: Dict = None,
    smc_data: Dict = None,
    market_context: Dict = None,
    progress_callback=None,
) -> Dict:
    """
    Train probabilistic model for a trading mode.
    
    Args:
        df: Historical OHLCV DataFrame (needs ATR, BB columns)
        mode: scalp, daytrade, swing, or investment
        whale_data: Optional whale positioning context
        progress_callback: Optional callback(pct, message)
    
    Returns:
        Training metrics
    """
    trainer = ProbabilisticMLTrainer()
    return trainer.train_mode(df, mode, whale_data, smc_data, market_context, progress_callback)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

_unified_engine = None

def get_unified_ml_prediction(
    df: pd.DataFrame,
    mode: str = 'daytrade',
    whale_data: Dict = None,
    smc_data: Dict = None,
    market_context: Dict = None,
) -> ProbabilisticPrediction:
    """
    Get ML prediction using unified engine (auto-selects best model).
    
    Example:
        pred = get_unified_ml_prediction(df, 'daytrade', whale_data)
        
        # Access probabilities
        print(f"P(continuation) = {pred.probabilities.get('continuation', 0):.2f}")
        print(f"P(fakeout) = {pred.probabilities.get('fakeout', 0):.2f}")
        
        # Access derived direction
        print(f"Direction: {pred.direction} ({pred.confidence:.0f}%)")
    """
    global _unified_engine
    if _unified_engine is None:
        _unified_engine = UnifiedMLEngine()
    
    return _unified_engine.predict(df, mode, whale_data, smc_data, market_context)


def get_probability_breakdown(
    df: pd.DataFrame,
    mode: str = 'daytrade',
    whale_data: Dict = None,
) -> Dict:
    """
    Get probability breakdown for display in UI.
    
    Returns:
        {
            'probabilities': {'continuation': 0.64, 'fakeout': 0.19, ...},
            'direction': 'LONG',
            'confidence': 72,
            'reasoning': 'P_cont=64% | P_fake=19% | Whales +15% → LONG',
            'label_meanings': {...}
        }
    """
    global _unified_engine
    if _unified_engine is None:
        _unified_engine = UnifiedMLEngine()
    
    return _unified_engine.get_detailed_probabilities(df, mode, whale_data)


# ═══════════════════════════════════════════════════════════════════════════════
# HOW THE NEW SYSTEM WORKS
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT THE ML TRAINS ON (45 INPUT FEATURES):
==========================================

Price & Structure (15):
- Returns (1,3,5,10 bars)
- High-Low range
- Close position in range
- Break of structure (flag + direction)
- Distance from highs/lows
- BB compression metrics
- ATR & volatility change

Positioning & Flow (12):
- OI change (1h, 24h)
- Price vs OI divergence
- Funding rate (level + delta)
- Long/short ratio
- Whale net position
- Retail net position
- Institutional buy/sell imbalance
- Smart money flow (CMF)

SMC / Context (12):
- Liquidity sweep flags (bull/bear)
- Order block proximity (at/near)
- FVG proximity (bull/bear)
- Accumulation score
- Distribution score
- Session timing (Asian/London/NY)
- Session overlap

Market Context (6):
- BTC correlation
- BTC trend
- HTF trend alignment
- Fear/Greed index
- Weekend flag
- Days since major news


WHAT THE ML PREDICTS (MODE-SPECIFIC LABELS):
============================================

Day Trade / Scalp:
- P(continuation): Will price continue in current direction?
- P(fakeout): Will initial move reverse?
- P(vol_expansion): Will volatility increase?

Swing:
- P(trend_holds): Will existing trend continue?
- P(reversal): Is a reversal likely?
- P(drawdown): Is significant drawdown expected?

Investment:
- P(accumulation): Is smart money accumulating?
- P(distribution): Is smart money distributing?
- P(large_drawdown): Is major correction possible?


WHAT THE ML OUTPUTS:
====================

Example (Day Trade):
    P_continuation = 0.64
    P_fakeout = 0.19
    P_vol_expansion = 0.72


HOW RULES INTERPRET → DIRECTION:
================================

1. High continuation + Low fakeout + Whale bullish → LONG
2. High continuation + Low fakeout + Whale bearish → SHORT
3. High fakeout → WAIT
4. Vol expansion + continuation → LONG/SHORT based on whale
5. Else → WAIT

ML informs. Rules decide.


HOW TO VERIFY RESULTS:
======================

1. Performance by Probability Bucket:
   - 60% bucket should hit ~60% of time
   - 80% bucket should hit ~80% of time

2. Return vs Probability:
   - Higher probabilities → Higher returns?

3. Drawdown vs Probability:
   - Higher fakeout P → Higher drawdowns?

4. Regime Stability:
   - Does model work in different market conditions?

5. Feature Importance:
   - What features drive predictions?
"""