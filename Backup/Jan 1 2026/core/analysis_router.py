"""
Analysis Mode Router
=====================
Routes analysis requests to the appropriate engine based on user selection.

Modes:
- RULE_BASED: Uses MASTER_RULES (current system)
- ML: Uses ML_ENGINE for predictions
- HYBRID: Combines both ML + Rules

Usage:
    from core.analysis_router import AnalysisRouter, AnalysisMode
    
    router = AnalysisRouter(mode=AnalysisMode.HYBRID)
    result = router.analyze(
        symbol='BTCUSDT',
        whale_pct=84,
        retail_pct=57,
        ...
    )
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import streamlit as st


class AnalysisMode(Enum):
    """Available analysis modes"""
    RULE_BASED = "rule_based"    # Current MASTER_RULES system
    ML = "ml"                     # ML predictions only
    HYBRID = "hybrid"            # Combined ML + Rules


@dataclass
class UnifiedAnalysisResult:
    """
    Unified result format that works for all modes.
    Ensures consistent output regardless of which engine is used.
    """
    # Core decision
    direction: str              # 'LONG', 'SHORT', 'WAIT'
    confidence: float           # 0-100 score
    action: str                 # Action text (e.g., "STRONG BUY")
    trade_direction: str        # Trading direction for execution
    
    # Price levels
    entry_price: float = 0
    tp1_price: float = 0
    tp1_pct: float = 0
    tp2_price: float = 0
    tp2_pct: float = 0
    sl_price: float = 0
    sl_pct: float = 0
    risk_reward: float = 0
    
    # Scores breakdown
    direction_score: int = 0
    squeeze_score: int = 0
    entry_score: int = 0
    
    # ML-specific (empty for rule-based)
    ml_confidence: float = 0
    ml_predicted_move: float = 0
    similar_trades_count: int = 0
    similar_win_rate: float = 0
    ml_top_features: List[tuple] = field(default_factory=list)
    
    # Rule-specific (empty for ML-only)
    rule_score: int = 0
    rule_warnings: List[str] = field(default_factory=list)
    
    # Stories (from rules or generated)
    whale_story: str = ""
    position_story: str = ""
    oi_story: str = ""
    conclusion: str = ""
    
    # Metadata
    mode_used: str = ""         # Which mode produced this result
    is_ml_available: bool = False
    is_fallback: bool = False


class AnalysisRouter:
    """
    Routes analysis to the appropriate engine based on selected mode.
    
    Provides a unified interface regardless of which engine is used.
    """
    
    def __init__(self, mode: AnalysisMode = AnalysisMode.RULE_BASED):
        self.mode = mode
        self._ml_engine = None
        self._hybrid_engine = None
        self._ml_available = False
        
        # Try to load ML engines
        self._load_ml_engines()
    
    def _load_ml_engines(self):
        """Load ML and Hybrid engines if available"""
        try:
            from ml.ml_engine import MLEngine
            from ml.hybrid_engine import HybridEngine
            
            self._ml_engine = MLEngine()
            self._hybrid_engine = HybridEngine()
            self._ml_available = self._ml_engine.is_model_loaded()
        except ImportError as e:
            print(f"ML engines not available: {e}")
            self._ml_available = False
        except Exception as e:
            print(f"Error loading ML engines: {e}")
            self._ml_available = False
    
    def is_ml_available(self) -> bool:
        """Check if ML models are loaded and ready"""
        return self._ml_available
    
    def get_effective_mode(self) -> AnalysisMode:
        """
        Get the mode that will actually be used.
        Falls back to RULE_BASED if ML not available.
        """
        if self.mode in [AnalysisMode.ML, AnalysisMode.HYBRID] and not self._ml_available:
            return AnalysisMode.RULE_BASED
        return self.mode
    
    def analyze(
        self,
        symbol: str,
        current_price: float,
        whale_pct: float,
        retail_pct: float,
        oi_change: float = 0,
        price_change: float = 0,
        position_pct: float = 50,
        ta_score: float = 50,
        money_flow_phase: str = "UNKNOWN",
        # SMC data
        at_bullish_ob: bool = False,
        at_bearish_ob: bool = False,
        near_bullish_ob: bool = False,
        near_bearish_ob: bool = False,
        at_support: bool = False,
        at_resistance: bool = False,
        # Market conditions
        fear_greed: int = 50,
        btc_change_24h: float = 0,
        btc_correlation: float = 0,
        # Historical
        historical_win_rate: float = None,
        historical_sample_size: int = 0,
        # Trading params
        timeframe: str = "15m",
        trading_mode: str = "day_trade",
    ) -> UnifiedAnalysisResult:
        """
        Main analysis method - routes to appropriate engine.
        
        Returns:
            UnifiedAnalysisResult with consistent format
        """
        
        effective_mode = self.get_effective_mode()
        
        if effective_mode == AnalysisMode.RULE_BASED:
            return self._analyze_rule_based(
                symbol=symbol,
                current_price=current_price,
                whale_pct=whale_pct,
                retail_pct=retail_pct,
                oi_change=oi_change,
                price_change=price_change,
                position_pct=position_pct,
                ta_score=ta_score,
                money_flow_phase=money_flow_phase,
                at_bullish_ob=at_bullish_ob,
                at_bearish_ob=at_bearish_ob,
                near_bullish_ob=near_bullish_ob,
                near_bearish_ob=near_bearish_ob,
                at_support=at_support,
                at_resistance=at_resistance,
                fear_greed=fear_greed,
                btc_change_24h=btc_change_24h,
                historical_win_rate=historical_win_rate,
                historical_sample_size=historical_sample_size,
            )
        
        elif effective_mode == AnalysisMode.ML:
            return self._analyze_ml(
                symbol=symbol,
                current_price=current_price,
                whale_pct=whale_pct,
                retail_pct=retail_pct,
                oi_change=oi_change,
                price_change=price_change,
                position_pct=position_pct,
                ta_score=ta_score,
                money_flow_phase=money_flow_phase,
                btc_correlation=btc_correlation,
                timeframe=timeframe,
                trading_mode=trading_mode,
            )
        
        else:  # HYBRID
            return self._analyze_hybrid(
                symbol=symbol,
                current_price=current_price,
                whale_pct=whale_pct,
                retail_pct=retail_pct,
                oi_change=oi_change,
                price_change=price_change,
                position_pct=position_pct,
                ta_score=ta_score,
                money_flow_phase=money_flow_phase,
                at_bullish_ob=at_bullish_ob,
                at_bearish_ob=at_bearish_ob,
                near_bullish_ob=near_bullish_ob,
                near_bearish_ob=near_bearish_ob,
                at_support=at_support,
                at_resistance=at_resistance,
                fear_greed=fear_greed,
                btc_change_24h=btc_change_24h,
                btc_correlation=btc_correlation,
                historical_win_rate=historical_win_rate,
                historical_sample_size=historical_sample_size,
                timeframe=timeframe,
                trading_mode=trading_mode,
            )
    
    def _analyze_rule_based(self, **kwargs) -> UnifiedAnalysisResult:
        """Run rule-based analysis using MASTER_RULES"""
        try:
            from core.MASTER_RULES import get_trade_decision
            
            decision = get_trade_decision(
                whale_pct=kwargs['whale_pct'],
                retail_pct=kwargs['retail_pct'],
                oi_change=kwargs.get('oi_change', 0),
                price_change=kwargs.get('price_change', 0),
                position_pct=kwargs.get('position_pct', 50),
                ta_score=kwargs.get('ta_score', 50),
                money_flow_phase=kwargs.get('money_flow_phase', 'UNKNOWN'),
                fear_greed=kwargs.get('fear_greed', 50),
                btc_change_24h=kwargs.get('btc_change_24h', 0),
                historical_win_rate=kwargs.get('historical_win_rate'),
                historical_sample_size=kwargs.get('historical_sample_size', 0),
                at_bullish_ob=kwargs.get('at_bullish_ob', False),
                at_bearish_ob=kwargs.get('at_bearish_ob', False),
                near_bullish_ob=kwargs.get('near_bullish_ob', False),
                near_bearish_ob=kwargs.get('near_bearish_ob', False),
                at_support=kwargs.get('at_support', False),
                at_resistance=kwargs.get('at_resistance', False),
            )
            
            return UnifiedAnalysisResult(
                direction=decision.direction,
                confidence=decision.total_score,
                action=decision.action,
                trade_direction=decision.trade_direction,
                direction_score=decision.direction_score,
                squeeze_score=decision.squeeze_score,
                entry_score=decision.entry_score,
                rule_score=decision.total_score,
                rule_warnings=decision.warnings,
                whale_story=decision.whale_story,
                position_story=decision.position_story,
                oi_story=decision.oi_story,
                conclusion=decision.conclusion,
                mode_used="RULE_BASED",
                is_ml_available=self._ml_available,
                is_fallback=False,
            )
            
        except Exception as e:
            print(f"Rule-based analysis error: {e}")
            return self._get_fallback_result(f"Rule error: {e}")
    
    def _analyze_ml(self, **kwargs) -> UnifiedAnalysisResult:
        """Run ML-only analysis"""
        if not self._ml_available or self._ml_engine is None:
            return self._analyze_rule_based(**kwargs)
        
        try:
            prediction = self._ml_engine.predict(
                whale_pct=kwargs['whale_pct'],
                retail_pct=kwargs['retail_pct'],
                oi_change=kwargs.get('oi_change', 0),
                price_change=kwargs.get('price_change', 0),
                position_pct=kwargs.get('position_pct', 50),
                ta_score=kwargs.get('ta_score', 50),
                btc_correlation=kwargs.get('btc_correlation', 0),
                timeframe=kwargs.get('timeframe', '15m'),
            )
            
            current_price = kwargs.get('current_price', 0)
            
            # Calculate price levels from ML percentages
            if prediction.direction == 'LONG':
                tp1_price = current_price * (1 + prediction.optimal_tp_pct / 100)
                sl_price = current_price * (1 - prediction.optimal_sl_pct / 100)
            elif prediction.direction == 'SHORT':
                tp1_price = current_price * (1 - prediction.optimal_tp_pct / 100)
                sl_price = current_price * (1 + prediction.optimal_sl_pct / 100)
            else:
                tp1_price = current_price
                sl_price = current_price
            
            # Convert ML action
            if prediction.confidence >= 0.80:
                action = f"STRONG {'BUY' if prediction.direction == 'LONG' else 'SELL' if prediction.direction == 'SHORT' else 'WAIT'}"
            elif prediction.confidence >= 0.65:
                action = f"{'BUY' if prediction.direction == 'LONG' else 'SELL' if prediction.direction == 'SHORT' else 'WAIT'}"
            else:
                action = "WAIT"
            
            # Generate ML-based stories
            whale_story = f"🤖 ML sees whale {kwargs['whale_pct']:.0f}% vs retail {kwargs['retail_pct']:.0f}%"
            position_story = f"🤖 Position {kwargs.get('position_pct', 50):.0f}% - ML confidence: {prediction.confidence*100:.0f}%"
            oi_story = f"🤖 OI {kwargs.get('oi_change', 0):+.1f}% | Price {kwargs.get('price_change', 0):+.1f}%"
            
            conclusion = f"🤖 ML predicts {prediction.direction} with {prediction.confidence*100:.0f}% confidence. Expected move: {prediction.predicted_move_pct:+.1f}%"
            
            return UnifiedAnalysisResult(
                direction=prediction.direction,
                confidence=prediction.confidence * 100,
                action=action,
                trade_direction=prediction.direction,
                entry_price=current_price,
                tp1_price=tp1_price,
                tp1_pct=prediction.optimal_tp_pct,
                sl_price=sl_price,
                sl_pct=prediction.optimal_sl_pct,
                risk_reward=prediction.risk_reward,
                ml_confidence=prediction.confidence,
                ml_predicted_move=prediction.predicted_move_pct,
                similar_trades_count=prediction.similar_trades_count,
                similar_win_rate=prediction.similar_win_rate,
                ml_top_features=prediction.top_features,
                whale_story=whale_story,
                position_story=position_story,
                oi_story=oi_story,
                conclusion=conclusion,
                mode_used="ML",
                is_ml_available=True,
                is_fallback=prediction.is_fallback,
            )
            
        except Exception as e:
            print(f"ML analysis error: {e}")
            # Fall back to rule-based
            result = self._analyze_rule_based(**kwargs)
            result.mode_used = "ML (fallback to rules)"
            result.is_fallback = True
            return result
    
    def _analyze_hybrid(self, **kwargs) -> UnifiedAnalysisResult:
        """Run hybrid analysis combining ML + Rules"""
        if not self._ml_available or self._hybrid_engine is None:
            return self._analyze_rule_based(**kwargs)
        
        try:
            hybrid_result = self._hybrid_engine.analyze(
                whale_pct=kwargs['whale_pct'],
                retail_pct=kwargs['retail_pct'],
                oi_change=kwargs.get('oi_change', 0),
                price_change=kwargs.get('price_change', 0),
                position_pct=kwargs.get('position_pct', 50),
                ta_score=kwargs.get('ta_score', 50),
                money_flow_phase=kwargs.get('money_flow_phase', 'UNKNOWN'),
                fear_greed=kwargs.get('fear_greed', 50),
                btc_change_24h=kwargs.get('btc_change_24h', 0),
                btc_correlation=kwargs.get('btc_correlation', 0),
                historical_win_rate=kwargs.get('historical_win_rate'),
                historical_sample_size=kwargs.get('historical_sample_size', 0),
                at_bullish_ob=kwargs.get('at_bullish_ob', False),
                at_bearish_ob=kwargs.get('at_bearish_ob', False),
                near_bullish_ob=kwargs.get('near_bullish_ob', False),
                near_bearish_ob=kwargs.get('near_bearish_ob', False),
                at_support=kwargs.get('at_support', False),
                at_resistance=kwargs.get('at_resistance', False),
                current_price=kwargs.get('current_price', 0),
                timeframe=kwargs.get('timeframe', '15m'),
            )
            
            return UnifiedAnalysisResult(
                direction=hybrid_result.direction,
                confidence=hybrid_result.confidence,
                action=hybrid_result.action,
                trade_direction=hybrid_result.direction,
                entry_price=hybrid_result.entry_price,
                tp1_price=hybrid_result.tp1_price,
                tp1_pct=hybrid_result.tp1_pct,
                sl_price=hybrid_result.sl_price,
                sl_pct=hybrid_result.sl_pct,
                risk_reward=hybrid_result.risk_reward,
                ml_confidence=hybrid_result.ml_confidence,
                ml_predicted_move=hybrid_result.ml_predicted_move,
                similar_trades_count=hybrid_result.similar_trades_count,
                similar_win_rate=hybrid_result.similar_win_rate,
                rule_score=hybrid_result.rule_score,
                rule_warnings=hybrid_result.rule_warnings,
                whale_story=hybrid_result.whale_story,
                position_story=hybrid_result.position_story,
                oi_story=hybrid_result.oi_story,
                conclusion=hybrid_result.conclusion,
                mode_used=f"HYBRID ({hybrid_result.source})",
                is_ml_available=True,
                is_fallback=False,
            )
            
        except Exception as e:
            print(f"Hybrid analysis error: {e}")
            result = self._analyze_rule_based(**kwargs)
            result.mode_used = "HYBRID (fallback to rules)"
            result.is_fallback = True
            return result
    
    def _get_fallback_result(self, error_msg: str) -> UnifiedAnalysisResult:
        """Return a safe fallback result on errors"""
        return UnifiedAnalysisResult(
            direction="WAIT",
            confidence=0,
            action="ERROR",
            trade_direction="WAIT",
            conclusion=f"⚠️ Analysis error: {error_msg}",
            mode_used="FALLBACK",
            is_ml_available=self._ml_available,
            is_fallback=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_analysis_mode() -> AnalysisMode:
    """Get current analysis mode from session state"""
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = AnalysisMode.RULE_BASED
    return st.session_state.analysis_mode


def set_analysis_mode(mode: AnalysisMode):
    """Set analysis mode in session state"""
    st.session_state.analysis_mode = mode


def get_router() -> AnalysisRouter:
    """Get or create analysis router with current mode"""
    mode = get_analysis_mode()
    
    # Cache router in session state
    if 'analysis_router' not in st.session_state:
        st.session_state.analysis_router = AnalysisRouter(mode=mode)
    
    # Update mode if changed
    if st.session_state.analysis_router.mode != mode:
        st.session_state.analysis_router.mode = mode
    
    return st.session_state.analysis_router


def render_mode_toggle(location: str = "sidebar") -> AnalysisMode:
    """
    Render the analysis mode toggle UI.
    
    Args:
        location: Where to render ("sidebar" or "main")
        
    Returns:
        Currently selected mode
    """
    current_mode = get_analysis_mode()
    router = get_router()
    ml_available = router.is_ml_available()
    
    # Mode descriptions
    mode_info = {
        AnalysisMode.RULE_BASED: {
            "label": "📊 Rule-Based",
            "desc": "Current system (MASTER_RULES)",
            "icon": "📊"
        },
        AnalysisMode.ML: {
            "label": "🤖 ML Model",
            "desc": "Predictive ML with optimal TP/SL",
            "icon": "🤖"
        },
        AnalysisMode.HYBRID: {
            "label": "⚡ Hybrid",
            "desc": "Best of ML + Rules combined",
            "icon": "⚡"
        }
    }
    
    container = st.sidebar if location == "sidebar" else st
    
    with container:
        st.markdown("### 🎛️ Analysis Engine")
        
        # Radio buttons for mode selection
        mode_options = list(AnalysisMode)
        mode_labels = [mode_info[m]["label"] for m in mode_options]
        
        current_index = mode_options.index(current_mode)
        
        selected_index = st.radio(
            "Select Mode:",
            range(len(mode_options)),
            index=current_index,
            format_func=lambda i: mode_labels[i],
            key="analysis_mode_radio",
            horizontal=True,
        )
        
        selected_mode = mode_options[selected_index]
        
        # Show ML availability status
        if not ml_available and selected_mode in [AnalysisMode.ML, AnalysisMode.HYBRID]:
            st.warning("⚠️ ML model not trained yet. Using Rule-Based as fallback.")
            st.caption("Train the model in Settings → ML Training")
        
        # Show mode description
        st.caption(mode_info[selected_mode]["desc"])
        
        # Update session state if changed
        if selected_mode != current_mode:
            set_analysis_mode(selected_mode)
        
        return selected_mode


def render_mode_badge(mode: AnalysisMode) -> str:
    """Return HTML badge for current mode"""
    badges = {
        AnalysisMode.RULE_BASED: '<span style="background: #2196F3; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">📊 Rules</span>',
        AnalysisMode.ML: '<span style="background: #9C27B0; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">🤖 ML</span>',
        AnalysisMode.HYBRID: '<span style="background: #FF9800; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">⚡ Hybrid</span>',
    }
    return badges.get(mode, "")
