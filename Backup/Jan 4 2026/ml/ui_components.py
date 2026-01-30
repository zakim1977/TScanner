"""
ML Toggle UI Component
=======================

Provides UI elements for switching between Rule-Based, ML, and Hybrid analysis modes.
"""

import streamlit as st
from typing import Tuple


def render_analysis_engine_toggle(section: str = "scanner") -> str:
    """
    Render the analysis engine toggle.
    
    Args:
        section: Which section this is for (scanner, single_analysis, trade_monitor)
    
    Returns:
        Selected engine mode: 'rules', 'ml', or 'hybrid'
    """
    
    # Get current mode from session state
    current_mode = st.session_state.get('analysis_engine', 'rules')
    
    # Mode options with descriptions
    modes = {
        'rules': ('📋 Rule-Based', 'MASTER_RULES scoring - interpretable, no training needed'),
        'ml': ('🤖 ML Model', 'Machine Learning predictions - learns from outcomes'),
        'hybrid': ('⚡ Hybrid', 'Combines both - highest accuracy when they agree'),
    }
    
    # Check if ML is available
    try:
        from ml.ml_engine import is_ml_available
        ml_available = is_ml_available()
    except:
        ml_available = False
    
    # Render toggle
    st.markdown("""
    <style>
    .engine-toggle {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .engine-option {
        display: inline-block;
        padding: 8px 16px;
        margin: 2px;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .engine-option.active {
        background: #3b82f6;
        color: white;
    }
    .engine-option.inactive {
        background: #374151;
        color: #9ca3af;
    }
    .engine-option.disabled {
        background: #1f2937;
        color: #4b5563;
        cursor: not-allowed;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button(
            "📋 Rules",
            key=f"{section}_rules_btn",
            type="primary" if current_mode == 'rules' else "secondary",
            use_container_width=True
        ):
            st.session_state.analysis_engine = 'rules'
            st.rerun()
    
    with col2:
        if ml_available:
            if st.button(
                "🤖 ML",
                key=f"{section}_ml_btn",
                type="primary" if current_mode == 'ml' else "secondary",
                use_container_width=True
            ):
                st.session_state.analysis_engine = 'ml'
                st.rerun()
        else:
            st.button(
                "🤖 ML (Train First)",
                key=f"{section}_ml_btn_disabled",
                disabled=True,
                use_container_width=True
            )
    
    with col3:
        if ml_available:
            if st.button(
                "⚡ Hybrid",
                key=f"{section}_hybrid_btn",
                type="primary" if current_mode == 'hybrid' else "secondary",
                use_container_width=True
            ):
                st.session_state.analysis_engine = 'hybrid'
                st.rerun()
        else:
            st.button(
                "⚡ Hybrid",
                key=f"{section}_hybrid_btn_disabled",
                disabled=True,
                use_container_width=True
            )
    
    with col4:
        # Show current mode description
        mode_label, mode_desc = modes.get(current_mode, modes['rules'])
        status = "✅ Active" if ml_available or current_mode == 'rules' else "⚠️ Train model first"
        st.markdown(f"""
        <div style='background: #0d0d1a; padding: 8px; border-radius: 6px;'>
            <span style='color: #3b82f6; font-weight: bold;'>{mode_label}</span>
            <span style='color: #888; font-size: 0.85em;'> | {status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    return current_mode


def render_ml_prediction_box(
    prediction,
    current_price: float = 0,
    show_levels: bool = True
) -> None:
    """
    Render ML prediction results in a styled box.
    
    Args:
        prediction: MLPrediction or HybridPrediction object
        current_price: Current price for level calculations
        show_levels: Whether to show TP/SL levels
    """
    
    # Determine colors based on direction
    if prediction.direction == 'LONG':
        dir_color = "#00ff88"
        dir_emoji = "🟢"
        bg_gradient = "linear-gradient(135deg, #0a2a0a 0%, #1a2a1a 100%)"
    elif prediction.direction == 'SHORT':
        dir_color = "#ff4444"
        dir_emoji = "🔴"
        bg_gradient = "linear-gradient(135deg, #2a0a0a 0%, #2a1a1a 100%)"
    else:
        dir_color = "#ffcc00"
        dir_emoji = "⚪"
        bg_gradient = "linear-gradient(135deg, #2a2a0a 0%, #1a1a1a 100%)"
    
    # Confidence color
    if prediction.confidence >= 75:
        conf_color = "#00ff88"
    elif prediction.confidence >= 60:
        conf_color = "#00d4aa"
    elif prediction.confidence >= 45:
        conf_color = "#ffcc00"
    else:
        conf_color = "#ff9500"
    
    # Main prediction box
    st.markdown(f"""
    <div style='background: {bg_gradient}; border: 2px solid {dir_color}; 
                border-radius: 12px; padding: 20px; margin-bottom: 15px;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 2em;'>{dir_emoji}</span>
                <span style='color: {dir_color}; font-size: 1.5em; font-weight: bold; margin-left: 10px;'>
                    {prediction.direction}
                </span>
            </div>
            <div style='text-align: right;'>
                <div style='color: {conf_color}; font-size: 2em; font-weight: bold;'>
                    {prediction.confidence:.0f}%
                </div>
                <div style='color: #888; font-size: 0.9em;'>ML Confidence</div>
            </div>
        </div>
        
        <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;'>
            <div style='color: #aaa; font-size: 0.9em;'>
                🧠 <strong>ML Reasoning:</strong> {prediction.reasoning}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Levels box (if showing)
    if show_levels and prediction.direction != 'WAIT':
        entry = current_price if current_price > 0 else getattr(prediction, 'entry', 0)
        
        if prediction.direction == 'LONG' and entry > 0:
            tp1 = entry * (1 + prediction.optimal_tp1_pct / 100)
            tp2 = entry * (1 + prediction.optimal_tp2_pct / 100)
            sl = entry * (1 - prediction.optimal_sl_pct / 100)
        elif prediction.direction == 'SHORT' and entry > 0:
            tp1 = entry * (1 - prediction.optimal_tp1_pct / 100)
            tp2 = entry * (1 - prediction.optimal_tp2_pct / 100)
            sl = entry * (1 + prediction.optimal_sl_pct / 100)
        else:
            tp1 = tp2 = sl = 0
        
        st.markdown(f"""
        <div style='background: #0d0d1a; border-radius: 10px; padding: 15px; margin-bottom: 15px;'>
            <div style='color: #888; font-size: 0.85em; margin-bottom: 10px;'>
                📊 ML-Optimized Levels (based on {getattr(prediction, 'similar_trades_count', 0)} similar trades)
            </div>
            <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;'>
                <div style='text-align: center;'>
                    <div style='color: #888; font-size: 0.8em;'>Entry</div>
                    <div style='color: #fff; font-weight: bold;'>${entry:,.4f}</div>
                </div>
                <div style='text-align: center;'>
                    <div style='color: #888; font-size: 0.8em;'>TP1 (+{prediction.optimal_tp1_pct:.1f}%)</div>
                    <div style='color: #00ff88; font-weight: bold;'>${tp1:,.4f}</div>
                </div>
                <div style='text-align: center;'>
                    <div style='color: #888; font-size: 0.8em;'>TP2 (+{prediction.optimal_tp2_pct:.1f}%)</div>
                    <div style='color: #00d4aa; font-weight: bold;'>${tp2:,.4f}</div>
                </div>
                <div style='text-align: center;'>
                    <div style='color: #888; font-size: 0.8em;'>SL (-{prediction.optimal_sl_pct:.1f}%)</div>
                    <div style='color: #ff6b6b; font-weight: bold;'>${sl:,.4f}</div>
                </div>
            </div>
            <div style='margin-top: 10px; text-align: center;'>
                <span style='color: #888;'>R:R Ratio: </span>
                <span style='color: {"#00ff88" if prediction.expected_rr >= 1.5 else "#ffcc00" if prediction.expected_rr >= 1 else "#ff6b6b"}; font-weight: bold;'>
                    {prediction.expected_rr:.2f}:1
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top features box
    if hasattr(prediction, 'top_features') and prediction.top_features:
        st.markdown("""
        <div style='background: #0d0d1a; border-radius: 10px; padding: 15px;'>
            <div style='color: #888; font-size: 0.85em; margin-bottom: 10px;'>
                📈 Top Contributing Factors
            </div>
        """, unsafe_allow_html=True)
        
        for name, value, importance in prediction.top_features[:5]:
            bar_width = min(100, importance * 100)
            st.markdown(f"""
            <div style='margin-bottom: 8px;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #aaa; font-size: 0.85em;'>{name}</span>
                    <span style='color: #fff; font-size: 0.85em;'>{value:.1f}</span>
                </div>
                <div style='background: #1a1a2e; border-radius: 4px; height: 6px; margin-top: 3px;'>
                    <div style='background: #3b82f6; width: {bar_width}%; height: 100%; border-radius: 4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_hybrid_comparison(hybrid_pred) -> None:
    """
    Render comparison between Rule-Based and ML predictions.
    """
    
    agree_color = "#00ff88" if hybrid_pred.engines_agree else "#ff9500"
    agree_text = "✅ AGREE" if hybrid_pred.engines_agree else "⚠️ CONFLICT"
    
    st.markdown(f"""
    <div style='background: #0d0d1a; border-radius: 10px; padding: 15px; margin-bottom: 15px;'>
        <div style='text-align: center; margin-bottom: 15px;'>
            <span style='color: {agree_color}; font-size: 1.2em; font-weight: bold;'>
                {agree_text}
            </span>
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div style='text-align: center; padding: 15px; background: #1a1a2e; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em;'>📋 Rule-Based</div>
                <div style='color: {"#00ff88" if hybrid_pred.rule_direction == "LONG" else "#ff6b6b" if hybrid_pred.rule_direction == "SHORT" else "#ffcc00"}; 
                            font-size: 1.3em; font-weight: bold; margin-top: 5px;'>
                    {hybrid_pred.rule_direction}
                </div>
                <div style='color: #aaa; font-size: 0.9em;'>Score: {hybrid_pred.rule_score:.0f}/100</div>
            </div>
            <div style='text-align: center; padding: 15px; background: #1a1a2e; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em;'>🤖 ML Model</div>
                <div style='color: {"#00ff88" if hybrid_pred.ml_direction == "LONG" else "#ff6b6b" if hybrid_pred.ml_direction == "SHORT" else "#ffcc00"}; 
                            font-size: 1.3em; font-weight: bold; margin-top: 5px;'>
                    {hybrid_pred.ml_direction}
                </div>
                <div style='color: #aaa; font-size: 0.9em;'>Confidence: {hybrid_pred.ml_confidence:.0f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_train_ml_button() -> bool:
    """
    Render button to train ML models.
    
    Returns:
        True if training was triggered
    """
    
    st.markdown("""
    <div style='background: #1a1a2e; border-radius: 10px; padding: 20px; margin: 15px 0;'>
        <div style='color: #888; margin-bottom: 10px;'>
            🤖 <strong>ML Models Not Trained</strong>
        </div>
        <div style='color: #aaa; font-size: 0.9em; margin-bottom: 15px;'>
            Train the ML model using your historical whale data and trade outcomes.
            This will enable ML and Hybrid analysis modes.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Train ML Models", type="primary", use_container_width=True):
        return True
    
    return False
