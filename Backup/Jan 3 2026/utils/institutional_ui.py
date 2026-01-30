"""
🏦 INSTITUTIONAL ANALYSIS UI COMPONENTS
========================================
Streamlit components for rendering institutional analysis with full education.
Used across Scanner, Monitor, and Single Analysis tabs.

Author: InvestorIQ
"""

import streamlit as st
from typing import Dict, List, Optional

# Import the institutional engine
from core.institutional_engine import (
    InstitutionalAnalysis, MetricReading, EDUCATION, PREMIUM_DATA_SOURCES,
    get_institutional_analysis, render_metric_with_education
)


def render_institutional_verdict(analysis: InstitutionalAnalysis, compact: bool = False):
    """
    Render the main institutional verdict banner.
    
    Args:
        analysis: InstitutionalAnalysis object
        compact: If True, render a smaller version for scanner cards
    """
    verdict_colors = {
        'BULLISH': '#00d4aa',
        'BEARISH': '#ff4444',
        'NEUTRAL': '#888888',
        'UNKNOWN': '#666666'
    }
    
    v_color = verdict_colors.get(analysis.verdict, '#888888')
    
    if compact:
        # Compact version for scanner cards
        st.markdown(f"""
        <div style='background: {v_color}22; border: 1px solid {v_color}; border-radius: 8px; 
                    padding: 8px 12px; margin: 5px 0; display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #fff; font-size: 0.9em;'>🏦 Institutional: <strong style='color: {v_color};'>{analysis.verdict}</strong></span>
            <span style='background: {v_color}44; color: {v_color}; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;'>
                {analysis.confidence}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Full version
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {v_color}22, {v_color}11); 
                    border: 2px solid {v_color}; border-radius: 12px; padding: 15px; margin-bottom: 15px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: #fff; font-size: 1.2em; font-weight: bold;'>🏦 Institutional Verdict</span>
                <span style='background: {v_color}; color: #000; padding: 8px 20px; border-radius: 20px; 
                             font-weight: bold; font-size: 1.1em;'>
                    {analysis.verdict} ({analysis.confidence}%)
                </span>
            </div>
            <div style='color: #ccc; margin-top: 10px; font-size: 0.95em;'>
                {analysis.action}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_metric_card(metric: MetricReading, show_education: bool = True):
    """
    Render a single metric with optional educational tooltip.
    """
    # Color based on bias
    if metric.bias == 'BULLISH':
        bg_color = '#0a2a1a'
        border_color = '#00d4aa'
    elif metric.bias == 'BEARISH':
        bg_color = '#2a0a0a'
        border_color = '#ff4444'
    else:
        bg_color = '#1a1a2e'
        border_color = '#333'
    
    # Icon based on strength
    strength_indicator = '●●●' if metric.strength == 'STRONG' else '●●○' if metric.strength == 'MODERATE' else '●○○'
    
    st.markdown(f"""
    <div style='background: {bg_color}; padding: 15px; border-radius: 10px; 
                border: 1px solid {border_color}; margin: 8px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
            <span style='color: #fff; font-weight: bold;'>{metric.icon} {metric.name}</span>
            <span style='color: {metric.color}; font-size: 1.2em; font-weight: bold;'>{metric.display_value}</span>
        </div>
        <div style='color: #ccc; font-size: 0.9em; line-height: 1.5;'>
            {metric.interpretation}
        </div>
        <div style='display: flex; justify-content: space-between; margin-top: 10px; padding-top: 8px; border-top: 1px solid #333;'>
            <span style='color: #666; font-size: 0.8em;'>Strength: {strength_indicator}</span>
            <span style='color: {metric.color}; font-size: 0.8em;'>Impact: {metric.score_impact:+d} pts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if show_education and metric.education:
        with st.expander(f"📚 Learn: {metric.name}", expanded=False):
            render_metric_education(metric.education)


def render_metric_education(education: Dict):
    """
    Render full educational content for a metric.
    """
    st.markdown(f"""
    <div style='background: #0d1117; padding: 15px; border-radius: 8px; border: 1px solid #30363d;'>
        <h4 style='color: #58a6ff; margin-top: 0;'>📖 What is it?</h4>
        <p style='color: #c9d1d9; line-height: 1.6;'>{education.get('what', 'N/A')}</p>
        
        <h4 style='color: #58a6ff;'>💡 Why it matters</h4>
        <p style='color: #c9d1d9; line-height: 1.6;'>{education.get('why_matters', 'N/A')}</p>
        
        <h4 style='color: #58a6ff;'>🎯 Trading Edge</h4>
        <p style='color: #79c0ff; line-height: 1.6; font-style: italic;'>{education.get('edge', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # How to read - show specific interpretations
    how_to_read = education.get('how_to_read', {})
    if how_to_read:
        st.markdown("#### 📊 How to Read")
        for scenario, interpretation in how_to_read.items():
            # Format the scenario name
            scenario_display = scenario.replace('_', ' ').title()
            st.markdown(f"**{scenario_display}:** {interpretation}")
    
    # Normal range
    if education.get('normal_range'):
        st.info(f"📏 **Normal Range:** {education.get('normal_range')}")
    
    # Data source
    if education.get('data_source'):
        st.caption(f"📡 Data Source: {education.get('data_source')}")
    
    # Premium note
    if education.get('premium_note'):
        st.warning(education.get('premium_note'))


def render_key_signals(signals: List[str]):
    """
    Render the key institutional signals as a highlighted list.
    """
    if not signals:
        st.info("No strong institutional signals detected")
        return
    
    st.markdown("### 🎯 Key Institutional Signals")
    
    for i, signal in enumerate(signals):
        # Determine signal type for styling
        if '🟢' in signal or 'LONG' in signal or 'BULLISH' in signal or 'BUYING' in signal:
            bg_color = '#0a2a1a'
            border_color = '#00d4aa'
        elif '🔴' in signal or 'SHORT' in signal or 'BEARISH' in signal or 'SELLING' in signal:
            bg_color = '#2a0a0a'
            border_color = '#ff4444'
        elif '⚠️' in signal or 'CONTRARIAN' in signal:
            bg_color = '#2a2a1a'
            border_color = '#ffcc00'
        else:
            bg_color = '#1a1a2e'
            border_color = '#333'
        
        priority = "PRIMARY" if i == 0 else "SECONDARY"
        
        st.markdown(f"""
        <div style='background: {bg_color}; padding: 12px 15px; border-radius: 8px; 
                    border-left: 4px solid {border_color}; margin: 8px 0;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='color: #fff; font-size: 1em;'>{signal}</span>
                <span style='color: #666; font-size: 0.75em; background: #1a1a2e; padding: 2px 8px; border-radius: 4px;'>
                    {priority}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_metrics_grid(metrics: List[MetricReading], columns: int = 2):
    """
    Render metrics in a grid layout with education expandable.
    """
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(metric, show_education=True)


def render_full_institutional_analysis(symbol: str, market_type: str = None, show_education: bool = True):
    """
    Complete institutional analysis rendering for a symbol.
    This is the main function to use in app.py
    
    Args:
        symbol: Trading symbol
        market_type: 'CRYPTO', 'STOCK', or 'ETF' (auto-detected if None)
        show_education: Whether to show educational content
    
    Returns:
        InstitutionalAnalysis object for integration with main signal
    """
    with st.spinner("Loading institutional data..."):
        analysis = get_institutional_analysis(symbol, market_type)
    
    if analysis.verdict == 'UNKNOWN':
        st.warning("🏦 Could not fetch institutional data. API may be unavailable.")
        return analysis
    
    # Market type badge
    market_badge_colors = {
        'CRYPTO': '#f7931a',
        'STOCK': '#00d4ff',
        'ETF': '#9d4edd'
    }
    badge_color = market_badge_colors.get(analysis.market_type, '#888')
    
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
        <span style='background: {badge_color}; color: #000; padding: 4px 12px; border-radius: 15px; 
                     font-weight: bold; font-size: 0.85em;'>{analysis.market_type}</span>
        <span style='color: #888; font-size: 0.85em;'>
            Data: {', '.join(analysis.data_sources)}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main verdict
    render_institutional_verdict(analysis)
    
    # Key signals
    if analysis.key_signals:
        render_key_signals(analysis.key_signals)
    
    # Action reasoning
    if analysis.action_reasoning:
        st.markdown(f"""
        <div style='background: #1a1a2e; padding: 12px 15px; border-radius: 8px; margin: 15px 0;'>
            <span style='color: #888;'>📊 Analysis Summary:</span>
            <span style='color: #fff;'> {analysis.action_reasoning}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics with education
    st.markdown("### 📊 Detailed Metrics")
    
    if analysis.metrics:
        # Use tabs for cleaner organization
        if analysis.market_type == 'CRYPTO':
            tab_names = ["📈 OI & Funding", "🐋 Positioning", "📚 Learn More"]
        else:
            tab_names = ["👔 Insiders & Shorts", "📊 Options & Institutions", "📚 Learn More"]
        
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            # First 2-3 metrics
            for metric in analysis.metrics[:3]:
                render_metric_card(metric, show_education=show_education)
        
        with tabs[1]:
            # Remaining metrics
            for metric in analysis.metrics[3:]:
                render_metric_card(metric, show_education=show_education)
        
        with tabs[2]:
            render_educational_overview(analysis.market_type)
    
    # Premium suggestion
    if analysis.premium_suggestion:
        st.markdown("---")
        st.info(analysis.premium_suggestion)
    
    return analysis


def render_compact_institutional(symbol: str, market_type: str = None):
    """
    Compact version for scanner cards - just verdict + main signal.
    """
    try:
        analysis = get_institutional_analysis(symbol, market_type)
        
        if analysis.verdict == 'UNKNOWN':
            return None
        
        render_institutional_verdict(analysis, compact=True)
        
        # Show just the top signal
        if analysis.key_signals:
            st.markdown(f"""
            <div style='color: #888; font-size: 0.85em; padding: 5px 0;'>
                💡 {analysis.key_signals[0]}
            </div>
            """, unsafe_allow_html=True)
        
        return analysis
    except:
        return None


def render_educational_overview(market_type: str):
    """
    Render a comprehensive educational overview for the market type.
    """
    st.markdown("### 📚 Understanding Institutional Analysis")
    
    if market_type == 'CRYPTO':
        st.markdown("""
        <div style='background: #0d1117; padding: 20px; border-radius: 10px; border: 1px solid #30363d;'>
            <h4 style='color: #58a6ff; margin-top: 0;'>🐋 Why Follow Whales?</h4>
            <p style='color: #c9d1d9; line-height: 1.8;'>
                In crypto futures, "whales" and "top traders" have consistently profitable track records.
                Binance tracks these traders separately from retail. When smart money positions differently
                from retail, they're usually right. <strong style='color: #79c0ff;'>Retail is often the exit liquidity.</strong>
            </p>
            
            <h4 style='color: #58a6ff;'>🔑 Key Principles</h4>
            <ul style='color: #c9d1d9; line-height: 2;'>
                <li><strong>Follow the whales</strong> - Their positioning predicts direction</li>
                <li><strong>Fade extreme retail</strong> - When retail is 65%+ one way, expect reversal</li>
                <li><strong>OI + Price tells the story</strong> - Is the move driven by new money or liquidations?</li>
                <li><strong>Extreme funding reverses</strong> - Overleveraged sides get liquidated</li>
            </ul>
            
            <h4 style='color: #58a6ff;'>⚠️ Important Notes</h4>
            <p style='color: #8b949e; line-height: 1.6;'>
                • Data updates every few minutes<br>
                • Works best in trending markets<br>
                • Always combine with technical analysis<br>
                • Whale data is ONE factor, not the only factor
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # STOCK or ETF
        st.markdown("""
        <div style='background: #0d1117; padding: 20px; border-radius: 10px; border: 1px solid #30363d;'>
            <h4 style='color: #58a6ff; margin-top: 0;'>🏦 Why Follow Institutions?</h4>
            <p style='color: #c9d1d9; line-height: 1.8;'>
                Institutional investors (funds, banks, pensions) manage trillions. They have research teams,
                insider access, and move markets. When insiders buy their own stock, they know something.
                <strong style='color: #79c0ff;'>Follow the money.</strong>
            </p>
            
            <h4 style='color: #58a6ff;'>🔑 Key Principles</h4>
            <ul style='color: #c9d1d9; line-height: 2;'>
                <li><strong>Insider buying is gold</strong> - Executives buying their own stock = confidence</li>
                <li><strong>High short interest = squeeze risk</strong> - Can fuel violent rallies</li>
                <li><strong>Options flow shows smart money bets</strong> - Large unusual activity may be informed</li>
                <li><strong>Institutional ownership = validation</strong> - But watch for changes</li>
            </ul>
            
            <h4 style='color: #58a6ff;'>⚠️ Data Limitations (Free Sources)</h4>
            <p style='color: #8b949e; line-height: 1.6;'>
                • Insider trades: Real-time from SEC (best free data)<br>
                • Short interest: 2-week delay from FINRA<br>
                • 13F filings: 45-day delay (quarterly)<br>
                • Options flow: Basic only - premium services show real-time unusual activity
            </p>
            
            <h4 style='color: #58a6ff;'>💎 Premium Data Options</h4>
            <p style='color: #c9d1d9; line-height: 1.6;'>
                For real-time options flow, dark pool data, and unusual activity:
            </p>
            <ul style='color: #79c0ff; line-height: 1.8;'>
                <li>Unusual Whales ($39-149/mo) - Best for options flow</li>
                <li>FlowAlgo ($99/mo) - Dark pool + options</li>
                <li>Cheddar Flow ($49-99/mo) - Clean UI, alerts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_institutional_integration_note(analysis: InstitutionalAnalysis, technical_verdict: str):
    """
    Show how institutional data integrates with technical analysis.
    Helps users understand when signals agree or conflict.
    """
    # Determine alignment
    tech_bullish = technical_verdict in ['BULLISH', 'LONG', 'BUY']
    inst_bullish = analysis.verdict == 'BULLISH'
    tech_bearish = technical_verdict in ['BEARISH', 'SHORT', 'SELL']
    inst_bearish = analysis.verdict == 'BEARISH'
    
    if (tech_bullish and inst_bullish) or (tech_bearish and inst_bearish):
        # Aligned
        st.markdown(f"""
        <div style='background: #0a2a1a; border: 2px solid #00d4aa; border-radius: 10px; padding: 15px; margin: 15px 0;'>
            <div style='color: #00d4aa; font-weight: bold; font-size: 1.1em; margin-bottom: 8px;'>
                ✅ SIGNALS ALIGNED - High Confidence Setup
            </div>
            <div style='color: #ccc; line-height: 1.6;'>
                Technical analysis and institutional data both point {analysis.verdict}.<br>
                <span style='color: #888;'>This alignment increases probability of success.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif (tech_bullish and inst_bearish) or (tech_bearish and inst_bullish):
        # Conflicting
        st.markdown(f"""
        <div style='background: #2a2a1a; border: 2px solid #ffcc00; border-radius: 10px; padding: 15px; margin: 15px 0;'>
            <div style='color: #ffcc00; font-weight: bold; font-size: 1.1em; margin-bottom: 8px;'>
                ⚠️ SIGNALS CONFLICTING - Proceed with Caution
            </div>
            <div style='color: #ccc; line-height: 1.6;'>
                Technical: <strong>{technical_verdict}</strong> | Institutional: <strong>{analysis.verdict}</strong><br>
                <span style='color: #888;'>Consider waiting for alignment or reducing position size.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Neutral institutional
        st.markdown(f"""
        <div style='background: #1a1a2e; border: 1px solid #333; border-radius: 10px; padding: 15px; margin: 15px 0;'>
            <div style='color: #888; font-weight: bold; font-size: 1.1em; margin-bottom: 8px;'>
                ⚪ INSTITUTIONAL NEUTRAL - Rely on Technicals
            </div>
            <div style='color: #ccc; line-height: 1.6;'>
                No strong institutional edge detected. Technical analysis is the primary signal.<br>
                <span style='color: #888;'>Watch for institutional signals to develop.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def get_score_adjustment_display(analysis: InstitutionalAnalysis) -> str:
    """
    Get a display string for how institutional data affects the score.
    """
    adj = analysis.signal_adjustment
    
    if adj > 10:
        return f"📈 Strong institutional boost (+{adj})"
    elif adj > 0:
        return f"📈 Institutional support (+{adj})"
    elif adj < -10:
        return f"📉 Strong institutional headwind ({adj})"
    elif adj < 0:
        return f"📉 Institutional caution ({adj})"
    else:
        return "⚪ No institutional adjustment"
