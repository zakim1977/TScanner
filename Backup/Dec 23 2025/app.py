"""
InvestorIQ - Smart Money Analysis Platform
Crypto • Stocks • ETFs | SMC-Based Technical Analysis
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import pytz

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS FROM MODULES
# ═══════════════════════════════════════════════════════════════════════════════

from config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON,
    TOP_CRYPTO_PAIRS, TRADING_MODES, REFRESH_OPTIONS,
    SIGNAL_THRESHOLDS, STRATEGY_EXPLANATIONS,
    get_timeframe_context,
    TOP_ETFS, TOP_STOCKS  # NEW: Stock/ETF lists
)

from styles import apply_custom_css

from core.data_fetcher import (
    fetch_ohlcv_smart, fetch_ohlcv_binance,
    fetch_binance_klines, fetch_binance_price, fetch_all_binance_usdt_pairs,
    fetch_binance_futures_pairs, get_default_futures_pairs,
    get_all_binance_pairs, get_current_price,
    # NEW: Stock/ETF support
    fetch_stock_data, get_stock_price, fetch_universal, 
    estimate_time_to_target as estimate_time
)

from core.signal_generator import SignalGenerator, TradeSignal, analyze_trend, analyze_momentum
from core.smc_detector import detect_smc, analyze_market_structure
from core.money_flow import calculate_money_flow, detect_whale_activity, detect_pre_breakout
from core.money_flow_context import get_money_flow_context, get_money_flow_education, MoneyFlowContext, MONEY_FLOW_EDUCATION
from core.level_calculator import calculate_smart_levels, get_all_levels, TradeLevels
from core.indicators import calculate_rsi, calculate_ema, calculate_atr, calculate_bbands, calculate_vwap

# Master Narrative Engine
from core.narrative_engine import MasterNarrative, analyze as narrative_analyze, AnalysisResult, Action, Sentiment

# Education Module - Unified learning across all modes
from core.education import (
    CONCEPT_DEFINITIONS, generate_price_story, get_concept_education,
    get_indicator_explanation, build_full_education_section
)

# Alert & Prediction System
from core.alert_system import (
    find_approaching_setups, check_trade_alerts,
    format_alert_html, format_predictive_signal_html,
    AlertType, SignalStage,
    # Institutional Activity Detection
    detect_stealth_accumulation, detect_stealth_distribution,
    detect_institutional_activity, format_institutional_activity_html
)

# 🐋 Whale & Institutional Data Module (NEW!)
from core.whale_institutional import (
    get_institutional_analysis as get_whale_analysis, get_whale_summary, format_institutional_html,
    interpret_oi_price, interpret_funding, interpret_long_short
)

# 🏦 Unified Institutional Engine (NEW - replaces scattered whale code)
from core.institutional_engine import (
    get_institutional_analysis, InstitutionalAnalysis, EDUCATION as INSTITUTIONAL_EDUCATION,
    integrate_institutional_into_signal, should_override_signal
)

# 🎨 Institutional UI Components
from utils.institutional_ui import (
    render_institutional_verdict, render_metric_card, render_key_signals,
    render_full_institutional_analysis, render_compact_institutional,
    render_institutional_integration_note, get_score_adjustment_display
)

# 📊 Stock Institutional Data (Quiver Quant)
from core.quiver_institutional import (
    get_stock_institutional_analysis, format_stock_institutional_html,
    get_congress_trades, get_insider_trades, get_short_interest
)

# 📊 Unified Scoring Engine - SINGLE SOURCE OF TRUTH for all scores
from core.unified_scoring import (
    calculate_unified_score, get_all_scores, calculate_predictive_score,
    TAStrength, WhaleVerdict, Confidence, FinalAction
)

from utils.formatters import (
    fmt_price, calc_roi, format_percentage, format_number,
    get_grade_emoji, get_grade_letter, get_quality_badge
)

from utils.trade_storage import (
    load_trade_history, save_trade_history, add_trade,
    get_active_trades, get_closed_trades, calculate_statistics,
    update_trade, close_trade, get_trade_by_symbol, sync_active_trades,
    delete_trade_by_symbol, export_trades_json, import_trades_json, get_download_link,
    load_user_settings, save_user_settings, get_saved_api_key, save_api_key
)

from utils.charts import create_trade_setup_chart, create_performance_chart

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="InvestIQ PRO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# ═══════════════════════════════════════════════════════════════════════════════
# PRO GRADE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

GRADE_CRITERIA = {
    'A+': {'min_score': 85, 'color': '#00ff88', 'emoji': '🔥', 'label': 'EXCEPTIONAL'},
    'A':  {'min_score': 70, 'color': '#00d4aa', 'emoji': '🟢', 'label': 'STRONG'},
    'B':  {'min_score': 55, 'color': '#ffcc00', 'emoji': '🟡', 'label': 'GOOD'},
    'C':  {'min_score': 40, 'color': '#ff9500', 'emoji': '🟠', 'label': 'MODERATE'},
    'D':  {'min_score': 0,  'color': '#ff4444', 'emoji': '🔴', 'label': 'WEAK'}
}

def get_pro_grade(score: int) -> dict:
    """Get professional grade info based on score"""
    for grade, info in GRADE_CRITERIA.items():
        if score >= info['min_score']:
            return {'grade': grade, **info}
    return {'grade': 'D', **GRADE_CRITERIA['D']}


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 COMBINED CONFIDENCE SCORE SYSTEM
# Separates Technical Analysis and Institutional scores for transparency
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_confidence_scores(
    signal, mf, smc, whale, pre_break, institutional,
    trade_mode: str = 'DayTrade',
    timeframe: str = '1h',
    oi_change: float = None,
    whale_pct: float = None
) -> dict:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    UNIFIED SCORING - SINGLE SOURCE OF TRUTH - NOW TIMEFRAME-AWARE!
    ═══════════════════════════════════════════════════════════════════════════════
    
    ARCHITECTURE:
    Layer 1: Raw Data (chart, API)
    Layer 2: TA Processor (this function calculates ta_score)
    Layer 3: Unified Scoring Engine (derives ALL scores from ta_score + whale verdict)
             → NOW WEIGHTED BY TRADE MODE!
    Layer 4: Display (uses output from Layer 3)
    
    TRADE MODE WEIGHTING:
    ┌─────────────┬───────────┬──────────────┬───────────────┐
    │ Mode        │ TA Weight │ Whale Weight │ Min OI Change │
    ├─────────────┼───────────┼──────────────┼───────────────┤
    │ Scalp       │    90%    │     10%      │     2.0%      │
    │ DayTrade    │    75%    │     25%      │     1.0%      │
    │ Swing       │    50%    │     50%      │     0.5%      │
    │ Investment  │    40%    │     60%      │     0.3%      │
    └─────────────┴───────────┴──────────────┴───────────────┘
    
    Args:
        signal, mf, smc, whale, pre_break, institutional: Analysis data
        trade_mode: User's selected trading mode ('Scalp', 'DayTrade', 'Swing', 'Investment')
        timeframe: User's selected timeframe ('1m', '5m', '15m', '1h', '4h', '1d', '1w')
        oi_change: OI change % (24h) - for validity check against mode threshold
        whale_pct: Top trader long % - for validity check against mode threshold
    
    Returns:
        Dict with all scores, properly weighted by trade mode
    ═══════════════════════════════════════════════════════════════════════════════
    """
    from core.unified_scoring import get_all_scores
    
    # ═══════════════════════════════════════════════════════════════════
    # LAYER 2: TA PROCESSOR - Calculate ta_score from chart patterns
    # ═══════════════════════════════════════════════════════════════════
    
    ta_score = 35  # Base score for having a signal
    ta_factors = []
    
    # Signal strength
    if signal and signal.confidence:
        signal_boost = min(25, signal.confidence // 4)  # Up to +25
        ta_score += signal_boost
        if signal.confidence >= 60:
            ta_factors.append(("Strong signal confidence", signal_boost))
    
    # Money Flow (OBV, CMF)
    if mf.get('is_accumulating'):
        ta_score += 15
        ta_factors.append(("Money flowing IN (accumulation)", 15))
    elif mf.get('is_distributing'):
        ta_score -= 10
        ta_factors.append(("Money flowing OUT (distribution)", -10))
    
    # CMF strength
    cmf = mf.get('cmf', 0)
    if cmf > 0.2:
        ta_score += 10
        ta_factors.append((f"Strong buying pressure (CMF: {cmf:.2f})", 10))
    elif cmf < -0.2:
        ta_score -= 10
        ta_factors.append((f"Strong selling pressure (CMF: {cmf:.2f})", -10))
    
    # Smart Money Concepts
    ob_data = smc.get('order_blocks', {})
    fvg_data = smc.get('fvg', {})
    
    if ob_data.get('at_bullish_ob'):
        ta_score += 15
        ta_factors.append(("At bullish Order Block", 15))
    elif ob_data.get('at_bearish_ob'):
        ta_score -= 10
        ta_factors.append(("At bearish Order Block", -10))
    
    if fvg_data.get('at_bullish_fvg'):
        ta_score += 10
        ta_factors.append(("At Fair Value Gap", 10))
    
    # Trend alignment
    if mf.get('trend_aligned'):
        ta_score += 10
        ta_factors.append(("Trend aligned", 10))
    
    # Volume spike
    if mf.get('volume_spike'):
        ta_score += 10
        ta_factors.append(("Volume spike detected", 10))
    
    # Pre-breakout probability
    if pre_break.get('probability', 0) >= 60:
        ta_score += 10
        ta_factors.append(("High breakout probability", 10))
    elif pre_break.get('probability', 0) >= 40:
        ta_score += 5
        ta_factors.append(("Moderate breakout setup", 5))
    
    # MFI overbought/oversold (can be negative for entries)
    mfi = mf.get('mfi', 50)
    if mfi > 80:
        ta_score -= 5
        ta_factors.append(("MFI overbought - may pullback", -5))
    elif mfi < 20:
        ta_score += 5
        ta_factors.append(("MFI oversold - bounce potential", 5))
    
    ta_score = max(0, min(100, ta_score))
    
    # ═══════════════════════════════════════════════════════════════════
    # LAYER 3: UNIFIED SCORING ENGINE - NOW TIMEFRAME-AWARE!
    # All scores (ta, inst, combined) are derived from ONE source
    # Weighted by trade_mode for proper relevance
    # ═══════════════════════════════════════════════════════════════════
    
    # Get unified verdict from whale dict (set during API fetch)
    unified_verdict = whale.get('unified_verdict', None)
    
    # Get OI change and whale % from whale data if not provided
    if oi_change is None:
        oi_data = whale.get('open_interest', {})
        oi_change = oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0
    
    if whale_pct is None:
        top_trader_data = whale.get('top_trader_ls', {})
        whale_pct = top_trader_data.get('long_pct', 50) if isinstance(top_trader_data, dict) else 50
    
    # Call the unified scoring engine - NOW WITH TRADE MODE & TIMEFRAME!
    result = get_all_scores(
        ta_score, 
        unified_verdict,
        trade_mode=trade_mode,
        timeframe=timeframe,
        oi_change=oi_change,
        whale_pct=whale_pct
    )
    
    # Return all scores from the unified engine
    return {
        'ta_score': result['ta_score'],
        'ta_factors': ta_factors,  # Keep our calculated factors
        'inst_score': result['inst_score'],  # DERIVED from whale verdict, weighted by mode!
        'inst_factors': result.get('inst_factors', []),
        'combined_score': result['combined_score'],
        'confidence_level': result['confidence_level'],
        'confidence_color': result['confidence_color'],
        'action_hint': result['action_hint'],
        'alignment': result['alignment'],
        'alignment_note': result['alignment_note'],
        'final_action': result.get('final_action', ''),
        'position_size': result.get('position_size', '50%'),
        'reasoning': result.get('reasoning', ''),
        'warnings': result.get('warnings', []),
        'whale_verdict': result.get('whale_verdict', 'neutral'),
        'whale_confidence': result.get('whale_confidence', 'low'),
        'trade_mode': result.get('trade_mode', trade_mode),
        'timeframe': result.get('timeframe', timeframe),
        'ta_weight': result.get('ta_weight', 0.5),
        'whale_weight': result.get('whale_weight', 0.5),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION ANALYSIS (AMD - Accumulation/Manipulation/Distribution)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_trading_sessions(df: pd.DataFrame) -> dict:
    """
    Analyze price action across trading sessions
    Asian (00:00-08:00 UTC), London (08:00-16:00 UTC), New York (13:00-21:00 UTC)
    """
    if df is None or len(df) < 24:
        return {
            'asian': {'direction': 'NEUTRAL', 'change_pct': 0},
            'london': {'direction': 'NEUTRAL', 'change_pct': 0},
            'newyork': {'direction': 'NEUTRAL', 'change_pct': 0},
            'current_session': 'Unknown',
            'pattern': 'Insufficient data'
        }
    
    # Get current UTC hour
    now_utc = datetime.utcnow()
    current_hour = now_utc.hour
    
    # Determine current session
    if 0 <= current_hour < 8:
        current_session = 'Asian'
    elif 8 <= current_hour < 13:
        current_session = 'London'
    elif 13 <= current_hour < 21:
        current_session = 'New York'
    else:
        current_session = 'After Hours'
    
    # Analyze last 24 hours by session
    df_recent = df.tail(48)  # Get more data for analysis
    
    # Simple session analysis based on recent candles
    asian_data = []
    london_data = []
    ny_data = []
    
    for idx, row in df_recent.iterrows():
        if hasattr(row, 'DateTime'):
            hour = row['DateTime'].hour if hasattr(row['DateTime'], 'hour') else 0
        else:
            hour = 12  # Default
        
        change = ((row['Close'] - row['Open']) / row['Open']) * 100 if row['Open'] > 0 else 0
        
        if 0 <= hour < 8:
            asian_data.append(change)
        elif 8 <= hour < 13:
            london_data.append(change)
        else:
            ny_data.append(change)
    
    def get_session_summary(changes):
        if not changes:
            return {'direction': 'NEUTRAL', 'change_pct': 0, 'emoji': '⚪'}
        avg = sum(changes) / len(changes)
        if avg > 0.1:
            return {'direction': 'ACC ↑', 'change_pct': avg, 'emoji': '🟢'}  # Accumulation
        elif avg < -0.1:
            return {'direction': 'DIST ↓', 'change_pct': avg, 'emoji': '🔴'}  # Distribution
        else:
            return {'direction': 'RANGE', 'change_pct': avg, 'emoji': '🟡'}  # Ranging
    
    asian = get_session_summary(asian_data)
    london = get_session_summary(london_data)
    newyork = get_session_summary(ny_data)
    
    # Determine pattern
    acc_count = sum(1 for s in [asian, london, newyork] if 'ACC' in s['direction'])
    dist_count = sum(1 for s in [asian, london, newyork] if 'DIST' in s['direction'])
    
    if acc_count >= 2:
        pattern = "🐂 Institutions accumulating (bullish bias)"
    elif dist_count >= 2:
        pattern = "🐻 Distribution phase (bearish bias)"
    elif asian['direction'] == 'ACC ↑' and newyork['direction'] == 'ACC ↑':
        pattern = "🎯 Smart money buying dips"
    elif london['direction'] == 'DIST ↓':
        pattern = "⚠️ London manipulation - wait for reversal"
    else:
        pattern = "📊 Mixed signals - trade with caution"
    
    return {
        'asian': asian,
        'london': london,
        'newyork': newyork,
        'current_session': current_session,
        'pattern': pattern
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 SHARED ANALYSIS FUNCTION - Single Source of Truth
# Used by BOTH Scanner and Single Analysis for consistent results
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_symbol_full(df, symbol: str, timeframe: str, market_type: str, trade_mode: str, fetch_whale_api: bool = False) -> dict:
    """
    Complete symbol analysis - SINGLE SOURCE OF TRUTH
    
    Both Scanner and Single Analysis call this function to ensure
    identical analysis logic and consistent results.
    
    Args:
        fetch_whale_api: If True, fetch real whale data from Binance API (slower but more accurate)
    
    Returns dict with all analysis data or None if analysis fails.
    """
    if df is None or len(df) < 50:
        return {'error': f'Invalid df: {len(df) if df is not None else "None"} rows', 'traceback': 'N/A'}
    
    step = "init"
    try:
        step = "get_price"
        current_price = df['Close'].iloc[-1]
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Base Technical Analysis
        # ═══════════════════════════════════════════════════════════════
        step = "money_flow"
        mf = calculate_money_flow(df)
        
        step = "smc"
        smc = detect_smc(df)
        
        step = "pre_break"
        pre_break = detect_pre_breakout(df)
        
        step = "whale_basic"
        whale = detect_whale_activity(df)  # Basic chart-based
        
        step = "sessions"
        sessions = analyze_trading_sessions(df)
        
        step = "institutional"
        institutional = detect_institutional_activity(df, mf)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Fetch Real Whale Data (ONLY if enabled!)
        # ═══════════════════════════════════════════════════════════════
        step = "whale_api_check"
        if "Crypto" in market_type and fetch_whale_api:
            try:
                real_whale_data = get_whale_analysis(symbol)
                
                if real_whale_data:
                    whale['open_interest'] = real_whale_data.get('open_interest', {})
                    whale['funding_rate'] = real_whale_data.get('funding', {}).get('rate', 0)
                    whale['top_trader_ls'] = real_whale_data.get('top_trader_ls', {})
                    whale['retail_ls'] = real_whale_data.get('retail_ls', {})
                    whale['real_whale_data'] = real_whale_data
                    
                    top_long = real_whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                    if top_long >= 55:
                        whale['whale_detected'] = True
                        whale['direction'] = 'BULLISH'
                        whale['confidence'] = min(90, int((top_long - 50) * 3))
                    elif top_long <= 45:
                        whale['whale_detected'] = True
                        whale['direction'] = 'BEARISH'
                        whale['confidence'] = min(90, int((50 - top_long) * 3))
                    else:
                        whale['direction'] = 'NEUTRAL'
                        whale['confidence'] = 30
                    
                    # Unified verdict for score adjustment
                    from core.education import get_unified_verdict
                    oi_change = real_whale_data.get('open_interest', {}).get('change_24h', 0)
                    price_change = real_whale_data.get('price_change_24h', 0)
                    retail_long = real_whale_data.get('retail_ls', {}).get('long_pct', 50)
                    funding_raw = real_whale_data.get('funding', {}).get('rate', 0)
                    
                    unified_verdict = get_unified_verdict(
                        oi_change, price_change, top_long, retail_long, funding_raw, 'WAIT'
                    )
                    whale['unified_verdict'] = unified_verdict
                    
                    # Apply verdict override
                    verdict_conf = unified_verdict.get('confidence', 'MEDIUM')
                    verdict_action = str(unified_verdict.get('unified_action', 'WAIT')).upper()
                    is_wait_or_avoid = any(x in verdict_action for x in ['AVOID', 'WAIT', 'RANGE', 'CAUTION'])
                    
                    if verdict_conf == 'LOW' or is_wait_or_avoid:
                        whale['direction'] = 'NEUTRAL'
                        whale['confidence'] = 30
                        whale['no_edge'] = True
                        whale['verdict_override'] = True
            except:
                pass
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Extract metrics for scoring
        # ═══════════════════════════════════════════════════════════════
        step = "extract_metrics"
        oi_data = whale.get('open_interest', {})
        oi_change = oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0
        top_trader_data = whale.get('top_trader_ls', {})
        whale_pct = top_trader_data.get('long_pct', 50) if isinstance(top_trader_data, dict) else 50
        retail_data = whale.get('retail_ls', {})
        retail_pct = retail_data.get('long_pct', 50) if isinstance(retail_data, dict) else 50
        price_change = whale.get('real_whale_data', {}).get('price_change_24h', 0) if whale.get('real_whale_data') else 0
        
        # SMC structure data
        smc_structure = smc.get('structure', {})
        swing_high = smc_structure.get('last_swing_high', 0)
        swing_low = smc_structure.get('last_swing_low', 0)
        structure_type = smc_structure.get('structure', 'Unknown')
        
        # Position in range
        if swing_high and swing_low and swing_high > swing_low:
            pos_in_range = ((current_price - swing_low) / (swing_high - swing_low)) * 100
            pos_in_range = max(0, min(100, pos_in_range))
        else:
            pos_in_range = 50
        
        # Money Flow Phase
        step = "flow_context"
        flow_ctx = get_money_flow_context(
            is_accumulating=mf.get('is_accumulating', False),
            is_distributing=mf.get('is_distributing', False),
            position_pct=pos_in_range,
            structure_type=structure_type
        )
        money_flow_phase = flow_ctx.phase
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Generate signal ONCE, then check if direction matches
        # ═══════════════════════════════════════════════════════════════
        step = "signal_gen"
        signal = SignalGenerator.generate_signal(df, symbol, timeframe)
        
        step = "confidence_scores"
        confidence_scores = calculate_confidence_scores(
            signal, mf, smc, whale, pre_break, institutional,
            trade_mode=trade_mode,
            timeframe=timeframe,
            oi_change=oi_change,
            whale_pct=whale_pct
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: 🎯 PREDICTIVE SCORE (determines direction!)
        # ═══════════════════════════════════════════════════════════════
        step = "predictive_setup"
        support_level = signal.stop_loss if signal and signal.direction == 'LONG' else None
        
        step = "predictive_calc"
        predictive_result = calculate_predictive_score(
            oi_change=oi_change,
            price_change=price_change,
            whale_pct=whale_pct,
            retail_pct=retail_pct,
            ta_score=confidence_scores.get('ta_score', 50),
            trade_mode=trade_mode,
            timeframe=timeframe,
            support_level=support_level,
            current_price=current_price,
            swing_high=swing_high,
            swing_low=swing_low,
            structure_type=structure_type,
            money_flow_phase=money_flow_phase
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Regenerate signal ONLY if direction mismatch
        # ═══════════════════════════════════════════════════════════════
        step = "direction_check"
        trade_direction = predictive_result.trade_direction if predictive_result else 'WAIT'
        
        # Check if we need to regenerate (only if direction differs)
        signal_dir = signal.direction if signal else None
        need_regen = (
            (trade_direction == 'LONG' and signal_dir != 'LONG') or
            (trade_direction == 'SHORT' and signal_dir != 'SHORT') or
            (signal is None and trade_direction in ['LONG', 'SHORT'])
        )
        
        step = "signal_regen"
        if need_regen and trade_direction in ['LONG', 'SHORT']:
            new_signal = SignalGenerator.generate_signal(df, symbol, timeframe, force_direction=trade_direction)
            if new_signal and new_signal.is_valid:
                signal = new_signal
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6b: Fallback to narrative_analyze if SignalGenerator fails
        # SAME AS SINGLE ANALYSIS - use narrative_analyze for levels
        # ═══════════════════════════════════════════════════════════════
        step = "narrative_fallback"
        narrative_result = None
        if signal is None or not getattr(signal, 'is_valid', True):
            try:
                # Same call as Single Analysis line 6727
                narrative_mode = 'daytrade' if trade_mode == 'day_trade' else trade_mode
                narrative_result = narrative_analyze(df, symbol, narrative_mode, timeframe)
                
                # Check if we got valid levels
                if narrative_result and narrative_result.entry and narrative_result.entry > 0:
                    n_entry = float(narrative_result.entry)
                    n_sl = float(narrative_result.stop_loss) if narrative_result.stop_loss else n_entry * 0.98
                    n_tp1 = float(narrative_result.tp1) if narrative_result.tp1 else n_entry * 1.03
                    n_tp2 = float(narrative_result.tp2) if narrative_result.tp2 else n_entry * 1.05
                    n_tp3 = float(narrative_result.tp3) if narrative_result.tp3 else n_entry * 1.08
                    n_risk = abs((n_entry - n_sl) / n_entry * 100) if n_entry > 0 else 2.0
                    n_rr = float(narrative_result.rr_ratio) if narrative_result.rr_ratio else 1.5
                    
                    # Create signal-like object from narrative result
                    class FallbackSignal:
                        pass
                    signal = FallbackSignal()
                    signal.symbol = symbol
                    signal.direction = trade_direction if trade_direction in ['LONG', 'SHORT'] else 'LONG'
                    signal.entry = n_entry
                    signal.stop_loss = n_sl
                    signal.tp1 = n_tp1
                    signal.tp2 = n_tp2
                    signal.tp3 = n_tp3
                    signal.risk_pct = n_risk
                    signal.rr_tp1 = n_rr
                    signal.rr_tp2 = n_rr * 1.5
                    signal.rr_tp3 = n_rr * 2.5
                    signal.rr_ratio = n_rr * 1.5  # Same as rr_tp2 (legacy compatibility)
                    signal.confidence = 60
                    signal.timeframe = timeframe
                    signal.is_valid = True
                    signal.sl_reason = 'Narrative'
                    signal.sl_type = 'narrative'
                    signal.entry_reason = 'Narrative'
                    signal.tp1_reason = 'Narrative'
                    signal.tp2_reason = 'Extended'
                    signal.tp3_reason = 'Max target'
                    signal.timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
            except Exception as e:
                pass  # Keep signal as None if fallback fails
        
        step = "return"
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Return all analysis data
        # ═══════════════════════════════════════════════════════════════
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': signal,
            'predictive_result': predictive_result,
            'score': predictive_result.final_score if predictive_result else 50,
            'trade_direction': trade_direction,
            'confidence_scores': confidence_scores,
            'mf': mf,
            'smc': smc,
            'whale': whale,
            'pre_break': pre_break,
            'institutional': institutional,
            'sessions': sessions,
            'df': df,
            'current_price': current_price,
            'money_flow_phase': money_flow_phase,
            'narrative_result': narrative_result  # Include for display
        }
        
    except Exception as e:
        # Store error for debugging with step info
        import traceback
        error_msg = f"[Step: {step}] {type(e).__name__}: {str(e)[:100]}"
        # Return error info instead of None for debugging
        return {'error': error_msg, 'traceback': traceback.format_exc()[:500]}

# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL BREAKDOWN GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signal_breakdown(df, signal, mf, smc, whale, pre_break, score) -> dict:
    """
    Generate detailed breakdown of why this signal is rated as it is
    Returns grade progression, reasons, and confidence factors
    """
    
    grade_info = get_pro_grade(score)
    
    # Initialize breakdown
    breakdown = {
        'grade': grade_info['grade'],
        'grade_color': grade_info['color'],
        'grade_emoji': grade_info['emoji'],
        'grade_label': grade_info['label'],
        'score': score,
        'progression': [],  # Grade steps achieved
        'bullish_factors': [],
        'bearish_factors': [],
        'neutral_factors': [],
        'risk_factors': [],
        'confidence_pct': 0,
        'roi_potential': {},
        'order_block': {},
        'volume_analysis': {},
        'trend_analysis': {},
        'action_recommendation': ''
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # GRADE PROGRESSION (like Pulse Flow: C → B → A → A+)
    # ═══════════════════════════════════════════════════════════════════════
    
    progression = []
    
    # Step 1: Base Signal (Grade D → C)
    if signal:
        progression.append({
            'step': 'D → C',
            'name': 'Signal Detected',
            'detail': f"{signal.direction} signal on {signal.timeframe}",
            'achieved': True
        })
    
    # Step 2: Order Block Confirmation (Grade C → B)
    ob_data = smc.get('order_blocks', {})
    at_bullish_ob = ob_data.get('at_bullish_ob', False)
    
    if at_bullish_ob or ob_data.get('bullish_ob'):
        progression.append({
            'step': 'C → B',
            'name': 'Order Block Hit',
            'detail': f"Price at demand zone",
            'achieved': True
        })
        breakdown['bullish_factors'].append("✅ At Order Block (institutional demand zone)")
        
        # Store OB data - use correct keys from detect_order_blocks
        breakdown['order_block'] = {
            'type': 'BULLISH DEMAND',
            'top': ob_data.get('bullish_ob_top', 0),
            'bottom': ob_data.get('bullish_ob_bottom', 0),
            'strength': 70 if at_bullish_ob else 50  # Higher if price is at OB
        }
    else:
        progression.append({
            'step': 'C → B',
            'name': 'Order Block Hit',
            'detail': 'Not at key OB level',
            'achieved': False
        })
        breakdown['neutral_factors'].append("⚪ Not at Order Block - entry may be chased")
    
    # Step 3: FVG + Structure (Grade B → A)
    fvg_data = smc.get('fvg', {})
    at_fvg = fvg_data.get('at_bullish_fvg', False)
    structure = smc.get('structure', {})
    
    if at_fvg or mf['is_accumulating']:
        progression.append({
            'step': 'B → A',
            'name': 'FVG + Money Flow',
            'detail': 'Imbalance zone + accumulation',
            'achieved': True
        })
        if at_fvg:
            breakdown['bullish_factors'].append("✅ At Fair Value Gap (price magnet zone)")
        if mf['is_accumulating']:
            breakdown['bullish_factors'].append("✅ Money flowing IN (OBV rising)")
    else:
        progression.append({
            'step': 'B → A',
            'name': 'FVG + Money Flow',
            'detail': 'No imbalance confirmation',
            'achieved': False
        })
    
    # Step 4: Volume Spike (Grade A → A+)
    if whale.get('whale_detected') or pre_break.get('probability', 0) >= 60:
        progression.append({
            'step': 'A → A+',
            'name': 'Volume Spike',
            'detail': f"Whale activity or breakout setup",
            'achieved': True
        })
        if whale.get('whale_detected'):
            breakdown['bullish_factors'].append(f"🐋 WHALE ACTIVITY: {whale.get('direction', 'Bullish')} volume spike")
        if pre_break.get('probability', 0) >= 60:
            breakdown['bullish_factors'].append(f"🚀 Pre-breakout detected ({pre_break['probability']}% probability)")
    else:
        progression.append({
            'step': 'A → A+',
            'name': 'Volume Spike',
            'detail': 'No exceptional volume',
            'achieved': False
        })
    
    breakdown['progression'] = progression
    
    # ═══════════════════════════════════════════════════════════════════════
    # VOLUME ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    breakdown['volume_analysis'] = {
        'mfi': mf['mfi'],
        'mfi_status': 'OVERSOLD 🔥' if mf['mfi'] < 20 else 'OVERBOUGHT ⚠️' if mf['mfi'] > 80 else 'NEUTRAL',
        'cmf': mf['cmf'],
        'cmf_status': 'INFLOW 🟢' if mf['cmf'] > 0.05 else 'OUTFLOW 🔴' if mf['cmf'] < -0.05 else 'NEUTRAL',
        'obv_trend': 'RISING 📈' if mf.get('obv_rising', False) else 'FALLING 📉' if mf.get('obv_falling', False) else 'FLAT',
        'flow_status': mf['flow_status']
    }
    
    # Add volume-based factors
    if mf['mfi'] < 30:
        breakdown['bullish_factors'].append(f"💰 MFI oversold ({mf['mfi']:.0f}) - potential bounce zone")
    elif mf['mfi'] > 70:
        breakdown['bearish_factors'].append(f"⚠️ MFI overbought ({mf['mfi']:.0f}) - may pull back")
    
    if mf['cmf'] > 0.1:
        breakdown['bullish_factors'].append(f"💵 Strong buying pressure (CMF: {mf['cmf']:.3f})")
    elif mf['cmf'] < -0.1:
        breakdown['bearish_factors'].append(f"💸 Selling pressure (CMF: {mf['cmf']:.3f})")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    
    if len(df) >= 50:
        ema_20 = calculate_ema(df['Close'], 20).iloc[-1]
        ema_50 = calculate_ema(df['Close'], 50).iloc[-1]
        current_price = df['Close'].iloc[-1]
        rsi = calculate_rsi(df['Close'], 14).iloc[-1]
        
        trend_direction = 'BULLISH' if current_price > ema_50 else 'BEARISH'
        ema_alignment = 'ALIGNED' if (current_price > ema_20 > ema_50) else 'MISALIGNED'
        
        breakdown['trend_analysis'] = {
            'direction': trend_direction,
            'ema_alignment': ema_alignment,
            'price_vs_ema20': ((current_price - ema_20) / ema_20) * 100,
            'price_vs_ema50': ((current_price - ema_50) / ema_50) * 100,
            'rsi': rsi,
            'rsi_status': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        }
        
        if current_price > ema_20 > ema_50:
            breakdown['bullish_factors'].append("📈 Trend aligned (Price > EMA20 > EMA50)")
        elif current_price < ema_20 < ema_50:
            breakdown['bearish_factors'].append("📉 Bearish trend structure")
        
        if rsi < 35:
            breakdown['bullish_factors'].append(f"📊 RSI oversold ({rsi:.0f}) - reversal likely")
        elif rsi > 65:
            breakdown['risk_factors'].append(f"⚠️ RSI elevated ({rsi:.0f}) - watch for pullback")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ROI POTENTIAL
    # ═══════════════════════════════════════════════════════════════════════
    
    if signal:
        tp1_roi = calc_roi(signal.tp1, signal.entry)
        tp2_roi = calc_roi(signal.tp2, signal.entry)
        tp3_roi = calc_roi(signal.tp3, signal.entry)
        
        breakdown['roi_potential'] = {
            'tp1': {'price': signal.tp1, 'roi': tp1_roi, 'probability': 70},
            'tp2': {'price': signal.tp2, 'roi': tp2_roi, 'probability': 50},
            'tp3': {'price': signal.tp3, 'roi': tp3_roi, 'probability': 30},
            'risk': signal.risk_pct,
            'rr_ratio': signal.rr_ratio,
            'expected_value': (tp1_roi * 0.7 + tp2_roi * 0.5 + tp3_roi * 0.3) / 3 - (signal.risk_pct * 0.3)
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # RISK FACTORS
    # ═══════════════════════════════════════════════════════════════════════
    
    if signal and signal.risk_pct > 5:
        breakdown['risk_factors'].append(f"⚠️ High risk: {signal.risk_pct:.1f}% to stop loss")
    
    if mf.get('is_distributing', False):
        breakdown['risk_factors'].append("🚨 Distribution detected - smart money may be selling")
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACTION RECOMMENDATION - Prioritize SCORE over factor counts
    # ═══════════════════════════════════════════════════════════════════════
    
    bull_count = len(breakdown['bullish_factors'])
    bear_count = len(breakdown['bearish_factors'])
    risk_count = len(breakdown['risk_factors'])
    
    # Score-based recommendation (score is the ultimate truth)
    if score >= 80:
        if risk_count == 0:
            breakdown['action_recommendation'] = "🟢 STRONG BUY - Multiple confirmations aligned"
        else:
            # High score but has risks (like overbought) - still bullish but cautious
            breakdown['action_recommendation'] = "🟢 BUY - Strong setup, minor caution on timing"
    elif score >= 65:
        if bull_count > bear_count:
            breakdown['action_recommendation'] = "🟢 BUY - Good setup with confirmations"
        else:
            breakdown['action_recommendation'] = "🟡 LEAN BULLISH - Setup valid but watch for entry"
    elif score >= 50:
        if bull_count > bear_count:
            breakdown['action_recommendation'] = "🟡 WATCH - Wait for better entry or small position"
        else:
            breakdown['action_recommendation'] = "🟡 NEUTRAL - Mixed signals, be selective"
    elif score >= 35:
        breakdown['action_recommendation'] = "🟠 WEAK - Limited edge, high selectivity needed"
    else:
        if bear_count > bull_count:
            breakdown['action_recommendation'] = "🔴 AVOID - Bearish factors outweigh bullish"
        else:
            breakdown['action_recommendation'] = "⚪ NO TRADE - Insufficient setup quality"
    
    # Calculate confidence percentage
    achieved_steps = sum(1 for p in progression if p['achieved'])
    breakdown['confidence_pct'] = (achieved_steps / len(progression)) * 100 if progression else 0
    
    return breakdown

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

# Load active trades from file on startup
# Only initialize if not already in session state (prevents losing data on rerun)
if 'active_trades' not in st.session_state:
    saved_trades = get_active_trades()
    st.session_state.active_trades = saved_trades if saved_trades else []
else:
    # Ensure we have the latest from file (in case another tab added trades)
    saved_trades = get_active_trades()
    # Merge: keep file version as source of truth
    st.session_state.active_trades = saved_trades if saved_trades else st.session_state.active_trades

# ═══════════════════════════════════════════════════════════════════════════════
# 🏆 CLOSED TRADES - Cache in session state for persistence across refreshes
# ═══════════════════════════════════════════════════════════════════════════════
if 'closed_trades' not in st.session_state:
    # Load from file on first run
    saved_closed = get_closed_trades()
    st.session_state.closed_trades = saved_closed if saved_closed else []
else:
    # Merge file + session state (session state may have newer trades not yet in file)
    file_closed = get_closed_trades()
    session_closed = st.session_state.closed_trades
    
    # Combine, avoiding duplicates (use symbol + added_at as unique key)
    existing_keys = set()
    merged = []
    
    # Add session state first (may be more recent)
    for t in session_closed:
        key = f"{t.get('symbol')}_{t.get('added_at')}"
        if key not in existing_keys:
            existing_keys.add(key)
            merged.append(t)
    
    # Add any from file not in session
    for t in file_closed:
        key = f"{t.get('symbol')}_{t.get('added_at')}"
        if key not in existing_keys:
            existing_keys.add(key)
            merged.append(t)
    
    st.session_state.closed_trades = merged

# Track which trade closures have been notified (avoid duplicate toasts)
if 'notified_closures' not in st.session_state:
    st.session_state.notified_closures = set()

# ═══════════════════════════════════════════════════════════════════════════════
# 📋 WATCHLIST - For "WAIT" setups (not ready to trade yet)
# ═══════════════════════════════════════════════════════════════════════════════
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []  # List of WatchlistItem dicts

# Track expanded state of monitor expanders (persists across reruns)
if 'monitor_expanded' not in st.session_state:
    st.session_state.monitor_expanded = {}

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []

if 'signal_history' not in st.session_state:
    st.session_state.signal_history = {}  # Track historical performance

# Single Analysis persistence
if 'show_single_analysis' not in st.session_state:
    st.session_state.show_single_analysis = False
if 'analysis_symbol' not in st.session_state:
    st.session_state.analysis_symbol = 'BTCUSDT'
if 'analysis_tf' not in st.session_state:
    st.session_state.analysis_tf = '15m'
if 'analysis_market' not in st.session_state:
    st.session_state.analysis_market = '🪙 Crypto'
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = 'day_trade'
if 'single_mode_config' not in st.session_state:
    st.session_state.single_mode_config = {}

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 CENTRALIZED SETTINGS - One source of truth for ALL filters
# ═══════════════════════════════════════════════════════════════════════════════

if 'settings' not in st.session_state:
    st.session_state.settings = {
        # Scanner settings
        'market': 'Crypto',
        'trading_mode': 'day_trade',
        'selected_timeframes': ['15m', '1h'],  # Multi-select timeframes
        'interval': '15m',  # Primary timeframe
        'num_assets': 200,
        'min_score': 70,  # Default to 70+ for quality signals
        # Filters - APPLY EVERYWHERE
        'long_only': True,
        'min_grade': 'B',  # Match min_score 70 = Grade B+
        'auto_add_aplus': True,
        'max_sl_pct': 5.0,  # Maximum stop loss % - reject if SL > this
        'min_rr_tp1': 1.0,  # Minimum R:R at TP1 - mode-specific
        'use_real_whale_api': False,  # Fetch real Binance whale data (slower but more accurate)
        # 📊 Stock Institutional API (Quiver Quant) - Load from saved file!
        'quiver_api_key': get_saved_api_key('quiver_api_key'),
        # Mode config (will be updated when mode changes)
        'mode_config': {
            'name': 'Day Trade',
            'icon': '📊',
            'max_sl': 5.0,
            'min_rr': 1.0,
            'confirm_tf': '4h',
            'trend_tf': '1d'
        }
    }

def get_setting(key, default=None):
    """Get a setting from centralized settings"""
    return st.session_state.settings.get(key, default)

def apply_direction_filter(items, direction_key='direction'):
    """Apply LONG only filter to any list of items with direction"""
    if not get_setting('long_only'):
        return items
    return [item for item in items if item.get(direction_key) == 'LONG']

def should_show_bearish():
    """Check if bearish signals should be shown"""
    return not get_setting('long_only')

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 InvestorIQ")
    
    app_mode = st.radio(
        "Mode",
        ["🔍 Scanner", "📈 Trade Monitor", "🔬 Single Analysis", "📊 Performance"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if app_mode == "🔍 Scanner":
        st.markdown("#### ⚙️ Settings")
        
        # Market selection
        market_type = st.selectbox(
            "Market",
            ["🪙 Crypto", "📈 Stocks", "📊 ETFs"],
            index=0
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # TIMEFRAME-SPECIFIC DESCRIPTIONS - What makes each TF unique
        # ═══════════════════════════════════════════════════════════════════════
        
        TIMEFRAME_DETAILS = {
            "1m": {
                "name": "1 Minute",
                "hold_time": "5-30 minutes",
                "expected_close": "5-30 minutes",
                "description": "Extreme scalping for experienced traders only",
                "pros": "Fastest profits, many opportunities daily",
                "cons": "Highest stress, needs constant attention, spread matters more",
                "best_for": "Full-time traders with fast execution",
                "tip": "⚠️ Only trade high-volume pairs. Spread can eat your profits on low-volume coins."
            },
            "5m": {
                "name": "5 Minute", 
                "hold_time": "30 min - 2 hours",
                "expected_close": "30 minutes to 2 hours",
                "description": "Active scalping with breathing room between candles",
                "pros": "Good balance of speed and clarity, cleaner signals than 1m",
                "cons": "Still requires frequent monitoring",
                "best_for": "Active traders who can check every 5-10 minutes",
                "tip": "✅ Most popular scalping timeframe. Good signal quality without extreme stress."
            },
            "15m": {
                "name": "15 Minute",
                "hold_time": "2-6 hours",
                "expected_close": "2-6 hours (same day)",
                "description": "Sweet spot for day trading - clear signals, manageable pace",
                "pros": "Excellent signal clarity, filters out 1m/5m noise, time to think",
                "cons": "Fewer setups than lower timeframes",
                "best_for": "Day traders who want quality over quantity",
                "tip": "⭐ RECOMMENDED for most traders. Best balance of signal quality and opportunity."
            },
            "1h": {
                "name": "1 Hour",
                "hold_time": "4-12 hours",
                "expected_close": "4-12 hours (within the day)",
                "description": "Relaxed day trading - can work a job and still trade",
                "pros": "High quality signals, less noise, can set alerts and walk away",
                "cons": "Slower, may miss intraday moves",
                "best_for": "People with jobs who trade around work schedule",
                "tip": "👔 Perfect for 9-5 workers. Check at lunch, after work, before bed."
            },
            "4h": {
                "name": "4 Hour",
                "hold_time": "2-7 days",
                "expected_close": "2-7 days",
                "description": "Classic swing trading - capture multi-day moves",
                "pros": "Very clean signals, great work-life balance, less emotional",
                "cons": "Need patience, overnight risk",
                "best_for": "Busy professionals who check charts 2x daily",
                "tip": "⭐ BEST for swing traders. Check morning and evening only."
            },
            "1d": {
                "name": "Daily",
                "hold_time": "1-4 weeks",
                "expected_close": "1-4 weeks",
                "description": "Position trading for patient, disciplined traders",
                "pros": "Highest quality signals, minimal time needed, less stress",
                "cons": "Requires larger stops, more capital needed, slower profits",
                "best_for": "Patient traders who want minimal screen time",
                "tip": "🧘 One chart check per day is enough. Great for mental health."
            },
            "1w": {
                "name": "Weekly",
                "hold_time": "Months to years",
                "expected_close": "1-6 months (or longer)",
                "description": "Long-term investing - ignore daily noise",
                "pros": "Captures major trends, almost no time needed, compound growth",
                "cons": "Very wide stops, ties up capital for long periods",
                "best_for": "Investors building long-term wealth",
                "tip": "💎 Perfect for retirement accounts. Set and forget."
            }
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # TRADING MODES - Each has clear purpose, description, and parameters
        # ═══════════════════════════════════════════════════════════════════════
        
        TRADING_MODE_CONFIG = {
            "scalp": {
                "name": "Scalp",
                "icon": "⚡",
                "description": "Quick in-and-out trades for small profits",
                "purpose": "Capture small price moves with tight stops. High frequency.",
                "timeframes": ["1m", "5m"],
                "max_sl": 2.5,
                "min_rr": 0.75,
                "hold_time": "Minutes to 2 hours",
                "trades_per_day": "5-20+",
                "check_freq": "Every few minutes",
                "confirm_tf": "15m",
                "trend_tf": "1h",
                "color": "#ff9500"
            },
            "day_trade": {
                "name": "Day Trade",
                "icon": "📊",
                "description": "Intraday positions closed same day",
                "purpose": "Capture intraday swings. No overnight risk.",
                "timeframes": ["15m", "1h"],
                "max_sl": 5.0,
                "min_rr": 1.0,
                "hold_time": "2-12 hours",
                "trades_per_day": "1-5",
                "check_freq": "Every 30-60 min",
                "confirm_tf": "4h",
                "trend_tf": "1d",
                "color": "#00d4ff"
            },
            "swing": {
                "name": "Swing Trade",
                "icon": "📈",
                "description": "Hold positions for days to capture bigger moves",
                "purpose": "Ride medium-term trends. Check charts 2x daily.",
                "timeframes": ["4h", "1d"],
                "max_sl": 8.0,
                "min_rr": 1.5,
                "hold_time": "2-14 days",
                "trades_per_day": "0-2",
                "check_freq": "Morning & evening",
                "confirm_tf": "1d",
                "trend_tf": "1w",
                "color": "#00d4aa"
            },
            "position": {
                "name": "Position Trade",
                "icon": "🏦",
                "description": "Longer-term trades following major trends",
                "purpose": "Capture big moves over weeks. Low maintenance.",
                "timeframes": ["1d", "1w"],
                "max_sl": 12.0,
                "min_rr": 2.0,
                "hold_time": "1-8 weeks",
                "trades_per_day": "0-1 per week",
                "check_freq": "Once daily",
                "confirm_tf": "1w",
                "trend_tf": "1M",
                "color": "#9d4edd"
            },
            "investment": {
                "name": "Investment",
                "icon": "💎",
                "description": "Long-term accumulation for wealth building",
                "purpose": "Buy & hold strategy. Ignore daily noise.",
                "timeframes": ["1w"],
                "max_sl": 20.0,
                "min_rr": 2.5,
                "hold_time": "Months to years",
                "trades_per_day": "Rare",
                "check_freq": "Weekly",
                "confirm_tf": "1M",
                "trend_tf": "1M",
                "color": "#ffd700"
            }
        }
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: SELECT TRADING MODE
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown("#### 🎯 Trading Mode")
        
        trading_mode = st.selectbox(
            "Select Your Style",
            list(TRADING_MODE_CONFIG.keys()),
            index=1,  # Default to day_trade
            format_func=lambda x: f"{TRADING_MODE_CONFIG[x]['icon']} {TRADING_MODE_CONFIG[x]['name']}"
        )
        
        mode_config = TRADING_MODE_CONFIG[trading_mode]
        
        # Show mode description
        st.markdown(f"""
        <div style='background: {mode_config['color']}20; border-left: 4px solid {mode_config['color']}; 
                    padding: 10px 12px; border-radius: 4px; margin: 8px 0;'>
            <div style='color: {mode_config['color']}; font-weight: bold; font-size: 0.95em;'>
                {mode_config['icon']} {mode_config['name']}
            </div>
            <div style='color: #333; font-size: 0.85em; margin-top: 5px;'>
                {mode_config['description']}
            </div>
            <div style='color: #666; font-size: 0.8em; margin-top: 3px; font-style: italic;'>
                💡 {mode_config['purpose']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: SELECT TIMEFRAME(S) - Based on mode
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown("#### ⏱️ Timeframe(s)")
        
        available_tfs = mode_config['timeframes']
        
        # Multi-select for timeframes
        if len(available_tfs) > 1:
            selected_tfs = st.multiselect(
                f"Select timeframe(s) for {mode_config['name']}",
                available_tfs,
                default=available_tfs,  # Default to all
                help="Select one or more timeframes to scan"
            )
            if not selected_tfs:
                selected_tfs = [available_tfs[0]]  # Default to first if none selected
        else:
            selected_tfs = available_tfs
            st.info(f"📌 {mode_config['name']} uses {available_tfs[0]} timeframe")
        
        # Show what will be scanned
        tf_display = " + ".join(selected_tfs)
        st.markdown(f"""
        <div style='background: #f0f8ff; border: 1px solid #87CEEB; border-radius: 6px; 
                    padding: 8px 10px; margin: 5px 0; font-size: 0.85em;'>
            <span style='color: #4169E1;'>🔍 Will scan: <strong>{mode_config['name']} {tf_display}</strong></span>
        </div>
        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TIMEFRAME-SPECIFIC DETAILS - Show unique info for each selected TF
        # ═══════════════════════════════════════════════════════════════════════
        
        for tf in selected_tfs:
            tf_info = TIMEFRAME_DETAILS.get(tf, {})
            if tf_info:
                st.markdown(f"""
                <div style='background: #ffffff; border: 1px solid {mode_config['color']}; border-radius: 8px; 
                            padding: 10px 12px; margin: 8px 0; border-left: 4px solid {mode_config['color']};'>
                    <div style='color: {mode_config['color']}; font-weight: bold; font-size: 0.95em;'>
                        {mode_config['icon']} {mode_config['name']} {tf} - {tf_info['name']}
                    </div>
                    <div style='color: #333; font-size: 0.85em; margin-top: 6px;'>
                        {tf_info['description']}
                    </div>
                    <div style='color: #444; font-size: 0.8em; margin-top: 8px;'>
                        <div style='background: #e8f5e9; padding: 6px 10px; border-radius: 4px; margin-bottom: 6px;'>
                            🎯 <strong style='color: #2e7d32;'>Expected Trade Duration: {tf_info['expected_close']}</strong>
                        </div>
                        <div style='color: #228B22;'>✅ <strong>Pros:</strong> {tf_info['pros']}</div>
                        <div style='color: #B22222;'>⚠️ <strong>Cons:</strong> {tf_info['cons']}</div>
                        <div>👤 <strong>Best For:</strong> {tf_info['best_for']}</div>
                    </div>
                    <div style='background: #fffde7; border-radius: 4px; padding: 6px 8px; margin-top: 8px; 
                                font-size: 0.8em; color: #5d4037;'>
                        💡 {tf_info['tip']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: SHOW MODE PARAMETERS
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown(f"""
        <div style='background: #ffffff; border: 1px solid #ddd; border-radius: 8px; 
                    padding: 10px 12px; margin: 8px 0;'>
            <div style='color: #333; font-weight: bold; font-size: 0.9em;'>📋 {mode_config['name']} Parameters</div>
            <div style='color: #444; font-size: 0.85em; margin-top: 6px;'>
                <div>🛡️ Max Stop Loss: <strong>{mode_config['max_sl']}%</strong></div>
                <div>🎯 Min R:R at TP1: <strong>{mode_config['min_rr']}:1</strong></div>
                <div>⏱️ Hold Time: <strong>{mode_config['hold_time']}</strong></div>
                <div>👀 Check Charts: <strong>{mode_config['check_freq']}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Multi-TF confirmation guidance
        st.markdown(f"""
        <div style='background: #fff3e6; border: 1px solid #ff9500; border-radius: 6px; 
                    padding: 8px 10px; margin: 8px 0; font-size: 0.8em;'>
            <div style='color: #cc5500; font-weight: bold;'>📊 Pro Tip: Multi-TF Check</div>
            <div style='color: #666; margin-top: 3px;'>
                Confirm on <strong>{mode_config['confirm_tf']}</strong> | 
                Trend on <strong>{mode_config['trend_tf']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🐋 WHALE DATA MODE - PROMINENT TOGGLE
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown("---")
        st.markdown("#### 🐋 Whale Data Source")
        
        use_real_whale = st.checkbox(
            "🔴 Real-Time Binance API", 
            value=st.session_state.settings.get('use_real_whale_api', False),
            help="ON = Live OI, Funding, Positioning data (slower but accurate)\nOFF = Chart-based detection (fast)"
        )
        
        # Visual indicator of current mode
        if use_real_whale:
            st.markdown("""
            <div style='background: #e8f5e9; border: 1px solid #4caf50; border-radius: 6px; 
                        padding: 8px 10px; margin: 5px 0; font-size: 0.8em;'>
                <span style='color: #2e7d32; font-weight: bold;'>✅ REAL API MODE</span>
                <span style='color: #555;'> - Live whale data from Binance (slower scan)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #e3f2fd; border: 1px solid #2196f3; border-radius: 6px; 
                        padding: 8px 10px; margin: 5px 0; font-size: 0.8em;'>
                <span style='color: #1565c0; font-weight: bold;'>⚡ FAST MODE</span>
                <span style='color: #555;'> - Chart-based whale detection (faster scan)</span>
            </div>
            """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 📊 STOCK INSTITUTIONAL API (Quiver Quant)
        # ═══════════════════════════════════════════════════════════════════════
        
        if "Stocks" in market_type or "ETFs" in market_type:
            st.markdown("---")
            st.markdown("#### 📊 Stock Institutional Data")
            
            # Load saved key as default
            saved_key = get_saved_api_key('quiver_api_key')
            
            quiver_key = st.text_input(
                "Quiver Quant API Token",
                value=st.session_state.settings.get('quiver_api_key', saved_key),
                type="password",
                help="Paste your TOKEN here (not username). Get it from api.quiverquant.com after subscribing."
            )
            
            # Auto-save when key changes
            if quiver_key and quiver_key != saved_key:
                save_api_key('quiver_api_key', quiver_key)
                st.markdown("""
                <div style='background: #e8f5e9; border: 1px solid #4caf50; border-radius: 6px; 
                            padding: 8px 10px; margin: 5px 0; font-size: 0.8em;'>
                    <span style='color: #2e7d32; font-weight: bold;'>✅ Token Saved!</span>
                    <span style='color: #555;'> - Will persist across refreshes</span>
                </div>
                """, unsafe_allow_html=True)
            elif quiver_key:
                st.markdown("""
                <div style='background: #e8f5e9; border: 1px solid #4caf50; border-radius: 6px; 
                            padding: 8px 10px; margin: 5px 0; font-size: 0.8em;'>
                    <span style='color: #2e7d32; font-weight: bold;'>✅ Token Set</span>
                    <span style='color: #555;'> - Congress, Insider, Short data enabled</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: #fff3e0; border: 1px solid #ff9800; border-radius: 6px; 
                            padding: 8px 10px; margin: 5px 0; font-size: 0.8em;'>
                    <span style='color: #e65100;'>🔑 Need Token:</span>
                    <span style='color: #555;'> Go to <a href='https://api.quiverquant.com/' target='_blank' style='color: #1565c0;'>api.quiverquant.com</a> → Subscribe → Copy Token</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            quiver_key = st.session_state.settings.get('quiver_api_key', '')
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: OTHER SETTINGS
        # ═══════════════════════════════════════════════════════════════════════
        
        st.markdown("---")
        
        num_assets = st.slider("Assets to Scan", 10, 500, st.session_state.settings['num_assets'],
                               help="Binance Futures has 450+ pairs. Increase to catch lower-volume coins like COWUSDT.")
        min_score = st.slider("Min Score", 0, 100, st.session_state.settings['min_score'])
        
        st.markdown("---")
        st.markdown("#### 🎯 Filters")
        
        long_only = st.checkbox("📈 LONG Only", value=st.session_state.settings['long_only'])
        
        # Grade filter REMOVED - Score is the only filter now
        # Grade is just a display label derived from score:
        # A+ = 85+, A = 70+, B = 55+, C = 40+, D = 0+
        
        # Max SL is auto-set by mode (but user can override)
        st.markdown(f"""
        <div style='background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; 
                    padding: 8px; margin: 5px 0; font-size: 0.8em;'>
            <span style='color: #856404;'>💡 Max SL auto-set to <b>{mode_config['max_sl']}%</b> for {mode_config['name']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        max_sl_pct = st.slider("Max Stop Loss %", 1.0, 25.0, mode_config['max_sl'], 0.5,
                               help=f"Recommended: {mode_config['max_sl']}% for {mode_config['name']}")
        
        st.markdown("---")
        st.markdown("#### 🤖 Auto-Add")
        
        auto_add = st.checkbox("Auto-add HIGH confidence to Monitor", value=st.session_state.settings['auto_add_aplus'])
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔄 SYNC TO CENTRALIZED SETTINGS - This is the ONE place settings change
        # ═══════════════════════════════════════════════════════════════════════
        st.session_state.settings.update({
            'market': market_type,
            'trading_mode': trading_mode,
            'selected_timeframes': selected_tfs,  # List of timeframes to scan
            'interval': selected_tfs[0],  # Primary timeframe (first one)
            'num_assets': num_assets,
            'min_score': min_score,
            'long_only': long_only,
            # 'min_grade': REMOVED - Use min_score only, grade is just a display label
            'auto_add_aplus': auto_add,
            'use_real_whale_api': use_real_whale,
            'max_sl_pct': max_sl_pct,
            'min_rr_tp1': mode_config['min_rr'],  # From mode config
            'mode_config': mode_config,  # Store full config for reference
            'quiver_api_key': quiver_key,  # Stock institutional data API
        })
    
    elif app_mode == "🔬 Single Analysis":
        st.markdown("#### 🔬 Analyze Single Asset")
        
        # ═══════════════════════════════════════════════════════════════════════
        # TRADING MODE CONFIG (same as Scanner for consistency)
        # ═══════════════════════════════════════════════════════════════════════
        
        SINGLE_ANALYSIS_MODES = {
            "scalp": {
                "name": "Scalp",
                "icon": "⚡",
                "description": "Quick in-and-out trades for small profits",
                "purpose": "Capture small price moves. Check every few minutes.",
                "timeframes": ["1m", "5m"],
                "default_tf": "5m",
                "max_sl": 2.5,
                "min_rr": 0.75,
                "hold_time": "Minutes to 2 hours",
                "color": "#ff9500"
            },
            "day_trade": {
                "name": "Day Trade",
                "icon": "📊",
                "description": "Intraday positions closed same day",
                "purpose": "Capture intraday swings. No overnight risk.",
                "timeframes": ["15m", "1h"],
                "default_tf": "15m",
                "max_sl": 5.0,
                "min_rr": 1.0,
                "hold_time": "2-12 hours",
                "color": "#00d4ff"
            },
            "swing": {
                "name": "Swing Trade",
                "icon": "📈",
                "description": "Hold positions for days to capture bigger moves",
                "purpose": "Ride medium-term trends. Check 2x daily.",
                "timeframes": ["4h", "1d"],
                "default_tf": "4h",
                "max_sl": 8.0,
                "min_rr": 1.5,
                "hold_time": "2-14 days",
                "color": "#00d4aa"
            },
            "position": {
                "name": "Position Trade",
                "icon": "🏦",
                "description": "Longer-term trades following major trends",
                "purpose": "Capture big moves over weeks.",
                "timeframes": ["1d", "1w"],
                "default_tf": "1d",
                "max_sl": 12.0,
                "min_rr": 2.0,
                "hold_time": "1-8 weeks",
                "color": "#9d4edd"
            },
            "investment": {
                "name": "Investment",
                "icon": "💎",
                "description": "Long-term accumulation for wealth building",
                "purpose": "Buy & hold. DCA zones, not scalp levels.",
                "timeframes": ["1w"],
                "default_tf": "1w",
                "max_sl": 20.0,
                "min_rr": 2.5,
                "hold_time": "Months to years",
                "color": "#ffd700"
            }
        }
        
        # Market selection first
        analysis_market = st.selectbox(
            "Market Type",
            ["🪙 Crypto", "📈 Stock", "📊 ETF"],
            index=0,
            key="single_market"
        )
        
        # 📊 QUIVER API TOKEN (for Stocks/ETFs)
        if "Stock" in analysis_market or "ETF" in analysis_market:
            st.markdown("---")
            st.markdown("##### 📊 Stock Institutional Data")
            
            # Load saved key as default
            saved_key = get_saved_api_key('quiver_api_key')
            
            quiver_key_single = st.text_input(
                "Quiver API Token",
                value=st.session_state.settings.get('quiver_api_key', saved_key),
                type="password",
                help="Get token from api.quiverquant.com ($10/mo)",
                key="quiver_key_single"
            )
            
            # Auto-save when key changes
            if quiver_key_single and quiver_key_single != saved_key:
                save_api_key('quiver_api_key', quiver_key_single)
                st.session_state.settings['quiver_api_key'] = quiver_key_single
                st.success("✅ Token saved!")
            elif quiver_key_single:
                st.session_state.settings['quiver_api_key'] = quiver_key_single
                st.markdown("<small style='color: #4caf50;'>✅ Token active</small>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <small style='color: #ff9800;'>
                    🔑 <a href='https://api.quiverquant.com/' target='_blank'>Get token</a> for Congress, Insider, Short data
                </small>
                """, unsafe_allow_html=True)
            st.markdown("---")
        
        if "Crypto" in analysis_market:
            default_symbol = "BTCUSDT"
        elif "Stock" in analysis_market:
            default_symbol = "AAPL"
        else:
            default_symbol = "SPY"
        
        symbol_input = st.text_input("Symbol", value=default_symbol, key="single_symbol")
        
        # Trading Mode selection (same as Scanner!)
        st.markdown("#### 🎯 Trading Mode")
        
        single_trading_mode = st.selectbox(
            "Select Your Style",
            list(SINGLE_ANALYSIS_MODES.keys()),
            index=1,  # Default to day_trade
            format_func=lambda x: f"{SINGLE_ANALYSIS_MODES[x]['icon']} {SINGLE_ANALYSIS_MODES[x]['name']}",
            key="single_trading_mode"
        )
        
        single_mode_config = SINGLE_ANALYSIS_MODES[single_trading_mode]
        
        # Show mode description
        st.markdown(f"""
        <div style='background: {single_mode_config['color']}20; border-left: 4px solid {single_mode_config['color']}; 
                    padding: 10px 12px; border-radius: 4px; margin: 8px 0;'>
            <div style='color: {single_mode_config['color']}; font-weight: bold; font-size: 0.95em;'>
                {single_mode_config['icon']} {single_mode_config['name']}
            </div>
            <div style='color: #333; font-size: 0.85em; margin-top: 5px;'>
                {single_mode_config['description']}
            </div>
            <div style='color: #666; font-size: 0.8em; margin-top: 3px; font-style: italic;'>
                💡 {single_mode_config['purpose']}
            </div>
            <div style='color: #888; font-size: 0.75em; margin-top: 5px;'>
                Hold: {single_mode_config['hold_time']} | Max SL: {single_mode_config['max_sl']}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Timeframe selection (based on mode)
        available_tfs = single_mode_config['timeframes']
        default_idx = available_tfs.index(single_mode_config['default_tf']) if single_mode_config['default_tf'] in available_tfs else 0
        
        analysis_tf = st.selectbox(
            "Timeframe",
            available_tfs,
            index=default_idx,
            key="single_tf"
        )
        
        # Store mode config for use in analysis
        if 'single_mode_config' not in st.session_state:
            st.session_state.single_mode_config = single_mode_config
        st.session_state.single_mode_config = single_mode_config
        
        # Analyze button - USE SESSION STATE TO PERSIST
        analyze_btn = st.button("🔬 ANALYZE", type="primary", use_container_width=True, key="analyze_btn")
        
        # Persist analysis state
        if analyze_btn:
            st.session_state.show_single_analysis = True
            st.session_state.analysis_symbol = symbol_input
            st.session_state.analysis_tf = analysis_tf
            st.session_state.analysis_market = analysis_market
            st.session_state.analysis_mode = single_trading_mode
        
        # Clear analysis button
        if st.session_state.get('show_single_analysis', False):
            if st.button("🗑️ Clear Analysis", use_container_width=True):
                st.session_state.show_single_analysis = False
                st.rerun()
        
    elif app_mode == "📈 Trade Monitor":
        st.markdown(f"#### 📊 Active: {len(st.session_state.active_trades)}")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.active_trades = []
            sync_active_trades([])  # Clear file too
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# 🧠 InvestorIQ")
st.markdown("<p style='color: #00d4ff;'>Smart Money Analysis for Crypto • Stocks • ETFs</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 SCANNER MODE - PRO VERSION
# ═══════════════════════════════════════════════════════════════════════════════

if app_mode == "🔍 Scanner":
    
    # Get mode and timeframe info from settings
    current_mode = get_setting('trading_mode', 'day_trade')
    selected_tfs = get_setting('selected_timeframes', ['15m'])
    mode_cfg = get_setting('mode_config', {})
    mode_name = mode_cfg.get('name', 'Day Trade') if mode_cfg else 'Day Trade'
    mode_icon = mode_cfg.get('icon', '📊') if mode_cfg else '📊'
    
    # Current session info
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    if 0 <= hour < 8:
        session_info = "🌏 Asian Session (Low Volume)"
    elif 8 <= hour < 13:
        session_info = "🇬🇧 London Session (High Volume)"
    elif 13 <= hour < 21:
        session_info = "🇺🇸 New York Session (Peak Volume)"
    else:
        session_info = "🌙 After Hours"
    
    # Show market type
    if "Crypto" in market_type:
        market_emoji = "🪙"
        market_name = "Crypto"
    elif "Stock" in market_type:
        market_emoji = "📈"
        market_name = "Stocks"
    else:
        market_emoji = "📊"
        market_name = "ETFs"
    
    # Show what we're scanning
    tf_display = " + ".join(selected_tfs)
    st.info(f"**{mode_icon} {mode_name}** | **{tf_display}** | {market_emoji} {market_name} | {session_info}")
    
    # 🎯 Show active filters status bar
    filters_active = []
    filters_active.append(f"{mode_icon} {mode_name} {tf_display}")
    if get_setting('long_only'):
        filters_active.append("📈 LONG Only")
    filters_active.append(f"Min Score: {get_setting('min_score', 40)}")
    filters_active.append(f"Max SL: {get_setting('max_sl_pct', 5.0)}%")
    filters_active.append(f"Min TP1 R:R: {get_setting('min_rr_tp1', 1.0)}:1")
    
    # API mode indicator
    use_real_api = get_setting('use_real_whale_api', False)
    api_indicator = "🐋 <span style='color: #00ff88; font-weight: bold;'>Real API ON</span>" if use_real_api else "⚡ <span style='color: #888;'>Fast Mode</span>"
    
    st.markdown(f"""
    <div style='background: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 8px 12px; margin: 5px 0;'>
        <span style='color: #888;'>🎯 Active Filters:</span> 
        <span style='color: #00d4ff;'>{' | '.join(filters_active)}</span>
        <span style='margin-left: 15px;'>{api_indicator}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Scan button
    if st.button("🚀 SCAN FOR SIGNALS", type="primary", use_container_width=True):
        
        progress = st.progress(0)
        status = st.empty()
        
        results = []
        filtered_signals = []  # 🔍 Track filtered signals for debugging
        scanned = 0
        
        # Get pairs based on market type
        if "Crypto" in market_type:
            status.markdown("📡 Fetching Binance Futures pairs (Global)...")
            # Use FUTURES pairs only - ensures whale/OI data is accurate
            # Binance Futures = Global (Dubai, Europe, Asia) - NOT US
            pairs = fetch_binance_futures_pairs(num_assets)
            if not pairs:
                pairs = get_default_futures_pairs()[:num_assets]
            fetch_func = fetch_binance_klines
                
        elif "Stock" in market_type:
            status.markdown("📡 Loading stock list...")
            pairs = TOP_STOCKS[:num_assets]
            fetch_func = fetch_stock_data
        else:  # ETFs
            status.markdown("📡 Loading ETF list...")
            pairs = TOP_ETFS[:num_assets]
            fetch_func = fetch_stock_data
        
        # Get selected timeframes from settings
        timeframes_to_scan = get_setting('selected_timeframes', ['15m'])
        total_scans = len(pairs) * len(timeframes_to_scan)
        scan_count = 0
        
        # Timer for progress estimation
        scan_start_time = time.time()
        use_real_api = get_setting('use_real_whale_api', False)
        
        # 🔍 DEBUG: Track errors from analyze_symbol_full
        analysis_errors = []
        
        # Loop through pairs AND timeframes
        for i, symbol in enumerate(pairs):
            for tf in timeframes_to_scan:
                scanned += 1
                scan_count += 1
                progress.progress(scan_count / total_scans)
                
                # Calculate elapsed and estimated remaining time
                elapsed = time.time() - scan_start_time
                if scan_count > 1:
                    avg_per_scan = elapsed / scan_count
                    remaining_scans = total_scans - scan_count
                    est_remaining = avg_per_scan * remaining_scans
                    time_info = f"⏱️ {elapsed:.0f}s elapsed | ~{est_remaining:.0f}s remaining"
                else:
                    time_info = "⏱️ Starting..."
                
                # Show API mode indicator if real API is on
                api_indicator = " | 🐋 Real API" if use_real_api else ""
                
                status.markdown(f"🔍 **[{scan_count}/{total_scans}]** <span style='color: #00d4ff;'>{symbol}</span> on <span style='color: #ffcc00;'>{mode_icon} {mode_name} {tf}</span> <span style='color: #888; font-size: 0.85em;'>{time_info}{api_indicator}</span>", unsafe_allow_html=True)
                
                try:
                    df = fetch_func(symbol, tf, 200)
                    
                    if df is not None and len(df) >= 50:
                        
                        # ═══════════════════════════════════════════════════════════
                        # 🎯 USE SHARED ANALYSIS FUNCTION (Same as Single Analysis!)
                        # ═══════════════════════════════════════════════════════════
                        
                        # Call shared function - SINGLE SOURCE OF TRUTH
                        analysis = analyze_symbol_full(df, symbol, tf, market_type, current_mode, fetch_whale_api=use_real_api)
                        
                        # Check for errors or None
                        if analysis is None:
                            continue
                        
                        # Check if error dict was returned
                        if 'error' in analysis:
                            # Track first 5 errors for any symbol
                            if len(analysis_errors) < 5:
                                analysis_errors.append({
                                    'symbol': symbol,
                                    'error': analysis['error'],
                                    'traceback': analysis.get('traceback', '')[:300]
                                })
                            continue
                        
                        # Extract results
                        signal = analysis['signal']
                        predictive_result = analysis['predictive_result']
                        score = analysis['score']
                        trade_direction = analysis['trade_direction']
                        confidence_scores = analysis['confidence_scores']
                        mf = analysis['mf']
                        smc = analysis['smc']
                        whale = analysis['whale']
                        pre_break = analysis['pre_break']
                        institutional = analysis['institutional']
                        sessions = analysis['sessions']
                        
                        # Skip if no valid signal (after all fallbacks)
                        if not signal or not getattr(signal, 'is_valid', True):
                            continue
                        
                        # ═══════════════════════════════════════════════════════════
                        # 🎯 APPLY FILTERS
                        # ═══════════════════════════════════════════════════════════
                        
                        # LONG ONLY FILTER
                        if get_setting("long_only"):
                            if trade_direction != 'LONG':
                                filtered_signals.append({
                                    'symbol': symbol, 'timeframe': tf,
                                    'reason': f"Direction filter: {trade_direction} (need LONG)",
                                    'score': score
                                })
                                continue
                        
                        # SCORE FILTER
                        min_score = get_setting('min_score', 40)
                        if score < min_score:
                            filtered_signals.append({
                                'symbol': symbol, 'timeframe': tf,
                                'reason': f"Score too low: {score} (need ≥{min_score})",
                                'score': score
                            })
                            continue
                        
                        # MAX SL FILTER
                        max_sl = get_setting('max_sl_pct', 5.0)
                        if signal.risk_pct > max_sl:
                            filtered_signals.append({
                                'symbol': symbol, 'timeframe': tf,
                                'reason': f"SL too far: {signal.risk_pct:.1f}% (max {max_sl}%)",
                                'score': score
                            })
                            continue
                        
                        # MIN R:R FILTER
                        min_rr = get_setting('min_rr_tp1', 1.0)
                        if signal.rr_tp1 < min_rr:
                            filtered_signals.append({
                                'symbol': symbol, 'timeframe': tf,
                                'reason': f"R:R too low: {signal.rr_tp1:.2f}:1 (need ≥{min_rr}:1)",
                                'score': score
                            })
                            continue
                        
                        # ═══════════════════════════════════════════════════════════
                        # ✅ PASSED ALL FILTERS - Add to results
                        # ═══════════════════════════════════════════════════════════
                        
                        breakdown = generate_signal_breakdown(df, signal, mf, smc, whale, pre_break, score)
                        
                        results.append({
                            'symbol': symbol,
                            'signal': signal,
                            'score': score,
                            'breakdown': breakdown,
                            'money_flow': mf,
                            'smc': smc,
                            'sessions': sessions,
                            'whale': whale,
                            'pre_break': pre_break,
                            'institutional': institutional,
                            'df': df,
                            'mode_name': mode_name,
                            'mode_icon': mode_icon,
                            'timeframe': tf,
                            'display_label': f"{mode_icon} {mode_name} {tf}",
                            'confidence_scores': confidence_scores,
                            'predictive_result': predictive_result
                        })
                            
                except Exception as e:
                    continue
        progress.progress(1.0)
        total_time = time.time() - scan_start_time
        status.success(f"✅ Scan complete! Found **{len(results)}** signals across {len(timeframes_to_scan)} timeframe(s) in **{total_time:.1f}s**")
        
        # 🔍 DEBUG: Show filtered signals (high score ones only)
        high_score_filtered = [f for f in filtered_signals if f.get('score', 0) >= 60]
        if high_score_filtered:
            with st.expander(f"🔍 **{len(high_score_filtered)} High-Score Signals Filtered** (score ≥60) - Click to see why", expanded=False):
                st.markdown("""
                <p style='color: #ff9800; margin-bottom: 10px;'>
                    These signals scored well but didn't pass other filters (SL%, R:R, direction)
                </p>
                """, unsafe_allow_html=True)
                
                for f in sorted(high_score_filtered, key=lambda x: x.get('score', 0), reverse=True)[:10]:
                    st.markdown(f"""
                    <div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px; margin-bottom: 5px; border-left: 3px solid #ff9800;'>
                        <span style='color: #00d4ff; font-weight: bold;'>{f['symbol']}</span>
                        <span style='color: #888;'> ({f['timeframe']})</span>
                        <span style='color: #00ff88;'> Score: {f['score']}</span>
                        <br><span style='color: #ff6b6b;'>❌ {f['reason']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 🔍 DEBUG: Show analysis errors (critical for debugging!)
        if analysis_errors:
            with st.expander(f"🚨 **Analysis Errors** - {len(analysis_errors)} errors found", expanded=True):
                st.markdown("""
                <p style='color: #ff6b6b; margin-bottom: 10px; font-weight: bold;'>
                    ⚠️ These errors are causing analyze_symbol_full() to fail:
                </p>
                """, unsafe_allow_html=True)
                
                for err in analysis_errors:
                    st.markdown(f"""
                    <div style='background: #2a1a1a; padding: 10px 12px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #ff6b6b;'>
                        <div style='color: #ff6b6b; font-weight: bold;'>{err['symbol']}: {err['error']}</div>
                        <pre style='color: #888; font-size: 0.75em; margin-top: 5px; white-space: pre-wrap;'>{err['traceback']}</pre>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 📊 SORT BY PREDICTIVE SCORE (highest first) - most important signals at top!
        results.sort(key=lambda x: x.get('predictive_result').final_score if x.get('predictive_result') else x.get('score', 0), reverse=True)
        st.session_state.scan_results = results
        
        # Auto-add HIGH CONFIDENCE setups from CENTRALIZED SETTINGS
        if get_setting('auto_add_aplus', True):
            added_count = 0
            for r in results[:5]:  # Check top 5 results
                pred_result = r.get('predictive_result')
                predictive_score = pred_result.final_score if pred_result else 0
                predictive_action = pred_result.final_action if pred_result else ''
                
                # Auto-add if predictive score >= 75 or STRONG BUY/SELL
                should_add = predictive_score >= 75 or 'STRONG' in predictive_action
                
                if should_add:
                    trade = {
                        'symbol': r['signal'].symbol,
                        'entry': r['signal'].entry,
                        'stop_loss': r['signal'].stop_loss,
                        'tp1': r['signal'].tp1,
                        'tp2': r['signal'].tp2,
                        'tp3': r['signal'].tp3,
                        'direction': r['signal'].direction,
                        'score': predictive_score,
                        'grade': r['breakdown']['grade'],
                        'timeframe': r.get('timeframe', '15m'),
                        'mode_name': r.get('mode_name', 'Day Trade'),
                        'predictive_action': predictive_action,
                        'direction_score': pred_result.direction_score if pred_result else 0,
                        'squeeze_score': pred_result.squeeze_score if pred_result else 0,
                        'timing_score': pred_result.timing_score if pred_result else 0,
                        'status': 'active',  # Required for persistence
                        'created_at': datetime.now().isoformat()
                    }
                    # Check if not already in active trades AND not in closed trades
                    already_active = any(t['symbol'] == trade['symbol'] for t in st.session_state.active_trades)
                    already_closed = any(t.get('symbol') == trade['symbol'] for t in st.session_state.get('closed_trades', []))
                    
                    if not already_active and not already_closed:
                        # Save to file and reload
                        add_trade(trade)
                        st.session_state.active_trades = get_active_trades()
                        added_count += 1
            
            if added_count > 0:
                st.toast(f"🎯 Auto-added {added_count} HIGH confidence setup(s) to Monitor!", icon="🎯")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS - PRO SIGNAL CARDS
    # ═══════════════════════════════════════════════════════════════════════
    
    if st.session_state.scan_results:
        results = st.session_state.scan_results
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🔮 APPROACHING SETUPS (Predictive)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Collect approaching setups from scanned assets
        approaching_setups = []
        seen_symbols = set()  # Deduplicate by symbol
        for r in results[:20]:  # Check top 20 results
            try:
                predictions = find_approaching_setups(r['df'], r['signal'].symbol)
                for p in predictions:
                    # Skip duplicates
                    if p.symbol in seen_symbols:
                        continue
                    if p.stage == SignalStage.APPROACHING and p.distance_pct > 1.5:
                        # Apply LONG Only filter to approaching setups too
                        if get_setting("long_only") and p.direction != 'LONG':
                            continue
                        p.current_score = r['score']  # Add context
                        approaching_setups.append(p)
                        seen_symbols.add(p.symbol)
            except:
                pass
        
        # Sort by confidence and distance
        approaching_setups.sort(key=lambda x: (x.confidence, -x.distance_pct), reverse=True)
        
        if approaching_setups:
            with st.expander(f"🔮 **{len(approaching_setups)} APPROACHING SETUPS** - Set limit orders!", expanded=False):
                st.markdown("""
                <p style='color: #9d4edd; margin-bottom: 15px;'>
                    These setups are <strong>NOT active yet</strong> - price is approaching key levels. 
                    Set limit orders to catch the move early!
                </p>
                """, unsafe_allow_html=True)
                
                for signal in approaching_setups[:8]:  # Show top 8
                    st.markdown(f"""
                    <div style='background: {signal.stage_color}15; border-left: 4px solid {signal.stage_color}; 
                                padding: 12px 15px; border-radius: 8px; margin: 8px 0;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='color: {signal.stage_color}; font-weight: bold; font-size: 1.1em;'>
                                    {signal.stage_emoji} {signal.symbol}
                                </span>
                                <span style='color: #888; margin-left: 10px;'>{signal.direction}</span>
                            </div>
                            <div style='color: #888;'>
                                Confidence: <strong style='color: #fff;'>{signal.confidence}%</strong>
                            </div>
                        </div>
                        <div style='color: #ccc; margin-top: 6px;'>{signal.description}</div>
                        <div style='color: #fff; margin-top: 8px; background: #1a1a2e; padding: 8px 10px; border-radius: 4px;'>
                            <strong>Level:</strong> {signal.level_price:,.2f} 
                            <span style='color: #ffcc00;'>({signal.distance_pct:.1f}% away)</span><br>
                            <strong>Action:</strong> {signal.action}
                        </div>
                        <div style='color: #666; margin-top: 6px; font-size: 0.85em;'>
                            If triggered → Entry: ${signal.suggested_entry:,.2f} | SL: ${signal.suggested_sl:,.2f} | TP1: ${signal.suggested_tp1:,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <p style='color: #666; font-size: 0.85em; margin-top: 10px;'>
                    💡 <strong>Tip:</strong> Set limit orders at these levels. If price reaches them, you'll get better entries than chasing.
                </p>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🦈 SMART MONEY / INSTITUTIONAL ACTIVITY SUMMARY
        # ═══════════════════════════════════════════════════════════════════════
        
        # Collect institutional activity signals from all results
        stealth_accum_coins = []
        stealth_dist_coins = []
        strong_inst_buying = []
        strong_inst_selling = []
        
        for r in results:
            inst = r.get('institutional', {})
            sig = r.get('signal')
            if inst and sig:
                if inst.get('stealth_accum', {}).get('detected'):
                    stealth_accum_coins.append({
                        'symbol': r['symbol'],
                        'score': inst.get('net_score', 0),
                        'reasons': inst.get('stealth_accum', {}).get('reasons', []),
                        'entry': sig.entry,
                        'stop_loss': sig.stop_loss,
                        'tp1': sig.tp1,
                        'current_price': r['df']['Close'].iloc[-1] if r.get('df') is not None else sig.entry
                    })
                # Only show distribution if NOT long_only filter
                if inst.get('stealth_dist', {}).get('detected') and should_show_bearish():
                    stealth_dist_coins.append({
                        'symbol': r['symbol'],
                        'score': inst.get('net_score', 0),
                        'reasons': inst.get('stealth_dist', {}).get('reasons', [])
                    })
                if inst.get('net_score', 0) >= 40:
                    strong_inst_buying.append(r['symbol'])
                elif inst.get('net_score', 0) <= -40 and should_show_bearish():
                    strong_inst_selling.append(r['symbol'])
        
        # Show if there's significant institutional activity
        show_smart_money = stealth_accum_coins or (stealth_dist_coins and should_show_bearish()) or strong_inst_buying or (strong_inst_selling and should_show_bearish())
        
        if show_smart_money:
            st.markdown("## 🦈 Smart Money Activity")
            st.markdown("<p style='color: #888;'>Detected BEFORE price moves - Institutional behavior signals</p>", unsafe_allow_html=True)
            
            if stealth_accum_coins:
                with st.expander(f"🦈 **STEALTH ACCUMULATION** ({len(stealth_accum_coins)} coins) - 🟢 Bullish!", expanded=False):
                    st.markdown("""
                    <div style='background: #00d4aa22; border-left: 4px solid #00d4aa; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'>
                        <strong style='color: #00d4aa;'>What this means:</strong>
                        <span style='color: #ccc;'>Volume is building while price stays flat. Smart money is quietly buying before a breakout.</span>
                        <div style='color: #00ff88; margin-top: 8px;'><strong>✅ ACTION:</strong> These are LONG opportunities - scroll down to find them in Active Signals!</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for coin in stealth_accum_coins[:5]:
                        reasons_html = " | ".join(coin['reasons'][:3])
                        st.markdown(f"""
                        <div style='background: #1a2a1a; border-left: 3px solid #00d4aa; padding: 12px 15px; border-radius: 6px; margin: 8px 0;'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <strong style='color: #00d4aa; font-size: 1.1em;'>🦈 {coin['symbol']}</strong>
                                <span style='color: #00ff88;'>Score: +{coin['score']}</span>
                            </div>
                            <div style='color: #888; font-size: 0.85em; margin-top: 4px;'>{reasons_html}</div>
                            <div style='background: #0a1a0a; padding: 8px; border-radius: 4px; margin-top: 8px;'>
                                <span style='color: #888;'>Entry:</span> <strong style='color: #00d4ff;'>{fmt_price(coin['entry'])}</strong>
                                <span style='color: #888; margin-left: 15px;'>SL:</span> <strong style='color: #ff6666;'>{fmt_price(coin['stop_loss'])}</strong>
                                <span style='color: #888; margin-left: 15px;'>TP1:</span> <strong style='color: #00d4aa;'>{fmt_price(coin['tp1'])}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<p style='color: #00d4aa; margin-top: 10px;'>👇 <strong>Find these coins in Active Signals below for full details and charts!</strong></p>", unsafe_allow_html=True)
            
            # Only show distribution if NOT filtering for LONG only
            if stealth_dist_coins and should_show_bearish():
                with st.expander(f"🦈 **STEALTH DISTRIBUTION** ({len(stealth_dist_coins)} coins) - 🔴 Bearish!", expanded=False):
                    st.markdown("""
                    <div style='background: #ff444422; border-left: 4px solid #ff4444; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'>
                        <strong style='color: #ff4444;'>What this means:</strong>
                        <span style='color: #ccc;'>Volume is building while price stays flat or rises slightly. Smart money is quietly selling into strength. These often precede drops.</span>
                        <div style='color: #ffaa00; margin-top: 8px;'><strong>⚠️ ACTION:</strong> AVOID buying these or consider SHORT positions</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for coin in stealth_dist_coins[:5]:
                        reasons_html = " | ".join(coin['reasons'][:3])
                        st.markdown(f"""
                        <div style='background: #2a1a1a; border-left: 3px solid #ff4444; padding: 10px 12px; border-radius: 6px; margin: 5px 0;'>
                            <strong style='color: #ff4444;'>🦈 {coin['symbol']}</strong>
                            <span style='color: #888; margin-left: 10px;'>Score: {coin['score']}</span>
                            <div style='color: #888; font-size: 0.85em; margin-top: 4px;'>{reasons_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if strong_inst_buying and not stealth_accum_coins:
                st.success(f"**Strong Institutional Buying:** {', '.join(strong_inst_buying[:5])}")
            
            if strong_inst_selling and not stealth_dist_coins and should_show_bearish():
                st.warning(f"**Strong Institutional Selling:** {', '.join(strong_inst_selling[:5])}")
            
            st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🎯 ACTIVE SETUPS (Ready Now)
        # ═══════════════════════════════════════════════════════════════════════
        # (Results already sorted by combined_score - highest first)
        
        st.markdown(f"## 🎯 {len(results)} Active Signals")
        st.markdown("<p style='color: #888;'>Price is AT these levels - entry available now (sorted by score, highest first)</p>", unsafe_allow_html=True)
        
        for r in results:
            sig = r['signal']
            bd = r['breakdown']  # breakdown
            mf = r['money_flow']
            sessions = r['sessions']
            
            # Get mode + timeframe label (e.g., "📊 Day Trade 15m")
            display_label = r.get('display_label', f"📊 {r.get('timeframe', '15m')}")
            timeframe = r.get('timeframe', '15m')
            
            # Signal card header with grade AND mode/timeframe
            header_color = bd['grade_color']
            
            # Get PREDICTIVE score info for header
            pred_result = r.get('predictive_result')
            pred_score = pred_result.final_score if pred_result else r.get('score', 0)
            pred_action = pred_result.final_action if pred_result else ''
            pred_direction = pred_result.direction if pred_result else ''
            
            # Color based on action
            if pred_result:
                if 'STRONG BUY' in pred_action:
                    action_emoji = "🚀"
                    action_color = "#00ff88"
                elif 'BUY' in pred_action:
                    action_emoji = "✅"
                    action_color = "#00d4aa"
                elif 'SELL' in pred_action:
                    action_emoji = "🔴"
                    action_color = "#ff6b6b"
                elif 'WAIT' in pred_action:
                    action_emoji = "⏳"
                    action_color = "#ffcc00"
                else:
                    action_emoji = "📊"
                    action_color = "#888"
            else:
                action_emoji = "📊"
                action_color = "#888"
            
            with st.expander(f"{bd['grade_emoji']} **{sig.symbol}** | {display_label} | {action_emoji} {pred_action} ({pred_score}/100)", expanded=False):
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 SIGNAL HEADER BOX (Same format as Single Analysis)
                # ═══════════════════════════════════════════════════════════
                
                pred_result = r.get('predictive_result')
                if pred_result:
                    # Get whale data for combined learning
                    whale_data = r.get('whale', {})
                    top_trader = whale_data.get('top_trader_ls', {})
                    retail_data = whale_data.get('retail_ls', {})
                    whale_pct = top_trader.get('long_pct', 50) if isinstance(top_trader, dict) else 50
                    retail_pct = retail_data.get('long_pct', 50) if isinstance(retail_data, dict) else 50
                    
                    oi_data = whale_data.get('open_interest', {})
                    oi_change = oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0
                    price_change = whale_data.get('real_whale_data', {}).get('price_change_24h', 0) if whale_data.get('real_whale_data') else 0
                    
                    # Get money flow phase
                    money_flow_phase = mf.get('flow_status', '')
                    
                    # Generate Combined Learning
                    from core.unified_scoring import generate_combined_learning
                    from utils.formatters import get_setup_info, render_signal_header_html, render_combined_learning_html
                    
                    combined_learning = generate_combined_learning(
                        signal_name=pred_result.final_action,
                        direction=pred_result.direction,
                        whale_pct=whale_pct,
                        retail_pct=retail_pct,
                        oi_change=oi_change,
                        price_change=price_change,
                        money_flow_phase=money_flow_phase,
                        structure_type=bd.get('structure', 'Mixed'),
                        position_pct=pred_result.position_pct if hasattr(pred_result, 'position_pct') else 50,
                        ta_score=pred_result.ta_score,
                    )
                    
                    # ═══════════════════════════════════════════════════════════
                    # 🎯 SINGLE SOURCE OF TRUTH: Use pred_result.trade_direction directly!
                    # No mapping needed - trade_direction is already 'LONG', 'SHORT', 'WAIT', etc.
                    # ═══════════════════════════════════════════════════════════
                    setup_direction_scan = pred_result.trade_direction if hasattr(pred_result, 'trade_direction') else 'WAIT'
                    
                    setup_text, setup_color, conclusion_color, conclusion_bg = get_setup_info(setup_direction_scan)
                    
                    # Render signal header
                    header_html = render_signal_header_html(
                        symbol=sig.symbol,
                        action_word=pred_result.final_action,
                        score=pred_result.final_score,
                        summary=pred_result.final_summary,
                        direction_score=pred_result.direction_score,
                        squeeze_score=pred_result.squeeze_score,
                        timing_score=pred_result.timing_score,
                        setup_text=setup_text,
                        setup_color=setup_color,
                        bg_color="#1a1a2e"
                    )
                    st.markdown(header_html, unsafe_allow_html=True)
                    
                    # Combined Learning expander
                    cl_html = render_combined_learning_html(
                        conclusion=combined_learning['conclusion'],
                        conclusion_action=combined_learning['conclusion_action'],
                        conclusion_color=conclusion_color,
                        conclusion_bg=conclusion_bg,
                        stories=combined_learning['stories'],
                        is_squeeze=combined_learning['is_squeeze'],
                        squeeze_type=combined_learning.get('squeeze_type'),
                        direction=combined_learning['direction'],
                        has_conflict=combined_learning.get('has_conflict', False),
                        conflicts=combined_learning.get('conflicts', []),
                        is_capitulation_long=combined_learning.get('is_capitulation_long', False),
                        whale_pct=whale_pct
                    )
                    
                    with st.expander("📖 **COMBINED LEARNING** - What's Really Happening?", expanded=False):
                        st.markdown(cl_html['conclusion_html'], unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("**📊 THE FULL STORY:**")
                        st.markdown(cl_html['stories_html'], unsafe_allow_html=True)
                        
                        if cl_html['squeeze_html']:
                            st.markdown(cl_html['squeeze_html'], unsafe_allow_html=True)
                        if cl_html['conflict_html']:
                            st.markdown(cl_html['conflict_html'], unsafe_allow_html=True)
                        if cl_html['capitulation_html']:
                            st.markdown(cl_html['capitulation_html'], unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 📖 QUICK NARRATIVE - What's happening in plain English
                # ═══════════════════════════════════════════════════════════
                
                # Build quick narrative based on detected factors
                narrative_parts = []
                
                # Money flow narrative
                if mf.get('is_accumulating'):
                    narrative_parts.append("💰 **Money is flowing IN** - Buyers are stepping in with volume support")
                elif mf.get('is_distributing'):
                    narrative_parts.append("💸 **Money is flowing OUT** - Sellers are active, be cautious")
                
                # SMC/Structure narrative
                if bd.get('order_block'):
                    ob_type = bd['order_block'].get('type', 'bullish')
                    if 'bullish' in ob_type.lower():
                        narrative_parts.append("🏦 **At Institutional Demand Zone** - Price is at an Order Block where big players previously bought")
                
                # Volume narrative
                if bd.get('volume_analysis', {}).get('volume_spike'):
                    narrative_parts.append("📊 **Volume Spike Detected** - Unusual activity suggests smart money is involved")
                
                # Momentum narrative
                mfi = mf.get('mfi', 50)
                if mfi < 30:
                    narrative_parts.append("📉 **Oversold Conditions** - Price has dropped significantly, potential bounce zone")
                elif mfi > 70:
                    narrative_parts.append("📈 **Overbought Conditions** - Price extended, may see pullback before continuation")
                
                # Entry quality narrative
                if bd['score'] >= 80:
                    narrative_parts.append("⭐ **High Quality Setup** - Multiple confirmations align for a strong trade")
                elif bd['score'] >= 60:
                    narrative_parts.append("✅ **Good Setup** - Solid confirmation with acceptable risk")
                
                # If no specific narrative, add default based on grade
                if not narrative_parts:
                    if bd['grade'] in ['A+', 'A']:
                        narrative_parts.append("✅ **Conditions are favorable** - Technical factors support this trade")
                    elif bd['grade'] == 'B':
                        narrative_parts.append("👀 **Decent setup** - Some positive factors but exercise caution")
                    else:
                        narrative_parts.append("⚠️ **Watch carefully** - Limited confirmation, higher risk")
                
                # Display narrative box
                if narrative_parts:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #1a2a3a, #0a1a2a); border-left: 4px solid #00d4ff; 
                                border-radius: 8px; padding: 12px 15px; margin-bottom: 15px;'>
                        <div style='color: #00d4ff; font-weight: bold; margin-bottom: 8px;'>📖 What's Happening</div>
                        <div style='color: #ccc; line-height: 1.8;'>
                            {'<br>'.join(narrative_parts)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 3-LAYER PREDICTIVE SCORE DISPLAY
                # ═══════════════════════════════════════════════════════════
                
                if pred_result:
                    # Colors for each layer
                    dir_color = "#00ff88" if pred_result.direction in ["BULLISH"] else "#00d4aa" if pred_result.direction == "WEAK_BULLISH" else "#ff6b6b" if pred_result.direction == "BEARISH" else "#ffcc00"
                    squeeze_color = "#00ff88" if pred_result.squeeze_potential == "HIGH" else "#00d4aa" if pred_result.squeeze_potential == "MEDIUM" else "#ff6b6b" if pred_result.squeeze_potential == "CONFLICT" else "#888"
                    timing_color = "#00ff88" if pred_result.entry_timing == "NOW" else "#ffcc00"
                    final_color = "#00ff88" if pred_result.final_score >= 75 else "#00d4aa" if pred_result.final_score >= 60 else "#ffcc00" if pred_result.final_score >= 45 else "#ff9500"
                    
                    # Truncate reasons for display
                    dir_reason = pred_result.direction_reason
                    timing_reason = pred_result.timing_reason
                    
                    # 3 Layers in columns
                    layer_cols = st.columns(3)
                    
                    with layer_cols[0]:
                        st.markdown(f"""
                        <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; border-left: 4px solid {dir_color}; min-height: 160px;'>
                            <div style='color: #888; font-size: 0.8em;'>📈 LAYER 1: Direction</div>
                            <div style='color: {dir_color}; font-size: 1.3em; font-weight: bold; margin: 5px 0;'>{pred_result.direction}</div>
                            <div style='color: #aaa; font-size: 0.8em;'>{pred_result.direction_confidence} confidence</div>
                            <div style='color: {dir_color}; font-size: 1.6em; font-weight: bold; margin-top: 8px;'>{pred_result.direction_score}/40</div>
                            <div style='color: #666; font-size: 0.7em; margin-top: 8px; border-top: 1px solid #333; padding-top: 6px;'>{dir_reason}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with layer_cols[1]:
                        st.markdown(f"""
                        <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; border-left: 4px solid {squeeze_color}; min-height: 160px;'>
                            <div style='color: #888; font-size: 0.8em;'>🔥 LAYER 2: Squeeze</div>
                            <div style='color: {squeeze_color}; font-size: 1.3em; font-weight: bold; margin: 5px 0;'>{pred_result.squeeze_potential}</div>
                            <div style='color: #aaa; font-size: 0.8em;'>W:{whale_pct:.0f}% vs R:{retail_pct:.0f}%</div>
                            <div style='color: {squeeze_color}; font-size: 1.6em; font-weight: bold; margin-top: 8px;'>{pred_result.squeeze_score}/30</div>
                            <div style='color: #666; font-size: 0.7em; margin-top: 8px; border-top: 1px solid #333; padding-top: 6px;'>Divergence: {pred_result.divergence_pct:+.0f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with layer_cols[2]:
                        # Move position color
                        move_color_scan = "#00ff88" if pred_result.move_position == "EARLY" else "#ffcc00" if pred_result.move_position == "MIDDLE" else "#ff6b6b" if pred_result.move_position == "LATE" else "#888"
                        st.markdown(f"""
                        <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; border-left: 4px solid {timing_color}; min-height: 160px;'>
                            <div style='color: #888; font-size: 0.8em;'>⏰ LAYER 3: Entry (TA+Pos)</div>
                            <div style='color: {timing_color}; font-size: 1.3em; font-weight: bold; margin: 5px 0;'>{pred_result.entry_timing}</div>
                            <div style='color: #aaa; font-size: 0.8em;'>TA: {pred_result.ta_score} | <span style='color: {move_color_scan};'>{pred_result.move_position}</span></div>
                            <div style='color: {timing_color}; font-size: 1.6em; font-weight: bold; margin-top: 8px;'>{pred_result.timing_score}/30</div>
                            <div style='color: #666; font-size: 0.7em; margin-top: 8px; border-top: 1px solid #333; padding-top: 6px;'>{timing_reason}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Summary with more top margin
                    st.markdown(f"""
                    <div style='background: {final_color}22; border-radius: 8px; padding: 12px; text-align: center; margin-top: 20px; margin-bottom: 15px;'>
                        <div style='color: {final_color}; font-size: 1em;'>{pred_result.final_summary}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Raw Data
                    oi_data = whale_data.get('open_interest', {})
                    oi_change = oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0
                    price_change = whale_data.get('real_whale_data', {}).get('price_change_24h', 0) if whale_data.get('real_whale_data') else 0
                    ta_score_raw = pred_result.ta_score
                    
                    oi_color = "#00ff88" if oi_change > 0.5 else "#ff6b6b" if oi_change < -0.5 else "#888"
                    price_clr = "#00ff88" if price_change > 0 else "#ff6b6b" if price_change < 0 else "#888"
                    
                    st.markdown("**📊 Raw Data**")
                    data_cols = st.columns(5)
                    with data_cols[0]:
                        st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.75em;'>OI 24h</div><div style='color:{oi_color};font-size:1.1em;font-weight:bold;'>{oi_change:+.1f}%</div></div>", unsafe_allow_html=True)
                    with data_cols[1]:
                        st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.75em;'>Price 24h</div><div style='color:{price_clr};font-size:1.1em;font-weight:bold;'>{price_change:+.1f}%</div></div>", unsafe_allow_html=True)
                    with data_cols[2]:
                        st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.75em;'>🐋 Whales</div><div style='color:#00d4ff;font-size:1.1em;font-weight:bold;'>{whale_pct:.0f}%</div></div>", unsafe_allow_html=True)
                    with data_cols[3]:
                        st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.75em;'>🐑 Retail</div><div style='color:#ffcc00;font-size:1.1em;font-weight:bold;'>{retail_pct:.0f}%</div></div>", unsafe_allow_html=True)
                    with data_cols[4]:
                        st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.75em;'>TA Score</div><div style='color:#aaa;font-size:1.1em;font-weight:bold;'>{ta_score_raw}</div></div>", unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 📊 MARKET CONTEXT - Structure + Money Flow + Move Position
                # ═══════════════════════════════════════════════════════════
                
                smc_data = r.get('smc', {})
                mf_data = r.get('money_flow', {})
                
                # Get market structure
                structure_info = smc_data.get('structure', {})
                structure_type = structure_info.get('structure', 'Unknown')
                swing_high = structure_info.get('last_swing_high', 0)
                swing_low = structure_info.get('last_swing_low', 0)
                
                # Use move position from pred_result (already calculated in scoring)
                move_label = pred_result.move_position if pred_result else "N/A"
                position_in_range = pred_result.move_position_pct if pred_result else 50
                
                # Move position styling - CONTEXT AWARE!
                # Check if structure is consolidating/ranging (no clear trend)
                is_consolidating = structure_type in ['Consolidating', 'Ranging', 'Mixed', 'Unknown']
                
                # Use ACTUAL position percentage for tips, not just label
                # (Label can be "EARLY" for shorts at highs, but tip should reflect price location)
                
                if move_label == "EARLY":
                    move_color = "#00ff88"
                    move_emoji = "🟢"
                    if is_consolidating:
                        # Check actual position to give correct tip
                        if position_in_range <= 30:
                            move_tip = "At range lows - good for range longs"
                        elif position_in_range >= 70:
                            move_tip = "At range highs - early for shorts"
                        else:
                            move_tip = "Good positioning for this setup"
                    else:
                        move_tip = "Great entry - catching the move early!"
                elif move_label == "MIDDLE":
                    move_color = "#ffcc00"
                    move_emoji = "🟡"
                    if is_consolidating:
                        move_tip = "Mid-range - wait for edge of range"
                    else:
                        move_tip = "OK entry - some room left"
                elif move_label in ["LATE", "CHASING"]:
                    move_color = "#ff6b6b"
                    move_emoji = "🔴"
                    if move_label == "CHASING":
                        move_color = "#ff4444"
                        move_emoji = "🚫"
                    if is_consolidating:
                        # Check actual position to give correct tip
                        if position_in_range >= 70:
                            move_tip = "At range highs - risky for longs"
                        elif position_in_range <= 30:
                            move_tip = "At range lows - risky for shorts"
                        else:
                            move_tip = "Chasing - wait for better entry"
                    else:
                        move_tip = "Chasing - wait for pullback"
                elif move_label in ["NEAR HIGH", "NEAR LOW", "MID-RANGE"]:
                    # Range-specific labels
                    if move_label == "NEAR HIGH":
                        move_color = "#ff9500"
                        move_emoji = "⬆️"
                        move_tip = "At range highs - good for shorts or wait"
                    elif move_label == "NEAR LOW":
                        move_color = "#00d4aa"
                        move_emoji = "⬇️"
                        move_tip = "At range lows - good for longs"
                    else:
                        move_color = "#888"
                        move_emoji = "↔️"
                        move_tip = "Mid-range - wait for edge"
                else:
                    move_color = "#888"
                    move_emoji = "↔️"
                    move_tip = "No clear trend"
                
                # Determine structure color and emoji
                if structure_type == 'Bullish':
                    struct_color = "#00ff88"
                    struct_emoji = "📈"
                    struct_detail = "HH + HL"
                elif structure_type == 'Bearish':
                    struct_color = "#ff6b6b"
                    struct_emoji = "📉"
                    struct_detail = "LH + LL"
                else:
                    struct_color = "#ffcc00"
                    struct_emoji = "↔️"
                    struct_detail = "Mixed"
                
                # Get money flow status WITH CONTEXT (position-aware!)
                is_accumulating = mf_data.get('is_accumulating', False)
                is_distributing = mf_data.get('is_distributing', False)
                
                # Use contextual flow analysis
                scanner_flow_context = get_money_flow_context(
                    is_accumulating=is_accumulating,
                    is_distributing=is_distributing,
                    position_pct=position_in_range,
                    structure_type=structure_type
                )
                
                flow_status = scanner_flow_context.phase
                flow_color = scanner_flow_context.phase_color
                flow_emoji = scanner_flow_context.phase_emoji
                flow_detail = scanner_flow_context.phase_detail
                
                # Build warning HTML if exists
                warning_html = ""
                if scanner_flow_context.warning:
                    warning_html = f"<div style='color: #ff9500; font-size: 0.75em; margin-top: 3px;'>⚠️ {scanner_flow_context.warning}</div>"
                
                # Get timeframe for display
                scanner_tf_display = r.get('timeframe', '15m')
                
                # Full MARKET CONTEXT box (matching Single Analysis)
                st.markdown(f"""
                <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #333;'>
                    <div style='color: #00d4ff; font-weight: bold; margin-bottom: 12px;'>📊 MARKET CONTEXT ({scanner_tf_display})</div>
                    <div style='display: flex; gap: 20px;'>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Structure</div>
                            <div style='color: {struct_color}; font-size: 1.1em; font-weight: bold;'>{struct_emoji} {structure_type}</div>
                            <div style='color: #666; font-size: 0.75em;'>{struct_detail}</div>
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Money Flow</div>
                            <div style='color: {flow_color}; font-size: 1.1em; font-weight: bold;'>{flow_emoji} {flow_status}</div>
                            <div style='color: #666; font-size: 0.75em;'>{flow_detail}</div>{warning_html}
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Move Position</div>
                            <div style='color: {move_color}; font-size: 1.1em; font-weight: bold;'>{move_emoji} {move_label}</div>
                            <div style='color: #666; font-size: 0.75em;'>{position_in_range:.0f}% of range</div>
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Swing Levels</div>
                            <div style='color: #aaa; font-size: 0.85em;'>H: <span style='color: #00d4aa;'>${swing_high:,.4f}</span></div>
                            <div style='color: #aaa; font-size: 0.85em;'>L: <span style='color: #ff6b6b;'>${swing_low:,.4f}</span></div>
                        </div>
                    </div>
                    <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;'>
                        <span style='color: {move_color}; font-size: 0.9em;'>💡 {move_tip}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 MULTI-TIMEFRAME TIP (Scanner)
                # ═══════════════════════════════════════════════════════════
                try:
                    from core.money_flow_context import get_multi_timeframe_tip
                    scanner_tf = r.get('timeframe', '15m')
                    scanner_mtf_tip = get_multi_timeframe_tip(
                        current_tf=scanner_tf,
                        current_phase=flow_status,
                        higher_tf_phase=None,  # Would need to fetch higher TF
                        predictive_signal=pred_result.final_action if pred_result else None
                    )
                    if scanner_mtf_tip.has_tip:
                        st.markdown(f"""
                        <div style='background: {scanner_mtf_tip.tip_color}15; border-left: 4px solid {scanner_mtf_tip.tip_color}; 
                                    padding: 12px; border-radius: 8px; margin: 10px 0;'>
                            <div style='color: {scanner_mtf_tip.tip_color}; font-weight: bold; margin-bottom: 5px;'>
                                {scanner_mtf_tip.tip_emoji} {scanner_mtf_tip.tip_title}
                            </div>
                            <div style='color: #ccc; font-size: 0.9em;'>
                                {scanner_mtf_tip.tip_detail}
                            </div>
                            <div style='color: {scanner_mtf_tip.tip_color}; font-size: 0.85em; font-style: italic; margin-top: 8px;'>
                                👉 {scanner_mtf_tip.action}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    pass  # Silently skip MTF tip if error
                
                # 📚 Learn expander for Money Flow - with clickable phases!
                with st.expander(f"📚 What is {flow_status}?", expanded=False):
                    edu = get_money_flow_education(flow_status)
                    st.markdown(f"**{edu.get('emoji', '')} {edu.get('title', flow_status)}**")
                    st.markdown(f"*{edu.get('short', '')}*")
                    st.markdown(edu.get('description', ''))
                    
                    if edu.get('wyckoff'):
                        st.info(f"📖 Wyckoff: {edu.get('wyckoff')}")
                    
                    # Structure vs Money Flow explanation
                    st.markdown("---")
                    st.markdown("""
                    **💡 Structure vs Money Flow:**
                    - **Structure (Consolidating):** Price pattern - no clear HH/HL or LH/LL
                    - **Money Flow (Consolidation):** Volume direction - no clear buying/selling
                    
                    *Both can show "consolidation" at same time = strong WAIT signal*
                    """)
                    
                    # Clickable phases (compact for Scanner)
                    st.markdown("**📚 Other Phases:**")
                    phase_options = list(MONEY_FLOW_EDUCATION.keys())
                    selected_phase = st.selectbox("Select phase to learn:", phase_options, index=phase_options.index(flow_status) if flow_status in phase_options else 0, key=f"scanner_phase_{sig.symbol}")
                    
                    if selected_phase != flow_status:
                        sel_edu = get_money_flow_education(selected_phase)
                        st.markdown(f"**{sel_edu.get('emoji', '')} {sel_edu.get('title', selected_phase)}**")
                        st.markdown(f"*{sel_edu.get('short', '')}*")
                        st.markdown(sel_edu.get('description', ''))
                
                # ═══════════════════════════════════════════════════════════
                # 📖 COMBINED LEARNING (Scanner - Collapsed)
                # ═══════════════════════════════════════════════════════════
                
                if pred_result:
                    try:
                        from core.unified_scoring import generate_combined_learning
                        
                        # Get data for combined learning
                        whale_data_cl = r.get('whale', {})
                        oi_data_cl = whale_data_cl.get('open_interest', {})
                        oi_change_cl = oi_data_cl.get('change_24h', 0) if isinstance(oi_data_cl, dict) else 0
                        price_change_cl = whale_data_cl.get('real_whale_data', {}).get('price_change_24h', 0) if whale_data_cl.get('real_whale_data') else 0
                        
                        top_trader_cl = whale_data_cl.get('top_trader_ls', {})
                        retail_cl = whale_data_cl.get('retail_ls', {})
                        whale_pct_cl = top_trader_cl.get('long_pct', 50) if isinstance(top_trader_cl, dict) else 50
                        retail_pct_cl = retail_cl.get('long_pct', 50) if isinstance(retail_cl, dict) else 50
                        
                        combined_learning_scanner = generate_combined_learning(
                            signal_name=pred_result.final_action,
                            direction=pred_result.direction,
                            whale_pct=whale_pct_cl,
                            retail_pct=retail_pct_cl,
                            oi_change=oi_change_cl,
                            price_change=price_change_cl,
                            money_flow_phase=flow_status,
                            structure_type=structure_type,
                            position_pct=position_in_range,
                            ta_score=pred_result.ta_score,
                        )
                        
                        # Conclusion color
                        if combined_learning_scanner['direction'] == 'LONG':
                            cl_color = "#00ff88"
                            cl_bg = "rgba(0, 255, 136, 0.1)"
                        elif combined_learning_scanner['direction'] == 'SHORT':
                            cl_color = "#ff6b6b"
                            cl_bg = "rgba(255, 107, 107, 0.1)"
                        else:
                            cl_color = "#ffcc00"
                            cl_bg = "rgba(255, 204, 0, 0.1)"
                        
                        with st.expander("📖 **COMBINED LEARNING** - Full Story", expanded=False):
                            # CONCLUSION AT TOP
                            st.markdown(f"""
                            <div style='background: {cl_bg}; border: 2px solid {cl_color}; border-radius: 10px; padding: 12px; margin-bottom: 12px;'>
                                <div style='color: {cl_color}; font-weight: bold;'>🎯 CONCLUSION</div>
                                <div style='color: #fff; font-size: 0.95em; margin-top: 5px;'>{combined_learning_scanner['conclusion']}</div>
                                <div style='color: {cl_color}; font-size: 0.9em; margin-top: 5px;'>Action: <strong>{combined_learning_scanner['conclusion_action']}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Stories
                            for title, content in combined_learning_scanner['stories']:
                                st.markdown(f"""
                                <div style='background: #1a1a2e; border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 3px solid #444;'>
                                    <div style='color: #888; font-size: 0.8em;'>{title}</div>
                                    <div style='color: #ccc; font-size: 0.9em;'>{content}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Squeeze alert if applicable
                            if combined_learning_scanner['is_squeeze']:
                                sq_color = "#00ff88" if combined_learning_scanner['squeeze_type'] == 'SHORT' else "#ff6b6b"
                                st.markdown(f"""
                                <div style='background: rgba(255, 0, 255, 0.1); border: 2px solid #ff00ff; border-radius: 8px; padding: 12px; margin-top: 10px;'>
                                    <div style='color: #ff00ff; font-weight: bold;'>⚡ SQUEEZE: {combined_learning_scanner['squeeze_type']}S getting liquidated!</div>
                                    <div style='color: #fff; font-size: 0.9em;'>Your trade: <strong style='color: {sq_color};'>{combined_learning_scanner['direction']}</strong></div>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        pass  # Skip if error
                
                # ═══════════════════════════════════════════════════════════
                # TOP ROW: Key Info + Chart
                # ═══════════════════════════════════════════════════════════
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Grade badge with mode/timeframe
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, {header_color}22, {header_color}11); 
                                border: 2px solid {header_color}; border-radius: 12px; padding: 15px; text-align: center;'>
                        <h1 style='margin: 0; color: {header_color}; font-size: 3em;'>{bd['grade']}</h1>
                        <p style='margin: 5px 0 0 0; color: {header_color};'>{bd['grade_label']}</p>
                        <p style='margin: 5px 0 0 0; color: #888;'>Score: {bd['score']}/100</p>
                        <p style='margin: 8px 0 0 0; background: #1a1a2e; padding: 4px 8px; border-radius: 4px; 
                                  color: #00d4ff; font-size: 0.85em;'>{display_label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action recommendation
                    st.markdown(f"### {bd['action_recommendation']}")
                    
                    # Entry and Stop boxes - dark theme
                    st.markdown(f"""
                    <div style='background: #1a2a3a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4ff;'>
                        <span style='color: #888;'>Entry:</span> 
                        <span style='color: #00d4ff; font-weight: bold;'>{fmt_price(sig.entry)}</span>
                    </div>
                    <div style='background: #2a1a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #ff4444;'>
                        <span style='color: #888;'>Stop:</span> 
                        <span style='color: #ff6666; font-weight: bold;'>{fmt_price(sig.stop_loss)} ({sig.risk_pct:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # TP Levels with R:R for EACH target - this is what matters!
                    tp1_roi = calc_roi(sig.tp1, sig.entry)
                    tp2_roi = calc_roi(sig.tp2, sig.entry)
                    tp3_roi = calc_roi(sig.tp3, sig.entry)
                    
                    # Get R:R values (with fallback for old signals)
                    rr_tp1 = getattr(sig, 'rr_tp1', tp1_roi / sig.risk_pct if sig.risk_pct > 0 else 0)
                    rr_tp2 = getattr(sig, 'rr_tp2', tp2_roi / sig.risk_pct if sig.risk_pct > 0 else 0)
                    rr_tp3 = getattr(sig, 'rr_tp3', tp3_roi / sig.risk_pct if sig.risk_pct > 0 else 0)
                    
                    # Color code R:R - green if >= 1, yellow if >= 0.5, red if < 0.5
                    rr1_color = "#00d4aa" if rr_tp1 >= 1 else "#ffcc00" if rr_tp1 >= 0.5 else "#ff4444"
                    rr2_color = "#00d4aa" if rr_tp2 >= 1 else "#ffcc00" if rr_tp2 >= 0.5 else "#ff4444"
                    rr3_color = "#00d4aa" if rr_tp3 >= 1 else "#ffcc00" if rr_tp3 >= 0.5 else "#ff4444"
                    
                    st.markdown(f"""
                    <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4aa;'>
                        <span style='color: #888;'>TP1:</span> 
                        <span style='color: #00d4aa; font-weight: bold;'>{fmt_price(sig.tp1)}</span>
                        <span style='color: #00aa88;'>(+{tp1_roi:.1f}%)</span>
                        <span style='color: {rr1_color}; float: right; font-weight: bold;'>R:R {rr_tp1:.1f}:1</span>
                    </div>
                    <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4aa;'>
                        <span style='color: #888;'>TP2:</span> 
                        <span style='color: #00d4aa; font-weight: bold;'>{fmt_price(sig.tp2)}</span>
                        <span style='color: #00aa88;'>(+{tp2_roi:.1f}%)</span>
                        <span style='color: {rr2_color}; float: right; font-weight: bold;'>R:R {rr_tp2:.1f}:1</span>
                    </div>
                    <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4aa;'>
                        <span style='color: #888;'>TP3:</span> 
                        <span style='color: #00d4aa; font-weight: bold;'>{fmt_price(sig.tp3)}</span>
                        <span style='color: #00aa88;'>(+{tp3_roi:.1f}%)</span>
                        <span style='color: {rr3_color}; float: right; font-weight: bold;'>R:R {rr_tp3:.1f}:1</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # R:R Quality indicator
                    if rr_tp1 >= 1:
                        rr_status = "✅ Good R:R at TP1"
                        rr_status_color = "#00d4aa"
                    elif rr_tp1 >= 0.5:
                        rr_status = "⚠️ Marginal R:R at TP1"
                        rr_status_color = "#ffcc00"
                    else:
                        rr_status = "❌ Poor R:R at TP1"
                        rr_status_color = "#ff4444"
                    
                    st.markdown(f"""
                    <div style='background: {rr_status_color}22; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid {rr_status_color};'>
                        <span style='color: {rr_status_color}; font-weight: bold;'>{rr_status}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # 📊 VOLATILITY WARNING - Is TP realistic for this timeframe?
                    # ═══════════════════════════════════════════════════════════
                    
                    # Calculate ATR-based volatility (typical move per candle)
                    try:
                        df = r['df']
                        if df is not None and len(df) >= 14:
                            # Calculate ATR as % of price
                            high = df['High'].values
                            low = df['Low'].values
                            close = df['Close'].values
                            
                            tr = []
                            for i in range(1, len(df)):
                                tr.append(max(
                                    high[i] - low[i],
                                    abs(high[i] - close[i-1]),
                                    abs(low[i] - close[i-1])
                                ))
                            
                            atr = sum(tr[-14:]) / 14  # 14-period ATR
                            current_price = close[-1]
                            atr_pct = (atr / current_price) * 100  # ATR as % of price
                            
                            # Expected candles to TP1 based on ATR
                            # Typical move per candle ≈ 0.5-1x ATR
                            candles_to_tp1 = tp1_roi / (atr_pct * 0.7) if atr_pct > 0 else 10
                            
                            # Timeframe to hours mapping
                            tf_hours = {
                                '1m': 1/60, '5m': 5/60, '15m': 0.25, '1h': 1,
                                '4h': 4, '1d': 24, '1w': 168
                            }
                            hours_per_candle = tf_hours.get(timeframe, 1)
                            estimated_hours = candles_to_tp1 * hours_per_candle
                            
                            # Volatility rating
                            if atr_pct > 5:
                                vol_label = "🔥 EXTREME"
                                vol_color = "#ff4444"
                            elif atr_pct > 3:
                                vol_label = "⚡ HIGH"
                                vol_color = "#ff9500"
                            elif atr_pct > 1.5:
                                vol_label = "📊 MODERATE"
                                vol_color = "#ffcc00"
                            else:
                                vol_label = "🐢 LOW"
                                vol_color = "#00d4aa"
                            
                            # Warning if TP1 seems ambitious
                            if candles_to_tp1 > 20:  # More than 20 candles
                                warning_msg = f"⚠️ TP1 may take longer than expected (~{candles_to_tp1:.0f} candles / ~{estimated_hours:.1f}h)"
                                warning_color = "#ff9500"
                            elif candles_to_tp1 > 10:
                                warning_msg = f"📊 TP1 is ~{candles_to_tp1:.0f} candles away (~{estimated_hours:.1f}h)"
                                warning_color = "#ffcc00"
                            else:
                                warning_msg = f"✅ TP1 within typical range (~{candles_to_tp1:.0f} candles)"
                                warning_color = "#00d4aa"
                            
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 10px 12px; border-radius: 6px; margin: 8px 0; border: 1px solid #333;'>
                                <div style='color: #888; font-size: 0.85em; margin-bottom: 6px;'>📊 Volatility Analysis</div>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <span style='color: {vol_color};'>{vol_label} ({atr_pct:.1f}% ATR)</span>
                                    <span style='color: {warning_color}; font-size: 0.85em;'>{warning_msg}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        pass  # Skip volatility warning if calculation fails
                
                with col2:
                    fig = create_trade_setup_chart(r['df'], sig, bd['score'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # ═══════════════════════════════════════════════════════════
                # GRADE PROGRESSION (C → B → A → A+)
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 📊 Signal Grade Progression")
                
                prog_cols = st.columns(4)
                for i, step in enumerate(bd['progression']):
                    with prog_cols[i]:
                        if step['achieved']:
                            st.success(f"**{step['step']}**")
                            st.markdown(f"✅ {step['name']}")
                        else:
                            st.warning(f"**{step['step']}**")
                            st.markdown(f"⏳ {step['name']}")
                        st.caption(step['detail'])
                
                # ═══════════════════════════════════════════════════════════
                # WHY THIS TRADE
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 🎯 Why This Trade")
                
                why_col1, why_col2, why_col3 = st.columns(3)
                
                with why_col1:
                    st.markdown("#### 🟢 Bullish Factors")
                    if bd['bullish_factors']:
                        for factor in bd['bullish_factors']:
                            st.markdown(f"• {factor}")
                    else:
                        st.markdown("*No strong bullish factors*")
                
                with why_col2:
                    st.markdown("#### 🔴 Bearish Factors")
                    if bd['bearish_factors']:
                        for factor in bd['bearish_factors']:
                            st.markdown(f"• {factor}")
                    else:
                        st.markdown("*No bearish factors* ✓")
                
                with why_col3:
                    st.markdown("#### ⚠️ Risk Factors")
                    if bd['risk_factors']:
                        for factor in bd['risk_factors']:
                            st.markdown(f"• {factor}")
                    else:
                        st.markdown("*Low risk setup* ✓")
                
                # ═══════════════════════════════════════════════════════════
                # 🐋 WHALE & INSTITUTIONAL ANALYSIS (UNIFIED!)
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 🐋 Whale & Institutional Analysis")
                
                try:
                    whale_data = get_whale_analysis(sig.symbol)
                    
                    # Check if we got real data
                    has_real_data = (
                        whale_data.get('signals') or 
                        whale_data.get('open_interest', {}).get('change_24h', 0) != 0 or
                        whale_data.get('funding', {}).get('rate', 0) != 0 or
                        whale_data.get('top_trader_ls', {}).get('long_pct', 50) != 50
                    )
                    
                    if has_real_data:
                        # ════════════════════════════════════════════════════════════
                        # USE NEW UNIFIED SCORING (not the broken old verdict!)
                        # ════════════════════════════════════════════════════════════
                        from core.institutional_scoring import (
                            analyze_institutional_data, get_scenario_display, TradeStatus
                        )
                        from core.education import SCENARIO_EDUCATION
                        
                        # Get raw metrics
                        oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                        price_change = whale_data.get('price_change_24h', 0)
                        whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                        retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
                        funding = whale_data.get('funding', {}).get('rate', 0)
                        
                        # Get tech direction
                        tech_dir = "BULLISH" if sig.direction == "LONG" else "BEARISH" if sig.direction == "SHORT" else "NEUTRAL"
                        
                        # Analyze with NEW unified system
                        inst_score = analyze_institutional_data(
                            oi_change, price_change, whale_long, retail_long, funding, tech_dir
                        )
                        
                        # Get scenario display info from rules engine
                        scenario_display = get_scenario_display(inst_score.scenario)
                        
                        # Map to education.py key for rich content
                        scenario_key_map = {
                            'SHORT_SQUEEZE': 'short_squeeze_setup',
                            'NEW_LONGS_WHALE_BULLISH': 'new_longs_whale_bullish',
                            'NEW_LONGS_NEUTRAL_WHALES': 'new_longs_neutral',
                            'LONG_SQUEEZE': 'long_squeeze_setup',
                            'NEW_SHORTS_WHALE_BEARISH': 'new_shorts_whale_bearish',
                            'NEW_SHORTS_NEUTRAL_WHALES': 'new_shorts_neutral',
                            'LONG_LIQUIDATION_WHALE_BULLISH': 'long_liquidation_whale_bullish',
                            'SHORT_COVERING_WHALE_BEARISH': 'short_covering_whale_bearish',
                            'SHORT_COVERING_NEUTRAL': 'short_covering_neutral',
                            'LONG_LIQUIDATION_NEUTRAL': 'long_liquidation_neutral',
                            'CONFLICTING_SIGNALS': 'conflicting_signals',
                            'NO_EDGE': 'no_edge'
                        }
                        edu_key = scenario_key_map.get(inst_score.scenario, 'no_edge')
                        scenario_edu = SCENARIO_EDUCATION.get(edu_key, SCENARIO_EDUCATION.get('no_edge', {}))
                        
                        # Use SCENARIO_EDUCATION for rich content
                        interpretation = scenario_edu.get('interpretation', scenario_display.get('description', ''))
                        recommendation = scenario_edu.get('recommendation', scenario_display.get('what_to_do', ''))
                        
                        # Convert markdown **bold** to HTML <b>bold</b>
                        import re
                        def md_to_html(text):
                            """Convert markdown bold to HTML bold"""
                            if not text:
                                return ''
                            # Replace **text** with <b>text</b>
                            text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
                            # Replace • with proper bullet
                            text = text.replace('•', '•')
                            return text.strip()
                        
                        interpretation = md_to_html(interpretation)
                        recommendation = md_to_html(recommendation)
                        
                        # Determine unified verdict color
                        if inst_score.direction_bias == "LONG":
                            verdict = "BULLISH"
                            v_color = "#00d4aa"
                        elif inst_score.direction_bias == "SHORT":
                            verdict = "BEARISH"
                            v_color = "#ff4444"
                        else:
                            verdict = "NEUTRAL"
                            v_color = "#888888"
                        
                        # Status color
                        status_colors = {
                            TradeStatus.READY: '#00d4aa',
                            TradeStatus.WAIT: '#ffcc00',
                            TradeStatus.AVOID: '#ff4444',
                            TradeStatus.CONFLICTING: '#ff8800'
                        }
                        status_color = status_colors.get(inst_score.status, '#888888')
                        status_text = inst_score.status.value
                        
                        # UNIFIED Verdict header
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {v_color}22, {v_color}11); 
                                    border: 2px solid {v_color}; border-radius: 12px; padding: 15px; margin-bottom: 15px;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;'>
                                <div>
                                    <span style='color: #fff; font-size: 1.2em; font-weight: bold;'>
                                        {scenario_display['emoji']} {inst_score.scenario_name}
                                    </span>
                                    <span style='color: #888; font-size: 0.9em; margin-left: 10px;'>
                                        ({inst_score.confidence} confidence)
                                    </span>
                                </div>
                                <span style='background: {status_color}; color: #000; padding: 8px 20px; border-radius: 20px; 
                                             font-weight: bold; font-size: 1.1em;'>
                                    {status_text}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ═══════════════════════════════════════════════════════════
                        # 📖 EDUCATIONAL BOXES - Same as Single Analysis
                        # ═══════════════════════════════════════════════════════════
                        edu_cols = st.columns([1, 1])
                        
                        with edu_cols[0]:
                            st.markdown(f"""
                            <div style='background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px; height: 100%;'>
                                <div style='color: #888; font-size: 0.9em; margin-bottom: 10px;'>🔍 What's Happening</div>
                                <div style='color: #ddd; font-size: 0.95em; line-height: 1.8; white-space: pre-line;'>{interpretation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with edu_cols[1]:
                            st.markdown(f"""
                            <div style='background: #1a2a1a; border: 1px solid #2a4a2a; border-radius: 12px; padding: 20px; height: 100%;'>
                                <div style='color: #00d4aa; font-size: 0.9em; margin-bottom: 10px;'>🎯 What To Do</div>
                                <div style='color: #ddd; font-size: 0.95em; line-height: 1.8; white-space: pre-line;'>{recommendation}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ═══════════════════════════════════════════════════════════
                        # 📊 METRIC CARDS WITH INFO TOOLTIPS (Same as Single Analysis)
                        # ═══════════════════════════════════════════════════════════
                        
                        st.markdown("#### 📊 Raw Data (Click ℹ️ to learn)")
                        
                        whale_cols = st.columns(5)
                        
                        # OI Change Card
                        with whale_cols[0]:
                            oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                            oi_color = "#00d4aa" if oi_change > 3 else "#ff4444" if oi_change < -3 else "#888"
                            oi_label = "📈 Rising" if oi_change > 3 else "📉 Falling" if oi_change < -3 else "➡️ Stable"
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                <div style='color: #888; font-size: 0.85em;'>📊 OI Change 24h</div>
                                <div style='color: {oi_color}; font-size: 1.8em; font-weight: bold;'>{oi_change:+.1f}%</div>
                                <div style='color: #666; font-size: 0.8em;'>{oi_label}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("ℹ️ What is OI?"):
                                st.markdown("""
                                **Open Interest** = Total outstanding futures contracts
                                
                                📈 **Rising OI** = New money entering market
                                📉 **Falling OI** = Money leaving market
                                
                                ⚠️ **OI alone doesn't tell direction!**
                                Must combine with PRICE to understand who's entering/exiting.
                                """)
                        
                        # Price Change Card
                        with whale_cols[1]:
                            price_change = whale_data.get('price_change_24h', 0)
                            price_color = "#00d4aa" if price_change > 0 else "#ff4444"
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                <div style='color: #888; font-size: 0.85em;'>💵 Price 24h</div>
                                <div style='color: {price_color}; font-size: 1.8em; font-weight: bold;'>{price_change:+.1f}%</div>
                                <div style='color: #666; font-size: 0.8em;'>{"📈 Up" if price_change > 0 else "📉 Down"}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("ℹ️ OI + Price"):
                                st.markdown("""
                                ### The 4 Combinations (MEMORIZE THIS!)
                                
                                | OI | Price | Signal | Meaning | Action |
                                |:---:|:---:|:---:|:---|:---|
                                | 📈 UP | 📈 UP | 🟢 **NEW LONGS** | Fresh buying, new money entering long | ✅ Bullish continuation |
                                | 📈 UP | 📉 DOWN | 🔴 **NEW SHORTS** | Fresh selling, new money entering short | ✅ Bearish continuation |
                                | 📉 DOWN | 📈 UP | 🟡 **SHORT COVERING** | Shorts closing, NOT new buying | ⚠️ Weak rally, may reverse |
                                | 📉 DOWN | 📉 DOWN | 🟡 **LONG LIQUIDATION** | Longs closing/stopped out, NOT new shorts | ⚠️ Dump may be ending |
                                
                                ---
                                
                                **💡 KEY INSIGHT:**
                                - **OI Rising** = New positions = Trend has CONVICTION
                                - **OI Falling** = Closing positions = Trend may EXHAUST
                                """)
                                # Show current interpretation
                                if oi_change > 1 and price_change > 0:
                                    st.success("🟢 NEW LONGS - Fresh buying, bullish continuation likely")
                                elif oi_change > 1 and price_change < 0:
                                    st.error("🔴 NEW SHORTS - Fresh selling, bearish continuation likely")
                                elif oi_change < -1 and price_change > 0:
                                    st.warning("🟡 SHORT COVERING - Rally is weak (shorts closing, not new buying)")
                                elif oi_change < -1 and price_change < 0:
                                    st.warning("🟡 LONG LIQUIDATION - Dump may be ending (forced selling, not new shorts)")
                                else:
                                    st.info("⚪ STABLE - No strong signal from OI + Price")
                        
                        # Funding Rate Card
                        with whale_cols[2]:
                            funding = whale_data.get('funding', {}).get('rate_pct', 0)
                            fund_status = "Longs Pay" if funding > 0 else "Shorts Pay" if funding < 0 else "Neutral"
                            fund_color = "#ff4444" if abs(funding) > 0.05 else "#ffcc00" if abs(funding) > 0.01 else "#888"
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                <div style='color: #888; font-size: 0.85em;'>💰 Funding Rate</div>
                                <div style='color: {fund_color}; font-size: 1.8em; font-weight: bold;'>{funding:.4f}%</div>
                                <div style='color: #666; font-size: 0.8em;'>{fund_status}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("ℹ️ What is Funding?"):
                                st.markdown("""
                                **Funding Rate** = Fee between longs/shorts every 8h
                                
                                💰 **Positive** = Longs pay shorts (bullish sentiment)
                                💰 **Negative** = Shorts pay longs (bearish sentiment)
                                
                                ⚠️ **CONTRARIAN at extremes:**
                                - Very positive (>0.1%) = Too many longs → Dump coming
                                - Very negative (<-0.1%) = Too many shorts → Pump coming
                                
                                💡 **Edge:** Extreme funding = crowded side gets liquidated
                                """)
                        
                        # Top Traders Card
                        with whale_cols[3]:
                            whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                            whale_color = "#00d4aa" if whale_long > 55 else "#ff4444" if whale_long < 45 else "#888"
                            whale_label = "🟢 LONG" if whale_long > 55 else "🔴 SHORT" if whale_long < 45 else "Balanced"
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                <div style='color: #888; font-size: 0.85em;'>🐋 Top Traders</div>
                                <div style='color: {whale_color}; font-size: 1.8em; font-weight: bold;'>{whale_long:.0f}%</div>
                                <div style='color: #666; font-size: 0.8em;'>{whale_label}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("ℹ️ Who are Top Traders?"):
                                st.markdown("""
                                **Top Traders** = Binance's most profitable futures traders
                                
                                This shows what % of their positions are LONG:
                                - **60%** means 60% long, 40% short
                                
                                🟢 **>55%** = Smart money bullish
                                🔴 **<45%** = Smart money bearish
                                
                                💡 **Edge:** When whales diverge from retail, follow whales!
                                
                                ⚠️ **BUT:** Whales being long ≠ "Enter now"
                                Wait for price confirmation!
                                """)
                        
                        # Retail Card
                        with whale_cols[4]:
                            retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
                            retail_color = "#ff4444" if retail_long > 65 else "#00d4aa" if retail_long < 35 else "#888"
                            retail_label = "⚠️ FADE" if retail_long > 65 or retail_long < 35 else "Balanced"
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                <div style='color: #888; font-size: 0.85em;'>🐑 Retail</div>
                                <div style='color: {retail_color}; font-size: 1.8em; font-weight: bold;'>{retail_long:.0f}%</div>
                                <div style='color: #666; font-size: 0.8em;'>{retail_label}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("ℹ️ Retail = Contrarian"):
                                st.markdown("""
                                **Retail Traders** = All regular traders on Binance
                                
                                ⚠️ **Retail is often WRONG at extremes!**
                                
                                - **>65% Long** = FADE THIS → Often marks TOPS
                                - **<35% Long** = FADE THIS → Often marks BOTTOMS
                                
                                💡 **Best setup:** Whales opposite to retail
                                = Retail becomes exit liquidity
                                """)
                        
                        # OI + Price Interpretation - THE KEY INSIGHT
                        oi_interp = whale_data.get('oi_interpretation', {})
                        if oi_interp and oi_interp.get('interpretation'):
                            interp_emoji = oi_interp.get('emoji', '📊')
                            interp_color = "#00d4aa" if interp_emoji == '🟢' else "#ff4444" if interp_emoji == '🔴' else "#ffcc00"
                            
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, {interp_color}22, {interp_color}11); 
                                        border-left: 4px solid {interp_color}; border-radius: 8px; padding: 15px; margin: 15px 0;'>
                                <div style='color: {interp_color}; font-weight: bold; font-size: 1.1em; margin-bottom: 8px;'>
                                    {interp_emoji} OI + Price Analysis: {oi_interp.get('signal', '')}
                                </div>
                                <div style='color: #ccc; font-size: 1em; margin-bottom: 8px;'>
                                    {oi_interp.get('interpretation', '')}
                                </div>
                                <div style='color: #888; font-size: 0.95em;'>
                                    <strong>Action:</strong> {oi_interp.get('action', '')}
                                </div>
                                <div style='color: #666; font-size: 0.85em; margin-top: 5px;'>
                                    Strength: {oi_interp.get('strength', 'N/A')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Funding interpretation
                        funding_interp = whale_data.get('funding_interpretation', {})
                        if funding_interp and funding_interp.get('is_contrarian'):
                            st.markdown(f"""
                            <div style='background: #2a2a1a; border-left: 4px solid #ffcc00; border-radius: 8px; padding: 12px; margin: 10px 0;'>
                                <div style='color: #ffcc00; font-weight: bold;'>
                                    ⚠️ CONTRARIAN SIGNAL: {funding_interp.get('action', '')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Long/Short interpretation
                        ls_interp = whale_data.get('ls_interpretation', {})
                        if ls_interp and ls_interp.get('edge') in ['HIGH', 'MEDIUM']:
                            edge_color = "#00d4aa" if 'LONG' in ls_interp.get('action', '') else "#ff4444"
                            st.markdown(f"""
                            <div style='background: {edge_color}22; border-left: 4px solid {edge_color}; border-radius: 8px; padding: 12px; margin: 10px 0;'>
                                <div style='color: {edge_color}; font-weight: bold;'>
                                    🎯 POSITIONING EDGE ({ls_interp.get('edge', '')}): {ls_interp.get('action', '')}
                                </div>
                                <div style='color: #888; font-size: 0.9em; margin-top: 5px;'>
                                    Retail: {ls_interp.get('retail_bias', 'N/A')} | Whales: {ls_interp.get('whale_bias', 'N/A')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # All signals summary
                        if whale_data.get('signals'):
                            with st.expander("📋 All Institutional Signals", expanded=False):
                                for signal in whale_data['signals']:
                                    st.markdown(f"• {signal}")
                    else:
                        st.info("🐋 Whale data unavailable - Binance Futures API may be restricted in your network")
                
                except Exception as e:
                    st.info(f"🐋 Could not fetch institutional data: {str(e)[:50]}")
                
                # ═══════════════════════════════════════════════════════════
                # ROI TARGETS
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 💰 ROI Targets")
                
                roi = bd['roi_potential']
                if roi:
                    roi_cols = st.columns(4)
                    
                    with roi_cols[0]:
                        st.metric("🎯 TP1", f"+{roi['tp1']['roi']:.1f}%", f"{roi['tp1']['probability']}% prob")
                        st.caption(fmt_price(roi['tp1']['price']))
                    
                    with roi_cols[1]:
                        st.metric("🎯 TP2", f"+{roi['tp2']['roi']:.1f}%", f"{roi['tp2']['probability']}% prob")
                        st.caption(fmt_price(roi['tp2']['price']))
                    
                    with roi_cols[2]:
                        st.metric("🎯 TP3", f"+{roi['tp3']['roi']:.1f}%", f"{roi['tp3']['probability']}% prob")
                        st.caption(fmt_price(roi['tp3']['price']))
                    
                    with roi_cols[3]:
                        ev = roi.get('expected_value', 0)
                        st.metric("📊 Expected Value", f"{ev:+.1f}%", "Risk-adjusted")
                        st.caption(f"Risk: {roi['risk']:.1f}%")
                
                # ═══════════════════════════════════════════════════════════
                # TIME ESTIMATION
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### ⏱️ Time Estimation")
                
                # Calculate time estimates for TP1 and SL
                current_price = sig.entry
                time_cols = st.columns(3)
                
                with time_cols[0]:
                    tp1_est = estimate_time(r['df'], current_price, sig.tp1, timeframe)
                    st.metric("⏱️ TP1 ETA", tp1_est['time_str'], f"{tp1_est['probability']}% prob")
                    st.caption(f"~{tp1_est['candles']} candles")
                
                with time_cols[1]:
                    sl_est = estimate_time(r['df'], current_price, sig.stop_loss, timeframe)
                    st.metric("⚠️ SL ETA", sl_est['time_str'], f"Conf: {sl_est['confidence']}")
                    st.caption(f"~{sl_est['candles']} candles")
                
                with time_cols[2]:
                    tp2_est = estimate_time(r['df'], current_price, sig.tp2, timeframe)
                    st.metric("⏱️ TP2 ETA", tp2_est['time_str'], f"{tp2_est['probability']}% prob")
                    st.caption(f"~{tp2_est['candles']} candles")
                
                # ═══════════════════════════════════════════════════════════
                # ORDER BLOCK DATA
                # ═══════════════════════════════════════════════════════════
                
                if bd['order_block']:
                    st.markdown("---")
                    st.markdown("### 🏦 Order Block Data")
                    
                    ob = bd['order_block']
                    ob_cols = st.columns(3)
                    
                    with ob_cols[0]:
                        st.markdown(f"**Type:** {ob['type']}")
                    with ob_cols[1]:
                        st.markdown(f"**OB Top:** {fmt_price(ob['top'])}")
                    with ob_cols[2]:
                        st.markdown(f"**OB Bottom:** {fmt_price(ob['bottom'])}")
                
                # ═══════════════════════════════════════════════════════════
                # SESSION ANALYSIS (AMD)
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 🌍 Session Analysis (AMD)")
                
                sess_cols = st.columns(4)
                
                with sess_cols[0]:
                    st.markdown(f"**🌏 Asian:**")
                    st.markdown(f"{sessions['asian']['emoji']} {sessions['asian']['direction']}")
                
                with sess_cols[1]:
                    st.markdown(f"**🇬🇧 London:**")
                    st.markdown(f"{sessions['london']['emoji']} {sessions['london']['direction']}")
                
                with sess_cols[2]:
                    st.markdown(f"**🇺🇸 New York:**")
                    st.markdown(f"{sessions['newyork']['emoji']} {sessions['newyork']['direction']}")
                
                with sess_cols[3]:
                    st.markdown(f"**📊 Current:**")
                    st.markdown(f"{sessions['current_session']}")
                
                st.info(f"**Pattern:** {sessions['pattern']}")
                
                # ═══════════════════════════════════════════════════════════
                # VOLUME METRICS
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 📊 Volume Analysis")
                
                vol = bd['volume_analysis']
                vol_cols = st.columns(4)
                
                with vol_cols[0]:
                    st.metric("MFI", f"{vol['mfi']:.0f}", vol['mfi_status'])
                
                with vol_cols[1]:
                    st.metric("CMF", f"{vol['cmf']:.3f}", vol['cmf_status'])
                
                with vol_cols[2]:
                    st.markdown(f"**OBV Trend:**")
                    st.markdown(vol['obv_trend'])
                
                with vol_cols[3]:
                    st.markdown(f"**Flow Status:**")
                    st.markdown(vol['flow_status'])
                
                # ═══════════════════════════════════════════════════════════
                # 🦈 INSTITUTIONAL ACTIVITY (PREDICTIVE)
                # ═══════════════════════════════════════════════════════════
                
                inst = r.get('institutional', {})
                if inst and inst.get('signals'):
                    st.markdown("---")
                    st.markdown("### 🦈 Institutional Activity")
                    
                    # Show activity summary
                    act_color = inst.get('color', '#888')
                    st.markdown(f"""
                    <div style='background: {act_color}22; border-left: 4px solid {act_color}; 
                                padding: 10px 15px; border-radius: 8px; margin-bottom: 10px;'>
                        <strong style='color: {act_color};'>{inst.get('recommendation', 'No clear signal')}</strong>
                        <div style='color: #888; margin-top: 4px;'>
                            Confidence: <strong>{inst.get('confidence', 0)}%</strong> | 
                            Bullish: <span style='color: #00d4aa;'>+{inst.get('bullish_score', 0)}</span> | 
                            Bearish: <span style='color: #ff4444;'>-{inst.get('bearish_score', 0)}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show individual signals
                    for sig_item in inst.get('signals', [])[:3]:
                        sig_color = sig_item.get('color', '#888')
                        st.markdown(f"""
                        <div style='background: #1a1a2e; border-left: 3px solid {sig_color}; 
                                    padding: 8px 12px; border-radius: 6px; margin: 5px 0;'>
                            <span style='color: {sig_color}; font-weight: bold;'>{sig_item.get('emoji', '📊')} {sig_item.get('message', '')}</span>
                            <div style='color: #888; font-size: 0.85em; margin-top: 2px;'>{sig_item.get('detail', '')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 🎓 LEARN - Educational explanation of concepts used
                # ═══════════════════════════════════════════════════════════
                
                with st.expander("🎓 **Learn: Understanding This Signal**", expanded=False):
                    
                    # Dynamically show education based on what factors are present
                    education_items = []
                    
                    # Order Block education
                    if bd.get('order_block'):
                        education_items.append({
                            'title': '🏦 Order Blocks (OB)',
                            'explanation': """An <b>Order Block</b> is the last candle before a strong move in the opposite direction.
It represents where institutional traders (banks, funds) placed large orders.

<b>Why it matters:</b> When price returns to an OB, it often bounces because
institutions defend their positions. Think of it as "unfilled orders" waiting.

<b>How to use:</b> Enter at bullish OB for longs, bearish OB for shorts.
Place stop loss just beyond the OB."""
                        })
                    
                    # Money flow education
                    if mf.get('is_accumulating') or mf.get('is_distributing'):
                        education_items.append({
                            'title': '💰 Money Flow (MFI/CMF/OBV)',
                            'explanation': f"""<b>MFI (Money Flow Index):</b> Like RSI but includes volume.
Current: {mf.get('mfi', 50):.0f} - {'Oversold (bullish)' if mf.get('mfi', 50) < 30 else 'Overbought (bearish)' if mf.get('mfi', 50) > 70 else 'Neutral'}

<b>CMF (Chaikin Money Flow):</b> Measures buying/selling pressure.
Current: {mf.get('cmf', 0):.3f} - {'Positive = Buyers in control' if mf.get('cmf', 0) > 0 else 'Negative = Sellers in control'}

<b>OBV (On-Balance Volume):</b> Cumulative volume showing money direction.
{'Rising OBV = Smart money buying' if mf.get('obv_rising') else 'Falling OBV = Smart money selling'}

<b>Key insight:</b> Price can be manipulated, but volume cannot lie.
When volume confirms price, the move is more likely to continue."""
                        })
                    
                    # R:R education
                    education_items.append({
                        'title': '⚖️ Risk:Reward Ratio (R:R)',
                        'explanation': f"""<b>What it means:</b> For every $1 you risk, how much can you gain?

This trade: Risk {sig.risk_pct:.1f}% to gain +{calc_roi(sig.tp1, sig.entry):.1f}% at TP1
R:R at TP1 = <b>{calc_roi(sig.tp1, sig.entry)/sig.risk_pct:.1f}:1</b>

<b>Good practice:</b> Only take trades with R:R >= 1.5:1
This means even with 40% win rate, you're profitable!

<b>Example:</b>
• 10 trades at 1.5:1 R:R, 40% win rate
• 4 wins × 1.5R = 6R gained
• 6 losses × 1R = 6R lost
• Net = Breakeven (and 40% is very achievable!)"""
                    })
                    
                    # Timeframe education
                    education_items.append({
                        'title': f'⏱️ Why {timeframe} Timeframe?',
                        'explanation': f"""You selected <b>{display_label}</b> mode.

<b>{timeframe} means:</b> Each candle = {timeframe} of price action

<b>Trade duration:</b> {TIMEFRAME_DETAILS.get(timeframe, {}).get('expected_close', 'Varies')}

<b>Pro tip:</b> Higher timeframes = cleaner signals but slower trades.
Lower timeframes = more opportunities but more noise.

<b>Multi-TF check:</b> Confirm on {mode_config.get('confirm_tf', '4h')} for higher probability."""
                    })
                    
                    # Display education
                    for item in education_items:
                        st.markdown(f"""
                        <div style='background: #0a1a2a; border-radius: 8px; padding: 15px; margin: 10px 0; 
                                    border: 1px solid #1a3a5a;'>
                            <div style='color: #00d4ff; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;'>
                                {item['title']}
                            </div>
                            <div style='color: #bbb; line-height: 1.7; white-space: pre-line;'>
                                {item['explanation']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 📈 ADD TO MONITOR / WATCHLIST BUTTONS
                # (Scenario info already shown in Whale section above)
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                
                # Get current status for button logic
                try:
                    from core.institutional_scoring import (
                        analyze_institutional_data, calculate_combined_score,
                        TradeStatus, get_scenario_display
                    )
                    
                    whale_data = get_whale_analysis(sig.symbol)
                    if whale_data and whale_data.get('open_interest', {}).get('change_24h', 0) != 0:
                        oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                        price_change = whale_data.get('price_change_24h', 0)
                        whale_pct = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                        retail_pct = whale_data.get('retail_ls', {}).get('long_pct', 50)
                        funding = whale_data.get('funding', {}).get('rate', 0)
                        
                        # Get tech direction
                        tech_dir = "BULLISH" if sig.direction == "LONG" else "BEARISH" if sig.direction == "SHORT" else "NEUTRAL"
                        
                        # Analyze institutional data
                        inst_score = analyze_institutional_data(
                            oi_change, price_change, whale_pct, retail_pct, funding, tech_dir
                        )
                        
                        # Calculate combined score
                        combined, status, details = calculate_combined_score(bd['score'], inst_score)
                        
                        # Get scenario display
                        scenario_display = get_scenario_display(inst_score.scenario)
                        
                        status_colors = {
                            'READY': '#00d4aa',
                            'WAIT': '#ffcc00', 
                            'WATCH': '#ff8800',
                            'AVOID': '#ff4444'
                        }
                        status_color = status_colors.get(status, '#888888')
                        
                        # ADD TO MONITOR or WATCHLIST based on status
                        if any(t['symbol'] == sig.symbol for t in st.session_state.active_trades):
                            st.success("✅ Already in Monitor")
                        elif any(w['symbol'] == sig.symbol for w in st.session_state.watchlist):
                            st.info("👁️ Already in Watchlist")
                        else:
                            btn_col1, btn_col2 = st.columns(2)
                            
                            with btn_col1:
                                # Always allow adding to Monitor (user's choice)
                                if st.button(f"➕ Add to Monitor", key=f"add_{sig.symbol}_{timeframe}", type="primary" if status == "READY" else "secondary"):
                                    trade = {
                                        'symbol': sig.symbol,
                                        'entry': sig.entry,
                                        'stop_loss': sig.stop_loss,
                                        'tp1': sig.tp1,
                                        'tp2': sig.tp2,
                                        'tp3': sig.tp3,
                                        'direction': sig.direction,
                                        'score': combined,  # Use combined score
                                        'grade': bd['grade'],
                                        'timeframe': timeframe,
                                        'status': 'active',
                                        'inst_status': status,  # Track institutional status
                                        'scenario': inst_score.scenario_name,
                                        'created_at': datetime.now().isoformat()
                                    }
                                    add_trade(trade)
                                    st.session_state.active_trades = get_active_trades()
                                    st.success(f"✅ Added {sig.symbol} to Monitor!")
                                    st.rerun()
                            
                            with btn_col2:
                                # Add to Watchlist option for WAIT scenarios
                                if status == "WAIT":
                                    if st.button(f"👁️ Add to Watchlist", key=f"watch_{sig.symbol}_{timeframe}"):
                                        # Get recent price levels for triggers
                                        df = r.get('df')
                                        recent_low = df['Low'].tail(20).min() if df is not None else sig.stop_loss
                                        recent_high = df['High'].tail(20).max() if df is not None else sig.tp1
                                        
                                        watch_item = {
                                            'symbol': sig.symbol,
                                            'timeframe': timeframe,
                                            'direction': inst_score.direction_bias,
                                            'scenario': inst_score.scenario.value,
                                            'entry_conditions': inst_score.wait_conditions,
                                            'conditions_met': [False] * len(inst_score.wait_conditions),
                                            'confirmation_triggers': inst_score.confirmation_triggers,
                                            'trigger_above': sig.entry * 1.02 if inst_score.direction_bias == 'LONG' else 0,
                                            'trigger_below': sig.entry * 0.98 if inst_score.direction_bias == 'SHORT' else 0,
                                            'invalidation_price': recent_low * 0.98 if inst_score.direction_bias == 'LONG' else recent_high * 1.02,
                                            'added_at': datetime.now().isoformat(),
                                            'price_at_add': sig.entry,
                                            'whale_pct_at_add': whale_pct,
                                            'notes': f"Waiting for {inst_score.scenario_name} confirmation"
                                        }
                                        st.session_state.watchlist.append(watch_item)
                                        st.info(f"👁️ Added {sig.symbol} to Watchlist - Will alert when conditions are met!")
                                        st.rerun()
                                else:
                                    st.caption("Add to Monitor when ready to trade")
                    else:
                        # No whale data - show normal add button
                        if any(t['symbol'] == sig.symbol for t in st.session_state.active_trades):
                            st.success("✅ Already in Monitor")
                        else:
                            if st.button(f"➕ Add {sig.symbol} to Monitor", key=f"add_{sig.symbol}_{timeframe}_nowh", type="primary"):
                                trade = {
                                    'symbol': sig.symbol,
                                    'entry': sig.entry,
                                    'stop_loss': sig.stop_loss,
                                    'tp1': sig.tp1,
                                    'tp2': sig.tp2,
                                    'tp3': sig.tp3,
                                    'direction': sig.direction,
                                    'score': bd['score'],
                                    'grade': bd['grade'],
                                    'timeframe': timeframe,
                                    'status': 'active',
                                    'created_at': datetime.now().isoformat()
                                }
                                add_trade(trade)
                                st.session_state.active_trades = get_active_trades()
                                st.success(f"✅ Added {sig.symbol} to Monitor!")
                                st.rerun()
                except Exception as inst_err:
                    # Fallback - no institutional data
                    if any(t['symbol'] == sig.symbol for t in st.session_state.active_trades):
                        st.success("✅ Already in Monitor")
                    else:
                        if st.button(f"➕ Add {sig.symbol} to Monitor", key=f"add_{sig.symbol}_{timeframe}_err", type="primary"):
                            trade = {
                                'symbol': sig.symbol,
                                'entry': sig.entry,
                                'stop_loss': sig.stop_loss,
                                'tp1': sig.tp1,
                                'tp2': sig.tp2,
                                'tp3': sig.tp3,
                                'direction': sig.direction,
                                'score': bd['score'],
                                'grade': bd['grade'],
                                'timeframe': timeframe,
                                'status': 'active',
                                'created_at': datetime.now().isoformat()
                            }
                            add_trade(trade)
                            st.session_state.active_trades = get_active_trades()
                            st.success(f"✅ Added {sig.symbol} to Monitor!")
                            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 TRADE MONITOR - PRO VERSION WITH ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

elif app_mode == "📈 Trade Monitor":
    st.markdown("## 📈 Trade Monitor")
    st.markdown("<p style='color: #888;'>Live tracking with alerts & recommendations</p>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # 💰 PRICE CACHING - Avoid reloading on every tab switch
    # ═══════════════════════════════════════════════════════════════════
    
    PRICE_CACHE_TTL = 30  # seconds
    
    # Initialize price cache
    if 'price_cache' not in st.session_state:
        st.session_state.price_cache = {}
    if 'price_cache_time' not in st.session_state:
        st.session_state.price_cache_time = None
    if 'force_price_refresh' not in st.session_state:
        st.session_state.force_price_refresh = False
    
    def get_cached_prices(symbols: list, force_refresh: bool = False) -> dict:
        """Get prices with caching. Only fetch if cache is stale or force refresh."""
        now = datetime.now()
        cache_age = (now - st.session_state.price_cache_time).total_seconds() if st.session_state.price_cache_time else float('inf')
        
        # Check if we need to refresh
        need_refresh = force_refresh or cache_age > PRICE_CACHE_TTL or not st.session_state.price_cache
        
        if need_refresh:
            # Fetch fresh prices
            new_prices = {}
            for symbol in symbols:
                try:
                    price = get_current_price(symbol)
                    if price > 0:
                        new_prices[symbol] = price
                except:
                    pass
            
            st.session_state.price_cache = new_prices
            st.session_state.price_cache_time = now
            st.session_state.force_price_refresh = False
        
        return st.session_state.price_cache
    
    # Get all symbols we need prices for
    trade_symbols = [t['symbol'] for t in st.session_state.active_trades]
    
    # Check if we should skip refresh (after remove action)
    skip_refresh = st.session_state.get('skip_price_refresh', False)
    if skip_refresh:
        st.session_state.skip_price_refresh = False  # Reset flag
        should_refresh = False
    else:
        # Check if we should refresh (button click or first load)
        should_refresh = st.session_state.get('force_price_refresh', False) or st.session_state.price_cache_time is None
    
    # Fetch prices (cached or fresh)
    cached_prices = get_cached_prices(trade_symbols, force_refresh=should_refresh)
    
    # ═══════════════════════════════════════════════════════════════════
    # ⚠️ CLOUD PERSISTENCE WARNING
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("""
    <div style='background: #2a2a1a; border-left: 4px solid #ffcc00; padding: 10px 15px; 
                border-radius: 8px; margin-bottom: 15px;'>
        <span style='color: #ffcc00;'>⚠️ <strong>Cloud Warning:</strong></span>
        <span style='color: #ccc;'> On Streamlit Cloud, trades reset on refresh. 
        Use <strong>Export</strong> to save your trades to your computer, 
        then <strong>Import</strong> to restore them.</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # 🔧 MANAGEMENT BUTTONS + EXPORT/IMPORT
    # ═══════════════════════════════════════════════════════════════════
    
    # Show cache status
    cache_age_display = ""
    if st.session_state.price_cache_time:
        age = (datetime.now() - st.session_state.price_cache_time).total_seconds()
        cache_age_display = f"Prices: {int(age)}s ago"
    
    mgmt_col1, mgmt_col2, mgmt_col3, mgmt_col4, mgmt_col5 = st.columns([1, 1, 1, 1, 1])
    
    with mgmt_col1:
        if st.button("🔄 Refresh Prices", help="Fetch latest prices from exchange"):
            st.session_state.force_price_refresh = True
            st.rerun()
    
    with mgmt_col2:
        if st.button("📂 Reload Trades", help="Reload trades from saved file"):
            st.session_state.active_trades = get_active_trades()
            st.toast(f"Reloaded {len(st.session_state.active_trades)} trades")
            st.rerun()
    
    with mgmt_col2:
        # Export button
        if st.session_state.active_trades:
            export_data = export_trades_json()
            st.download_button(
                label="📥 Export",
                data=export_data,
                file_name=f"investoriq_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                help="Download your trades to your computer"
            )
        else:
            st.button("📥 Export", disabled=True, help="No trades to export")
    
    with mgmt_col3:
        if st.button("🗑️ Clear All", help="Remove all active trades"):
            if st.session_state.active_trades:
                all_trades = load_trade_history()
                closed_only = [t for t in all_trades if t.get('status') == 'closed']
                save_trade_history(closed_only)
                st.session_state.active_trades = []
                st.success("Cleared all active trades")
                st.rerun()
    
    # Import section
    with st.expander("📤 **Import Trades from File**", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload your exported trades JSON file",
            type=['json'],
            help="Select a previously exported trades file"
        )
        
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode('utf-8')
                imported_trades = json.loads(content)
                
                st.success(f"Found {len(imported_trades)} trades in file")
                
                # Preview
                active_count = len([t for t in imported_trades if t.get('status') == 'active'])
                closed_count = len([t for t in imported_trades if t.get('status') == 'closed'])
                
                st.markdown(f"""
                <div style='background: #1a2a3a; padding: 10px; border-radius: 6px;'>
                    <span style='color: #00d4aa;'>✅ Active: {active_count}</span> | 
                    <span style='color: #888;'>📊 Closed: {closed_count}</span>
                </div>
                """, unsafe_allow_html=True)
                
                col_import1, col_import2 = st.columns(2)
                
                with col_import1:
                    if st.button("✅ Import & Replace All", type="primary"):
                        if import_trades_json(content):
                            st.session_state.active_trades = get_active_trades()
                            st.success(f"Imported {len(st.session_state.active_trades)} active trades!")
                            st.rerun()
                        else:
                            st.error("Failed to import trades")
                
                with col_import2:
                    if st.button("➕ Add to Existing"):
                        existing = load_trade_history()
                        # Merge, avoiding duplicates by symbol
                        existing_symbols = {t.get('symbol') for t in existing if t.get('status') == 'active'}
                        
                        new_trades = [t for t in imported_trades 
                                     if t.get('symbol') not in existing_symbols or t.get('status') != 'active']
                        
                        combined = existing + new_trades
                        save_trade_history(combined)
                        st.session_state.active_trades = get_active_trades()
                        st.success(f"Added {len(new_trades)} new trades!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # ═══════════════════════════════════════════════════════════════════
    # 🛡️ MENTAL STOP STRATEGY EXPLANATION
    # ═══════════════════════════════════════════════════════════════════
    
    with st.expander("💡 **Why use Mental Stops instead of Exchange Stops?**", expanded=False):
        st.markdown("### 🛡️ The Mental Stop Strategy")
        
        st.warning("**Problem:** Binance (and other exchanges) can SEE your stop loss orders. Their algorithms hunt stops before real moves.")
        
        st.success("**Solution:** Don't put actual stop orders on the exchange!")
        
        st.markdown("""
        **How to use this Monitor:**
        1. Enter your trade on Binance **WITHOUT a stop loss order**
        2. Add the trade to this Monitor with your planned SL level
        3. Monitor will alert you when price approaches your SL
        4. When you see 🚨 **DANGER ZONE**, manually close your position
        """)
        
        st.warning("⚠️ **Risk:** If you're away from your screen, you could lose more than planned. Only use this if you can monitor actively!")
    
    # ═══════════════════════════════════════════════════════════════════
    # 📊 OI + PRICE INTERPRETATION GUIDE (NEW!)
    # ═══════════════════════════════════════════════════════════════════
    
    with st.expander("📊 **Understanding Whale Data (OI + Price)**", expanded=False):
        st.markdown("""
        ### 🐋 How to Read Institutional Data
        
        #### 📊 Open Interest (OI) + Price Relationship
        
        This is the **KEY** to understanding what whales are doing:
        
        | OI Change | Price Change | What It Means | Action |
        |-----------|--------------|---------------|--------|
        | 📈 **OI UP** | 📈 **Price UP** | **New LONGS entering** - Fresh money buying | ✅ Bullish continuation - Follow trend |
        | 📈 **OI UP** | 📉 **Price DOWN** | **New SHORTS entering** - Fresh money shorting | ❌ Bearish continuation - Follow trend |
        | 📉 **OI DOWN** | 📈 **Price UP** | **Short covering** - Forced buying (weak) | ⚠️ Weak rally - May reverse |
        | 📉 **OI DOWN** | 📉 **Price DOWN** | **Long liquidation** - Forced selling (weak) | ⚠️ Weak dump - May reverse |
        
        #### 💰 Funding Rate (Contrarian Indicator!)
        
        Funding rate tells you who is overleveraged:
        
        | Funding | Meaning | Action |
        |---------|---------|--------|
        | **> 0.1%** | Longs are overleveraged | 🔴 **CONTRARIAN SHORT** - Dump coming, longs will get liquidated |
        | **< -0.1%** | Shorts are overleveraged | 🟢 **CONTRARIAN LONG** - Pump coming, shorts will get squeezed |
        | **-0.01% to 0.01%** | Balanced | Follow other signals |
        
        #### 🐑 Retail vs 🐋 Top Traders
        
        **KEY INSIGHT**: Retail is usually WRONG at extremes. Top Traders have better information.
        
        | Situation | Action |
        |-----------|--------|
        | Retail 70% LONG + Whales SHORT | 🔴 **FADE RETAIL** - Go short, whales are smarter |
        | Retail 70% SHORT + Whales LONG | 🟢 **FADE RETAIL** - Go long, whales are buying |
        | Both aligned | Follow the direction with confidence |
        
        #### 🔥 Best Setups (High Probability)
        
        1. **Top traders LONG + Retail SHORT + Funding negative** = STRONG LONG
        2. **Top traders SHORT + Retail LONG + Funding positive** = STRONG SHORT
        3. **OI rising + Taker buying + Price consolidating** = Breakout coming
        4. **Extreme funding + Liquidation cascade** = Reversal imminent
        """)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 📋 WATCHLIST - WAIT Setups (Not Ready Yet)
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if st.session_state.watchlist:
        st.markdown("### 📋 Watchlist - Waiting for Confirmation")
        st.markdown("<p style='color: #888;'>These setups need confirmation before entry</p>", unsafe_allow_html=True)
        
        # Import scoring functions
        from core.institutional_scoring import check_watchlist_triggers, get_scenario_display, ScenarioType
        
        watchlist_to_remove = []
        watchlist_triggered = []
        
        for i, item in enumerate(st.session_state.watchlist):
            # Get current data
            try:
                current_price = get_current_price(item['symbol'])
                whale_data = get_whale_analysis(item['symbol'])
                current_whale_pct = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
            except:
                current_price = item.get('price_at_add', 0)
                current_whale_pct = 50
                oi_change = 0
            
            # Check triggers
            triggered, invalidated, message = check_watchlist_triggers(
                type('obj', (object,), item)(),  # Convert dict to object-like
                current_price,
                current_whale_pct,
                oi_change
            )
            
            # Get scenario details
            scenario_type = item.get('scenario', ScenarioType.NO_EDGE)
            if isinstance(scenario_type, str):
                scenario_type = ScenarioType(scenario_type) if scenario_type in [e.value for e in ScenarioType] else ScenarioType.NO_EDGE
            scenario_info = get_scenario_display(scenario_type)
            
            # Display watchlist item
            direction_color = "#00d4aa" if item['direction'] == 'LONG' else "#ff4444"
            
            with st.expander(f"{scenario_info['emoji']} {item['symbol']} - {item['direction']} (Waiting)", expanded=not invalidated):
                
                # Status row
                if triggered:
                    st.success(f"✅ **TRIGGERED!** {message}")
                    st.markdown(f"""
                    <div style='background: #0a2a1a; border: 2px solid #00d4aa; border-radius: 10px; padding: 15px; margin: 10px 0;'>
                        <div style='color: #00d4aa; font-size: 1.2em; font-weight: bold;'>🎯 READY TO ENTER {item['direction']}</div>
                        <div style='color: #ddd; margin-top: 10px;'>
                            Conditions met! Consider entering this trade now.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    watchlist_triggered.append(i)
                    
                elif invalidated:
                    st.error(f"❌ **INVALIDATED** {message}")
                    watchlist_to_remove.append(i)
                    
                else:
                    st.info(f"⏳ {message}")
                
                # Info columns
                wcol1, wcol2, wcol3 = st.columns(3)
                
                with wcol1:
                    st.markdown(f"**Scenario:** {scenario_info['description']}")
                    st.markdown(f"**Direction:** :{('green' if item['direction'] == 'LONG' else 'red')}[{item['direction']}]")
                    
                with wcol2:
                    price_change = ((current_price - item['price_at_add']) / item['price_at_add'] * 100) if item['price_at_add'] > 0 else 0
                    st.markdown(f"**Added at:** ${item['price_at_add']:.4f}")
                    st.markdown(f"**Current:** ${current_price:.4f} ({price_change:+.2f}%)")
                    
                with wcol3:
                    st.markdown(f"**Whale % (then):** {item.get('whale_pct_at_add', 50):.0f}%")
                    st.markdown(f"**Whale % (now):** {current_whale_pct:.0f}%")
                
                # Conditions
                st.markdown("**Waiting for:**")
                for j, condition in enumerate(item.get('entry_conditions', [])):
                    met = item.get('conditions_met', [False] * len(item.get('entry_conditions', [])))[j] if j < len(item.get('conditions_met', [])) else False
                    icon = "✅" if met else "⬜"
                    st.markdown(f"  {icon} {condition}")
                
                # Action buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                
                with btn_col1:
                    if st.button(f"🗑️ Remove", key=f"remove_watch_{i}"):
                        watchlist_to_remove.append(i)
                        st.rerun()
                        
                with btn_col2:
                    if triggered and st.button(f"➡️ Add to Trades", key=f"add_trade_{i}"):
                        # TODO: Add trade creation logic
                        st.success(f"Go to Scanner to set up {item['symbol']} trade")
                        watchlist_to_remove.append(i)
        
        # Clean up watchlist
        if watchlist_to_remove:
            st.session_state.watchlist = [
                item for i, item in enumerate(st.session_state.watchlist) 
                if i not in watchlist_to_remove
            ]
        
        st.markdown("---")
    
    if not st.session_state.active_trades:
        st.info("No active trades. Go to Scanner to find setups!")
    else:
        # ═══════════════════════════════════════════════════════════════════
        # 🚨 DANGER ZONE ALERTS (Top Priority)
        # Uses "Risk Budget Consumed" - how much of your planned risk is used
        # ═══════════════════════════════════════════════════════════════════
        
        danger_trades = []
        
        for trade in st.session_state.active_trades:
            price = get_current_price(trade['symbol'])
            if price > 0:
                entry = trade['entry']
                sl = trade['stop_loss']
                
                # Calculate distances
                if trade['direction'] == 'LONG':
                    initial_risk = ((entry - sl) / entry) * 100  # Risk at entry
                    current_dist_to_sl = ((price - sl) / price) * 100  # Current distance
                    pnl_pct = ((price - entry) / entry) * 100
                else:
                    initial_risk = ((sl - entry) / entry) * 100
                    current_dist_to_sl = ((sl - price) / price) * 100
                    pnl_pct = ((entry - price) / entry) * 100
                
                # Calculate risk budget consumed (0% = at entry, 100% = at SL)
                if initial_risk > 0:
                    risk_consumed = ((initial_risk - current_dist_to_sl) / initial_risk) * 100
                else:
                    risk_consumed = 0
                
                # DANGER ZONE criteria:
                # 1. Very close to SL (< 0.8% absolute) - Always critical regardless
                # 2. More than 60% of risk budget consumed (well on way to SL)
                # 3. More than 40% risk consumed AND losing more than 1%
                
                is_critical = current_dist_to_sl < 0.8
                is_danger = risk_consumed > 60 or (risk_consumed > 40 and pnl_pct < -1.0)
                
                if is_critical or is_danger:
                    danger_trades.append({
                        'trade': trade,
                        'price': price,
                        'dist_to_sl': current_dist_to_sl,
                        'pnl_pct': pnl_pct,
                        'risk_consumed': risk_consumed,
                        'is_critical': is_critical
                    })
        
        if danger_trades:
            st.markdown("""
            <div style='background: linear-gradient(90deg, #ff4444, #ff0000); padding: 20px; border-radius: 12px; 
                        margin-bottom: 20px; animation: pulse 1s infinite;'>
                <h2 style='color: white; margin: 0; text-align: center;'>🚨 DANGER ZONE - CLOSE MANUALLY NOW! 🚨</h2>
                <p style='color: #ffcccc; text-align: center; margin: 10px 0 0 0;'>
                    These trades are approaching stop loss - close on Binance before they get hunted!
                </p>
            </div>
            <style>
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.8; }
                    100% { opacity: 1; }
                }
            </style>
            """, unsafe_allow_html=True)
            
            for d in danger_trades:
                trade = d['trade']
                price = d['price']
                dist = d['dist_to_sl']
                risk_consumed = d.get('risk_consumed', 0)
                pnl = d.get('pnl_pct', 0)
                is_critical = d.get('is_critical', False)
                
                banner_color = "#ff0000" if is_critical else "#ff4444"
                
                st.markdown(f"""
                <div style='background: #2a0a0a; border: 2px solid {banner_color}; border-radius: 8px; 
                            padding: 15px; margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='color: {banner_color}; font-size: 1.3em; font-weight: bold;'>
                                🔴 {trade['symbol']}
                            </span>
                            <span style='color: #ff6666; margin-left: 15px;'>
                                <strong>{risk_consumed:.0f}%</strong> of risk budget consumed ({dist:.2f}% to SL)
                            </span>
                        </div>
                        <div style='color: #ffcccc;'>
                            Price: <strong>${price:,.4f}</strong> | 
                            SL: <strong>${trade['stop_loss']:,.4f}</strong>
                        </div>
                    </div>
                    <div style='color: #ff9999; margin-top: 10px; font-size: 0.9em;'>
                        ⚡ ACTION: Open Binance → Close this position NOW → They will hunt it soon!
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # ALERTS SECTION
        # ═══════════════════════════════════════════════════════════════════
        
        all_alerts = []
        
        # Collect alerts from ACTIVE trades only (not closed)
        for trade in st.session_state.active_trades:
            # Skip if trade is closed (shouldn't be in active_trades, but double-check)
            if trade.get('closed', False) or trade.get('status') in ['WIN', 'LOSS', 'PARTIAL', 'closed']:
                continue
                
            price = get_current_price(trade['symbol'])
            if price > 0:
                alerts = check_trade_alerts(trade, price)
                all_alerts.extend(alerts)
        
        # Show alerts if any
        if all_alerts:
            # Sort by urgency
            high_alerts = [a for a in all_alerts if a.urgency == "high"]
            med_alerts = [a for a in all_alerts if a.urgency == "medium"]
            
            if high_alerts:
                st.markdown("### 🚨 ALERTS")
                for alert in high_alerts:
                    st.markdown(format_alert_html(alert), unsafe_allow_html=True)
                    if alert.sound:
                        # Visual indicator for sound alert
                        st.markdown(f"<small style='color: #ff4444;'>🔊 Alert triggered!</small>", unsafe_allow_html=True)
            
            if med_alerts:
                with st.expander(f"📢 {len(med_alerts)} Other Notifications", expanded=False):
                    for alert in med_alerts:
                        st.markdown(format_alert_html(alert), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # SUMMARY METRICS
        # ═══════════════════════════════════════════════════════════════════
        
        total_pnl = 0
        winning = 0
        losing = 0
        breakeven = 0
        in_danger = len(danger_trades)  # From earlier danger zone check
        
        for trade in st.session_state.active_trades:
            price = get_current_price(trade['symbol'])
            if price > 0:
                entry = trade['entry']
                pnl = ((price - entry) / entry) * 100 if trade['direction'] == 'LONG' else ((entry - price) / entry) * 100
                total_pnl += pnl
                # Better thresholds: >0.1% = winning, <-0.1% = losing, between = breakeven
                if pnl > 0.1:
                    winning += 1
                elif pnl < -0.1:
                    losing += 1
                else:
                    breakeven += 1  # Essentially flat
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📊 Trades", len(st.session_state.active_trades))
        m2.metric("🟢 Winning", winning)
        m3.metric("🔴 Losing", losing)
        m4.metric("🚨 In Danger", in_danger, delta=None if in_danger == 0 else "CLOSE!")
        m5.metric("💰 Total P&L", f"{total_pnl:+.2f}%")
        
        # ═══════════════════════════════════════════════════════════════════
        # 🏆 COMPLETED TRADES SUMMARY (Professional Metrics)
        # ═══════════════════════════════════════════════════════════════════
        
        # Use session state for closed trades (survives refresh)
        closed_trades = st.session_state.get('closed_trades', [])
        if not closed_trades:
            closed_trades = get_closed_trades()
        
        if closed_trades:
            st.markdown("#### 🏆 Completed Trades (Professional Tracking)")
            
            stats = calculate_statistics(closed_trades)
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Completed", stats['total_trades'])
            c2.metric("🎯 TP1 Hit Rate", f"{stats.get('tp1_hit_rate', 0):.0f}%", 
                     help="Trade selection quality")
            c3.metric("📊 Avg R", f"{stats.get('avg_r_multiple', 0):.1f}R")
            c4.metric("Total P&L", f"{stats['total_pnl']:+.1f}%")
            c5.metric("Profit Factor", f"{stats['profit_factor']:.1f}" if stats['profit_factor'] != float('inf') else "∞")
            
            # Show last 5 completed trades with outcome types
            with st.expander(f"📋 Recent Completed ({len(closed_trades)} total) - Click for details", expanded=False):
                for trade in reversed(closed_trades[-5:]):  # Last 5
                    pnl = trade.get('final_pnl', trade.get('blended_pnl', trade.get('pnl_pct', 0)))
                    outcome = trade.get('outcome_type', '')
                    r_mult = trade.get('r_multiple', 0)
                    
                    # Outcome-based styling
                    if outcome == "FULL_WIN":
                        emoji, color, label = "🏆", "#00ff88", "FULL WIN"
                    elif outcome in ["PARTIAL_WIN_TP2"]:
                        emoji, color, label = "✅", "#00d4aa", "PARTIAL"
                    elif outcome in ["BREAKEVEN", "PARTIAL_WIN_TP1"]:
                        emoji, color, label = "🟡", "#ffcc00", "BE"
                    elif outcome == "FULL_LOSS":
                        emoji, color, label = "🔴", "#ff4444", "LOSS"
                    else:
                        emoji = "✅" if pnl > 0 else "🔴"
                        color = "#00d4aa" if pnl > 0 else "#ff4444"
                        label = "WIN" if pnl > 0 else "LOSS"
                    
                    st.markdown(f"""
                    <div style='background: {color}15; border-left: 3px solid {color}; padding: 8px 12px; 
                                margin: 4px 0; border-radius: 0 6px 6px 0; display: flex; justify-content: space-between;'>
                        <span>{emoji} <strong>{trade.get('symbol', 'N/A')}</strong> 
                        <span style='color: #888;'>({trade.get('direction', '')} {trade.get('timeframe', '')})</span>
                        <span style='background: {color}33; color: {color}; padding: 1px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;'>{label}</span></span>
                        <span style='color: {color}; font-weight: bold;'>{pnl:+.1f}% <span style='color: #888;'>({r_mult:.1f}R)</span></span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<p style='color: #888; font-size: 0.85em; margin-top: 10px;'>📊 Go to <strong>Performance</strong> tab for full analysis</p>", unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # 📊 TRADES BY MODE BREAKDOWN (NEW!)
        # ═══════════════════════════════════════════════════════════════════
        
        trades_by_mode = {}
        for trade in st.session_state.active_trades:
            mode = trade.get('mode_name') or trade.get('mode', 'Unknown')
            if mode not in trades_by_mode:
                trades_by_mode[mode] = []
            trades_by_mode[mode].append(trade)
        
        if len(trades_by_mode) > 1:
            st.markdown("#### 📋 By Trading Mode")
            
            mode_colors = {
                'Scalp': '#ff9500', 'scalp': '#ff9500',
                'Day Trade': '#00d4ff', 'day_trade': '#00d4ff',
                'Swing': '#00d4aa', 'swing': '#00d4aa',
                'Position': '#9d4edd', 'position': '#9d4edd',
                'Investment': '#ffd700', 'investment': '#ffd700',
                'Unknown': '#888888'
            }
            
            mode_cols = st.columns(min(len(trades_by_mode), 5))
            
            for idx, (mode, trades) in enumerate(trades_by_mode.items()):
                color = mode_colors.get(mode, '#888888')
                with mode_cols[idx % len(mode_cols)]:
                    st.markdown(f"""
                    <div style='background: {color}22; border: 1px solid {color}; border-radius: 8px; 
                                padding: 8px; text-align: center; margin: 3px 0;'>
                        <div style='color: {color}; font-weight: bold; font-size: 0.9em;'>{mode}</div>
                        <div style='color: #fff; font-size: 1.3em; font-weight: bold;'>{len(trades)}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ═══════════════════════════════════════════════════════════════════
        # INDIVIDUAL TRADES WITH ALERTS
        # Sort by most recent first (newest trades on top)
        # ═══════════════════════════════════════════════════════════════════
        
        # Sort active trades by added_at (most recent first)
        def get_trade_timestamp(t):
            ts = t.get('added_at') or t.get('created_at') or ''
            return ts if ts else '1970-01-01'  # Old trades without timestamp go to bottom
        
        sorted_trades = sorted(
            enumerate(st.session_state.active_trades), 
            key=lambda x: get_trade_timestamp(x[1]), 
            reverse=True  # Newest first
        )
        
        for i, trade in sorted_trades:
            price = get_current_price(trade['symbol'])
            
            if price > 0:
                entry = trade['entry']
                pnl = ((price - entry) / entry) * 100 if trade['direction'] == 'LONG' else ((entry - price) / entry) * 100
                
                # ═══════════════════════════════════════════════════════════
                # PERSISTENT TARGET HIT TRACKING
                # Checks: 1) Current price, 2) Historical high, 3) Saved state
                # ═══════════════════════════════════════════════════════════
                
                # Check if targets are hit NOW based on current price
                tp1_hit_now = price >= trade['tp1'] if trade['direction'] == 'LONG' else price <= trade['tp1']
                tp2_hit_now = price >= trade['tp2'] if trade['direction'] == 'LONG' else price <= trade['tp2']
                tp3_hit_now = price >= trade['tp3'] if trade['direction'] == 'LONG' else price <= trade['tp3']
                sl_hit = price <= trade['stop_loss'] if trade['direction'] == 'LONG' else price >= trade['stop_loss']
                
                # Get SAVED hit status from trade record (persists across sessions)
                tp1_hit = trade.get('tp1_hit', False)
                tp2_hit = trade.get('tp2_hit', False)
                tp3_hit = trade.get('tp3_hit', False)
                highest_pnl = trade.get('highest_pnl', 0)
                
                # Check HISTORICAL HIGH to detect if targets were hit while not watching
                # This catches cases where TP was hit but we missed it
                try:
                    if not tp1_hit or not tp2_hit or not tp3_hit:
                        # Fetch recent candles to check high prices
                        df_check = fetch_binance_klines(trade['symbol'], '5m', 200)  # ~16 hours of data
                        if df_check is not None and len(df_check) > 0:
                            # Get the high since trade was added
                            added_time = trade.get('added_at', '')
                            if added_time:
                                try:
                                    from datetime import datetime
                                    trade_time = datetime.fromisoformat(added_time.replace('Z', '+00:00'))
                                    # Filter candles after trade was added
                                    df_check['datetime'] = pd.to_datetime(df_check['Open_time'], unit='ms')
                                    df_since_entry = df_check[df_check['datetime'] >= trade_time]
                                    if len(df_since_entry) > 0:
                                        if trade['direction'] == 'LONG':
                                            historical_high = df_since_entry['High'].max()
                                            historical_low = df_since_entry['Low'].min()
                                        else:
                                            historical_high = df_since_entry['High'].max()  # For short, high is bad
                                            historical_low = df_since_entry['Low'].min()   # Low is good for short
                                    else:
                                        historical_high = df_check['High'].max()
                                        historical_low = df_check['Low'].min()
                                except:
                                    historical_high = df_check['High'].max()
                                    historical_low = df_check['Low'].min()
                            else:
                                historical_high = df_check['High'].max()
                                historical_low = df_check['Low'].min()
                            
                            # Check if targets were hit historically
                            if trade['direction'] == 'LONG':
                                if not tp1_hit and historical_high >= trade['tp1']:
                                    tp1_hit = True
                                if not tp2_hit and historical_high >= trade['tp2']:
                                    tp2_hit = True
                                if not tp3_hit and historical_high >= trade['tp3']:
                                    tp3_hit = True
                            else:  # SHORT
                                if not tp1_hit and historical_low <= trade['tp1']:
                                    tp1_hit = True
                                if not tp2_hit and historical_low <= trade['tp2']:
                                    tp2_hit = True
                                if not tp3_hit and historical_low <= trade['tp3']:
                                    tp3_hit = True
                except Exception as e:
                    pass  # If historical check fails, continue with saved state
                
                # Update from current price (if hit now)
                if tp1_hit_now:
                    tp1_hit = True
                if tp2_hit_now:
                    tp2_hit = True
                if tp3_hit_now:
                    tp3_hit = True
                
                # Track highest PnL
                if pnl > highest_pnl:
                    highest_pnl = pnl
                
                # SAVE to trade record if changed (persists to JSON file)
                needs_save = False
                if tp1_hit and not trade.get('tp1_hit', False):
                    trade['tp1_hit'] = True
                    needs_save = True
                if tp2_hit and not trade.get('tp2_hit', False):
                    trade['tp2_hit'] = True
                    needs_save = True
                if tp3_hit and not trade.get('tp3_hit', False):
                    trade['tp3_hit'] = True
                    needs_save = True
                if highest_pnl > trade.get('highest_pnl', 0):
                    trade['highest_pnl'] = highest_pnl
                    needs_save = True
                
                # Save to file if any target was newly hit
                if needs_save:
                    st.session_state.active_trades[i] = trade
                    sync_active_trades(st.session_state.active_trades)
                
                # Distance calculations
                dist_to_tp1 = ((trade['tp1'] - price) / price * 100) if trade['direction'] == 'LONG' else ((price - trade['tp1']) / price * 100)
                dist_to_sl = ((price - trade['stop_loss']) / price * 100) if trade['direction'] == 'LONG' else ((trade['stop_loss'] - price) / price * 100)
                
                # ═══════════════════════════════════════════════════════════
                # AUTO-CLOSE COMPLETED TRADES (TP3 hit or SL hit)
                # PROFESSIONAL PARTIAL PROFIT TRACKING
                # ═══════════════════════════════════════════════════════════
                
                # Check if trade should be auto-closed
                should_auto_close = False
                close_reason = ""
                close_pnl = 0
                
                # Calculate R-multiple (how many R's risked/gained)
                if trade['direction'] == 'LONG':
                    initial_risk_pct = ((trade['entry'] - trade['stop_loss']) / trade['entry']) * 100
                    tp1_reward_pct = ((trade['tp1'] - trade['entry']) / trade['entry']) * 100
                    tp2_reward_pct = ((trade['tp2'] - trade['entry']) / trade['entry']) * 100
                    tp3_reward_pct = ((trade['tp3'] - trade['entry']) / trade['entry']) * 100
                else:
                    initial_risk_pct = ((trade['stop_loss'] - trade['entry']) / trade['entry']) * 100
                    tp1_reward_pct = ((trade['entry'] - trade['tp1']) / trade['entry']) * 100
                    tp2_reward_pct = ((trade['entry'] - trade['tp2']) / trade['entry']) * 100
                    tp3_reward_pct = ((trade['entry'] - trade['tp3']) / trade['entry']) * 100
                
                # ═══════════════════════════════════════════════════════════
                # PROFESSIONAL BLENDED P&L CALCULATION (33/33/33 Strategy)
                # ═══════════════════════════════════════════════════════════
                
                def calculate_blended_pnl(tp1_hit, tp2_hit, tp3_hit, sl_hit, 
                                         tp1_pct, tp2_pct, tp3_pct, current_pnl):
                    """
                    Calculate realistic P&L assuming 33/33/33 partial profit taking.
                    
                    Professional strategy:
                    - TP1: Take 33%, move SL to breakeven
                    - TP2: Take 33%, move SL to TP1  
                    - TP3: Take remaining 33%
                    
                    Returns: (blended_pnl, outcome_type, r_multiple)
                    """
                    if not tp1_hit:
                        # Never hit TP1 - full loss or still active
                        if sl_hit:
                            return (-initial_risk_pct, "FULL_LOSS", -1.0)
                        else:
                            return (current_pnl, "ACTIVE", current_pnl / initial_risk_pct if initial_risk_pct > 0 else 0)
                    
                    # TP1 was hit - trade was SUCCESSFUL (good entry identification)
                    portion1 = tp1_pct * 0.33  # First 33% closed at TP1
                    
                    if tp3_hit:
                        # FULL WIN - all targets hit
                        portion2 = tp2_pct * 0.33
                        portion3 = tp3_pct * 0.34
                        blended = portion1 + portion2 + portion3
                        r_mult = blended / initial_risk_pct if initial_risk_pct > 0 else 0
                        return (blended, "FULL_WIN", r_mult)
                    
                    if tp2_hit:
                        # TP2 hit - check if stopped at TP1 or still running
                        portion2 = tp2_pct * 0.33
                        if sl_hit or (current_pnl < tp2_pct * 0.5):  # Likely stopped at TP1
                            portion3 = tp1_pct * 0.34  # Remaining stopped at TP1
                            blended = portion1 + portion2 + portion3
                            r_mult = blended / initial_risk_pct if initial_risk_pct > 0 else 0
                            return (blended, "PARTIAL_WIN_TP2", r_mult)
                        else:
                            portion3 = current_pnl * 0.34  # Still running
                            blended = portion1 + portion2 + portion3
                            r_mult = blended / initial_risk_pct if initial_risk_pct > 0 else 0
                            return (blended, "PARTIAL_WIN_TP2_RUNNING", r_mult)
                    
                    # Only TP1 hit
                    if sl_hit or current_pnl <= 0.1:  # Stopped at breakeven
                        portion2 = 0  # Remaining 67% stopped at breakeven
                        blended = portion1
                        r_mult = blended / initial_risk_pct if initial_risk_pct > 0 else 0
                        return (blended, "BREAKEVEN", r_mult)
                    else:
                        # TP1 hit, still running toward TP2
                        portion2 = current_pnl * 0.67  # Remaining at current price
                        blended = portion1 + portion2
                        r_mult = blended / initial_risk_pct if initial_risk_pct > 0 else 0
                        return (blended, "PARTIAL_WIN_TP1_RUNNING", r_mult)
                
                # Calculate blended P&L
                blended_pnl, outcome_type, r_multiple = calculate_blended_pnl(
                    tp1_hit, tp2_hit, tp3_hit, sl_hit,
                    tp1_reward_pct, tp2_reward_pct, tp3_reward_pct, pnl
                )
                
                # Store for display
                trade['blended_pnl'] = blended_pnl
                trade['outcome_type'] = outcome_type
                trade['r_multiple'] = r_multiple
                
                # ═══════════════════════════════════════════════════════════
                # AUTO-CLOSE CONDITIONS
                # ═══════════════════════════════════════════════════════════
                
                if tp3_hit and not trade.get('closed', False):
                    # TP3 HIT = FULL WIN! Auto-close this trade
                    should_auto_close = True
                    close_reason = "TP3_HIT"
                    close_pnl = blended_pnl  # Use blended P&L
                
                if sl_hit and not trade.get('closed', False):
                    # SL HIT - Could be LOSS or BREAKEVEN depending on TP1
                    should_auto_close = True
                    if tp1_hit:
                        close_reason = "STOPPED_AFTER_TP1"  # Breakeven or small win
                    else:
                        close_reason = "SL_HIT"  # Full loss
                    close_pnl = blended_pnl
                
                # Auto-close the trade if conditions met
                if should_auto_close:
                    trade['closed'] = True
                    trade['closed_at'] = datetime.now().isoformat()
                    trade['close_reason'] = close_reason
                    trade['final_pnl'] = close_pnl
                    trade['blended_pnl'] = blended_pnl
                    trade['outcome_type'] = outcome_type
                    trade['r_multiple'] = r_multiple
                    trade['tp1_hit'] = tp1_hit
                    trade['tp2_hit'] = tp2_hit
                    trade['tp3_hit'] = tp3_hit
                    
                    # Determine status based on outcome
                    if outcome_type == "FULL_WIN":
                        trade['status'] = 'WIN'
                    elif outcome_type in ["PARTIAL_WIN_TP2", "PARTIAL_WIN_TP1", "BREAKEVEN"]:
                        trade['status'] = 'PARTIAL'  # TP1 hit = successful identification
                    elif outcome_type == "STOPPED_AFTER_TP1":
                        trade['status'] = 'PARTIAL'  # Still a successful trade
                    else:
                        trade['status'] = 'LOSS'
                    
                    # STEP 1: Save closed trade to SESSION STATE (survives page refresh)
                    if 'closed_trades' not in st.session_state:
                        st.session_state.closed_trades = []
                    # Remove old version if exists, then add new
                    st.session_state.closed_trades = [t for t in st.session_state.closed_trades 
                                                      if not (t.get('symbol') == trade['symbol'] and t.get('added_at') == trade.get('added_at'))]
                    st.session_state.closed_trades.append(trade)
                    
                    # STEP 2: Save to FILE (backup for longer persistence)
                    all_trades = load_trade_history()
                    # Remove old version of this trade if exists
                    all_trades = [t for t in all_trades if not (t.get('symbol') == trade['symbol'] and t.get('added_at') == trade.get('added_at'))]
                    # Add the closed trade
                    all_trades.append(trade)
                    save_trade_history(all_trades)
                    
                    # STEP 3: Remove from active trades in session state
                    st.session_state.active_trades = [t for t in st.session_state.active_trades 
                                                      if not (t.get('symbol') == trade['symbol'] and t.get('added_at') == trade.get('added_at'))]
                    
                    # STEP 4: Show notification ONLY IF NOT ALREADY NOTIFIED
                    closure_key = f"{trade['symbol']}_{trade.get('added_at', '')}"
                    if closure_key not in st.session_state.get('notified_closures', set()):
                        outcome_emoji = "🏆" if outcome_type == "FULL_WIN" else "✅" if "PARTIAL" in str(outcome_type) or "BREAKEVEN" in str(outcome_type) else "🔴"
                        st.toast(f"{outcome_emoji} {trade['symbol']} closed! {close_pnl:+.1f}% ({r_multiple:.1f}R) → Moved to Performance", icon=outcome_emoji)
                        
                        # Mark as notified
                        if 'notified_closures' not in st.session_state:
                            st.session_state.notified_closures = set()
                        st.session_state.notified_closures.add(closure_key)
                    
                    # Refresh to update display
                    st.rerun()
                
                # Status display - Professional outcome types
                if trade.get('closed', False):
                    otype = trade.get('outcome_type', '')
                    final = trade.get('final_pnl', 0)
                    r_mult = trade.get('r_multiple', 0)
                    
                    if otype == "FULL_WIN":
                        status = f"🏆 FULL WIN! +{final:.1f}% ({r_mult:.1f}R)"
                        emoji = "🏆"
                        status_color = "#00ff88"
                    elif otype in ["PARTIAL_WIN_TP2", "PARTIAL_WIN_TP2_RUNNING"]:
                        status = f"✅ PARTIAL WIN (TP2) +{final:.1f}% ({r_mult:.1f}R)"
                        emoji = "✅"
                        status_color = "#00d4aa"
                    elif otype == "BREAKEVEN" or trade.get('close_reason') == "STOPPED_AFTER_TP1":
                        status = f"🟡 BREAKEVEN +{final:.1f}% ({r_mult:.1f}R)"
                        emoji = "🟡"
                        status_color = "#ffcc00"
                    elif otype == "FULL_LOSS":
                        status = f"🔴 LOSS {final:.1f}% ({r_mult:.1f}R)"
                        emoji = "🔴"
                        status_color = "#ff4444"
                    else:
                        status = f"📊 CLOSED {final:+.1f}%"
                        emoji = "📊"
                        status_color = "#888888"
                elif sl_hit:
                    status = f"🔴 STOPPED OUT ({outcome_type})"
                    emoji = "🔴"
                    status_color = "#ff4444"
                elif tp3_hit:
                    status = f"🏆 TP3 HIT - FULL WIN! ({r_multiple:.1f}R)"
                    emoji = "🏆"
                    status_color = "#00ff88"
                elif tp2_hit:
                    if tp2_hit_now:
                        status = f"🎯 TP2 HIT - Take Profit! ({r_multiple:.1f}R)"
                    else:
                        status = f"🎯 TP2 HIT (retraced) ({r_multiple:.1f}R)"
                    emoji = "🎯"
                    status_color = "#00d4aa"
                elif tp1_hit:
                    if tp1_hit_now:
                        status = f"✅ TP1 HIT - Take 33%! ({r_multiple:.1f}R)"
                    else:
                        status = f"✅ TP1 HIT (retraced) ({r_multiple:.1f}R)"
                    emoji = "✅"
                    status_color = "#00d4aa" if tp1_hit_now else "#ffcc00"
                elif pnl > 5:
                    status = "🟢 STRONG PROFIT"
                    emoji = "🟢"
                    status_color = "#00d4aa"
                elif pnl > 0:
                    status = "🟢 IN PROFIT"
                    emoji = "🟢"
                    status_color = "#00d4aa"
                elif pnl > -3:
                    status = "🟡 ACTIVE"
                    emoji = "🟡"
                    status_color = "#ffcc00"
                else:
                    status = "🟠 DRAWDOWN"
                    emoji = "🟠"
                    status_color = "#ff9500"
                
                # Convert to Python bool to avoid numpy bool issue
                should_expand = bool(pnl < -5 or tp1_hit)
                
                # Track expander state - use session state to persist across reruns
                expander_key = f"exp_{trade['symbol']}"
                if expander_key not in st.session_state.monitor_expanded:
                    # First time: use calculated should_expand
                    st.session_state.monitor_expanded[expander_key] = should_expand
                
                # Always use tracked state (preserves user interactions)
                is_expanded = st.session_state.monitor_expanded.get(expander_key, should_expand)
                with st.expander(f"{emoji} {trade['symbol']} | P&L: {pnl:+.2f}% | {status}", expanded=is_expanded):
                    
                    # ═══════════════════════════════════════════════════════════
                    # 🏷️ MODE + TIMEFRAME BADGES (NEW!)
                    # ═══════════════════════════════════════════════════════════
                    
                    # Get mode and timeframe
                    mode_name = trade.get('mode_name') or trade.get('mode', '')
                    timeframe = trade.get('timeframe', 'N/A')
                    grade = trade.get('grade', 'N/A')
                    created = trade.get('created_at', '')[:10] if trade.get('created_at') else 'N/A'
                    
                    # IMPORTANT: Derive correct mode from timeframe if mismatch
                    # This fixes issues like "Swing + 5m" which is incorrect
                    tf_to_mode = {
                        '1m': 'Scalp', '3m': 'Scalp', '5m': 'Scalp',
                        '15m': 'Day Trade', '30m': 'Day Trade', '1h': 'Day Trade',
                        '4h': 'Swing', '1d': 'Swing',
                        '1w': 'Investment', '1M': 'Investment'
                    }
                    correct_mode = tf_to_mode.get(timeframe, mode_name or 'Unknown')
                    
                    # Use correct mode (override if mismatch)
                    if mode_name and mode_name != correct_mode and timeframe in tf_to_mode:
                        mode_name = correct_mode  # Fix the mismatch
                    elif not mode_name:
                        mode_name = correct_mode
                    
                    # Mode colors
                    mode_colors = {
                        'Scalp': '#ff9500', 'scalp': '#ff9500',
                        'Day Trade': '#00d4ff', 'day_trade': '#00d4ff',
                        'Swing': '#00d4aa', 'swing': '#00d4aa',
                        'Position': '#9d4edd', 'position': '#9d4edd',
                        'Investment': '#ffd700', 'investment': '#ffd700'
                    }
                    mode_color = mode_colors.get(mode_name, '#888888')
                    
                    st.markdown(f"""
                    <div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;'>
                        <span style='background: {mode_color}; color: #000; padding: 4px 12px; 
                                     border-radius: 15px; font-weight: bold; font-size: 0.85em;'>
                            {mode_name}
                        </span>
                        <span style='background: #333; color: #fff; padding: 4px 12px; 
                                     border-radius: 15px; font-size: 0.85em;'>
                            ⏱️ {timeframe}
                        </span>
                        <span style='background: #1a1a2e; color: #00d4ff; padding: 4px 12px; 
                                     border-radius: 15px; font-size: 0.85em; border: 1px solid #333;'>
                            Grade: {grade}
                        </span>
                        <span style='background: #1a1a2e; color: #888; padding: 4px 12px; 
                                     border-radius: 15px; font-size: 0.8em; border: 1px solid #333;'>
                            📅 {created}
                        </span>
                        <span style='background: #1a1a2e; color: #888; padding: 4px 12px; 
                                     border-radius: 15px; font-size: 0.8em; border: 1px solid #333;'>
                            {trade['direction']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # 🐋 WHALE DATA SECTION (NEW!)
                    # ═══════════════════════════════════════════════════════════
                    
                    # Only fetch for crypto (has USDT suffix typically)
                    if 'USDT' in trade['symbol'].upper() or not any(c.isalpha() and c.isupper() for c in trade['symbol'][-3:]):
                        try:
                            whale_data = get_whale_analysis(trade['symbol'])
                            
                            # Check if we got real data (not just defaults)
                            has_real_data = (
                                whale_data.get('signals') or 
                                whale_data.get('open_interest', {}).get('change_24h', 0) != 0 or
                                whale_data.get('funding', {}).get('rate', 0) != 0
                            )
                            
                            if has_real_data:
                                # Get raw metrics
                                oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                                price_change = whale_data.get('price_change_24h', 0)
                                funding = whale_data.get('funding', {}).get('rate_pct', 0)
                                whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                                retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
                                
                                oi_interp = whale_data.get('oi_interpretation', {})
                                
                                # Use NEW unified scenario system (not broken old verdict!)
                                from core.institutional_scoring import analyze_institutional_data, get_scenario_display
                                tech_dir = "BULLISH" if trade['direction'] == "LONG" else "BEARISH"
                                inst_score = analyze_institutional_data(
                                    oi_change, price_change, whale_long, retail_long, 
                                    whale_data.get('funding', {}).get('rate', 0), tech_dir
                                )
                                scenario_display = get_scenario_display(inst_score.scenario)
                                
                                # Determine color from NEW system
                                if inst_score.direction_bias == "LONG":
                                    v_color = '#00d4aa'
                                    verdict_text = "BULLISH"
                                elif inst_score.direction_bias == "SHORT":
                                    v_color = '#ff4444'
                                    verdict_text = "BEARISH"
                                else:
                                    v_color = '#888888'
                                    verdict_text = "NEUTRAL"
                                
                                st.markdown(f"""
                                <div style='background: #1a1a2e; border-radius: 8px; padding: 12px; margin-bottom: 12px;
                                            border-left: 3px solid {v_color};'>
                                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                                        <span style='color: #00d4ff; font-weight: bold;'>🐋 Whale Analysis</span>
                                        <span style='background: {v_color}33; color: {v_color}; padding: 3px 10px; 
                                                     border-radius: 12px; font-size: 0.85em; font-weight: bold;'>
                                            {inst_score.scenario_name}
                                        </span>
                                    </div>
                                    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;'>
                                        <div style='background: #252540; padding: 6px; border-radius: 4px; text-align: center;'>
                                            <div style='color: #888; font-size: 0.7em;'>OI 24h</div>
                                            <div style='color: {"#00d4aa" if oi_change > 3 else "#ff4444" if oi_change < -3 else "#888"}; font-weight: bold;'>{oi_change:+.1f}%</div>
                                        </div>
                                        <div style='background: #252540; padding: 6px; border-radius: 4px; text-align: center;'>
                                            <div style='color: #888; font-size: 0.7em;'>Funding</div>
                                            <div style='color: {"#ff4444" if abs(funding) > 0.05 else "#888"}; font-weight: bold;'>{funding:.3f}%</div>
                                        </div>
                                        <div style='background: #252540; padding: 6px; border-radius: 4px; text-align: center;'>
                                            <div style='color: #888; font-size: 0.7em;'>🐋 Whales</div>
                                            <div style='color: {"#00d4aa" if whale_long > 55 else "#ff4444" if whale_long < 45 else "#888"}; font-weight: bold;'>{whale_long:.0f}% L</div>
                                        </div>
                                        <div style='background: #252540; padding: 6px; border-radius: 4px; text-align: center;'>
                                            <div style='color: #888; font-size: 0.7em;'>🐑 Retail</div>
                                            <div style='color: {"#ff4444" if retail_long > 65 else "#00d4aa" if retail_long < 35 else "#888"}; font-weight: bold;'>{retail_long:.0f}% L</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # OI + Price Interpretation (the key insight!)
                                if oi_interp and oi_interp.get('interpretation'):
                                    interp_emoji = oi_interp.get('emoji', '📊')
                                    interp_color = "#00d4aa" if interp_emoji == '🟢' else "#ff4444" if interp_emoji == '🔴' else "#ffcc00"
                                    st.markdown(f"""
                                    <div style='background: {interp_color}11; border-left: 3px solid {interp_color}; 
                                                padding: 8px 12px; border-radius: 6px; margin-bottom: 10px;'>
                                        <div style='color: {interp_color}; font-weight: bold; font-size: 0.9em;'>
                                            {interp_emoji} {oi_interp.get('interpretation', '')}
                                        </div>
                                        <div style='color: #888; font-size: 0.8em; margin-top: 4px;'>
                                            → {oi_interp.get('action', '')}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # ═══════════════════════════════════════════════════════════
                                # 🎯 COMBINED CONCLUSION (What all metrics mean together!)
                                # ═══════════════════════════════════════════════════════════
                                
                                # Build combined interpretation
                                conclusions = []
                                trade_bias = trade['direction']  # LONG or SHORT
                                
                                # 1. OI + Price alignment with trade
                                oi_signal = oi_interp.get('signal', '')
                                if trade_bias == 'LONG':
                                    if oi_signal == 'NEW_LONGS':
                                        conclusions.append(("✅", "OI supports your LONG - New longs entering"))
                                    elif oi_signal == 'NEW_SHORTS':
                                        conclusions.append(("⚠️", "OI against your LONG - New shorts entering"))
                                    elif oi_signal == 'SHORT_COVERING':
                                        conclusions.append(("🟡", "Rally may be weak (short covering) - Watch for reversal"))
                                    elif oi_signal == 'LONG_LIQUIDATION':
                                        conclusions.append(("❌", "Longs being liquidated - Your LONG at risk"))
                                else:  # SHORT
                                    if oi_signal == 'NEW_SHORTS':
                                        conclusions.append(("✅", "OI supports your SHORT - New shorts entering"))
                                    elif oi_signal == 'NEW_LONGS':
                                        conclusions.append(("⚠️", "OI against your SHORT - New longs entering"))
                                    elif oi_signal == 'LONG_LIQUIDATION':
                                        conclusions.append(("🟡", "Dump may be weak (long liquidation) - Watch for reversal"))
                                    elif oi_signal == 'SHORT_COVERING':
                                        conclusions.append(("❌", "Shorts being covered - Your SHORT at risk"))
                                
                                # 2. Funding alignment
                                if abs(funding) > 0.05:
                                    if funding > 0 and trade_bias == 'SHORT':
                                        conclusions.append(("✅", f"Funding ({funding:.3f}%) = Longs overleveraged → Supports your SHORT"))
                                    elif funding > 0 and trade_bias == 'LONG':
                                        conclusions.append(("⚠️", f"Funding ({funding:.3f}%) = Longs overleveraged → Risk for your LONG"))
                                    elif funding < 0 and trade_bias == 'LONG':
                                        conclusions.append(("✅", f"Funding ({funding:.3f}%) = Shorts overleveraged → Supports your LONG"))
                                    elif funding < 0 and trade_bias == 'SHORT':
                                        conclusions.append(("⚠️", f"Funding ({funding:.3f}%) = Shorts overleveraged → Risk for your SHORT"))
                                
                                # 3. Whale positioning
                                if trade_bias == 'LONG':
                                    if whale_long > 55:
                                        conclusions.append(("✅", f"Whales are bullish ({whale_long:.0f}% Long) → Aligns with your LONG"))
                                    elif whale_long < 45:
                                        conclusions.append(("⚠️", f"Whales are bearish ({whale_long:.0f}% Long) → Against your LONG"))
                                else:
                                    if whale_long < 45:
                                        conclusions.append(("✅", f"Whales are bearish ({whale_long:.0f}% Long) → Aligns with your SHORT"))
                                    elif whale_long > 55:
                                        conclusions.append(("⚠️", f"Whales are bullish ({whale_long:.0f}% Long) → Against your SHORT"))
                                
                                # 4. Retail divergence (fade retail)
                                if retail_long > 65 and trade_bias == 'SHORT':
                                    conclusions.append(("✅", f"Retail heavily long ({retail_long:.0f}%) → Good for your SHORT (fade retail)"))
                                elif retail_long < 35 and trade_bias == 'LONG':
                                    conclusions.append(("✅", f"Retail heavily short ({retail_long:.0f}%) → Good for your LONG (fade retail)"))
                                elif retail_long > 65 and trade_bias == 'LONG':
                                    conclusions.append(("🟡", f"You're with the retail crowd ({retail_long:.0f}% Long) - Be cautious"))
                                elif retail_long < 35 and trade_bias == 'SHORT':
                                    conclusions.append(("🟡", f"You're with the retail crowd ({100-retail_long:.0f}% Short) - Be cautious"))
                                
                                # Display combined conclusion
                                if conclusions:
                                    positive = sum(1 for c in conclusions if c[0] == "✅")
                                    negative = sum(1 for c in conclusions if c[0] in ["⚠️", "❌"])
                                    
                                    if positive >= 2 and negative == 0:
                                        summary_text = f"🟢 **STRONG** - Whale data confirms your {trade_bias}"
                                        summary_color = "#00d4aa"
                                    elif positive > negative:
                                        summary_text = f"🟢 **SUPPORTIVE** - Whale data leans toward your {trade_bias}"
                                        summary_color = "#00d4aa"
                                    elif negative > positive:
                                        summary_text = f"🔴 **CAUTION** - Whale data conflicts with your {trade_bias}"
                                        summary_color = "#ff4444"
                                    else:
                                        summary_text = f"🟡 **MIXED** - Whale signals are conflicting"
                                        summary_color = "#ffcc00"
                                    
                                    st.markdown(f"""
                                    <div style='background: #1a1a2e; border: 1px solid {summary_color}; border-radius: 8px; 
                                                padding: 10px; margin-bottom: 10px;'>
                                        <div style='color: {summary_color}; font-weight: bold; margin-bottom: 8px;'>
                                            🎯 Combined Conclusion for YOUR {trade_bias}:
                                        </div>
                                        <div style='color: {summary_color}; font-size: 1em; margin-bottom: 8px;'>
                                            {summary_text}
                                        </div>
                                        <div style='color: #aaa; font-size: 0.85em;'>
                                            {'<br>'.join([f"{c[0]} {c[1]}" for c in conclusions])}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # 📖 Full Story (Collapsible) - uses Combined Learning
                                    try:
                                        from core.unified_scoring import generate_combined_learning
                                        
                                        # Calculate position in range
                                        position_pct_mon = 50
                                        entry_p = trade.get('entry', 0)
                                        stop_p = trade.get('stop', 0)
                                        tp1_p = trade.get('tp1', 0)
                                        if entry_p and stop_p and tp1_p:
                                            if trade_bias == 'LONG' and tp1_p > stop_p:
                                                range_s = tp1_p - stop_p
                                                position_pct_mon = ((price - stop_p) / range_s) * 100 if range_s > 0 else 50
                                            elif trade_bias == 'SHORT' and stop_p > tp1_p:
                                                range_s = stop_p - tp1_p
                                                position_pct_mon = ((stop_p - price) / range_s) * 100 if range_s > 0 else 50
                                        position_pct_mon = max(0, min(100, position_pct_mon))
                                        
                                        cl_mon = generate_combined_learning(
                                            signal_name=trade_bias,
                                            direction=trade_bias,
                                            whale_pct=whale_long,
                                            retail_pct=retail_long,
                                            oi_change=oi_data_mon.get('change_24h', 0),
                                            price_change=whale_data.get('real_whale_data', {}).get('price_change_24h', 0) if whale_data.get('real_whale_data') else 0,
                                            money_flow_phase='',
                                            structure_type='',
                                            position_pct=position_pct_mon,
                                            ta_score=50,
                                        )
                                        
                                        with st.expander("📖 **Full Story** - Learn What's Happening", expanded=False):
                                            for tl, ct in cl_mon['stories']:
                                                st.markdown(f"""
                                                <div style='background: #0d0d1a; border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 3px solid #333;'>
                                                    <div style='color: #666; font-size: 0.8em;'>{tl}</div>
                                                    <div style='color: #bbb; font-size: 0.9em;'>{ct}</div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                    except:
                                        pass
                        except Exception as e:
                            # Silently skip whale data if unavailable
                            pass
                    
                    # Trade-specific alerts
                    trade_alerts = check_trade_alerts(trade, price)
                    if trade_alerts:
                        for alert in trade_alerts:
                            if alert.urgency in ["high", "medium"]:
                                st.markdown(f"""
                                <div style='background: {alert.color}22; border-left: 3px solid {alert.color}; 
                                            padding: 8px 12px; border-radius: 6px; margin-bottom: 10px;'>
                                    <span style='color: {alert.color}; font-weight: bold;'>{alert.emoji} {alert.message}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.markdown("### 💵 Position")
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px; border-radius: 8px;'>
                            <div><span style='color: #888;'>Current:</span> <strong style='color: #00d4ff;'>{fmt_price(price)}</strong></div>
                            <div><span style='color: #888;'>Entry:</span> <strong>{fmt_price(trade['entry'])}</strong></div>
                            <div><span style='color: #888;'>P&L:</span> <strong style='color: {status_color};'>{pnl:+.2f}%</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c2:
                        st.markdown("### 🎯 Targets")
                        
                        # Calculate ROIs
                        tp1_roi = calc_roi(trade['tp1'], trade['entry'])
                        tp2_roi = calc_roi(trade['tp2'], trade['entry'])
                        tp3_roi = calc_roi(trade['tp3'], trade['entry'])
                        
                        # Calculate risk % for R:R
                        trade_risk_pct = trade.get('risk', 0)
                        if trade_risk_pct <= 0:
                            # Fallback: calculate from entry and stop_loss
                            trade_risk_pct = abs((trade['entry'] - trade['stop_loss']) / trade['entry'] * 100) if trade['entry'] > 0 else 5.0
                        
                        # Calculate R:R for each TP level
                        rr_tp1 = tp1_roi / trade_risk_pct if trade_risk_pct > 0 else 0
                        rr_tp2 = tp2_roi / trade_risk_pct if trade_risk_pct > 0 else 0
                        rr_tp3 = tp3_roi / trade_risk_pct if trade_risk_pct > 0 else 0
                        
                        # Color code R:R - green >= 1, yellow >= 0.5, red < 0.5
                        rr1_color = "#00d4aa" if rr_tp1 >= 1 else "#ffcc00" if rr_tp1 >= 0.5 else "#ff4444"
                        rr2_color = "#00d4aa" if rr_tp2 >= 1 else "#ffcc00" if rr_tp2 >= 0.5 else "#ff4444"
                        rr3_color = "#00d4aa" if rr_tp3 >= 1 else "#ffcc00" if rr_tp3 >= 0.5 else "#ff4444"
                        
                        # TP1 - Show if hit AND if currently at target or retraced
                        if tp1_hit:
                            if tp1_hit_now:
                                tp1_status = "✅ HIT"
                                tp1_style = "background: #1a3a1a; border: 1px solid #00d4aa;"
                            else:
                                tp1_status = "✅ HIT (retraced)"
                                tp1_style = "background: #2a2a1a; border: 1px solid #ffcc00;"
                        else:
                            tp1_status = f"⏳ {dist_to_tp1:.1f}% away"
                            tp1_style = "background: #1a2a1a;"
                        
                        st.markdown(f"""
                        <div style='{tp1_style} padding: 8px 12px; border-radius: 6px; margin-bottom: 5px;'>
                            <strong>TP1:</strong> {fmt_price(trade['tp1'])} 
                            <span style='color: #00d4aa;'>(+{tp1_roi:.1f}%)</span>
                            <span style='color: {rr1_color}; font-weight: bold;'>R:R {rr_tp1:.1f}:1</span>
                            <span style='color: {"#00d4aa" if tp1_hit else "#888"}; float: right;'>{tp1_status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # TP2
                        if tp2_hit:
                            if tp2_hit_now:
                                tp2_status = "✅ HIT"
                                tp2_style = "background: #1a3a1a; border: 1px solid #00d4aa;"
                            else:
                                tp2_status = "✅ HIT (retraced)"
                                tp2_style = "background: #2a2a1a; border: 1px solid #ffcc00;"
                        else:
                            tp2_status = "⏳"
                            tp2_style = "background: #1a2a1a;"
                        
                        st.markdown(f"""
                        <div style='{tp2_style} padding: 8px 12px; border-radius: 6px; margin-bottom: 5px;'>
                            <strong>TP2:</strong> {fmt_price(trade['tp2'])} 
                            <span style='color: #00d4aa;'>(+{tp2_roi:.1f}%)</span>
                            <span style='color: {rr2_color}; font-weight: bold;'>R:R {rr_tp2:.1f}:1</span>
                            <span style='color: {"#00d4aa" if tp2_hit else "#888"}; float: right;'>{tp2_status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # TP3
                        if tp3_hit:
                            if tp3_hit_now:
                                tp3_status = "✅ HIT"
                                tp3_style = "background: #1a3a1a; border: 1px solid #00d4aa;"
                            else:
                                tp3_status = "✅ HIT (retraced)"
                                tp3_style = "background: #2a2a1a; border: 1px solid #ffcc00;"
                        else:
                            tp3_status = "⏳"
                            tp3_style = "background: #1a2a1a;"
                        
                        st.markdown(f"""
                        <div style='{tp3_style} padding: 8px 12px; border-radius: 6px;'>
                            <strong>TP3:</strong> {fmt_price(trade['tp3'])} 
                            <span style='color: #00d4aa;'>(+{tp3_roi:.1f}%)</span>
                            <span style='color: {rr3_color}; font-weight: bold;'>R:R {rr_tp3:.1f}:1</span>
                            <span style='color: {"#00d4aa" if tp3_hit else "#888"}; float: right;'>{tp3_status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c3:
                        st.markdown("### ⚠️ Risk")
                        
                        # Calculate risk budget consumed
                        entry = trade['entry']
                        sl = trade['stop_loss']
                        
                        if trade['direction'] == 'LONG':
                            initial_risk = ((entry - sl) / entry) * 100
                        else:
                            initial_risk = ((sl - entry) / entry) * 100
                        
                        # Risk consumed: 0% = at entry, 100% = at SL
                        if initial_risk > 0:
                            risk_consumed = ((initial_risk - dist_to_sl) / initial_risk) * 100
                        else:
                            risk_consumed = 0
                        
                        # IMPROVED LOGIC based on risk budget consumed
                        # < 20% consumed: Safe (just entered or winning)
                        # 20-40% consumed: Watch (some movement toward SL)
                        # 40-60% consumed: Getting Close (halfway)
                        # 60-80% consumed: Danger Zone (majority of risk used)
                        # > 80% consumed: Critical (almost at SL)
                        
                        if dist_to_sl < 0.8 or risk_consumed > 80:
                            sl_color = "#ff0000"
                            sl_status = "🚨 CRITICAL - CLOSE NOW!"
                            sl_bg = "#2a0a0a"
                        elif risk_consumed > 60:
                            sl_color = "#ff4444"
                            sl_status = f"⚠️ DANGER ZONE ({risk_consumed:.0f}% risk used)"
                            sl_bg = "#2a1a1a"
                        elif risk_consumed > 40:
                            sl_color = "#ff9500"
                            sl_status = f"🟠 Halfway ({risk_consumed:.0f}% risk used)"
                            sl_bg = "#2a2a1a"
                        elif risk_consumed > 20:
                            sl_color = "#ffcc00"
                            sl_status = f"🟡 Watch ({risk_consumed:.0f}% risk used)"
                            sl_bg = "#1a1a1a"
                        else:
                            sl_color = "#00d4aa"
                            sl_status = f"🟢 Safe ({risk_consumed:.0f}% risk used)"
                            sl_bg = "#1a2a1a"
                        
                        if sl_hit:
                            sl_status = "❌ STOPPED OUT"
                            sl_color = "#ff0000"
                            sl_bg = "#2a0a0a"
                        
                        st.markdown(f"""
                        <div style='background: {sl_bg}; padding: 10px; border-radius: 8px; border: 1px solid {sl_color};'>
                            <div><strong>Stop Loss:</strong> {fmt_price(trade['stop_loss'])}</div>
                            <div style='margin: 8px 0; padding: 6px; background: {sl_color}22; border-radius: 4px;'>
                                <span style='color: {sl_color}; font-weight: bold;'>{sl_status}</span>
                            </div>
                            <div style='color: {sl_color}; font-size: 1.2em; font-weight: bold;'>
                                {dist_to_sl:.2f}% to SL
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show "CLOSE NOW" guidance ONLY if actually in danger (using risk_consumed)
                        # NOT just because dist_to_sl < 2% (that's initial risk, not danger!)
                        if (risk_consumed > 60 or dist_to_sl < 0.8) and not sl_hit:
                            st.markdown(f"""
                            <div style='background: #ff444422; border: 1px solid #ff4444; border-radius: 6px; 
                                        padding: 8px; margin-top: 8px; text-align: center;'>
                                <span style='color: #ff4444; font-weight: bold;'>⚡ Close on Binance NOW!</span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # 📊 REAL-TIME GRADE WITH BREAKDOWN
                    # ═══════════════════════════════════════════════════════════
                    
                    # Fetch current data for real-time grade
                    try:
                        df_grade = fetch_binance_klines(trade['symbol'], trade.get('timeframe', '15m'), 100)
                        if df_grade is not None and len(df_grade) >= 30:
                            from core.money_flow import calculate_money_flow
                            from core.smc_detector import detect_smc
                            from core.education import calculate_realtime_grade
                            from core.indicators import calculate_ema, calculate_vwap
                            
                            mf_grade = calculate_money_flow(df_grade)
                            smc_grade = detect_smc(df_grade)
                            
                            # Build trend data
                            ema_9 = calculate_ema(df_grade['Close'], 9).iloc[-1]
                            ema_20 = calculate_ema(df_grade['Close'], 20).iloc[-1]
                            ema_50 = calculate_ema(df_grade['Close'], 50).iloc[-1]
                            
                            trend_data = {
                                'ema_bullish': ema_9 > ema_20 > ema_50,
                                'ema_bearish': ema_9 < ema_20 < ema_50
                            }
                            
                            # VWAP - pass individual columns
                            vwap_data = calculate_vwap(
                                df_grade['High'], 
                                df_grade['Low'], 
                                df_grade['Close'], 
                                df_grade['Volume']
                            )
                            
                            # Calculate real-time grade
                            grade_breakdown = calculate_realtime_grade(
                                mf_grade, smc_grade, trend_data, vwap_data
                            )
                            
                            # Show grade breakdown in expander
                            original_grade = trade.get('grade', 'N/A')
                            grade_changed = original_grade != 'N/A' and original_grade != grade_breakdown.grade
                            
                            with st.expander(f"📊 **Grade: {grade_breakdown.grade_emoji} {grade_breakdown.grade}** (Score: {grade_breakdown.total_score}/100)" + 
                                           (f" ⚠️ Changed from {original_grade}!" if grade_changed else "")):
                                
                                # Summary
                                summary_color = "#00d4aa" if grade_breakdown.total_score >= 55 else "#ffcc00" if grade_breakdown.total_score >= 45 else "#ff4444"
                                st.markdown(f"""
                                <div style='background: {summary_color}22; border-left: 4px solid {summary_color}; padding: 10px; border-radius: 8px; margin-bottom: 15px;'>
                                    <strong style='color: {summary_color};'>{grade_breakdown.summary}</strong>
                                    <div style='color: #888; margin-top: 5px; font-size: 0.9em;'>
                                        Bullish: +{grade_breakdown.bullish_points} pts | Bearish: -{grade_breakdown.bearish_points} pts | Net: {grade_breakdown.bullish_points - grade_breakdown.bearish_points:+d}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Factor breakdown
                                st.markdown("#### 📋 Contributing Factors")
                                
                                for factor in grade_breakdown.factors:
                                    point_color = "#00d4aa" if factor['points'].startswith('+') else "#ff4444"
                                    st.markdown(f"""
                                    <div style='background: #1a1a2e; padding: 8px 12px; border-radius: 6px; margin: 4px 0; 
                                                display: flex; justify-content: space-between; align-items: center;'>
                                        <div>
                                            <span style='font-size: 1.1em;'>{factor['emoji']}</span>
                                            <strong style='margin-left: 8px;'>{factor['name']}</strong>
                                            <div style='color: #888; font-size: 0.85em; margin-left: 28px;'>{factor['description']}</div>
                                        </div>
                                        <span style='color: {point_color}; font-weight: bold; font-size: 1.1em;'>{factor['points']}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if grade_changed:
                                    st.warning(f"⚠️ Grade changed from **{original_grade}** at entry to **{grade_breakdown.grade}** now. Conditions have shifted!")
                        else:
                            # No data - show original grade
                            st.markdown(f"**Grade:** {trade.get('grade', 'N/A')} *(from entry)*")
                    except Exception as e:
                        # Fallback to original grade if calculation fails
                        st.markdown(f"**Grade:** {trade.get('grade', 'N/A')} *(from entry)*")
                        # Uncomment below to debug:
                        # st.caption(f"Grade calc error: {str(e)}")
                    
                    # ═══════════════════════════════════════════════════════════
                    # SMART RECOMMENDATIONS
                    # ═══════════════════════════════════════════════════════════
                    
                    st.markdown("---")
                    st.markdown("### 💡 Recommendation")
                    
                    # Check if trade is closed first
                    if trade.get('closed', False):
                        if trade.get('close_reason') == 'TP3_HIT':
                            st.success(f"""
                            **🏆 TRADE COMPLETED - FULL WIN!**
                            
                            • Entry: {fmt_price(trade['entry'])}
                            • Exit: {fmt_price(trade['tp3'])} (TP3)
                            • Profit: **+{trade.get('final_pnl', 0):.1f}%**
                            
                            Congratulations! This trade hit all targets. 🎉
                            """)
                        else:  # SL_HIT
                            st.error(f"""
                            **Trade Closed - Stop Loss Hit**
                            
                            • Entry: {fmt_price(trade['entry'])}
                            • Exit: {fmt_price(trade['stop_loss'])} (SL)
                            • Loss: **{trade.get('final_pnl', 0):.1f}%**
                            
                            Review what happened and move on. Losses are part of trading.
                            """)
                    # Generate recommendation based on current state
                    elif sl_hit:
                        st.error("**Trade closed.** Stop loss was hit. Review what went wrong and move on.")
                    elif tp3_hit:
                        st.success("**🏆 WINNER!** Full target reached. Trade should be auto-closed.")
                    elif tp2_hit:
                        if tp2_hit_now:
                            st.success("**Take Profit:** TP2 hit! Close 50-75% of remaining position. Trail stop on rest.")
                        else:
                            st.warning("**TP2 was hit but retraced.** If still holding, move stop to at least TP1 level.")
                    elif tp1_hit:
                        if tp1_hit_now:
                            st.success("**Partial Profit:** TP1 hit! Take 33% profit. Move stop to breakeven on rest.")
                        else:
                            st.warning(f"**TP1 was hit but retraced to {pnl:+.1f}%.** If you took partial profit at TP1, you're managing well. Hold remaining with breakeven stop.")
                    elif pnl > 10:
                        st.info("**Strong profit.** Consider trailing your stop loss to lock in gains.")
                    elif pnl > 5:
                        st.info("**Healthy profit.** Move stop to breakeven to create a 'free trade'.")
                    elif pnl > 0:
                        st.info("**In profit.** Hold with original stop. Let the trade develop.")
                    elif pnl > -3:
                        # Check if DCA opportunity
                        if dist_to_sl > 4:
                            st.warning(f"**DCA Opportunity?** Price down {abs(pnl):.1f}% but {dist_to_sl:.1f}% from SL. Consider adding small amount if thesis intact.")
                        else:
                            st.info("**Active trade.** Monitor closely. Stay patient.")
                    elif pnl > -5:
                        st.warning(f"**Drawdown.** Trade underwater by {abs(pnl):.1f}%. Re-evaluate thesis. Is original reason still valid?")
                    else:
                        if dist_to_sl < 2:
                            st.error(f"**⚠️ DANGER!** Only {dist_to_sl:.1f}% from stop loss. Prepare for possible exit.")
                        else:
                            st.error(f"**Significant loss.** Down {abs(pnl):.1f}%. Consider reducing position size to manage risk.")
                    
                    # Show highest P&L achieved if retraced significantly (only for open trades)
                    if not trade.get('closed', False) and highest_pnl > pnl + 2:
                        st.caption(f"📈 Peak P&L: +{highest_pnl:.1f}% (currently {pnl:+.1f}%)")
                    
                    # Action buttons
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button(f"🗑️ Remove", key=f"remove_{trade['symbol']}_{i}"):
                            st.session_state.active_trades.pop(i)
                            sync_active_trades(st.session_state.active_trades)
                            # Skip price refresh on next rerun - use cached prices
                            st.session_state.skip_price_refresh = True
                            st.rerun()
                    
                    with btn_col2:
                        if tp1_hit and not tp2_hit:
                            if st.button(f"✅ Mark TP1 Taken", key=f"tp1_{trade['symbol']}_{i}"):
                                st.session_state.active_trades[i]['tp1_taken'] = True
                                sync_active_trades(st.session_state.active_trades)
                                st.success("Marked TP1 as taken!")
                    
                    with btn_col3:
                        # Use checkbox toggle instead of button - more stable, no jarring refresh
                        show_analysis_key = f"show_analysis_{trade['symbol']}"
                        if show_analysis_key not in st.session_state:
                            st.session_state[show_analysis_key] = False
                        
                        # Toggle button that doesn't cause jarring rerun
                        current_state = st.session_state.get(show_analysis_key, False)
                        button_label = "📊 Hide Analysis" if current_state else "📊 Full Analysis"
                        if st.button(button_label, key=f"analyze_{trade['symbol']}_{i}"):
                            st.session_state[show_analysis_key] = not current_state
                            # Keep this expander open after clicking Full Analysis
                            expander_key = f"exp_{trade['symbol']}"
                            st.session_state.monitor_expanded[expander_key] = True
                    
                    # ═══════════════════════════════════════════════════════════
                    # ENHANCED PROGRESS BAR WITH % LABELS
                    # ═══════════════════════════════════════════════════════════
                    total_range = trade['tp3'] - trade['stop_loss']
                    if total_range > 0:
                        # Calculate positions
                        sl_pos = 0
                        entry_pos = (trade['entry'] - trade['stop_loss']) / total_range
                        tp1_pos = (trade['tp1'] - trade['stop_loss']) / total_range
                        tp2_pos = (trade['tp2'] - trade['stop_loss']) / total_range
                        tp3_pos = 1.0
                        current_pos = max(0, min(1, (price - trade['stop_loss']) / total_range))
                        
                        # Calculate % to next targets (from current price)
                        dist_to_tp1_pct = ((trade['tp1'] - price) / price * 100) if price < trade['tp1'] else 0
                        dist_to_tp2_pct = ((trade['tp2'] - price) / price * 100) if price < trade['tp2'] else 0
                        dist_to_tp3_pct = ((trade['tp3'] - price) / price * 100) if price < trade['tp3'] else 0
                        
                        # Calculate SL and TP percentages (from entry price)
                        # For LONG: SL is below entry (negative), TPs are above (positive)
                        # For SHORT: SL is above entry, TPs are below - but we still show risk as loss
                        if trade['direction'] == 'LONG':
                            sl_pct_from_entry = abs((trade['entry'] - trade['stop_loss']) / trade['entry'] * 100)
                            tp1_pct_from_entry = ((trade['tp1'] - trade['entry']) / trade['entry'] * 100)
                            tp2_pct_from_entry = ((trade['tp2'] - trade['entry']) / trade['entry'] * 100)
                            tp3_pct_from_entry = ((trade['tp3'] - trade['entry']) / trade['entry'] * 100)
                        else:  # SHORT
                            sl_pct_from_entry = abs((trade['stop_loss'] - trade['entry']) / trade['entry'] * 100)
                            tp1_pct_from_entry = abs((trade['entry'] - trade['tp1']) / trade['entry'] * 100)
                            tp2_pct_from_entry = abs((trade['entry'] - trade['tp2']) / trade['entry'] * 100)
                            tp3_pct_from_entry = abs((trade['entry'] - trade['tp3']) / trade['entry'] * 100)
                        
                        # Create visual progress bar with markers
                        st.markdown(f"""
                        <div style='background: #1a1a2e; border-radius: 8px; padding: 10px; margin: 10px 0;'>
                            <div style='display: flex; justify-content: space-between; font-size: 0.8em; margin-bottom: 5px;'>
                                <span style='color: #ff4444;'>SL {fmt_price(trade['stop_loss'])} (-{sl_pct_from_entry:.1f}%)</span>
                                <span style='color: #888;'>Entry {fmt_price(trade['entry'])}</span>
                                <span style='color: #00d4aa;'>TP1 {fmt_price(trade['tp1'])} (+{tp1_pct_from_entry:.1f}%)</span>
                                <span style='color: #00d4aa;'>TP2 {fmt_price(trade['tp2'])} (+{tp2_pct_from_entry:.1f}%)</span>
                                <span style='color: #00ff88;'>TP3 {fmt_price(trade['tp3'])} (+{tp3_pct_from_entry:.1f}%)</span>
                            </div>
                            <div style='position: relative; height: 24px; background: linear-gradient(to right, 
                                #ff4444 0%, #ff4444 2%, 
                                #333 2%, #333 {entry_pos*100:.0f}%, 
                                #1a4a3a {entry_pos*100:.0f}%, #1a4a3a {tp1_pos*100:.0f}%,
                                #1a6a4a {tp1_pos*100:.0f}%, #1a6a4a {tp2_pos*100:.0f}%,
                                #00ff88 {tp2_pos*100:.0f}%, #00ff88 100%); 
                                border-radius: 4px;'>
                                <!-- Current position marker -->
                                <div style='position: absolute; left: {current_pos*100:.1f}%; top: -2px; transform: translateX(-50%);'>
                                    <div style='width: 4px; height: 28px; background: #00d4ff; border-radius: 2px;'></div>
                                </div>
                                <!-- Entry marker -->
                                <div style='position: absolute; left: {entry_pos*100:.1f}%; top: 0; transform: translateX(-50%);'>
                                    <div style='width: 2px; height: 24px; background: #888;'></div>
                                </div>
                                <!-- TP markers -->
                                <div style='position: absolute; left: {tp1_pos*100:.1f}%; top: 0; transform: translateX(-50%);'>
                                    <div style='width: 2px; height: 24px; background: #00d4aa;'></div>
                                </div>
                                <div style='position: absolute; left: {tp2_pos*100:.1f}%; top: 0; transform: translateX(-50%);'>
                                    <div style='width: 2px; height: 24px; background: #00d4aa;'></div>
                                </div>
                            </div>
                            <div style='text-align: center; margin-top: 5px; color: #00d4ff; font-size: 0.9em;'>
                                📍 Current: <b>{fmt_price(price)}</b> | 
                                {('✅ TP1 Hit!' if tp1_hit_now else '✅ TP1 Hit (retraced)') if tp1_hit else ('🎯 ' + f'{dist_to_tp1_pct:.1f}% to TP1' if dist_to_tp1_pct > 0 else '✅ At TP1')} | 
                                {'⚠️ ' + f'{dist_to_sl:.1f}% to SL' if dist_to_sl < 3 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # INLINE FULL ANALYSIS (when button clicked)
                    # ═══════════════════════════════════════════════════════════
                    if st.session_state.get(f"show_analysis_{trade['symbol']}", False):
                        with st.expander(f"📊 Full Analysis: {trade['symbol']}", expanded=True):
                            analysis_status = st.empty()
                            analysis_status.info(f"Loading analysis for {trade['symbol']}...")
                            
                            try:
                                # Fetch fresh data
                                df_analysis = fetch_binance_klines(trade['symbol'], trade.get('timeframe', '15m'), 200)
                                
                                if df_analysis is not None and len(df_analysis) >= 50:
                                    # Generate fresh signal and analysis
                                    fresh_signal = SignalGenerator.generate_signal(df_analysis, trade['symbol'], trade.get('timeframe', '15m'))
                                    mf_analysis = calculate_money_flow(df_analysis)
                                    smc_analysis = detect_smc(df_analysis)
                                    
                                    analysis_status.empty()
                                    
                                    # ═══════════════════════════════════════════════════════════
                                    # 🚨 CHECK DANGER ZONE FIRST - Using Risk Budget Consumed
                                    # ═══════════════════════════════════════════════════════════
                                    
                                    current_price_check = df_analysis['Close'].iloc[-1]
                                    entry_check = trade['entry']
                                    sl_check = trade['stop_loss']
                                    
                                    # Calculate distance to SL and risk budget consumed
                                    if trade['direction'] == 'LONG':
                                        distance_to_sl = ((current_price_check - sl_check) / current_price_check) * 100
                                        initial_risk = ((entry_check - sl_check) / entry_check) * 100
                                        pnl_check = ((current_price_check - entry_check) / entry_check) * 100
                                    else:
                                        distance_to_sl = ((sl_check - current_price_check) / current_price_check) * 100
                                        initial_risk = ((sl_check - entry_check) / entry_check) * 100
                                        pnl_check = ((entry_check - current_price_check) / entry_check) * 100
                                    
                                    # Risk budget consumed: 0% = at entry, 100% = at SL
                                    if initial_risk > 0:
                                        risk_budget_consumed = ((initial_risk - distance_to_sl) / initial_risk) * 100
                                    else:
                                        risk_budget_consumed = 0
                                    
                                    # DANGER ZONE only if:
                                    # 1. Very close to SL (< 0.8% absolute) - Always critical
                                    # 2. More than 60% of risk budget consumed
                                    is_critical = distance_to_sl < 0.8
                                    is_in_danger_zone = is_critical or risk_budget_consumed > 60
                                    
                                    # ═══════════════════════════════════════════════════════════
                                    # 📖 MASTER NARRATIVE - Educational Story
                                    # ═══════════════════════════════════════════════════════════
                                    
                                    # Import narrative engine
                                    from core.narrative_engine import MasterNarrative, Action
                                    
                                    # Get current mode from settings
                                    current_mode = st.session_state.settings.get('trading_mode', 'day_trade')
                                    timeframe_tf = trade.get('timeframe', '15m')
                                    
                                    # Generate narrative analysis
                                    narrator = MasterNarrative()
                                    existing_pos = {
                                        'entry': trade['entry'],
                                        'direction': trade['direction'],
                                        'size': 1.0  # Normalized
                                    }
                                    
                                    narrative = narrator.analyze(
                                        df_analysis, 
                                        trade['symbol'], 
                                        current_mode, 
                                        timeframe_tf,
                                        existing_position=existing_pos
                                    )
                                    
                                    # Calculate COMBINED confidence score (same as Scanner/Single Analysis)
                                    whale_monitor = detect_whale_activity(df_analysis)
                                    pre_break_monitor = detect_pre_breakout(df_analysis)
                                    inst_monitor = detect_institutional_activity(df_analysis, mf_analysis)
                                    
                                    # 🐋 FETCH REAL WHALE API DATA (for knowledge base scoring!)
                                    try:
                                        real_whale = get_whale_analysis(trade['symbol'])
                                        if real_whale and real_whale.get('top_trader_ls', {}).get('long_pct', 50) != 50:
                                            # Enrich whale_monitor with real API data
                                            whale_monitor['open_interest'] = real_whale.get('open_interest', {})
                                            whale_monitor['funding_rate'] = real_whale.get('funding', {}).get('rate', 0)
                                            whale_monitor['top_trader_ls'] = real_whale.get('top_trader_ls', {})
                                            whale_monitor['retail_ls'] = real_whale.get('retail_ls', {})
                                            
                                            # Calculate unified verdict for knowledge base
                                            from core.education import get_unified_verdict
                                            oi_change = real_whale.get('open_interest', {}).get('change_24h', 0)
                                            price_change = real_whale.get('price_change_24h', 0)
                                            top_long = real_whale.get('top_trader_ls', {}).get('long_pct', 50)
                                            retail_long = real_whale.get('retail_ls', {}).get('long_pct', 50)
                                            funding_raw = real_whale.get('funding', {}).get('rate', 0)
                                            tech_dir = 'BUY' if trade['direction'] == 'LONG' else 'SELL'
                                            
                                            unified = get_unified_verdict(
                                                oi_change, price_change, top_long, retail_long, funding_raw, tech_dir
                                            )
                                            whale_monitor['unified_verdict'] = unified
                                            whale_monitor['real_whale_data'] = real_whale
                                            
                                            # Apply verdict override (same as Single Analysis)
                                            verdict_conf = unified.get('confidence', 'MEDIUM')
                                            verdict_action = str(unified.get('unified_action', unified.get('action', 'WAIT'))).upper()
                                            is_wait_or_avoid = any(x in verdict_action for x in ['AVOID', 'WAIT', 'RANGE', 'CAUTION'])
                                            
                                            if verdict_conf == 'LOW' or is_wait_or_avoid:
                                                whale_monitor['direction'] = 'NEUTRAL'
                                                whale_monitor['confidence'] = 30
                                                whale_monitor['no_edge'] = True
                                                whale_monitor['verdict_override'] = True
                                    except:
                                        pass  # Use chart-based if API fails
                                    
                                    # Get OI and whale % for threshold validation
                                    oi_data_monitor = whale_monitor.get('open_interest', {})
                                    oi_change_monitor = oi_data_monitor.get('change_24h', 0) if isinstance(oi_data_monitor, dict) else 0
                                    top_trader_monitor = whale_monitor.get('top_trader_ls', {})
                                    whale_pct_monitor = top_trader_monitor.get('long_pct', 50) if isinstance(top_trader_monitor, dict) else 50
                                    
                                    # Get retail % for predictive scoring
                                    retail_data_monitor = whale_monitor.get('retail_ls', {})
                                    retail_pct_monitor = retail_data_monitor.get('long_pct', 50) if isinstance(retail_data_monitor, dict) else 50
                                    price_change_monitor = whale_monitor.get('real_whale_data', {}).get('price_change_24h', 0) if whale_monitor.get('real_whale_data') else 0
                                    
                                    monitor_conf_scores = calculate_confidence_scores(
                                        fresh_signal, mf_analysis, smc_analysis, whale_monitor, pre_break_monitor, inst_monitor,
                                        trade_mode=current_mode,
                                        timeframe=timeframe_tf,
                                        oi_change=oi_change_monitor,
                                        whale_pct=whale_pct_monitor
                                    )
                                    
                                    # Get support level from trade or fresh signal
                                    trade_direction = trade.get('direction', 'LONG')
                                    support_lvl = trade.get('stop_loss') if trade_direction == 'LONG' else None
                                    if not support_lvl and fresh_signal and fresh_signal.direction == 'LONG':
                                        support_lvl = fresh_signal.stop_loss
                                    
                                    # Get move position data from SMC
                                    smc_struct_monitor = smc_analysis.get('structure', {}) if smc_analysis else {}
                                    swing_high_monitor = smc_struct_monitor.get('last_swing_high', 0)
                                    swing_low_monitor = smc_struct_monitor.get('last_swing_low', 0)
                                    structure_type_monitor = smc_struct_monitor.get('structure', 'Unknown')
                                    current_price_monitor = df_analysis['Close'].iloc[-1] if df_analysis is not None and len(df_analysis) > 0 else 0
                                    
                                    # Calculate position in range for money flow context
                                    if swing_high_monitor and swing_low_monitor and swing_high_monitor > swing_low_monitor:
                                        pos_in_range_monitor = ((current_price_monitor - swing_low_monitor) / (swing_high_monitor - swing_low_monitor)) * 100
                                        pos_in_range_monitor = max(0, min(100, pos_in_range_monitor))
                                    else:
                                        pos_in_range_monitor = 50
                                    
                                    # 🎯 Calculate Money Flow Phase for predictive signal
                                    is_accum_monitor = mf_analysis.get('is_accumulating', False)
                                    is_distrib_monitor = mf_analysis.get('is_distributing', False)
                                    flow_ctx_monitor = get_money_flow_context(
                                        is_accumulating=is_accum_monitor,
                                        is_distributing=is_distrib_monitor,
                                        position_pct=pos_in_range_monitor,
                                        structure_type=structure_type_monitor
                                    )
                                    money_flow_phase_monitor = flow_ctx_monitor.phase
                                    
                                    # 🎯 NEW 3-LAYER PREDICTIVE SCORE FOR TRADE MONITOR
                                    monitor_predictive = calculate_predictive_score(
                                        oi_change=oi_change_monitor,
                                        price_change=price_change_monitor,
                                        whale_pct=whale_pct_monitor,
                                        retail_pct=retail_pct_monitor,
                                        ta_score=monitor_conf_scores.get('ta_score', 50),
                                        trade_mode=current_mode,
                                        timeframe=timeframe_tf,
                                        support_level=support_lvl,
                                        current_price=current_price_monitor,
                                        swing_high=swing_high_monitor,
                                        swing_low=swing_low_monitor,
                                        structure_type=structure_type_monitor,
                                        money_flow_phase=money_flow_phase_monitor  # 🚀 For predictive signal!
                                    )
                                    
                                    monitor_combined = monitor_predictive.final_score
                                    monitor_ta = monitor_conf_scores.get('ta_score', 50)
                                    monitor_inst = monitor_conf_scores.get('inst_score', 50)
                                    
                                    # ═══════════════════════════════════════════════════════════
                                    # 🚨 DANGER ZONE OVERRIDE - Don't suggest ADD when near SL!
                                    # ═══════════════════════════════════════════════════════════
                                    
                                    if is_in_danger_zone:
                                        # Override action if in danger zone
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, #4a0000 0%, #2a0000 100%); 
                                                    border-left: 4px solid #ff4444; 
                                                    padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                                <div>
                                                    <span style='font-size: 1.5em;'>🚨</span>
                                                    <span style='color: #ff4444; font-size: 1.3em; font-weight: bold; margin-left: 10px;'>
                                                        DANGER ZONE - EXIT or HOLD TIGHT
                                                    </span>
                                                </div>
                                                <div style='text-align: right;'>
                                                    <span style='color: #888; font-size: 0.9em;'>Risk Budget Used</span><br>
                                                    <span style='color: #ff4444; font-size: 1.2em; font-weight: bold;'>{risk_budget_consumed:.0f}%</span>
                                                </div>
                                            </div>
                                            <div style='color: #ffcccc; margin-top: 10px; font-size: 1.1em;'>
                                                ⚠️ You've used {risk_budget_consumed:.0f}% of your risk budget ({distance_to_sl:.2f}% to SL). Do NOT add to this position.
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Danger Zone specific advice
                                        st.markdown(f"""
                                        <div style='background: #1a0a0a; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                                            <div style='color: #ff6666; font-weight: bold; margin-bottom: 8px;'>🎯 What To Do NOW</div>
                                            <div style='color: #ddd; line-height: 1.6;'>
                                                <strong>Option 1:</strong> Exit now and take the small loss ({pnl_check:+.2f}%)<br>
                                                <strong>Option 2:</strong> Hold tight with finger on exit button - if SL level breaks, close IMMEDIATELY<br>
                                                <strong>DO NOT:</strong> Add more to this position - that's how big losses happen
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        # Normal display with PREDICTIVE SCORE
                                        pred_color = "#00ff88" if monitor_predictive.final_score >= 75 else "#00d4aa" if monitor_predictive.final_score >= 60 else "#ffcc00" if monitor_predictive.final_score >= 45 else "#ff9500"
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, #0d1b2a, #1b263b); border-left: 4px solid {pred_color}; 
                                                    padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                                <div>
                                                    <span style='font-size: 1.5em;'>🎯</span>
                                                    <span style='color: {pred_color}; font-size: 1.3em; font-weight: bold; margin-left: 10px;'>
                                                        {monitor_predictive.final_action}
                                                    </span>
                                                </div>
                                                <div style='text-align: right;'>
                                                    <span style='color: #888; font-size: 0.9em;'>Predictive Score</span><br>
                                                    <span style='color: {pred_color}; font-size: 1.2em; font-weight: bold;'>{monitor_predictive.final_score}/100</span>
                                                </div>
                                            </div>
                                            <div style='color: #ccc; margin-top: 10px; font-size: 0.95em;'>
                                                {monitor_predictive.final_summary}
                                            </div>
                                            <div style='color: #888; margin-top: 8px; font-size: 0.85em;'>
                                                Direction: {monitor_predictive.direction_score}/40 | Squeeze: {monitor_predictive.squeeze_score}/30 | Entry: {monitor_predictive.timing_score}/30
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Summary Box
                                        st.markdown(f"""
                                        <div style='background: #1a1a2e; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                                            <div style='color: #00d4ff; font-weight: bold; margin-bottom: 8px;'>📖 What's Happening</div>
                                            <div style='color: #ddd; line-height: 1.6;'>{narrative.summary}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # ═══════════════════════════════════════════════════════════
                                        # 🎯 MULTI-TIMEFRAME TIP (Trade Monitor)
                                        # ═══════════════════════════════════════════════════════════
                                        try:
                                            from core.money_flow_context import get_multi_timeframe_tip
                                            monitor_mtf_tip = get_multi_timeframe_tip(
                                                current_tf=timeframe_tf,
                                                current_phase=money_flow_phase_monitor,
                                                higher_tf_phase=None,  # Would need to fetch higher TF
                                                predictive_signal=monitor_predictive.final_action
                                            )
                                            if monitor_mtf_tip.has_tip:
                                                st.markdown(f"""
                                                <div style='background: {monitor_mtf_tip.tip_color}15; border-left: 4px solid {monitor_mtf_tip.tip_color}; 
                                                            padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
                                                    <div style='color: {monitor_mtf_tip.tip_color}; font-weight: bold; margin-bottom: 5px;'>
                                                        {monitor_mtf_tip.tip_emoji} {monitor_mtf_tip.tip_title}
                                                    </div>
                                                    <div style='color: #ccc; font-size: 0.9em;'>
                                                        {monitor_mtf_tip.tip_detail}
                                                    </div>
                                                    <div style='color: {monitor_mtf_tip.tip_color}; font-size: 0.85em; font-style: italic; margin-top: 8px;'>
                                                        👉 {monitor_mtf_tip.action}
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except Exception as e:
                                            pass  # Silently skip MTF tip if error
                                    
                                    # Insights (Educational Points) - SHOW FOR BOTH DANGER AND NON-DANGER
                                    if narrative.insights:
                                        st.markdown("### 🎓 Key Insights")
                                        
                                        bull_insights = [i for i in narrative.insights if i.sentiment.value == 'bullish']
                                        bear_insights = [i for i in narrative.insights if i.sentiment.value == 'bearish']
                                        neutral_insights = [i for i in narrative.insights if i.sentiment.value == 'neutral']
                                        
                                        insight_cols = st.columns(2)
                                        
                                        with insight_cols[0]:
                                            st.markdown("**🟢 Bullish Factors**")
                                            if bull_insights:
                                                for insight in bull_insights[:4]:
                                                    st.markdown(f"""
                                                    <div style='background: #0a2a1a; border-left: 3px solid #00d4aa; 
                                                                padding: 8px 12px; border-radius: 4px; margin: 5px 0;'>
                                                        <div style='color: #00d4aa; font-weight: bold;'>{insight.title}</div>
                                                        <div style='color: #aaa; font-size: 0.85em; margin-top: 4px;'>{insight.explanation}</div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.markdown("<span style='color: #666;'>No bullish signals detected</span>", unsafe_allow_html=True)
                                        
                                        with insight_cols[1]:
                                            st.markdown("**🔴 Bearish Factors**")
                                            if bear_insights:
                                                for insight in bear_insights[:4]:
                                                    st.markdown(f"""
                                                    <div style='background: #2a0a0a; border-left: 3px solid #ff4444; 
                                                                padding: 8px 12px; border-radius: 4px; margin: 5px 0;'>
                                                        <div style='color: #ff6666; font-weight: bold;'>{insight.title}</div>
                                                        <div style='color: #aaa; font-size: 0.85em; margin-top: 4px;'>{insight.explanation}</div>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                            else:
                                                st.markdown("<span style='color: #666;'>No bearish signals detected</span>", unsafe_allow_html=True)
                                    
                                    # Action Plan - DIFFERENT FOR DANGER VS NON-DANGER
                                    if is_in_danger_zone:
                                        # DANGER ZONE action plan
                                        st.markdown("### 📋 Action Plan")
                                        st.markdown(f"""
                                        <div style='background: #2a0a0a; border-radius: 8px; padding: 15px; border: 1px solid #ff4444;'>
                                            <div style='color: #ff6666; margin: 8px 0;'>🚨 <strong>PRIORITY:</strong> You are {distance_to_sl:.2f}% from Stop Loss - manage risk FIRST</div>
                                            <div style='color: #ffcccc; margin: 8px 0;'>❌ Do NOT add to this position</div>
                                            <div style='color: #ffcccc; margin: 8px 0;'>⚡ Consider exiting if conditions don't improve quickly</div>
                                            <div style='color: #ffcccc; margin: 8px 0;'>👀 Watch support at ${trade['stop_loss']:.4f}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        # Normal action plan
                                        if narrative.action_steps:
                                            st.markdown("### 📋 Action Plan")
                                            st.markdown(f"""
                                            <div style='background: #1a2a3a; border-radius: 8px; padding: 15px;'>
                                            """, unsafe_allow_html=True)
                                            for step in narrative.action_steps:
                                                st.markdown(f"<div style='color: #ccc; margin: 8px 0;'>{step}</div>", unsafe_allow_html=True)
                                            st.markdown("</div>", unsafe_allow_html=True)
                                    
                                    # Risk Assessment
                                    risk_color = "#ff4444" if narrative.risk_level == "High" else "#ffcc00" if narrative.risk_level == "Medium" else "#00d4aa"
                                    st.markdown(f"""
                                    <div style='background: #1a1a1a; border: 1px solid {risk_color}; border-radius: 8px; 
                                                padding: 12px; margin-top: 15px;'>
                                        <span style='color: {risk_color}; font-weight: bold;'>⚠️ Risk Level: {narrative.risk_level}</span>
                                        <span style='color: #888; margin-left: 15px;'>{narrative.risk_notes}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    
                                    # Show chart
                                    if fresh_signal:
                                        fig_analysis = create_trade_setup_chart(df_analysis, fresh_signal, trade.get('score', 0))
                                        st.plotly_chart(fig_analysis, use_container_width=True)
                                    
                                    # Quick stats
                                    a_col1, a_col2, a_col3 = st.columns(3)
                                    with a_col1:
                                        st.markdown("**📊 Money Flow**")
                                        st.write(f"MFI: {mf_analysis['mfi']:.0f}")
                                        cmf_val = mf_analysis['cmf']
                                        if pd.isna(cmf_val):
                                            cmf_val = 0
                                        st.write(f"CMF: {cmf_val:.3f}")
                                        
                                        # Calculate position in range for context
                                        if swing_high_monitor and swing_low_monitor and swing_high_monitor > swing_low_monitor:
                                            monitor_pos_pct = ((current_price_monitor - swing_low_monitor) / (swing_high_monitor - swing_low_monitor)) * 100
                                            monitor_pos_pct = max(0, min(100, monitor_pos_pct))
                                        else:
                                            monitor_pos_pct = 50
                                        
                                        # Get contextual flow
                                        monitor_flow_context = get_money_flow_context(
                                            is_accumulating=mf_analysis.get('is_accumulating', False),
                                            is_distributing=mf_analysis.get('is_distributing', False),
                                            position_pct=monitor_pos_pct,
                                            structure_type=structure_type_monitor
                                        )
                                        st.write(f"Status: {monitor_flow_context.phase_emoji} {monitor_flow_context.phase}")
                                        
                                        # Learn button - with clickable phases!
                                        with st.expander("📚 Learn", expanded=False):
                                            edu = get_money_flow_education(monitor_flow_context.phase)
                                            st.markdown(f"**{edu.get('emoji', '')} {edu.get('title', monitor_flow_context.phase)}**")
                                            st.markdown(f"*{edu.get('short', '')}*")
                                            st.markdown(edu.get('description', ''))
                                            
                                            if edu.get('wyckoff'):
                                                st.info(f"📖 Wyckoff: {edu.get('wyckoff')}")
                                            
                                            # Structure vs Money Flow explanation
                                            st.markdown("---")
                                            st.markdown("""
                                            **💡 Structure vs Money Flow:**
                                            - **Structure:** Price pattern (HH/HL vs LH/LL)
                                            - **Money Flow:** Volume direction (buying/selling)
                                            """)
                                            
                                            # Dropdown to see other phases
                                            st.markdown("**📚 Other Phases:**")
                                            phase_options = list(MONEY_FLOW_EDUCATION.keys())
                                            current_idx = phase_options.index(monitor_flow_context.phase) if monitor_flow_context.phase in phase_options else 0
                                            selected_phase = st.selectbox("Select to learn:", phase_options, index=current_idx, key=f"monitor_phase_{symbol}")
                                            
                                            if selected_phase != monitor_flow_context.phase:
                                                sel_edu = get_money_flow_education(selected_phase)
                                                st.markdown(f"**{sel_edu.get('emoji', '')} {sel_edu.get('title', selected_phase)}**")
                                                st.markdown(f"*{sel_edu.get('short', '')}*")
                                                st.markdown(sel_edu.get('description', ''))
                                    
                                    with a_col2:
                                        st.markdown("**🏦 SMC Levels**")
                                        if smc_analysis['order_blocks'].get('at_bullish_ob'):
                                            st.write("✅ At Bullish OB")
                                        if smc_analysis['fvg'].get('at_bullish_fvg'):
                                            st.write("✅ At Bullish FVG")
                                        st.write(f"Structure: {smc_analysis.get('structure', {}).get('bias', 'N/A')}")
                                    
                                    with a_col3:
                                        # Calculate VWAP
                                        try:
                                            vwap_data = calculate_vwap(
                                                df_analysis['High'], df_analysis['Low'], 
                                                df_analysis['Close'], df_analysis['Volume']
                                            )
                                            st.markdown("**⚖️ VWAP**")
                                            st.write(f"VWAP: {fmt_price(vwap_data['vwap'])}")
                                            st.write(f"Position: {vwap_data['position']}")
                                            st.write(f"Distance: {vwap_data['distance_pct']:+.1f}%")
                                        except:
                                            st.markdown("**📈 Current**")
                                            current_price_analysis = df_analysis['Close'].iloc[-1]
                                            st.write(f"Price: {fmt_price(current_price_analysis)}")
                                    
                                    # ═══════════════════════════════════════════════════════════
                                    # 📚 EDUCATION SECTION - "How We Got Here" + Definitions
                                    # ═══════════════════════════════════════════════════════════
                                    
                                    st.markdown("---")
                                    
                                    # Generate Price Story
                                    try:
                                        vwap_data = calculate_vwap(
                                            df_analysis['High'], df_analysis['Low'], 
                                            df_analysis['Close'], df_analysis['Volume']
                                        )
                                    except:
                                        vwap_data = None
                                    
                                    price_story = generate_price_story(
                                        df_analysis, 
                                        trade.get('timeframe', '15m'),
                                        smc_analysis,
                                        mf_analysis,
                                        vwap_data
                                    )
                                    
                                    # Display Story
                                    with st.expander("📖 **How We Got Here** - Price Action Story", expanded=False):
                                        st.markdown(f"**Timeframe:** {trade.get('timeframe', '15m')} | **Current Price:** {fmt_price(price_story.current_price)}")
                                        st.markdown("---")
                                        
                                        for part in price_story.story_parts:
                                            st.markdown(part)
                                        
                                        # ═══════════════════════════════════════════════════════════
                                        # 🐋 WHALE DATA IN PRICE STORY (NEW!)
                                        # ═══════════════════════════════════════════════════════════
                                        
                                        try:
                                            whale_story_data = get_whale_analysis(trade['symbol'])
                                            has_whale = (
                                                whale_story_data.get('signals') or 
                                                whale_story_data.get('open_interest', {}).get('change_24h', 0) != 0
                                            )
                                            
                                            if has_whale:
                                                oi_change = whale_story_data.get('open_interest', {}).get('change_24h', 0)
                                                price_chg = whale_story_data.get('price_change_24h', 0)
                                                funding = whale_story_data.get('funding', {}).get('rate_pct', 0)
                                                whale_long = whale_story_data.get('top_trader_ls', {}).get('long_pct', 50)
                                                retail_long = whale_story_data.get('retail_ls', {}).get('long_pct', 50)
                                                oi_interp = whale_story_data.get('oi_interpretation', {})
                                                
                                                # Use NEW unified scenario system (not broken old verdict!)
                                                from core.institutional_scoring import analyze_institutional_data, get_scenario_display
                                                inst_score = analyze_institutional_data(
                                                    oi_change, price_chg, whale_long, retail_long, funding, 'NEUTRAL'
                                                )
                                                scenario_display = get_scenario_display(inst_score.scenario)
                                                
                                                st.markdown("---")
                                                st.markdown("### 🐋 Institutional Activity")
                                                
                                                # OI + Price interpretation
                                                if oi_interp.get('interpretation'):
                                                    emoji = oi_interp.get('emoji', '📊')
                                                    signal = oi_interp.get('signal', '')
                                                    st.markdown(f"**{emoji} {signal}:** {oi_interp.get('interpretation', '')}")
                                                
                                                # Funding interpretation
                                                if abs(funding) > 0.05:
                                                    if funding > 0:
                                                        st.markdown(f"💰 **Funding Alert:** Longs paying {funding:.3f}% - overleveraged, dump risk")
                                                    else:
                                                        st.markdown(f"💰 **Funding Alert:** Shorts paying {abs(funding):.3f}% - overleveraged, pump potential")
                                                
                                                # Whale vs Retail divergence
                                                if abs(whale_long - retail_long) > 15:
                                                    if whale_long > retail_long:
                                                        st.markdown(f"🐋 **Whale/Retail Divergence:** Whales are MORE bullish ({whale_long:.0f}% L) than retail ({retail_long:.0f}% L) - FOLLOW WHALES")
                                                    else:
                                                        st.markdown(f"🐋 **Whale/Retail Divergence:** Whales are LESS bullish ({whale_long:.0f}% L) than retail ({retail_long:.0f}% L) - FADE RETAIL")
                                                
                                                # UNIFIED Scenario-based verdict (not broken old system!)
                                                if inst_score.direction_bias == "LONG":
                                                    v_color = '#00d4aa'
                                                elif inst_score.direction_bias == "SHORT":
                                                    v_color = '#ff4444'
                                                else:
                                                    v_color = '#888'
                                                
                                                st.markdown(f"""
                                                <div style='background: {v_color}22; border-left: 3px solid {v_color}; 
                                                            padding: 10px; border-radius: 6px; margin-top: 10px;'>
                                                    <strong style='color: {v_color};'>{scenario_display['emoji']} {inst_score.scenario_name}</strong>
                                                    <span style='color: #888; margin-left: 10px;'>({inst_score.confidence} confidence)</span>
                                                </div>
                                                """, unsafe_allow_html=True)
                                        except:
                                            pass  # Skip whale data if unavailable
                                        
                                        st.markdown("---")
                                        st.markdown(price_story.overall_narrative)
                                    
                                    # Education - Concept Definitions
                                    with st.expander("📚 **Learn: Concepts Used in This Analysis**", expanded=False):
                                        
                                        # Build education content
                                        education = build_full_education_section(
                                            smc_analysis, mf_analysis, vwap_data, show_definitions=True
                                        )
                                        
                                        # Tabs for different concept categories
                                        edu_tabs = st.tabs(["💰 Money Flow", "🏦 SMC", "⚖️ VWAP", "📈 Trend", "🐋 Whale Data"])
                                        
                                        with edu_tabs[0]:  # Money Flow
                                            st.markdown("### Money Flow Indicators")
                                            st.markdown("*These show WHERE the money is going - into or out of the asset*")
                                            
                                            # MFI
                                            mfi_def = CONCEPT_DEFINITIONS.get('mfi', {})
                                            st.markdown(f"#### {mfi_def.get('emoji', '')} {mfi_def.get('name', 'MFI')}")
                                            st.info(mfi_def.get('short', ''))
                                            
                                            mfi_val = mf_analysis.get('mfi', 50)
                                            mfi_explanation = get_indicator_explanation('mfi', mfi_val)
                                            st.markdown(f"**Current Value:** {mfi_val:.0f}")
                                            st.markdown(mfi_explanation)
                                            
                                            with st.expander("📖 Full MFI Definition"):
                                                st.markdown(mfi_def.get('definition', ''))
                                            
                                            st.markdown("---")
                                            
                                            # CMF
                                            cmf_def = CONCEPT_DEFINITIONS.get('cmf', {})
                                            st.markdown(f"#### {cmf_def.get('emoji', '')} {cmf_def.get('name', 'CMF')}")
                                            st.info(cmf_def.get('short', ''))
                                            
                                            cmf_val = mf_analysis.get('cmf', 0)
                                            if pd.isna(cmf_val):
                                                cmf_val = 0
                                            cmf_explanation = get_indicator_explanation('cmf', cmf_val)
                                            st.markdown(f"**Current Value:** {cmf_val:.3f}")
                                            st.markdown(cmf_explanation)
                                            
                                            with st.expander("📖 Full CMF Definition"):
                                                st.markdown(cmf_def.get('definition', ''))
                                            
                                            st.markdown("---")
                                            
                                            # OBV - Now with contextual explanation!
                                            obv_def = CONCEPT_DEFINITIONS.get('obv', {})
                                            st.markdown(f"#### {obv_def.get('emoji', '')} {obv_def.get('name', 'OBV')}")
                                            st.info(obv_def.get('short', ''))
                                            
                                            obv_rising = mf_analysis.get('obv_rising', False)
                                            obv_status = "Rising 📈" if obv_rising else "Falling 📉"
                                            st.markdown(f"**Current Status:** {obv_status}")
                                            
                                            # Get price direction for divergence detection
                                            price_rising = None
                                            if len(df_analysis) >= 20:
                                                price_change = (df_analysis['Close'].iloc[-1] - df_analysis['Close'].iloc[-20]) / df_analysis['Close'].iloc[-20] * 100
                                                price_rising = price_change > 0.5  # More than 0.5% up = rising
                                            
                                            # Get CMF for context
                                            cmf_positive = mf_analysis.get('cmf', 0) > 0
                                            
                                            # Import and use OBV explanation function
                                            from core.education import get_obv_explanation
                                            obv_explanation = get_obv_explanation(obv_rising, price_rising, cmf_positive)
                                            st.markdown(obv_explanation)
                                            
                                            with st.expander("📖 Full OBV Definition"):
                                                st.markdown(obv_def.get('definition', ''))
                                        
                                        with edu_tabs[1]:  # SMC
                                            st.markdown("### Smart Money Concepts (SMC)")
                                            st.markdown("*How institutions trade - following the 'smart money'*")
                                            
                                            # Order Blocks
                                            ob_def = CONCEPT_DEFINITIONS.get('order_block', {})
                                            st.markdown(f"#### {ob_def.get('emoji', '')} {ob_def.get('name', 'Order Block')}")
                                            st.info(ob_def.get('short', ''))
                                            
                                            ob_data = smc_analysis.get('order_blocks', {})
                                            if ob_data.get('at_bullish_ob'):
                                                st.success("✅ **Currently AT a Bullish Order Block** - High probability long entry zone")
                                            elif ob_data.get('at_bearish_ob'):
                                                st.warning("⚠️ **Currently AT a Bearish Order Block** - Expect resistance")
                                            else:
                                                st.markdown("*Not currently at an Order Block*")
                                            
                                            with st.expander("📖 Full Order Block Definition"):
                                                st.markdown(ob_def.get('definition', ''))
                                            
                                            st.markdown("---")
                                            
                                            # FVG
                                            fvg_def = CONCEPT_DEFINITIONS.get('fvg', {})
                                            st.markdown(f"#### {fvg_def.get('emoji', '')} {fvg_def.get('name', 'FVG')}")
                                            st.info(fvg_def.get('short', ''))
                                            
                                            fvg_data = smc_analysis.get('fvg', {})
                                            if fvg_data.get('at_bullish_fvg'):
                                                st.success("✅ **Currently AT a Bullish FVG** - Gap being filled, expect bounce")
                                            elif fvg_data.get('at_bearish_fvg'):
                                                st.warning("⚠️ **Currently AT a Bearish FVG** - Gap being filled, expect rejection")
                                            else:
                                                st.markdown("*Not currently at a Fair Value Gap*")
                                            
                                            with st.expander("📖 Full FVG Definition"):
                                                st.markdown(fvg_def.get('definition', ''))
                                            
                                            st.markdown("---")
                                            
                                            # Liquidity
                                            liq_def = CONCEPT_DEFINITIONS.get('liquidity', {})
                                            st.markdown(f"#### {liq_def.get('emoji', '')} {liq_def.get('name', 'Liquidity')}")
                                            st.info(liq_def.get('short', ''))
                                            
                                            with st.expander("📖 Full Liquidity Definition"):
                                                st.markdown(liq_def.get('definition', ''))
                                        
                                        with edu_tabs[2]:  # VWAP
                                            st.markdown("### VWAP - The Institutional Benchmark")
                                            st.markdown("*This is how big players measure their execution quality*")
                                            
                                            vwap_def = CONCEPT_DEFINITIONS.get('vwap', {})
                                            st.markdown(f"#### {vwap_def.get('emoji', '')} {vwap_def.get('name', 'VWAP')}")
                                            st.info(vwap_def.get('short', ''))
                                            
                                            if vwap_data:
                                                st.markdown(f"**Current VWAP:** {fmt_price(vwap_data.get('vwap', 0))}")
                                                st.markdown(f"**Price Position:** {vwap_data.get('position_text', 'N/A')}")
                                                st.markdown(f"**Distance from VWAP:** {vwap_data.get('distance_pct', 0):+.1f}%")
                                                
                                                # VWAP bands (with safety checks)
                                                if 'upper_2' in vwap_data:
                                                    st.markdown("**VWAP Bands:**")
                                                    st.markdown(f"- Upper +2σ: {fmt_price(vwap_data.get('upper_2', 0))}")
                                                    st.markdown(f"- Upper +1σ: {fmt_price(vwap_data.get('upper_1', 0))}")
                                                    st.markdown(f"- **VWAP: {fmt_price(vwap_data.get('vwap', 0))}**")
                                                    st.markdown(f"- Lower -1σ: {fmt_price(vwap_data.get('lower_1', 0))}")
                                                    st.markdown(f"- Lower -2σ: {fmt_price(vwap_data.get('lower_2', 0))}")
                                                else:
                                                    st.info("VWAP bands calculation not available")
                                            else:
                                                st.warning("VWAP data not available")
                                            
                                            with st.expander("📖 Full VWAP Definition"):
                                                st.markdown(vwap_def.get('definition', ''))
                                        
                                        with edu_tabs[3]:  # Trend
                                            st.markdown("### Trend Analysis")
                                            st.markdown("*Understanding the overall market direction*")
                                            
                                            # Market Structure
                                            ms_def = CONCEPT_DEFINITIONS.get('market_structure', {})
                                            st.markdown(f"#### {ms_def.get('emoji', '')} {ms_def.get('name', 'Market Structure')}")
                                            st.info(ms_def.get('short', ''))
                                            
                                            structure = smc_analysis.get('structure', {})
                                            bias = structure.get('bias', 'N/A')
                                            if bias == 'Long':
                                                st.success("✅ **Bullish Structure** - Higher highs and higher lows")
                                            elif bias == 'Short':
                                                st.error("🔴 **Bearish Structure** - Lower highs and lower lows")
                                            else:
                                                st.info("⚪ **Neutral Structure** - No clear trend")
                                            
                                            with st.expander("📖 Full Market Structure Definition"):
                                                st.markdown(ms_def.get('definition', ''))
                                            
                                            st.markdown("---")
                                            
                                            # EMA
                                            ema_def = CONCEPT_DEFINITIONS.get('ema', {})
                                            st.markdown(f"#### {ema_def.get('emoji', '')} {ema_def.get('name', 'EMA')}")
                                            st.info(ema_def.get('short', ''))
                                            
                                            # Calculate current EMAs
                                            try:
                                                ema_9 = calculate_ema(df_analysis['Close'], 9).iloc[-1]
                                                ema_20 = calculate_ema(df_analysis['Close'], 20).iloc[-1]
                                                ema_50 = calculate_ema(df_analysis['Close'], 50).iloc[-1]
                                                current_p = df_analysis['Close'].iloc[-1]
                                                
                                                st.markdown(f"**Current Price:** {fmt_price(current_p)}")
                                                st.markdown(f"**EMA 9:** {fmt_price(ema_9)}")
                                                st.markdown(f"**EMA 20:** {fmt_price(ema_20)}")
                                                st.markdown(f"**EMA 50:** {fmt_price(ema_50)}")
                                                
                                                if current_p > ema_9 > ema_20 > ema_50:
                                                    st.success("✅ **Perfect Bullish Stack** - Price > EMA9 > EMA20 > EMA50")
                                                elif current_p < ema_9 < ema_20 < ema_50:
                                                    st.error("🔴 **Perfect Bearish Stack** - Price < EMA9 < EMA20 < EMA50")
                                                else:
                                                    st.info("⚪ **Mixed EMA alignment** - No perfect stack")
                                            except:
                                                st.warning("EMA calculation not available")
                                            
                                            with st.expander("📖 Full EMA Definition"):
                                                st.markdown(ema_def.get('definition', ''))
                                        
                                        # ═══════════════════════════════════════════════════════════
                                        # 🐋 WHALE/INSTITUTIONAL TAB (NEW!)
                                        # ═══════════════════════════════════════════════════════════
                                        
                                        with edu_tabs[4]:  # Whale Data
                                            st.markdown("### 🐋 Whale & Institutional Data")
                                            st.markdown("*What the big players (whales, funds, market makers) are doing*")
                                            
                                            # Fetch whale data for this symbol
                                            try:
                                                whale_edu_data = get_whale_analysis(trade['symbol'])
                                                has_whale_data = (
                                                    whale_edu_data.get('signals') or 
                                                    whale_edu_data.get('open_interest', {}).get('change_24h', 0) != 0
                                                )
                                                
                                                if has_whale_data:
                                                    # OI + Price Explanation
                                                    st.markdown("#### 📊 Open Interest + Price")
                                                    st.info("OI shows how many futures contracts are open. Combined with price, it reveals WHO is entering/exiting.")
                                                    
                                                    oi_change = whale_edu_data.get('open_interest', {}).get('change_24h', 0)
                                                    price_chg = whale_edu_data.get('price_change_24h', 0)
                                                    
                                                    st.markdown(f"**Current OI Change:** {oi_change:+.1f}%")
                                                    st.markdown(f"**Current Price Change:** {price_chg:+.1f}%")
                                                    
                                                    # Interpretation table
                                                    st.markdown("""
                                                    **How to Read OI + Price:**
                                                    
                                                    | OI | Price | Meaning | Strength |
                                                    |:--:|:-----:|---------|----------|
                                                    | ↑ | ↑ | **New LONGS** entering | Strong bullish |
                                                    | ↑ | ↓ | **New SHORTS** entering | Strong bearish |
                                                    | ↓ | ↑ | Short covering (weak rally) | Weak - may reverse |
                                                    | ↓ | ↓ | Long liquidation (weak dump) | Weak - may reverse |
                                                    """)
                                                    
                                                    oi_interp = whale_edu_data.get('oi_interpretation', {})
                                                    if oi_interp.get('interpretation'):
                                                        interp_color = "#00d4aa" if oi_interp.get('signal') in ['NEW_LONGS', 'SHORT_COVERING'] else "#ff4444"
                                                        st.markdown(f"""
                                                        <div style='background: {interp_color}22; border-left: 3px solid {interp_color}; 
                                                                    padding: 10px; border-radius: 6px; margin: 10px 0;'>
                                                            <strong style='color: {interp_color};'>{oi_interp.get('emoji', '')} Your Current Situation:</strong><br>
                                                            {oi_interp.get('interpretation', '')}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    
                                                    with st.expander("📖 Full OI + Price Definition"):
                                                        st.markdown("""
                                                        **Open Interest (OI)** represents the total number of outstanding futures contracts.
                                                        
                                                        - When OI RISES, new money is entering the market (new positions being opened)
                                                        - When OI FALLS, money is leaving (positions being closed)
                                                        
                                                        **Combined with Price:**
                                                        - OI↑ + Price↑: Fresh longs opening → Bullish continuation
                                                        - OI↑ + Price↓: Fresh shorts opening → Bearish continuation  
                                                        - OI↓ + Price↑: Shorts forced to cover → Rally may be weak
                                                        - OI↓ + Price↓: Longs forced to close → Dump may be weak
                                                        
                                                        **Why this matters:** Understanding if moves are driven by NEW conviction (OI↑) 
                                                        or FORCED liquidations (OI↓) helps predict if the move will continue or reverse.
                                                        """)
                                                    
                                                    st.markdown("---")
                                                    
                                                    # Funding Rate
                                                    st.markdown("#### 💰 Funding Rate (Contrarian!)")
                                                    st.info("Funding is what longs pay shorts (or vice versa) to keep futures price aligned with spot. Extreme values signal overleveraging.")
                                                    
                                                    funding = whale_edu_data.get('funding', {}).get('rate_pct', 0)
                                                    st.markdown(f"**Current Funding:** {funding:.4f}%")
                                                    
                                                    if funding > 0.05:
                                                        st.warning("⚠️ **Longs are overleveraged** - Contrarian SHORT signal (dump likely)")
                                                    elif funding < -0.05:
                                                        st.success("✅ **Shorts are overleveraged** - Contrarian LONG signal (pump likely)")
                                                    else:
                                                        st.info("⚪ **Funding is neutral** - No extreme positioning")
                                                    
                                                    with st.expander("📖 Full Funding Rate Definition"):
                                                        st.markdown("""
                                                        **Funding Rate** is a periodic payment between long and short position holders.
                                                        
                                                        - **Positive funding (>0):** Longs pay shorts → Market is bullish/overleveraged
                                                        - **Negative funding (<0):** Shorts pay longs → Market is bearish/overleveraged
                                                        
                                                        **CONTRARIAN SIGNAL:**
                                                        - High positive (>0.1%): Too many longs → Expect dump
                                                        - High negative (<-0.1%): Too many shorts → Expect pump
                                                        
                                                        **Why this works:** Extreme funding creates incentive for market makers to 
                                                        move price against the overleveraged side to collect liquidations.
                                                        """)
                                                    
                                                    st.markdown("---")
                                                    
                                                    # Whale vs Retail
                                                    st.markdown("#### 🐋 Whales vs 🐑 Retail")
                                                    st.info("Top Traders (whales) often have better information than retail. When they diverge, follow the whales.")
                                                    
                                                    whale_long = whale_edu_data.get('top_trader_ls', {}).get('long_pct', 50)
                                                    retail_long = whale_edu_data.get('retail_ls', {}).get('long_pct', 50)
                                                    
                                                    st.markdown(f"**🐋 Top Traders:** {whale_long:.0f}% Long / {100-whale_long:.0f}% Short")
                                                    st.markdown(f"**🐑 Retail:** {retail_long:.0f}% Long / {100-retail_long:.0f}% Short")
                                                    
                                                    # Check for divergence
                                                    if retail_long > 65 and whale_long < 50:
                                                        st.error("🚨 **FADE RETAIL** - Retail heavily long but whales aren't. Consider SHORT.")
                                                    elif retail_long < 35 and whale_long > 50:
                                                        st.success("✅ **FADE RETAIL** - Retail heavily short but whales are long. Consider LONG.")
                                                    elif whale_long > 60:
                                                        st.success("✅ **FOLLOW WHALES** - Top traders are heavily long. Bullish bias.")
                                                    elif whale_long < 40:
                                                        st.warning("⚠️ **FOLLOW WHALES** - Top traders are heavily short. Bearish bias.")
                                                    else:
                                                        st.info("⚪ **No clear divergence** - Whales and retail roughly aligned.")
                                                    
                                                    with st.expander("📖 Why Retail is Usually Wrong"):
                                                        st.markdown("""
                                                        **Top Traders vs Retail:**
                                                        
                                                        Binance tracks two groups:
                                                        1. **Top Traders:** Accounts with highest PnL and volume (likely professionals)
                                                        2. **Retail:** All other accounts (regular traders)
                                                        
                                                        **Why fade retail at extremes:**
                                                        - Retail tends to chase moves (buy high, sell low)
                                                        - Market makers target retail liquidity
                                                        - When 70%+ of retail is on one side, they become the fuel for the opposite move
                                                        
                                                        **When retail is 70% long:** Their stops are below → Market likely to hunt those stops
                                                        **When retail is 70% short:** Their stops are above → Market likely to squeeze up
                                                        """)
                                                else:
                                                    st.info("🐋 Whale data unavailable for this symbol. Works best for major USDT pairs on Binance Futures.")
                                            except Exception as e:
                                                st.info("🐋 Whale data requires Binance Futures API access")
                                    
                                    st.markdown("---")
                                    
                                    # Info about closing
                                    st.caption("💡 Click 'Hide Analysis' button above to close this panel")
                                else:
                                    analysis_status.error("Could not fetch data for analysis")
                            except Exception as e:
                                analysis_status.error(f"Analysis error: {str(e)[:50]}")

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

elif app_mode == "📊 Performance":
    st.markdown("## 📊 Professional Performance Dashboard")
    st.markdown("<p style='color: #888;'>Track your edge with institutional-grade metrics</p>", unsafe_allow_html=True)
    
    # Use session state for closed trades (survives refresh within session)
    closed = st.session_state.get('closed_trades', [])
    
    # Also check file in case session state is empty (e.g., new browser tab)
    if not closed:
        closed = get_closed_trades()
        st.session_state.closed_trades = closed
    
    stats = calculate_statistics(closed)
    
    # ═══════════════════════════════════════════════════════════════════
    # PRIMARY METRICS - What matters most
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("### 🎯 Key Performance Indicators")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # TP1 Hit Rate - This is the MOST important metric for trade SELECTION
    tp1_color = "#00ff88" if stats.get('tp1_hit_rate', 0) >= 50 else "#ffcc00" if stats.get('tp1_hit_rate', 0) >= 35 else "#ff4444"
    c1.metric(
        "🎯 TP1 Hit Rate", 
        f"{stats.get('tp1_hit_rate', 0):.0f}%",
        help="Measures trade SELECTION quality. Did you identify good entries?"
    )
    
    # Avg R-Multiple - Risk-adjusted performance
    r_mult = stats.get('avg_r_multiple', 0)
    r_color = "#00ff88" if r_mult >= 1.5 else "#ffcc00" if r_mult >= 0.5 else "#ff4444"
    c2.metric(
        "📊 Avg R-Multiple", 
        f"{r_mult:.2f}R",
        help="Average return per unit of risk. Above 1R = profitable system"
    )
    
    # Profit Factor
    pf = stats.get('profit_factor', 0)
    c3.metric(
        "⚖️ Profit Factor", 
        f"{pf:.2f}" if pf != float('inf') else "∞",
        help="Gross Profit / Gross Loss. Above 1.5 = good system"
    )
    
    # Expectancy
    exp = stats.get('expectancy', 0)
    c4.metric(
        "💰 Expectancy", 
        f"{exp:+.2f}%",
        help="Expected profit per trade. Positive = edge exists"
    )
    
    # ═══════════════════════════════════════════════════════════════════
    # OUTCOME BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("### 📈 Trade Outcomes")
    
    c5, c6, c7, c8, c9 = st.columns(5)
    c5.metric("Total Trades", stats['total_trades'])
    c6.metric("🏆 Full Wins", stats['wins'], help="TP3 hit - all targets reached")
    c7.metric("✅ Partials", stats.get('partials', 0), help="TP1/TP2 hit, then stopped - STILL PROFITABLE")
    c8.metric("🔴 Losses", stats['losses'], help="Stopped out before TP1")
    c9.metric("💰 Total P&L", f"{stats['total_pnl']:+.1f}%")
    
    # Success rate explanation
    total = stats['total_trades']
    if total > 0:
        success_rate = ((stats['wins'] + stats.get('partials', 0)) / total) * 100
        st.markdown(f"""
        <div style='background: #1a2a3a; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <div style='color: #00d4ff; font-weight: bold; font-size: 1.1em;'>
                📊 Success Rate: {success_rate:.0f}% 
                <span style='color: #888; font-weight: normal;'>
                    ({stats['wins']} full wins + {stats.get('partials', 0)} partial wins out of {total} trades)
                </span>
            </div>
            <div style='color: #aaa; margin-top: 8px; font-size: 0.9em;'>
                <strong>Key Insight:</strong> Any trade that hits TP1 is a SUCCESSFUL trade identification. 
                What happens after (TP2, TP3, or stopped at breakeven) is about trade MANAGEMENT, not selection quality.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════
    # DETAILED METRICS
    # ═══════════════════════════════════════════════════════════════════
    
    st.markdown("### 📉 Risk Analysis")
    
    c10, c11, c12, c13 = st.columns(4)
    c10.metric("Avg Win", f"+{stats['avg_win']:.1f}%" if stats['avg_win'] > 0 else "0%")
    c11.metric("Avg Loss", f"-{stats['avg_loss']:.1f}%" if stats['avg_loss'] > 0 else "0%")
    c12.metric("Best Trade", f"+{stats['best_trade']:.1f}%" if stats['best_trade'] > 0 else f"{stats['best_trade']:.1f}%")
    c13.metric("Worst Trade", f"{stats['worst_trade']:+.1f}%")  # +/- sign automatically
    
    # closed is already loaded from session state above
    if closed:
        # Equity curve
        fig = create_performance_chart(closed)
        st.plotly_chart(fig, use_container_width=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # TRADE HISTORY WITH OUTCOME TYPES
        # ═══════════════════════════════════════════════════════════════════
        
        st.markdown("### 📋 Trade History")
        
        for trade in reversed(closed):  # Most recent first
            pnl = trade.get('final_pnl', trade.get('blended_pnl', trade.get('pnl_pct', 0)))
            outcome = trade.get('outcome_type', '')
            r_mult = trade.get('r_multiple', 0)
            
            # Determine display based on outcome type
            if outcome == "FULL_WIN":
                emoji = "🏆"
                color = "#00ff88"
                outcome_label = "FULL WIN"
            elif outcome in ["PARTIAL_WIN_TP2", "PARTIAL_WIN_TP2_RUNNING"]:
                emoji = "✅"
                color = "#00d4aa"
                outcome_label = "PARTIAL (TP2)"
            elif outcome in ["PARTIAL_WIN_TP1", "PARTIAL_WIN_TP1_RUNNING", "BREAKEVEN"]:
                emoji = "🟡"
                color = "#ffcc00"
                outcome_label = "BREAKEVEN"
            elif outcome == "FULL_LOSS":
                emoji = "🔴"
                color = "#ff4444"
                outcome_label = "LOSS"
            else:
                # Fallback for older trades
                emoji = "🏆" if trade.get('close_reason') == 'TP3_HIT' else "✅" if pnl > 0 else "🔴"
                color = "#00ff88" if trade.get('close_reason') == 'TP3_HIT' else "#00d4aa" if pnl > 0 else "#ff4444"
                outcome_label = trade.get('close_reason', 'CLOSED').replace('_', ' ')
            
            closed_at = trade.get('closed_at', trade.get('exit_time', 'N/A'))
            if isinstance(closed_at, str) and len(closed_at) > 10:
                closed_at = closed_at[:10]  # Just the date
            
            # TP progression indicators
            tp1 = "✅" if trade.get('tp1_hit') else "❌"
            tp2 = "✅" if trade.get('tp2_hit') else "❌"
            tp3 = "✅" if trade.get('tp3_hit') else "❌"
            
            st.markdown(f"""
            <div style='background: {color}15; border-left: 3px solid {color}; padding: 10px 15px; 
                        margin: 5px 0; border-radius: 0 8px 8px 0;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <span style='font-size: 1.2em;'>{emoji}</span>
                        <strong style='margin-left: 8px;'>{trade.get('symbol', 'N/A')}</strong>
                        <span style='color: #888; margin-left: 10px;'>{trade.get('direction', 'N/A')} | {trade.get('timeframe', 'N/A')}</span>
                        <span style='background: {color}33; color: {color}; padding: 2px 8px; border-radius: 4px; margin-left: 10px; font-size: 0.8em;'>
                            {outcome_label}
                        </span>
                    </div>
                    <div style='text-align: right;'>
                        <span style='color: {color}; font-size: 1.3em; font-weight: bold;'>{pnl:+.1f}%</span>
                        <span style='color: #888; margin-left: 8px;'>({r_mult:.1f}R)</span>
                        <div style='color: #888; font-size: 0.8em;'>{closed_at}</div>
                    </div>
                </div>
                <div style='color: #aaa; font-size: 0.85em; margin-top: 5px; display: flex; justify-content: space-between;'>
                    <span>Entry: ${trade.get('entry', 0):.4f}</span>
                    <span>Targets: TP1{tp1} TP2{tp2} TP3{tp3}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # EDUCATIONAL SECTION
        # ═══════════════════════════════════════════════════════════════════
        
        with st.expander("📚 Understanding Professional Performance Metrics", expanded=False):
            st.markdown("""
            ### 🎯 The 33/33/33 Partial Profit Strategy
            
            Professional traders don't go "all in, all out". They scale out:
            
            | Stage | Action | Position | Stop Loss |
            |-------|--------|----------|-----------|
            | **Entry** | Open trade | 100% | Original SL |
            | **TP1 Hit** | Take 33% profit | 67% remaining | Move to Breakeven |
            | **TP2 Hit** | Take 33% profit | 33% remaining | Move to TP1 |
            | **TP3 Hit** | Close all | 0% | - |
            
            ### 📊 What Each Metric Means
            
            **🎯 TP1 Hit Rate** - The MOST important metric!
            - Measures: Are you identifying good entries?
            - Target: 50%+ means you have an edge in trade SELECTION
            - If low: Work on entry criteria, not trade management
            
            **📊 R-Multiple** - Risk-adjusted returns
            - Measures: How much you make per unit of risk
            - Example: Risk 1% to make 2% = 2R trade
            - Target: Average above 1R means profitable system
            
            **⚖️ Profit Factor** - System quality
            - Formula: Total Profits ÷ Total Losses
            - Target: Above 1.5 = good system, Above 2.0 = excellent
            
            **💰 Expectancy** - Edge per trade
            - Formula: (Win% × Avg Win) - (Loss% × Avg Loss)
            - Target: Positive = you have an edge
            
            ### 🏆 Trade Outcome Categories
            
            | Outcome | What Happened | Your Fault? |
            |---------|---------------|-------------|
            | **FULL WIN** | All targets hit | Great trade! |
            | **PARTIAL TP2** | TP2 hit, stopped at TP1 | No - good management |
            | **BREAKEVEN** | TP1 hit, stopped at BE | No - you secured profit |
            | **LOSS** | Stopped before TP1 | Maybe - review entry |
            
            **Key Insight:** If you hit TP1, your trade IDENTIFICATION was correct. 
            What happens after is about the MARKET, not your analysis.
            """)
    else:
        st.info("""
        📊 **No completed trades yet!**
        
        Trades are automatically closed when:
        - TP3 is hit (Full Win)
        - Stop Loss is hit (Loss or Breakeven if TP1 was hit first)
        
        Your P&L is calculated using the **33/33/33 partial profit strategy** for realistic results.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 SINGLE ANALYSIS MODE - WITH MASTER NARRATIVE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

elif app_mode == "🔬 Single Analysis":
    
    # Get values from session state (persisted from sidebar)
    show_analysis = st.session_state.get('show_single_analysis', False)
    symbol_input = st.session_state.get('analysis_symbol', 'BTCUSDT')
    analysis_tf = st.session_state.get('analysis_tf', '15m')
    analysis_market = st.session_state.get('analysis_market', '🪙 Crypto')
    analysis_mode_key = st.session_state.get('analysis_mode', 'day_trade')
    single_mode_config = st.session_state.get('single_mode_config', {})
    
    # Determine narrative mode from trading mode
    is_investment = analysis_mode_key == 'investment'
    narrative_mode = 'investment' if is_investment else 'swing'
    
    # Header based on mode
    if single_mode_config:
        mode_icon = single_mode_config.get('icon', '📊')
        mode_name = single_mode_config.get('name', 'Trading')
        mode_color = single_mode_config.get('color', '#00d4ff')
        st.markdown(f"## {mode_icon} {mode_name} Analysis")
        st.markdown(f"<p style='color: #aaa;'>{single_mode_config.get('description', 'Analyzing...')}</p>", unsafe_allow_html=True)
    elif is_investment:
        st.markdown("## 💎 Long-Term Investment Analysis")
        st.markdown("<p style='color: #aaa;'>Accumulate • Hold • Trim recommendations with educational insights</p>", unsafe_allow_html=True)
    else:
        st.markdown("## 📊 Trading Analysis")
        st.markdown("<p style='color: #aaa;'>Entry • Exit • Risk Management with detailed reasoning</p>", unsafe_allow_html=True)
    
    # Show analysis when session state says to
    if show_analysis:
        with st.spinner(f"Generating comprehensive analysis for {symbol_input}..."):
            
            # Fetch data based on market type
            if "Crypto" in analysis_market:
                df = fetch_binance_klines(symbol_input, analysis_tf, 200)
            else:
                df = fetch_stock_data(symbol_input, analysis_tf, 200)
            
            if df is not None and len(df) >= 50:
                
                # Run Master Narrative Analysis
                result = narrative_analyze(df, symbol_input, narrative_mode, analysis_tf)
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 CALCULATE COMBINED CONFIDENCE SCORE
                # Uses REAL Binance API whale data for accurate institutional scoring
                # ═══════════════════════════════════════════════════════════
                
                # Generate signal and all analysis data
                signal = SignalGenerator.generate_signal(df, symbol_input, analysis_tf)
                mf = calculate_money_flow(df)
                smc = detect_smc(df)
                pre_break = detect_pre_breakout(df)
                whale = detect_whale_activity(df)  # Basic chart-based
                institutional = detect_institutional_activity(df, mf)
                
                # 🐋 FETCH REAL WHALE DATA FROM BINANCE API (for Crypto only)
                # This ensures confidence scores match the whale data shown below
                stock_inst_data = None  # For stocks/ETFs
                unified_verdict = None  # Store for later display
                
                if "Crypto" in analysis_market:
                    try:
                        real_whale_data = get_whale_analysis(symbol_input)
                        
                        # Enrich whale dict with REAL API data
                        if real_whale_data:
                            # OI data
                            whale['open_interest'] = real_whale_data.get('open_interest', {})
                            
                            # Funding rate
                            whale['funding_rate'] = real_whale_data.get('funding', {}).get('rate', 0)
                            
                            # Top trader positioning (most important!)
                            whale['top_trader_ls'] = real_whale_data.get('top_trader_ls', {})
                            
                            # Retail positioning
                            whale['retail_ls'] = real_whale_data.get('retail_ls', {})
                            
                            # Determine whale direction from real data
                            top_long = real_whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                            if top_long >= 55:
                                whale['whale_detected'] = True
                                whale['direction'] = 'BULLISH'
                                whale['confidence'] = min(90, int((top_long - 50) * 3))
                            elif top_long <= 45:
                                whale['whale_detected'] = True
                                whale['direction'] = 'BEARISH'
                                whale['confidence'] = min(90, int((50 - top_long) * 3))
                            else:
                                whale['direction'] = 'NEUTRAL'
                                whale['confidence'] = 30
                            
                            # 🎯 CALCULATE UNIFIED VERDICT NOW (for score adjustment)
                            from core.education import get_unified_verdict
                            oi_change = real_whale_data.get('open_interest', {}).get('change_24h', 0)
                            price_change = real_whale_data.get('price_change_24h', 0)
                            funding_raw = real_whale_data.get('funding', {}).get('rate', 0)
                            retail_long = real_whale_data.get('retail_ls', {}).get('long_pct', 50)
                            tech_action_str = result.action.value if hasattr(result.action, 'value') else str(result.action) if result else 'WAIT'
                            
                            unified_verdict = get_unified_verdict(
                                oi_change, price_change, top_long, retail_long, funding_raw, tech_action_str
                            )
                            
                            # Store for later display
                            whale['unified_verdict'] = unified_verdict
                            whale['real_whale_data'] = real_whale_data
                            
                            # 🚨 ADJUST WHALE CONFIDENCE BASED ON UNIFIED VERDICT
                            # If verdict says LOW confidence or AVOID/WAIT, reduce institutional impact
                            verdict_conf = unified_verdict.get('confidence', 'MEDIUM')
                            verdict_action = str(unified_verdict.get('unified_action', unified_verdict.get('action', 'WAIT'))).upper()
                            
                            # Check for non-confirmatory actions
                            is_wait_or_avoid = any(x in verdict_action for x in ['AVOID', 'WAIT', 'RANGE', 'CAUTION'])
                            
                            if verdict_conf == 'LOW' or is_wait_or_avoid:
                                # No clear edge from whale data - set to neutral
                                whale['direction'] = 'NEUTRAL'
                                whale['confidence'] = 30  # Low confidence
                                whale['no_edge'] = True
                                whale['verdict_override'] = True  # Flag for score calculation
                            elif verdict_conf == 'MEDIUM':
                                whale['confidence'] = min(whale['confidence'], 50)
                                whale['verdict_override'] = True
                            # HIGH confidence keeps original values
                            
                    except Exception as e:
                        pass  # Use basic whale data if API fails
                
                # 📊 FETCH STOCK INSTITUTIONAL DATA (for Stocks/ETFs)
                elif "Stock" in analysis_market or "ETF" in analysis_market:
                    quiver_key = get_setting('quiver_api_key', '')
                    if quiver_key:
                        try:
                            stock_inst_data = get_stock_institutional_analysis(quiver_key, symbol_input)
                            
                            # Use stock institutional score to modify inst_score
                            if stock_inst_data and stock_inst_data.get('available'):
                                stock_score = stock_inst_data.get('combined_score', 50)
                                # Blend with chart-based institutional detection
                                # Stock data is more reliable, so weight it higher
                                inst_score_override = stock_score
                                institutional['stock_inst_score'] = stock_score
                                institutional['stock_inst_data'] = stock_inst_data
                        except Exception as e:
                            pass  # Use chart-based if API fails
                
                # Get OI change and whale % for threshold validation
                oi_data_single = whale.get('open_interest', {})
                oi_change_single = oi_data_single.get('change_24h', 0) if isinstance(oi_data_single, dict) else 0
                top_trader_single = whale.get('top_trader_ls', {})
                whale_pct_single = top_trader_single.get('long_pct', 50) if isinstance(top_trader_single, dict) else 50
                
                # Get retail % for 3-layer scoring
                retail_data_single = whale.get('retail_ls', {})
                retail_pct_single = retail_data_single.get('long_pct', 50) if isinstance(retail_data_single, dict) else 50
                
                # Get price change
                price_change_single = whale.get('real_whale_data', {}).get('price_change_24h', 0) if whale.get('real_whale_data') else 0
                
                # Calculate confidence scores with enriched whale data
                # NOW TIMEFRAME-AWARE! Weighted by trade mode.
                confidence_scores = calculate_confidence_scores(
                    signal, mf, smc, whale, pre_break, institutional,
                    trade_mode=analysis_mode_key,
                    timeframe=analysis_tf,
                    oi_change=oi_change_single,
                    whale_pct=whale_pct_single
                )
                
                # Get support level for "wait for dip" message
                support_lvl_single = None
                if signal and signal.direction == 'LONG':
                    support_lvl_single = signal.stop_loss
                
                # Get move position data from SMC
                smc_struct_single = smc.get('structure', {}) if smc else {}
                swing_high_single = smc_struct_single.get('last_swing_high', 0)
                swing_low_single = smc_struct_single.get('last_swing_low', 0)
                structure_type_single = smc_struct_single.get('structure', 'Unknown')
                current_price_single = df['Close'].iloc[-1] if df is not None and len(df) > 0 else 0
                
                # Calculate position in range for money flow context
                if swing_high_single and swing_low_single and swing_high_single > swing_low_single:
                    pos_in_range_single = ((current_price_single - swing_low_single) / (swing_high_single - swing_low_single)) * 100
                    pos_in_range_single = max(0, min(100, pos_in_range_single))
                else:
                    pos_in_range_single = 50
                
                # 🎯 Calculate Money Flow Phase FIRST for predictive signal
                is_accum_single = mf.get('is_accumulating', False)
                is_distrib_single = mf.get('is_distributing', False)
                flow_context_early = get_money_flow_context(
                    is_accumulating=is_accum_single,
                    is_distributing=is_distrib_single,
                    position_pct=pos_in_range_single,
                    structure_type=structure_type_single
                )
                money_flow_phase_single = flow_context_early.phase
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 NEW 3-LAYER PREDICTIVE SCORE
                # ═══════════════════════════════════════════════════════════
                predictive_result = calculate_predictive_score(
                    oi_change=oi_change_single,
                    price_change=price_change_single,
                    whale_pct=whale_pct_single,
                    retail_pct=retail_pct_single,
                    ta_score=confidence_scores.get('ta_score', 50),
                    trade_mode=analysis_mode_key,
                    timeframe=analysis_tf,
                    support_level=support_lvl_single,
                    current_price=current_price_single,
                    swing_high=swing_high_single,
                    swing_low=swing_low_single,
                    structure_type=structure_type_single,
                    money_flow_phase=money_flow_phase_single  # 🚀 For predictive signal!
                )
                
                # ═══════════════════════════════════════════════════════════════════
                # 🎯 CRITICAL: Check if signal direction matches predictive direction
                # If they conflict, regenerate signal with correct direction for levels
                # ═══════════════════════════════════════════════════════════════════
                
                # Use Layer 1 Direction (BULLISH/BEARISH/NEUTRAL) - this is the truth!
                pred_direction_single = predictive_result.direction if predictive_result else 'NEUTRAL'
                
                # Determine if we need LONG or SHORT levels
                pred_is_bullish_single = pred_direction_single in ['BULLISH', 'WEAK_BULLISH', 'TA_BULLISH']
                pred_is_bearish_single = pred_direction_single in ['BEARISH', 'WEAK_BEARISH', 'TA_BEARISH']
                
                # Check for direction conflict
                signal_is_long_single = signal.direction == 'LONG' if signal else True
                needs_regeneration_single = (pred_is_bullish_single and not signal_is_long_single) or (pred_is_bearish_single and signal_is_long_single)
                
                if needs_regeneration_single and signal:
                    # Determine which direction we need
                    force_dir_single = 'LONG' if pred_is_bullish_single else 'SHORT'
                    
                    # Regenerate signal with correct direction
                    new_signal_single = SignalGenerator.generate_signal(df, symbol_input, analysis_tf, force_direction=force_dir_single)
                    
                    if new_signal_single and new_signal_single.is_valid:
                        signal = new_signal_single  # Use new signal with correct levels
                        # Update support level for predictive
                        if signal.direction == 'LONG':
                            support_lvl_single = signal.stop_loss
                    else:
                        # 🚨 Regeneration failed - flag this mismatch for display
                        # Don't override with artificial levels - show warning instead
                        signal.levels_mismatch = True
                        signal.expected_direction = force_dir_single
                
                ta_score = confidence_scores.get('ta_score', 50)
                inst_score = confidence_scores.get('inst_score', 50)
                combined_score = confidence_scores.get('combined_score', 50)
                conf_level = confidence_scores.get('confidence_level', 'MODERATE')
                conf_color = confidence_scores.get('confidence_color', '#ffcc00')
                action_hint = confidence_scores.get('action_hint', '')
                alignment = confidence_scores.get('alignment', 'NEUTRAL')
                alignment_note = confidence_scores.get('alignment_note', '')
                
                # ═══════════════════════════════════════════════════════════
                # ═══════════════════════════════════════════════════════════
                # 📖 UNIFIED ACTION BOX + COMBINED LEARNING
                # One clear answer at top, details collapsible below
                # ═══════════════════════════════════════════════════════════
                
                style = result.style
                final_score = predictive_result.final_score
                action_word = predictive_result.final_action
                
                from core.unified_scoring import generate_combined_learning
                
                # Get all data for combined learning
                # FIX: Use correct paths to OI and price data!
                oi_data_cl = whale.get('open_interest', {}) if whale else {}
                oi_24h = oi_data_cl.get('change_24h', 0) if isinstance(oi_data_cl, dict) else 0
                real_whale_cl = whale.get('real_whale_data', {}) if whale else {}
                price_24h = real_whale_cl.get('price_change_24h', 0) if isinstance(real_whale_cl, dict) else 0
                whale_pct_learn = whale_pct_single if 'whale_pct_single' in dir() else 50
                retail_pct_learn = retail_pct_single if 'retail_pct_single' in dir() else (100 - whale_pct_learn)
                position_pct_learn = predictive_result.position_pct if hasattr(predictive_result, 'position_pct') else 50
                
                # Get position from raw data if available
                if 'current_price' in dir() and 'swing_high' in dir() and 'swing_low' in dir():
                    if swing_high and swing_low and swing_high > swing_low:
                        position_pct_learn = ((current_price - swing_low) / (swing_high - swing_low)) * 100
                
                combined_learning = generate_combined_learning(
                    signal_name=predictive_result.final_action,
                    direction=predictive_result.direction,
                    whale_pct=whale_pct_learn,
                    retail_pct=retail_pct_learn,
                    oi_change=oi_24h,
                    price_change=price_24h,
                    money_flow_phase=money_flow_phase_single if 'money_flow_phase_single' in dir() else '',
                    structure_type=result.structure if hasattr(result, 'structure') else 'Mixed',
                    position_pct=position_pct_learn,
                    ta_score=result.ta_score if hasattr(result, 'ta_score') else 50,
                )
                
                # Determine action display based on combined_learning direction
                cl_direction = combined_learning['direction']
                cl_action = combined_learning['conclusion_action']
                
                # Use shared function for setup info
                from utils.formatters import get_setup_info, render_signal_header_html, render_combined_learning_html
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 SINGLE SOURCE OF TRUTH: Use predictive_result.trade_direction directly!
                # No mapping needed - trade_direction is already 'LONG', 'SHORT', 'WAIT', etc.
                # ═══════════════════════════════════════════════════════════
                setup_direction = predictive_result.trade_direction if hasattr(predictive_result, 'trade_direction') else 'WAIT'
                
                setup_text, setup_color, conclusion_color, conclusion_bg = get_setup_info(setup_direction)
                
                # ═══════════════════════════════════════════════════════════
                # HEADER WITH RECOMMENDATION (Using shared component)
                # ═══════════════════════════════════════════════════════════
                
                header_html = render_signal_header_html(
                    symbol=symbol_input,
                    action_word=action_word,
                    score=final_score,
                    summary=predictive_result.final_summary,
                    direction_score=predictive_result.direction_score,
                    squeeze_score=predictive_result.squeeze_score,
                    timing_score=predictive_result.timing_score,
                    setup_text=setup_text,
                    setup_color=setup_color,
                    bg_color=style['bg']
                )
                st.markdown(header_html, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 📖 COMBINED LEARNING (Using shared component)
                # ═══════════════════════════════════════════════════════════
                
                cl_html = render_combined_learning_html(
                    conclusion=combined_learning['conclusion'],
                    conclusion_action=combined_learning['conclusion_action'],
                    conclusion_color=conclusion_color,
                    conclusion_bg=conclusion_bg,
                    stories=combined_learning['stories'],
                    is_squeeze=combined_learning['is_squeeze'],
                    squeeze_type=combined_learning.get('squeeze_type'),
                    direction=combined_learning['direction'],
                    has_conflict=combined_learning.get('has_conflict', False),
                    conflicts=combined_learning.get('conflicts', []),
                    is_capitulation_long=combined_learning.get('is_capitulation_long', False),
                    whale_pct=whale_pct_learn
                )
                
                with st.expander("📖 **COMBINED LEARNING** - What's Really Happening?", expanded=False):
                    st.markdown(cl_html['conclusion_html'], unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown("**📊 THE FULL STORY:**")
                    st.markdown(cl_html['stories_html'], unsafe_allow_html=True)
                    
                    if cl_html['squeeze_html']:
                        st.markdown(cl_html['squeeze_html'], unsafe_allow_html=True)
                    if cl_html['conflict_html']:
                        st.markdown(cl_html['conflict_html'], unsafe_allow_html=True)
                    if cl_html['capitulation_html']:
                        st.markdown(cl_html['capitulation_html'], unsafe_allow_html=True)
                
                # 🚨 WARNING: If whale verdict says AVOID but TA looks good
                if unified_verdict and whale.get('no_edge'):
                    verdict_action = unified_verdict.get('action', '')
                    if 'AVOID' in str(verdict_action).upper() or 'WAIT' in str(verdict_action).upper():
                        st.warning(f"""
                        ⚠️ **WHALE DATA CAUTION**: Real-time Binance data shows **no clear institutional edge**.
                        
                        - Technical Analysis: Strong ({ta_score}/100)
                        - Whale/Institutional: Neutral (no directional bias)
                        
                        **Recommendation**: Use TA-based levels with **tight stops**. Whale data doesn't confirm this move.
                        """)
                elif unified_verdict and unified_verdict.get('confidence') == 'LOW':
                    st.info(f"""
                    ℹ️ **LOW Institutional Confidence**: Whale positioning is mixed. 
                    Trade with caution and use proper risk management.
                    """)
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 3-LAYER PREDICTIVE SCORE DISPLAY (NEW!)
                # ═══════════════════════════════════════════════════════════
                
                # Colors for each layer
                dir_color = "#00ff88" if predictive_result.direction in ["BULLISH"] else "#00d4aa" if predictive_result.direction == "WEAK_BULLISH" else "#ff6b6b" if predictive_result.direction == "BEARISH" else "#ffcc00"
                squeeze_color = "#00ff88" if predictive_result.squeeze_potential == "HIGH" else "#00d4aa" if predictive_result.squeeze_potential == "MEDIUM" else "#ff6b6b" if predictive_result.squeeze_potential == "CONFLICT" else "#888"
                timing_color = "#00ff88" if predictive_result.entry_timing == "NOW" else "#ffcc00"
                final_color = "#00ff88" if predictive_result.final_score >= 75 else "#00d4aa" if predictive_result.final_score >= 60 else "#ffcc00" if predictive_result.final_score >= 45 else "#ff9500"
                
                # Truncate reasons safely
                dir_reason_short = predictive_result.direction_reason
                timing_reason_short = predictive_result.timing_reason
                
                # 3 Layers in columns (score already shown in header - no need for redundant box)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid {dir_color}; min-height: 180px;'>
                        <div style='color: #888; font-size: 0.85em;'>📈 LAYER 1: Direction</div>
                        <div style='color: {dir_color}; font-size: 1.5em; font-weight: bold; margin: 8px 0;'>{predictive_result.direction}</div>
                        <div style='color: #aaa; font-size: 0.85em;'>{predictive_result.direction_confidence} confidence</div>
                        <div style='color: {dir_color}; font-size: 2em; font-weight: bold; margin-top: 10px;'>{predictive_result.direction_score}/40</div>
                        <div style='color: #666; font-size: 0.75em; margin-top: 10px; border-top: 1px solid #333; padding-top: 8px;'>{dir_reason_short}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid {squeeze_color}; min-height: 180px;'>
                        <div style='color: #888; font-size: 0.85em;'>🔥 LAYER 2: Squeeze</div>
                        <div style='color: {squeeze_color}; font-size: 1.5em; font-weight: bold; margin: 8px 0;'>{predictive_result.squeeze_potential}</div>
                        <div style='color: #aaa; font-size: 0.85em;'>W:{whale_pct_single:.0f}% vs R:{retail_pct_single:.0f}%</div>
                        <div style='color: {squeeze_color}; font-size: 2em; font-weight: bold; margin-top: 10px;'>{predictive_result.squeeze_score}/30</div>
                        <div style='color: #666; font-size: 0.75em; margin-top: 10px; border-top: 1px solid #333; padding-top: 8px;'>Divergence: {predictive_result.divergence_pct:+.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Move position color
                    move_color = "#00ff88" if predictive_result.move_position == "EARLY" else "#ffcc00" if predictive_result.move_position == "MIDDLE" else "#ff6b6b" if predictive_result.move_position == "LATE" else "#888"
                    st.markdown(f"""
                    <div style='background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid {timing_color}; min-height: 180px;'>
                        <div style='color: #888; font-size: 0.85em;'>⏰ LAYER 3: Entry (TA + Position)</div>
                        <div style='color: {timing_color}; font-size: 1.5em; font-weight: bold; margin: 8px 0;'>{predictive_result.entry_timing}</div>
                        <div style='color: #aaa; font-size: 0.85em;'>TA: {predictive_result.ta_score} | Pos: <span style='color: {move_color};'>{predictive_result.move_position}</span></div>
                        <div style='color: {timing_color}; font-size: 2em; font-weight: bold; margin-top: 10px;'>{predictive_result.timing_score}/30</div>
                        <div style='color: #666; font-size: 0.75em; margin-top: 10px; border-top: 1px solid #333; padding-top: 8px;'>{timing_reason_short}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary with proper top margin
                st.markdown(f"""
                <div style='background: {final_color}22; border-radius: 8px; padding: 15px; text-align: center; margin-top: 25px; margin-bottom: 15px;'>
                    <div style='color: {final_color}; font-size: 1.1em;'>{predictive_result.final_summary}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Raw Data in columns
                st.markdown("**📊 Raw Data**")
                data_cols = st.columns(5)
                
                oi_color = "#00ff88" if oi_change_single > 0.5 else "#ff6b6b" if oi_change_single < -0.5 else "#888"
                price_color = "#00ff88" if price_change_single > 0 else "#ff6b6b" if price_change_single < 0 else "#888"
                
                with data_cols[0]:
                    st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.8em;'>OI 24h</div><div style='color:{oi_color};font-size:1.3em;font-weight:bold;'>{oi_change_single:+.1f}%</div></div>", unsafe_allow_html=True)
                with data_cols[1]:
                    st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.8em;'>Price 24h</div><div style='color:{price_color};font-size:1.3em;font-weight:bold;'>{price_change_single:+.1f}%</div></div>", unsafe_allow_html=True)
                with data_cols[2]:
                    st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.8em;'>🐋 Whales</div><div style='color:#00d4ff;font-size:1.3em;font-weight:bold;'>{whale_pct_single:.0f}%</div></div>", unsafe_allow_html=True)
                with data_cols[3]:
                    st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.8em;'>🐑 Retail</div><div style='color:#ffcc00;font-size:1.3em;font-weight:bold;'>{retail_pct_single:.0f}%</div></div>", unsafe_allow_html=True)
                with data_cols[4]:
                    st.markdown(f"<div style='text-align:center;'><div style='color:#666;font-size:0.8em;'>TA Score</div><div style='color:#aaa;font-size:1.3em;font-weight:bold;'>{ta_score}</div></div>", unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 📊 MARKET CONTEXT - Structure + Money Flow
                # ═══════════════════════════════════════════════════════════
                
                # Get market structure
                structure_info = smc.get('structure', {})
                structure_type = structure_info.get('structure', 'Unknown')
                swing_high = structure_info.get('last_swing_high', 0)
                swing_low = structure_info.get('last_swing_low', 0)
                
                # Use move position from predictive_result (already calculated)
                move_label = predictive_result.move_position
                position_in_range = predictive_result.move_position_pct
                
                # Check if structure is consolidating/ranging (no clear trend)
                is_consolidating = structure_type in ['Consolidating', 'Ranging', 'Mixed', 'Unknown']
                
                # Move position styling - CONTEXT AWARE!
                # Use ACTUAL position percentage for tips, not just label
                # (Label can be "EARLY" for shorts at highs, but tip should reflect price location)
                
                if move_label == "EARLY":
                    move_color = "#00ff88"
                    move_emoji = "🟢"
                    if is_consolidating:
                        # Check actual position to give correct tip
                        if position_in_range <= 30:
                            move_tip = "At range lows - good for range longs"
                        elif position_in_range >= 70:
                            move_tip = "At range highs - early for shorts"
                        else:
                            move_tip = "Good positioning for this setup"
                    else:
                        move_tip = "Great entry - catching the move!"
                elif move_label == "MIDDLE":
                    move_color = "#ffcc00"
                    move_emoji = "🟡"
                    if is_consolidating:
                        move_tip = "Mid-range - wait for edge of range"
                    else:
                        move_tip = "OK entry - some room left"
                elif move_label in ["LATE", "CHASING"]:
                    move_color = "#ff6b6b"
                    move_emoji = "🔴"
                    if is_consolidating:
                        # Check actual position to give correct tip
                        if position_in_range >= 70:
                            move_tip = "At range highs - risky for longs, consider shorts"
                        elif position_in_range <= 30:
                            move_tip = "At range lows - risky for shorts"
                        else:
                            move_tip = "Chasing - wait for better entry"
                    else:
                        move_tip = "Chasing - wait for pullback"
                elif move_label in ["NEAR HIGH", "NEAR LOW", "MID-RANGE"]:
                    # Range-specific labels
                    if move_label == "NEAR HIGH":
                        move_color = "#ff9500"
                        move_emoji = "⬆️"
                        move_tip = "At range highs - good for shorts or wait"
                    elif move_label == "NEAR LOW":
                        move_color = "#00d4aa"
                        move_emoji = "⬇️"
                        move_tip = "At range lows - good for longs"
                    else:
                        move_color = "#888"
                        move_emoji = "↔️"
                        move_tip = "Mid-range - wait for edge"
                else:
                    move_color = "#888"
                    move_emoji = "↔️"
                    move_tip = "No clear trend"
                
                # Determine structure color and emoji
                if structure_type == 'Bullish':
                    struct_color = "#00ff88"
                    struct_emoji = "📈"
                    struct_detail = "HH + HL"
                elif structure_type == 'Bearish':
                    struct_color = "#ff6b6b"
                    struct_emoji = "📉"
                    struct_detail = "LH + LL"
                else:
                    struct_color = "#ffcc00"
                    struct_emoji = "↔️"
                    struct_detail = "Mixed"
                
                # Get money flow status WITH CONTEXT (position-aware!)
                is_accumulating = mf.get('is_accumulating', False)
                is_distributing = mf.get('is_distributing', False)
                
                # Use contextual flow analysis
                flow_context = get_money_flow_context(
                    is_accumulating=is_accumulating,
                    is_distributing=is_distributing,
                    position_pct=position_in_range,
                    structure_type=structure_type
                )
                
                flow_status = flow_context.phase
                flow_color = flow_context.phase_color
                flow_emoji = flow_context.phase_emoji
                flow_detail = flow_context.phase_detail
                
                # Build warning HTML separately (if exists)
                warning_html = ""
                if flow_context.warning:
                    warning_html = f"<div style='color: #ff9500; font-size: 0.75em; margin-top: 3px;'>⚠️ {flow_context.warning}</div>"
                
                # Build the complete HTML as a single string
                market_context_html = f"""<div style='background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 15px 0; border: 1px solid #333;'>
                    <div style='color: #00d4ff; font-weight: bold; margin-bottom: 12px;'>📊 MARKET CONTEXT ({analysis_tf})</div>
                    <div style='display: flex; gap: 20px;'>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Structure</div>
                            <div style='color: {struct_color}; font-size: 1.2em; font-weight: bold;'>{struct_emoji} {structure_type}</div>
                            <div style='color: #666; font-size: 0.8em;'>{struct_detail}</div>
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Money Flow</div>
                            <div style='color: {flow_color}; font-size: 1.2em; font-weight: bold;'>{flow_emoji} {flow_status}</div>
                            <div style='color: #666; font-size: 0.8em;'>{flow_detail}</div>{warning_html}
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Move Position</div>
                            <div style='color: {move_color}; font-size: 1.2em; font-weight: bold;'>{move_emoji} {move_label}</div>
                            <div style='color: #666; font-size: 0.8em;'>{position_in_range:.0f}% of range</div>
                        </div>
                        <div style='flex: 1;'>
                            <div style='color: #888; font-size: 0.85em;'>Swing Levels</div>
                            <div style='color: #aaa; font-size: 0.9em;'>H: <span style='color: #00d4aa;'>${swing_high:,.4f}</span></div>
                            <div style='color: #aaa; font-size: 0.9em;'>L: <span style='color: #ff6b6b;'>${swing_low:,.4f}</span></div>
                        </div>
                    </div>
                    <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;'>
                        <span style='color: {move_color}; font-size: 0.9em;'>💡 {move_tip}</span>
                    </div>
                </div>"""
                
                st.markdown(market_context_html, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 🎯 MULTI-TIMEFRAME TIP (When predictive diverges from current TF)
                # ═══════════════════════════════════════════════════════════
                try:
                    from core.money_flow_context import get_multi_timeframe_tip
                    mtf_tip = get_multi_timeframe_tip(
                        current_tf=analysis_tf,
                        current_phase=flow_status,
                        higher_tf_phase=None,  # We don't have higher TF data here
                        predictive_signal=predictive_result.final_action
                    )
                    
                    if mtf_tip.has_tip:
                        st.markdown(f"""
                        <div style='background: {mtf_tip.tip_color}15; border-left: 4px solid {mtf_tip.tip_color}; 
                                    padding: 12px; border-radius: 8px; margin-top: 10px;'>
                            <div style='color: {mtf_tip.tip_color}; font-weight: bold; margin-bottom: 5px;'>
                                {mtf_tip.tip_emoji} {mtf_tip.tip_title}
                            </div>
                            <div style='color: #aaa; font-size: 0.9em; margin-bottom: 8px;'>
                                {mtf_tip.tip_detail}
                            </div>
                            <div style='color: {mtf_tip.tip_color}; font-size: 0.85em; font-style: italic;'>
                                👉 {mtf_tip.action}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as mtf_err:
                    pass  # Skip if MTF tip fails
                
                # ═══════════════════════════════════════════════════════════
                # 📚 LEARN: Money Flow Phases (Expandable)
                # ═══════════════════════════════════════════════════════════
                with st.expander(f"📚 Learn: What is {flow_status}?", expanded=False):
                    edu = get_money_flow_education(flow_status)
                    
                    st.markdown(f"""
                    <div style='background: #0d1117; padding: 15px; border-radius: 8px; border-left: 3px solid {flow_color};'>
                        <div style='font-size: 1.3em; font-weight: bold; color: {flow_color}; margin-bottom: 10px;'>
                            {edu.get('emoji', '📊')} {edu.get('title', flow_status)}
                        </div>
                        <div style='color: #aaa; margin-bottom: 15px;'>{edu.get('short', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Main description
                    st.markdown(edu.get('description', ''))
                    
                    # Wyckoff reference
                    if edu.get('wyckoff'):
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                            <span style='color: #00d4ff;'>📖 Wyckoff:</span> 
                            <span style='color: #888;'>{edu.get('wyckoff')}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ═══════════════════════════════════════════════════════════
                    # CLICKABLE PHASES - Put first so user sees immediately!
                    # ═══════════════════════════════════════════════════════════
                    st.markdown("---")
                    st.markdown("### 📚 Learn All Phases (Click to Expand)")
                    
                    # Create tabs for each phase category
                    phase_tabs = st.tabs(["🟢 Bullish Phases", "🔴 Bearish Phases", "⚪ Neutral Phases"])
                    
                    bullish_phases = ["ACCUMULATION", "MARKUP"]
                    bearish_phases = ["DISTRIBUTION", "PROFIT TAKING", "FOMO / DIST RISK"]
                    neutral_phases = ["CONSOLIDATION", "RE-ACCUMULATION", "EXHAUSTION", "CAPITULATION"]
                    
                    with phase_tabs[0]:  # Bullish
                        for phase_key in bullish_phases:
                            phase_edu = MONEY_FLOW_EDUCATION.get(phase_key, {})
                            is_current = phase_key == flow_status
                            with st.expander(f"{phase_edu.get('emoji', '')} {phase_key}" + (" ← YOU ARE HERE" if is_current else ""), expanded=is_current):
                                st.markdown(f"**{phase_edu.get('title', phase_key)}**")
                                st.markdown(f"*{phase_edu.get('short', '')}*")
                                st.markdown(phase_edu.get('description', ''))
                                if phase_edu.get('wyckoff'):
                                    st.info(f"📖 Wyckoff: {phase_edu.get('wyckoff')}")
                    
                    with phase_tabs[1]:  # Bearish
                        for phase_key in bearish_phases:
                            phase_edu = MONEY_FLOW_EDUCATION.get(phase_key, {})
                            is_current = phase_key == flow_status
                            with st.expander(f"{phase_edu.get('emoji', '')} {phase_key}" + (" ← YOU ARE HERE" if is_current else ""), expanded=is_current):
                                st.markdown(f"**{phase_edu.get('title', phase_key)}**")
                                st.markdown(f"*{phase_edu.get('short', '')}*")
                                st.markdown(phase_edu.get('description', ''))
                                if phase_edu.get('wyckoff'):
                                    st.info(f"📖 Wyckoff: {phase_edu.get('wyckoff')}")
                    
                    with phase_tabs[2]:  # Neutral
                        for phase_key in neutral_phases:
                            phase_edu = MONEY_FLOW_EDUCATION.get(phase_key, {})
                            is_current = phase_key == flow_status
                            with st.expander(f"{phase_edu.get('emoji', '')} {phase_key}" + (" ← YOU ARE HERE" if is_current else ""), expanded=is_current):
                                st.markdown(f"**{phase_edu.get('title', phase_key)}**")
                                st.markdown(f"*{phase_edu.get('short', '')}*")
                                st.markdown(phase_edu.get('description', ''))
                                if phase_edu.get('wyckoff'):
                                    st.info(f"📖 Wyckoff: {phase_edu.get('wyckoff')}")
                    
                    # Important note about Structure vs Money Flow - at the bottom
                    st.markdown("---")
                    st.markdown("""
                    <div style='background: #1a2a3a; padding: 12px; border-radius: 8px; border-left: 3px solid #00d4ff;'>
                        <div style='color: #00d4ff; font-weight: bold; margin-bottom: 8px;'>💡 Structure vs Money Flow - What's the Difference?</div>
                        <div style='color: #aaa; font-size: 0.9em;'>
                            <b>Structure (Consolidating):</b> Price pattern - no clear Higher Highs/Lows or Lower Highs/Lows<br>
                            <b>Money Flow (Consolidation):</b> Volume direction - no clear buying OR selling pressure<br><br>
                            <em>Both can show "consolidation" at the same time - that's a strong "WAIT" signal!</em>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # CHART AND LEVELS
                # ═══════════════════════════════════════════════════════════
                
                col_chart, col_levels = st.columns([2, 1])
                
                with col_chart:
                    # Signal already generated above for confidence calculation
                    if signal:
                        fig = create_trade_setup_chart(df, signal, combined_score)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Basic chart without signal lines
                        import plotly.graph_objects as go
                        fig = go.Figure(data=[go.Candlestick(
                            x=df['DateTime'],
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close']
                        )])
                        fig.update_layout(template='plotly_dark', height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col_levels:
                    if is_investment:
                        # INVESTMENT MODE - DCA Zones
                        st.markdown("### 💰 DCA Accumulation Zones")
                        st.markdown("<p style='color: #888; font-size: 0.9em;'>Spread your investment across these zones</p>", unsafe_allow_html=True)
                        
                        for zone in result.dca_zones:
                            st.markdown(f"""
                            <div style='background: #1a1a2e; padding: 8px 12px; margin: 5px 0; border-radius: 6px;
                                        border-left: 3px solid #00d4aa;'>
                                <strong>{zone['label']}</strong>: {fmt_price(zone['price'])}<br>
                                <span style='color: #888;'>Allocate: {zone['allocation']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("### 📈 Profit Taking Zones")
                        for zone in result.trim_zones:
                            st.markdown(f"""
                            <div style='background: #1a2a1a; padding: 8px 12px; margin: 5px 0; border-radius: 6px;
                                        border-left: 3px solid #ffcc00;'>
                                <strong>{zone['label']}</strong>: {fmt_price(zone['price'])}<br>
                                <span style='color: #888;'>{zone['action']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # TRADING MODE - Entry/SL/TP
                        # ⚠️ USE SignalGenerator levels for consistency with Scanner!
                        st.markdown("### 🎯 Trade Setup")
                        
                        # 🚨 Check for levels mismatch warning
                        levels_mismatch = getattr(signal, 'levels_mismatch', False) if signal else False
                        expected_dir = getattr(signal, 'expected_direction', None) if signal else None
                        
                        if levels_mismatch and expected_dir:
                            st.markdown(f"""
                            <div style='background: #ff6b6b20; border: 2px solid #ff6b6b; border-radius: 8px; padding: 12px; margin-bottom: 15px;'>
                                <div style='color: #ff6b6b; font-weight: bold; font-size: 1.1em;'>
                                    ⚠️ LEVELS MISMATCH
                                </div>
                                <div style='color: #ffcc00; margin-top: 8px;'>
                                    Predictive analysis suggests <strong>{expected_dir}</strong>, but no valid {expected_dir} structure levels found.
                                </div>
                                <div style='color: #aaa; margin-top: 8px; font-size: 0.9em;'>
                                    The levels shown are based on current technical structure. 
                                    <strong>Wait for better setup</strong> or use these levels with caution.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Use signal levels if available (SAME SOURCE AS SCANNER)
                        # Fall back to narrative result only if no signal
                        if signal:
                            display_entry = signal.entry
                            display_sl = signal.stop_loss
                            display_tp1 = signal.tp1
                            display_tp2 = signal.tp2
                            display_tp3 = signal.tp3
                            display_risk_pct = signal.risk_pct
                            display_rr = getattr(signal, 'rr_tp1', 0) or result.rr_ratio
                        else:
                            display_entry = result.entry
                            display_sl = result.stop_loss
                            display_tp1 = result.tp1
                            display_tp2 = result.tp2
                            display_tp3 = result.tp3
                            display_risk_pct = result.risk_pct
                            display_rr = result.rr_ratio
                        
                        st.markdown(f"""
                        <div style='background: #1a2a3a; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                            <strong style='color: #00d4ff;'>Entry:</strong> {fmt_price(display_entry)}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style='background: #3a1a1a; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                            <strong style='color: #ff4444;'>Stop Loss:</strong> {fmt_price(display_sl)}<br>
                            <span style='color: #888;'>Risk: {display_risk_pct:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Take Profit Targets:**")
                        tp1_pct = ((display_tp1 - display_entry) / display_entry * 100) if display_entry > 0 else 0
                        tp2_pct = ((display_tp2 - display_entry) / display_entry * 100) if display_entry > 0 else 0
                        tp3_pct = ((display_tp3 - display_entry) / display_entry * 100) if display_entry > 0 else 0
                        
                        # Calculate R:R for each TP level
                        rr_tp1 = tp1_pct / display_risk_pct if display_risk_pct > 0 else 0
                        rr_tp2 = tp2_pct / display_risk_pct if display_risk_pct > 0 else 0
                        rr_tp3 = tp3_pct / display_risk_pct if display_risk_pct > 0 else 0
                        
                        # Color code R:R - green >= 1, yellow >= 0.5, red < 0.5
                        rr1_color = "#00d4aa" if rr_tp1 >= 1 else "#ffcc00" if rr_tp1 >= 0.5 else "#ff4444"
                        rr2_color = "#00d4aa" if rr_tp2 >= 1 else "#ffcc00" if rr_tp2 >= 0.5 else "#ff4444"
                        rr3_color = "#00d4aa" if rr_tp3 >= 1 else "#ffcc00" if rr_tp3 >= 0.5 else "#ff4444"
                        
                        st.markdown(f"""
                        <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4aa;'>
                            <span style='color: #888;'>TP1:</span> 
                            <span style='color: #00d4aa; font-weight: bold;'>{fmt_price(display_tp1)}</span>
                            <span style='color: #00aa88;'>(+{tp1_pct:.1f}%)</span>
                            <span style='color: {rr1_color}; float: right; font-weight: bold;'>R:R {rr_tp1:.1f}:1</span>
                        </div>
                        <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4aa;'>
                            <span style='color: #888;'>TP2:</span> 
                            <span style='color: #00d4aa; font-weight: bold;'>{fmt_price(display_tp2)}</span>
                            <span style='color: #00aa88;'>(+{tp2_pct:.1f}%)</span>
                            <span style='color: {rr2_color}; float: right; font-weight: bold;'>R:R {rr_tp2:.1f}:1</span>
                        </div>
                        <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin: 5px 0; border-left: 3px solid #00d4aa;'>
                            <span style='color: #888;'>TP3:</span> 
                            <span style='color: #00d4aa; font-weight: bold;'>{fmt_price(display_tp3)}</span>
                            <span style='color: #00aa88;'>(+{tp3_pct:.1f}%)</span>
                            <span style='color: {rr3_color}; float: right; font-weight: bold;'>R:R {rr_tp3:.1f}:1</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if display_rr > 0:
                            st.markdown(f"**Risk/Reward:** 1:{display_rr:.1f}")
                    
                    # Risk Assessment
                    st.markdown("---")
                    risk_color = "#ff4444" if result.risk_level == "High" else "#ffcc00" if result.risk_level == "Medium" else "#00d4aa"
                    st.markdown(f"""
                    <div style='background: #1a1a2e; padding: 10px; border-radius: 6px;'>
                        <strong>Risk Level:</strong> <span style='color: {risk_color};'>{result.risk_level}</span><br>
                        <span style='color: #888; font-size: 0.9em;'>{result.risk_notes}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # ACTION PLAN - USE SAME LEVELS AS TRADE SETUP!
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 📋 Action Plan")
                st.markdown("<p style='color: #888;'>Step-by-step guide based on current conditions</p>", unsafe_allow_html=True)
                
                # Create action steps using SIGNAL levels (same as Trade Setup)
                # This ensures consistency between Trade Setup box and Action Plan
                if signal:
                    plan_entry = signal.entry
                    plan_sl = signal.stop_loss
                    plan_tp1 = signal.tp1
                    plan_tp2 = signal.tp2
                    plan_tp3 = signal.tp3
                    plan_risk = signal.risk_pct
                    plan_rr = getattr(signal, 'rr_tp1', 0) or result.rr_ratio
                else:
                    plan_entry = result.entry
                    plan_sl = result.stop_loss
                    plan_tp1 = result.tp1
                    plan_tp2 = result.tp2
                    plan_tp3 = result.tp3
                    plan_risk = result.risk_pct
                    plan_rr = result.rr_ratio
                
                # Build consistent action steps with R:R for each TP
                plan_tp1_pct = ((plan_tp1 - plan_entry) / plan_entry * 100) if plan_entry > 0 else 0
                plan_tp2_pct = ((plan_tp2 - plan_entry) / plan_entry * 100) if plan_entry > 0 else 0
                plan_tp3_pct = ((plan_tp3 - plan_entry) / plan_entry * 100) if plan_entry > 0 else 0
                
                plan_rr_tp1 = plan_tp1_pct / plan_risk if plan_risk > 0 else 0
                plan_rr_tp2 = plan_tp2_pct / plan_risk if plan_risk > 0 else 0
                plan_rr_tp3 = plan_tp3_pct / plan_risk if plan_risk > 0 else 0
                
                consistent_action_steps = [
                    f"🎯 Entry: {fmt_price(plan_entry)}",
                    f"🛑 Stop Loss: {fmt_price(plan_sl)} ({plan_risk:.1f}% risk)",
                    f"✅ TP1: {fmt_price(plan_tp1)} (+{plan_tp1_pct:.1f}%) R:R {plan_rr_tp1:.1f}:1 - Take 33%",
                    f"✅ TP2: {fmt_price(plan_tp2)} (+{plan_tp2_pct:.1f}%) R:R {plan_rr_tp2:.1f}:1 - Take 33%",
                    f"✅ TP3: {fmt_price(plan_tp3)} (+{plan_tp3_pct:.1f}%) R:R {plan_rr_tp3:.1f}:1 - Take remaining",
                ]
                
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    for step in consistent_action_steps[:len(consistent_action_steps)//2 + 1]:
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px 15px; margin: 8px 0; border-radius: 8px;
                                    border-left: 3px solid {style['color']};'>
                            {step}
                        </div>
                        """, unsafe_allow_html=True)
                
                with action_col2:
                    for step in consistent_action_steps[len(consistent_action_steps)//2 + 1:]:
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px 15px; margin: 8px 0; border-radius: 8px;
                                    border-left: 3px solid {style['color']};'>
                            {step}
                        </div>
                        """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # 🐋 WHALE & INSTITUTIONAL ANALYSIS (Single Analysis)
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 🐋 Whale & Institutional Analysis")
                st.markdown("<p style='color: #888;'>Real-time futures data: OI, Funding, Top Trader Positioning</p>", unsafe_allow_html=True)
                
                # Determine market type
                if "Crypto" in analysis_market:
                    # Use pre-fetched whale data and unified verdict from earlier
                    whale_data = whale.get('real_whale_data')
                    unified = whale.get('unified_verdict')
                    
                    # Check if we have real data
                    has_real_data = whale_data is not None and unified is not None
                    
                    if has_real_data:
                        try:
                            # Import education for scenario info
                            from core.education import (
                                INSTITUTIONAL_EDUCATION, 
                                SCENARIO_EDUCATION, identify_current_scenario
                            )
                            
                            # Extract all data
                            oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                            price_change = whale_data.get('price_change_24h', 0)
                            funding = whale_data.get('funding', {}).get('rate_pct', 0)
                            funding_raw = whale_data.get('funding', {}).get('rate', 0)
                            whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                            retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
                            
                            # Use pre-calculated unified verdict
                            scenario = unified['scenario']
                            conf_colors = {'HIGH': '#00d4aa', 'MEDIUM': '#ffcc00', 'LOW': '#ff8800'}
                            conf_color = conf_colors.get(unified['confidence'], '#888888')
                            
                            # ═══════════════════════════════════════════════════════
                            # 🎯 UNIFIED VERDICT AT THE TOP (Most Important!)
                            # ═══════════════════════════════════════════════════════
                            
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, {conf_color}22, {conf_color}11); 
                                        border: 3px solid {conf_color}; border-radius: 15px; padding: 25px; margin-bottom: 20px;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                                    <span style='color: #fff; font-size: 1.4em; font-weight: bold;'>
                                        🎯 {scenario.get('name', 'Market Analysis')}
                                    </span>
                                    <span style='background: {conf_color}; color: #000; padding: 10px 20px; border-radius: 20px; font-weight: bold;'>
                                        {unified['confidence']} Confidence
                                    </span>
                                </div>
                                <div style='color: #aaa; font-size: 0.9em; margin-bottom: 15px;'>
                                    📋 Conditions: {scenario.get('conditions', 'N/A')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Recommendation box - THE MOST IMPORTANT PART
                            rec_cols = st.columns([1, 1])
                            
                            # Convert markdown **bold** to HTML <b>bold</b>
                            import re
                            def md_to_html(text):
                                """Convert markdown bold to HTML bold"""
                                if not text:
                                    return ''
                                text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
                                return text.strip()
                            
                            with rec_cols[0]:
                                interpretation = md_to_html(scenario.get('interpretation', ''))
                                st.markdown(f"""
                                <div style='background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px; height: 100%;'>
                                    <div style='color: #888; font-size: 0.9em; margin-bottom: 10px;'>🔍 What's Happening</div>
                                    <div style='color: #ddd; font-size: 0.95em; line-height: 1.8; white-space: pre-line;'>{interpretation}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with rec_cols[1]:
                                recommendation = md_to_html(scenario.get('recommendation', ''))
                                st.markdown(f"""
                                <div style='background: #1a2a1a; border: 1px solid #2a4a2a; border-radius: 12px; padding: 20px; height: 100%;'>
                                    <div style='color: #00d4aa; font-size: 0.9em; margin-bottom: 10px;'>🎯 What To Do</div>
                                    <div style='color: #ddd; font-size: 0.95em; line-height: 1.8; white-space: pre-line;'>{recommendation}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Alignment indicator
                            st.markdown(f"""
                            <div style='background: {conf_color}15; border: 1px solid {conf_color}50; border-radius: 8px; padding: 12px; margin: 15px 0; text-align: center;'>
                                <span style='color: {conf_color}; font-weight: bold;'>{unified['alignment_note']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # ═══════════════════════════════════════════════════════
                            # 📊 METRIC CARDS WITH INFO TOOLTIPS
                            # ═══════════════════════════════════════════════════════
                            
                            st.markdown("#### 📊 Raw Data (Click ℹ️ to learn)")
                            
                            whale_cols = st.columns(5)
                            
                            # OI Change Card
                            with whale_cols[0]:
                                oi_color = "#00d4aa" if oi_change > 3 else "#ff4444" if oi_change < -3 else "#888"
                                oi_label = "📈 Rising" if oi_change > 3 else "📉 Falling" if oi_change < -3 else "➡️ Stable"
                                st.markdown(f"""
                                <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                    <div style='color: #888; font-size: 0.85em;'>📊 OI Change 24h</div>
                                    <div style='color: {oi_color}; font-size: 1.8em; font-weight: bold;'>{oi_change:+.1f}%</div>
                                    <div style='color: #666; font-size: 0.8em;'>{oi_label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("ℹ️ What is OI?"):
                                    st.markdown("""
                                    **Open Interest** = Total outstanding futures contracts
                                    
                                    📈 **Rising OI** = New money entering market
                                    📉 **Falling OI** = Money leaving market
                                    
                                    ⚠️ **OI alone doesn't tell direction!**
                                    Must combine with PRICE to understand who's entering/exiting.
                                    """)
                            
                            # Price Change Card
                            with whale_cols[1]:
                                price_color = "#00d4aa" if price_change > 0 else "#ff4444"
                                st.markdown(f"""
                                <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                    <div style='color: #888; font-size: 0.85em;'>💵 Price 24h</div>
                                    <div style='color: {price_color}; font-size: 1.8em; font-weight: bold;'>{price_change:+.1f}%</div>
                                    <div style='color: #666; font-size: 0.8em;'>{"📈 Up" if price_change > 0 else "📉 Down"}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("ℹ️ OI + Price"):
                                    st.markdown("""
                                    ### The 4 Combinations (MEMORIZE THIS!)
                                    
                                    | OI | Price | Signal | Meaning | Action |
                                    |:---:|:---:|:---:|:---|:---|
                                    | 📈 UP | 📈 UP | 🟢 **NEW LONGS** | Fresh buying, new money entering long | ✅ Bullish continuation |
                                    | 📈 UP | 📉 DOWN | 🔴 **NEW SHORTS** | Fresh selling, new money entering short | ✅ Bearish continuation |
                                    | 📉 DOWN | 📈 UP | 🟡 **SHORT COVERING** | Shorts closing, NOT new buying | ⚠️ Weak rally, may reverse |
                                    | 📉 DOWN | 📉 DOWN | 🟡 **LONG LIQUIDATION** | Longs closing/stopped out, NOT new shorts | ⚠️ Dump may be ending |
                                    
                                    ---
                                    
                                    **💡 KEY INSIGHT:**
                                    - **OI Rising** = New positions = Trend has CONVICTION
                                    - **OI Falling** = Closing positions = Trend may EXHAUST
                                    
                                    **🎯 CURRENT SITUATION:**
                                    """)
                                    # Show current interpretation
                                    if oi_change > 0 and price_change > 0:
                                        st.success("🟢 NEW LONGS - Fresh buying, bullish continuation likely")
                                    elif oi_change > 0 and price_change < 0:
                                        st.error("🔴 NEW SHORTS - Fresh selling, bearish continuation likely")
                                    elif oi_change < 0 and price_change > 0:
                                        st.warning("🟡 SHORT COVERING - Rally is weak (shorts closing, not new buying)")
                                    elif oi_change < 0 and price_change < 0:
                                        st.warning("🟡 LONG LIQUIDATION - Dump may be ending (forced selling, not new shorts)")
                                    else:
                                        st.info("⚪ NEUTRAL - No clear signal")
                            
                            # Funding Rate Card
                            with whale_cols[2]:
                                fund_status = "Longs Pay" if funding > 0 else "Shorts Pay" if funding < 0 else "Neutral"
                                fund_color = "#ff4444" if abs(funding) > 0.05 else "#ffcc00" if abs(funding) > 0.01 else "#888"
                                st.markdown(f"""
                                <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                    <div style='color: #888; font-size: 0.85em;'>💰 Funding Rate</div>
                                    <div style='color: {fund_color}; font-size: 1.8em; font-weight: bold;'>{funding:.4f}%</div>
                                    <div style='color: #666; font-size: 0.8em;'>{fund_status}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("ℹ️ What is Funding?"):
                                    st.markdown("""
                                    **Funding Rate** = Fee between longs/shorts every 8h
                                    
                                    💰 **Positive** = Longs pay shorts (bullish sentiment)
                                    💰 **Negative** = Shorts pay longs (bearish sentiment)
                                    
                                    ⚠️ **CONTRARIAN at extremes:**
                                    - Very positive (>0.1%) = Too many longs → Dump coming
                                    - Very negative (<-0.1%) = Too many shorts → Pump coming
                                    
                                    💡 **Edge:** Extreme funding = crowded side gets liquidated
                                    """)
                            
                            # Top Traders Card
                            with whale_cols[3]:
                                whale_color = "#00d4aa" if whale_long > 55 else "#ff4444" if whale_long < 45 else "#888"
                                whale_label = "🟢 LONG" if whale_long > 55 else "🔴 SHORT" if whale_long < 45 else "Balanced"
                                st.markdown(f"""
                                <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                    <div style='color: #888; font-size: 0.85em;'>🐋 Top Traders</div>
                                    <div style='color: {whale_color}; font-size: 1.8em; font-weight: bold;'>{whale_long:.0f}%</div>
                                    <div style='color: #666; font-size: 0.8em;'>{whale_label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("ℹ️ Who are Top Traders?"):
                                    st.markdown("""
                                    **Top Traders** = Binance's most profitable futures traders
                                    
                                    This shows what % of their positions are LONG:
                                    - **60%** means 60% long, 40% short
                                    
                                    🟢 **>55%** = Smart money bullish
                                    🔴 **<45%** = Smart money bearish
                                    
                                    💡 **Edge:** When whales diverge from retail, follow whales!
                                    
                                    ⚠️ **BUT:** Whales being long ≠ "Enter now"
                                    Wait for price confirmation!
                                    """)
                            
                            # Retail Card
                            with whale_cols[4]:
                                retail_color = "#ff4444" if retail_long > 65 else "#00d4aa" if retail_long < 35 else "#888"
                                retail_label = "⚠️ FADE" if retail_long > 65 or retail_long < 35 else "Balanced"
                                st.markdown(f"""
                                <div style='background: #1a1a2e; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #333;'>
                                    <div style='color: #888; font-size: 0.85em;'>🐑 Retail</div>
                                    <div style='color: {retail_color}; font-size: 1.8em; font-weight: bold;'>{retail_long:.0f}%</div>
                                    <div style='color: #666; font-size: 0.8em;'>{retail_label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                with st.expander("ℹ️ Retail = Contrarian"):
                                    st.markdown("""
                                    **Retail Traders** = All regular traders on Binance
                                    
                                    ⚠️ **Retail is often WRONG at extremes!**
                                    
                                    - **>65% Long** = FADE THIS → Often marks TOPS
                                    - **<35% Long** = FADE THIS → Often marks BOTTOMS
                                    
                                    💡 **Best setup:** Whales opposite to retail
                                    = Retail becomes exit liquidity
                                    """)
                            
                            # ═══════════════════════════════════════════════════════
                            # 📈 OI + PRICE INTERPRETATION DETAIL
                            # ═══════════════════════════════════════════════════════
                            
                            oi_interp = whale_data.get('oi_interpretation', {})
                            if oi_interp and oi_interp.get('interpretation'):
                                interp_emoji = oi_interp.get('emoji', '📊')
                                interp_color = "#00d4aa" if interp_emoji == '🟢' else "#ff4444" if interp_emoji == '🔴' else "#ffcc00"
                                
                                with st.expander(f"{interp_emoji} OI + Price Detail: {oi_interp.get('signal', '')}", expanded=False):
                                    st.markdown(f"""
                                    **Interpretation:** {oi_interp.get('interpretation', '')}
                                    
                                    **Action:** {oi_interp.get('action', '')}
                                    
                                    **Signal Strength:** {oi_interp.get('strength', 'N/A')}
                                    """)
                            
                            # Positioning edge detail
                            ls_interp = whale_data.get('ls_interpretation', {})
                            if ls_interp and ls_interp.get('edge') in ['HIGH', 'MEDIUM']:
                                edge_color = "#00d4aa" if 'LONG' in ls_interp.get('action', '') else "#ff4444"
                                with st.expander(f"🎯 Positioning Edge: {ls_interp.get('edge', '')}", expanded=False):
                                    st.markdown(f"""
                                    **Action:** {ls_interp.get('action', '')}
                                    
                                    **Retail Bias:** {ls_interp.get('retail_bias', 'N/A')}
                                    **Whale Bias:** {ls_interp.get('whale_bias', 'N/A')}
                                    """)
                            
                            # ML Placeholder
                            st.markdown(f"""
                            <div style='background: #1a1a2e; border: 1px dashed #444; border-radius: 8px; padding: 15px; margin: 15px 0;'>
                                <span style='color: #666;'>🤖 <strong>Phase 2 Coming:</strong> Historical pattern analysis - 
                                "This scenario: X% win rate, avg +Y% when bullish"</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 🎮 THE WHALE VS RETAIL GAME - Core Education
                            with st.expander("🎮 Learn: The Whale vs Retail Game", expanded=False):
                                st.markdown("""
                                ### 🎮 How Markets Really Work
                                
                                Markets are a **wealth transfer mechanism**. Understanding who's on the other side of your trade is essential.
                                
                                ---
                                
                                #### The Four Phases:
                                
                                **📉 1. ACCUMULATION (Bottoms)**
                                | Player | Behavior |
                                |--------|----------|
                                | 🐑 Retail | Panic selling, "crypto is dead" |
                                | 🐋 Whales | Quietly buying, absorbing sell pressure |
                                | 🎯 You | Watch for whale % rising while price falls |
                                
                                **📈 2. MARKUP (Uptrends)**
                                | Player | Behavior |
                                |--------|----------|
                                | 🐑 Retail | Starting to notice, FOMO building |
                                | 🐋 Whales | Holding, letting price rise |
                                | 🎯 You | Follow trend, buy dips |
                                
                                **🔝 3. DISTRIBUTION (Tops)**
                                | Player | Behavior |
                                |--------|----------|
                                | 🐑 Retail | "To the moon!", max FOMO, all-in |
                                | 🐋 Whales | Quietly selling to retail buyers |
                                | 🎯 You | Watch for whale % falling while retail buys |
                                
                                **📉 4. MARKDOWN (Downtrends)**
                                | Player | Behavior |
                                |--------|----------|
                                | 🐑 Retail | Holding bags → panic sell at lows |
                                | 🐋 Whales | Already sold, waiting to buy again |
                                | 🎯 You | Stay out or short, wait for accumulation |
                                
                                ---
                                
                                ### 🎯 THE CORE TRUTH
                                
                                Money flows from **impatient** to **patient**.
                                Money flows from **emotional** to **rational**.
                                Money flows from **RETAIL** to **WHALES**.
                                
                                **Your job:** Don't be exit liquidity.
                                
                                ---
                                
                                ### 📊 Current Situation Explained
                                
                                **Technical says SELL** = Price is falling, bearish patterns
                                **Whales are LONG** = Smart money is accumulating
                                
                                This is **PHASE 1: ACCUMULATION** in progress!
                                
                                Retail is panic selling (that's you if you sell now).
                                Whales are buying those panic sells.
                                
                                **The trap:** Selling now = selling to whales at the bottom.
                                **The play:** Wait for confirmation, then go LONG with whales.
                                
                                ---
                                
                                ### 💡 Golden Rules
                                
                                1. **When retail is euphoric** → Be cautious (distribution)
                                2. **When retail is panicking** → Be ready to buy (accumulation)
                                3. **Best trades feel uncomfortable** - Buying fear, selling greed
                                4. **Follow whales**, fade retail extremes
                                """)
                            
                            # All signals in expander
                            if whale_data.get('signals'):
                                with st.expander("📋 View All Institutional Signals", expanded=False):
                                    for signal in whale_data['signals']:
                                        st.markdown(f"• {signal}")
                        except Exception as e:
                            st.warning(f"🐋 Error displaying whale data: {str(e)[:60]}")
                    else:
                        st.info("🐋 Whale data unavailable - Binance Futures API may be restricted. Data will appear when API is accessible.")
                else:
                    # 📊 STOCK/ETF INSTITUTIONAL DATA (Quiver Quant)
                    # Use already-fetched data from earlier (stored in institutional dict)
                    stock_data = institutional.get('stock_inst_data') if institutional else None
                    
                    if stock_data and stock_data.get('available'):
                        # Display using NATIVE Streamlit components (not raw HTML)
                        score = stock_data.get('combined_score', 50)
                        sentiment = stock_data.get('overall_sentiment', 'NEUTRAL')
                        sent_color = stock_data.get('sentiment_color', '#888')
                        
                        # Header
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #0d1b2a, #1b263b); border: 2px solid {sent_color}; 
                                    border-radius: 12px; padding: 20px; margin: 15px 0; text-align: center;'>
                            <span style='color: {sent_color}; font-size: 1.3em; font-weight: bold;'>
                                📊 STOCK INSTITUTIONAL SCORE: {score}/100 ({sentiment})
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Key Factors
                        factors = stock_data.get('factors', [])
                        if factors:
                            st.markdown("**Key Factors:**")
                            for factor, impact in factors:
                                impact_icon = "🟢" if impact > 0 else "🔴" if impact < 0 else "⚪"
                                impact_str = f"+{impact}" if impact > 0 else str(impact)
                                st.markdown(f"- {impact_icon} {factor}: **{impact_str}**")
                        else:
                            st.markdown("*No significant factors detected*")
                        
                        # Data cards in columns
                        data_cols = st.columns(3)
                        
                        # Congress
                        congress = stock_data.get('congress', {})
                        with data_cols[0]:
                            if congress and 'error' not in congress:
                                c_sent = congress.get('sentiment', 'NEUTRAL')
                                st.metric(
                                    "🏛️ Congress", 
                                    c_sent,
                                    f"{congress.get('buys', 0)} buys / {congress.get('sells', 0)} sells"
                                )
                            else:
                                st.metric("🏛️ Congress", "N/A", "No data")
                        
                        # Insider
                        insider = stock_data.get('insider', {})
                        with data_cols[1]:
                            if insider and 'error' not in insider:
                                i_sent = insider.get('sentiment', 'NEUTRAL')
                                c_suite = insider.get('c_suite_buys', 0)
                                st.metric(
                                    "👔 Insider", 
                                    i_sent,
                                    f"C-Suite buys: {c_suite}"
                                )
                            else:
                                st.metric("👔 Insider", "N/A", "No data")
                        
                        # Short Interest
                        short = stock_data.get('short_interest', {})
                        with data_cols[2]:
                            if short and 'error' not in short:
                                short_pct = short.get('short_percent', 0)
                                squeeze = short.get('squeeze_potential', 'N/A')
                                st.metric(
                                    "🩳 Short Interest", 
                                    f"{short_pct:.1f}%",
                                    f"Squeeze: {squeeze}"
                                )
                            else:
                                st.metric("🩳 Short Interest", "N/A", "No data")
                        
                        # Action hint
                        action_hint = stock_data.get('action_hint', '')
                        if action_hint:
                            st.info(f"💡 **{action_hint}**")
                        
                        # Detailed breakdowns in expanders
                        detail_cols = st.columns(3)
                        
                        # Congress Trading Details
                        if congress and 'error' not in congress:
                            with detail_cols[0]:
                                with st.expander("🏛️ Congress Trading Details"):
                                    st.markdown(f"""
                                    **{congress.get('buys', 0)} Buys** / **{congress.get('sells', 0)} Sells** (60 days)
                                    
                                    {congress.get('interpretation', '')}
                                    """)
                                    
                                    if congress.get('notable_traders_active'):
                                        st.success(f"🌟 Notable traders: {', '.join(congress['notable_traders_active'][:3])}")
                                    
                                    recent = congress.get('recent_trades', [])[:5]
                                    if recent:
                                        st.markdown("**Recent Trades:**")
                                        for t in recent:
                                            action = "🟢 BUY" if "Purchase" in t.get('Transaction', '') else "🔴 SELL"
                                            st.markdown(f"- {action} by {t.get('Representative', 'Unknown')[:20]}")
                        
                        # Insider Trading Details  
                        if insider and 'error' not in insider:
                            with detail_cols[1]:
                                with st.expander("👔 Insider Trading Details"):
                                    st.markdown(f"""
                                    **{insider.get('buys', 0)} Buys** / **{insider.get('sells', 0)} Sells** (60 days)
                                    
                                    **C-Suite Activity:**
                                    - Buys: {insider.get('c_suite_buys', 0)}
                                    - Sells: {insider.get('c_suite_sells', 0)}
                                    
                                    {insider.get('interpretation', '')}
                                    """)
                                    
                                    net = insider.get('net_value', 0)
                                    if net > 0:
                                        st.success(f"💰 Net insider buying: ${net:,.0f}")
                                    elif net < 0:
                                        st.warning(f"💸 Net insider selling: ${abs(net):,.0f}")
                        
                        # Short Interest Details
                        if short and 'error' not in short:
                            with detail_cols[2]:
                                with st.expander("🩳 Short Interest Details"):
                                    st.markdown(f"""
                                    **Short % of Float:** {short.get('short_percent', 0):.1f}%
                                    
                                    **Days to Cover:** {short.get('days_to_cover', 0):.1f}
                                    
                                    **Squeeze Potential:** {short.get('squeeze_potential', 'Unknown')}
                                    
                                    {short.get('interpretation', '')}
                                    """)
                                    
                                    if short.get('squeeze_potential') == 'HIGH':
                                        st.warning("⚠️ HIGH squeeze potential - watch for catalysts!")
                        
                        # Educational section
                        with st.expander("📚 Learn: Stock Institutional Signals"):
                            st.markdown("""
                            ### 🏛️ Congress Trading
                            Congress members often trade ahead of legislation. Notable traders like 
                            Nancy Pelosi have historically outperformed the market.
                            
                            **Signal:** Congress buying = Bullish (they know what's coming)
                            
                            ---
                            
                            ### 👔 Insider Trading
                            CEO/CFO buying their own stock is one of the strongest bullish signals.
                            They know the company better than anyone.
                            
                            **Signal:** C-Suite buying = Very Bullish
                            **Note:** Selling is less meaningful (often just diversification)
                            
                            ---
                            
                            ### 🩳 Short Interest
                            High short interest (>20%) means many are betting against the stock.
                            
                            **Contrarian Signal:** High shorts + positive catalyst = SQUEEZE potential
                            
                            **Days to Cover:** How many days to close all shorts at average volume.
                            Higher = More squeeze potential.
                            """)
                    else:
                        # No stock data available
                        quiver_key = get_setting('quiver_api_key', '')
                        if not quiver_key:
                            st.markdown("""
                            <div style='background: #2a2a3a; border-left: 4px solid #ffcc00; padding: 15px; border-radius: 8px;'>
                                <div style='color: #ffcc00; font-weight: bold;'>📊 Stock Institutional Data Available!</div>
                                <div style='color: #ccc; margin-top: 8px;'>
                                    Get access to Congress trading, Insider trades, and Short interest.
                                </div>
                                <div style='margin-top: 10px;'>
                                    <a href='https://api.quiverquant.com/' target='_blank' style='color: #00d4ff;'>
                                        1. Get API token at api.quiverquant.com ($10/mo) →
                                    </a>
                                </div>
                                <div style='color: #888; margin-top: 5px;'>
                                    2. Paste token in sidebar under "Stock Institutional Data"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("📊 No institutional data available for this symbol.")
                
                # ═══════════════════════════════════════════════════════════
                # DETAILED INSIGHTS (Educational)
                # ═══════════════════════════════════════════════════════════
                st.markdown("---")
                st.markdown("### 📚 Analysis Breakdown")
                st.markdown("<p style='color: #888;'>Understanding WHY - each factor explained</p>", unsafe_allow_html=True)
                
                # Group insights by sentiment
                bullish_insights = [i for i in result.insights if i.sentiment == Sentiment.BULLISH]
                bearish_insights = [i for i in result.insights if i.sentiment == Sentiment.BEARISH]
                neutral_insights = [i for i in result.insights if i.sentiment == Sentiment.NEUTRAL]
                
                insight_tabs = st.tabs([
                    f"🟢 Bullish ({len(bullish_insights)})", 
                    f"🔴 Bearish ({len(bearish_insights)})",
                    f"⚪ Neutral ({len(neutral_insights)})"
                ])
                
                with insight_tabs[0]:
                    if bullish_insights:
                        for insight in bullish_insights:
                            with st.expander(f"🟢 {insight.title} (+{insight.impact} points)", expanded=False):
                                st.markdown(f"""
                                <div style='background: #0a2a1a; padding: 15px; border-radius: 8px;'>
                                    <p style='color: #ddd; line-height: 1.6;'>{insight.explanation}</p>
                                    <p style='color: #666; font-size: 0.85em; margin-top: 10px;'>
                                        Category: {insight.category.title()} | Impact: +{insight.impact}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No bullish factors detected")
                
                with insight_tabs[1]:
                    if bearish_insights:
                        for insight in bearish_insights:
                            with st.expander(f"🔴 {insight.title} ({insight.impact} points)", expanded=False):
                                st.markdown(f"""
                                <div style='background: #2a0a0a; padding: 15px; border-radius: 8px;'>
                                    <p style='color: #ddd; line-height: 1.6;'>{insight.explanation}</p>
                                    <p style='color: #666; font-size: 0.85em; margin-top: 10px;'>
                                        Category: {insight.category.title()} | Impact: {insight.impact}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No bearish factors detected")
                
                with insight_tabs[2]:
                    if neutral_insights:
                        for insight in neutral_insights:
                            with st.expander(f"⚪ {insight.title}", expanded=False):
                                st.markdown(f"""
                                <div style='background: #1a1a2a; padding: 15px; border-radius: 8px;'>
                                    <p style='color: #ddd; line-height: 1.6;'>{insight.explanation}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No neutral observations")
                
                # ═══════════════════════════════════════════════════════════
                # KEY METRICS
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 📊 Key Metrics")
                
                m = result.metrics
                met_cols = st.columns(5)
                
                with met_cols[0]:
                    rsi_val = m.get('rsi', 50)
                    rsi_status = "Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral"
                    st.metric("RSI", f"{rsi_val:.0f}", rsi_status)
                
                with met_cols[1]:
                    cmf_val = m.get('cmf', 0)
                    cmf_status = "Inflow" if cmf_val > 0.05 else "Outflow" if cmf_val < -0.05 else "Neutral"
                    st.metric("CMF", f"{cmf_val:.2f}", cmf_status)
                
                with met_cols[2]:
                    mfi_val = m.get('mfi', 50)
                    mfi_status = "Oversold" if mfi_val < 20 else "Overbought" if mfi_val > 80 else "Neutral"
                    st.metric("MFI", f"{mfi_val:.0f}", mfi_status)
                
                with met_cols[3]:
                    trend = m.get('trend', 'Unknown')
                    st.metric("Trend", trend)
                
                with met_cols[4]:
                    pos = m.get('price_position', 50)
                    st.metric("Range Position", f"{pos:.0f}%")
                
                # ═══════════════════════════════════════════════════════════
                # ADD TO MONITOR
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                
                # Get the signal for accurate trade levels
                signal = SignalGenerator.generate_signal(df, symbol_input, analysis_tf)
                
                # Get current price for fallback
                current_price = df['Close'].iloc[-1]
                atr = df['High'].rolling(14).max().iloc[-1] - df['Low'].rolling(14).min().iloc[-1]
                atr = atr / 14  # Rough ATR estimate
                
                col_add, col_info = st.columns([1, 2])
                
                # Check if already in monitor (from file, not just session)
                all_active = get_active_trades()
                existing_symbols = [t.get('symbol', '') for t in all_active]
                already_added = symbol_input.upper() in [s.upper() for s in existing_symbols]
                
                with col_add:
                    if already_added:
                        st.success(f"✅ {symbol_input} already in Monitor!")
                        st.info("👉 Go to Trade Monitor to view it")
                    else:
                        add_key = f"add_single_{symbol_input}_{analysis_tf}"
                        if st.button(f"➕ Add {symbol_input} to Monitor", type="primary", use_container_width=True, key=add_key):
                            # Use levels from narrative result (same as Action Plan shows)
                            # Priority: result levels > signal levels > default 5%
                            if hasattr(result, 'entry') and result.entry and result.entry > 0:
                                # Use narrative engine levels (matches Action Plan)
                                entry = float(result.entry)
                                stop_loss = float(result.stop_loss) if hasattr(result, 'stop_loss') and result.stop_loss else float(result.entry * 0.95)
                                tp1 = float(result.tp1) if hasattr(result, 'tp1') and result.tp1 else float(result.entry * 1.05)
                                tp2 = float(result.tp2) if hasattr(result, 'tp2') and result.tp2 else float(result.entry * 1.10)
                                tp3 = float(result.tp3) if hasattr(result, 'tp3') and result.tp3 else float(result.entry * 1.15)
                                direction = result.action.value.upper() if result.action.value in ['long', 'buy'] else 'LONG'
                            elif signal:
                                # Fallback to signal levels
                                entry = float(signal.entry)
                                stop_loss = float(signal.stop_loss)
                                tp1 = float(signal.tp1)
                                tp2 = float(signal.tp2)
                                tp3 = float(signal.tp3)
                                direction = signal.direction
                            else:
                                # Last resort: default levels
                                entry = float(current_price)
                                stop_loss = float(current_price * 0.95)
                                tp1 = float(current_price * 1.05)
                                tp2 = float(current_price * 1.10)
                                tp3 = float(current_price * 1.15)
                                direction = 'LONG'
                            
                            trade = {
                                'symbol': symbol_input.upper(),
                                'entry': entry,
                                'stop_loss': stop_loss,
                                'tp1': tp1,
                                'tp2': tp2,
                                'tp3': tp3,
                                'direction': direction,
                                'timeframe': analysis_tf,
                                'score': combined_score,
                                'grade': get_grade_letter(combined_score),
                                'market': analysis_market,
                                'mode': narrative_mode,
                                'action': result.action.value,
                                'status': 'active',
                                'created_at': datetime.now().isoformat()
                            }
                            
                            # Save to file
                            success = add_trade(trade)
                            
                            if success:
                                # Reload from file to update session state
                                st.session_state.active_trades = get_active_trades()
                                st.success(f"✅ Added {symbol_input} to Monitor!")
                                st.info(f"👉 Go to **Trade Monitor** in the sidebar to view your trade")
                                st.balloons()
                            else:
                                # Check why it failed
                                st.warning(f"⚠️ {symbol_input} may already be in Monitor or save failed")
                                st.info("👉 Go to Trade Monitor to check")
                
                with col_info:
                    # Show levels from result (same as Action Plan)
                    if hasattr(result, 'entry') and result.entry and result.entry > 0:
                        display_entry = result.entry
                        display_sl = result.stop_loss if hasattr(result, 'stop_loss') and result.stop_loss else result.entry * 0.95
                        display_tp1 = result.tp1 if hasattr(result, 'tp1') and result.tp1 else result.entry * 1.05
                        display_risk = result.risk_pct if hasattr(result, 'risk_pct') and result.risk_pct else 5.0
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px; border-radius: 8px;'>
                            <span style='color: #888;'>Entry:</span> <strong style='color: #00d4ff;'>{fmt_price(display_entry)}</strong> | 
                            <span style='color: #888;'>SL:</span> <strong style='color: #ff4444;'>{fmt_price(display_sl)}</strong> ({display_risk:.1f}%) | 
                            <span style='color: #888;'>TP1:</span> <strong style='color: #00d4aa;'>{fmt_price(display_tp1)}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    elif signal:
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px; border-radius: 8px;'>
                            <span style='color: #888;'>Entry:</span> <strong style='color: #00d4ff;'>{fmt_price(signal.entry)}</strong> | 
                            <span style='color: #888;'>SL:</span> <strong style='color: #ff4444;'>{fmt_price(signal.stop_loss)}</strong> | 
                            <span style='color: #888;'>TP1:</span> <strong style='color: #00d4aa;'>{fmt_price(signal.tp1)}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px; border-radius: 8px;'>
                            <span style='color: #888;'>Entry:</span> <strong style='color: #00d4ff;'>{fmt_price(current_price)}</strong> | 
                            <span style='color: #888;'>SL:</span> <strong style='color: #ff4444;'>{fmt_price(current_price * 0.95)}</strong> (5%) | 
                            <span style='color: #888;'>TP1:</span> <strong style='color: #00d4aa;'>{fmt_price(current_price * 1.05)}</strong> (5%)
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("⚠️ Using default 5% levels - adjust in Monitor if needed")
                        
            else:
                st.error(f"Could not fetch data for {symbol_input}. Check the symbol and try again.")
    else:
        # No analysis requested yet - show instructions
        st.markdown("""
        <div style='background: #1a1a2e; padding: 30px; border-radius: 12px; text-align: center; margin: 20px 0;'>
            <div style='font-size: 3em; margin-bottom: 15px;'>🔬</div>
            <h3 style='color: #00d4ff; margin-bottom: 10px;'>Ready to Analyze</h3>
            <p style='color: #888;'>1. Select your trading mode in the sidebar</p>
            <p style='color: #888;'>2. Enter a symbol (e.g., BTCUSDT, AAPL, SPY)</p>
            <p style='color: #888;'>3. Click <strong>ANALYZE</strong> to get started</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>🧠 InvestorIQ | Smart Money Analysis | Not Financial Advice</p>", unsafe_allow_html=True)
