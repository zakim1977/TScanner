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
    get_all_binance_pairs, get_current_price,
    # NEW: Stock/ETF support
    fetch_stock_data, get_stock_price, fetch_universal, 
    estimate_time_to_target as estimate_time
)

from core.signal_generator import SignalGenerator, TradeSignal, analyze_trend, analyze_momentum
from core.smc_detector import detect_smc, analyze_market_structure
from core.money_flow import calculate_money_flow, detect_whale_activity, detect_pre_breakout
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

from utils.formatters import (
    fmt_price, calc_roi, format_percentage, format_number,
    get_grade_emoji, get_grade_letter, get_quality_badge
)

from utils.trade_storage import (
    load_trade_history, save_trade_history, add_trade,
    get_active_trades, get_closed_trades, calculate_statistics,
    update_trade, close_trade, get_trade_by_symbol, sync_active_trades,
    delete_trade_by_symbol, export_trades_json, import_trades_json, get_download_link
)

from utils.charts import create_trade_setup_chart, create_performance_chart

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="InvestorIQ PRO",
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
    bullish_ob = ob_data.get('bullish_ob')
    
    if at_bullish_ob or bullish_ob:
        progression.append({
            'step': 'C → B',
            'name': 'Order Block Hit',
            'detail': f"Price at demand zone",
            'achieved': True
        })
        breakdown['bullish_factors'].append("✅ At Order Block (institutional demand zone)")
        
        # Store OB data
        if bullish_ob:
            breakdown['order_block'] = {
                'type': 'BULLISH DEMAND',
                'top': bullish_ob.get('top', 0),
                'bottom': bullish_ob.get('bottom', 0),
                'strength': bullish_ob.get('strength', 0)
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
        'num_assets': 50,
        'min_score': 40,
        # Filters - APPLY EVERYWHERE
        'long_only': True,
        'min_grade': 'C',
        'auto_add_aplus': True,
        'max_sl_pct': 5.0,  # Maximum stop loss % - reject if SL > this
        'min_rr_tp1': 1.0,  # Minimum R:R at TP1 - mode-specific
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
        ["🔍 Scanner", "📈 Trade Monitor", "📊 Performance", "🔬 Single Analysis"],
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
        # STEP 4: OTHER SETTINGS
        # ═══════════════════════════════════════════════════════════════════════
        
        num_assets = st.slider("Assets to Scan", 10, 200, st.session_state.settings['num_assets'])
        min_score = st.slider("Min Score", 0, 100, st.session_state.settings['min_score'])
        
        st.markdown("---")
        st.markdown("#### 🎯 Filters")
        
        long_only = st.checkbox("📈 LONG Only", value=st.session_state.settings['long_only'])
        grade_filter = st.selectbox("Min Grade", ['D', 'C', 'B', 'A', 'A+'], 
                                    index=['D', 'C', 'B', 'A', 'A+'].index(st.session_state.settings['min_grade']))
        
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
        
        auto_add = st.checkbox("Auto-add A+ setups", value=st.session_state.settings['auto_add_aplus'])
        
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
            'min_grade': grade_filter,
            'auto_add_aplus': auto_add,
            'max_sl_pct': max_sl_pct,
            'min_rr_tp1': mode_config['min_rr'],  # From mode config
            'mode_config': mode_config,  # Store full config for reference
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
    filters_active.append(f"Min Grade: {get_setting('min_grade', 'C')}")
    filters_active.append(f"Min Score: {get_setting('min_score', 40)}")
    filters_active.append(f"Max SL: {get_setting('max_sl_pct', 5.0)}%")
    filters_active.append(f"Min TP1 R:R: {get_setting('min_rr_tp1', 1.0)}:1")
    
    st.markdown(f"""
    <div style='background: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 8px 12px; margin: 5px 0;'>
        <span style='color: #888;'>🎯 Active Filters:</span> 
        <span style='color: #00d4ff;'>{' | '.join(filters_active)}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Scan button
    if st.button("🚀 SCAN FOR SIGNALS", type="primary", use_container_width=True):
        
        progress = st.progress(0)
        status = st.empty()
        
        results = []
        scanned = 0
        
        # Get pairs based on market type
        if "Crypto" in market_type:
            status.markdown("📡 Fetching pairs from Binance...")
            pairs = fetch_all_binance_usdt_pairs(num_assets)
            if not pairs:
                pairs = TOP_CRYPTO_PAIRS[:num_assets]
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
        
        # Loop through pairs AND timeframes
        for i, symbol in enumerate(pairs):
            for tf in timeframes_to_scan:
                scanned += 1
                scan_count += 1
                progress.progress(scan_count / total_scans)
                status.markdown(f"🔍 **[{scan_count}/{total_scans}]** <span style='color: #00d4ff;'>{symbol}</span> on <span style='color: #ffcc00;'>{mode_icon} {mode_name} {tf}</span>", unsafe_allow_html=True)
                
                try:
                    df = fetch_func(symbol, tf, 200)
                    
                    if df is not None and len(df) >= 50:
                        signal = SignalGenerator.generate_signal(df, symbol, tf)
                        
                        if signal:
                            # Skip shorts if filter enabled
                            if get_setting("long_only") and signal.direction != 'LONG':
                                continue
                            
                            mf = calculate_money_flow(df)
                            smc = detect_smc(df)
                            pre_break = detect_pre_breakout(df)
                            whale = detect_whale_activity(df)
                            sessions = analyze_trading_sessions(df)
                            
                            # 🦈 Institutional Activity Detection (PREDICTIVE)
                            institutional = detect_institutional_activity(df, mf)
                            stealth_accum = institutional.get('stealth_accum', {})
                            stealth_dist = institutional.get('stealth_dist', {})
                            
                            # Calculate score - now with institutional bonus
                            score = signal.confidence
                            if mf['is_accumulating']: score += 15
                            if smc['order_blocks'].get('at_bullish_ob'): score += 20
                            if smc['fvg'].get('at_bullish_fvg'): score += 15
                            if pre_break['probability'] >= 50: score += 10
                            if whale['whale_detected']: score += 15
                            
                            # 🦈 PREDICTIVE BONUS - Stealth accumulation gets high priority
                            if stealth_accum.get('detected') and signal.direction == 'LONG':
                                score += 20  # Big bonus for predicting move before it happens
                            if institutional.get('net_score', 0) >= 30:
                                score += 10  # Strong institutional buying
                            
                            score = min(100, score)
                            
                            # Apply filters from CENTRALIZED SETTINGS
                            if score < get_setting('min_score', 40):
                                continue
                            
                            # ⚠️ MAX STOP LOSS FILTER - reject trades with SL too far
                            max_sl = get_setting('max_sl_pct', 5.0)
                            if signal.risk_pct > max_sl:
                                continue  # SL too far - skip this trade
                            
                            # ⚠️ MIN TP1 R:R FILTER - mode-specific requirement
                            min_rr_tp1 = get_setting('min_rr_tp1', 1.0)
                            signal_rr_tp1 = getattr(signal, 'rr_tp1', 0)
                            if signal_rr_tp1 < min_rr_tp1:
                                continue  # TP1 R:R too low for this mode
                            
                            grade = get_pro_grade(score)
                            grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
                            if grade_order.get(grade['grade'], 0) < grade_order.get(get_setting('min_grade', 'C'), 0):
                                continue
                            
                            # Generate full breakdown
                            breakdown = generate_signal_breakdown(df, signal, mf, smc, whale, pre_break, score)
                            
                            # Add result WITH mode and timeframe info
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
                                # 🆕 Mode and Timeframe info
                                'mode_name': mode_name,
                                'mode_icon': mode_icon,
                                'timeframe': tf,
                                'display_label': f"{mode_icon} {mode_name} {tf}"  # e.g., "📊 Day Trade 15m"
                            })
                            
                except Exception as e:
                    continue
        
        progress.progress(1.0)
        status.success(f"✅ Scan complete! Found **{len(results)}** signals across {len(timeframes_to_scan)} timeframe(s)")
        
        results.sort(key=lambda x: x['score'], reverse=True)
        st.session_state.scan_results = results
        
        # Auto-add A+ setups from CENTRALIZED SETTINGS
        if get_setting('auto_add_aplus', True):
            for r in results[:3]:
                if r['breakdown']['grade'] == 'A+':
                    trade = {
                        'symbol': r['signal'].symbol,
                        'entry': r['signal'].entry,
                        'stop_loss': r['signal'].stop_loss,
                        'tp1': r['signal'].tp1,
                        'tp2': r['signal'].tp2,
                        'tp3': r['signal'].tp3,
                        'direction': r['signal'].direction,
                        'score': r['score'],
                        'grade': r['breakdown']['grade'],
                        'timeframe': r.get('timeframe', '15m'),
                        'mode_name': r.get('mode_name', 'Day Trade'),
                        'status': 'active',  # Required for persistence
                        'created_at': datetime.now().isoformat()
                    }
                    if not any(t['symbol'] == trade['symbol'] for t in st.session_state.active_trades):
                        # Save to file and reload
                        add_trade(trade)
                        st.session_state.active_trades = get_active_trades()
    
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
        for r in results[:20]:  # Check top 20 results
            try:
                predictions = find_approaching_setups(r['df'], r['signal'].symbol)
                for p in predictions:
                    if p.stage == SignalStage.APPROACHING and p.distance_pct > 1.5:
                        # Apply LONG Only filter to approaching setups too
                        if get_setting("long_only") and p.direction != 'LONG':
                            continue
                        p.current_score = r['score']  # Add context
                        approaching_setups.append(p)
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
        
        st.markdown(f"## 🎯 {len(results)} Active Signals")
        st.markdown("<p style='color: #888;'>Price is AT these levels - entry available now</p>", unsafe_allow_html=True)
        
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
            
            with st.expander(f"{bd['grade_emoji']} **{sig.symbol}** | {display_label} | Grade {bd['grade']} | Score: {bd['score']}", expanded=False):
                
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
                                if st.button(f"➕ Add to Monitor", key=f"add_{sig.symbol}", type="primary" if status == "READY" else "secondary"):
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
                                    if st.button(f"👁️ Add to Watchlist", key=f"watch_{sig.symbol}"):
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
                            if st.button(f"➕ Add {sig.symbol} to Monitor", key=f"add_{sig.symbol}", type="primary"):
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
                        if st.button(f"➕ Add {sig.symbol} to Monitor", key=f"add_{sig.symbol}", type="primary"):
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
    
    mgmt_col1, mgmt_col2, mgmt_col3, mgmt_col4 = st.columns([1, 1, 1, 2])
    
    with mgmt_col1:
        if st.button("🔄 Reload", help="Force reload trades from saved file"):
            st.session_state.active_trades = get_active_trades()
            st.success(f"Reloaded {len(st.session_state.active_trades)} trades")
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
        
        # Collect alerts from all trades
        for trade in st.session_state.active_trades:
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
        # ═══════════════════════════════════════════════════════════════════
        
        for i, trade in enumerate(st.session_state.active_trades):
            price = get_current_price(trade['symbol'])
            
            if price > 0:
                entry = trade['entry']
                pnl = ((price - entry) / entry) * 100 if trade['direction'] == 'LONG' else ((entry - price) / entry) * 100
                
                # TP/SL status
                tp1_hit = price >= trade['tp1'] if trade['direction'] == 'LONG' else price <= trade['tp1']
                tp2_hit = price >= trade['tp2'] if trade['direction'] == 'LONG' else price <= trade['tp2']
                tp3_hit = price >= trade['tp3'] if trade['direction'] == 'LONG' else price <= trade['tp3']
                sl_hit = price <= trade['stop_loss'] if trade['direction'] == 'LONG' else price >= trade['stop_loss']
                
                # Distance calculations
                dist_to_tp1 = ((trade['tp1'] - price) / price * 100) if trade['direction'] == 'LONG' else ((price - trade['tp1']) / price * 100)
                dist_to_sl = ((price - trade['stop_loss']) / price * 100) if trade['direction'] == 'LONG' else ((trade['stop_loss'] - price) / price * 100)
                
                # Status
                if sl_hit:
                    status = "🔴 STOPPED OUT"
                    emoji = "🔴"
                    status_color = "#ff4444"
                elif tp3_hit:
                    status = "🏆 TP3 HIT - FULL TARGET!"
                    emoji = "🏆"
                    status_color = "#00ff88"
                elif tp2_hit:
                    status = "🎯 TP2 HIT - Take Profit!"
                    emoji = "🎯"
                    status_color = "#00d4aa"
                elif tp1_hit:
                    status = "✅ TP1 HIT - Partial Profit"
                    emoji = "✅"
                    status_color = "#00d4aa"
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
                        
                        # TP1
                        tp1_icon = "✅" if tp1_hit else "⏳"
                        tp1_dist = "" if tp1_hit else f" - {dist_to_tp1:.1f}% away"
                        st.markdown(f"""
                        <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin-bottom: 5px;'>
                            {tp1_icon} <strong>TP1:</strong> {fmt_price(trade['tp1'])} 
                            <span style='color: #00d4aa;'>(+{tp1_roi:.1f}%)</span>
                            <span style='color: #888;'>{tp1_dist}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # TP2
                        tp2_icon = "✅" if tp2_hit else "⏳"
                        st.markdown(f"""
                        <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px; margin-bottom: 5px;'>
                            {tp2_icon} <strong>TP2:</strong> {fmt_price(trade['tp2'])} 
                            <span style='color: #00d4aa;'>(+{tp2_roi:.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # TP3
                        tp3_icon = "✅" if tp3_hit else "⏳"
                        st.markdown(f"""
                        <div style='background: #1a2a1a; padding: 8px 12px; border-radius: 6px;'>
                            {tp3_icon} <strong>TP3:</strong> {fmt_price(trade['tp3'])} 
                            <span style='color: #00d4aa;'>(+{tp3_roi:.1f}%)</span>
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
                    
                    # Generate recommendation based on current state
                    if sl_hit:
                        st.error("**Trade closed.** Stop loss was hit. Review what went wrong and move on.")
                    elif tp3_hit:
                        st.success("**🏆 WINNER!** Full target reached. Close remaining position and celebrate!")
                    elif tp2_hit:
                        st.success("**Take Profit:** TP2 hit! Close 50-75% of remaining position. Trail stop on rest.")
                    elif tp1_hit:
                        st.success("**Partial Profit:** TP1 hit! Take 33% profit. Move stop to breakeven on rest.")
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
                    
                    # Action buttons
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                            st.session_state.active_trades.pop(i)
                            sync_active_trades(st.session_state.active_trades)
                            st.rerun()
                    
                    with btn_col2:
                        if tp1_hit and not tp2_hit:
                            if st.button(f"✅ Mark TP1 Taken", key=f"tp1_{i}"):
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
                        if st.button(button_label, key=f"analyze_{i}"):
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
                                {'🎯 ' + f'{dist_to_tp1_pct:.1f}% to TP1' if dist_to_tp1_pct > 0 else '✅ TP1 Hit!'} | 
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
                                        # Normal narrative display (not in danger)
                                        action_style = narrative.style
                                        st.markdown(f"""
                                        <div style='background: {action_style['bg']}; border-left: 4px solid {action_style['color']}; 
                                                    padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                                <div>
                                                    <span style='font-size: 1.5em;'>{action_style['emoji']}</span>
                                                    <span style='color: {action_style['color']}; font-size: 1.3em; font-weight: bold; margin-left: 10px;'>
                                                        {narrative.action.value}
                                                    </span>
                                                </div>
                                                <div style='text-align: right;'>
                                                    <span style='color: #888; font-size: 0.9em;'>Confidence</span><br>
                                                    <span style='color: {action_style['color']}; font-size: 1.2em; font-weight: bold;'>{narrative.confidence}%</span>
                                                </div>
                                            </div>
                                            <div style='color: #ccc; margin-top: 10px; font-size: 1.1em;'>{narrative.headline}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Summary Box
                                        st.markdown(f"""
                                        <div style='background: #1a1a2e; border-radius: 8px; padding: 15px; margin-bottom: 15px;'>
                                            <div style='color: #00d4ff; font-weight: bold; margin-bottom: 8px;'>📖 What's Happening</div>
                                            <div style='color: #ddd; line-height: 1.6;'>{narrative.summary}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
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
                                        st.write(f"Status: {'🟢 Inflow' if mf_analysis['is_accumulating'] else '🔴 Outflow' if mf_analysis['is_distributing'] else '⚪ Neutral'}")
                                    
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
    st.markdown("## 📊 Performance Dashboard")
    
    stats = calculate_statistics()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", stats['total_trades'])
    c2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    c3.metric("Total P&L", f"{stats['total_pnl']:+.1f}%")
    c4.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
    
    closed = get_closed_trades()
    if closed:
        fig = create_performance_chart(closed)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Complete some trades to see performance!")

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
                # HEADER WITH RECOMMENDATION
                # ═══════════════════════════════════════════════════════════
                
                style = result.style
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {style['bg']} 0%, #16213e 100%); 
                            padding: 25px; border-radius: 12px; margin-bottom: 20px;
                            border-left: 5px solid {style['color']};'>
                    <h2 style='margin: 0; color: {style['color']}; font-size: 1.8em;'>
                        {result.headline}
                    </h2>
                    <p style='color: #ccc; margin: 10px 0 0 0; font-size: 1.1em;'>
                        {result.summary}
                    </p>
                    <p style='color: #888; margin: 10px 0 0 0;'>
                        Confidence: {result.confidence}% | Bullish Factors: {result.bullish_score} | Bearish Factors: {result.bearish_score}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # ═══════════════════════════════════════════════════════════
                # CHART AND LEVELS
                # ═══════════════════════════════════════════════════════════
                
                col_chart, col_levels = st.columns([2, 1])
                
                with col_chart:
                    # Generate signal for chart
                    signal = SignalGenerator.generate_signal(df, symbol_input, analysis_tf)
                    if signal:
                        fig = create_trade_setup_chart(df, signal, result.confidence)
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
                        st.markdown("### 🎯 Trade Setup")
                        
                        st.markdown(f"""
                        <div style='background: #1a2a3a; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                            <strong style='color: #00d4ff;'>Entry:</strong> {fmt_price(result.entry)}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style='background: #3a1a1a; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                            <strong style='color: #ff4444;'>Stop Loss:</strong> {fmt_price(result.stop_loss)}<br>
                            <span style='color: #888;'>Risk: {result.risk_pct:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("**Take Profit Targets:**")
                        st.write(f"✅ TP1: {fmt_price(result.tp1)} (+{((result.tp1 - result.entry) / result.entry * 100):.1f}%)")
                        st.write(f"✅ TP2: {fmt_price(result.tp2)} (+{((result.tp2 - result.entry) / result.entry * 100):.1f}%)")
                        st.write(f"✅ TP3: {fmt_price(result.tp3)} (+{((result.tp3 - result.entry) / result.entry * 100):.1f}%)")
                        
                        if result.rr_ratio > 0:
                            st.markdown(f"**Risk/Reward:** 1:{result.rr_ratio:.1f}")
                    
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
                # ACTION PLAN
                # ═══════════════════════════════════════════════════════════
                
                st.markdown("---")
                st.markdown("### 📋 Action Plan")
                st.markdown("<p style='color: #888;'>Step-by-step guide based on current conditions</p>", unsafe_allow_html=True)
                
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    for i, step in enumerate(result.action_steps[:len(result.action_steps)//2 + 1], 1):
                        st.markdown(f"""
                        <div style='background: #1a1a2e; padding: 10px 15px; margin: 8px 0; border-radius: 8px;
                                    border-left: 3px solid {style['color']};'>
                            {step}
                        </div>
                        """, unsafe_allow_html=True)
                
                with action_col2:
                    for step in result.action_steps[len(result.action_steps)//2 + 1:]:
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
                    try:
                        whale_data = get_whale_analysis(symbol_input)
                        
                        # Check if we got real data
                        has_real_data = (
                            whale_data.get('signals') or 
                            whale_data.get('open_interest', {}).get('change_24h', 0) != 0 or
                            whale_data.get('funding', {}).get('rate', 0) != 0 or
                            whale_data.get('top_trader_ls', {}).get('long_pct', 50) != 50
                        )
                        
                        if has_real_data:
                            # Import education
                            from core.education import (
                                get_unified_verdict, INSTITUTIONAL_EDUCATION, 
                                SCENARIO_EDUCATION, identify_current_scenario
                            )
                            
                            # Extract all data first
                            oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
                            price_change = whale_data.get('price_change_24h', 0)
                            funding = whale_data.get('funding', {}).get('rate_pct', 0)
                            funding_raw = whale_data.get('funding', {}).get('rate', 0)
                            whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
                            retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
                            # Note: Using unified scenario system, not old verdict
                            
                            # Get unified verdict FIRST
                            tech_action_str = result.action.value if hasattr(result.action, 'value') else str(result.action) if result else 'WAIT'
                            unified = get_unified_verdict(
                                oi_change, price_change, whale_long, retail_long, funding_raw, tech_action_str
                            )
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
                        else:
                            st.info("🐋 Whale data unavailable - Binance Futures API may be restricted. Data will appear when API is accessible.")
                    
                    except Exception as e:
                        st.warning(f"🐋 Could not fetch institutional data: {str(e)[:60]}")
                else:
                    # For Stocks/ETFs - show different message
                    st.info("📊 Stock/ETF institutional data: Coming soon! Will include insider trading, short interest, and options flow.")
                
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
                                'score': int(result.confidence),
                                'grade': get_grade_letter(result.confidence),
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
