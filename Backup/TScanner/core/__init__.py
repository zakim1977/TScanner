"""
Core modules for InvestorIQ
"""

from .data_fetcher import (
    fetch_ohlcv_binance, fetch_ohlcv_smart, 
    fetch_binance_klines, fetch_binance_price, fetch_all_binance_usdt_pairs,
    fetch_binance_futures_pairs, get_default_futures_pairs,
    get_all_binance_pairs, get_current_price, test_binance_connection,
    # Stock/ETF support
    fetch_stock_data, get_stock_price, get_popular_etfs, get_popular_stocks,
    fetch_universal, estimate_time_to_target
)
from .indicators import calculate_rsi, calculate_ema, calculate_atr, calculate_macd, calculate_bbands
from .signal_generator import SignalGenerator, SignalGeneratorV2, TradeSetup, TradeSignal, analyze_trend, analyze_momentum
from .smc_detector import detect_smc, detect_order_blocks, detect_fvg, detect_liquidity_sweep, analyze_market_structure
from .money_flow import calculate_money_flow, detect_whale_activity, detect_pre_breakout
from .money_flow_context import get_money_flow_context, get_smart_action, get_money_flow_education, MoneyFlowContext, MONEY_FLOW_EDUCATION, get_multi_timeframe_tip, MultiTimeframeTip, get_higher_timeframe, TIMEFRAME_HIERARCHY
from .level_calculator import calculate_smart_levels, get_all_levels, TradeLevels

# Master Narrative Engine
from .narrative_engine import MasterNarrative, analyze as narrative_analyze, AnalysisResult, Action, Sentiment

# Alert & Prediction System
from .alert_system import (
    PredictiveAnalyzer, TradeAlertMonitor,
    find_approaching_setups, check_trade_alerts,
    format_alert_html, format_predictive_signal_html,
    Alert, AlertType, PredictiveSignal, SignalStage,
    # Institutional Activity Detection
    detect_stealth_accumulation, detect_stealth_distribution,
    detect_institutional_activity, format_institutional_activity_html
)

# 🐋 Whale & Institutional Data (NEW!)
from .whale_institutional import (
    get_institutional_analysis as get_whale_analysis, get_whale_summary, format_institutional_html,
    interpret_oi_price, interpret_funding, interpret_long_short,
    WhaleSignal, InstitutionalData
)

# 🏦 Unified Institutional Engine (NEW - replaces scattered whale code)
from .institutional_engine import (
    get_institutional_analysis, InstitutionalAnalysis, InstitutionalBias,
    MetricReading, EDUCATION as INSTITUTIONAL_EDUCATION,
    analyze_crypto_institutional, analyze_stock_institutional,
    integrate_institutional_into_signal, should_override_signal,
    render_metric_with_education, get_education_for_metric, get_premium_providers
)

# 📊 Institutional Scoring (Combined Tech + Whale)
from .institutional_scoring import (
    analyze_institutional_data, calculate_combined_score,
    create_watchlist_item, check_watchlist_triggers,
    InstitutionalScore, WatchlistItem, TradeStatus, ScenarioType,
    get_scenario_display, SCENARIO_DETAILS
)

# 🎯 NEW: Professional Rules Engine (Configuration-Driven)
from .rules_engine import (
    InstitutionalRulesEngine, analyze as analyze_institutional,
    MarketData, AnalysisResult as RulesAnalysisResult,
    Direction, Status, Confidence, get_engine, DEFAULT_CONFIG
)
