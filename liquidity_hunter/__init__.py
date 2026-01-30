"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         LIQUIDITY HUNTER MODULE                                ║
║                                                                                ║
║  Data-driven liquidity sweep detection with ML quality prediction              ║
║  Follow the whale footprints + ML to filter high-quality sweeps                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from .liquidity_hunter import (
    normalize_columns,
    find_liquidity_levels,
    detect_sweep,
    check_approaching_liquidity,
    get_binance_liquidations,
    estimate_liquidation_zones,
    generate_liquidity_trade_plan,
    scan_for_liquidity_setups,
    full_liquidity_analysis,
    # NEW: Enhanced detection and entry quality
    detect_round_numbers,
    calculate_level_strength_score,
    calculate_entry_quality,
    filter_scanner_results_by_quality,
    get_fresh_only_results
)

# Core UI imports (always available)
from .liquidity_hunter_ui import (
    render_liquidity_header,
    render_liquidity_map,
    render_sweep_status,
    render_approaching_levels,
    render_trade_plan,
    render_whale_positioning,
    render_scanner_results,
    render_liquidity_education,
    render_scanner_controls,
    filter_scanner_results,
    render_ml_training_panel,
    render_ml_prediction_badge,
    # NEW: Entry quality UI components
    render_entry_quality_badge,
    render_entry_quality_breakdown,
    render_scanner_quality_summary,
    # ETF Money Flow UI
    render_etf_flow_badge,
    render_etf_flow_panel
)

# NEW: Improved sequence visualization (optional - may not exist in older versions)
try:
    from .liquidity_hunter_ui import (
        render_liquidity_sequence,
        render_next_targets_improved
    )
except ImportError:
    # Fallback stub functions
    def render_liquidity_sequence(analysis: dict):
        """Stub - update liquidity_hunter_ui.py to get this feature"""
        import streamlit as st
        st.info("Update liquidity_hunter_ui.py to enable sequence visualization")
    
    def render_next_targets_improved(analysis: dict):
        """Stub - update liquidity_hunter_ui.py to get this feature"""
        import streamlit as st
        st.info("Update liquidity_hunter_ui.py to enable improved targets")

# Liquidity Sequence Visualization (pure Streamlit - optional)
try:
    from .liquidity_sequence import (
        render_full_liquidity_sequence,
        render_sequence_diagram_st,
        render_actionable_summary_st,
        render_expected_sequence_st,
        analyze_liquidity_sequence
    )
except ImportError:
    # Fallback - these features just won't be available
    render_full_liquidity_sequence = None
    analyze_liquidity_sequence = None

# ML module (optional)
try:
    from .liquidity_hunter_ml import (
        predict_sweep_quality,
        store_sweep_trade,
        get_training_stats,
        get_predictor,
        get_model_status,
        SweepPredictor,
        generate_training_from_history,
        bulk_generate_training_data,
        # Stock functions
        get_stock_institutional_score,
        generate_training_from_history_stock,
        bulk_generate_training_data_stock,
        # ETF functions
        generate_training_from_history_etf,
        bulk_generate_training_data_etf,
        init_db
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Quality Model - ETF/Stock variants
try:
    from .quality_model import (
        train_quality_model_etf,
        train_quality_model_stock,
        get_quality_prediction_etf,
        get_quality_prediction_stock,
        get_quality_model_etf_status,
        get_quality_model_stock_status
    )
except ImportError:
    pass

# Price Action Analyzer (SMC-based prediction)
try:
    from .price_action_analyzer import (
        analyze_sweep_reaction,
        analyze_sweep_with_price_action,
        get_price_action_summary
    )
    PRICE_ACTION_AVAILABLE = True
except ImportError:
    PRICE_ACTION_AVAILABLE = False

# Pure Price Action Model (no sweep dependency)
try:
    from .price_action_model import (
        train_price_action_model,
        get_price_action_model,
        get_price_action_prediction,
        get_pa_model_status,
        PriceActionModel,
        extract_price_action_features,
        generate_pa_samples
    )
    PA_MODEL_AVAILABLE = True
except ImportError:
    PA_MODEL_AVAILABLE = False

__all__ = [
    # Core functions
    'normalize_columns',
    'find_liquidity_levels',
    'detect_sweep',
    'check_approaching_liquidity',
    'get_binance_liquidations',
    'estimate_liquidation_zones',
    'generate_liquidity_trade_plan',
    'scan_for_liquidity_setups',
    'full_liquidity_analysis',
    # NEW: Enhanced detection and entry quality
    'detect_round_numbers',
    'calculate_level_strength_score',
    'calculate_entry_quality',
    'filter_scanner_results_by_quality',
    'get_fresh_only_results',
    # UI functions
    'render_liquidity_header',
    'render_liquidity_map',
    'render_sweep_status',
    'render_approaching_levels',
    'render_trade_plan',
    'render_whale_positioning',
    'render_scanner_results',
    'render_liquidity_education',
    # NEW: Entry quality UI components
    'render_entry_quality_badge',
    'render_entry_quality_breakdown',
    'render_scanner_quality_summary',
    # Improved sequence visualization
    'render_liquidity_sequence',
    'render_next_targets_improved',
    # Price Action Analyzer
    'analyze_sweep_reaction',
    'analyze_sweep_with_price_action',
    'get_price_action_summary'
]
