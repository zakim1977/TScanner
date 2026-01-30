"""
Utility modules for Crypto Scanner Pro
"""

from .formatters import fmt_price, calc_roi, format_percentage, format_number
from .trade_storage import (
    load_trade_history, save_trade_history, add_trade, update_trade,
    close_trade, get_active_trades, get_closed_trades, calculate_statistics,
    get_trade_by_symbol
)
from .charts import create_trade_setup_chart, create_performance_chart

# 🏦 Institutional UI Components (NEW!)
from .institutional_ui import (
    render_institutional_verdict, render_metric_card, render_key_signals,
    render_full_institutional_analysis, render_compact_institutional,
    render_educational_overview, render_institutional_integration_note,
    get_score_adjustment_display
)
