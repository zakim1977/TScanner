"""
Trading Card Module - Actionable Trade Display
==============================================
Shows simplified, actionable trade information
Discord-bot style but with OUR structural levels

Integrates with existing InvestorIQ system
"""

try:
    import streamlit as st
except ImportError:
    st = None  # For testing without streamlit

from typing import Optional, Dict

try:
    from core.trade_manager import SmartTradeLevels, EnhancedTrade
except ImportError:
    # Fallback for testing
    SmartTradeLevels = None
    EnhancedTrade = None


def format_price(price: float) -> str:
    """Format price appropriately"""
    if price is None or price == 0:
        return "$0.00"
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}".rstrip('0').rstrip('.')
    else:
        return f"${price:.6f}"


def get_grade_color(grade: str) -> str:
    """Get color for grade"""
    colors = {
        'A+': '#00ff88',  # Bright green
        'A': '#00d4aa',   # Green
        'B+': '#ffcc00',  # Yellow
        'B': '#ff9500',   # Orange
        'C': '#ff4444',   # Red
        'D': '#888888',   # Gray
    }
    return colors.get(grade, '#888888')


def get_rr_color(rr: float) -> str:
    """Get color for R:R ratio"""
    if rr >= 2.0:
        return '#00ff88'
    elif rr >= 1.5:
        return '#00d4aa'
    elif rr >= 1.0:
        return '#ffcc00'
    else:
        return '#ff4444'


def get_sl_type_badge(sl_type: str) -> tuple:
    """Get badge color and label for SL type"""
    badges = {
        'below_ob': ('#00ff88', '🏦 ORDER BLOCK'),
        'above_ob': ('#00ff88', '🏦 ORDER BLOCK'),
        'below_fvg': ('#00d4aa', '📊 FVG'),
        'above_fvg': ('#00d4aa', '📊 FVG'),
        'below_swing': ('#ffcc00', '📈 SWING'),
        'above_swing': ('#ffcc00', '📈 SWING'),
        'below_support': ('#ffcc00', '🛡️ SUPPORT'),
        'above_resistance': ('#ffcc00', '🛡️ RESISTANCE'),
        'below_vwap': ('#00d4ff', '📊 VWAP'),
        'above_vwap': ('#00d4ff', '📊 VWAP'),
        'below_ema': ('#ff9500', '📉 EMA'),
        'above_ema': ('#ff9500', '📉 EMA'),
        'atr_fallback': ('#ff4444', '⚠️ ATR FALLBACK'),
    }
    
    for key, (color, label) in badges.items():
        if key in sl_type:
            return color, label
    
    return '#888888', '❓ UNKNOWN'


def render_trading_card(
    symbol: str,
    grade: str,
    confidence: int,
    direction: str,
    levels: SmartTradeLevels,
    position_size: float = 2500,
    auto_added: bool = False
) -> None:
    """
    Render the Trading Card at top of analysis.
    Shows actionable info in clean format.
    """
    
    grade_color = get_grade_color(grade)
    direction_emoji = "🟢" if direction == "LONG" else "🔴"
    direction_text = "BUY" if direction == "LONG" else "SELL"
    
    # Calculate dollar amounts
    risk_dollars = position_size * (levels.risk_pct / 100)
    tp1_profit = position_size * (levels.tp1_rr * levels.risk_pct / 100)
    tp2_profit = position_size * (levels.tp2_rr * levels.risk_pct / 100)
    tp3_profit = position_size * (levels.tp3_rr * levels.risk_pct / 100)
    
    # SL type badge
    sl_badge_color, sl_badge_label = get_sl_type_badge(levels.sl_type)
    
    # Auto-added badge
    auto_badge_html = f"""<span style='background: {grade_color}; color: black; padding: 3px 10px; 
                          border-radius: 4px; margin-left: 10px; font-size: 0.85em;'>⭐ AUTO-LOGGED</span>""" if auto_added else ""
    
    # Main card
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid {grade_color};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    '>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <div>
                <span style='font-size: 1.5em; font-weight: bold; color: {grade_color};'>
                    {direction_emoji} {symbol} - Grade {grade} ({confidence}%)
                </span>
                {auto_badge_html}
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <!-- Left: Signal & Entry -->
            <div>
                <div style='color: #888; font-size: 0.85em; margin-bottom: 5px;'>SIGNAL</div>
                <div style='font-size: 1.3em; color: {grade_color}; font-weight: bold; margin-bottom: 10px;'>
                    {direction_text} @ {format_price(levels.entry)}
                </div>
                
                <div style='color: #888; font-size: 0.85em; margin-bottom: 5px;'>STOP LOSS</div>
                <div style='color: #ff6b6b; margin-bottom: 5px;'>
                    {format_price(levels.stop_loss)}
                    <span style='background: {sl_badge_color}; color: black; padding: 2px 8px; 
                          border-radius: 4px; margin-left: 8px; font-size: 0.75em;'>{sl_badge_label}</span>
                </div>
                <div style='color: #666; font-size: 0.8em; margin-bottom: 10px;'>
                    {levels.sl_reason} • Risk: {levels.risk_pct:.1f}%
                </div>
            </div>
            
            <!-- Right: Your Trade -->
            <div style='background: #0d1117; padding: 15px; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em; margin-bottom: 10px;'>YOUR ${position_size:,.0f} TRADE</div>
                
                <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                    <span style='color: #ff6b6b;'>Risk:</span>
                    <span style='color: #ff6b6b; font-weight: bold;'>-${risk_dollars:.0f}</span>
                </div>
                
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span style='color: {get_rr_color(levels.tp1_rr)};'>TP1 ({levels.tp1_rr:.1f}:1):</span>
                    <span style='color: {get_rr_color(levels.tp1_rr)}; font-weight: bold;'>+${tp1_profit:.0f}</span>
                </div>
                <div style='color: #666; font-size: 0.75em; margin-bottom: 8px; text-align: right;'>
                    @ {format_price(levels.tp1)} • {levels.tp1_reason}
                </div>
                
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span style='color: {get_rr_color(levels.tp2_rr)};'>TP2 ({levels.tp2_rr:.1f}:1):</span>
                    <span style='color: {get_rr_color(levels.tp2_rr)}; font-weight: bold;'>+${tp2_profit:.0f}</span>
                </div>
                <div style='color: #666; font-size: 0.75em; margin-bottom: 8px; text-align: right;'>
                    @ {format_price(levels.tp2)} • {levels.tp2_reason}
                </div>
                
                <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                    <span style='color: {get_rr_color(levels.tp3_rr)};'>TP3 ({levels.tp3_rr:.1f}:1):</span>
                    <span style='color: {get_rr_color(levels.tp3_rr)}; font-weight: bold;'>+${tp3_profit:.0f}</span>
                </div>
                <div style='color: #666; font-size: 0.75em; text-align: right;'>
                    @ {format_price(levels.tp3)} • {levels.tp3_reason}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_mini_trading_card(
    symbol: str,
    grade: str,
    direction: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    risk_pct: float,
    tp1_rr: float
) -> None:
    """
    Render a compact trading card for scanner results.
    """
    grade_color = get_grade_color(grade)
    direction_emoji = "🟢" if direction == "LONG" else "🔴"
    
    st.markdown(f"""
    <div style='background: #1a1a2e; border-left: 4px solid {grade_color}; padding: 12px; border-radius: 8px; margin: 5px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-weight: bold; color: {grade_color};'>{direction_emoji} {symbol}</span>
                <span style='background: {grade_color}; color: black; padding: 2px 6px; border-radius: 4px; 
                      margin-left: 8px; font-size: 0.75em;'>{grade}</span>
            </div>
            <div style='color: #888; font-size: 0.85em;'>
                Entry: {format_price(entry)} | SL: {format_price(stop_loss)} ({risk_pct:.1f}%)
            </div>
            <div style='color: {get_rr_color(tp1_rr)};'>
                TP1: {format_price(tp1)} ({tp1_rr:.1f}:1)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_monitor_card_enhanced(trade: EnhancedTrade, current_price: float) -> None:
    """
    Render a monitored trade card with MAE/MFE and live P&L.
    Enhanced version with more data.
    """
    
    # Calculate current P&L
    if trade.direction == "LONG":
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
    else:
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100
    
    pnl_dollars = trade.position_size * (pnl_pct / 100)
    
    # Status colors
    if pnl_pct >= 0:
        pnl_color = "#00d4aa"
        pnl_emoji = "🟢"
    else:
        pnl_color = "#ff6b6b"
        pnl_emoji = "🔴"
    
    # Grade color
    grade_color = get_grade_color(trade.grade)
    
    # Status badge
    status_colors = {
        'ACTIVE': '#00d4ff',
        'TP1_HIT': '#00ff88',
        'TP2_HIT': '#00ff88',
        'PROFIT_PROTECTED': '#ffcc00',
        'TRAILING': '#ffcc00',
        'CLOSED': '#888888'
    }
    status_color = status_colors.get(trade.status, '#888888')
    
    # Distance to SL/TP
    if trade.direction == "LONG":
        dist_to_sl = (current_price - trade.current_sl) / current_price * 100
        dist_to_tp1 = (trade.tp1 - current_price) / current_price * 100 if not trade.tp1_hit else 0
    else:
        dist_to_sl = (trade.current_sl - current_price) / current_price * 100
        dist_to_tp1 = (current_price - trade.tp1) / current_price * 100 if not trade.tp1_hit else 0
    
    # SL type badge
    sl_badge_color, sl_badge_label = get_sl_type_badge(trade.sl_type)
    
    # Confirmation badge
    conf_badge = f"🔄 x{trade.confirmation_count}" if trade.confirmation_count > 1 else ""
    
    st.markdown(f"""
    <div style='
        background: #1a1a2e;
        border: 1px solid #333;
        border-left: 4px solid {pnl_color};
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    '>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <div>
                <span style='font-size: 1.2em; font-weight: bold; color: #ffffff;'>
                    {pnl_emoji} {trade.symbol}
                </span>
                <span style='background: {grade_color}; color: black; padding: 2px 8px; border-radius: 4px; margin-left: 8px; font-size: 0.8em;'>
                    {trade.grade}
                </span>
                <span style='background: {status_color}; color: black; padding: 2px 8px; border-radius: 4px; margin-left: 5px; font-size: 0.8em;'>
                    {trade.status}
                </span>
                <span style='color: #888; margin-left: 8px; font-size: 0.85em;'>
                    {conf_badge}
                </span>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 1.3em; font-weight: bold; color: {pnl_color};'>
                    {pnl_pct:+.2f}%
                </div>
                <div style='color: {pnl_color}; font-size: 0.9em;'>
                    ${pnl_dollars:+,.2f}
                </div>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.85em;'>
            <div>
                <div style='color: #666;'>Entry</div>
                <div style='color: #fff;'>{format_price(trade.entry_price)}</div>
            </div>
            <div>
                <div style='color: #666;'>Current</div>
                <div style='color: #fff;'>{format_price(current_price)}</div>
            </div>
            <div>
                <div style='color: #666;'>Stop Loss</div>
                <div style='color: #ff6b6b;'>{format_price(trade.current_sl)}</div>
                <div style='color: #666; font-size: 0.8em;'>{dist_to_sl:.1f}% away</div>
            </div>
            <div>
                <div style='color: #666;'>{'TP1 ✅' if trade.tp1_hit else 'TP1'}</div>
                <div style='color: #00d4aa;'>{format_price(trade.tp1)}</div>
                <div style='color: #666; font-size: 0.8em;'>{dist_to_tp1:.1f}% away</div>
            </div>
        </div>
        
        <div style='margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;'>
            <div style='display: flex; justify-content: space-between; font-size: 0.8em;'>
                <span style='color: #666;'>
                    MAE: <span style='color: #ff6b6b;'>{trade.worst_drawdown_pct:.1f}%</span> | 
                    MFE: <span style='color: #00d4aa;'>{trade.highest_profit_pct:.1f}%</span>
                </span>
                <span style='color: #666;'>
                    {trade.timeframe} | {trade.direction} |
                    <span style='color: {sl_badge_color};'>{sl_badge_label}</span>
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stats_summary(stats: Dict) -> None:
    """
    Render statistics summary card.
    """
    
    if stats.get('closed_count', 0) == 0:
        st.info("📊 No closed trades yet. Statistics will appear after trades complete.")
        return
    
    # Determine overall performance color
    win_rate = stats.get('win_rate', 0)
    if win_rate >= 60:
        perf_color = "#00ff88"
        perf_emoji = "🔥"
    elif win_rate >= 50:
        perf_color = "#00d4aa"
        perf_emoji = "✅"
    elif win_rate >= 40:
        perf_color = "#ffcc00"
        perf_emoji = "⚠️"
    else:
        perf_color = "#ff6b6b"
        perf_emoji = "❌"
    
    total_pnl = stats.get('total_pnl_dollars', 0)
    profit_factor = stats.get('profit_factor', 0)
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid {perf_color};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    '>
        <div style='font-size: 1.3em; font-weight: bold; color: {perf_color}; margin-bottom: 15px;'>
            {perf_emoji} SYSTEM PERFORMANCE (A/A+ Signals)
        </div>
        
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;'>
            <div style='text-align: center; background: #0d1117; padding: 15px; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em;'>Total Trades</div>
                <div style='font-size: 1.5em; color: #00d4ff; font-weight: bold;'>{stats.get('total_trades', 0)}</div>
                <div style='color: #666; font-size: 0.8em;'>{stats.get('active_count', 0)} active</div>
            </div>
            <div style='text-align: center; background: #0d1117; padding: 15px; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em;'>Win Rate</div>
                <div style='font-size: 1.5em; color: {perf_color}; font-weight: bold;'>{win_rate:.1f}%</div>
                <div style='color: #666; font-size: 0.8em;'>{stats.get('win_count', 0)}W / {stats.get('loss_count', 0)}L</div>
            </div>
            <div style='text-align: center; background: #0d1117; padding: 15px; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em;'>Total P&L</div>
                <div style='font-size: 1.5em; color: {"#00d4aa" if total_pnl >= 0 else "#ff6b6b"}; font-weight: bold;'>
                    ${total_pnl:+,.0f}
                </div>
                <div style='color: #666; font-size: 0.8em;'>{stats.get('total_pnl_pct', 0):+.1f}%</div>
            </div>
            <div style='text-align: center; background: #0d1117; padding: 15px; border-radius: 8px;'>
                <div style='color: #888; font-size: 0.85em;'>Profit Factor</div>
                <div style='font-size: 1.5em; color: {"#00d4aa" if profit_factor >= 1.5 else "#ffcc00" if profit_factor >= 1.0 else "#ff6b6b"}; font-weight: bold;'>
                    {profit_factor:.2f}
                </div>
                <div style='color: #666; font-size: 0.8em;'>{"Good" if profit_factor >= 1.5 else "OK" if profit_factor >= 1.0 else "Poor"}</div>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
            <div>
                <div style='color: #888; margin-bottom: 10px;'>📈 Winning Trades</div>
                <div style='color: #00d4aa;'>Avg Win: ${stats.get('avg_win_dollars', 0):+,.0f} ({stats.get('avg_win_pct', 0):+.1f}%)</div>
                <div style='color: #00d4aa;'>Best: ${stats.get('best_trade_dollars', 0):+,.0f}</div>
            </div>
            <div>
                <div style='color: #888; margin-bottom: 10px;'>📉 Losing Trades</div>
                <div style='color: #ff6b6b;'>Avg Loss: ${stats.get('avg_loss_dollars', 0):,.0f}</div>
                <div style='color: #ff6b6b;'>Worst: ${stats.get('worst_trade_dollars', 0):,.0f}</div>
            </div>
        </div>
        
        <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;'>
            <div style='color: #888; margin-bottom: 10px;'>📊 MAE/MFE Analysis (for future optimization)</div>
            <div style='color: #666;'>
                Avg Worst Drawdown: {stats.get('avg_mae', 0):.1f}% | Avg Best Profit Seen: {stats.get('avg_mfe', 0):.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_grade_breakdown(grade_stats: Dict) -> None:
    """Render performance breakdown by grade"""
    if not grade_stats:
        return
    
    st.markdown("### 📊 Performance by Grade")
    
    cols = st.columns(len(grade_stats))
    
    for i, (grade, stats) in enumerate(grade_stats.items()):
        with cols[i]:
            color = get_grade_color(grade)
            win_rate = stats.get('win_rate', 0)
            st.markdown(f"""
            <div style='background: #1a1a2e; padding: 15px; border-radius: 8px; text-align: center; border-top: 3px solid {color};'>
                <div style='font-size: 1.5em; color: {color}; font-weight: bold;'>{grade}</div>
                <div style='color: #888; margin: 10px 0;'>{stats.get('total', 0)} trades</div>
                <div style='color: {"#00d4aa" if win_rate >= 50 else "#ff6b6b"};'>
                    {win_rate:.0f}% win rate
                </div>
                <div style='color: #888; font-size: 0.85em;'>
                    Avg: {stats.get('avg_pnl', 0):+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
