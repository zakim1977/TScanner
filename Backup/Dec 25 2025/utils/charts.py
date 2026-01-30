"""
Chart Generation Module
Plotly charts for trade setups and performance
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional

from .formatters import fmt_price, calc_roi

# ═══════════════════════════════════════════════════════════════════════════════
# CHART COLORS
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = {
    'bg': '#0e1117',
    'grid': '#1e2130',
    'text': '#ffffff',
    'green': '#00d4aa',
    'red': '#ff4444',
    'blue': '#00d4ff',
    'yellow': '#ffcc00',
    'purple': '#9d4edd',
    'orange': '#ff9500',
    'ema_20': '#ffcc00',
    'ema_50': '#00d4ff'
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SETUP CHART
# ═══════════════════════════════════════════════════════════════════════════════

def create_trade_setup_chart(df: pd.DataFrame, signal, master_score: int = 0,
                             show_volume: bool = True) -> go.Figure:
    """
    Create a professional trade setup chart with Entry, SL, TP levels
    
    Args:
        df: OHLCV DataFrame
        signal: TradeSignal object
        master_score: Signal score (0-100)
        show_volume: Whether to show volume subplot
        
    Returns:
        Plotly Figure
    """
    if df is None or len(df) < 20:
        return go.Figure()
    
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
    else:
        fig = go.Figure()
    
    # Candlestick chart
    candlestick = go.Candlestick(
        x=df['DateTime'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=COLORS['green'],
        decreasing_line_color=COLORS['red']
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add EMAs
    if len(df) >= 20:
        ema_20 = df['Close'].ewm(span=20, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df['DateTime'],
            y=ema_20,
            name='EMA 20',
            line=dict(color=COLORS['ema_20'], width=1)
        ), row=1, col=1) if show_volume else fig.add_trace(go.Scatter(
            x=df['DateTime'],
            y=ema_20,
            name='EMA 20',
            line=dict(color=COLORS['ema_20'], width=1)
        ))
    
    if len(df) >= 50:
        ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
        fig.add_trace(go.Scatter(
            x=df['DateTime'],
            y=ema_50,
            name='EMA 50',
            line=dict(color=COLORS['ema_50'], width=1)
        ), row=1, col=1) if show_volume else fig.add_trace(go.Scatter(
            x=df['DateTime'],
            y=ema_50,
            name='EMA 50',
            line=dict(color=COLORS['ema_50'], width=1)
        ))
    
    # Add trade levels if signal provided
    if signal:
        # Entry line
        fig.add_hline(
            y=signal.entry,
            line_dash="solid",
            line_color=COLORS['blue'],
            line_width=2,
            annotation_text=f"ENTRY: {fmt_price(signal.entry)}",
            annotation_position="right",
            annotation_font_color=COLORS['text']
        )
        
        # Stop Loss line
        tp1_roi = calc_roi(signal.tp1, signal.entry)
        tp2_roi = calc_roi(signal.tp2, signal.entry)
        tp3_roi = calc_roi(signal.tp3, signal.entry)
        
        fig.add_hline(
            y=signal.stop_loss,
            line_dash="dash",
            line_color=COLORS['red'],
            line_width=2,
            annotation_text=f"SL: {fmt_price(signal.stop_loss)} (-{signal.risk_pct:.1f}%)",
            annotation_position="right",
            annotation_font_color=COLORS['red']
        )
        
        # Take Profit lines
        fig.add_hline(
            y=signal.tp1,
            line_dash="dash",
            line_color=COLORS['green'],
            line_width=1,
            annotation_text=f"TP1: {fmt_price(signal.tp1)} (+{tp1_roi:.1f}%)",
            annotation_position="right",
            annotation_font_color=COLORS['green']
        )
        
        fig.add_hline(
            y=signal.tp2,
            line_dash="dash",
            line_color=COLORS['green'],
            line_width=1,
            annotation_text=f"TP2: {fmt_price(signal.tp2)} (+{tp2_roi:.1f}%)",
            annotation_position="right",
            annotation_font_color=COLORS['green']
        )
        
        fig.add_hline(
            y=signal.tp3,
            line_dash="dash",
            line_color=COLORS['green'],
            line_width=1,
            annotation_text=f"TP3: {fmt_price(signal.tp3)} (+{tp3_roi:.1f}%)",
            annotation_position="right",
            annotation_font_color=COLORS['green']
        )
    
    # Add volume bars
    if show_volume:
        colors = [COLORS['green'] if c >= o else COLORS['red'] 
                  for c, o in zip(df['Close'], df['Open'])]
        
        fig.add_trace(go.Bar(
            x=df['DateTime'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{signal.symbol if signal else 'Chart'} | Score: {master_score}/100" if master_score else "",
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor=COLORS['grid'],
        showgrid=True
    )
    
    fig.update_yaxes(
        gridcolor=COLORS['grid'],
        showgrid=True
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE CHART
# ═══════════════════════════════════════════════════════════════════════════════

def create_performance_chart(trades: list) -> go.Figure:
    """
    Create equity curve chart from trade history
    
    Args:
        trades: List of closed trades
        
    Returns:
        Plotly Figure
    """
    if not trades:
        fig = go.Figure()
        fig.add_annotation(
            text="No trades yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=COLORS['text'])
        )
        fig.update_layout(
            paper_bgcolor=COLORS['bg'],
            plot_bgcolor=COLORS['bg']
        )
        return fig
    
    # Calculate cumulative P&L
    equity = [100]  # Start with 100
    dates = []
    
    for trade in trades:
        # Get P&L from either final_pnl (auto-close) or pnl_pct (manual close)
        pnl = trade.get('final_pnl', trade.get('pnl_pct', 0))
        new_equity = equity[-1] * (1 + pnl / 100)
        equity.append(new_equity)
        dates.append(trade.get('closed_at', trade.get('exit_time', trade.get('created_at', ''))))
    
    # Create figure
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=list(range(len(equity))),
        y=equity,
        mode='lines+markers',
        name='Equity',
        line=dict(color=COLORS['blue'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Add reference line at starting capital
    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="gray",
        annotation_text="Starting Capital"
    )
    
    fig.update_layout(
        title='Equity Curve',
        xaxis_title='Trade #',
        yaxis_title='Equity %',
        template='plotly_dark',
        paper_bgcolor=COLORS['bg'],
        plot_bgcolor=COLORS['bg'],
        font=dict(color=COLORS['text']),
        height=400
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MINI CHART (for cards)
# ═══════════════════════════════════════════════════════════════════════════════

def create_mini_chart(df: pd.DataFrame, height: int = 150) -> go.Figure:
    """
    Create a mini sparkline chart for signal cards
    """
    if df is None or len(df) < 10:
        return go.Figure()
    
    # Use last 50 candles
    df = df.tail(50)
    
    # Determine color based on trend
    is_bullish = df['Close'].iloc[-1] > df['Close'].iloc[0]
    color = COLORS['green'] if is_bullish else COLORS['red']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=df['Close'],
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba({61 if is_bullish else 255}, {214 if is_bullish else 68}, {170 if is_bullish else 68}, 0.1)"
    ))
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION RANGE CHART
# ═══════════════════════════════════════════════════════════════════════════════

def create_position_range_chart(entry: float, sl: float, tp1: float, tp2: float, 
                                tp3: float, current: float) -> go.Figure:
    """
    Create a horizontal bar showing position in trade range
    """
    fig = go.Figure()
    
    # Calculate positions as percentages of range
    total_range = tp3 - sl
    
    def to_pct(price):
        return ((price - sl) / total_range) * 100 if total_range > 0 else 50
    
    # Background bar (SL to TP3)
    fig.add_trace(go.Bar(
        y=['Position'],
        x=[100],
        orientation='h',
        marker_color='#1a1a2e',
        showlegend=False
    ))
    
    # Current position marker
    current_pct = to_pct(current)
    
    # Color based on position
    if current < entry:
        marker_color = COLORS['red']
    elif current >= tp1:
        marker_color = COLORS['green']
    else:
        marker_color = COLORS['blue']
    
    fig.add_vline(x=current_pct, line_color=marker_color, line_width=3)
    
    # Add level markers
    fig.add_vline(x=to_pct(entry), line_color=COLORS['blue'], line_dash='dash')
    fig.add_vline(x=to_pct(tp1), line_color=COLORS['green'], line_dash='dot')
    fig.add_vline(x=to_pct(tp2), line_color=COLORS['green'], line_dash='dot')
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=50,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False)
    )
    
    return fig
