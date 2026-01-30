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
    'ema_50': '#00d4ff',
    # SMC Colors
    'bullish_ob': 'rgba(0, 212, 170, 0.25)',      # Green for bullish OB
    'bearish_ob': 'rgba(255, 68, 68, 0.25)',      # Red for bearish OB
    'bullish_fvg': 'rgba(0, 150, 255, 0.2)',      # Blue for bullish FVG
    'bearish_fvg': 'rgba(255, 150, 0, 0.2)',      # Orange for bearish FVG
    'vwap': '#ff00ff',                             # Magenta for VWAP
    'swing_high': '#00d4ff',                       # Cyan for swing high
    'swing_low': '#ffcc00',                        # Yellow for swing low
}

# ═══════════════════════════════════════════════════════════════════════════════
# VWAP CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price"""
    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        return df['Close']  # Fallback to close price
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SETUP CHART
# ═══════════════════════════════════════════════════════════════════════════════

def create_trade_setup_chart(df: pd.DataFrame, signal, master_score: int = 0,
                             show_volume: bool = True, smc_data: dict = None,
                             show_smc: bool = True, htf_obs: dict = None,
                             htf_timeframe: str = None) -> go.Figure:
    """
    Create a professional trade setup chart with Entry, SL, TP levels
    and SMC elements (Order Blocks, FVG, VWAP, Swing levels)
    
    Args:
        df: OHLCV DataFrame
        signal: TradeSignal object
        master_score: Signal score (0-100)
        show_volume: Whether to show volume subplot
        smc_data: Dict with order_blocks, fvg, structure data from detect_smc()
        show_smc: Whether to show SMC overlays
        
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SMC OVERLAYS - Order Blocks, FVG, VWAP, Swing Levels
    # ═══════════════════════════════════════════════════════════════════════════
    
    if show_smc:
        x0 = df['DateTime'].iloc[0]
        x1 = df['DateTime'].iloc[-1]
        
        # --- VWAP Line ---
        vwap = calculate_vwap(df)
        trace_kwargs = dict(
            x=df['DateTime'],
            y=vwap,
            name='VWAP',
            line=dict(color=COLORS['vwap'], width=1.5, dash='dot')
        )
        if show_volume:
            fig.add_trace(go.Scatter(**trace_kwargs), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))
        
        # --- Swing High/Low Lines ---
        if smc_data:
            structure = smc_data.get('structure', {})
            swing_high = structure.get('last_swing_high', 0)
            swing_low = structure.get('last_swing_low', 0)
            
            if swing_high and swing_high > 0:
                fig.add_hline(
                    y=swing_high,
                    line_dash="solid",
                    line_color=COLORS['swing_high'],
                    line_width=1.5,
                    annotation_text=f"S.High",
                    annotation_position="left",
                    annotation_font_color=COLORS['swing_high'],
                    annotation_font_size=10
                )
            
            if swing_low and swing_low > 0:
                fig.add_hline(
                    y=swing_low,
                    line_dash="solid",
                    line_color=COLORS['swing_low'],
                    line_width=1.5,
                    annotation_text=f"S.Low",
                    annotation_position="left",
                    annotation_font_color=COLORS['swing_low'],
                    annotation_font_size=10
                )
            
            # --- Order Blocks ---
            ob_data = smc_data.get('order_blocks', {})
            
            # Bullish Order Block (demand zone)
            if ob_data.get('bullish_ob'):
                ob_high = ob_data.get('bullish_ob_top', 0)  # Use correct field name
                ob_low = ob_data.get('bullish_ob_bottom', 0)  # Use correct field name
                if ob_high > 0 and ob_low > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1,
                        y0=ob_low, y1=ob_high,
                        fillcolor=COLORS['bullish_ob'],
                        line=dict(color='rgba(0, 212, 170, 0.6)', width=1),
                        layer="below",
                        name="Bullish OB"
                    )
                    # Add label
                    fig.add_annotation(
                        x=x0, y=(ob_high + ob_low) / 2,
                        text="B.OB",
                        showarrow=False,
                        font=dict(color=COLORS['green'], size=9),
                        xanchor="left"
                    )
            
            # Bearish Order Block (supply zone)
            if ob_data.get('bearish_ob'):
                ob_high = ob_data.get('bearish_ob_top', 0)  # Use correct field name
                ob_low = ob_data.get('bearish_ob_bottom', 0)  # Use correct field name
                if ob_high > 0 and ob_low > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1,
                        y0=ob_low, y1=ob_high,
                        fillcolor=COLORS['bearish_ob'],
                        line=dict(color='rgba(255, 68, 68, 0.6)', width=1),
                        layer="below",
                        name="Bearish OB"
                    )
                    # Add label
                    fig.add_annotation(
                        x=x0, y=(ob_high + ob_low) / 2,
                        text="S.OB",
                        showarrow=False,
                        font=dict(color=COLORS['red'], size=9),
                        xanchor="left"
                    )
            
            # --- Fair Value Gaps (FVG) ---
            fvg_data = smc_data.get('fvg', {})
            
            # Bullish FVG
            if fvg_data.get('bullish_fvg'):
                fvg_high = fvg_data.get('bullish_fvg_high', 0)
                fvg_low = fvg_data.get('bullish_fvg_low', 0)
                if fvg_high > 0 and fvg_low > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1,
                        y0=fvg_low, y1=fvg_high,
                        fillcolor=COLORS['bullish_fvg'],
                        line=dict(color='rgba(0, 150, 255, 0.5)', width=1, dash='dot'),
                        layer="below",
                        name="Bullish FVG"
                    )
                    fig.add_annotation(
                        x=x1, y=(fvg_high + fvg_low) / 2,
                        text="B.FVG",
                        showarrow=False,
                        font=dict(color='#0096ff', size=9),
                        xanchor="right"
                    )
            
            # Bearish FVG
            if fvg_data.get('bearish_fvg'):
                fvg_high = fvg_data.get('bearish_fvg_high', 0)
                fvg_low = fvg_data.get('bearish_fvg_low', 0)
                if fvg_high > 0 and fvg_low > 0:
                    fig.add_shape(
                        type="rect",
                        x0=x0, x1=x1,
                        y0=fvg_low, y1=fvg_high,
                        fillcolor=COLORS['bearish_fvg'],
                        line=dict(color='rgba(255, 150, 0, 0.5)', width=1, dash='dot'),
                        layer="below",
                        name="Bearish FVG"
                    )
                    fig.add_annotation(
                        x=x1, y=(fvg_high + fvg_low) / 2,
                        text="S.FVG",
                        showarrow=False,
                        font=dict(color='#ff9600', size=9),
                        xanchor="right"
                    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HTF ORDER BLOCKS (for TP Targeting) - Dashed zones
    # ═══════════════════════════════════════════════════════════════════════════
    
    if htf_obs and show_smc:
        x0 = df['DateTime'].iloc[0]
        x1 = df['DateTime'].iloc[-1]
        htf_label = f" ({htf_timeframe})" if htf_timeframe else " (HTF)"
        
        # HTF Bearish OBs (resistance above - LONG targets)
        for i, ob in enumerate(htf_obs.get('bearish_obs', [])[:3]):  # Max 3
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=ob['bottom'], y1=ob['top'],
                fillcolor='rgba(255, 68, 68, 0.1)',  # Lighter than LTF
                line=dict(color='rgba(255, 68, 68, 0.4)', width=1, dash='dash'),
                layer="below",
            )
            if i == 0:  # Only label first one
                fig.add_annotation(
                    x=x1, y=ob['mid'],
                    text=f"HTF S.OB",
                    showarrow=False,
                    font=dict(color='#ff6666', size=8),
                    xanchor="right"
                )
        
        # HTF Bullish OBs (support below - SHORT targets)
        for i, ob in enumerate(htf_obs.get('bullish_obs', [])[:3]):  # Max 3
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=ob['bottom'], y1=ob['top'],
                fillcolor='rgba(0, 212, 170, 0.1)',  # Lighter than LTF
                line=dict(color='rgba(0, 212, 170, 0.4)', width=1, dash='dash'),
                layer="below",
            )
            if i == 0:  # Only label first one
                fig.add_annotation(
                    x=x1, y=ob['mid'],
                    text=f"HTF B.OB",
                    showarrow=False,
                    font=dict(color='#66ffaa', size=8),
                    xanchor="right"
                )
    
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