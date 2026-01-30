"""
InvestorIQ - Global Education Renderer
=======================================
Centralized education content that renders properly everywhere.
All education uses HTML (not markdown) to avoid rendering issues.

Usage:
    from utils.education_renderer import render_education_section, get_education_content
    
    # In Scanner, Monitor, or Single Analysis:
    render_education_section(st, context_data)
"""

import streamlit as st
from typing import Dict, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# EDUCATION CONTENT - HTML FORMATTED (no markdown **)
# ═══════════════════════════════════════════════════════════════════════════════

EDUCATION_CONTENT = {
    # ─────────────────────────────────────────────────────────────────────────
    # ORDER BLOCKS
    # ─────────────────────────────────────────────────────────────────────────
    "order_block": {
        "title": "Order Blocks (OB)",
        "emoji": "🏦",
        "content": """
            <b>What it is:</b> The last candle before a strong move - where institutions placed orders.<br><br>
            <b>Why it matters:</b> When price returns to an OB, it often bounces because institutions defend their positions.<br><br>
            <b>How to use:</b><br>
            • LONG: Buy when price returns to Bullish OB<br>
            • SHORT: Sell when price returns to Bearish OB<br>
            • Stop Loss: Just beyond the OB (if it breaks, setup invalid)
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # FAIR VALUE GAP
    # ─────────────────────────────────────────────────────────────────────────
    "fvg": {
        "title": "Fair Value Gap (FVG)",
        "emoji": "📊",
        "content": """
            <b>What it is:</b> A price imbalance where price moved too fast, leaving a "gap" in fair value.<br><br>
            <b>Why it matters:</b> Markets seek balance - price tends to return and fill these gaps before continuing.<br><br>
            <b>How to use:</b><br>
            • Wait for price to return to FVG zone<br>
            • Enter when FVG gets "mitigated" (touched)<br>
            • Combine with Order Blocks for higher probability
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # MONEY FLOW (with dynamic values)
    # ─────────────────────────────────────────────────────────────────────────
    "money_flow": {
        "title": "Money Flow (MFI/CMF/OBV)",
        "emoji": "💰",
        "content_template": """
            <b>MFI (Money Flow Index):</b> Like RSI but includes volume.<br>
            Current: {mfi:.0f} - {mfi_status}<br><br>
            
            <b>CMF (Chaikin Money Flow):</b> Measures buying/selling pressure.<br>
            Current: {cmf:.3f} - {cmf_status}<br><br>
            
            <b>OBV (On-Balance Volume):</b> Cumulative volume direction.<br>
            {obv_status}<br><br>
            
            <b>Key insight:</b> Price can be manipulated, but volume cannot lie.
            When volume confirms price, the move is more likely to continue.
        """,
        "content_static": """
            <b>MFI (Money Flow Index):</b> RSI weighted by volume (0-100).<br>
            • Below 20 = Oversold (bullish)<br>
            • Above 80 = Overbought (bearish)<br><br>
            
            <b>CMF (Chaikin Money Flow):</b> Net buying/selling pressure.<br>
            • Positive = Buyers in control<br>
            • Negative = Sellers in control<br><br>
            
            <b>OBV (On-Balance Volume):</b><br>
            • Rising OBV = Smart money buying<br>
            • Falling OBV = Smart money selling<br><br>
            
            <b>Key insight:</b> Volume precedes price. When OBV diverges from price, a move is coming.
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # WHALE DATA (CRYPTO)
    # ─────────────────────────────────────────────────────────────────────────
    "whale_crypto": {
        "title": "Whale & Institutional Data",
        "emoji": "🐋",
        "content_template": """
            <b>Open Interest + Price:</b><br>
            OI: {oi_change:+.1f}% | Price: {price_change:+.1f}%<br>
            {oi_signal}<br><br>
            
            <b>Funding Rate:</b> {funding:.4f}%<br>
            {funding_signal}<br><br>
            
            <b>Positioning:</b><br>
            🐋 Whales: {whale_long:.0f}% Long | 🐑 Retail: {retail_long:.0f}% Long<br>
            {retail_signal}<br><br>
            
            <b>How to read OI + Price:</b><br>
            <table style='color:#ccc; width:100%; font-size:0.9em;'>
                <tr><td>OI↑ + Price↑</td><td>=</td><td style='color:#00d4aa'>New longs (strong bullish)</td></tr>
                <tr><td>OI↑ + Price↓</td><td>=</td><td style='color:#ff4444'>New shorts (strong bearish)</td></tr>
                <tr><td>OI↓ + Price↑</td><td>=</td><td style='color:#ffcc00'>Shorts covering (weak rally)</td></tr>
                <tr><td>OI↓ + Price↓</td><td>=</td><td style='color:#ffcc00'>Longs liquidating (weak dump)</td></tr>
            </table>
        """,
        "content_static": """
            <b>Open Interest (OI):</b> Total open futures contracts.<br>
            • OI Rising = New money entering<br>
            • OI Falling = Money leaving (closing positions)<br><br>
            
            <b>Funding Rate:</b> Payment between longs/shorts every 8h.<br>
            • Positive funding = Longs pay shorts (too bullish)<br>
            • Negative funding = Shorts pay longs (too bearish)<br>
            • Extreme funding = Contrarian signal<br><br>
            
            <b>Top Trader Positioning:</b><br>
            • Whales >55% Long = Smart money bullish<br>
            • Whales <45% Long = Smart money bearish<br><br>
            
            <b>Fade Retail Strategy:</b><br>
            • Retail >65% Long = Consider SHORT<br>
            • Retail <35% Long = Consider LONG
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # STOCK INSTITUTIONAL DATA
    # ─────────────────────────────────────────────────────────────────────────
    "whale_stock": {
        "title": "Institutional Analysis (Stocks)",
        "emoji": "📊",
        "content_template": """
            <b>Short Interest:</b> {short_pct:.1f}% of float<br>
            {short_signal}<br><br>
            
            <b>Put/Call Ratio:</b> {pc_ratio:.2f}<br>
            {pc_signal}<br><br>
            
            <b>Insider Activity:</b> {insider}<br>
            {insider_signal}<br><br>
            
            <b>Institutional Ownership:</b> {inst_pct:.0f}%<br><br>
            
            <b>Key insights:</b><br>
            • Short Interest >20% + catalyst = Squeeze potential<br>
            • P/C Ratio <0.7 = Heavy call buying (bullish)<br>
            • Insiders buying with own money = Strong conviction
        """,
        "content_static": """
            <b>Short Interest:</b> % of shares sold short.<br>
            • Below 10% = Normal<br>
            • 10-20% = Elevated, watch for squeeze<br>
            • Above 20% = High squeeze potential<br><br>
            
            <b>Put/Call Ratio:</b> Options market sentiment.<br>
            • Below 0.7 = Bullish (more calls)<br>
            • Above 1.0 = Bearish (more puts)<br>
            • Extreme = Contrarian signal<br><br>
            
            <b>Insider Trading (Form 4):</b><br>
            • Insiders BUYING = Very bullish (they know best)<br>
            • Insiders SELLING = Check why (taxes or bearish?)<br><br>
            
            <b>Institutional %:</b> Higher = more stable, more efficient
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # RISK:REWARD
    # ─────────────────────────────────────────────────────────────────────────
    "risk_reward": {
        "title": "Risk:Reward Ratio (R:R)",
        "emoji": "⚖️",
        "content_template": """
            <b>What it means:</b> For every $1 you risk, how much can you gain?<br><br>
            
            <b>This trade:</b> Risk {risk:.1f}% to gain +{reward:.1f}% at TP1<br>
            R:R at TP1 = <span style='color:#00d4ff; font-weight:bold;'>{rr:.1f}:1</span><br><br>
            
            <b>Good practice:</b> Only take trades with R:R >= 1.5:1<br>
            This means even with 40% win rate, you're profitable!<br><br>
            
            <b>Example (1.5:1 R:R, 40% win rate):</b><br>
            • 4 wins × 1.5R = 6R gained<br>
            • 6 losses × 1R = 6R lost<br>
            • Net = Breakeven at just 40% accuracy!
        """,
        "content_static": """
            <b>What it means:</b> For every $1 you risk, how much can you gain?<br><br>
            
            <b>Good R:R targets:</b><br>
            • 1.5:1 = Minimum acceptable<br>
            • 2:1 = Good<br>
            • 3:1+ = Excellent (but harder to hit)<br><br>
            
            <b>Why it matters:</b><br>
            With 2:1 R:R, you only need 34% win rate to be profitable!<br><br>
            
            <b>Pro tip:</b> Never risk more to make less (R:R < 1:1).
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # TIMEFRAME
    # ─────────────────────────────────────────────────────────────────────────
    "timeframe": {
        "title": "Timeframe Selection",
        "emoji": "⏱️",
        "content_template": """
            You selected <b>{mode_name}</b> mode on <b>{timeframe}</b>.<br><br>
            
            <b>{timeframe} means:</b> Each candle = {timeframe} of price action<br><br>
            
            <b>Expected hold time:</b> {hold_time}<br><br>
            
            <b>Pro tip:</b> Higher timeframes = cleaner signals, slower trades.
            Lower timeframes = more opportunities, more noise.<br><br>
            
            <b>Multi-TF confirmation:</b> Check {confirm_tf} for higher probability.
        """,
        "content_static": """
            <b>Timeframe Guide:</b><br><br>
            
            <table style='color:#ccc; width:100%;'>
                <tr><td>1m/5m</td><td>Scalping</td><td>Minutes to hours</td></tr>
                <tr><td>15m/1h</td><td>Day Trading</td><td>Hours to 1 day</td></tr>
                <tr><td>4h</td><td>Swing Trading</td><td>Days to weeks</td></tr>
                <tr><td>1d</td><td>Position Trading</td><td>Weeks to months</td></tr>
            </table><br>
            
            <b>Pro tip:</b> Trade in direction of higher timeframe trend.
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # VWAP
    # ─────────────────────────────────────────────────────────────────────────
    "vwap": {
        "title": "VWAP (Volume Weighted Average Price)",
        "emoji": "📈",
        "content_static": """
            <b>What it is:</b> Average price weighted by volume - THE institutional benchmark.<br><br>
            
            <b>Why institutions use it:</b><br>
            • Large orders are judged against VWAP<br>
            • "Did we buy below VWAP?" = Good execution<br><br>
            
            <b>Trading rules:</b><br>
            • Price > VWAP = Bullish (buyers paying premium)<br>
            • Price < VWAP = Bearish (sellers accepting discount)<br><br>
            
            <b>VWAP as S/R:</b><br>
            • In uptrends: VWAP acts as SUPPORT (buy dips)<br>
            • In downtrends: VWAP acts as RESISTANCE (sell rallies)<br><br>
            
            <b>Pro tip:</b> First test of VWAP often holds. Third test often breaks.
        """
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # MARKET STRUCTURE
    # ─────────────────────────────────────────────────────────────────────────
    "market_structure": {
        "title": "Market Structure",
        "emoji": "📐",
        "content_static": """
            <b>What it is:</b> The pattern of highs and lows that define trend.<br><br>
            
            <b>Bullish Structure:</b><br>
            • Higher Highs (HH) + Higher Lows (HL)<br>
            • Each swing makes a new high<br>
            • Buy dips to HL<br><br>
            
            <b>Bearish Structure:</b><br>
            • Lower Highs (LH) + Lower Lows (LL)<br>
            • Each swing makes a new low<br>
            • Sell rallies to LH<br><br>
            
            <b>Break of Structure (BOS):</b><br>
            When price breaks the pattern, trend may be changing.<br>
            • BOS up in downtrend = Potential reversal long<br>
            • BOS down in uptrend = Potential reversal short
        """
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_education_card(st_obj, key: str, dynamic_values: Dict = None) -> None:
    """
    Render a single education card with proper HTML formatting.
    
    Args:
        st_obj: Streamlit object (st)
        key: Key from EDUCATION_CONTENT
        dynamic_values: Optional dict of values to format into template
    """
    if key not in EDUCATION_CONTENT:
        return
    
    edu = EDUCATION_CONTENT[key]
    title = edu.get('title', 'Education')
    emoji = edu.get('emoji', '📚')
    
    # Get content - use template if dynamic values provided, else static
    if dynamic_values and 'content_template' in edu:
        try:
            content = edu['content_template'].format(**dynamic_values)
        except:
            content = edu.get('content_static', edu.get('content', ''))
    else:
        content = edu.get('content_static', edu.get('content', ''))
    
    # Render with consistent styling
    st_obj.markdown(f"""
    <div style='background: #0a1a2a; border-radius: 8px; padding: 15px; margin: 10px 0; 
                border: 1px solid #1a3a5a;'>
        <div style='color: #00d4ff; font-weight: bold; font-size: 1.1em; margin-bottom: 10px;'>
            {emoji} {title}
        </div>
        <div style='color: #ccc; line-height: 1.7;'>
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_money_flow_education(st_obj, mf_data: Dict) -> None:
    """Render Money Flow education with live values."""
    mfi = mf_data.get('mfi', 50)
    cmf = mf_data.get('cmf', 0)
    obv_rising = mf_data.get('obv_rising', False)
    
    mfi_status = 'Overbought (bearish)' if mfi > 70 else 'Oversold (bullish)' if mfi < 30 else 'Neutral'
    cmf_status = 'Buyers in control' if cmf > 0 else 'Sellers in control'
    obv_status = 'Rising = Smart money buying' if obv_rising else 'Falling = Smart money selling'
    
    render_education_card(st_obj, 'money_flow', {
        'mfi': mfi,
        'mfi_status': mfi_status,
        'cmf': cmf,
        'cmf_status': cmf_status,
        'obv_status': obv_status
    })


def render_whale_education_crypto(st_obj, whale_data: Dict) -> None:
    """Render Whale education for crypto with live values."""
    oi_change = whale_data.get('open_interest', {}).get('change_24h', 0)
    price_change = whale_data.get('price_change_24h', 0)
    funding = whale_data.get('funding', {}).get('rate_pct', 0)
    whale_long = whale_data.get('top_trader_ls', {}).get('long_pct', 50)
    retail_long = whale_data.get('retail_ls', {}).get('long_pct', 50)
    
    # OI + Price interpretation
    if oi_change > 3 and price_change > 0:
        oi_signal = "🟢 NEW LONGS entering - Strong bullish"
    elif oi_change > 3 and price_change < 0:
        oi_signal = "🔴 NEW SHORTS entering - Strong bearish"
    elif oi_change < -3 and price_change > 0:
        oi_signal = "🟡 Short covering - Rally may be weak"
    elif oi_change < -3 and price_change < 0:
        oi_signal = "🟡 Long liquidation - Dump may be weak"
    else:
        oi_signal = "Neutral flow"
    
    # Funding interpretation
    if funding > 0.05:
        funding_signal = "⚠️ Longs overleveraged - dump risk"
    elif funding < -0.05:
        funding_signal = "✅ Shorts overleveraged - pump potential"
    else:
        funding_signal = "Neutral"
    
    # Retail interpretation
    if retail_long > 65:
        retail_signal = "⚠️ FADE RETAIL: Crowd heavily long"
    elif retail_long < 35:
        retail_signal = "✅ FADE RETAIL: Crowd heavily short"
    else:
        retail_signal = "No extreme positioning"
    
    render_education_card(st_obj, 'whale_crypto', {
        'oi_change': oi_change,
        'price_change': price_change,
        'oi_signal': oi_signal,
        'funding': funding,
        'funding_signal': funding_signal,
        'whale_long': whale_long,
        'retail_long': retail_long,
        'retail_signal': retail_signal
    })


def render_whale_education_stock(st_obj, stock_data: Dict) -> None:
    """Render institutional education for stocks with live values."""
    short_pct = stock_data.get('short_interest', {}).get('short_pct_float', 0)
    pc_ratio = stock_data.get('options', {}).get('put_call_ratio', 1.0)
    insider = stock_data.get('insider', {}).get('sentiment', 'NEUTRAL')
    inst_pct = stock_data.get('institutional', {}).get('institutional_pct', 0)
    
    # Interpretations
    if short_pct > 20:
        short_signal = "🚀 HIGH - Squeeze potential!"
    elif short_pct > 15:
        short_signal = "⚠️ Elevated"
    else:
        short_signal = "Normal"
    
    if pc_ratio < 0.7:
        pc_signal = "🟢 Bullish (heavy call buying)"
    elif pc_ratio > 1.0:
        pc_signal = "🔴 Bearish (heavy put buying)"
    else:
        pc_signal = "Neutral"
    
    if insider in ['STRONG_BUY', 'BULLISH']:
        insider_signal = "✅ Insiders BUYING - Bullish"
    elif insider in ['BEARISH', 'SELLING']:
        insider_signal = "⚠️ Insiders SELLING"
    else:
        insider_signal = "Mixed activity"
    
    render_education_card(st_obj, 'whale_stock', {
        'short_pct': short_pct,
        'short_signal': short_signal,
        'pc_ratio': pc_ratio,
        'pc_signal': pc_signal,
        'insider': insider,
        'insider_signal': insider_signal,
        'inst_pct': inst_pct
    })


def render_rr_education(st_obj, risk_pct: float, tp1_roi: float) -> None:
    """Render Risk:Reward education with live values."""
    rr = tp1_roi / risk_pct if risk_pct > 0 else 0
    
    render_education_card(st_obj, 'risk_reward', {
        'risk': risk_pct,
        'reward': tp1_roi,
        'rr': rr
    })


def render_timeframe_education(st_obj, timeframe: str, mode_name: str, 
                                hold_time: str, confirm_tf: str) -> None:
    """Render timeframe education with context."""
    render_education_card(st_obj, 'timeframe', {
        'timeframe': timeframe,
        'mode_name': mode_name,
        'hold_time': hold_time,
        'confirm_tf': confirm_tf
    })


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EDUCATION SECTION RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_full_education_section(st_obj, context: Dict) -> None:
    """
    Render complete education section based on what's present in context.
    
    Args:
        st_obj: Streamlit object
        context: Dict containing:
            - signal: TradeSignal object
            - money_flow: Dict from calculate_money_flow()
            - smc_data: Dict from detect_smc()
            - whale_data: Optional Dict from get_institutional_analysis()
            - stock_data: Optional Dict from get_stock_institutional_analysis()
            - timeframe: str
            - mode_config: Dict with mode settings
            - is_crypto: bool
    """
    
    # Order Block education
    smc = context.get('smc_data', {})
    if smc.get('order_blocks', {}).get('at_bullish_ob') or smc.get('order_blocks', {}).get('at_bearish_ob'):
        render_education_card(st_obj, 'order_block')
    
    # FVG education
    if smc.get('fvg', {}).get('at_bullish_fvg') or smc.get('fvg', {}).get('at_bearish_fvg'):
        render_education_card(st_obj, 'fvg')
    
    # Money Flow education
    mf = context.get('money_flow', {})
    if mf.get('is_accumulating') or mf.get('is_distributing') or mf.get('mfi', 50) > 70 or mf.get('mfi', 50) < 30:
        render_money_flow_education(st_obj, mf)
    
    # Whale/Stock education
    is_crypto = context.get('is_crypto', True)
    if is_crypto:
        whale = context.get('whale_data')
        if whale and (whale.get('signals') or whale.get('open_interest', {}).get('change_24h', 0) != 0):
            render_whale_education_crypto(st_obj, whale)
    else:
        stock = context.get('stock_data')
        if stock and (stock.get('signals') or stock.get('score', 0) != 0):
            render_whale_education_stock(st_obj, stock)
    
    # R:R education (always show)
    signal = context.get('signal')
    if signal:
        tp1_roi = ((signal.tp1 - signal.entry) / signal.entry) * 100 if signal.entry > 0 else 0
        if signal.direction == 'SHORT':
            tp1_roi = ((signal.entry - signal.tp1) / signal.entry) * 100
        render_rr_education(st_obj, signal.risk_pct, abs(tp1_roi))
    
    # Timeframe education
    timeframe = context.get('timeframe', '4h')
    mode_config = context.get('mode_config', {})
    render_timeframe_education(
        st_obj, 
        timeframe,
        mode_config.get('name', 'Trading'),
        mode_config.get('hold_time', 'Varies'),
        mode_config.get('confirm_tf', '4h')
    )
