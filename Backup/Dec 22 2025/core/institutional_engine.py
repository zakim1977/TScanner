"""
🏦 UNIFIED INSTITUTIONAL ANALYSIS ENGINE
==========================================
A single source of truth for institutional/whale analysis across:
- Crypto (Binance Futures data)
- Stocks & ETFs (SEC, Yahoo Finance, FINRA)

KEY PRINCIPLES:
1. Institutions/Whales have better information and resources
2. They move FIRST, retail follows (or gets trapped)
3. Follow the smart money, fade the crowd
4. This data FEEDS INTO the main signal - no contradictions

Author: InvestorIQ
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime

# Import existing modules for data fetching - use relative imports
from .whale_institutional import (
    fetch_open_interest as get_open_interest, 
    fetch_funding_rate as get_funding_rate, 
    fetch_top_trader_ratio as get_top_trader_ls_ratio,
    fetch_long_short_ratio as get_global_ls_ratio, 
    fetch_taker_buy_sell as get_taker_volume, 
    fetch_price_change as get_price_change,
    interpret_oi_price,
    interpret_funding, 
    interpret_long_short
)
from .stock_institutional import (
    get_stock_institutional_analysis, get_insider_trading,
    get_short_interest, get_options_sentiment as get_options_flow, get_institutional_ownership
)


# ═══════════════════════════════════════════════════════════════════════════════
# EDUCATIONAL CONTENT - The Heart of Understanding
# ═══════════════════════════════════════════════════════════════════════════════

EDUCATION = {
    # ─────────────────────────────────────────────────────────────────────────
    # CRYPTO METRICS
    # ─────────────────────────────────────────────────────────────────────────
    'open_interest': {
        'name': 'Open Interest (OI)',
        'what': 'Total number of outstanding futures contracts that haven\'t been settled.',
        'why_matters': 'Shows how much money is committed to the market. Rising OI = new money entering, Falling OI = money leaving.',
        'how_to_read': {
            'rising_price_rising_oi': '🟢 BULLISH: New longs entering - Trend continuation likely',
            'rising_price_falling_oi': '🟡 WEAK: Short covering rally - Not new buying, may reverse',
            'falling_price_rising_oi': '🔴 BEARISH: New shorts entering - Trend continuation likely',
            'falling_price_falling_oi': '🟡 WEAK: Long liquidation - Forced selling, may reverse'
        },
        'edge': 'OI + Price together reveal WHETHER moves are driven by new conviction or position closing.',
        'normal_range': '±3% daily change is normal, >8% is significant',
        'icon': '📊'
    },
    
    'funding_rate': {
        'name': 'Funding Rate',
        'what': 'Fee exchanged between long and short holders every 8 hours on perpetual futures.',
        'why_matters': 'Shows which side is more crowded. Positive = longs pay shorts. Negative = shorts pay longs.',
        'how_to_read': {
            'high_positive': '⚠️ CONTRARIAN SHORT: Longs overleveraged (>0.1%), expect dump as longs get liquidated',
            'moderate_positive': '🟢 Slight bullish bias, longs willing to pay premium',
            'neutral': '⚪ Balanced market, no funding edge',
            'moderate_negative': '🔴 Slight bearish bias, shorts willing to pay premium',
            'high_negative': '⚠️ CONTRARIAN LONG: Shorts overleveraged (<-0.1%), expect pump as shorts get squeezed'
        },
        'edge': 'Extreme funding often precedes reversals. The crowded side usually loses.',
        'normal_range': '±0.01% is normal, ±0.05% is elevated, ±0.1% is extreme',
        'icon': '💰'
    },
    
    'top_traders': {
        'name': 'Top Trader Positioning',
        'what': 'Long/Short ratio of Binance\'s top traders by profit (the "smart money" in crypto).',
        'why_matters': 'These traders are consistently profitable. Their positioning shows where experienced money is betting.',
        'how_to_read': {
            'above_60': '🟢 BULLISH: Smart money is heavily long (>60%)',
            '55_to_60': '🟢 Moderately bullish positioning',
            '45_to_55': '⚪ Neutral - no clear directional bias',
            '40_to_45': '🔴 Moderately bearish positioning',
            'below_40': '🔴 BEARISH: Smart money is heavily short (<40%)'
        },
        'edge': 'When top traders and retail diverge, top traders are usually right.',
        'display': 'Shows as "60% L" meaning 60% long positions, 40% short',
        'icon': '🐋'
    },
    
    'retail_positioning': {
        'name': 'Retail Positioning',
        'what': 'Long/Short ratio of all retail traders on Binance Futures.',
        'why_matters': 'Retail traders are often wrong at extremes. They buy tops and sell bottoms.',
        'how_to_read': {
            'above_65': '⚠️ CONTRARIAN SHORT: Retail extremely bullish - Often a top signal',
            '55_to_65': '🟡 Retail leaning bullish',
            '45_to_55': '⚪ Neutral retail sentiment',
            '35_to_45': '🟡 Retail leaning bearish',
            'below_35': '⚠️ CONTRARIAN LONG: Retail extremely bearish - Often a bottom signal'
        },
        'edge': 'Fade retail at extremes. When everyone is bullish, who\'s left to buy?',
        'display': 'Shows as "47% L" meaning 47% long, 53% short',
        'icon': '🐑'
    },
    
    'whale_vs_retail': {
        'name': 'Whale vs Retail Divergence',
        'what': 'The difference between top trader positioning and retail positioning.',
        'why_matters': 'The biggest edge comes when smart money and retail disagree.',
        'how_to_read': {
            'whales_long_retail_short': '🟢 STRONG LONG: Whales buying what retail is selling',
            'whales_short_retail_long': '🔴 STRONG SHORT: Whales selling what retail is buying',
            'both_same_direction': '⚪ No divergence edge - Follow other signals'
        },
        'edge': 'Divergence trades have the highest win rate. Retail is the exit liquidity.',
        'icon': '⚔️'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # STOCK/ETF METRICS  
    # ─────────────────────────────────────────────────────────────────────────
    'insider_trading': {
        'name': 'Insider Trading (Form 4)',
        'what': 'Executives and directors buying/selling their own company stock, reported to SEC.',
        'why_matters': 'Insiders know the company best. When they buy with their OWN money, they\'re confident.',
        'how_to_read': {
            'cluster_buying': '🟢 VERY BULLISH: Multiple insiders buying = Strong confidence',
            'single_buy': '🟢 Bullish: Insider buy shows confidence',
            'routine_selling': '⚪ Normal: Often pre-planned (10b5-1 plans) - Ignore',
            'unusual_selling': '🔴 BEARISH: Multiple insiders selling outside plans = Concern'
        },
        'edge': 'Insider buying is the most reliable bullish signal for stocks. They rarely buy unless confident.',
        'data_source': 'SEC EDGAR Form 4 (free, real-time)',
        'icon': '👔'
    },
    
    'short_interest': {
        'name': 'Short Interest',
        'what': 'Percentage of tradeable shares currently sold short (betting on decline).',
        'why_matters': 'High short interest = potential short squeeze. Shorts must eventually buy back.',
        'how_to_read': {
            'above_20': '⚠️ SQUEEZE RISK: Very high (>20%) - Explosive upside possible',
            '10_to_20': '🟡 Elevated: Significant bearish betting, squeeze possible',
            '5_to_10': '⚪ Moderate: Normal skepticism',
            'below_5': '⚪ Low: Little bearish conviction'
        },
        'edge': 'High short interest + positive catalyst = violent short squeeze.',
        'data_source': 'FINRA (bi-weekly, delayed 2 weeks)',
        'icon': '📉'
    },
    
    'put_call_ratio': {
        'name': 'Put/Call Ratio',
        'what': 'Ratio of put options (bearish bets) to call options (bullish bets) traded.',
        'why_matters': 'Shows options market sentiment. High ratio = bearish, Low ratio = bullish.',
        'how_to_read': {
            'above_1.2': '🔴 BEARISH: More puts than calls - Market expects decline',
            '0.8_to_1.2': '⚪ Neutral: Balanced options activity',
            '0.5_to_0.8': '🟢 Bullish: More calls than puts - Market expects rise',
            'below_0.5': '⚠️ EXTREME BULLISH: May be contrarian bearish signal'
        },
        'edge': 'Extreme readings are contrarian signals. But moderate readings follow the trend.',
        'data_source': 'Yahoo Finance (free, daily)',
        'icon': '📊'
    },
    
    'institutional_ownership': {
        'name': 'Institutional Ownership',
        'what': 'Percentage of shares held by institutions (funds, banks, pensions).',
        'why_matters': 'High institutional ownership = professional validation. Low = under the radar.',
        'how_to_read': {
            'above_80': '✅ High conviction: Major institutions are invested',
            '50_to_80': '✅ Good coverage: Healthy institutional interest',
            '20_to_50': '🟡 Moderate: May be growing or declining',
            'below_20': '⚠️ Low: Either undiscovered gem or avoided for reason'
        },
        'edge': 'Look for INCREASING institutional ownership - they\'re building positions.',
        'data_source': '13F Filings (quarterly, delayed 45 days)',
        'icon': '🏦'
    },
    
    'options_flow': {
        'name': 'Unusual Options Activity',
        'what': 'Large or unusual options trades that may indicate informed positioning.',
        'why_matters': 'Big money often uses options for leverage. Unusual activity may signal knowledge.',
        'how_to_read': {
            'large_calls': '🟢 BULLISH: Big bets on upside',
            'large_puts': '🔴 BEARISH: Big bets on downside or hedging',
            'sweep_orders': '⚠️ URGENT: Buyer wanted immediate fills - High conviction'
        },
        'edge': 'Follow unusual activity, but verify with other signals. Not all big trades are informed.',
        'data_source': 'Free: Limited. Premium: Unusual Whales, FlowAlgo, Cheddar Flow',
        'premium_note': '💎 Premium data sources provide real-time unusual options flow',
        'icon': '🎯'
    }
}

# Premium data source recommendations
PREMIUM_DATA_SOURCES = {
    'options_flow': {
        'name': 'Options Flow Data',
        'free_limitations': 'Basic put/call ratio only, no real-time flow',
        'premium_providers': [
            {'name': 'Unusual Whales', 'url': 'unusualwhales.com', 'price': '$39-149/mo', 'best_for': 'Real-time unusual activity'},
            {'name': 'FlowAlgo', 'url': 'flowalgo.com', 'price': '$99/mo', 'best_for': 'Dark pool + options flow'},
            {'name': 'Cheddar Flow', 'url': 'cheddarflow.com', 'price': '$49-99/mo', 'best_for': 'Clean UI, alerts'}
        ]
    },
    'dark_pool': {
        'name': 'Dark Pool Data',
        'free_limitations': 'Not available free',
        'premium_providers': [
            {'name': 'FlowAlgo', 'url': 'flowalgo.com', 'price': '$99/mo', 'best_for': 'Comprehensive dark pool'},
            {'name': 'Quiver Quant', 'url': 'quiverquant.com', 'price': 'Free tier available', 'best_for': 'Alternative data'}
        ]
    },
    'congress_trading': {
        'name': 'Congress Trading',
        'free_limitations': 'Delayed data only',
        'premium_providers': [
            {'name': 'Quiver Quant', 'url': 'quiverquant.com', 'price': 'Free tier', 'best_for': 'Congress, Senate trades'},
            {'name': 'Capitol Trades', 'url': 'capitoltrades.com', 'price': 'Free', 'best_for': 'Clean UI'}
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class InstitutionalBias(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    LEAN_BULLISH = "LEAN_BULLISH"
    NEUTRAL = "NEUTRAL"
    LEAN_BEARISH = "LEAN_BEARISH"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"


@dataclass
class MetricReading:
    """A single metric with value and educational context"""
    name: str
    value: any
    display_value: str
    interpretation: str
    bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    strength: str  # 'STRONG', 'MODERATE', 'WEAK'
    color: str  # Hex color for UI
    icon: str
    education: Dict  # Full educational content
    score_impact: int  # How much this affects overall score (-20 to +20)


@dataclass 
class InstitutionalAnalysis:
    """Complete institutional analysis result"""
    symbol: str
    market_type: str  # 'CRYPTO', 'STOCK', 'ETF'
    timestamp: datetime
    
    # Overall verdict (THIS feeds into main signal)
    verdict: str  # 'BULLISH', 'BEARISH', 'NEUTRAL', 'WAIT'
    confidence: int  # 0-100
    score: int  # -100 to +100
    
    # Individual metrics with education
    metrics: List[MetricReading] = field(default_factory=list)
    
    # Key signals (top 3-5 insights)
    key_signals: List[str] = field(default_factory=list)
    
    # Recommended action
    action: str = ""
    action_reasoning: str = ""
    
    # Integration with main signal
    signal_adjustment: int = 0  # Points to add/subtract from main signal score
    should_wait: bool = False  # If True, suggests waiting regardless of technicals
    
    # Education summary
    main_insight: str = ""
    learn_more: List[Dict] = field(default_factory=list)
    
    # Data source info
    data_sources: List[str] = field(default_factory=list)
    premium_suggestion: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_crypto_institutional(symbol: str) -> InstitutionalAnalysis:
    """
    Analyze crypto institutional/whale data from Binance.
    Returns unified analysis with education.
    """
    result = InstitutionalAnalysis(
        symbol=symbol,
        market_type='CRYPTO',
        timestamp=datetime.now(),
        verdict='NEUTRAL',
        confidence=50,
        score=0,
        data_sources=['Binance Futures API']
    )
    
    score = 0
    metrics = []
    key_signals = []
    
    try:
        # 1. OPEN INTEREST + PRICE
        oi_data = get_open_interest(symbol)
        oi_change = oi_data.get('change_24h', 0)
        price_change = get_price_change(symbol)  # Fetch separately
        
        oi_interp = interpret_oi_price(oi_change, price_change)
        
        # Determine bias and score
        if oi_interp['signal'] == 'NEW_LONGS':
            oi_bias = 'BULLISH'
            oi_score = 20 if oi_interp['strength'] == 'STRONG' else 12
            oi_color = '#00d4aa'
        elif oi_interp['signal'] == 'NEW_SHORTS':
            oi_bias = 'BEARISH'
            oi_score = -20 if oi_interp['strength'] == 'STRONG' else -12
            oi_color = '#ff4444'
        elif oi_interp['signal'] == 'SHORT_COVERING':
            oi_bias = 'NEUTRAL'
            oi_score = 5  # Slight bullish but weak
            oi_color = '#ffcc00'
        elif oi_interp['signal'] == 'LONG_LIQUIDATION':
            oi_bias = 'NEUTRAL'
            oi_score = -5  # Slight bearish but may reverse
            oi_color = '#ffcc00'
        else:
            oi_bias = 'NEUTRAL'
            oi_score = 0
            oi_color = '#888888'
        
        score += oi_score
        
        metrics.append(MetricReading(
            name='Open Interest + Price',
            value={'oi_change': oi_change, 'price_change': price_change},
            display_value=f"OI: {oi_change:+.1f}% | Price: {price_change:+.1f}%",
            interpretation=oi_interp['interpretation'],
            bias=oi_bias,
            strength=oi_interp['strength'],
            color=oi_color,
            icon='📊',
            education=EDUCATION['open_interest'],
            score_impact=oi_score
        ))
        
        if oi_interp['strength'] in ['STRONG', 'MODERATE']:
            key_signals.append(f"{oi_interp['emoji']} {oi_interp['signal']}: {oi_interp['action']}")
        
        # 2. FUNDING RATE
        funding_data = get_funding_rate(symbol)
        funding_rate = funding_data.get('rate', 0)
        funding_pct = funding_rate * 100
        
        funding_interp = interpret_funding(funding_rate)
        
        if funding_interp['is_contrarian']:
            if 'LONG' in funding_interp['action']:
                fund_bias = 'BULLISH'
                fund_score = 15
                fund_color = '#00d4aa'
            else:
                fund_bias = 'BEARISH'
                fund_score = -15
                fund_color = '#ff4444'
            key_signals.append(f"⚠️ CONTRARIAN: {funding_interp['action']}")
        elif funding_interp['signal'] == 'BULLISH':
            fund_bias = 'BULLISH'
            fund_score = 5
            fund_color = '#00d4aa'
        elif funding_interp['signal'] == 'BEARISH':
            fund_bias = 'BEARISH'
            fund_score = -5
            fund_color = '#ff4444'
        else:
            fund_bias = 'NEUTRAL'
            fund_score = 0
            fund_color = '#888888'
        
        score += fund_score
        
        # Determine reading level for education
        if abs(funding_pct) > 0.1:
            reading_key = 'high_positive' if funding_pct > 0 else 'high_negative'
        elif abs(funding_pct) > 0.05:
            reading_key = 'moderate_positive' if funding_pct > 0 else 'moderate_negative'
        else:
            reading_key = 'neutral'
        
        metrics.append(MetricReading(
            name='Funding Rate',
            value=funding_rate,
            display_value=f"{funding_pct:.4f}%",
            interpretation=funding_interp['interpretation'],
            bias=fund_bias,
            strength='STRONG' if funding_interp['is_contrarian'] else 'MODERATE',
            color=fund_color,
            icon='💰',
            education=EDUCATION['funding_rate'],
            score_impact=fund_score
        ))
        
        # 3. TOP TRADERS (Whales)
        top_trader_data = get_top_trader_ls_ratio(symbol)
        top_long = top_trader_data.get('long_ratio', 50)
        
        if top_long > 60:
            whale_bias = 'BULLISH'
            whale_score = 15
            whale_color = '#00d4aa'
            whale_strength = 'STRONG'
            whale_interp = f"Smart money is {top_long:.0f}% LONG - Bullish positioning"
        elif top_long > 55:
            whale_bias = 'BULLISH'
            whale_score = 8
            whale_color = '#00d4aa'
            whale_strength = 'MODERATE'
            whale_interp = f"Smart money leaning LONG ({top_long:.0f}%)"
        elif top_long < 40:
            whale_bias = 'BEARISH'
            whale_score = -15
            whale_color = '#ff4444'
            whale_strength = 'STRONG'
            whale_interp = f"Smart money is {100-top_long:.0f}% SHORT - Bearish positioning"
        elif top_long < 45:
            whale_bias = 'BEARISH'
            whale_score = -8
            whale_color = '#ff4444'
            whale_strength = 'MODERATE'
            whale_interp = f"Smart money leaning SHORT ({100-top_long:.0f}%)"
        else:
            whale_bias = 'NEUTRAL'
            whale_score = 0
            whale_color = '#888888'
            whale_strength = 'WEAK'
            whale_interp = f"Smart money neutral ({top_long:.0f}% long)"
        
        score += whale_score
        
        metrics.append(MetricReading(
            name='Top Traders (Whales)',
            value=top_long,
            display_value=f"{top_long:.0f}% LONG",
            interpretation=whale_interp,
            bias=whale_bias,
            strength=whale_strength,
            color=whale_color,
            icon='🐋',
            education=EDUCATION['top_traders'],
            score_impact=whale_score
        ))
        
        if whale_strength == 'STRONG':
            direction = 'LONG' if whale_bias == 'BULLISH' else 'SHORT'
            key_signals.append(f"🐋 Top Traders {top_long:.0f}% {direction} - Follow the whales")
        
        # 4. RETAIL POSITIONING
        retail_data = get_global_ls_ratio(symbol)
        retail_long = retail_data.get('long_ratio', 50)
        
        # Retail is a CONTRARIAN indicator
        if retail_long > 65:
            retail_bias = 'BEARISH'  # Contrarian
            retail_score = -10
            retail_color = '#ff4444'
            retail_strength = 'STRONG'
            retail_interp = f"⚠️ Retail extremely bullish ({retail_long:.0f}% long) - Contrarian BEARISH"
        elif retail_long > 55:
            retail_bias = 'NEUTRAL'
            retail_score = -3
            retail_color = '#ffcc00'
            retail_strength = 'WEAK'
            retail_interp = f"Retail leaning bullish ({retail_long:.0f}% long)"
        elif retail_long < 35:
            retail_bias = 'BULLISH'  # Contrarian
            retail_score = 10
            retail_color = '#00d4aa'
            retail_strength = 'STRONG'
            retail_interp = f"⚠️ Retail extremely bearish ({100-retail_long:.0f}% short) - Contrarian BULLISH"
        elif retail_long < 45:
            retail_bias = 'NEUTRAL'
            retail_score = 3
            retail_color = '#ffcc00'
            retail_strength = 'WEAK'
            retail_interp = f"Retail leaning bearish ({100-retail_long:.0f}% short)"
        else:
            retail_bias = 'NEUTRAL'
            retail_score = 0
            retail_color = '#888888'
            retail_strength = 'WEAK'
            retail_interp = f"Retail neutral ({retail_long:.0f}% long)"
        
        score += retail_score
        
        metrics.append(MetricReading(
            name='Retail Positioning',
            value=retail_long,
            display_value=f"{retail_long:.0f}% LONG",
            interpretation=retail_interp,
            bias=retail_bias,
            strength=retail_strength,
            color=retail_color,
            icon='🐑',
            education=EDUCATION['retail_positioning'],
            score_impact=retail_score
        ))
        
        # 5. WHALE VS RETAIL DIVERGENCE (Most important!)
        ls_interp = interpret_long_short(retail_long, top_long)
        
        if ls_interp['edge'] == 'HIGH':
            if 'LONG' in ls_interp['action']:
                div_bias = 'BULLISH'
                div_score = 20
                div_color = '#00d4aa'
            else:
                div_bias = 'BEARISH'
                div_score = -20
                div_color = '#ff4444'
            key_signals.insert(0, f"⚔️ DIVERGENCE: {ls_interp['action']}")  # Top priority
        elif ls_interp['edge'] == 'MEDIUM':
            if 'LONG' in ls_interp['action']:
                div_bias = 'BULLISH'
                div_score = 10
                div_color = '#00d4aa'
            else:
                div_bias = 'BEARISH'
                div_score = -10
                div_color = '#ff4444'
            key_signals.append(f"🐋 {ls_interp['action']}")
        else:
            div_bias = 'NEUTRAL'
            div_score = 0
            div_color = '#888888'
        
        score += div_score
        
        metrics.append(MetricReading(
            name='Whale vs Retail Divergence',
            value={'whale': top_long, 'retail': retail_long},
            display_value=f"Whales: {top_long:.0f}%L | Retail: {retail_long:.0f}%L",
            interpretation=ls_interp['interpretation'],
            bias=div_bias,
            strength=ls_interp['edge'],
            color=div_color,
            icon='⚔️',
            education=EDUCATION['whale_vs_retail'],
            score_impact=div_score
        ))
        
        # Calculate final verdict
        if score >= 40:
            result.verdict = 'BULLISH'
            result.confidence = min(90, 60 + score // 2)
            result.action = 'LONG BIAS - Institutional data supports bullish positioning'
        elif score >= 20:
            result.verdict = 'BULLISH'
            result.confidence = min(75, 50 + score // 2)
            result.action = 'LEAN LONG - Moderate institutional bullish signals'
        elif score <= -40:
            result.verdict = 'BEARISH'
            result.confidence = min(90, 60 + abs(score) // 2)
            result.action = 'SHORT BIAS - Institutional data supports bearish positioning'
        elif score <= -20:
            result.verdict = 'BEARISH'
            result.confidence = min(75, 50 + abs(score) // 2)
            result.action = 'LEAN SHORT - Moderate institutional bearish signals'
        else:
            result.verdict = 'NEUTRAL'
            result.confidence = 50
            result.action = 'NO CLEAR EDGE - Wait for stronger institutional signals'
            result.should_wait = True
        
        result.score = score
        result.metrics = metrics
        result.key_signals = key_signals[:5]  # Top 5 signals
        result.signal_adjustment = score // 5  # Convert to adjustment points
        
        # Main insight for education
        if key_signals:
            result.main_insight = key_signals[0]
        else:
            result.main_insight = "No strong institutional signals detected"
        
        result.action_reasoning = _generate_crypto_reasoning(score, metrics)
        
    except Exception as e:
        result.verdict = 'UNKNOWN'
        result.confidence = 0
        result.action = f'Data unavailable: {str(e)[:50]}'
        result.main_insight = "Could not fetch institutional data"
    
    return result


def _generate_crypto_reasoning(score: int, metrics: List[MetricReading]) -> str:
    """Generate human-readable reasoning for the verdict"""
    bullish_factors = [m for m in metrics if m.bias == 'BULLISH' and m.strength in ['STRONG', 'MODERATE']]
    bearish_factors = [m for m in metrics if m.bias == 'BEARISH' and m.strength in ['STRONG', 'MODERATE']]
    
    parts = []
    
    if bullish_factors:
        names = [f.name for f in bullish_factors]
        parts.append(f"Bullish: {', '.join(names)}")
    
    if bearish_factors:
        names = [f.name for f in bearish_factors]
        parts.append(f"Bearish: {', '.join(names)}")
    
    if not parts:
        return "No strong signals from institutional data"
    
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK/ETF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_stock_institutional(symbol: str) -> InstitutionalAnalysis:
    """
    Analyze stock/ETF institutional data from SEC, Yahoo Finance, etc.
    Returns unified analysis with education.
    """
    result = InstitutionalAnalysis(
        symbol=symbol,
        market_type='STOCK',
        timestamp=datetime.now(),
        verdict='NEUTRAL',
        confidence=50,
        score=0,
        data_sources=['SEC EDGAR', 'Yahoo Finance', 'FINRA']
    )
    
    score = 0
    metrics = []
    key_signals = []
    
    try:
        # 1. INSIDER TRADING (Most reliable signal for stocks!)
        insider_data = get_insider_trading(symbol)
        
        net_activity = insider_data.get('net_shares', 0)
        total_buys = insider_data.get('total_buys', 0)
        total_sells = insider_data.get('total_sells', 0)
        sentiment = insider_data.get('sentiment', 'NEUTRAL')
        
        if sentiment in ['STRONG_BUY', 'BULLISH'] or total_buys >= 3:
            insider_bias = 'BULLISH'
            insider_score = 25 if total_buys >= 3 else 15
            insider_color = '#00d4aa'
            insider_strength = 'STRONG' if total_buys >= 3 else 'MODERATE'
            insider_interp = f"🟢 Insiders BUYING ({total_buys} buys) - Strong confidence signal"
            key_signals.append(f"👔 INSIDER BUYING: {total_buys} transactions - Bullish signal")
        elif sentiment in ['BEARISH', 'SELLING'] and total_sells >= 3:
            insider_bias = 'BEARISH'
            insider_score = -15
            insider_color = '#ff4444'
            insider_strength = 'MODERATE'
            insider_interp = f"🔴 Insiders SELLING ({total_sells} sells) - May indicate concern"
            key_signals.append(f"👔 INSIDER SELLING: {total_sells} transactions - Caution")
        else:
            insider_bias = 'NEUTRAL'
            insider_score = 0
            insider_color = '#888888'
            insider_strength = 'WEAK'
            insider_interp = f"No significant insider activity (Buys: {total_buys}, Sells: {total_sells})"
        
        score += insider_score
        
        metrics.append(MetricReading(
            name='Insider Trading',
            value={'buys': total_buys, 'sells': total_sells},
            display_value=f"Buys: {total_buys} | Sells: {total_sells}",
            interpretation=insider_interp,
            bias=insider_bias,
            strength=insider_strength,
            color=insider_color,
            icon='👔',
            education=EDUCATION['insider_trading'],
            score_impact=insider_score
        ))
        
        # 2. SHORT INTEREST
        short_data = get_short_interest(symbol)
        short_pct = short_data.get('short_pct_float', 0)
        squeeze_potential = short_data.get('squeeze_potential', 'LOW')
        
        if short_pct > 20:
            short_bias = 'BULLISH'  # High squeeze potential
            short_score = 15
            short_color = '#00d4aa'
            short_strength = 'STRONG'
            short_interp = f"⚠️ Very high short interest ({short_pct:.1f}%) - Squeeze potential!"
            key_signals.append(f"📉 SHORT SQUEEZE SETUP: {short_pct:.1f}% shorted")
        elif short_pct > 10:
            short_bias = 'NEUTRAL'
            short_score = 5
            short_color = '#ffcc00'
            short_strength = 'MODERATE'
            short_interp = f"Elevated short interest ({short_pct:.1f}%) - Watch for squeeze"
        elif short_pct > 5:
            short_bias = 'NEUTRAL'
            short_score = 0
            short_color = '#888888'
            short_strength = 'WEAK'
            short_interp = f"Moderate short interest ({short_pct:.1f}%)"
        else:
            short_bias = 'NEUTRAL'
            short_score = 0
            short_color = '#888888'
            short_strength = 'WEAK'
            short_interp = f"Low short interest ({short_pct:.1f}%)"
        
        score += short_score
        
        metrics.append(MetricReading(
            name='Short Interest',
            value=short_pct,
            display_value=f"{short_pct:.1f}% of float",
            interpretation=short_interp,
            bias=short_bias,
            strength=short_strength,
            color=short_color,
            icon='📉',
            education=EDUCATION['short_interest'],
            score_impact=short_score
        ))
        
        # 3. PUT/CALL RATIO
        options_data = get_options_flow(symbol)
        pc_ratio = options_data.get('put_call_ratio', 1.0)
        options_sentiment = options_data.get('sentiment', 'NEUTRAL')
        
        if pc_ratio > 1.2:
            pc_bias = 'BEARISH'
            pc_score = -10
            pc_color = '#ff4444'
            pc_strength = 'MODERATE'
            pc_interp = f"🔴 Put/Call {pc_ratio:.2f} - Options market bearish"
        elif pc_ratio < 0.5:
            pc_bias = 'NEUTRAL'  # Extreme bullish = contrarian
            pc_score = -5
            pc_color = '#ffcc00'
            pc_strength = 'MODERATE'
            pc_interp = f"⚠️ Put/Call {pc_ratio:.2f} - Extremely bullish, may be contrarian signal"
        elif pc_ratio < 0.8:
            pc_bias = 'BULLISH'
            pc_score = 10
            pc_color = '#00d4aa'
            pc_strength = 'MODERATE'
            pc_interp = f"🟢 Put/Call {pc_ratio:.2f} - Options market bullish"
        else:
            pc_bias = 'NEUTRAL'
            pc_score = 0
            pc_color = '#888888'
            pc_strength = 'WEAK'
            pc_interp = f"Put/Call {pc_ratio:.2f} - Neutral options sentiment"
        
        score += pc_score
        
        metrics.append(MetricReading(
            name='Put/Call Ratio',
            value=pc_ratio,
            display_value=f"{pc_ratio:.2f}",
            interpretation=pc_interp,
            bias=pc_bias,
            strength=pc_strength,
            color=pc_color,
            icon='📊',
            education=EDUCATION['put_call_ratio'],
            score_impact=pc_score
        ))
        
        # 4. INSTITUTIONAL OWNERSHIP
        inst_data = get_institutional_ownership(symbol)
        inst_pct = inst_data.get('institutional_pct', 0)
        inst_change = inst_data.get('change_qoq', 0)
        
        if inst_pct > 80:
            inst_bias = 'BULLISH'
            inst_score = 10
            inst_color = '#00d4aa'
            inst_strength = 'MODERATE'
            inst_interp = f"✅ High institutional ownership ({inst_pct:.0f}%) - Professional validation"
        elif inst_pct > 50:
            inst_bias = 'NEUTRAL'
            inst_score = 5
            inst_color = '#888888'
            inst_strength = 'WEAK'
            inst_interp = f"Good institutional coverage ({inst_pct:.0f}%)"
        elif inst_pct > 20:
            inst_bias = 'NEUTRAL'
            inst_score = 0
            inst_color = '#888888'
            inst_strength = 'WEAK'
            inst_interp = f"Moderate institutional interest ({inst_pct:.0f}%)"
        else:
            inst_bias = 'NEUTRAL'
            inst_score = -5
            inst_color = '#ffcc00'
            inst_strength = 'WEAK'
            inst_interp = f"⚠️ Low institutional ownership ({inst_pct:.0f}%) - Under radar or avoided"
        
        # Bonus for increasing ownership
        if inst_change > 5:
            inst_score += 5
            inst_interp += f" | 📈 Increasing (+{inst_change:.1f}% QoQ)"
            if inst_bias == 'NEUTRAL':
                inst_bias = 'BULLISH'
        elif inst_change < -5:
            inst_score -= 5
            inst_interp += f" | 📉 Decreasing ({inst_change:.1f}% QoQ)"
            if inst_bias == 'NEUTRAL':
                inst_bias = 'BEARISH'
        
        score += inst_score
        
        metrics.append(MetricReading(
            name='Institutional Ownership',
            value=inst_pct,
            display_value=f"{inst_pct:.0f}%",
            interpretation=inst_interp,
            bias=inst_bias,
            strength=inst_strength,
            color=inst_color,
            icon='🏦',
            education=EDUCATION['institutional_ownership'],
            score_impact=inst_score
        ))
        
        # 5. OPTIONS FLOW (Limited free data - suggest premium)
        metrics.append(MetricReading(
            name='Options Flow',
            value=None,
            display_value='Limited data',
            interpretation='Basic put/call ratio only. Real-time unusual activity requires premium data.',
            bias='NEUTRAL',
            strength='WEAK',
            color='#888888',
            icon='🎯',
            education=EDUCATION['options_flow'],
            score_impact=0
        ))
        
        # Add premium suggestion
        result.premium_suggestion = "💎 For real-time options flow, unusual activity alerts, and dark pool data, consider: Unusual Whales ($39/mo), FlowAlgo ($99/mo), or Cheddar Flow ($49/mo)"
        
        # Calculate final verdict
        if score >= 30:
            result.verdict = 'BULLISH'
            result.confidence = min(85, 55 + score // 2)
            result.action = 'LONG BIAS - Institutional data supports bullish positioning'
        elif score >= 15:
            result.verdict = 'BULLISH'
            result.confidence = min(70, 50 + score // 2)
            result.action = 'LEAN LONG - Moderate institutional bullish signals'
        elif score <= -30:
            result.verdict = 'BEARISH'
            result.confidence = min(85, 55 + abs(score) // 2)
            result.action = 'SHORT BIAS - Institutional data supports bearish positioning'
        elif score <= -15:
            result.verdict = 'BEARISH'
            result.confidence = min(70, 50 + abs(score) // 2)
            result.action = 'LEAN SHORT - Moderate institutional bearish signals'
        else:
            result.verdict = 'NEUTRAL'
            result.confidence = 50
            result.action = 'NO CLEAR EDGE - Wait for stronger signals or check premium data sources'
            result.should_wait = True
        
        result.score = score
        result.metrics = metrics
        result.key_signals = key_signals[:5]
        result.signal_adjustment = score // 4
        
        if key_signals:
            result.main_insight = key_signals[0]
        else:
            result.main_insight = "No strong institutional signals from free data sources"
        
        result.action_reasoning = _generate_stock_reasoning(score, metrics)
        
    except Exception as e:
        result.verdict = 'UNKNOWN'
        result.confidence = 0
        result.action = f'Data unavailable: {str(e)[:50]}'
    
    return result


def _generate_stock_reasoning(score: int, metrics: List[MetricReading]) -> str:
    """Generate human-readable reasoning for stock verdict"""
    bullish = [m for m in metrics if m.bias == 'BULLISH' and m.strength in ['STRONG', 'MODERATE']]
    bearish = [m for m in metrics if m.bias == 'BEARISH' and m.strength in ['STRONG', 'MODERATE']]
    
    parts = []
    if bullish:
        parts.append(f"Bullish: {', '.join([f.name for f in bullish])}")
    if bearish:
        parts.append(f"Bearish: {', '.join([f.name for f in bearish])}")
    
    return " | ".join(parts) if parts else "No strong signals from institutional data"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def get_institutional_analysis(symbol: str, market_type: str = None) -> InstitutionalAnalysis:
    """
    Main entry point - automatically detects market type and returns analysis.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL', 'SPY')
        market_type: Optional override ('CRYPTO', 'STOCK', 'ETF')
    
    Returns:
        InstitutionalAnalysis with unified format
    """
    # Auto-detect market type if not provided
    if market_type is None:
        if symbol.upper().endswith('USDT') or symbol.upper().endswith('USD') or symbol.upper().endswith('BUSD'):
            market_type = 'CRYPTO'
        elif symbol.upper() in ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'ARKK', 'XLF', 'XLE', 'XLK']:
            market_type = 'ETF'
        else:
            market_type = 'STOCK'
    
    if market_type == 'CRYPTO':
        return analyze_crypto_institutional(symbol)
    else:
        result = analyze_stock_institutional(symbol)
        result.market_type = market_type
        return result


def get_education_for_metric(metric_key: str) -> Dict:
    """Get full educational content for a specific metric"""
    return EDUCATION.get(metric_key, {})


def get_premium_providers(data_type: str) -> Dict:
    """Get premium data provider recommendations"""
    return PREMIUM_DATA_SOURCES.get(data_type, {})


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_institutional_into_signal(base_score: int, institutional: InstitutionalAnalysis) -> Tuple[int, str]:
    """
    Integrate institutional analysis into main signal score.
    
    Args:
        base_score: The technical analysis score (0-100)
        institutional: InstitutionalAnalysis result
    
    Returns:
        Tuple of (adjusted_score, explanation)
    """
    adjustment = institutional.signal_adjustment
    
    # Cap adjustment to ±15 points
    adjustment = max(-15, min(15, adjustment))
    
    new_score = base_score + adjustment
    new_score = max(0, min(100, new_score))
    
    if adjustment > 0:
        explanation = f"📈 +{adjustment} from institutional data ({institutional.verdict})"
    elif adjustment < 0:
        explanation = f"📉 {adjustment} from institutional data ({institutional.verdict})"
    else:
        explanation = "⚪ No adjustment from institutional data"
    
    return new_score, explanation


def should_override_signal(institutional: InstitutionalAnalysis) -> Tuple[bool, str]:
    """
    Check if institutional data should override technical signal.
    
    Returns:
        Tuple of (should_override, reason)
    """
    # Strong divergence signals should raise caution
    if institutional.score >= 40 or institutional.score <= -40:
        if institutional.score > 0:
            return False, f"✅ Strong institutional support ({institutional.verdict})"
        else:
            return True, f"⚠️ Strong institutional warning - Consider waiting despite technicals"
    
    if institutional.should_wait:
        return False, "⚪ No strong institutional edge - Rely on technicals"
    
    return False, "✅ Institutional data aligns with technicals"


# ═══════════════════════════════════════════════════════════════════════════════
# UI RENDERING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def render_metric_with_education(metric: MetricReading) -> Dict:
    """
    Prepare metric data for UI rendering with full educational context.
    
    Returns dict ready for Streamlit rendering.
    """
    edu = metric.education
    
    return {
        'name': metric.name,
        'value': metric.display_value,
        'interpretation': metric.interpretation,
        'color': metric.color,
        'icon': metric.icon,
        'bias': metric.bias,
        'strength': metric.strength,
        'score_impact': metric.score_impact,
        
        # Educational content
        'education': {
            'what': edu.get('what', ''),
            'why_matters': edu.get('why_matters', ''),
            'how_to_read': edu.get('how_to_read', {}),
            'edge': edu.get('edge', ''),
            'normal_range': edu.get('normal_range', ''),
            'data_source': edu.get('data_source', ''),
            'premium_note': edu.get('premium_note', '')
        }
    }
