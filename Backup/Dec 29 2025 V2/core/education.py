"""
InvestorIQ - Unified Education Module
=====================================
Provides educational explanations for all trading concepts:
- Smart Money Concepts (SMC): Order Blocks, FVG, Liquidity
- Money Flow: MFI, CMF, OBV
- VWAP: Institutional trading levels
- Price Action: Support/Resistance, Trend Structure

This module is used across Scanner, Monitor, and Single Analysis
to provide consistent education while trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT DEFINITIONS (Static Education)
# ═══════════════════════════════════════════════════════════════════════════════

CONCEPT_DEFINITIONS = {
    # ─────────────────────────────────────────────────────────────────────────
    # SMART MONEY CONCEPTS
    # ─────────────────────────────────────────────────────────────────────────
    "order_block": {
        "name": "Order Block (OB)",
        "emoji": "🏦",
        "short": "Institutional entry zone - where big players bought/sold",
        "definition": """
An **Order Block** is the last opposing candle before a strong impulsive move.

**Bullish OB:** Last red candle before strong green move up
**Bearish OB:** Last green candle before strong red move down

**Why it works:** 
Institutions can't fill large orders at once. They leave "unfilled orders" at these levels.
When price returns, they defend their positions → price bounces.

**How to trade:**
- LONG: Buy when price returns to Bullish OB
- SHORT: Sell when price returns to Bearish OB
- Stop Loss: Just beyond the OB (if it breaks, setup invalid)
""",
        "signal_bullish": "Price at Bullish Order Block - Institutional demand zone",
        "signal_bearish": "Price at Bearish Order Block - Institutional supply zone"
    },
    
    "fvg": {
        "name": "Fair Value Gap (FVG)",
        "emoji": "📊",
        "short": "Price imbalance zone - gaps that tend to get filled",
        "definition": """
A **Fair Value Gap** (or Imbalance) is a 3-candle pattern where the middle candle's body 
doesn't overlap with the wicks of the surrounding candles.

**Bullish FVG:** Gap created during strong up-move (below price)
**Bearish FVG:** Gap created during strong down-move (above price)

**Why it works:**
Markets seek "fair value". When price moves too fast, it creates an imbalance.
Price tends to return to fill these gaps before continuing.

**How to trade:**
- Wait for price to return to FVG
- Enter when FVG gets "mitigated" (touched)
- Often combines with Order Blocks for high-probability setups
""",
        "signal_bullish": "Price at Bullish FVG - Expect fill then continuation up",
        "signal_bearish": "Price at Bearish FVG - Expect fill then continuation down"
    },
    
    "liquidity": {
        "name": "Liquidity",
        "emoji": "💧",
        "short": "Clusters of stop losses that get hunted",
        "definition": """
**Liquidity** refers to stop loss orders clustered at obvious levels 
(round numbers, swing highs/lows, trendlines).

**Types:**
- **Buy-side Liquidity (BSL):** Stops above highs - gets swept in upward hunts
- **Sell-side Liquidity (SSL):** Stops below lows - gets swept in downward hunts

**Why it matters:**
Smart money needs liquidity to fill large orders. They engineer "stop hunts" 
to grab retail stops before the real move.

**The Trap:**
1. Price breaks obvious level (your stop gets hit)
2. You exit at a loss
3. Price immediately reverses
4. Smart money bought your stop, now rides the real move

**How to avoid:**
- Place stops below Order Blocks, not obvious levels
- Wait for liquidity sweep BEFORE entering
- Use wider stops in less obvious places
""",
        "signal_bullish": "Sell-side liquidity swept - Smart money grabbed stops, likely reversal up",
        "signal_bearish": "Buy-side liquidity swept - Smart money grabbed stops, likely reversal down"
    },
    
    "market_structure": {
        "name": "Market Structure",
        "emoji": "📈",
        "short": "Higher highs/lows (uptrend) or lower highs/lows (downtrend)",
        "definition": """
**Market Structure** is the pattern of highs and lows that defines the trend.

**Bullish Structure:**
- Higher Highs (HH): Each peak higher than the last
- Higher Lows (HL): Each dip higher than the last
→ Buyers in control

**Bearish Structure:**
- Lower Highs (LH): Each peak lower than the last  
- Lower Lows (LL): Each dip lower than the last
→ Sellers in control

**Break of Structure (BOS):**
When price breaks a significant high/low, it confirms trend continuation.

**Change of Character (ChoCH):**
When price breaks structure in the OPPOSITE direction - signals potential reversal.
""",
        "signal_bullish": "Bullish structure - Higher highs and higher lows",
        "signal_bearish": "Bearish structure - Lower highs and lower lows"
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # MONEY FLOW INDICATORS
    # ─────────────────────────────────────────────────────────────────────────
    "mfi": {
        "name": "Money Flow Index (MFI)",
        "emoji": "💰",
        "short": "RSI with volume - shows buying/selling pressure",
        "definition": """
**MFI (Money Flow Index)** is like RSI but incorporates volume.
It measures the flow of money INTO or OUT OF an asset.

**Formula:** Uses typical price × volume to calculate money flow

**Interpretation:**
- **MFI > 80:** Overbought - Too much buying, expect pullback
- **MFI < 20:** Oversold - Too much selling, expect bounce
- **MFI 40-60:** Neutral zone

**Why it's better than RSI:**
RSI only looks at price. MFI adds VOLUME confirmation.
High volume moves are more significant than low volume moves.

**Divergences:**
- Price makes new high, MFI makes lower high → Bearish divergence (weakness)
- Price makes new low, MFI makes higher low → Bullish divergence (strength)
""",
        "levels": {"oversold": 20, "overbought": 80, "neutral_low": 40, "neutral_high": 60}
    },
    
    "cmf": {
        "name": "Chaikin Money Flow (CMF)",
        "emoji": "📊",
        "short": "Measures accumulation vs distribution pressure",
        "definition": """
**CMF (Chaikin Money Flow)** measures the amount of Money Flow Volume 
over a specific period (usually 20-21 days).

**Range:** -1 to +1 (usually between -0.5 and +0.5)

**Interpretation:**
- **CMF > 0:** Accumulation - Buyers are dominant
- **CMF < 0:** Distribution - Sellers are dominant
- **CMF > 0.25:** Strong buying pressure
- **CMF < -0.25:** Strong selling pressure

**Key insight:**
CMF shows WHERE in the candle's range the close occurred, weighted by volume.
- Close near high + high volume = Strong accumulation
- Close near low + high volume = Strong distribution

**Trading signals:**
- CMF crosses above 0 → Bullish (start of accumulation)
- CMF crosses below 0 → Bearish (start of distribution)
- Divergence with price → Potential reversal coming
""",
        "levels": {"strong_buy": 0.25, "buy": 0, "sell": 0, "strong_sell": -0.25}
    },
    
    "obv": {
        "name": "On-Balance Volume (OBV)",
        "emoji": "📈",
        "short": "Cumulative volume showing smart money direction",
        "definition": """
**OBV (On-Balance Volume)** is a cumulative indicator that adds volume on up days 
and subtracts volume on down days.

**Calculation:**
- If close > previous close: OBV = Previous OBV + Volume
- If close < previous close: OBV = Previous OBV - Volume

**Key Principle:** Volume precedes price

**Interpretation:**
- **OBV rising + Price rising:** Healthy uptrend confirmed
- **OBV falling + Price falling:** Healthy downtrend confirmed
- **OBV rising + Price flat/falling:** Accumulation! Smart money buying
- **OBV falling + Price flat/rising:** Distribution! Smart money selling

**The "Smart Money" tell:**
When OBV diverges from price, smart money is positioning BEFORE the move.
- OBV up, Price flat → They're buying, expect breakout UP
- OBV down, Price flat → They're selling, expect breakdown DOWN
""",
        "signal_bullish": "OBV rising - Volume confirming upward pressure",
        "signal_bearish": "OBV falling - Volume confirming downward pressure"
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # VWAP
    # ─────────────────────────────────────────────────────────────────────────
    "vwap": {
        "name": "VWAP (Volume Weighted Average Price)",
        "emoji": "⚖️",
        "short": "Average price weighted by volume - institutional benchmark",
        "definition": """
**VWAP** is the average price an asset has traded at throughout the day, 
weighted by volume. It's the #1 benchmark for institutional traders.

**Why institutions use VWAP:**
- Large orders are judged against VWAP
- "Did we buy below VWAP?" = Good execution
- "Did we sell above VWAP?" = Good execution

**Trading interpretation:**
- **Price > VWAP:** Bullish - Buyers paying above average
- **Price < VWAP:** Bearish - Sellers accepting below average

**VWAP as Support/Resistance:**
- In uptrends: VWAP acts as SUPPORT (buy dips to VWAP)
- In downtrends: VWAP acts as RESISTANCE (sell rallies to VWAP)

**VWAP Bands (Standard Deviations):**
- +1σ/-1σ: Normal trading range
- +2σ/-2σ: Extended/Oversold, expect mean reversion
- Beyond ±2σ: Extreme, high probability of snap-back

**Pro tip:**
First test of VWAP often holds. Second test is weaker. Third test often breaks.
""",
        "signal_bullish": "Price above VWAP - Buyers in control, dips to VWAP are buys",
        "signal_bearish": "Price below VWAP - Sellers in control, rallies to VWAP are sells"
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # TREND INDICATORS
    # ─────────────────────────────────────────────────────────────────────────
    "ema": {
        "name": "EMA (Exponential Moving Average)",
        "emoji": "〰️",
        "short": "Smoothed price average giving more weight to recent prices",
        "definition": """
**EMA** gives more weight to recent prices, making it more responsive than SMA.

**Common EMAs:**
- **EMA 9:** Very short-term, scalping
- **EMA 20:** Short-term trend
- **EMA 50:** Medium-term trend
- **EMA 200:** Long-term trend (institutional benchmark)

**EMA Stacking (Golden Setup):**
- **Bullish:** Price > EMA 9 > EMA 20 > EMA 50 > EMA 200
- **Bearish:** Price < EMA 9 < EMA 20 < EMA 50 < EMA 200

**Trading signals:**
- Price crosses above EMA → Bullish
- Price crosses below EMA → Bearish
- EMA 9 crosses EMA 20 → Short-term trend change
- EMA 50 crosses EMA 200 → "Golden Cross" (bullish) or "Death Cross" (bearish)

**Dynamic Support/Resistance:**
In trends, EMAs act as support (uptrend) or resistance (downtrend).
EMA 20 and 50 are most commonly used for entries.
""",
        "signal_bullish": "EMAs stacked bullish (9>20>50) - Strong uptrend",
        "signal_bearish": "EMAs stacked bearish (9<20<50) - Strong downtrend"
    },
    
    "rsi": {
        "name": "RSI (Relative Strength Index)",
        "emoji": "📉",
        "short": "Momentum oscillator showing overbought/oversold conditions",
        "definition": """
**RSI** measures the speed and magnitude of recent price changes.

**Range:** 0 to 100

**Traditional levels:**
- **RSI > 70:** Overbought - May pull back
- **RSI < 30:** Oversold - May bounce
- **RSI 40-60:** Neutral zone

**Reality check:**
In strong trends, RSI can stay overbought/oversold for extended periods.
- In uptrends: RSI often stays 40-80 range
- In downtrends: RSI often stays 20-60 range

**Divergences (Most Powerful Signal):**
- **Bullish Divergence:** Price lower low, RSI higher low → Reversal UP likely
- **Bearish Divergence:** Price higher high, RSI lower high → Reversal DOWN likely

**Hidden Divergences (Trend Continuation):**
- **Hidden Bullish:** Price higher low, RSI lower low → Uptrend continues
- **Hidden Bearish:** Price lower high, RSI higher high → Downtrend continues
""",
        "levels": {"oversold": 30, "overbought": 70}
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# STORY GENERATOR - "How We Got Here"
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PriceStory:
    """Narrative of how price reached current level"""
    timeframe: str
    current_price: float
    story_parts: List[str] = field(default_factory=list)
    key_events: List[Dict] = field(default_factory=list)
    overall_narrative: str = ""


def generate_price_story(df: pd.DataFrame, timeframe: str, 
                         smc_data: Dict, money_flow: Dict, 
                         vwap_data: Dict = None) -> PriceStory:
    """
    Generate a narrative of how price reached current levels
    
    This tells the "story" of the chart - what happened and why
    """
    story = PriceStory(
        timeframe=timeframe,
        current_price=df['Close'].iloc[-1]
    )
    
    # Analyze recent price action
    lookback_20 = df.tail(20)
    lookback_50 = df.tail(50)
    
    current_price = df['Close'].iloc[-1]
    price_20_ago = df['Close'].iloc[-20] if len(df) >= 20 else df['Close'].iloc[0]
    price_50_ago = df['Close'].iloc[-50] if len(df) >= 50 else df['Close'].iloc[0]
    
    change_20 = ((current_price - price_20_ago) / price_20_ago) * 100
    change_50 = ((current_price - price_50_ago) / price_50_ago) * 100
    
    high_50 = lookback_50['High'].max()
    low_50 = lookback_50['Low'].min()
    range_position = ((current_price - low_50) / (high_50 - low_50)) * 100 if high_50 != low_50 else 50
    
    # ─────────────────────────────────────────────────────────────────────────
    # Build the story
    # ─────────────────────────────────────────────────────────────────────────
    
    # Part 1: Overall trend context
    if change_50 > 15:
        story.story_parts.append(f"📈 **Strong uptrend**: Price is up {change_50:.1f}% over the last 50 candles on {timeframe}.")
    elif change_50 > 5:
        story.story_parts.append(f"📈 **Uptrend**: Price has gained {change_50:.1f}% over the last 50 candles.")
    elif change_50 < -15:
        story.story_parts.append(f"📉 **Strong downtrend**: Price is down {abs(change_50):.1f}% over the last 50 candles on {timeframe}.")
    elif change_50 < -5:
        story.story_parts.append(f"📉 **Downtrend**: Price has dropped {abs(change_50):.1f}% over the last 50 candles.")
    else:
        story.story_parts.append(f"↔️ **Ranging/Consolidation**: Price is relatively flat ({change_50:+.1f}%) over 50 candles.")
    
    # Part 2: Recent momentum (last 20 candles)
    if change_20 > 5 and change_50 > 0:
        story.story_parts.append(f"🚀 **Accelerating up**: Recent momentum (+{change_20:.1f}% in 20 candles) confirms the trend.")
    elif change_20 < -5 and change_50 < 0:
        story.story_parts.append(f"📉 **Accelerating down**: Recent momentum ({change_20:.1f}% in 20 candles) confirms the downtrend.")
    elif change_20 > 3 and change_50 < 0:
        story.story_parts.append(f"🔄 **Potential reversal**: Price bouncing (+{change_20:.1f}% recently) within overall downtrend.")
        story.key_events.append({"type": "reversal_signal", "direction": "bullish"})
    elif change_20 < -3 and change_50 > 0:
        story.story_parts.append(f"⚠️ **Pullback**: Price pulling back ({change_20:.1f}% recently) within overall uptrend.")
        story.key_events.append({"type": "pullback", "direction": "opportunity"})
    
    # Part 3: Position in range
    if range_position > 80:
        story.story_parts.append(f"📍 **Near range high**: Currently at {range_position:.0f}% of 50-candle range. Extended.")
    elif range_position < 20:
        story.story_parts.append(f"📍 **Near range low**: Currently at {range_position:.0f}% of 50-candle range. Potentially oversold.")
    else:
        story.story_parts.append(f"📍 **Mid-range**: Currently at {range_position:.0f}% of recent trading range.")
    
    # Part 4: SMC Context
    if smc_data:
        ob = smc_data.get('order_blocks', {})
        fvg = smc_data.get('fvg', {})
        
        if ob.get('at_bullish_ob'):
            story.story_parts.append("🏦 **At Demand Zone**: Price returned to a Bullish Order Block where institutions previously bought. This is a high-probability long entry.")
            story.key_events.append({"type": "order_block", "direction": "bullish"})
        elif ob.get('at_bearish_ob'):
            story.story_parts.append("🏦 **At Supply Zone**: Price returned to a Bearish Order Block where institutions previously sold. Expect resistance.")
            story.key_events.append({"type": "order_block", "direction": "bearish"})
        
        if fvg.get('at_bullish_fvg'):
            story.story_parts.append("📊 **FVG Fill**: Price mitigated (filled) a bullish Fair Value Gap. Imbalance corrected, may continue up.")
        elif fvg.get('at_bearish_fvg'):
            story.story_parts.append("📊 **FVG Fill**: Price mitigated (filled) a bearish Fair Value Gap. Imbalance corrected, may continue down.")
    
    # Part 5: Money Flow Context
    if money_flow:
        mfi = money_flow.get('mfi', 50)
        cmf = money_flow.get('cmf', 0)
        is_accumulating = money_flow.get('is_accumulating', False)
        is_distributing = money_flow.get('is_distributing', False)
        
        if is_accumulating:
            story.story_parts.append(f"💰 **Accumulation Phase**: Smart money is buying. MFI: {mfi:.0f}, CMF: {cmf:.3f} (positive). Volume supports upward move.")
            story.key_events.append({"type": "accumulation", "strength": "strong" if cmf > 0.1 else "moderate"})
        elif is_distributing:
            story.story_parts.append(f"💸 **Distribution Phase**: Smart money is selling. MFI: {mfi:.0f}, CMF: {cmf:.3f} (negative). Selling pressure present.")
            story.key_events.append({"type": "distribution", "strength": "strong" if cmf < -0.1 else "moderate"})
        elif mfi < 25:
            story.story_parts.append(f"📉 **Oversold**: MFI at {mfi:.0f} indicates exhausted selling. Bounce likely.")
        elif mfi > 75:
            story.story_parts.append(f"📈 **Overbought**: MFI at {mfi:.0f} indicates exhausted buying. Pullback possible.")
    
    # Part 6: VWAP Context
    if vwap_data:
        vwap_pos = vwap_data.get('position', 'NEUTRAL')
        vwap_text = vwap_data.get('position_text', '')
        vwap_dist = vwap_data.get('distance_pct', 0)
        
        story.story_parts.append(f"⚖️ **VWAP Status**: {vwap_text} ({vwap_dist:+.1f}% from VWAP)")
        
        if vwap_pos in ['FAR_ABOVE', 'FAR_BELOW']:
            story.key_events.append({"type": "vwap_extreme", "direction": "mean_reversion_likely"})
    
    # Build overall narrative
    bullish_events = len([e for e in story.key_events if e.get('direction') in ['bullish', 'opportunity']])
    bearish_events = len([e for e in story.key_events if e.get('direction') == 'bearish'])
    
    if bullish_events > bearish_events:
        story.overall_narrative = "🟢 **Overall**: Bullish factors outweigh bearish. Conditions favor long positions."
    elif bearish_events > bullish_events:
        story.overall_narrative = "🔴 **Overall**: Bearish factors outweigh bullish. Caution advised for longs."
    else:
        story.overall_narrative = "⚪ **Overall**: Mixed signals. Wait for clearer direction or trade with caution."
    
    return story


# ═══════════════════════════════════════════════════════════════════════════════
# EDUCATION DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_concept_education(concept_key: str) -> Dict:
    """Get education content for a specific concept"""
    return CONCEPT_DEFINITIONS.get(concept_key, {})


def get_indicator_explanation(indicator: str, value: float, context: str = "") -> str:
    """
    Generate contextual explanation for an indicator value
    
    Args:
        indicator: 'mfi', 'cmf', 'rsi', 'obv', 'vwap'
        value: Current value of the indicator
        context: Additional context (e.g., 'falling', 'rising')
    
    Returns:
        Educational explanation string
    """
    explanations = {
        'mfi': {
            'oversold': f"MFI at {value:.0f} is **oversold** (<20). This means selling pressure is exhausted. Historically, this is where smart money starts buying. Watch for reversal signals.",
            'overbought': f"MFI at {value:.0f} is **overbought** (>80). Buying pressure is exhausted. Price may pull back or consolidate. Don't chase here.",
            'neutral': f"MFI at {value:.0f} is in **neutral territory** (20-80). No extreme reading. Follow the trend."
        },
        'cmf': {
            'strong_buy': f"CMF at {value:.3f} shows **strong accumulation** (>0.25). Institutions are aggressively buying. This is the kind of volume-backed move that has legs.",
            'buy': f"CMF at {value:.3f} is **positive**, indicating net buying pressure. Money is flowing into this asset. Bullish.",
            'sell': f"CMF at {value:.3f} is **negative**, indicating net selling pressure. Money is flowing out. Be cautious with longs.",
            'strong_sell': f"CMF at {value:.3f} shows **strong distribution** (<-0.25). Institutions are aggressively selling. Avoid longs until this reverses."
        },
        'rsi': {
            'oversold': f"RSI at {value:.0f} is **oversold** (<30). Price has fallen significantly. Bounce likely, but confirm with other signals.",
            'overbought': f"RSI at {value:.0f} is **overbought** (>70). Price has risen significantly. Pullback possible.",
            'neutral': f"RSI at {value:.0f} is **neutral**. No extreme momentum reading."
        }
    }
    
    if indicator == 'mfi':
        if value < 20:
            return explanations['mfi']['oversold']
        elif value > 80:
            return explanations['mfi']['overbought']
        else:
            return explanations['mfi']['neutral']
    
    elif indicator == 'cmf':
        if value > 0.25:
            return explanations['cmf']['strong_buy']
        elif value > 0:
            return explanations['cmf']['buy']
        elif value > -0.25:
            return explanations['cmf']['sell']
        else:
            return explanations['cmf']['strong_sell']
    
    elif indicator == 'rsi':
        if value < 30:
            return explanations['rsi']['oversold']
        elif value > 70:
            return explanations['rsi']['overbought']
        else:
            return explanations['rsi']['neutral']
    
    return f"{indicator.upper()}: {value}"


def get_obv_explanation(obv_rising: bool, price_rising: bool = None, cmf_positive: bool = None) -> str:
    """
    Generate contextual OBV explanation based on OBV direction and price context
    
    Args:
        obv_rising: Is OBV trending up?
        price_rising: Is price trending up? (for divergence detection)
        cmf_positive: Is CMF positive? (for confirmation)
    
    Returns:
        Contextual explanation string
    """
    if obv_rising:
        if price_rising is True:
            # OBV up + Price up = Confirmation (bullish)
            return ("OBV is **rising** 📈 while price is also rising. This is **healthy confirmation** - "
                   "volume supports the uptrend. The move has conviction behind it. "
                   "**Conclusion:** Bullish - trend is supported by volume. Hold or add on dips.")
        elif price_rising is False:
            # OBV up + Price down = Bullish Divergence
            return ("OBV is **rising** 📈 but price is falling. This is a **bullish divergence** - "
                   "smart money is accumulating while retail sells. Institutions are quietly buying. "
                   "**Conclusion:** Watch for reversal UP. This often precedes a bounce.")
        else:
            # OBV up, price unknown
            return ("OBV is **rising** 📈 indicating net buying pressure over recent candles. "
                   "More volume occurred on up-moves than down-moves. "
                   "**Conclusion:** Bullish - buyers are in control based on volume.")
    else:
        if price_rising is True:
            # OBV down + Price up = Bearish Divergence (warning!)
            return ("OBV is **falling** 📉 but price is rising. This is a **bearish divergence** ⚠️ - "
                   "price is rising on decreasing volume conviction. Smart money may be distributing. "
                   "**Conclusion:** Be cautious! The rally may not be sustainable. Consider taking profits.")
        elif price_rising is False:
            # OBV down + Price down = Confirmation (bearish)
            return ("OBV is **falling** 📉 while price is also falling. This **confirms the downtrend** - "
                   "volume supports the selling pressure. The move has conviction. "
                   "**Conclusion:** Bearish - avoid longs until OBV shows accumulation.")
        else:
            # OBV down, price unknown
            return ("OBV is **falling** 📉 indicating net selling pressure over recent candles. "
                   "More volume occurred on down-moves than up-moves. "
                   "**Conclusion:** Bearish warning - sellers are more aggressive based on volume.")


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME GRADE CALCULATION WITH BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GradeBreakdown:
    """Real-time grade calculation with full breakdown"""
    total_score: int
    grade: str
    grade_emoji: str
    factors: List[Dict]  # List of {name, points, emoji, description}
    bullish_points: int
    bearish_points: int
    summary: str
    
    
def calculate_realtime_grade(money_flow: Dict, smc_data: Dict, 
                             trend_data: Dict = None, vwap_data: Dict = None,
                             price_data: Dict = None) -> GradeBreakdown:
    """
    Calculate real-time grade with full breakdown of contributing factors.
    
    This shows exactly WHY the grade is what it is, so users understand
    the trade quality.
    
    Returns:
        GradeBreakdown with score, grade letter, and all contributing factors
    """
    factors = []
    bullish_points = 0
    bearish_points = 0
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. MONEY FLOW ANALYSIS (Max +/- 35 points)
    # ═══════════════════════════════════════════════════════════════════
    
    if money_flow:
        cmf = money_flow.get('cmf', 0)
        mfi = money_flow.get('mfi', 50)
        obv_rising = money_flow.get('obv_rising', False)
        
        # CMF (most important - institutional flow)
        if cmf > 0.25:
            points = 20
            bullish_points += points
            factors.append({
                'name': 'CMF (Strong Accumulation)',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': f'CMF at {cmf:.3f} - Institutions heavily buying'
            })
        elif cmf > 0.10:
            points = 12
            bullish_points += points
            factors.append({
                'name': 'CMF (Accumulation)',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': f'CMF at {cmf:.3f} - Net buying pressure'
            })
        elif cmf > 0:
            points = 5
            bullish_points += points
            factors.append({
                'name': 'CMF (Slight Inflow)',
                'points': f'+{points}',
                'emoji': '🟡',
                'description': f'CMF at {cmf:.3f} - Mild buying pressure'
            })
        elif cmf > -0.10:
            points = 5
            bearish_points += points
            factors.append({
                'name': 'CMF (Slight Outflow)',
                'points': f'-{points}',
                'emoji': '🟡',
                'description': f'CMF at {cmf:.3f} - Mild selling pressure'
            })
        elif cmf > -0.25:
            points = 12
            bearish_points += points
            factors.append({
                'name': 'CMF (Distribution)',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': f'CMF at {cmf:.3f} - Net selling pressure'
            })
        else:
            points = 20
            bearish_points += points
            factors.append({
                'name': 'CMF (Heavy Distribution)',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': f'CMF at {cmf:.3f} - Institutions heavily selling'
            })
        
        # OBV
        if obv_rising:
            points = 10
            bullish_points += points
            factors.append({
                'name': 'OBV Rising',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': 'Volume confirming upward movement'
            })
        else:
            points = 10
            bearish_points += points
            factors.append({
                'name': 'OBV Falling',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': 'Volume confirming downward pressure'
            })
        
        # MFI extremes
        if mfi < 20:
            points = 10
            bullish_points += points
            factors.append({
                'name': 'MFI Oversold',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': f'MFI at {mfi:.0f} - Bounce likely'
            })
        elif mfi > 80:
            points = 10
            bearish_points += points
            factors.append({
                'name': 'MFI Overbought',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': f'MFI at {mfi:.0f} - Pullback likely'
            })
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. SMC ANALYSIS (Max +/- 30 points)
    # ═══════════════════════════════════════════════════════════════════
    
    if smc_data:
        ob = smc_data.get('order_blocks', {})
        fvg = smc_data.get('fvg', {})
        
        # Order Blocks
        if ob.get('at_bullish_ob'):
            points = 20
            bullish_points += points
            factors.append({
                'name': 'At Bullish Order Block',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': 'Price at institutional demand zone'
            })
        elif ob.get('at_bearish_ob'):
            points = 15
            bearish_points += points
            factors.append({
                'name': 'At Bearish Order Block',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': 'Price at resistance - expect rejection'
            })
        
        # FVG
        if fvg.get('at_bullish_fvg'):
            points = 15
            bullish_points += points
            factors.append({
                'name': 'At Bullish FVG',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': 'Price at unfilled gap - support zone'
            })
        elif fvg.get('at_bearish_fvg'):
            points = 10
            bearish_points += points
            factors.append({
                'name': 'At Bearish FVG',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': 'Price at unfilled gap - resistance'
            })
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. TREND ANALYSIS (Max +/- 25 points)
    # ═══════════════════════════════════════════════════════════════════
    
    if trend_data:
        ema_stacked_bullish = trend_data.get('ema_bullish', False)
        ema_stacked_bearish = trend_data.get('ema_bearish', False)
        above_vwap = trend_data.get('above_vwap', False)
        
        if ema_stacked_bullish:
            points = 15
            bullish_points += points
            factors.append({
                'name': 'EMAs Stacked Bullish',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': '9 > 20 > 50 EMA - Strong uptrend structure'
            })
        elif ema_stacked_bearish:
            points = 15
            bearish_points += points
            factors.append({
                'name': 'EMAs Stacked Bearish',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': '9 < 20 < 50 EMA - Downtrend structure'
            })
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. VWAP ANALYSIS (Max +/- 10 points)
    # ═══════════════════════════════════════════════════════════════════
    
    if vwap_data:
        position = vwap_data.get('position', '')
        if 'ABOVE' in position:
            points = 10
            bullish_points += points
            factors.append({
                'name': 'Above VWAP',
                'points': f'+{points}',
                'emoji': '🟢',
                'description': 'Buyers in control - institutional benchmark'
            })
        elif 'BELOW' in position:
            points = 10
            bearish_points += points
            factors.append({
                'name': 'Below VWAP',
                'points': f'-{points}',
                'emoji': '🔴',
                'description': 'Sellers in control - caution'
            })
    
    # ═══════════════════════════════════════════════════════════════════
    # CALCULATE FINAL SCORE AND GRADE
    # ═══════════════════════════════════════════════════════════════════
    
    net_score = bullish_points - bearish_points
    
    # Normalize to 0-100 scale (centered at 50)
    # Max possible is around +/- 100, so we scale
    total_score = 50 + (net_score * 0.5)  # 0-100 range
    total_score = max(0, min(100, total_score))
    
    # Determine grade
    if total_score >= 85:
        grade = "A+"
        grade_emoji = "🏆"
    elif total_score >= 75:
        grade = "A"
        grade_emoji = "🟢"
    elif total_score >= 65:
        grade = "B+"
        grade_emoji = "🟢"
    elif total_score >= 55:
        grade = "B"
        grade_emoji = "🟡"
    elif total_score >= 45:
        grade = "C"
        grade_emoji = "🟡"
    elif total_score >= 35:
        grade = "D"
        grade_emoji = "🟠"
    else:
        grade = "F"
        grade_emoji = "🔴"
    
    # Create summary
    if net_score >= 30:
        summary = "Strong bullish setup - Multiple factors aligned"
    elif net_score >= 15:
        summary = "Bullish bias - More bulls than bears"
    elif net_score >= -15:
        summary = "Mixed signals - No clear edge"
    elif net_score >= -30:
        summary = "Bearish bias - Caution advised"
    else:
        summary = "Strong bearish setup - Consider exiting"
    
    return GradeBreakdown(
        total_score=int(total_score),
        grade=grade,
        grade_emoji=grade_emoji,
        factors=factors,
        bullish_points=bullish_points,
        bearish_points=bearish_points,
        summary=summary
    )


def build_full_education_section(smc_data: Dict, money_flow: Dict, 
                                  vwap_data: Dict = None,
                                  show_definitions: bool = True) -> Dict:
    """
    Build complete education section for display
    
    Returns dict with:
    - active_concepts: List of concepts that are relevant NOW
    - explanations: Contextual explanations for each
    - definitions: Static definitions (if show_definitions=True)
    """
    result = {
        'active_concepts': [],
        'explanations': [],
        'definitions': {}
    }
    
    # Check which SMC concepts are active
    if smc_data:
        ob = smc_data.get('order_blocks', {})
        fvg = smc_data.get('fvg', {})
        
        if ob.get('at_bullish_ob') or ob.get('bullish_ob'):
            result['active_concepts'].append('order_block')
            result['explanations'].append({
                'concept': 'Order Block',
                'status': 'ACTIVE - Bullish',
                'explanation': "Price is at an institutional demand zone. Big players bought here before. They may defend this level.",
                'action': "This is a high-probability LONG entry zone"
            })
        
        if ob.get('at_bearish_ob') or ob.get('bearish_ob'):
            result['active_concepts'].append('order_block')
            result['explanations'].append({
                'concept': 'Order Block',
                'status': 'ACTIVE - Bearish',
                'explanation': "Price is at an institutional supply zone. Big players sold here before. Expect resistance.",
                'action': "This is a high-probability SHORT entry or exit zone"
            })
        
        if fvg.get('at_bullish_fvg'):
            result['active_concepts'].append('fvg')
            result['explanations'].append({
                'concept': 'Fair Value Gap',
                'status': 'FILLING - Bullish',
                'explanation': "Price is filling a gap created during a previous up-move. This 'imbalance' attracts price.",
                'action': "Expect support here, then continuation upward"
            })
    
    # Money flow explanations
    if money_flow:
        mfi = money_flow.get('mfi', 50)
        cmf = money_flow.get('cmf', 0)
        
        result['active_concepts'].append('mfi')
        result['explanations'].append({
            'concept': 'MFI (Money Flow Index)',
            'value': f"{mfi:.0f}",
            'explanation': get_indicator_explanation('mfi', mfi),
            'status': 'Oversold' if mfi < 20 else 'Overbought' if mfi > 80 else 'Neutral'
        })
        
        result['active_concepts'].append('cmf')
        result['explanations'].append({
            'concept': 'CMF (Chaikin Money Flow)',
            'value': f"{cmf:.3f}",
            'explanation': get_indicator_explanation('cmf', cmf),
            'status': 'Accumulation' if cmf > 0 else 'Distribution'
        })
    
    # VWAP explanation
    if vwap_data:
        result['active_concepts'].append('vwap')
        result['explanations'].append({
            'concept': 'VWAP',
            'value': f"${vwap_data.get('vwap', 0):.4f}",
            'explanation': vwap_data.get('position_text', ''),
            'status': vwap_data.get('position', 'NEUTRAL'),
            'distance': f"{vwap_data.get('distance_pct', 0):+.1f}%"
        })
    
    # Add definitions if requested
    if show_definitions:
        for concept in result['active_concepts']:
            if concept in CONCEPT_DEFINITIONS:
                result['definitions'][concept] = CONCEPT_DEFINITIONS[concept]
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 🐋 INSTITUTIONAL / WHALE EDUCATION
# ═══════════════════════════════════════════════════════════════════════════════

INSTITUTIONAL_EDUCATION = {
    
    # ─────────────────────────────────────────────────────────────────────────
    # OPEN INTEREST
    # ─────────────────────────────────────────────────────────────────────────
    'open_interest': {
        'name': 'Open Interest (OI)',
        'icon': '📊',
        'what': 'Total number of outstanding futures contracts that haven\'t been closed.',
        'why': 'Shows how much money is committed. Rising OI = new money entering. Falling OI = money leaving.',
        'readings': {
            'rising_strong': ('> +5%', '🟢 Strong new positions opening - High conviction'),
            'rising': ('+3% to +5%', '🟢 Moderate new positions - Growing interest'),
            'stable': ('-3% to +3%', '⚪ Stable - Market in equilibrium'),
            'falling': ('-5% to -3%', '🟡 Positions closing - Reduced conviction'),
            'falling_strong': ('< -5%', '🔴 Major position closing - Traders exiting')
        },
        'edge': 'OI alone doesn\'t tell direction. Combine with PRICE to understand WHO is entering/exiting.',
        'pro_tip': 'Sudden OI spikes during consolidation = big move coming soon.'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # OI + PRICE COMBINATIONS
    # ─────────────────────────────────────────────────────────────────────────
    'oi_price': {
        'name': 'OI + Price Analysis',
        'icon': '🔍',
        'what': 'The relationship between OI changes and Price changes reveals WHO is trading.',
        'why': 'This is the MOST IMPORTANT institutional signal. It shows if moves are real conviction or forced liquidations.',
        'combinations': {
            'new_longs': {
                'condition': 'OI Rising + Price Rising',
                'emoji': '🟢',
                'meaning': 'NEW LONGS ENTERING - Fresh buying, bullish continuation',
                'action': 'LONG bias - Trend has conviction behind it'
            },
            'new_shorts': {
                'condition': 'OI Rising + Price Falling', 
                'emoji': '🔴',
                'meaning': 'NEW SHORTS ENTERING - Fresh selling, bearish continuation',
                'action': 'SHORT bias - Trend has conviction behind it'
            },
            'short_covering': {
                'condition': 'OI Falling + Price Rising',
                'emoji': '🟡',
                'meaning': 'SHORT COVERING - NOT new buying! Shorts closing.',
                'action': 'CAUTION on longs - Rally may be weak and reverse'
            },
            'long_liquidation': {
                'condition': 'OI Falling + Price Falling',
                'emoji': '🟡', 
                'meaning': 'LONG LIQUIDATION - NOT new selling! Longs getting stopped out.',
                'action': 'Watch for reversal - Forced selling may be exhausting'
            }
        },
        'edge': 'NEW positions = trend continues. CLOSING positions = potential reversal.',
        'pro_tip': 'Long liquidation cascades often mark local bottoms. Start watching for reversal.'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # FUNDING RATE
    # ─────────────────────────────────────────────────────────────────────────
    'funding_rate': {
        'name': 'Funding Rate',
        'icon': '💰',
        'what': 'Fee exchanged between longs and shorts every 8 hours on perpetual futures.',
        'why': 'Shows which side is crowded. Crowded side PAYS the other side.',
        'readings': {
            'extreme_positive': ('> +0.1%', '⚠️ CONTRARIAN BEARISH - Longs overleveraged, expect dump'),
            'positive': ('+0.01% to +0.1%', '🟢 Bullish sentiment - Longs paying premium'),
            'neutral': ('-0.01% to +0.01%', '⚪ Balanced - No funding edge'),
            'negative': ('-0.1% to -0.01%', '🔴 Bearish sentiment - Shorts paying premium'),
            'extreme_negative': ('< -0.1%', '⚠️ CONTRARIAN BULLISH - Shorts overleveraged, expect pump')
        },
        'edge': 'EXTREME funding is a CONTRARIAN indicator. The crowded side gets liquidated.',
        'pro_tip': '3+ days of extreme funding often precedes violent reversals.'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOP TRADERS (WHALES)
    # ─────────────────────────────────────────────────────────────────────────
    'top_traders': {
        'name': 'Top Traders (Whales)',
        'icon': '🐋',
        'what': 'Long/Short positioning of Binance\'s most profitable futures traders.',
        'why': 'These traders consistently make money. Their positioning shows where smart money is betting.',
        'readings': {
            'very_bullish': ('> 60% Long', '🟢 Smart money heavily bullish - Strong conviction'),
            'bullish': ('55-60% Long', '🟢 Leaning bullish - Moderate conviction'),
            'neutral': ('45-55% Long', '⚪ Balanced - No clear directional bias'),
            'bearish': ('40-45% Long', '🔴 Leaning bearish - Moderate conviction'),
            'very_bearish': ('< 40% Long', '🔴 Smart money heavily short - Strong conviction')
        },
        'edge': 'When whales and retail diverge, follow the whales. They\'re usually right.',
        'pro_tip': 'Best setups: Whales positioned OPPOSITE to retail = retail becomes exit liquidity.'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # RETAIL POSITIONING
    # ─────────────────────────────────────────────────────────────────────────
    'retail': {
        'name': 'Retail Traders',
        'icon': '🐑',
        'what': 'Long/Short positioning of all retail traders on Binance Futures.',
        'why': 'Retail traders are often wrong at EXTREMES. They FOMO at tops and panic at bottoms.',
        'readings': {
            'extreme_bullish': ('> 65% Long', '⚠️ FADE THIS - Retail too bullish, often marks tops'),
            'bullish': ('55-65% Long', '🟡 Retail leaning bullish'),
            'neutral': ('45-55% Long', '⚪ Balanced - No clear sentiment'),
            'bearish': ('35-45% Long', '🟡 Retail leaning bearish'),
            'extreme_bearish': ('< 35% Long', '⚠️ FADE THIS - Retail too bearish, often marks bottoms')
        },
        'edge': 'Retail extremes are CONTRARIAN signals. When everyone is bullish, who\'s left to buy?',
        'pro_tip': 'Best trades: Retail extreme + Whales opposite = High probability reversal.'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # DIVERGENCE
    # ─────────────────────────────────────────────────────────────────────────
    'divergence': {
        'name': 'Whale vs Retail Divergence',
        'icon': '⚔️',
        'what': 'The difference between how whales and retail are positioned.',
        'why': 'The BIGGEST edge comes when smart money and retail disagree.',
        'scenarios': {
            'whales_long_retail_short': {
                'meaning': 'Whales buying what retail is selling',
                'action': '🟢 HIGH EDGE LONG - Follow whales',
                'confidence': 'HIGH'
            },
            'whales_short_retail_long': {
                'meaning': 'Whales selling what retail is buying',
                'action': '🔴 HIGH EDGE SHORT - Follow whales',
                'confidence': 'HIGH'
            },
            'aligned': {
                'meaning': 'Both groups positioned similarly',
                'action': '⚪ No divergence edge - Use other signals',
                'confidence': 'LOW'
            }
        },
        'edge': 'Divergence trades have the highest win rate. Retail is exit liquidity.',
        'pro_tip': 'Need >15% divergence to be actionable. Small divergences are noise.'
    },
    
    # ─────────────────────────────────────────────────────────────────────────
    # THE WHALE VS RETAIL GAME (Core Concept)
    # ─────────────────────────────────────────────────────────────────────────
    'whale_retail_game': {
        'name': 'The Whale vs Retail Game',
        'icon': '🎮',
        'what': 'How smart money (whales) and retail traders interact in markets.',
        'why': 'Understanding this dynamic is KEY to not being exit liquidity.',
        'the_game': {
            'accumulation': {
                'phase': '📉 ACCUMULATION (Bottom Formation)',
                'price_action': 'Price falling or consolidating at lows',
                'retail_behavior': 'Panic selling, capitulating, "crypto is dead"',
                'whale_behavior': 'Quietly buying, absorbing sell pressure',
                'your_action': 'Watch for whale % rising while retail sells'
            },
            'markup': {
                'phase': '📈 MARKUP (Uptrend)',
                'price_action': 'Price rising steadily',
                'retail_behavior': 'Starting to notice, "maybe I should buy"',
                'whale_behavior': 'Holding positions, letting price rise',
                'your_action': 'Follow the trend, buy dips'
            },
            'distribution': {
                'phase': '🔝 DISTRIBUTION (Top Formation)',
                'price_action': 'Price at highs, choppy',
                'retail_behavior': 'FOMO buying, "to the moon!", all-in',
                'whale_behavior': 'Quietly selling to retail buyers',
                'your_action': 'Watch for whale % falling while retail buys'
            },
            'markdown': {
                'phase': '📉 MARKDOWN (Downtrend)',
                'price_action': 'Price falling hard',
                'retail_behavior': 'Holding bags, then panic selling at lows',
                'whale_behavior': 'Already sold, waiting to accumulate again',
                'your_action': 'Stay out or short, wait for accumulation'
            }
        },
        'key_insight': '''
🎯 THE CORE TRUTH:
Markets are a wealth transfer mechanism.
Money flows from impatient to patient.
Money flows from emotional to rational.
Money flows from RETAIL to WHALES.

Your job: Don't be exit liquidity.
When retail is euphoric → Be cautious (distribution)
When retail is panicking → Be ready to buy (accumulation)
        ''',
        'edge': 'Trade WITH whales, not against them. When in doubt, fade retail.',
        'pro_tip': 'The best trades feel uncomfortable. Buying when everyone is scared. Selling when everyone is greedy.'
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO COMBINATIONS (Foundation for ML)
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_EDUCATION = {
    
    'long_liquidation_whale_bullish': {
        'name': 'Long Liquidation + Whale Bullish',
        'emoji': '🟡',
        'conditions': 'OI Falling + Price Falling + Whales >55% Long',
        'interpretation': '''
**WHAT'S HAPPENING:**
Price dropping, longs getting liquidated (OI falling).
BUT smart money (whales) remains bullish.

**WHAT THIS MEANS:**
• Weak hands being shaken out
• Forced selling, NOT new shorts entering
• Whales NOT panicking - holding or buying the dip
• Dump may be nearing exhaustion

**🐋 vs 🐑 DYNAMIC:**
Right now, retail is panic selling → Whales are accumulating.
This is HOW wealth transfers from retail to whales.
Retail sells low (fear), whales buy low (opportunity).
        ''',
        'recommendation': '''
🎯 **WAIT then LONG**

⚠️ Technical says SELL but that's the TRAP!
Selling now = selling to whales at the bottom.

**CORRECT PLAY:**
• DON'T sell now (you'd be exit liquidity for whales)
• DON'T buy now (still falling knife)
• WAIT for: Volume spike, support test, or price stabilization
• When confirmed: Enter LONG, stop below recent low

**CONFIRMATION SIGNS:**
✅ Price stops making new lows
✅ Volume spike on green candle
✅ Funding becomes more negative (shorts getting greedy)

⚠️ ABORT if whale % drops below 55% (they're capitulating)
        ''',
        'confidence': 'MEDIUM',
        'ml_placeholder': '📊 Historical data: Scenario occurred X times, Y% success rate'
    },
    
    'new_longs_whale_bullish': {
        'name': 'New Longs + Whale Bullish',
        'emoji': '🟢',
        'conditions': 'OI Rising + Price Rising + Whales >55% Long',
        'interpretation': '''
**WHAT'S HAPPENING:**
Price rising WITH new money entering (OI rising).
Whales are positioned bullish.

**WHAT THIS MEANS:**
• Fresh buying conviction
• Smart money agrees with the move
• Trend has legs - backed by new positions
• Continuation likely
        ''',
        'recommendation': '''
🎯 **LONG (High Conviction)**

• Enter on pullbacks to support
• Stop below recent swing low
• Trail stop as trend continues
• Add on dips

✅ This is a HIGH CONVICTION bullish setup.
        ''',
        'confidence': 'HIGH',
        'ml_placeholder': '📊 Historical data: Scenario occurred X times, Y% success rate'
    },
    
    'short_squeeze_setup': {
        'name': 'Short Squeeze Setup',
        'emoji': '🚀',
        'conditions': 'Funding Very Negative + Whales Bullish + Retail Bearish',
        'interpretation': '''
**WHAT'S HAPPENING:**
Shorts paying high funding (crowded).
Whales long while retail is short.

**WHAT THIS MEANS:**
• Too many shorts = squeeze fuel
• Smart money opposite to crowd
• Any up move triggers cascading liquidations
• Explosive upside likely
        ''',
        'recommendation': '''
🎯 **LONG (Squeeze Play)**

• High probability setup
• Enter immediately or on dip
• Tight stop below recent low
• Target: 5-10% or until funding normalizes

🚀 SQUEEZE ALERT: These moves can be violent!
        ''',
        'confidence': 'HIGH',
        'ml_placeholder': '📊 Historical data: Short squeezes succeed 70%+ of time'
    },
    
    # ═══════════════════════════════════════════════════════════════════════
    # BEARISH SCENARIOS (Mirror of BULLISH - Follow Whales SHORT)
    # ═══════════════════════════════════════════════════════════════════════
    
    'new_shorts_whale_bearish': {
        'name': 'New Shorts + Whale Bearish',
        'emoji': '🔴',
        'conditions': 'OI Rising + Price Falling + Whales <45% Long',
        'interpretation': '''
**WHAT'S HAPPENING:**
Price falling WITH new money entering shorts (OI rising).
Whales are positioned bearish.

**WHAT THIS MEANS:**
• Fresh selling conviction
• Smart money agrees with the move
• Downtrend has legs - backed by new positions
• Continuation likely
        ''',
        'recommendation': '''
🎯 **SHORT (High Conviction)**

• Enter on bounces to resistance
• Stop above recent swing high
• Trail stop as trend continues
• Add on rallies

✅ This is a HIGH CONVICTION bearish setup.
        ''',
        'confidence': 'HIGH',
        'ml_placeholder': '📊 Historical data: New shorts + falling price continues 65-75% of time.'
    },
    
    'long_squeeze_setup': {
        'name': 'Long Squeeze Setup',
        'emoji': '💥',
        'conditions': 'Funding Very Positive + Whales Bearish + Retail Bullish',
        'interpretation': '''
**WHAT'S HAPPENING:**
Longs paying high funding (crowded).
Whales short while retail is long.

**WHAT THIS MEANS:**
• Too many longs = liquidation fuel
• Smart money opposite to crowd
• Any down move triggers cascading liquidations
• Violent downside likely
        ''',
        'recommendation': '''
🎯 **SHORT (Squeeze Play)**

• High probability setup
• Enter immediately or on bounce
• Tight stop above recent high
• Target: 5-10% or until funding normalizes

💥 DUMP ALERT: These moves can be violent!
        ''',
        'confidence': 'HIGH',
        'ml_placeholder': '📊 Historical data: Long squeezes succeed 70%+ of time'
    },
    
    'short_covering_whale_bearish': {
        'name': 'Short Covering + Whale Bearish',
        'emoji': '🟡',
        'conditions': 'OI Falling + Price Rising + Whales <45% Long',
        'interpretation': '''
**WHAT'S HAPPENING:**
Price rising but OI is FALLING.
This means shorts are COVERING (closing), NOT new longs entering.
Whales remain bearish.

**WHAT THIS MEANS:**
• Rally is WEAK - no new buying conviction
• Just shorts closing positions
• Whales NOT chasing this rally
• Rally may exhaust and reverse

**🐋 vs 🐑 DYNAMIC:**
Retail sees rally and thinks "bottom is in!"
Whales see weak rally and prepare to short more.
        ''',
        'recommendation': '''
🎯 **WAIT then SHORT**

⚠️ Don't short into strength immediately!
This rally is weak but may continue a bit more.

**CORRECT PLAY:**
• DON'T short now (may squeeze you first)
• DON'T buy now (whales will sell to you)
• WAIT for: Rally exhaustion, resistance test, volume fade
• When confirmed: Enter SHORT, stop above recent high

**CONFIRMATION SIGNS:**
✅ Price stops making new highs
✅ Volume fading on green candles
✅ Funding becomes more positive (longs getting greedy)

⚠️ ABORT if whale % goes above 50% (they're turning bullish)
        ''',
        'confidence': 'MEDIUM',
        'ml_placeholder': '📊 Historical data: Scenario occurred X times, Y% success rate'
    },
    
    'conflicting_signals': {
        'name': 'Tech vs Institutional Conflict',
        'emoji': '⚠️',
        'conditions': 'Technical = WAIT/Bearish + Institutional = Bullish (or vice versa)',
        'interpretation': '''
**WHAT'S HAPPENING:**
Technical analysis (price patterns) says one thing.
Institutional positioning (whale data) says another.

**WHY THIS HAPPENS:**
• Technical lags - it shows CURRENT price action
• Institutional leads - it shows where smart money IS POSITIONED
• Whales often accumulate WHILE price is still falling
• That's how they buy cheap!

**🐋 vs 🐑 CONTEXT:**
Technical bearish + Whales bullish often means:
→ We're in ACCUMULATION phase
→ Retail sees falling price and panics
→ Whales see cheap prices and accumulate
→ Technical will turn bullish LATER
        ''',
        'recommendation': '''
🎯 **WAIT or SMALL POSITION**

⚠️ This is NOT "signals disagree so do nothing"
This is "smart money is early, be patient"

**UNDERSTAND THE TRAP:**
• Selling now = selling to whales at the bottom
• Buying now = catching a falling knife

**CORRECT PLAY:**
**Option A (Conservative):**
• Wait for technical to confirm whale direction
• Enter when both signals align
• Higher confidence, slightly worse entry

**Option B (Aggressive):**
• Small position following whales NOW
• Accept you might catch some falling knife
• Add size when technicals confirm
• Better entry if whales are right

⚠️ Size down vs high confidence setups!
        ''',
        'confidence': 'LOW',
        'ml_placeholder': '📊 Historical data: Conflicts resolve in whale direction ~60% of time'
    },
    
    'new_longs_neutral': {
        'name': 'New Longs Entering',
        'emoji': '🟢',
        'conditions': 'OI Rising + Price Rising + Whales Neutral',
        'interpretation': '''
**WHAT'S HAPPENING:**
New money is entering LONG positions (OI rising + price rising).
Whales are neutral - haven't committed strongly yet.

**WHAT THIS MEANS:**
• Fresh buying conviction from market
• OI + Price pattern is BULLISH
• Whales haven't confirmed but aren't opposing
• Good signal but less conviction than whale-confirmed
        ''',
        'recommendation': '''
🎯 **LONG (Medium Conviction)**

• Enter on pullbacks to support
• Smaller position than high-conviction setups
• Stop below recent swing low
• Add size if whales turn bullish (>55%)

✅ This is a valid bullish setup, just without whale confirmation yet.
        ''',
        'confidence': 'MEDIUM',
        'ml_placeholder': '📊 Historical data: Scenario occurred X times, Y% success rate'
    },
    
    'new_shorts_neutral': {
        'name': 'New Shorts Entering',
        'emoji': '🔴',
        'conditions': 'OI Rising + Price Falling + Whales Neutral',
        'interpretation': '''
**WHAT'S HAPPENING:**
New money is entering SHORT positions (OI rising + price falling).
Whales are neutral - haven't committed strongly yet.

**WHAT THIS MEANS:**
• Fresh selling pressure from market
• OI + Price pattern is BEARISH
• Whales haven't confirmed but aren't opposing
• Good signal but less conviction than whale-confirmed
        ''',
        'recommendation': '''
🎯 **SHORT (Medium Conviction)**

• Enter on bounces to resistance
• Smaller position than high-conviction setups
• Stop above recent swing high
• Add size if whales turn bearish (<45%)

✅ This is a valid bearish setup, just without whale confirmation yet.
        ''',
        'confidence': 'MEDIUM',
        'ml_placeholder': '📊 Historical data: Scenario occurred X times, Y% success rate'
    },
    
    'short_covering_neutral': {
        'name': 'Short Covering Rally',
        'emoji': '🟡',
        'conditions': 'OI Falling + Price Rising',
        'interpretation': '''
**WHAT'S HAPPENING:**
Price rising but OI falling - shorts are closing, not new longs entering.
Direction is unclear.

**WHAT THIS MEANS:**
• Rally may be temporary - no fresh buying conviction
• Could fade once short covering exhausts
• Or could attract new longs if momentum continues
        ''',
        'recommendation': '''
🎯 **WAIT for clarity**

• Don't chase the rally
• Wait for OI to start rising (new longs entering)
• Or wait for reversal signs (failed breakout)

⚠️ Short covering rallies can be traps.
        ''',
        'confidence': 'LOW',
        'ml_placeholder': '📊 Historical data: Short covering rallies fade ~55% of time'
    },
    
    'long_liquidation_neutral': {
        'name': 'Long Liquidation Cascade',
        'emoji': '🟡',
        'conditions': 'OI Falling + Price Falling',
        'interpretation': '''
**WHAT'S HAPPENING:**
Price falling and OI falling - longs are being liquidated.
No whale support visible.

**WHAT THIS MEANS:**
• Forced selling in progress
• Could exhaust and reverse
• Or could cascade further if support breaks
        ''',
        'recommendation': '''
🎯 **WAIT - Don't catch the knife**

• Don't buy the dip without whale support
• Wait for: price stabilization, volume spike, or OI rising
• Watch for whale positioning change (>55% = accumulation starting)

⚠️ Forced selling can accelerate. Stay patient.
        ''',
        'confidence': 'LOW',
        'ml_placeholder': '📊 Historical data: Liquidation cascades need ~2-3 support tests'
    },
    
    # ─────────────────────────────────────────────────────────────────────────────
    # WHALE BIAS WITH NEUTRAL OI (New scenarios!)
    # ─────────────────────────────────────────────────────────────────────────────
    
    'whale_bullish_oi_neutral': {
        'name': 'Whale Bullish (OI Stable)',
        'emoji': '🟢',
        'conditions': 'Whales 55%+ LONG, OI stable',
        'interpretation': '''
**WHAT'S HAPPENING:**
Whales are positioned bullish (55%+ long).
But OI isn't moving significantly - no new money entering yet.

**WHAT THIS MEANS:**
• Smart money is bullish but waiting
• Could be accumulation phase
• Price may move once OI confirms
        ''',
        'recommendation': '''
🎯 **CAUTIOUS LONG - Wait for OI confirmation**

• Whales see value at these levels
• Enter small, add when OI rises with price
• Watch for breakout with volume
• Use technical levels for entry/stops

💡 Whale positioning often leads price action.
        ''',
        'confidence': 'MEDIUM',
        'ml_placeholder': '📊 Historical data: Whale positioning often precedes moves by 1-3 candles'
    },
    
    'whale_bearish_oi_neutral': {
        'name': 'Whale Bearish (OI Stable)',
        'emoji': '🔴',
        'conditions': 'Whales 45%- LONG (short bias), OI stable',
        'interpretation': '''
**WHAT'S HAPPENING:**
Whales are positioned bearish (45% or less long).
But OI isn't moving significantly - no new shorts entering yet.

**WHAT THIS MEANS:**
• Smart money is bearish but waiting
• Could be distribution phase
• Price may drop once OI confirms downward
        ''',
        'recommendation': '''
🎯 **CAUTIOUS SHORT - Wait for OI confirmation**

• Whales see downside risk
• Don't buy dips here
• Watch for breakdown with volume
• Use technical levels for entry/stops

💡 Whale positioning often leads price action.
        ''',
        'confidence': 'MEDIUM',
        'ml_placeholder': '📊 Historical data: Whale positioning often precedes moves by 1-3 candles'
    },
    
    'no_edge': {
        'name': 'No Clear Edge',
        'emoji': '⚪',
        'conditions': 'All metrics neutral (45-55%)',
        'interpretation': '''
**WHAT'S HAPPENING:**
All institutional metrics are neutral.
No strong conviction either direction.

**WHAT THIS MEANS:**
• Market in equilibrium
• Waiting for catalyst
• No edge from positioning
        ''',
        'recommendation': '''
🎯 **WAIT or RANGE TRADE**

• No institutional edge to exploit
• Use technical levels only
• Buy support, sell resistance
• Tight stops

💡 Wait for extremes before directional bets.
        ''',
        'confidence': 'LOW',
        'ml_placeholder': '📊 Historical data: Neutral often precedes breakouts'
    }
}


def identify_current_scenario(oi_change: float, price_change: float, 
                             whale_pct: float, retail_pct: float, 
                             funding: float) -> str:
    """
    Identify which scenario matches current market conditions.
    
    ═══════════════════════════════════════════════════════════════════════════════
    THIS FUNCTION NOW DELEGATES TO rules_engine.py - THE SINGLE SOURCE OF TRUTH
    All thresholds are read from institutional_config.json
    ═══════════════════════════════════════════════════════════════════════════════
    """
    # Import from rules_engine (single source of truth)
    from .rules_engine import analyze
    
    # Get analysis from rules engine
    result = analyze(oi_change, price_change, whale_pct, retail_pct, funding)
    
    # Map rules_engine scenario_id to education.py keys
    # Rules engine uses UPPER_CASE, education uses lower_case
    scenario_mapping = {
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
        'SHORT_COVERING_NO_CONVICTION': 'short_covering_neutral',
        'LONG_LIQUIDATION_NO_SUPPORT': 'long_liquidation_neutral',
        'WHALE_BULLISH_OI_NEUTRAL': 'whale_bullish_oi_neutral',
        'WHALE_BEARISH_OI_NEUTRAL': 'whale_bearish_oi_neutral',
        'CONFLICTING_SIGNALS': 'conflicting_signals',
        'NO_EDGE': 'no_edge'
    }
    
    return scenario_mapping.get(result.scenario_id, 'no_edge')


def get_unified_verdict(oi_change: float, price_change: float,
                        whale_pct: float, retail_pct: float,
                        funding: float, tech_action: str = 'WAIT',
                        position_pct: float = 50, ta_score: int = 50) -> Dict:
    """
    Generate a unified verdict combining all signals.
    
    ═══════════════════════════════════════════════════════════════════════════════
    NOW USES MASTER_RULES AS THE SINGLE SOURCE OF TRUTH!
    All decisions flow through MASTER_RULES.get_trade_decision()
    Education content is mapped from MASTER_RULES output
    ═══════════════════════════════════════════════════════════════════════════════
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # USE MASTER_RULES AS SINGLE SOURCE OF TRUTH
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from .MASTER_RULES import get_trade_decision
        
        decision = get_trade_decision(
            whale_pct=whale_pct,
            retail_pct=retail_pct,
            oi_change=oi_change,
            price_change=price_change,
            position_pct=position_pct,
            ta_score=ta_score
        )
        
        # Map MASTER_RULES action to scenario key for education
        action_to_scenario = {
            'STRONG_LONG': 'new_longs_whale_bullish',
            'LONG_SETUP': 'whale_bullish_oi_neutral',
            'BUILDING_LONG': 'whale_bullish_oi_neutral',
            'MONITOR_LONG': 'whale_bullish_oi_neutral',
            'CAUTION_LONG': 'conflicting_signals',
            'SHORT_COVERING': 'short_covering_neutral',
            'WAIT': 'no_edge',
            'WAIT_FOR_DIP': 'whale_bullish_oi_neutral',
            'WAIT_FOR_BOUNCE': 'whale_bearish_oi_neutral',
            'MONITOR_SHORT': 'whale_bearish_oi_neutral',
            'BUILDING_SHORT': 'whale_bearish_oi_neutral',
            'SHORT_SETUP': 'new_shorts_whale_bearish',
            'STRONG_SHORT': 'new_shorts_whale_bearish',
        }
        
        scenario_key = action_to_scenario.get(decision.action, 'no_edge')
        
        # Build scenario info from MASTER_RULES output
        scenario_info = {
            'name': decision.conclusion.split(' - ')[0].replace('⚠️ ', '').replace('🟢 ', '').replace('🔴 ', '').replace('⏳ ', '').replace('🎯 ', ''),
            'conditions': f"OI: {oi_change:+.1f}% | Price: {price_change:+.1f}% | Whales: {whale_pct:.0f}%",
            'interpretation': decision.whale_story + "\n\n" + decision.oi_story,
            'recommendation': decision.position_story + "\n\n**Action:** " + decision.conclusion_action,
            'confidence': decision.confidence,
        }
        
        # Use MASTER_RULES trade_direction as unified_action
        unified_action = decision.trade_direction
        
        # Check alignment with technicals
        tech_bullish = any(w in tech_action.upper() for w in ['BUY', 'LONG', 'ACCUMULATE'])
        tech_bearish = any(w in tech_action.upper() for w in ['SELL', 'SHORT', 'TRIM'])
        inst_bullish = decision.trade_direction == 'LONG'
        inst_bearish = decision.trade_direction == 'SHORT'
        
        if (tech_bullish and inst_bullish) or (tech_bearish and inst_bearish):
            alignment = 'ALIGNED'
            alignment_note = '✅ Technical and Institutional signals AGREE'
        elif (tech_bullish and inst_bearish) or (tech_bearish and inst_bullish):
            alignment = 'CONFLICTING'
            alignment_note = '⚠️ Technical and Institutional signals DISAGREE'
        elif decision.trade_direction == 'WAIT':
            alignment = 'WAIT'
            alignment_note = f"⏳ {decision.conclusion_action}"
        else:
            alignment = 'NEUTRAL'
            alignment_note = '⚪ No strong conflict between signals'
        
        # Add warnings to alignment note if any
        if decision.warnings:
            alignment_note = decision.warnings[0]
        
        return {
            'scenario_key': scenario_key,
            'scenario': scenario_info,
            'unified_action': unified_action,
            'confidence': decision.confidence,
            'alignment': alignment,
            'alignment_note': alignment_note,
            'tech_action': tech_action,
            'institutional_bias': 'BULLISH' if whale_pct >= 55 else 'BEARISH' if whale_pct <= 45 else 'NEUTRAL',
            'master_decision': decision,  # Include full decision for reference
        }
        
    except Exception as e:
        # Fallback to old logic if MASTER_RULES fails
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FALLBACK: Old logic (should rarely happen)
    # ═══════════════════════════════════════════════════════════════════════════
    
    scenario_key = identify_current_scenario(
        oi_change, price_change, whale_pct, retail_pct, funding
    )
    
    scenario_info = SCENARIO_EDUCATION.get(scenario_key, SCENARIO_EDUCATION.get('no_edge', {
        'name': 'No Clear Edge',
        'conditions': 'Mixed signals',
        'interpretation': 'Wait for clarity',
        'recommendation': 'Wait',
        'confidence': 'LOW'
    }))
    
    # Determine unified action based on confidence
    if scenario_info.get('confidence') == 'HIGH':
        if 'LONG' in str(scenario_info.get('recommendation', '')):
            unified_action = 'LONG'
        elif 'SHORT' in str(scenario_info.get('recommendation', '')):
            unified_action = 'SHORT'
        else:
            unified_action = 'WAIT'
    elif scenario_info.get('confidence') == 'MEDIUM':
        unified_action = 'WAIT_THEN_' + ('LONG' if whale_pct >= 55 else 'SHORT' if whale_pct <= 45 else 'WAIT')
    else:
        unified_action = 'WAIT'
    
    # Check alignment with technicals
    tech_bullish = any(w in tech_action.upper() for w in ['BUY', 'LONG', 'ACCUMULATE'])
    tech_bearish = any(w in tech_action.upper() for w in ['SELL', 'SHORT', 'TRIM'])
    inst_bullish = whale_pct >= 55
    inst_bearish = whale_pct <= 45
    
    if (tech_bullish and inst_bullish) or (tech_bearish and inst_bearish):
        alignment = 'ALIGNED'
        alignment_note = '✅ Technical and Institutional signals AGREE'
    elif (tech_bullish and inst_bearish) or (tech_bearish and inst_bullish):
        alignment = 'CONFLICTING'
        alignment_note = '⚠️ Technical and Institutional signals DISAGREE'
    else:
        alignment = 'NEUTRAL'
        alignment_note = '⚪ No strong conflict between signals'
    
    return {
        'scenario_key': scenario_key,
        'scenario': scenario_info,
        'unified_action': unified_action,
        'confidence': scenario_info.get('confidence', 'LOW'),
        'alignment': alignment,
        'alignment_note': alignment_note,
        'tech_action': tech_action,
        'institutional_bias': 'BULLISH' if inst_bullish else 'BEARISH' if inst_bearish else 'NEUTRAL'
    }


def get_metric_tooltip(metric_key: str) -> str:
    """Get a short tooltip for a metric (for hover/info icon)"""
    edu = INSTITUTIONAL_EDUCATION.get(metric_key, {})
    if not edu:
        return ''
    
    return f"{edu.get('icon', '')} **{edu.get('name', metric_key)}**\n\n{edu.get('what', '')}\n\n💡 {edu.get('edge', '')}"


def get_full_metric_education(metric_key: str) -> Dict:
    """Get full education content for a metric (for expander)"""
    return INSTITUTIONAL_EDUCATION.get(metric_key, {})


def get_oi_price_quick_reference(oi_change: float, price_change: float) -> Dict:
    """
    Get quick reference card for OI + Price combination.
    Returns current interpretation plus the full reference table.
    """
    # Determine current signal
    if oi_change > 1 and price_change > 0:
        signal = 'NEW_LONGS'
        emoji = '🟢'
        title = 'NEW LONGS ENTERING'
        meaning = 'Fresh buying - new money entering long positions'
        action = 'Bullish continuation likely - trend has conviction'
        color = '#00d4aa'
    elif oi_change > 1 and price_change < 0:
        signal = 'NEW_SHORTS'
        emoji = '🔴'
        title = 'NEW SHORTS ENTERING'
        meaning = 'Fresh selling - new money entering short positions'
        action = 'Bearish continuation likely - trend has conviction'
        color = '#ff4444'
    elif oi_change < -1 and price_change > 0:
        signal = 'SHORT_COVERING'
        emoji = '🟡'
        title = 'SHORT COVERING'
        meaning = 'Shorts closing positions - NOT new buying!'
        action = 'Rally may be WEAK - no new conviction, may reverse'
        color = '#ffcc00'
    elif oi_change < -1 and price_change < 0:
        signal = 'LONG_LIQUIDATION'
        emoji = '🟡'
        title = 'LONG LIQUIDATION'
        meaning = 'Longs getting stopped out - NOT new selling!'
        action = 'Dump may be ENDING - forced selling exhausting'
        color = '#ffcc00'
    else:
        signal = 'NEUTRAL'
        emoji = '⚪'
        title = 'NEUTRAL / UNCLEAR'
        meaning = 'No strong OI + Price signal'
        action = 'Use other indicators'
        color = '#888888'
    
    return {
        'signal': signal,
        'emoji': emoji,
        'title': title,
        'meaning': meaning,
        'action': action,
        'color': color,
        'oi_change': oi_change,
        'price_change': price_change,
        'reference_table': """
| OI | Price | Signal | Meaning | What To Do |
|:---:|:---:|:---|:---|:---|
| 📈 UP | 📈 UP | 🟢 NEW LONGS | Fresh buying entering | ✅ Bullish continuation |
| 📈 UP | 📉 DOWN | 🔴 NEW SHORTS | Fresh selling entering | ❌ Bearish continuation |
| 📉 DOWN | 📈 UP | 🟡 SHORT COVERING | NOT new buying! Shorts closing | ⚠️ Weak rally, may reverse |
| 📉 DOWN | 📉 DOWN | 🟡 LONG LIQUIDATION | NOT new selling! Longs closing | ⚠️ Dump ending, watch for reversal |
        """,
        'key_insight': """
**KEY INSIGHT:**
- OI Rising = New positions = Trend has CONVICTION = Follow it
- OI Falling = Closing positions = Trend may EXHAUST = Watch for reversal
        """
    }


def render_oi_price_education_html(oi_change: float, price_change: float) -> str:
    """Generate HTML for OI + Price education card"""
    ref = get_oi_price_quick_reference(oi_change, price_change)
    
    return f"""
    <div style='background: {ref['color']}22; border: 2px solid {ref['color']}; border-radius: 12px; padding: 20px; margin: 15px 0;'>
        <div style='color: {ref['color']}; font-size: 1.3em; font-weight: bold; margin-bottom: 10px;'>
            {ref['emoji']} {ref['title']}
        </div>
        <div style='color: #ddd; font-size: 1em; margin-bottom: 8px;'>
            {ref['meaning']}
        </div>
        <div style='color: #aaa; font-size: 0.95em;'>
            <strong>→ Action:</strong> {ref['action']}
        </div>
        <div style='color: #666; font-size: 0.85em; margin-top: 10px;'>
            OI: {ref['oi_change']:+.1f}% | Price: {ref['price_change']:+.1f}%
        </div>
    </div>
    """

