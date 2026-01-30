"""
InvestorIQ - Master Narrative Engine
=====================================
The brain of InvestorIQ - generates professional, educational analysis
that applies consistently across Scanner, Single Analysis, and Trade Monitor.

This engine explains the "WHY" behind every recommendation, helping you
learn while you invest.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class Action(Enum):
    """Recommended actions"""
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    LIGHT_ACCUMULATE = "LIGHT ACCUMULATE"
    HOLD = "HOLD"
    LIGHT_TRIM = "LIGHT TRIM"
    TRIM = "TRIM"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"
    WAIT = "WAIT"
    ADD_TO_POSITION = "ADD TO POSITION"
    TAKE_PARTIAL_PROFIT = "TAKE PARTIAL PROFIT"

class Sentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

# Action colors and emojis
ACTION_STYLES = {
    Action.STRONG_BUY: {"color": "#00ff88", "emoji": "🟢", "bg": "#003322"},
    Action.BUY: {"color": "#00d4aa", "emoji": "🟢", "bg": "#002a22"},
    Action.ACCUMULATE: {"color": "#00d4aa", "emoji": "✅", "bg": "#002a22"},
    Action.LIGHT_ACCUMULATE: {"color": "#88d4aa", "emoji": "➕", "bg": "#002a22"},
    Action.HOLD: {"color": "#ffcc00", "emoji": "⏸️", "bg": "#332a00"},
    Action.LIGHT_TRIM: {"color": "#ffaa00", "emoji": "➖", "bg": "#332200"},
    Action.TRIM: {"color": "#ff7700", "emoji": "⚠️", "bg": "#331a00"},
    Action.SELL: {"color": "#ff4444", "emoji": "🔴", "bg": "#330000"},
    Action.STRONG_SELL: {"color": "#ff0000", "emoji": "🔴", "bg": "#440000"},
    Action.WAIT: {"color": "#aaaaaa", "emoji": "⏳", "bg": "#222222"},
    Action.ADD_TO_POSITION: {"color": "#00d4ff", "emoji": "➕", "bg": "#002233"},
    Action.TAKE_PARTIAL_PROFIT: {"color": "#ffcc00", "emoji": "💰", "bg": "#332a00"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InsightPoint:
    """A single insight/observation in the analysis"""
    sentiment: Sentiment
    category: str      # 'price', 'momentum', 'volume', 'trend', 'structure', 'position'
    title: str         # Short title
    explanation: str   # Detailed educational explanation
    impact: int        # Score impact (-30 to +30)
    
    @property
    def emoji(self) -> str:
        if self.sentiment == Sentiment.BULLISH:
            return "🟢"
        elif self.sentiment == Sentiment.BEARISH:
            return "🔴"
        return "⚪"


@dataclass  
class PriceLevel:
    """A significant price level"""
    price: float
    label: str
    level_type: str  # 'entry', 'stop', 'target', 'dca', 'support', 'resistance'
    note: str = ""


@dataclass
class AnalysisResult:
    """Complete analysis result used across all modes"""
    # Basic info
    symbol: str
    mode: str  # 'scalp', 'day_trade', 'swing', 'investment'
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Current state
    current_price: float = 0
    
    # Scores
    bullish_score: int = 0
    bearish_score: int = 0
    net_score: int = 0
    confidence: int = 50
    
    # Recommendation
    action: Action = Action.HOLD
    
    # Narrative components
    headline: str = ""
    summary: str = ""
    insights: List[InsightPoint] = field(default_factory=list)
    
    # Price levels
    levels: List[PriceLevel] = field(default_factory=list)
    
    # For trading modes
    entry: float = 0
    stop_loss: float = 0
    tp1: float = 0
    tp2: float = 0
    tp3: float = 0
    risk_pct: float = 0
    reward_pct: float = 0
    rr_ratio: float = 0
    
    # For investment mode
    dca_zones: List[Dict] = field(default_factory=list)
    trim_zones: List[Dict] = field(default_factory=list)
    
    # For position monitoring
    position_pnl: float = 0
    position_advice: str = ""
    should_add: bool = False
    should_trim: bool = False
    
    # For timing advice (uptrend but overbought)
    wait_for_pullback: bool = False
    pullback_reason: List[str] = field(default_factory=list)
    
    # Action plan (step by step)
    action_steps: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "Medium"
    risk_notes: str = ""
    
    # Key metrics for display
    metrics: Dict = field(default_factory=dict)
    
    @property
    def style(self) -> Dict:
        # Special case: wait_for_pullback is BULLISH timing issue, not bearish
        if self.wait_for_pullback:
            return {"color": "#ffcc00", "emoji": "⏳", "bg": "#332a00"}  # Yellow - bullish but wait
        return ACTION_STYLES.get(self.action, ACTION_STYLES[Action.HOLD])


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER NARRATIVE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MasterNarrative:
    """
    The Master Narrative Engine for InvestorIQ
    
    Generates consistent, professional, educational analysis for:
    - Scanner: Quick assessment of opportunities
    - Single Analysis: Deep dive into any asset
    - Trade Monitor: Ongoing position management
    
    Every recommendation comes with clear reasoning to help you learn.
    """
    
    def __init__(self):
        self.insights: List[InsightPoint] = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ANALYSIS FUNCTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        mode: str,  # 'scalp', 'day_trade', 'swing', 'investment'
        timeframe: str = '1d',
        existing_position: Optional[Dict] = None
    ) -> AnalysisResult:
        """
        Generate complete analysis with narrative
        
        Args:
            df: OHLCV DataFrame
            symbol: Asset symbol (e.g., 'BTCUSDT', 'AAPL', 'SPY')
            mode: Trading/investment mode
            timeframe: Timeframe being analyzed
            existing_position: If monitoring, the current position details
                               {'entry': float, 'size': float, 'direction': str}
        
        Returns:
            AnalysisResult with complete narrative and recommendations
        """
        self.insights = []
        
        # Calculate all metrics
        m = self._calculate_all_metrics(df)
        
        # Create result object
        result = AnalysisResult(
            symbol=symbol,
            mode=mode,
            current_price=m['price'],
            metrics=m
        )
        
        # Run analysis based on mode
        self._analyze_price_position(m, mode, result)
        self._analyze_trend_structure(m, mode, result)
        self._analyze_momentum(m, mode, result)
        self._analyze_volume_flow(m, mode, result)
        self._analyze_volatility_risk(m, mode, result)
        
        # If monitoring existing position
        if existing_position:
            self._analyze_existing_position(m, existing_position, mode, result)
        
        # Calculate final scores
        result.insights = self.insights
        result.bullish_score = sum(i.impact for i in self.insights if i.sentiment == Sentiment.BULLISH)
        result.bearish_score = abs(sum(i.impact for i in self.insights if i.sentiment == Sentiment.BEARISH))
        result.net_score = result.bullish_score - result.bearish_score
        result.confidence = min(100, max(0, 50 + result.net_score))
        
        # Determine action
        result.action = self._determine_action(result, m, mode, existing_position)
        
        # Generate narrative text
        result.headline = self._create_headline(result, m, mode)
        result.summary = self._create_summary(result, m, mode)
        
        # Set price levels
        self._set_price_levels(result, m, mode)
        
        # Generate action plan
        result.action_steps = self._create_action_plan(result, m, mode, existing_position)
        
        # Risk assessment
        result.risk_level, result.risk_notes = self._assess_risk(m, mode)
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC CALCULATIONS - NOW USING ta LIBRARY VIA indicators.py
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_all_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive metrics using ta library"""
        from core.indicators import (
            calculate_rsi, calculate_macd, calculate_ema, calculate_sma,
            calculate_atr, calculate_bbands, calculate_obv, calculate_mfi,
            calculate_cmf, find_support_resistance
        )
        
        c = df['Close']
        h = df['High']
        l = df['Low']
        v = df['Volume']
        
        price = c.iloc[-1]
        
        # === PRICE METRICS ===
        high_all = h.max()
        low_all = l.min()
        price_range = high_all - low_all
        price_position = ((price - low_all) / price_range * 100) if price_range > 0 else 50
        
        avg_price = c.mean()
        
        # Price change
        change_1d = ((price - c.iloc[-2]) / c.iloc[-2] * 100) if len(c) > 1 else 0
        change_5d = ((price - c.iloc[-5]) / c.iloc[-5] * 100) if len(c) > 5 else 0
        change_20d = ((price - c.iloc[-20]) / c.iloc[-20] * 100) if len(c) > 20 else 0
        
        # === MOVING AVERAGES (using ta library) ===
        ema_9 = calculate_ema(c, 9)
        ema_20 = calculate_ema(c, 20)
        ema_50 = calculate_ema(c, 50)
        sma_200 = calculate_sma(c, 200) if len(c) >= 200 else calculate_sma(c, 50)
        
        # Safe get last values
        def safe_last(series, default=0):
            try:
                val = series.iloc[-1]
                return default if pd.isna(val) else val
            except:
                return default
        
        ema_9_val = safe_last(ema_9, price)
        ema_20_val = safe_last(ema_20, price)
        ema_50_val = safe_last(ema_50, price)
        sma_200_val = safe_last(sma_200, price)
        
        above_ema20 = price > ema_20_val
        above_ema50 = price > ema_50_val
        above_sma200 = price > sma_200_val
        ema_bullish_stack = ema_9_val > ema_20_val > ema_50_val
        ema_bearish_stack = ema_9_val < ema_20_val < ema_50_val
        
        # === RSI (using ta library) ===
        rsi = calculate_rsi(c, 14)
        rsi_val = safe_last(rsi, 50)
        rsi_prev = safe_last(rsi.shift(1), 50) if len(rsi) > 1 else 50
        
        # RSI divergence check
        price_higher = price > c.iloc[-5] if len(c) > 5 else False
        rsi_lower = rsi_val < safe_last(rsi.shift(4), 50) if len(rsi) > 4 else False
        bearish_div = price_higher and rsi_lower
        
        price_lower = price < c.iloc[-5] if len(c) > 5 else False
        rsi_higher = rsi_val > safe_last(rsi.shift(4), 50) if len(rsi) > 4 else False
        bullish_div = price_lower and rsi_higher
        
        # === MACD (using ta library) ===
        macd_line, macd_signal_line, macd_hist = calculate_macd(c)
        macd_val = safe_last(macd_line)
        macd_sig_val = safe_last(macd_signal_line)
        macd_hist_val = safe_last(macd_hist)
        
        macd_cross_up = (macd_val > macd_sig_val and 
                        safe_last(macd_line.shift(1)) <= safe_last(macd_signal_line.shift(1)))
        macd_cross_down = (macd_val < macd_sig_val and 
                          safe_last(macd_line.shift(1)) >= safe_last(macd_signal_line.shift(1)))
        
        # === VOLUME METRICS (using ta library) ===
        vol_avg = v.rolling(20).mean().iloc[-1] if len(v) >= 20 else v.mean()
        vol_ratio = v.iloc[-1] / vol_avg if vol_avg > 0 else 1
        
        obv = calculate_obv(c, v)
        obv_sma = obv.rolling(10).mean() if len(obv) >= 10 else obv
        obv_rising = safe_last(obv) > safe_last(obv_sma)
        
        mfi = calculate_mfi(h, l, c, v, 14)
        mfi_val = safe_last(mfi, 50)
        
        cmf = calculate_cmf(h, l, c, v, 20)
        cmf_val = safe_last(cmf, 0)
        
        # === VOLATILITY (using ta library) ===
        atr = calculate_atr(h, l, c, 14)
        atr_val = safe_last(atr)
        atr_pct = (atr_val / price * 100) if price > 0 else 0
        
        bb_upper, bb_middle, bb_lower = calculate_bbands(c, 20, 2.0)
        bb_upper_val = safe_last(bb_upper, price * 1.02)
        bb_lower_val = safe_last(bb_lower, price * 0.98)
        bb_range = bb_upper_val - bb_lower_val
        bb_position = ((price - bb_lower_val) / bb_range * 100) if bb_range > 0 else 50
        
        # === TREND ===
        if ema_bullish_stack and above_sma200:
            trend = "Strong Uptrend"
            trend_score = 2
        elif above_ema20 and above_ema50:
            trend = "Uptrend"
            trend_score = 1
        elif ema_bearish_stack and not above_sma200:
            trend = "Strong Downtrend"
            trend_score = -2
        elif not above_ema20 and not above_ema50:
            trend = "Downtrend"
            trend_score = -1
        else:
            trend = "Sideways"
            trend_score = 0
        
        # === SUPPORT/RESISTANCE ===
        sr = find_support_resistance(h, l, c)
        support = sr['support']
        resistance = sr['resistance']
        
        dist_to_support = ((price - support) / price * 100) if price > 0 else 0
        dist_to_resistance = ((resistance - price) / price * 100) if price > 0 else 0
        
        return {
            # Price
            'price': price,
            'high_all': high_all,
            'low_all': low_all,
            'price_position': price_position,
            'avg_price': avg_price,
            'change_1d': change_1d,
            'change_5d': change_5d,
            'change_20d': change_20d,
            
            # MAs
            'ema_9': ema_9_val,
            'ema_20': ema_20_val,
            'ema_50': ema_50_val,
            'sma_200': sma_200_val,
            'above_ema20': above_ema20,
            'above_ema50': above_ema50,
            'above_sma200': above_sma200,
            'ema_bullish_stack': ema_bullish_stack,
            'ema_bearish_stack': ema_bearish_stack,
            
            # Momentum
            'rsi': rsi_val,
            'rsi_prev': rsi_prev,
            'bullish_div': bullish_div,
            'bearish_div': bearish_div,
            'macd': macd_val,
            'macd_signal': macd_sig_val,
            'macd_hist': macd_hist_val,
            'macd_cross_up': macd_cross_up,
            'macd_cross_down': macd_cross_down,
            
            # Volume
            'volume': v.iloc[-1],
            'vol_avg': vol_avg,
            'vol_ratio': vol_ratio,
            'obv_rising': obv_rising,
            'mfi': mfi_val,
            'cmf': cmf_val,
            
            # Volatility
            'atr': atr_val,
            'atr_pct': atr_pct,
            'bb_upper': bb_upper_val,
            'bb_lower': bb_lower_val,
            'bb_position': bb_position,
            
            # Trend
            'trend': trend,
            'trend_score': trend_score,
            
            # S/R
            'support': support,
            'resistance': resistance,
            'dist_to_support': dist_to_support,
            'dist_to_resistance': dist_to_resistance,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS MODULES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_price_position(self, m: Dict, mode: str, result: AnalysisResult):
        """Analyze where price is relative to its range"""
        pos = m['price_position']
        price = m['price']
        
        if mode == 'investment':
            # INVESTMENT MODE - Focus on value
            if pos <= 15:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='price',
                    title='Deep Value Zone',
                    explanation=f'Price is in the bottom 15% of its range - this is where long-term investors find the best opportunities. '
                               f'Historically, buying at these levels has generated superior returns over 3-5 year periods. '
                               f'This is an ideal time to accumulate aggressively.',
                    impact=30
                ))
            elif pos <= 30:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='price',
                    title='Strong Value',
                    explanation=f'Trading in the lower 30% of range - good value territory. '
                               f'Smart money typically builds positions at these levels. '
                               f'Consider systematic accumulation (DCA) to build your position.',
                    impact=20
                ))
            elif pos <= 45:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='price',
                    title='Fair Value',
                    explanation=f'Price below the midpoint offers reasonable entry. '
                               f'Suitable for regular monthly investments. '
                               f'Not a screaming buy, but not overvalued either.',
                    impact=10
                ))
            elif pos >= 90:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BEARISH,
                    category='price',
                    title='Extremely Extended',
                    explanation=f'Near all-time highs ({pos:.0f}% from bottom). '
                               f'This is where investors should be TAKING profits, not adding. '
                               f'Wait for a 15-20% pullback before considering new positions. '
                               f'If you hold, consider trimming 25-50%.',
                    impact=-30
                ))
            elif pos >= 75:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BEARISH,
                    category='price',
                    title='Overvalued Territory',
                    explanation=f'In the upper quartile - elevated risk of pullback. '
                               f'New money should wait. Existing holders consider trimming. '
                               f'Better opportunities will come on a correction.',
                    impact=-20
                ))
            elif pos >= 60:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.NEUTRAL,
                    category='price',
                    title='Above Average',
                    explanation=f'Slightly elevated at {pos:.0f}% of range. '
                               f'Not ideal for large new positions. '
                               f'Consider smaller allocations or wait for dip.',
                    impact=-5
                ))
        else:
            # TRADING MODES - Focus on levels
            if m['dist_to_support'] < 2:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='price',
                    title='At Support Level',
                    explanation=f'Price testing support at {self._fmt(m["support"])}. '
                               f'Support levels are where buyers historically step in. '
                               f'This creates a high-probability long entry with clear risk (below support). '
                               f'Risk/reward is favorable here.',
                    impact=20
                ))
            elif m['dist_to_support'] < 5:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='price',
                    title='Near Support',
                    explanation=f'Close to support zone. Good area to look for long entries. '
                               f'Wait for confirmation (bounce, bullish candle) before entering.',
                    impact=10
                ))
            
            if m['dist_to_resistance'] < 2:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BEARISH,
                    category='price',
                    title='At Resistance',
                    explanation=f'Testing resistance at {self._fmt(m["resistance"])}. '
                               f'Resistance is where sellers historically appear. '
                               f'Either wait for breakout confirmation or take profits here. '
                               f'Breakout traders: wait for close above with volume.',
                    impact=-15
                ))
    
    def _analyze_trend_structure(self, m: Dict, mode: str, result: AnalysisResult):
        """Analyze trend and market structure"""
        trend = m['trend']
        
        # Long-term trend (200 SMA)
        if mode == 'investment':
            if m['above_sma200']:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='trend',
                    title='Long-Term Uptrend Intact',
                    explanation=f'Price above 200-day average confirms we are in a bull market regime. '
                               f'In bull markets, buying dips is the winning strategy. '
                               f'The trend is your friend - accumulate on weakness.',
                    impact=20
                ))
            else:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BEARISH,
                    category='trend',
                    title='Below Long-Term Average',
                    explanation=f'Trading below 200-day average suggests structural weakness. '
                               f'This doesn\'t mean don\'t buy, but use smaller sizes and have patience. '
                               f'Best to DCA slowly rather than go all-in.',
                    impact=-10
                ))
        
        # EMA structure
        if m['ema_bullish_stack']:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='trend',
                title='Bullish Trend Structure',
                explanation=f'EMAs stacked bullish (9 > 20 > 50) - this is the ideal trend structure. '
                           f'Momentum is aligned with direction. '
                           f'{"Buy dips to EMAs" if mode == "investment" else "Look for pullbacks to EMA 20 for entries"}.',
                impact=15
            ))
        elif m['ema_bearish_stack']:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='trend',
                title='Bearish Trend Structure',
                explanation=f'EMAs stacked bearish (9 < 20 < 50) - downtrend confirmed. '
                           f'{"This is actually good for accumulators - lower prices ahead for buying" if mode == "investment" else "Avoid longs. Wait for structure to improve"}.',
                impact=-15 if mode != 'investment' else -5
            ))
        elif trend == 'Sideways':
            self.insights.append(InsightPoint(
                sentiment=Sentiment.NEUTRAL,
                category='trend',
                title='Consolidation Phase',
                explanation=f'No clear trend - price is ranging. '
                           f'{"Wait for breakout direction before committing large capital" if mode == "investment" else "Trade the range: buy support, sell resistance"}.',
                impact=0
            ))
    
    def _analyze_momentum(self, m: Dict, mode: str, result: AnalysisResult):
        """Analyze momentum indicators"""
        rsi = m['rsi']
        mfi = m['mfi']
        
        # RSI Analysis
        if rsi <= 25:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='momentum',
                title='RSI: Extreme Oversold',
                explanation=f'RSI at {rsi:.0f} - this is capitulation territory. '
                           f'When everyone is selling in panic, smart money is buying. '
                           f'Historically, RSI below 25 precedes strong reversals. '
                           f'{"Aggressive accumulation zone" if mode == "investment" else "High probability long setup"}.',
                impact=25
            ))
        elif rsi <= 35:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='momentum',
                title='RSI: Oversold',
                explanation=f'RSI at {rsi:.0f} indicates selling pressure is exhausted. '
                           f'Momentum is poised to shift bullish. '
                           f'Good area to start building positions.',
                impact=15
            ))
        elif rsi >= 80:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='momentum',
                title='RSI: Extreme Overbought',
                explanation=f'RSI at {rsi:.0f} - extreme greed/euphoria. '
                           f'When everyone is buying in FOMO, smart money is selling. '
                           f'High probability of pullback. '
                           f'{"Take profits, do NOT add here" if mode == "investment" else "Avoid new longs, look for shorts"}.',
                impact=-25
            ))
        elif rsi >= 70:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='momentum',
                title='RSI: Overbought',
                explanation=f'RSI at {rsi:.0f} shows stretched momentum. '
                           f'Doesn\'t mean sell everything, but be cautious. '
                           f'Not the time for new entries.',
                impact=-15
            ))
        
        # MFI (Money Flow Index - volume-weighted RSI)
        if mfi <= 20:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='momentum',
                title='Money Flow: Capitulation',
                explanation=f'MFI at {mfi:.0f} - money is flowing OUT heavily. '
                           f'This exhaustion often marks bottoms. '
                           f'When volume confirms oversold, reversals are more reliable.',
                impact=20
            ))
        elif mfi >= 80:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='momentum',
                title='Money Flow: Euphoria',
                explanation=f'MFI at {mfi:.0f} - buying climax on heavy volume. '
                           f'This often precedes distribution phase. '
                           f'Be prepared for profit-taking.',
                impact=-15
            ))
        
        # Divergences
        if m['bullish_div']:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='momentum',
                title='Bullish RSI Divergence',
                explanation=f'Price made lower low but RSI made higher low - classic bullish divergence. '
                           f'This signals weakening selling pressure and often precedes reversals. '
                           f'One of the most reliable reversal signals.',
                impact=20
            ))
        elif m['bearish_div']:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='momentum',
                title='Bearish RSI Divergence',
                explanation=f'Price made higher high but RSI made lower high - bearish divergence. '
                           f'Momentum not confirming price - warning sign. '
                           f'Often precedes corrections.',
                impact=-15
            ))
        
        # MACD
        if m['macd_cross_up']:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='momentum',
                title='MACD Bullish Cross',
                explanation=f'MACD just crossed above signal line - momentum shifting bullish. '
                           f'This is a buy signal for trend followers. '
                           f'More reliable when occurring below zero line.',
                impact=15
            ))
        elif m['macd_cross_down']:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='momentum',
                title='MACD Bearish Cross',
                explanation=f'MACD crossed below signal - momentum turning bearish. '
                           f'Trend followers would exit longs here. '
                           f'Wait for new bullish cross before re-entering.',
                impact=-15
            ))
    
    def _analyze_volume_flow(self, m: Dict, mode: str, result: AnalysisResult):
        """Analyze volume and institutional flow"""
        cmf = m['cmf']
        obv_rising = m['obv_rising']
        vol_ratio = m['vol_ratio']
        
        # CMF - Institutional flow
        if cmf >= 0.20:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='volume',
                title='Heavy Accumulation',
                explanation=f'CMF at {cmf:.2f} - strong institutional buying detected. '
                           f'Big money is accumulating. When institutions buy, follow them. '
                           f'This is the "smart money" signal to accumulate.',
                impact=25
            ))
        elif cmf >= 0.08:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='volume',
                title='Positive Money Flow',
                explanation=f'CMF at {cmf:.2f} - more buying pressure than selling. '
                           f'Healthy demand supporting price. '
                           f'Confirms bullish bias.',
                impact=12
            ))
        elif cmf <= -0.20:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='volume',
                title='Heavy Distribution',
                explanation=f'CMF at {cmf:.2f} - institutions are selling aggressively. '
                           f'When smart money exits, retail usually follows. '
                           f'{"Wait for CMF to turn positive before accumulating" if mode == "investment" else "Avoid longs until selling exhausts"}.',
                impact=-25
            ))
        elif cmf <= -0.08:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='volume',
                title='Negative Money Flow',
                explanation=f'CMF at {cmf:.2f} - selling pressure dominates. '
                           f'Supply exceeds demand. '
                           f'Be cautious with new positions.',
                impact=-12
            ))
        
        # OBV confirmation
        if obv_rising and m['trend_score'] >= 1:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='volume',
                title='Volume Confirms Uptrend',
                explanation=f'OBV rising with price - volume validates the move. '
                           f'Healthy trends have volume confirmation. '
                           f'This uptrend has legs.',
                impact=10
            ))
        elif not obv_rising and m['trend_score'] >= 1:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='volume',
                title='Volume Divergence Warning',
                explanation=f'Price rising but OBV falling - volume not confirming. '
                           f'This divergence warns the rally may be weak. '
                           f'Be ready for pullback.',
                impact=-10
            ))
        
        # Volume spike
        if vol_ratio >= 2.5:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.NEUTRAL,
                category='volume',
                title='Volume Spike Detected',
                explanation=f'Volume is {vol_ratio:.1f}x average - unusual activity. '
                           f'Big volume often marks turning points. '
                           f'Watch the direction - this move has conviction.',
                impact=5 if m['change_1d'] > 0 else -5
            ))
    
    def _analyze_volatility_risk(self, m: Dict, mode: str, result: AnalysisResult):
        """Analyze volatility and Bollinger Band position"""
        bb_pos = m['bb_position']
        atr_pct = m['atr_pct']
        
        # Bollinger Band extremes
        if bb_pos <= 5:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='structure',
                title='At Lower Bollinger Band',
                explanation=f'Price touching lower Bollinger Band - statistically oversold. '
                           f'~95% of price action occurs within the bands. '
                           f'Mean reversion likely - good entry zone.',
                impact=15
            ))
        elif bb_pos >= 95:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='structure',
                title='At Upper Bollinger Band',
                explanation=f'Price at upper Bollinger Band - statistically stretched. '
                           f'Either expect mean reversion or breakout. '
                           f'Not ideal for new entries - wait for pullback.',
                impact=-10
            ))
    
    def _analyze_existing_position(self, m: Dict, position: Dict, mode: str, result: AnalysisResult):
        """Analyze an existing position for the Trade Monitor"""
        entry = position.get('entry', 0)
        direction = position.get('direction', 'LONG').upper()
        
        if entry <= 0:
            return
        
        price = m['price']
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = ((price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - price) / entry) * 100
        
        result.position_pnl = pnl_pct
        
        # Position-specific analysis
        if pnl_pct >= 30:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.NEUTRAL,
                category='position',
                title='Significant Profit - Consider Trimming',
                explanation=f'Position up {pnl_pct:.1f}% - excellent gain! '
                           f'Rule of thumb: take 25-50% profit to lock in gains. '
                           f'Let the rest ride with a trailing stop. '
                           f'Never let a big winner turn into a loser.',
                impact=0
            ))
            result.should_trim = True
            result.position_advice = "Consider taking 25-50% profit"
            
        elif pnl_pct >= 15:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='position',
                title='Healthy Profit - Manage Risk',
                explanation=f'Position up {pnl_pct:.1f}%. '
                           f'Move stop loss to breakeven to create a "free trade". '
                           f'You can\'t lose money with stop at entry.',
                impact=5
            ))
            result.position_advice = "Move stop to breakeven"
            
        elif pnl_pct >= 5:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BULLISH,
                category='position',
                title='In Profit - Hold',
                explanation=f'Position up {pnl_pct:.1f}%. '
                           f'Trade working as planned. Hold with original stop. '
                           f'Let the trade play out.',
                impact=3
            ))
            result.position_advice = "Hold position"
            
        elif pnl_pct >= -5:
            # Small loss or breakeven
            # Only recommend ADD if price is BETTER than entry (negative PnL for longs)
            # Why add at the SAME price you already bought?
            price_is_better = pnl_pct < -0.5  # At least 0.5% below entry for longs
            
            if price_is_better and m['cmf'] > 0.05 and m['rsi'] < 60:
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='position',
                    title='Consider Adding',
                    explanation=f'Position at {pnl_pct:+.1f}% - better price than your entry! '
                               f'Positive money flow (CMF {m["cmf"]:.2f}) supports the thesis. '
                               f'Adding here improves your average cost.',
                    impact=5
                ))
                result.should_add = True
                result.position_advice = "Consider adding at this better price"
            elif abs(pnl_pct) < 0.5:
                # At breakeven - don't recommend add, just hold
                result.position_advice = "Hold - at entry price, monitor for move"
            else:
                result.position_advice = "Hold - monitor closely"
                
        elif pnl_pct >= -15:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='position',
                title='Drawdown - Evaluate',
                explanation=f'Position down {abs(pnl_pct):.1f}%. '
                           f'Ask: "Would I enter this trade today?" '
                           f'If no, consider reducing size. '
                           f'If yes, hold with discipline.',
                impact=-10
            ))
            result.position_advice = "Re-evaluate thesis"
            
        else:
            self.insights.append(InsightPoint(
                sentiment=Sentiment.BEARISH,
                category='position',
                title='Significant Loss - Review',
                explanation=f'Position down {abs(pnl_pct):.1f}% - significant drawdown. '
                           f'Either your thesis is wrong or you need more patience. '
                           f'Consider: cut losses and re-enter at better level, '
                           f'or reduce size to manage emotional impact.',
                impact=-20
            ))
            result.should_trim = True
            result.position_advice = "Review position - consider reducing"
        
        # Additional add/trim signals based on technicals
        if not result.should_add and not result.should_trim:
            # Only recommend ADD if price is actually LOWER than entry (for longs)
            price_is_better = pnl_pct < -0.5  # At least 0.5% better price
            
            if price_is_better and m['rsi'] < 35 and m['cmf'] > 0 and pnl_pct < 10:
                result.should_add = True
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.BULLISH,
                    category='position',
                    title='Potential Add Opportunity',
                    explanation=f'RSI oversold ({m["rsi"]:.0f}) with positive money flow at a better price than entry. '
                               f'Adding here improves your average cost significantly.',
                    impact=10
                ))
            elif m['rsi'] > 75 and pnl_pct > 10:
                result.should_trim = True
                self.insights.append(InsightPoint(
                    sentiment=Sentiment.NEUTRAL,
                    category='position',
                    title='Take Partial Profit',
                    explanation=f'RSI overbought with profit - scale out opportunity. '
                               f'Taking some profit here is smart risk management.',
                    impact=0
                ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION DETERMINATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _determine_action(self, result: AnalysisResult, m: Dict, mode: str, 
                          existing_position: Optional[Dict]) -> Action:
        """Determine the recommended action"""
        net = result.net_score
        
        # If monitoring position
        if existing_position:
            if result.should_trim and result.position_pnl > 20:
                return Action.TAKE_PARTIAL_PROFIT
            elif result.should_trim:
                return Action.TRIM
            elif result.should_add:
                return Action.ADD_TO_POSITION
            else:
                return Action.HOLD
        
        # Investment mode
        if mode == 'investment':
            if net >= 50:
                return Action.STRONG_BUY
            elif net >= 30:
                return Action.ACCUMULATE
            elif net >= 15:
                return Action.LIGHT_ACCUMULATE
            elif net >= -15:
                return Action.HOLD
            elif net >= -30:
                return Action.LIGHT_TRIM
            elif net >= -50:
                return Action.TRIM
            else:
                return Action.STRONG_SELL
        
        # Trading modes
        else:
            # SMART LOGIC: Differentiate between "bad trend" and "good trend but bad timing"
            trend = m.get('trend', '')
            rsi = m.get('rsi', 50)
            cmf = m.get('cmf', 0)
            
            # Check if it's an uptrend but overbought (bad TIMING, not bad TRADE)
            is_uptrend = 'uptrend' in trend.lower() or m.get('trend_score', 0) >= 50
            is_overbought = rsi > 70
            is_distribution = cmf < -0.05
            
            # Uptrend but overbought/distribution = WAIT FOR PULLBACK, not SELL
            if is_uptrend and (is_overbought or is_distribution) and net < 0 and net >= -50:
                result.wait_for_pullback = True
                result.pullback_reason = []
                if is_overbought:
                    result.pullback_reason.append(f"RSI overbought ({rsi:.0f})")
                if is_distribution:
                    result.pullback_reason.append(f"Money outflow (CMF {cmf:.2f})")
                return Action.WAIT  # Changed from SELL to WAIT
            
            # Normal logic for other cases
            if net >= 40:
                return Action.STRONG_BUY
            elif net >= 20:
                return Action.BUY
            elif net >= -20:
                return Action.WAIT
            elif net >= -40:
                # Only SELL if trend is actually bearish
                if is_uptrend:
                    result.wait_for_pullback = True
                    return Action.WAIT
                return Action.SELL
            else:
                return Action.STRONG_SELL
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NARRATIVE TEXT GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _create_headline(self, result: AnalysisResult, m: Dict, mode: str) -> str:
        """Create the main headline"""
        action = result.action.value
        symbol = result.symbol
        conf = result.confidence
        style = result.style
        
        if mode == 'investment':
            if result.action in [Action.STRONG_BUY, Action.ACCUMULATE, Action.LIGHT_ACCUMULATE]:
                return f"{style['emoji']} {symbol}: {action} - Good time to build position"
            elif result.action == Action.HOLD:
                return f"{style['emoji']} {symbol}: {action} - Maintain current allocation"
            elif result.action in [Action.TAKE_PARTIAL_PROFIT, Action.LIGHT_TRIM, Action.TRIM]:
                return f"{style['emoji']} {symbol}: {action} - Consider reducing exposure"
            else:
                return f"{style['emoji']} {symbol}: {action} - Risk elevated"
        else:
            if result.action in [Action.STRONG_BUY, Action.BUY]:
                return f"{style['emoji']} {symbol}: {action} ({conf}% confidence)"
            elif result.action == Action.WAIT:
                # Check if it's wait-for-pullback vs no-setup
                if result.wait_for_pullback:
                    return f"⏳ {symbol}: WAIT FOR PULLBACK - Trend is UP but timing is wrong"
                return f"{style['emoji']} {symbol}: No Clear Setup - Wait"
            elif result.action == Action.ADD_TO_POSITION:
                return f"{style['emoji']} {symbol}: Consider Adding to Position"
            elif result.action == Action.TAKE_PARTIAL_PROFIT:
                return f"{style['emoji']} {symbol}: Take Partial Profits"
            else:
                return f"{style['emoji']} {symbol}: {action} ({conf}% confidence)"
    
    def _create_summary(self, result: AnalysisResult, m: Dict, mode: str) -> str:
        """Create executive summary paragraph"""
        bull_pts = len([i for i in result.insights if i.sentiment == Sentiment.BULLISH])
        bear_pts = len([i for i in result.insights if i.sentiment == Sentiment.BEARISH])
        
        price_desc = ""
        if m['price_position'] < 30:
            price_desc = "trading near lows"
        elif m['price_position'] > 70:
            price_desc = "near highs"
        else:
            price_desc = "mid-range"
        
        if mode == 'investment':
            if result.action in [Action.STRONG_BUY, Action.ACCUMULATE]:
                return (f"Analysis reveals {bull_pts} bullish factors vs {bear_pts} bearish, "
                       f"with price {price_desc}. Current conditions favor accumulation. "
                       f"This is a good time to add to your position through DCA.")
            elif result.action == Action.LIGHT_ACCUMULATE:
                return (f"Moderately bullish with {bull_pts} positive factors. "
                       f"Consider small additions to position. Not all-in territory, but favorable.")
            elif result.action == Action.HOLD:
                return (f"Mixed signals ({bull_pts} bullish, {bear_pts} bearish). "
                       f"Hold existing position but wait for clearer setup before adding capital.")
            else:
                return (f"Caution warranted with {bear_pts} bearish factors dominating. "
                       f"Consider taking profits or reducing position. "
                       f"Better entry prices likely ahead.")
        else:
            if result.action in [Action.STRONG_BUY, Action.BUY]:
                return (f"Technical setup is bullish with {bull_pts} confirming factors. "
                       f"Entry conditions present with clearly defined risk.")
            elif result.action == Action.WAIT:
                # Check if it's wait-for-pullback vs no-setup
                if result.wait_for_pullback:
                    reasons = ', '.join(result.pullback_reason) if result.pullback_reason else "overbought conditions"
                    return (f"⚠️ The TREND is bullish (structure supports longs), BUT the TIMING is wrong. "
                           f"Currently: {reasons}. "
                           f"Wait for a pullback to support levels before entering long.")
                return (f"No clear edge ({bull_pts} bull vs {bear_pts} bear factors). "
                       f"Wait for momentum shift or test of key levels before entering.")
            else:
                return (f"Bearish setup with {bear_pts} warning signals. "
                       f"Avoid long positions until conditions improve.")
    
    def _set_price_levels(self, result: AnalysisResult, m: Dict, mode: str):
        """Calculate and set all price levels"""
        price = m['price']
        atr = m['atr']
        support = m['support']
        resistance = m['resistance']
        
        result.entry = price
        
        if mode == 'investment':
            # DCA zones for accumulation
            result.dca_zones = [
                {'label': 'Current Price', 'price': price, 'allocation': '20-25%', 'note': 'If ready to start'},
                {'label': 'Zone 1 (-5%)', 'price': price * 0.95, 'allocation': '25%', 'note': 'Light pullback'},
                {'label': 'Zone 2 (-10%)', 'price': price * 0.90, 'allocation': '25%', 'note': 'Correction'},
                {'label': 'Zone 3 (-15%)', 'price': price * 0.85, 'allocation': '25%', 'note': 'Deep value'},
                {'label': 'Zone 4 (-20%)', 'price': price * 0.80, 'allocation': '25%', 'note': 'Maximum fear'},
            ]
            
            # Trim zones
            result.trim_zones = [
                {'label': '+20% Profit', 'price': price * 1.20, 'action': 'Trim 25%'},
                {'label': '+35% Profit', 'price': price * 1.35, 'action': 'Trim another 25%'},
                {'label': '+50% Profit', 'price': price * 1.50, 'action': 'Evaluate full exit'},
            ]
            
            # Long-term targets
            result.tp1 = price * 1.15  # +15%
            result.tp2 = price * 1.30  # +30%
            result.tp3 = price * 1.50  # +50%
            result.stop_loss = price * 0.75  # -25% (wide for investment)
            
        else:
            # Trading levels
            result.stop_loss = max(support * 0.995, price - (atr * 2))
            risk = abs(price - result.stop_loss)
            
            result.tp1 = price + (risk * 1.5)  # 1.5R
            result.tp2 = price + (risk * 2.5)  # 2.5R
            result.tp3 = price + (risk * 4.0)  # 4R
            
            result.risk_pct = (risk / price) * 100
            result.reward_pct = ((result.tp2 - price) / price) * 100
            result.rr_ratio = result.reward_pct / result.risk_pct if result.risk_pct > 0 else 0
    
    def _create_action_plan(self, result: AnalysisResult, m: Dict, mode: str,
                            existing_position: Optional[Dict]) -> List[str]:
        """Generate specific action steps"""
        steps = []
        price = m['price']
        
        if existing_position:
            # Position management steps
            if result.action == Action.ADD_TO_POSITION:
                steps.append(f"➕ Consider adding 10-25% to position at {self._fmt(price)}")
                steps.append(f"📊 This will improve your average entry price")
                steps.append(f"⚠️ Only add if your original thesis is still valid")
            elif result.action == Action.TAKE_PARTIAL_PROFIT:
                steps.append(f"💰 Take 25-50% profit at current price {self._fmt(price)}")
                steps.append(f"📈 Move stop to breakeven on remaining position")
                steps.append(f"🎯 Let winners run, but lock in gains")
            elif result.action == Action.TRIM:
                steps.append(f"⚠️ Consider reducing position by 25-50%")
                steps.append(f"📉 Conditions have deteriorated")
                steps.append(f"💡 Re-evaluate at support: {self._fmt(m['support'])}")
            elif result.action == Action.HOLD:
                steps.append(f"⏸️ Maintain current position")
                steps.append(f"👀 Monitor: Support {self._fmt(m['support'])} | Resistance {self._fmt(m['resistance'])}")
                if result.position_pnl > 0:
                    steps.append(f"✅ Trade working (+{result.position_pnl:.1f}%)")
            return steps
        
        if mode == 'investment':
            if result.action in [Action.STRONG_BUY, Action.ACCUMULATE]:
                steps.append(f"✅ ACCUMULATE: Deploy capital across DCA zones")
                steps.append(f"📊 Start with 20-25% at current price {self._fmt(price)}")
                steps.append(f"⏰ Set limit orders at each DCA zone below")
                steps.append(f"📈 Long-term targets: {self._fmt(result.tp1)} / {self._fmt(result.tp2)} / {self._fmt(result.tp3)}")
            elif result.action == Action.LIGHT_ACCUMULATE:
                steps.append(f"➕ Light accumulation appropriate")
                steps.append(f"📊 Deploy 10-15% at current levels")
                steps.append(f"⏳ Save capital for better entries on dips")
            elif result.action == Action.HOLD:
                steps.append(f"⏸️ Hold existing positions")
                steps.append(f"💤 Wait for RSI < 40 or -10% pullback to add")
                steps.append(f"📊 Current position is fine, just don't chase")
            elif result.action in [Action.LIGHT_TRIM, Action.TRIM]:
                steps.append(f"💰 Consider trimming 20-30% of position")
                steps.append(f"📊 Lock in profits at extended levels")
                steps.append(f"⏳ Wait for pullback to re-enter trimmed portion")
            elif result.action == Action.STRONG_SELL:
                steps.append(f"⚠️ Significant risk - reduce exposure 50%+")
                steps.append(f"🛑 Protect capital - better prices ahead")
        else:
            # Trading
            if result.action in [Action.STRONG_BUY, Action.BUY]:
                steps.append(f"🎯 Entry: {self._fmt(price)}")
                steps.append(f"🛑 Stop Loss: {self._fmt(result.stop_loss)} ({result.risk_pct:.1f}% risk)")
                steps.append(f"✅ TP1: {self._fmt(result.tp1)} - Take 33%")
                steps.append(f"✅ TP2: {self._fmt(result.tp2)} - Take 33%")
                steps.append(f"✅ TP3: {self._fmt(result.tp3)} - Take remaining")
                steps.append(f"📊 Risk/Reward: 1:{result.rr_ratio:.1f}")
            elif result.action == Action.WAIT:
                # Different advice for wait-for-pullback vs no-setup
                if result.wait_for_pullback:
                    steps.append(f"⏳ WAIT FOR PULLBACK - Trend is UP but bad timing!")
                    reasons = ', '.join(result.pullback_reason) if result.pullback_reason else "overbought"
                    steps.append(f"⚠️ Problem: {reasons}")
                    steps.append(f"✅ Bullish when price pulls back to {self._fmt(m['support'])} area")
                    steps.append(f"🎯 Set alert at support: {self._fmt(m['support'])}")
                else:
                    steps.append(f"⏳ No entry signal - patience")
                    steps.append(f"👀 Watch for: Bounce at {self._fmt(m['support'])}")
                    steps.append(f"👀 Or: Breakout above {self._fmt(m['resistance'])}")
            else:
                steps.append(f"🔴 Avoid long positions")
                steps.append(f"📉 Consider short if breaks {self._fmt(m['support'])}")
        
        return steps
    
    def _assess_risk(self, m: Dict, mode: str) -> Tuple[str, str]:
        """Assess risk level and provide explanation"""
        atr_pct = m['atr_pct']
        rsi = m['rsi']
        bb_pos = m['bb_position']
        
        risk_factors = []
        
        if atr_pct > 5:
            risk_factors.append(f"High volatility ({atr_pct:.1f}% daily range)")
        if rsi > 75 or rsi < 25:
            risk_factors.append(f"Extreme RSI ({rsi:.0f})")
        if bb_pos > 95 or bb_pos < 5:
            risk_factors.append("At Bollinger Band extreme")
        if m['vol_ratio'] > 2:
            risk_factors.append(f"Unusual volume ({m['vol_ratio']:.1f}x)")
        
        if len(risk_factors) >= 3:
            return "High", "Multiple risk factors: " + ", ".join(risk_factors)
        elif len(risk_factors) >= 1:
            return "Medium", "Risk factors: " + ", ".join(risk_factors)
        else:
            return "Low", "No significant risk factors detected"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _fmt(self, price: float) -> str:
        """Format price - 2 decimals for >= $1, 4 for < $1"""
        if price is None or price == 0:
            return "$0.00"
        if price >= 1:
            return f"${price:,.2f}"
        return f"${price:.4f}"


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(df: pd.DataFrame, symbol: str, mode: str, timeframe: str = '1d',
            existing_position: Optional[Dict] = None) -> AnalysisResult:
    """
    Convenience function to run analysis
    
    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        symbol: Asset symbol
        mode: 'scalp', 'day_trade', 'swing', 'investment'
        timeframe: Timeframe string
        existing_position: Optional dict with 'entry', 'direction', 'size'
    
    Returns:
        AnalysisResult with complete narrative
    """
    engine = MasterNarrative()
    return engine.analyze(df, symbol, mode, timeframe, existing_position)

