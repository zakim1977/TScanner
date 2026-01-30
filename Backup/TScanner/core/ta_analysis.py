"""
TA Analysis Module - FULLY POWERED BY 'ta' LIBRARY
=================================================
This is the SINGLE SOURCE OF TRUTH for all technical analysis.
ALL indicators, ALL scoring, ALL signals come from here.

Uses: pip install ta
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import ta library
try:
    import ta
    from ta import add_all_ta_features
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
    from ta.volume import (
        OnBalanceVolumeIndicator, MFIIndicator, ChaikinMoneyFlowIndicator,
        VolumeWeightedAveragePrice, ForceIndexIndicator
    )
    TA_AVAILABLE = True
    print("✅ ta library loaded successfully")
except ImportError:
    TA_AVAILABLE = False
    print("❌ ta library not found. Install with: pip install ta")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TAResult:
    """Complete technical analysis result"""
    # Basic info
    symbol: str = ""
    timeframe: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Current price
    price: float = 0
    
    # Scores
    bullish_score: int = 0
    bearish_score: int = 0
    net_score: int = 0
    total_score: int = 50  # 0-100, 50 = neutral
    
    # Grade and Action
    grade: str = "C"
    action: str = "HOLD"
    confidence: int = 50
    
    # All factors that contributed to score
    factors: List[Dict] = field(default_factory=list)
    
    # Raw indicator values (for display)
    indicators: Dict = field(default_factory=dict)
    
    # Signals
    is_bullish: bool = False
    is_bearish: bool = False
    
    # Trade levels
    entry: float = 0
    stop_loss: float = 0
    tp1: float = 0
    tp2: float = 0
    tp3: float = 0
    risk_pct: float = 0
    
    # Narrative
    summary: str = ""
    action_plan: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class TAAnalyzer:
    """
    Complete Technical Analysis using ta library.
    One class to rule them all!
    """
    
    def __init__(self):
        self.df = None
        self.result = None
    
    def analyze(self, df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> TAResult:
        """
        Run COMPLETE technical analysis on a DataFrame.
        
        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
            symbol: Trading pair symbol
            timeframe: Timeframe string (e.g., '1h', '4h')
            
        Returns:
            TAResult with all analysis
        """
        if not TA_AVAILABLE:
            return self._fallback_analysis(df, symbol, timeframe)
        
        self.df = df.copy()
        self.result = TAResult(symbol=symbol, timeframe=timeframe)
        
        # Step 1: Add ALL ta indicators to DataFrame
        self._add_all_indicators()
        
        # Step 2: Extract current values
        self._extract_current_values()
        
        # Step 3: Calculate scores
        self._calculate_scores()
        
        # Step 4: Determine grade and action
        self._determine_grade_and_action()
        
        # Step 5: Calculate trade levels
        self._calculate_trade_levels()
        
        # Step 6: Generate narrative
        self._generate_narrative()
        
        return self.result
    
    def _add_all_indicators(self):
        """Add ALL ta library indicators to the DataFrame"""
        df = self.df
        h, l, c, v = df['High'], df['Low'], df['Close'], df['Volume']
        
        # ═══════════════════════════════════════════════════════════════
        # MOMENTUM INDICATORS
        # ═══════════════════════════════════════════════════════════════
        
        # RSI
        df['rsi'] = RSIIndicator(close=c, window=14).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=h, low=l, close=c, window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # MACD
        macd = MACD(close=c, window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Williams %R
        df['williams_r'] = WilliamsRIndicator(high=h, low=l, close=c, lbp=14).williams_r()
        
        # CCI
        df['cci'] = CCIIndicator(high=h, low=l, close=c, window=20).cci()
        
        # ═══════════════════════════════════════════════════════════════
        # TREND INDICATORS
        # ═══════════════════════════════════════════════════════════════
        
        # EMAs
        df['ema_9'] = EMAIndicator(close=c, window=9).ema_indicator()
        df['ema_20'] = EMAIndicator(close=c, window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(close=c, window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close=c, window=200).ema_indicator() if len(c) >= 200 else EMAIndicator(close=c, window=50).ema_indicator()
        
        # SMAs
        df['sma_20'] = SMAIndicator(close=c, window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=c, window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=c, window=200).sma_indicator() if len(c) >= 200 else SMAIndicator(close=c, window=50).sma_indicator()
        
        # ADX (trend strength)
        adx = ADXIndicator(high=h, low=l, close=c, window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()  # +DI
        df['adx_neg'] = adx.adx_neg()  # -DI
        
        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY INDICATORS
        # ═══════════════════════════════════════════════════════════════
        
        # Bollinger Bands
        bb = BollingerBands(close=c, window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pct'] = bb.bollinger_pband()  # %B (position within bands)
        
        # ATR
        df['atr'] = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
        
        # Keltner Channel
        kc = KeltnerChannel(high=h, low=l, close=c, window=20, window_atr=10)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        # ═══════════════════════════════════════════════════════════════
        # VOLUME INDICATORS
        # ═══════════════════════════════════════════════════════════════
        
        # OBV
        df['obv'] = OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
        
        # MFI (Money Flow Index)
        df['mfi'] = MFIIndicator(high=h, low=l, close=c, volume=v, window=14).money_flow_index()
        
        # CMF (Chaikin Money Flow)
        df['cmf'] = ChaikinMoneyFlowIndicator(high=h, low=l, close=c, volume=v, window=20).chaikin_money_flow()
        
        # VWAP
        try:
            df['vwap'] = VolumeWeightedAveragePrice(high=h, low=l, close=c, volume=v).volume_weighted_average_price()
        except:
            # Fallback VWAP calculation
            typical_price = (h + l + c) / 3
            df['vwap'] = (typical_price * v).cumsum() / v.cumsum()
        
        # Force Index
        df['force_index'] = ForceIndexIndicator(close=c, volume=v, window=13).force_index()
        
        # Volume SMA
        df['volume_sma'] = v.rolling(20).mean()
        df['volume_ratio'] = v / df['volume_sma']
        
        self.df = df
    
    def _extract_current_values(self):
        """Extract current (latest) indicator values"""
        df = self.df
        
        def safe_last(col, default=0):
            try:
                val = df[col].iloc[-1]
                return default if pd.isna(val) else float(val)
            except:
                return default
        
        price = safe_last('Close')
        self.result.price = price
        
        # Store all indicator values
        self.result.indicators = {
            # Price
            'price': price,
            'price_change_pct': ((price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0,
            
            # Momentum
            'rsi': safe_last('rsi', 50),
            'stoch_k': safe_last('stoch_k', 50),
            'stoch_d': safe_last('stoch_d', 50),
            'macd': safe_last('macd'),
            'macd_signal': safe_last('macd_signal'),
            'macd_diff': safe_last('macd_diff'),
            'williams_r': safe_last('williams_r', -50),
            'cci': safe_last('cci'),
            
            # Trend
            'ema_9': safe_last('ema_9', price),
            'ema_20': safe_last('ema_20', price),
            'ema_50': safe_last('ema_50', price),
            'ema_200': safe_last('ema_200', price),
            'sma_20': safe_last('sma_20', price),
            'sma_50': safe_last('sma_50', price),
            'sma_200': safe_last('sma_200', price),
            'adx': safe_last('adx', 25),
            'adx_pos': safe_last('adx_pos'),
            'adx_neg': safe_last('adx_neg'),
            
            # Volatility
            'bb_upper': safe_last('bb_upper', price * 1.02),
            'bb_middle': safe_last('bb_middle', price),
            'bb_lower': safe_last('bb_lower', price * 0.98),
            'bb_width': safe_last('bb_width'),
            'bb_pct': safe_last('bb_pct', 0.5),  # 0-1, 0.5 = middle
            'atr': safe_last('atr'),
            'atr_pct': (safe_last('atr') / price * 100) if price > 0 else 0,
            
            # Volume
            'obv': safe_last('obv'),
            'obv_slope': (df['obv'].iloc[-1] - df['obv'].iloc[-5]) if len(df) >= 5 else 0,
            'mfi': safe_last('mfi', 50),
            'cmf': safe_last('cmf'),
            'vwap': safe_last('vwap', price),
            'force_index': safe_last('force_index'),
            'volume_ratio': safe_last('volume_ratio', 1),
            
            # Derived
            'above_vwap': price > safe_last('vwap', price),
            'above_ema_20': price > safe_last('ema_20', price),
            'above_ema_50': price > safe_last('ema_50', price),
            'above_sma_200': price > safe_last('sma_200', price),
            'ema_bullish': safe_last('ema_9', price) > safe_last('ema_20', price) > safe_last('ema_50', price),
            'ema_bearish': safe_last('ema_9', price) < safe_last('ema_20', price) < safe_last('ema_50', price),
            'macd_bullish': safe_last('macd') > safe_last('macd_signal'),
            'strong_trend': safe_last('adx', 25) > 25,
        }
        
        # Support/Resistance
        lookback = min(20, len(df) - 1)
        self.result.indicators['support'] = df['Low'].tail(lookback).min()
        self.result.indicators['resistance'] = df['High'].tail(lookback).max()
    
    def _calculate_scores(self):
        """Calculate bullish/bearish scores from indicators"""
        ind = self.result.indicators
        bullish = 0
        bearish = 0
        factors = []
        
        # ═══════════════════════════════════════════════════════════════
        # RSI SCORING (max ±20 points)
        # ═══════════════════════════════════════════════════════════════
        rsi = ind['rsi']
        if rsi < 30:
            pts = 20
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'RSI Oversold ({rsi:.0f})', 'points': pts,
                          'explanation': 'RSI below 30 indicates oversold conditions. Historically a buying opportunity.'})
        elif rsi < 40:
            pts = 10
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'RSI Low ({rsi:.0f})', 'points': pts,
                          'explanation': 'RSI in lower range suggests potential upside.'})
        elif rsi > 70:
            pts = 20
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'RSI Overbought ({rsi:.0f})', 'points': pts,
                          'explanation': 'RSI above 70 indicates overbought conditions. Price may pull back.'})
        elif rsi > 60:
            pts = 10
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'RSI High ({rsi:.0f})', 'points': pts,
                          'explanation': 'RSI elevated, momentum may slow.'})
        
        # ═══════════════════════════════════════════════════════════════
        # MACD SCORING (max ±15 points)
        # ═══════════════════════════════════════════════════════════════
        if ind['macd_bullish']:
            pts = 15
            bullish += pts
            factors.append({'type': 'bullish', 'name': 'MACD Bullish', 'points': pts,
                          'explanation': 'MACD above signal line indicates bullish momentum.'})
        else:
            pts = 15
            bearish += pts
            factors.append({'type': 'bearish', 'name': 'MACD Bearish', 'points': pts,
                          'explanation': 'MACD below signal line indicates bearish momentum.'})
        
        # ═══════════════════════════════════════════════════════════════
        # EMA STACK SCORING (max ±20 points)
        # ═══════════════════════════════════════════════════════════════
        if ind['ema_bullish']:
            pts = 20
            bullish += pts
            factors.append({'type': 'bullish', 'name': 'EMAs Stacked Bullish (9>20>50)', 'points': pts,
                          'explanation': 'Perfect bullish EMA alignment. Strong uptrend structure.'})
        elif ind['ema_bearish']:
            pts = 20
            bearish += pts
            factors.append({'type': 'bearish', 'name': 'EMAs Stacked Bearish (9<20<50)', 'points': pts,
                          'explanation': 'Perfect bearish EMA alignment. Strong downtrend structure.'})
        elif ind['above_ema_20'] and ind['above_ema_50']:
            pts = 10
            bullish += pts
            factors.append({'type': 'bullish', 'name': 'Above Key EMAs', 'points': pts,
                          'explanation': 'Price above EMA 20 and 50. Bullish bias.'})
        elif not ind['above_ema_20'] and not ind['above_ema_50']:
            pts = 10
            bearish += pts
            factors.append({'type': 'bearish', 'name': 'Below Key EMAs', 'points': pts,
                          'explanation': 'Price below EMA 20 and 50. Bearish bias.'})
        
        # ═══════════════════════════════════════════════════════════════
        # CMF - MONEY FLOW (max ±25 points) - MOST IMPORTANT!
        # ═══════════════════════════════════════════════════════════════
        cmf = ind['cmf']
        if cmf > 0.20:
            pts = 25
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'Strong Money Inflow (CMF {cmf:.2f})', 'points': pts,
                          'explanation': 'Heavy institutional buying. Smart money accumulating.'})
        elif cmf > 0.10:
            pts = 15
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'Money Inflow (CMF {cmf:.2f})', 'points': pts,
                          'explanation': 'Positive money flow. More buying than selling pressure.'})
        elif cmf > 0.05:
            pts = 8
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'Slight Inflow (CMF {cmf:.2f})', 'points': pts,
                          'explanation': 'Mild buying pressure detected.'})
        elif cmf < -0.20:
            pts = 25
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'Strong Money Outflow (CMF {cmf:.2f})', 'points': pts,
                          'explanation': 'Heavy institutional selling. Smart money distributing.'})
        elif cmf < -0.10:
            pts = 15
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'Money Outflow (CMF {cmf:.2f})', 'points': pts,
                          'explanation': 'Negative money flow. More selling than buying pressure.'})
        elif cmf < -0.05:
            pts = 8
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'Slight Outflow (CMF {cmf:.2f})', 'points': pts,
                          'explanation': 'Mild selling pressure detected.'})
        
        # ═══════════════════════════════════════════════════════════════
        # MFI SCORING (max ±15 points)
        # ═══════════════════════════════════════════════════════════════
        mfi = ind['mfi']
        if mfi < 20:
            pts = 15
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'MFI Oversold ({mfi:.0f})', 'points': pts,
                          'explanation': 'Money Flow Index indicates extreme selling exhaustion.'})
        elif mfi > 80:
            pts = 15
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'MFI Overbought ({mfi:.0f})', 'points': pts,
                          'explanation': 'Money Flow Index indicates extreme buying exhaustion.'})
        
        # ═══════════════════════════════════════════════════════════════
        # OBV SCORING (max ±10 points)
        # ═══════════════════════════════════════════════════════════════
        obv_slope = ind['obv_slope']
        if obv_slope > 0:
            pts = 10
            bullish += pts
            factors.append({'type': 'bullish', 'name': 'OBV Rising', 'points': pts,
                          'explanation': 'On-Balance Volume rising. Volume confirming price direction.'})
        else:
            pts = 10
            bearish += pts
            factors.append({'type': 'bearish', 'name': 'OBV Falling', 'points': pts,
                          'explanation': 'On-Balance Volume falling. Volume not confirming - divergence warning.'})
        
        # ═══════════════════════════════════════════════════════════════
        # VWAP SCORING (max ±10 points)
        # ═══════════════════════════════════════════════════════════════
        if ind['above_vwap']:
            pts = 10
            bullish += pts
            factors.append({'type': 'bullish', 'name': 'Above VWAP', 'points': pts,
                          'explanation': 'Price above Volume Weighted Average Price. Institutional buyers in control.'})
        else:
            pts = 10
            bearish += pts
            factors.append({'type': 'bearish', 'name': 'Below VWAP', 'points': pts,
                          'explanation': 'Price below Volume Weighted Average Price. Institutional sellers in control.'})
        
        # ═══════════════════════════════════════════════════════════════
        # BOLLINGER BANDS SCORING (max ±15 points)
        # ═══════════════════════════════════════════════════════════════
        bb_pct = ind['bb_pct'] * 100  # Convert to 0-100
        if bb_pct < 15:
            pts = 15
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'At BB Lower ({bb_pct:.0f}%)', 'points': pts,
                          'explanation': 'Price at lower Bollinger Band. Mean reversion likely - oversold.'})
        elif bb_pct < 30:
            pts = 8
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'Near BB Lower ({bb_pct:.0f}%)', 'points': pts,
                          'explanation': 'Price in lower range of Bollinger Bands.'})
        elif bb_pct > 85:
            pts = 15
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'At BB Upper ({bb_pct:.0f}%)', 'points': pts,
                          'explanation': 'Price at upper Bollinger Band. Mean reversion likely - overbought.'})
        elif bb_pct > 70:
            pts = 8
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'Near BB Upper ({bb_pct:.0f}%)', 'points': pts,
                          'explanation': 'Price in upper range of Bollinger Bands.'})
        
        # ═══════════════════════════════════════════════════════════════
        # STOCHASTIC SCORING (max ±10 points)
        # ═══════════════════════════════════════════════════════════════
        stoch_k = ind['stoch_k']
        if stoch_k < 20:
            pts = 10
            bullish += pts
            factors.append({'type': 'bullish', 'name': f'Stochastic Oversold ({stoch_k:.0f})', 'points': pts,
                          'explanation': 'Stochastic in oversold territory. Bounce likely.'})
        elif stoch_k > 80:
            pts = 10
            bearish += pts
            factors.append({'type': 'bearish', 'name': f'Stochastic Overbought ({stoch_k:.0f})', 'points': pts,
                          'explanation': 'Stochastic in overbought territory. Pullback likely.'})
        
        # ═══════════════════════════════════════════════════════════════
        # ADX TREND STRENGTH BONUS (max ±5 points)
        # ═══════════════════════════════════════════════════════════════
        if ind['strong_trend']:
            adx = ind['adx']
            if bullish > bearish:
                pts = 5
                bullish += pts
                factors.append({'type': 'bullish', 'name': f'Strong Trend (ADX {adx:.0f})', 'points': pts,
                              'explanation': 'ADX > 25 confirms strong trend. Momentum on bulls side.'})
            elif bearish > bullish:
                pts = 5
                bearish += pts
                factors.append({'type': 'bearish', 'name': f'Strong Trend (ADX {adx:.0f})', 'points': pts,
                              'explanation': 'ADX > 25 confirms strong trend. Momentum on bears side.'})
        
        # ═══════════════════════════════════════════════════════════════
        # CALCULATE FINAL SCORE
        # ═══════════════════════════════════════════════════════════════
        net_score = bullish - bearish
        
        # Normalize to 0-100 (50 = neutral)
        # Max possible: ~140 bullish or ~140 bearish
        total_score = 50 + (net_score * 0.35)
        total_score = max(0, min(100, total_score))
        
        # Store results
        self.result.bullish_score = bullish
        self.result.bearish_score = bearish
        self.result.net_score = net_score
        self.result.total_score = int(total_score)
        self.result.factors = factors
        self.result.is_bullish = bullish > bearish
        self.result.is_bearish = bearish > bullish
    
    def _determine_grade_and_action(self):
        """Determine grade and recommended action from score"""
        score = self.result.total_score
        
        # Grade
        if score >= 80:
            self.result.grade = 'A+'
            self.result.action = 'STRONG BUY'
            self.result.confidence = 90
        elif score >= 70:
            self.result.grade = 'A'
            self.result.action = 'BUY'
            self.result.confidence = 80
        elif score >= 60:
            self.result.grade = 'B+'
            self.result.action = 'LEAN BULLISH'
            self.result.confidence = 70
        elif score >= 55:
            self.result.grade = 'B'
            self.result.action = 'SLIGHT BULLISH'
            self.result.confidence = 60
        elif score >= 45:
            self.result.grade = 'C'
            self.result.action = 'NEUTRAL'
            self.result.confidence = 50
        elif score >= 40:
            self.result.grade = 'C-'
            self.result.action = 'SLIGHT BEARISH'
            self.result.confidence = 40
        elif score >= 30:
            self.result.grade = 'D'
            self.result.action = 'LEAN BEARISH'
            self.result.confidence = 30
        elif score >= 20:
            self.result.grade = 'D-'
            self.result.action = 'SELL'
            self.result.confidence = 20
        else:
            self.result.grade = 'F'
            self.result.action = 'STRONG SELL'
            self.result.confidence = 10
    
    def _calculate_trade_levels(self):
        """Calculate entry, stop loss, and take profit levels"""
        ind = self.result.indicators
        price = self.result.price
        atr = ind['atr']
        
        # Use ATR for dynamic levels
        if atr == 0:
            atr = price * 0.02  # Fallback to 2%
        
        if self.result.is_bullish:
            # LONG setup
            self.result.entry = price
            self.result.stop_loss = max(ind['support'] * 0.99, price - (atr * 2))
            self.result.tp1 = price + (atr * 1.5)
            self.result.tp2 = price + (atr * 2.5)
            self.result.tp3 = min(ind['resistance'] * 1.01, price + (atr * 4))
        else:
            # SHORT setup or wait
            self.result.entry = price
            self.result.stop_loss = min(ind['resistance'] * 1.01, price + (atr * 2))
            self.result.tp1 = price - (atr * 1.5)
            self.result.tp2 = price - (atr * 2.5)
            self.result.tp3 = max(ind['support'] * 0.99, price - (atr * 4))
        
        # Calculate risk percentage
        self.result.risk_pct = abs(price - self.result.stop_loss) / price * 100
    
    def _generate_narrative(self):
        """Generate human-readable summary and action plan"""
        r = self.result
        ind = r.indicators
        
        # Summary
        bull_count = len([f for f in r.factors if f['type'] == 'bullish'])
        bear_count = len([f for f in r.factors if f['type'] == 'bearish'])
        
        if r.total_score >= 60:
            r.summary = (f"Bullish setup with {bull_count} confirming signals vs {bear_count} bearish. "
                        f"Grade {r.grade} indicates favorable conditions for long positions.")
        elif r.total_score <= 40:
            r.summary = (f"Bearish setup with {bear_count} warning signals vs {bull_count} bullish. "
                        f"Grade {r.grade} indicates unfavorable conditions. Avoid longs or consider shorts.")
        else:
            r.summary = (f"Mixed signals with {bull_count} bullish vs {bear_count} bearish factors. "
                        f"Grade {r.grade} suggests waiting for clearer direction.")
        
        # Action Plan
        if r.action in ['STRONG BUY', 'BUY']:
            r.action_plan = [
                f"✅ Entry: {r.entry:.4f}",
                f"🛑 Stop Loss: {r.stop_loss:.4f} ({r.risk_pct:.1f}% risk)",
                f"🎯 TP1: {r.tp1:.4f}",
                f"🎯 TP2: {r.tp2:.4f}",
                f"🎯 TP3: {r.tp3:.4f}",
            ]
        elif r.action in ['STRONG SELL', 'SELL', 'LEAN BEARISH']:
            r.action_plan = [
                f"🔴 Avoid long positions",
                f"👀 Watch for bounce at support: {ind['support']:.4f}",
                f"⚠️ Resistance at: {ind['resistance']:.4f}",
            ]
        else:
            r.action_plan = [
                f"⏳ No clear entry signal - patience",
                f"👀 Watch support: {ind['support']:.4f}",
                f"👀 Watch resistance: {ind['resistance']:.4f}",
            ]
    
    def _fallback_analysis(self, df: pd.DataFrame, symbol: str, timeframe: str) -> TAResult:
        """Fallback if ta library not available"""
        result = TAResult(symbol=symbol, timeframe=timeframe)
        result.price = df['Close'].iloc[-1]
        result.summary = "⚠️ ta library not installed. Run: pip install ta"
        result.action = "INSTALL TA LIBRARY"
        result.grade = "?"
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_with_ta(df: pd.DataFrame, symbol: str = "", timeframe: str = "") -> TAResult:
    """
    Quick function to analyze a DataFrame with ta library.
    
    Usage:
        result = analyze_with_ta(df, 'BTCUSDT', '4h')
        print(result.grade)  # 'A+'
        print(result.action)  # 'STRONG BUY'
        print(result.factors)  # List of all factors
    """
    analyzer = TAAnalyzer()
    return analyzer.analyze(df, symbol, timeframe)


def get_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ALL ta indicators to a DataFrame.
    Returns the DataFrame with ~50 new columns.
    """
    analyzer = TAAnalyzer()
    analyzer.df = df.copy()
    analyzer._add_all_indicators()
    return analyzer.df


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("TA Analysis Module loaded successfully!")
    print(f"ta library available: {TA_AVAILABLE}")
