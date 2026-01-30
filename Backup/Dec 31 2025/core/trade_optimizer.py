"""
Trade Optimizer Module for InvestorIQ
=====================================

A constraint-based optimizer for trade levels (Entry, SL, TP).
Similar to MIP solver - finds optimal solution within constraints.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                      LEVEL COLLECTOR                            │
├─────────────────────────────────────────────────────────────────┤
│  Current TF    │  HTF          │  Volume       │  Historical    │
│  ├─ OB         │  ├─ OB        │  ├─ POC       │  ├─ Prev Day H │
│  ├─ FVG        │  ├─ FVG       │  ├─ VWAP      │  ├─ Prev Day L │
│  ├─ Swing H/L  │  ├─ Structure │  ├─ VAH/VAL   │  ├─ Prev Day C │
│  └─ BOS/CHoCH  │  └─ Swing H/L │  └─ HVN/LVN   │  └─ Weekly H/L │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CONSTRAINT ENGINE                          │
├─────────────────────────────────────────────────────────────────┤
│  • Direction: LONG → TPs above entry, SL below                  │
│  • HTF Alignment: Don't trade against HTF structure             │
│  • Mode Limits: Max SL % by trading mode                        │
│  • R:R Minimum: At least 1:1 for TP1                           │
│  • Confluence: Prefer levels with multiple touches              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRADE OPTIMIZER                            │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: Collected levels + Constraints + Direction              │
│  OUTPUT: Optimal Entry, SL, TP1, TP2, TP3                      │
│  OR: WAIT (no valid solution within constraints)                │
└─────────────────────────────────────────────────────────────────┘

SL STRATEGY:
- Option 1: Anti-hunt (below structure + ATR buffer + ugly number)
- Option 2: Structure break (below key level that invalidates trade)
- Choose: Whichever is CLOSER but still valid for the mode

NO ARBITRARY PERCENTAGES OR MULTIPLIERS.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class TradingMode(Enum):
    SCALP = "scalp"
    DAY_TRADE = "day_trade"
    SWING = "swing"
    INVESTMENT = "investment"


class LevelType(Enum):
    """Types of price levels we collect"""
    # Current TF - SMC
    OB_BULLISH = "ob_bullish"
    OB_BEARISH = "ob_bearish"
    FVG_BULLISH = "fvg_bullish"
    FVG_BEARISH = "fvg_bearish"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    BOS = "bos"
    CHOCH = "choch"
    
    # HTF
    HTF_OB_BULLISH = "htf_ob_bullish"
    HTF_OB_BEARISH = "htf_ob_bearish"
    HTF_SWING_HIGH = "htf_swing_high"
    HTF_SWING_LOW = "htf_swing_low"
    
    # Volume
    POC = "poc"  # Point of Control
    VWAP = "vwap"
    VAH = "vah"  # Value Area High
    VAL = "val"  # Value Area Low
    
    # Historical
    PREV_DAY_HIGH = "prev_day_high"
    PREV_DAY_LOW = "prev_day_low"
    PREV_DAY_CLOSE = "prev_day_close"
    PREV_WEEK_HIGH = "prev_week_high"
    PREV_WEEK_LOW = "prev_week_low"
    
    # Fibonacci - Institutional retracement levels
    FIB_382 = "fib_382"   # 38.2% - shallow retracement
    FIB_500 = "fib_500"   # 50% - equilibrium  
    FIB_618 = "fib_618"   # 61.8% - golden ratio (most important!)
    FIB_786 = "fib_786"   # 78.6% - deep retracement
    
    # Liquidity - Where stops cluster
    LIQUIDITY_HIGH = "liquidity_high"  # Above swing highs (buy stops)
    LIQUIDITY_LOW = "liquidity_low"    # Below swing lows (sell stops)
    
    # Session Levels - Intraday key levels
    SESSION_HIGH = "session_high"      # Current session high
    SESSION_LOW = "session_low"        # Current session low
    ASIA_HIGH = "asia_high"
    ASIA_LOW = "asia_low"
    
    # Psychological
    ROUND_NUMBER = "round_number"      # $100, $50, $10, etc.


class LevelRole(Enum):
    """How a level can be used"""
    SUPPORT = "support"      # For LONG: Entry, SL reference
    RESISTANCE = "resistance"  # For LONG: TP target
    BOTH = "both"            # Can act as either


@dataclass
class PriceLevel:
    """A single price level with metadata"""
    price: float
    level_type: LevelType
    role: LevelRole
    source_tf: str  # Timeframe it came from
    confidence: float = 1.0  # 0-1, higher = more reliable
    touches: int = 1  # How many times price reacted here
    description: str = ""
    
    def __lt__(self, other):
        return self.price < other.price


@dataclass
class TradeSetup:
    """The optimized trade setup"""
    direction: str  # 'LONG', 'SHORT', or 'WAIT'
    entry: float
    entry_type: str  # 'market' or 'limit'
    entry_level: Optional[PriceLevel] = None
    
    stop_loss: float = 0
    sl_type: str = ""  # 'anti_hunt' or 'structure_break'
    sl_level: Optional[PriceLevel] = None
    
    tp1: float = 0
    tp1_level: Optional[PriceLevel] = None
    tp2: float = 0
    tp2_level: Optional[PriceLevel] = None
    tp3: float = 0
    tp3_level: Optional[PriceLevel] = None
    
    risk_reward: float = 0
    confidence: float = 0
    reasoning: str = ""
    
    # Why WAIT (if applicable)
    wait_reason: str = ""
    
    @property
    def sl(self) -> float:
        """Alias for stop_loss for convenience"""
        return self.stop_loss


@dataclass 
class Constraints:
    """
    Professional trading constraints by mode.
    
    Based on research from professional traders:
    - SL: 1-2x ATR is standard
    - TP: 1.5-3x ATR for TP1, can extend further for TP2/TP3
    - Structure within these ranges is preferred
    - If no structure, use ATR-based defaults
    
    Sources:
    - "Stop-loss levels: Set stops at 1-2 times the ATR value"
    - "Take-profit targets: Aim for 1-3 times the ATR value"
    - "1:2 risk-reward ratio is solid for most traders"
    - "Swing traders: aim for 3x+ risk target"
    """
    # Maximum % stop loss (hard cap for mode)
    max_sl_pct: float
    
    # Minimum R:R ratio for TP1 (trade must be worth it)
    min_rr: float
    
    # Entry timing
    max_wait_candles: int
    prefer_limit: bool
    
    # MINIMUM % stop loss (floor - can't be tighter than this!)
    # This is critical when ATR is small in consolidation
    min_sl_pct: float = 0.5
    
    # ATR-based RANGES for finding structure
    # SL Range: Find structure between min and max ATR
    min_sl_atr: float = 0.5    # Minimum SL distance (avoid being too tight)
    max_sl_atr: float = 2.0    # Maximum SL distance
    
    # TP1 Range: First target
    min_tp1_atr: float = 1.0   # Minimum TP1 (must meet R:R)
    max_tp1_atr: float = 4.0   # Maximum TP1 
    
    # TP2 Range: Second target  
    min_tp2_atr: float = 2.0
    max_tp2_atr: float = 7.0
    
    # TP3 Range: Extended target
    min_tp3_atr: float = 4.0
    max_tp3_atr: float = 10.0
    
    # PERCENTAGE FLOORS - minimum TP distance regardless of ATR
    # These ensure targets make sense for the trading mode
    min_tp1_pct: float = 1.0   # Minimum 1% for TP1
    min_tp2_pct: float = 2.0   # Minimum 2% for TP2
    min_tp3_pct: float = 3.0   # Minimum 3% for TP3


# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL MODE CONFIGURATION
# Based on research from professional traders, prop firms, and ChatGPT synthesis
# 
# KEY INSIGHT: "Professional day traders rarely target >3–5%"
#              "If they do → it's not a day trade anymore"
# ═══════════════════════════════════════════════════════════════════════════════

MODE_CONSTRAINTS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # SCALP: Quick in/out, tight levels, high win rate expected
    # Timeframes: 1m, 5m
    # Hold: Minutes to 1 hour
    # Professional: 0.3-1% SL, 0.5-2% TP
    # ═══════════════════════════════════════════════════════════════════════════
    TradingMode.SCALP: Constraints(
        max_sl_pct=1.0,           # Max 1% SL for scalping (crypto)
        min_sl_pct=0.3,           # Min 0.3% SL (floor in consolidation)
        min_rr=1.5,               # Need decent R:R for quick trades
        max_wait_candles=3,
        prefer_limit=True,
        # SL: 0.5-1 ATR 
        min_sl_atr=0.5,
        max_sl_atr=1.0,
        # TPs: 0.5-2% targets (tight)
        min_tp1_atr=0.8,   max_tp1_atr=2.0,
        min_tp2_atr=1.5,   max_tp2_atr=3.5,
        min_tp3_atr=2.5,   max_tp3_atr=5.0,
        # Percentage floors (minimum regardless of ATR)
        min_tp1_pct=0.5,   # Min 0.5% for TP1
        min_tp2_pct=1.0,   # Min 1% for TP2
        min_tp3_pct=1.5,   # Min 1.5% for TP3
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DAY TRADE: Intraday positions, closed by end of day
    # Timeframes: 15m, 1h
    # Hold: 1 hour to 1 day
    # Professional (Crypto): 0.8-2.5% SL, 1.5-5% TP, R:R 1:2
    # "Professional day traders rarely target >3–5%"
    # ═══════════════════════════════════════════════════════════════════════════
    TradingMode.DAY_TRADE: Constraints(
        max_sl_pct=2.5,           # Max 2.5% SL (ChatGPT: 0.8-2.5%)
        min_sl_pct=0.8,           # Min 0.8% SL (floor - CRITICAL!)
        min_rr=1.5,               # R:R 1:1.5 minimum (ChatGPT: 1:2)
        max_wait_candles=8,
        prefer_limit=True,
        # SL: 1-2 ATR
        min_sl_atr=0.8,
        max_sl_atr=2.0,
        # TPs: 1.5-8% targets (extended for HTF targets)
        min_tp1_atr=1.5,   max_tp1_atr=4.0,    # ~1.5-4%
        min_tp2_atr=2.5,   max_tp2_atr=7.0,    # ~2.5-7% (extended for HTF)
        min_tp3_atr=4.0,   max_tp3_atr=10.0,   # ~4-10% (extended for major HTF targets)
        # Percentage floors (minimum regardless of ATR)
        min_tp1_pct=1.5,   # Min 1.5% for TP1
        min_tp2_pct=2.5,   # Min 2.5% for TP2
        min_tp3_pct=4.0,   # Min 4% for TP3
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SWING: Multi-day positions, catch bigger moves
    # Timeframes: 4h, 1d
    # Hold: Days to weeks
    # Professional (Crypto): 3-8% SL, 6-15% TP
    # ═══════════════════════════════════════════════════════════════════════════
    TradingMode.SWING: Constraints(
        max_sl_pct=8.0,           # Max 8% SL
        min_sl_pct=3.0,           # Min 3% SL (floor - swing needs room)
        min_rr=1.5,               # R:R 1:1.5 minimum for swing
        max_wait_candles=20,
        prefer_limit=True,
        # SL: 1.5-3 ATR 
        min_sl_atr=1.5,
        max_sl_atr=3.0,
        # TPs: Realistic swing targets (days to weeks)
        min_tp1_atr=1.5,   max_tp1_atr=4.0,    # ~3-6%
        min_tp2_atr=2.5,   max_tp2_atr=6.0,    # ~5-10%
        min_tp3_atr=4.0,   max_tp3_atr=8.0,    # ~8-15%
        # Percentage floors - CRITICAL for swing!
        min_tp1_pct=3.0,   # Min 3% for TP1 (must be worth holding days)
        min_tp2_pct=5.0,   # Min 5% for TP2
        min_tp3_pct=8.0,   # Min 8% for TP3
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INVESTMENT: Long-term positions, major structure levels
    # Timeframes: 1d, 1w
    # Hold: Weeks to months
    # Professional (Crypto): 15-35% SL, 50-300% TP
    # ═══════════════════════════════════════════════════════════════════════════
    TradingMode.INVESTMENT: Constraints(
        max_sl_pct=25.0,          # Max 25% SL (ChatGPT: 15-35%)
        min_sl_pct=8.0,           # Min 8% SL (floor - investments need room)
        min_rr=2.0,               # Must be worth the hold time
        max_wait_candles=50,
        prefer_limit=True,
        # SL: 2-4 ATR
        min_sl_atr=2.0,
        max_sl_atr=4.0,
        # TPs: 50-300% targets (long term)
        min_tp1_atr=5.0,   max_tp1_atr=15.0,   # ~20-60%
        min_tp2_atr=10.0,  max_tp2_atr=25.0,   # ~40-100%
        min_tp3_atr=15.0,  max_tp3_atr=40.0,   # ~60-160%
        # Percentage floors - investment needs big targets
        min_tp1_pct=10.0,  # Min 10% for TP1 (worth holding weeks)
        min_tp2_pct=20.0,  # Min 20% for TP2
        min_tp3_pct=30.0,  # Min 30% for TP3
    ),
}

TIMEFRAME_MODE_MAP = {
    '1m': TradingMode.SCALP,
    '5m': TradingMode.SCALP,
    '15m': TradingMode.DAY_TRADE,
    '1h': TradingMode.DAY_TRADE,
    '4h': TradingMode.SWING,
    '1d': TradingMode.SWING,
    '1w': TradingMode.INVESTMENT,
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODE → DEFAULT TIMEFRAME (Professional Standards)
# User selects MODE, system auto-selects optimal TF
# ═══════════════════════════════════════════════════════════════════════════════
MODE_DEFAULT_TIMEFRAME = {
    TradingMode.SCALP: '5m',        # Quick entries, 5m is standard
    TradingMode.DAY_TRADE: '15m',   # Intraday, 15m balances signal quality
    TradingMode.SWING: '4h',        # Multi-day, 4h is professional standard
    TradingMode.INVESTMENT: '1d',   # Long-term, daily charts
}

# HTF Context for each mode (for auto-fetching higher timeframe)
MODE_HTF_CONTEXT = {
    TradingMode.SCALP: '1h',        # Scalp on 5m, HTF context from 1h
    TradingMode.DAY_TRADE: '4h',    # Day trade on 15m, HTF context from 4h
    TradingMode.SWING: '1d',        # Swing on 4h, HTF context from 1d
    TradingMode.INVESTMENT: '1w',   # Investment on 1d, HTF context from 1w
}


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LevelCollector:
    """
    Collects all possible price levels from multiple sources.
    
    Sources:
    - Current TF SMC (OB, FVG, Swings, BOS/CHoCH)
    - HTF SMC (OB, Swings, Structure)
    - Volume Profile (POC, VWAP, VAH/VAL)
    - Historical (Previous day/week H/L/C)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        current_price: float,
        timeframe: str,
        htf_df: Optional[pd.DataFrame] = None,
        htf_timeframe: Optional[str] = None,
    ):
        self.df = df
        self.current_price = current_price
        self.timeframe = timeframe
        self.htf_df = htf_df
        self.htf_timeframe = htf_timeframe
        
        self.levels: List[PriceLevel] = []
        self.atr = self._calculate_atr()
    
    def _calculate_atr(self, period: int = 14) -> float:
        """Calculate ATR for the current timeframe"""
        if self.df is None or len(self.df) < period:
            return self.current_price * 0.02  # Fallback: 2%
        
        high = self.df['High'] if 'High' in self.df.columns else self.df['high']
        low = self.df['Low'] if 'Low' in self.df.columns else self.df['low']
        close = self.df['Close'] if 'Close' in self.df.columns else self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def collect_from_smc(self, smc_data: Dict) -> None:
        """
        Collect levels from SMC analysis.
        
        Expected smc_data keys:
        - bullish_ob_top, bullish_ob_bottom (or bullish_ob flag)
        - bearish_ob_top, bearish_ob_bottom (or bearish_ob flag)
        - fvg_bullish_top, fvg_bullish_bottom (or fvg_bullish flag)
        - fvg_bearish_top, fvg_bearish_bottom (or fvg_bearish flag)
        - swing_high, swing_low
        - bos, choch, bos_level, choch_level
        """
        # Bullish OB (support zone) - check if price exists, not boolean flag
        bullish_ob_top = smc_data.get('bullish_ob_top', 0)
        bullish_ob_bottom = smc_data.get('bullish_ob_bottom', 0)
        if bullish_ob_top > 0 or bullish_ob_bottom > 0:
            if bullish_ob_top > 0:
                self.levels.append(PriceLevel(
                    price=bullish_ob_top,
                    level_type=LevelType.OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    confidence=0.9,
                    description=f"Bullish OB ({self.timeframe})"
                ))
            if bullish_ob_bottom > 0:
                self.levels.append(PriceLevel(
                    price=bullish_ob_bottom,
                    level_type=LevelType.OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    confidence=0.85,
                    description=f"Bullish OB bottom ({self.timeframe})"
                ))
        
        # Bearish OB (resistance zone)
        bearish_ob_top = smc_data.get('bearish_ob_top', 0)
        bearish_ob_bottom = smc_data.get('bearish_ob_bottom', 0)
        if bearish_ob_top > 0 or bearish_ob_bottom > 0:
            if bearish_ob_bottom > 0:
                self.levels.append(PriceLevel(
                    price=bearish_ob_bottom,
                    level_type=LevelType.OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    confidence=0.9,
                    description=f"Bearish OB ({self.timeframe})"
                ))
            if bearish_ob_top > 0:
                self.levels.append(PriceLevel(
                    price=bearish_ob_top,
                    level_type=LevelType.OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    confidence=0.85,
                    description=f"Bearish OB top ({self.timeframe})"
                ))
        
        # FVG Bullish (support - gap to fill)
        fvg_bullish_top = smc_data.get('fvg_bullish_top', 0)
        fvg_bullish_bottom = smc_data.get('fvg_bullish_bottom', 0)
        if fvg_bullish_top > 0:
            self.levels.append(PriceLevel(
                price=fvg_bullish_top,
                level_type=LevelType.FVG_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf=self.timeframe,
                confidence=0.8,
                description=f"Bullish FVG ({self.timeframe})"
            ))
        if fvg_bullish_bottom > 0:
            self.levels.append(PriceLevel(
                price=fvg_bullish_bottom,
                level_type=LevelType.FVG_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf=self.timeframe,
                confidence=0.75,
                description=f"Bullish FVG bottom ({self.timeframe})"
            ))
        
        # FVG Bearish (resistance - gap to fill)
        fvg_bearish_top = smc_data.get('fvg_bearish_top', 0)
        fvg_bearish_bottom = smc_data.get('fvg_bearish_bottom', 0)
        if fvg_bearish_bottom > 0:
            self.levels.append(PriceLevel(
                price=fvg_bearish_bottom,
                level_type=LevelType.FVG_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf=self.timeframe,
                confidence=0.8,
                description=f"Bearish FVG ({self.timeframe})"
            ))
        if fvg_bearish_top > 0:
            self.levels.append(PriceLevel(
                price=fvg_bearish_top,
                level_type=LevelType.FVG_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf=self.timeframe,
                confidence=0.75,
                description=f"Bearish FVG top ({self.timeframe})"
            ))
        
        # Swing High (resistance)
        if smc_data.get('swing_high', 0) > 0:
            self.levels.append(PriceLevel(
                price=smc_data['swing_high'],
                level_type=LevelType.SWING_HIGH,
                role=LevelRole.RESISTANCE,
                source_tf=self.timeframe,
                confidence=0.7,
                description=f"Swing High ({self.timeframe})"
            ))
        
        # Swing Low (support)
        if smc_data.get('swing_low', 0) > 0:
            self.levels.append(PriceLevel(
                price=smc_data['swing_low'],
                level_type=LevelType.SWING_LOW,
                role=LevelRole.SUPPORT,
                source_tf=self.timeframe,
                confidence=0.7,
                description=f"Swing Low ({self.timeframe})"
            ))
    
    def collect_from_htf(self, htf_data: Dict) -> None:
        """
        Collect levels from Higher Timeframe analysis.
        HTF levels have higher confidence.
        
        For Bearish OB (supply zone):
        - Bottom: First resistance (price enters here)
        - Mid: Key rejection level
        - Top: Major resistance (full rejection)
        
        For Bullish OB (demand zone):
        - Top: First support (price enters here)  
        - Mid: Key support level
        - Bottom: Major support (full support)
        """
        if not htf_data:
            return
        
        htf_tf = self.htf_timeframe or "HTF"
        
        # ═══════════════════════════════════════════════════════════════
        # HTF BULLISH OB (Demand Zone) - For LONG SL reference, SHORT TPs
        # ═══════════════════════════════════════════════════════════════
        bullish_top = htf_data.get('bullish_ob_top', 0)
        bullish_bottom = htf_data.get('bullish_ob_bottom', 0)
        
        if bullish_top > 0:
            self.levels.append(PriceLevel(
                price=bullish_top,
                level_type=LevelType.HTF_OB_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf=htf_tf,
                confidence=0.95,
                description=f"HTF Bullish OB top ({htf_tf})"
            ))
        
        if bullish_bottom > 0:
            self.levels.append(PriceLevel(
                price=bullish_bottom,
                level_type=LevelType.HTF_OB_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf=htf_tf,
                confidence=0.90,
                description=f"HTF Bullish OB bottom ({htf_tf})"
            ))
        
        # Mid-OB level (key institutional level)
        if bullish_top > 0 and bullish_bottom > 0:
            mid_bullish = (bullish_top + bullish_bottom) / 2
            self.levels.append(PriceLevel(
                price=mid_bullish,
                level_type=LevelType.HTF_OB_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf=htf_tf,
                confidence=0.85,
                description=f"HTF Bullish OB mid ({htf_tf})"
            ))
        
        # ═══════════════════════════════════════════════════════════════
        # HTF BEARISH OB (Supply Zone) - For LONG TPs, SHORT SL reference
        # ═══════════════════════════════════════════════════════════════
        bearish_top = htf_data.get('bearish_ob_top', 0)
        bearish_bottom = htf_data.get('bearish_ob_bottom', 0)
        
        if bearish_bottom > 0:
            self.levels.append(PriceLevel(
                price=bearish_bottom,
                level_type=LevelType.HTF_OB_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf=htf_tf,
                confidence=0.90,
                description=f"HTF Bearish OB bottom ({htf_tf})"
            ))
        
        if bearish_top > 0:
            self.levels.append(PriceLevel(
                price=bearish_top,
                level_type=LevelType.HTF_OB_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf=htf_tf,
                confidence=0.95,  # Top is strongest resistance
                description=f"HTF Bearish OB top ({htf_tf})"
            ))
        
        # Mid-OB level (key institutional level) 
        if bearish_top > 0 and bearish_bottom > 0:
            mid_bearish = (bearish_top + bearish_bottom) / 2
            self.levels.append(PriceLevel(
                price=mid_bearish,
                level_type=LevelType.HTF_OB_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf=htf_tf,
                confidence=0.88,
                description=f"HTF Bearish OB mid ({htf_tf})"
            ))
        
        # ═══════════════════════════════════════════════════════════════
        # HTF SWING LEVELS
        # ═══════════════════════════════════════════════════════════════
        if htf_data.get('htf_swing_high', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['htf_swing_high'],
                level_type=LevelType.HTF_SWING_HIGH,
                role=LevelRole.RESISTANCE,
                source_tf=htf_tf,
                confidence=0.85,
                description=f"HTF Swing High ({htf_tf})"
            ))
        
        if htf_data.get('htf_swing_low', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['htf_swing_low'],
                level_type=LevelType.HTF_SWING_LOW,
                role=LevelRole.SUPPORT,
                source_tf=htf_tf,
                confidence=0.85,
                description=f"HTF Swing Low ({htf_tf})"
            ))
    
    def collect_from_volume(self, volume_data: Dict) -> None:
        """
        Collect levels from Volume Profile.
        
        Expected keys:
        - poc: Point of Control
        - vwap: VWAP
        - vah: Value Area High
        - val: Value Area Low
        """
        if not volume_data:
            return
        
        if volume_data.get('poc', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['poc'],
                level_type=LevelType.POC,
                role=LevelRole.BOTH,  # POC acts as magnet
                source_tf=self.timeframe,
                confidence=0.9,
                description="POC - Point of Control (fair value)"
            ))
        
        if volume_data.get('vwap', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['vwap'],
                level_type=LevelType.VWAP,
                role=LevelRole.BOTH,
                source_tf=self.timeframe,
                confidence=0.85,
                description="VWAP (institutional reference)"
            ))
        
        if volume_data.get('vah', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['vah'],
                level_type=LevelType.VAH,
                role=LevelRole.RESISTANCE,
                source_tf=self.timeframe,
                confidence=0.8,
                description="VAH - Value Area High"
            ))
        
        if volume_data.get('val', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['val'],
                level_type=LevelType.VAL,
                role=LevelRole.SUPPORT,
                source_tf=self.timeframe,
                confidence=0.8,
                description="VAL - Value Area Low"
            ))
    
    def collect_from_historical(self) -> None:
        """
        Collect levels from historical price data.
        Previous day/week highs and lows.
        """
        if self.df is None or len(self.df) < 2:
            return
        
        high_col = 'High' if 'High' in self.df.columns else 'high'
        low_col = 'Low' if 'Low' in self.df.columns else 'low'
        close_col = 'Close' if 'Close' in self.df.columns else 'close'
        
        # For intraday timeframes, calculate previous day levels
        if self.timeframe in ['1m', '5m', '15m', '1h']:
            # Get unique dates
            if hasattr(self.df.index, 'date'):
                dates = self.df.index.date
                unique_dates = sorted(set(dates))
                
                if len(unique_dates) >= 2:
                    prev_day = unique_dates[-2]
                    prev_day_data = self.df[self.df.index.date == prev_day]
                    
                    if len(prev_day_data) > 0:
                        prev_high = prev_day_data[high_col].max()
                        prev_low = prev_day_data[low_col].min()
                        prev_close = prev_day_data[close_col].iloc[-1]
                        
                        self.levels.append(PriceLevel(
                            price=prev_high,
                            level_type=LevelType.PREV_DAY_HIGH,
                            role=LevelRole.RESISTANCE,
                            source_tf="D",
                            confidence=0.75,
                            description="Previous Day High"
                        ))
                        
                        self.levels.append(PriceLevel(
                            price=prev_low,
                            level_type=LevelType.PREV_DAY_LOW,
                            role=LevelRole.SUPPORT,
                            source_tf="D",
                            confidence=0.75,
                            description="Previous Day Low"
                        ))
                        
                        self.levels.append(PriceLevel(
                            price=prev_close,
                            level_type=LevelType.PREV_DAY_CLOSE,
                            role=LevelRole.BOTH,
                            source_tf="D",
                            confidence=0.7,
                            description="Previous Day Close"
                        ))
    
    def collect_fibonacci(self) -> None:
        """
        Calculate Fibonacci retracement levels from recent swing high/low.
        
        Uses the most recent significant swing to calculate:
        - 38.2%, 50%, 61.8%, 78.6% retracements
        
        These are institutional levels where price often reverses.
        """
        if self.df is None or len(self.df) < 20:
            return
        
        try:
            high_col = 'High' if 'High' in self.df.columns else 'high'
            low_col = 'Low' if 'Low' in self.df.columns else 'low'
            
            # Find recent swing high and low (last 50 candles)
            lookback = min(50, len(self.df))
            recent_df = self.df.tail(lookback)
            
            swing_high = recent_df[high_col].max()
            swing_low = recent_df[low_col].min()
            
            if swing_high <= swing_low:
                return
            
            range_size = swing_high - swing_low
            
            # Determine trend direction for fib calculation
            # If price is closer to high = downtrend (retrace from high)
            # If price is closer to low = uptrend (retrace from low)
            mid_point = (swing_high + swing_low) / 2
            
            if self.current_price > mid_point:
                # Price near highs - calculate retracements DOWN from swing high
                fib_levels = {
                    LevelType.FIB_382: swing_high - (range_size * 0.382),
                    LevelType.FIB_500: swing_high - (range_size * 0.500),
                    LevelType.FIB_618: swing_high - (range_size * 0.618),
                    LevelType.FIB_786: swing_high - (range_size * 0.786),
                }
                fib_role = LevelRole.SUPPORT  # These act as support in uptrend
            else:
                # Price near lows - calculate retracements UP from swing low
                fib_levels = {
                    LevelType.FIB_382: swing_low + (range_size * 0.382),
                    LevelType.FIB_500: swing_low + (range_size * 0.500),
                    LevelType.FIB_618: swing_low + (range_size * 0.618),
                    LevelType.FIB_786: swing_low + (range_size * 0.786),
                }
                fib_role = LevelRole.RESISTANCE  # These act as resistance in downtrend
            
            fib_names = {
                LevelType.FIB_382: "Fib 38.2%",
                LevelType.FIB_500: "Fib 50%",
                LevelType.FIB_618: "Fib 61.8% (Golden)",
                LevelType.FIB_786: "Fib 78.6%",
            }
            
            fib_confidence = {
                LevelType.FIB_382: 0.65,
                LevelType.FIB_500: 0.70,
                LevelType.FIB_618: 0.85,  # Golden ratio - highest confidence
                LevelType.FIB_786: 0.60,
            }
            
            for level_type, price in fib_levels.items():
                if price > 0:
                    self.levels.append(PriceLevel(
                        price=price,
                        level_type=level_type,
                        role=fib_role,
                        source_tf=self.timeframe,
                        confidence=fib_confidence[level_type],
                        description=fib_names[level_type]
                    ))
        except Exception:
            pass
    
    def collect_liquidity_zones(self) -> None:
        """
        Identify liquidity zones where stop losses cluster.
        
        - Above swing highs = buy stops (liquidity for shorts)
        - Below swing lows = sell stops (liquidity for longs)
        
        Smart money hunts these zones before reversing.
        """
        if self.df is None or len(self.df) < 20:
            return
        
        try:
            high_col = 'High' if 'High' in self.df.columns else 'high'
            low_col = 'Low' if 'Low' in self.df.columns else 'low'
            
            # ATR for offset calculation
            atr = self.atr if self.atr > 0 else self.current_price * 0.01
            
            # Find significant swing highs (local maxima)
            lookback = min(100, len(self.df))
            recent_df = self.df.tail(lookback)
            
            highs = recent_df[high_col].values
            lows = recent_df[low_col].values
            
            # Find swing highs (higher than 3 candles on each side)
            for i in range(3, len(highs) - 3):
                if highs[i] > max(highs[i-3:i]) and highs[i] > max(highs[i+1:i+4]):
                    # Liquidity sits just above swing high
                    liquidity_price = highs[i] + (atr * 0.3)  # Small offset above
                    if liquidity_price > self.current_price:
                        self.levels.append(PriceLevel(
                            price=liquidity_price,
                            level_type=LevelType.LIQUIDITY_HIGH,
                            role=LevelRole.RESISTANCE,
                            source_tf=self.timeframe,
                            confidence=0.70,
                            description=f"Buy Stops above {highs[i]:.4f}"
                        ))
            
            # Find swing lows (lower than 3 candles on each side)
            for i in range(3, len(lows) - 3):
                if lows[i] < min(lows[i-3:i]) and lows[i] < min(lows[i+1:i+4]):
                    # Liquidity sits just below swing low
                    liquidity_price = lows[i] - (atr * 0.3)  # Small offset below
                    if liquidity_price < self.current_price:
                        self.levels.append(PriceLevel(
                            price=liquidity_price,
                            level_type=LevelType.LIQUIDITY_LOW,
                            role=LevelRole.SUPPORT,
                            source_tf=self.timeframe,
                            confidence=0.70,
                            description=f"Sell Stops below {lows[i]:.4f}"
                        ))
        except Exception:
            pass
    
    def collect_round_numbers(self) -> None:
        """
        Identify nearby psychological round number levels.
        
        Round numbers act as magnets and often cause reactions.
        Works for any price scale (BTC at $100k, altcoins at $1, etc.)
        """
        try:
            price = self.current_price
            if price <= 0:
                return
            
            # Determine appropriate round number interval based on price
            if price >= 10000:
                intervals = [1000, 500, 100]  # BTC style
            elif price >= 1000:
                intervals = [100, 50, 25]
            elif price >= 100:
                intervals = [10, 5, 1]
            elif price >= 10:
                intervals = [1, 0.5, 0.25]
            elif price >= 1:
                intervals = [0.1, 0.05, 0.01]
            else:
                intervals = [0.01, 0.005, 0.001]
            
            # Only use the largest interval that gives nearby levels
            interval = intervals[0]
            
            # Find round numbers within 5% of current price
            range_pct = 0.05
            lower_bound = price * (1 - range_pct)
            upper_bound = price * (1 + range_pct)
            
            # Calculate nearest round numbers
            base = (price // interval) * interval
            
            round_levels = []
            for mult in range(-3, 4):
                level = base + (mult * interval)
                if lower_bound < level < upper_bound and level != price:
                    round_levels.append(level)
            
            for level in round_levels[:4]:  # Max 4 round numbers
                role = LevelRole.RESISTANCE if level > price else LevelRole.SUPPORT
                self.levels.append(PriceLevel(
                    price=level,
                    level_type=LevelType.ROUND_NUMBER,
                    role=role,
                    source_tf="PSYCH",
                    confidence=0.60,
                    description=f"Round Number ${level:,.2f}"
                ))
        except Exception:
            pass
    
    def get_support_levels(self) -> List[PriceLevel]:
        """Get all levels that can act as support (below current price)"""
        return sorted([
            l for l in self.levels 
            if l.role in [LevelRole.SUPPORT, LevelRole.BOTH] 
            and l.price < self.current_price
        ], reverse=True)  # Nearest first (highest below price)
    
    def get_resistance_levels(self) -> List[PriceLevel]:
        """Get all levels that can act as resistance (above current price)"""
        return sorted([
            l for l in self.levels 
            if l.role in [LevelRole.RESISTANCE, LevelRole.BOTH] 
            and l.price > self.current_price
        ])  # Nearest first (lowest above price)
    
    def get_current_tf_support(self) -> List[PriceLevel]:
        """
        Get ONLY current timeframe support levels (no HTF, no historical).
        Use this for TP calculation to stay within timeframe-appropriate range.
        """
        htf_types = {LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH, 
                     LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                     LevelType.PREV_DAY_HIGH, LevelType.PREV_DAY_LOW, 
                     LevelType.PREV_DAY_CLOSE, LevelType.PREV_WEEK_HIGH, 
                     LevelType.PREV_WEEK_LOW}
        return sorted([
            l for l in self.levels 
            if l.role in [LevelRole.SUPPORT, LevelRole.BOTH] 
            and l.price < self.current_price
            and l.level_type not in htf_types
        ], reverse=True)
    
    def get_current_tf_resistance(self) -> List[PriceLevel]:
        """
        Get ONLY current timeframe resistance levels (no HTF, no historical).
        Use this for TP calculation to stay within timeframe-appropriate range.
        """
        htf_types = {LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH, 
                     LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                     LevelType.PREV_DAY_HIGH, LevelType.PREV_DAY_LOW, 
                     LevelType.PREV_DAY_CLOSE, LevelType.PREV_WEEK_HIGH, 
                     LevelType.PREV_WEEK_LOW}
        return sorted([
            l for l in self.levels 
            if l.role in [LevelRole.RESISTANCE, LevelRole.BOTH] 
            and l.price > self.current_price
            and l.level_type not in htf_types
        ])
    
    def get_levels_by_strategy(self, strategy: str, role: str = 'both') -> List[PriceLevel]:
        """
        Get levels prioritized by winning strategy.
        
        Args:
            strategy: 'OB', 'FVG', 'VWAP', 'FIB', 'LIQ', etc.
            role: 'support', 'resistance', or 'both'
        """
        # Map strategy to level types (from backtester results)
        strategy_types = {
            # SMC strategies
            'OB': {LevelType.OB_BULLISH, LevelType.OB_BEARISH},
            'FVG': {LevelType.FVG_BULLISH, LevelType.FVG_BEARISH},
            'HL': {LevelType.SWING_HIGH, LevelType.SWING_LOW},  # Higher Low uses swings
            'LH': {LevelType.SWING_HIGH, LevelType.SWING_LOW},  # Lower High uses swings
            'BOS': {LevelType.SWING_HIGH, LevelType.SWING_LOW},  # Break of Structure
            # Volume strategies
            'VWAP': {LevelType.VWAP, LevelType.POC},
            'POC': {LevelType.POC, LevelType.VWAP},
            # Fibonacci
            'FIB': {LevelType.FIB_382, LevelType.FIB_500, LevelType.FIB_618, LevelType.FIB_786},
            'FIBONACCI': {LevelType.FIB_382, LevelType.FIB_500, LevelType.FIB_618, LevelType.FIB_786},
            # Liquidity
            'LIQ': {LevelType.LIQUIDITY_HIGH, LevelType.LIQUIDITY_LOW},
            'LIQUIDITY': {LevelType.LIQUIDITY_HIGH, LevelType.LIQUIDITY_LOW},
            # General
            'SWING': {LevelType.SWING_HIGH, LevelType.SWING_LOW},
            'S/R': {LevelType.SWING_HIGH, LevelType.SWING_LOW, LevelType.POC, LevelType.ROUND_NUMBER},
            'ROUND': {LevelType.ROUND_NUMBER},
            'EMA': set(),  # EMA doesn't map to price levels
        }
        
        target_types = strategy_types.get(strategy.upper(), set())
        
        # Filter by role
        if role == 'support':
            roles = [LevelRole.SUPPORT, LevelRole.BOTH]
            price_filter = lambda p: p < self.current_price
        elif role == 'resistance':
            roles = [LevelRole.RESISTANCE, LevelRole.BOTH]
            price_filter = lambda p: p > self.current_price
        else:
            roles = [LevelRole.SUPPORT, LevelRole.RESISTANCE, LevelRole.BOTH]
            price_filter = lambda p: True
        
        # Get matching levels
        matching = [l for l in self.levels 
                    if l.level_type in target_types and l.role in roles and price_filter(l.price)]
        
        # Sort by distance from current price
        return sorted(matching, key=lambda l: abs(l.price - self.current_price))
    
    def get_all_levels(self) -> List[PriceLevel]:
        """Get all collected levels, sorted by price"""
        return sorted(self.levels)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class TradeOptimizer:
    """
    Constraint-based optimizer for trade levels.
    
    Given collected levels and constraints, finds optimal:
    - Entry (market or limit at support/resistance)
    - Stop Loss (anti-hunt or structure break)
    - Take Profits (TP1, TP2, TP3 at resistance/support levels)
    
    Returns WAIT if no valid solution exists.
    """
    
    def __init__(
        self,
        collector: LevelCollector,
        mode: TradingMode,
        htf_bias: Optional[str] = None,  # 'BULLISH', 'BEARISH', 'NEUTRAL'
        winning_strategy: Optional[str] = None,  # 'OB', 'FVG', 'VWAP', etc.
    ):
        self.collector = collector
        self.mode = mode
        self.constraints = MODE_CONSTRAINTS[mode]
        self.htf_bias = htf_bias or 'NEUTRAL'
        self.winning_strategy = winning_strategy  # From backtest results
        
        self.current_price = collector.current_price
        self.atr = collector.atr
    
    def optimize(self, direction: str) -> TradeSetup:
        """
        Find optimal trade setup for given direction.
        
        Args:
            direction: 'LONG' or 'SHORT'
            
        Returns:
            TradeSetup with optimal levels, or direction='WAIT' if no valid solution
        """
        # ═══════════════════════════════════════════════════════════════════
        # CONSTRAINT 1: HTF Alignment (check but calculate levels anyway)
        # ═══════════════════════════════════════════════════════════════════
        htf_conflict = not self._check_htf_alignment(direction)
        htf_wait_reason = f"HTF is {self.htf_bias} - don't trade {direction} against higher timeframe" if htf_conflict else ""
        
        # ═══════════════════════════════════════════════════════════════════
        # GET LEVELS FOR ENTRY/SL (can use all levels including HTF)
        # ═══════════════════════════════════════════════════════════════════
        if direction == 'LONG':
            entry_levels = self.collector.get_support_levels()  # Buy at support
            sl_side = 'below'
        else:  # SHORT
            entry_levels = self.collector.get_resistance_levels()  # Sell at resistance
            sl_side = 'above'
        
        # Prioritize winning strategy levels for ENTRY (not TPs!)
        # e.g., VWAP strategy = enter at VWAP, FVG strategy = enter at FVG
        if self.winning_strategy:
            strategy_entry_levels = self.collector.get_levels_by_strategy(
                self.winning_strategy, 
                role='support' if direction == 'LONG' else 'resistance'
            )
            if strategy_entry_levels:
                # Put strategy levels first for entry consideration
                strategy_prices = {l.price for l in strategy_entry_levels}
                other_entry = [l for l in entry_levels if l.price not in strategy_prices]
                entry_levels = strategy_entry_levels + other_entry
        
        # ═══════════════════════════════════════════════════════════════════
        # GET LEVELS FOR TPs (all current TF structure - NOT strategy-specific)
        # TPs should target the BEST structure regardless of entry method
        # e.g., Enter at VWAP → TP at Swing High, Enter at FVG → TP at OB
        # ═══════════════════════════════════════════════════════════════════
        if direction == 'LONG':
            # TPs at resistance above entry
            target_levels = self.collector.get_current_tf_resistance()
        else:
            # TPs at support below entry
            target_levels = self.collector.get_current_tf_support()
        
        # ═══════════════════════════════════════════════════════════════════
        # FIND ENTRY
        # ═══════════════════════════════════════════════════════════════════
        entry, entry_type, entry_level = self._find_entry(direction, entry_levels)
        
        # ═══════════════════════════════════════════════════════════════════
        # FIND STOP LOSS
        # ═══════════════════════════════════════════════════════════════════
        sl, sl_type, sl_level = self._find_stop_loss(direction, entry, entry_levels)
        
        # ═══════════════════════════════════════════════════════════════════
        # ENFORCE SL CONSTRAINTS - BOTH MIN AND MAX!
        # - Too tight SL = unrealistic R:R, will get stopped out on noise
        # - Too wide SL = excessive risk for the trading mode
        # ═══════════════════════════════════════════════════════════════════
        atr = self.collector.atr if self.collector else entry * 0.01
        sl_pct = abs(entry - sl) / entry * 100
        sl_wait_reason = ""
        
        # ═══════════════════════════════════════════════════════════════════
        # MINIMUM SL ENFORCEMENT - Use HIGHER of ATR-based or % floor
        # This prevents tiny SL in consolidation (when ATR is small)
        # ═══════════════════════════════════════════════════════════════════
        
        # ATR-based minimum
        atr_min_distance = atr * self.constraints.min_sl_atr
        atr_min_pct = (atr_min_distance / entry) * 100 if entry > 0 else 1
        
        # Percentage floor (absolute minimum regardless of ATR)
        pct_floor = self.constraints.min_sl_pct
        
        # Use the HIGHER of the two - critical for consolidation!
        effective_min_sl_pct = max(atr_min_pct, pct_floor)
        
        if sl_pct < effective_min_sl_pct:
            # SL is too tight - expand it to minimum
            min_sl_distance = entry * (effective_min_sl_pct / 100)
            if direction == 'LONG':
                sl = entry - min_sl_distance
            else:
                sl = entry + min_sl_distance
            sl_level = None  # No longer structure-based
            sl_type = f"min_{effective_min_sl_pct:.1f}%"
            sl_pct = effective_min_sl_pct
        
        # MAXIMUM SL ENFORCEMENT - SL can't be too wide!
        if sl_pct > self.constraints.max_sl_pct:
            # CAP the SL to max allowed
            max_sl_distance = entry * (self.constraints.max_sl_pct / 100)
            if direction == 'LONG':
                sl = entry - max_sl_distance
            else:
                sl = entry + max_sl_distance
            sl_level = None  # No longer structure-based
            sl_type = f"max_{self.constraints.max_sl_pct}%"
            sl_pct = self.constraints.max_sl_pct
        
        # ═══════════════════════════════════════════════════════════════════
        # SMART TP CALCULATION - Pure Structure Based with Hierarchy
        # ═══════════════════════════════════════════════════════════════════
        # Hierarchy (how professionals do it):
        # 1. Current TF structure (OB, FVG, Swing) - if not clustered
        # 2. HTF structure - when current TF is in consolidation  
        # 3. Volume profile (POC, VAH, VAL) - institutional levels
        # 4. Fibonacci levels - institutional retracement/extension
        # 5. ATR-based fallback - ONLY if NO structure found anywhere
        
        atr = self.collector.atr if self.collector else 0
        min_spacing = atr * 1.0 if atr > 0 else entry * 0.003  # Minimum 1 ATR between levels
        
        def get_all_tp_candidates(direction: str) -> List[PriceLevel]:
            """
            Collect ALL available structure levels in priority order.
            Deduplicate clustered levels (within 0.1% of each other).
            
            PROFESSIONAL HIERARCHY (from SMC):
            - Entry/SL → Current TF (tight, precise)
            - TPs → HTF structure FIRST (major institutional targets)
            
            "Take-profit levels should be based on liquidity targets"
            "Use higher timeframe for WHERE to take profit"
            """
            all_levels = []
            
            # ═══════════════════════════════════════════════════════════════
            # 1. HTF STRUCTURE FIRST - Major institutional targets
            # Professionals target HTF OBs, swing levels, liquidity pools
            # CRITICAL: LONG targets RESISTANCE (Bearish OB, Swing High)
            #           SHORT targets SUPPORT (Bullish OB, Swing Low)
            # ═══════════════════════════════════════════════════════════════
            if direction == 'LONG':
                # LONG TPs target RESISTANCE levels above price
                htf_resistance_types = {LevelType.HTF_OB_BEARISH, LevelType.HTF_SWING_HIGH}
                htf = [l for l in self.collector.levels 
                       if l.level_type in htf_resistance_types and l.price > self.collector.current_price]
            else:
                # SHORT TPs target SUPPORT levels below price
                htf_support_types = {LevelType.HTF_OB_BULLISH, LevelType.HTF_SWING_LOW}
                htf = [l for l in self.collector.levels 
                       if l.level_type in htf_support_types and l.price < self.collector.current_price]
            all_levels.extend(htf)
            
            # ═══════════════════════════════════════════════════════════════
            # 2. LIQUIDITY LEVELS - Previous highs/lows where stops rest
            # "Target the next liquidity area" - institutional magnets
            # ═══════════════════════════════════════════════════════════════
            liquidity_types = {LevelType.LIQUIDITY_HIGH, LevelType.LIQUIDITY_LOW,
                              LevelType.PREV_DAY_HIGH, LevelType.PREV_DAY_LOW,
                              LevelType.PREV_WEEK_HIGH, LevelType.PREV_WEEK_LOW}
            if direction == 'LONG':
                liquidity = [l for l in self.collector.levels 
                            if l.level_type in liquidity_types and l.price > self.collector.current_price]
            else:
                liquidity = [l for l in self.collector.levels 
                            if l.level_type in liquidity_types and l.price < self.collector.current_price]
            all_levels.extend(liquidity)
            
            # ═══════════════════════════════════════════════════════════════
            # 3. CURRENT TF STRUCTURE - Only if no HTF structure available
            # Used as fallback when HTF doesn't have clear levels
            # ═══════════════════════════════════════════════════════════════
            if direction == 'LONG':
                current_tf = self.collector.get_current_tf_resistance()
            else:
                current_tf = self.collector.get_current_tf_support()
            all_levels.extend(current_tf)
            
            # 4. Volume profile (POC, VAH, VAL, VWAP)
            volume_types = {LevelType.POC, LevelType.VAH, LevelType.VAL, LevelType.VWAP}
            if direction == 'LONG':
                volume = [l for l in self.collector.levels 
                         if l.level_type in volume_types and l.price > self.collector.current_price]
            else:
                volume = [l for l in self.collector.levels 
                         if l.level_type in volume_types and l.price < self.collector.current_price]
            all_levels.extend(volume)
            
            # 5. Fibonacci and Round Numbers (LAST RESORT ONLY!)
            # These should ONLY be used if NO structure exists
            fib_types = {LevelType.FIB_382, LevelType.FIB_500, LevelType.FIB_618, 
                        LevelType.FIB_786, LevelType.ROUND_NUMBER}
            if direction == 'LONG':
                fib = [l for l in self.collector.levels 
                      if l.level_type in fib_types and l.price > self.collector.current_price]
            else:
                fib = [l for l in self.collector.levels 
                      if l.level_type in fib_types and l.price < self.collector.current_price]
            all_levels.extend(fib)
            
            # ═══════════════════════════════════════════════════════════════
            # DEDUPLICATE - But PREFER STRUCTURE over non-structure!
            # When two levels are close, keep the ORDER BLOCK / SWING, not round number
            # ═══════════════════════════════════════════════════════════════
            if not all_levels or entry <= 0:
                return []
            
            # Define structure types (REAL levels from chart)
            current_tf_structure = {
                LevelType.OB_BULLISH, LevelType.OB_BEARISH,
                LevelType.FVG_BULLISH, LevelType.FVG_BEARISH,
                LevelType.SWING_HIGH, LevelType.SWING_LOW,
            }
            htf_structure = {
                LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
            }
            historical_structure = {
                LevelType.PREV_DAY_HIGH, LevelType.PREV_DAY_LOW,
            }
            
            # Priority score: current TF = 12, HTF = 10, historical = 8, volume = 5, other = 1
            # Current TF is HIGHEST because user can validate on their chart!
            def get_priority(level):
                if level.level_type in current_tf_structure:
                    return 12  # Current TF - validatable on chart!
                elif level.level_type in htf_structure:
                    return 10  # HTF - major targets
                elif level.level_type in historical_structure:
                    return 8   # Historical levels
                elif level.level_type in volume_types:
                    return 5
                else:
                    return 1  # Round numbers, liquidity estimates
            
            # Sort by distance first
            if direction == 'LONG':
                all_levels.sort(key=lambda l: l.price)
            else:
                all_levels.sort(key=lambda l: l.price, reverse=True)
            
            # Remove duplicates - but PREFER structure!
            unique = []
            for level in all_levels:
                is_duplicate = False
                replace_idx = -1
                for idx, existing in enumerate(unique):
                    distance_pct = abs(level.price - existing.price) / entry * 100
                    if distance_pct < 0.5:  # Within 0.5% = cluster (wider to catch more)
                        is_duplicate = True
                        # If new level has higher priority, replace existing
                        if get_priority(level) > get_priority(existing):
                            replace_idx = idx
                        break
                
                if replace_idx >= 0:
                    unique[replace_idx] = level  # Replace with better level
                elif not is_duplicate:
                    unique.append(level)
            
            return unique
        

        # Get all TP candidates
        tp_candidates = get_all_tp_candidates(direction)

        
        
        # ═══════════════════════════════════════════════════════════════════
        # MODE-BASED MINIMUM R:R FOR TP1
        # ═══════════════════════════════════════════════════════════════════
        # TP1 must meet minimum R:R for the trading mode
        # Skip structure levels that don't provide adequate reward
        mode_min_rr_tp1 = {
            TradingMode.SCALP: 0.5,       # Quick trades, can accept lower R:R
            TradingMode.DAY_TRADE: 1.0,   # Standard - don't risk more than you make
            TradingMode.SWING: 1.5,       # Holding longer, need better reward
            TradingMode.INVESTMENT: 2.0,  # Capital locked up, need significant reward
        }
        min_rr_tp1 = mode_min_rr_tp1.get(self.mode, 1.0)
        
        # Calculate risk for R:R calculation
        risk = abs(entry - sl) if sl > 0 else 0
        
        # ═══════════════════════════════════════════════════════════════════
        # PROFESSIONAL RANGE-BASED TP SELECTION
        # For each TP, define ATR-based range and find nearest structure within it
        # If no structure in range, use ATR-based default
        # ═══════════════════════════════════════════════════════════════════
        
        tp1, tp1_level = 0, None
        tp2, tp2_level = 0, None  
        tp3, tp3_level = 0, None
        
        # Define STRUCTURE types that can override min distance
        structure_types = {
            LevelType.OB_BULLISH, LevelType.OB_BEARISH,
            LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
            LevelType.FVG_BULLISH, LevelType.FVG_BEARISH,
            LevelType.SWING_HIGH, LevelType.SWING_LOW,
            LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
        }
        
        if atr > 0:
            # Calculate ATR-based ranges for each TP
            # TP1 Range
            min_tp1_dist = atr * self.constraints.min_tp1_atr
            max_tp1_dist = atr * self.constraints.max_tp1_atr
            
            # TP2 Range  
            min_tp2_dist = atr * self.constraints.min_tp2_atr
            max_tp2_dist = atr * self.constraints.max_tp2_atr
            
            # TP3 Range
            min_tp3_dist = atr * self.constraints.min_tp3_atr
            max_tp3_dist = atr * self.constraints.max_tp3_atr
            
            # ═══════════════════════════════════════════════════════════════
            # PERCENTAGE FLOORS - use HIGHER of ATR or % minimum
            # This ensures Swing TP1 is at least 3%, not 1.7%!
            # ═══════════════════════════════════════════════════════════════
            min_tp1_pct_dist = entry * (self.constraints.min_tp1_pct / 100)
            min_tp2_pct_dist = entry * (self.constraints.min_tp2_pct / 100)
            min_tp3_pct_dist = entry * (self.constraints.min_tp3_pct / 100)
            
            # Use the HIGHER minimum (floor enforcement)
            min_tp1_dist = max(min_tp1_dist, min_tp1_pct_dist)
            min_tp2_dist = max(min_tp2_dist, min_tp2_pct_dist)
            min_tp3_dist = max(min_tp3_dist, min_tp3_pct_dist)
            
            if direction == 'LONG':
                # Define ranges
                tp1_min = entry + min_tp1_dist
                tp1_max = entry + max_tp1_dist
                tp2_min = entry + min_tp2_dist
                tp2_max = entry + max_tp2_dist
                tp3_min = entry + min_tp3_dist
                tp3_max = entry + max_tp3_dist
                
                # Define HTF types (for preferring HTF on TP2/TP3)
                htf_types = {
                    LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                    LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                }
                
                # ═══════════════════════════════════════════════════════════
                # TP1: Current TF structure FIRST (quick target you can validate)
                # Only use HTF if NO current TF structure exists
                # CRITICAL: Must be >= tp1_min (percentage floor)
                # ═══════════════════════════════════════════════════════════
                
                # FIRST: Look for current TF structure only
                for level in tp_candidates:
                    is_structure = level.level_type in structure_types
                    is_htf = level.level_type in htf_types
                    
                    # Skip HTF for TP1 first pass - we want current TF
                    if is_htf:
                        continue
                    
                    # Must be within range AND above minimum!
                    if is_structure and level.price >= tp1_min and level.price <= tp1_max:
                        reward = level.price - entry
                        rr = reward / risk if risk > 0 else 0
                        
                        if rr >= 0.3:
                            tp1, tp1_level = level.price, level
                            break  # Found current TF structure - use it!
                
                # SECOND: Only if no current TF found, use HTF
                if tp1 == 0:
                    for level in tp_candidates:
                        is_htf = level.level_type in htf_types
                        
                        # Must be within range AND above minimum!
                        if is_htf and level.price >= tp1_min and level.price <= tp1_max:
                            reward = level.price - entry
                            rr = reward / risk if risk > 0 else 0
                            
                            if rr >= 0.3:
                                tp1, tp1_level = level.price, level
                                break
                
                # ═══════════════════════════════════════════════════════════
                # TP2: HTF structure PREFERRED (institutional target)
                # Only use current TF if no HTF available
                # ═══════════════════════════════════════════════════════════
                htf_tp2_candidates = [l for l in tp_candidates 
                                      if l.level_type in htf_types 
                                      and l.price >= tp2_min  # Enforce minimum!
                                      and l.price <= tp2_max
                                      and l.price > (tp1 if tp1 > 0 else entry)]
                
                if htf_tp2_candidates:
                    # Use nearest HTF level for TP2
                    tp2, tp2_level = htf_tp2_candidates[0].price, htf_tp2_candidates[0]
                else:
                    # Fallback to current TF structure
                    for level in tp_candidates:
                        if level.level_type in structure_types:
                            if tp2_min <= level.price <= tp2_max and level.price > (tp1 if tp1 > 0 else entry):
                                tp2, tp2_level = level.price, level
                                break
                
                # ═══════════════════════════════════════════════════════════
                # TP3: HTF structure PREFERRED (major target)
                # ═══════════════════════════════════════════════════════════
                htf_tp3_candidates = [l for l in tp_candidates 
                                      if l.level_type in htf_types 
                                      and l.price >= tp3_min  # Enforce minimum!
                                      and l.price <= tp3_max
                                      and l.price > (tp2 if tp2 > 0 else tp1 if tp1 > 0 else entry)]
                
                if htf_tp3_candidates:
                    # Use nearest HTF level for TP3
                    tp3, tp3_level = htf_tp3_candidates[0].price, htf_tp3_candidates[0]
                else:
                    # Fallback to current TF structure
                    for level in tp_candidates:
                        if level.level_type in structure_types:
                            if tp3_min <= level.price <= tp3_max and level.price > (tp2 if tp2 > 0 else entry):
                                tp3, tp3_level = level.price, level
                                break
                
                # ═══════════════════════════════════════════════════════════
                # FALLBACK: Any levels in range if structure didn't fill
                # ═══════════════════════════════════════════════════════════
                for level in tp_candidates:
                    if tp1 == 0 and tp1_min <= level.price <= tp1_max:
                        reward = level.price - entry
                        rr = reward / risk if risk > 0 else 0
                        if rr >= self.constraints.min_rr:
                            tp1, tp1_level = level.price, level
                    elif tp2 == 0 and tp2_min <= level.price <= tp2_max and level.price > (tp1 if tp1 > 0 else tp1_min):
                        tp2, tp2_level = level.price, level
                    elif tp3 == 0 and tp3_min <= level.price <= tp3_max and level.price > (tp2 if tp2 > 0 else tp2_min):
                        tp3, tp3_level = level.price, level
                
                # ATR-based defaults if no structure in range
                if tp1 == 0:
                    # Use minimum that meets R:R, capped at max
                    min_reward = risk * self.constraints.min_rr if risk > 0 else min_tp1_dist
                    tp1 = entry + max(min_reward, min_tp1_dist)
                    tp1 = min(tp1, tp1_max)  # Cap at max
                    
                if tp2 == 0:
                    tp2 = max(tp1 + min_spacing, entry + (min_tp2_dist + max_tp2_dist) / 2)
                    tp2 = min(tp2, tp2_max)
                    
                if tp3 == 0:
                    tp3 = max(tp2 + min_spacing, entry + (min_tp3_dist + max_tp3_dist) / 2)
                    tp3 = min(tp3, tp3_max)
                    
            else:  # SHORT
                # Define ranges
                tp1_min = entry - max_tp1_dist
                tp1_max = entry - min_tp1_dist
                tp2_min = entry - max_tp2_dist
                tp2_max = entry - min_tp2_dist
                tp3_min = entry - max_tp3_dist
                tp3_max = entry - min_tp3_dist
                
                # Define HTF types (for preferring HTF on TP2/TP3)
                htf_types = {
                    LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                    LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                }
                
                # ═══════════════════════════════════════════════════════════
                # ═══════════════════════════════════════════════════════════
                # TP1: Current TF structure FIRST (quick target you can validate)
                # Only use HTF if NO current TF structure exists
                # ═══════════════════════════════════════════════════════════
                
                # FIRST: Look for current TF structure only
                for level in tp_candidates:
                    is_structure = level.level_type in structure_types
                    is_htf = level.level_type in htf_types
                    
                    # Skip HTF for TP1 first pass - we want current TF
                    if is_htf:
                        continue
                    
                    if is_structure and level.price < entry and level.price >= tp1_min:
                        reward = entry - level.price
                        rr = reward / risk if risk > 0 else 0
                        
                        if rr >= 0.3:
                            tp1, tp1_level = level.price, level
                            break  # Found current TF structure - use it!
                
                # SECOND: Only if no current TF found, use HTF
                if tp1 == 0:
                    for level in tp_candidates:
                        is_htf = level.level_type in htf_types
                        
                        if is_htf and level.price < entry and level.price >= tp1_min:
                            reward = entry - level.price
                            rr = reward / risk if risk > 0 else 0
                            
                            if rr >= 0.3:
                                tp1, tp1_level = level.price, level
                                break
                
                # ═══════════════════════════════════════════════════════════
                # TP2: HTF structure PREFERRED (institutional target)
                # ═══════════════════════════════════════════════════════════
                htf_tp2_candidates = [l for l in tp_candidates 
                                      if l.level_type in htf_types 
                                      and l.price < (tp1 if tp1 > 0 else entry)
                                      and l.price >= tp2_min]
                
                if htf_tp2_candidates:
                    tp2, tp2_level = htf_tp2_candidates[0].price, htf_tp2_candidates[0]
                else:
                    for level in tp_candidates:
                        if level.level_type in structure_types:
                            if tp2_min <= level.price <= tp2_max and level.price < (tp1 if tp1 > 0 else entry):
                                tp2, tp2_level = level.price, level
                                break
                
                # ═══════════════════════════════════════════════════════════
                # TP3: HTF structure PREFERRED (major target)
                # ═══════════════════════════════════════════════════════════
                htf_tp3_candidates = [l for l in tp_candidates 
                                      if l.level_type in htf_types 
                                      and l.price < (tp2 if tp2 > 0 else tp1 if tp1 > 0 else entry)
                                      and l.price >= tp3_min]
                
                if htf_tp3_candidates:
                    tp3, tp3_level = htf_tp3_candidates[0].price, htf_tp3_candidates[0]
                else:
                    for level in tp_candidates:
                        if level.level_type in structure_types:
                            if tp3_min <= level.price <= tp3_max and level.price < (tp2 if tp2 > 0 else entry):
                                tp3, tp3_level = level.price, level
                                break
                
                # ═══════════════════════════════════════════════════════════
                # FALLBACK: Any levels in range
                # ═══════════════════════════════════════════════════════════
                for level in tp_candidates:
                    if tp1 == 0 and tp1_min <= level.price <= tp1_max:
                        reward = entry - level.price
                        rr = reward / risk if risk > 0 else 0
                        if rr >= self.constraints.min_rr:
                            tp1, tp1_level = level.price, level
                    elif tp2 == 0 and tp2_min <= level.price <= tp2_max and level.price < (tp1 if tp1 > 0 else tp1_max):
                        tp2, tp2_level = level.price, level
                    elif tp3 == 0 and tp3_min <= level.price <= tp3_max and level.price < (tp2 if tp2 > 0 else tp2_max):
                        tp3, tp3_level = level.price, level
                
                # ATR-based defaults if no structure in range
                if tp1 == 0:
                    min_reward = risk * self.constraints.min_rr if risk > 0 else min_tp1_dist
                    tp1 = entry - max(min_reward, min_tp1_dist)
                    tp1 = max(tp1, tp1_min)  # Cap at min for SHORT
                    
                if tp2 == 0:
                    tp2 = min(tp1 - min_spacing, entry - (min_tp2_dist + max_tp2_dist) / 2)
                    tp2 = max(tp2, tp2_min)
                    
                if tp3 == 0:
                    tp3 = min(tp2 - min_spacing, entry - (min_tp3_dist + max_tp3_dist) / 2)
                    tp3 = max(tp3, tp3_min)
        
        else:
            # No ATR available - use simple percentage fallback
            if direction == 'LONG':
                tp1 = entry * 1.02  # +2%
                tp2 = entry * 1.04  # +4%
                tp3 = entry * 1.06  # +6%
            else:
                tp1 = entry * 0.98  # -2%
                tp2 = entry * 0.96  # -4%
                tp3 = entry * 0.94  # -6%
        
        # ═══════════════════════════════════════════════════════════════════
        # ENSURE TP ORDERING (TP1 < TP2 < TP3 for LONG, opposite for SHORT)
        # ═══════════════════════════════════════════════════════════════════
        if direction == 'LONG':
            if tp2 <= tp1:
                tp2 = tp1 + min_spacing
            if tp3 <= tp2:
                tp3 = tp2 + min_spacing
        else:
            if tp2 >= tp1:
                tp2 = tp1 - min_spacing
            if tp3 >= tp2:
                tp3 = tp2 - min_spacing
        
        # ═══════════════════════════════════════════════════════════════════
        # HARD CAP ENFORCEMENT - TPs MUST stay within ATR limits!
        # This is critical - no TP should ever exceed mode constraints
        # ═══════════════════════════════════════════════════════════════════
        if atr > 0:
            max_tp1_dist = atr * self.constraints.max_tp1_atr
            max_tp2_dist = atr * self.constraints.max_tp2_atr
            max_tp3_dist = atr * self.constraints.max_tp3_atr
            
            if direction == 'LONG':
                tp1 = min(tp1, entry + max_tp1_dist)
                tp2 = min(tp2, entry + max_tp2_dist)
                tp3 = min(tp3, entry + max_tp3_dist)
                # Re-enforce ordering after caps
                if tp2 <= tp1:
                    tp2 = tp1 + min_spacing
                if tp3 <= tp2:
                    tp3 = tp2 + min_spacing
                # Final cap enforcement
                tp2 = min(tp2, entry + max_tp2_dist)
                tp3 = min(tp3, entry + max_tp3_dist)
            else:
                tp1 = max(tp1, entry - max_tp1_dist)
                tp2 = max(tp2, entry - max_tp2_dist)
                tp3 = max(tp3, entry - max_tp3_dist)
                # Re-enforce ordering after caps
                if tp2 >= tp1:
                    tp2 = tp1 - min_spacing
                if tp3 >= tp2:
                    tp3 = tp2 - min_spacing
                # Final cap enforcement
                tp2 = max(tp2, entry - max_tp2_dist)
                tp3 = max(tp3, entry - max_tp3_dist)
        
        # ═══════════════════════════════════════════════════════════════════
        # VALIDATE R:R
        # ═══════════════════════════════════════════════════════════════════
        reward_tp1 = abs(tp1 - entry) if tp1 > 0 else 0
        rr = reward_tp1 / risk if risk > 0 else 0
        
        rr_wait_reason = ""
        if rr < self.constraints.min_rr:
            rr_wait_reason = f"R:R of {rr:.1f}:1 below minimum {self.constraints.min_rr}:1 for {self.mode.value}"
        
        # ═══════════════════════════════════════════════════════════════════
        # DETERMINE FINAL DIRECTION (WAIT or actual direction)
        # ═══════════════════════════════════════════════════════════════════
        wait_reasons = [r for r in [htf_wait_reason, sl_wait_reason, rr_wait_reason] if r]
        final_direction = 'WAIT' if wait_reasons else direction
        final_wait_reason = wait_reasons[0] if wait_reasons else ""  # Use first reason
        
        # ═══════════════════════════════════════════════════════════════════
        # BUILD TRADE SETUP (always includes levels)
        # ═══════════════════════════════════════════════════════════════════
        reasoning_parts = []
        if entry_level:
            reasoning_parts.append(f"Entry: {entry_level.description}")
        if sl_level:
            reasoning_parts.append(f"SL: {sl_level.description} ({sl_type})")
        if tp1_level:
            reasoning_parts.append(f"TP1: {tp1_level.description}")
        
        return TradeSetup(
            direction=final_direction,
            entry=entry,
            entry_type=entry_type,
            entry_level=entry_level,
            stop_loss=sl,
            sl_type=sl_type,
            sl_level=sl_level,
            tp1=tp1,
            tp1_level=tp1_level,
            tp2=tp2,
            tp2_level=tp2_level,
            tp3=tp3,
            tp3_level=tp3_level,
            risk_reward=rr,
            confidence=self._calculate_confidence(entry_level, sl_level, tp1_level),
            reasoning=" | ".join(reasoning_parts),
            wait_reason=final_wait_reason
        )
    
    def _check_htf_alignment(self, direction: str) -> bool:
        """Check if direction aligns with HTF bias"""
        if self.htf_bias == 'NEUTRAL':
            return True
        if direction == 'LONG' and self.htf_bias == 'BEARISH':
            return False
        if direction == 'SHORT' and self.htf_bias == 'BULLISH':
            return False
        return True
    
    def _find_entry(
        self, 
        direction: str, 
        entry_levels: List[PriceLevel]
    ) -> Tuple[float, str, Optional[PriceLevel]]:
        """
        Find optimal entry point.
        
        For LONG: Look for support levels below current price (limit buy)
        For SHORT: Look for resistance levels above current price (limit sell)
        
        If no good level within reasonable distance, use market entry.
        """
        if not entry_levels:
            return self.current_price, 'market', None
        
        # Find nearest level within reasonable distance (1 ATR)
        max_distance = self.atr * 1.5
        
        for level in entry_levels:
            distance = abs(self.current_price - level.price)
            if distance <= max_distance:
                return level.price, 'limit', level
        
        # No good limit entry, use market
        return self.current_price, 'market', None
    
    def _find_stop_loss(
        self,
        direction: str,
        entry: float,
        support_levels: List[PriceLevel]
    ) -> Tuple[float, str, Optional[PriceLevel]]:
        """
        Find optimal stop loss using two strategies:
        
        1. Structure Break: Place SL just beyond a key structure level
           - Invalidates the trade thesis if hit
           - Tighter SL = better R:R
           
        2. Anti-Hunt: Place SL well beyond obvious levels
           - Avoids stop hunts at round numbers
           - Uses ATR buffer + ugly number
        
        Choose whichever is CLOSER but still valid.
        """
        if direction == 'LONG':
            # SL goes below entry
            # Find structure levels below entry
            structure_levels = [l for l in support_levels if l.price < entry]
            
            if structure_levels:
                # Anti-Hunt SL: BELOW nearest structure with meaningful buffer
                # Stop hunts sweep below structure - we place SL where hunts end
                nearest_structure = structure_levels[0]  # Already sorted nearest first
                
                # Buffer = 0.5-1.0 ATR below structure (where stop hunts end)
                anti_hunt_buffer = self.atr * 0.7  # 0.7 ATR buffer below structure
                anti_hunt_sl = nearest_structure.price - anti_hunt_buffer
                anti_hunt_sl = self._make_ugly_number(anti_hunt_sl, direction)
                
                # Update description to show it's anti-hunt
                sl_desc = f"Anti-hunt below {nearest_structure.description}"
            else:
                # No structure - use ATR-based anti-hunt from entry
                anti_hunt_sl = entry - (self.atr * 1.5)
                anti_hunt_sl = self._make_ugly_number(anti_hunt_sl, direction)
                sl_desc = "Anti-hunt (ATR-based)"
                nearest_structure = None
            
            # Check mode limit
            max_sl = entry * (1 - self.constraints.max_sl_pct / 100)
            
            if anti_hunt_sl >= max_sl:
                # Create a new level with anti-hunt description
                if nearest_structure:
                    from copy import copy
                    sl_level = copy(nearest_structure)
                    sl_level.description = sl_desc
                    return anti_hunt_sl, 'anti_hunt', sl_level
                else:
                    return anti_hunt_sl, 'anti_hunt', None
            else:
                # Use mode max
                return max_sl, f'max_{self.constraints.max_sl_pct}%', None
        
        else:  # SHORT
            # SL goes above entry
            structure_levels = [l for l in support_levels if l.price > entry]
            
            if structure_levels:
                # Anti-Hunt SL: ABOVE nearest structure with buffer
                nearest_structure = structure_levels[-1]  # For SHORT, highest is nearest
                
                anti_hunt_buffer = self.atr * 0.7
                anti_hunt_sl = nearest_structure.price + anti_hunt_buffer
                anti_hunt_sl = self._make_ugly_number(anti_hunt_sl, direction)
                
                sl_desc = f"Anti-hunt above {nearest_structure.description}"
            else:
                anti_hunt_sl = entry + (self.atr * 1.5)
                anti_hunt_sl = self._make_ugly_number(anti_hunt_sl, direction)
                sl_desc = "Anti-hunt (ATR-based)"
                nearest_structure = None
            
            max_sl = entry * (1 + self.constraints.max_sl_pct / 100)
            
            if anti_hunt_sl <= max_sl:
                if nearest_structure:
                    from copy import copy
                    sl_level = copy(nearest_structure)
                    sl_level.description = sl_desc
                    return anti_hunt_sl, 'anti_hunt', sl_level
                else:
                    return anti_hunt_sl, 'anti_hunt', None
            else:
                return max_sl, f'max_{self.constraints.max_sl_pct}%', None
    
    def _find_tp(
        self,
        direction: str,
        entry: float,
        target_levels: List[PriceLevel],
        tp_number: int
    ) -> Tuple[float, Optional[PriceLevel]]:
        """
        Find take profit level.
        
        TP1, TP2, TP3 are assigned to successive target levels.
        If not enough levels, extend from the last one.
        """
        if not target_levels:
            return 0, None
        
        # Filter levels that are valid targets
        if direction == 'LONG':
            valid_targets = [l for l in target_levels if l.price > entry]
        else:
            valid_targets = [l for l in target_levels if l.price < entry]
        
        if not valid_targets:
            return 0, None
        
        # Assign by index (TP1 = nearest, TP2 = second, etc.)
        idx = tp_number - 1
        
        if idx < len(valid_targets):
            return valid_targets[idx].price, valid_targets[idx]
        elif len(valid_targets) > 0:
            # Extend from last level
            last = valid_targets[-1]
            extension = abs(last.price - entry) * 0.5 * (idx - len(valid_targets) + 1)
            if direction == 'LONG':
                return last.price + extension, None
            else:
                return last.price - extension, None
        
        return 0, None
    
    def _make_ugly_number(self, price: float, direction: str) -> float:
        """
        Convert price to an "ugly" number that avoids common stop-hunt levels.
        
        Avoids: .00, .25, .50, .75, .10, .20, .30, etc.
        Creates: .03, .07, .13, .17, .23, .27, .33, etc.
        """
        if price <= 0:
            return price
        
        # Determine decimal places based on price magnitude
        if price >= 1000:
            decimal_places = 2
        elif price >= 10:
            decimal_places = 3
        elif price >= 1:
            decimal_places = 4
        else:
            decimal_places = 6
        
        # Round to decimal places
        rounded = round(price, decimal_places)
        
        # Add small offset to avoid round numbers
        # For LONG SL (below), subtract a bit more
        # For SHORT SL (above), add a bit more
        offset_multiplier = 10 ** (-decimal_places)
        ugly_offsets = [3, 7, 13, 17, 23, 27, 33, 37, 43, 47]
        
        # Pick an offset based on the price to make it consistent
        offset_idx = int(price * 100) % len(ugly_offsets)
        offset = ugly_offsets[offset_idx] * offset_multiplier
        
        if direction == 'LONG':
            return rounded - offset
        else:
            return rounded + offset
    
    def _calculate_confidence(
        self,
        entry_level: Optional[PriceLevel],
        sl_level: Optional[PriceLevel],
        tp1_level: Optional[PriceLevel]
    ) -> float:
        """Calculate overall confidence based on level quality"""
        confidence = 50  # Base
        
        if entry_level:
            confidence += entry_level.confidence * 15
        if sl_level:
            confidence += sl_level.confidence * 10
        if tp1_level:
            confidence += tp1_level.confidence * 15
        
        # HTF alignment bonus
        if self.htf_bias != 'NEUTRAL':
            confidence += 10
        
        return min(100, confidence)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def get_optimized_trade(
    df: pd.DataFrame,
    current_price: float,
    direction: str,
    timeframe: str,
    smc_data: Dict,
    htf_data: Optional[Dict] = None,
    volume_data: Optional[Dict] = None,
    htf_bias: Optional[str] = None,
    mode: Optional[TradingMode] = None,
    winning_strategy: Optional[str] = None,  # From backtest: 'OB', 'FVG', 'VWAP', etc.
) -> TradeSetup:
    """
    Main interface - get optimized trade setup.
    
    Args:
        df: OHLCV DataFrame
        current_price: Current market price
        direction: 'LONG' or 'SHORT'
        timeframe: Current timeframe string
        smc_data: SMC analysis dict (OB, FVG, swings)
        htf_data: Higher timeframe analysis dict (optional)
        volume_data: Volume profile dict (optional)
        htf_bias: HTF structure bias ('BULLISH', 'BEARISH', 'NEUTRAL')
        mode: Trading mode (auto-detected from timeframe if None)
        winning_strategy: Backtested winning strategy to prioritize levels
        
    Returns:
        TradeSetup with optimal levels or WAIT if no valid solution
    """
    # Auto-detect mode from timeframe
    if mode is None:
        mode = TIMEFRAME_MODE_MAP.get(timeframe, TradingMode.DAY_TRADE)
    
    # Collect all levels
    collector = LevelCollector(
        df=df,
        current_price=current_price,
        timeframe=timeframe,
    )
    
    collector.collect_from_smc(smc_data)
    
    if htf_data:
        collector.collect_from_htf(htf_data)
    
    if volume_data:
        collector.collect_from_volume(volume_data)
    
    collector.collect_from_historical()
    
    # Additional level collection
    collector.collect_fibonacci()
    collector.collect_liquidity_zones()
    collector.collect_round_numbers()
    
    # Optimize with winning strategy priority
    optimizer = TradeOptimizer(
        collector=collector,
        mode=mode,
        htf_bias=htf_bias,
        winning_strategy=winning_strategy,
    )
    
    return optimizer.optimize(direction)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'Open': close - np.random.rand(100) * 0.5,
        'High': close + np.random.rand(100),
        'Low': close - np.random.rand(100),
        'Close': close,
        'Volume': np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    # Sample SMC data
    smc_data = {
        'bullish_ob': True,
        'bullish_ob_top': 99.5,
        'bullish_ob_bottom': 99.0,
        'bearish_ob': True,
        'bearish_ob_bottom': 101.5,
        'bearish_ob_top': 102.0,
        'swing_low': 98.5,
        'swing_high': 103.0,
    }
    
    # Get optimized trade
    setup = get_optimized_trade(
        df=df,
        current_price=100.0,
        direction='LONG',
        timeframe='15m',
        smc_data=smc_data,
        htf_bias='BULLISH',
    )
    
    print(f"Direction: {setup.direction}")
    print(f"Entry: {setup.entry} ({setup.entry_type})")
    print(f"Stop Loss: {setup.stop_loss} ({setup.sl_type})")
    print(f"TP1: {setup.tp1}")
    print(f"TP2: {setup.tp2}")
    print(f"TP3: {setup.tp3}")
    print(f"R:R: {setup.risk_reward:.2f}:1")
    print(f"Confidence: {setup.confidence:.0f}%")
    print(f"Reasoning: {setup.reasoning}")
    
    if setup.wait_reason:
        print(f"WAIT: {setup.wait_reason}")
