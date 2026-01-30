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
    """The optimized trade setup with educational descriptions"""
    direction: str  # 'LONG', 'SHORT', or 'WAIT'
    entry: float
    entry_type: str  # 'market' or 'limit'
    entry_level: Optional[PriceLevel] = None
    
    stop_loss: float = 0
    sl_type: str = ""  # 'anti_hunt', 'structure_break', or mode constraint
    sl_level: Optional[PriceLevel] = None
    sl_description: str = ""  # Educational description for display
    
    tp1: float = 0
    tp1_type: str = ""  # Structure type or 'ML_X.X%'
    tp1_level: Optional[PriceLevel] = None
    tp1_description: str = ""  # Educational description
    tp2: float = 0
    tp2_type: str = ""
    tp2_level: Optional[PriceLevel] = None
    tp2_description: str = ""
    tp3: float = 0
    tp3_type: str = ""
    tp3_level: Optional[PriceLevel] = None
    tp3_description: str = ""
    
    risk_reward: float = 0
    risk_pct: float = 0  # SL distance as percentage
    rr_tp1: float = 0    # R:R to TP1
    confidence: float = 0
    reasoning: str = ""
    
    # Trading mode info
    mode: Optional['TradingMode'] = None
    htf_aligned: bool = True
    
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
        min_sl_pct=0.8,           # Min 0.8% SL (prevent too tight stops)
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
        
        try:
            high = self.df['High'] if 'High' in self.df.columns else self.df['high']
            low = self.df['Low'] if 'Low' in self.df.columns else self.df['low']
            close = self.df['Close'] if 'Close' in self.df.columns else self.df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean().iloc[-1]
            
            # Handle NaN or invalid values
            if pd.isna(atr) or atr <= 0:
                return self.current_price * 0.02  # Fallback: 2%
            
            return float(atr)
        except Exception:
            return self.current_price * 0.02  # Fallback: 2%
    
    def collect_from_smc(self, smc_data: Dict) -> None:
        """
        Collect levels from SMC analysis.
        
        Expected smc_data keys:
        - bullish_ob_top, bullish_ob_bottom (or bullish_ob flag)
        - bearish_ob_top, bearish_ob_bottom (or bearish_ob flag)
        - bullish_obs, bearish_obs (LISTS of all OBs for TP targeting)
        - fvg_bullish_top, fvg_bullish_bottom (or fvg_bullish flag)
        - fvg_bearish_top, fvg_bearish_bottom (or fvg_bearish flag)
        - bullish_fvgs, bearish_fvgs (LISTS of all FVGs)
        - swing_high, swing_low
        - bos, choch, bos_level, choch_level
        """
        # ═══════════════════════════════════════════════════════════════
        # COLLECT ALL BEARISH OBs (resistance for LONG TPs)
        # ═══════════════════════════════════════════════════════════════
        bearish_obs = smc_data.get('bearish_obs', [])
        if bearish_obs:
            for ob in bearish_obs:
                ob_top = ob.get('top', 0)
                ob_bottom = ob.get('bottom', 0)
                if ob_bottom > 0 and ob_bottom > self.current_price:
                    self.levels.append(PriceLevel(
                        price=ob_bottom,
                        level_type=LevelType.OB_BEARISH,
                        role=LevelRole.RESISTANCE,
                        source_tf=self.timeframe,
                        confidence=0.9,
                        description=f"Bearish OB ({self.timeframe})"
                    ))
                if ob_top > 0 and ob_top > self.current_price:
                    self.levels.append(PriceLevel(
                        price=ob_top,
                        level_type=LevelType.OB_BEARISH,
                        role=LevelRole.RESISTANCE,
                        source_tf=self.timeframe,
                        confidence=0.85,
                        description=f"Bearish OB top ({self.timeframe})"
                    ))
        else:
            # Fallback to single OB values
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
        
        # ═══════════════════════════════════════════════════════════════
        # COLLECT ALL BULLISH OBs (support for SHORT TPs / LONG entries)
        # ═══════════════════════════════════════════════════════════════
        bullish_obs = smc_data.get('bullish_obs', [])
        if bullish_obs:
            for ob in bullish_obs:
                ob_top = ob.get('top', 0)
                ob_bottom = ob.get('bottom', 0)
                if ob_top > 0 and ob_top < self.current_price:
                    self.levels.append(PriceLevel(
                        price=ob_top,
                        level_type=LevelType.OB_BULLISH,
                        role=LevelRole.SUPPORT,
                        source_tf=self.timeframe,
                        confidence=0.9,
                        description=f"Bullish OB ({self.timeframe})"
                    ))
                if ob_bottom > 0 and ob_bottom < self.current_price:
                    self.levels.append(PriceLevel(
                        price=ob_bottom,
                        level_type=LevelType.OB_BULLISH,
                        role=LevelRole.SUPPORT,
                        source_tf=self.timeframe,
                        confidence=0.85,
                        description=f"Bullish OB bottom ({self.timeframe})"
                    ))
        else:
            # Fallback to single OB values
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
        
        # ═══════════════════════════════════════════════════════════════
        # COLLECT ALL BEARISH FVGs (resistance for LONG TPs)
        # ═══════════════════════════════════════════════════════════════
        bearish_fvgs = smc_data.get('bearish_fvgs', [])
        if bearish_fvgs:
            for fvg in bearish_fvgs:
                fvg_top = fvg.get('top', fvg.get('high', 0))
                fvg_bottom = fvg.get('bottom', fvg.get('low', 0))
                if fvg_bottom > 0 and fvg_bottom > self.current_price:
                    self.levels.append(PriceLevel(
                        price=fvg_bottom,
                        level_type=LevelType.FVG_BEARISH,
                        role=LevelRole.RESISTANCE,
                        source_tf=self.timeframe,
                        confidence=0.8,
                        description=f"Bearish FVG ({self.timeframe})"
                    ))
        else:
            # Fallback to single FVG values
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
        
        # ═══════════════════════════════════════════════════════════════
        # COLLECT ALL BULLISH FVGs (support for SHORT TPs / LONG entries)
        # ═══════════════════════════════════════════════════════════════
        bullish_fvgs = smc_data.get('bullish_fvgs', [])
        if bullish_fvgs:
            for fvg in bullish_fvgs:
                fvg_top = fvg.get('top', fvg.get('high', 0))
                if fvg_top > 0 and fvg_top < self.current_price:
                    self.levels.append(PriceLevel(
                        price=fvg_top,
                        level_type=LevelType.FVG_BULLISH,
                        role=LevelRole.SUPPORT,
                        source_tf=self.timeframe,
                        confidence=0.8,
                        description=f"Bullish FVG ({self.timeframe})"
                    ))
        else:
            # Fallback to single FVG values
            fvg_bullish_top = smc_data.get('fvg_bullish_top', 0)
            if fvg_bullish_top > 0:
                self.levels.append(PriceLevel(
                    price=fvg_bullish_top,
                    level_type=LevelType.FVG_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    confidence=0.8,
                    description=f"Bullish FVG ({self.timeframe})"
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
    
    Supports ML blending based on analysis_mode:
    - 'rules': Pure structure-based levels
    - 'ml': ML-predicted levels as primary
    - 'hybrid': Blend structure + ML (ML adjusts R:R requirements)
    
    Returns WAIT if no valid solution exists.
    """
    
    def __init__(
        self,
        collector: LevelCollector,
        mode: TradingMode,
        htf_bias: Optional[str] = None,  # 'BULLISH', 'BEARISH', 'NEUTRAL'
        winning_strategy: Optional[str] = None,  # 'OB', 'FVG', 'VWAP', etc.
        ml_prediction: Optional[Dict] = None,  # ML predictions
        analysis_mode: str = 'hybrid',  # 'rules', 'ml', 'hybrid'
    ):
        self.collector = collector
        self.mode = mode
        self.trading_mode = mode  # Alias for consistency
        self.constraints = MODE_CONSTRAINTS[mode]  # Base constraints from mode
        self.htf_bias = htf_bias or 'NEUTRAL'
        self.winning_strategy = winning_strategy  # From backtest results
        self.ml_prediction = ml_prediction
        self.analysis_mode = analysis_mode
        
        self.current_price = collector.current_price
        self.atr = collector.atr
        
        # ═══════════════════════════════════════════════════════════════
        # ML TARGET CALCULATION - Use ML predictions DIRECTLY from trained model
        # Mode-specific model already knows appropriate levels for the trading mode
        # ═══════════════════════════════════════════════════════════════
        self.ml_targets = None
        if ml_prediction and analysis_mode in ['ml', 'hybrid']:
            optimal_tp1_pct = ml_prediction.get('optimal_tp1_pct', 0)
            optimal_tp2_pct = ml_prediction.get('optimal_tp2_pct', 0)
            optimal_tp3_pct = ml_prediction.get('optimal_tp3_pct', 0)
            optimal_sl_pct = ml_prediction.get('optimal_sl_pct', 0)
            
            if optimal_tp1_pct > 0 and optimal_sl_pct > 0:
                # Use ML values directly - trust the mode-specific trained model
                self.ml_targets = {
                    'sl_pct': optimal_sl_pct,
                    'tp1_pct': optimal_tp1_pct,
                    'tp2_pct': optimal_tp2_pct if optimal_tp2_pct > 0 else optimal_tp1_pct * 1.618,
                    'tp3_pct': optimal_tp3_pct if optimal_tp3_pct > 0 else optimal_tp1_pct * 2.618,
                    'rr': optimal_tp1_pct / optimal_sl_pct,
                }
    
    def optimize(self, direction: str) -> TradeSetup:
        """
        Find optimal trade setup for given direction.
        
        Args:
            direction: 'LONG' or 'SHORT'
            
        Returns:
            TradeSetup with optimal levels, or direction='WAIT' if no valid solution
        """
        try:
            return self._optimize_internal(direction)
        except Exception as e:
            # FALLBACK: Return mode-based defaults on ANY error
            print(f"⚠️ Optimizer error: {e}")
            import traceback
            traceback.print_exc()
            
            # Generate safe fallback values
            entry = self.current_price if self.current_price and self.current_price > 0 else 100
            mode_sl_pct = self.constraints.max_sl_pct if self.constraints else 2.5
            mode_tp_pct = self.constraints.min_tp1_pct if self.constraints else 1.5
            
            if direction == 'LONG':
                sl = entry * (1 - mode_sl_pct / 100)
                tp1 = entry * (1 + mode_tp_pct / 100)
                tp2 = entry * (1 + mode_tp_pct * 2 / 100)
                tp3 = entry * (1 + mode_tp_pct * 3 / 100)
            else:
                sl = entry * (1 + mode_sl_pct / 100)
                tp1 = entry * (1 - mode_tp_pct / 100)
                tp2 = entry * (1 - mode_tp_pct * 2 / 100)
                tp3 = entry * (1 - mode_tp_pct * 3 / 100)
            
            mode_name = self.mode.value if self.mode else 'Day Trade'
            
            return TradeSetup(
                direction=direction,
                entry=entry,
                entry_type='market',
                entry_level=None,
                stop_loss=sl,
                sl_type=f'fallback_{mode_sl_pct}%',
                sl_level=None,
                sl_description=f"📍 Source: {mode_name} fallback {mode_sl_pct}%\n⚠️ Optimizer error occurred\n💡 Using safe mode defaults",
                tp1=tp1,
                tp1_type=f'fallback_{mode_tp_pct}%',
                tp1_level=None,
                tp1_description=f"📍 Source: {mode_name} minimum {mode_tp_pct}%\n💡 Fallback target",
                tp2=tp2,
                tp2_type='fallback_extension',
                tp2_level=None,
                tp2_description=f"📍 Source: Mode-based extension\n💡 Extended from TP1",
                tp3=tp3,
                tp3_type='fallback_extension',
                tp3_level=None,
                tp3_description=f"📍 Source: Mode-based extension\n💡 Extended from TP2",
                risk_reward=mode_tp_pct / mode_sl_pct if mode_sl_pct > 0 else 0,
                risk_pct=mode_sl_pct,
                rr_tp1=mode_tp_pct / mode_sl_pct if mode_sl_pct > 0 else 0,
                confidence=0.3,
                reasoning="Fallback: Optimizer error - using mode defaults",
                mode=self.mode,
                htf_aligned=True,
                wait_reason=""
            )
    
    def _optimize_internal(self, direction: str) -> TradeSetup:
        """
        Internal optimization logic - called by optimize() with error handling.
        """
        # ═══════════════════════════════════════════════════════════════════
        # INPUT VALIDATION - Ensure we have valid data to work with
        # ═══════════════════════════════════════════════════════════════════
        if not self.current_price or self.current_price <= 0:
            raise ValueError(f"Invalid current_price: {self.current_price}")
        
        if not self.collector:
            raise ValueError("No level collector provided")
        
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
        # FIND STOP LOSS (with educational description)
        # _find_stop_loss handles: Anti-hunt → ML comparison → min/max enforcement
        # ═══════════════════════════════════════════════════════════════════
        sl, sl_type, sl_level, sl_description = self._find_stop_loss(direction, entry, entry_levels)
        
        # Calculate SL percentage for reference
        atr = self.collector.atr if self.collector else entry * 0.01
        sl_pct = abs(self.current_price - sl) / self.current_price * 100 if self.current_price > 0 else 0
        sl_wait_reason = ""
        
        # NOTE: Min/Max SL enforcement is now handled INSIDE _find_stop_loss
        # No duplicate enforcement needed here!
        
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
            Collect structure levels in PRIORITY ORDER based on TRADING MODE.
            
            ═══════════════════════════════════════════════════════════════════
            MODE-SPECIFIC STRUCTURE PRIORITY:
            ═══════════════════════════════════════════════════════════════════
            
            SCALP (1m, 5m):
              → Current TF OB > Current TF FVG > Current TF Swing
              → NO HTF targets (too far), NO Fib (too far)
              
            DAY TRADE (15m, 1h):
              → Current TF OB > HTF OB > Current TF FVG > Swing > Fib
              → HTF = 4h context
              
            SWING (4h, 1d):
              → HTF OB > HTF Swing > Fib > Current TF OB
              → HTF = 1d/1w context, Fib is primary target method
              
            INVESTMENT (1w):
              → HTF Swing > Fib Extensions > Major Levels
              → Long-term structural targets
            ═══════════════════════════════════════════════════════════════════
            """
            all_levels = []
            mode = self.trading_mode
            
            # ═══════════════════════════════════════════════════════════════
            # Define level type groups
            # ═══════════════════════════════════════════════════════════════
            current_tf_ob = {LevelType.OB_BULLISH, LevelType.OB_BEARISH}
            current_tf_fvg = {LevelType.FVG_BULLISH, LevelType.FVG_BEARISH}
            current_tf_swing = {LevelType.SWING_HIGH, LevelType.SWING_LOW}
            htf_ob = {LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH}
            htf_swing = {LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW}
            fib_levels = {LevelType.FIB_382, LevelType.FIB_500, LevelType.FIB_618, LevelType.FIB_786}
            liquidity = {LevelType.LIQUIDITY_HIGH, LevelType.LIQUIDITY_LOW,
                        LevelType.PREV_DAY_HIGH, LevelType.PREV_DAY_LOW,
                        LevelType.PREV_WEEK_HIGH, LevelType.PREV_WEEK_LOW}
            volume_profile = {LevelType.POC, LevelType.VAH, LevelType.VAL, LevelType.VWAP}
            
            # Helper to filter by direction
            def filter_by_direction(level_types: set) -> List[PriceLevel]:
                if direction == 'LONG':
                    # LONG targets RESISTANCE above current price
                    resistance_map = {
                        LevelType.OB_BULLISH: False, LevelType.OB_BEARISH: True,  # Bearish OB = resistance
                        LevelType.FVG_BULLISH: False, LevelType.FVG_BEARISH: True,
                        LevelType.SWING_HIGH: True, LevelType.SWING_LOW: False,
                        LevelType.HTF_OB_BULLISH: False, LevelType.HTF_OB_BEARISH: True,
                        LevelType.HTF_SWING_HIGH: True, LevelType.HTF_SWING_LOW: False,
                    }
                    return [l for l in self.collector.levels 
                            if l.level_type in level_types 
                            and l.price > self.collector.current_price
                            and resistance_map.get(l.level_type, True)]
                else:
                    # SHORT targets SUPPORT below current price
                    support_map = {
                        LevelType.OB_BULLISH: True, LevelType.OB_BEARISH: False,  # Bullish OB = support
                        LevelType.FVG_BULLISH: True, LevelType.FVG_BEARISH: False,
                        LevelType.SWING_HIGH: False, LevelType.SWING_LOW: True,
                        LevelType.HTF_OB_BULLISH: True, LevelType.HTF_OB_BEARISH: False,
                        LevelType.HTF_SWING_HIGH: False, LevelType.HTF_SWING_LOW: True,
                    }
                    return [l for l in self.collector.levels 
                            if l.level_type in level_types 
                            and l.price < self.collector.current_price
                            and support_map.get(l.level_type, True)]
            
            def filter_general(level_types: set) -> List[PriceLevel]:
                """For non-directional levels (Fib, Volume, Liquidity)"""
                if direction == 'LONG':
                    return [l for l in self.collector.levels 
                            if l.level_type in level_types and l.price > self.collector.current_price]
                else:
                    return [l for l in self.collector.levels 
                            if l.level_type in level_types and l.price < self.collector.current_price]
            
            # ═══════════════════════════════════════════════════════════════
            # SCALP MODE - Current TF only, tight targets
            # Priority: Current OB > Current FVG > Current Swing
            # NO HTF, NO Fib (too distant for scalp)
            # ═══════════════════════════════════════════════════════════════
            if mode == TradingMode.SCALP:
                all_levels.extend(filter_by_direction(current_tf_ob))
                all_levels.extend(filter_by_direction(current_tf_fvg))
                all_levels.extend(filter_by_direction(current_tf_swing))
                # Liquidity for scalp (intraday highs/lows)
                all_levels.extend(filter_general(liquidity))
            
            # ═══════════════════════════════════════════════════════════════
            # DAY TRADE MODE - Current TF + HTF, include Fib
            # Priority: Current OB > HTF OB > Current FVG > Swing > Fib
            # ═══════════════════════════════════════════════════════════════
            elif mode == TradingMode.DAY_TRADE:
                all_levels.extend(filter_by_direction(current_tf_ob))
                all_levels.extend(filter_by_direction(htf_ob))
                all_levels.extend(filter_by_direction(current_tf_fvg))
                all_levels.extend(filter_by_direction(current_tf_swing))
                all_levels.extend(filter_by_direction(htf_swing))
                all_levels.extend(filter_general(fib_levels))  # Fib included
                all_levels.extend(filter_general(liquidity))
                all_levels.extend(filter_general(volume_profile))
            
            # ═══════════════════════════════════════════════════════════════
            # SWING MODE - HTF first, Fib is primary method
            # Priority: HTF OB > HTF Swing > Fib > Current TF OB
            # ═══════════════════════════════════════════════════════════════
            elif mode == TradingMode.SWING:
                all_levels.extend(filter_by_direction(htf_ob))
                all_levels.extend(filter_by_direction(htf_swing))
                all_levels.extend(filter_general(fib_levels))  # Fib is primary!
                all_levels.extend(filter_by_direction(current_tf_ob))
                all_levels.extend(filter_by_direction(current_tf_fvg))
                all_levels.extend(filter_by_direction(current_tf_swing))
                all_levels.extend(filter_general(liquidity))
            
            # ═══════════════════════════════════════════════════════════════
            # INVESTMENT MODE - HTF Swing + Fib Extensions primary
            # Priority: HTF Swing > Fib > Current Swing > Liquidity
            # ═══════════════════════════════════════════════════════════════
            elif mode == TradingMode.INVESTMENT:
                all_levels.extend(filter_by_direction(htf_swing))
                all_levels.extend(filter_general(fib_levels))  # Fib primary
                all_levels.extend(filter_by_direction(htf_ob))
                all_levels.extend(filter_by_direction(current_tf_swing))
                all_levels.extend(filter_general(liquidity))
            
            # ═══════════════════════════════════════════════════════════════
            # DEDUPLICATE - Prefer STRUCTURE over calculated levels
            # When two levels are close, keep OB/Swing over Fib/Round
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
                LevelType.PREV_WEEK_HIGH, LevelType.PREV_WEEK_LOW,
                LevelType.LIQUIDITY_HIGH, LevelType.LIQUIDITY_LOW,
            }
            volume_types = {
                LevelType.POC, LevelType.VAH, LevelType.VAL, LevelType.VWAP,
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
        # ML MODE - SL is ALWAYS structure-based (anti-hunt)
        # ML only affects TPs, never SL - this ensures consistency across all modes
        # ═══════════════════════════════════════════════════════════════════
        # NOTE: Previous code here overrode SL with ML prediction - REMOVED
        # SL is now handled exclusively by _find_stop_loss() which uses anti-hunt
        
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
        
        # ═══════════════════════════════════════════════════════════════════
        # HYBRID MODE: Blend structure levels with ML targets
        # Strategy: Find structure levels, but prefer those closest to ML targets
        # ═══════════════════════════════════════════════════════════════════
        ml_tp1_target = None
        ml_tp2_target = None
        ml_tp3_target = None
        
        if self.analysis_mode == 'hybrid' and self.ml_targets:
            ml_rr = self.ml_targets['rr']
            # Increase R:R filter based on ML
            if ml_rr > min_rr_tp1:
                min_rr_tp1 = max(min_rr_tp1, ml_rr * 0.7)
            
            # Calculate ML target prices for hybrid scoring
            # TP1 from ML trained, TP2/TP3 from Fib extension
            ml_tp1_pct = self.ml_targets['tp1_pct']
            ml_tp2_pct = self.ml_targets['tp2_pct']  # Fib 1.618×
            ml_tp3_pct = self.ml_targets['tp3_pct']  # Fib 2.618×
            
            if direction == 'LONG':
                ml_tp1_target = entry * (1 + ml_tp1_pct / 100)
                ml_tp2_target = entry * (1 + ml_tp2_pct / 100)
                ml_tp3_target = entry * (1 + ml_tp3_pct / 100)
            else:
                ml_tp1_target = entry * (1 - ml_tp1_pct / 100)
                ml_tp2_target = entry * (1 - ml_tp2_pct / 100)
                ml_tp3_target = entry * (1 - ml_tp3_pct / 100)
        
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
        
        # ═══════════════════════════════════════════════════════════════
        # MODE-SPECIFIC STRUCTURE TYPES
        # Scalp: Current TF only (no HTF, no Fib)
        # Day Trade/Swing/Investment: Include HTF and Fib
        # ═══════════════════════════════════════════════════════════════
        if self.trading_mode == TradingMode.SCALP:
            # SCALP: Current TF + HTF for better TP targets
            structure_types = {
                LevelType.OB_BULLISH, LevelType.OB_BEARISH,
                LevelType.FVG_BULLISH, LevelType.FVG_BEARISH,
                LevelType.SWING_HIGH, LevelType.SWING_LOW,
                LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                # NO Fib for scalp (still too far)
            }
        elif self.trading_mode == TradingMode.DAY_TRADE:
            # DAY TRADE: Current TF + HTF + Fib
            structure_types = {
                LevelType.OB_BULLISH, LevelType.OB_BEARISH,
                LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                LevelType.FVG_BULLISH, LevelType.FVG_BEARISH,
                LevelType.SWING_HIGH, LevelType.SWING_LOW,
                LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                LevelType.FIB_382, LevelType.FIB_500, 
                LevelType.FIB_618, LevelType.FIB_786,
            }
        elif self.trading_mode == TradingMode.SWING:
            # SWING: HTF primary, Fib important
            structure_types = {
                LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                LevelType.FIB_382, LevelType.FIB_500, 
                LevelType.FIB_618, LevelType.FIB_786,
                LevelType.OB_BULLISH, LevelType.OB_BEARISH,
                LevelType.FVG_BULLISH, LevelType.FVG_BEARISH,
                LevelType.SWING_HIGH, LevelType.SWING_LOW,
            }
        else:  # INVESTMENT
            # INVESTMENT: HTF Swing + Fib primary
            structure_types = {
                LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                LevelType.FIB_382, LevelType.FIB_500, 
                LevelType.FIB_618, LevelType.FIB_786,
                LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                LevelType.SWING_HIGH, LevelType.SWING_LOW,
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
                
                # Define HTF types - now enabled for ALL modes including Scalp
                htf_types = {
                    LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                    LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
                }
                
                # ═══════════════════════════════════════════════════════════
                # MODE-SPECIFIC TP SELECTION using priority-sorted candidates
                # Priority order depends on trading mode (see _get_mode_tp_priority)
                # ═══════════════════════════════════════════════════════════
                
                # Sort candidates by mode-specific priority
                sorted_candidates = self._sort_tp_candidates_by_mode_priority(
                    tp_candidates, 'LONG', entry, min_price=0
                )
                
                # TP1: First valid level in priority order within TP1 range
                for level in sorted_candidates:
                    if tp1_min <= level.price <= tp1_max:
                        reward = level.price - entry
                        rr = reward / risk if risk > 0 else 0
                        if rr >= 0.3:
                            # HYBRID MODE: Check ML proximity
                            if self.analysis_mode == 'hybrid' and ml_tp1_target:
                                proximity = abs(level.price - ml_tp1_target) / entry * 100
                                if proximity < 1.0:  # Within 1% of ML target
                                    tp1, tp1_level = level.price, level
                                    break
                                elif tp1 == 0:
                                    tp1, tp1_level = level.price, level
                            else:
                                tp1, tp1_level = level.price, level
                                break
                
                # TP2: Next valid level BEYOND TP1 in priority order
                for level in sorted_candidates:
                    if level.price > (tp1 if tp1 > 0 else entry) and tp2_min <= level.price <= tp2_max:
                        tp2, tp2_level = level.price, level
                        break
                
                # TP3: Next valid level BEYOND TP2 in priority order
                for level in sorted_candidates:
                    if level.price > (tp2 if tp2 > 0 else tp1 if tp1 > 0 else entry) and tp3_min <= level.price <= tp3_max:
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
                
                # ═══════════════════════════════════════════════════════════
                # THIRD: If still no TP, look for ANY structure above entry
                # BUT MUST STILL meet minimum percentage requirement!
                # ═══════════════════════════════════════════════════════════
                if tp1 == 0:
                    for level in tp_candidates:
                        is_structure = level.level_type in structure_types
                        is_htf = level.level_type in htf_types
                        
                        # MUST be above minimum AND have reasonable R:R
                        if (is_structure or is_htf) and level.price >= tp1_min:
                            reward = level.price - entry
                            rr = reward / risk if risk > 0 else 0
                            if rr >= 0.5:  # At least 0.5:1 R:R
                                tp1, tp1_level = level.price, level
                                break
                
                if tp2 == 0 and tp1 > 0:
                    for level in tp_candidates:
                        is_structure = level.level_type in structure_types
                        is_htf = level.level_type in htf_types
                        
                        if (is_structure or is_htf) and level.price > tp1:
                            tp2, tp2_level = level.price, level
                            break
                
                if tp3 == 0 and tp2 > 0:
                    for level in tp_candidates:
                        is_structure = level.level_type in structure_types
                        is_htf = level.level_type in htf_types
                        
                        if (is_structure or is_htf) and level.price > tp2:
                            tp3, tp3_level = level.price, level
                            break
                
                # ATR-based defaults ONLY if no structure found at all
                # HYBRID MODE: Use ML targets as fallback instead of pure ATR
                if tp1 == 0:
                    if self.analysis_mode == 'hybrid' and ml_tp1_target:
                        tp1 = ml_tp1_target  # Use ML target
                    else:
                        # Use minimum that meets R:R, capped at max
                        min_reward = risk * self.constraints.min_rr if risk > 0 else min_tp1_dist
                        tp1 = entry + max(min_reward, min_tp1_dist)
                        tp1 = min(tp1, tp1_max)  # Cap at max
                    
                if tp2 == 0:
                    if self.analysis_mode == 'hybrid' and ml_tp2_target:
                        tp2 = ml_tp2_target  # Use ML target
                    else:
                        tp2 = max(tp1 + min_spacing, entry + (min_tp2_dist + max_tp2_dist) / 2)
                        tp2 = min(tp2, tp2_max)
                    # CRITICAL: Ensure TP2 > TP1 regardless of capping
                    if tp2 <= tp1:
                        tp2 = tp1 + min_spacing
                    
                if tp3 == 0:
                    if self.analysis_mode == 'hybrid' and ml_tp3_target:
                        tp3 = ml_tp3_target  # Use ML target
                    else:
                        tp3 = max(tp2 + min_spacing, entry + (min_tp3_dist + max_tp3_dist) / 2)
                        tp3 = min(tp3, tp3_max)
                    # CRITICAL: Ensure TP3 > TP2 regardless of capping
                    if tp3 <= tp2:
                        tp3 = tp2 + min_spacing
                    
            else:  # SHORT
                # Define ranges
                tp1_min = entry - max_tp1_dist
                tp1_max = entry - min_tp1_dist
                tp2_min = entry - max_tp2_dist
                tp2_max = entry - min_tp2_dist
                tp3_min = entry - max_tp3_dist
                tp3_max = entry - min_tp3_dist
                
                # ═══════════════════════════════════════════════════════════
                # MODE-SPECIFIC TP SELECTION using priority-sorted candidates
                # Priority order depends on trading mode (see _get_mode_tp_priority)
                # ═══════════════════════════════════════════════════════════
                
                # Sort candidates by mode-specific priority (for SHORT, price < entry)
                sorted_candidates = self._sort_tp_candidates_by_mode_priority(
                    tp_candidates, 'SHORT', entry, min_price=entry
                )
                
                # TP1: First valid level in priority order within TP1 range
                for level in sorted_candidates:
                    if tp1_min <= level.price <= tp1_max:
                        reward = entry - level.price
                        rr = reward / risk if risk > 0 else 0
                        if rr >= 0.3:
                            # HYBRID MODE: Check ML proximity
                            if self.analysis_mode == 'hybrid' and ml_tp1_target:
                                proximity = abs(level.price - ml_tp1_target) / entry * 100
                                if proximity < 1.0:  # Within 1% of ML target
                                    tp1, tp1_level = level.price, level
                                    break
                                elif tp1 == 0:
                                    tp1, tp1_level = level.price, level
                            else:
                                tp1, tp1_level = level.price, level
                                break
                
                # TP2: Next valid level BEYOND TP1 (lower for SHORT) in priority order
                for level in sorted_candidates:
                    if level.price < (tp1 if tp1 > 0 else entry) and tp2_min <= level.price <= tp2_max:
                        tp2, tp2_level = level.price, level
                        break
                
                # TP3: Next valid level BEYOND TP2 (lower for SHORT) in priority order
                for level in sorted_candidates:
                    if level.price < (tp2 if tp2 > 0 else tp1 if tp1 > 0 else entry) and tp3_min <= level.price <= tp3_max:
                        tp3, tp3_level = level.price, level
                        break
                
                # ═══════════════════════════════════════════════════════════
                # ATR-BASED FALLBACK - Only if NO structure found
                # Uses ATR spacing (NOT percentage!) to maintain proper spacing
                # ═══════════════════════════════════════════════════════════
                if tp1 == 0:
                    if self.analysis_mode == 'hybrid' and ml_tp1_target:
                        tp1 = ml_tp1_target
                    else:
                        min_reward = risk * self.constraints.min_rr if risk > 0 else min_tp1_dist
                        tp1 = entry - max(min_reward, min_tp1_dist)
                        tp1 = max(tp1, tp1_min)
                    
                if tp2 == 0:
                    if self.analysis_mode == 'hybrid' and ml_tp2_target:
                        tp2 = ml_tp2_target
                    else:
                        # Use ATR-based spacing from TP1
                        tp2 = tp1 - min_spacing
                        tp2 = max(tp2, tp2_min)
                    
                if tp3 == 0:
                    if self.analysis_mode == 'hybrid' and ml_tp3_target:
                        tp3 = ml_tp3_target
                    else:
                        # Use ATR-based spacing from TP2
                        tp3 = tp2 - min_spacing
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
        # NOTE: Ordering enforcement moved to AFTER caps to prevent caps from
        # undoing the ordering fix. See "FINAL ORDERING ENFORCEMENT" below.
        # ═══════════════════════════════════════════════════════════════════
        
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
            else:
                tp1 = max(tp1, entry - max_tp1_dist)
                tp2 = max(tp2, entry - max_tp2_dist)
                tp3 = max(tp3, entry - max_tp3_dist)
        
        # ═══════════════════════════════════════════════════════════════════
        # FINAL ORDERING ENFORCEMENT - TP1 < TP2 < TP3 for LONG (MUST BE LAST!)
        # This runs AFTER all caps - ordering takes priority over caps
        # ═══════════════════════════════════════════════════════════════════
        min_spacing_pct = 0.002  # Minimum 0.2% between TPs
        if direction == 'LONG':
            # TP2 must be above TP1
            if tp2 <= tp1:
                tp2 = tp1 * (1 + min_spacing_pct)  # At least 0.2% above TP1
                tp2_level = None  # Clear level since we overrode
            # TP3 must be above TP2
            if tp3 <= tp2:
                tp3 = tp2 * (1 + min_spacing_pct)  # At least 0.2% above TP2
                tp3_level = None
        else:  # SHORT
            # TP2 must be below TP1
            if tp2 >= tp1:
                tp2 = tp1 * (1 - min_spacing_pct)  # At least 0.2% below TP1
                tp2_level = None
            # TP3 must be below TP2
            if tp3 >= tp2:
                tp3 = tp2 * (1 - min_spacing_pct)  # At least 0.2% below TP2
                tp3_level = None
        
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
        # FINAL CONSTRAINT ENFORCEMENT - Using CURRENT PRICE (what user sees!)
        # Entry might be limit entry, but user sees risk from current price
        # ═══════════════════════════════════════════════════════════════════
        current_price = self.current_price
        
        # Enforce SL constraint from CURRENT PRICE
        sl_from_current_pct = abs(current_price - sl) / current_price * 100 if current_price > 0 else 0
        if sl_from_current_pct > self.constraints.max_sl_pct:
            # SL is too far from current price - cap it
            max_sl_dist_current = current_price * (self.constraints.max_sl_pct / 100)
            if direction == 'LONG':
                sl = current_price - max_sl_dist_current
            else:
                sl = current_price + max_sl_dist_current
            sl_type = f"max_{self.constraints.max_sl_pct:.1f}%"
            sl_level = None
            sl_pct = self.constraints.max_sl_pct
        
        # Enforce TP1 minimum from CURRENT PRICE
        tp1_from_current_pct = abs(tp1 - current_price) / current_price * 100 if current_price > 0 and tp1 > 0 else 0
        if tp1_from_current_pct < self.constraints.min_tp1_pct:
            # TP1 is too close - expand it to minimum
            min_tp1_dist_current = current_price * (self.constraints.min_tp1_pct / 100)
            if direction == 'LONG':
                tp1 = current_price + min_tp1_dist_current
            else:
                tp1 = current_price - min_tp1_dist_current
            tp1_type_str = f"min_{self.constraints.min_tp1_pct:.1f}%"
            tp1_level = None
            
        # Recalculate risk for return value (from entry, not current_price)
        risk = abs(entry - sl) if sl > 0 else 0
        rr = abs(tp1 - entry) / risk if risk > 0 and tp1 > 0 else 0
        sl_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
        
        # ═══════════════════════════════════════════════════════════════════
        # BUILD TRADE SETUP with EDUCATIONAL DESCRIPTIONS
        # ═══════════════════════════════════════════════════════════════════
        current_price = self.current_price
        
        # TP1 DESCRIPTION
        if tp1_level:
            tp1_type_str = tp1_level.description
            tp1_pct = abs(tp1 - current_price) / current_price * 100 if current_price > 0 else 0
            tp1_desc = (
                f"📍 Source: {tp1_level.description}\n"
                f"✅ Structure-based target (+{tp1_pct:.1f}%)"
            )
        elif tp1 > 0:
            tp1_pct = abs(tp1 - current_price) / current_price * 100 if current_price > 0 else 0
            if tp1_pct <= self.constraints.min_tp1_pct * 1.1:
                tp1_type_str = f"min_{self.constraints.min_tp1_pct:.1f}%"
                tp1_desc = (
                    f"📍 Source: {self.trading_mode.value} minimum {self.constraints.min_tp1_pct:.1f}%\n"
                    f"⚠️ No OB/FVG found in {self.trading_mode.value} range\n"
                    f"💡 Using mode minimum target"
                )
            else:
                tp1_type_str = f"ATR_{tp1_pct:.1f}%"
                tp1_desc = (
                    f"📍 Source: ATR-based target (+{tp1_pct:.1f}%)\n"
                    f"⚠️ No structure found in valid range\n"
                    f"💡 Using ATR calculation as fallback"
                )
        else:
            tp1_type_str = ""
            tp1_desc = ""
        
        # TP2 DESCRIPTION
        if tp2_level:
            tp2_type_str = tp2_level.description
            tp2_pct = abs(tp2 - current_price) / current_price * 100 if current_price > 0 else 0
            tp2_desc = (
                f"📍 Source: {tp2_level.description}\n"
                f"✅ Structure-based target (+{tp2_pct:.1f}%)"
            )
        elif tp2 > 0:
            tp2_pct = abs(tp2 - current_price) / current_price * 100 if current_price > 0 else 0
            tp2_type_str = f"ATR_{tp2_pct:.1f}%"
            tp2_desc = (
                f"📍 Source: ATR-based extension (+{tp2_pct:.1f}%)\n"
                f"💡 Extended from TP1 using ATR spacing"
            )
        else:
            tp2_type_str = ""
            tp2_desc = ""
        
        # TP3 DESCRIPTION  
        if tp3_level:
            tp3_type_str = tp3_level.description
            tp3_pct = abs(tp3 - current_price) / current_price * 100 if current_price > 0 else 0
            tp3_desc = (
                f"📍 Source: {tp3_level.description}\n"
                f"✅ Structure-based target (+{tp3_pct:.1f}%)"
            )
        elif tp3 > 0:
            tp3_pct = abs(tp3 - current_price) / current_price * 100 if current_price > 0 else 0
            tp3_type_str = f"ATR_{tp3_pct:.1f}%"
            tp3_desc = (
                f"📍 Source: ATR-based extension (+{tp3_pct:.1f}%)\n"
                f"💡 Extended from TP2 using ATR spacing"
            )
        else:
            tp3_type_str = ""
            tp3_desc = ""
        
        # For Hybrid mode, indicate ML influence
        if self.analysis_mode == 'hybrid' and self.ml_targets:
            tp1_type_str = f"Hybrid: {tp1_type_str}"
            if tp1_desc:
                tp1_desc += f"\n🤖 ML-guided selection"
        
        # Build reasoning summary
        reasoning_parts = []
        if entry_level:
            reasoning_parts.append(f"Entry: {entry_level.description}")
        reasoning_parts.append(f"SL: {sl_type}")
        reasoning_parts.append(f"TP1: {tp1_type_str}")
        
        # ═══════════════════════════════════════════════════════════════════
        # ABSOLUTELY BULLETPROOF TP ORDERING
        # This is the FINAL CHECK before returning - NOTHING can bypass this
        # ═══════════════════════════════════════════════════════════════════
        atr = self.collector.atr if self.collector else self.current_price * 0.01
        min_tp_spacing = atr * 0.5  # 0.5 ATR minimum spacing between TPs
        
        # Determine direction from SL position (CANNOT be wrong)
        is_long = sl < entry
        
        # LONG TRADE: All TPs MUST be above entry, TP3 > TP2 > TP1
        if is_long:
            # Step 1: Force all TPs above entry
            if tp1 <= entry:
                tp1 = entry + (entry * 0.005)  # At least +0.5% above entry
            if tp2 <= entry:
                tp2 = entry + (entry * 0.01)   # At least +1.0% above entry
            if tp3 <= entry:
                tp3 = entry + (entry * 0.015)  # At least +1.5% above entry
            
            # Step 2: Force TP3 > TP2 > TP1 (ascending order)
            if tp2 <= tp1:
                tp2 = tp1 + min_tp_spacing
                tp2_type_str = "ATR-extension"
                tp2_level = None
                tp2_pct = abs(tp2 - entry) / entry * 100 if entry > 0 else 0
                tp2_desc = f"📍 Source: ATR-based extension (+{tp2_pct:.1f}%)\n💡 Extended from TP1 (ordering fix)"
            
            if tp3 <= tp2:
                tp3 = tp2 + min_tp_spacing
                tp3_type_str = "ATR-extension"
                tp3_level = None
                tp3_pct = abs(tp3 - entry) / entry * 100 if entry > 0 else 0
                tp3_desc = f"📍 Source: ATR-based extension (+{tp3_pct:.1f}%)\n💡 Extended from TP2 (ordering fix)"
        
        # SHORT TRADE: All TPs MUST be below entry, TP3 < TP2 < TP1
        else:
            # Step 1: Force all TPs below entry
            if tp1 >= entry:
                tp1 = entry - (entry * 0.005)  # At least -0.5% below entry
            if tp2 >= entry:
                tp2 = entry - (entry * 0.01)   # At least -1.0% below entry
            if tp3 >= entry:
                tp3 = entry - (entry * 0.015)  # At least -1.5% below entry
            
            # Step 2: Force TP3 < TP2 < TP1 (descending order)
            if tp2 >= tp1:
                tp2 = tp1 - min_tp_spacing
                tp2_type_str = "ATR-extension"
                tp2_level = None
                tp2_pct = abs(entry - tp2) / entry * 100 if entry > 0 else 0
                tp2_desc = f"📍 Source: ATR-based extension (+{tp2_pct:.1f}%)\n💡 Extended from TP1 (ordering fix)"
            
            if tp3 >= tp2:
                tp3 = tp2 - min_tp_spacing
                tp3_type_str = "ATR-extension"
                tp3_level = None
                tp3_pct = abs(entry - tp3) / entry * 100 if entry > 0 else 0
                tp3_desc = f"📍 Source: ATR-based extension (+{tp3_pct:.1f}%)\n💡 Extended from TP2 (ordering fix)"
        
        return TradeSetup(
            direction=final_direction,
            entry=entry,
            entry_type=entry_type,
            entry_level=entry_level,
            stop_loss=sl,
            sl_type=sl_type,
            sl_level=sl_level,
            sl_description=sl_description,
            tp1=tp1,
            tp1_type=tp1_type_str,
            tp1_level=tp1_level,
            tp1_description=tp1_desc,
            tp2=tp2,
            tp2_type=tp2_type_str,
            tp2_level=tp2_level,
            tp2_description=tp2_desc,
            tp3=tp3,
            tp3_type=tp3_type_str,
            tp3_level=tp3_level,
            tp3_description=tp3_desc,
            risk_reward=rr,
            risk_pct=sl_pct,
            rr_tp1=rr,
            confidence=self._calculate_confidence(entry_level, sl_level, tp1_level),
            reasoning=" | ".join(reasoning_parts),
            mode=self.mode,
            htf_aligned=not bool(htf_wait_reason),
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
    
    def _get_ml_optimized_buffer(self, atr: float, current_price: float) -> Tuple[float, str]:
        """
        Get ML-optimized anti-hunt buffer multiplier.
        
        Structure level is rules-based, but buffer is ML-optimized for:
        1. Less predictability (randomization component)
        2. Volatility adaptation
        3. Session-based adjustment (more hunts during NY/London)
        4. Mode-appropriate scaling
        
        Returns: (buffer_multiplier, source_description)
        """
        import random
        from datetime import datetime
        
        # ═══════════════════════════════════════════════════════════════
        # BASE BUFFER BY MODE
        # ═══════════════════════════════════════════════════════════════
        mode_base = {
            TradingMode.SCALP: 0.6,       # Base 0.6 ATR for scalp
            TradingMode.DAY_TRADE: 0.8,   # Base 0.8 ATR for day trade
            TradingMode.SWING: 1.2,       # Base 1.2 ATR for swing
            TradingMode.INVESTMENT: 1.8,  # Base 1.8 ATR for investment
        }
        base_buffer = mode_base.get(self.trading_mode, 0.8)
        
        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY ADJUSTMENT
        # High volatility = wider buffer needed
        # ═══════════════════════════════════════════════════════════════
        volatility_pct = (atr / current_price * 100) if current_price > 0 else 1.0
        
        # Classify volatility regime
        if volatility_pct > 3.0:
            vol_multiplier = 1.4  # High volatility - 40% wider
            vol_regime = "HIGH_VOL"
        elif volatility_pct > 1.5:
            vol_multiplier = 1.2  # Medium volatility - 20% wider
            vol_regime = "MED_VOL"
        elif volatility_pct < 0.5:
            vol_multiplier = 0.9  # Low volatility - can be tighter
            vol_regime = "LOW_VOL"
        else:
            vol_multiplier = 1.0  # Normal
            vol_regime = "NORMAL"
        
        # ═══════════════════════════════════════════════════════════════
        # SESSION ADJUSTMENT
        # NY/London sessions have more stop hunts
        # ═══════════════════════════════════════════════════════════════
        try:
            current_hour = datetime.utcnow().hour
            
            # NY session (13:00-21:00 UTC) - most manipulated
            if 13 <= current_hour < 21:
                session_multiplier = 1.25  # 25% wider during NY
                session_name = "NY_SESSION"
            # London session (7:00-15:00 UTC) - second most
            elif 7 <= current_hour < 15:
                session_multiplier = 1.15  # 15% wider during London
                session_name = "LONDON"
            # Asia session (0:00-8:00 UTC) - less manipulation
            elif 0 <= current_hour < 8:
                session_multiplier = 1.0
                session_name = "ASIA"
            else:
                session_multiplier = 1.05
                session_name = "TRANSITION"
        except:
            session_multiplier = 1.0
            session_name = "UNKNOWN"
        
        # ═══════════════════════════════════════════════════════════════
        # ML PREDICTION (if available from trained model)
        # ═══════════════════════════════════════════════════════════════
        ml_adjustment = 1.0
        ml_used = False
        
        if self.ml_targets and 'optimal_sl_pct' in self.ml_targets:
            # ML suggests optimal SL distance - convert to buffer hint
            ml_sl_pct = self.ml_targets.get('optimal_sl_pct', 0)
            if ml_sl_pct > 0:
                # If ML suggests wider SL, increase buffer
                expected_sl_pct = base_buffer * volatility_pct
                if ml_sl_pct > expected_sl_pct * 1.5:
                    ml_adjustment = 1.3  # ML says go wider
                    ml_used = True
                elif ml_sl_pct < expected_sl_pct * 0.7:
                    ml_adjustment = 0.85  # ML says can be tighter
                    ml_used = True
        
        # ═══════════════════════════════════════════════════════════════
        # RANDOMIZATION - Makes buffer less predictable
        # ±10% random variation to avoid clustered stops
        # ═══════════════════════════════════════════════════════════════
        random_factor = random.uniform(0.92, 1.08)  # ±8% random
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL CALCULATION
        # ═══════════════════════════════════════════════════════════════
        final_buffer = base_buffer * vol_multiplier * session_multiplier * ml_adjustment * random_factor
        
        # Clamp to reasonable range per mode
        mode_limits = {
            TradingMode.SCALP: (0.5, 1.5),       # 0.5-1.5 ATR for scalp
            TradingMode.DAY_TRADE: (0.6, 2.0),   # 0.6-2.0 ATR for day trade
            TradingMode.SWING: (0.8, 2.5),       # 0.8-2.5 ATR for swing
            TradingMode.INVESTMENT: (1.0, 3.0),  # 1.0-3.0 ATR for investment
        }
        min_buf, max_buf = mode_limits.get(self.trading_mode, (0.5, 2.0))
        final_buffer = max(min_buf, min(max_buf, final_buffer))
        
        # Build source description
        source_parts = [self.trading_mode.value]
        if vol_regime != "NORMAL":
            source_parts.append(vol_regime)
        if session_multiplier > 1.0:
            source_parts.append(session_name)
        if ml_used:
            source_parts.append("ML-adjusted")
        
        source_desc = " + ".join(source_parts)
        
        return final_buffer, source_desc
    
    def _get_mode_tp_priority(self, level: 'PriceLevel') -> int:
        """
        Get priority score for a level based on trading mode.
        LOWER score = HIGHER priority (used for sorting).
        
        MODE-SPECIFIC TP STRUCTURE PRIORITY:
        ═══════════════════════════════════════════════════════════════
        SCALP: Current OB > Current FVG > Current Swing
               NO Fib (too distant), NO HTF (too distant)
        
        DAY TRADE: Current OB > HTF OB > Current FVG > Swing > Fib
                   HTF provides context but current TF preferred
        
        SWING: HTF OB > HTF Swing > Fib > Current OB
               Fib is PRIMARY target method for swing trades
        
        INVESTMENT: HTF Swing > Fib > HTF OB > Current Swing
                    Fib is PRIMARY, long-term structure focus
        ═══════════════════════════════════════════════════════════════
        """
        level_type = level.level_type
        
        if self.trading_mode == TradingMode.SCALP:
            # SCALP: Current OB > FVG > Swing > HTF (HTF as backup)
            priority_map = {
                # Current OB - highest priority (1-2)
                LevelType.OB_BULLISH: 1,
                LevelType.OB_BEARISH: 2,
                # Current FVG - second priority (3-4)
                LevelType.FVG_BULLISH: 3,
                LevelType.FVG_BEARISH: 4,
                # Current Swing - third priority (5-6)
                LevelType.SWING_HIGH: 5,
                LevelType.SWING_LOW: 6,
                # HTF as backup (7-10)
                LevelType.HTF_OB_BULLISH: 7,
                LevelType.HTF_OB_BEARISH: 8,
                LevelType.HTF_SWING_HIGH: 9,
                LevelType.HTF_SWING_LOW: 10,
                # Fib - still excluded for scalp (99 = skip)
                LevelType.FIB_382: 99,
                LevelType.FIB_500: 99,
                LevelType.FIB_618: 99,
                LevelType.FIB_786: 99,
            }
        
        elif self.trading_mode == TradingMode.DAY_TRADE:
            # DAY TRADE: Current OB > HTF OB > FVG > Swing > Fib
            priority_map = {
                # Current OB - highest priority (1-2)
                LevelType.OB_BULLISH: 1,
                LevelType.OB_BEARISH: 2,
                # HTF OB - second priority (3-4)
                LevelType.HTF_OB_BULLISH: 3,
                LevelType.HTF_OB_BEARISH: 4,
                # Current FVG - third priority (5-6)
                LevelType.FVG_BULLISH: 5,
                LevelType.FVG_BEARISH: 6,
                # Swing levels - fourth priority (7-10)
                LevelType.SWING_HIGH: 7,
                LevelType.SWING_LOW: 8,
                LevelType.HTF_SWING_HIGH: 9,
                LevelType.HTF_SWING_LOW: 10,
                # Fib - lowest priority for day trade (11-14)
                LevelType.FIB_618: 11,  # 61.8% most common
                LevelType.FIB_500: 12,
                LevelType.FIB_382: 13,
                LevelType.FIB_786: 14,
            }
        
        elif self.trading_mode == TradingMode.SWING:
            # SWING: HTF OB > HTF Swing > Fib > Current OB
            # Fib is PRIMARY target method for swing!
            priority_map = {
                # HTF OB - highest priority (1-2)
                LevelType.HTF_OB_BULLISH: 1,
                LevelType.HTF_OB_BEARISH: 2,
                # HTF Swing - second priority (3-4)
                LevelType.HTF_SWING_HIGH: 3,
                LevelType.HTF_SWING_LOW: 4,
                # Fib - third priority but PRIMARY method (5-8)
                LevelType.FIB_618: 5,   # Golden ratio - most important
                LevelType.FIB_500: 6,   # 50% retracement
                LevelType.FIB_382: 7,
                LevelType.FIB_786: 8,
                # Current OB - fourth priority (9-10)
                LevelType.OB_BULLISH: 9,
                LevelType.OB_BEARISH: 10,
                # Current FVG/Swing - lowest (11-14)
                LevelType.FVG_BULLISH: 11,
                LevelType.FVG_BEARISH: 12,
                LevelType.SWING_HIGH: 13,
                LevelType.SWING_LOW: 14,
            }
        
        else:  # INVESTMENT
            # INVESTMENT: HTF Swing > Fib > HTF OB > Current Swing
            # Fib is PRIMARY for long-term targets
            priority_map = {
                # HTF Swing - highest priority (1-2)
                LevelType.HTF_SWING_HIGH: 1,
                LevelType.HTF_SWING_LOW: 2,
                # Fib - second priority but PRIMARY method (3-6)
                LevelType.FIB_618: 3,   # Golden ratio
                LevelType.FIB_500: 4,
                LevelType.FIB_786: 5,   # Deep retracement
                LevelType.FIB_382: 6,
                # HTF OB - third priority (7-8)
                LevelType.HTF_OB_BULLISH: 7,
                LevelType.HTF_OB_BEARISH: 8,
                # Current Swing - fourth priority (9-10)
                LevelType.SWING_HIGH: 9,
                LevelType.SWING_LOW: 10,
                # Current OB/FVG - lowest (11-14)
                LevelType.OB_BULLISH: 11,
                LevelType.OB_BEARISH: 12,
                LevelType.FVG_BULLISH: 13,
                LevelType.FVG_BEARISH: 14,
            }
        
        return priority_map.get(level_type, 50)  # Default to low priority
    
    def _sort_tp_candidates_by_mode_priority(
        self, 
        candidates: List['PriceLevel'], 
        direction: str,
        entry: float,
        min_price: float = 0
    ) -> List['PriceLevel']:
        """
        Sort TP candidates by mode-specific structure priority.
        
        Returns levels sorted by:
        1. Mode-specific priority (OB vs FVG vs Fib etc.)
        2. Distance from entry (nearest first within same priority)
        
        Also filters out levels that don't meet min_price requirement.
        """
        if not candidates:
            return []
        
        # Filter by direction
        if direction == 'LONG':
            # For LONG: price must be above entry
            valid = [l for l in candidates if l.price > entry]
        else:  # SHORT
            # For SHORT: price must be below entry
            valid = [l for l in candidates if l.price < entry]
        
        if not valid:
            return []
        
        # Sort by: (1) mode priority, (2) distance from entry (nearest first)
        def sort_key(level):
            priority = self._get_mode_tp_priority(level)
            distance = abs(level.price - entry)
            return (priority, distance)
        
        valid.sort(key=sort_key)
        
        # Filter out levels with priority 99 (excluded for this mode)
        valid = [l for l in valid if self._get_mode_tp_priority(l) < 99]
        
        return valid
    
    def _find_stop_loss(
        self,
        direction: str,
        entry: float,
        support_levels: List[PriceLevel]
    ) -> Tuple[float, str, Optional[PriceLevel], str]:
        """
        Find optimal stop loss - ALWAYS ANTI-HUNT BASED
        
        SL is ALWAYS structure-based with anti-hunt buffer.
        ML optimizes the BUFFER MULTIPLIER only (not the structure level).
        
        Priority:
        1️⃣ Anti-Hunt SL at SIGNIFICANT structure (OB > FVG > Fib > Swing)
        2️⃣ Mode minimum (if no valid structure found)
        3️⃣ Enforce min/max constraints
        
        Returns: (sl_price, sl_type, sl_level, sl_description)
        """
        current_price = self.current_price
        atr = self.collector.atr if self.collector else current_price * 0.01
        
        # ═══════════════════════════════════════════════════════════════
        # CALCULATE EFFECTIVE MINIMUM (considers ATR for volatility)
        # ═══════════════════════════════════════════════════════════════
        atr_based_min = atr * self.constraints.min_sl_atr / current_price * 100 if current_price > 0 else 0
        effective_min_sl_pct = max(atr_based_min, self.constraints.min_sl_pct)
        
        # ═══════════════════════════════════════════════════════════════
        # ML-OPTIMIZED ANTI-HUNT BUFFER
        # Structure is rules-based, buffer is ML-optimized for unpredictability
        # ═══════════════════════════════════════════════════════════════
        buffer_multiplier, buffer_source = self._get_ml_optimized_buffer(atr, current_price)
        anti_hunt_buffer = atr * buffer_multiplier
        
        # ═══════════════════════════════════════════════════════════════
        # STRUCTURE PRIORITY - Most significant levels for stop hunts
        # Institutions defend these levels, so SL beyond them is safer
        # ═══════════════════════════════════════════════════════════════
        level_priority = {
            # HTF levels - highest priority (institutions watch these)
            LevelType.HTF_OB_BULLISH: 1,
            LevelType.HTF_OB_BEARISH: 1,
            LevelType.HTF_SWING_HIGH: 2,
            LevelType.HTF_SWING_LOW: 2,
            
            # Current TF Order Blocks - high priority
            LevelType.OB_BULLISH: 3,
            LevelType.OB_BEARISH: 3,
            
            # Fibonacci levels - institutional retracement
            LevelType.FIB_618: 4,  # Golden ratio - most important
            LevelType.FIB_500: 5,  # Equilibrium
            LevelType.FIB_786: 6,  # Deep retracement
            LevelType.FIB_382: 7,  # Shallow retracement
            
            # FVG - imbalance zones
            LevelType.FVG_BULLISH: 8,
            LevelType.FVG_BEARISH: 8,
            
            # Volume profile
            LevelType.POC: 9,
            LevelType.VAH: 10,
            LevelType.VAL: 10,
            LevelType.VWAP: 11,
            
            # Previous day/week levels
            LevelType.PREV_DAY_HIGH: 12,
            LevelType.PREV_DAY_LOW: 12,
            LevelType.PREV_WEEK_HIGH: 13,
            LevelType.PREV_WEEK_LOW: 13,
            
            # Swing levels - lowest priority (often just noise)
            LevelType.SWING_HIGH: 20,
            LevelType.SWING_LOW: 20,
            
            # Session levels
            LevelType.SESSION_HIGH: 15,
            LevelType.SESSION_LOW: 15,
            LevelType.ASIA_HIGH: 16,
            LevelType.ASIA_LOW: 16,
            
            # Other
            LevelType.LIQUIDITY_HIGH: 14,
            LevelType.LIQUIDITY_LOW: 14,
            LevelType.ROUND_NUMBER: 18,
            LevelType.BOS: 19,
            LevelType.CHOCH: 19,
            LevelType.PREV_DAY_CLOSE: 17,
        }
        
        def get_priority(level: PriceLevel) -> int:
            """Get priority score for a level (lower = more significant)"""
            return level_priority.get(level.level_type, 99)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: FIND ALL VALID STRUCTURE LEVELS
        # Filter to levels within valid SL range, then sort by significance
        # ═══════════════════════════════════════════════════════════════
        anti_hunt_sl = None
        anti_hunt_pct = None
        anti_hunt_level = None
        anti_hunt_desc = ""
        
        if direction == 'LONG':
            # Get all levels below entry
            structure_levels = [l for l in support_levels if l.price < entry]
            
            # Calculate SL for each level and filter to valid range
            valid_levels = []
            for level in structure_levels:
                test_sl = level.price - anti_hunt_buffer
                test_sl = self._make_ugly_number(test_sl, direction)
                test_pct = abs(current_price - test_sl) / current_price * 100
                
                # Must be within mode constraints (min to max)
                if effective_min_sl_pct <= test_pct <= self.constraints.max_sl_pct:
                    valid_levels.append((level, test_sl, test_pct))
            
            # Sort by priority (most significant first)
            valid_levels.sort(key=lambda x: get_priority(x[0]))
            
            # Use most significant level
            if valid_levels:
                best_level, best_sl, best_pct = valid_levels[0]
                anti_hunt_sl = best_sl
                anti_hunt_pct = best_pct
                from copy import copy
                anti_hunt_level = copy(best_level)
                anti_hunt_level.description = f"Anti-hunt below {best_level.description}"
                anti_hunt_desc = (
                    f"📍 Source: Anti-hunt below {best_level.description}\n"
                    f"💡 {buffer_multiplier:.2f}× ATR buffer ({buffer_source})\n"
                    f"✅ Risk: {anti_hunt_pct:.1f}% - Significant structure"
                )
                
        else:  # SHORT
            # Get all levels above entry
            structure_levels = [l for l in support_levels if l.price > entry]
            
            # Calculate SL for each level and filter to valid range
            valid_levels = []
            for level in structure_levels:
                test_sl = level.price + anti_hunt_buffer
                test_sl = self._make_ugly_number(test_sl, direction)
                test_pct = abs(test_sl - current_price) / current_price * 100
                
                # Must be within mode constraints
                if effective_min_sl_pct <= test_pct <= self.constraints.max_sl_pct:
                    valid_levels.append((level, test_sl, test_pct))
            
            # Sort by priority (most significant first)
            valid_levels.sort(key=lambda x: get_priority(x[0]))
            
            # Use most significant level
            if valid_levels:
                best_level, best_sl, best_pct = valid_levels[0]
                anti_hunt_sl = best_sl
                anti_hunt_pct = best_pct
                from copy import copy
                anti_hunt_level = copy(best_level)
                anti_hunt_level.description = f"Anti-hunt above {best_level.description}"
                anti_hunt_desc = (
                    f"📍 Source: Anti-hunt above {best_level.description}\n"
                    f"💡 {buffer_multiplier:.2f}× ATR buffer ({buffer_source})\n"
                    f"✅ Risk: {anti_hunt_pct:.1f}% - Significant structure"
                )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: SELECT SL - Significant Structure or Mode Minimum
        # ═══════════════════════════════════════════════════════════════
        final_sl = None
        final_type = ""
        final_level = None
        final_desc = ""
        
        # Case 1: Found significant structure within valid range
        if anti_hunt_sl and anti_hunt_pct:
            final_sl = anti_hunt_sl
            final_type = 'anti_hunt'
            final_level = anti_hunt_level
            final_desc = anti_hunt_desc
        
        # Case 2: No structure found → use mode minimum
        else:
            if direction == 'LONG':
                final_sl = current_price * (1 - effective_min_sl_pct / 100)
            else:
                final_sl = current_price * (1 + effective_min_sl_pct / 100)
            final_sl = self._make_ugly_number(final_sl, direction)
            final_type = f'min_{effective_min_sl_pct:.1f}%'
            final_level = None
            final_desc = (
                f"📍 Source: {self.trading_mode.value} minimum {effective_min_sl_pct:.1f}%\n"
                f"⚠️ No structure found for anti-hunt\n"
                f"💡 Using mode minimum as default"
            )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: ENFORCE MODE CONSTRAINTS (min/max)
        # ═══════════════════════════════════════════════════════════════
        final_pct = abs(current_price - final_sl) / current_price * 100 if current_price > 0 else 0
        
        # Check MAXIMUM - cap if too risky
        if final_pct > self.constraints.max_sl_pct:
            if direction == 'LONG':
                final_sl = current_price * (1 - self.constraints.max_sl_pct / 100)
            else:
                final_sl = current_price * (1 + self.constraints.max_sl_pct / 100)
            final_sl = self._make_ugly_number(final_sl, direction)
            final_type = f'max_{self.constraints.max_sl_pct}%'
            final_level = None
            final_desc = (
                f"📍 Source: {self.trading_mode.value} max {self.constraints.max_sl_pct}% cap\n"
                f"⚠️ Capped: Structure SL was {final_pct:.1f}% - too risky\n"
                f"💡 Using mode maximum for risk management"
            )
        
        # Check MINIMUM - expand if too tight
        elif final_pct < effective_min_sl_pct:
            original_pct = final_pct
            if direction == 'LONG':
                final_sl = current_price * (1 - effective_min_sl_pct / 100)
            else:
                final_sl = current_price * (1 + effective_min_sl_pct / 100)
            final_sl = self._make_ugly_number(final_sl, direction)
            final_type = f'min_{effective_min_sl_pct:.1f}%'
            final_level = None
            final_desc = (
                f"📍 Source: {self.trading_mode.value} minimum {effective_min_sl_pct:.1f}%\n"
                f"⚠️ Expanded: Original {original_pct:.1f}% was too tight\n"
                f"💡 Minimum ensures room for market noise"
            )
        
        return final_sl, final_type, final_level, final_desc
    
    def _find_tp(
        self,
        direction: str,
        entry: float,
        target_levels: List[PriceLevel],
        tp_number: int,
        previous_tp: float = 0  # NEW: Pass previous TP to ensure ordering
    ) -> Tuple[float, Optional[PriceLevel]]:
        """
        Find take profit level from structure (OB > FVG > Fib > Swing).
        
        TP1, TP2, TP3 are assigned to successive target levels.
        CRITICAL: Each TP must be BEYOND the previous one.
        If not enough structure levels, extend using ATR.
        
        Args:
            direction: 'LONG' or 'SHORT'
            entry: Entry price
            target_levels: List of structure levels (OBs, FVGs, Fibs, Swings)
            tp_number: 1, 2, or 3
            previous_tp: Previous TP price (to ensure TP2 > TP1, TP3 > TP2)
        """
        atr = self.collector.atr if self.collector else entry * 0.01
        
        if not target_levels:
            # No structure - use ATR extension from entry or previous TP
            base_price = previous_tp if previous_tp > 0 else entry
            extension = atr * tp_number * 1.5  # 1.5, 3.0, 4.5 ATR
            if direction == 'LONG':
                return base_price + extension, None
            else:
                return base_price - extension, None
        
        # Filter levels that are valid targets (beyond entry AND beyond previous TP)
        if direction == 'LONG':
            min_price = max(entry, previous_tp) if previous_tp > 0 else entry
            valid_targets = [l for l in target_levels if l.price > min_price]
            # Sort by price ascending (nearest first for LONG)
            valid_targets.sort(key=lambda x: x.price)
        else:  # SHORT
            max_price = min(entry, previous_tp) if previous_tp > 0 else entry
            valid_targets = [l for l in target_levels if l.price < max_price]
            # Sort by price descending (nearest first for SHORT)
            valid_targets.sort(key=lambda x: x.price, reverse=True)
        
        if not valid_targets:
            # No valid structure beyond previous TP - use ATR extension
            base_price = previous_tp if previous_tp > 0 else entry
            extension = atr * 1.5  # Extend by 1.5 ATR from previous
            if direction == 'LONG':
                return base_price + extension, None
            else:
                return base_price - extension, None
        
        # Return the nearest valid target (first after sorting)
        # Each call gets the NEXT level beyond previous_tp
        return valid_targets[0].price, valid_targets[0]
    
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
    ml_prediction: Optional[Dict] = None,  # ML predicted levels {'optimal_tp1_pct', 'optimal_sl_pct', etc.}
    analysis_mode: str = 'hybrid',  # 'rules', 'ml', 'hybrid'
) -> TradeSetup:
    """
    Main interface - get optimized trade setup.
    
    MASTER FUNCTION for trade levels - used by Scanner AND Single Analysis.
    Blends structure-based levels with ML predictions based on analysis_mode.
    
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
        ml_prediction: ML prediction dict with optimal_tp1_pct, optimal_sl_pct (optional)
        analysis_mode: 'rules' (structure only), 'ml' (ML primary), 'hybrid' (blend both)
        
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
    
    # Optimize with winning strategy priority AND ML blending
    optimizer = TradeOptimizer(
        collector=collector,
        mode=mode,
        htf_bias=htf_bias,
        winning_strategy=winning_strategy,
        ml_prediction=ml_prediction,
        analysis_mode=analysis_mode,
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
