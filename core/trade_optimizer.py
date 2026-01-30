"""
Trade Optimizer - MIP-Based Optimization using PuLP
====================================================

REPLACES the old if-else cascade optimizer with true MIP optimization.

Key Changes:
- TPs selected via Mixed Integer Programming (maximizes structure score)
- SL is ALWAYS anti-hunt (structure + ATR buffer + ugly number)
- No hard min/max filters - soft scoring penalties instead
- Mode-appropriate distances via Gaussian scoring

Install: pip install pulp --break-system-packages
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import math
import random

# Try to import PuLP
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("⚠️ PuLP not installed. Run: pip install pulp --break-system-packages")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TradingMode(Enum):
    SCALP = "scalp"
    DAY_TRADE = "day_trade"
    SWING = "swing"
    INVESTMENT = "investment"


class LevelType(Enum):
    """Types of price levels"""
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
    POC = "poc"
    VWAP = "vwap"
    VAH = "vah"
    VAL = "val"
    
    # Fibonacci
    FIB_382 = "fib_382"
    FIB_500 = "fib_500"
    FIB_618 = "fib_618"
    FIB_786 = "fib_786"
    FIB_EXTENSION = "fib_extension"  # For 127.2%, 161.8%, 200% extensions
    
    # Liquidity
    LIQUIDITY_HIGH = "liquidity_high"
    LIQUIDITY_LOW = "liquidity_low"
    
    # Historical
    PREV_DAY_HIGH = "prev_day_high"
    PREV_DAY_LOW = "prev_day_low"
    PREV_DAY_CLOSE = "prev_day_close"
    PREV_WEEK_HIGH = "prev_week_high"
    PREV_WEEK_LOW = "prev_week_low"
    
    # Session
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    ASIA_HIGH = "asia_high"
    ASIA_LOW = "asia_low"
    
    # Other
    ROUND_NUMBER = "round_number"
    ATR_BASED = "atr_based"


class LevelRole(Enum):
    """How a level can be used"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    BOTH = "both"


@dataclass
class PriceLevel:
    """A price level with metadata"""
    price: float
    level_type: LevelType
    role: LevelRole = LevelRole.BOTH
    source_tf: str = ""
    confidence: float = 1.0
    touches: int = 1
    description: str = ""
    
    def __hash__(self):
        return hash((self.price, self.level_type.value))
    
    def __lt__(self, other):
        return self.price < other.price


@dataclass
class TradeSetup:
    """Optimized trade setup - COMPATIBLE with old interface"""
    direction: str  # 'LONG', 'SHORT', 'WAIT'
    entry: float = 0
    entry_type: str = "market"
    entry_level: Optional[PriceLevel] = None
    
    stop_loss: float = 0
    sl_type: str = ""
    sl_level: Optional[PriceLevel] = None
    sl_description: str = ""
    
    tp1: float = 0
    tp1_type: str = ""
    tp1_level: Optional[PriceLevel] = None
    tp1_description: str = ""
    
    tp2: float = 0
    tp2_type: str = ""
    tp2_level: Optional[PriceLevel] = None
    tp2_description: str = ""
    
    tp3: float = 0
    tp3_type: str = ""
    tp3_level: Optional[PriceLevel] = None
    tp3_description: str = ""
    
    risk_reward: float = 0
    risk_pct: float = 0
    rr_tp1: float = 0
    confidence: float = 0
    reasoning: str = ""
    
    mode: Optional[TradingMode] = None
    htf_aligned: bool = True
    wait_reason: str = ""
    
    # Additional for compatibility
    total_score: float = 0
    
    @property
    def sl(self) -> float:
        return self.stop_loss


@dataclass
class ModeConstraints:
    """Constraints per trading mode"""
    max_sl_pct: float
    min_sl_pct: float
    min_rr: float
    ideal_tp1_pct: float
    ideal_tp2_pct: float
    ideal_tp3_pct: float
    max_tp1_pct: float
    max_tp2_pct: float
    max_tp3_pct: float


@dataclass
class EntryConstraints:
    """Constraints for limit entry optimization per trading mode"""
    min_entry_dist_pct: float   # Minimum distance below current (too close = no benefit)
    ideal_entry_dist_pct: float # Ideal distance for best R:R
    max_entry_dist_pct: float   # Maximum realistic pullback to wait for
    max_wait_candles: int       # How long to wait for limit fill


@dataclass
class LimitEntryResult:
    """Result from limit entry optimization"""
    price: float
    level_type: str
    source: str                 # 'OB', 'FVG', 'SWING', 'ATR'
    zone_top: float = 0         # Top of the zone (for validation)
    zone_bottom: float = 0      # Bottom of the zone
    distance_pct: float = 0     # Distance from current price
    rr_improvement: float = 0   # R:R improvement vs market entry
    score: float = 0            # Optimization score
    recency_score: float = 0    # How fresh is this level (0-100)
    description: str = ""
    reason: str = ""
    is_valid: bool = True
    warning: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODE_CONSTRAINTS = {
    TradingMode.SCALP: ModeConstraints(
        max_sl_pct=1.0, min_sl_pct=0.3, min_rr=1.5,
        ideal_tp1_pct=0.8, ideal_tp2_pct=1.2, ideal_tp3_pct=1.8,
        max_tp1_pct=2.0, max_tp2_pct=3.0, max_tp3_pct=4.0,
    ),
    TradingMode.DAY_TRADE: ModeConstraints(
        max_sl_pct=2.5, min_sl_pct=0.8, min_rr=1.5,
        ideal_tp1_pct=2.0, ideal_tp2_pct=3.5, ideal_tp3_pct=5.0,
        max_tp1_pct=5.0, max_tp2_pct=8.0, max_tp3_pct=12.0,
    ),
    TradingMode.SWING: ModeConstraints(
        max_sl_pct=8.0, min_sl_pct=2.0, min_rr=1.5,
        ideal_tp1_pct=5.0, ideal_tp2_pct=8.0, ideal_tp3_pct=12.0,
        max_tp1_pct=15.0, max_tp2_pct=25.0, max_tp3_pct=40.0,
    ),
    TradingMode.INVESTMENT: ModeConstraints(
        max_sl_pct=15.0, min_sl_pct=5.0, min_rr=1.5,
        ideal_tp1_pct=15.0, ideal_tp2_pct=25.0, ideal_tp3_pct=40.0,
        max_tp1_pct=50.0, max_tp2_pct=80.0, max_tp3_pct=120.0,
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

# Entry constraints per mode - how far to wait for limit entry
ENTRY_CONSTRAINTS = {
    TradingMode.SCALP: EntryConstraints(
        min_entry_dist_pct=0.2,    # At least 0.2% below to be worth it
        ideal_entry_dist_pct=0.5,  # Sweet spot for scalp
        max_entry_dist_pct=1.5,    # Don't wait for more than 1.5%
        max_wait_candles=10,       # Quick fills expected
    ),
    TradingMode.DAY_TRADE: EntryConstraints(
        min_entry_dist_pct=0.3,
        ideal_entry_dist_pct=1.0,
        max_entry_dist_pct=3.5,
        max_wait_candles=20,
    ),
    TradingMode.SWING: EntryConstraints(
        min_entry_dist_pct=0.5,
        ideal_entry_dist_pct=2.0,
        max_entry_dist_pct=8.0,
        max_wait_candles=50,
    ),
    TradingMode.INVESTMENT: EntryConstraints(
        min_entry_dist_pct=1.0,
        ideal_entry_dist_pct=5.0,
        max_entry_dist_pct=15.0,
        max_wait_candles=100,
    ),
}

# Entry level scoring (different from TP scoring - recency matters more)
ENTRY_LEVEL_SCORES = {
    # FRESH OBs from impulsive moves - HIGHEST priority
    'FRESH_OB': 120,           # OB created in last 10 candles
    'FRESH_FVG': 100,          # FVG from recent impulse
    
    # Current TF OBs - PRIMARY entry points
    LevelType.OB_BULLISH: 100,  # OB top is optimal LONG entry
    LevelType.OB_BEARISH: 100,  # OB bottom is optimal SHORT entry
    
    # FVG equilibrium - good but less precise than OB
    'FVG_EQUILIBRIUM': 85,
    
    # Current TF FVGs
    LevelType.FVG_BULLISH: 80,
    LevelType.FVG_BEARISH: 80,
    
    # HTF levels (good but may be far)
    LevelType.HTF_OB_BULLISH: 90,
    LevelType.HTF_OB_BEARISH: 90,
    
    # Swings (less precise)
    LevelType.SWING_LOW: 60,
    LevelType.SWING_HIGH: 60,
    
    # Volume levels
    LevelType.POC: 55,
    LevelType.VAL: 50,
    LevelType.VAH: 50,
    
    # Fallbacks
    LevelType.FIB_618: 30,
    LevelType.FIB_500: 25,
    LevelType.ROUND_NUMBER: 15,
    LevelType.ATR_BASED: 10,
}

# Structure quality scores (higher = better)
STRUCTURE_SCORES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 1: SMC STRUCTURE (Always preferred for TPs)
    # These are REAL institutional levels - always use if available
    # ═══════════════════════════════════════════════════════════════════════════
    LevelType.OB_BULLISH: 100,
    LevelType.OB_BEARISH: 100,
    LevelType.HTF_OB_BULLISH: 95,
    LevelType.HTF_OB_BEARISH: 95,
    LevelType.FVG_BULLISH: 90,
    LevelType.FVG_BEARISH: 90,
    LevelType.HTF_SWING_HIGH: 85,
    LevelType.HTF_SWING_LOW: 85,
    LevelType.SWING_HIGH: 80,
    LevelType.SWING_LOW: 80,
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 2: VOLUME STRUCTURE (Good secondary targets)
    # ═══════════════════════════════════════════════════════════════════════════
    LevelType.POC: 70,
    LevelType.VWAP: 65,
    LevelType.VAH: 60,
    LevelType.VAL: 60,
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIER 3: FALLBACK LEVELS (Only when NO structure available)
    # Low scores ensure they NEVER beat real structure
    # ═══════════════════════════════════════════════════════════════════════════
    LevelType.FIB_EXTENSION: 25,  # Reduced - only use if no OB/FVG/Swing
    LevelType.FIB_618: 20,
    LevelType.FIB_500: 18,
    LevelType.FIB_382: 16,
    LevelType.FIB_786: 14,
    LevelType.LIQUIDITY_HIGH: 20,
    LevelType.LIQUIDITY_LOW: 20,
    LevelType.PREV_DAY_HIGH: 15,
    LevelType.PREV_DAY_LOW: 15,
    LevelType.ROUND_NUMBER: 10,   # Very low - psychological only, not structure
    LevelType.ATR_BASED: 5,       # Last resort
}

# Define which levels are "real structure" vs fallback
SMC_STRUCTURE_TYPES = {
    LevelType.OB_BULLISH, LevelType.OB_BEARISH,
    LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
    LevelType.FVG_BULLISH, LevelType.FVG_BEARISH,
    LevelType.SWING_HIGH, LevelType.SWING_LOW,
    LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW,
    LevelType.POC, LevelType.VWAP, LevelType.VAH, LevelType.VAL,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL COLLECTOR - Gathers all structure levels
# ═══════════════════════════════════════════════════════════════════════════════

class LevelCollector:
    """Collects price levels from various sources"""
    
    def __init__(self, df: pd.DataFrame, current_price: float, timeframe: str):
        self.df = df
        self.current_price = current_price
        self.timeframe = timeframe
        self.levels: List[PriceLevel] = []
        
        # Calculate ATR
        if len(df) >= 14:
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            self.atr = tr.rolling(14).mean().iloc[-1]
        else:
            self.atr = current_price * 0.02
    
    def collect_from_smc(self, smc_data: Dict) -> None:
        """Collect levels from SMC data"""
        if not smc_data:
            return
        
        # SMART UNWRAP: Handle both raw detect_smc output and pre-unwrapped format
        # Raw format has 'order_blocks' key, unwrapped has 'bearish_obs' directly
        if 'order_blocks' in smc_data and isinstance(smc_data['order_blocks'], dict):
            # Raw format from detect_smc - unwrap it
            ob_data = smc_data['order_blocks']
            smc_data = {
                'bearish_obs': ob_data.get('bearish_obs', []),
                'bullish_obs': ob_data.get('bullish_obs', []),
                'bearish_ob_top': ob_data.get('bearish_ob_top', 0),
                'bearish_ob_bottom': ob_data.get('bearish_ob_bottom', 0),
                'bullish_ob_top': ob_data.get('bullish_ob_top', 0),
                'bullish_ob_bottom': ob_data.get('bullish_ob_bottom', 0),
                'swing_high': smc_data.get('structure', {}).get('last_swing_high', 0),
                'swing_low': smc_data.get('structure', {}).get('last_swing_low', 0),
            }
        
        # Bearish OBs (resistance)
        for ob in smc_data.get('bearish_obs', []):
            if ob.get('bottom', 0) > 0:
                self.levels.append(PriceLevel(
                    price=ob['bottom'],
                    level_type=LevelType.OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    confidence=0.9,
                    description=f"Bearish OB ({self.timeframe})"
                ))
            if ob.get('top', 0) > 0:
                self.levels.append(PriceLevel(
                    price=ob['top'],
                    level_type=LevelType.OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    confidence=0.85,
                    description=f"Bearish OB top ({self.timeframe})"
                ))
        
        # Fallback single bearish OB
        if not smc_data.get('bearish_obs'):
            if smc_data.get('bearish_ob_bottom', 0) > 0:
                self.levels.append(PriceLevel(
                    price=smc_data['bearish_ob_bottom'],
                    level_type=LevelType.OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    description=f"Bearish OB ({self.timeframe})"
                ))
            if smc_data.get('bearish_ob_top', 0) > 0:
                self.levels.append(PriceLevel(
                    price=smc_data['bearish_ob_top'],
                    level_type=LevelType.OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    description=f"Bearish OB top ({self.timeframe})"
                ))
        
        # Bullish OBs (support)
        for ob in smc_data.get('bullish_obs', []):
            if ob.get('top', 0) > 0:
                self.levels.append(PriceLevel(
                    price=ob['top'],
                    level_type=LevelType.OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    confidence=0.9,
                    description=f"Bullish OB ({self.timeframe})"
                ))
            if ob.get('bottom', 0) > 0:
                self.levels.append(PriceLevel(
                    price=ob['bottom'],
                    level_type=LevelType.OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    confidence=0.85,
                    description=f"Bullish OB bottom ({self.timeframe})"
                ))
        
        # Fallback single bullish OB
        if not smc_data.get('bullish_obs'):
            if smc_data.get('bullish_ob_top', 0) > 0:
                self.levels.append(PriceLevel(
                    price=smc_data['bullish_ob_top'],
                    level_type=LevelType.OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    description=f"Bullish OB ({self.timeframe})"
                ))
            if smc_data.get('bullish_ob_bottom', 0) > 0:
                self.levels.append(PriceLevel(
                    price=smc_data['bullish_ob_bottom'],
                    level_type=LevelType.OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    description=f"Bullish OB bottom ({self.timeframe})"
                ))
        
        # FVGs
        for fvg in smc_data.get('bearish_fvgs', []):
            if fvg.get('bottom', 0) > 0:
                self.levels.append(PriceLevel(
                    price=fvg['bottom'],
                    level_type=LevelType.FVG_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf=self.timeframe,
                    description=f"Bearish FVG ({self.timeframe})"
                ))
        
        for fvg in smc_data.get('bullish_fvgs', []):
            if fvg.get('top', 0) > 0:
                self.levels.append(PriceLevel(
                    price=fvg['top'],
                    level_type=LevelType.FVG_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf=self.timeframe,
                    description=f"Bullish FVG ({self.timeframe})"
                ))
        
        # Single FVG fallback
        if smc_data.get('fvg_bearish_bottom', 0) > 0:
            self.levels.append(PriceLevel(
                price=smc_data['fvg_bearish_bottom'],
                level_type=LevelType.FVG_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf=self.timeframe,
                description=f"Bearish FVG ({self.timeframe})"
            ))
        if smc_data.get('fvg_bullish_top', 0) > 0:
            self.levels.append(PriceLevel(
                price=smc_data['fvg_bullish_top'],
                level_type=LevelType.FVG_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf=self.timeframe,
                description=f"Bullish FVG ({self.timeframe})"
            ))
        
        # Swings
        if smc_data.get('swing_high', 0) > 0:
            self.levels.append(PriceLevel(
                price=smc_data['swing_high'],
                level_type=LevelType.SWING_HIGH,
                role=LevelRole.RESISTANCE,
                source_tf=self.timeframe,
                description=f"Swing High ({self.timeframe})"
            ))
        if smc_data.get('swing_low', 0) > 0:
            self.levels.append(PriceLevel(
                price=smc_data['swing_low'],
                level_type=LevelType.SWING_LOW,
                role=LevelRole.SUPPORT,
                source_tf=self.timeframe,
                description=f"Swing Low ({self.timeframe})"
            ))
    
    def collect_from_htf(self, htf_data: Dict) -> None:
        """Collect levels from HTF data"""
        if not htf_data:
            return
        
        # HTF OBs
        for ob in htf_data.get('bearish_obs', []):
            if ob.get('bottom', 0) > 0:
                self.levels.append(PriceLevel(
                    price=ob['bottom'],
                    level_type=LevelType.HTF_OB_BEARISH,
                    role=LevelRole.RESISTANCE,
                    source_tf='HTF',
                    confidence=0.95,
                    description="HTF Bearish OB"
                ))
        
        for ob in htf_data.get('bullish_obs', []):
            if ob.get('top', 0) > 0:
                self.levels.append(PriceLevel(
                    price=ob['top'],
                    level_type=LevelType.HTF_OB_BULLISH,
                    role=LevelRole.SUPPORT,
                    source_tf='HTF',
                    confidence=0.95,
                    description="HTF Bullish OB"
                ))
        
        # Single HTF OB fallback
        if htf_data.get('bearish_ob_bottom', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['bearish_ob_bottom'],
                level_type=LevelType.HTF_OB_BEARISH,
                role=LevelRole.RESISTANCE,
                source_tf='HTF',
                description="HTF Bearish OB"
            ))
        if htf_data.get('bullish_ob_top', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['bullish_ob_top'],
                level_type=LevelType.HTF_OB_BULLISH,
                role=LevelRole.SUPPORT,
                source_tf='HTF',
                description="HTF Bullish OB"
            ))
        
        # HTF Swings
        if htf_data.get('htf_swing_high', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['htf_swing_high'],
                level_type=LevelType.HTF_SWING_HIGH,
                role=LevelRole.RESISTANCE,
                source_tf='HTF',
                description="HTF Swing High"
            ))
        if htf_data.get('htf_swing_low', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['htf_swing_low'],
                level_type=LevelType.HTF_SWING_LOW,
                role=LevelRole.SUPPORT,
                source_tf='HTF',
                description="HTF Swing Low"
            ))
        
        # HTF Volume Profile (POC, VWAP)
        if htf_data.get('htf_poc', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['htf_poc'],
                level_type=LevelType.POC,
                role=LevelRole.BOTH,
                source_tf='HTF',
                confidence=0.85,
                description="HTF POC (4H)"
            ))
        if htf_data.get('htf_vwap', 0) > 0:
            self.levels.append(PriceLevel(
                price=htf_data['htf_vwap'],
                level_type=LevelType.VWAP,
                role=LevelRole.BOTH,
                source_tf='HTF',
                confidence=0.8,
                description="HTF VWAP (4H)"
            ))
    
    def collect_from_volume(self, volume_data: Dict) -> None:
        """Collect levels from volume profile (external data)"""
        if not volume_data:
            return
        
        if volume_data.get('poc', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['poc'],
                level_type=LevelType.POC,
                role=LevelRole.BOTH,
                description="POC"
            ))
        if volume_data.get('vah', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['vah'],
                level_type=LevelType.VAH,
                role=LevelRole.RESISTANCE,
                description="VAH"
            ))
        if volume_data.get('val', 0) > 0:
            self.levels.append(PriceLevel(
                price=volume_data['val'],
                level_type=LevelType.VAL,
                role=LevelRole.SUPPORT,
                description="VAL"
            ))
    
    def collect_volume_profile(self, lookback: int = None) -> None:
        """
        Calculate and collect volume profile levels from DataFrame.
        
        Calculates:
        - POC (Point of Control) - highest volume price level
        - VAH (Value Area High) - upper bound of 70% volume
        - VAL (Value Area Low) - lower bound of 70% volume
        - VWAP (Volume Weighted Average Price)
        
        Args:
            lookback: Number of candles to analyze (default: all data)
        """
        if self.df is None or len(self.df) < 20:
            return
        
        df = self.df.tail(lookback) if lookback else self.df
        
        try:
            # ═══════════════════════════════════════════════════════════════
            # VWAP - Volume Weighted Average Price
            # ═══════════════════════════════════════════════════════════════
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).sum() / df['Volume'].sum()
            
            if vwap > 0:
                self.levels.append(PriceLevel(
                    price=float(vwap),
                    level_type=LevelType.VWAP,
                    role=LevelRole.BOTH,
                    description=f"VWAP ({self.timeframe})"
                ))
            
            # ═══════════════════════════════════════════════════════════════
            # VOLUME PROFILE - POC, VAH, VAL
            # ═══════════════════════════════════════════════════════════════
            price_min = df['Low'].min()
            price_max = df['High'].max()
            price_range = price_max - price_min
            
            if price_range <= 0:
                return
            
            # Create price bins (50 bins across the range)
            num_bins = 50
            bin_size = price_range / num_bins
            bins = {}
            
            for idx in range(len(df)):
                row = df.iloc[idx]
                candle_low = row['Low']
                candle_high = row['High']
                candle_volume = row['Volume']
                
                # Distribute volume across price bins this candle touched
                for bin_idx in range(num_bins):
                    bin_low = price_min + bin_idx * bin_size
                    bin_high = bin_low + bin_size
                    bin_mid = (bin_low + bin_high) / 2
                    
                    # Check if candle overlaps this bin
                    if candle_low <= bin_high and candle_high >= bin_low:
                        # Calculate overlap proportion
                        overlap_low = max(candle_low, bin_low)
                        overlap_high = min(candle_high, bin_high)
                        overlap_pct = (overlap_high - overlap_low) / (candle_high - candle_low) if candle_high > candle_low else 1
                        
                        # Add proportional volume to bin
                        if bin_mid not in bins:
                            bins[bin_mid] = 0
                        bins[bin_mid] += candle_volume * overlap_pct
            
            if not bins:
                return
            
            # Find POC (highest volume price)
            poc_price = max(bins.keys(), key=lambda x: bins[x])
            total_volume = sum(bins.values())
            
            self.levels.append(PriceLevel(
                price=float(poc_price),
                level_type=LevelType.POC,
                role=LevelRole.BOTH,
                confidence=0.9,
                description=f"POC ({self.timeframe})"
            ))
            
            # Calculate Value Area (70% of volume)
            sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
            cumulative_volume = 0
            value_area_prices = []
            
            for price, volume in sorted_bins:
                cumulative_volume += volume
                value_area_prices.append(price)
                if cumulative_volume >= total_volume * 0.7:
                    break
            
            if value_area_prices:
                vah = max(value_area_prices)
                val = min(value_area_prices)
                
                if vah > self.current_price:
                    self.levels.append(PriceLevel(
                        price=float(vah),
                        level_type=LevelType.VAH,
                        role=LevelRole.RESISTANCE,
                        description=f"VAH ({self.timeframe})"
                    ))
                
                if val < self.current_price:
                    self.levels.append(PriceLevel(
                        price=float(val),
                        level_type=LevelType.VAL,
                        role=LevelRole.SUPPORT,
                        description=f"VAL ({self.timeframe})"
                    ))
        
        except Exception as e:
            pass  # Silently fail - volume profile is optional
    
    def collect_previous_day_levels(self) -> None:
        """
        Collect previous day's key levels.
        
        Calculates from daily data:
        - Previous Day High/Low
        - Previous Day POC (if enough intraday data)
        """
        if self.df is None or len(self.df) < 2:
            return
        
        try:
            # If we have timestamp index, group by day
            if hasattr(self.df.index, 'date'):
                df_with_date = self.df.copy()
                df_with_date['date'] = self.df.index.date
                
                dates = df_with_date['date'].unique()
                if len(dates) >= 2:
                    prev_date = sorted(dates)[-2]
                    prev_day_data = df_with_date[df_with_date['date'] == prev_date]
                    
                    if len(prev_day_data) > 0:
                        prev_high = prev_day_data['High'].max()
                        prev_low = prev_day_data['Low'].min()
                        prev_close = prev_day_data['Close'].iloc[-1]
                        
                        self.levels.append(PriceLevel(
                            price=float(prev_high),
                            level_type=LevelType.PREV_DAY_HIGH,
                            role=LevelRole.RESISTANCE,
                            description="Prev Day High"
                        ))
                        self.levels.append(PriceLevel(
                            price=float(prev_low),
                            level_type=LevelType.PREV_DAY_LOW,
                            role=LevelRole.SUPPORT,
                            description="Prev Day Low"
                        ))
                        
                        # Previous day POC
                        if len(prev_day_data) >= 10:
                            typical = (prev_day_data['High'] + prev_day_data['Low'] + prev_day_data['Close']) / 3
                            prev_poc = (typical * prev_day_data['Volume']).sum() / prev_day_data['Volume'].sum()
                            
                            self.levels.append(PriceLevel(
                                price=float(prev_poc),
                                level_type=LevelType.POC,
                                role=LevelRole.BOTH,
                                description="Prev Day POC"
                            ))
        except:
            pass
    
    def collect_from_historical(self) -> None:
        """Collect historical levels from DataFrame"""
        if self.df is None or len(self.df) < 2:
            return
        
        # Previous day levels (if we have daily data)
        try:
            prev_high = self.df['High'].iloc[-2]
            prev_low = self.df['Low'].iloc[-2]
            prev_close = self.df['Close'].iloc[-2]
            
            self.levels.append(PriceLevel(
                price=prev_high,
                level_type=LevelType.PREV_DAY_HIGH,
                role=LevelRole.RESISTANCE,
                description="Previous High"
            ))
            self.levels.append(PriceLevel(
                price=prev_low,
                level_type=LevelType.PREV_DAY_LOW,
                role=LevelRole.SUPPORT,
                description="Previous Low"
            ))
        except:
            pass
    
    def collect_fibonacci(self) -> None:
        """Calculate and add Fibonacci RETRACEMENT and EXTENSION levels"""
        if self.df is None or len(self.df) < 20:
            return
        
        recent_high = self.df['High'].tail(50).max()
        recent_low = self.df['Low'].tail(50).min()
        
        fib_range = recent_high - recent_low
        
        # RETRACEMENT levels (support during pullbacks)
        fib_retracements = [
            (0.382, LevelType.FIB_382, "Fib 38.2%"),
            (0.500, LevelType.FIB_500, "Fib 50%"),
            (0.618, LevelType.FIB_618, "Fib 61.8%"),
            (0.786, LevelType.FIB_786, "Fib 78.6%"),
        ]
        
        for ratio, level_type, desc in fib_retracements:
            price = recent_high - (fib_range * ratio)
            self.levels.append(PriceLevel(
                price=price,
                level_type=level_type,
                role=LevelRole.BOTH,
                description=desc
            ))
        
        # ═══════════════════════════════════════════════════════════════════════
        # EXTENSION levels (targets for breakouts above recent high)
        # Critical for finding TPs when price is near highs!
        # ═══════════════════════════════════════════════════════════════════════
        fib_extensions = [
            (1.272, "Fib Ext 127.2%"),
            (1.414, "Fib Ext 141.4%"),
            (1.618, "Fib Ext 161.8%"),
            (2.000, "Fib Ext 200%"),
        ]
        
        for ratio, desc in fib_extensions:
            price = recent_low + (fib_range * ratio)
            # Only add if above current price (useful as TP)
            if price > self.current_price:
                self.levels.append(PriceLevel(
                    price=price,
                    level_type=LevelType.FIB_EXTENSION,
                    role=LevelRole.RESISTANCE,
                    confidence=0.7,
                    description=desc
                ))
    
    def collect_liquidity_zones(self) -> None:
        """Add liquidity zones based on swing points"""
        pass  # Simplified - can be expanded
    
    def collect_round_numbers(self) -> None:
        """Add round number levels - MORE levels for better TP selection"""
        price = self.current_price
        
        # Determine round number interval based on price
        if price >= 10000:
            # For BTC: add $1000 intervals ($96k, $97k, $98k, $99k, $100k)
            interval = 1000
            num_above = 6  # More levels above for TPs
            num_below = 3
        elif price >= 1000:
            interval = 100
            num_above = 5
            num_below = 3
        elif price >= 100:
            interval = 10
            num_above = 4
            num_below = 2
        elif price >= 10:
            interval = 1
            num_above = 4
            num_below = 2
        elif price >= 1:
            interval = 0.1
            num_above = 4
            num_below = 2
        else:
            interval = 0.01
            num_above = 4
            num_below = 2
        
        # Find base round number
        base = math.floor(price / interval) * interval
        
        # Add levels BELOW price (support)
        for i in range(num_below):
            level_price = base - (i * interval)
            if level_price > 0:
                self.levels.append(PriceLevel(
                    price=level_price,
                    level_type=LevelType.ROUND_NUMBER,
                    role=LevelRole.SUPPORT,
                    confidence=0.6 if i == 0 else 0.5,
                    description=f"Round {level_price:,.0f}"
                ))
        
        # Add levels ABOVE price (resistance/TPs)
        for i in range(1, num_above + 1):
            level_price = base + (i * interval)
            self.levels.append(PriceLevel(
                price=level_price,
                level_type=LevelType.ROUND_NUMBER,
                role=LevelRole.RESISTANCE,
                confidence=0.6 if i == 1 else 0.5,
                description=f"Round {level_price:,.0f}"
            ))
    
    def get_support_levels(self) -> List[PriceLevel]:
        """Get all support levels below current price"""
        return [l for l in self.levels 
                if l.price < self.current_price and 
                l.role in [LevelRole.SUPPORT, LevelRole.BOTH]]
    
    def get_resistance_levels(self) -> List[PriceLevel]:
        """Get all resistance levels above current price"""
        return [l for l in self.levels 
                if l.price > self.current_price and 
                l.role in [LevelRole.RESISTANCE, LevelRole.BOTH]]


# ═══════════════════════════════════════════════════════════════════════════════
# MIP TRADE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class TradeOptimizer:
    """
    MIP-based Trade Optimizer
    
    Uses PuLP for TP selection, anti-hunt logic for SL.
    """
    
    def __init__(
        self,
        collector: LevelCollector,
        mode: TradingMode,
        htf_bias: str = None,
        winning_strategy: str = None,
        ml_prediction: Dict = None,
        analysis_mode: str = 'hybrid',
    ):
        self.collector = collector
        self.current_price = collector.current_price
        self.mode = mode
        self.constraints = MODE_CONSTRAINTS[mode]
        self.atr = collector.atr
        self.htf_bias = htf_bias or 'NEUTRAL'
        self.winning_strategy = winning_strategy
        self.ml_prediction = ml_prediction
        self.analysis_mode = analysis_mode
        
        # Categorize levels
        self.tp_candidates: List[PriceLevel] = []
        self.sl_candidates: List[PriceLevel] = []
    
    def _categorize_levels(self, direction: str):
        """Categorize collected levels by direction"""
        for level in self.collector.levels:
            if direction == 'LONG':
                if level.price > self.current_price:
                    self.tp_candidates.append(level)
                elif level.price < self.current_price:
                    self.sl_candidates.append(level)
            else:  # SHORT
                if level.price < self.current_price:
                    self.tp_candidates.append(level)
                elif level.price > self.current_price:
                    self.sl_candidates.append(level)
    
    def _add_atr_fallbacks(self, direction: str, entry: float):
        """Add ATR-based fallback levels"""
        if direction == 'LONG':
            for i, mult in enumerate([1.5, 2.5, 4.0], 1):
                self.tp_candidates.append(PriceLevel(
                    price=entry + self.atr * mult,
                    level_type=LevelType.ATR_BASED,
                    description=f"ATR TP{i}"
                ))
            self.sl_candidates.append(PriceLevel(
                price=entry - self.atr * 1.5,
                level_type=LevelType.ATR_BASED,
                description="ATR SL"
            ))
        else:
            for i, mult in enumerate([1.5, 2.5, 4.0], 1):
                self.tp_candidates.append(PriceLevel(
                    price=entry - self.atr * mult,
                    level_type=LevelType.ATR_BASED,
                    description=f"ATR TP{i}"
                ))
            self.sl_candidates.append(PriceLevel(
                price=entry + self.atr * 1.5,
                level_type=LevelType.ATR_BASED,
                description="ATR SL"
            ))
    
    def _score_level(self, level: PriceLevel, role: str, entry: float) -> float:
        """Score a level for a given role"""
        score = 0.0
        
        # Structure quality
        score += STRUCTURE_SCORES.get(level.level_type, 10)
        
        # Mode fit
        distance_pct = abs(level.price - entry) / entry * 100 if entry > 0 else 0
        
        if role == 'TP1':
            ideal = self.constraints.ideal_tp1_pct
            max_dist = self.constraints.max_tp1_pct
        elif role == 'TP2':
            ideal = self.constraints.ideal_tp2_pct
            max_dist = self.constraints.max_tp2_pct
        elif role == 'TP3':
            ideal = self.constraints.ideal_tp3_pct
            max_dist = self.constraints.max_tp3_pct
        else:
            ideal = (self.constraints.min_sl_pct + self.constraints.max_sl_pct) / 2
            max_dist = self.constraints.max_sl_pct
        
        # Gaussian scoring
        if ideal > 0:
            deviation = abs(distance_pct - ideal) / ideal
            score += 50 * math.exp(-deviation ** 2)
        
        # Penalty for too far
        if distance_pct > max_dist:
            overshoot = (distance_pct - max_dist) / max_dist
            score -= 30 * overshoot
        
        # HTF bonus
        if level.level_type in {LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
                                LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW}:
            score += 20
        
        # Confidence
        score += level.confidence * 10
        
        return max(0, score)
    
    def _find_best_sl_structure(self, direction: str, entry: float) -> Optional[PriceLevel]:
        """Find best structure level for SL"""
        sl_priority = {
            LevelType.HTF_OB_BULLISH: 1,
            LevelType.HTF_OB_BEARISH: 1,
            LevelType.OB_BULLISH: 2,
            LevelType.OB_BEARISH: 2,
            LevelType.FVG_BULLISH: 3,
            LevelType.FVG_BEARISH: 3,
            LevelType.FIB_618: 4,
            LevelType.SWING_LOW: 5,
            LevelType.SWING_HIGH: 5,
            LevelType.HTF_SWING_LOW: 5,
            LevelType.HTF_SWING_HIGH: 5,
        }
        
        max_sl_dist = entry * (self.constraints.max_sl_pct / 100)
        min_sl_dist = entry * (self.constraints.min_sl_pct / 100)
        
        valid_levels = []
        for level in self.sl_candidates:
            dist = abs(entry - level.price)
            if dist <= max_sl_dist * 0.8 and dist >= min_sl_dist * 0.3:
                priority = sl_priority.get(level.level_type, 99)
                valid_levels.append((level, priority))
        
        if not valid_levels:
            return None
        
        valid_levels.sort(key=lambda x: x[1])
        return valid_levels[0][0]
    
    def _calculate_anti_hunt_sl(self, structure: PriceLevel, direction: str, entry: float) -> Tuple[float, str]:
        """Calculate anti-hunt SL with buffer and ugly number"""
        buffer_multipliers = {
            TradingMode.SCALP: 0.3,
            TradingMode.DAY_TRADE: 0.5,
            TradingMode.SWING: 0.7,
            TradingMode.INVESTMENT: 1.0,
        }
        buffer_mult = buffer_multipliers.get(self.mode, 0.5)
        
        # Random factor for unpredictability
        random_factor = 0.9 + random.random() * 0.2
        buffer = self.atr * buffer_mult * random_factor
        
        if direction == 'LONG':
            raw_sl = structure.price - buffer
        else:
            raw_sl = structure.price + buffer
        
        ugly_sl = self._make_ugly_number(raw_sl, direction)
        
        # Enforce constraints
        sl_pct = abs(entry - ugly_sl) / entry * 100 if entry > 0 else 0
        
        if sl_pct > self.constraints.max_sl_pct:
            max_dist = entry * (self.constraints.max_sl_pct / 100)
            if direction == 'LONG':
                ugly_sl = self._make_ugly_number(entry - max_dist, direction)
            else:
                ugly_sl = self._make_ugly_number(entry + max_dist, direction)
            desc = f"Anti-hunt below {structure.description} (capped)"
        elif sl_pct < self.constraints.min_sl_pct:
            min_dist = entry * (self.constraints.min_sl_pct / 100)
            if direction == 'LONG':
                ugly_sl = self._make_ugly_number(entry - min_dist, direction)
            else:
                ugly_sl = self._make_ugly_number(entry + min_dist, direction)
            desc = f"Anti-hunt (expanded to min)"
        else:
            desc = f"Anti-hunt below {structure.description}"
        
        return ugly_sl, desc
    
    def _make_ugly_number(self, price: float, direction: str) -> float:
        """Convert to ugly number avoiding round numbers"""
        if price <= 0:
            return price
        
        if price >= 1000:
            decimal_places = 2
        elif price >= 10:
            decimal_places = 3
        elif price >= 1:
            decimal_places = 4
        else:
            decimal_places = 6
        
        rounded = round(price, decimal_places)
        ugly_offsets = [3, 7, 13, 17, 23, 27, 33, 37, 43, 47]
        offset_multiplier = 10 ** (-decimal_places)
        offset_idx = int(price * 100) % len(ugly_offsets)
        offset = ugly_offsets[offset_idx] * offset_multiplier
        
        if direction == 'LONG':
            return rounded - offset
        else:
            return rounded + offset
    
    def optimize(self, direction: str) -> TradeSetup:
        """Run optimization"""
        entry = self.current_price
        
        # Categorize and add fallbacks
        self._categorize_levels(direction)
        self._add_atr_fallbacks(direction, entry)
        
        # Deduplicate
        self.tp_candidates = list({l.price: l for l in self.tp_candidates}.values())
        self.sl_candidates = list({l.price: l for l in self.sl_candidates}.values())
        
        # Sort
        if direction == 'LONG':
            self.tp_candidates.sort(key=lambda x: x.price)
            self.sl_candidates.sort(key=lambda x: x.price, reverse=True)
        else:
            self.tp_candidates.sort(key=lambda x: x.price, reverse=True)
            self.sl_candidates.sort(key=lambda x: x.price)
        
        # Get anti-hunt SL
        sl_structure = self._find_best_sl_structure(direction, entry)
        if sl_structure:
            sl_price, sl_desc = self._calculate_anti_hunt_sl(sl_structure, direction, entry)
            sl_level = PriceLevel(price=sl_price, level_type=sl_structure.level_type, description=sl_desc)
        else:
            min_dist = entry * (self.constraints.min_sl_pct / 100)
            if direction == 'LONG':
                sl_price = self._make_ugly_number(entry - min_dist, direction)
            else:
                sl_price = self._make_ugly_number(entry + min_dist, direction)
            sl_level = PriceLevel(price=sl_price, level_type=LevelType.ATR_BASED, description="Mode minimum SL")
            sl_desc = sl_level.description
        
        risk = abs(entry - sl_level.price)
        sl_pct = abs(entry - sl_level.price) / entry * 100 if entry > 0 else 0
        
        # Use MIP or greedy for TPs
        if PULP_AVAILABLE and len(self.tp_candidates) >= 3:
            tp1, tp2, tp3, tp1_level, tp2_level, tp3_level, total_score = self._optimize_tps_mip(direction, entry, risk)
        else:
            tp1, tp2, tp3, tp1_level, tp2_level, tp3_level, total_score = self._optimize_tps_greedy(direction, entry, risk)
        
        # Calculate R:R
        tp1_rr = abs(tp1 - entry) / risk if risk > 0 and tp1 > 0 else 0
        
        # Check R:R constraint
        if tp1_rr < self.constraints.min_rr:
            return TradeSetup(
                direction='WAIT',
                wait_reason=f"R:R {tp1_rr:.1f}:1 below minimum {self.constraints.min_rr}:1",
                mode=self.mode,
            )
        
        # Build descriptions with ACTUAL level type names
        tp1_pct = abs(tp1 - entry) / entry * 100 if entry > 0 else 0
        tp2_pct = abs(tp2 - entry) / entry * 100 if entry > 0 else 0
        tp3_pct = abs(tp3 - entry) / entry * 100 if entry > 0 else 0
        
        # Map level type to readable name
        level_type_names = {
            'ob_bearish': 'Bearish OB',
            'ob_bullish': 'Bullish OB',
            'htf_ob_bearish': 'HTF Bearish OB',
            'htf_ob_bullish': 'HTF Bullish OB',
            'fvg_bearish': 'Bearish FVG',
            'fvg_bullish': 'Bullish FVG',
            'swing_high': 'Swing High',
            'swing_low': 'Swing Low',
            'htf_swing_high': 'HTF Swing High',
            'htf_swing_low': 'HTF Swing Low',
            'poc': 'POC',
            'vwap': 'VWAP',
            'vah': 'VAH',
            'val': 'VAL',
            'fib_extension': 'Fib Extension',
            'fib_618': 'Fib 61.8%',
            'fib_500': 'Fib 50%',
            'fib_382': 'Fib 38.2%',
            'round_number': 'Round Number',
            'prev_day_high': 'Prev Day High',
            'prev_day_low': 'Prev Day Low',
            'atr_based': 'ATR Extension',
        }
        
        def get_level_name(level):
            type_name = level_type_names.get(level.level_type.value, level.level_type.value)
            return type_name
        
        tp1_type_name = get_level_name(tp1_level)
        tp2_type_name = get_level_name(tp2_level)
        tp3_type_name = get_level_name(tp3_level)
        
        tp1_desc = f"📍 Source: {tp1_level.description}\n✅ {tp1_type_name} (+{tp1_pct:.1f}%)"
        tp2_desc = f"📍 Source: {tp2_level.description}\n✅ {tp2_type_name} (+{tp2_pct:.1f}%)"
        tp3_desc = f"📍 Source: {tp3_level.description}\n✅ {tp3_type_name} (+{tp3_pct:.1f}%)"
        sl_description = f"📍 Source: {sl_desc}\n🛡️ Anti-hunt SL ({sl_pct:.1f}%)"
        
        return TradeSetup(
            direction=direction,
            entry=entry,
            entry_type='market',
            
            stop_loss=sl_level.price,
            sl_type=sl_level.description,
            sl_level=sl_level,
            sl_description=sl_description,
            
            tp1=tp1,
            tp1_type=tp1_level.description,
            tp1_level=tp1_level,
            tp1_description=tp1_desc,
            
            tp2=tp2,
            tp2_type=tp2_level.description,
            tp2_level=tp2_level,
            tp2_description=tp2_desc,
            
            tp3=tp3,
            tp3_type=tp3_level.description,
            tp3_level=tp3_level,
            tp3_description=tp3_desc,
            
            risk_reward=tp1_rr,
            risk_pct=sl_pct,
            rr_tp1=tp1_rr,
            confidence=min(100, total_score / 3),
            reasoning=f"MIP: SL={sl_desc}, TP1={tp1_level.description}",
            mode=self.mode,
            total_score=total_score,
        )
    
    def _optimize_tps_mip(self, direction: str, entry: float, risk: float):
        """MIP optimization for TPs using PuLP"""
        prob = pulp.LpProblem("TPOptimizer", pulp.LpMaximize)
        
        tp_levels = {i: l for i, l in enumerate(self.tp_candidates)}
        n_tp = len(tp_levels)
        
        x_tp1 = pulp.LpVariable.dicts("tp1", range(n_tp), cat='Binary')
        x_tp2 = pulp.LpVariable.dicts("tp2", range(n_tp), cat='Binary')
        x_tp3 = pulp.LpVariable.dicts("tp3", range(n_tp), cat='Binary')
        
        tp1_scores = {i: self._score_level(l, 'TP1', entry) for i, l in tp_levels.items()}
        tp2_scores = {i: self._score_level(l, 'TP2', entry) for i, l in tp_levels.items()}
        tp3_scores = {i: self._score_level(l, 'TP3', entry) for i, l in tp_levels.items()}
        
        # Objective
        prob += (
            pulp.lpSum([tp1_scores[i] * x_tp1[i] for i in range(n_tp)]) +
            pulp.lpSum([tp2_scores[i] * x_tp2[i] for i in range(n_tp)]) +
            pulp.lpSum([tp3_scores[i] * x_tp3[i] for i in range(n_tp)])
        )
        
        # Constraints
        prob += pulp.lpSum([x_tp1[i] for i in range(n_tp)]) == 1
        prob += pulp.lpSum([x_tp2[i] for i in range(n_tp)]) == 1
        prob += pulp.lpSum([x_tp3[i] for i in range(n_tp)]) == 1
        
        # Ordering
        for i in range(n_tp):
            for j in range(n_tp):
                if direction == 'LONG':
                    if tp_levels[i].price >= tp_levels[j].price:
                        prob += x_tp1[i] + x_tp2[j] <= 1
                    if tp_levels[i].price >= tp_levels[j].price:
                        prob += x_tp2[i] + x_tp3[j] <= 1
                else:
                    if tp_levels[i].price <= tp_levels[j].price:
                        prob += x_tp1[i] + x_tp2[j] <= 1
                    if tp_levels[i].price <= tp_levels[j].price:
                        prob += x_tp2[i] + x_tp3[j] <= 1
        
        # R:R constraint for TP1
        min_rr = self.constraints.min_rr
        for i in range(n_tp):
            if direction == 'LONG':
                reward = tp_levels[i].price - entry
            else:
                reward = entry - tp_levels[i].price
            rr = reward / risk if risk > 0 else 0
            if rr < min_rr:
                prob += x_tp1[i] == 0
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            return self._optimize_tps_greedy(direction, entry, risk)
        
        tp1_idx = next((i for i in range(n_tp) if pulp.value(x_tp1[i]) == 1), 0)
        tp2_idx = next((i for i in range(n_tp) if pulp.value(x_tp2[i]) == 1), 1)
        tp3_idx = next((i for i in range(n_tp) if pulp.value(x_tp3[i]) == 1), 2)
        
        return (
            tp_levels[tp1_idx].price, tp_levels[tp2_idx].price, tp_levels[tp3_idx].price,
            tp_levels[tp1_idx], tp_levels[tp2_idx], tp_levels[tp3_idx],
            pulp.value(prob.objective)
        )
    
    def _optimize_tps_greedy(self, direction: str, entry: float, risk: float):
        """Greedy fallback for TPs - PRIORITIZES SMC STRUCTURE"""
        
        # ═══════════════════════════════════════════════════════════════════════
        # STRUCTURE FIRST: Separate SMC structure from fallback levels
        # Always prefer OB, FVG, Swing over Fib/Round numbers
        # ═══════════════════════════════════════════════════════════════════════
        structure_levels = [l for l in self.tp_candidates if l.level_type in SMC_STRUCTURE_TYPES]
        fallback_levels = [l for l in self.tp_candidates if l.level_type not in SMC_STRUCTURE_TYPES]
        
        # Score and sort structure levels first
        scored_structure = [(l, self._score_level(l, 'TP1', entry)) for l in structure_levels]
        scored_structure.sort(key=lambda x: x[1], reverse=True)
        
        # Score and sort fallback levels
        scored_fallback = [(l, self._score_level(l, 'TP1', entry)) for l in fallback_levels]
        scored_fallback.sort(key=lambda x: x[1], reverse=True)
        
        # Prioritize structure: try structure first, then fallback
        scored = scored_structure + scored_fallback
        
        selected = []
        prev_price = entry if direction == 'LONG' else float('inf')
        
        for level, score in scored:
            if direction == 'LONG':
                if level.price > prev_price:
                    if len(selected) == 0:
                        reward = level.price - entry
                        rr = reward / risk if risk > 0 else 0
                        if rr < self.constraints.min_rr:
                            continue
                    selected.append(level)
                    prev_price = level.price
            else:
                if level.price < prev_price:
                    if len(selected) == 0:
                        reward = entry - level.price
                        rr = reward / risk if risk > 0 else 0
                        if rr < self.constraints.min_rr:
                            continue
                    selected.append(level)
                    prev_price = level.price
            
            if len(selected) >= 3:
                break
        
        # Pad if needed - calculate from LAST SELECTED level, not entry!
        while len(selected) < 3:
            # Base price is last selected TP (or entry if none selected)
            base_price = selected[-1].price if selected else entry
            
            # Add 1 ATR increment for each subsequent TP
            increment = self.atr * 1.0  # 1 ATR between TPs
            
            if direction == 'LONG':
                price = base_price + increment
            else:
                price = base_price - increment
            
            selected.append(PriceLevel(
                price=price, 
                level_type=LevelType.ATR_BASED, 
                description=f"ATR TP{len(selected)+1}"
            ))
        
        total_score = sum(self._score_level(l, f'TP{i+1}', entry) for i, l in enumerate(selected[:3]))
        
        return (
            selected[0].price, selected[1].price, selected[2].price,
            selected[0], selected[1], selected[2],
            total_score
        )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # LIMIT ENTRY OPTIMIZATION
    # Structure-based entry selection with recency scoring
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def optimize_limit_entry(
        self, 
        direction: str, 
        stop_loss: float = 0,
        tp1: float = 0,
    ) -> Optional[LimitEntryResult]:
        """
        Optimize limit entry based on structure levels.
        
        Key principles:
        1. FRESH structures from impulsive moves score highest
        2. For LONG: Entry at TOP of bullish OB (not bottom)
        3. Entry must be WITHIN structure zone, not below/above it
        4. Mode-appropriate distances (Scalp=close, Swing=further)
        
        Args:
            direction: 'LONG' or 'SHORT'
            stop_loss: Pre-calculated SL for R:R calculation
            tp1: Pre-calculated TP1 for R:R calculation
            
        Returns:
            LimitEntryResult or None if no valid entry found
        """
        entry_constraints = ENTRY_CONSTRAINTS.get(self.mode, ENTRY_CONSTRAINTS[TradingMode.DAY_TRADE])
        
        # Collect entry candidates
        candidates = self._collect_entry_candidates(direction, entry_constraints)
        
        if not candidates:
            return None
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            score = self._score_entry_level(
                candidate, 
                direction, 
                entry_constraints,
                stop_loss,
                tp1
            )
            candidate['total_score'] = score
            scored_candidates.append(candidate)
        
        # Sort by score (highest first)
        scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Select best candidate
        best = scored_candidates[0]
        
        # Calculate R:R improvement
        rr_market = 0
        rr_limit = 0
        if stop_loss > 0 and tp1 > 0:
            if direction == 'LONG':
                market_risk = abs(self.current_price - stop_loss)
                market_reward = abs(tp1 - self.current_price)
                limit_risk = abs(best['price'] - stop_loss)
                limit_reward = abs(tp1 - best['price'])
            else:
                market_risk = abs(stop_loss - self.current_price)
                market_reward = abs(self.current_price - tp1)
                limit_risk = abs(stop_loss - best['price'])
                limit_reward = abs(best['price'] - tp1)
            
            rr_market = market_reward / market_risk if market_risk > 0 else 0
            rr_limit = limit_reward / limit_risk if limit_risk > 0 else 0
        
        rr_improvement = rr_limit - rr_market
        
        # Validate: Limit entry should improve R:R
        is_valid = True
        warning = ""
        
        if rr_improvement < 0.1:
            warning = "Limit entry provides minimal R:R improvement"
        
        # Validate: Entry is in correct direction
        if direction == 'LONG' and best['price'] >= self.current_price:
            is_valid = False
            warning = "Limit entry must be below current price for LONG"
        elif direction == 'SHORT' and best['price'] <= self.current_price:
            is_valid = False
            warning = "Limit entry must be above current price for SHORT"
        
        # Validate: Entry is above SL for LONG, below SL for SHORT
        if stop_loss > 0:
            if direction == 'LONG' and best['price'] <= stop_loss:
                is_valid = False
                warning = "Limit entry is below stop loss!"
            elif direction == 'SHORT' and best['price'] >= stop_loss:
                is_valid = False
                warning = "Limit entry is above stop loss!"
        
        return LimitEntryResult(
            price=best['price'],
            level_type=best.get('level_type', 'UNKNOWN'),
            source=best.get('source', 'structure'),
            zone_top=best.get('zone_top', best['price']),
            zone_bottom=best.get('zone_bottom', best['price']),
            distance_pct=best.get('distance_pct', 0),
            rr_improvement=rr_improvement,
            score=best['total_score'],
            recency_score=best.get('recency_score', 50),
            description=best.get('description', ''),
            reason=f"{best.get('description', 'Structure')} ({best.get('distance_pct', 0):.1f}% {'below' if direction == 'LONG' else 'above'})",
            is_valid=is_valid,
            warning=warning,
        )
    
    def _collect_entry_candidates(
        self, 
        direction: str, 
        constraints: 'EntryConstraints'
    ) -> List[Dict]:
        """
        Collect potential entry levels from structure.
        
        For LONG: Support levels below current price
        For SHORT: Resistance levels above current price
        """
        candidates = []
        
        min_dist = self.current_price * (constraints.min_entry_dist_pct / 100)
        max_dist = self.current_price * (constraints.max_entry_dist_pct / 100)
        
        for level in self.collector.levels:
            if direction == 'LONG':
                # For LONG: Need levels BELOW current price
                if level.price >= self.current_price:
                    continue
                    
                distance = self.current_price - level.price
                if distance < min_dist or distance > max_dist:
                    continue
                
                distance_pct = (distance / self.current_price) * 100
                
                # For bullish OB: Use TOP of zone (optimal entry)
                # This is where price will react
                entry_price = level.price
                zone_top = level.price
                zone_bottom = level.price * 0.99  # Approximate
                
                # Check if this is an OB with zone data
                if hasattr(level, 'zone_bottom') and level.zone_bottom > 0:
                    zone_bottom = level.zone_bottom
                
            else:  # SHORT
                # For SHORT: Need levels ABOVE current price
                if level.price <= self.current_price:
                    continue
                    
                distance = level.price - self.current_price
                if distance < min_dist or distance > max_dist:
                    continue
                
                distance_pct = (distance / self.current_price) * 100
                
                # For bearish OB: Use BOTTOM of zone (optimal entry)
                entry_price = level.price
                zone_top = level.price * 1.01  # Approximate
                zone_bottom = level.price
                
                if hasattr(level, 'zone_top') and level.zone_top > 0:
                    zone_top = level.zone_top
            
            # Estimate recency from description or source_tf
            recency_score = self._estimate_recency(level)
            
            candidates.append({
                'price': entry_price,
                'level_type': level.level_type.value if hasattr(level.level_type, 'value') else str(level.level_type),
                'source': 'structure',
                'zone_top': zone_top,
                'zone_bottom': zone_bottom,
                'distance_pct': distance_pct,
                'recency_score': recency_score,
                'description': level.description,
                'confidence': level.confidence,
                'original_level': level,
            })
        
        # Add FVG equilibrium entries (50% of gap)
        self._add_fvg_equilibrium_entries(candidates, direction, constraints)
        
        # Add ATR fallback if no structure found
        if not candidates:
            self._add_atr_entry_fallback(candidates, direction, constraints)
        
        return candidates
    
    def _estimate_recency(self, level: PriceLevel) -> float:
        """
        Estimate how recent/fresh a level is (0-100).
        Fresh levels from impulsive moves score higher.
        """
        recency = 50  # Default
        
        desc_lower = level.description.lower()
        
        # Fresh indicators
        if 'fresh' in desc_lower or 'recent' in desc_lower:
            recency = 90
        elif 'impulse' in desc_lower or 'impulsive' in desc_lower:
            recency = 85
        
        # HTF levels are older by definition
        if 'htf' in desc_lower or level.level_type in {
            LevelType.HTF_OB_BULLISH, LevelType.HTF_OB_BEARISH,
            LevelType.HTF_SWING_HIGH, LevelType.HTF_SWING_LOW
        }:
            recency = max(30, recency - 20)
        
        # OBs are generally more reliable than swings
        if level.level_type in {LevelType.OB_BULLISH, LevelType.OB_BEARISH}:
            recency = min(100, recency + 10)
        
        return recency
    
    def _add_fvg_equilibrium_entries(
        self, 
        candidates: List[Dict], 
        direction: str,
        constraints: 'EntryConstraints'
    ) -> None:
        """Add FVG equilibrium (50%) as entry option."""
        for level in self.collector.levels:
            if direction == 'LONG' and level.level_type == LevelType.FVG_BULLISH:
                # FVG has high and low - equilibrium is middle
                # Note: We may only have the top stored, estimate bottom
                fvg_top = level.price
                fvg_bottom = level.price * 0.995  # Approximate 0.5% gap
                
                equilibrium = (fvg_top + fvg_bottom) / 2
                
                if equilibrium < self.current_price:
                    distance = self.current_price - equilibrium
                    distance_pct = (distance / self.current_price) * 100
                    
                    if constraints.min_entry_dist_pct <= distance_pct <= constraints.max_entry_dist_pct:
                        candidates.append({
                            'price': equilibrium,
                            'level_type': 'FVG_EQUILIBRIUM',
                            'source': 'fvg_eq',
                            'zone_top': fvg_top,
                            'zone_bottom': fvg_bottom,
                            'distance_pct': distance_pct,
                            'recency_score': 75,
                            'description': 'FVG 50% (Equilibrium)',
                            'confidence': 0.85,
                        })
            
            elif direction == 'SHORT' and level.level_type == LevelType.FVG_BEARISH:
                fvg_bottom = level.price
                fvg_top = level.price * 1.005
                
                equilibrium = (fvg_top + fvg_bottom) / 2
                
                if equilibrium > self.current_price:
                    distance = equilibrium - self.current_price
                    distance_pct = (distance / self.current_price) * 100
                    
                    if constraints.min_entry_dist_pct <= distance_pct <= constraints.max_entry_dist_pct:
                        candidates.append({
                            'price': equilibrium,
                            'level_type': 'FVG_EQUILIBRIUM',
                            'source': 'fvg_eq',
                            'zone_top': fvg_top,
                            'zone_bottom': fvg_bottom,
                            'distance_pct': distance_pct,
                            'recency_score': 75,
                            'description': 'FVG 50% (Equilibrium)',
                            'confidence': 0.85,
                        })
    
    def _add_atr_entry_fallback(
        self, 
        candidates: List[Dict], 
        direction: str,
        constraints: 'EntryConstraints'
    ) -> None:
        """Add ATR-based entry as last resort."""
        # Use 1 ATR as pullback target
        atr_entry = self.current_price - self.atr if direction == 'LONG' else self.current_price + self.atr
        
        distance_pct = (self.atr / self.current_price) * 100
        
        if constraints.min_entry_dist_pct <= distance_pct <= constraints.max_entry_dist_pct:
            candidates.append({
                'price': atr_entry,
                'level_type': 'ATR_PULLBACK',
                'source': 'atr',
                'zone_top': atr_entry,
                'zone_bottom': atr_entry,
                'distance_pct': distance_pct,
                'recency_score': 30,  # Low - not structure-based
                'description': f'ATR Pullback ({distance_pct:.1f}%)',
                'confidence': 0.5,
            })
    
    def _score_entry_level(
        self, 
        candidate: Dict, 
        direction: str,
        constraints: 'EntryConstraints',
        stop_loss: float = 0,
        tp1: float = 0,
    ) -> float:
        """
        Score an entry level candidate.
        
        Factors:
        1. Structure quality (OB > FVG > Swing > ATR)
        2. Recency (fresh from impulsive move = better)
        3. Distance fit (ideal distance for mode)
        4. R:R improvement
        5. Zone validation (entry within zone, not outside)
        """
        score = 0.0
        
        # ═══════════════════════════════════════════════════════════════════════
        # 1. STRUCTURE QUALITY (0-100 points)
        # ═══════════════════════════════════════════════════════════════════════
        level_type = candidate.get('level_type', '')
        
        # Try to get score from ENTRY_LEVEL_SCORES
        if isinstance(level_type, str):
            # String type - check special types first
            if level_type == 'FRESH_OB':
                score += 100
            elif level_type == 'FRESH_FVG':
                score += 95
            elif level_type == 'FVG_EQUILIBRIUM':
                score += 85
            elif level_type == 'ATR_PULLBACK':
                score += 10
            else:
                # Try to match to LevelType enum
                try:
                    lt = LevelType(level_type)
                    score += ENTRY_LEVEL_SCORES.get(lt, 30)
                except ValueError:
                    score += 30  # Default
        else:
            score += ENTRY_LEVEL_SCORES.get(level_type, 30)
        
        # ═══════════════════════════════════════════════════════════════════════
        # 2. RECENCY BONUS (0-50 points)
        # Fresh structures from impulsive moves are more reliable
        # ═══════════════════════════════════════════════════════════════════════
        recency = candidate.get('recency_score', 50)
        score += recency * 0.5  # Max 50 points
        
        # ═══════════════════════════════════════════════════════════════════════
        # 3. DISTANCE FIT - Gaussian scoring (0-40 points)
        # Entry should be at ideal distance for the mode
        # ═══════════════════════════════════════════════════════════════════════
        distance_pct = candidate.get('distance_pct', 0)
        ideal_dist = constraints.ideal_entry_dist_pct
        
        if ideal_dist > 0:
            # Gaussian: peaks at ideal distance
            deviation = abs(distance_pct - ideal_dist) / ideal_dist
            distance_score = 40 * math.exp(-deviation ** 2)
            score += distance_score
        
        # Penalty for being too close (not worth the wait)
        if distance_pct < constraints.min_entry_dist_pct:
            score -= 20
        
        # Penalty for being too far (unrealistic)
        if distance_pct > constraints.max_entry_dist_pct:
            overshoot = (distance_pct - constraints.max_entry_dist_pct) / constraints.max_entry_dist_pct
            score -= 30 * overshoot
        
        # ═══════════════════════════════════════════════════════════════════════
        # 4. R:R IMPROVEMENT BONUS (0-30 points)
        # Better entry = better R:R
        # ═══════════════════════════════════════════════════════════════════════
        if stop_loss > 0 and tp1 > 0:
            entry_price = candidate['price']
            
            if direction == 'LONG':
                risk = abs(entry_price - stop_loss)
                reward = abs(tp1 - entry_price)
            else:
                risk = abs(stop_loss - entry_price)
                reward = abs(entry_price - tp1)
            
            if risk > 0:
                rr = reward / risk
                # Bonus for good R:R
                if rr >= 3.0:
                    score += 30
                elif rr >= 2.0:
                    score += 20
                elif rr >= 1.5:
                    score += 10
                elif rr < 1.0:
                    score -= 20  # Penalty for bad R:R
        
        # ═══════════════════════════════════════════════════════════════════════
        # 5. ZONE VALIDATION
        # Entry should be WITHIN the OB zone, not outside
        # ═══════════════════════════════════════════════════════════════════════
        zone_top = candidate.get('zone_top', 0)
        zone_bottom = candidate.get('zone_bottom', 0)
        entry_price = candidate['price']
        description = candidate.get('description', '').lower()
        
        if zone_top > 0 and zone_bottom > 0:
            # Check if entry is within zone
            if zone_bottom <= entry_price <= zone_top:
                score += 15  # Bonus for being in zone
            else:
                # Entry outside zone - penalize
                if direction == 'LONG' and entry_price < zone_bottom:
                    score -= 25  # Entry below OB = less likely to fill
                elif direction == 'SHORT' and entry_price > zone_top:
                    score -= 25  # Entry above OB
        
        # BONUS: For LONG, prefer TOP of OB (conservative entry, high fill probability)
        # For SHORT, prefer BOTTOM of OB
        if direction == 'LONG':
            if 'bullish ob' in description and 'bottom' not in description:
                score += 20  # TOP of bullish OB = optimal LONG entry
            elif 'bottom' in description:
                score -= 10  # Bottom of OB = more aggressive, lower fill probability
        else:  # SHORT
            if 'bearish ob' in description and 'top' not in description:
                score += 20  # BOTTOM of bearish OB = optimal SHORT entry
            elif 'top' in description:
                score -= 10
        
        # ═══════════════════════════════════════════════════════════════════════
        # 6. CONFIDENCE BONUS
        # ═══════════════════════════════════════════════════════════════════════
        confidence = candidate.get('confidence', 0.5)
        score += confidence * 10
        
        return max(0, score)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE - COMPATIBLE WITH OLD CODE
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
    winning_strategy: Optional[str] = None,
    ml_prediction: Optional[Dict] = None,
    analysis_mode: str = 'hybrid',
) -> TradeSetup:
    """
    Main interface - get optimized trade setup using MIP.
    
    COMPATIBLE with old interface - drop-in replacement.
    """
    # Auto-detect mode
    if mode is None:
        mode = TIMEFRAME_MODE_MAP.get(timeframe, TradingMode.DAY_TRADE)
    
    # Collect levels
    collector = LevelCollector(df=df, current_price=current_price, timeframe=timeframe)
    collector.collect_from_smc(smc_data)
    
    if htf_data:
        collector.collect_from_htf(htf_data)
    
    # Volume profile - use external data if provided, otherwise auto-calculate
    if volume_data:
        collector.collect_from_volume(volume_data)
    else:
        # Auto-calculate volume profile from DataFrame
        collector.collect_volume_profile()
    
    # Previous day levels (POC, High, Low)
    collector.collect_previous_day_levels()
    
    collector.collect_from_historical()
    collector.collect_fibonacci()
    collector.collect_round_numbers()
    
    # Optimize
    optimizer = TradeOptimizer(
        collector=collector,
        mode=mode,
        htf_bias=htf_bias,
        winning_strategy=winning_strategy,
        ml_prediction=ml_prediction,
        analysis_mode=analysis_mode,
    )
    
    return optimizer.optimize(direction)


def get_optimized_limit_entry(
    df: pd.DataFrame,
    current_price: float,
    direction: str,
    timeframe: str,
    smc_data: Dict,
    stop_loss: float = 0,
    tp1: float = 0,
    htf_data: Optional[Dict] = None,
    volume_data: Optional[Dict] = None,
    mode: Optional[TradingMode] = None,
) -> Optional[LimitEntryResult]:
    """
    Get optimized limit entry based on structure.
    
    This is the main interface for limit entry optimization.
    Uses the same level collection as get_optimized_trade() but
    with entry-specific scoring that prioritizes:
    1. Fresh OBs from recent impulsive moves
    2. Proper zone placement (TOP of bullish OB for LONG)
    3. Mode-appropriate distances
    4. R:R improvement over market entry
    
    Args:
        df: OHLCV DataFrame
        current_price: Current market price
        direction: 'LONG' or 'SHORT'
        timeframe: Analysis timeframe
        smc_data: SMC structure data
        stop_loss: Pre-calculated stop loss (for R:R calculation)
        tp1: Pre-calculated TP1 (for R:R calculation)
        htf_data: Higher timeframe data (optional)
        volume_data: Volume profile data (optional)
        mode: Trading mode (auto-detected from timeframe if not provided)
        
    Returns:
        LimitEntryResult with optimal limit entry, or None if no valid entry
        
    Example:
        result = get_optimized_limit_entry(
            df=df,
            current_price=0.1187,
            direction='LONG',
            timeframe='15m',
            smc_data=smc,
            stop_loss=0.1140,
            tp1=0.1250,
        )
        
        if result and result.is_valid:
            print(f"Limit Entry: ${result.price:.4f}")
            print(f"Source: {result.description}")
            print(f"R:R Improvement: +{result.rr_improvement:.1f}")
    """
    # Auto-detect mode
    if mode is None:
        mode = TIMEFRAME_MODE_MAP.get(timeframe, TradingMode.DAY_TRADE)
    
    # Collect levels
    collector = LevelCollector(df=df, current_price=current_price, timeframe=timeframe)
    collector.collect_from_smc(smc_data)
    
    if htf_data:
        collector.collect_from_htf(htf_data)
    
    if volume_data:
        collector.collect_from_volume(volume_data)
    else:
        collector.collect_volume_profile()
    
    collector.collect_previous_day_levels()
    collector.collect_from_historical()
    collector.collect_fibonacci()
    
    # Create optimizer
    optimizer = TradeOptimizer(
        collector=collector,
        mode=mode,
    )
    
    # Categorize levels for the direction
    optimizer._categorize_levels(direction)
    
    # Optimize limit entry
    return optimizer.optimize_limit_entry(
        direction=direction,
        stop_loss=stop_loss,
        tp1=tp1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test with LITUSDT-like scenario
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    close = 2.19 + np.cumsum(np.random.randn(100) * 0.01)
    
    df = pd.DataFrame({
        'Open': close - np.random.rand(100) * 0.01,
        'High': close + np.random.rand(100) * 0.02,
        'Low': close - np.random.rand(100) * 0.02,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    smc_data = {
        'bearish_obs': [
            {'top': 2.33, 'bottom': 2.28},
            {'top': 2.45, 'bottom': 2.40},
        ],
        'bullish_obs': [
            {'top': 2.15, 'bottom': 2.10},
        ],
        'swing_high': 2.35,
        'swing_low': 2.05,
    }
    
    setup = get_optimized_trade(
        df=df,
        current_price=2.19,
        direction='LONG',
        timeframe='15m',
        smc_data=smc_data,
        htf_bias='BULLISH',
    )
    
    print(f"Direction: {setup.direction}")
    print(f"Entry: ${setup.entry:.4f}")
    print(f"SL: ${setup.stop_loss:.4f} ({setup.risk_pct:.1f}%) - {setup.sl_type}")
    print(f"TP1: ${setup.tp1:.4f} R:R {setup.rr_tp1:.1f}:1 - {setup.tp1_type}")
    print(f"TP2: ${setup.tp2:.4f} - {setup.tp2_type}")
    print(f"TP3: ${setup.tp3:.4f} - {setup.tp3_type}")
    print(f"Score: {setup.total_score:.1f}")
    
    # Test Limit Entry Optimizer
    print("\n" + "=" * 70)
    print("LIMIT ENTRY OPTIMIZER TEST")
    print("=" * 70)
    
    limit_result = get_optimized_limit_entry(
        df=df,
        current_price=2.19,
        direction='LONG',
        timeframe='15m',
        smc_data=smc_data,
        stop_loss=setup.stop_loss,
        tp1=setup.tp1,
    )
    
    if limit_result:
        print(f"Limit Entry: ${limit_result.price:.4f}")
        print(f"Source: {limit_result.description}")
        print(f"Level Type: {limit_result.level_type}")
        print(f"Distance: {limit_result.distance_pct:.1f}% below current")
        print(f"R:R Improvement: +{limit_result.rr_improvement:.2f}")
        print(f"Score: {limit_result.score:.1f}")
        print(f"Recency: {limit_result.recency_score:.0f}/100")
        print(f"Valid: {limit_result.is_valid}")
        if limit_result.warning:
            print(f"Warning: {limit_result.warning}")
    else:
        print("No valid limit entry found")
    
    # Test ZBTUSDT-like scenario
    print("\n" + "=" * 70)
    print("ZBTUSDT SCENARIO TEST")
    print("=" * 70)
    
    zbt_smc = {
        'bullish_obs': [
            {'top': 0.1175, 'bottom': 0.1155},  # Fresh OB from impulsive move
            {'top': 0.1140, 'bottom': 0.1130},  # Older OB (should score lower)
        ],
        'bullish_fvgs': [
            {'top': 0.1170, 'bottom': 0.1160},  # FVG in the move
        ],
        'swing_low': 0.1120,
    }
    
    zbt_limit = get_optimized_limit_entry(
        df=df,
        current_price=0.1187,
        direction='LONG',
        timeframe='15m',
        smc_data=zbt_smc,
        stop_loss=0.1140,
        tp1=0.1250,
    )
    
    if zbt_limit:
        print(f"Current Price: $0.1187")
        print(f"Optimized Limit: ${zbt_limit.price:.4f}")
        print(f"Source: {zbt_limit.description}")
        print(f"Zone: ${zbt_limit.zone_bottom:.4f} - ${zbt_limit.zone_top:.4f}")
        print(f"Distance: {zbt_limit.distance_pct:.1f}% below current")
        print(f"R:R Improvement: +{zbt_limit.rr_improvement:.2f}")
        print(f"Score: {zbt_limit.score:.1f}")
        print(f"Should be ~$0.1175 (TOP of fresh OB), NOT $0.1141 (below structure)")