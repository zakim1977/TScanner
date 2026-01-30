"""
InvestorIQ Data Models
======================
Single source of truth for all data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    BULLISH = "BULLISH"
    LEAN_BULLISH = "LEAN_BULLISH"
    NEUTRAL = "NEUTRAL"
    LEAN_BEARISH = "LEAN_BEARISH"
    BEARISH = "BEARISH"
    
    @property
    def color(self) -> str:
        return {
            Direction.BULLISH: "#00ff88",
            Direction.LEAN_BULLISH: "#00d4aa",
            Direction.NEUTRAL: "#888888",
            Direction.LEAN_BEARISH: "#ff9966",
            Direction.BEARISH: "#ff6b6b",
        }.get(self, "#888888")
    
    @property
    def is_bullish(self) -> bool:
        return self in [Direction.BULLISH, Direction.LEAN_BULLISH]
    
    @property
    def is_bearish(self) -> bool:
        return self in [Direction.BEARISH, Direction.LEAN_BEARISH]


class Trade(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WAIT = "WAIT"
    
    @property
    def color(self) -> str:
        return {"LONG": "#00ff88", "SHORT": "#ff6b6b", "WAIT": "#ffcc00"}.get(self.value, "#ffcc00")


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Position(Enum):
    EARLY = "EARLY"
    MIDDLE = "MIDDLE"
    LATE = "LATE"
    
    @property
    def color(self) -> str:
        return {"EARLY": "#00ff88", "MIDDLE": "#ffcc00", "LATE": "#ff6b6b"}.get(self.value, "#888888")


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

class T:
    """Thresholds - All magic numbers in ONE place"""
    EXTREME_BULLISH = 75
    STRONG_BULLISH = 70
    BULLISH = 65
    LEAN_BULLISH = 60
    LEAN_BEARISH = 40
    BEARISH = 35
    STRONG_BEARISH = 30
    EXTREME_BEARISH = 25
    EARLY_MAX = 35
    LATE_MIN = 65
    EXPLOSION_READY = 70
    EXPLOSION_HIGH = 50


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WhaleData:
    """Whale positioning"""
    whale_pct: float = 50.0
    retail_pct: float = 50.0
    oi_change_24h: float = 0.0
    oi_change_1h: float = 0.0
    price_change_24h: float = 0.0
    funding_rate: float = 0.0
    
    @property
    def divergence(self) -> float:
        return self.whale_pct - self.retail_pct
    
    @property
    def direction(self) -> Direction:
        if self.whale_pct >= T.STRONG_BULLISH: return Direction.BULLISH
        elif self.whale_pct >= T.LEAN_BULLISH: return Direction.LEAN_BULLISH
        elif self.whale_pct <= T.STRONG_BEARISH: return Direction.BEARISH
        elif self.whale_pct <= T.LEAN_BEARISH: return Direction.LEAN_BEARISH
        return Direction.NEUTRAL
    
    @property
    def confidence(self) -> Confidence:
        if self.whale_pct >= T.STRONG_BULLISH or self.whale_pct <= T.STRONG_BEARISH:
            return Confidence.HIGH
        elif self.whale_pct >= T.LEAN_BULLISH or self.whale_pct <= T.LEAN_BEARISH:
            return Confidence.MEDIUM
        return Confidence.LOW
    
    @property
    def is_trap(self) -> bool:
        return self.divergence < 0
    
    @property
    def direction_score(self) -> int:
        if self.whale_pct >= T.EXTREME_BULLISH or self.whale_pct <= T.EXTREME_BEARISH: return 40
        elif self.whale_pct >= T.STRONG_BULLISH or self.whale_pct <= T.STRONG_BEARISH: return 36
        elif self.whale_pct >= T.BULLISH or self.whale_pct <= T.BEARISH: return 32
        elif self.whale_pct >= T.LEAN_BULLISH or self.whale_pct <= T.LEAN_BEARISH: return 28
        return 15
    
    @property
    def squeeze_score(self) -> int:
        if self.divergence >= 15: return 20
        elif self.divergence >= 10: return 14
        elif self.divergence >= 5: return 8
        elif self.divergence < 0: return max(-10, int(self.divergence))
        return 0
    
    @property
    def squeeze_label(self) -> str:
        if self.divergence >= 15: return "HIGH"
        elif self.divergence >= 10: return "MEDIUM"
        elif self.divergence >= 5: return "LOW"
        elif self.divergence < 0: return "CONFLICT"
        return "NONE"


@dataclass
class ExplosionData:
    """Explosion readiness"""
    score: int = 0
    ready: bool = False
    direction: Optional[str] = None
    state: str = "UNKNOWN"
    entry_valid: bool = False
    squeeze_pct: float = 50.0
    signals: List[str] = field(default_factory=list)
    bb_squeeze: float = 0.0
    volume_buildup: float = 0.0
    oi_buildup: float = 0.0


@dataclass
class MLData:
    """ML prediction"""
    direction: str = "WAIT"
    confidence: float = 0.0
    top_factors: List[str] = field(default_factory=list)
    
    @property
    def color(self) -> str:
        return "#00ff88" if self.direction == "LONG" else "#ff6b6b" if self.direction == "SHORT" else "#ffcc00"


@dataclass
class FVGData:
    """Fair Value Gap - LuxAlgo style"""
    bullish_fvg: bool = False
    bearish_fvg: bool = False
    bullish_fvg_top: float = 0.0
    bullish_fvg_bottom: float = 0.0
    bearish_fvg_top: float = 0.0
    bearish_fvg_bottom: float = 0.0
    at_bullish_fvg: bool = False
    at_bearish_fvg: bool = False


@dataclass
class SMCData:
    """Smart Money Concepts - Complete"""
    swing_high: float = 0.0
    swing_low: float = 0.0
    structure: str = "NEUTRAL"
    bias: str = "Neutral"
    bos: bool = False
    choch: bool = False
    
    bullish_ob_top: float = 0.0
    bullish_ob_bottom: float = 0.0
    bearish_ob_top: float = 0.0
    bearish_ob_bottom: float = 0.0
    
    bullish_obs: List[Dict] = field(default_factory=list)
    bearish_obs: List[Dict] = field(default_factory=list)
    
    at_bullish_ob: bool = False
    at_bearish_ob: bool = False
    near_bullish_ob: bool = False
    near_bearish_ob: bool = False
    at_support: bool = False
    at_resistance: bool = False
    
    fvg: FVGData = field(default_factory=FVGData)
    
    liquidity_swept_high: bool = False
    liquidity_swept_low: bool = False
    
    @property
    def has_bullish_ob(self) -> bool:
        return self.bullish_ob_top > 0
    
    @property
    def has_bearish_ob(self) -> bool:
        return self.bearish_ob_top > 0


@dataclass
class HTFData:
    """Higher timeframe data"""
    timeframe: str = ""
    structure: str = "NEUTRAL"
    money_flow_phase: str = "CONSOLIDATION"
    bullish_obs: List[Dict] = field(default_factory=list)
    bearish_obs: List[Dict] = field(default_factory=list)
    swing_high: float = 0.0
    swing_low: float = 0.0
    
    @property
    def is_bullish(self) -> bool:
        return self.structure in ["BULLISH", "LEAN_BULLISH", "bullish"]
    
    @property
    def is_bearish(self) -> bool:
        return self.structure in ["BEARISH", "LEAN_BEARISH", "bearish"]


@dataclass
class TradeSetup:
    """Trade setup with levels"""
    direction: str = "WAIT"
    entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    rr1: float = 0.0
    rr2: float = 0.0
    rr3: float = 0.0
    entry_reason: str = ""
    sl_reason: str = ""
    tp1_reason: str = ""
    strategy: str = ""
    
    @property
    def risk_pct(self) -> float:
        if self.entry > 0 and self.stop_loss > 0:
            return abs(self.entry - self.stop_loss) / self.entry * 100
        return 0.0


@dataclass
class MoneyFlow:
    """Money flow data"""
    phase: str = "CONSOLIDATION"
    is_accumulating: bool = False
    is_distributing: bool = False
    flow_score: float = 50.0


@dataclass
class CombinedLearning:
    """Combined learning stories"""
    conclusion: str = ""
    conclusion_action: str = "WAIT"
    stories: List[Tuple[str, str]] = field(default_factory=list)
    
    layer1_direction: str = "NEUTRAL"
    layer1_score: int = 15
    layer1_confidence: str = "LOW"
    layer1_reason: str = ""
    
    layer2_squeeze: str = "NONE"
    layer2_score: int = 0
    layer2_reason: str = ""
    
    layer3_entry: str = "WAIT"
    layer3_score: int = 10
    layer3_reason: str = ""
    
    is_squeeze: bool = False
    has_conflict: bool = False
    conflicts: List[str] = field(default_factory=list)


@dataclass
class RulesDecision:
    """Decision from MASTER_RULES"""
    action: str = "WAIT"
    trade_direction: str = "WAIT"
    confidence: str = "LOW"
    direction_score: int = 15
    squeeze_score: int = 0
    entry_score: int = 10
    total_score: int = 25
    direction_label: str = "NEUTRAL"
    squeeze_label: str = "NONE"
    position_label: str = "MIDDLE"
    main_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    is_valid_long: bool = True
    is_valid_short: bool = True
    whale_story: str = ""
    oi_story: str = ""
    position_story: str = ""
    conclusion: str = ""
    conclusion_action: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RESULT - THE SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    """
    THE SINGLE SOURCE OF TRUTH
    Used by Scanner, Single Analysis, Trade Monitor - ALL use this same object.
    """
    symbol: str = ""
    timeframe: str = ""
    market_type: str = "Crypto"
    trading_mode: str = "Day Trade"
    engine_mode: str = "hybrid"
    price: float = 0.0
    
    whale: WhaleData = field(default_factory=WhaleData)
    explosion: ExplosionData = field(default_factory=ExplosionData)
    smc: SMCData = field(default_factory=SMCData)
    htf: HTFData = field(default_factory=HTFData)
    money_flow: MoneyFlow = field(default_factory=MoneyFlow)
    
    ml: Optional[MLData] = None
    rules: Optional[RulesDecision] = None
    
    setup: TradeSetup = field(default_factory=TradeSetup)
    learning: CombinedLearning = field(default_factory=CombinedLearning)
    
    trade: Trade = Trade.WAIT
    action: str = "WAIT"
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    setup_type: Optional[str] = None
    
    position: Position = Position.MIDDLE
    position_pct: float = 50.0
    ta_score: int = 50
    
    df: Any = None
    error: Optional[str] = None
    btc_correlation: float = 0.0
    btc_trend: str = "NEUTRAL"
    
    @property
    def direction_score(self) -> int:
        return self.rules.direction_score if self.rules else self.whale.direction_score
    
    @property
    def squeeze_score(self) -> int:
        return self.rules.squeeze_score if self.rules else self.whale.squeeze_score
    
    @property
    def timing_score(self) -> int:
        if self.rules: return self.rules.entry_score
        base = {Position.EARLY: 10, Position.MIDDLE: 7, Position.LATE: 3}.get(self.position, 5)
        return base + int(self.ta_score * 0.2)
    
    @property
    def total_score(self) -> int:
        base = self.rules.total_score if self.rules else (self.direction_score + max(0, self.squeeze_score) + self.timing_score)
        if self.explosion.ready: base += 15
        elif self.explosion.score >= 50: base += 10
        if self.trade == Trade.WAIT: base = min(base, 55)
        return max(0, min(100, base))
    
    @property
    def direction(self) -> Direction:
        return self.whale.direction
    
    @property
    def direction_label(self) -> str:
        return self.rules.direction_label if self.rules else self.whale.direction.value
    
    @property
    def confidence(self) -> Confidence:
        return self.whale.confidence
    
    @property
    def squeeze_label(self) -> str:
        return self.rules.squeeze_label if self.rules else self.whale.squeeze_label
    
    @property 
    def is_valid(self) -> bool:
        return self.error is None
    
    @property
    def ml_rules_aligned(self) -> bool:
        if not self.ml: return True
        ml_long = self.ml.direction == "LONG"
        rules_bullish = self.direction.is_bullish
        return (ml_long and rules_bullish) or (not ml_long and not rules_bullish)
    
    @property
    def action_emoji(self) -> str:
        return "🟢" if self.trade == Trade.LONG else "🔴" if self.trade == Trade.SHORT else "⏳"
    
    @property
    def action_color(self) -> str:
        return self.trade.color
    
    @property
    def summary(self) -> str:
        if self.trade == Trade.LONG:
            return f"LONG SETUP - {self.reason or f'Whales {self.whale.whale_pct:.0f}% bullish'}"
        elif self.trade == Trade.SHORT:
            return f"SHORT SETUP - {self.reason or f'Whales {self.whale.whale_pct:.0f}% bearish'}"
        return f"WAIT - {self.reason or 'No clear edge'}"
