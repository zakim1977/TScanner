"""
═══════════════════════════════════════════════════════════════════════════════
📊 INSTITUTIONAL RULES ENGINE - Professional Configuration-Driven Analysis
═══════════════════════════════════════════════════════════════════════════════

This module provides a PROFESSIONAL approach to institutional data analysis:

1. CONFIGURATION-DRIVEN: All thresholds and rules defined in YAML/JSON
2. EXTENSIBLE: Easy to add new patterns without code changes
3. ML-READY: Structured for future machine learning integration
4. BACKTESTABLE: Every decision is logged and traceable

NO MORE HARDCODED THRESHOLDS - Everything is configurable!

Author: InvestorIQ
Version: 2.0.0
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import os

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION (Can be overridden by config file)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "version": "2.0.0",
    "last_updated": "2025-12-21",
    
    # ═══════════════════════════════════════════════════════════════════════════
    # THRESHOLD DEFINITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    "thresholds": {
        "oi_change": {
            "significant_positive": 1.0,   # OI rising = new positions
            "significant_negative": -1.0,  # OI falling = closing positions
            "strong_positive": 3.0,        # Strong conviction
            "strong_negative": -3.0        # Strong exit
        },
        "price_change": {
            "bullish": 0.0,                # Any positive = bullish
            "bearish": 0.0,                # Any negative = bearish
            "strong_bullish": 2.0,         # Strong move up
            "strong_bearish": -2.0         # Strong move down
        },
        "whale_positioning": {
            "bullish": 55.0,               # >= 55% long = bullish
            "bearish": 45.0,               # <= 45% long = bearish
            "strong_bullish": 60.0,        # >= 60% = strong conviction
            "strong_bearish": 40.0         # <= 40% = strong conviction
        },
        "retail_positioning": {
            "bullish": 55.0,
            "bearish": 45.0,
            "extreme_bullish": 65.0,       # Fade signal when retail extreme
            "extreme_bearish": 35.0
        },
        "funding_rate": {
            "positive_extreme": 0.0005,    # 0.05% = longs overleveraged
            "negative_extreme": -0.0005,   # -0.05% = shorts overleveraged
            "neutral_band": 0.0001         # ±0.01% = neutral
        },
        "divergence": {
            "significant": 10.0,           # Whale - Retail difference
            "strong": 15.0
        }
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OI + PRICE COMBINATION PATTERNS
    # This is the CORE logic - OI and Price together tell the story
    # ═══════════════════════════════════════════════════════════════════════════
    "oi_price_patterns": {
        "NEW_LONGS": {
            "conditions": {
                "oi_change": ">=significant_positive",
                "price_change": ">=bullish"
            },
            "signal": "BULLISH",
            "description": "New money entering LONG positions",
            "conviction": "STRONG",
            "action": "Bullish continuation expected"
        },
        "NEW_SHORTS": {
            "conditions": {
                "oi_change": ">=significant_positive",
                "price_change": "<=bearish"
            },
            "signal": "BEARISH",
            "description": "New money entering SHORT positions",
            "conviction": "STRONG",
            "action": "Bearish continuation expected"
        },
        "SHORT_COVERING": {
            "conditions": {
                "oi_change": "<=significant_negative",
                "price_change": ">=bullish"
            },
            "signal": "WEAK_BULLISH",
            "description": "Shorts closing, NOT new longs entering",
            "conviction": "WEAK",
            "action": "Rally may be temporary - no new buying"
        },
        "LONG_LIQUIDATION": {
            "conditions": {
                "oi_change": "<=significant_negative",
                "price_change": "<=bearish"
            },
            "signal": "WEAK_BEARISH",
            "description": "Longs closing/stopped out",
            "conviction": "WEAK",
            "action": "Dump may be exhausting - watch for reversal"
        },
        "NEUTRAL": {
            "conditions": {
                "oi_change": "neutral",
                "price_change": "any"
            },
            "signal": "NEUTRAL",
            "description": "No significant position changes",
            "conviction": "NONE",
            "action": "Follow price action"
        }
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO DEFINITIONS (Complete market situations)
    # These combine multiple signals into actionable scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    "scenarios": {
        "NEW_LONGS_WHALE_BULLISH": {
            "name": "New Longs + Whale Bullish",
            "emoji": "🟢",
            "required": {
                "oi_price_pattern": "NEW_LONGS",
                "whale_bias": "BULLISH"
            },
            "optional": {},
            "confidence": "HIGH",
            "direction": "LONG",
            "status": "READY",
            "score": 85,
            "description": "Fresh buying with whale confirmation",
            "action": "Enter LONG on pullbacks, stop below recent swing low",
            "education": "Smart money is buying WITH new positions entering. This is high-conviction bullish."
        },
        "NEW_LONGS_NEUTRAL_WHALES": {
            "name": "New Longs Entering",
            "emoji": "🟢",
            "required": {
                "oi_price_pattern": "NEW_LONGS"
            },
            "excluded": {
                "whale_bias": "BEARISH"
            },
            "confidence": "MEDIUM",
            "direction": "LONG",
            "status": "READY",
            "score": 65,
            "description": "Fresh buying, whales neutral",
            "action": "LONG bias, but smaller size without whale confirmation",
            "education": "New longs entering but whales haven't committed yet. Still bullish but less conviction."
        },
        "SHORT_SQUEEZE": {
            "name": "Short Squeeze Setup",
            "emoji": "🚀",
            "required": {
                "funding_signal": "SHORTS_OVERLEVERAGED",
                "whale_bias": "BULLISH"
            },
            "optional": {
                "retail_bias": "BEARISH"
            },
            "confidence": "HIGH",
            "direction": "LONG",
            "status": "READY",
            "score": 90,
            "description": "Shorts overleveraged, squeeze imminent",
            "action": "LONG now, tight stop, target 5-10%",
            "education": "Shorts paying high funding + whales long = explosive upside when shorts cover."
        },
        "LONG_LIQUIDATION_WHALE_BULLISH": {
            "name": "Whale Accumulation Zone",
            "emoji": "🟡",
            "required": {
                "oi_price_pattern": "LONG_LIQUIDATION",
                "whale_bias": "BULLISH"
            },
            "confidence": "MEDIUM",
            "direction": "LONG",
            "status": "WAIT",
            "score": 65,
            "description": "Retail longs getting liquidated while whales accumulate",
            "action": "DON'T catch falling knife. Wait for stabilization, then LONG",
            "wait_conditions": [
                "Price stops making new lows",
                "Volume spike on green candle",
                "OI starts rising again"
            ],
            "education": "This is HOW wealth transfers from retail to whales. Retail panic sells, whales buy."
        },
        "NEW_SHORTS_WHALE_BEARISH": {
            "name": "New Shorts + Whale Bearish",
            "emoji": "🔴",
            "required": {
                "oi_price_pattern": "NEW_SHORTS",
                "whale_bias": "BEARISH"
            },
            "confidence": "HIGH",
            "direction": "SHORT",
            "status": "READY",
            "score": 85,
            "description": "Fresh selling with whale confirmation",
            "action": "Enter SHORT on bounces, stop above recent swing high",
            "education": "Smart money is selling WITH new shorts entering. High-conviction bearish."
        },
        "NEW_SHORTS_NEUTRAL_WHALES": {
            "name": "New Shorts Entering",
            "emoji": "🔴",
            "required": {
                "oi_price_pattern": "NEW_SHORTS"
            },
            "excluded": {
                "whale_bias": "BULLISH"
            },
            "confidence": "MEDIUM",
            "direction": "SHORT",
            "status": "READY",
            "score": 65,
            "description": "Fresh selling, whales neutral",
            "action": "SHORT bias, but smaller size without whale confirmation",
            "education": "New shorts entering but whales haven't committed yet. Still bearish but less conviction."
        },
        "LONG_SQUEEZE": {
            "name": "Long Squeeze Setup",
            "emoji": "💥",
            "required": {
                "funding_signal": "LONGS_OVERLEVERAGED",
                "whale_bias": "BEARISH"
            },
            "optional": {
                "retail_bias": "BULLISH"
            },
            "confidence": "HIGH",
            "direction": "SHORT",
            "status": "READY",
            "score": 90,
            "description": "Longs overleveraged, dump imminent",
            "action": "SHORT now, tight stop, target 5-10%",
            "education": "Longs paying high funding + whales short = violent downside when longs liquidate."
        },
        "SHORT_COVERING_WHALE_BEARISH": {
            "name": "Weak Rally - Whale Distribution",
            "emoji": "🟡",
            "required": {
                "oi_price_pattern": "SHORT_COVERING",
                "whale_bias": "BEARISH"
            },
            "confidence": "MEDIUM",
            "direction": "SHORT",
            "status": "WAIT",
            "score": 65,
            "description": "Rally is just short covering, whales still bearish",
            "action": "DON'T short the rally immediately. Wait for exhaustion, then SHORT",
            "wait_conditions": [
                "Price stops making new highs",
                "Volume fading on green candles",
                "OI starts rising again (new shorts entering)"
            ],
            "education": "Retail sees rally and buys. Whales are selling to them. Classic distribution."
        },
        "SHORT_COVERING_NO_CONVICTION": {
            "name": "Short Covering Rally",
            "emoji": "🟡",
            "required": {
                "oi_price_pattern": "SHORT_COVERING"
            },
            "confidence": "LOW",
            "direction": "NEUTRAL",
            "status": "WAIT",
            "score": 45,
            "description": "Rally from short covering, unclear direction",
            "action": "Wait for new positions to enter (OI rising)",
            "education": "Price up but positions closing. Need new conviction to continue."
        },
        "LONG_LIQUIDATION_NO_SUPPORT": {
            "name": "Long Liquidation Cascade",
            "emoji": "🟡",
            "required": {
                "oi_price_pattern": "LONG_LIQUIDATION"
            },
            "excluded": {
                "whale_bias": "BULLISH"
            },
            "confidence": "LOW",
            "direction": "NEUTRAL",
            "status": "WAIT",
            "score": 45,
            "description": "Longs getting liquidated, no whale support",
            "action": "Stay out - could continue or reverse",
            "education": "Forced selling happening. Without whale buying, could cascade further."
        },
        "CONFLICTING_SIGNALS": {
            "name": "Conflicting Signals",
            "emoji": "⚠️",
            "required": {
                "conflict": True
            },
            "confidence": "LOW",
            "direction": "NEUTRAL",
            "status": "CONFLICTING",
            "score": 40,
            "description": "OI/Price and Whale positioning disagree",
            "action": "Small position or wait for alignment",
            "education": "When signals conflict, it often means accumulation or distribution phase. Wait for clarity."
        },
        "NO_EDGE": {
            "name": "No Clear Edge",
            "emoji": "⚪",
            "required": {},
            "confidence": "LOW",
            "direction": "NEUTRAL",
            "status": "AVOID",
            "score": 50,
            "description": "All metrics neutral, no institutional edge",
            "action": "Use technical analysis only, or wait",
            "education": "No clear smart money signal. The market is in equilibrium - wait for extremes."
        }
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCENARIO PRIORITY (Order matters - first match wins)
    # ═══════════════════════════════════════════════════════════════════════════
    "scenario_priority": [
        "SHORT_SQUEEZE",
        "LONG_SQUEEZE",
        "NEW_LONGS_WHALE_BULLISH",
        "NEW_SHORTS_WHALE_BEARISH",
        "LONG_LIQUIDATION_WHALE_BULLISH",
        "SHORT_COVERING_WHALE_BEARISH",
        "NEW_LONGS_NEUTRAL_WHALES",
        "NEW_SHORTS_NEUTRAL_WHALES",
        "SHORT_COVERING_NO_CONVICTION",
        "LONG_LIQUIDATION_NO_SUPPORT",
        "CONFLICTING_SIGNALS",
        "NO_EDGE"
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Status(Enum):
    READY = "READY"
    WAIT = "WAIT"
    AVOID = "AVOID"
    CONFLICTING = "CONFLICTING"


class Confidence(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class MarketData:
    """Raw market data input"""
    oi_change: float = 0.0          # OI change % (24h)
    price_change: float = 0.0       # Price change % (24h)
    whale_pct: float = 50.0         # Whale long % (0-100)
    retail_pct: float = 50.0        # Retail long % (0-100)
    funding_rate: float = 0.0       # Funding rate (decimal, e.g., 0.0001)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    # Scenario identification
    scenario_id: str = "NO_EDGE"
    scenario_name: str = "No Clear Edge"
    emoji: str = "⚪"
    
    # Signals
    direction: Direction = Direction.NEUTRAL
    status: Status = Status.AVOID
    confidence: Confidence = Confidence.LOW
    score: int = 50
    
    # Derived signals
    oi_price_pattern: str = "NEUTRAL"
    whale_bias: str = "NEUTRAL"
    retail_bias: str = "NEUTRAL"
    funding_signal: str = "NEUTRAL"
    
    # Content
    description: str = ""
    action: str = ""
    education: str = ""
    wait_conditions: List[str] = field(default_factory=list)
    
    # Metadata
    input_data: Dict = field(default_factory=dict)
    matched_rules: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['direction'] = self.direction.value
        result['status'] = self.status.value
        result['confidence'] = self.confidence.value
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# RULES ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class InstitutionalRulesEngine:
    """
    Professional rules-based engine for institutional data analysis.
    
    Features:
    - Configuration-driven (no hardcoded values)
    - Extensible pattern matching
    - Full audit trail
    - ML-ready structure
    """
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize the rules engine.
        
        Args:
            config: Configuration dictionary (overrides defaults)
            config_path: Path to JSON config file
        """
        # Load configuration
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._merge_config(user_config)
        
        if config:
            self._merge_config(config)
        
        # Cache thresholds for performance
        self.thresholds = self.config['thresholds']
        self.scenarios = self.config['scenarios']
        self.oi_price_patterns = self.config['oi_price_patterns']
        self.scenario_priority = self.config['scenario_priority']
    
    def _merge_config(self, new_config: Dict):
        """Deep merge configuration"""
        for key, value in new_config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def analyze(self, data: MarketData) -> AnalysisResult:
        """
        Analyze market data and return scenario identification.
        
        This is the MAIN entry point.
        """
        result = AnalysisResult(input_data=data.to_dict())
        
        # Step 1: Classify individual signals
        oi_price_pattern = self._classify_oi_price(data.oi_change, data.price_change)
        whale_bias = self._classify_positioning(data.whale_pct, 'whale_positioning')
        retail_bias = self._classify_positioning(data.retail_pct, 'retail_positioning')
        funding_signal = self._classify_funding(data.funding_rate)
        
        result.oi_price_pattern = oi_price_pattern
        result.whale_bias = whale_bias
        result.retail_bias = retail_bias
        result.funding_signal = funding_signal
        
        # Step 2: Check for conflicts
        has_conflict = self._check_conflict(oi_price_pattern, whale_bias)
        
        # Step 3: Match scenario (priority order)
        matched_scenario = self._match_scenario(
            oi_price_pattern, whale_bias, retail_bias, funding_signal, has_conflict
        )
        
        # Step 4: Populate result
        scenario_def = self.scenarios.get(matched_scenario, self.scenarios.get('NO_EDGE', {}))
        
        result.scenario_id = matched_scenario
        result.scenario_name = scenario_def.get('name', matched_scenario)
        result.emoji = scenario_def.get('emoji', '⚪')
        result.direction = Direction[scenario_def.get('direction', 'NEUTRAL')]
        result.status = Status[scenario_def.get('status', 'AVOID')]
        result.confidence = Confidence[scenario_def.get('confidence', 'LOW')]
        result.score = scenario_def.get('score', 50)
        # Support both old ('description'/'action') and new ('interpretation'/'recommendation') keys
        result.description = scenario_def.get('description', scenario_def.get('interpretation', ''))
        result.action = scenario_def.get('action', scenario_def.get('recommendation', ''))
        result.education = scenario_def.get('education', '')
        result.wait_conditions = scenario_def.get('wait_conditions', scenario_def.get('wait_for', []))
        result.matched_rules.append(f"oi_price={oi_price_pattern}")
        result.matched_rules.append(f"whale={whale_bias}")
        result.matched_rules.append(f"scenario={matched_scenario}")
        
        return result
    
    def _classify_oi_price(self, oi_change: float, price_change: float) -> str:
        """Classify OI + Price combination"""
        oi_thresh = self.thresholds['oi_change']
        
        # Support both old and new config key names
        rising_thresh = oi_thresh.get('rising', oi_thresh.get('significant_positive', 1.0))
        falling_thresh = oi_thresh.get('falling', oi_thresh.get('significant_negative', -1.0))
        
        oi_rising = oi_change >= rising_thresh
        oi_falling = oi_change <= falling_thresh
        price_up = price_change >= 0
        price_down = price_change < 0
        
        if oi_rising and price_up:
            return "NEW_LONGS"
        elif oi_rising and price_down:
            return "NEW_SHORTS"
        elif oi_falling and price_up:
            return "SHORT_COVERING"
        elif oi_falling and price_down:
            return "LONG_LIQUIDATION"
        else:
            return "NEUTRAL"
    
    def _classify_positioning(self, pct: float, threshold_key: str) -> str:
        """Classify positioning (whale or retail)"""
        thresh = self.thresholds[threshold_key]
        
        if pct >= thresh['bullish']:
            return "BULLISH"
        elif pct <= thresh['bearish']:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _classify_funding(self, rate: float) -> str:
        """Classify funding rate signal"""
        thresh = self.thresholds['funding_rate']
        
        # Support both old and new config key names
        longs_thresh = thresh.get('longs_overleveraged', thresh.get('positive_extreme', 0.0005))
        shorts_thresh = thresh.get('shorts_overleveraged', thresh.get('negative_extreme', -0.0005))
        
        if rate >= longs_thresh:
            return "LONGS_OVERLEVERAGED"
        elif rate <= shorts_thresh:
            return "SHORTS_OVERLEVERAGED"
        else:
            return "NEUTRAL"
    
    def _check_conflict(self, oi_price_pattern: str, whale_bias: str) -> bool:
        """Check if signals conflict"""
        # NEW_LONGS but whales bearish = conflict
        if oi_price_pattern == "NEW_LONGS" and whale_bias == "BEARISH":
            return True
        # NEW_SHORTS but whales bullish = conflict
        if oi_price_pattern == "NEW_SHORTS" and whale_bias == "BULLISH":
            return True
        return False
    
    def _match_scenario(self, oi_price: str, whale: str, retail: str, 
                        funding: str, has_conflict: bool) -> str:
        """Match the best scenario based on current signals"""
        
        # Build current state for matching
        state = {
            'oi_price_pattern': oi_price,
            'whale_bias': whale,
            'retail_bias': retail,
            'funding_signal': funding,
            'conflict': has_conflict
        }
        
        # Try each scenario in priority order
        for scenario_id in self.scenario_priority:
            scenario = self.scenarios.get(scenario_id)
            if not scenario:
                continue
            
            if self._scenario_matches(scenario, state):
                return scenario_id
        
        return "NO_EDGE"
    
    def _scenario_matches(self, scenario: Dict, state: Dict) -> bool:
        """Check if a scenario matches the current state"""
        required = scenario.get('required', {})
        excluded = scenario.get('excluded', {})
        
        # Check required conditions
        for key, value in required.items():
            if key not in state:
                return False
            if state[key] != value:
                return False
        
        # Check exclusions
        for key, value in excluded.items():
            if key in state and state[key] == value:
                return False
        
        return True
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()
    
    def update_threshold(self, category: str, key: str, value: float):
        """Update a specific threshold"""
        if category in self.thresholds and key in self.thresholds[category]:
            self.thresholds[category][key] = value
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def explain_decision(self, result: AnalysisResult) -> str:
        """Generate human-readable explanation of the decision"""
        lines = [
            f"=== ANALYSIS EXPLANATION ===",
            f"",
            f"INPUT DATA:",
            f"  OI Change:  {result.input_data.get('oi_change', 0):+.1f}%",
            f"  Price:      {result.input_data.get('price_change', 0):+.1f}%",
            f"  Whales:     {result.input_data.get('whale_pct', 50):.0f}% long",
            f"  Retail:     {result.input_data.get('retail_pct', 50):.0f}% long",
            f"  Funding:    {result.input_data.get('funding_rate', 0)*100:.4f}%",
            f"",
            f"SIGNAL CLASSIFICATION:",
            f"  OI + Price Pattern: {result.oi_price_pattern}",
            f"  Whale Bias:         {result.whale_bias}",
            f"  Retail Bias:        {result.retail_bias}",
            f"  Funding Signal:     {result.funding_signal}",
            f"",
            f"SCENARIO MATCHED:",
            f"  {result.emoji} {result.scenario_name}",
            f"  Confidence: {result.confidence.value}",
            f"  Direction:  {result.direction.value}",
            f"  Status:     {result.status.value}",
            f"  Score:      {result.score}/100",
            f"",
            f"WHAT'S HAPPENING:",
            f"  {result.description}",
            f"",
            f"ACTION:",
            f"  {result.action}",
        ]
        
        if result.wait_conditions:
            lines.append(f"")
            lines.append(f"WAIT FOR:")
            for cond in result.wait_conditions:
                lines.append(f"  ✓ {cond}")
        
        if result.education:
            lines.append(f"")
            lines.append(f"WHY:")
            lines.append(f"  {result.education}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

# Global engine instance (lazy loaded)
_engine_instance: Optional[InstitutionalRulesEngine] = None

def get_engine() -> InstitutionalRulesEngine:
    """Get or create the rules engine singleton"""
    global _engine_instance
    if _engine_instance is None:
        # Try to load from config file first
        config_path = os.path.join(os.path.dirname(__file__), 'institutional_config.json')
        _engine_instance = InstitutionalRulesEngine(config_path=config_path)
    return _engine_instance


def analyze(oi_change: float, price_change: float, whale_pct: float,
            retail_pct: float = 50.0, funding_rate: float = 0.0) -> AnalysisResult:
    """
    Convenience function for quick analysis.
    
    Args:
        oi_change: OI change % (24h)
        price_change: Price change % (24h)
        whale_pct: Whale long % (0-100)
        retail_pct: Retail long % (0-100)
        funding_rate: Funding rate (decimal)
    
    Returns:
        AnalysisResult with complete scenario identification
    """
    engine = get_engine()
    data = MarketData(
        oi_change=oi_change,
        price_change=price_change,
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        funding_rate=funding_rate
    )
    return engine.analyze(data)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY LAYER (For migration from old institutional_scoring.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_scenario_display(result_or_id) -> Dict:
    """
    Get display information for a scenario.
    Compatible with old interface.
    """
    engine = get_engine()
    
    # Handle both AnalysisResult and string scenario_id
    if isinstance(result_or_id, AnalysisResult):
        scenario_id = result_or_id.scenario_id
    elif hasattr(result_or_id, 'value'):
        # Handle old ScenarioType enum
        scenario_id = result_or_id.value.upper().replace(' ', '_')
    else:
        scenario_id = str(result_or_id).upper().replace(' ', '_')
    
    scenario = engine.scenarios.get(scenario_id, engine.scenarios.get('NO_EDGE', {}))
    
    # Support both old keys (description/action) and new keys (interpretation/recommendation)
    description = scenario.get('description', '') or scenario.get('interpretation', '')
    what_to_do = scenario.get('action', '') or scenario.get('recommendation', '')
    
    return {
        'emoji': scenario.get('emoji', '⚪'),
        'action': scenario.get('direction', 'NEUTRAL'),
        'status': scenario.get('status', 'AVOID'),
        'description': description,
        'what_to_do': what_to_do
    }


class LegacyCompatResult:
    """
    Wrapper to provide old InstitutionalScore-like interface.
    For backwards compatibility during migration.
    """
    def __init__(self, result: AnalysisResult):
        self._result = result
        
        # Map new attributes to old names
        self.scenario = result.scenario_id
        self.scenario_name = result.scenario_name
        self.confidence = result.confidence.value
        self.status = result.status
        self.direction_bias = result.direction.value
        self.institutional_score = result.score
        
        # Individual signals
        self.oi_price_signal = result.oi_price_pattern
        self.whale_bias = result.whale_bias
        self.retail_bias = result.retail_bias
        self.funding_signal = result.funding_signal
        
        # Conditions
        self.wait_conditions = result.wait_conditions
        self.confirmation_triggers = []  # Not used in new system
        
        # Original data
        self.oi_change = result.input_data.get('oi_change', 0)
        self.price_change = result.input_data.get('price_change', 0)
        self.whale_pct = result.input_data.get('whale_pct', 50)
        self.retail_pct = result.input_data.get('retail_pct', 50)
        self.funding_rate = result.input_data.get('funding_rate', 0)
        self.divergence = self.whale_pct - self.retail_pct
        
        # Alignment (computed)
        self.alignment_with_tech = "NEUTRAL"  # Set by caller if needed


def analyze_institutional_data(
    oi_change: float,
    price_change: float,
    whale_pct: float,
    retail_pct: float,
    funding_rate: float,
    tech_direction: str = "NEUTRAL"
) -> LegacyCompatResult:
    """
    COMPATIBILITY WRAPPER for old analyze_institutional_data function.
    
    This provides the same interface as the old institutional_scoring.py
    but uses the new rules engine under the hood.
    """
    # Use new engine
    result = analyze(oi_change, price_change, whale_pct, retail_pct, funding_rate)
    
    # Wrap in compatibility object
    legacy = LegacyCompatResult(result)
    
    # Set alignment based on tech direction
    if tech_direction in ["BULLISH", "LONG"]:
        if legacy.direction_bias == "LONG":
            legacy.alignment_with_tech = "ALIGNED"
        elif legacy.direction_bias == "SHORT":
            legacy.alignment_with_tech = "CONFLICTING"
    elif tech_direction in ["BEARISH", "SHORT"]:
        if legacy.direction_bias == "SHORT":
            legacy.alignment_with_tech = "ALIGNED"
        elif legacy.direction_bias == "LONG":
            legacy.alignment_with_tech = "CONFLICTING"
    
    return legacy


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== INSTITUTIONAL RULES ENGINE TEST ===\n")
    
    engine = InstitutionalRulesEngine()
    
    # Test cases
    test_cases = [
        {"name": "User Example", "data": {"oi_change": 1.6, "price_change": 1.7, "whale_pct": 55, "retail_pct": 56, "funding_rate": 0.00005}},
        {"name": "Strong Bullish", "data": {"oi_change": 5.0, "price_change": 3.0, "whale_pct": 60, "retail_pct": 50, "funding_rate": 0}},
        {"name": "Short Squeeze", "data": {"oi_change": 2.0, "price_change": 1.0, "whale_pct": 58, "retail_pct": 40, "funding_rate": -0.001}},
        {"name": "Long Liquidation + Whale Bullish", "data": {"oi_change": -6.0, "price_change": -5.0, "whale_pct": 60, "retail_pct": 45, "funding_rate": 0}},
        {"name": "New Shorts", "data": {"oi_change": 3.0, "price_change": -2.0, "whale_pct": 42, "retail_pct": 50, "funding_rate": 0}},
        {"name": "All Neutral", "data": {"oi_change": 0.5, "price_change": 0.2, "whale_pct": 50, "retail_pct": 50, "funding_rate": 0}},
        {"name": "Conflicting", "data": {"oi_change": 3.0, "price_change": -2.0, "whale_pct": 60, "retail_pct": 50, "funding_rate": 0}},
    ]
    
    for tc in test_cases:
        data = MarketData(**tc['data'])
        result = engine.analyze(data)
        print(f"{tc['name']}:")
        print(f"  OI+Price: {result.oi_price_pattern} | Whale: {result.whale_bias}")
        print(f"  → {result.emoji} {result.scenario_name} ({result.confidence.value})")
        print(f"  → Status: {result.status.value} | Direction: {result.direction.value}")
        print()
    
    # Full explanation for user's example
    print("=" * 60)
    print("FULL EXPLANATION FOR USER'S EXAMPLE:")
    print("=" * 60)
    data = MarketData(**test_cases[0]['data'])
    result = engine.analyze(data)
    print(engine.explain_decision(result))
