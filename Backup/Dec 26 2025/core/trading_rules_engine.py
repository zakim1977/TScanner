"""
Trading Rules Engine
====================
Centralized rule management for InvestorIQ.

Features:
1. YAML-based configuration (easy to edit)
2. Rule priority and chaining
3. Audit trail (which rules fired)
4. Easy A/B testing of parameters
5. Auto-optimization ready (Optuna integration)

This replaces scattered if/else logic with declarative rules.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG FILE PATH
# ═══════════════════════════════════════════════════════════════════════════════

# Look for config in multiple locations
CONFIG_PATHS = [
    'config/trading_config.yaml',           # Standard location
    '../config/trading_config.yaml',        # If running from core/
    'TScanner/config/trading_config.yaml',  # If running from parent
    os.path.join(os.path.dirname(__file__), '..', 'config', 'trading_config.yaml'),  # Relative to this file
]

DEFAULT_CONFIG = """
# InvestorIQ Trading Rules Configuration
# ======================================
# Edit these values to tune the system behavior

version: "1.0"

# ─────────────────────────────────────────────────────────────────────────────
# WHALE THRESHOLDS - When do we consider whales bullish/bearish?
# ─────────────────────────────────────────────────────────────────────────────
whale_thresholds:
  extreme_bullish: 75    # Very high conviction
  strong_bullish: 70     # Strong conviction
  bullish: 65            # Clear bullish
  lean_bullish: 60       # Slight bullish lean
  neutral_high: 55       # Upper neutral
  neutral_low: 45        # Lower neutral  
  lean_bearish: 40       # Slight bearish lean
  bearish: 35            # Clear bearish
  strong_bearish: 30     # Strong conviction
  extreme_bearish: 25    # Very high conviction

# ─────────────────────────────────────────────────────────────────────────────
# DIVERGENCE THRESHOLDS - Whale vs Retail difference
# ─────────────────────────────────────────────────────────────────────────────
divergence_thresholds:
  extreme: 20            # Massive divergence - strong squeeze
  squeeze_setup: 15      # High divergence - squeeze potential
  clear_edge: 10         # Clear edge
  slight_edge: 5         # Minor edge
  no_edge: 0             # No divergence benefit

# ─────────────────────────────────────────────────────────────────────────────
# POSITION THRESHOLDS - Where in the range is price?
# ─────────────────────────────────────────────────────────────────────────────
position_thresholds:
  early_long: 35         # Best for longs (near lows)
  late_long: 65          # Getting late for longs
  early_short: 65        # Best for shorts (near highs)
  late_short: 35         # Getting late for shorts

# ─────────────────────────────────────────────────────────────────────────────
# SCORING WEIGHTS - How much does each factor contribute?
# ─────────────────────────────────────────────────────────────────────────────
scoring_weights:
  layer1_direction:
    max_points: 40
    extreme_conviction: 40
    strong_conviction: 36
    clear_direction: 32
    lean_direction: 28
    slight_lean: 22
    neutral: 15
    
  layer2_squeeze:
    max_points: 30
    extreme_divergence: 22
    high_divergence: 18
    medium_divergence: 14
    low_divergence: 10
    whale_conviction_bonus: 10
    
  layer3_entry:
    max_points: 30
    ta_strong: 15
    ta_good: 12
    ta_moderate: 9
    ta_weak: 6
    position_optimal: 18
    position_middle: 12
    position_late: 5
    position_chasing: 0

# ─────────────────────────────────────────────────────────────────────────────
# PENALTIES - When to reduce scores
# ─────────────────────────────────────────────────────────────────────────────
penalties:
  retail_overleveraged:
    # When retail is MORE positioned than whales - warning!
    threshold_extreme: 70   # Retail >= 70% when going long
    threshold_high: 65
    threshold_moderate: 60
    penalty_extreme: 12
    penalty_high: 8
    penalty_moderate: 5
    
  counter_trend:
    # Going against strong structure
    penalty: 10
    
  chasing:
    # Entering at poor position
    penalty: 8

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL RULES - Priority-ordered rules for signal generation
# ─────────────────────────────────────────────────────────────────────────────
signal_rules:
  # High priority rules checked FIRST
  - name: "CAPITULATION_LONG"
    priority: 100
    conditions:
      money_flow_phase: "CAPITULATION"
      whale_pct_min: 65
      position_pct_max: 35
    action: "EARLY_LONG"
    confidence: "HIGH"
    reason: "Capitulation + Whales positioned + Near lows = REVERSAL"
    
  - name: "RETAIL_TRAP_WARNING"
    priority: 95
    conditions:
      whale_pct_min: 55
      whale_pct_max: 68
      retail_pct_min: 70
      divergence_max: -5
    action: "WAIT"
    confidence: "MEDIUM"
    reason: "Retail overleveraged - whales not as bullish"
    reduce_score: true
    penalty: 15
    
  - name: "SHORT_SQUEEZE_SETUP"
    priority: 90
    conditions:
      whale_pct_min: 70
      retail_pct_max: 50
      divergence_min: 15
    action: "STRONG_LONG"
    confidence: "HIGH"
    reason: "Massive divergence - shorts about to be liquidated"
    
  - name: "LONG_SQUEEZE_SETUP"
    priority: 90
    conditions:
      whale_pct_max: 30
      retail_pct_min: 50
      divergence_max: -15
    action: "STRONG_SHORT"
    confidence: "HIGH"
    reason: "Massive divergence - longs about to be liquidated"
    
  - name: "DISTRIBUTION_SHORT"
    priority: 85
    conditions:
      money_flow_phase: "DISTRIBUTION"
      whale_pct_max: 35
      position_pct_min: 65
    action: "EARLY_SHORT"
    confidence: "HIGH"
    reason: "Distribution + Whales short + Near highs = TOP"

# ─────────────────────────────────────────────────────────────────────────────
# BTC CORRELATION - How to handle altcoin/BTC relationship
# ─────────────────────────────────────────────────────────────────────────────
btc_correlation:
  critical_threshold: 0.85    # Very high correlation
  high_threshold: 0.70        # High correlation
  moderate_threshold: 0.50    # Moderate correlation
  low_threshold: 0.30         # Low correlation
  
  # Adjustments based on alignment
  aligned_bonus: 5
  counter_trend_penalty: 10
  ignore_warning_below: 0.50  # Don't warn if correlation is low
"""


@dataclass
class RuleResult:
    """Result of evaluating a rule"""
    rule_name: str
    matched: bool
    action: Optional[str] = None
    confidence: Optional[str] = None
    reason: Optional[str] = None
    score_adjustment: int = 0
    priority: int = 0


@dataclass 
class EvaluationContext:
    """All the data needed to evaluate rules"""
    whale_pct: float = 50
    retail_pct: float = 50
    divergence: float = 0
    oi_change: float = 0
    price_change: float = 0
    position_pct: float = 50
    ta_score: float = 50
    money_flow_phase: str = "UNKNOWN"
    structure: str = "UNKNOWN"
    btc_correlation: float = 0.75
    btc_trend: str = "NEUTRAL"
    
    def to_dict(self) -> Dict:
        return {
            'whale_pct': self.whale_pct,
            'retail_pct': self.retail_pct,
            'divergence': self.divergence,
            'oi_change': self.oi_change,
            'price_change': self.price_change,
            'position_pct': self.position_pct,
            'ta_score': self.ta_score,
            'money_flow_phase': self.money_flow_phase,
            'structure': self.structure,
            'btc_correlation': self.btc_correlation,
            'btc_trend': self.btc_trend,
        }


class TradingRulesEngine:
    """
    Central rules engine for InvestorIQ.
    
    Usage:
        engine = TradingRulesEngine()
        engine.load_config()  # or load_config('custom_config.yaml')
        
        context = EvaluationContext(
            whale_pct=63,
            retail_pct=70,
            divergence=-7,
            ...
        )
        
        result = engine.evaluate(context)
        print(result.fired_rules)
        print(result.final_action)
        print(result.score_adjustments)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict = {}
        self.config_path = config_path
        self._rule_cache: Dict = {}
        
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file or use defaults"""
        path = config_path or self.config_path
        
        # If no path specified, search for config file
        if not path:
            for search_path in CONFIG_PATHS:
                if os.path.exists(search_path):
                    path = search_path
                    break
        
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f)
                print(f"✅ Loaded trading config from: {path}")
        else:
            # Use default config
            self.config = yaml.safe_load(DEFAULT_CONFIG)
            
        # Build rule cache for fast lookup
        self._build_rule_cache()
    
    def save_config(self, path: str) -> None:
        """Save current config to file"""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Example:
            engine.get('whale_thresholds.strong_bullish')  # Returns 70
            engine.get('penalties.retail_overleveraged.threshold_extreme')  # Returns 70
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self._build_rule_cache()  # Rebuild cache
    
    def _build_rule_cache(self) -> None:
        """Pre-compile rules for faster evaluation"""
        self._rule_cache = {}
        rules = self.config.get('signal_rules', [])
        
        # Sort by priority (highest first)
        self._sorted_rules = sorted(rules, key=lambda r: r.get('priority', 0), reverse=True)
    
    def _check_condition(self, condition: str, value: Any, context: EvaluationContext) -> bool:
        """Check if a single condition is met"""
        ctx_dict = context.to_dict()
        
        # Parse condition like "whale_pct_min" -> check if whale_pct >= value
        if condition.endswith('_min'):
            field = condition[:-4]
            return ctx_dict.get(field, 0) >= value
        elif condition.endswith('_max'):
            field = condition[:-4]
            return ctx_dict.get(field, 100) <= value
        elif condition in ctx_dict:
            return ctx_dict[condition] == value
        
        return False
    
    def _evaluate_rule(self, rule: Dict, context: EvaluationContext) -> RuleResult:
        """Evaluate a single rule against context"""
        conditions = rule.get('conditions', {})
        
        # Check all conditions
        all_match = True
        for cond_name, cond_value in conditions.items():
            if not self._check_condition(cond_name, cond_value, context):
                all_match = False
                break
        
        if all_match:
            return RuleResult(
                rule_name=rule.get('name', 'UNKNOWN'),
                matched=True,
                action=rule.get('action'),
                confidence=rule.get('confidence'),
                reason=rule.get('reason'),
                score_adjustment=-rule.get('penalty', 0) if rule.get('reduce_score') else 0,
                priority=rule.get('priority', 0)
            )
        
        return RuleResult(rule_name=rule.get('name', 'UNKNOWN'), matched=False)
    
    def evaluate(self, context: EvaluationContext) -> Dict:
        """
        Evaluate all rules and return comprehensive result.
        
        Returns:
            {
                'fired_rules': [RuleResult, ...],
                'final_action': str,
                'final_confidence': str,
                'reasons': [str, ...],
                'score_adjustment': int,
                'audit_trail': [...]
            }
        """
        fired_rules: List[RuleResult] = []
        audit_trail: List[str] = []
        
        # Evaluate rules in priority order
        for rule in self._sorted_rules:
            result = self._evaluate_rule(rule, context)
            audit_trail.append(f"Rule '{rule.get('name')}': {'MATCHED' if result.matched else 'not matched'}")
            
            if result.matched:
                fired_rules.append(result)
        
        # Determine final action from highest priority matched rule
        final_action = None
        final_confidence = None
        reasons = []
        total_adjustment = 0
        
        for rule in fired_rules:
            if rule.action and final_action is None:
                final_action = rule.action
                final_confidence = rule.confidence
            if rule.reason:
                reasons.append(rule.reason)
            total_adjustment += rule.score_adjustment
        
        return {
            'fired_rules': fired_rules,
            'fired_rule_names': [r.rule_name for r in fired_rules],
            'final_action': final_action,
            'final_confidence': final_confidence,
            'reasons': reasons,
            'score_adjustment': total_adjustment,
            'audit_trail': audit_trail,
        }
    
    def get_whale_direction(self, whale_pct: float) -> Dict:
        """Get direction based on whale positioning using config thresholds"""
        wt = self.config.get('whale_thresholds', {})
        sw = self.config.get('scoring_weights', {}).get('layer1_direction', {})
        
        if whale_pct >= wt.get('extreme_bullish', 75):
            return {'direction': 'BULLISH', 'confidence': 'HIGH', 'score': sw.get('extreme_conviction', 40)}
        elif whale_pct >= wt.get('strong_bullish', 70):
            return {'direction': 'BULLISH', 'confidence': 'HIGH', 'score': sw.get('strong_conviction', 36)}
        elif whale_pct >= wt.get('bullish', 65):
            return {'direction': 'BULLISH', 'confidence': 'HIGH', 'score': sw.get('clear_direction', 32)}
        elif whale_pct >= wt.get('lean_bullish', 60):
            return {'direction': 'LEAN_BULLISH', 'confidence': 'MEDIUM', 'score': sw.get('lean_direction', 28)}
        elif whale_pct >= wt.get('neutral_high', 55):
            return {'direction': 'LEAN_BULLISH', 'confidence': 'LOW', 'score': sw.get('slight_lean', 22)}
        elif whale_pct <= wt.get('extreme_bearish', 25):
            return {'direction': 'BEARISH', 'confidence': 'HIGH', 'score': sw.get('extreme_conviction', 40)}
        elif whale_pct <= wt.get('strong_bearish', 30):
            return {'direction': 'BEARISH', 'confidence': 'HIGH', 'score': sw.get('strong_conviction', 36)}
        elif whale_pct <= wt.get('bearish', 35):
            return {'direction': 'BEARISH', 'confidence': 'HIGH', 'score': sw.get('clear_direction', 32)}
        elif whale_pct <= wt.get('lean_bearish', 40):
            return {'direction': 'LEAN_BEARISH', 'confidence': 'MEDIUM', 'score': sw.get('lean_direction', 28)}
        elif whale_pct <= wt.get('neutral_low', 45):
            return {'direction': 'LEAN_BEARISH', 'confidence': 'LOW', 'score': sw.get('slight_lean', 22)}
        else:
            return {'direction': 'NEUTRAL', 'confidence': 'LOW', 'score': sw.get('neutral', 15)}
    
    def check_retail_warning(self, whale_pct: float, retail_pct: float, direction: str) -> Dict:
        """Check if retail is overleveraged (warning signal)"""
        penalties = self.config.get('penalties', {}).get('retail_overleveraged', {})
        divergence = whale_pct - retail_pct
        
        result = {'warning': False, 'penalty': 0, 'reason': ''}
        
        # Only warn if divergence is negative (retail more positioned)
        if divergence >= -5:
            return result
        
        # Check for long positions
        if direction in ['BULLISH', 'LEAN_BULLISH']:
            if retail_pct >= penalties.get('threshold_extreme', 70):
                result = {
                    'warning': True,
                    'penalty': penalties.get('penalty_extreme', 12),
                    'reason': f'⚠️ Retail ({retail_pct:.0f}%) MORE bullish than Whales ({whale_pct:.0f}%)! Overleveraged.'
                }
            elif retail_pct >= penalties.get('threshold_high', 65):
                result = {
                    'warning': True,
                    'penalty': penalties.get('penalty_high', 8),
                    'reason': f'⚠️ Retail ({retail_pct:.0f}%) more bullish than Whales ({whale_pct:.0f}%)'
                }
            elif retail_pct >= penalties.get('threshold_moderate', 60):
                result = {
                    'warning': True,
                    'penalty': penalties.get('penalty_moderate', 5),
                    'reason': f'Note: Retail slightly more bullish ({retail_pct:.0f}% vs {whale_pct:.0f}%)'
                }
        
        # Check for short positions
        elif direction in ['BEARISH', 'LEAN_BEARISH']:
            retail_short = 100 - retail_pct
            whale_short = 100 - whale_pct
            if retail_short >= penalties.get('threshold_extreme', 70):
                result = {
                    'warning': True,
                    'penalty': penalties.get('penalty_extreme', 12),
                    'reason': f'⚠️ Retail ({retail_short:.0f}%S) MORE bearish than Whales ({whale_short:.0f}%S)! Overleveraged.'
                }
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[TradingRulesEngine] = None

def get_rules_engine() -> TradingRulesEngine:
    """Get or create the singleton rules engine"""
    global _engine
    if _engine is None:
        _engine = TradingRulesEngine()
        _engine.load_config()
    return _engine


def get_config(key_path: str, default: Any = None) -> Any:
    """Convenience function to get config values"""
    return get_rules_engine().get(key_path, default)


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA INTEGRATION (for auto-tuning)
# ═══════════════════════════════════════════════════════════════════════════════

def create_optuna_study(backtest_fn: Callable, n_trials: int = 100):
    """
    Create an Optuna study to optimize trading parameters.
    
    Usage:
        def backtest(params):
            # Run backtest with params, return win_rate
            return win_rate
            
        best_params = create_optuna_study(backtest, n_trials=100)
    """
    try:
        import optuna
    except ImportError:
        print("Install optuna: pip install optuna")
        return None
    
    def objective(trial):
        params = {
            'whale_strong_bullish': trial.suggest_int('whale_strong_bullish', 65, 80),
            'whale_lean_bullish': trial.suggest_int('whale_lean_bullish', 55, 65),
            'divergence_squeeze': trial.suggest_int('divergence_squeeze', 10, 25),
            'divergence_edge': trial.suggest_int('divergence_edge', 5, 15),
            'retail_warning_threshold': trial.suggest_int('retail_warning_threshold', 65, 80),
            'position_early_long': trial.suggest_int('position_early_long', 25, 45),
        }
        
        return backtest_fn(params)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Use these throughout the app!
# ═══════════════════════════════════════════════════════════════════════════════

def whale_threshold(level: str) -> float:
    """
    Get whale threshold value.
    
    Usage:
        if whale_pct >= whale_threshold('strong_bullish'):
            # Strong bullish...
    
    Levels: extreme_bullish, strong_bullish, bullish, lean_bullish, 
            neutral_high, neutral_low, lean_bearish, bearish, 
            strong_bearish, extreme_bearish
    """
    return get_config(f'whale_thresholds.{level}', 50)


def divergence_threshold(level: str) -> float:
    """
    Get divergence threshold value.
    
    Levels: extreme, squeeze_setup, clear_edge, slight_edge, no_edge
    """
    return get_config(f'divergence_thresholds.{level}', 10)


def position_threshold(level: str) -> float:
    """
    Get position threshold value.
    
    Levels: early_long, late_long, early_short, late_short
    """
    return get_config(f'position_thresholds.{level}', 50)


def scoring_points(layer: str, level: str) -> int:
    """
    Get scoring points for a specific level.
    
    Usage:
        points = scoring_points('layer1_direction', 'strong_conviction')  # Returns 36
        points = scoring_points('layer2_squeeze', 'extreme_divergence')   # Returns 22
    """
    return get_config(f'scoring.{layer}.{level}', 10)


def penalty_value(category: str, level: str) -> int:
    """
    Get penalty value.
    
    Usage:
        penalty = penalty_value('retail_overleveraged', 'extreme_penalty')  # Returns 12
    """
    return get_config(f'penalties.{category}.{level}', 5)


def trading_mode_config(mode: str) -> Dict:
    """
    Get trading mode configuration.
    
    Usage:
        config = trading_mode_config('day_trade')
        max_sl = config['max_stop_loss_pct']  # 4.0
    """
    return get_config(f'trading_modes.{mode}', {})


def btc_correlation_threshold(level: str) -> float:
    """
    Get BTC correlation threshold.
    
    Levels: critical, high, moderate, low
    """
    return get_config(f'btc_correlation.{level}', 0.5)


def filter_value(key: str) -> Any:
    """
    Get filter value.
    
    Keys: min_score_to_show, min_rr_tp1, min_confidence, require_whale_data
    """
    return get_config(f'filters.{key}')


def evaluate_signal(
    whale_pct: float,
    retail_pct: float,
    position_pct: float = 50,
    ta_score: float = 50,
    oi_change: float = 0,
    price_change: float = 0,
    money_flow_phase: str = "UNKNOWN",
    structure: str = "UNKNOWN",
    btc_correlation: float = 0.75,
    btc_trend: str = "NEUTRAL"
) -> Dict:
    """
    Convenience function to evaluate a signal with all rules.
    
    Usage:
        result = evaluate_signal(
            whale_pct=63,
            retail_pct=70,
            position_pct=17,
            ta_score=65
        )
        
        print(result['final_action'])      # 'WAIT'
        print(result['fired_rule_names'])  # ['RETAIL_TRAP_LONG']
        print(result['score_adjustment'])  # -15
    """
    engine = get_rules_engine()
    
    context = EvaluationContext(
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        divergence=whale_pct - retail_pct,
        oi_change=oi_change,
        price_change=price_change,
        position_pct=position_pct,
        ta_score=ta_score,
        money_flow_phase=money_flow_phase,
        structure=structure,
        btc_correlation=btc_correlation,
        btc_trend=btc_trend,
    )
    
    return engine.evaluate(context)


def get_direction_and_score(whale_pct: float, retail_pct: float) -> Dict:
    """
    Get direction, score, and any warnings based on whale/retail positioning.
    
    Returns:
        {
            'direction': 'LEAN_BULLISH',
            'confidence': 'MEDIUM',
            'base_score': 28,
            'warning': {...} or None,
            'adjusted_score': 18,
            'reason': '...'
        }
    """
    engine = get_rules_engine()
    
    # Get base direction from whales
    direction_info = engine.get_whale_direction(whale_pct)
    
    # Check for retail warning
    warning = engine.check_retail_warning(whale_pct, retail_pct, direction_info['direction'])
    
    # Calculate adjusted score
    adjusted_score = direction_info['score']
    if warning['warning']:
        adjusted_score = max(15, adjusted_score - warning['penalty'])
    
    return {
        'direction': direction_info['direction'],
        'confidence': direction_info['confidence'],
        'base_score': direction_info['score'],
        'warning': warning if warning['warning'] else None,
        'adjusted_score': adjusted_score,
        'reason': warning['reason'] if warning['warning'] else f"Whales {whale_pct:.0f}% positioned",
    }


def should_warn_retail(whale_pct: float, retail_pct: float, direction: str) -> bool:
    """
    Quick check if retail is overleveraged.
    
    Usage:
        if should_warn_retail(63, 70, 'LEAN_BULLISH'):
            st.warning("Retail overleveraged!")
    """
    engine = get_rules_engine()
    warning = engine.check_retail_warning(whale_pct, retail_pct, direction)
    return warning['warning']


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    engine = TradingRulesEngine()
    engine.load_config()
    
    # Test with PORTALUSDT scenario
    context = EvaluationContext(
        whale_pct=63,
        retail_pct=70,
        divergence=-7,
        position_pct=17,
        ta_score=65,
        money_flow_phase="ACCUMULATION"
    )
    
    print("=" * 60)
    print("Testing PORTALUSDT Scenario")
    print("=" * 60)
    print(f"Whale: {context.whale_pct}%, Retail: {context.retail_pct}%")
    print(f"Divergence: {context.divergence}%")
    print()
    
    # Get direction
    direction = engine.get_whale_direction(context.whale_pct)
    print(f"Direction: {direction}")
    
    # Check retail warning
    warning = engine.check_retail_warning(context.whale_pct, context.retail_pct, direction['direction'])
    print(f"Retail Warning: {warning}")
    
    # Evaluate all rules
    result = engine.evaluate(context)
    print()
    print("Fired Rules:", result['fired_rule_names'])
    print("Final Action:", result['final_action'])
    print("Score Adjustment:", result['score_adjustment'])
    print("Reasons:", result['reasons'])
