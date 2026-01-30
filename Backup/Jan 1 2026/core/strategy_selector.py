"""
Strategy Selector Module for InvestorIQ
========================================
Intelligently selects the best entry/exit strategy based on:
- Trading Mode (Scalp, DayTrade, Swing, Investment)
- Timeframe
- Market Structure
- Available SMC levels
- Volatility conditions

Strategies Supported:
- OB: Order Block entry (wait for price to return to OB)
- LH/HL: Lower High / Higher Low structure entry
- FVG: Fair Value Gap fill entry
- VWAP: Volume Weighted Average Price bounce/break
- EN: Engulfing pattern entry
- SB: Structure Break momentum entry
- LIQ: Liquidity sweep entry (stop hunt reversal)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class TradingMode(Enum):
    SCALP = "scalp"
    DAY_TRADE = "day_trade"
    SWING = "swing"
    INVESTMENT = "investment"


class EntryStrategy(Enum):
    OB = "Order Block"
    OB_HTF = "HTF Order Block"
    LH = "Lower High"
    HL = "Higher Low"
    FVG = "Fair Value Gap"
    VWAP = "VWAP"
    EN = "Engulfing"
    SB = "Structure Break"
    LIQ = "Liquidity Sweep"
    MARKET = "Market Entry"


@dataclass
class StrategyRecommendation:
    """Result of strategy selection"""
    primary_strategy: EntryStrategy
    secondary_strategy: Optional[EntryStrategy]
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confidence: float  # 0-100
    reasoning: str
    entry_type: str  # 'limit' or 'market'
    time_in_force: str  # 'GTC', 'IOC', 'DAY'
    max_wait_candles: int  # How long to wait for limit fill
    strategy_tags: List[str]  # e.g., ['OB', 'LH', 'VWAP']


class StrategySelector:
    """
    Intelligently selects the best trading strategy based on conditions.
    
    Usage:
        selector = StrategySelector(df, mode=TradingMode.DAY_TRADE, timeframe='15m')
        recommendation = selector.get_best_strategy(direction='LONG', smc_data=smc)
    """
    
    # Strategy suitability matrix by trading mode
    MODE_STRATEGIES = {
        TradingMode.SCALP: {
            'preferred': [EntryStrategy.FVG, EntryStrategy.OB, EntryStrategy.EN],
            'acceptable': [EntryStrategy.VWAP, EntryStrategy.LIQ],
            'avoid': [EntryStrategy.OB_HTF, EntryStrategy.SB],
            'max_wait_candles': 3,
            'entry_preference': 'limit',  # Better R:R
        },
        TradingMode.DAY_TRADE: {
            'preferred': [EntryStrategy.OB, EntryStrategy.FVG, EntryStrategy.LH, EntryStrategy.HL],
            'acceptable': [EntryStrategy.VWAP, EntryStrategy.EN, EntryStrategy.LIQ],
            'avoid': [EntryStrategy.MARKET],
            'max_wait_candles': 8,
            'entry_preference': 'limit',
        },
        TradingMode.SWING: {
            'preferred': [EntryStrategy.OB_HTF, EntryStrategy.OB, EntryStrategy.LH, EntryStrategy.HL],
            'acceptable': [EntryStrategy.FVG, EntryStrategy.SB, EntryStrategy.LIQ],
            'avoid': [EntryStrategy.EN],  # Too short-term
            'max_wait_candles': 20,
            'entry_preference': 'limit',
        },
        TradingMode.INVESTMENT: {
            'preferred': [EntryStrategy.OB_HTF, EntryStrategy.LH, EntryStrategy.HL],
            'acceptable': [EntryStrategy.VWAP, EntryStrategy.SB],
            'avoid': [EntryStrategy.FVG, EntryStrategy.EN, EntryStrategy.LIQ],
            'max_wait_candles': 50,
            'entry_preference': 'limit',
        },
    }
    
    # Timeframe to mode mapping
    TIMEFRAME_MODE_MAP = {
        '1m': TradingMode.SCALP,
        '5m': TradingMode.SCALP,
        '15m': TradingMode.DAY_TRADE,
        '1h': TradingMode.DAY_TRADE,
        '4h': TradingMode.SWING,
        '1d': TradingMode.SWING,
        '1w': TradingMode.INVESTMENT,
    }
    
    def __init__(
        self,
        df: pd.DataFrame,
        mode: Optional[TradingMode] = None,
        timeframe: str = '15m',
        htf_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize Strategy Selector.
        
        Args:
            df: OHLCV DataFrame for analysis timeframe
            mode: Trading mode (auto-detected from timeframe if not provided)
            timeframe: Current timeframe string
            htf_df: Higher timeframe DataFrame (optional, for HTF OB)
        """
        self.df = df
        self.timeframe = timeframe
        self.htf_df = htf_df
        
        # Auto-detect mode from timeframe if not provided
        if mode is None:
            self.mode = self.TIMEFRAME_MODE_MAP.get(timeframe, TradingMode.DAY_TRADE)
        else:
            self.mode = mode
            
        # Calculate basic metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate ATR, volatility, and other metrics"""
        if self.df is None or len(self.df) < 20:
            self.atr = 0
            self.volatility = 0
            self.current_price = 0
            return
            
        # ATR
        high = self.df['High'] if 'High' in self.df.columns else self.df['high']
        low = self.df['Low'] if 'Low' in self.df.columns else self.df['low']
        close = self.df['Close'] if 'Close' in self.df.columns else self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = tr.rolling(14).mean().iloc[-1]
        
        # Volatility (ATR as % of price)
        self.current_price = close.iloc[-1]
        self.volatility = (self.atr / self.current_price) * 100 if self.current_price > 0 else 0
        
        # Recent high/low
        self.recent_high = high.iloc[-20:].max()
        self.recent_low = low.iloc[-20:].min()
        
        # Backtested results cache
        self._backtest_results = None
    
    def run_backtest(self) -> Dict:
        """
        Run backtest on historical data to find best strategy and optimal levels.
        Results are cached for reuse.
        
        Returns:
            Dict with best_strategy, win_rate, optimal levels
        """
        if self._backtest_results is not None:
            return self._backtest_results
        
        try:
            from .strategy_backtester import backtest_symbol
            
            # Need at least 200 candles for meaningful backtest
            if len(self.df) < 200:
                return self._default_backtest_result()
            
            result = backtest_symbol(self.df, self.timeframe)
            self._backtest_results = result
            return result
            
        except Exception as e:
            # If backtester not available or fails, return defaults
            print(f"Backtest error: {e}")
            return self._default_backtest_result()
    
    def _default_backtest_result(self) -> Dict:
        """Default backtest result when backtesting unavailable"""
        return {
            'best_strategy': 'OB',
            'win_rate': 50.0,
            'profit_factor': 1.0,
            'optimal_sl_pct': 1.5,
            'optimal_tp1_pct': 2.0,
            'optimal_tp2_pct': 3.5,
            'optimal_tp3_pct': 5.0,
            'all_results': {}
        }
    
    def get_backtested_levels(self) -> Tuple[float, float, float, float]:
        """
        Get optimal SL/TP levels from backtesting.
        
        Returns:
            (sl_pct, tp1_pct, tp2_pct, tp3_pct)
        """
        bt_result = self.run_backtest()
        return (
            bt_result.get('optimal_sl_pct', 1.5),
            bt_result.get('optimal_tp1_pct', 2.0),
            bt_result.get('optimal_tp2_pct', 3.5),
            bt_result.get('optimal_tp3_pct', 5.0),
        )
    
    def get_best_strategy_from_backtest(self) -> str:
        """Get the best performing strategy from backtesting"""
        bt_result = self.run_backtest()
        return bt_result.get('best_strategy', 'OB')
    
    def _calculate_vwap(self) -> float:
        """Calculate VWAP"""
        if self.df is None or 'Volume' not in self.df.columns and 'volume' not in self.df.columns:
            return 0
            
        vol_col = 'Volume' if 'Volume' in self.df.columns else 'volume'
        high = self.df['High'] if 'High' in self.df.columns else self.df['high']
        low = self.df['Low'] if 'Low' in self.df.columns else self.df['low']
        close = self.df['Close'] if 'Close' in self.df.columns else self.df['close']
        
        typical_price = (high + low + close) / 3
        vwap = (typical_price * self.df[vol_col]).cumsum() / self.df[vol_col].cumsum()
        return vwap.iloc[-1]
    
    def _score_strategy(
        self,
        strategy: EntryStrategy,
        direction: str,
        smc_data: Dict,
        entry_level: float,
    ) -> Tuple[float, str]:
        """
        Score a strategy from 0-100 based on current conditions.
        
        Returns:
            Tuple of (score, reasoning)
        """
        score = 50  # Base score
        reasons = []
        mode_config = self.MODE_STRATEGIES[self.mode]
        
        # Mode preference bonus/penalty
        if strategy in mode_config['preferred']:
            score += 20
            reasons.append(f"Preferred for {self.mode.value}")
        elif strategy in mode_config['acceptable']:
            score += 5
            reasons.append(f"Acceptable for {self.mode.value}")
        elif strategy in mode_config['avoid']:
            score -= 30
            reasons.append(f"Not ideal for {self.mode.value}")
        
        # Entry level quality (distance from current price)
        if entry_level > 0 and self.current_price > 0:
            distance_pct = abs(self.current_price - entry_level) / self.current_price * 100
            
            # Sweet spot: 0.5% - 2% for most modes
            if self.mode == TradingMode.SCALP:
                if 0.1 <= distance_pct <= 0.5:
                    score += 15
                    reasons.append(f"Optimal scalp distance ({distance_pct:.2f}%)")
                elif distance_pct > 1.0:
                    score -= 10
                    reasons.append(f"Too far for scalp ({distance_pct:.2f}%)")
            elif self.mode == TradingMode.DAY_TRADE:
                if 0.3 <= distance_pct <= 1.5:
                    score += 15
                    reasons.append(f"Good pullback level ({distance_pct:.2f}%)")
                elif distance_pct > 3.0:
                    score -= 10
                    reasons.append(f"Far entry ({distance_pct:.2f}%)")
            elif self.mode == TradingMode.SWING:
                if 0.5 <= distance_pct <= 3.0:
                    score += 15
                    reasons.append(f"Quality swing entry ({distance_pct:.2f}%)")
        
        # Strategy-specific scoring
        if strategy == EntryStrategy.OB:
            # Check if OB exists and is fresh
            if smc_data.get('bullish_ob' if direction == 'LONG' else 'bearish_ob'):
                score += 15
                reasons.append("Valid OB detected")
            else:
                score -= 20
                reasons.append("No OB available")
                
        elif strategy == EntryStrategy.OB_HTF:
            if smc_data.get('htf_bullish_ob' if direction == 'LONG' else 'htf_bearish_ob'):
                score += 20
                reasons.append("HTF OB confluence")
            else:
                score -= 15
                reasons.append("No HTF OB")
                
        elif strategy == EntryStrategy.FVG:
            if smc_data.get('fvg_bullish' if direction == 'LONG' else 'fvg_bearish'):
                score += 12
                reasons.append("FVG present")
                # FVG works better in trending markets
                if self.volatility > 2:
                    score += 5
                    reasons.append("Good volatility for FVG")
            else:
                score -= 15
                reasons.append("No FVG")
                
        elif strategy == EntryStrategy.VWAP:
            vwap = self._calculate_vwap()
            if vwap > 0:
                vwap_distance = abs(self.current_price - vwap) / self.current_price * 100
                if vwap_distance < 1.0:
                    score += 15
                    reasons.append(f"Near VWAP ({vwap_distance:.2f}%)")
                elif vwap_distance < 2.0:
                    score += 8
                    reasons.append(f"VWAP reachable ({vwap_distance:.2f}%)")
                else:
                    score -= 10
                    reasons.append(f"Far from VWAP ({vwap_distance:.2f}%)")
                    
        elif strategy == EntryStrategy.LH or strategy == EntryStrategy.HL:
            # Check structure
            if smc_data.get('structure') == 'bullish' and strategy == EntryStrategy.HL:
                score += 15
                reasons.append("HL in bullish structure")
            elif smc_data.get('structure') == 'bearish' and strategy == EntryStrategy.LH:
                score += 15
                reasons.append("LH in bearish structure")
            else:
                score -= 10
                reasons.append("Structure doesn't match")
                
        elif strategy == EntryStrategy.LIQ:
            # Liquidity sweep works better at extremes
            if smc_data.get('liquidity_swept'):
                score += 20
                reasons.append("Recent liquidity sweep detected")
            else:
                score -= 15
                reasons.append("No liquidity sweep")
                
        elif strategy == EntryStrategy.EN:
            # Engulfing is momentum-based
            if smc_data.get('engulfing_bullish' if direction == 'LONG' else 'engulfing_bearish'):
                score += 12
                reasons.append("Engulfing pattern")
            else:
                score -= 10
                reasons.append("No engulfing")
                
        elif strategy == EntryStrategy.SB:
            # Structure break for momentum
            if smc_data.get('bos') or smc_data.get('choch'):
                score += 15
                reasons.append("Structure break confirmed")
            else:
                score -= 10
                reasons.append("No structure break")
        
        # Cap score
        score = max(0, min(100, score))
        
        return score, " | ".join(reasons)
    
    def get_best_strategy(
        self,
        direction: str,
        smc_data: Dict,
        current_price: Optional[float] = None,
    ) -> StrategyRecommendation:
        """
        Get the best strategy recommendation based on current conditions.
        
        Args:
            direction: 'LONG' or 'SHORT'
            smc_data: Dict containing SMC analysis (OBs, FVGs, structure, etc.)
            current_price: Current price (uses last close if not provided)
            
        Returns:
            StrategyRecommendation with full trade setup
        """
        if current_price:
            self.current_price = current_price
            
        mode_config = self.MODE_STRATEGIES[self.mode]
        
        # Get all possible entry levels
        entry_levels = self._get_entry_levels(direction, smc_data)
        
        # Score each strategy
        scored_strategies = []
        for strategy, entry_level in entry_levels.items():
            if entry_level is None or entry_level <= 0:
                continue
            
            # Entry levels are already validated in _get_entry_levels()
            # All entries here are valid for direction
                
            score, reasoning = self._score_strategy(
                strategy, direction, smc_data, entry_level
            )
            scored_strategies.append({
                'strategy': strategy,
                'entry': entry_level,
                'score': score,
                'reasoning': reasoning,
            })
        
        # Sort by score
        scored_strategies.sort(key=lambda x: x['score'], reverse=True)
        
        # Get best and second best
        if not scored_strategies:
            # Fallback to market entry
            return self._create_market_entry(direction, smc_data)
            
        best = scored_strategies[0]
        secondary = scored_strategies[1] if len(scored_strategies) > 1 else None
        
        # Calculate SL and TPs
        # CRITICAL: 
        # - SL is relative to ENTRY (limit entry - where we'll actually buy)
        # - TPs should target resistance levels above CURRENT PRICE (realistic targets)
        # - R:R will be calculated from limit entry (better R:R is the whole point!)
        #
        # We pass current_price so TPs target levels above market (not just above limit)
        # But SL is still calculated from the actual entry level
        sl, tp1, tp2, tp3 = self._calculate_levels(
            direction, best['entry'], smc_data, tp_ref_price=self.current_price
        )
        
        # Determine entry type
        entry_type = 'limit' if best['strategy'] != EntryStrategy.MARKET else 'market'
        
        # Build strategy tags
        tags = [best['strategy'].value]
        if secondary:
            tags.append(secondary['strategy'].value)
        
        return StrategyRecommendation(
            primary_strategy=best['strategy'],
            secondary_strategy=secondary['strategy'] if secondary else None,
            entry_price=best['entry'],
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            confidence=best['score'],
            reasoning=best['reasoning'],
            entry_type=entry_type,
            time_in_force='GTC' if self.mode in [TradingMode.SWING, TradingMode.INVESTMENT] else 'DAY',
            max_wait_candles=mode_config['max_wait_candles'],
            strategy_tags=tags,
        )
    
    def _get_entry_levels(self, direction: str, smc_data: Dict) -> Dict[EntryStrategy, float]:
        """
        Get potential entry levels for each strategy.
        
        CRITICAL: All limit entries must be BETTER than market:
        - LONG: Entry must be BELOW current price (buy cheaper)
        - SHORT: Entry must be ABOVE current price (sell higher)
        """
        levels = {}
        
        # Order Block
        if direction == 'LONG':
            ob_top = smc_data.get('bullish_ob_top', 0)
            ob_bottom = smc_data.get('bullish_ob_bottom', 0)
            # LONG: OB must be BELOW current price
            if ob_top > 0 and ob_top < self.current_price:
                levels[EntryStrategy.OB] = ob_top
            elif ob_bottom > 0 and ob_bottom < self.current_price:
                levels[EntryStrategy.OB] = ob_bottom
        else:
            ob_top = smc_data.get('bearish_ob_top', 0)
            ob_bottom = smc_data.get('bearish_ob_bottom', 0)
            # SHORT: OB must be ABOVE current price
            if ob_bottom > 0 and ob_bottom > self.current_price:
                levels[EntryStrategy.OB] = ob_bottom
            elif ob_top > 0 and ob_top > self.current_price:
                levels[EntryStrategy.OB] = ob_top
        
        # HTF Order Block - MUST VALIDATE DIRECTION
        if direction == 'LONG':
            htf_ob = smc_data.get('htf_bullish_ob_top', 0)
            # LONG: HTF OB must be BELOW current price
            if htf_ob > 0 and htf_ob < self.current_price:
                levels[EntryStrategy.OB_HTF] = htf_ob
        else:
            htf_ob = smc_data.get('htf_bearish_ob_bottom', 0)
            # SHORT: HTF OB must be ABOVE current price
            if htf_ob > 0 and htf_ob > self.current_price:
                levels[EntryStrategy.OB_HTF] = htf_ob
        
        # FVG - MUST VALIDATE DIRECTION
        if direction == 'LONG':
            fvg_level = smc_data.get('fvg_bullish_top', 0)
            # LONG: FVG must be BELOW current price
            if fvg_level > 0 and fvg_level < self.current_price:
                levels[EntryStrategy.FVG] = fvg_level
        else:
            fvg_level = smc_data.get('fvg_bearish_bottom', 0)
            # SHORT: FVG must be ABOVE current price
            if fvg_level > 0 and fvg_level > self.current_price:
                levels[EntryStrategy.FVG] = fvg_level
        
        # VWAP
        vwap = self._calculate_vwap()
        if vwap > 0:
            # For LONG, VWAP should be below price; for SHORT, above
            if (direction == 'LONG' and vwap < self.current_price) or \
               (direction == 'SHORT' and vwap > self.current_price):
                levels[EntryStrategy.VWAP] = vwap
        
        # LH/HL (structure-based)
        if direction == 'LONG':
            # Higher Low entry - must be BELOW current price
            recent_swing_low = smc_data.get('swing_low', self.recent_low)
            if recent_swing_low > 0 and recent_swing_low < self.current_price:
                levels[EntryStrategy.HL] = recent_swing_low
        else:
            # Lower High entry - must be ABOVE current price
            recent_swing_high = smc_data.get('swing_high', self.recent_high)
            if recent_swing_high > 0 and recent_swing_high > self.current_price:
                levels[EntryStrategy.LH] = recent_swing_high
        
        # Liquidity - MUST VALIDATE DIRECTION
        liq_level = smc_data.get('liquidity_level', 0)
        if liq_level > 0:
            # LONG: liquidity sweep below price; SHORT: above price
            if (direction == 'LONG' and liq_level < self.current_price) or \
               (direction == 'SHORT' and liq_level > self.current_price):
                levels[EntryStrategy.LIQ] = liq_level
        
        # Structure Break (market entry after confirmation)
        if smc_data.get('bos') or smc_data.get('choch'):
            levels[EntryStrategy.SB] = self.current_price
        
        # Engulfing (market entry on pattern)
        if smc_data.get('engulfing_bullish' if direction == 'LONG' else 'engulfing_bearish'):
            levels[EntryStrategy.EN] = self.current_price
        
        # Market entry fallback
        levels[EntryStrategy.MARKET] = self.current_price
        
        return levels
    
    def _calculate_levels(
        self,
        direction: str,
        entry: float,
        smc_data: Dict,
        tp_ref_price: float = None,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate SL and TP levels based on STRUCTURE with ANTI-HUNT protection.
        
        Args:
            direction: 'LONG' or 'SHORT'
            entry: Entry price (limit or market) - used for SL calculation
            smc_data: SMC analysis data
            tp_ref_price: Price to find TP levels above/below (default: entry)
                               For limit entries, pass current market price so TPs
                               target levels above market (realistic targets to hit)
                               
        KEY INSIGHT FOR LIMIT ENTRIES:
        - SL is structure-based from ENTRY (limit entry)
        - TPs are structure levels above MARKET price (so they're reachable)
        - R:R calculated from LIMIT entry gives better ratios (that's the point!)
        
        ANTI-HUNT SL Strategy:
        1. Find obvious level (swing low for LONG)
        2. Add significant buffer BELOW (1+ ATR, not just padding)
        3. Round to "ugly" numbers (avoid .00, .50, .25, round numbers)
        4. Ensure minimum distance by trading mode
        
        TPs are based on STRUCTURE, validated by backtesting - NOT arbitrary caps.
        """
        
        # Use tp_ref_price for finding TP structure levels
        # This ensures TPs are above market price for LONG (realistic targets)
        tp_ref = tp_ref_price if tp_ref_price and tp_ref_price > 0 else entry
        
        # ═══════════════════════════════════════════════════════════════════
        # ANTI-HUNT SL CONFIGURATION BY MODE
        # ═══════════════════════════════════════════════════════════════════
        anti_hunt_config = {
            TradingMode.SCALP: {'min_sl_pct': 0.5, 'buffer_atr': 0.5},
            TradingMode.DAY_TRADE: {'min_sl_pct': 1.0, 'buffer_atr': 1.0},
            TradingMode.SWING: {'min_sl_pct': 2.0, 'buffer_atr': 1.5},
            TradingMode.INVESTMENT: {'min_sl_pct': 3.0, 'buffer_atr': 2.0},
        }
        
        config = anti_hunt_config.get(self.mode, anti_hunt_config[TradingMode.DAY_TRADE])
        min_sl_pct = config['min_sl_pct']
        buffer_atr = config['buffer_atr']
        
        # ATR buffer for ANTI-HUNT (much larger than before!)
        atr_buffer = self.atr * buffer_atr if self.atr > 0 else entry * (min_sl_pct / 100)
        
        if direction == 'LONG':
            # ═══════════════════════════════════════════════════════════
            # ANTI-HUNT STOP LOSS - Well below obvious levels
            # ═══════════════════════════════════════════════════════════
            swing_low = smc_data.get('swing_low', 0)
            bullish_ob_bottom = smc_data.get('bullish_ob_bottom', 0)
            
            # Find the obvious level where everyone puts their stop
            if swing_low > 0 and swing_low < entry:
                obvious_sl = swing_low
            elif bullish_ob_bottom > 0 and bullish_ob_bottom < entry:
                obvious_sl = bullish_ob_bottom
            else:
                obvious_sl = entry * 0.99  # Fallback
            
            # ANTI-HUNT: Go BELOW the obvious level by ATR buffer
            raw_sl = obvious_sl - atr_buffer
            
            # Enforce minimum SL distance
            min_sl = entry * (1 - min_sl_pct / 100)
            if raw_sl > min_sl:
                raw_sl = min_sl
            
            # Make it an "ugly" number (anti-hunt)
            sl = self._make_ugly_number(raw_sl, entry, direction)
            
            # ═══════════════════════════════════════════════════════════
            # TAKE PROFITS - Structure-based targets
            # Use tp_ref (current market price) for finding levels
            # This ensures TPs are above MARKET price, not just above limit entry
            # ═══════════════════════════════════════════════════════════
            
            # Get ALL structure levels from SMC - NO ARBITRARY FILTERING
            fvg_top = smc_data.get('fvg_bearish_bottom', 0)  # FVG to fill
            htf_bearish_ob = smc_data.get('htf_bearish_ob_bottom', 0)  # HTF supply
            swing_high = smc_data.get('swing_high', 0)
            bearish_ob_bottom = smc_data.get('bearish_ob_bottom', 0)  # Current TF supply
            
            # Collect ALL valid resistance levels above tp_ref (market price)
            # Let SMC tell us where structure is - don't filter by arbitrary %
            resistance_levels = []
            if fvg_top > tp_ref:
                resistance_levels.append(('FVG', fvg_top))
            if bearish_ob_bottom > tp_ref:
                resistance_levels.append(('OB', bearish_ob_bottom))
            if htf_bearish_ob > tp_ref:
                resistance_levels.append(('HTF_OB', htf_bearish_ob))
            if swing_high > tp_ref:
                resistance_levels.append(('SWING', swing_high))
            
            # Sort by price (nearest first)
            resistance_levels.sort(key=lambda x: x[1])
            
            # Assign TPs from ACTUAL STRUCTURE only
            if len(resistance_levels) >= 3:
                tp1 = resistance_levels[0][1]
                tp2 = resistance_levels[1][1]
                tp3 = resistance_levels[2][1]
            elif len(resistance_levels) == 2:
                tp1 = resistance_levels[0][1]
                tp2 = resistance_levels[1][1]
                # TP3 = extend from the structure we have
                tp3 = tp2 + (tp2 - tp1)
            elif len(resistance_levels) == 1:
                # Only one structure level - use it and extend
                tp1 = resistance_levels[0][1]
                dist_to_tp1 = tp1 - tp_ref
                tp2 = tp1 + dist_to_tp1 * 0.5
                tp3 = tp1 + dist_to_tp1 * 1.0
            else:
                # NO STRUCTURE FOUND - use swing_high as fallback if available
                # This shouldn't happen if SMC is working correctly
                if swing_high > tp_ref:
                    tp1 = swing_high
                    dist = swing_high - tp_ref
                    tp2 = swing_high + dist * 0.5
                    tp3 = swing_high + dist * 1.0
                else:
                    # Absolute last resort - ATR based (but this means SMC found nothing)
                    tp1 = tp_ref + (self.atr * 2) if self.atr > 0 else tp_ref * 1.02
                    tp2 = tp_ref + (self.atr * 3) if self.atr > 0 else tp_ref * 1.03
                    tp3 = tp_ref + (self.atr * 4) if self.atr > 0 else tp_ref * 1.04
                
        else:  # SHORT
            # ═══════════════════════════════════════════════════════════
            # ANTI-HUNT STOP LOSS - Well above obvious levels
            # ═══════════════════════════════════════════════════════════
            swing_high = smc_data.get('swing_high', 0)
            bearish_ob_top = smc_data.get('bearish_ob_top', 0)
            
            # Find the obvious level where everyone puts their stop
            if swing_high > 0 and swing_high > entry:
                obvious_sl = swing_high
            elif bearish_ob_top > 0 and bearish_ob_top > entry:
                obvious_sl = bearish_ob_top
            else:
                obvious_sl = entry * 1.01  # Fallback
            
            # ANTI-HUNT: Go ABOVE the obvious level by ATR buffer
            raw_sl = obvious_sl + atr_buffer
            
            # Enforce minimum SL distance
            min_sl = entry * (1 + min_sl_pct / 100)
            if raw_sl < min_sl:
                raw_sl = min_sl
            
            # Make it an "ugly" number (anti-hunt)
            sl = self._make_ugly_number(raw_sl, entry, direction)
            
            # ═══════════════════════════════════════════════════════════
            # TAKE PROFITS - Structure-based targets
            # Use tp_ref (market price) to find support levels below market
            # NO ARBITRARY FILTERING - use what SMC finds
            # ═══════════════════════════════════════════════════════════
            
            fvg_bottom = smc_data.get('fvg_bullish_top', 0)
            htf_bullish_ob = smc_data.get('htf_bullish_ob_top', 0)
            swing_low = smc_data.get('swing_low', 0)
            bullish_ob_top = smc_data.get('bullish_ob_top', 0)
            
            # Collect ALL valid support levels below tp_ref (market price)
            support_levels = []
            if 0 < fvg_bottom < tp_ref:
                support_levels.append(('FVG', fvg_bottom))
            if 0 < bullish_ob_top < tp_ref:
                support_levels.append(('OB', bullish_ob_top))
            if 0 < htf_bullish_ob < tp_ref:
                support_levels.append(('HTF_OB', htf_bullish_ob))
            if 0 < swing_low < tp_ref:
                support_levels.append(('SWING', swing_low))
            
            # Sort by price (nearest first = highest price for SHORT)
            support_levels.sort(key=lambda x: x[1], reverse=True)
            
            # Assign TPs from ACTUAL STRUCTURE only
            if len(support_levels) >= 3:
                tp1 = support_levels[0][1]
                tp2 = support_levels[1][1]
                tp3 = support_levels[2][1]
            elif len(support_levels) == 2:
                tp1 = support_levels[0][1]
                tp2 = support_levels[1][1]
                # TP3 = extend from the structure we have
                tp3 = tp2 - (tp1 - tp2)
            elif len(support_levels) == 1:
                # Only one structure level - use it and extend
                tp1 = support_levels[0][1]
                dist_to_tp1 = tp_ref - tp1
                tp2 = tp1 - dist_to_tp1 * 0.5
                tp3 = tp1 - dist_to_tp1 * 1.0
            else:
                # NO STRUCTURE FOUND - use swing_low as fallback if available
                if 0 < swing_low < tp_ref:
                    tp1 = swing_low
                    dist = tp_ref - swing_low
                    tp2 = swing_low - dist * 0.5
                    tp3 = swing_low - dist * 1.0
                else:
                    # Absolute last resort - ATR based
                    tp1 = tp_ref - (self.atr * 2) if self.atr > 0 else tp_ref * 0.98
                    tp2 = tp_ref - (self.atr * 3) if self.atr > 0 else tp_ref * 0.97
                    tp3 = tp_ref - (self.atr * 4) if self.atr > 0 else tp_ref * 0.96
        
        # TPs come from STRUCTURE - no arbitrary caps
        # The backtester validates which levels actually work
        return sl, tp1, tp2, tp3
    
    def _make_ugly_number(self, price: float, entry: float, direction: str) -> float:
        """
        Convert a price to an "ugly" number that avoids common stop-hunt levels.
        
        Avoids numbers ending in: .00, .25, .50, .75, .10, .20, .30, .40, .60, .70, .80, .90
        Creates numbers ending in: .03, .07, .13, .17, .23, .27, .33, .37, .43, .47, etc.
        
        These are levels where retail stops DON'T cluster.
        """
        # Determine precision based on price level
        if price >= 1000:
            precision = 2
            ugly_endings = [0.03, 0.07, 0.13, 0.17, 0.23, 0.27, 0.33, 0.37, 
                          0.43, 0.47, 0.53, 0.57, 0.63, 0.67, 0.73, 0.77, 
                          0.83, 0.87, 0.93, 0.97]
            base_unit = 1.0
        elif price >= 100:
            precision = 2
            ugly_endings = [0.03, 0.07, 0.13, 0.17, 0.23, 0.27, 0.33, 0.37,
                          0.43, 0.47, 0.53, 0.57, 0.63, 0.67, 0.73, 0.77,
                          0.83, 0.87, 0.93, 0.97]
            base_unit = 1.0
        elif price >= 10:
            precision = 3
            ugly_endings = [0.037, 0.073, 0.137, 0.173, 0.237, 0.273, 0.337, 0.373,
                          0.437, 0.473, 0.537, 0.573, 0.637, 0.673, 0.737, 0.773,
                          0.837, 0.873, 0.937, 0.973]
            base_unit = 1.0
        elif price >= 1:
            precision = 4
            ugly_endings = [0.0037, 0.0073, 0.0137, 0.0173, 0.0237, 0.0273, 0.0337, 0.0373,
                          0.0437, 0.0473, 0.0537, 0.0573, 0.0637, 0.0673, 0.0737, 0.0773,
                          0.0837, 0.0873, 0.0937, 0.0973]
            base_unit = 0.1
        else:  # < 1
            precision = 6
            ugly_endings = [0.000037, 0.000073, 0.000137, 0.000173, 0.000237, 0.000273,
                          0.000337, 0.000373, 0.000437, 0.000473, 0.000537, 0.000573]
            base_unit = 0.001
        
        # Get the integer base of the price
        base = int(price / base_unit) * base_unit
        
        # Find closest ugly ending
        decimal_part = price - base
        
        # Pick ugly ending that's closest to our decimal part but in the right direction
        if direction == 'LONG':
            # SL below price - find ugly ending <= decimal part
            valid_endings = [e for e in ugly_endings if e <= decimal_part + 0.0001]
            if valid_endings:
                ugly_ending = max(valid_endings)  # Closest below
            else:
                # Go to previous base unit
                base -= base_unit
                ugly_ending = max(ugly_endings)
        else:
            # SL above price - find ugly ending >= decimal part
            valid_endings = [e for e in ugly_endings if e >= decimal_part - 0.0001]
            if valid_endings:
                ugly_ending = min(valid_endings)  # Closest above
            else:
                # Go to next base unit
                base += base_unit
                ugly_ending = min(ugly_endings)
        
        ugly_price = base + ugly_ending
        
        return round(ugly_price, precision)
    
    def _create_market_entry(self, direction: str, smc_data: Dict) -> StrategyRecommendation:
        """Create a market entry recommendation when no better option exists"""
        sl, tp1, tp2, tp3 = self._calculate_levels(direction, self.current_price, smc_data)
        
        return StrategyRecommendation(
            primary_strategy=EntryStrategy.MARKET,
            secondary_strategy=None,
            entry_price=self.current_price,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            confidence=40,  # Low confidence for market entry
            reasoning="No optimal entry level found - market entry as fallback",
            entry_type='market',
            time_in_force='IOC',
            max_wait_candles=0,
            strategy_tags=['MARKET'],
        )
    
    def get_strategy_summary(self, direction: str, smc_data: Dict) -> str:
        """
        Get a human-readable summary of recommended strategies.
        Similar to the Discord bot output.
        """
        rec = self.get_best_strategy(direction, smc_data)
        
        summary = f"""
╔══════════════════════════════════════╗
║  STRATEGY RECOMMENDATION             ║
╠══════════════════════════════════════╣
║ Mode: {self.mode.value.upper():^32} ║
║ Direction: {direction:^27} ║
╠══════════════════════════════════════╣
║ Primary: {rec.primary_strategy.value:^28} ║
║ Secondary: {rec.secondary_strategy.value if rec.secondary_strategy else 'N/A':^26} ║
║ Confidence: {rec.confidence:.0f}%{' ':>23} ║
╠══════════════════════════════════════╣
║ Entry: ${rec.entry_price:<28.4f} ║
║ Stop Loss: ${rec.stop_loss:<24.4f} ║
║ TP1: ${rec.tp1:<30.4f} ║
║ TP2: ${rec.tp2:<30.4f} ║
║ TP3: ${rec.tp3:<30.4f} ║
╠══════════════════════════════════════╣
║ Entry Type: {rec.entry_type.upper():^24} ║
║ Max Wait: {rec.max_wait_candles} candles{' ':>17} ║
║ Best Strategy: {', '.join(rec.strategy_tags):^21} ║
╚══════════════════════════════════════╝

Reasoning: {rec.reasoning}
"""
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK STRATEGY FUNCTIONS (for integration)
# ═══════════════════════════════════════════════════════════════════════════════

def get_best_entry_strategy(
    df: pd.DataFrame,
    direction: str,
    smc_data: Dict,
    timeframe: str = '15m',
    mode: Optional[TradingMode] = None,
) -> StrategyRecommendation:
    """
    Quick function to get best entry strategy.
    
    Args:
        df: OHLCV DataFrame
        direction: 'LONG' or 'SHORT'
        smc_data: SMC analysis dictionary
        timeframe: Timeframe string
        mode: Trading mode (auto-detected if None)
        
    Returns:
        StrategyRecommendation
    """
    selector = StrategySelector(df, mode=mode, timeframe=timeframe)
    return selector.get_best_strategy(direction, smc_data)


def format_strategy_tags(recommendation: StrategyRecommendation) -> str:
    """Format strategy tags like Discord bot: OB, LH, EN, VWAP, SB"""
    return ", ".join(recommendation.strategy_tags)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example with mock data
    import numpy as np
    
    # Create sample OHLCV data
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
        'swing_low': 98.5,
        'swing_high': 102.0,
        'structure': 'bullish',
        'bos': True,
    }
    
    # Get recommendation
    selector = StrategySelector(df, timeframe='15m')
    rec = selector.get_best_strategy('LONG', smc_data)
    
    print(selector.get_strategy_summary('LONG', smc_data))