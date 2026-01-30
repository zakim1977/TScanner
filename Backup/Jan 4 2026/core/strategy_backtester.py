"""
Strategy Backtester Module

Uses the backtesting.py package to properly test SMC strategies
and determine which works best for each timeframe AND SESSION.

Sessions matter because:
- Asia (00:00-08:00 UTC): Low volatility, range-bound → FVG, Range plays
- London (08:00-16:00 UTC): High volatility, breakouts → OB, Structure Breaks  
- New York (13:00-21:00 UTC): Trend continuation → HL/LH, Liquidity sweeps
- Overlap (13:00-16:00 UTC): Maximum volatility → Momentum plays

NO arbitrary caps - let the DATA decide what works!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')


class TradingSession(Enum):
    """Trading sessions with UTC hours"""
    ASIA = "asia"           # 00:00 - 08:00 UTC
    LONDON = "london"       # 08:00 - 16:00 UTC  
    NEW_YORK = "new_york"   # 13:00 - 21:00 UTC
    OVERLAP = "overlap"     # 13:00 - 16:00 UTC (London + NY)
    ALL = "all"             # Full 24h


# Session time ranges (UTC hours)
SESSION_HOURS = {
    TradingSession.ASIA: (0, 8),
    TradingSession.LONDON: (8, 16),
    TradingSession.NEW_YORK: (13, 21),
    TradingSession.OVERLAP: (13, 16),
    TradingSession.ALL: (0, 24),
}


def detect_current_session() -> TradingSession:
    """Detect current trading session based on UTC time"""
    from datetime import datetime, timezone
    
    utc_hour = datetime.now(timezone.utc).hour
    
    # Check overlap first (most specific)
    if 13 <= utc_hour < 16:
        return TradingSession.OVERLAP
    elif 13 <= utc_hour < 21:
        return TradingSession.NEW_YORK
    elif 8 <= utc_hour < 16:
        return TradingSession.LONDON
    elif 0 <= utc_hour < 8:
        return TradingSession.ASIA
    else:
        return TradingSession.ASIA  # Late night = Asia


def filter_by_session(df: pd.DataFrame, session: TradingSession) -> pd.DataFrame:
    """Filter DataFrame to only include candles from specified session"""
    if session == TradingSession.ALL:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        return df  # Can't filter without datetime index
    
    start_hour, end_hour = SESSION_HOURS[session]
    
    # Filter by hour
    mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
    filtered = df[mask]
    
    # Need minimum data for backtest
    if len(filtered) < 100:
        return df  # Fall back to full data
    
    return filtered


class StrategyType(Enum):
    """SMC Strategy types we'll backtest"""
    ORDER_BLOCK = "OB"
    FVG = "FVG"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    STRUCTURE_BREAK = "BOS"
    VWAP = "VWAP"
    FIBONACCI = "FIB"
    LIQUIDITY = "LIQ"
    EMA_CROSS = "EMA"  # Baseline comparison


@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    strategy_type: StrategyType
    timeframe: str
    session: TradingSession  # NEW: Which session this was tested on
    direction: str  # LONG or SHORT
    
    # Core metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    
    # TP/SL metrics
    avg_win_pct: float
    avg_loss_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    
    # Optimal levels discovered
    optimal_sl_pct: float
    optimal_tp1_pct: float
    optimal_tp2_pct: float
    optimal_tp3_pct: float
    
    # Time metrics
    avg_trade_duration: str
    
    def __str__(self):
        return f"{self.strategy_type.value} {self.timeframe} {self.session.value}: {self.win_rate:.1f}% WR, {self.profit_factor:.2f} PF"


# ═══════════════════════════════════════════════════════════════════════════════
# BASE STRATEGY CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SMCBaseStrategy(Strategy):
    """
    Base class for all SMC strategies.
    Child classes implement entry logic, this handles SL/TP management.
    """
    
    # Parameters to optimize
    sl_atr_mult = 1.5
    tp1_atr_mult = 1.5
    tp2_atr_mult = 3.0
    tp3_atr_mult = 5.0
    
    # Position sizing
    risk_pct = 2.0  # Risk 2% per trade
    
    def init(self):
        """Initialize indicators"""
        # ATR for SL/TP calculation
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        
        def calc_atr():
            h = pd.Series(high)
            l = pd.Series(low)
            c = pd.Series(close)
            tr1 = h - l
            tr2 = abs(h - c.shift(1))
            tr3 = abs(l - c.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(14).mean().fillna(method='bfill').values
        
        def calc_ema_fast():
            return pd.Series(close).ewm(span=9).mean().values
        
        def calc_ema_slow():
            return pd.Series(close).ewm(span=21).mean().values
        
        def calc_swing_high():
            h = pd.Series(high)
            result = h.rolling(5, center=True).max()
            is_swing = (h == result)
            swing_highs = h.where(is_swing).ffill().fillna(h.max())
            return swing_highs.values
        
        def calc_swing_low():
            l = pd.Series(low)
            result = l.rolling(5, center=True).min()
            is_swing = (l == result)
            swing_lows = l.where(is_swing).ffill().fillna(l.min())
            return swing_lows.values
        
        self.atr = self.I(calc_atr, name='ATR')
        self.ema_fast = self.I(calc_ema_fast, name='EMA9')
        self.ema_slow = self.I(calc_ema_slow, name='EMA21')
        self.swing_high = self.I(calc_swing_high, name='SwingH')
        self.swing_low = self.I(calc_swing_low, name='SwingL')
    
    def should_go_long(self) -> bool:
        """Override in child class"""
        return False
    
    def should_go_short(self) -> bool:
        """Override in child class"""
        return False
    
    def next(self):
        """Main trading logic"""
        if len(self.data) < 20:
            return
            
        price = self.data.Close[-1]
        atr = self.atr[-1] if self.atr[-1] > 0 else price * 0.01
        
        # Check for entries if not in position
        if not self.position:
            if self.should_go_long():
                sl = price - (atr * self.sl_atr_mult)
                tp = price + (atr * self.tp1_atr_mult)
                self.buy(sl=sl, tp=tp)
                
            elif self.should_go_short():
                sl = price + (atr * self.sl_atr_mult)
                tp = price - (atr * self.tp1_atr_mult)
                self.sell(sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER BLOCK STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class OrderBlockStrategy(SMCBaseStrategy):
    """
    Order Block entry strategy.
    Enters when price returns to an OB zone.
    """
    
    def init(self):
        super().init()
        
        # Detect Order Blocks
        high = self.data.High
        low = self.data.Low
        open_ = self.data.Open
        close = self.data.Close
        
        def calc_bullish_ob():
            o = pd.Series(open_)
            c = pd.Series(close)
            h = pd.Series(high)
            l = pd.Series(low)
            
            is_bearish = c < o
            is_bullish = c > o
            ob_signal = is_bearish.shift(1) & is_bullish & (c > h.shift(1))
            ob_low = l.shift(1).where(ob_signal)
            return ob_low.ffill().fillna(0).values
        
        def calc_bearish_ob():
            o = pd.Series(open_)
            c = pd.Series(close)
            h = pd.Series(high)
            l = pd.Series(low)
            
            is_bearish = c < o
            is_bullish = c > o
            ob_signal = is_bullish.shift(1) & is_bearish & (c < l.shift(1))
            ob_high = h.shift(1).where(ob_signal)
            return ob_high.ffill().fillna(0).values
        
        self.bullish_ob = self.I(calc_bullish_ob, name='BullOB')
        self.bearish_ob = self.I(calc_bearish_ob, name='BearOB')
    
    def should_go_long(self) -> bool:
        """Enter long when price touches bullish OB"""
        ob_level = self.bullish_ob[-1]
        if ob_level <= 0:
            return False
        
        price = self.data.Close[-1]
        
        # Price must be near OB (within 0.5%)
        distance = abs(price - ob_level) / price
        
        # Trend filter: above slow EMA
        trend_ok = price > self.ema_slow[-1]
        
        return distance < 0.005 and trend_ok
    
    def should_go_short(self) -> bool:
        """Enter short when price touches bearish OB"""
        ob_level = self.bearish_ob[-1]
        if ob_level <= 0:
            return False
        
        price = self.data.Close[-1]
        
        distance = abs(price - ob_level) / price
        trend_ok = price < self.ema_slow[-1]
        
        return distance < 0.005 and trend_ok


# ═══════════════════════════════════════════════════════════════════════════════
# FVG (FAIR VALUE GAP) STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class FVGStrategy(SMCBaseStrategy):
    """
    Fair Value Gap entry strategy.
    Enters when price fills an FVG.
    """
    
    def init(self):
        super().init()
        
        high = self.data.High
        low = self.data.Low
        
        def calc_bullish_fvg():
            h = pd.Series(high)
            l = pd.Series(low)
            # Bullish FVG: gap up
            gap = l - h.shift(2)
            fvg_exists = gap > 0
            fvg_level = h.shift(2).where(fvg_exists)
            return fvg_level.ffill().fillna(0).values
        
        def calc_bearish_fvg():
            h = pd.Series(high)
            l = pd.Series(low)
            # Bearish FVG: gap down
            gap = l.shift(2) - h
            fvg_exists = gap > 0
            fvg_level = l.shift(2).where(fvg_exists)
            return fvg_level.ffill().fillna(0).values
        
        self.bullish_fvg = self.I(calc_bullish_fvg, name='BullFVG')
        self.bearish_fvg = self.I(calc_bearish_fvg, name='BearFVG')
    
    def should_go_long(self) -> bool:
        fvg_level = self.bullish_fvg[-1]
        if fvg_level <= 0:
            return False
        
        price = self.data.Close[-1]
        
        # Price filling the gap from above
        filled = self.data.Low[-1] <= fvg_level
        trend_ok = price > self.ema_slow[-1]
        
        return filled and trend_ok
    
    def should_go_short(self) -> bool:
        fvg_level = self.bearish_fvg[-1]
        if fvg_level <= 0:
            return False
        
        price = self.data.Close[-1]
        
        filled = self.data.High[-1] >= fvg_level
        trend_ok = price < self.ema_slow[-1]
        
        return filled and trend_ok


# ═══════════════════════════════════════════════════════════════════════════════
# HIGHER LOW / LOWER HIGH STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class HigherLowStrategy(SMCBaseStrategy):
    """
    Higher Low entry for longs, Lower High for shorts.
    Classic structure-based entries.
    """
    
    def should_go_long(self) -> bool:
        """Enter on Higher Low"""
        if len(self.data) < 10:
            return False
        
        # Current swing low must be higher than previous
        current_low = self.swing_low[-1]
        prev_low = self.swing_low[-5] if len(self.swing_low) > 5 else current_low
        
        higher_low = current_low > prev_low
        
        # Price bouncing off the higher low
        price = self.data.Close[-1]
        near_low = abs(price - current_low) / price < 0.01
        
        # Trend filter
        trend_ok = self.ema_fast[-1] > self.ema_slow[-1]
        
        return higher_low and near_low and trend_ok
    
    def should_go_short(self) -> bool:
        """Enter on Lower High"""
        if len(self.data) < 10:
            return False
        
        current_high = self.swing_high[-1]
        prev_high = self.swing_high[-5] if len(self.swing_high) > 5 else current_high
        
        lower_high = current_high < prev_high
        
        price = self.data.Close[-1]
        near_high = abs(price - current_high) / price < 0.01
        trend_ok = self.ema_fast[-1] < self.ema_slow[-1]
        
        return lower_high and near_high and trend_ok


# ═══════════════════════════════════════════════════════════════════════════════
# EMA CROSSOVER (BASELINE FOR COMPARISON)
# ═══════════════════════════════════════════════════════════════════════════════

class EMACrossStrategy(SMCBaseStrategy):
    """Simple EMA crossover as baseline comparison"""
    
    def should_go_long(self) -> bool:
        if len(self.data) < 3:
            return False
        # EMA fast crosses above EMA slow
        return self.ema_fast[-2] < self.ema_slow[-2] and self.ema_fast[-1] > self.ema_slow[-1]
    
    def should_go_short(self) -> bool:
        if len(self.data) < 3:
            return False
        # EMA slow crosses above EMA fast
        return self.ema_fast[-2] > self.ema_slow[-2] and self.ema_fast[-1] < self.ema_slow[-1]


class VWAPStrategy(SMCBaseStrategy):
    """
    VWAP Mean Reversion Strategy
    
    - LONG: Price crosses above VWAP after being below (mean reversion)
    - SHORT: Price crosses below VWAP after being above
    
    VWAP acts as institutional fair value - deviations tend to revert.
    """
    
    def _get_vwap(self):
        """Calculate VWAP from data"""
        if len(self.data) < 2:
            return None
        typical_price = (self.data.High + self.data.Low + self.data.Close) / 3
        cumulative_tp_vol = pd.Series(typical_price * self.data.Volume).cumsum()
        cumulative_vol = pd.Series(self.data.Volume).cumsum()
        return (cumulative_tp_vol / cumulative_vol).values
    
    def should_go_long(self) -> bool:
        if len(self.data) < 3:
            return False
        vwap = self._get_vwap()
        if vwap is None:
            return False
        # Price crosses above VWAP from below
        close = self.data.Close
        return close[-2] < vwap[-2] and close[-1] > vwap[-1]
    
    def should_go_short(self) -> bool:
        if len(self.data) < 3:
            return False
        vwap = self._get_vwap()
        if vwap is None:
            return False
        # Price crosses below VWAP from above
        close = self.data.Close
        return close[-2] > vwap[-2] and close[-1] < vwap[-1]


class FibonacciStrategy(SMCBaseStrategy):
    """
    Fibonacci Retracement Strategy
    
    - LONG: Price bounces from 61.8% or 78.6% retracement in uptrend
    - SHORT: Price rejects from 38.2% or 50% retracement in downtrend
    
    The 61.8% (golden ratio) is the most significant institutional level.
    """
    
    def _get_fib_levels(self):
        """Calculate fib levels from recent swing"""
        if len(self.data) < 20:
            return {}
        
        lookback = min(50, len(self.data))
        recent_high = max(self.data.High[-lookback:])
        recent_low = min(self.data.Low[-lookback:])
        range_size = recent_high - recent_low
        
        if range_size <= 0:
            return {}
        
        return {
            '382': recent_high - (range_size * 0.382),
            '500': recent_high - (range_size * 0.500),
            '618': recent_high - (range_size * 0.618),
            '786': recent_high - (range_size * 0.786),
        }
    
    def should_go_long(self) -> bool:
        if len(self.data) < 20:
            return False
        
        fib_levels = self._get_fib_levels()
        if not fib_levels:
            return False
        
        close = self.data.Close
        low = self.data.Low
        fib_618 = fib_levels.get('618', 0)
        fib_786 = fib_levels.get('786', 0)
        
        # Price touches 61.8% or 78.6% and bounces (bullish candle)
        touched_fib = (low[-1] <= fib_618 * 1.002) or (low[-1] <= fib_786 * 1.002)
        bullish_bounce = close[-1] > close[-2]  # Current candle is bullish
        
        return touched_fib and bullish_bounce
    
    def should_go_short(self) -> bool:
        if len(self.data) < 20:
            return False
        
        fib_levels = self._get_fib_levels()
        if not fib_levels:
            return False
        
        close = self.data.Close
        high = self.data.High
        fib_382 = fib_levels.get('382', float('inf'))
        fib_500 = fib_levels.get('500', float('inf'))
        
        # Price rallies to 38.2% or 50% and rejects (bearish candle)
        touched_fib = (high[-1] >= fib_382 * 0.998) or (high[-1] >= fib_500 * 0.998)
        bearish_reject = close[-1] < close[-2]  # Current candle is bearish
        
        return touched_fib and bearish_reject


class LiquidityStrategy(SMCBaseStrategy):
    """
    Liquidity Sweep Strategy
    
    - LONG: Price sweeps below swing low (grabs sell stops) then reverses up
    - SHORT: Price sweeps above swing high (grabs buy stops) then reverses down
    
    Smart money hunts liquidity before major moves. This strategy enters
    after the sweep is complete and reversal is confirmed.
    """
    
    def _find_swing_points(self):
        """Find recent swing highs and lows"""
        if len(self.data) < 10:
            return [], []
        
        highs = list(self.data.High)
        lows = list(self.data.Low)
        
        swing_highs = []
        swing_lows = []
        
        for i in range(3, len(highs) - 1):
            # Swing high: higher than 3 before and 1 after
            if highs[i] > max(highs[max(0,i-3):i]) and highs[i] > highs[i+1]:
                swing_highs.append(highs[i])
            # Swing low: lower than 3 before and 1 after
            if lows[i] < min(lows[max(0,i-3):i]) and lows[i] < lows[i+1]:
                swing_lows.append(lows[i])
        
        return swing_highs, swing_lows
    
    def should_go_long(self) -> bool:
        if len(self.data) < 10:
            return False
        
        swing_highs, swing_lows = self._find_swing_points()
        if not swing_lows:
            return False
        
        low = self.data.Low
        close = self.data.Close
        
        # Recent swing low (highest of last 3 swing lows = nearest)
        recent_swing_low = max(swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]
        
        # Price swept below swing low (grabbed liquidity) then closed back above
        swept = low[-2] < recent_swing_low  # Previous candle swept below
        recovered = close[-1] > recent_swing_low  # Current candle recovered
        bullish = close[-1] > close[-2]  # Bullish momentum
        
        return swept and recovered and bullish
    
    def should_go_short(self) -> bool:
        if len(self.data) < 10:
            return False
        
        swing_highs, swing_lows = self._find_swing_points()
        if not swing_highs:
            return False
        
        high = self.data.High
        close = self.data.Close
        
        # Recent swing high (lowest of last 3 swing highs = nearest)
        recent_swing_high = min(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
        
        # Price swept above swing high (grabbed liquidity) then closed back below
        swept = high[-2] > recent_swing_high  # Previous candle swept above
        recovered = close[-1] < recent_swing_high  # Current candle recovered
        bearish = close[-1] < close[-2]  # Bearish momentum
        
        return swept and recovered and bearish


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_CLASSES = {
    StrategyType.ORDER_BLOCK: OrderBlockStrategy,
    StrategyType.FVG: FVGStrategy,
    StrategyType.HIGHER_LOW: HigherLowStrategy,
    StrategyType.EMA_CROSS: EMACrossStrategy,
    StrategyType.VWAP: VWAPStrategy,
    StrategyType.FIBONACCI: FibonacciStrategy,
    StrategyType.LIQUIDITY: LiquidityStrategy,
}


class StrategyBacktester:
    """
    Main backtester that tests all strategies on historical data
    and determines the best one for each timeframe AND SESSION.
    
    Sessions matter because different market participants are active:
    - Asia: Range-bound, lower volatility
    - London: Breakouts, high volatility
    - New York: Trend continuation
    - Overlap: Maximum volatility
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        timeframe: str,
        session: TradingSession = TradingSession.ALL,
    ):
        """
        Initialize with OHLCV data.
        
        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
            timeframe: e.g., '1m', '5m', '15m', '1h', '4h', '1d'
            session: Trading session to test (filters data by UTC hours)
        """
        self.timeframe = timeframe
        self.session = session
        
        # Prepare and filter data by session
        prepared_df = self._prepare_data(df)
        self.df = filter_by_session(prepared_df, session)
        
        self.results: Dict[StrategyType, BacktestResult] = {}
        
        # Track session-specific results
        self.session_results: Dict[TradingSession, Dict[StrategyType, BacktestResult]] = {}
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting.py format"""
        df = df.copy()
        
        # Ensure proper column names
        column_map = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        for old, new in column_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
            else:
                df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    
    def run_backtest(
        self,
        strategy_type: StrategyType,
        cash: float = 10000,
        commission: float = 0.001,  # 0.1% for crypto
    ) -> Optional[BacktestResult]:
        """Run backtest for a single strategy"""
        
        if strategy_type not in STRATEGY_CLASSES:
            print(f"Strategy {strategy_type} not implemented")
            return None
        
        strategy_class = STRATEGY_CLASSES[strategy_type]
        
        try:
            bt = Backtest(
                self.df,
                strategy_class,
                cash=cash,
                commission=commission,
                exclusive_orders=True,
                trade_on_close=True,
            )
            
            stats = bt.run()
            
            # Extract results
            result = BacktestResult(
                strategy_type=strategy_type,
                timeframe=self.timeframe,
                session=self.session,  # Include session
                direction="BOTH",  # Test both directions
                total_trades=stats['# Trades'],
                win_rate=stats['Win Rate [%]'] if not pd.isna(stats['Win Rate [%]']) else 0,
                profit_factor=stats['Profit Factor'] if not pd.isna(stats['Profit Factor']) else 0,
                sharpe_ratio=stats['Sharpe Ratio'] if not pd.isna(stats['Sharpe Ratio']) else 0,
                max_drawdown=stats['Max. Drawdown [%]'] if not pd.isna(stats['Max. Drawdown [%]']) else 0,
                avg_win_pct=stats['Best Trade [%]'] / 2 if not pd.isna(stats['Best Trade [%]']) else 0,
                avg_loss_pct=abs(stats['Worst Trade [%]']) / 2 if not pd.isna(stats['Worst Trade [%]']) else 0,
                best_trade_pct=stats['Best Trade [%]'] if not pd.isna(stats['Best Trade [%]']) else 0,
                worst_trade_pct=stats['Worst Trade [%]'] if not pd.isna(stats['Worst Trade [%]']) else 0,
                optimal_sl_pct=self._calc_optimal_sl(stats),
                optimal_tp1_pct=self._calc_optimal_tp1(stats),
                optimal_tp2_pct=self._calc_optimal_tp2(stats),
                optimal_tp3_pct=self._calc_optimal_tp3(stats),
                avg_trade_duration=str(stats['Avg. Trade Duration']) if 'Avg. Trade Duration' in stats else "N/A",
            )
            
            self.results[strategy_type] = result
            return result
            
        except Exception as e:
            print(f"Error backtesting {strategy_type}: {e}")
            return None
    
    def _calc_optimal_sl(self, stats) -> float:
        """Calculate optimal SL from ACTUAL backtest data"""
        worst = stats.get('Worst Trade [%]')
        if worst and not pd.isna(worst):
            return abs(worst)
        return 2.0
    
    def _calc_optimal_tp1(self, stats) -> float:
        """Calculate optimal TP1 from ACTUAL backtest data"""
        avg_trade = stats.get('Avg. Trade [%]')
        if avg_trade and not pd.isna(avg_trade) and avg_trade > 0:
            return abs(avg_trade) * 1.5
        return 1.5
    
    def _calc_optimal_tp2(self, stats) -> float:
        """Calculate optimal TP2 - MUST be higher than TP1"""
        tp1 = self._calc_optimal_tp1(stats)
        best_trade = stats.get('Best Trade [%]')
        
        if best_trade and not pd.isna(best_trade) and best_trade > 0:
            # TP2 = midpoint between TP1 and best, but MUST be > TP1
            tp2 = max(tp1 * 1.5, abs(best_trade) * 0.5)
        else:
            tp2 = tp1 * 2.0  # Default: 2x TP1
        
        # ENFORCE: TP2 > TP1
        return max(tp2, tp1 * 1.5)
    
    def _calc_optimal_tp3(self, stats) -> float:
        """Calculate optimal TP3 - MUST be higher than TP2"""
        tp2 = self._calc_optimal_tp2(stats)
        best_trade = stats.get('Best Trade [%]')
        
        if best_trade and not pd.isna(best_trade) and best_trade > 0:
            tp3 = max(tp2 * 1.5, abs(best_trade) * 0.8)
        else:
            tp3 = tp2 * 1.5  # Default: 1.5x TP2
        
        # ENFORCE: TP3 > TP2
        return max(tp3, tp2 * 1.3)
    
    def run_all_strategies(self) -> Dict[StrategyType, BacktestResult]:
        """Run backtest for all available strategies"""
        for strategy_type in STRATEGY_CLASSES.keys():
            self.run_backtest(strategy_type)
        return self.results
    
    def get_best_strategy(self) -> Optional[Tuple[StrategyType, BacktestResult]]:
        """Get the best performing strategy based on combined score"""
        if not self.results:
            return None
        
        def score_strategy(result: BacktestResult) -> float:
            """Combined score: win rate, profit factor, sharpe ratio"""
            if result.total_trades < 10:
                return 0  # Not enough trades
            
            # Weighted score
            score = (
                result.win_rate * 0.3 +  # 30% weight
                min(result.profit_factor, 3) * 20 +  # Cap PF at 3, 20pts per PF
                max(result.sharpe_ratio, 0) * 10 +  # 10pts per sharpe
                (100 - abs(result.max_drawdown)) * 0.2  # Penalize drawdown
            )
            return score
        
        best = max(self.results.items(), key=lambda x: score_strategy(x[1]))
        return best
    
    def get_optimal_levels(self, strategy_type: StrategyType) -> Dict[str, float]:
        """Get optimal SL/TP levels for a strategy"""
        if strategy_type not in self.results:
            return {
                'sl_pct': 1.5,
                'tp1_pct': 1.5,
                'tp2_pct': 3.0,
                'tp3_pct': 5.0,
            }
        
        result = self.results[strategy_type]
        return {
            'sl_pct': result.optimal_sl_pct,
            'tp1_pct': result.optimal_tp1_pct,
            'tp2_pct': result.optimal_tp2_pct,
            'tp3_pct': result.optimal_tp3_pct,
        }
    
    def print_results(self):
        """Print formatted results"""
        print("\n" + "═" * 70)
        print(f"BACKTEST RESULTS - {self.timeframe}")
        print("═" * 70)
        
        for strategy_type, result in sorted(
            self.results.items(),
            key=lambda x: x[1].win_rate,
            reverse=True
        ):
            print(f"\n{strategy_type.value} ({result.total_trades} trades):")
            print(f"  Win Rate:      {result.win_rate:.1f}%")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
            print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown:  {result.max_drawdown:.1f}%")
            print(f"  Best Trade:    +{result.best_trade_pct:.1f}%")
            print(f"  Worst Trade:   {result.worst_trade_pct:.1f}%")
        
        best = self.get_best_strategy()
        if best:
            print(f"\n🏆 BEST STRATEGY: {best[0].value}")
            levels = self.get_optimal_levels(best[0])
            print(f"   Optimal SL:  {levels['sl_pct']:.1f}%")
            print(f"   Optimal TP1: {levels['tp1_pct']:.1f}%")
            print(f"   Optimal TP2: {levels['tp2_pct']:.1f}%")
            print(f"   Optimal TP3: {levels['tp3_pct']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_symbol(
    df: pd.DataFrame,
    timeframe: str,
    session: TradingSession = TradingSession.ALL,
) -> Dict[str, any]:
    """
    Run full backtest for a symbol and return best strategy with optimal levels.
    
    Args:
        df: Historical OHLCV data
        timeframe: e.g., '15m', '1h', '4h'
        session: Trading session to test (auto-filters data by UTC hours)
    
    Returns:
        {
            'best_strategy': 'OB',
            'session': 'london',
            'win_rate': 62.5,
            'profit_factor': 1.85,
            'optimal_sl_pct': 1.2,
            'optimal_tp1_pct': 1.8,
            'optimal_tp2_pct': 3.5,
            'optimal_tp3_pct': 5.5,
            'all_results': {...}
        }
    """
    backtester = StrategyBacktester(df, timeframe, session)
    backtester.run_all_strategies()
    
    best = backtester.get_best_strategy()
    if not best:
        return {
            'best_strategy': 'OB',
            'session': session.value,
            'win_rate': 50.0,
            'profit_factor': 1.0,
            'optimal_sl_pct': 1.5,
            'optimal_tp1_pct': 2.0,
            'optimal_tp2_pct': 3.5,
            'optimal_tp3_pct': 5.0,
            'all_results': {}
        }
    
    strategy_type, result = best
    levels = backtester.get_optimal_levels(strategy_type)
    
    return {
        'best_strategy': strategy_type.value,
        'session': session.value,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'sharpe_ratio': result.sharpe_ratio,
        'total_trades': result.total_trades,
        'optimal_sl_pct': levels['sl_pct'],
        'optimal_tp1_pct': levels['tp1_pct'],
        'optimal_tp2_pct': levels['tp2_pct'],
        'optimal_tp3_pct': levels['tp3_pct'],
        'all_results': {
            st.value: {
                'win_rate': r.win_rate,
                'profit_factor': r.profit_factor,
                'trades': r.total_trades,
            }
            for st, r in backtester.results.items()
        }
    }


def backtest_all_sessions(
    df: pd.DataFrame,
    timeframe: str,
) -> Dict[str, Dict]:
    """
    Run backtest for ALL trading sessions and return best strategy for each.
    
    This is the KEY function - tells you which strategy works best in EACH session!
    
    Returns:
        {
            'asia': {'best_strategy': 'FVG', 'win_rate': 65.0, ...},
            'london': {'best_strategy': 'OB', 'win_rate': 72.0, ...},
            'new_york': {'best_strategy': 'HL', 'win_rate': 68.0, ...},
            'overlap': {'best_strategy': 'OB', 'win_rate': 75.0, ...},
            'current_session': 'london',
            'recommended': {...}  # Best for current session
        }
    """
    results = {}
    
    # Test each session
    for session in [TradingSession.ASIA, TradingSession.LONDON, 
                    TradingSession.NEW_YORK, TradingSession.OVERLAP]:
        try:
            result = backtest_symbol(df, timeframe, session)
            results[session.value] = result
        except Exception as e:
            print(f"Error backtesting {session.value}: {e}")
            results[session.value] = None
    
    # Detect current session
    current = detect_current_session()
    results['current_session'] = current.value
    
    # Get recommendation for current session
    if current.value in results and results[current.value]:
        results['recommended'] = results[current.value]
    else:
        # Fall back to full data backtest
        results['recommended'] = backtest_symbol(df, timeframe, TradingSession.ALL)
    
    return results


def get_session_strategy(
    df: pd.DataFrame,
    timeframe: str,
    session: Optional[TradingSession] = None,
) -> Dict:
    """
    Get the best strategy for a specific session (or auto-detect current session).
    
    This is what you call from the app - it automatically uses the right session!
    
    Args:
        df: Historical OHLCV data
        timeframe: e.g., '15m'
        session: Optional - if None, auto-detects current session
        
    Returns:
        Best strategy and levels for the session
    """
    if session is None:
        session = detect_current_session()
    
    return backtest_symbol(df, timeframe, session)


if __name__ == "__main__":
    # Test with sample data
    print("Creating sample data...")
    
    # Create data with proper timestamps for session testing
    dates = pd.date_range('2024-01-01', periods=2000, freq='15min', tz='UTC')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.randn(2000) * 0.002
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': prices,
        'High': prices * (1 + np.abs(np.random.randn(2000)) * 0.005),
        'Low': prices * (1 - np.abs(np.random.randn(2000)) * 0.005),
        'Close': prices * (1 + np.random.randn(2000) * 0.003),
        'Volume': np.random.randint(1000, 10000, 2000),
    }, index=dates)
    
    print("\n" + "═" * 60)
    print("SESSION-BASED BACKTESTING")
    print("═" * 60)
    
    # Test each session
    for session in [TradingSession.ASIA, TradingSession.LONDON, 
                    TradingSession.NEW_YORK, TradingSession.OVERLAP]:
        print(f"\n📊 {session.value.upper()} SESSION:")
        result = backtest_symbol(df, '15m', session)
        print(f"   Best Strategy: {result['best_strategy']}")
        print(f"   Win Rate: {result['win_rate']:.1f}%")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")
        print(f"   Optimal SL: {result['optimal_sl_pct']:.1f}%")
        print(f"   Optimal TP1: {result['optimal_tp1_pct']:.1f}%")
    
    print("\n" + "═" * 60)
    print("CURRENT SESSION RECOMMENDATION")
    print("═" * 60)
    current = detect_current_session()
    print(f"Current Session: {current.value.upper()}")
    
    rec = get_session_strategy(df, '15m')
    print(f"Recommended Strategy: {rec['best_strategy']}")
    print(f"Win Rate: {rec['win_rate']:.1f}%")
    print(f"Optimal Levels: SL {rec['optimal_sl_pct']:.1f}%, TP1 {rec['optimal_tp1_pct']:.1f}%")