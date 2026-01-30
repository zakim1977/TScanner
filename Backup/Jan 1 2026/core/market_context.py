"""
═══════════════════════════════════════════════════════════════════════════════
MARKET CONTEXT - Single Container for ALL Gathered Metrics
═══════════════════════════════════════════════════════════════════════════════

This is STEP 1 of the unified architecture:
1. GATHER ALL METRICS → MarketContext
2. DECISION ENGINE → TradeDecision  
3. LEVEL GENERATION → TradeLevels (only if actionable)
4. DISPLAY → UI

MarketContext holds everything needed to make a trading decision.
It's passed to MASTER_RULES which returns the final decision.

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class MarketContext:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SINGLE CONTAINER FOR ALL MARKET METRICS
    ═══════════════════════════════════════════════════════════════════════════
    
    This dataclass holds EVERYTHING needed to make a trading decision.
    It's gathered ONCE and passed to the decision engine.
    
    No more scattered data across multiple dicts!
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # IDENTITY
    # ═══════════════════════════════════════════════════════════════════════
    symbol: str
    timeframe: str
    trading_mode: str  # 'scalp', 'day_trade', 'swing', 'investment'
    market_type: str   # 'crypto', 'stock', 'etf'
    timestamp: datetime = field(default_factory=datetime.now)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRICE DATA
    # ═══════════════════════════════════════════════════════════════════════
    current_price: float = 0.0
    price_change_24h: float = 0.0  # % change
    
    # ═══════════════════════════════════════════════════════════════════════
    # WHALE/INSTITUTIONAL DATA (Leading Indicators)
    # ═══════════════════════════════════════════════════════════════════════
    # Crypto: From Binance API
    whale_pct: float = 50.0        # Top trader long % (0-100)
    retail_pct: float = 50.0       # Retail long % (0-100)
    oi_change_24h: float = 0.0     # Open Interest change %
    funding_rate: float = 0.0      # Funding rate
    
    # Stocks/ETFs: From Quiver API
    insider_score: Optional[float] = None    # 0-100
    congress_score: Optional[float] = None   # 0-100
    short_interest_pct: Optional[float] = None
    institutional_score: Optional[float] = None  # Combined score
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURE & POSITION
    # ═══════════════════════════════════════════════════════════════════════
    swing_high: float = 0.0
    swing_low: float = 0.0
    position_in_range: float = 50.0  # 0=at lows, 100=at highs
    structure_type: str = 'Mixed'    # 'Bullish', 'Bearish', 'Mixed', 'Ranging'
    trend: str = 'Neutral'           # 'Bullish', 'Bearish', 'Neutral'
    
    # ═══════════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS (Lagging - for confirmation only)
    # ═══════════════════════════════════════════════════════════════════════
    ta_score: int = 50           # Overall TA score 0-100
    rsi: float = 50.0
    mfi: float = 50.0
    cmf: float = 0.0
    atr_pct: float = 2.0         # ATR as % of price (volatility)
    volume_ratio: float = 1.0    # Current vol / avg vol
    
    # ═══════════════════════════════════════════════════════════════════════
    # MONEY FLOW CONTEXT
    # ═══════════════════════════════════════════════════════════════════════
    money_flow_phase: str = 'CONSOLIDATION'  # ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN
    is_accumulating: bool = False
    is_distributing: bool = False
    
    # ═══════════════════════════════════════════════════════════════════════
    # SMC PATTERNS (Optional)
    # ═══════════════════════════════════════════════════════════════════════
    has_fvg: bool = False
    has_order_block: bool = False
    has_liquidity_sweep: bool = False
    smc_bias: str = 'Neutral'  # 'Bullish', 'Bearish', 'Neutral'
    
    # ═══════════════════════════════════════════════════════════════════════
    # BTC CORRELATION (Crypto only)
    # ═══════════════════════════════════════════════════════════════════════
    btc_correlation: Optional[float] = None  # -1 to 1
    btc_trend: str = 'Neutral'
    btc_price: float = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # RAW DATA (for reference/debugging)
    # ═══════════════════════════════════════════════════════════════════════
    raw_whale_data: Dict[str, Any] = field(default_factory=dict)
    raw_smc_data: Dict[str, Any] = field(default_factory=dict)
    raw_money_flow: Dict[str, Any] = field(default_factory=dict)
    raw_institutional: Dict[str, Any] = field(default_factory=dict)
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTED PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def divergence(self) -> float:
        """Whale% - Retail% (positive = whales more bullish)"""
        return self.whale_pct - self.retail_pct
    
    @property
    def position_label(self) -> str:
        """Convert position_in_range to label"""
        if self.position_in_range <= 35:
            return 'EARLY'
        elif self.position_in_range >= 65:
            return 'LATE'
        else:
            return 'MIDDLE'
    
    @property
    def is_crypto(self) -> bool:
        return self.market_type.lower() == 'crypto'
    
    @property
    def is_stock(self) -> bool:
        return self.market_type.lower() in ['stock', 'etf']
    
    @property
    def has_whale_data(self) -> bool:
        """Check if we have real whale data (not defaults)"""
        return self.whale_pct != 50.0 or self.retail_pct != 50.0
    
    @property
    def has_institutional_data(self) -> bool:
        """Check if we have stock institutional data"""
        return self.insider_score is not None or self.congress_score is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trading_mode': self.trading_mode,
            'market_type': self.market_type,
            'current_price': self.current_price,
            'price_change_24h': self.price_change_24h,
            'whale_pct': self.whale_pct,
            'retail_pct': self.retail_pct,
            'oi_change_24h': self.oi_change_24h,
            'position_in_range': self.position_in_range,
            'position_label': self.position_label,
            'divergence': self.divergence,
            'ta_score': self.ta_score,
            'money_flow_phase': self.money_flow_phase,
            'structure_type': self.structure_type,
            'trend': self.trend,
        }


def build_market_context(
    symbol: str,
    timeframe: str,
    trading_mode: str,
    market_type: str,
    df,  # DataFrame with OHLCV
    whale_data: Dict[str, Any] = None,
    smc_data: Dict[str, Any] = None,
    money_flow: Dict[str, Any] = None,
    institutional: Dict[str, Any] = None,
    ta_score: int = 50,
    btc_context: Dict[str, Any] = None,
) -> MarketContext:
    """
    ═══════════════════════════════════════════════════════════════════════════
    BUILD MARKET CONTEXT FROM RAW DATA
    ═══════════════════════════════════════════════════════════════════════════
    
    This is the GATHERER function. It takes all raw data sources and builds
    a single MarketContext object.
    
    Call this ONCE, then pass the result to MASTER_RULES.
    """
    
    whale_data = whale_data or {}
    smc_data = smc_data or {}
    money_flow = money_flow or {}
    institutional = institutional or {}
    btc_context = btc_context or {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRICE DATA
    # ═══════════════════════════════════════════════════════════════════════
    current_price = float(df['Close'].iloc[-1]) if df is not None and len(df) > 0 else 0.0
    
    # Get price change from whale data or calculate
    price_change_24h = 0.0
    if whale_data.get('real_whale_data'):
        price_change_24h = whale_data['real_whale_data'].get('price_change_24h', 0)
    elif df is not None and len(df) > 24:
        price_change_24h = ((df['Close'].iloc[-1] / df['Close'].iloc[-24]) - 1) * 100
    
    # ═══════════════════════════════════════════════════════════════════════
    # WHALE DATA (Crypto)
    # ═══════════════════════════════════════════════════════════════════════
    top_trader = whale_data.get('top_trader_ls', {})
    retail_data = whale_data.get('retail_ls', {})
    oi_data = whale_data.get('open_interest', {})
    
    whale_pct = top_trader.get('long_pct', 50) if isinstance(top_trader, dict) else 50
    retail_pct = retail_data.get('long_pct', 50) if isinstance(retail_data, dict) else 50
    oi_change = oi_data.get('change_24h', 0) if isinstance(oi_data, dict) else 0
    funding_rate = whale_data.get('funding_rate', 0)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STOCK INSTITUTIONAL DATA
    # ═══════════════════════════════════════════════════════════════════════
    stock_inst = institutional.get('stock_inst_data', {})
    insider_score = stock_inst.get('insider_score') if stock_inst.get('available') else None
    congress_score = stock_inst.get('congress_score') if stock_inst.get('available') else None
    short_interest = stock_inst.get('short_interest_pct') if stock_inst.get('available') else None
    inst_score = stock_inst.get('combined_score') if stock_inst.get('available') else None
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURE & POSITION
    # ═══════════════════════════════════════════════════════════════════════
    smc_struct = smc_data.get('structure', {}) if smc_data else {}
    swing_high = smc_struct.get('last_swing_high', 0) or 0
    swing_low = smc_struct.get('last_swing_low', 0) or 0
    structure_type = smc_struct.get('structure', 'Mixed')
    trend = smc_struct.get('trend', 'Neutral')
    
    # Calculate position in range
    if swing_high and swing_low and swing_high > swing_low:
        position_in_range = ((current_price - swing_low) / (swing_high - swing_low)) * 100
        position_in_range = max(0, min(100, position_in_range))
    else:
        position_in_range = 50
    
    # ═══════════════════════════════════════════════════════════════════════
    # MONEY FLOW
    # ═══════════════════════════════════════════════════════════════════════
    is_accumulating = money_flow.get('is_accumulating', False)
    is_distributing = money_flow.get('is_distributing', False)
    mfi = money_flow.get('mfi', 50)
    cmf = money_flow.get('cmf', 0)
    rsi = money_flow.get('rsi', 50)
    atr_pct = money_flow.get('atr_pct', 2.0)
    volume_ratio = money_flow.get('volume_ratio', 1.0)
    
    # Determine money flow phase
    if is_accumulating:
        money_flow_phase = 'ACCUMULATION'
    elif is_distributing:
        money_flow_phase = 'DISTRIBUTION'
    else:
        flow_status = money_flow.get('flow_status', '')
        if 'MARKUP' in flow_status.upper():
            money_flow_phase = 'MARKUP'
        elif 'MARKDOWN' in flow_status.upper():
            money_flow_phase = 'MARKDOWN'
        else:
            money_flow_phase = 'CONSOLIDATION'
    
    # ═══════════════════════════════════════════════════════════════════════
    # SMC PATTERNS
    # ═══════════════════════════════════════════════════════════════════════
    has_fvg = bool(smc_data.get('fvg'))
    has_order_block = bool(smc_data.get('order_blocks'))
    has_liquidity_sweep = bool(smc_data.get('liquidity_sweeps'))
    smc_bias = smc_data.get('bias', 'Neutral') if smc_data else 'Neutral'
    
    # ═══════════════════════════════════════════════════════════════════════
    # BTC CORRELATION
    # ═══════════════════════════════════════════════════════════════════════
    btc_correlation = btc_context.get('correlation')
    btc_trend = btc_context.get('trend', 'Neutral')
    btc_price = btc_context.get('price', 0)
    
    # ═══════════════════════════════════════════════════════════════════════
    # BUILD CONTEXT
    # ═══════════════════════════════════════════════════════════════════════
    
    return MarketContext(
        # Identity
        symbol=symbol,
        timeframe=timeframe,
        trading_mode=trading_mode,
        market_type=market_type,
        
        # Price
        current_price=current_price,
        price_change_24h=price_change_24h,
        
        # Whale/Institutional
        whale_pct=whale_pct,
        retail_pct=retail_pct,
        oi_change_24h=oi_change,
        funding_rate=funding_rate,
        insider_score=insider_score,
        congress_score=congress_score,
        short_interest_pct=short_interest,
        institutional_score=inst_score,
        
        # Structure
        swing_high=swing_high,
        swing_low=swing_low,
        position_in_range=position_in_range,
        structure_type=structure_type,
        trend=trend,
        
        # Technical
        ta_score=ta_score,
        rsi=rsi,
        mfi=mfi,
        cmf=cmf,
        atr_pct=atr_pct,
        volume_ratio=volume_ratio,
        
        # Money Flow
        money_flow_phase=money_flow_phase,
        is_accumulating=is_accumulating,
        is_distributing=is_distributing,
        
        # SMC
        has_fvg=has_fvg,
        has_order_block=has_order_block,
        has_liquidity_sweep=has_liquidity_sweep,
        smc_bias=smc_bias,
        
        # BTC
        btc_correlation=btc_correlation,
        btc_trend=btc_trend,
        btc_price=btc_price,
        
        # Raw data for reference
        raw_whale_data=whale_data,
        raw_smc_data=smc_data,
        raw_money_flow=money_flow,
        raw_institutional=institutional,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MODE REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_MODE_REQUIREMENTS = {
    'scalp': {
        'max_sl_pct': 1.5,
        'min_rr': 1.5,
        'timeframes': ['1m', '5m'],
        'hold_time': '< 1 hour',
    },
    'day_trade': {
        'max_sl_pct': 3.0,
        'min_rr': 2.0,
        'timeframes': ['15m', '1h'],
        'hold_time': '1-24 hours',
    },
    'swing': {
        'max_sl_pct': 8.0,
        'min_rr': 2.5,
        'timeframes': ['4h', '1d'],
        'hold_time': '2-14 days',
    },
    'investment': {
        'max_sl_pct': 15.0,
        'min_rr': 3.0,
        'timeframes': ['1w'],
        'hold_time': '> 14 days',
    },
}


def get_mode_from_timeframe(timeframe: str) -> str:
    """Auto-detect trading mode from timeframe"""
    tf_lower = timeframe.lower()
    if tf_lower in ['1m', '5m']:
        return 'scalp'
    elif tf_lower in ['15m', '1h']:
        return 'day_trade'
    elif tf_lower in ['4h', '1d']:
        return 'swing'
    elif tf_lower in ['1w', '1M']:
        return 'investment'
    else:
        return 'day_trade'  # Default


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test MarketContext
    ctx = MarketContext(
        symbol='BTCUSDT',
        timeframe='1h',
        trading_mode='day_trade',
        market_type='crypto',
        current_price=88000,
        whale_pct=68,
        retail_pct=55,
        oi_change_24h=2.5,
        position_in_range=30,
        ta_score=65,
    )
    
    print("MarketContext Test:")
    print(f"  Symbol: {ctx.symbol}")
    print(f"  Divergence: {ctx.divergence}")
    print(f"  Position Label: {ctx.position_label}")
    print(f"  Has Whale Data: {ctx.has_whale_data}")
    print(f"  Is Crypto: {ctx.is_crypto}")
    print()
    print(f"  Dict: {ctx.to_dict()}")
