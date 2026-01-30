"""
Trade Manager - Clean SMC-Based Trade Level Calculator
=======================================================
Uses smartmoneyconcepts package for structure detection.
Calculates SL/TP based on trading mode and timeframe.

NO hardcoded multipliers - ONLY structure-based levels!
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Try to import smartmoneyconcepts, fall back to manual detection if not available
try:
    from smartmoneyconcepts import smc
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("⚠️ smartmoneyconcepts not installed. Install with: pip install smartmoneyconcepts")
    print("   Using fallback SMC detection (less accurate)")
    smc = None


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MODE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_MODES = {
    'Scalp': {
        'timeframes': ['1m', '5m'],
        'min_rr_tp1': 1.0,      # Minimum R:R for TP1
        'min_rr_tp2': 1.5,      # Minimum R:R for TP2
        'min_rr_tp3': 2.0,      # Minimum R:R for TP3
        'max_sl_pct': 1.0,      # Maximum stop loss %
        'swing_length': 5,      # For swing detection
        'ob_lookback': 20,      # Order block lookback
        'description': 'Quick trades, tight stops'
    },
    'Day Trade': {
        'timeframes': ['15m', '1h'],
        'min_rr_tp1': 1.5,
        'min_rr_tp2': 2.5,
        'min_rr_tp3': 4.0,
        'max_sl_pct': 3.0,
        'swing_length': 10,
        'ob_lookback': 50,
        'description': 'Intraday positions'
    },
    'Swing': {
        'timeframes': ['4h', '1d'],
        'min_rr_tp1': 2.0,
        'min_rr_tp2': 3.5,
        'min_rr_tp3': 5.0,
        'max_sl_pct': 5.0,
        'swing_length': 15,
        'ob_lookback': 100,
        'description': 'Multi-day positions'
    },
    'Investment': {
        'timeframes': ['1d', '1w'],
        'min_rr_tp1': 2.5,
        'min_rr_tp2': 4.0,
        'min_rr_tp3': 6.0,
        'max_sl_pct': 10.0,
        'swing_length': 20,
        'ob_lookback': 150,
        'description': 'Long-term positions'
    }
}

# Timeframe to trading mode mapping
TIMEFRAME_TO_MODE = {
    '1m': 'Scalp',
    '5m': 'Scalp',
    '15m': 'Day Trade',
    '30m': 'Day Trade',
    '1h': 'Day Trade',
    '4h': 'Swing',
    '1d': 'Swing',
    '1w': 'Investment',
    '1M': 'Investment'
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructureLevel:
    """A price level from SMC analysis"""
    price: float
    level_type: str  # 'swing_high', 'swing_low', 'ob_top', 'ob_bottom', 'fvg', 'liquidity'
    strength: int    # 1-5
    distance_pct: float  # Distance from current price
    description: str


@dataclass
class TradeLevels:
    """Complete trade setup"""
    direction: str  # 'LONG' or 'SHORT'
    
    # Entry
    entry: float
    entry_reason: str
    
    # Stop Loss
    stop_loss: float
    sl_reason: str
    sl_type: str
    risk_pct: float
    
    # Take Profits
    tp1: float
    tp1_reason: str
    tp1_rr: float
    
    tp2: float
    tp2_reason: str
    tp2_rr: float
    
    tp3: float
    tp3_reason: str
    tp3_rr: float
    
    # Validation
    valid: bool
    validation_msg: str
    warnings: List[str]
    
    # Mode info
    trading_mode: str
    timeframe: str


# ═══════════════════════════════════════════════════════════════════════════════
# SMC ANALYZER - Uses smartmoneyconcepts package
# ═══════════════════════════════════════════════════════════════════════════════

class SMCAnalyzer:
    """
    Analyzes market structure using smartmoneyconcepts package.
    Provides clean interface to all SMC detection methods.
    """
    
    def __init__(self, df: pd.DataFrame, trading_mode: str = 'Day Trade'):
        """
        Initialize SMC analyzer.
        
        Args:
            df: OHLCV DataFrame with lowercase columns ['open', 'high', 'low', 'close', 'volume']
            trading_mode: One of 'Scalp', 'Day Trade', 'Swing', 'Investment'
        """
        self.df = self._prepare_dataframe(df)
        self.trading_mode = trading_mode
        self.config = TRADING_MODES.get(trading_mode, TRADING_MODES['Day Trade'])
        
        self.current_price = self.df['close'].iloc[-1]
        
        # Run all SMC detection
        self._detect_all()
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has correct format for smartmoneyconcepts"""
        df = df.copy()
        
        # Rename columns to lowercase if needed
        column_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        }
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add volume if missing (some methods need it)
        if 'volume' not in df.columns:
            df['volume'] = 1000000  # Default volume
        
        return df
    
    def _detect_all(self):
        """Run all SMC detection methods"""
        swing_length = self.config['swing_length']
        
        if SMC_AVAILABLE:
            # Use smartmoneyconcepts package
            self.swing_hl = smc.swing_highs_lows(self.df, swing_length=swing_length)
            self.order_blocks = smc.ob(self.df, self.swing_hl, close_mitigation=False)
            self.fvg = smc.fvg(self.df, join_consecutive=True)
            self.bos_choch = smc.bos_choch(self.df, self.swing_hl, close_break=True)
            self.liquidity = smc.liquidity(self.df, self.swing_hl, range_percent=0.01)
        else:
            # Fallback: Manual detection
            self.swing_hl = self._detect_swings_fallback(swing_length)
            self.order_blocks = self._detect_ob_fallback()
            self.fvg = self._detect_fvg_fallback()
            self.bos_choch = pd.DataFrame({'BOS': [0]*len(self.df), 'CHOCH': [0]*len(self.df), 'Level': [np.nan]*len(self.df)})
            self.liquidity = pd.DataFrame()
    
    def _detect_swings_fallback(self, swing_length: int) -> pd.DataFrame:
        """Fallback swing detection when smartmoneyconcepts not available"""
        df = self.df
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        swing_hl = pd.DataFrame({
            'HighLow': [np.nan] * n,
            'Level': [np.nan] * n
        })
        
        for i in range(swing_length, n - swing_length):
            # Check for swing high
            is_swing_high = True
            for j in range(1, swing_length + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_hl.iloc[i] = [1, highs[i]]
            
            # Check for swing low
            is_swing_low = True
            for j in range(1, swing_length + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_hl.iloc[i] = [-1, lows[i]]
        
        return swing_hl
    
    def _detect_ob_fallback(self) -> pd.DataFrame:
        """Fallback order block detection"""
        df = self.df
        n = len(df)
        
        ob_data = {
            'OB': [0] * n,
            'Top': [np.nan] * n,
            'Bottom': [np.nan] * n,
            'OBVolume': [0] * n,
            'MitigatedIndex': [0] * n,
            'Percentage': [0.5] * n
        }
        
        # Simple OB detection: strong candle followed by reversal
        for i in range(2, n - 1):
            # Bullish OB: Down candle before up move
            if df['close'].iloc[i-1] < df['open'].iloc[i-1]:  # Red candle
                if df['close'].iloc[i] > df['open'].iloc[i]:  # Green candle after
                    if df['close'].iloc[i] > df['high'].iloc[i-1]:  # Breaks above
                        ob_data['OB'][i-1] = 1  # Bullish OB
                        ob_data['Top'][i-1] = df['high'].iloc[i-1]
                        ob_data['Bottom'][i-1] = df['low'].iloc[i-1]
            
            # Bearish OB: Up candle before down move
            if df['close'].iloc[i-1] > df['open'].iloc[i-1]:  # Green candle
                if df['close'].iloc[i] < df['open'].iloc[i]:  # Red candle after
                    if df['close'].iloc[i] < df['low'].iloc[i-1]:  # Breaks below
                        ob_data['OB'][i-1] = -1  # Bearish OB
                        ob_data['Top'][i-1] = df['high'].iloc[i-1]
                        ob_data['Bottom'][i-1] = df['low'].iloc[i-1]
        
        return pd.DataFrame(ob_data)
    
    def _detect_fvg_fallback(self) -> pd.DataFrame:
        """Fallback FVG detection"""
        df = self.df
        n = len(df)
        
        fvg_data = {
            'FVG': [0] * n,
            'Top': [np.nan] * n,
            'Bottom': [np.nan] * n,
            'MitigatedIndex': [0] * n
        }
        
        for i in range(2, n):
            # Bullish FVG: gap between candle[i-2] high and candle[i] low
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                fvg_data['FVG'][i-1] = 1
                fvg_data['Top'][i-1] = df['low'].iloc[i]
                fvg_data['Bottom'][i-1] = df['high'].iloc[i-2]
            
            # Bearish FVG: gap between candle[i-2] low and candle[i] high
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                fvg_data['FVG'][i-1] = -1
                fvg_data['Top'][i-1] = df['low'].iloc[i-2]
                fvg_data['Bottom'][i-1] = df['high'].iloc[i]
        
        return pd.DataFrame(fvg_data)
    
    def get_swing_highs(self) -> List[StructureLevel]:
        """Get all swing highs above current price"""
        levels = []
        
        for i in range(len(self.swing_hl)):
            if self.swing_hl['HighLow'].iloc[i] == 1:  # 1 = swing high
                price = self.swing_hl['Level'].iloc[i]
                if pd.notna(price) and price > self.current_price:
                    dist_pct = ((price - self.current_price) / self.current_price) * 100
                    levels.append(StructureLevel(
                        price=price,
                        level_type='swing_high',
                        strength=3,
                        distance_pct=dist_pct,
                        description='Swing High'
                    ))
        
        return sorted(levels, key=lambda x: x.price)
    
    def get_swing_lows(self) -> List[StructureLevel]:
        """Get all swing lows below current price"""
        levels = []
        
        for i in range(len(self.swing_hl)):
            if self.swing_hl['HighLow'].iloc[i] == -1:  # -1 = swing low
                price = self.swing_hl['Level'].iloc[i]
                if pd.notna(price) and price < self.current_price:
                    dist_pct = ((self.current_price - price) / self.current_price) * 100
                    levels.append(StructureLevel(
                        price=price,
                        level_type='swing_low',
                        strength=3,
                        distance_pct=dist_pct,
                        description='Swing Low'
                    ))
        
        return sorted(levels, key=lambda x: -x.price)  # Highest first (closest)
    
    def get_bullish_obs(self) -> List[StructureLevel]:
        """Get bullish order blocks (support zones below price)"""
        levels = []
        
        for i in range(len(self.order_blocks)):
            if self.order_blocks['OB'].iloc[i] == 1 and self.order_blocks['MitigatedIndex'].iloc[i] == 0:
                top = self.order_blocks['Top'].iloc[i]
                bottom = self.order_blocks['Bottom'].iloc[i]
                strength = self.order_blocks['Percentage'].iloc[i]
                
                if pd.notna(top) and top < self.current_price:  # Below current price = support
                    dist_pct = ((self.current_price - top) / self.current_price) * 100
                    levels.append(StructureLevel(
                        price=top,  # Top of OB = first touch for support
                        level_type='bullish_ob',
                        strength=int(min(5, max(1, strength * 5))) if pd.notna(strength) else 3,
                        distance_pct=dist_pct,
                        description=f'Bullish OB ({bottom:.4f}-{top:.4f})'
                    ))
        
        return sorted(levels, key=lambda x: -x.price)  # Highest first (closest)
    
    def get_bearish_obs(self) -> List[StructureLevel]:
        """Get bearish order blocks (resistance zones above price)"""
        levels = []
        
        for i in range(len(self.order_blocks)):
            if self.order_blocks['OB'].iloc[i] == -1 and self.order_blocks['MitigatedIndex'].iloc[i] == 0:
                top = self.order_blocks['Top'].iloc[i]
                bottom = self.order_blocks['Bottom'].iloc[i]
                strength = self.order_blocks['Percentage'].iloc[i]
                
                if pd.notna(bottom) and bottom > self.current_price:  # Above current price = resistance
                    dist_pct = ((bottom - self.current_price) / self.current_price) * 100
                    levels.append(StructureLevel(
                        price=bottom,  # Bottom of OB = first touch for resistance
                        level_type='bearish_ob',
                        strength=int(min(5, max(1, strength * 5))) if pd.notna(strength) else 3,
                        distance_pct=dist_pct,
                        description=f'Bearish OB ({bottom:.4f}-{top:.4f})'
                    ))
        
        return sorted(levels, key=lambda x: x.price)  # Lowest first (closest)
    
    def get_bullish_fvg(self) -> List[StructureLevel]:
        """Get bullish FVGs (support zones)"""
        levels = []
        
        for i in range(len(self.fvg)):
            if self.fvg['FVG'].iloc[i] == 1 and self.fvg['MitigatedIndex'].iloc[i] == 0:
                top = self.fvg['Top'].iloc[i]
                bottom = self.fvg['Bottom'].iloc[i]
                
                if pd.notna(top) and top < self.current_price:
                    dist_pct = ((self.current_price - top) / self.current_price) * 100
                    levels.append(StructureLevel(
                        price=top,
                        level_type='bullish_fvg',
                        strength=2,
                        distance_pct=dist_pct,
                        description=f'Bullish FVG ({bottom:.4f}-{top:.4f})'
                    ))
        
        return sorted(levels, key=lambda x: -x.price)
    
    def get_bearish_fvg(self) -> List[StructureLevel]:
        """Get bearish FVGs (resistance zones)"""
        levels = []
        
        for i in range(len(self.fvg)):
            if self.fvg['FVG'].iloc[i] == -1 and self.fvg['MitigatedIndex'].iloc[i] == 0:
                top = self.fvg['Top'].iloc[i]
                bottom = self.fvg['Bottom'].iloc[i]
                
                if pd.notna(bottom) and bottom > self.current_price:
                    dist_pct = ((bottom - self.current_price) / self.current_price) * 100
                    levels.append(StructureLevel(
                        price=bottom,
                        level_type='bearish_fvg',
                        strength=2,
                        distance_pct=dist_pct,
                        description=f'Bearish FVG ({bottom:.4f}-{top:.4f})'
                    ))
        
        return sorted(levels, key=lambda x: x.price)
    
    def get_market_structure(self) -> Dict:
        """Get current market structure (BOS/CHoCH status)"""
        last_bos = None
        last_choch = None
        
        # Find most recent BOS
        for i in range(len(self.bos_choch) - 1, -1, -1):
            bos_val = self.bos_choch['BOS'].iloc[i]
            if pd.notna(bos_val) and bos_val != 0:
                last_bos = {
                    'direction': 'BULLISH' if bos_val == 1 else 'BEARISH',
                    'level': self.bos_choch['Level'].iloc[i]
                }
                break
        
        # Find most recent CHoCH
        for i in range(len(self.bos_choch) - 1, -1, -1):
            choch_val = self.bos_choch['CHOCH'].iloc[i]
            if pd.notna(choch_val) and choch_val != 0:
                last_choch = {
                    'direction': 'BULLISH' if choch_val == 1 else 'BEARISH',
                    'level': self.bos_choch['Level'].iloc[i]
                }
                break
        
        # Determine overall structure
        if last_choch:
            structure = last_choch['direction']
        elif last_bos:
            structure = last_bos['direction']
        else:
            structure = 'NEUTRAL'
        
        return {
            'structure': structure,
            'last_bos': last_bos,
            'last_choch': last_choch
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE CALCULATOR - Calculates SL/TP from structure
# ═══════════════════════════════════════════════════════════════════════════════

class TradeCalculator:
    """
    Calculates trade levels (SL/TP) from SMC structure.
    Applies trading mode requirements for validation.
    """
    
    def __init__(self, analyzer: SMCAnalyzer, direction: str, entry_price: float = None):
        """
        Initialize trade calculator.
        
        Args:
            analyzer: SMCAnalyzer instance with detected structure
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price (defaults to current price)
        """
        self.analyzer = analyzer
        self.direction = direction.upper()
        self.entry = entry_price or analyzer.current_price
        self.config = analyzer.config
        self.warnings = []
    
    def calculate_stop_loss(self) -> Tuple[float, str, str]:
        """
        Calculate stop loss based on structure.
        
        Returns:
            (sl_price, sl_reason, sl_type)
        """
        max_sl_pct = self.config['max_sl_pct']
        
        if self.direction == 'LONG':
            # For LONG: SL below support (swing low, bullish OB, bullish FVG)
            candidates = []
            
            # 1. Swing lows
            for level in self.analyzer.get_swing_lows():
                if level.distance_pct <= max_sl_pct:
                    candidates.append((level.price * 0.998, level.strength, 'Below Swing Low', 'swing_low'))
            
            # 2. Bullish Order Blocks
            for level in self.analyzer.get_bullish_obs():
                if level.distance_pct <= max_sl_pct:
                    candidates.append((level.price * 0.998, level.strength + 1, f'Below {level.description}', 'bullish_ob'))
            
            # 3. Bullish FVGs
            for level in self.analyzer.get_bullish_fvg():
                if level.distance_pct <= max_sl_pct:
                    candidates.append((level.price * 0.998, level.strength, f'Below {level.description}', 'bullish_fvg'))
            
            if candidates:
                # Select highest strength, closest to entry
                candidates.sort(key=lambda x: (-x[1], -x[0]))  # Highest strength, highest price
                sl, strength, reason, sl_type = candidates[0]
                return sl, reason, sl_type
            else:
                # Fallback: ATR-based
                atr = self._calculate_atr()
                sl = self.entry - (atr * 2)
                self.warnings.append(f"No structure found for SL, using 2x ATR")
                return sl, '2x ATR (no structure)', 'atr_fallback'
        
        else:  # SHORT
            # For SHORT: SL above resistance (swing high, bearish OB, bearish FVG)
            candidates = []
            
            # 1. Swing highs
            for level in self.analyzer.get_swing_highs():
                if level.distance_pct <= max_sl_pct:
                    candidates.append((level.price * 1.002, level.strength, 'Above Swing High', 'swing_high'))
            
            # 2. Bearish Order Blocks
            for level in self.analyzer.get_bearish_obs():
                if level.distance_pct <= max_sl_pct:
                    candidates.append((level.price * 1.002, level.strength + 1, f'Above {level.description}', 'bearish_ob'))
            
            # 3. Bearish FVGs
            for level in self.analyzer.get_bearish_fvg():
                if level.distance_pct <= max_sl_pct:
                    candidates.append((level.price * 1.002, level.strength, f'Above {level.description}', 'bearish_fvg'))
            
            if candidates:
                # Select highest strength, closest to entry
                candidates.sort(key=lambda x: (-x[1], x[0]))  # Highest strength, lowest price
                sl, strength, reason, sl_type = candidates[0]
                return sl, reason, sl_type
            else:
                # Fallback: ATR-based
                atr = self._calculate_atr()
                sl = self.entry + (atr * 2)
                self.warnings.append(f"No structure found for SL, using 2x ATR")
                return sl, '2x ATR (no structure)', 'atr_fallback'
    
    def calculate_take_profits(self, stop_loss: float) -> List[Tuple[float, str, float]]:
        """
        Calculate take profit levels based on structure.
        
        Args:
            stop_loss: Stop loss price (to calculate R:R)
            
        Returns:
            List of (tp_price, tp_reason, tp_rr)
        """
        risk = abs(self.entry - stop_loss)
        if risk == 0:
            return []
        
        min_rr = [self.config['min_rr_tp1'], self.config['min_rr_tp2'], self.config['min_rr_tp3']]
        
        if self.direction == 'LONG':
            # For LONG: TPs at resistance (swing high, bearish OB, bearish FVG)
            targets = []
            
            # Collect all resistance levels
            for level in self.analyzer.get_swing_highs():
                rr = (level.price - self.entry) / risk
                targets.append((level.price, level.description, rr, level.strength))
            
            for level in self.analyzer.get_bearish_obs():
                rr = (level.price - self.entry) / risk
                targets.append((level.price, level.description, rr, level.strength + 1))
            
            for level in self.analyzer.get_bearish_fvg():
                rr = (level.price - self.entry) / risk
                targets.append((level.price, level.description, rr, level.strength))
            
            # Sort by price (ascending - closest first)
            targets.sort(key=lambda x: x[0])
            
        else:  # SHORT
            # For SHORT: TPs at support (swing low, bullish OB, bullish FVG)
            targets = []
            
            for level in self.analyzer.get_swing_lows():
                rr = (self.entry - level.price) / risk
                targets.append((level.price, level.description, rr, level.strength))
            
            for level in self.analyzer.get_bullish_obs():
                rr = (self.entry - level.price) / risk
                targets.append((level.price, level.description, rr, level.strength + 1))
            
            for level in self.analyzer.get_bullish_fvg():
                rr = (self.entry - level.price) / risk
                targets.append((level.price, level.description, rr, level.strength))
            
            # Sort by price (descending - closest first)
            targets.sort(key=lambda x: -x[0])
        
        # Select TPs that meet minimum R:R requirements
        tps = []
        used_prices = set()
        
        for i, min_r in enumerate(min_rr):
            # Find first target that meets min R:R and isn't already used
            found = False
            for price, reason, rr, strength in targets:
                if rr >= min_r and price not in used_prices:
                    tps.append((price, reason, rr))
                    used_prices.add(price)
                    found = True
                    break
            
            if not found:
                # Fallback: R:R based
                if self.direction == 'LONG':
                    fallback_price = self.entry + (risk * min_r * 1.2)
                else:
                    fallback_price = self.entry - (risk * min_r * 1.2)
                
                tps.append((fallback_price, f'{min_r*1.2:.1f}R (no structure)', min_r * 1.2))
                self.warnings.append(f"TP{i+1}: No structure meeting {min_r}:1 R:R")
        
        # Ensure correct order
        if self.direction == 'LONG':
            tps.sort(key=lambda x: x[0])  # Ascending
        else:
            tps.sort(key=lambda x: -x[0])  # Descending
        
        return tps
    
    def _calculate_atr(self, period: int = 14) -> float:
        """Calculate ATR for fallback calculations"""
        df = self.analyzer.df
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            abs(high - close),
            abs(low - close)
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def calculate_all(self) -> TradeLevels:
        """Calculate complete trade setup"""
        # Calculate SL
        sl, sl_reason, sl_type = self.calculate_stop_loss()
        risk = abs(self.entry - sl)
        risk_pct = (risk / self.entry) * 100
        
        # Calculate TPs
        tps = self.calculate_take_profits(sl)
        
        # Validate
        valid = True
        validation_msg = "Trade setup valid"
        
        if risk_pct > self.config['max_sl_pct']:
            valid = False
            validation_msg = f"SL {risk_pct:.1f}% exceeds max {self.config['max_sl_pct']}%"
        
        if len(tps) < 3:
            valid = False
            validation_msg = "Could not find 3 valid TPs"
        
        return TradeLevels(
            direction=self.direction,
            entry=self.entry,
            entry_reason='Current Price',
            stop_loss=sl,
            sl_reason=sl_reason,
            sl_type=sl_type,
            risk_pct=risk_pct,
            tp1=tps[0][0] if len(tps) > 0 else self.entry,
            tp1_reason=tps[0][1] if len(tps) > 0 else 'N/A',
            tp1_rr=tps[0][2] if len(tps) > 0 else 0,
            tp2=tps[1][0] if len(tps) > 1 else self.entry,
            tp2_reason=tps[1][1] if len(tps) > 1 else 'N/A',
            tp2_rr=tps[1][2] if len(tps) > 1 else 0,
            tp3=tps[2][0] if len(tps) > 2 else self.entry,
            tp3_reason=tps[2][1] if len(tps) > 2 else 'N/A',
            tp3_rr=tps[2][2] if len(tps) > 2 else 0,
            valid=valid,
            validation_msg=validation_msg,
            warnings=self.warnings,
            trading_mode=self.analyzer.trading_mode,
            timeframe=''
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_trade_levels(
    df: pd.DataFrame,
    direction: str,
    timeframe: str = '15m',
    trading_mode: str = None,
    entry_price: float = None
) -> TradeLevels:
    """
    Calculate trade levels using SMC analysis.
    
    Args:
        df: OHLCV DataFrame
        direction: 'LONG' or 'SHORT'
        timeframe: Chart timeframe
        trading_mode: Override auto-detected mode (optional)
        entry_price: Entry price (optional, defaults to current)
        
    Returns:
        TradeLevels with SL/TP based on structure
    """
    # Auto-detect trading mode from timeframe if not specified
    if trading_mode is None:
        trading_mode = TIMEFRAME_TO_MODE.get(timeframe, 'Day Trade')
    
    # Run SMC analysis
    analyzer = SMCAnalyzer(df, trading_mode)
    
    # Calculate trade levels
    calculator = TradeCalculator(analyzer, direction, entry_price)
    levels = calculator.calculate_all()
    levels.timeframe = timeframe
    
    return levels


def get_trading_mode_config(mode: str) -> Dict:
    """Get configuration for a trading mode"""
    return TRADING_MODES.get(mode, TRADING_MODES['Day Trade'])


def get_mode_for_timeframe(timeframe: str) -> str:
    """Get recommended trading mode for a timeframe"""
    return TIMEFRAME_TO_MODE.get(timeframe, 'Day Trade')


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    print("=" * 60)
    print("TRADE MANAGER TEST")
    print("=" * 60)
    
    # Test LONG trade
    levels = calculate_trade_levels(df, 'LONG', '15m')
    
    print(f"\nDirection: {levels.direction}")
    print(f"Trading Mode: {levels.trading_mode}")
    print(f"Entry: ${levels.entry:.4f}")
    print(f"\nStop Loss: ${levels.stop_loss:.4f} ({levels.sl_reason})")
    print(f"Risk: {levels.risk_pct:.2f}%")
    print(f"\nTP1: ${levels.tp1:.4f} (R:R {levels.tp1_rr:.1f}:1) - {levels.tp1_reason}")
    print(f"TP2: ${levels.tp2:.4f} (R:R {levels.tp2_rr:.1f}:1) - {levels.tp2_reason}")
    print(f"TP3: ${levels.tp3:.4f} (R:R {levels.tp3_rr:.1f}:1) - {levels.tp3_reason}")
    print(f"\nValid: {levels.valid}")
    print(f"Message: {levels.validation_msg}")
    if levels.warnings:
        print(f"Warnings: {levels.warnings}")
    
    print("\n" + "=" * 60)