"""
Deep Learning Pattern Detection System
=======================================
Detects chart patterns using CNN (images) and LSTM (sequences).

Patterns Detected:
- Reversal: Double Top (M), Double Bottom (W), Head & Shoulders, Inverse H&S
- Continuation: Bull Flag, Bear Flag, Triangle, Wedge, Pennant
- Breakout: Cup & Handle, Ascending/Descending Triangle

Architecture:
1. CNN Path: Converts OHLCV to candlestick image, classifies pattern
2. LSTM Path: Processes raw sequence, predicts continuation/reversal
3. Combined: Ensemble of both for final prediction

Modes:
- Scalp: 50 candles, focus on quick patterns (flags, pennants)
- DayTrade: 100 candles, all patterns
- Swing: 150 candles, larger formations (H&S, cup & handle)
- Investment: 200 candles, major trend patterns

Markets:
- Crypto: Higher volatility thresholds
- Stocks: Standard thresholds
- ETFs: Lower volatility thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
HAS_TORCH = False
TORCH_ERROR = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError as e:
    TORCH_ERROR = f"PyTorch not installed: {e}"
except OSError as e:
    # Windows DLL loading error
    TORCH_ERROR = f"PyTorch DLL error (Windows): {e}"
except Exception as e:
    TORCH_ERROR = f"PyTorch load error: {e}"

if TORCH_ERROR:
    print(f"⚠️ {TORCH_ERROR}")
    print("Fix: Reinstall PyTorch with the correct version for your system")

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class PatternType(Enum):
    """All detectable chart patterns"""
    # Reversal Patterns (Bearish)
    DOUBLE_TOP = "double_top"           # M shape
    HEAD_SHOULDERS = "head_shoulders"   # H&S
    TRIPLE_TOP = "triple_top"
    RISING_WEDGE = "rising_wedge"       # Bearish
    
    # Reversal Patterns (Bullish)
    DOUBLE_BOTTOM = "double_bottom"     # W shape
    INV_HEAD_SHOULDERS = "inv_head_shoulders"
    TRIPLE_BOTTOM = "triple_bottom"
    FALLING_WEDGE = "falling_wedge"     # Bullish
    
    # Continuation Patterns (Bullish)
    BULL_FLAG = "bull_flag"
    BULL_PENNANT = "bull_pennant"
    ASC_TRIANGLE = "ascending_triangle"
    CUP_HANDLE = "cup_handle"
    
    # Continuation Patterns (Bearish)
    BEAR_FLAG = "bear_flag"
    BEAR_PENNANT = "bear_pennant"
    DESC_TRIANGLE = "descending_triangle"
    
    # Neutral
    SYM_TRIANGLE = "symmetric_triangle"
    RECTANGLE = "rectangle"
    NO_PATTERN = "no_pattern"


PATTERN_LABELS = [p.value for p in PatternType]

PATTERN_DIRECTION = {
    PatternType.DOUBLE_TOP: "BEARISH",
    PatternType.HEAD_SHOULDERS: "BEARISH",
    PatternType.TRIPLE_TOP: "BEARISH",
    PatternType.RISING_WEDGE: "BEARISH",
    PatternType.DOUBLE_BOTTOM: "BULLISH",
    PatternType.INV_HEAD_SHOULDERS: "BULLISH",
    PatternType.TRIPLE_BOTTOM: "BULLISH",
    PatternType.FALLING_WEDGE: "BULLISH",
    PatternType.BULL_FLAG: "BULLISH",
    PatternType.BULL_PENNANT: "BULLISH",
    PatternType.ASC_TRIANGLE: "BULLISH",
    PatternType.CUP_HANDLE: "BULLISH",
    PatternType.BEAR_FLAG: "BEARISH",
    PatternType.BEAR_PENNANT: "BEARISH",
    PatternType.DESC_TRIANGLE: "BEARISH",
    PatternType.SYM_TRIANGLE: "NEUTRAL",
    PatternType.RECTANGLE: "NEUTRAL",
    PatternType.NO_PATTERN: "NEUTRAL",
}

# Mode-specific pattern focus
MODE_PATTERNS = {
    'scalp': [
        PatternType.BULL_FLAG, PatternType.BEAR_FLAG,
        PatternType.BULL_PENNANT, PatternType.BEAR_PENNANT,
        PatternType.NO_PATTERN
    ],
    'daytrade': [
        PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM,
        PatternType.BULL_FLAG, PatternType.BEAR_FLAG,
        PatternType.ASC_TRIANGLE, PatternType.DESC_TRIANGLE,
        PatternType.SYM_TRIANGLE, PatternType.NO_PATTERN
    ],
    'swing': [
        PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM,
        PatternType.HEAD_SHOULDERS, PatternType.INV_HEAD_SHOULDERS,
        PatternType.RISING_WEDGE, PatternType.FALLING_WEDGE,
        PatternType.CUP_HANDLE, PatternType.NO_PATTERN
    ],
    # Investment: Include ALL major patterns since they're rare on long timeframes
    'investment': [
        PatternType.DOUBLE_TOP, PatternType.DOUBLE_BOTTOM,
        PatternType.HEAD_SHOULDERS, PatternType.INV_HEAD_SHOULDERS,
        PatternType.TRIPLE_TOP, PatternType.TRIPLE_BOTTOM,
        PatternType.RISING_WEDGE, PatternType.FALLING_WEDGE,
        PatternType.CUP_HANDLE, PatternType.ASC_TRIANGLE,
        PatternType.DESC_TRIANGLE, PatternType.NO_PATTERN
    ]
}

# Candle sequence length per mode
MODE_SEQUENCE_LENGTH = {
    'scalp': 50,
    'daytrade': 100,
    'swing': 150,
    'investment': 200
}

# Market-specific volatility multipliers
# Higher = more lenient pattern detection (larger price movements required)
MARKET_VOLATILITY = {
    'crypto': 1.5,    # Higher thresholds (crypto is volatile)
    'stocks': 1.2,    # Slightly higher than 1.0 for more patterns
    'etfs': 1.0       # Standard thresholds (was 0.7, too strict)
}

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET-SPECIFIC PATTERN THRESHOLDS
# These control how strict/loose each pattern detection is per market
# For ETFs: We WANT more double_bottoms (accumulation) and fewer double_tops
# ═══════════════════════════════════════════════════════════════════════════════
PATTERN_THRESHOLDS = {
    'crypto': {
        'double_top': {
            'similarity_pct': 0.03,      # Peaks within 3%
            'depth_pct': 0.04,           # Valley 4% below peaks
            'min_distance': 10,          # Min candles between peaks
        },
        'double_bottom': {
            'similarity_pct': 0.03,      # Lows within 3%
            'depth_pct': 0.04,           # Peak 4% above lows
            'min_distance': 10,
        },
        'head_shoulders': {
            'shoulder_diff_pct': 0.02,   # Shoulders within 2%
            'head_prominence_pct': 0.03, # Head 3% above shoulders
        },
    },
    'stocks': {
        'double_top': {
            'similarity_pct': 0.025,
            'depth_pct': 0.035,
            'min_distance': 8,
        },
        'double_bottom': {
            'similarity_pct': 0.025,
            'depth_pct': 0.035,
            'min_distance': 8,
        },
        'head_shoulders': {
            'shoulder_diff_pct': 0.02,
            'head_prominence_pct': 0.025,
        },
    },
    'etfs': {
        # ETF-SPECIFIC: Stricter double_top, Looser double_bottom for accumulation bias
        'double_top': {
            'similarity_pct': 0.015,     # STRICTER: Peaks must be within 1.5% (was 2%)
            'depth_pct': 0.04,           # STRICTER: Valley must be 4% below (was 3%)
            'min_distance': 15,          # STRICTER: More candles between peaks
            'min_confidence': 70,        # STRICTER: Higher confidence required
        },
        'double_bottom': {
            'similarity_pct': 0.03,      # LOOSER: Lows within 3% (was 2%)
            'depth_pct': 0.02,           # LOOSER: Peak only 2% above (was 3%)
            'min_distance': 8,           # LOOSER: Fewer candles required
            'min_confidence': 50,        # LOOSER: Lower confidence OK
        },
        'head_shoulders': {
            'shoulder_diff_pct': 0.015,
            'head_prominence_pct': 0.02,
        },
        'inv_head_shoulders': {
            'shoulder_diff_pct': 0.025,  # LOOSER for bullish pattern
            'head_prominence_pct': 0.015,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatternDetection:
    """Result of pattern detection"""
    pattern: PatternType
    confidence: float           # 0-100
    direction: str              # BULLISH, BEARISH, NEUTRAL
    target_pct: float           # Expected move %
    stop_pct: float             # Suggested stop %
    formation_start: int        # Candle index where pattern starts
    formation_end: int          # Candle index where pattern ends
    neckline: Optional[float] = None  # For H&S patterns
    breakout_level: Optional[float] = None


@dataclass 
class CombinedSignal:
    """Combined signal from Traditional ML + Pattern Detection"""
    # Traditional ML
    trad_direction: str         # LONG, SHORT, WAIT
    trad_confidence: float      # 0-100
    trad_probabilities: Dict    # {label: probability}
    
    # Pattern Detection
    pattern: PatternType
    pattern_confidence: float   # 0-100
    pattern_direction: str      # BULLISH, BEARISH, NEUTRAL
    
    # Combined
    final_direction: str        # LONG, SHORT, WAIT
    final_confidence: float     # 0-100
    agreement: str              # STRONG_AGREE, AGREE, NEUTRAL, CONFLICT
    
    # Targets
    suggested_tp_pct: float
    suggested_sl_pct: float


# ═══════════════════════════════════════════════════════════════════════════════
# SEQUENCE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_sequence(df: pd.DataFrame, seq_length: int = 100) -> np.ndarray:
    """
    Normalize OHLCV data for LSTM input.
    
    Returns shape: (seq_length, 5) for OHLCV
    """
    if len(df) < seq_length:
        # Pad with first values
        padding = pd.DataFrame([df.iloc[0]] * (seq_length - len(df)))
        df = pd.concat([padding, df], ignore_index=True)
    elif len(df) > seq_length:
        df = df.tail(seq_length).reset_index(drop=True)
    
    # Get OHLCV
    ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(float)
    
    # Normalize price relative to first close
    first_close = ohlcv[0, 3]
    if first_close > 0:
        ohlcv[:, :4] = (ohlcv[:, :4] / first_close - 1) * 100  # Percentage change
    
    # Normalize volume
    vol_mean = ohlcv[:, 4].mean()
    if vol_mean > 0:
        ohlcv[:, 4] = ohlcv[:, 4] / vol_mean
    
    return ohlcv.astype(np.float32)


def create_pattern_image(df: pd.DataFrame, img_size: int = 64) -> np.ndarray:
    """
    Convert OHLCV to grayscale candlestick image for CNN.
    
    Returns shape: (1, img_size, img_size) grayscale image
    """
    seq_length = len(df)
    
    # Normalize prices to 0-1 range
    prices = df[['Open', 'High', 'Low', 'Close']].values
    price_min = prices.min()
    price_max = prices.max()
    price_range = price_max - price_min
    
    if price_range == 0:
        price_range = 1
    
    # Create blank image
    img = np.zeros((img_size, img_size), dtype=np.float32)
    
    # Draw candlesticks
    candle_width = max(1, img_size // seq_length)
    
    for i, (_, row) in enumerate(df.iterrows()):
        x = int(i * img_size / seq_length)
        
        # Normalize OHLC
        o = int((row['Open'] - price_min) / price_range * (img_size - 1))
        h = int((row['High'] - price_min) / price_range * (img_size - 1))
        l = int((row['Low'] - price_min) / price_range * (img_size - 1))
        c = int((row['Close'] - price_min) / price_range * (img_size - 1))
        
        # Invert y-axis (image coordinates)
        o, h, l, c = img_size - 1 - o, img_size - 1 - h, img_size - 1 - l, img_size - 1 - c
        
        # Draw wick
        for y in range(min(h, l), max(h, l) + 1):
            if 0 <= y < img_size and 0 <= x < img_size:
                img[y, x] = 0.5
        
        # Draw body
        body_top = min(o, c)
        body_bottom = max(o, c)
        intensity = 1.0 if row['Close'] >= row['Open'] else 0.3  # Green vs Red
        
        for y in range(body_top, body_bottom + 1):
            for dx in range(candle_width):
                if 0 <= y < img_size and 0 <= x + dx < img_size:
                    img[y, x + dx] = intensity
    
    return img.reshape(1, img_size, img_size)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED PATTERN DETECTION (Baseline + Label Generation)
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedPatternDetector:
    """
    Rule-based pattern detection for:
    1. Baseline comparison
    2. Generating training labels for deep learning
    """
    
    def __init__(self, market: str = 'crypto'):
        self.market = market.lower()
        self.vol_mult = MARKET_VOLATILITY.get(self.market, 1.0)
        # Get market-specific pattern thresholds
        self.pattern_thresholds = PATTERN_THRESHOLDS.get(self.market, PATTERN_THRESHOLDS['stocks'])
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect all patterns in the dataframe"""
        patterns = []
        
        # Run all detectors
        patterns.extend(self._detect_double_top(df))
        patterns.extend(self._detect_double_bottom(df))
        patterns.extend(self._detect_head_shoulders(df))
        patterns.extend(self._detect_flags(df))
        patterns.extend(self._detect_triangles(df))
        patterns.extend(self._detect_wedges(df))
        # Investment mode patterns
        patterns.extend(self._detect_triple_top(df))
        patterns.extend(self._detect_triple_bottom(df))
        patterns.extend(self._detect_cup_handle(df))
        
        # Sort by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        # If no patterns found, return NO_PATTERN
        if not patterns:
            patterns.append(PatternDetection(
                pattern=PatternType.NO_PATTERN,
                confidence=50.0,
                direction="NEUTRAL",
                target_pct=0,
                stop_pct=0,
                formation_start=0,
                formation_end=len(df) - 1
            ))
        
        return patterns
    
    def _find_swing_points(self, df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """Find swing highs and lows"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Swing high
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                highs.append((i, df['High'].iloc[i]))
            
            # Swing low
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                lows.append((i, df['Low'].iloc[i]))
        
        return highs, lows
    
    def _detect_double_top(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Double Top (M) pattern - BEARISH reversal"""
        patterns = []
        highs, lows = self._find_swing_points(df)
        
        if len(highs) < 2 or len(lows) < 1:
            return patterns
        
        # Get market-specific thresholds for double_top
        dt_thresholds = self.pattern_thresholds.get('double_top', {})
        similarity_pct = dt_thresholds.get('similarity_pct', 0.02) * self.vol_mult
        depth_pct = dt_thresholds.get('depth_pct', 0.03) * self.vol_mult
        min_distance = dt_thresholds.get('min_distance', 10)
        min_confidence = dt_thresholds.get('min_confidence', 50)
        
        # Look for two similar highs with a valley between
        for i in range(len(highs) - 1):
            h1_idx, h1_price = highs[i]
            h2_idx, h2_price = highs[i + 1]
            
            # Check minimum distance between peaks
            if h2_idx - h1_idx < min_distance:
                continue
            
            # Check if highs are similar (using market-specific threshold)
            if abs(h1_price - h2_price) / h1_price > similarity_pct:
                continue
            
            # Find valley between
            valley_lows = [l for l in lows if h1_idx < l[0] < h2_idx]
            if not valley_lows:
                continue
            
            valley_idx, valley_price = min(valley_lows, key=lambda x: x[1])
            
            # Check valley is significant (using market-specific threshold)
            peak_avg = (h1_price + h2_price) / 2
            if (peak_avg - valley_price) / peak_avg < depth_pct:
                continue
            
            # Calculate confidence
            similarity = 1 - abs(h1_price - h2_price) / h1_price
            depth = (peak_avg - valley_price) / peak_avg
            confidence = min(100, (similarity * 50 + depth * 100) * 100)
            
            # Apply minimum confidence filter (stricter for ETFs)
            if confidence < min_confidence:
                continue
            
            patterns.append(PatternDetection(
                pattern=PatternType.DOUBLE_TOP,
                confidence=confidence,
                direction="BEARISH",
                target_pct=-(peak_avg - valley_price) / peak_avg * 100,
                stop_pct=(h2_price - peak_avg) / peak_avg * 100 + 1,
                formation_start=h1_idx,
                formation_end=h2_idx,
                neckline=valley_price
            ))
        
        return patterns
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Double Bottom (W) pattern - BULLISH reversal (accumulation)"""
        patterns = []
        highs, lows = self._find_swing_points(df)
        
        if len(lows) < 2 or len(highs) < 1:
            return patterns
        
        # Get market-specific thresholds for double_bottom
        # For ETFs, these are LOOSER to detect more accumulation opportunities
        db_thresholds = self.pattern_thresholds.get('double_bottom', {})
        similarity_pct = db_thresholds.get('similarity_pct', 0.02) * self.vol_mult
        depth_pct = db_thresholds.get('depth_pct', 0.03) * self.vol_mult
        min_distance = db_thresholds.get('min_distance', 8)
        min_confidence = db_thresholds.get('min_confidence', 50)
        
        for i in range(len(lows) - 1):
            l1_idx, l1_price = lows[i]
            l2_idx, l2_price = lows[i + 1]
            
            # Check minimum distance between lows
            if l2_idx - l1_idx < min_distance:
                continue
            
            # Check if lows are similar (using market-specific threshold)
            if abs(l1_price - l2_price) / l1_price > similarity_pct:
                continue
            
            # Find peak between
            peak_highs = [h for h in highs if l1_idx < h[0] < l2_idx]
            if not peak_highs:
                continue
            
            peak_idx, peak_price = max(peak_highs, key=lambda x: x[1])
            
            # Check peak is significant (using market-specific threshold - LOOSER for ETFs)
            low_avg = (l1_price + l2_price) / 2
            if (peak_price - low_avg) / low_avg < depth_pct:
                continue
            
            similarity = 1 - abs(l1_price - l2_price) / l1_price
            depth = (peak_price - low_avg) / low_avg
            confidence = min(100, (similarity * 50 + depth * 100) * 100)
            
            # Apply minimum confidence filter
            if confidence < min_confidence:
                continue
            
            patterns.append(PatternDetection(
                pattern=PatternType.DOUBLE_BOTTOM,
                confidence=confidence,
                direction="BULLISH",
                target_pct=(peak_price - low_avg) / low_avg * 100,
                stop_pct=-(low_avg - l2_price) / low_avg * 100 - 1,
                formation_start=l1_idx,
                formation_end=l2_idx,
                neckline=peak_price
            ))
        
        return patterns
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Head and Shoulders patterns"""
        patterns = []
        highs, lows = self._find_swing_points(df)
        
        if len(highs) < 3:
            return patterns
        
        # Look for pattern: shoulder - head - shoulder
        for i in range(len(highs) - 2):
            left_idx, left_price = highs[i]
            head_idx, head_price = highs[i + 1]
            right_idx, right_price = highs[i + 2]
            
            # Head must be highest
            if not (head_price > left_price and head_price > right_price):
                continue
            
            # Shoulders should be similar
            if abs(left_price - right_price) / left_price > 0.03 * self.vol_mult:
                continue
            
            # Head should be significantly higher
            shoulder_avg = (left_price + right_price) / 2
            if (head_price - shoulder_avg) / shoulder_avg < 0.02 * self.vol_mult:
                continue
            
            # Find neckline (lows between shoulders)
            neck_lows = [l for l in lows if left_idx < l[0] < right_idx]
            if len(neck_lows) < 2:
                continue
            
            neckline = sum(l[1] for l in neck_lows) / len(neck_lows)
            
            confidence = min(100, 70 + (head_price - shoulder_avg) / shoulder_avg * 100)
            
            patterns.append(PatternDetection(
                pattern=PatternType.HEAD_SHOULDERS,
                confidence=confidence,
                direction="BEARISH",
                target_pct=-(head_price - neckline) / head_price * 100,
                stop_pct=(head_price - shoulder_avg) / shoulder_avg * 100 + 1,
                formation_start=left_idx,
                formation_end=right_idx,
                neckline=neckline
            ))
        
        # Inverse H&S (check lows instead)
        if len(lows) >= 3:
            for i in range(len(lows) - 2):
                left_idx, left_price = lows[i]
                head_idx, head_price = lows[i + 1]
                right_idx, right_price = lows[i + 2]
                
                if not (head_price < left_price and head_price < right_price):
                    continue
                
                if abs(left_price - right_price) / left_price > 0.03 * self.vol_mult:
                    continue
                
                shoulder_avg = (left_price + right_price) / 2
                if (shoulder_avg - head_price) / shoulder_avg < 0.02 * self.vol_mult:
                    continue
                
                neck_highs = [h for h in highs if left_idx < h[0] < right_idx]
                if len(neck_highs) < 2:
                    continue
                
                neckline = sum(h[1] for h in neck_highs) / len(neck_highs)
                
                confidence = min(100, 70 + (shoulder_avg - head_price) / shoulder_avg * 100)
                
                patterns.append(PatternDetection(
                    pattern=PatternType.INV_HEAD_SHOULDERS,
                    confidence=confidence,
                    direction="BULLISH",
                    target_pct=(neckline - head_price) / head_price * 100,
                    stop_pct=-(shoulder_avg - head_price) / shoulder_avg * 100 - 1,
                    formation_start=left_idx,
                    formation_end=right_idx,
                    neckline=neckline
                ))
        
        return patterns
    
    def _detect_flags(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Bull and Bear flags"""
        patterns = []
        
        if len(df) < 20:
            return patterns
        
        # Split into pole and flag portions
        pole_len = len(df) // 3
        flag_start = pole_len
        
        pole = df.iloc[:pole_len]
        flag = df.iloc[flag_start:]
        
        # Calculate pole move
        pole_move = (pole['Close'].iloc[-1] - pole['Close'].iloc[0]) / pole['Close'].iloc[0]
        
        # Pole must be significant (>3%)
        if abs(pole_move) < 0.03 * self.vol_mult:
            return patterns
        
        # Flag should be consolidating (lower volatility)
        pole_range = (pole['High'].max() - pole['Low'].min()) / pole['Close'].mean()
        flag_range = (flag['High'].max() - flag['Low'].min()) / flag['Close'].mean()
        
        if flag_range > pole_range * 0.7:
            return patterns  # Not consolidating
        
        # Flag should be counter-trend or flat
        flag_move = (flag['Close'].iloc[-1] - flag['Close'].iloc[0]) / flag['Close'].iloc[0]
        
        if pole_move > 0:  # Bull flag
            if flag_move > pole_move * 0.5:  # Flag shouldn't retrace too much upward
                return patterns
            
            confidence = min(100, 60 + abs(pole_move) * 200)
            patterns.append(PatternDetection(
                pattern=PatternType.BULL_FLAG,
                confidence=confidence,
                direction="BULLISH",
                target_pct=pole_move * 100,  # Measured move
                stop_pct=-flag_range * 100 - 1,
                formation_start=0,
                formation_end=len(df) - 1
            ))
        else:  # Bear flag
            if flag_move < pole_move * 0.5:
                return patterns
            
            confidence = min(100, 60 + abs(pole_move) * 200)
            patterns.append(PatternDetection(
                pattern=PatternType.BEAR_FLAG,
                confidence=confidence,
                direction="BEARISH",
                target_pct=pole_move * 100,
                stop_pct=flag_range * 100 + 1,
                formation_start=0,
                formation_end=len(df) - 1
            ))
        
        return patterns
    
    def _detect_triangles(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect triangle patterns"""
        patterns = []
        highs, lows = self._find_swing_points(df, window=3)
        
        if len(highs) < 3 or len(lows) < 3:
            return patterns
        
        # Get trend of highs and lows
        high_prices = [h[1] for h in highs[-4:]]
        low_prices = [l[1] for l in lows[-4:]]
        
        if len(high_prices) < 2 or len(low_prices) < 2:
            return patterns
        
        high_trend = (high_prices[-1] - high_prices[0]) / high_prices[0]
        low_trend = (low_prices[-1] - low_prices[0]) / low_prices[0]
        
        # Converging = triangle
        if high_trend < -0.01 and low_trend > 0.01:
            # Symmetric triangle
            patterns.append(PatternDetection(
                pattern=PatternType.SYM_TRIANGLE,
                confidence=60,
                direction="NEUTRAL",
                target_pct=abs(high_prices[0] - low_prices[0]) / df['Close'].iloc[-1] * 50,
                stop_pct=2,
                formation_start=highs[0][0],
                formation_end=len(df) - 1
            ))
        elif high_trend < -0.01 and abs(low_trend) < 0.01:
            # Descending triangle (bearish)
            patterns.append(PatternDetection(
                pattern=PatternType.DESC_TRIANGLE,
                confidence=65,
                direction="BEARISH",
                target_pct=-(high_prices[0] - low_prices[-1]) / df['Close'].iloc[-1] * 100,
                stop_pct=abs(high_trend) * 100 + 1,
                formation_start=highs[0][0],
                formation_end=len(df) - 1
            ))
        elif abs(high_trend) < 0.01 and low_trend > 0.01:
            # Ascending triangle (bullish)
            patterns.append(PatternDetection(
                pattern=PatternType.ASC_TRIANGLE,
                confidence=65,
                direction="BULLISH",
                target_pct=(high_prices[-1] - low_prices[0]) / df['Close'].iloc[-1] * 100,
                stop_pct=-low_trend * 100 - 1,
                formation_start=lows[0][0],
                formation_end=len(df) - 1
            ))
        
        return patterns
    
    def _detect_wedges(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect wedge patterns"""
        patterns = []
        highs, lows = self._find_swing_points(df, window=3)
        
        if len(highs) < 3 or len(lows) < 3:
            return patterns
        
        high_prices = [h[1] for h in highs[-4:]]
        low_prices = [l[1] for l in lows[-4:]]
        
        if len(high_prices) < 2 or len(low_prices) < 2:
            return patterns
        
        high_trend = (high_prices[-1] - high_prices[0]) / high_prices[0]
        low_trend = (low_prices[-1] - low_prices[0]) / low_prices[0]
        
        # Both trending same direction but converging
        if high_trend > 0.02 and low_trend > 0.02 and high_trend > low_trend:
            # Rising wedge (bearish)
            patterns.append(PatternDetection(
                pattern=PatternType.RISING_WEDGE,
                confidence=60,
                direction="BEARISH",
                target_pct=-(high_prices[-1] - low_prices[0]) / df['Close'].iloc[-1] * 100,
                stop_pct=high_trend * 100 + 1,
                formation_start=min(highs[0][0], lows[0][0]),
                formation_end=len(df) - 1
            ))
        elif high_trend < -0.02 and low_trend < -0.02 and low_trend < high_trend:
            # Falling wedge (bullish)
            patterns.append(PatternDetection(
                pattern=PatternType.FALLING_WEDGE,
                confidence=60,
                direction="BULLISH",
                target_pct=(high_prices[0] - low_prices[-1]) / df['Close'].iloc[-1] * 100,
                stop_pct=abs(low_trend) * 100 + 1,
                formation_start=min(highs[0][0], lows[0][0]),
                formation_end=len(df) - 1
            ))
        
        return patterns
    
    def _detect_triple_top(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Triple Top patterns (bearish reversal)"""
        patterns = []
        highs, lows = self._find_swing_points(df)
        
        if len(highs) < 3:
            return patterns
        
        # Look for 3 peaks at similar levels
        for i in range(len(highs) - 2):
            h1_idx, h1_price = highs[i]
            h2_idx, h2_price = highs[i + 1]
            h3_idx, h3_price = highs[i + 2]
            
            # All three highs should be within 2.5% of each other
            avg_high = (h1_price + h2_price + h3_price) / 3
            tolerance = 0.025 * self.vol_mult
            
            if (abs(h1_price - avg_high) / avg_high > tolerance or
                abs(h2_price - avg_high) / avg_high > tolerance or
                abs(h3_price - avg_high) / avg_high > tolerance):
                continue
            
            # Find support level (lows between peaks)
            support_lows = [l for l in lows if h1_idx < l[0] < h3_idx]
            if len(support_lows) < 2:
                continue
            
            support_level = sum(l[1] for l in support_lows) / len(support_lows)
            
            # Pattern height for target calculation
            pattern_height = avg_high - support_level
            
            # Confidence based on how well the peaks align
            alignment_score = 1 - (max(abs(h1_price - avg_high), abs(h2_price - avg_high), abs(h3_price - avg_high)) / avg_high) / tolerance
            confidence = min(100, 60 + alignment_score * 30)
            
            patterns.append(PatternDetection(
                pattern=PatternType.TRIPLE_TOP,
                confidence=confidence,
                direction="BEARISH",
                target_pct=-(pattern_height / avg_high) * 100,
                stop_pct=(pattern_height / avg_high) * 50,
                formation_start=h1_idx,
                formation_end=h3_idx,
                neckline=support_level
            ))
        
        return patterns
    
    def _detect_triple_bottom(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Triple Bottom patterns (bullish reversal)"""
        patterns = []
        highs, lows = self._find_swing_points(df)
        
        if len(lows) < 3:
            return patterns
        
        # Look for 3 troughs at similar levels
        for i in range(len(lows) - 2):
            l1_idx, l1_price = lows[i]
            l2_idx, l2_price = lows[i + 1]
            l3_idx, l3_price = lows[i + 2]
            
            # All three lows should be within 2.5% of each other
            avg_low = (l1_price + l2_price + l3_price) / 3
            tolerance = 0.025 * self.vol_mult
            
            if (abs(l1_price - avg_low) / avg_low > tolerance or
                abs(l2_price - avg_low) / avg_low > tolerance or
                abs(l3_price - avg_low) / avg_low > tolerance):
                continue
            
            # Find resistance level (highs between troughs)
            resistance_highs = [h for h in highs if l1_idx < h[0] < l3_idx]
            if len(resistance_highs) < 2:
                continue
            
            resistance_level = sum(h[1] for h in resistance_highs) / len(resistance_highs)
            
            # Pattern height for target calculation
            pattern_height = resistance_level - avg_low
            
            # Confidence based on how well the troughs align
            alignment_score = 1 - (max(abs(l1_price - avg_low), abs(l2_price - avg_low), abs(l3_price - avg_low)) / avg_low) / tolerance
            confidence = min(100, 60 + alignment_score * 30)
            
            patterns.append(PatternDetection(
                pattern=PatternType.TRIPLE_BOTTOM,
                confidence=confidence,
                direction="BULLISH",
                target_pct=(pattern_height / avg_low) * 100,
                stop_pct=(pattern_height / avg_low) * 50,
                formation_start=l1_idx,
                formation_end=l3_idx,
                neckline=resistance_level
            ))
        
        return patterns
    
    def _detect_cup_handle(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect Cup and Handle patterns (bullish continuation)"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        highs, lows = self._find_swing_points(df, window=5)
        
        if len(highs) < 2 or len(lows) < 1:
            return patterns
        
        # Look for cup pattern: high -> low -> high at similar level
        for i in range(len(highs) - 1):
            left_rim_idx, left_rim_price = highs[i]
            right_rim_idx, right_rim_price = highs[i + 1]
            
            # Find the lowest point between the two rims (cup bottom)
            cup_lows = [l for l in lows if left_rim_idx < l[0] < right_rim_idx]
            if not cup_lows:
                continue
            
            cup_bottom_idx, cup_bottom_price = min(cup_lows, key=lambda x: x[1])
            
            # Rims should be at similar levels (within 3%)
            rim_avg = (left_rim_price + right_rim_price) / 2
            if abs(left_rim_price - right_rim_price) / rim_avg > 0.03 * self.vol_mult:
                continue
            
            # Cup should have decent depth (at least 10% below rim)
            cup_depth = (rim_avg - cup_bottom_price) / rim_avg
            if cup_depth < 0.10 * self.vol_mult or cup_depth > 0.50:
                continue
            
            # Cup should be U-shaped (bottom in middle third)
            cup_width = right_rim_idx - left_rim_idx
            if cup_width < 5:
                continue
            bottom_position = (cup_bottom_idx - left_rim_idx) / cup_width
            if bottom_position < 0.3 or bottom_position > 0.7:
                continue
            
            # Look for handle (small pullback after right rim)
            handle_end = min(right_rim_idx + int(cup_width * 0.3), len(df))
            handle_data = df.iloc[right_rim_idx:handle_end]
            if len(handle_data) < 3:
                # No handle yet, but cup is valid
                confidence = 55 + cup_depth * 50
            else:
                # Check for handle (should not go below 50% of cup depth)
                handle_low = handle_data['Low'].min()
                handle_depth = (right_rim_price - handle_low) / right_rim_price
                
                if handle_depth > cup_depth * 0.5:
                    continue  # Handle too deep
                
                confidence = 65 + cup_depth * 40 + (1 - handle_depth / cup_depth) * 10
            
            confidence = min(100, confidence)
            
            patterns.append(PatternDetection(
                pattern=PatternType.CUP_HANDLE,
                confidence=confidence,
                direction="BULLISH",
                target_pct=cup_depth * 100,  # Target = cup depth above rim
                stop_pct=cup_depth * 50,
                formation_start=left_rim_idx,
                formation_end=right_rim_idx,
                neckline=rim_avg
            ))
        
        return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING MODELS
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    
    class PatternCNN(nn.Module):
        """
        CNN for pattern detection from candlestick images.
        Input: (batch, 1, 64, 64) grayscale image
        Output: (batch, num_patterns) class logits
        """
        
        def __init__(self, num_classes: int = len(PATTERN_LABELS)):
            super().__init__()
            
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            
            # After 4 pools: 64 -> 32 -> 16 -> 8 -> 4
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            
            x = x.view(-1, 256 * 4 * 4)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.fc3(x)
            
            return x
    
    
    class PatternLSTM(nn.Module):
        """
        LSTM for pattern detection from price sequences.
        Input: (batch, seq_len, 5) for OHLCV
        Output: (batch, num_patterns) class logits
        """
        
        def __init__(
            self,
            input_size: int = 5,
            hidden_size: int = 128,
            num_layers: int = 2,
            num_classes: int = len(PATTERN_LABELS),
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            
            self.fc1 = nn.Linear(hidden_size * 2, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            # LSTM
            lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
            
            # Attention
            attn_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)
            
            # Classifier
            out = self.dropout(F.relu(self.fc1(context)))
            out = self.dropout(F.relu(self.fc2(out)))
            out = self.fc3(out)
            
            return out
    
    
    class HybridPatternModel(nn.Module):
        """
        Combines CNN (image) + LSTM (sequence) for robust pattern detection.
        """
        
        def __init__(self, num_classes: int = len(PATTERN_LABELS)):
            super().__init__()
            
            self.cnn = PatternCNN(num_classes=256)  # Feature extractor
            self.lstm = PatternLSTM(num_classes=256)  # Feature extractor
            
            # Modify last layers to output features instead of classes
            self.cnn.fc3 = nn.Linear(256, 256)
            self.lstm.fc3 = nn.Linear(128, 256)
            
            # Fusion layers
            self.fusion = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, image, sequence):
            cnn_features = self.cnn(image)
            lstm_features = self.lstm(sequence)
            
            combined = torch.cat([cnn_features, lstm_features], dim=1)
            output = self.fusion(combined)
            
            return output


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    
    class PatternDataset(Dataset):
        """Dataset for pattern detection training"""
        
        def __init__(
            self,
            dataframes: List[pd.DataFrame],
            labels: List[int],
            seq_length: int = 100,
            img_size: int = 64,
            augment: bool = True
        ):
            self.dataframes = dataframes
            self.labels = labels
            self.seq_length = seq_length
            self.img_size = img_size
            self.augment = augment
        
        def __len__(self):
            return len(self.dataframes)
        
        def __getitem__(self, idx):
            df = self.dataframes[idx].copy()
            label = self.labels[idx]
            
            # Augmentation
            if self.augment and np.random.random() > 0.5:
                # Random time shift
                shift = np.random.randint(-10, 10)
                if shift > 0 and len(df) > self.seq_length + shift:
                    df = df.iloc[shift:].reset_index(drop=True)
            
            # Create image
            image = create_pattern_image(df.tail(self.seq_length), self.img_size)
            
            # Create sequence
            sequence = normalize_sequence(df, self.seq_length)
            
            return (
                torch.tensor(image, dtype=torch.float32),
                torch.tensor(sequence, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class DeepPatternTrainer:
    """
    Trainer for deep learning pattern detection models.
    """
    
    def __init__(
        self,
        mode: str = 'daytrade',
        market: str = 'crypto',
        model_type: str = 'hybrid'  # 'cnn', 'lstm', or 'hybrid'
    ):
        self.mode = mode
        self.market = market
        self.model_type = model_type
        self.seq_length = MODE_SEQUENCE_LENGTH.get(mode, 100)
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get patterns for this mode
        self.patterns = MODE_PATTERNS.get(mode, list(PatternType))
        self.pattern_labels = [p.value for p in self.patterns]
    
    def _create_model(self, num_classes: int):
        """Create the appropriate model"""
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        if self.model_type == 'cnn':
            return PatternCNN(num_classes=num_classes).to(self.device)
        elif self.model_type == 'lstm':
            return PatternLSTM(num_classes=num_classes).to(self.device)
        else:  # hybrid
            return HybridPatternModel(num_classes=num_classes).to(self.device)
    
    def generate_training_data(
        self,
        df: pd.DataFrame,
        chunk_size: int = None,
        overlap: int = None,
        min_confidence: float = 35,  # Lower threshold for more samples
        include_no_pattern: bool = True  # Balance with NO_PATTERN samples
    ) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Generate training samples from historical data using rule-based detection.
        
        Args:
            df: Full historical dataframe
            chunk_size: Size of each sample (default: seq_length)
            overlap: Overlap between samples (default: chunk_size // 2)
            min_confidence: Minimum confidence threshold (default: 35)
            include_no_pattern: Include NO_PATTERN samples (default: True)
        
        Returns:
            List of dataframes and their labels
        """
        if chunk_size is None:
            chunk_size = self.seq_length
        if overlap is None:
            # Use 75% overlap for more samples (was 50%)
            overlap = int(chunk_size * 0.75)
        
        detector = RuleBasedPatternDetector(market=self.market)
        
        samples = []
        labels = []
        no_pattern_samples = []  # Track NO_PATTERN separately
        
        step = max(10, chunk_size - overlap)  # Minimum step of 10
        
        for i in range(0, len(df) - chunk_size, step):
            chunk = df.iloc[i:i + chunk_size].reset_index(drop=True)
            
            # Detect patterns
            patterns = detector.detect_all_patterns(chunk)
            
            if patterns:
                best_pattern = patterns[0]
                
                # Check if it's a real pattern or NO_PATTERN
                if best_pattern.pattern == PatternType.NO_PATTERN:
                    if include_no_pattern:
                        no_pattern_samples.append(chunk)
                elif best_pattern.confidence >= min_confidence:
                    # Filter to mode-specific patterns
                    if best_pattern.pattern in self.patterns:
                        samples.append(chunk)
                        labels.append(best_pattern.pattern.value)
        
        # Balance NO_PATTERN samples with real patterns
        # Add up to 50% of real pattern count as NO_PATTERN
        if include_no_pattern and no_pattern_samples and samples:
            max_no_pattern = max(len(samples) // 2, 20)  # At least 20 or 50% of patterns
            import random
            random.shuffle(no_pattern_samples)
            for chunk in no_pattern_samples[:max_no_pattern]:
                samples.append(chunk)
                labels.append(PatternType.NO_PATTERN.value)
        
        # If still too few samples, include lower confidence patterns
        if len(samples) < 50 and min_confidence > 20:
            print(f"⚠️ Only {len(samples)} samples found. Trying lower confidence threshold...")
            extra_samples, extra_labels = self.generate_training_data(
                df, chunk_size, overlap, 
                min_confidence=20, 
                include_no_pattern=False  # Avoid recursion
            )
            samples.extend(extra_samples)
            labels.extend(extra_labels)
        
        return samples, labels
    
    def train(
        self,
        samples: List[pd.DataFrame],
        labels: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        progress_callback=None
    ) -> Dict:
        """
        Train the pattern detection model.
        
        Returns:
            Training metrics
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        # Encode labels
        self.label_encoder.fit(self.pattern_labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Check class distribution - need at least 2 samples per class for stratified split
        from collections import Counter
        label_counts = Counter(encoded_labels)
        min_count = min(label_counts.values())
        
        # Filter out classes with too few samples
        if min_count < 2:
            # Find classes with enough samples
            valid_classes = [cls for cls, count in label_counts.items() if count >= 2]
            
            if len(valid_classes) < 2:
                raise ValueError(f"Not enough training data. Need at least 2 samples per pattern class. "
                               f"Try: more symbols, more days, or different timeframe. "
                               f"Current distribution: {dict(label_counts)}")
            
            # Filter samples to only include valid classes
            mask = [lbl in valid_classes for lbl in encoded_labels]
            samples = [s for s, m in zip(samples, mask) if m]
            encoded_labels = encoded_labels[mask]
            
            # Re-encode with only valid classes
            valid_labels = [self.label_encoder.inverse_transform([c])[0] for c in valid_classes]
            self.label_encoder.fit(valid_labels)
            original_labels = [labels[i] for i, m in enumerate(mask) if m]
            encoded_labels = self.label_encoder.transform(original_labels)
            
            print(f"⚠️ Filtered rare patterns. Training with {len(valid_classes)} pattern types, {len(samples)} samples")
        
        # Split data (use stratify only if we have enough samples)
        try:
            train_samples, val_samples, train_labels, val_labels = train_test_split(
                samples, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
        except ValueError:
            # Fallback: split without stratify if still failing
            train_samples, val_samples, train_labels, val_labels = train_test_split(
                samples, encoded_labels, test_size=0.2, random_state=42
            )
            print("⚠️ Using non-stratified split due to limited samples")
        
        # Create datasets
        train_dataset = PatternDataset(
            train_samples, train_labels.tolist(),
            seq_length=self.seq_length, augment=True
        )
        val_dataset = PatternDataset(
            val_samples, val_labels.tolist(),
            seq_length=self.seq_length, augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        num_classes = len(self.label_encoder.classes_)
        self.model = self._create_model(num_classes)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # COMPUTE CLASS WEIGHTS for imbalanced data
        # This helps the model learn rare patterns (like double_bottom in ETFs)
        # ═══════════════════════════════════════════════════════════════════════════
        class_counts = np.bincount(train_labels, minlength=num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        # Inverse frequency weighting: rare classes get higher weight
        class_weights = 1.0 / class_counts
        # Normalize so sum = num_classes (balanced weights)
        class_weights = class_weights / class_weights.sum() * num_classes
        # Cap maximum weight to avoid overweighting extremely rare classes
        class_weights = np.minimum(class_weights, 10.0)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"   ⚖️ Class weights computed for {num_classes} classes (handles imbalance)")
        
        # Loss and optimizer - WITH class weights!
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_acc = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            
            for images, sequences, batch_labels in train_loader:
                images = images.to(self.device)
                sequences = sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_type == 'cnn':
                    outputs = self.model(images)
                elif self.model_type == 'lstm':
                    outputs = self.model(sequences)
                else:
                    outputs = self.model(images, sequences)
                
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validate
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, sequences, batch_labels in val_loader:
                    images = images.to(self.device)
                    sequences = sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    if self.model_type == 'cnn':
                        outputs = self.model(images)
                    elif self.model_type == 'lstm':
                        outputs = self.model(sequences)
                    else:
                        outputs = self.model(images, sequences)
                    
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                improved = " ⬆️ BEST"
            else:
                improved = ""
            
            if progress_callback:
                progress_callback(
                    (epoch + 1) / epochs,
                    f"Epoch {epoch+1}/{epochs} | Loss: {val_loss:.4f} | Acc: {val_acc:.1%}{improved}"
                )
        
        return {
            'mode': self.mode,
            'market': self.market,
            'model_type': self.model_type,
            'n_samples': len(samples),
            'n_classes': num_classes,
            'classes': self.label_encoder.classes_.tolist(),
            'best_val_acc': best_val_acc,
            'final_val_acc': history['val_acc'][-1],
            'history': history
        }
    
    def predict(self, df: pd.DataFrame) -> PatternDetection:
        """
        Predict pattern in the given dataframe.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Prepare data
        image = create_pattern_image(df.tail(self.seq_length), 64)
        sequence = normalize_sequence(df, self.seq_length)
        
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'cnn':
                outputs = self.model(image_tensor)
            elif self.model_type == 'lstm':
                outputs = self.model(seq_tensor)
            else:
                outputs = self.model(image_tensor, seq_tensor)
            
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        pattern_name = self.label_encoder.inverse_transform([predicted.item()])[0]
        pattern_type = PatternType(pattern_name)
        
        return PatternDetection(
            pattern=pattern_type,
            confidence=confidence.item() * 100,
            direction=PATTERN_DIRECTION.get(pattern_type, "NEUTRAL"),
            target_pct=0,  # Would need rule-based for targets
            stop_pct=0,
            formation_start=0,
            formation_end=len(df) - 1
        )
    
    def save(self, filepath: str, accuracy: float = None):
        """Save model and metadata including accuracy"""
        if self.model is None:
            raise ValueError("No model to save")
        
        bundle = {
            'model_state': self.model.state_dict(),
            'label_encoder': self.label_encoder,
            'mode': self.mode,
            'market': self.market,
            'model_type': self.model_type,
            'seq_length': self.seq_length,
            'pattern_labels': self.pattern_labels,
            'accuracy': accuracy or getattr(self, 'best_val_acc', 0),
            'best_val_acc': accuracy or getattr(self, 'best_val_acc', 0),
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
    
    def load(self, filepath: str):
        """Load model and metadata"""
        with open(filepath, 'rb') as f:
            bundle = pickle.load(f)
        
        self.mode = bundle['mode']
        self.market = bundle['market']
        self.model_type = bundle['model_type']
        self.seq_length = bundle['seq_length']
        self.pattern_labels = bundle['pattern_labels']
        self.label_encoder = bundle['label_encoder']
        
        num_classes = len(self.label_encoder.classes_)
        self.model = self._create_model(num_classes)
        self.model.load_state_dict(bundle['model_state'])
        self.model.eval()


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class CombinedPredictor:
    """
    Combines Traditional ML + Pattern Detection for final signal.
    """
    
    def __init__(
        self,
        mode: str = 'daytrade',
        market: str = 'crypto'
    ):
        self.mode = mode
        self.market = market
        
        self.traditional_ml = None
        self.pattern_detector = None
        self.rule_detector = RuleBasedPatternDetector(market=market)
        
        # Weight for combining signals
        self.trad_weight = 0.5
        self.pattern_weight = 0.5
    
    def load_traditional_ml(self, model_path: str):
        """Load traditional ML model"""
        try:
            from .probabilistic_ml import ProbabilisticMLTrainer
        except ImportError:
            from probabilistic_ml import ProbabilisticMLTrainer
        
        self.traditional_ml = ProbabilisticMLTrainer()
        self.traditional_ml.load_models(self.mode)
    
    def load_pattern_detector(self, model_path: str):
        """Load deep learning pattern detector"""
        self.pattern_detector = DeepPatternTrainer(
            mode=self.mode,
            market=self.market
        )
        self.pattern_detector.load(model_path)
    
    def predict(
        self,
        df: pd.DataFrame,
        whale_data: dict = None,
        smc_data: dict = None
    ) -> CombinedSignal:
        """
        Generate combined prediction from both systems.
        """
        # Get traditional ML prediction
        trad_direction = "WAIT"
        trad_confidence = 50.0
        trad_probs = {}
        
        if self.traditional_ml is not None:
            try:
                trad_result = self.traditional_ml.predict(df, whale_data, smc_data)
                trad_direction = trad_result.get('direction', 'WAIT')
                trad_confidence = trad_result.get('confidence', 50)
                trad_probs = trad_result.get('probabilities', {})
            except:
                pass
        
        # Get pattern detection
        if self.pattern_detector is not None and self.pattern_detector.model is not None:
            pattern_result = self.pattern_detector.predict(df)
        else:
            # Fallback to rule-based
            patterns = self.rule_detector.detect_all_patterns(df)
            pattern_result = patterns[0] if patterns else PatternDetection(
                pattern=PatternType.NO_PATTERN,
                confidence=50,
                direction="NEUTRAL",
                target_pct=0,
                stop_pct=0,
                formation_start=0,
                formation_end=len(df) - 1
            )
        
        # Combine signals
        final_direction, final_confidence, agreement = self._combine_signals(
            trad_direction, trad_confidence,
            pattern_result.direction, pattern_result.confidence
        )
        
        # Calculate suggested targets
        if pattern_result.target_pct != 0:
            tp_pct = abs(pattern_result.target_pct)
            sl_pct = abs(pattern_result.stop_pct)
        else:
            # Default based on mode
            mode_targets = {
                'scalp': (0.5, 0.3),
                'daytrade': (1.5, 0.8),
                'swing': (5.0, 2.5),
                'investment': (15.0, 7.0)
            }
            tp_pct, sl_pct = mode_targets.get(self.mode, (2.0, 1.0))
        
        return CombinedSignal(
            trad_direction=trad_direction,
            trad_confidence=trad_confidence,
            trad_probabilities=trad_probs,
            pattern=pattern_result.pattern,
            pattern_confidence=pattern_result.confidence,
            pattern_direction=pattern_result.direction,
            final_direction=final_direction,
            final_confidence=final_confidence,
            agreement=agreement,
            suggested_tp_pct=tp_pct,
            suggested_sl_pct=sl_pct
        )
    
    def _combine_signals(
        self,
        trad_dir: str,
        trad_conf: float,
        pattern_dir: str,
        pattern_conf: float
    ) -> Tuple[str, float, str]:
        """
        Combine traditional ML and pattern signals.
        
        Returns: (final_direction, final_confidence, agreement_level)
        """
        # Normalize directions
        trad_bullish = trad_dir == "LONG"
        trad_bearish = trad_dir == "SHORT"
        
        pattern_bullish = pattern_dir == "BULLISH"
        pattern_bearish = pattern_dir == "BEARISH"
        pattern_neutral = pattern_dir == "NEUTRAL"
        
        # Agreement levels
        if (trad_bullish and pattern_bullish) or (trad_bearish and pattern_bearish):
            agreement = "STRONG_AGREE"
            # Boost confidence
            final_conf = min(100, (trad_conf * self.trad_weight + pattern_conf * self.pattern_weight) * 1.2)
            final_dir = "LONG" if trad_bullish else "SHORT"
            
        elif pattern_neutral:
            agreement = "NEUTRAL"
            final_conf = trad_conf * 0.9  # Slight reduction without pattern confirmation
            final_dir = trad_dir
            
        elif trad_dir == "WAIT":
            agreement = "PATTERN_ONLY"
            final_conf = pattern_conf * 0.8  # Pattern alone less reliable
            final_dir = "LONG" if pattern_bullish else "SHORT" if pattern_bearish else "WAIT"
            
        elif (trad_bullish and pattern_bearish) or (trad_bearish and pattern_bullish):
            agreement = "CONFLICT"
            # Use stronger signal
            if trad_conf > pattern_conf:
                final_dir = trad_dir
                final_conf = trad_conf * 0.6  # Reduced confidence due to conflict
            else:
                final_dir = "LONG" if pattern_bullish else "SHORT"
                final_conf = pattern_conf * 0.6
        else:
            agreement = "AGREE"
            final_conf = trad_conf * self.trad_weight + pattern_conf * self.pattern_weight
            final_dir = trad_dir if trad_dir != "WAIT" else ("LONG" if pattern_bullish else "SHORT" if pattern_bearish else "WAIT")
        
        return final_dir, final_conf, agreement


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("DEEP LEARNING PATTERN DETECTION SYSTEM")
    print("=" * 60)
    
    print(f"\nPyTorch Available: {HAS_TORCH}")
    
    if HAS_TORCH:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        # Test models
        print("\nTesting model architectures...")
        
        cnn = PatternCNN(num_classes=18)
        lstm = PatternLSTM(num_classes=18)
        hybrid = HybridPatternModel(num_classes=18)
        
        # Test forward pass
        test_img = torch.randn(2, 1, 64, 64)
        test_seq = torch.randn(2, 100, 5)
        
        print(f"CNN output shape: {cnn(test_img).shape}")
        print(f"LSTM output shape: {lstm(test_seq).shape}")
        print(f"Hybrid output shape: {hybrid(test_img, test_seq).shape}")
    
    print("\n" + "=" * 60)
    print("PATTERNS BY MODE")
    print("=" * 60)
    
    for mode, patterns in MODE_PATTERNS.items():
        print(f"\n{mode.upper()}:")
        for p in patterns:
            dir = PATTERN_DIRECTION.get(p, "?")
            print(f"  {p.value}: {dir}")