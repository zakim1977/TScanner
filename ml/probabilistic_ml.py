"""
Probabilistic ML System
========================

WHAT THE ML TRAINS ON (INPUT FEATURES):
- Price & Structure: Returns, Range, Position, BoS, Compression, ATR
- Positioning & Flow: OI, Funding, Long/Short ratio, Whale/Retail
- SMC / Context: Liquidity sweep, OB proximity, FVG, Accumulation, Sessions

WHAT THE ML PREDICTS (LABELS):
- Day Trade: P_continuation, P_fakeout, P_vol_expansion
- Swing: P_trend_holds, P_reversal, P_drawdown
- Investment: P_accumulation, P_distribution, P_large_drawdown

WHAT THE ML OUTPUTS:
- Multiple probability values per scenario
- Rules layer interprets probabilities to make LONG/SHORT/WAIT decision

HOW YOU VERIFY:
- Performance by probability bucket
- Return vs probability correlation
- Regime stability analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# SMOTE for handling class imbalance (improves F1 for rare labels)
try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

ENHANCED_FEATURES = [
    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE & STRUCTURE (15 features)
    # ═══════════════════════════════════════════════════════════════════════════
    'return_1bar',              # Return over 1 bar (%)
    'return_3bar',              # Return over 3 bars (%)
    'return_5bar',              # Return over 5 bars (%)
    'return_10bar',             # Return over 10 bars (%)
    'high_low_range',           # (High - Low) / Close (%)
    'close_position_in_range',  # (Close - Low) / (High - Low) [0-1]
    'distance_from_high',       # (High - Close) / Close (%)
    'distance_from_low',        # (Close - Low) / Close (%)
    'break_of_structure',       # 1 if BoS detected, 0 otherwise
    'structure_direction',      # -1=bearish BoS, 0=none, 1=bullish BoS
    'bb_squeeze_pct',           # BB compression percentile [0-100]
    'bb_width_pct',             # BB width as % of price
    'compression_candles',      # How many candles in compression
    'atr_pct',                  # ATR as % of price
    'atr_change_ratio',         # Current ATR / ATR 10 bars ago
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EMA TREND FEATURES (5 features) - CRITICAL FOR TREND DETECTION!
    # ═══════════════════════════════════════════════════════════════════════════
    'price_vs_ema20',           # (Close - EMA20) / Close * 100 (% distance)
    'price_vs_ema50',           # (Close - EMA50) / Close * 100 (% distance)
    'ema20_vs_ema50',           # (EMA20 - EMA50) / EMA50 * 100 (EMA alignment)
    'ema_trend_score',          # -2=strong bear, -1=bear, 0=neutral, 1=bull, 2=strong bull
    'falling_knife',            # 1 if price < EMA20 < EMA50 AND RSI < 35 (DANGER!)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSITIONING & FLOW (12 features)
    # ═══════════════════════════════════════════════════════════════════════════
    'oi_change_1h',             # OI change last 1 hour (%)
    'oi_change_24h',            # OI change last 24 hours (%)
    'price_vs_oi_divergence',   # Price change - OI change (divergence signal)
    'funding_rate',             # Current funding rate
    'funding_rate_delta',       # Funding rate change (momentum)
    'long_short_ratio',         # Long accounts / Short accounts
    'whale_long_pct',           # Whale long positioning (%)
    'retail_long_pct',          # Retail long positioning (%)
    'whale_retail_divergence',  # Whale% - Retail% (key signal!)
    'whale_dominance',          # (Whale% - 50) / 50 normalized
    'institutional_buy_imbalance',  # Inst buy volume - sell volume
    'smart_money_flow',         # CMF or similar
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SMC / CONTEXT (12 features)
    # ═══════════════════════════════════════════════════════════════════════════
    'liquidity_sweep_bull',     # 1 if bullish liquidity sweep detected
    'liquidity_sweep_bear',     # 1 if bearish liquidity sweep detected
    'at_bullish_ob',            # 1 if at bullish order block
    'at_bearish_ob',            # 1 if at bearish order block
    'near_bullish_ob',          # 1 if near (within 1 ATR) bullish OB
    'near_bearish_ob',          # 1 if near bearish OB
    'in_fvg_bullish',           # 1 if in bullish FVG
    'in_fvg_bearish',           # 1 if in bearish FVG
    'accumulation_score',       # 0-100 accumulation strength
    'distribution_score',       # 0-100 distribution strength
    'session_encoded',          # -1=Asian, 0=London, 1=NY
    'session_overlap',          # 1 if in overlap session (high volume)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MARKET CONTEXT (6 features)
    # ═══════════════════════════════════════════════════════════════════════════
    'btc_correlation',          # Correlation with BTC
    'btc_trend',                # -1=bearish, 0=neutral, 1=bullish
    'htf_trend_alignment',      # -1=against, 0=neutral, 1=aligned with LTF
    'fear_greed_index',         # Market sentiment 0-100
    'volatility_regime',        # -1=low vol, 0=normal, 1=high vol
    'momentum_regime',          # -1=bearish momentum, 0=neutral, 1=bullish
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEMPORAL FEATURES (6 features) - CRITICAL FOR STOCKS!
    # ═══════════════════════════════════════════════════════════════════════════
    'day_of_week',              # 0=Mon, 4=Fri (weekly patterns)
    'month',                    # 1-12 (monthly/seasonal patterns)
    'is_month_end',             # 1 if last 3 trading days (window dressing)
    'is_quarter_end',           # 1 if last week of quarter (rebalancing)
    'is_first_week',            # 1 if first week of month (inflows)
    'days_in_trend',            # How long current trend has lasted
]

NUM_ENHANCED_FEATURES = len(ENHANCED_FEATURES)  # 56 features (was 51, added 5 EMA trend features)


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK-SPECIFIC FEATURES (Different from crypto - capture what moves stocks)
# ═══════════════════════════════════════════════════════════════════════════════
# Key insight: Crypto features (whale %, OI, funding) don't exist for stocks
# Stocks need: mean reversion signals, volume profile, market correlation

STOCK_FEATURES = [
    # Mean Reversion (Stocks revert to mean, unlike crypto)
    'distance_from_vwap_pct',       # How far from VWAP
    'distance_from_sma20_pct',      # Distance from 20 SMA
    'distance_from_sma50_pct',      # Distance from 50 SMA
    'rsi_14',                        # Classic RSI
    'rsi_deviation',                 # RSI - 50 (centered)
    'bb_position',                   # Position within BB [0-1]
    'bb_squeeze_score',              # How tight are bands (0-100)
    'z_score_20',                    # (Price - SMA20) / StdDev20
    'z_score_50',                    # (Price - SMA50) / StdDev50
    'mean_reversion_score',          # Combined mean-reversion signal
    
    # Momentum & Trend Quality
    'adx_14',                        # Trend strength (not direction)
    'trend_efficiency',              # |net move| / sum(|moves|)
    'consecutive_direction_bars',    # How many bars same direction
    'macd_histogram',                # MACD momentum
    'macd_hist_slope',               # Is MACD accelerating?
    'momentum_10',                   # 10-bar momentum
    'momentum_20',                   # 20-bar momentum
    'momentum_divergence',           # Price HH but momentum LH?
    
    # Volume Profile
    'relative_volume',               # Volume / Avg Volume
    'volume_trend',                  # Is volume increasing?
    'price_volume_correlation',      # Are price and volume aligned?
    'accumulation_distribution',     # A/D line slope
    'obv_slope',                     # On Balance Volume trend
    'volume_breakout',               # 1 if volume > 2x average
    'buying_pressure',               # (Close - Low) / Range * Volume
    
    # Volatility Regime
    'atr_percentile',                # Current ATR vs historical
    'volatility_contraction',        # Is vol contracting?
    'volatility_expansion',          # Is vol expanding?
    'range_percentile',              # Today's range vs historical
    'gap_pct',                       # Gap from previous close
    'intraday_volatility',           # High-Low range / Open
    
    # Structure & Support/Resistance
    'distance_from_52w_high_pct',    # How far from 52-week high
    'distance_from_52w_low_pct',     # How far from 52-week low
    'position_in_range',             # Where in 20-bar range
    'at_support',                    # Near support level
    'at_resistance',                 # Near resistance level
    'break_of_structure',            # BoS detected
    'higher_high',                   # Making HH
    'lower_low',                     # Making LL
    
    # Institutional Signals (From Quiver - regime indicator)
    'congress_sentiment',            # -1 to 1
    'insider_sentiment',             # -1 to 1
    'short_interest_pct',            # Short interest %
    'short_interest_change',         # SI rising or falling
    'institutional_activity',        # Combined signal
    
    # Market Context (Stocks correlated to SPY)
    'spy_correlation',               # Correlation with SPY
    'spy_relative_strength',         # Stock - SPY return
    'sector_relative_strength',      # Stock - Sector return
    'market_regime',                 # Bull/Bear/Range
    'vix_level',                     # VIX (fear gauge)
    'vix_trend',                     # VIX rising or falling
    
    # Temporal (Calendar effects real in stocks)
    'day_of_week',                   # Monday/Friday effects
    'is_month_end',                  # Window dressing
    'is_quarter_end',                # Rebalancing
    'days_to_earnings',              # Catalyst proximity
]

NUM_STOCK_FEATURES = len(STOCK_FEATURES)  # 54 features


# ═══════════════════════════════════════════════════════════════════════════════
# MODE-SPECIFIC LABEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# BEST PRACTICES FOR ML TRADING:
# 1. Thresholds = minimum move to count as "success" during training
#    - Too low: Model learns noise, not signal
#    - Too high: Too few positive samples, model can't learn
#
# 2. Lookahead = prediction horizon (entry validation window)
#    - Shorter = higher accuracy (easier to predict near-term)
#    - Longer = more time to hit target, but more noise
#    - PRINCIPLE: Validate ENTRY quality, not full trade duration
#
# 3. Market-specific:
#    - Crypto: 3-8% daily moves typical, higher thresholds
#    - Stocks: 0.5-2% daily moves typical, lower thresholds
#
# ═══════════════════════════════════════════════════════════════════════════════

# CRYPTO thresholds (high volatility - BTC, ETH, altcoins)
MODE_LABELS_CRYPTO = {
    'scalp': {
        'labels': ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion'],
        'lookahead': 6,       # 6 candles = 30 min on 5m
        'thresholds': {
            'continuation': 1.0,    # 1.0% move - meaningful for scalp
            'fakeout': 0.8,         # 0.8% fake then reverse
            'vol_expansion': 1.5,   # ATR expands 1.5x
        }
    },
    'daytrade': {
        'labels': ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion'],
        'lookahead': 12,      # 12 candles = 3 hours on 15m (entry validation)
        'thresholds': {
            'continuation': 2.5,    # 2.5% move - YOUR TARGET
            'fakeout': 1.5,         # 1.5% fake then reverse
            'vol_expansion': 1.8,   # Volatility expansion
        }
    },
    'swing': {
        'labels': ['trend_holds_bull', 'trend_holds_bear', 'reversal_to_bull', 'reversal_to_bear', 'drawdown'],
        'lookahead': 12,      # 12 candles = 48 hours on 4h
        'thresholds': {
            'trend_holds': 5.0,     # 5% continuation
            'reversal': 4.0,        # 4% reversal
            'drawdown': 3.0,        # 3% adverse = risk threshold
        }
    },
    'investment': {
        'labels': ['accumulation', 'distribution', 'reversal_to_bull', 'reversal_to_bear', 'large_drawdown'],
        'lookahead': 10,      # INCREASED: 10 candles (was 7) - more time for moves
        'thresholds': {
            'accumulation': 3.0,    # LOWERED AGAIN: 3% (was 5%)
            'distribution': 3.0,    # LOWERED AGAIN: 3% (was 5%)
            'reversal': 2.5,        # LOWERED AGAIN: 2.5% (was 4%)
            'large_drawdown': 5.0,  # LOWERED AGAIN: 5% (was 8%)
        }
    }
}

# STOCK thresholds - QUALITY GATES approach
# Stocks are mean-reverting - predict OPPORTUNITY QUALITY, not direction!
# This is how institutions actually trade stocks.
MODE_LABELS_STOCK = {
    'scalp': {
        'labels': ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion'],
        'lookahead': 6,
        'thresholds': {
            'continuation': 0.8,
            'fakeout': 0.6,
            'vol_expansion': 1.4,
        }
    },
    'daytrade': {
        'labels': ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion'],
        'lookahead': 18,
        'thresholds': {
            'continuation': 1.0,
            'fakeout': 0.6,
            'vol_expansion': 1.4,
        }
    },
    'swing': {
        # ═══════════════════════════════════════════════════════════════════════
        # STOCK SWING: QUALITY + MEAN REVERSION APPROACH
        # Key insight: Stocks mean-revert, crypto trends. Different labels needed!
        # Predict QUALITY (good R:R) and REGIME (mean reversion) not just direction
        # ═══════════════════════════════════════════════════════════════════════
        'labels': [
            'setup_quality_high',     # TP hit before SL - good entry
            'setup_quality_low',      # SL hit before TP - bad entry
            'mean_reversion_long',    # Oversold bounce opportunity
            'mean_reversion_short',   # Overbought fade opportunity
            'breakout_valid',         # Breakout that holds (not fakeout)
            'volatility_expansion',   # Vol about to expand (opportunity)
        ],
        'lookahead': 15,
        'thresholds': {
            'quality_tp_pct': 2.0,       # Target profit %
            'quality_sl_pct': 1.0,       # Stop loss %
            'mean_rev_threshold': 1.3,   # LOWERED from 2.0 for better class balance
            'breakout_hold_pct': 1.5,    # Breakout must hold this %
            'vol_expansion_ratio': 1.5,  # Volatility expansion threshold
            'rsi_oversold': 35,          # RSI threshold for oversold
            'rsi_overbought': 65,        # RSI threshold for overbought
            'sma_deviation_pct': 3.0,    # % deviation from SMA to trigger
        }
    },
    'investment': {
        'labels': ['accumulation', 'distribution', 'reversal_to_bull', 'reversal_to_bear', 'large_drawdown'],
        'lookahead': 20,
        'thresholds': {
            'accumulation': 2.5,
            'distribution': 2.5,
            'reversal': 2.0,
            'large_drawdown': 4.0,
        }
    }
}

# Default to CRYPTO (backward compatibility)
MODE_LABELS = MODE_LABELS_CRYPTO


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET PHASE LABELS (Wyckoff-based)
# ═══════════════════════════════════════════════════════════════════════════════
# ML can predict which Wyckoff phase the market is in AND which phase it will
# transition to. This is more predictive than just detecting current phase.
#
# BULLISH PHASES:
#   - ACCUMULATION: Smart money buying quietly at lows
#   - RE_ACCUMULATION: Pause in uptrend, smart money reloading
#   - MARKUP: Active uptrend, price rising
#
# BEARISH PHASES:
#   - DISTRIBUTION: Smart money selling to retail at highs
#   - PROFIT_TAKING: Some selling, not full distribution
#   - MARKDOWN: Active downtrend, price falling
#   - CAPITULATION: Panic selling at lows (reversal signal!)
#
# PHASE TRANSITIONS (what ML predicts):
#   ACCUMULATION → MARKUP (bullish breakout coming)
#   MARKUP → DISTRIBUTION (top forming)
#   DISTRIBUTION → MARKDOWN (breakdown coming)
#   MARKDOWN → CAPITULATION (bottom forming)
#   CAPITULATION → ACCUMULATION (cycle restart)
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_LABELS = [
    'ACCUMULATION',      # 0 - Smart money buying quietly at lows
    'RE_ACCUMULATION',   # 1 - Re-loading during uptrend pullback
    'MARKUP',            # 2 - Trending up, momentum phase
    'DISTRIBUTION',      # 3 - Smart money selling to retail at highs
    'PROFIT_TAKING',     # 4 - Some selling, not full distribution
    'MARKDOWN',          # 5 - Trending down
    'CAPITULATION',      # 6 - Panic selling at lows (reversal!)
]

PHASE_LABEL_TO_IDX = {label: i for i, label in enumerate(PHASE_LABELS)}

# Phase to direction mapping (for bias)
PHASE_BIAS = {
    'ACCUMULATION': 'BULLISH',
    'RE_ACCUMULATION': 'BULLISH',
    'MARKUP': 'BULLISH',
    'DISTRIBUTION': 'BEARISH',
    'PROFIT_TAKING': 'BEARISH',
    'MARKDOWN': 'BEARISH',
    'CAPITULATION': 'BULLISH',  # Contrarian - bottom signal!
}

# Likely next phase transitions (for prediction display)
PHASE_TRANSITIONS = {
    'ACCUMULATION': ['MARKUP', 'RE_ACCUMULATION'],
    'RE_ACCUMULATION': ['MARKUP', 'DISTRIBUTION'],
    'MARKUP': ['DISTRIBUTION', 'RE_ACCUMULATION'],
    'DISTRIBUTION': ['MARKDOWN', 'PROFIT_TAKING'],
    'PROFIT_TAKING': ['MARKDOWN', 'DISTRIBUTION', 'MARKUP'],
    'MARKDOWN': ['CAPITULATION', 'DISTRIBUTION'],
    'CAPITULATION': ['ACCUMULATION', 'MARKDOWN'],
}


def get_mode_labels(market_type: str = 'crypto') -> dict:
    """Get mode labels for specific market type."""
    if market_type.lower() in ['stock', 'stocks', 'etf', 'etfs']:
        return MODE_LABELS_STOCK
    return MODE_LABELS_CRYPTO


def auto_tune_thresholds(
    df: pd.DataFrame, 
    mode: str, 
    target_positive_rate: float = 0.35,
    progress_callback=None,
    market_type: str = 'crypto',
) -> dict:
    """
    Automatically calculate optimal thresholds from actual price data.
    
    This finds thresholds that give ~35% positive rate (optimal for ML).
    No more guessing - data-driven thresholds!
    
    Args:
        df: Training DataFrame with OHLCV data
        mode: Trading mode ('scalp', 'daytrade', 'swing', 'investment')
        target_positive_rate: Target positive rate (default 35%)
        progress_callback: Optional callback for progress updates
        market_type: 'crypto' or 'stock' - different approach for stocks!
        
    Returns:
        dict with optimal thresholds for this mode
    """
    if progress_callback:
        progress_callback(0.05, "Auto-tuning thresholds from data...")
    
    is_stock = market_type.lower() in ['stock', 'stocks', 'etf', 'etfs']
    
    # Mode lookahead mapping - stocks get longer lookahead
    if is_stock:
        lookahead_map = {
            'scalp': 6,
            'daytrade': 18,
            'swing': 18,      # Increased for stocks!
            'investment': 20,
        }
    else:
        lookahead_map = {
            'scalp': 6,
            'daytrade': 18,
            'swing': 10,
            'investment': 10,  # Match MODE_LABELS_CRYPTO
        }
    lookahead = lookahead_map.get(mode, 10)
    
    # Calculate forward returns
    if 'Close' not in df.columns:
        print("⚠️ No Close column - using default thresholds")
        return None
    
    df_calc = df.copy()
    
    # Calculate forward highs and lows for quality gate analysis
    df_calc['fwd_return'] = (df_calc['Close'].shift(-lookahead) - df_calc['Close']) / df_calc['Close'] * 100
    df_calc['fwd_max_up'] = df_calc['High'].rolling(lookahead).max().shift(-lookahead)
    df_calc['fwd_max_up_pct'] = (df_calc['fwd_max_up'] - df_calc['Close']) / df_calc['Close'] * 100
    df_calc['fwd_max_down'] = df_calc['Low'].rolling(lookahead).min().shift(-lookahead)
    df_calc['fwd_max_down_pct'] = (df_calc['Close'] - df_calc['fwd_max_down']) / df_calc['Close'] * 100
    
    # Drop NaN
    returns = df_calc['fwd_return'].dropna()
    
    if len(returns) < 100:
        print(f"⚠️ Not enough data for auto-tune ({len(returns)} samples) - using defaults")
        return None
    
    abs_returns = returns.abs()
    
    # ═══════════════════════════════════════════════════════════════════════
    # STOCK SWING: Quality + Mean Reversion thresholds
    # Key insight: Stocks mean-revert. Predict quality not direction.
    # ═══════════════════════════════════════════════════════════════════════
    if is_stock and mode == 'swing':
        print(f"\n📊 Auto-tuning STOCK SWING (Quality + Mean Reversion)...")
        
        max_ups = df_calc['fwd_max_up_pct'].dropna()
        max_downs = df_calc['fwd_max_down_pct'].dropna()
        
        # Target TP: What move is achievable ~15% of the time?
        quality_tp_pct = np.percentile(max_ups, 85)  # 15% reach this
        quality_tp_pct = round(max(quality_tp_pct, 1.5), 2)
        
        # Stop SL: What adverse move happens ~25% of the time?
        quality_sl_pct = np.percentile(max_downs, 75)  # 25% hit this
        quality_sl_pct = round(max(quality_sl_pct, 0.75), 2)
        
        # Mean reversion: Calculate Z-scores and find threshold
        if 'Close' in df_calc.columns:
            sma20 = df_calc['Close'].rolling(20).mean()
            std20 = df_calc['Close'].rolling(20).std()
            z_scores = ((df_calc['Close'] - sma20) / std20).dropna()
            
            # Find Z-score that ~15% of samples exceed (extreme readings)
            mean_rev_threshold = np.percentile(z_scores.abs(), 85)
            mean_rev_threshold = round(max(min(mean_rev_threshold, 3.0), 1.5), 2)
        else:
            mean_rev_threshold = 2.0
        
        # Calculate expected positive rates
        quality_high_count = 0
        quality_low_count = 0
        mean_rev_long_count = 0
        mean_rev_short_count = 0
        total = 0
        
        for i in range(20, len(df_calc) - lookahead):
            current = df_calc['Close'].iloc[i]
            max_up = df_calc['fwd_max_up_pct'].iloc[i]
            max_down = df_calc['fwd_max_down_pct'].iloc[i]
            final_ret = df_calc['fwd_return'].iloc[i]
            
            if pd.isna(max_up) or pd.isna(max_down):
                continue
            
            total += 1
            
            # Check TP/SL hits
            tp_possible = max_up >= quality_tp_pct
            sl_possible = max_down >= quality_sl_pct
            
            if tp_possible and (not sl_possible or max_up > max_down):
                quality_high_count += 1
            elif sl_possible and (not tp_possible or max_down >= max_up):
                quality_low_count += 1
            
            # Check mean reversion
            sma = df_calc['Close'].iloc[i-19:i+1].mean()
            std = df_calc['Close'].iloc[i-19:i+1].std()
            z = (current - sma) / std if std > 0 else 0
            
            if z < -mean_rev_threshold and final_ret > quality_tp_pct * 0.5:
                mean_rev_long_count += 1
            elif z > mean_rev_threshold and final_ret < -quality_tp_pct * 0.5:
                mean_rev_short_count += 1
        
        if total > 0:
            print(f"   quality_tp_pct:      {quality_tp_pct:.2f}% ({quality_high_count/total:.1%} quality_high rate)")
            print(f"   quality_sl_pct:      {quality_sl_pct:.2f}% ({quality_low_count/total:.1%} quality_low rate)")
            print(f"   mean_rev_threshold:  {mean_rev_threshold:.2f} Z ({mean_rev_long_count/total:.1%} long, {mean_rev_short_count/total:.1%} short)")
            print(f"   Data: {total:,} samples, {lookahead} candles lookahead")
        
        return {
            'quality_tp_pct': quality_tp_pct,
            'quality_sl_pct': quality_sl_pct,
            'mean_rev_threshold': mean_rev_threshold,
            'breakout_hold_pct': round(quality_tp_pct * 0.7, 2),
            'vol_expansion_ratio': 1.5,
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # INVESTMENT MODE: Data-driven thresholds for accumulation/distribution
    # Target: 8-15% positive rate per label for good ML learning
    # ═══════════════════════════════════════════════════════════════════════
    if mode == 'investment':
        print(f"\n📊 Auto-tuning INVESTMENT thresholds ({market_type})...")
        
        # Target positive rates for each label (8-15% is optimal for ML)
        target_acc_rate = 0.12      # 12% accumulation
        target_dist_rate = 0.12    # 12% distribution  
        target_rev_rate = 0.10     # 10% reversal
        target_dd_rate = 0.08      # 8% large drawdown
        
        # Calculate position in range for each bar (for accumulation/distribution context)
        range_lookback = 50
        df_calc['range_high'] = df_calc['High'].rolling(range_lookback).max()
        df_calc['range_low'] = df_calc['Low'].rolling(range_lookback).min()
        df_calc['position_in_range'] = (df_calc['Close'] - df_calc['range_low']) / (df_calc['range_high'] - df_calc['range_low'] + 1e-10) * 100
        
        # Calculate recent return (for reversal detection)
        df_calc['recent_return'] = (df_calc['Close'] - df_calc['Close'].shift(5)) / df_calc['Close'].shift(5) * 100
        
        # Get positive and negative returns separately
        pos_returns = returns[returns > 0]
        neg_returns = returns[returns < 0].abs()
        
        # ═══════════════════════════════════════════════════════════════════
        # ACCUMULATION: Find threshold where X% of "at lows + goes up" qualifies
        # ═══════════════════════════════════════════════════════════════════
        # Filter to bars at lows (bottom 60%)
        at_lows_mask = df_calc['position_in_range'] < 60
        at_lows_returns = df_calc.loc[at_lows_mask, 'fwd_return'].dropna()
        at_lows_pos = at_lows_returns[at_lows_returns > 0]
        
        if len(at_lows_pos) > 50:
            # Find percentile that gives target rate
            acc_percentile = (1 - target_acc_rate / (len(at_lows_pos) / len(at_lows_returns))) * 100
            acc_percentile = max(20, min(80, acc_percentile))  # Clamp to reasonable range
            acc_threshold = np.percentile(at_lows_pos, acc_percentile)
        else:
            acc_threshold = np.percentile(pos_returns, 65) if len(pos_returns) > 50 else 3.0
        
        acc_threshold = round(max(acc_threshold, 1.5), 2)  # Min 1.5%
        
        # ═══════════════════════════════════════════════════════════════════
        # DISTRIBUTION: Find threshold where X% of "at highs + goes down" qualifies
        # ═══════════════════════════════════════════════════════════════════
        at_highs_mask = df_calc['position_in_range'] > 40
        at_highs_returns = df_calc.loc[at_highs_mask, 'fwd_return'].dropna()
        at_highs_neg = at_highs_returns[at_highs_returns < 0].abs()
        
        if len(at_highs_neg) > 50:
            dist_percentile = (1 - target_dist_rate / (len(at_highs_neg) / len(at_highs_returns))) * 100
            dist_percentile = max(20, min(80, dist_percentile))
            dist_threshold = np.percentile(at_highs_neg, dist_percentile)
        else:
            dist_threshold = np.percentile(neg_returns, 65) if len(neg_returns) > 50 else 3.0
        
        dist_threshold = round(max(dist_threshold, 1.5), 2)
        
        # ═══════════════════════════════════════════════════════════════════
        # REVERSAL: Find threshold for reversal detection
        # ═══════════════════════════════════════════════════════════════════
        # Reversal to bull: was going down (recent_return < -1%), then goes up
        was_down_mask = df_calc['recent_return'] < -1
        was_down_returns = df_calc.loc[was_down_mask, 'fwd_return'].dropna()
        was_down_pos = was_down_returns[was_down_returns > 0]
        
        if len(was_down_pos) > 50:
            rev_percentile = (1 - target_rev_rate / (len(was_down_pos) / len(was_down_returns))) * 100
            rev_percentile = max(30, min(75, rev_percentile))
            rev_threshold = np.percentile(was_down_pos, rev_percentile)
        else:
            rev_threshold = np.percentile(pos_returns, 70) if len(pos_returns) > 50 else 2.5
        
        rev_threshold = round(max(rev_threshold, 1.0), 2)
        
        # ═══════════════════════════════════════════════════════════════════
        # LARGE DRAWDOWN: Find threshold for max drawdown
        # ═══════════════════════════════════════════════════════════════════
        max_downs = df_calc['fwd_max_down_pct'].dropna()
        
        if len(max_downs) > 50:
            # Find the drawdown level that X% of samples exceed
            dd_percentile = (1 - target_dd_rate) * 100
            dd_threshold = np.percentile(max_downs, dd_percentile)
        else:
            dd_threshold = np.percentile(neg_returns, 85) if len(neg_returns) > 50 else 5.0
        
        dd_threshold = round(max(dd_threshold, 3.0), 2)
        
        # ═══════════════════════════════════════════════════════════════════
        # Validate and print results
        # ═══════════════════════════════════════════════════════════════════
        # Calculate actual positive rates with these thresholds
        actual_acc_rate = ((df_calc['fwd_return'] > acc_threshold) & (df_calc['position_in_range'] < 60)).mean()
        actual_dist_rate = ((df_calc['fwd_return'] < -dist_threshold) & (df_calc['position_in_range'] > 40)).mean()
        actual_rev_rate = ((df_calc['fwd_return'] > rev_threshold) & (df_calc['recent_return'] < -1)).mean()
        actual_dd_rate = (df_calc['fwd_max_down_pct'] > dd_threshold).mean()
        
        print(f"   ✅ Accumulation:    {acc_threshold:.2f}% threshold → {actual_acc_rate:.1%} positive rate")
        print(f"   ✅ Distribution:    {dist_threshold:.2f}% threshold → {actual_dist_rate:.1%} positive rate")
        print(f"   ✅ Reversal:        {rev_threshold:.2f}% threshold → {actual_rev_rate:.1%} positive rate")
        print(f"   ✅ Large Drawdown:  {dd_threshold:.2f}% threshold → {actual_dd_rate:.1%} positive rate")
        print(f"   📈 Data: {len(returns):,} samples, {lookahead} candles lookahead")
        
        return {
            'accumulation': acc_threshold,
            'distribution': dist_threshold,
            'reversal': rev_threshold,
            'large_drawdown': dd_threshold,
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # CRYPTO / Other modes: Directional thresholds
    # ═══════════════════════════════════════════════════════════════════════
    
    # Calculate percentiles for target positive rate
    trend_percentile = (1 - target_positive_rate) * 100
    reversal_percentile = (1 - target_positive_rate * 0.85) * 100
    drawdown_percentile = (1 - target_positive_rate * 0.75) * 100
    
    trend_threshold = np.percentile(abs_returns, trend_percentile)
    reversal_threshold = np.percentile(abs_returns, reversal_percentile)
    drawdown_threshold = np.percentile(abs_returns, drawdown_percentile)
    
    # Ensure minimum thresholds
    trend_threshold = max(trend_threshold, 0.3)
    reversal_threshold = max(reversal_threshold, 0.2)
    drawdown_threshold = max(drawdown_threshold, 0.2)
    
    trend_threshold = round(trend_threshold, 2)
    reversal_threshold = round(reversal_threshold, 2)
    drawdown_threshold = round(drawdown_threshold, 2)
    
    actual_trend_rate = (abs_returns >= trend_threshold).mean()
    actual_reversal_rate = (abs_returns >= reversal_threshold).mean()
    actual_drawdown_rate = (abs_returns >= drawdown_threshold).mean()
    
    print(f"📊 Auto-tuned thresholds for {mode} ({market_type}):")
    print(f"   Trend/Continuation: {trend_threshold:.2f}% ({actual_trend_rate:.1%} positive rate)")
    print(f"   Reversal/Fakeout:   {reversal_threshold:.2f}% ({actual_reversal_rate:.1%} positive rate)")
    print(f"   Drawdown:           {drawdown_threshold:.2f}% ({actual_drawdown_rate:.1%} positive rate)")
    print(f"   Data analyzed: {len(returns):,} samples, {lookahead} candles lookahead")
    
    # Return thresholds based on mode
    if mode in ['scalp', 'daytrade']:
        return {
            'continuation': trend_threshold,
            'fakeout': drawdown_threshold,
            'vol_expansion': 1.4,
        }
    elif mode == 'swing':
        return {
            'trend_holds': trend_threshold,
            'reversal': reversal_threshold,
            'drawdown': drawdown_threshold,
        }
    # Note: investment mode is handled above with dedicated auto-tune logic
    
    return {
        'trend_holds': trend_threshold,
        'reversal': reversal_threshold,
        'drawdown': drawdown_threshold,
    }


# Global cache for auto-tuned thresholds (avoid recalculating)
_AUTO_TUNED_THRESHOLDS = {}


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC OUTPUT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProbabilisticPrediction:
    """
    Output from probabilistic ML model.
    
    Example for Day Trade:
        P_continuation = 0.64
        P_fakeout = 0.19
        P_vol_expansion = 0.72
        
    Now includes expected price targets!
    """
    mode: str
    probabilities: Dict[str, float]  # e.g., {'continuation': 0.64, 'fakeout': 0.19, ...}
    
    # Derived direction from rules interpretation
    direction: str                    # 'LONG', 'SHORT', 'WAIT'
    confidence: float                 # 0-100
    
    # Why this decision
    reasoning: str
    
    # Feature importance for this prediction
    top_features: List[Tuple[str, float, float]]
    
    # Meta
    model_version: str
    prediction_time: str
    
    # NEW: Expected price targets based on thresholds
    expected_move_pct: float = 0.0       # Minimum expected move (the threshold)
    min_tp_pct: float = 0.0              # Minimum TP (= threshold)
    suggested_sl_pct: float = 0.0        # Suggested SL (= drawdown threshold)
    risk_reward: float = 0.0             # TP / SL ratio
    
    # F1 scores per label (from trained model metadata)
    f1_scores: Dict[str, float] = field(default_factory=dict)  # {'continuation_bull': 0.62, ...}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE PREDICTION (NEW!)
    # ═══════════════════════════════════════════════════════════════════════════
    current_phase: str = 'UNKNOWN'           # Detected current phase
    phase_probabilities: Dict[str, float] = field(default_factory=dict)  # {'ACCUMULATION': 0.15, 'MARKUP': 0.45, ...}
    next_phase: str = 'UNKNOWN'              # Predicted next phase (highest prob transition)
    next_phase_confidence: float = 0.0       # Confidence in next phase prediction
    phase_bias: str = 'NEUTRAL'              # BULLISH/BEARISH based on phase
    
    def __repr__(self):
        probs = " | ".join([f"P_{k}={v:.2f}" for k, v in self.probabilities.items()])
        return f"ProbabilisticPrediction({self.mode}): {probs} → {self.direction} ({self.confidence:.0f}%)"
    
    def get_targets(self, current_price: float) -> Dict:
        """Calculate actual price targets from current price."""
        if self.direction == 'LONG':
            tp1 = current_price * (1 + self.min_tp_pct / 100)
            tp2 = current_price * (1 + self.min_tp_pct * 1.5 / 100)  # 1.5x minimum
            tp3 = current_price * (1 + self.min_tp_pct * 2 / 100)    # 2x minimum
            sl = current_price * (1 - self.suggested_sl_pct / 100)
        elif self.direction == 'SHORT':
            tp1 = current_price * (1 - self.min_tp_pct / 100)
            tp2 = current_price * (1 - self.min_tp_pct * 1.5 / 100)
            tp3 = current_price * (1 - self.min_tp_pct * 2 / 100)
            sl = current_price * (1 + self.suggested_sl_pct / 100)
        else:  # WAIT
            return {'tp1': None, 'tp2': None, 'tp3': None, 'sl': None}
        
        return {
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'tp3': round(tp3, 4),
            'sl': round(sl, 4),
            'tp1_pct': self.min_tp_pct,
            'tp2_pct': self.min_tp_pct * 1.5,
            'tp3_pct': self.min_tp_pct * 2,
            'sl_pct': self.suggested_sl_pct,
            'risk_reward': self.risk_reward,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_enhanced_features(
    df: pd.DataFrame,
    idx: int,
    whale_data: Dict = None,
    smc_data: Dict = None,
    market_context: Dict = None,
    market_type: str = 'crypto',
) -> np.ndarray:
    """
    Extract enhanced feature vector from DataFrame row.
    
    Args:
        df: OHLCV DataFrame with indicators
        idx: Index to extract features for
        whale_data: Whale/retail positioning data
        smc_data: SMC structure data (OBs, FVGs, liquidity)
        market_context: BTC correlation, fear/greed, etc.
        market_type: 'crypto' or 'stock' - stocks use TA features instead of whale data
    
    Returns:
        Feature vector of shape (NUM_ENHANCED_FEATURES,)
    """
    features = np.zeros(NUM_ENHANCED_FEATURES)
    
    whale_data = whale_data or {}
    smc_data = smc_data or {}
    market_context = market_context or {}
    
    is_stock = market_type.lower() in ['stock', 'stocks', 'etf', 'etfs']
    
    try:
        row = df.iloc[idx]
        close = row['Close']
        high = row['High']
        low = row['Low']
        volume = row['Volume'] if 'Volume' in row else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # PRICE & STRUCTURE
        # ═══════════════════════════════════════════════════════════════════════
        
        # Returns
        if idx >= 1:
            features[0] = (close - df['Close'].iloc[idx-1]) / df['Close'].iloc[idx-1] * 100
        if idx >= 3:
            features[1] = (close - df['Close'].iloc[idx-3]) / df['Close'].iloc[idx-3] * 100
        if idx >= 5:
            features[2] = (close - df['Close'].iloc[idx-5]) / df['Close'].iloc[idx-5] * 100
        if idx >= 10:
            features[3] = (close - df['Close'].iloc[idx-10]) / df['Close'].iloc[idx-10] * 100
        
        # Range metrics
        features[4] = (high - low) / close * 100 if close > 0 else 0  # high_low_range
        features[5] = (close - low) / (high - low) if (high - low) > 0 else 0.5  # close_position
        features[6] = (high - close) / close * 100 if close > 0 else 0  # distance_from_high
        features[7] = (close - low) / close * 100 if close > 0 else 0  # distance_from_low
        
        # Break of structure (from structure column or detect)
        if 'structure_bos' in df.columns:
            features[8] = 1 if df['structure_bos'].iloc[idx] != 0 else 0
            features[9] = df['structure_bos'].iloc[idx]  # direction
        else:
            # Simple BoS detection: HH+HL = bullish, LH+LL = bearish
            if idx >= 3:
                recent_highs = df['High'].iloc[idx-3:idx+1]
                recent_lows = df['Low'].iloc[idx-3:idx+1]
                hh = recent_highs.iloc[-1] > recent_highs.iloc[:-1].max()
                hl = recent_lows.iloc[-1] > recent_lows.iloc[:-1].min()
                lh = recent_highs.iloc[-1] < recent_highs.iloc[:-1].max()
                ll = recent_lows.iloc[-1] < recent_lows.iloc[:-1].min()
                
                if hh and hl:
                    features[8], features[9] = 1, 1  # Bullish BoS
                elif lh and ll:
                    features[8], features[9] = 1, -1  # Bearish BoS
        
        # Compression metrics
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            bb_upper = df['BB_upper'].iloc[idx]
            bb_lower = df['BB_lower'].iloc[idx]
            bb_width = (bb_upper - bb_lower) / close * 100 if close > 0 else 5
            features[11] = bb_width
            
            # Squeeze percentile (lower width = higher compression)
            if idx >= 20:
                bb_widths = ((df['BB_upper'] - df['BB_lower']) / df['Close'] * 100).iloc[idx-20:idx+1]
                features[10] = 100 - (bb_widths.iloc[-1] / bb_widths.max() * 100)  # Inverse percentile
            
            # Compression candles
            if bb_width < 3:  # Tight bands
                features[12] = sum(1 for i in range(max(0, idx-20), idx) 
                                   if (df['BB_upper'].iloc[i] - df['BB_lower'].iloc[i]) / df['Close'].iloc[i] * 100 < 3)
        
        # ATR
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[idx]
            features[13] = atr / close * 100 if close > 0 else 2
            if idx >= 10 and df['ATR'].iloc[idx-10] > 0:
                features[14] = atr / df['ATR'].iloc[idx-10]
        
        # ═══════════════════════════════════════════════════════════════════════
        # EMA TREND FEATURES - CRITICAL FOR TREND DETECTION!
        # These prevent LONG signals in falling knife scenarios
        # ═══════════════════════════════════════════════════════════════════════
        
        # Get EMA values (try multiple column name conventions)
        ema20 = None
        ema50 = None
        rsi = df['RSI'].iloc[idx] if 'RSI' in df.columns else 50
        
        # Try different EMA column names
        for ema20_col in ['EMA_20', 'EMA20', 'ema_20', 'ema20']:
            if ema20_col in df.columns:
                ema20 = df[ema20_col].iloc[idx]
                break
        for ema50_col in ['EMA_50', 'EMA50', 'ema_50', 'ema50']:
            if ema50_col in df.columns:
                ema50 = df[ema50_col].iloc[idx]
                break
        
        # Calculate EMA if not present (fallback)
        if ema20 is None and idx >= 20:
            ema20 = df['Close'].iloc[idx-19:idx+1].ewm(span=20, adjust=False).mean().iloc[-1]
        if ema50 is None and idx >= 50:
            ema50 = df['Close'].iloc[idx-49:idx+1].ewm(span=50, adjust=False).mean().iloc[-1]
        
        # Feature 15: price_vs_ema20 (% distance)
        if ema20 is not None and ema20 > 0:
            features[15] = (close - ema20) / ema20 * 100
        
        # Feature 16: price_vs_ema50 (% distance)
        if ema50 is not None and ema50 > 0:
            features[16] = (close - ema50) / ema50 * 100
        
        # Feature 17: ema20_vs_ema50 (EMA alignment)
        if ema20 is not None and ema50 is not None and ema50 > 0:
            features[17] = (ema20 - ema50) / ema50 * 100
        
        # Feature 18: ema_trend_score (-2 to +2)
        # -2 = price < EMA20 < EMA50 (strong downtrend)
        # -1 = price < EMA20, but EMA20 > EMA50 (weakening uptrend)
        # 0 = mixed/neutral
        # +1 = price > EMA20, but EMA20 < EMA50 (weakening downtrend) 
        # +2 = price > EMA20 > EMA50 (strong uptrend)
        ema_trend_score = 0
        if ema20 is not None and ema50 is not None:
            price_above_ema20 = close > ema20
            price_above_ema50 = close > ema50
            ema20_above_ema50 = ema20 > ema50
            
            if price_above_ema20 and price_above_ema50 and ema20_above_ema50:
                ema_trend_score = 2  # Strong uptrend
            elif price_above_ema20 and ema20_above_ema50:
                ema_trend_score = 1  # Uptrend
            elif not price_above_ema20 and not price_above_ema50 and not ema20_above_ema50:
                ema_trend_score = -2  # Strong downtrend (FALLING KNIFE!)
            elif not price_above_ema20 and not ema20_above_ema50:
                ema_trend_score = -1  # Downtrend
        features[18] = ema_trend_score
        
        # Feature 19: falling_knife (DANGER SIGNAL!)
        # 1 if: price < EMA20 < EMA50 AND RSI < 35
        falling_knife = 0
        if ema20 is not None and ema50 is not None:
            if close < ema20 and ema20 < ema50 and rsi < 35:
                falling_knife = 1
        features[19] = falling_knife
        
        # ═══════════════════════════════════════════════════════════════════════
        # POSITIONING & FLOW (Different for Crypto vs Stocks)
        # Indices 20-31 (shifted +5 for EMA features)
        # ═══════════════════════════════════════════════════════════════════════
        
        if is_stock:
            # STOCK-SPECIFIC FEATURES (replace whale data with TA-derived signals)
            
            # Volume analysis (stocks volume is more meaningful)
            if volume > 0 and idx >= 20:
                avg_volume = df['Volume'].iloc[idx-20:idx].mean()
                features[20] = (volume / avg_volume - 1) * 100 if avg_volume > 0 else 0  # Volume ratio
                features[21] = (df['Volume'].iloc[idx-5:idx].mean() / avg_volume - 1) * 100 if avg_volume > 0 else 0  # 5-day vol trend
            
            # Price momentum
            if idx >= 20:
                features[22] = (close - df['Close'].iloc[idx-20]) / df['Close'].iloc[idx-20] * 100  # 20-day return
            
            # RSI-based signals
            if 'RSI' in df.columns:
                rsi_val = df['RSI'].iloc[idx]
                features[23] = rsi_val - 50  # RSI deviation from neutral
                features[24] = rsi_val - df['RSI'].iloc[idx-5] if idx >= 5 else 0  # RSI momentum
            
            # MACD signals
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[idx]
                signal = df['MACD_signal'].iloc[idx]
                features[25] = (macd - signal) * 100  # MACD histogram
                if idx >= 3:
                    prev_hist = df['MACD'].iloc[idx-3] - df['MACD_signal'].iloc[idx-3]
                    features[26] = ((macd - signal) - prev_hist) * 100  # Histogram change
            
            # Moving average positions
            if 'SMA_20' in df.columns:
                features[27] = (close - df['SMA_20'].iloc[idx]) / df['SMA_20'].iloc[idx] * 100
            if 'SMA_50' in df.columns:
                features[28] = (close - df['SMA_50'].iloc[idx]) / df['SMA_50'].iloc[idx] * 100
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                features[29] = (df['EMA_12'].iloc[idx] - df['EMA_26'].iloc[idx]) / close * 100
            
            # Gap analysis (important for stocks)
            if idx >= 1:
                prev_close = df['Close'].iloc[idx-1]
                open_price = row['Open'] if 'Open' in row else close
                features[30] = (open_price - prev_close) / prev_close * 100  # Gap %
            
        else:
            # CRYPTO FEATURES (whale/OI data)
            features[20] = whale_data.get('oi_change_1h', 0)
            features[21] = whale_data.get('oi_change_24h', 0)
            
            # Price vs OI divergence
            price_change = features[3] if idx >= 10 else features[0]
            oi_change = whale_data.get('oi_change_24h', 0)
            features[22] = price_change - oi_change
            
            # Funding
            features[23] = whale_data.get('funding_rate', 0) * 100
            features[24] = whale_data.get('funding_rate_delta', 0) * 100
            
            # Long/Short ratio
            features[25] = whale_data.get('long_short_ratio', 1.0)
            
            # Whale/Retail
            features[26] = whale_data.get('whale_long_pct', 50)
            features[27] = whale_data.get('retail_long_pct', 50)
            features[28] = features[26] - features[27]  # Divergence
            features[29] = (features[26] - 50) / 50  # Whale dominance normalized
            
            # Institutional flow
            features[30] = whale_data.get('institutional_buy_imbalance', 0)
        
        # Smart money flow (CMF) - works for both
        if 'CMF' in df.columns:
            features[31] = df['CMF'].iloc[idx] * 100
        
        # ═══════════════════════════════════════════════════════════════════════
        # SMC / CONTEXT - Read from dataframe columns (pre-calculated) or smc_data dict
        # Indices 32-43 (shifted +5 for EMA features)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Liquidity sweeps
        if 'smc_liquidity_sweep_bull' in df.columns:
            features[32] = df['smc_liquidity_sweep_bull'].iloc[idx]
        else:
            features[32] = 1 if smc_data.get('liquidity_sweep_bull', False) else 0
        
        if 'smc_liquidity_sweep_bear' in df.columns:
            features[33] = df['smc_liquidity_sweep_bear'].iloc[idx]
        else:
            features[33] = 1 if smc_data.get('liquidity_sweep_bear', False) else 0
        
        # Order blocks
        if 'smc_at_bullish_ob' in df.columns:
            features[34] = df['smc_at_bullish_ob'].iloc[idx]
        else:
            features[34] = 1 if smc_data.get('at_bullish_ob', False) else 0
        
        if 'smc_at_bearish_ob' in df.columns:
            features[35] = df['smc_at_bearish_ob'].iloc[idx]
        else:
            features[35] = 1 if smc_data.get('at_bearish_ob', False) else 0
        
        if 'smc_near_bullish_ob' in df.columns:
            features[36] = df['smc_near_bullish_ob'].iloc[idx]
        else:
            features[36] = 1 if smc_data.get('near_bullish_ob', False) else 0
        
        if 'smc_near_bearish_ob' in df.columns:
            features[37] = df['smc_near_bearish_ob'].iloc[idx]
        else:
            features[37] = 1 if smc_data.get('near_bearish_ob', False) else 0
        
        # FVGs
        if 'smc_in_fvg_bullish' in df.columns:
            features[38] = df['smc_in_fvg_bullish'].iloc[idx]
        else:
            features[38] = 1 if smc_data.get('in_fvg_bullish', False) else 0
        
        if 'smc_in_fvg_bearish' in df.columns:
            features[39] = df['smc_in_fvg_bearish'].iloc[idx]
        else:
            features[39] = 1 if smc_data.get('in_fvg_bearish', False) else 0
        
        # Accumulation/Distribution scores
        if 'smc_accumulation_score' in df.columns:
            features[40] = df['smc_accumulation_score'].iloc[idx]
        else:
            features[40] = smc_data.get('accumulation_score', 50)
        
        if 'smc_distribution_score' in df.columns:
            features[41] = df['smc_distribution_score'].iloc[idx]
        else:
            features[41] = smc_data.get('distribution_score', 50)
        
        # Session timing (features 42-43)
        if 'hour' in df.columns:
            hour = df['hour'].iloc[idx]
        else:
            hour = 12  # Default to noon
        
        if 0 <= hour < 8:
            features[42] = -1  # Asian
        elif 8 <= hour < 14:
            features[42] = 0   # London
        else:
            features[42] = 1   # NY
        
        features[43] = 1 if 13 <= hour <= 17 else 0  # London-NY overlap
        
        # ═══════════════════════════════════════════════════════════════════════
        # MARKET CONTEXT & REGIME (Auto-calculated from data)
        # Indices 44-49 (shifted +5 for EMA features)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Calculate regime from data if not provided
        if not market_context:
            regime = calculate_market_regime(df, idx)
            market_context = regime
        
        # Regime features (KEY for bull/bear adaptation)
        features[44] = market_context.get('btc_correlation', 0.5)
        features[45] = market_context.get('btc_trend', market_context.get('regime_trend', 0))  # -1=bear, 0=neutral, 1=bull
        features[46] = market_context.get('htf_trend_alignment', market_context.get('trend_strength', 0) / 100)
        features[47] = market_context.get('fear_greed_index', 50)
        features[48] = market_context.get('volatility_regime', market_context.get('regime_volatility', 0))
        features[49] = market_context.get('momentum_regime', market_context.get('regime_momentum', 0))
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEMPORAL FEATURES (Critical for stocks!)
        # Indices 50-55 (shifted +5 for EMA features)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Get timestamp from dataframe
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[idx]
        elif 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'].iloc[idx])
        elif 'DateTime' in df.columns:
            ts = pd.to_datetime(df['DateTime'].iloc[idx])
        else:
            ts = pd.Timestamp.now()  # Fallback
        
        # Day of week (0=Mon, 4=Fri)
        features[50] = ts.dayofweek if hasattr(ts, 'dayofweek') else 2
        
        # Month (1-12)
        features[51] = ts.month if hasattr(ts, 'month') else 6
        
        # Is month end? (last 3 trading days)
        if hasattr(ts, 'day') and hasattr(ts, 'month'):
            import calendar
            days_in_month = calendar.monthrange(ts.year, ts.month)[1]
            features[52] = 1.0 if ts.day >= days_in_month - 2 else 0.0
        else:
            features[52] = 0.0
        
        # Is quarter end? (last week of quarter)
        if hasattr(ts, 'month') and hasattr(ts, 'day'):
            is_quarter_end_month = ts.month in [3, 6, 9, 12]
            features[53] = 1.0 if is_quarter_end_month and ts.day >= 23 else 0.0
        else:
            features[53] = 0.0
        
        # Is first week of month?
        features[54] = 1.0 if hasattr(ts, 'day') and ts.day <= 7 else 0.0
        
        # Days in trend (count consecutive days in same direction)
        if idx >= 5:
            trend_days = 0
            current_direction = 1 if close > df['Close'].iloc[idx-1] else -1
            for j in range(idx-1, max(0, idx-30), -1):
                day_dir = 1 if df['Close'].iloc[j] > df['Close'].iloc[j-1] else -1
                if day_dir == current_direction:
                    trend_days += 1
                else:
                    break
            features[55] = min(trend_days, 30)  # Cap at 30
        else:
            features[55] = 0
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
    
    return features.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK-SPECIFIC FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_stock_features(
    df: pd.DataFrame,
    idx: int,
    institutional_data: Dict = None,
    market_data: Dict = None
) -> np.ndarray:
    """
    Extract stock-specific features that capture what actually moves stocks.
    
    Key differences from crypto features:
    - Mean reversion signals (stocks revert, crypto trends)
    - Volume profile (volume tells truth in stocks)
    - Market correlation (stocks move with SPY)
    - Institutional footprint (from Quiver)
    """
    features = np.zeros(NUM_STOCK_FEATURES)
    institutional_data = institutional_data or {}
    market_data = market_data or {}
    
    try:
        row = df.iloc[idx]
        close = row['Close']
        high = row['High']
        low = row['Low']
        open_price = row['Open'] if 'Open' in row else close
        volume = row['Volume'] if 'Volume' in row else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # MEAN REVERSION SIGNALS (Indices 0-9)
        # ═══════════════════════════════════════════════════════════════════════
        
        # SMA calculations
        if idx >= 50:
            sma20 = df['Close'].iloc[idx-19:idx+1].mean()
            sma50 = df['Close'].iloc[idx-49:idx+1].mean()
            std20 = df['Close'].iloc[idx-19:idx+1].std()
            std50 = df['Close'].iloc[idx-49:idx+1].std()
            
            features[1] = (close - sma20) / sma20 * 100 if sma20 > 0 else 0  # dist from sma20
            features[2] = (close - sma50) / sma50 * 100 if sma50 > 0 else 0  # dist from sma50
            features[7] = (close - sma20) / std20 if std20 > 0 else 0  # z_score_20
            features[8] = (close - sma50) / std50 if std50 > 0 else 0  # z_score_50
        
        # VWAP distance
        if 'VWAP' in df.columns:
            vwap = df['VWAP'].iloc[idx]
            features[0] = (close - vwap) / vwap * 100 if vwap > 0 else 0
        
        # RSI
        rsi = 50
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[idx]
        elif idx >= 14:
            deltas = df['Close'].diff().iloc[idx-13:idx+1]
            gains = deltas.where(deltas > 0, 0).mean()
            losses = -deltas.where(deltas < 0, 0).mean()
            rs = gains / losses if losses > 0 else 100
            rsi = 100 - (100 / (1 + rs))
        features[3] = rsi
        features[4] = rsi - 50  # Centered RSI
        
        # Bollinger Band position
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            bb_upper = df['BB_upper'].iloc[idx]
            bb_lower = df['BB_lower'].iloc[idx]
            bb_range = bb_upper - bb_lower
            features[5] = (close - bb_lower) / bb_range if bb_range > 0 else 0.5
            
            if idx >= 20:
                bb_widths = (df['BB_upper'] - df['BB_lower']).iloc[idx-19:idx+1]
                current_width = bb_range
                features[6] = 100 * (1 - current_width / bb_widths.max()) if bb_widths.max() > 0 else 50
        
        # Combined mean reversion score
        z_score = features[7]
        rsi_score = (50 - features[3]) / 50 if features[3] != 0 else 0
        bb_score = (0.5 - features[5]) * 2
        features[9] = (z_score * 0.4 + rsi_score * 0.3 + bb_score * 0.3) * 33 + 50
        
        # ═══════════════════════════════════════════════════════════════════════
        # MOMENTUM & TREND QUALITY (Indices 10-17)
        # ═══════════════════════════════════════════════════════════════════════
        
        # ADX
        if 'ADX' in df.columns:
            features[10] = df['ADX'].iloc[idx]
        else:
            features[10] = 25
        
        # Trend efficiency
        if idx >= 10:
            net_move = abs(close - df['Close'].iloc[idx-10])
            total_move = df['Close'].diff().abs().iloc[idx-9:idx+1].sum()
            features[11] = net_move / total_move if total_move > 0 else 0.5
        
        # Consecutive direction bars
        if idx >= 5:
            directions = np.sign(df['Close'].diff().iloc[idx-4:idx+1])
            current_dir = directions.iloc[-1]
            consecutive = 1
            for d in reversed(directions.iloc[:-1].values):
                if d == current_dir:
                    consecutive += 1
                else:
                    break
            features[12] = consecutive * current_dir
        
        # MACD
        if 'MACD_hist' in df.columns:
            features[13] = df['MACD_hist'].iloc[idx]
            if idx >= 1:
                features[14] = df['MACD_hist'].iloc[idx] - df['MACD_hist'].iloc[idx-1]
        
        # Momentum
        if idx >= 10:
            features[15] = (close - df['Close'].iloc[idx-10]) / df['Close'].iloc[idx-10] * 100
        if idx >= 20:
            features[16] = (close - df['Close'].iloc[idx-20]) / df['Close'].iloc[idx-20] * 100
        
        # Momentum divergence
        if idx >= 20:
            price_hh = close > df['Close'].iloc[idx-20:idx].max()
            mom_values = df['Close'].pct_change(10).iloc[idx-10:idx]
            momentum_lh = features[15] < (mom_values.max() * 100) if len(mom_values) > 0 else False
            features[17] = 1 if price_hh and momentum_lh else (-1 if not price_hh and not momentum_lh else 0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # VOLUME PROFILE (Indices 18-24)
        # ═══════════════════════════════════════════════════════════════════════
        
        if volume > 0 and idx >= 20:
            avg_volume = df['Volume'].iloc[idx-19:idx].mean()
            
            features[18] = volume / avg_volume if avg_volume > 0 else 1  # Relative volume
            
            recent_vol = df['Volume'].iloc[idx-4:idx+1].mean()
            older_vol = df['Volume'].iloc[idx-9:idx-4].mean()
            features[19] = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0  # Volume trend
            
            price_changes = df['Close'].pct_change().iloc[idx-9:idx+1]
            vol_changes = df['Volume'].pct_change().iloc[idx-9:idx+1]
            features[20] = price_changes.corr(vol_changes) if len(price_changes) > 2 else 0
            
            features[23] = 1 if volume > 2 * avg_volume else 0  # Volume breakout
            features[24] = ((close - low) / (high - low) if high != low else 0.5) * (volume / 1e6)  # Buying pressure
        
        # ═══════════════════════════════════════════════════════════════════════
        # VOLATILITY REGIME (Indices 25-30)
        # ═══════════════════════════════════════════════════════════════════════
        
        if 'ATR' in df.columns and idx >= 50:
            atr = df['ATR'].iloc[idx]
            historical_atrs = df['ATR'].iloc[idx-49:idx+1]
            
            features[25] = (historical_atrs < atr).sum() / len(historical_atrs) * 100  # ATR percentile
            
            recent_atr = historical_atrs.iloc[-10:].mean()
            older_atr = historical_atrs.iloc[-20:-10].mean()
            features[26] = 1 if recent_atr < older_atr * 0.8 else 0  # Vol contraction
            features[27] = 1 if recent_atr > older_atr * 1.2 else 0  # Vol expansion
        
        if idx >= 20:
            today_range = high - low
            historical_ranges = (df['High'] - df['Low']).iloc[idx-19:idx]
            features[28] = (historical_ranges < today_range).sum() / len(historical_ranges) * 100
        
        if idx >= 1:
            prev_close = df['Close'].iloc[idx-1]
            features[29] = (open_price - prev_close) / prev_close * 100  # Gap
        
        features[30] = (high - low) / open_price * 100 if open_price > 0 else 0  # Intraday vol
        
        # ═══════════════════════════════════════════════════════════════════════
        # STRUCTURE & S/R (Indices 31-38)
        # ═══════════════════════════════════════════════════════════════════════
        
        if idx >= 252:
            high_52w = df['High'].iloc[idx-251:idx+1].max()
            low_52w = df['Low'].iloc[idx-251:idx+1].min()
            features[31] = (high_52w - close) / high_52w * 100
            features[32] = (close - low_52w) / low_52w * 100
        
        if idx >= 20:
            high_20 = df['High'].iloc[idx-19:idx+1].max()
            low_20 = df['Low'].iloc[idx-19:idx+1].min()
            range_20 = high_20 - low_20
            features[33] = (close - low_20) / range_20 * 100 if range_20 > 0 else 50
            features[34] = 1 if (close - low_20) / low_20 * 100 < 2 else 0  # At support
            features[35] = 1 if (high_20 - close) / close * 100 < 2 else 0  # At resistance
        
        if idx >= 5:
            recent_high = df['High'].iloc[idx-4:idx].max()
            recent_low = df['Low'].iloc[idx-4:idx].min()
            features[37] = 1 if high > recent_high else 0  # Higher high
            features[38] = 1 if low < recent_low else 0  # Lower low
        
        # ═══════════════════════════════════════════════════════════════════════
        # INSTITUTIONAL SIGNALS (Indices 39-43)
        # ═══════════════════════════════════════════════════════════════════════
        
        features[39] = institutional_data.get('congress_sentiment', 0)
        features[40] = institutional_data.get('insider_sentiment', 0)
        features[41] = institutional_data.get('short_interest_pct', 0)
        features[42] = institutional_data.get('short_interest_change', 0)
        features[43] = (features[39] + features[40] - features[42]) / 3  # Combined
        
        # ═══════════════════════════════════════════════════════════════════════
        # MARKET CONTEXT (Indices 44-49)
        # ═══════════════════════════════════════════════════════════════════════
        
        features[44] = market_data.get('spy_correlation', 0.5)
        features[45] = market_data.get('spy_relative_strength', 0)
        features[46] = market_data.get('sector_relative_strength', 0)
        features[47] = market_data.get('market_regime', 0)
        features[48] = market_data.get('vix_level', 20)
        features[49] = market_data.get('vix_trend', 0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TEMPORAL (Indices 50-53)
        # ═══════════════════════════════════════════════════════════════════════
        
        if hasattr(df.index, 'dayofweek'):
            try:
                dt = df.index[idx]
                features[50] = dt.dayofweek
                features[51] = 1 if dt.day >= 25 else 0
                features[52] = 1 if dt.month in [3, 6, 9, 12] and dt.day >= 25 else 0
            except:
                pass
        
        features[53] = institutional_data.get('days_to_earnings', 999)
        
    except Exception as e:
        print(f"[STOCK_FEATURES] Error at idx {idx}: {e}")
    
    return features.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_market_regime(df: pd.DataFrame, idx: int) -> dict:
    """
    Calculate market regime features from price data.
    This allows the model to automatically adjust to bull/bear markets.
    
    Returns:
        dict with regime features
    """
    regime = {
        'btc_trend': 0,           # -1=bear, 0=neutral, 1=bull
        'trend_strength': 0,      # 0-100
        'volatility_regime': 0,   # -1=low vol, 0=normal, 1=high vol
        'momentum_regime': 0,     # -1=bearish momentum, 0=neutral, 1=bullish
    }
    
    try:
        close = df['Close'].iloc[idx]
        
        # Calculate moving averages for trend
        if idx >= 30:
            ma_10 = df['Close'].iloc[idx-9:idx+1].mean()
            ma_30 = df['Close'].iloc[idx-29:idx+1].mean()
            
            # Trend direction
            if close > ma_10 > ma_30:
                regime['btc_trend'] = 1  # Bullish
                regime['trend_strength'] = min(100, ((close - ma_30) / ma_30 * 100) * 10)
            elif close < ma_10 < ma_30:
                regime['btc_trend'] = -1  # Bearish
                regime['trend_strength'] = min(100, ((ma_30 - close) / ma_30 * 100) * 10)
            else:
                regime['btc_trend'] = 0  # Neutral/Ranging
                regime['trend_strength'] = 0
        
        # Volatility regime (ATR relative to average)
        if 'ATR' in df.columns and idx >= 20:
            current_atr = df['ATR'].iloc[idx]
            avg_atr = df['ATR'].iloc[idx-19:idx+1].mean()
            if avg_atr > 0:
                vol_ratio = current_atr / avg_atr
                if vol_ratio > 1.5:
                    regime['volatility_regime'] = 1  # High volatility
                elif vol_ratio < 0.7:
                    regime['volatility_regime'] = -1  # Low volatility
        
        # Momentum regime (RSI based)
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[idx]
            if rsi > 60:
                regime['momentum_regime'] = 1  # Bullish momentum
            elif rsi < 40:
                regime['momentum_regime'] = -1  # Bearish momentum
        
    except Exception as e:
        pass
    
    return regime


def add_regime_features_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime columns to dataframe for training."""
    df = df.copy()
    
    # Initialize columns
    df['regime_trend'] = 0
    df['regime_strength'] = 0
    df['regime_volatility'] = 0
    df['regime_momentum'] = 0
    
    for i in range(len(df)):
        regime = calculate_market_regime(df, i)
        df.loc[df.index[i], 'regime_trend'] = regime['btc_trend']
        df.loc[df.index[i], 'regime_strength'] = regime['trend_strength']
        df.loc[df.index[i], 'regime_volatility'] = regime['volatility_regime']
        df.loc[df.index[i], 'regime_momentum'] = regime['momentum_regime']
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LABEL GENERATION (MODE-SPECIFIC)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_probabilistic_labels(
    df: pd.DataFrame,
    mode: str,
    market_type: str = 'crypto',
    custom_thresholds: dict = None,
) -> pd.DataFrame:
    """
    Generate mode-specific probabilistic labels based on what ACTUALLY happened.
    
    Args:
        df: DataFrame with OHLCV data
        mode: Trading mode ('scalp', 'daytrade', 'swing', 'investment')
        market_type: 'crypto' or 'stock' - different thresholds!
        custom_thresholds: Optional auto-tuned thresholds (overrides defaults)
    
    DIRECTIONAL LABELS:
    - reversal_to_bull: Was going down, reversed UP
    - reversal_to_bear: Was going up, reversed DOWN
    - fakeout_to_bull: Faked down then went UP
    - fakeout_to_bear: Faked up then went DOWN
    
    Day Trade / Scalp labels:
    - continuation_bull/bear: Price continued in direction
    - fakeout_to_bull/bear: Initial move reversed (DIRECTIONAL)
    - vol_expansion: Volatility increased significantly
    
    Swing labels:
    - trend_holds_bull/bear: Existing trend continued
    - reversal_to_bull/bear: Trend reversed (DIRECTIONAL)
    - drawdown: Significant adverse move occurred
    
    Investment labels:
    - accumulation: Smart money accumulating, price rising
    - distribution: Smart money distributing, price falling
    - reversal_to_bull/bear: Major trend reversal (DIRECTIONAL)
    - large_drawdown: Major correction occurred
    """
    # Get market-specific thresholds
    mode_labels = get_mode_labels(market_type)
    mode_config = mode_labels.get(mode, mode_labels['daytrade'])
    labels = mode_config['labels']
    lookahead = mode_config['lookahead']
    
    # Use custom thresholds if provided (auto-tuned), otherwise use defaults
    if custom_thresholds:
        thresholds = custom_thresholds
        print(f"   🎯 Using AUTO-TUNED thresholds: {thresholds}")
    else:
        thresholds = mode_config['thresholds']
        print(f"   📋 Using DEFAULT thresholds: {thresholds}")
    
    # Initialize label columns
    for label in labels:
        df[f'label_{label}'] = 0
    
    for i in range(len(df) - lookahead):
        current_price = df['Close'].iloc[i]
        current_atr = df['ATR'].iloc[i] if 'ATR' in df.columns else current_price * 0.02
        
        # Previous price for trend detection
        prev_price = df['Close'].iloc[i-1] if i > 0 else current_price
        was_going_up = current_price > prev_price
        was_going_down = current_price < prev_price
        
        # Future prices
        future_highs = df['High'].iloc[i+1:i+lookahead+1].values
        future_lows = df['Low'].iloc[i+1:i+lookahead+1].values
        future_closes = df['Close'].iloc[i+1:i+lookahead+1].values
        
        # Calculate outcomes
        max_up = ((future_highs.max() - current_price) / current_price) * 100
        max_down = ((current_price - future_lows.min()) / current_price) * 100
        final_return = ((future_closes[-1] - current_price) / current_price) * 100
        
        # Recent return: How price moved in the last few bars BEFORE this point
        # Used for reversal detection (was price going down/up before?)
        recent_lookback = min(5, i)  # Look back 5 bars or less if at start
        if recent_lookback > 0:
            recent_price = df['Close'].iloc[i - recent_lookback]
            recent_return = ((current_price - recent_price) / recent_price) * 100
        else:
            recent_return = 0
        
        # Calculate volatility expansion
        future_ranges = future_highs - future_lows
        avg_future_range = future_ranges.mean()
        vol_expansion_ratio = avg_future_range / current_atr if current_atr > 0 else 1
        
        # Detect initial move direction (first 2 candles)
        initial_move = ((future_closes[1] - current_price) / current_price) * 100 if len(future_closes) > 1 else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # DAY TRADE / SCALP LABELS (Directional)
        # ═══════════════════════════════════════════════════════════════════════
        if mode in ['daytrade', 'scalp']:
            # Bullish Continuation: Price moved UP meaningfully
            if final_return > thresholds['continuation']:
                df.loc[df.index[i], 'label_continuation_bull'] = 1
            
            # Bearish Continuation: Price moved DOWN meaningfully
            if final_return < -thresholds['continuation']:
                df.loc[df.index[i], 'label_continuation_bear'] = 1
            
            # DIRECTIONAL Fakeout: Initial move one way, then reversed
            # Fakeout to BULL: Faked down (max_down) then ended UP
            if max_down > thresholds['fakeout'] and final_return > thresholds['fakeout'] * 0.5:
                df.loc[df.index[i], 'label_fakeout_to_bull'] = 1
            
            # Fakeout to BEAR: Faked up (max_up) then ended DOWN
            if max_up > thresholds['fakeout'] and final_return < -thresholds['fakeout'] * 0.5:
                df.loc[df.index[i], 'label_fakeout_to_bear'] = 1
            
            # Vol expansion: Significant volatility increase
            if vol_expansion_ratio > thresholds['vol_expansion']:
                df.loc[df.index[i], 'label_vol_expansion'] = 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # SWING LABELS - DIFFERENT FOR CRYPTO VS STOCK!
        # Crypto: Directional (trend_holds_bull, etc.) - whales give us edge
        # Stock: Quality + Mean Reversion - stocks behave differently
        # ═══════════════════════════════════════════════════════════════════════
        elif mode == 'swing':
            is_stock = market_type.lower() in ['stock', 'stocks', 'etf', 'etfs']
            
            if is_stock:
                # ═══════════════════════════════════════════════════════════════
                # STOCK SWING: QUALITY + MEAN REVERSION LABELS
                # Key insight: Stocks mean-revert. Predict quality not direction.
                # ═══════════════════════════════════════════════════════════════
                tp_pct = thresholds.get('quality_tp_pct', 2.0)
                sl_pct = thresholds.get('quality_sl_pct', 1.0)
                mean_rev_z = thresholds.get('mean_rev_threshold', 2.0)
                vol_exp_ratio = thresholds.get('vol_expansion_ratio', 1.5)
                
                # --- SETUP QUALITY LABELS ---
                # Check which hit first: TP or SL
                tp_hit_bar = None
                sl_hit_bar = None
                
                for j in range(len(future_closes)):
                    if tp_hit_bar is None and future_highs[j] >= current_price * (1 + tp_pct/100):
                        tp_hit_bar = j
                    if sl_hit_bar is None and future_lows[j] <= current_price * (1 - sl_pct/100):
                        sl_hit_bar = j
                
                # Quality HIGH: TP hit before SL (good entry)
                if tp_hit_bar is not None:
                    if sl_hit_bar is None or tp_hit_bar <= sl_hit_bar:
                        df.loc[df.index[i], 'label_setup_quality_high'] = 1
                
                # Quality LOW: SL hit before TP (bad entry)
                if sl_hit_bar is not None:
                    if tp_hit_bar is None or sl_hit_bar < tp_hit_bar:
                        df.loc[df.index[i], 'label_setup_quality_low'] = 1
                
                # --- MEAN REVERSION LABELS (MULTI-PATH - Improved F1) ---
                # Multiple paths: Z-score, RSI, SMA deviation. Target: 8-15% positive rate
                if i >= 20:
                    sma20 = df['Close'].iloc[i-19:i+1].mean()
                    std20 = df['Close'].iloc[i-19:i+1].std()
                    z_score = (current_price - sma20) / std20 if std20 > 0 else 0
                    
                    # Get RSI and thresholds
                    rsi = df['RSI'].iloc[i] if 'RSI' in df.columns else 50
                    rsi_oversold = thresholds.get('rsi_oversold', 35)
                    rsi_overbought = thresholds.get('rsi_overbought', 65)
                    sma_dev_pct = thresholds.get('sma_deviation_pct', 3.0)
                    
                    # MEAN REVERSION LONG: Multiple qualifying paths
                    is_mean_rev_long = False
                    if z_score < -mean_rev_z * 0.7 and final_return > tp_pct * 0.3:
                        is_mean_rev_long = True
                    if rsi < rsi_oversold and final_return > tp_pct * 0.25:
                        is_mean_rev_long = True
                    if z_score < -1.0 and final_return > tp_pct * 0.7:
                        is_mean_rev_long = True
                    price_below_sma = (sma20 - current_price) / sma20 * 100 if sma20 > 0 else 0
                    if price_below_sma > sma_dev_pct and final_return > tp_pct * 0.4:
                        is_mean_rev_long = True
                    if is_mean_rev_long:
                        df.loc[df.index[i], 'label_mean_reversion_long'] = 1
                    
                    # MEAN REVERSION SHORT: Multiple qualifying paths
                    is_mean_rev_short = False
                    if z_score > mean_rev_z * 0.7 and final_return < -tp_pct * 0.3:
                        is_mean_rev_short = True
                    if rsi > rsi_overbought and final_return < -tp_pct * 0.25:
                        is_mean_rev_short = True
                    if z_score > 1.0 and final_return < -tp_pct * 0.7:
                        is_mean_rev_short = True
                    price_above_sma = (current_price - sma20) / sma20 * 100 if sma20 > 0 else 0
                    if price_above_sma > sma_dev_pct and final_return < -tp_pct * 0.4:
                        is_mean_rev_short = True
                    if is_mean_rev_short:
                        df.loc[df.index[i], 'label_mean_reversion_short'] = 1
                
                # --- BREAKOUT VALIDITY LABEL ---
                if i >= 20:
                    prev_high = df['High'].iloc[i-20:i].max()
                    current_high = df['High'].iloc[i]
                    is_breakout = current_high > prev_high
                    
                    if is_breakout:
                        # Did breakout HOLD?
                        bars_above = sum(1 for c in future_closes if c > prev_high)
                        if bars_above >= lookahead * 0.7:  # Held for 70% of bars
                            df.loc[df.index[i], 'label_breakout_valid'] = 1
                
                # --- VOLATILITY EXPANSION LABEL ---
                if 'ATR' in df.columns and i + lookahead < len(df):
                    current_atr = df['ATR'].iloc[i]
                    future_atr_avg = df['ATR'].iloc[i+1:i+lookahead+1].mean()
                    if future_atr_avg > current_atr * vol_exp_ratio:
                        df.loc[df.index[i], 'label_volatility_expansion'] = 1
            
            else:
                # ═══════════════════════════════════════════════════════════════
                # CRYPTO SWING: DIRECTIONAL LABELS (original - works well 72% F1)
                # ═══════════════════════════════════════════════════════════════
                trend_thresh = thresholds.get('trend_holds', 3.0)
                rev_thresh = thresholds.get('reversal', 2.0)
                dd_thresh = thresholds.get('drawdown', 2.0)
                
                # Bullish trend holds
                if final_return > trend_thresh:
                    df.loc[df.index[i], 'label_trend_holds_bull'] = 1
                
                # Bearish trend holds
                if final_return < -trend_thresh:
                    df.loc[df.index[i], 'label_trend_holds_bear'] = 1
                
                # Reversal to bull
                if (was_going_down or max_down > max_up * 0.5) and final_return > rev_thresh:
                    df.loc[df.index[i], 'label_reversal_to_bull'] = 1
                
                # Reversal to bear
                if (was_going_up or max_up > max_down * 0.5) and final_return < -rev_thresh:
                    df.loc[df.index[i], 'label_reversal_to_bear'] = 1
                
                # Drawdown
                if max_down > dd_thresh:
                    df.loc[df.index[i], 'label_drawdown'] = 1
        
        # ═══════════════════════════════════════════════════════════════════════
        # INVESTMENT LABELS (Enhanced Wyckoff-based Accumulation)
        # ═══════════════════════════════════════════════════════════════════════
        elif mode == 'investment':
            # ═══════════════════════════════════════════════════════════════
            # ACCUMULATION: VERY RELAXED for better class balance
            # Target: 8-15% positive rate
            # ═══════════════════════════════════════════════════════════════
            
            # 1. Calculate position in range (last 50 bars)
            lookback = min(50, i)
            if lookback > 10:
                range_high = df['High'].iloc[i-lookback:i+1].max()
                range_low = df['Low'].iloc[i-lookback:i+1].min()
                position_in_range = (current_price - range_low) / (range_high - range_low + 1e-10) * 100
            else:
                position_in_range = 50
            
            # ═══════════════════════════════════════════════════════════════
            # ACCUMULATION - SIMPLE: Goes up significantly = accumulation
            # ═══════════════════════════════════════════════════════════════
            goes_up = final_return > thresholds['accumulation']
            moderate_up = final_return > thresholds['accumulation'] * 0.7  # Even smaller moves count
            at_lows = position_in_range < 60  # VERY RELAXED: Bottom 60%
            
            # Path 1: Any significant up move from lower half
            if goes_up and at_lows:
                df.loc[df.index[i], 'label_accumulation'] = 1
            
            # Path 2: Moderate up move from bottom third
            elif moderate_up and position_in_range < 40:
                df.loc[df.index[i], 'label_accumulation'] = 1
            
            # Path 3: Strong up move from anywhere (momentum)
            elif final_return > thresholds['accumulation'] * 1.3:
                df.loc[df.index[i], 'label_accumulation'] = 1
            
            # ═══════════════════════════════════════════════════════════════
            # DISTRIBUTION - SIMPLE: Goes down significantly = distribution
            # ═══════════════════════════════════════════════════════════════
            goes_down = final_return < -thresholds['distribution']
            moderate_down = final_return < -thresholds['distribution'] * 0.7
            at_highs = position_in_range > 40  # VERY RELAXED: Top 60%
            
            # Path 1: Any significant down move from upper half
            if goes_down and at_highs:
                df.loc[df.index[i], 'label_distribution'] = 1
            
            # Path 2: Moderate down move from top third
            elif moderate_down and position_in_range > 60:
                df.loc[df.index[i], 'label_distribution'] = 1
            
            # Path 3: Strong down move from anywhere
            elif final_return < -thresholds['distribution'] * 1.3:
                df.loc[df.index[i], 'label_distribution'] = 1
            
            # ═══════════════════════════════════════════════════════════════
            # REVERSAL TO BULL - More permissive
            # ═══════════════════════════════════════════════════════════════
            rev_thresh = thresholds.get('reversal', 2.0)
            
            # Path 1: Was going down, reversed up
            if was_going_down and final_return > rev_thresh:
                df.loc[df.index[i], 'label_reversal_to_bull'] = 1
            
            # Path 2: At lows and goes up (even without prior downtrend)
            elif position_in_range < 35 and final_return > rev_thresh:
                df.loc[df.index[i], 'label_reversal_to_bull'] = 1
            
            # Path 3: Strong up after any down move
            elif recent_return < -1 and final_return > rev_thresh * 0.8:
                df.loc[df.index[i], 'label_reversal_to_bull'] = 1
            
            # ═══════════════════════════════════════════════════════════════
            # REVERSAL TO BEAR - More permissive
            # ═══════════════════════════════════════════════════════════════
            # Path 1: Was going up, reversed down
            if was_going_up and final_return < -rev_thresh:
                df.loc[df.index[i], 'label_reversal_to_bear'] = 1
            
            # Path 2: At highs and goes down (even without prior uptrend)
            elif position_in_range > 65 and final_return < -rev_thresh:
                df.loc[df.index[i], 'label_reversal_to_bear'] = 1
            
            # Path 3: Strong down after any up move
            elif recent_return > 1 and final_return < -rev_thresh * 0.8:
                df.loc[df.index[i], 'label_reversal_to_bear'] = 1
            
            # ═══════════════════════════════════════════════════════════════
            # LARGE DRAWDOWN - More sensitive
            # ═══════════════════════════════════════════════════════════════
            # Path 1: Max drawdown exceeds threshold
            if max_down > thresholds['large_drawdown']:
                df.loc[df.index[i], 'label_large_drawdown'] = 1
            
            # Path 2: Final return is very negative
            elif final_return < -thresholds['large_drawdown'] * 0.7:
                df.loc[df.index[i], 'label_large_drawdown'] = 1
            
            # Path 3: Significant volatility with downside
            elif max_down > thresholds['large_drawdown'] * 0.6 and final_return < 0:
                df.loc[df.index[i], 'label_large_drawdown'] = 1
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET PHASE LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_phase_labels(
    df: pd.DataFrame,
    lookahead: int = 12,
    market_type: str = 'crypto',
) -> pd.DataFrame:
    """
    Generate market phase labels based on price action and volume patterns.
    
    PHASE DETECTION LOGIC:
    - ACCUMULATION: Price at lows + volume declining + OI building
    - RE_ACCUMULATION: Pullback in uptrend + holding support
    - MARKUP: Rising prices + increasing volume + bullish structure
    - DISTRIBUTION: Price at highs + volume spikes + OI declining
    - PROFIT_TAKING: Moderate selling + no panic
    - MARKDOWN: Falling prices + increasing volume + bearish structure
    - CAPITULATION: Sharp drop + volume spike + oversold conditions
    
    Args:
        df: DataFrame with OHLCV and indicator data
        lookahead: How many bars ahead to verify phase transition
        market_type: 'crypto' or 'stock'
        
    Returns:
        DataFrame with 'label_phase' column (0-6 encoded)
    """
    df = df.copy()
    
    # Initialize phase column
    df['label_phase'] = 3  # Default to DISTRIBUTION (neutral-ish)
    
    # Need certain columns
    if 'Close' not in df.columns:
        return df
    
    # Calculate required metrics if not present
    if 'RSI' not in df.columns and len(df) > 14:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
    
    if 'ATR' not in df.columns and len(df) > 14:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
    
    # Calculate position in 50-bar range
    df['range_high'] = df['High'].rolling(50).max()
    df['range_low'] = df['Low'].rolling(50).min()
    df['position_in_range'] = (df['Close'] - df['range_low']) / (df['range_high'] - df['range_low'] + 1e-10) * 100
    
    # Calculate trend (20-bar SMA slope)
    df['sma20'] = df['Close'].rolling(20).mean()
    df['trend'] = df['sma20'].diff(5)  # 5-bar slope
    
    # Calculate volume trend
    if 'Volume' in df.columns:
        df['vol_sma'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_sma'].replace(0, 1)
    else:
        df['vol_ratio'] = 1.0
    
    # Calculate momentum
    df['momentum'] = df['Close'].pct_change(10) * 100  # 10-bar momentum
    
    for i in range(50, len(df) - lookahead):
        try:
            pos = df['position_in_range'].iloc[i]
            trend = df['trend'].iloc[i] if pd.notna(df['trend'].iloc[i]) else 0
            vol_ratio = df['vol_ratio'].iloc[i] if pd.notna(df['vol_ratio'].iloc[i]) else 1
            momentum = df['momentum'].iloc[i] if pd.notna(df['momentum'].iloc[i]) else 0
            rsi = df['RSI'].iloc[i] if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[i]) else 50
            
            # Future price for validation
            future_return = ((df['Close'].iloc[i + lookahead] - df['Close'].iloc[i]) / df['Close'].iloc[i]) * 100
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE CLASSIFICATION LOGIC
            # ═══════════════════════════════════════════════════════════════
            
            phase = 3  # Default: DISTRIBUTION
            
            # CAPITULATION (6): Panic selling at lows
            # - Position < 20% (at lows)
            # - RSI < 30 (oversold)
            # - Volume spike > 1.5x
            # - Sharp negative momentum
            if pos < 20 and rsi < 30 and vol_ratio > 1.5 and momentum < -5:
                phase = 6  # CAPITULATION
            
            # ACCUMULATION (0): Smart money buying at lows
            # - Position < 35% (low in range)
            # - Flat or declining trend (consolidation)
            # - Future goes up (validation)
            elif pos < 35 and abs(trend) < df['ATR'].iloc[i] * 0.1 and future_return > 2:
                phase = 0  # ACCUMULATION
            
            # RE_ACCUMULATION (1): Pullback in uptrend
            # - Position 35-65% (middle, pulled back)
            # - Overall trend still up
            # - Future goes up (validation)
            elif 35 <= pos <= 65 and trend > 0 and momentum < 0 and future_return > 1:
                phase = 1  # RE_ACCUMULATION
            
            # MARKUP (2): Active uptrend
            # - Position > 40% (not at lows)
            # - Positive trend
            # - Positive momentum
            # - RSI > 50
            elif pos > 40 and trend > 0 and momentum > 0 and rsi > 50:
                phase = 2  # MARKUP
            
            # PROFIT_TAKING (4): Some selling, not panic
            # - Position > 60% (high in range)
            # - Slowing momentum
            # - Volume normal
            elif pos > 60 and momentum < 0 and abs(momentum) < 5 and vol_ratio < 1.5:
                phase = 4  # PROFIT_TAKING
            
            # DISTRIBUTION (3): Smart money selling at highs
            # - Position > 70% (near highs)
            # - Flat or declining trend starting
            # - Future goes down (validation)
            elif pos > 70 and trend < df['ATR'].iloc[i] * 0.05 and future_return < -2:
                phase = 3  # DISTRIBUTION
            
            # MARKDOWN (5): Active downtrend
            # - Negative trend
            # - Negative momentum
            # - RSI < 50
            elif trend < 0 and momentum < 0 and rsi < 50:
                phase = 5  # MARKDOWN
            
            df.loc[df.index[i], 'label_phase'] = phase
            
        except Exception as e:
            continue
    
    # Clean up temporary columns
    for col in ['range_high', 'range_low', 'position_in_range', 'sma20', 'trend', 'vol_sma', 'vol_ratio', 'momentum']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return df


def detect_current_phase(
    df: pd.DataFrame,
    idx: int = -1,
    whale_data: Dict = None,
    mf_data: Dict = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Detect the current market phase from price data and indicators.
    
    Returns:
        Tuple of (phase_name, confidence_scores_per_phase)
    """
    if idx < 0:
        idx = len(df) + idx
    
    if idx < 50 or idx >= len(df):
        return 'UNKNOWN', {}
    
    try:
        close = df['Close'].iloc[idx]
        
        # Calculate position in range
        high_50 = df['High'].iloc[max(0, idx-50):idx+1].max()
        low_50 = df['Low'].iloc[max(0, idx-50):idx+1].min()
        pos = (close - low_50) / (high_50 - low_50 + 1e-10) * 100
        
        # Get RSI if available
        rsi = df['RSI'].iloc[idx] if 'RSI' in df.columns else 50
        
        # Get momentum (10-bar return)
        momentum = ((close - df['Close'].iloc[idx-10]) / df['Close'].iloc[idx-10]) * 100 if idx >= 10 else 0
        
        # Get trend (20-bar SMA slope)
        sma20 = df['Close'].iloc[max(0, idx-20):idx+1].mean()
        prev_sma20 = df['Close'].iloc[max(0, idx-25):idx-5+1].mean() if idx >= 25 else sma20
        trend = sma20 - prev_sma20
        
        # Get whale data if available
        whale_pct = whale_data.get('whale_long_pct', 50) if whale_data else 50
        retail_pct = whale_data.get('retail_long_pct', 50) if whale_data else 50
        oi_change = whale_data.get('oi_change', 0) if whale_data else 0
        
        # Get money flow phase if available
        mf_phase = mf_data.get('phase', 'UNKNOWN') if mf_data else 'UNKNOWN'
        
        # Calculate confidence for each phase
        scores = {phase: 0.0 for phase in PHASE_LABELS}
        
        # ═══════════════════════════════════════════════════════════════
        # SCORE EACH PHASE BASED ON CONDITIONS
        # ═══════════════════════════════════════════════════════════════
        
        # ACCUMULATION: At lows, whales accumulating
        if pos < 35:
            scores['ACCUMULATION'] += 30
        if whale_pct > retail_pct:
            scores['ACCUMULATION'] += 20
        if oi_change > 0 and momentum < 0:
            scores['ACCUMULATION'] += 15
        if rsi < 40:
            scores['ACCUMULATION'] += 10
        if mf_phase in ['ACCUMULATION', 'RE-ACCUMULATION']:
            scores['ACCUMULATION'] += 25
        
        # RE_ACCUMULATION: Pullback in uptrend
        if 35 <= pos <= 65:
            scores['RE_ACCUMULATION'] += 20
        if trend > 0 and momentum < 0:
            scores['RE_ACCUMULATION'] += 30
        if whale_pct > 55:
            scores['RE_ACCUMULATION'] += 15
        if mf_phase == 'RE-ACCUMULATION':
            scores['RE_ACCUMULATION'] += 35
        
        # MARKUP: Trending up
        if pos > 40:
            scores['MARKUP'] += 15
        if trend > 0 and momentum > 0:
            scores['MARKUP'] += 35
        if rsi > 50:
            scores['MARKUP'] += 15
        if mf_phase == 'MARKUP':
            scores['MARKUP'] += 35
        
        # DISTRIBUTION: At highs, whales selling
        if pos > 70:
            scores['DISTRIBUTION'] += 30
        if whale_pct < retail_pct:
            scores['DISTRIBUTION'] += 25
        if oi_change < 0 and momentum > 0:
            scores['DISTRIBUTION'] += 15
        if mf_phase in ['DISTRIBUTION', 'FOMO / DIST RISK']:
            scores['DISTRIBUTION'] += 30
        
        # PROFIT_TAKING: Some selling
        if pos > 60:
            scores['PROFIT_TAKING'] += 20
        if momentum < 0 and abs(momentum) < 5:
            scores['PROFIT_TAKING'] += 25
        if mf_phase == 'PROFIT TAKING':
            scores['PROFIT_TAKING'] += 35
        
        # MARKDOWN: Trending down
        if trend < 0 and momentum < 0:
            scores['MARKDOWN'] += 35
        if rsi < 50:
            scores['MARKDOWN'] += 15
        if pos < 60:
            scores['MARKDOWN'] += 10
        if mf_phase == 'MARKDOWN':
            scores['MARKDOWN'] += 35
        
        # CAPITULATION: Panic at lows
        if pos < 20:
            scores['CAPITULATION'] += 25
        if rsi < 30:
            scores['CAPITULATION'] += 25
        if momentum < -5:
            scores['CAPITULATION'] += 20
        if mf_phase == 'CAPITULATION':
            scores['CAPITULATION'] += 30
        
        # Normalize scores to probabilities
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            scores = {k: 1/7 for k in scores}  # Uniform if no signal
        
        # Get phase with highest score
        current_phase = max(scores, key=scores.get)
        
        return current_phase, scores
        
    except Exception as e:
        return 'UNKNOWN', {}


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILISTIC ML TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class ProbabilisticMLTrainer:
    """
    Trains probabilistic ML models for each trading mode.
    
    Each model outputs probabilities for mode-specific outcomes:
    - Day Trade: P_continuation, P_fakeout, P_vol_expansion
    - Swing: P_trend_holds, P_reversal, P_drawdown
    - Investment: P_accumulation, P_distribution, P_large_drawdown
    """
    
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    
    def __init__(self):
        self.models = {}  # {mode: {'model': classifier, 'scaler': scaler, 'metadata': {}}}
        self._ensure_model_dir()
    
    def _ensure_model_dir(self):
        os.makedirs(self.MODEL_DIR, exist_ok=True)
    
    def train_mode(
        self,
        df: pd.DataFrame,
        mode: str,
        whale_data: Dict = None,
        smc_data: Dict = None,
        market_context: Dict = None,
        progress_callback=None,
        market_type: str = 'crypto',
        auto_tune: bool = True,
        target_positive_rate: float = 0.35,
    ) -> Dict:
        """
        Train probabilistic model for a specific mode.
        
        Args:
            market_type: 'crypto' or 'stock' - uses different thresholds!
            auto_tune: If True, automatically calculate optimal thresholds from data
            target_positive_rate: Target positive rate for auto-tuning (default 35%)
        
        Returns:
            Training metrics and model info
        """
        mode_labels = get_mode_labels(market_type)
        mode_config = mode_labels.get(mode, mode_labels['daytrade'])
        labels = mode_config['labels']
        
        # ═══════════════════════════════════════════════════════════════════════
        # AUTO-TUNE THRESHOLDS FROM DATA
        # ═══════════════════════════════════════════════════════════════════════
        custom_thresholds = None
        if auto_tune:
            if progress_callback:
                progress_callback(0.05, f"Auto-tuning thresholds for {mode} ({market_type})...")
            
            custom_thresholds = auto_tune_thresholds(
                df, mode, target_positive_rate, progress_callback, market_type
            )
            
            # Cache for later use (e.g., prediction targets)
            cache_key = f"{market_type}_{mode}"
            _AUTO_TUNED_THRESHOLDS[cache_key] = custom_thresholds
        
        if progress_callback:
            progress_callback(0.1, f"Generating {mode} labels ({market_type})...")
        
        # Generate labels with auto-tuned or default thresholds
        df = generate_probabilistic_labels(
            df, mode, market_type=market_type, custom_thresholds=custom_thresholds
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE PHASE LABELS (NEW!)
        # ═══════════════════════════════════════════════════════════════════════
        if progress_callback:
            progress_callback(0.15, "Generating market phase labels...")
        
        df = generate_phase_labels(df, lookahead=mode_config['lookahead'], market_type=market_type)
        
        if progress_callback:
            progress_callback(0.3, "Extracting features...")
        
        # Extract features for all rows
        X = []
        y = []
        y_phase = []  # NEW: Phase labels
        
        valid_rows = len(df) - mode_config['lookahead']
        
        for i in range(valid_rows):
            # Look up whale data for this specific row if available
            row_whale_data = {}
            if whale_data and isinstance(whale_data, dict):
                # Check if it's a lookup dict keyed by (symbol, idx)
                symbol = df['symbol'].iloc[i] if 'symbol' in df.columns else ''
                lookup_key = (symbol, str(df.index[i]))
                if lookup_key in whale_data:
                    row_whale_data = whale_data[lookup_key]
                elif symbol in whale_data:
                    # Direct symbol lookup
                    row_whale_data = whale_data[symbol]
                else:
                    # Use as-is (single dict for all rows)
                    row_whale_data = whale_data
            
            features = extract_enhanced_features(
                df, i, row_whale_data, smc_data, market_context, market_type
            )
            X.append(features)
            
            # Multi-label target
            label_values = [df[f'label_{label}'].iloc[i] for label in labels]
            y.append(label_values)
            
            # NEW: Extract phase label
            phase_label = int(df['label_phase'].iloc[i]) if 'label_phase' in df.columns else 3
            y_phase.append(phase_label)
        
        X = np.array(X)
        y = np.array(y)
        y_phase = np.array(y_phase)  # NEW: Phase labels array
        
        # Handle NaN and inf values
        # Replace inf with nan, then fill nan with column means
        X = np.where(np.isinf(X), np.nan, X)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)  # If all nan, use 0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
        if progress_callback:
            progress_callback(0.5, f"Training {mode} model...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split (random split works better than walk-forward for mixed market conditions)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # NEW: Split phase labels with same indices
        _, _, y_phase_train, y_phase_test = train_test_split(
            X_scaled, y_phase, test_size=0.2, random_state=42
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # SMOTE OVERSAMPLING FOR RARE CLASSES (Improves F1 for mean_reversion)
        # ═══════════════════════════════════════════════════════════════════════
        X_train_balanced = {}
        y_train_balanced = {}
        
        if HAS_IMBLEARN:
            for i, label in enumerate(labels):
                pos_rate = y_train[:, i].mean()
                if 0.005 < pos_rate < 0.10:  # Rare classes only
                    try:
                        n_pos = int(y_train[:, i].sum())
                        k = min(5, n_pos - 1)
                        if k >= 1:
                            smote = BorderlineSMOTE(
                                sampling_strategy=min(0.20, pos_rate * 8),
                                random_state=42, k_neighbors=k
                            )
                            X_res, y_res = smote.fit_resample(X_train, y_train[:, i])
                            X_train_balanced[label] = X_res
                            y_train_balanced[label] = y_res
                            if progress_callback:
                                progress_callback(0.52, f"SMOTE {label}: {pos_rate:.1%} → {y_res.mean():.1%}")
                    except Exception as e:
                        print(f"SMOTE failed for {label}: {e}")
        
        # Train multi-output classifier
        base_clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Import ensemble models
        try:
            from .ensemble_models import train_all_models, get_best_models_per_label, HAS_XGBOOST, HAS_LIGHTGBM, HAS_CATBOOST
        except ImportError:
            try:
                from ensemble_models import train_all_models, get_best_models_per_label, HAS_XGBOOST, HAS_LIGHTGBM, HAS_CATBOOST
            except ImportError:
                # Fallback - no ensemble
                HAS_XGBOOST = False
                HAS_LIGHTGBM = False
                HAS_CATBOOST = False
                train_all_models = None
        
        # Train ALL models for each label and pick best
        models_per_label = {}
        metrics_per_label = {}
        all_model_results = {}  # Store results from ALL models
        
        for i, label in enumerate(labels):
            # ═══════════════════════════════════════════════════════════════════════
            # DYNAMIC CLASS WEIGHTING - Key for improving F1 on imbalanced data
            # This forces the model to be more selective (better precision)
            # ═══════════════════════════════════════════════════════════════════════
            pos_count = y_train[:, i].sum()
            neg_count = len(y_train) - pos_count
            
            if pos_count > 0:
                # Formula: weight positive class to balance with negative
                # Cap at 10.0 to prevent extreme weights
                scale_pos_weight = min(len(y_train) / (2 * pos_count), 10.0)
            else:
                scale_pos_weight = 1.0
            
            # Use SMOTE-balanced data if available for this label
            if label in X_train_balanced:
                X_train_label = X_train_balanced[label]
                y_train_label = y_train_balanced[label]
                effective_scale = min(scale_pos_weight, 3.0)  # Lower cap after SMOTE
            else:
                X_train_label = X_train
                y_train_label = y_train[:, i]
                effective_scale = scale_pos_weight
            
            # Create sample weights array
            sample_weights = np.where(y_train_label == 1, effective_scale, 1.0)
            
            if progress_callback:
                base_progress = 0.5 + (0.4 * i / len(labels))
                progress_callback(base_progress, f"Training 32 models for '{label}' ({i+1}/{len(labels)})...")
            
            # Create inner progress callback that shows model-by-model progress
            def model_progress(model_pct, model_msg):
                if progress_callback:
                    # Allocate progress: 50% base + 40% for labels * (label_position + model_progress within label)
                    label_progress = 0.4 * i / len(labels)
                    model_progress_within = 0.4 / len(labels) * model_pct
                    total = 0.5 + label_progress + model_progress_within
                    progress_callback(total, f"[{label}] {model_msg}")
            
            # Train ALL models and get results (if ensemble available)
            # Try with sample_weight first, fallback without if not supported
            if train_all_models is not None:
                try:
                    # Try with sample_weight (new feature for better F1)
                    label_results = train_all_models(
                        X_train_label, y_train_label,
                        X_test, y_test[:, i],
                        label_name=label,
                        progress_callback=model_progress,
                        sample_weight=sample_weights
                    )
                except TypeError as e:
                    # Fallback: ensemble_models doesn't support sample_weight yet
                    if 'sample_weight' in str(e):
                        print(f"[ML_TRAIN] Note: ensemble_models doesn't support sample_weight yet, training without it")
                        label_results = train_all_models(
                            X_train_label, y_train_label,
                            X_test, y_test[:, i],
                            label_name=label,
                            progress_callback=model_progress
                        )
                    else:
                        raise e
            else:
                label_results = {}  # Fallback to single model
            
            all_model_results[label] = label_results
            
            # Get best model
            if label_results:
                best_name = list(label_results.keys())[0]  # Already sorted by F1
                best_data = label_results[best_name]
                
                models_per_label[label] = best_data['model']
                metrics_per_label[label] = {
                    'accuracy': best_data['accuracy'],
                    'precision': best_data['precision'],
                    'recall': best_data['recall'],
                    'f1': best_data['f1'],
                    'positive_rate': y_train[:, i].mean(),
                    'test_positive_rate': y_test[:, i].mean(),
                    'n_samples': len(y_train),
                    'n_test': len(y_test),
                    'best_model': best_name,
                    'all_models': {k: {'f1': v['f1'], 'precision': v['precision'], 'recall': v['recall']} 
                                   for k, v in list(label_results.items())[:10]}  # Top 10
                }
            else:
                # Fallback to GradientBoosting
                sample_weights = np.where(y_train[:, i] == 1, scale_pos_weight, 1.0)
                clf = GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    min_samples_leaf=20,
                    subsample=0.8,
                    random_state=42
                )
                clf.fit(X_train, y_train[:, i], sample_weight=sample_weights)
                
                y_pred = clf.predict(X_test)
                y_true = y_test[:, i]
                
                models_per_label[label] = clf
                metrics_per_label[label] = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0),
                    'positive_rate': y_train[:, i].mean(),
                    'test_positive_rate': y_true.mean(),
                    'n_samples': len(y_train),
                    'n_test': len(y_test),
                    'best_model': 'GradientBoosting (fallback)',
                    'all_models': {}
                }
            
            if progress_callback:
                best = metrics_per_label[label].get('best_model', 'Unknown')
                f1 = metrics_per_label[label]['f1']
                progress_callback(0.5 + 0.4 * (i + 1) / len(labels), f"{label}: {best} F1={f1:.1%}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # TRAIN PHASE CLASSIFIER (NEW!)
        # Multi-class classifier for 7 Wyckoff phases
        # ═══════════════════════════════════════════════════════════════════════
        if progress_callback:
            progress_callback(0.92, "Training phase classifier...")
        
        phase_model = None
        phase_metrics = {}
        
        try:
            # Train phase classifier (multi-class, not multi-label)
            phase_clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            phase_clf.fit(X_train, y_phase_train)
            
            # Evaluate
            y_phase_pred = phase_clf.predict(X_test)
            
            phase_metrics = {
                'accuracy': accuracy_score(y_phase_test, y_phase_pred),
                'f1_macro': f1_score(y_phase_test, y_phase_pred, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_phase_test, y_phase_pred, average='weighted', zero_division=0),
                'n_classes': len(PHASE_LABELS),
            }
            
            phase_model = phase_clf
            
            if progress_callback:
                progress_callback(0.95, f"Phase classifier: F1={phase_metrics['f1_macro']:.1%}")
                
        except Exception as e:
            print(f"Phase classifier training failed: {e}")
            phase_metrics = {'error': str(e)}
        
        # Store model bundle
        model_bundle = {
            'models': models_per_label,
            'phase_model': phase_model,  # NEW: Phase classifier
            'scaler': scaler,
            'labels': labels,
            'mode': mode,
            'market_type': market_type,
            'thresholds': custom_thresholds,  # Save auto-tuned thresholds!
            'metadata': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'metrics': metrics_per_label,
                'phase_metrics': phase_metrics,  # NEW: Phase metrics
                'avg_f1': sum(m.get('f1', 0) for m in metrics_per_label.values()) / len(metrics_per_label) if metrics_per_label else 0,
                'trained_at': datetime.now().isoformat(),
                'thresholds': custom_thresholds,  # Also in metadata for easy access
                'auto_tuned': auto_tune,
            }
        }
        
        # Store with market_type key to keep crypto/stock separate in memory
        model_key = f"{mode}_{market_type}"
        self.models[model_key] = model_bundle
        
        # Save model with market_type in filename (crypto vs stock)
        model_path = os.path.join(self.MODEL_DIR, f'prob_model_{mode}_{market_type}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        # Also save metrics summary as JSON for quick display
        # Extract best model names from per-label metrics
        best_models = {}
        for label, label_metrics in metrics_per_label.items():
            best_models[label] = label_metrics.get('best_model', 'Unknown')
        
        metrics_summary = {
            'mode': mode,
            'market_type': market_type,
            'n_samples': len(X),
            'avg_f1': model_bundle['metadata']['avg_f1'],
            'labels': labels,
            'per_label': metrics_per_label,  # Per-label breakdown
            'best_models': best_models,  # Best model per label
            'thresholds': custom_thresholds,  # Include in JSON too!
            'trained_at': datetime.now().isoformat(),
        }
        metrics_path = os.path.join(self.MODEL_DIR, f'metrics_{mode}_{market_type}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        if progress_callback:
            progress_callback(1.0, f"Model saved: {model_path}")
        
        return {
            'mode': mode,
            'market_type': market_type,
            'metrics': metrics_per_label,
            'n_samples': len(X),
            'labels': labels,
        }
    
    def predict(
        self,
        df: pd.DataFrame,
        mode: str,
        idx: int = -1,
        whale_data: Dict = None,
        smc_data: Dict = None,
        market_context: Dict = None,
        market_type: str = 'crypto',
    ) -> ProbabilisticPrediction:
        """
        Get probabilistic prediction for a specific row.
        
        Args:
            market_type: 'crypto' or 'stock' - loads correct model!
        
        Returns:
            ProbabilisticPrediction with probabilities for each outcome
        """
        # Model key includes market_type to keep crypto/stock/etf separate
        model_key = f"{mode}_{market_type}"
        
        # DEBUG: Log what we're looking for
        print(f"[PROB_ML] predict() called: mode={mode}, market_type={market_type}, model_key={model_key}")
        print(f"[PROB_ML] MODEL_DIR: {self.MODEL_DIR}")
        
        # Load model if not in memory
        if model_key not in self.models:
            # Try market-specific model first
            model_path = os.path.join(self.MODEL_DIR, f'prob_model_{mode}_{market_type}.pkl')
            print(f"[PROB_ML] Trying: {model_path} -> exists={os.path.exists(model_path)}")
            
            # For ETF, also try 'etfs' variant (plural naming convention)
            if not os.path.exists(model_path) and market_type.lower() == 'etf':
                model_path = os.path.join(self.MODEL_DIR, f'prob_model_{mode}_etfs.pkl')
                print(f"[PROB_ML] Trying ETFs variant: {model_path} -> exists={os.path.exists(model_path)}")
            
            # Fallback to generic model (backward compatibility)
            if not os.path.exists(model_path):
                model_path = os.path.join(self.MODEL_DIR, f'prob_model_{mode}.pkl')
                print(f"[PROB_ML] Trying generic: {model_path} -> exists={os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                print(f"[PROB_ML] ✅ Loading model from: {model_path}")
                with open(model_path, 'rb') as f:
                    self.models[model_key] = pickle.load(f)
            else:
                # NO HEURISTIC - Return NOT_TRAINED so user knows to train
                print(f"[PROB_ML] ❌ No model found for {model_key}")
                return self._not_trained_response(mode)
        else:
            print(f"[PROB_ML] Model already in memory: {model_key}")
        
        model_bundle = self.models[model_key]
        models = model_bundle['models']
        scaler = model_bundle['scaler']
        labels = model_bundle['labels']
        
        # Handle negative index
        if idx < 0:
            idx = len(df) + idx
        
        # Extract features
        features = extract_enhanced_features(
            df, idx, whale_data, smc_data, market_context, market_type
        ).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        
        # Get probabilities for each label
        probabilities = {}
        for label in labels:
            clf = models[label]
            proba = clf.predict_proba(features_scaled)[0]
            # Get probability of positive class
            if len(proba) > 1:
                probabilities[label] = proba[1]
            else:
                probabilities[label] = proba[0]
        
        # Feature importances (average across models)
        feature_importances = np.zeros(NUM_ENHANCED_FEATURES)
        for label in labels:
            if hasattr(models[label], 'feature_importances_'):
                feature_importances += models[label].feature_importances_
        feature_importances /= len(labels)
        
        top_idx = np.argsort(feature_importances)[-5:][::-1]
        top_features = [(ENHANCED_FEATURES[i], float(features[0][i]), float(feature_importances[i])) 
                        for i in top_idx]
        
        # ═══════════════════════════════════════════════════════════════════════
        # RULES INTERPRET PROBABILITIES → DIRECTION
        # ═══════════════════════════════════════════════════════════════════════
        direction, confidence, reasoning = self._interpret_probabilities(
            probabilities, mode, whale_data
        )
        
        # Get thresholds - first try saved model, then cache, then defaults
        saved_thresholds = model_bundle.get('thresholds')
        cache_key = f"{market_type}_{mode}"
        cached_thresholds = _AUTO_TUNED_THRESHOLDS.get(cache_key)
        
        if saved_thresholds:
            thresholds = saved_thresholds
        elif cached_thresholds:
            thresholds = cached_thresholds
        else:
            thresholds = get_mode_labels(market_type).get(mode, {}).get('thresholds', {})
        
        # Calculate expected move and targets based on thresholds
        if mode in ['swing']:
            min_tp_pct = thresholds.get('trend_holds', 1.5)
            suggested_sl_pct = thresholds.get('drawdown', 1.0)
        elif mode in ['daytrade', 'scalp']:
            min_tp_pct = thresholds.get('continuation', 1.0)
            suggested_sl_pct = thresholds.get('fakeout', 0.6)
        elif mode == 'investment':
            min_tp_pct = thresholds.get('accumulation', 5.0)
            suggested_sl_pct = thresholds.get('reversal', 4.0)
        else:
            min_tp_pct = 2.0
            suggested_sl_pct = 1.0
        
        risk_reward = min_tp_pct / suggested_sl_pct if suggested_sl_pct > 0 else 2.0
        
        # Extract F1 scores from model metadata
        f1_scores = {}
        metadata = model_bundle.get('metadata', {})
        metrics = metadata.get('metrics', {})
        for label, m in metrics.items():
            f1_scores[label] = m.get('f1', 0.0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE PREDICTION (NEW!)
        # ═══════════════════════════════════════════════════════════════════════
        current_phase = 'UNKNOWN'
        phase_probabilities = {}
        next_phase = 'UNKNOWN'
        next_phase_confidence = 0.0
        phase_bias = 'NEUTRAL'
        
        try:
            phase_model = model_bundle.get('phase_model')
            
            if phase_model is not None:
                # Get phase probabilities from ML model
                phase_probs = phase_model.predict_proba(features.reshape(1, -1))[0]
                
                # Map to phase names
                for i, phase in enumerate(PHASE_LABELS):
                    if i < len(phase_probs):
                        phase_probabilities[phase] = float(phase_probs[i])
                
                # Get predicted phase (highest probability)
                phase_idx = phase_model.predict(features.reshape(1, -1))[0]
                if 0 <= phase_idx < len(PHASE_LABELS):
                    current_phase = PHASE_LABELS[phase_idx]
                    next_phase_confidence = phase_probabilities.get(current_phase, 0) * 100
                
                # Determine phase bias
                phase_bias = PHASE_BIAS.get(current_phase, 'NEUTRAL')
                
                # Predict next phase (from transitions)
                if current_phase in PHASE_TRANSITIONS:
                    possible_next = PHASE_TRANSITIONS[current_phase]
                    # Find the one with highest probability among transitions
                    next_probs = [(p, phase_probabilities.get(p, 0)) for p in possible_next]
                    if next_probs:
                        next_phase, _ = max(next_probs, key=lambda x: x[1])
            else:
                # Fallback: Use rules-based phase detection
                current_phase, phase_probabilities = detect_current_phase(
                    df, idx, whale_data, None
                )
                phase_bias = PHASE_BIAS.get(current_phase, 'NEUTRAL')
                if current_phase in PHASE_TRANSITIONS:
                    next_phase = PHASE_TRANSITIONS[current_phase][0]
                next_phase_confidence = phase_probabilities.get(current_phase, 0) * 100
                
        except Exception as e:
            # Silently handle phase prediction errors
            pass
        
        return ProbabilisticPrediction(
            mode=mode,
            probabilities=probabilities,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            top_features=top_features,
            model_version=f"probabilistic_{mode}_v2",  # Updated version
            prediction_time=datetime.now().isoformat(),
            expected_move_pct=min_tp_pct,
            min_tp_pct=min_tp_pct,
            suggested_sl_pct=suggested_sl_pct,
            risk_reward=risk_reward,
            f1_scores=f1_scores,
            # NEW: Phase prediction fields
            current_phase=current_phase,
            phase_probabilities=phase_probabilities,
            next_phase=next_phase,
            next_phase_confidence=next_phase_confidence,
            phase_bias=phase_bias,
        )
    
    def _interpret_probabilities(
        self,
        probabilities: Dict[str, float],
        mode: str,
        whale_data: Dict = None,
    ) -> Tuple[str, float, str]:
        """
        Rules layer: Interpret probabilities to make LONG/SHORT/WAIT decision.
        
        ML informs. Rules decide.
        
        IMPORTANT: Label keys are DIRECTIONAL:
        - scalp/daytrade: continuation_bull, continuation_bear, fakeout_to_bull, fakeout_to_bear
        - swing: trend_holds_bull, trend_holds_bear, reversal_to_bull, reversal_to_bear
        - investment: accumulation, distribution, reversal_to_bull, reversal_to_bear
        """
        whale_data = whale_data or {}
        whale_pct = whale_data.get('whale_long_pct', 50)
        retail_pct = whale_data.get('retail_long_pct', 50)
        divergence = whale_pct - retail_pct
        
        reasoning_parts = []
        
        if mode in ['daytrade', 'scalp']:
            # DIRECTIONAL labels - use bull vs bear comparison
            p_cont_bull = probabilities.get('continuation_bull', 0)
            p_cont_bear = probabilities.get('continuation_bear', 0)
            p_fake_bull = probabilities.get('fakeout_to_bull', 0)
            p_fake_bear = probabilities.get('fakeout_to_bear', 0)
            p_vol = probabilities.get('vol_expansion', 0)
            
            reasoning_parts.append(f"UP={p_cont_bull:.0%}")
            reasoning_parts.append(f"DOWN={p_cont_bear:.0%}")
            reasoning_parts.append(f"FakeUP={p_fake_bull:.0%}")
            reasoning_parts.append(f"FakeDN={p_fake_bear:.0%}")
            
            # Net direction from ML
            bull_score = p_cont_bull + p_fake_bull * 0.5  # Fakeout to bull is also bullish
            bear_score = p_cont_bear + p_fake_bear * 0.5
            
            # Decision rules
            if p_cont_bull > 0.15 and p_cont_bull > p_cont_bear and p_fake_bear < 0.10:
                # Strong continuation UP, low fakeout risk
                direction = 'LONG'
                confidence = min(95, p_cont_bull * 100 + (divergence / 3 if divergence > 0 else 0))
                reasoning_parts.append(f"ML says UP")
            elif p_cont_bear > 0.15 and p_cont_bear > p_cont_bull and p_fake_bull < 0.10:
                # Strong continuation DOWN, low fakeout risk
                direction = 'SHORT'
                confidence = min(95, p_cont_bear * 100 + (abs(divergence) / 3 if divergence < 0 else 0))
                reasoning_parts.append(f"ML says DOWN")
            elif p_fake_bull > 0.10 or p_fake_bear > 0.10:
                # Fakeout expected - be careful
                if p_fake_bull > p_fake_bear and p_fake_bull > 0.15:
                    direction = 'LONG'  # Will fake down then go UP
                    confidence = p_fake_bull * 80
                    reasoning_parts.append("Fakeout to BULL expected")
                elif p_fake_bear > p_fake_bull and p_fake_bear > 0.15:
                    direction = 'SHORT'  # Will fake up then go DOWN
                    confidence = p_fake_bear * 80
                    reasoning_parts.append("Fakeout to BEAR expected")
                else:
                    direction = 'WAIT'
                    confidence = 50
                    reasoning_parts.append("Fakeout risk - unclear direction")
            elif p_vol > 0.20 and (p_cont_bull > 0.10 or p_cont_bear > 0.10):
                # Volatility expansion with some direction
                direction = 'LONG' if bull_score > bear_score else 'SHORT' if bear_score > bull_score else 'WAIT'
                confidence = min(85, p_vol * 70 + max(p_cont_bull, p_cont_bear) * 30)
                reasoning_parts.append("Vol expansion expected")
            else:
                direction = 'WAIT'
                confidence = 40
                reasoning_parts.append("No clear ML signal")
        
        elif mode == 'swing':
            # Check if these are STOCK labels (quality-based) or CRYPTO labels (directional)
            has_stock_labels = 'setup_quality_high' in probabilities
            
            if has_stock_labels:
                # ═══════════════════════════════════════════════════════════════
                # STOCK SWING: Quality + Mean Reversion interpretation
                # ═══════════════════════════════════════════════════════════════
                p_quality_high = probabilities.get('setup_quality_high', 0)
                p_quality_low = probabilities.get('setup_quality_low', 0)
                p_mr_long = probabilities.get('mean_reversion_long', 0)
                p_mr_short = probabilities.get('mean_reversion_short', 0)
                p_breakout = probabilities.get('breakout_valid', 0)
                p_vol_exp = probabilities.get('volatility_expansion', 0)
                
                reasoning_parts.append(f"Quality={p_quality_high:.0%}")
                reasoning_parts.append(f"MR_Long={p_mr_long:.0%}")
                reasoning_parts.append(f"MR_Short={p_mr_short:.0%}")
                reasoning_parts.append(f"Breakout={p_breakout:.0%}")
                
                # Quality filter - avoid bad setups
                quality_ok = p_quality_high > p_quality_low or p_quality_high > 0.15
                
                # Decision logic - prioritize mean reversion (stocks mean-revert)
                if p_mr_long > 0.15 and p_mr_long > p_mr_short and quality_ok:
                    direction = 'LONG'
                    confidence = min(85, p_mr_long * 100 + p_quality_high * 20)
                    reasoning_parts.append("Mean reversion LONG (oversold bounce)")
                elif p_mr_short > 0.15 and p_mr_short > p_mr_long and quality_ok:
                    direction = 'SHORT'
                    confidence = min(85, p_mr_short * 100 + p_quality_high * 20)
                    reasoning_parts.append("Mean reversion SHORT (overbought fade)")
                elif p_breakout > 0.20 and p_quality_high > 0.15:
                    # Breakout with quality confirmation
                    direction = 'LONG'  # Breakouts are typically long
                    confidence = min(80, p_breakout * 80 + p_quality_high * 30)
                    reasoning_parts.append("Valid breakout setup")
                elif p_quality_low > 0.25:
                    direction = 'WAIT'
                    confidence = p_quality_low * 100
                    reasoning_parts.append("Low quality setup - AVOID")
                elif p_vol_exp > 0.25 and quality_ok:
                    # Vol expansion coming but no clear direction
                    direction = 'WAIT'
                    confidence = 60
                    reasoning_parts.append("Vol expansion coming - wait for direction")
                else:
                    direction = 'WAIT'
                    confidence = 40
                    reasoning_parts.append("No clear stock signal")
            
            else:
                # ═══════════════════════════════════════════════════════════════
                # CRYPTO SWING: Directional interpretation (original)
                # ═══════════════════════════════════════════════════════════════
                p_trend_bull = probabilities.get('trend_holds_bull', 0)
                p_trend_bear = probabilities.get('trend_holds_bear', 0)
                p_rev_bull = probabilities.get('reversal_to_bull', 0)
                p_rev_bear = probabilities.get('reversal_to_bear', 0)
                p_dd = probabilities.get('drawdown', 0)
                
                reasoning_parts.append(f"TrendUP={p_trend_bull:.0%}")
                reasoning_parts.append(f"TrendDN={p_trend_bear:.0%}")
                reasoning_parts.append(f"RevUP={p_rev_bull:.0%}")
                reasoning_parts.append(f"RevDN={p_rev_bear:.0%}")
                
                if p_trend_bull > 0.15 and p_trend_bull > p_trend_bear and p_rev_bear < 0.10 and p_dd < 0.15:
                    direction = 'LONG'
                    confidence = min(90, p_trend_bull * 100 - p_dd * 30)
                    reasoning_parts.append("Uptrend continuation")
                elif p_trend_bear > 0.15 and p_trend_bear > p_trend_bull and p_rev_bull < 0.10 and p_dd < 0.15:
                    direction = 'SHORT'
                    confidence = min(90, p_trend_bear * 100 - p_dd * 30)
                    reasoning_parts.append("Downtrend continuation")
                elif p_rev_bull > 0.15 and p_rev_bull > p_rev_bear:
                    direction = 'LONG'
                    confidence = p_rev_bull * 85
                    reasoning_parts.append("Reversal to BULL expected")
                elif p_rev_bear > 0.15 and p_rev_bear > p_rev_bull:
                    direction = 'SHORT'
                    confidence = p_rev_bear * 85
                    reasoning_parts.append("Reversal to BEAR expected")
                elif p_dd > 0.20:
                    direction = 'WAIT'
                    confidence = p_dd * 100
                    reasoning_parts.append("High drawdown risk!")
                else:
                    direction = 'WAIT'
                    confidence = 40
                    reasoning_parts.append("No clear swing signal")
        
        elif mode == 'investment':
            # Investment labels (already correct - not directional for main signals)
            p_acc = probabilities.get('accumulation', 0)
            p_dist = probabilities.get('distribution', 0)
            p_rev_bull = probabilities.get('reversal_to_bull', 0)
            p_rev_bear = probabilities.get('reversal_to_bear', 0)
            p_dd = probabilities.get('large_drawdown', 0)
            
            reasoning_parts.append(f"Accum={p_acc:.0%}")
            reasoning_parts.append(f"Dist={p_dist:.0%}")
            reasoning_parts.append(f"DD={p_dd:.0%}")
            
            if p_acc > 0.15 and p_acc > p_dist and p_dd < 0.15:
                direction = 'LONG'
                confidence = min(85, p_acc * 100 - p_dd * 25)
                reasoning_parts.append("Accumulation phase → LONG")
            elif p_dist > 0.15 and p_dist > p_acc:
                direction = 'SHORT'
                confidence = min(85, p_dist * 100)
                reasoning_parts.append("Distribution phase → SHORT")
            elif p_rev_bull > 0.15 and p_rev_bull > p_rev_bear:
                direction = 'LONG'
                confidence = p_rev_bull * 80
                reasoning_parts.append("Reversal to BULL")
            elif p_rev_bear > 0.15 and p_rev_bear > p_rev_bull:
                direction = 'SHORT'
                confidence = p_rev_bear * 80
                reasoning_parts.append("Reversal to BEAR")
            elif p_dd > 0.20:
                direction = 'WAIT'
                confidence = p_dd * 100
                reasoning_parts.append("Large drawdown risk!")
            else:
                direction = 'WAIT'
                confidence = 40
        
        else:
            direction = 'WAIT'
            confidence = 50
            reasoning_parts.append("Unknown mode")
        
        return direction, confidence, " | ".join(reasoning_parts)
    
    def _not_trained_response(self, mode: str) -> ProbabilisticPrediction:
        """
        Return clear NOT_TRAINED response when model doesn't exist.
        NO FAKE PREDICTIONS. User must train the model.
        """
        return ProbabilisticPrediction(
            mode=mode,
            probabilities={},
            direction='NOT_TRAINED',
            confidence=0,
            reasoning=f"⚠️ Model not trained for {mode} mode. Go to ML Training tab.",
            top_features=[],
            model_version="not_trained",
            prediction_time=datetime.now().isoformat(),
            f1_scores={},
            # Phase fields with defaults
            current_phase='UNKNOWN',
            phase_probabilities={},
            next_phase='UNKNOWN',
            next_phase_confidence=0.0,
            phase_bias='NEUTRAL',
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION & PERFORMANCE TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProbabilityBucket:
    """Performance stats for a probability bucket"""
    bucket_range: Tuple[float, float]  # e.g., (0.6, 0.7)
    n_predictions: int
    actual_hit_rate: float  # What % actually happened
    avg_return: float
    avg_drawdown: float


def verify_probability_calibration(
    predictions: List[Dict],  # List of {probabilities: {}, actual_outcomes: {}}
    label: str,
    n_buckets: int = 5,
) -> List[ProbabilityBucket]:
    """
    Verify that predicted probabilities match actual outcomes.
    
    A well-calibrated model should have:
    - 60% predictions in 0.6-0.7 bucket → ~60-70% actual occurrence
    - 80% predictions in 0.8-0.9 bucket → ~80-90% actual occurrence
    """
    bucket_size = 1.0 / n_buckets
    buckets = []
    
    for i in range(n_buckets):
        bucket_start = i * bucket_size
        bucket_end = (i + 1) * bucket_size
        
        # Filter predictions in this bucket
        in_bucket = [
            p for p in predictions
            if bucket_start <= p['probabilities'].get(label, 0) < bucket_end
        ]
        
        if len(in_bucket) > 0:
            # Calculate actual hit rate
            actual_hits = sum(1 for p in in_bucket if p['actual_outcomes'].get(label, 0) == 1)
            actual_rate = actual_hits / len(in_bucket)
            
            # Calculate returns and drawdowns
            avg_return = np.mean([p.get('return', 0) for p in in_bucket])
            avg_drawdown = np.mean([p.get('drawdown', 0) for p in in_bucket])
        else:
            actual_rate = 0
            avg_return = 0
            avg_drawdown = 0
        
        buckets.append(ProbabilityBucket(
            bucket_range=(bucket_start, bucket_end),
            n_predictions=len(in_bucket),
            actual_hit_rate=actual_rate,
            avg_return=avg_return,
            avg_drawdown=avg_drawdown,
        ))
    
    return buckets


def print_calibration_report(buckets: List[ProbabilityBucket], label: str):
    """Print calibration report"""
    print(f"\n{'='*60}")
    print(f"PROBABILITY CALIBRATION: {label}")
    print(f"{'='*60}")
    print(f"{'Bucket':<15} {'N':>8} {'Expected':>12} {'Actual':>12} {'Δ':>8}")
    print("-" * 60)
    
    for bucket in buckets:
        expected = (bucket.bucket_range[0] + bucket.bucket_range[1]) / 2
        delta = bucket.actual_hit_rate - expected
        delta_str = f"{delta:+.1%}"
        
        print(f"{bucket.bucket_range[0]:.0%}-{bucket.bucket_range[1]:.0%}"
              f"{bucket.n_predictions:>12}"
              f"{expected:>12.1%}"
              f"{bucket.actual_hit_rate:>12.1%}"
              f"{delta_str:>8}")
    
    print("="*60)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH EXISTING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def get_probabilistic_prediction(
    df: pd.DataFrame,
    mode: str,
    whale_data: Dict = None,
    smc_data: Dict = None,
    market_context: Dict = None,
) -> ProbabilisticPrediction:
    """
    Convenience function to get probabilistic prediction.
    
    Usage:
        pred = get_probabilistic_prediction(df, 'daytrade', whale_data)
        print(f"P_continuation: {pred.probabilities['continuation']:.2f}")
        print(f"Direction: {pred.direction} ({pred.confidence:.0f}%)")
    """
    trainer = ProbabilisticMLTrainer()
    return trainer.predict(df, mode, -1, whale_data, smc_data, market_context)


def get_all_model_metrics() -> Dict:
    """
    Load metrics for all trained models.
    
    Returns:
        Dict of {model_key: metrics_dict} for all trained models
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    metrics = {}
    
    if not os.path.exists(model_dir):
        return metrics
    
    # Load from JSON files (quick)
    for filename in os.listdir(model_dir):
        if filename.startswith('metrics_') and filename.endswith('.json'):
            try:
                with open(os.path.join(model_dir, filename), 'r') as f:
                    data = json.load(f)
                    key = f"{data['mode']}_{data['market_type']}"
                    metrics[key] = data
            except:
                pass
    
    # Fallback: Load from PKL files if no JSON
    if not metrics:
        for filename in os.listdir(model_dir):
            if filename.startswith('prob_model_') and filename.endswith('.pkl'):
                try:
                    with open(os.path.join(model_dir, filename), 'rb') as f:
                        bundle = pickle.load(f)
                        mode = bundle.get('mode', filename.split('_')[2])
                        market_type = bundle.get('market_type', 'crypto')
                        key = f"{mode}_{market_type}"
                        meta = bundle.get('metadata', {})
                        metrics[key] = {
                            'mode': mode,
                            'market_type': market_type,
                            'n_samples': meta.get('n_samples', 0),
                            'avg_f1': meta.get('avg_f1', 0),
                            'metrics': meta.get('metrics', {}),
                            'trained_at': meta.get('trained_at', 'Unknown'),
                        }
                except:
                    pass
    
    return metrics


def get_model_status() -> str:
    """
    Get formatted status of all trained models.
    
    Returns:
        Formatted string showing all models and their F1 scores
    """
    metrics = get_all_model_metrics()
    
    if not metrics:
        return "No models trained yet."
    
    lines = ["📊 **Trained Models:**\n"]
    
    for key in sorted(metrics.keys()):
        m = metrics[key]
        mode = m.get('mode', '?').upper()
        market = m.get('market_type', '?').upper()
        f1 = m.get('avg_f1', 0) * 100
        samples = m.get('n_samples', 0)
        trained = m.get('trained_at', 'Unknown')[:10]  # Just date
        
        lines.append(f"  • **{mode} {market}**: {f1:.1f}% F1 ({samples:,} samples) - {trained}")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Probabilistic ML System")
    print("="*60)
    print(f"Enhanced Features: {NUM_ENHANCED_FEATURES}")
    print(f"\nModes and Labels:")
    for mode, config in MODE_LABELS.items():
        print(f"  {mode}: {config['labels']}")
    
    print(f"\nFeature Categories:")
    print(f"  - Price & Structure: 15 features")
    print(f"  - Positioning & Flow: 12 features")
    print(f"  - SMC / Context: 12 features")
    print(f"  - Market Context: 6 features")
    print(f"  - Total: {NUM_ENHANCED_FEATURES} features")