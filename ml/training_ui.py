"""
Probabilistic ML Training UI
Uses native Streamlit components - styling handled by styles.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .probabilistic_ml import ProbabilisticMLTrainer, MODE_LABELS, NUM_ENHANCED_FEATURES
except ImportError:
    from probabilistic_ml import ProbabilisticMLTrainer, MODE_LABELS, NUM_ENHANCED_FEATURES

# Import whale data store
try:
    from core.whale_data_store import get_whale_store
except ImportError:
    try:
        from ..core.whale_data_store import get_whale_store
    except ImportError:
        get_whale_store = None

# Import stock data store
try:
    from core.stock_data_store import get_stock_store
except ImportError:
    try:
        from ..core.stock_data_store import get_stock_store
    except ImportError:
        get_stock_store = None

# Import SMC detector for Order Blocks, FVG, Liquidity Sweeps
try:
    from core.smc_detector import detect_smc, analyze_market_structure
except ImportError:
    try:
        from ..core.smc_detector import detect_smc, analyze_market_structure
    except ImportError:
        detect_smc = None
        analyze_market_structure = None

# Import SMCAnalyzer which uses smartmoneyconcepts package
try:
    from core.trade_manager import SMCAnalyzer
    SMC_ANALYZER_AVAILABLE = True
except ImportError:
    try:
        from ..core.trade_manager import SMCAnalyzer
        SMC_ANALYZER_AVAILABLE = True
    except ImportError:
        SMCAnalyzer = None
        SMC_ANALYZER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MODE_TIMEFRAMES = {
    'scalp': '5m',
    'daytrade': '15m',
    'swing': '4h',
    'investment': '1d',
}

# Recommended training period for each mode (include bull+bear cycles!)
# ENHANCED: More data = better coverage of different market conditions
MODE_TRAINING_DAYS = {
    'scalp': 90,        # 90 days of 5m data (for varied conditions)
    'daytrade': 365,    # 1 year of 15m data (full market cycle)
    'swing': 1095,      # 3 years of 4h data (multiple cycles) - IMPORTANT!
    'investment': 1825, # 5 years of daily data (bull + bear + recovery + bull)
}

MODE_DESCRIPTIONS = {
    'scalp': "Quick trades (minutes). Predicts: bullish continuation, bearish continuation, fakeout, volatility expansion",
    'daytrade': "Intraday trades (hours). Predicts: bullish continuation, bearish continuation, fakeout, volatility expansion",
    'swing': "Multi-day trades. Predicts: bullish trend holds, bearish trend holds, reversal, drawdown risk",
    'investment': "Long-term positions. Predicts: accumulation, distribution, large drawdown",
}

DEFAULT_CRYPTO = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 
                  'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'LTCUSDT', 'ATOMUSDT', 'UNIUSDT', 'NEARUSDT',
                  'AAVEUSDT', 'APTUSDT', 'ARBUSDT', 'FILUSDT', 'INJUSDT', 'OPUSDT', 'SEIUSDT', 'SUIUSDT', 'TIAUSDT', 'WLDUSDT']

# ENHANCED: 50 stocks for much better training coverage
DEFAULT_STOCKS = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Finance
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Industrial
    'CAT', 'BA', 'HON', 'UPS', 'RTX',
    # Semiconductor
    'AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'AMAT',
    # Other Tech
    'CRM', 'ADBE', 'ORCL', 'NOW', 'PYPL'
]

DEFAULT_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA', 'VWO', 'EEM', 'GLD',
                'SLV', 'TLT', 'XLF', 'XLK', 'XLE', 'XLV', 'ARKK', 'SOXX', 'SMH', 'VNQ',
                'XLI', 'XLY', 'XLP', 'XLU', 'XLRE', 'KRE', 'XBI', 'IGV', 'HACK', 'BOTZ']


# ═══════════════════════════════════════════════════════════════════════════════
# WHALE DATA MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_whale_with_price(price_df, whale_df) -> dict:
    """
    Merge historical whale data with price data by symbol and timestamp.
    
    Returns a dictionary keyed by (symbol, timestamp_str) with whale features.
    This allows the training to look up whale data for each price candle.
    """
    import pandas as pd_local
    from datetime import timedelta
    
    merged = {}
    
    try:
        price_df = price_df.copy()
        
        # Find timestamp column in price_df
        timestamp_col = None
        for col in ['timestamp', 'DateTime', 'datetime', 'Date', 'date', 'time']:
            if col in price_df.columns:
                timestamp_col = col
                break
        
        # If no timestamp column, check if index is datetime
        if timestamp_col is None:
            if isinstance(price_df.index, pd_local.DatetimeIndex):
                price_df['timestamp'] = price_df.index
            else:
                # Try to find any datetime-like column
                for col in price_df.columns:
                    if 'time' in col.lower() or 'date' in col.lower():
                        timestamp_col = col
                        break
                
                if timestamp_col is None:
                    print(f"No timestamp column found. Columns: {price_df.columns.tolist()}")
                    return {}
        
        # Convert to datetime
        if timestamp_col:
            price_df['timestamp'] = pd_local.to_datetime(price_df[timestamp_col], format='mixed')
        
        # Ensure whale timestamp is datetime (mixed format - some have T, some don't)
        whale_df = whale_df.copy()
        whale_df['timestamp'] = pd_local.to_datetime(whale_df['timestamp'], format='mixed')
        
        # Group whale data by symbol for faster lookup
        whale_by_symbol = whale_df.groupby('symbol')
        
        matched_count = 0
        for symbol in price_df['symbol'].unique():
            if symbol not in whale_by_symbol.groups:
                continue
                
            symbol_whale = whale_by_symbol.get_group(symbol).copy()
            symbol_whale = symbol_whale.sort_values('timestamp')
            
            symbol_price = price_df[price_df['symbol'] == symbol].copy()
            
            for idx, price_row in symbol_price.iterrows():
                price_time = price_row['timestamp']
                
                # Find closest whale snapshot (within 24 hours for better matching)
                time_diffs = abs(symbol_whale['timestamp'] - price_time)
                min_diff_idx = time_diffs.idxmin()
                min_diff = time_diffs[min_diff_idx]
                
                if min_diff <= timedelta(hours=24):  # Increased from 2 hours
                    whale_row = symbol_whale.loc[min_diff_idx]
                    
                    # Create key for lookup
                    key = (symbol, str(idx))
                    
                    # Store whale features
                    merged[key] = {
                        'whale_long_pct': whale_row.get('whale_long_pct', 50) if hasattr(whale_row, 'get') else whale_row['whale_long_pct'] if 'whale_long_pct' in whale_row.index else 50,
                        'retail_long_pct': whale_row.get('retail_long_pct', 50) if hasattr(whale_row, 'get') else whale_row['retail_long_pct'] if 'retail_long_pct' in whale_row.index else 50,
                        'oi_change_24h': whale_row.get('oi_change_24h', 0) if hasattr(whale_row, 'get') else whale_row['oi_change_24h'] if 'oi_change_24h' in whale_row.index else 0,
                        'funding_rate': whale_row.get('funding_rate', 0) if hasattr(whale_row, 'get') else whale_row['funding_rate'] if 'funding_rate' in whale_row.index else 0,
                        'position_in_range': whale_row.get('position_in_range', 50) if hasattr(whale_row, 'get') else whale_row['position_in_range'] if 'position_in_range' in whale_row.index else 50,
                    }
                    matched_count += 1
        
        print(f"Merge complete: {matched_count} matches found")
    except Exception as e:
        print(f"Error merging whale data: {e}")
        import traceback
        traceback.print_exc()
    
    return merged


def merge_stock_with_price(price_df, stock_df) -> dict:
    """
    Merge historical stock institutional data with price data by symbol and timestamp.
    
    Returns a dictionary keyed by (symbol, timestamp_str) with institutional features.
    Maps stock institutional data to whale-like features for consistency.
    """
    import pandas as pd_local
    from datetime import timedelta
    
    merged = {}
    
    try:
        # Ensure timestamp columns are datetime
        if 'timestamp' not in price_df.columns and 'Date' in price_df.columns:
            price_df = price_df.copy()
            price_df['timestamp'] = pd_local.to_datetime(price_df['Date'], format='mixed')
        elif 'timestamp' in price_df.columns:
            price_df = price_df.copy()
            price_df['timestamp'] = pd_local.to_datetime(price_df['timestamp'], format='mixed')
        else:
            if isinstance(price_df.index, pd_local.DatetimeIndex):
                price_df = price_df.copy()
                price_df['timestamp'] = price_df.index
            else:
                return {}
        
        # Convert stock timestamps
        stock_df = stock_df.copy()
        stock_df['timestamp'] = pd_local.to_datetime(stock_df['timestamp'], format='mixed')
        
        # Group stock data by symbol
        stock_by_symbol = stock_df.groupby('symbol')
        
        for symbol in price_df['symbol'].unique():
            if symbol not in stock_by_symbol.groups:
                continue
                
            symbol_stock = stock_by_symbol.get_group(symbol).copy()
            symbol_stock = symbol_stock.sort_values('timestamp')
            
            symbol_price = price_df[price_df['symbol'] == symbol].copy()
            
            for idx, price_row in symbol_price.iterrows():
                price_time = price_row['timestamp']
                
                # Find closest stock snapshot (within 1 day for daily data)
                time_diffs = abs(symbol_stock['timestamp'] - price_time)
                min_diff_idx = time_diffs.idxmin()
                min_diff = time_diffs[min_diff_idx]
                
                if min_diff <= timedelta(days=1):
                    stock_row = symbol_stock.loc[min_diff_idx]
                    
                    key = (symbol, str(idx))
                    
                    # Map stock institutional data to whale-like features
                    congress = stock_row['congress_score'] if 'congress_score' in stock_row.index else 50
                    insider = stock_row['insider_score'] if 'insider_score' in stock_row.index else 50
                    short_interest = stock_row['short_interest_pct'] if 'short_interest_pct' in stock_row.index else 5
                    
                    # Combined institutional sentiment (like whale positioning)
                    inst_sentiment = (congress * 0.4 + insider * 0.4 + (100 - short_interest * 10) * 0.2)
                    inst_sentiment = max(0, min(100, inst_sentiment))
                    
                    merged[key] = {
                        # Map to whale features for consistency
                        'whale_long_pct': inst_sentiment,  # Institutional sentiment
                        'retail_long_pct': 100 - inst_sentiment,  # Inverse
                        'oi_change_24h': -short_interest,  # Short interest as negative signal
                        'funding_rate': 0,  # Not applicable to stocks
                        'position_in_range': stock_row['position_pct'] if 'position_pct' in stock_row.index else 50,
                        # Also store original values
                        'congress_score': congress,
                        'insider_score': insider,
                        'short_interest_pct': short_interest,
                    }
    except Exception as e:
        print(f"Error merging stock data: {e}")
    
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_crypto_data(symbols, timeframe, days, progress_bar):
    try:
        from core.data_fetcher import fetch_binance_klines
    except ImportError:
        st.error("Could not import data_fetcher")
        return pd.DataFrame()
    
    all_data = []
    for i, symbol in enumerate(symbols):
        try:
            tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}
            limit = min(int((days * 1440) / tf_minutes.get(timeframe, 60)), 1000)
            df = fetch_binance_klines(symbol, timeframe, limit)
            if df is not None and len(df) > 50:
                df['symbol'] = symbol
                all_data.append(df)
        except:
            pass
        progress_bar.progress((i + 1) / len(symbols))
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def fetch_stock_data(symbols, timeframe, days, progress_bar):
    try:
        from core.data_fetcher import fetch_stock_data as fetch_stock
    except ImportError:
        st.error("Could not import stock data fetcher")
        return pd.DataFrame()
    
    all_data = []
    
    # Yahoo Finance doesn't have native 4h candles - need to resample from 1h
    needs_resample = (timeframe == '4h')
    fetch_tf = '1h' if needs_resample else timeframe
    
    # Map timeframes to Yahoo format
    tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d', '1w': '1wk'}
    yahoo_tf = tf_map.get(fetch_tf, '1d')
    
    for i, symbol in enumerate(symbols):
        try:
            # Fetch more data if we need to resample (4x for 4h)
            fetch_days = days * 4 if needs_resample else days * 10
            df = fetch_stock(symbol, yahoo_tf, fetch_days)
            
            if df is not None and len(df) > 50:
                # Resample 1h to 4h if needed
                if needs_resample and 'DateTime' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['DateTime'])
                    df = df.set_index('DateTime')
                    df = df.resample('4h').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                    df = df.reset_index()
                    print(f"  ✓ Resampled {symbol} to 4h: {len(df)} candles")
                
                df['symbol'] = symbol
                all_data.append(df)
        except Exception as e:
            print(f"  ⚠️ Error fetching {symbol}: {e}")
        progress_bar.progress((i + 1) / len(symbols))
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def add_technical_indicators(df):
    """Add all technical indicators needed for ML training."""
    if df.empty:
        return df
    df = df.copy()
    
    # ATR
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    
    # CMF (Chaikin Money Flow)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 1e-10)
    df['CMF'] = (mfm.fillna(0) * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum().replace(0, 1e-10)
    
    # Volume ratio
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, 1e-10)
    
    # === NEW: Missing indicators for stock ML ===
    
    # SMAs
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # EMAs
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Hour for session timing
    df['hour'] = pd.to_datetime(df['DateTime']).dt.hour if 'DateTime' in df.columns else 12
    
    return df.dropna()


def calculate_smc_features(df: pd.DataFrame, idx: int, trading_mode: str = 'Day Trade') -> dict:
    """
    Calculate SMC features for a specific row in the dataframe.
    
    Args:
        df: OHLCV DataFrame with indicators
        idx: Index to calculate SMC for (uses data up to this point)
        trading_mode: Trading mode for threshold adjustments
    
    Returns:
        Dict with SMC features for ML
    """
    if detect_smc is None:
        return {}
    
    # Need at least 50 candles for SMC detection
    if idx < 50:
        return {}
    
    try:
        # Get data window up to current index (simulate real-time)
        window_df = df.iloc[max(0, idx-100):idx+1].copy().reset_index(drop=True)
        
        if len(window_df) < 50:
            return {}
        
        current_price = window_df['Close'].iloc[-1]
        current_high = window_df['High'].iloc[-1]
        current_low = window_df['Low'].iloc[-1]
        
        # Detect SMC patterns
        smc_result = detect_smc(window_df, use_ict=True, trading_mode=trading_mode)
        
        order_blocks = smc_result.get('order_blocks', {})
        fvg = smc_result.get('fvg', {})
        liquidity = smc_result.get('liquidity_sweep', {})
        structure = smc_result.get('structure', {})
        
        # Calculate proximity to order blocks
        atr = window_df['ATR'].iloc[-1] if 'ATR' in window_df.columns else (current_high - current_low)
        proximity_threshold = atr * 1.5
        
        # Check bullish OBs
        at_bullish_ob = False
        near_bullish_ob = False
        bullish_obs = order_blocks.get('bullish_obs', [])
        for ob in bullish_obs:
            ob_top = ob.get('top', 0)
            ob_bottom = ob.get('bottom', 0)
            if ob_bottom <= current_price <= ob_top:
                at_bullish_ob = True
            elif abs(current_price - ob_top) <= proximity_threshold:
                near_bullish_ob = True
        
        # Check bearish OBs
        at_bearish_ob = False
        near_bearish_ob = False
        bearish_obs = order_blocks.get('bearish_obs', [])
        for ob in bearish_obs:
            ob_top = ob.get('top', 0)
            ob_bottom = ob.get('bottom', 0)
            if ob_bottom <= current_price <= ob_top:
                at_bearish_ob = True
            elif abs(current_price - ob_bottom) <= proximity_threshold:
                near_bearish_ob = True
        
        # Check FVGs
        in_fvg_bullish = False
        in_fvg_bearish = False
        bullish_fvgs = fvg.get('bullish', [])
        bearish_fvgs = fvg.get('bearish', [])
        
        for f in bullish_fvgs:
            if f.get('bottom', 0) <= current_price <= f.get('top', 0):
                in_fvg_bullish = True
                break
        
        for f in bearish_fvgs:
            if f.get('bottom', 0) <= current_price <= f.get('top', 0):
                in_fvg_bearish = True
                break
        
        # Liquidity sweeps
        liquidity_sweep_bull = liquidity.get('sweep_low', False)
        liquidity_sweep_bear = liquidity.get('sweep_high', False)
        
        # Structure bias for accumulation/distribution
        bias = structure.get('bias', 'Neutral')
        if bias == 'Bullish':
            accumulation_score = 70
            distribution_score = 30
        elif bias == 'Bearish':
            accumulation_score = 30
            distribution_score = 70
        else:
            accumulation_score = 50
            distribution_score = 50
        
        return {
            'at_bullish_ob': at_bullish_ob,
            'at_bearish_ob': at_bearish_ob,
            'near_bullish_ob': near_bullish_ob,
            'near_bearish_ob': near_bearish_ob,
            'in_fvg_bullish': in_fvg_bullish,
            'in_fvg_bearish': in_fvg_bearish,
            'liquidity_sweep_bull': liquidity_sweep_bull,
            'liquidity_sweep_bear': liquidity_sweep_bear,
            'accumulation_score': accumulation_score,
            'distribution_score': distribution_score,
            'structure_bullish': 1 if bias == 'Bullish' else 0,
            'structure_bearish': 1 if bias == 'Bearish' else 0,
        }
        
    except Exception as e:
        # Silent fail - return empty dict
        return {}


def add_smc_columns(df: pd.DataFrame, trading_mode: str = 'Day Trade', progress_callback=None) -> pd.DataFrame:
    """
    Pre-calculate SMC features for entire dataframe using smartmoneyconcepts package.
    
    The SMC package returns DataFrames where:
    - Most rows are NaN (no event at that candle)
    - OBs/FVGs are detected at ONE candle but valid for FUTURE candles until mitigated
    - We need to check if current price is within any PAST (unmitigated) OB/FVG range
    """
    if not SMC_ANALYZER_AVAILABLE:
        print("⚠️ SMCAnalyzer not available - SMC features will be zeros")
        return df
    
    if df.empty:
        return df
    
    df = df.copy()
    
    # Initialize SMC columns
    smc_columns = [
        'smc_at_bullish_ob', 'smc_at_bearish_ob',
        'smc_near_bullish_ob', 'smc_near_bearish_ob',
        'smc_in_fvg_bullish', 'smc_in_fvg_bearish',
        'smc_liquidity_sweep_bull', 'smc_liquidity_sweep_bear',
        'smc_accumulation_score', 'smc_distribution_score',
        'smc_structure_bullish', 'smc_structure_bearish',
        'smc_bos_bullish', 'smc_bos_bearish',
        'smc_choch_bullish', 'smc_choch_bearish',
    ]
    
    for col in smc_columns:
        df[col] = 0
    
    df['smc_accumulation_score'] = 50
    df['smc_distribution_score'] = 50
    
    # Process by symbol if multiple symbols in df
    symbols = df['symbol'].unique() if 'symbol' in df.columns else ['ALL']
    total_symbols = len(symbols)
    
    for sym_idx, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(f"SMC: Processing {symbol} ({sym_idx+1}/{total_symbols})...")
        
        if symbol == 'ALL':
            symbol_mask = df.index
            symbol_df = df
        else:
            symbol_mask = df['symbol'] == symbol
            symbol_df = df[symbol_mask].copy()
        
        if len(symbol_df) < 50:
            continue
        
        try:
            # Use SMCAnalyzer (wraps smartmoneyconcepts package)
            analyzer = SMCAnalyzer(symbol_df, trading_mode=trading_mode)
            
            # Get results from package
            order_blocks = analyzer.order_blocks
            fvg = analyzer.fvg
            bos_choch = analyzer.bos_choch
            liquidity = analyzer.liquidity if hasattr(analyzer, 'liquidity') else pd.DataFrame()
            
            # Extract valid OBs (non-NaN rows)
            bullish_obs = []  # List of (top, bottom, mitigated_idx)
            bearish_obs = []
            
            if order_blocks is not None and 'OB' in order_blocks.columns:
                for idx in range(len(order_blocks)):
                    ob_type = order_blocks['OB'].iloc[idx]
                    if pd.isna(ob_type):
                        continue
                    
                    top = order_blocks['Top'].iloc[idx]
                    bottom = order_blocks['Bottom'].iloc[idx]
                    mitigated = order_blocks['MitigatedIndex'].iloc[idx]
                    
                    if pd.isna(top) or pd.isna(bottom):
                        continue
                    
                    mitigated_idx = int(mitigated) if not pd.isna(mitigated) else 99999
                    
                    if ob_type == 1:  # Bullish OB
                        bullish_obs.append((idx, top, bottom, mitigated_idx))
                    elif ob_type == -1:  # Bearish OB
                        bearish_obs.append((idx, top, bottom, mitigated_idx))
            
            # Extract valid FVGs
            bullish_fvgs = []
            bearish_fvgs = []
            
            if fvg is not None and 'FVG' in fvg.columns:
                for idx in range(len(fvg)):
                    fvg_type = fvg['FVG'].iloc[idx]
                    if pd.isna(fvg_type):
                        continue
                    
                    top = fvg['Top'].iloc[idx]
                    bottom = fvg['Bottom'].iloc[idx]
                    mitigated = fvg['MitigatedIndex'].iloc[idx]
                    
                    if pd.isna(top) or pd.isna(bottom):
                        continue
                    
                    mitigated_idx = int(mitigated) if not pd.isna(mitigated) else 99999
                    
                    if fvg_type == 1:  # Bullish FVG
                        bullish_fvgs.append((idx, top, bottom, mitigated_idx))
                    elif fvg_type == -1:  # Bearish FVG
                        bearish_fvgs.append((idx, top, bottom, mitigated_idx))
            
            # Now map to each candle
            symbol_indices = symbol_df.index.tolist()
            
            for i, actual_idx in enumerate(symbol_indices):
                current_price = symbol_df['Close'].iloc[i]
                atr = symbol_df['ATR'].iloc[i] if 'ATR' in symbol_df.columns else (symbol_df['High'].iloc[i] - symbol_df['Low'].iloc[i])
                proximity = atr * 1.5
                
                # Check Bullish OBs (formed BEFORE current candle, not yet mitigated)
                for ob_idx, top, bottom, mitigated_idx in bullish_obs:
                    if ob_idx < i and i < mitigated_idx:  # OB formed before, not mitigated yet
                        if bottom <= current_price <= top:
                            df.loc[actual_idx, 'smc_at_bullish_ob'] = 1
                        elif current_price < top and (top - current_price) <= proximity:
                            df.loc[actual_idx, 'smc_near_bullish_ob'] = 1
                
                # Check Bearish OBs
                for ob_idx, top, bottom, mitigated_idx in bearish_obs:
                    if ob_idx < i and i < mitigated_idx:
                        if bottom <= current_price <= top:
                            df.loc[actual_idx, 'smc_at_bearish_ob'] = 1
                        elif current_price > bottom and (current_price - bottom) <= proximity:
                            df.loc[actual_idx, 'smc_near_bearish_ob'] = 1
                
                # Check Bullish FVGs
                for fvg_idx, top, bottom, mitigated_idx in bullish_fvgs:
                    if fvg_idx < i and i < mitigated_idx:
                        if bottom <= current_price <= top:
                            df.loc[actual_idx, 'smc_in_fvg_bullish'] = 1
                
                # Check Bearish FVGs
                for fvg_idx, top, bottom, mitigated_idx in bearish_fvgs:
                    if fvg_idx < i and i < mitigated_idx:
                        if bottom <= current_price <= top:
                            df.loc[actual_idx, 'smc_in_fvg_bearish'] = 1
                
                # BOS/CHOCH - these are events AT specific candles
                if bos_choch is not None and i < len(bos_choch):
                    if 'BOS' in bos_choch.columns:
                        bos_val = bos_choch['BOS'].iloc[i]
                        if not pd.isna(bos_val):
                            if bos_val == 1:
                                df.loc[actual_idx, 'smc_bos_bullish'] = 1
                                df.loc[actual_idx, 'smc_structure_bullish'] = 1
                                df.loc[actual_idx, 'smc_accumulation_score'] = 70
                                df.loc[actual_idx, 'smc_distribution_score'] = 30
                            elif bos_val == -1:
                                df.loc[actual_idx, 'smc_bos_bearish'] = 1
                                df.loc[actual_idx, 'smc_structure_bearish'] = 1
                                df.loc[actual_idx, 'smc_accumulation_score'] = 30
                                df.loc[actual_idx, 'smc_distribution_score'] = 70
                    
                    if 'CHOCH' in bos_choch.columns:
                        choch_val = bos_choch['CHOCH'].iloc[i]
                        if not pd.isna(choch_val):
                            if choch_val == 1:
                                df.loc[actual_idx, 'smc_choch_bullish'] = 1
                            elif choch_val == -1:
                                df.loc[actual_idx, 'smc_choch_bearish'] = 1
                
                # Liquidity sweeps - events AT specific candles
                if liquidity is not None and len(liquidity) > 0 and i < len(liquidity):
                    if 'Swept' in liquidity.columns:
                        swept = liquidity['Swept'].iloc[i]
                        liq_type = liquidity['Liquidity'].iloc[i] if 'Liquidity' in liquidity.columns else np.nan
                        if not pd.isna(swept) and swept == 1:
                            if not pd.isna(liq_type):
                                if liq_type == -1:  # Swept low = bullish
                                    df.loc[actual_idx, 'smc_liquidity_sweep_bull'] = 1
                                elif liq_type == 1:  # Swept high = bearish
                                    df.loc[actual_idx, 'smc_liquidity_sweep_bear'] = 1
                        
        except Exception as e:
            print(f"⚠️ SMC error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL STATUS
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_status():
    """Get status of all trained probabilistic models with F1 scores."""
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    status = {}
    
    for mode in ['scalp', 'daytrade', 'swing', 'investment']:
        for market in ['crypto', 'stock', 'etf']:
            key = f"{mode}_{market}"
            
            # Check new naming first: prob_model_swing_crypto.pkl
            model_path = os.path.join(model_dir, f'prob_model_{mode}_{market}.pkl')
            
            # Fallback to old naming: prob_model_swing.pkl (only for crypto)
            if not os.path.exists(model_path) and market == 'crypto':
                model_path = os.path.join(model_dir, f'prob_model_{mode}.pkl')
            
            if os.path.exists(model_path):
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                
                f1_score = None
                n_samples = None
                
                # Try to get F1 score from JSON first (fast)
                json_path = os.path.join(model_dir, f'metrics_{mode}_{market}.json')
                if os.path.exists(json_path):
                    try:
                        import json
                        with open(json_path, 'r') as f:
                            metrics = json.load(f)
                            f1_score = metrics.get('avg_f1', 0)
                            if f1_score and f1_score < 1:
                                f1_score *= 100
                            n_samples = metrics.get('n_samples', 0)
                    except:
                        pass
                
                # Fallback: Load from pickle and calculate avg_f1
                if f1_score is None or f1_score == 0:
                    try:
                        import pickle
                        with open(model_path, 'rb') as f:
                            bundle = pickle.load(f)
                            meta = bundle.get('metadata', {})
                            
                            # Try direct avg_f1 first
                            f1_score = meta.get('avg_f1', 0)
                            
                            # Calculate from per-label metrics if not available
                            if (not f1_score or f1_score == 0) and 'metrics' in meta:
                                label_metrics = meta['metrics']
                                if label_metrics:
                                    f1_scores = []
                                    for label, lm in label_metrics.items():
                                        lf1 = lm.get('f1', 0)
                                        if lf1 > 0:
                                            f1_scores.append(lf1)
                                    if f1_scores:
                                        f1_score = sum(f1_scores) / len(f1_scores)
                            
                            if f1_score and f1_score < 1:
                                f1_score *= 100
                            n_samples = meta.get('n_samples', 0)
                    except:
                        pass
                
                status[key] = {
                    'trained': True, 
                    'date': mtime.strftime('%Y-%m-%d'), 
                    'f1_score': f1_score or 0,
                    'n_samples': n_samples or 0,
                }
            else:
                status[key] = {'trained': False, 'date': None, 'f1_score': 0, 'n_samples': 0}
    
    return status


def get_detailed_metrics(mode: str, market: str) -> dict:
    """Get detailed per-label metrics for a trained model."""
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'probabilistic')
    
    result = {}
    
    # Try JSON first (faster)
    json_path = os.path.join(model_dir, f'metrics_{mode}_{market}.json')
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r') as f:
                metrics = json.load(f)
                # Handle both 'per_label' (new) and 'metrics' (old) keys
                per_label = metrics.get('per_label') or metrics.get('metrics', {})
                result = {
                    'per_label': per_label,
                    'avg_f1': metrics.get('avg_f1', 0),
                    'n_samples': metrics.get('n_samples', 0),
                    'best_models': metrics.get('best_models', {}),
                    # Phase model info (from JSON if available)
                    'phase_trained': metrics.get('phase_trained', False),
                    'phase_metrics': metrics.get('phase_metrics', {}),
                }
        except:
            pass
    
    # If phase_trained not in JSON (old format), check pickle for phase_model
    model_path = os.path.join(model_dir, f'prob_model_{mode}_{market}.pkl')
    if not os.path.exists(model_path) and market == 'crypto':
        model_path = os.path.join(model_dir, f'prob_model_{mode}.pkl')
    
    if os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
                meta = bundle.get('metadata', {})
                
                # If we don't have result from JSON, build from pickle
                if not result:
                    per_label = meta.get('metrics', {})
                    best_models = {}
                    for label, label_metrics in per_label.items():
                        best_models[label] = label_metrics.get('best_model', 'Unknown')
                    
                    result = {
                        'per_label': per_label,
                        'avg_f1': meta.get('avg_f1', 0),
                        'n_samples': meta.get('n_samples', 0),
                        'best_models': best_models,
                    }
                
                # ALWAYS check pickle for phase_model (may exist even if JSON doesn't have it)
                phase_model = bundle.get('phase_model')
                phase_metrics_pkl = meta.get('phase_metrics', {})
                
                # Override if pickle has phase_model but JSON didn't
                if phase_model is not None:
                    result['phase_trained'] = True
                    if phase_metrics_pkl:
                        result['phase_metrics'] = phase_metrics_pkl
                elif 'phase_trained' not in result:
                    result['phase_trained'] = False
                    result['phase_metrics'] = {}
                    
        except Exception as e:
            pass
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(mode, timeframe, market_type, symbols, days):
    st.subheader(f"🚀 Training {mode.upper()} Model")
    st.write(f"Market: {market_type} | Timeframe: {timeframe} | Symbols: {len(symbols)}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch
    status_text.write(f"📡 Fetching {market_type} data...")
    df = fetch_crypto_data(symbols, timeframe, days, progress_bar) if market_type == "Crypto" else fetch_stock_data(symbols, timeframe, days, progress_bar)
    
    if df.empty:
        st.error("No data fetched. Check symbols and connection.")
        return
    
    st.success(f"Fetched {len(df):,} candles from {df['symbol'].nunique()} symbols")
    
    # Indicators
    status_text.write("📊 Adding indicators...")
    progress_bar.progress(0.4)
    df = add_technical_indicators(df)
    
    # Remove NaN values
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df) < 500:
        st.error(f"Not enough data. Got {len(df)}, need 500+")
        return
    
    st.success(f"{len(df):,} candles with indicators")
    
    # Add SMC features (Order Blocks, FVG, BOS/CHOCH) - CRITICAL FOR ML!
    if SMC_ANALYZER_AVAILABLE:
        status_text.write("🎯 Calculating SMC features (OB, FVG, BOS)...")
        progress_bar.progress(0.45)
        
        trading_mode_map = {
            'scalp': 'Scalp',
            'daytrade': 'Day Trade', 
            'swing': 'Swing',
            'investment': 'Investment'
        }
        smc_trading_mode = trading_mode_map.get(mode, 'Day Trade')
        
        try:
            df = add_smc_columns(df, trading_mode=smc_trading_mode, 
                                progress_callback=lambda msg: status_text.write(msg))
            
            # Count SMC detections
            smc_cols = [c for c in df.columns if c.startswith('smc_')]
            # Count binary detections (exclude score columns which are 0-100)
            binary_smc_cols = [c for c in smc_cols if 'score' not in c]
            smc_counts = {col: int(df[col].sum()) for col in binary_smc_cols if df[col].dtype in ['int64', 'float64', 'int32', 'float32']}
            st.success(f"🎯 SMC features added: {sum(smc_counts.values()):,} detections")
            
            with st.expander("📊 SMC Detection Counts"):
                for col, count in sorted(smc_counts.items()):
                    if count > 0:
                        st.write(f"  {col}: {count:,}")
        except Exception as e:
            st.warning(f"⚠️ SMC calculation error: {e}. Training without SMC features.")
    else:
        st.warning("⚠️ SMCAnalyzer not available. Training without SMC features.")
    
    # Load historical whale data
    whale_history_df = None
    if market_type == "Crypto" and get_whale_store is not None:
        status_text.write("🐋 Loading historical whale data...")
        progress_bar.progress(0.5)
        
        try:
            whale_store = get_whale_store()
            all_snapshots = whale_store.get_all_snapshots(lookback_days=days, max_records=50000)
            
            if all_snapshots:
                import pandas
                whale_history_df = pandas.DataFrame(all_snapshots)
                whale_history_df['timestamp'] = pandas.to_datetime(whale_history_df['timestamp'], format='mixed')
                st.success(f"🐋 Loaded {len(whale_history_df):,} whale snapshots from history")
            else:
                st.warning("⚠️ No whale history found. Training with price data only.")
        except Exception as e:
            st.warning(f"⚠️ Could not load whale data: {e}. Training with price data only.")
    
    # Load historical stock institutional data
    stock_history_df = None
    if market_type in ["Stocks", "ETFs"] and get_stock_store is not None:
        status_text.write("🏛️ Loading historical institutional data...")
        progress_bar.progress(0.5)
        
        try:
            stock_store = get_stock_store()
            all_snapshots = stock_store.get_all_snapshots(lookback_days=days, max_records=50000)
            
            if all_snapshots:
                import pandas
                stock_history_df = pandas.DataFrame(all_snapshots)
                st.success(f"🏛️ Loaded {len(stock_history_df):,} institutional snapshots (Congress/Insider/Short Interest)")
            else:
                st.warning("⚠️ No institutional history found. Training with price data only.")
        except Exception as e:
            st.warning(f"⚠️ Could not load institutional data: {e}. Training with price data only.")
    
    # Merge whale data with price data (Crypto)
    merged_whale_data = None
    if whale_history_df is not None and len(whale_history_df) > 0:
        status_text.write("🔗 Merging whale data with price data...")
        progress_bar.progress(0.55)
        
        try:
            # Debug: Show data structure
            st.write(f"**Debug:** Price columns: {df.columns.tolist()[:8]}...")
            st.write(f"**Debug:** Whale columns: {whale_history_df.columns.tolist()[:8]}...")
            
            # Debug: Show what symbols we have
            price_symbols = set(df['symbol'].unique()) if 'symbol' in df.columns else set()
            whale_symbols = set(whale_history_df['symbol'].unique()) if 'symbol' in whale_history_df.columns else set()
            
            st.write(f"**Debug:** Price symbols: {list(price_symbols)[:5]}")
            st.write(f"**Debug:** Whale symbols: {list(whale_symbols)[:5]}")
            
            # Check for matches
            matching_symbols = price_symbols & whale_symbols
            
            if not matching_symbols:
                st.warning(f"⚠️ No matching symbols!")
                
                # Try to fix symbol mismatch (BTCUSDT vs BTC)
                symbol_map = {}
                for ps in price_symbols:
                    base = ps.replace('USDT', '').replace('BUSD', '').replace('USD', '')
                    for ws in whale_symbols:
                        if base in ws or ws in base or ps == ws:
                            symbol_map[ps] = ws
                            break
                
                if symbol_map:
                    st.info(f"🔄 Symbol mapping: {symbol_map}")
                    # Remap whale_df symbols to match price_df
                    whale_history_df_remapped = whale_history_df.copy()
                    reverse_map = {v: k for k, v in symbol_map.items()}
                    whale_history_df_remapped['symbol'] = whale_history_df_remapped['symbol'].map(lambda x: reverse_map.get(x, x))
                    merged_whale_data = merge_whale_with_price(df, whale_history_df_remapped)
            else:
                st.info(f"✓ Found {len(matching_symbols)} matching symbols: {list(matching_symbols)[:5]}...")
                merged_whale_data = merge_whale_with_price(df, whale_history_df)
            
            if merged_whale_data:
                st.success(f"🔗 Merged whale data: {len(merged_whale_data)} matched records")
            else:
                st.warning(f"⚠️ Merge returned 0 records. Timestamps might not overlap.")
                # Show timestamp ranges
                if 'timestamp' in whale_history_df.columns:
                    whale_ts = pd.to_datetime(whale_history_df['timestamp'], format='mixed')
                    st.write(f"**Whale time range:** {whale_ts.min()} to {whale_ts.max()}")
        except Exception as e:
            import traceback
            st.warning(f"⚠️ Could not merge whale data: {e}")
            st.code(traceback.format_exc())
    
    # Merge stock institutional data with price data (Stocks/ETFs)
    merged_inst_data = None
    if stock_history_df is not None and len(stock_history_df) > 0:
        status_text.write("🔗 Merging institutional data with price data...")
        progress_bar.progress(0.55)
        
        try:
            merged_inst_data = merge_stock_with_price(df, stock_history_df)
            if merged_inst_data:
                st.success(f"🔗 Merged institutional data: {len(merged_inst_data)} matched records")
        except Exception as e:
            st.warning(f"⚠️ Could not merge institutional data: {e}")
    
    # Use whichever data was loaded (whale for crypto, institutional for stocks)
    training_inst_data = merged_whale_data or merged_inst_data
    
    # Train
    status_text.write(f"🧠 Training {mode} model...")
    progress_bar.progress(0.6)
    
    # Create detailed progress display
    model_progress_container = st.container()
    with model_progress_container:
        model_status = st.empty()
        model_detail = st.empty()
    
    # Progress callback for detailed updates
    def training_progress(progress: float, message: str):
        # Update the main progress bar (60% to 95%)
        actual_progress = 0.6 + (progress * 0.35)
        progress_bar.progress(min(actual_progress, 0.95))
        model_status.write(f"🔄 {message}")
    
    try:
        trainer = ProbabilisticMLTrainer()
        
        # Get actual model count
        try:
            from ml.ensemble_models import get_all_models
            actual_model_count = len(get_all_models())
        except:
            actual_model_count = 32  # Default estimate
        
        # Show what we're about to train
        model_detail.info(f"📦 Training {actual_model_count} models per label: GradientBoosting, XGBoost, LightGBM, CatBoost, RandomForest, ExtraTrees, AdaBoost, MLP, KNN...")
        
        # Determine market_type for correct thresholds AND save path
        # CRITICAL: Keep ETFs separate from Stocks so models don't overwrite each other!
        if market_type == "Crypto":
            ml_market_type = 'crypto'
        elif market_type == "ETFs":
            ml_market_type = 'etf'
        else:
            ml_market_type = 'stock'
        
        metrics = trainer.train_mode(
            df=df, 
            mode=mode, 
            whale_data=training_inst_data, 
            smc_data=None, 
            market_context=None,
            progress_callback=training_progress,
            market_type=ml_market_type,
            auto_tune=True,  # Auto-calculate optimal thresholds from data!
            target_positive_rate=0.35,  # Target 35% positive rate
        )
        
        progress_bar.progress(1.0)
        status_text.write("✅ Model trained!")
        model_status.write(f"✅ All {actual_model_count} models trained and best selected per label!")
        model_detail.empty()
        
        st.subheader("📊 Training Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Samples", f"{metrics['n_samples']:,}")
        col2.metric("Mode", metrics['mode'].upper())
        col3.metric("Labels", len(metrics['labels']))
        
        # Calculate average F1 across labels
        avg_f1 = sum(m.get('f1', 0) for m in metrics['metrics'].values()) / len(metrics['metrics'])
        col4.metric("Avg F1 Score", f"{avg_f1:.1%}")
        
        # ═══════════════════════════════════════════════════════════════
        # 🔮 PHASE MODEL METRICS (NEW!)
        # ═══════════════════════════════════════════════════════════════
        phase_trained = metrics.get('phase_trained', False)
        phase_metrics = metrics.get('phase_metrics', {})
        
        if phase_trained and phase_metrics:
            st.divider()
            st.write("**🔮 Phase Prediction Model (Wyckoff Phases):**")
            
            phase_cols = st.columns(4)
            with phase_cols[0]:
                st.metric("Phase Model", "✅ Trained", delta="Active")
            with phase_cols[1]:
                phase_f1 = phase_metrics.get('f1_macro', 0)
                color = "normal" if phase_f1 >= 0.4 else "off"
                st.metric("F1 (Macro)", f"{phase_f1:.1%}", delta_color=color)
            with phase_cols[2]:
                phase_acc = phase_metrics.get('accuracy', 0)
                st.metric("Accuracy", f"{phase_acc:.1%}")
            with phase_cols[3]:
                n_classes = phase_metrics.get('n_classes', 7)
                st.metric("Phases", f"{n_classes}")
            
            st.caption("Phases: ACCUMULATION → MARKUP → DISTRIBUTION → MARKDOWN → CAPITULATION (cycle)")
            st.success("✅ Phase predictions will show as 'Phase (ML)' in Single Analysis & Scanner")
        else:
            st.divider()
            st.warning("⚠️ Phase model not trained - will use rules-based 'Money Flow' detection")
        
        st.divider()
        
        # Detailed metrics per label with BEST MODEL shown
        st.write("**Performance by Label (Best Model Selected):**")
        
        # Create a table
        metrics_data = []
        for label, m in metrics['metrics'].items():
            metrics_data.append({
                'Label': label,
                'Best Model': m.get('best_model', 'GradientBoosting')[:25],
                'F1 Score': f"{m.get('f1', 0):.1%}",
                'Precision': f"{m.get('precision', 0):.1%}",
                'Recall': f"{m.get('recall', 0):.1%}",
                'Accuracy': f"{m.get('accuracy', 0):.1%}",
                'Positive Rate': f"{m.get('positive_rate', 0):.1%}",
            })
        
        import pandas as pd
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Show model comparison for each label
        st.divider()
        st.write("**📊 Model Comparison (Top 10 per Label):**")
        
        for label, m in metrics['metrics'].items():
            all_models = m.get('all_models', {})
            if all_models:
                with st.expander(f"🔍 {label} - All Models Tested"):
                    model_comparison = []
                    for model_name, model_metrics in all_models.items():
                        model_comparison.append({
                            'Model': model_name,
                            'F1': f"{model_metrics.get('f1', 0):.1%}",
                            'Precision': f"{model_metrics.get('precision', 0):.1%}",
                            'Recall': f"{model_metrics.get('recall', 0):.1%}",
                        })
                    
                    if model_comparison:
                        df_comparison = pd.DataFrame(model_comparison)
                        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                        
                        # Best model highlight
                        best = model_comparison[0]['Model'] if model_comparison else 'Unknown'
                        st.success(f"✅ Best for {label}: **{best}**")
        
        # Interpretation guide
        with st.expander("📖 What do these metrics mean?"):
            st.write("""
            - **F1 Score**: Balance between precision and recall (higher = better). Target: >50%
            - **Precision**: When model predicts positive, how often is it correct? (avoid false signals)
            - **Recall**: Of all actual positives, how many did the model catch? (don't miss opportunities)
            - **Accuracy**: Overall correct predictions (can be misleading with imbalanced data)
            - **Positive Rate**: How often this label occurs in training data (ideal: 25-40%)
            """)
            
            st.write("**Class Balance Guide:**")
            st.write("- Positive Rate < 10%: Too rare, model may struggle")
            st.write("- Positive Rate 20-40%: Good balance for learning")
            st.write("- Positive Rate > 50%: May need stricter thresholds")
            
            st.divider()
            
            st.write("**🔄 Regime Awareness:**")
            st.write("""
            The model automatically calculates market regime from:
            - **Trend Direction**: 10/30 MA crossover → bull/bear/neutral
            - **Volatility Regime**: ATR vs 20-period average
            - **Momentum Regime**: RSI > 60 = bullish, RSI < 40 = bearish
            
            This means at prediction time, the model automatically adjusts based on current conditions!
            """)
        
        # Show if any labels need attention
        problem_labels = []
        for label, m in metrics['metrics'].items():
            pos_rate = m.get('positive_rate', 0)
            f1 = m.get('f1', 0)
            if pos_rate < 0.15:
                problem_labels.append(f"{label}: Low positive rate ({pos_rate:.1%})")
            elif f1 < 0.4:
                problem_labels.append(f"{label}: Low F1 ({f1:.1%})")
        
        if problem_labels:
            st.warning(f"⚠️ Labels needing attention: {', '.join(problem_labels)}")
        
        st.balloons()
    except Exception as e:
        st.error(f"Training failed: {e}")
        import traceback
        st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE - UNIFIED TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def render_training_page():
    st.title("🧠 ML Training Center")
    
    st.write("""
    **Two ML Systems Working Together:**
    - **Traditional ML (32 models):** Analyzes whale data, indicators, SMC → Predicts continuation, fakeout, reversal
    - **Pattern Detection (Deep Learning):** CNN/LSTM → Detects Double Top, H&S, Flags, Triangles, etc.
    """)
    
    # Import pattern detection with comprehensive error handling
    HAS_TORCH = False
    TORCH_ERROR = None
    MODE_PATTERNS = {}
    PATTERN_DIRECTION = {}
    get_pattern_model_status = lambda: {}
    train_pattern_model = None
    
    try:
        from .deep_pattern_detection import (
            DeepPatternTrainer, RuleBasedPatternDetector,
            PatternType, PATTERN_DIRECTION, MODE_PATTERNS,
            MODE_SEQUENCE_LENGTH, HAS_TORCH
        )
        try:
            from .deep_pattern_detection import TORCH_ERROR
        except ImportError:
            TORCH_ERROR = None
        from .pattern_training_ui import get_pattern_model_status, train_pattern_model
    except ImportError:
        try:
            from deep_pattern_detection import (
                DeepPatternTrainer, RuleBasedPatternDetector,
                PatternType, PATTERN_DIRECTION, MODE_PATTERNS,
                MODE_SEQUENCE_LENGTH, HAS_TORCH
            )
            try:
                from deep_pattern_detection import TORCH_ERROR
            except ImportError:
                TORCH_ERROR = None
            from pattern_training_ui import get_pattern_model_status, train_pattern_model
        except Exception as e:
            TORCH_ERROR = str(e)
    except Exception as e:
        TORCH_ERROR = str(e)
    
    # Import ensemble info
    HAS_XGBOOST = HAS_LIGHTGBM = HAS_CATBOOST = False
    get_all_models = lambda: {}
    
    try:
        from .ensemble_models import HAS_XGBOOST, HAS_LIGHTGBM, HAS_CATBOOST, get_all_models
    except ImportError:
        try:
            from ensemble_models import HAS_XGBOOST, HAS_LIGHTGBM, HAS_CATBOOST, get_all_models
        except Exception:
            pass
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIBRARY STATUS
    # ═══════════════════════════════════════════════════════════════════════════
    
    with st.expander("📦 ML Libraries Status", expanded=bool(TORCH_ERROR)):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("XGBoost", "✅" if HAS_XGBOOST else "❌")
        col2.metric("LightGBM", "✅" if HAS_LIGHTGBM else "❌")
        col3.metric("CatBoost", "✅" if HAS_CATBOOST else "❌")
        col4.metric("PyTorch", "✅" if HAS_TORCH else "❌")
        
        # Show PyTorch error if any
        if TORCH_ERROR:
            st.error(f"⚠️ PyTorch Error: {TORCH_ERROR}")
            st.write("**To fix PyTorch on Windows:**")
            st.write("**Step 1:** Uninstall existing PyTorch")
            st.code("pip uninstall torch torchvision torchaudio -y", language="bash")
            st.write("**Step 2:** Reinstall (choose ONE):")
            st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", language="bash")
            st.write("OR with CUDA (NVIDIA GPU):")
            st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", language="bash")
            st.write("**Step 3:** Install Visual C++ Redistributable:")
            st.write("Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        
        if not all([HAS_XGBOOST, HAS_LIGHTGBM, HAS_CATBOOST]):
            st.code("pip install xgboost lightgbm catboost", language="bash")
        
        models = get_all_models()
        st.write(f"**Traditional ML:** {len(models)} models available")
        st.write("**Pattern Detection:** CNN, LSTM, Hybrid architectures" if HAS_TORCH else "**Pattern Detection:** ❌ Requires PyTorch")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MODEL STATUS - CARD-BASED DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.subheader("📊 Current Model Status")
    
    trad_status = get_model_status()
    pattern_status = get_pattern_model_status()
    
    def get_combined_score(trad_f1, pattern_acc):
        """Calculate combined score (weighted average)"""
        if trad_f1 > 0 and pattern_acc > 0:
            return trad_f1 * 0.6 + pattern_acc * 0.4
        elif trad_f1 > 0:
            return trad_f1
        elif pattern_acc > 0:
            return pattern_acc
        return 0
    
    def render_model_cards(market_key):
        """Render 4 model cards for a market with expandable details"""
        cols = st.columns(4)
        
        mode_names = {
            'scalp': '⚡ Scalp',
            'daytrade': '📊 Day Trade',
            'swing': '📈 Swing',
            'investment': '🏦 Investment'
        }
        
        for i, mode in enumerate(['scalp', 'daytrade', 'swing', 'investment']):
            # Traditional ML
            trad_key = f"{mode}_{market_key}"
            trad_info = trad_status.get(trad_key, {})
            trad_trained = trad_info.get('trained', False)
            trad_f1 = trad_info.get('f1_score', 0)
            trad_samples = trad_info.get('n_samples', 0)
            trad_date = trad_info.get('date', '')
            
            # Pattern Detection
            pattern_key = f"{mode}_{market_key}"
            pattern_info = pattern_status.get(pattern_key, {})
            pattern_trained = pattern_info.get('trained', False)
            pattern_acc = pattern_info.get('accuracy', 0)
            pattern_date = pattern_info.get('date', '')
            
            # Combined score
            combined = get_combined_score(trad_f1, pattern_acc)
            
            # Last trained
            last_trained = trad_date or pattern_date or "—"
            
            with cols[i]:
                with st.container(border=True):
                    st.caption(f"{market_key.upper()}")
                    st.subheader(mode_names[mode])
                    
                    # Traditional ML Score
                    if trad_f1 > 0:
                        st.metric("Traditional ML (F1)", f"{trad_f1:.1f}%")
                    else:
                        st.metric("Traditional ML (F1)", "—")
                    
                    # Pattern Detection Score
                    if pattern_acc > 0:
                        st.metric("Pattern Detection", f"{pattern_acc:.1f}%")
                    else:
                        st.metric("Pattern Detection", "—")
                    
                    # Combined Score
                    if combined > 0:
                        st.metric("Combined", f"{combined:.1f}%")
                    else:
                        st.metric("Combined", "—")
                    
                    st.divider()
                    st.caption(f"Samples: {trad_samples:,}" if trad_samples else "Samples: —")
                    st.caption(f"Trained: {last_trained}")
                    
                    # Expandable details
                    if trad_trained:
                        with st.expander("🔍 Per-Label Details"):
                            details = get_detailed_metrics(mode, market_key)
                            per_label = details.get('per_label', {})
                            best_models = details.get('best_models', {})
                            
                            if per_label:
                                for label, metrics in per_label.items():
                                    f1 = metrics.get('f1', 0)
                                    if f1 < 1:
                                        f1 *= 100
                                    prec = metrics.get('precision', 0)
                                    if prec < 1:
                                        prec *= 100
                                    recall = metrics.get('recall', 0)
                                    if recall < 1:
                                        recall *= 100
                                    pos_rate = metrics.get('positive_rate', 0)
                                    if pos_rate < 1:
                                        pos_rate *= 100
                                    
                                    # Color code F1
                                    if f1 >= 60:
                                        color = "🟢"
                                    elif f1 >= 45:
                                        color = "🟡"
                                    else:
                                        color = "🔴"
                                    
                                    best_model = best_models.get(label, "—")
                                    
                                    st.markdown(f"**{label}** {color}")
                                    st.caption(f"F1: {f1:.1f}% | Prec: {prec:.1f}% | Rec: {recall:.1f}%")
                                    st.caption(f"Pos Rate: {pos_rate:.1f}% | Model: {best_model}")
                                    st.markdown("---")
                                
                                # Show Phase Model Status (NEW!)
                                phase_trained = details.get('phase_trained', False)
                                phase_metrics = details.get('phase_metrics', {})
                                
                                st.markdown("**🔮 Phase Model (Wyckoff)**")
                                if phase_trained and phase_metrics:
                                    phase_f1 = phase_metrics.get('f1_macro', 0)
                                    if phase_f1 < 1:
                                        phase_f1 *= 100
                                    phase_acc = phase_metrics.get('accuracy', 0)
                                    if phase_acc < 1:
                                        phase_acc *= 100
                                    st.caption(f"✅ Trained | F1: {phase_f1:.1f}% | Acc: {phase_acc:.1f}%")
                                    st.caption("Shows as 'Phase (ML)' in analysis")
                                else:
                                    st.caption("❌ Not trained - uses 'Money Flow' rules")
                            else:
                                st.caption("No detailed metrics available")
    
    # Create tabs for each market
    tab_crypto, tab_stocks, tab_etfs = st.tabs(["🪙 CRYPTO", "📈 STOCKS", "📊 ETFs"])
    
    with tab_crypto:
        render_model_cards("crypto")
    
    with tab_stocks:
        render_model_cards("stock")
    
    with tab_etfs:
        render_model_cards("etf")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAINING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.subheader("🎯 Train Models")
    
    # Row 1: Mode and Market
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.selectbox(
            "Trading Mode",
            ['scalp', 'daytrade', 'swing', 'investment'],
            format_func=lambda x: {
                'scalp': '⚡ SCALP - Quick patterns (Flags, Pennants)',
                'daytrade': '📊 DAYTRADE - All patterns',
                'swing': '📈 SWING - Major patterns (H&S, Wedges)',
                'investment': '🏦 INVESTMENT - Trend patterns'
            }.get(x, x.upper())
        )
    
    with col2:
        market = st.selectbox(
            "Market",
            ['crypto', 'stocks', 'etfs'],
            format_func=lambda x: {
                'crypto': '₿ Crypto',
                'stocks': '📈 Stocks', 
                'etfs': '📊 ETFs'
            }.get(x, x.upper())
        )
    
    # Mode info
    timeframe = MODE_TIMEFRAMES[mode]
    default_days = MODE_TRAINING_DAYS[mode]
    
    st.info(f"**{mode.upper()}:** {MODE_DESCRIPTIONS[mode]} | Timeframe: {timeframe} | Recommended: {default_days} days")
    
    # Symbols
    default_syms = DEFAULT_CRYPTO if market == 'crypto' else DEFAULT_STOCKS if market == 'stocks' else DEFAULT_ETFS
    
    with st.expander("📋 Training Symbols", expanded=True):
        symbols_input = st.text_area(
            "Symbols (one per line)", 
            value='\n'.join(default_syms), 
            height=150,
            label_visibility='collapsed'
        )
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        st.caption(f"{len(symbols)} symbols selected")
    
    # Training days - allow up to 5 years for Investment mode
    max_days = 1825 if mode == 'investment' else 1095 if mode == 'swing' else 730
    days = st.slider("Days of History", 30, max_days, min(default_days, max_days), help="More days = better coverage of bull+bear cycles. Investment: 3-5 years recommended. Swing: 2-3 years.")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WHAT TO TRAIN
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.subheader("🔧 What to Train")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_traditional = st.checkbox(
            "📊 Traditional ML (32 models)",
            value=True,
            help="GradientBoosting, XGBoost, LightGBM, RandomForest, Neural Networks, etc."
        )
        if train_traditional:
            st.caption(f"Labels: {', '.join(MODE_LABELS[mode]['labels'])}")
            st.caption("⏱️ ~2-5 minutes")
    
    with col2:
        train_patterns = st.checkbox(
            "🎯 Pattern Detection (Deep Learning)",
            value=HAS_TORCH,
            disabled=not HAS_TORCH,
            help="CNN/LSTM to detect Double Top, H&S, Flags, Triangles, etc."
        )
        if train_patterns:
            pattern_model_type = st.selectbox(
                "Architecture",
                ['hybrid', 'cnn', 'lstm'],
                format_func=lambda x: {
                    'hybrid': '🔀 Hybrid (Best)',
                    'cnn': '🖼️ CNN',
                    'lstm': '📊 LSTM'
                }.get(x, x),
                label_visibility='collapsed'
            )
            st.caption("⏱️ ~5-10 minutes")
        else:
            pattern_model_type = 'hybrid'
        
        if not HAS_TORCH:
            st.warning("⚠️ PyTorch not available")
            st.code("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", language="bash")
    
    # Show what patterns will be trained
    if train_patterns and HAS_TORCH:
        with st.expander("🏷️ Patterns for this mode"):
            patterns = MODE_PATTERNS.get(mode, [])
            cols = st.columns(3)
            for i, p in enumerate(patterns):
                dir_val = PATTERN_DIRECTION.get(p, "NEUTRAL")
                emoji = '🟢' if dir_val == 'BULLISH' else '🔴' if dir_val == 'BEARISH' else '⚪'
                cols[i % 3].write(f"{emoji} {p.value.replace('_', ' ').title()}")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRAIN BUTTON
    # ═══════════════════════════════════════════════════════════════════════════
    
    if not train_traditional and not train_patterns:
        st.warning("Select at least one model type to train")
        train_disabled = True
    elif len(symbols) < 3:
        st.warning("Select at least 3 symbols")
        train_disabled = True
    else:
        train_disabled = False
    
    if st.button(
        f"🚀 Train {mode.upper()} Models ({market.upper()})",
        type="primary",
        use_container_width=True,
        disabled=train_disabled
    ):
        # Track success of each training step
        traditional_success = False
        pattern_success = False
        
        # Train Traditional ML
        if train_traditional:
            st.markdown("---")
            # Get actual model count
            try:
                from ml.ensemble_models import get_all_models
                actual_count = len(get_all_models())
            except:
                actual_count = 32  # Default
            st.subheader(f"📊 Training Traditional ML ({actual_count} Models)")
            train_model(mode, timeframe, market.capitalize() if market != 'etfs' else 'ETFs', symbols, days)
            traditional_success = True  # Assume success if no exception
        
        # Train Pattern Detection
        if train_patterns and HAS_TORCH:
            st.markdown("---")
            st.subheader("🎯 Training Pattern Detection (Deep Learning)")
            pattern_success = train_pattern_model(mode, market, pattern_model_type, symbols, timeframe, days)
        
        # Summary - only show if at least one thing succeeded
        st.markdown("---")
        
        trained_items = []
        if train_traditional and traditional_success:
            trained_items.append("Traditional ML")
        if train_patterns and pattern_success:
            trained_items.append("Pattern Detection")
        
        if trained_items:
            st.success("✅ Training Complete!")
            st.write(f"**Trained:** {', '.join(trained_items)} for {mode.upper()} / {market.upper()}")
            st.balloons()
        else:
            st.warning("⚠️ No models were trained successfully. Check errors above.")



def render_training_ui():
    """Wrapper for compatibility"""
    render_training_page()


def render_feature_info():
    st.write(f"The model uses {NUM_ENHANCED_FEATURES} features across 4 categories.")


def main():
    st.set_page_config(page_title="ML Training", page_icon="🧠", layout="wide")
    render_training_page()


if __name__ == "__main__":
    main()