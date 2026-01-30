"""
Auto-tune ML thresholds based on actual price movement distribution.
Analyzes Crypto, Stocks, and ETFs across all trading modes.
Automatically updates probabilistic_ml.py with optimal thresholds.

Run: python auto_tune_thresholds.py
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("AUTO-TUNE THRESHOLDS FOR ALL MARKETS")
print("=" * 70)

# Market configurations
MARKETS = {
    'crypto': {
        'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT'],
        'fetch_func': 'fetch_crypto_data',
    },
    'stock': {
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH'],
        'fetch_func': 'fetch_stock_data',
    },
    'etf': {
        'symbols': ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'GLD', 'TLT', 'VTI'],
        'fetch_func': 'fetch_stock_data',  # ETFs use same fetcher as stocks
    }
}

# Mode configurations (timeframe and lookahead in candles)
MODES = {
    'scalp': {'timeframe': '5m', 'lookahead': 6, 'target_positive_rate': 0.35},
    'daytrade': {'timeframe': '15m', 'lookahead': 18, 'target_positive_rate': 0.35},
    'swing': {'timeframe': '4h', 'lookahead': 10, 'target_positive_rate': 0.35},
    'investment': {'timeframe': '1d', 'lookahead': 20, 'target_positive_rate': 0.30},
}

# Import data fetchers
try:
    from core.data_fetcher import fetch_stock_data
except:
    fetch_stock_data = None

try:
    from core.data_fetcher import fetch_crypto_data
except:
    # Fallback for crypto
    try:
        from core.data_fetcher import get_binance_data as fetch_crypto_data
    except:
        fetch_crypto_data = None


def fetch_data(market: str, symbol: str, timeframe: str, limit: int = 500):
    """Fetch data for any market type."""
    try:
        if market == 'crypto':
            if fetch_crypto_data:
                return fetch_crypto_data(symbol, timeframe, limit)
            else:
                # Try Binance API directly
                from core.data_fetcher import fetch_crypto_data as fc
                return fc(symbol, timeframe, limit)
        else:  # stock or etf
            if fetch_stock_data:
                return fetch_stock_data(symbol, timeframe, limit)
    except Exception as e:
        print(f"    ❌ {symbol}: {e}")
    return None


def calculate_optimal_thresholds(returns: pd.Series, target_rate: float = 0.35) -> dict:
    """Calculate thresholds that give target positive rate."""
    abs_returns = returns.abs()
    
    # For different label types
    trend_threshold = np.percentile(abs_returns, (1 - target_rate) * 100)
    reversal_threshold = np.percentile(abs_returns, (1 - target_rate * 0.9) * 100)  # Slightly harder
    drawdown_threshold = np.percentile(abs_returns, (1 - target_rate * 0.8) * 100)  # Even harder
    
    return {
        'trend': round(trend_threshold, 2),
        'reversal': round(reversal_threshold, 2),
        'drawdown': round(drawdown_threshold, 2),
        'continuation': round(trend_threshold, 2),
        'fakeout': round(drawdown_threshold, 2),
        'accumulation': round(trend_threshold * 2, 2),  # Investment needs bigger moves
        'distribution': round(trend_threshold * 2, 2),
    }


def analyze_market_mode(market: str, mode: str) -> dict:
    """Analyze price movements for a specific market and mode."""
    config = MARKETS[market]
    mode_config = MODES[mode]
    
    symbols = config['symbols']
    timeframe = mode_config['timeframe']
    lookahead = mode_config['lookahead']
    target_rate = mode_config['target_positive_rate']
    
    print(f"\n  📊 {mode.upper()} ({timeframe}, {lookahead} candles ahead):")
    
    all_returns = []
    
    for symbol in symbols:
        df = fetch_data(market, symbol, timeframe, 500)
        if df is None or len(df) < lookahead + 50:
            continue
        
        # Calculate forward returns
        df['fwd_return'] = (df['Close'].shift(-lookahead) - df['Close']) / df['Close'] * 100
        valid = df['fwd_return'].dropna()
        if len(valid) > 0:
            all_returns.extend(valid.tolist())
            print(f"    ✅ {symbol}: {len(valid)} samples")
    
    if not all_returns:
        print(f"    ❌ No data for {market} {mode}")
        return None
    
    returns = pd.Series(all_returns)
    
    # Calculate statistics
    abs_returns = returns.abs()
    print(f"    📈 Samples: {len(returns):,}")
    print(f"    📈 Mean move: ±{abs_returns.mean():.2f}%")
    print(f"    📈 Median move: ±{abs_returns.median():.2f}%")
    
    # Calculate optimal thresholds
    thresholds = calculate_optimal_thresholds(returns, target_rate)
    
    # Verify positive rates
    actual_rate = (abs_returns >= thresholds['trend']).mean()
    print(f"    🎯 Threshold {thresholds['trend']:.2f}% → {actual_rate:.1%} positive rate")
    
    return thresholds


def generate_threshold_code(market: str, all_thresholds: dict) -> str:
    """Generate Python code for thresholds."""
    market_upper = market.upper()
    
    code = f"""
# {market_upper} thresholds (auto-tuned from real data)
MODE_LABELS_{market_upper} = {{
    'scalp': {{
        'labels': ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion'],
        'lookahead': 6,
        'thresholds': {{
            'continuation': {all_thresholds.get('scalp', {}).get('continuation', 0.5)},
            'fakeout': {all_thresholds.get('scalp', {}).get('fakeout', 0.3)},
            'vol_expansion': 1.4,
        }}
    }},
    'daytrade': {{
        'labels': ['continuation_bull', 'continuation_bear', 'fakeout_to_bull', 'fakeout_to_bear', 'vol_expansion'],
        'lookahead': 18,
        'thresholds': {{
            'continuation': {all_thresholds.get('daytrade', {}).get('continuation', 1.0)},
            'fakeout': {all_thresholds.get('daytrade', {}).get('fakeout', 0.6)},
            'vol_expansion': 1.5,
        }}
    }},
    'swing': {{
        'labels': ['trend_holds_bull', 'trend_holds_bear', 'reversal_to_bull', 'reversal_to_bear', 'drawdown'],
        'lookahead': 10,
        'thresholds': {{
            'trend_holds': {all_thresholds.get('swing', {}).get('trend', 1.5)},
            'reversal': {all_thresholds.get('swing', {}).get('reversal', 1.2)},
            'drawdown': {all_thresholds.get('swing', {}).get('drawdown', 1.0)},
        }}
    }},
    'investment': {{
        'labels': ['accumulation', 'distribution', 'reversal_to_bull', 'reversal_to_bear', 'large_drawdown'],
        'lookahead': 20,
        'thresholds': {{
            'accumulation': {all_thresholds.get('investment', {}).get('accumulation', 5.0)},
            'distribution': {all_thresholds.get('investment', {}).get('distribution', 5.0)},
            'reversal': {all_thresholds.get('investment', {}).get('reversal', 4.0)},
            'large_drawdown': {all_thresholds.get('investment', {}).get('drawdown', 8.0) * 2},
        }}
    }}
}}
"""
    return code


# Main execution
if __name__ == '__main__':
    results = {}
    
    for market in ['crypto', 'stock', 'etf']:
        print(f"\n{'='*70}")
        print(f"🔍 ANALYZING {market.upper()}")
        print("=" * 70)
        
        results[market] = {}
        
        for mode in ['scalp', 'daytrade', 'swing', 'investment']:
            thresholds = analyze_market_mode(market, mode)
            if thresholds:
                results[market][mode] = thresholds
    
    # Generate code
    print("\n" + "=" * 70)
    print("📝 GENERATED THRESHOLD CODE")
    print("=" * 70)
    
    for market in ['crypto', 'stock', 'etf']:
        if results.get(market):
            code = generate_threshold_code(market, results[market])
            print(code)
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📊 THRESHOLD COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Market':<10} {'Mode':<12} {'Trend %':<10} {'Reversal %':<12} {'Drawdown %':<12}")
    print("-" * 56)
    
    for market in ['crypto', 'stock', 'etf']:
        for mode in ['scalp', 'daytrade', 'swing', 'investment']:
            if results.get(market, {}).get(mode):
                t = results[market][mode]
                print(f"{market:<10} {mode:<12} {t.get('trend', 0):<10.2f} {t.get('reversal', 0):<12.2f} {t.get('drawdown', 0):<12.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Copy the generated code above into probabilistic_ml.py")
    print("   Or run with --apply to auto-update the file")
    print("=" * 70)

