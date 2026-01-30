"""
Debug script to test CONTINUATION mode direction flip.
Run this to verify the scanner is producing correct trade_direction.
"""

import sys
sys.path.insert(0, '.')

from liquidity_hunter.liquidity_hunter import (
    TRADE_STRATEGY_MODE,
    scan_for_liquidity_setups,
    full_liquidity_analysis
)
from core.data_fetcher import fetch_binance_klines

print("=" * 60)
print(f"TRADE_STRATEGY_MODE: {TRADE_STRATEGY_MODE}")
print("=" * 60)

# Test symbol - use same timeframe as app (4h for liquidity scanner)
TEST_SYMBOL = "IQUSDT"
TIMEFRAME = "4h"  # App uses 4h for liquidity detection

print(f"\n🔍 Testing {TEST_SYMBOL}...")

# 1. Test via scanner
print("\n--- SCANNER TEST ---")

def fetch_data(symbol, tf):
    return fetch_binance_klines(symbol, tf, 100)

results = scan_for_liquidity_setups(
    symbols=[TEST_SYMBOL],
    fetch_data_func=fetch_data,
    timeframe=TIMEFRAME,
    trading_mode='day_trade',
    max_symbols=1
)

if results:
    r = results[0]
    sweep = r.get('sweep', {})
    raw_dir = sweep.get('direction', 'N/A') if sweep else 'N/A'
    trade_dir = r.get('trade_direction', 'N/A')

    print(f"  Symbol: {r.get('symbol')}")
    print(f"  Status: {r.get('status')}")
    print(f"  Sweep detected: {sweep.get('detected', False) if sweep else False}")
    print(f"  RAW direction (what was swept): {raw_dir}")
    print(f"  TRADE direction (what we trade): {trade_dir}")
    print(f"  Entry recommendation: {r.get('entry_recommendation')}")
    print(f"  ML probability: {r.get('ml_probability')}")

    # Check if flip is correct
    if TRADE_STRATEGY_MODE == "CONTINUATION" and raw_dir in ['LONG', 'SHORT']:
        expected_trade = "SHORT" if raw_dir == "LONG" else "LONG"
        if trade_dir == expected_trade:
            print(f"  ✅ Direction flip CORRECT: {raw_dir} → {trade_dir}")
        else:
            print(f"  ❌ Direction flip WRONG: {raw_dir} should → {expected_trade}, got {trade_dir}")
else:
    print("  No results from scanner")

# 2. Test via single analysis
print("\n--- SINGLE ANALYSIS TEST ---")

df = fetch_binance_klines(TEST_SYMBOL, TIMEFRAME, 100)
if df is not None and len(df) > 0:
    analysis = full_liquidity_analysis(TEST_SYMBOL, df, whale_pct=50, trading_mode='day_trade')

    sweep_status = analysis.get('sweep_status', {})
    trade_plan = analysis.get('trade_plan', {})

    raw_dir_single = sweep_status.get('direction', 'N/A')
    trade_dir_single = trade_plan.get('direction', 'N/A')

    print(f"  Symbol: {TEST_SYMBOL}")
    print(f"  Sweep detected: {sweep_status.get('detected', False)}")
    print(f"  RAW direction (from sweep_status): {raw_dir_single}")
    print(f"  TRADE direction (from trade_plan): {trade_dir_single}")
    print(f"  Entry recommendation: {trade_plan.get('entry_recommendation', 'N/A')}")

    # Check if flip is correct
    if TRADE_STRATEGY_MODE == "CONTINUATION" and raw_dir_single in ['LONG', 'SHORT']:
        expected_trade = "SHORT" if raw_dir_single == "LONG" else "LONG"
        if trade_dir_single == expected_trade:
            print(f"  ✅ Direction flip CORRECT: {raw_dir_single} → {trade_dir_single}")
        else:
            print(f"  ❌ Direction flip WRONG: {raw_dir_single} should → {expected_trade}, got {trade_dir_single}")
else:
    print("  Failed to fetch data")

# 3. Compare
print("\n--- COMPARISON ---")
if results and df is not None:
    scanner_trade = results[0].get('trade_direction', 'N/A')
    single_trade = trade_plan.get('direction', 'N/A')

    if scanner_trade == single_trade:
        print(f"✅ MATCH: Scanner ({scanner_trade}) == Single Analysis ({single_trade})")
    else:
        print(f"❌ MISMATCH: Scanner ({scanner_trade}) != Single Analysis ({single_trade})")
        print("\n⚠️  This is the bug we need to fix!")

print("\n" + "=" * 60)
