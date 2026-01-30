"""
Test script to verify BCHUSDT sweep detection on Jan 24, 2026
Target: Price reached ~586, should detect LONG sweep with ENTER
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timedelta
from core.data_fetcher import fetch_binance_klines
from liquidity_hunter.liquidity_hunter import (
    find_liquidity_levels,
    detect_sweep,
    calculate_entry_quality,
    get_binance_liquidations,
    normalize_columns
)
import numpy as np

def test_bch_sweep():
    print("=" * 70)
    print("BCHUSDT SWEEP DETECTION TEST - Jan 24, 2026")
    print("=" * 70)

    # Fetch data
    symbol = 'BCHUSDT'
    timeframe = '4h'

    print(f"\n1. Fetching {symbol} {timeframe} data...")
    df = fetch_binance_klines(symbol, interval=timeframe, limit=200)

    if df is None or len(df) < 50:
        print("ERROR: Could not fetch data")
        return

    df = normalize_columns(df)
    print(f"   Got {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Show recent candles
    print(f"\n2. Recent price action:")
    recent = df.tail(10)[['open', 'high', 'low', 'close', 'volume']]
    for idx, row in recent.iterrows():
        print(f"   {idx}: O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f}")

    current_price = float(df['close'].iloc[-1])
    print(f"\n   Current price: ${current_price:.2f}")

    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    volumes = df['volume']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    print(f"   ATR(14): ${atr:.2f}")

    # Find liquidity levels
    print(f"\n3. Finding liquidity levels...")
    liquidity_levels = find_liquidity_levels(df, lookback=50)

    print(f"\n   LOWS (potential LONG targets):")
    for level in liquidity_levels.get('lows', [])[:5]:
        print(f"   - ${level['price']:.2f} (touches: {level.get('touches', 1)}, type: {level.get('type', 'swing')})")

    print(f"\n   EQUAL LOWS (high priority):")
    for level in liquidity_levels.get('equal_lows', [])[:5]:
        print(f"   - ${level['price']:.2f} (touches: {level.get('touches', 2)})")

    print(f"\n   HIGHS (potential SHORT targets):")
    for level in liquidity_levels.get('highs', [])[:5]:
        print(f"   - ${level['price']:.2f} (touches: {level.get('touches', 1)}, type: {level.get('type', 'swing')})")

    # Get whale data
    print(f"\n4. Getting whale data...")
    whale_data = get_binance_liquidations(symbol)
    whale_pct = whale_data.get('whale_long', 50)
    print(f"   Whale long %: {whale_pct}%")

    # Detect sweep
    print(f"\n5. Detecting sweep...")
    lookback_candles = 40  # swing mode
    sweep_status = detect_sweep(df, liquidity_levels, atr, lookback_candles, whale_pct)

    if sweep_status.get('detected'):
        print(f"\n   ✅ SWEEP DETECTED!")
        print(f"   Direction: {sweep_status.get('direction')}")
        print(f"   Level price: ${sweep_status.get('level_swept', 0):.2f}")
        print(f"   Level type: {sweep_status.get('level_type', 'unknown')}")
        print(f"   Candles ago: {sweep_status.get('candles_ago', 0)}")
        print(f"   Confidence: {sweep_status.get('confidence', 0)}%")
        print(f"   Source: {sweep_status.get('source', 'unknown')}")

        # Price Action Analysis
        if sweep_status.get('price_action'):
            pa = sweep_status['price_action']
            pred = pa.get('prediction', {})
            print(f"\n   📊 PRICE ACTION ANALYSIS:")
            print(f"   PA Score: {pred.get('score', 0)}/100")
            print(f"   Prediction: {pred.get('prediction', 'UNKNOWN')}")
            print(f"   Confidence: {pred.get('confidence', 'LOW')}")
            print(f"   Action: {pred.get('action', 'WAIT')}")

            if pred.get('key_signals'):
                print(f"\n   Key Signals:")
                for sig in pred.get('key_signals', []):
                    print(f"   ✅ {sig}")

            if pred.get('warnings'):
                print(f"\n   PA Warnings:")
                for w in pred.get('warnings', []):
                    print(f"   ⚠️ {w}")

            # Component breakdown
            comp = pred.get('component_scores', {})
            if comp:
                print(f"\n   Component Scores:")
                print(f"   - Candle Pattern: {comp.get('candle_pattern', 0)}")
                print(f"   - Structure: {comp.get('structure', 0)}")
                print(f"   - Order Blocks: {comp.get('order_blocks', 0)}")
                print(f"   - FVG: {comp.get('fvg', 0)}")
                print(f"   - Volume: {comp.get('volume', 0)}")
                print(f"   - Momentum: {comp.get('momentum', 0)}")

        # Calculate entry quality
        print(f"\n6. Calculating entry quality...")
        candles_ago = sweep_status.get('candles_ago', 1)
        sweep_candle_idx = len(df) - candles_ago
        sweep_volume = float(volumes.iloc[sweep_candle_idx]) if 0 <= sweep_candle_idx < len(df) else float(volumes.iloc[-1])
        avg_volume = float(volumes.rolling(20).mean().iloc[-1])
        whale_delta = (whale_pct - 50) / 5

        entry_quality = calculate_entry_quality(
            sweep_status=sweep_status,
            current_price=current_price,
            whale_pct=whale_pct,
            whale_delta=whale_delta,
            volume_on_sweep=sweep_volume,
            avg_volume=avg_volume,
            df=df
        )

        print(f"\n   Entry Quality Score: {entry_quality.get('score', 0)}")
        print(f"   Grade: {entry_quality.get('grade', 'N/A')}")
        print(f"   Entry Window: {entry_quality.get('entry_window', 'N/A')}")
        print(f"   RECOMMENDATION: {entry_quality.get('recommendation', 'UNKNOWN')}")

        print(f"\n   Components:")
        components = entry_quality.get('components', {})
        for key, val in components.items():
            print(f"   - {key}: {val}")

        print(f"\n   Warnings:")
        for w in entry_quality.get('warnings', []):
            print(f"   - {w}")

    else:
        print(f"\n   ❌ NO SWEEP DETECTED")
        print(f"   Reason: {sweep_status.get('reason', 'No levels broken')}")

        # Show what levels are nearby
        print(f"\n   Nearby levels to watch:")
        all_lows = liquidity_levels.get('lows', []) + liquidity_levels.get('equal_lows', [])
        for level in sorted(all_lows, key=lambda x: abs(x['price'] - current_price))[:3]:
            dist = (current_price - level['price']) / current_price * 100
            print(f"   - ${level['price']:.2f} ({dist:+.1f}% away)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_bch_sweep()
