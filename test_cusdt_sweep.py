"""
Test script to verify CUSDT sweep detection on Jan 25, 2026
Test times (Abu Dhabi UTC+4):
- 1:30 AM Abu Dhabi = 21:30 UTC Jan 24, 2026
- 13:30 PM Abu Dhabi = 09:30 UTC Jan 25, 2026
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from core.data_fetcher import fetch_binance_klines
from liquidity_hunter.liquidity_hunter import (
    find_liquidity_levels,
    detect_sweep,
    calculate_entry_quality,
    get_binance_liquidations,
    normalize_columns,
    full_liquidity_analysis
)
from app import get_whale_delta
import numpy as np


def fetch_klines_with_timestamps(symbol, interval='1h', limit=300):
    """Fetch klines with proper timestamps from Binance"""
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert to proper types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Set index to open_time
            df.set_index('open_time', inplace=True)

            return df
    except Exception as e:
        print(f"Error fetching klines: {e}")

    return None

# Abu Dhabi is UTC+4
ABU_DHABI_TZ_OFFSET = timedelta(hours=4)

def get_abu_dhabi_time(utc_time):
    """Convert UTC to Abu Dhabi time"""
    return utc_time + ABU_DHABI_TZ_OFFSET

def test_cusdt_at_time(symbol, timeframe, target_time_utc, label):
    """Test sweep detection at a specific time"""
    print("\n" + "=" * 70)
    print(f"TEST: {symbol} at {label}")
    print(f"Target UTC: {target_time_utc}")
    print(f"Target Abu Dhabi: {get_abu_dhabi_time(target_time_utc)}")
    print("=" * 70)

    # Fetch data with proper timestamps
    print(f"\n1. Fetching {symbol} {timeframe} data...")
    df = fetch_klines_with_timestamps(symbol, interval=timeframe, limit=300)

    if df is None or len(df) < 50:
        print(f"ERROR: Could not fetch data for {symbol}")
        return None

    df = normalize_columns(df)
    print(f"   Got {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Find the candle closest to target time
    target_idx = None
    for i, idx in enumerate(df.index):
        candle_time = idx.to_pydatetime().replace(tzinfo=None)
        if candle_time <= target_time_utc:
            target_idx = i

    if target_idx is None:
        print(f"   Target time {target_time_utc} not found in data range")
        print(f"   Data starts at {df.index[0]}")
        # Use most recent data instead
        target_idx = len(df) - 1
        print(f"   Using most recent candle instead")
    else:
        actual_time = df.index[target_idx]
        abu_dhabi = get_abu_dhabi_time(actual_time.to_pydatetime().replace(tzinfo=None))
        print(f"   Found candle at: {actual_time} UTC ({abu_dhabi.strftime('%H:%M')} Abu Dhabi)")

    # Use data up to target time for analysis
    df_at_time = df.iloc[:target_idx + 1].copy()

    if len(df_at_time) < 50:
        print(f"   Not enough data, using full dataset")
        df_at_time = df.copy()

    # Show recent candles at that time
    print(f"\n2. Price action at target time:")
    recent = df_at_time.tail(10)[['open', 'high', 'low', 'close', 'volume']]
    for idx, row in recent.iterrows():
        abu_dhabi = get_abu_dhabi_time(idx.to_pydatetime().replace(tzinfo=None))
        time_str = abu_dhabi.strftime('%m-%d %H:%M')
        print(f"   {time_str} AD: O={row['open']:.6f} H={row['high']:.6f} L={row['low']:.6f} C={row['close']:.6f}")

    current_price = float(df_at_time['close'].iloc[-1])
    print(f"\n   Price at target time: ${current_price:.6f}")

    # Calculate ATR
    high = df_at_time['high']
    low = df_at_time['low']
    close = df_at_time['close']
    volumes = df_at_time['volume']
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    print(f"   ATR(14): ${atr:.6f}")

    # Get whale data
    print(f"\n3. Getting whale data...")
    whale_data = get_binance_liquidations(symbol)
    whale_pct = whale_data.get('whale_long', 50)
    print(f"   Whale long %: {whale_pct}%")

    # Get whale delta
    whale_delta_data = get_whale_delta(symbol, 'swing')
    whale_delta_val = 0
    whale_delta_4h = None
    whale_delta_24h = None
    whale_delta_7d = None
    whale_acceleration = None
    whale_early_signal = None
    is_fresh_accumulation = False

    if whale_delta_data and whale_delta_data.get('data_available'):
        whale_delta_val = whale_delta_data.get('whale_delta_24h') or whale_delta_data.get('whale_delta', 0)
        whale_delta_4h = whale_delta_data.get('whale_delta_4h')
        whale_delta_24h = whale_delta_data.get('whale_delta_24h')
        whale_delta_7d = whale_delta_data.get('whale_delta_7d')
        whale_acceleration = whale_delta_data.get('whale_acceleration')
        whale_early_signal = whale_delta_data.get('whale_early_signal')
        is_fresh_accumulation = whale_delta_data.get('is_fresh_accumulation', False)
        print(f"   Whale Delta 4h: {whale_delta_4h:+.1f}%" if whale_delta_4h else "   Whale Delta 4h: N/A")
        print(f"   Whale Delta 24h: {whale_delta_24h:+.1f}%" if whale_delta_24h else "   Whale Delta 24h: N/A")
        print(f"   Whale Delta 7d: {whale_delta_7d:+.1f}%" if whale_delta_7d else "   Whale Delta 7d: N/A")
        print(f"   Acceleration: {whale_acceleration}")
        print(f"   Early Signal: {whale_early_signal}")

    # Find liquidity levels
    print(f"\n4. Finding liquidity levels...")
    liquidity_levels = find_liquidity_levels(df_at_time, lookback=50)

    print(f"\n   LOWS (potential LONG targets after sweep):")
    for level in liquidity_levels.get('lows', [])[:5]:
        dist_pct = (level['price'] - current_price) / current_price * 100
        print(f"   - ${level['price']:.6f} ({dist_pct:+.1f}%, touches: {level.get('touches', 1)}, type: {level.get('type', 'swing')})")

    print(f"\n   EQUAL LOWS (high priority):")
    for level in liquidity_levels.get('equal_lows', [])[:5]:
        dist_pct = (level['price'] - current_price) / current_price * 100
        print(f"   - ${level['price']:.6f} ({dist_pct:+.1f}%, touches: {level.get('touches', 2)})")

    print(f"\n   HIGHS (potential SHORT targets after sweep):")
    for level in liquidity_levels.get('highs', [])[:5]:
        dist_pct = (level['price'] - current_price) / current_price * 100
        print(f"   - ${level['price']:.6f} ({dist_pct:+.1f}%, touches: {level.get('touches', 1)}, type: {level.get('type', 'swing')})")

    # Detect sweep
    print(f"\n5. Detecting sweep...")
    lookback_candles = 40  # swing mode
    sweep_status = detect_sweep(df_at_time, liquidity_levels, atr, lookback_candles, whale_pct)

    if sweep_status.get('detected'):
        print(f"\n   ✅ SWEEP DETECTED!")
        print(f"   Direction: {sweep_status.get('direction')}")
        print(f"   Level price: ${sweep_status.get('level_swept', 0):.6f}")
        print(f"   Level type: {sweep_status.get('level_type', 'unknown')}")
        print(f"   Candles ago: {sweep_status.get('candles_ago', 0)}")
        print(f"   Confidence: {sweep_status.get('confidence', 0)}%")

        # CONTINUATION mode: Sweep direction → Trade OPPOSITE
        sweep_dir = sweep_status.get('direction', '')
        trade_dir = 'SHORT' if sweep_dir == 'LONG' else 'LONG' if sweep_dir == 'SHORT' else 'UNKNOWN'
        print(f"\n   🎯 CONTINUATION MODE:")
        print(f"   Sweep of {sweep_dir} → Trade {trade_dir}")

        # Calculate entry quality with ML
        print(f"\n6. Calculating entry quality (ML)...")
        candles_ago = sweep_status.get('candles_ago', 1)
        sweep_candle_idx = len(df_at_time) - candles_ago
        sweep_volume = float(volumes.iloc[sweep_candle_idx]) if 0 <= sweep_candle_idx < len(df_at_time) else float(volumes.iloc[-1])
        avg_volume = float(volumes.rolling(20).mean().iloc[-1])

        entry_quality = calculate_entry_quality(
            sweep_status=sweep_status,
            current_price=current_price,
            whale_pct=whale_pct,
            whale_delta=whale_delta_val,
            volume_on_sweep=sweep_volume,
            avg_volume=avg_volume,
            df=df_at_time,
            whale_delta_4h=whale_delta_4h,
            whale_delta_24h=whale_delta_24h,
            whale_delta_7d=whale_delta_7d,
            whale_acceleration=whale_acceleration,
            whale_early_signal=whale_early_signal,
            is_fresh_accumulation=is_fresh_accumulation,
            symbol=symbol,
            # OI data not available in historical test - defaults will be used
            oi_change_24h=None,
            price_change_24h=None
        )

        ml_prob = entry_quality.get('ml_probability', 0)
        recommendation = entry_quality.get('recommendation', 'UNKNOWN')

        print(f"\n   {'='*50}")
        print(f"   🤖 ML PROBABILITY: {ml_prob:.0%}")
        print(f"   📋 RECOMMENDATION: {recommendation}")
        print(f"   🎯 TRADE DIRECTION: {trade_dir}")
        print(f"   {'='*50}")

        print(f"\n   Entry Window: {entry_quality.get('entry_window', 'N/A')}")
        print(f"   Candles Until Close: {entry_quality.get('candles_until_close', 0)}")

        print(f"\n   Warnings/Signals:")
        for w in entry_quality.get('warnings', []):
            print(f"   {w}")

        return {
            'symbol': symbol,
            'time': label,
            'price': current_price,
            'sweep_detected': True,
            'sweep_direction': sweep_dir,
            'trade_direction': trade_dir,
            'ml_probability': ml_prob,
            'recommendation': recommendation,
            'entry_window': entry_quality.get('entry_window'),
            'whale_pct': whale_pct,
            'whale_delta_24h': whale_delta_24h
        }
    else:
        print(f"\n   ❌ NO SWEEP DETECTED")
        print(f"   Reason: {sweep_status.get('reason', 'No levels broken')}")

        return {
            'symbol': symbol,
            'time': label,
            'price': current_price,
            'sweep_detected': False,
            'sweep_direction': None,
            'trade_direction': None,
            'ml_probability': 0,
            'recommendation': 'NO_SETUP',
            'entry_window': None,
            'whale_pct': whale_pct,
            'whale_delta_24h': whale_delta_24h
        }

def main():
    symbol = 'CUSDT'  # C Protocol token
    timeframe = '1h'   # 1h timeframe for more granular analysis

    print("\n" + "=" * 70)
    print(f"CUSDT/CLUSDT SWEEP TEST - Jan 25, 2026")
    print(f"Symbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print("=" * 70)

    # Define target times (UTC)
    # Abu Dhabi is UTC+4, so:
    # 1:30 AM Abu Dhabi = 21:30 UTC (previous day)
    # 13:30 PM Abu Dhabi = 09:30 UTC (same day)

    time1_utc = datetime(2026, 1, 24, 21, 0)  # 1:00 AM Abu Dhabi Jan 25
    time2_utc = datetime(2026, 1, 25, 9, 0)   # 13:00 PM Abu Dhabi Jan 25

    results = []

    # Test at 1:30 AM Abu Dhabi
    result1 = test_cusdt_at_time(symbol, timeframe, time1_utc, "1:00 AM Abu Dhabi (Jan 25)")
    if result1:
        results.append(result1)

    # Test at 13:00 PM Abu Dhabi
    result2 = test_cusdt_at_time(symbol, timeframe, time2_utc, "13:00 PM Abu Dhabi (Jan 25)")
    if result2:
        results.append(result2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\n{r['time']}:")
        print(f"  Price: ${r['price']:.6f}")
        print(f"  Sweep Detected: {'Yes' if r['sweep_detected'] else 'No'}")
        if r['sweep_detected']:
            print(f"  Sweep Direction: {r['sweep_direction']} → Trade {r['trade_direction']}")
            print(f"  ML Probability: {r['ml_probability']:.0%}")
            print(f"  Recommendation: {r['recommendation']}")
            print(f"  Entry Window: {r['entry_window']}")
        print(f"  Whale %: {r['whale_pct']}%")
        print(f"  Whale Delta 24h: {r['whale_delta_24h']:+.1f}%" if r['whale_delta_24h'] else "  Whale Delta 24h: N/A")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
