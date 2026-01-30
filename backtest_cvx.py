"""
Backtest CVX at specific time: Jan 25, 2026 00:30 Abu Dhabi time
Price was $2.206 - would ML have predicted LONG?
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pytz

# Import our modules
from liquidity_hunter.liquidity_hunter import (
    find_liquidity_levels, detect_sweep, calculate_entry_quality,
    normalize_columns, TRADE_STRATEGY_MODE
)
from liquidity_hunter.quality_model import get_quality_prediction, get_quality_model_status
from core.data_fetcher import fetch_ohlcv_binance

def fetch_historical_data(symbol: str, timeframe: str, end_time: datetime, limit: int = 200):
    """Fetch historical OHLCV data ending at specific time."""
    # Convert end_time to milliseconds
    end_ms = int(end_time.timestamp() * 1000)

    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': timeframe,
        'endTime': end_ms,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if not data:
            print(f"No data returned for {symbol}")
            return None

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df.set_index('timestamp', inplace=True)
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def get_whale_data_at_time(symbol: str, target_time: datetime):
    """
    Get whale positioning at specific time.
    Note: Historical whale data may not be available, so we estimate.
    """
    # Try to fetch historical top trader data
    url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"

    # Get data around the target time
    end_ms = int(target_time.timestamp() * 1000)

    params = {
        'symbol': symbol,
        'period': '4h',
        'limit': 50,
        'endTime': end_ms
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data and len(data) > 0:
            # Get the closest data point to target time
            latest = data[-1]
            whale_pct = float(latest.get('longAccount', 0.5)) * 100

            # Calculate delta from earlier data
            if len(data) >= 6:  # 24h ago (6 x 4h)
                earlier = data[-6]
                whale_24h_ago = float(earlier.get('longAccount', 0.5)) * 100
                whale_delta = whale_pct - whale_24h_ago
            else:
                whale_delta = 0

            return {
                'whale_pct': whale_pct,
                'whale_delta': whale_delta,
                'data_available': True
            }
    except Exception as e:
        print(f"Whale data error: {e}")

    return {'whale_pct': 50, 'whale_delta': 0, 'data_available': False}

def run_backtest():
    """Run backtest for CVX at Jan 25, 2026 00:30 Abu Dhabi time."""

    symbol = "CVXUSDT"

    # Abu Dhabi time (UTC+4): Jan 25, 2026 00:30
    # Convert to UTC: Jan 24, 2026 20:30 UTC
    abu_dhabi_tz = pytz.timezone('Asia/Dubai')
    abu_dhabi_time = abu_dhabi_tz.localize(datetime(2026, 1, 25, 0, 30))
    target_time = abu_dhabi_time.astimezone(pytz.UTC).replace(tzinfo=None)  # Jan 24, 20:30 UTC

    target_price = 2.206
    timeframe = "4h"

    print("=" * 70)
    print(f"BACKTEST: {symbol}")
    print(f"Abu Dhabi Time: Jan 25, 2026 00:30 (UTC+4)")
    print(f"UTC Time: {target_time}")
    print(f"Target Price: ${target_price}")
    print(f"Strategy Mode: {TRADE_STRATEGY_MODE}")
    print("=" * 70)

    # Check ML model status
    model_status = get_quality_model_status()
    print(f"\nML Model Status: {'Trained' if model_status.get('is_trained') else 'NOT TRAINED'}")

    # Fetch historical data
    print(f"\nFetching {timeframe} data ending at {target_time}...")
    df = fetch_historical_data(symbol, timeframe, target_time, limit=200)

    if df is None or len(df) < 50:
        print("ERROR: Could not fetch sufficient historical data")
        return

    print(f"Got {len(df)} candles")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")

    # Verify the price
    last_close = df['close'].iloc[-1]
    print(f"Last close in data: ${last_close:.3f}")
    print(f"Expected price: ${target_price:.3f}")
    print(f"Difference: ${abs(last_close - target_price):.3f}")

    # Normalize columns
    df = normalize_columns(df)

    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    current_price = float(close.iloc[-1])

    print(f"\nCurrent Price: ${current_price:.3f}")
    print(f"ATR: ${atr:.4f}")

    # Get whale data
    print("\nFetching whale data...")
    whale_info = get_whale_data_at_time(symbol, target_time)
    whale_pct = whale_info['whale_pct']
    whale_delta = whale_info['whale_delta']

    print(f"Whale %: {whale_pct:.1f}%")
    print(f"Whale Delta: {whale_delta:+.2f}%")
    print(f"Data Available: {whale_info['data_available']}")

    # Find liquidity levels
    print("\nFinding liquidity levels...")
    liquidity_levels = find_liquidity_levels(df)

    lows = liquidity_levels.get('lows', [])
    highs = liquidity_levels.get('highs', [])
    swept_lows = liquidity_levels.get('recently_swept_lows', [])
    swept_highs = liquidity_levels.get('recently_swept_highs', [])

    print(f"Unswept Lows: {len(lows)}")
    print(f"Unswept Highs: {len(highs)}")
    print(f"Recently Swept Lows: {len(swept_lows)}")
    print(f"Recently Swept Highs: {len(swept_highs)}")

    if swept_lows:
        print(f"  Swept Lows: {[(round(l['price'], 3), l.get('swept_candles_ago', '?')) for l in swept_lows[:5]]}")
    if swept_highs:
        print(f"  Swept Highs: {[(round(h['price'], 3), h.get('swept_candles_ago', '?')) for h in swept_highs[:5]]}")

    # Detect sweep
    print("\nDetecting sweep...")
    lookback_candles = 25  # Day trade mode
    sweep_status = detect_sweep(df, liquidity_levels, atr, lookback_candles, whale_pct)

    if sweep_status.get('detected'):
        print(f"✅ SWEEP DETECTED!")
        print(f"   Type: {sweep_status.get('type')}")
        print(f"   Direction: {sweep_status.get('direction')}")
        print(f"   Level: ${sweep_status.get('level_swept', 0):.3f}")
        print(f"   Candles Ago: {sweep_status.get('candles_ago')}")
        print(f"   Source: {sweep_status.get('source')}")

        # Calculate entry quality (which includes ML prediction)
        print("\nCalculating entry quality with ML...")

        # Get volume data
        volumes = df['volume']
        avg_volume = float(volumes.rolling(20).mean().iloc[-1])
        candles_ago = sweep_status.get('candles_ago', 1)
        sweep_idx = len(df) - candles_ago
        sweep_volume = float(volumes.iloc[sweep_idx]) if 0 <= sweep_idx < len(df) else avg_volume

        entry_quality = calculate_entry_quality(
            sweep_status=sweep_status,
            current_price=current_price,
            whale_pct=whale_pct,
            whale_delta=whale_delta,
            volume_on_sweep=sweep_volume,
            avg_volume=avg_volume,
            df=df,
            symbol=symbol
        )

        print("\n" + "=" * 70)
        print("ENTRY QUALITY RESULTS")
        print("=" * 70)
        print(f"Score: {entry_quality.get('score', 0)}")
        print(f"Grade: {entry_quality.get('grade', 'N/A')}")
        print(f"Entry Window: {entry_quality.get('entry_window', 'N/A')}")
        print(f"Recommendation: {entry_quality.get('recommendation', 'N/A')}")
        print(f"ML Probability: {entry_quality.get('ml_probability', 0):.1%}")
        print(f"ML Decision: {entry_quality.get('ml_decision', 'N/A')}")

        # Trade direction in CONTINUATION mode
        raw_direction = sweep_status.get('direction')
        if TRADE_STRATEGY_MODE == "CONTINUATION":
            trade_direction = "SHORT" if raw_direction == "LONG" else "LONG"
        else:
            trade_direction = raw_direction

        print(f"\nRaw Sweep Direction: {raw_direction}")
        print(f"Trade Direction ({TRADE_STRATEGY_MODE}): {trade_direction}")

        # Warnings
        warnings = entry_quality.get('warnings', [])
        if warnings:
            print(f"\nWarnings:")
            for w in warnings:
                print(f"  {w}")

        print("\n" + "=" * 70)
        print("BACKTEST CONCLUSION")
        print("=" * 70)

        ml_prob = entry_quality.get('ml_probability', 0)
        if ml_prob >= 0.6:
            print(f"✅ ML would have said STRONG_YES ({ml_prob:.1%}) for {trade_direction}")
        elif ml_prob >= 0.5:
            print(f"✅ ML would have said YES ({ml_prob:.1%}) for {trade_direction}")
        elif ml_prob >= 0.4:
            print(f"⚠️ ML would have said MAYBE ({ml_prob:.1%}) for {trade_direction}")
        else:
            print(f"❌ ML would have said NO ({ml_prob:.1%}) for {trade_direction}")

    else:
        print("❌ No sweep detected at this time")
        print("The ML would not have triggered a trade signal.")

if __name__ == "__main__":
    run_backtest()
