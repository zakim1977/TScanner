"""
Backtest Quality Model on Specific Historical Cases
====================================================
Test the ML model's predictions on past data to verify it works correctly.

Usage:
    python backtest_quality_model.py

Modify TEST_CASES below to add your own test cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES - MODIFY THIS SECTION TO ADD YOUR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

# Abu Dhabi timezone (UTC+4)
ABU_DHABI_TZ = pytz.timezone('Asia/Dubai')

TEST_CASES = [
    {
        'symbol': 'DOTUSDT',
        'datetime_local': '2026-01-29 11:15',  # Abu Dhabi time
        'timezone': ABU_DHABI_TZ,
        'description': 'DOT Jan 29 morning test'
    },
    {
        'symbol': 'DOTUSDT',
        'datetime_local': '2026-01-27 18:30',  # Abu Dhabi time
        'timezone': ABU_DHABI_TZ,
        'description': 'DOT Jan 27 evening test'
    },
    # Add more test cases here:
    # {
    #     'symbol': 'BTCUSDT',
    #     'datetime_local': '2026-01-25 14:00',
    #     'timezone': ABU_DHABI_TZ,
    #     'description': 'BTC test case'
    # },
]

# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def convert_to_utc(local_datetime_str: str, local_tz) -> datetime:
    """Convert local datetime string to UTC."""
    local_dt = local_tz.localize(datetime.strptime(local_datetime_str, '%Y-%m-%d %H:%M'))
    return local_dt.astimezone(pytz.UTC)


def fetch_historical_data(symbol: str, target_utc: datetime, lookback_candles: int = 200):
    """Fetch 4h candles using the existing data fetcher (handles SSL properly)."""
    from core.data_fetcher import fetch_binance_klines

    # Calculate how many candles we need total
    # We fetch a large amount and then find the target time within it
    total_candles = lookback_candles + 100  # Extra buffer

    try:
        df = fetch_binance_klines(symbol, '4h', limit=total_candles)

        if df is None or len(df) < 50:
            return None, f"Insufficient data: got {len(df) if df is not None else 0} candles"

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            elif 'open_time' in df.columns:
                df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')

        print(f"Fetched {len(df)} candles from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

        # Check if target time is within our data range
        target_naive = target_utc.replace(tzinfo=None)
        data_start = df['datetime'].iloc[0]
        data_end = df['datetime'].iloc[-1]

        if target_naive < data_start or target_naive > data_end:
            print(f"WARNING: Target time {target_naive} is outside data range ({data_start} to {data_end})")
            print(f"Using closest available data...")

        return df, None

    except Exception as e:
        return None, f"Fetch error: {str(e)}"


def find_candle_at_time(df: pd.DataFrame, target_utc: datetime):
    """Find the candle that contains the target time."""
    df = df.copy()

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Ensure datetime column is proper datetime type
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    target_naive = target_utc.replace(tzinfo=None)

    # Find candle where target falls within [open_time, open_time + 4h)
    for idx in range(len(df)):
        candle_start = pd.to_datetime(df.iloc[idx]['datetime'])
        candle_end = candle_start + timedelta(hours=4)

        if candle_start <= target_naive < candle_end:
            return idx, df.iloc[idx]

    # If exact match not found, find closest
    df['time_diff'] = abs((df['datetime'] - target_naive).dt.total_seconds())
    closest_idx = df['time_diff'].idxmin()

    # Get the integer position for iloc
    if isinstance(closest_idx, int):
        return closest_idx, df.iloc[closest_idx]
    else:
        pos = df.index.get_loc(closest_idx)
        return pos, df.loc[closest_idx]


def run_quality_prediction(symbol: str, df: pd.DataFrame, candle_idx: int):
    """Run the quality model prediction at the given candle."""
    from liquidity_hunter.quality_model import get_quality_prediction
    from liquidity_hunter.liquidity_hunter import find_liquidity_levels, detect_sweep, normalize_columns

    # Use data up to (and including) the target candle
    df_slice = df.iloc[:candle_idx + 1].copy()
    df_slice = normalize_columns(df_slice)

    if len(df_slice) < 50:
        return None, "Not enough historical data"

    # Calculate ATR
    high = df_slice['high'].values
    low = df_slice['low'].values
    close = df_slice['close'].values

    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]),
                              abs(low[1:] - close[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

    current_price = float(df_slice['close'].iloc[-1])

    # Find liquidity levels
    levels = find_liquidity_levels(df_slice, lookback=50)

    # Detect sweeps
    sweep_status = detect_sweep(df_slice, levels, atr, lookback_candles=10, whale_pct=50)

    # Get whale data (use defaults for backtest)
    whale_pct = 50.0  # Neutral
    whale_delta = 0.0

    results = {
        'current_price': current_price,
        'atr': atr,
        'levels': levels,
        'sweep_status': sweep_status,
    }

    # Check if there's an active/recent sweep
    has_sweep = (
        sweep_status.get('has_sweep') or
        sweep_status.get('status') in ['SWEEP_ACTIVE', 'SWEEP_RECENT'] or
        sweep_status.get('direction') is not None
    )

    if has_sweep:
        sweep_type = sweep_status.get('sweep_type', sweep_status.get('level_type', 'UNKNOWN'))
        sweep_level = sweep_status.get('sweep_level', sweep_status.get('level_price', current_price))
        sweep_direction = sweep_status.get('direction', '')
        candles_ago = sweep_status.get('candles_ago', 5)

        # Mark as having sweep
        results['sweep_status']['has_sweep'] = True

        # Determine if it's a LOW or HIGH sweep
        is_low_sweep = (
            'LOW' in str(sweep_type).upper() or
            sweep_direction == 'LONG'  # LONG direction usually means sweep of LOW
        )

        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE SWEEP FEATURES FROM DATA
        # ═══════════════════════════════════════════════════════════════════
        sweep_candle_idx = len(df_slice) - 1 - candles_ago if candles_ago else len(df_slice) - 1
        sweep_candle_idx = max(0, min(sweep_candle_idx, len(df_slice) - 1))

        # Get sweep candle data
        sweep_candle = df_slice.iloc[sweep_candle_idx]
        sweep_open = float(sweep_candle['open'])
        sweep_high = float(sweep_candle['high'])
        sweep_low = float(sweep_candle['low'])
        sweep_close = float(sweep_candle['close'])
        sweep_volume = float(sweep_candle['volume']) if 'volume' in sweep_candle else 1.0

        # Calculate average volume
        avg_volume = df_slice['volume'].iloc[-20:].mean() if 'volume' in df_slice.columns else 1.0

        # Sweep depth (how far past the level)
        if is_low_sweep:
            sweep_depth = (sweep_level - sweep_low) / atr if atr > 0 else 0
        else:
            sweep_depth = (sweep_high - sweep_level) / atr if atr > 0 else 0
        sweep_depth = max(0, sweep_depth)

        # Wick ratio (rejection strength)
        candle_range = sweep_high - sweep_low
        if candle_range > 0:
            if is_low_sweep:
                # For low sweep, lower wick shows rejection
                lower_wick = min(sweep_open, sweep_close) - sweep_low
                sweep_wick_ratio = lower_wick / candle_range
            else:
                # For high sweep, upper wick shows rejection
                upper_wick = sweep_high - max(sweep_open, sweep_close)
                sweep_wick_ratio = upper_wick / candle_range
        else:
            sweep_wick_ratio = 0

        # Body ratio
        body = abs(sweep_close - sweep_open)
        sweep_body_ratio = body / candle_range if candle_range > 0 else 0.5

        # Volume ratio
        sweep_volume_ratio = sweep_volume / avg_volume if avg_volume > 0 else 1.0

        # Price change since sweep
        sweep_close_price = float(df_slice.iloc[sweep_candle_idx]['close'])
        price_change_since_sweep = (current_price - sweep_close_price) / sweep_close_price * 100

        # Check if price aligned with expected direction
        if is_low_sweep:
            # After sweep of low, price should go UP for LONG
            rule_price_aligned = 1 if price_change_since_sweep > -1 else 0
        else:
            # After sweep of high, price should go DOWN for SHORT
            rule_price_aligned = 1 if price_change_since_sweep < 1 else 0

        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE STRUCTURE FEATURES
        # ═══════════════════════════════════════════════════════════════════
        # Check for lower highs (bearish structure)
        highs = df_slice['high'].iloc[-10:].values
        structure_lower_highs = 1 if len(highs) >= 3 and highs[-1] < highs[-2] < highs[-3] else 0

        # Check for higher lows (bullish structure)
        lows = df_slice['low'].iloc[-10:].values
        structure_higher_lows = 1 if len(lows) >= 3 and lows[-1] > lows[-2] > lows[-3] else 0

        # Failed BOS (structure break failed)
        structure_failed_bos = 0  # Would need more complex analysis

        # Trend aligned with trade direction
        if is_low_sweep:
            structure_trend_aligned = structure_higher_lows
        else:
            structure_trend_aligned = structure_lower_highs

        # ═══════════════════════════════════════════════════════════════════
        # GET ML PREDICTIONS WITH FULL FEATURES
        # ═══════════════════════════════════════════════════════════════════
        level_type_for_ml = 'SWING_LOW' if is_low_sweep else 'SWING_HIGH'

        # NEW: Follow-through per candle (% change per candle since sweep)
        follow_through_per_candle = abs(price_change_since_sweep) / candles_ago if candles_ago > 0 else 0

        # NEW: Distance from sweep level
        distance_from_sweep_pct = abs(current_price - sweep_level) / sweep_level * 100 if sweep_level > 0 else 10

        # NEW: Indecision candle (body ≈ wick)
        indecision_candle = 1 if abs(sweep_body_ratio - sweep_wick_ratio) < 0.2 and sweep_body_ratio > 0.3 else 0

        # NEW: Weak follow-through (5+ candles ago, <2% move)
        rule_weak_follow_through = 1 if candles_ago >= 5 and abs(price_change_since_sweep) < 2.0 else 0

        # NEW: Close to sweep level (within 3%)
        rule_close_to_sweep_level = 1 if distance_from_sweep_pct < 3.0 else 0

        # Common sweep features for both predictions
        sweep_features = {
            'sweep_depth_atr': sweep_depth,
            'sweep_wick_ratio': sweep_wick_ratio,
            'sweep_body_ratio': sweep_body_ratio,
            'candles_since_sweep': candles_ago,
            'sweep_volume_ratio': sweep_volume_ratio,
            'price_change_since_sweep': price_change_since_sweep,
            'rule_price_aligned': rule_price_aligned,
            'structure_lower_highs': structure_lower_highs,
            'structure_higher_lows': structure_higher_lows,
            'structure_failed_bos': structure_failed_bos,
            'structure_trend_aligned': structure_trend_aligned,
            # NEW features (Jan 2026)
            'follow_through_per_candle': follow_through_per_candle,
            'distance_from_sweep_pct': distance_from_sweep_pct,
            'indecision_candle': indecision_candle,
            'rule_weak_follow_through': rule_weak_follow_through,
            'rule_close_to_sweep_level': rule_close_to_sweep_level,
        }

        # Get prediction - the model evaluates BOTH directions internally
        # and returns probabilities for both
        prediction = get_quality_prediction(
            symbol=symbol, direction='LONG', level_type=level_type_for_ml,
            level_price=sweep_level, current_price=current_price, atr=atr,
            whale_pct=whale_pct, whale_delta=whale_delta,
            **sweep_features
        )

        # Extract direction-specific probabilities (model returns both)
        best_direction = prediction.get('best_direction', 'LONG')
        prob_long = prediction.get('long_probability', 0.5)
        prob_short = prediction.get('short_probability', 0.5)
        best_prob = prediction.get('probability', 0.5)
        decision = prediction.get('decision', 'UNKNOWN')

        # Determine decision for each direction
        def get_decision(prob):
            if prob >= 0.6:
                return 'STRONG_YES'
            elif prob >= 0.5:
                return 'YES'
            elif prob >= 0.4:
                return 'MAYBE'
            else:
                return 'NO'

        results['prediction_long'] = {
            'probability': prob_long,
            'decision': get_decision(prob_long),
            'take_trade': prob_long >= 0.5
        }
        results['prediction_short'] = {
            'probability': prob_short,
            'decision': get_decision(prob_short),
            'take_trade': prob_short >= 0.5
        }
        results['is_low_sweep'] = is_low_sweep
        results['sweep_features'] = sweep_features

        # Use the model's recommended direction (higher probability)
        results['recommended'] = best_direction
        results['ml_probability'] = best_prob
        results['ml_decision'] = decision

    return results, None


def check_outcome(df: pd.DataFrame, candle_idx: int, direction: str, atr: float, rr_ratio: float = 2.0):
    """Check what actually happened after the signal."""
    entry_price = float(df.iloc[candle_idx]['close'])

    # Calculate SL and TP
    if direction == 'LONG':
        stop_loss = entry_price - atr
        take_profit = entry_price + (atr * rr_ratio)
    else:  # SHORT
        stop_loss = entry_price + atr
        take_profit = entry_price - (atr * rr_ratio)

    # Check subsequent candles
    future_candles = df.iloc[candle_idx + 1:candle_idx + 41]  # Next 40 candles (~7 days)

    outcome = {
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'direction': direction,
        'result': 'PENDING',
        'exit_price': None,
        'exit_candle': None,
        'pnl_percent': 0,
        'max_favorable': 0,  # Max move in favorable direction
        'max_adverse': 0,    # Max move against us (drawdown)
        'price_path': [],    # Track price movement
    }

    max_high = entry_price
    min_low = entry_price

    for i, (idx, candle) in enumerate(future_candles.iterrows()):
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])

        max_high = max(max_high, high)
        min_low = min(min_low, low)

        # Track price path for first 10 candles
        if i < 10:
            outcome['price_path'].append({
                'candle': i + 1,
                'high': high,
                'low': low,
                'close': close
            })

        if direction == 'LONG':
            outcome['max_favorable'] = (max_high - entry_price) / entry_price * 100
            outcome['max_adverse'] = (entry_price - min_low) / entry_price * 100

            # Check if SL hit first
            if low <= stop_loss:
                outcome['result'] = 'LOSS'
                outcome['exit_price'] = stop_loss
                outcome['exit_candle'] = i + 1
                outcome['pnl_percent'] = -1.0  # 1 ATR loss
                break
            # Check if TP hit
            if high >= take_profit:
                outcome['result'] = 'WIN'
                outcome['exit_price'] = take_profit
                outcome['exit_candle'] = i + 1
                outcome['pnl_percent'] = rr_ratio  # 2 ATR profit at 2:1 R:R
                break
        else:  # SHORT
            outcome['max_favorable'] = (entry_price - min_low) / entry_price * 100
            outcome['max_adverse'] = (max_high - entry_price) / entry_price * 100

            # Check if SL hit first
            if high >= stop_loss:
                outcome['result'] = 'LOSS'
                outcome['exit_price'] = stop_loss
                outcome['exit_candle'] = i + 1
                outcome['pnl_percent'] = -1.0
                break
            # Check if TP hit
            if low <= take_profit:
                outcome['result'] = 'WIN'
                outcome['exit_price'] = take_profit
                outcome['exit_candle'] = i + 1
                outcome['pnl_percent'] = rr_ratio
                break

    return outcome


def run_backtest(test_case: dict):
    """Run backtest for a single test case."""
    symbol = test_case['symbol']
    local_dt_str = test_case['datetime_local']
    local_tz = test_case['timezone']
    description = test_case.get('description', '')

    print(f"\n{'='*70}")
    print(f"BACKTEST: {symbol} @ {local_dt_str} (Local Time)")
    print(f"Description: {description}")
    print(f"{'='*70}")

    # Convert to UTC
    target_utc = convert_to_utc(local_dt_str, local_tz)
    print(f"UTC Time: {target_utc.strftime('%Y-%m-%d %H:%M')} UTC")

    # Fetch data
    print(f"\nFetching historical data for {symbol}...")
    df, error = fetch_historical_data(symbol, target_utc)

    if error:
        print(f"ERROR: {error}")
        return None

    print(f"Fetched {len(df)} candles")

    # Find the target candle
    candle_idx, target_candle = find_candle_at_time(df, target_utc)
    # Get OHLC values from series (columns are lowercase after normalization)
    try:
        tc_open = float(target_candle['open'])
        tc_high = float(target_candle['high'])
        tc_low = float(target_candle['low'])
        tc_close = float(target_candle['close'])
        tc_time = target_candle['datetime']
    except (KeyError, TypeError):
        tc_open = tc_high = tc_low = tc_close = 'N/A'
        tc_time = 'N/A'

    print(f"\nTarget Candle (index {candle_idx}):")
    print(f"  Time:  {tc_time}")
    if tc_open != 'N/A':
        print(f"  Open:  ${tc_open:.4f}")
        print(f"  High:  ${tc_high:.4f}")
        print(f"  Low:   ${tc_low:.4f}")
        print(f"  Close: ${tc_close:.4f}")
    else:
        print("  OHLC: N/A (check data)")

    # Run prediction
    print(f"\nRunning Quality Model prediction...")
    results, error = run_quality_prediction(symbol, df, candle_idx)

    if error:
        print(f"ERROR: {error}")
        return None

    # Display results
    print(f"\n{'─'*70}")
    print("SWEEP STATUS:")
    print(f"{'─'*70}")
    sweep = results.get('sweep_status', {})

    # Check various ways sweep might be indicated
    has_sweep = (
        sweep.get('has_sweep') or
        sweep.get('status') in ['SWEEP_ACTIVE', 'SWEEP_RECENT'] or
        sweep.get('sweep_type') is not None or
        sweep.get('direction') is not None
    )

    if has_sweep:
        print(f"  ✅ SWEEP DETECTED!")
        print(f"  Status: {sweep.get('status', 'N/A')}")
        print(f"  Direction: {sweep.get('direction', 'N/A')}")
        print(f"  Type: {sweep.get('sweep_type', sweep.get('level_type', 'N/A'))}")
        print(f"  Level: ${sweep.get('sweep_level', sweep.get('level_price', 0)):.4f}")
        print(f"  Candles Ago: {sweep.get('candles_ago', 'N/A')}")
        print(f"  Confidence: {sweep.get('confidence', 'N/A')}")

        # Mark that we have a sweep for ML prediction
        results['sweep_status']['has_sweep'] = True
    else:
        print(f"  ❌ No active sweep at this time")
        print(f"  Status: {sweep.get('status', 'MONITORING')}")
        print(f"  (Sweep detection shows recent sweeps but may be too old for entry)")

    # Show all detected liquidity levels for context
    levels = results.get('levels', {})
    print(f"\n{'─'*70}")
    print("DETECTED LIQUIDITY LEVELS:")
    print(f"{'─'*70}")
    current_price = results.get('current_price', 0)
    print(f"  Current Price: ${current_price:.4f}")

    highs = levels.get('highs', [])
    lows = levels.get('lows', [])

    if highs:
        print(f"\n  HIGHS (resistance/sell liquidity):")
        for h in highs[:5]:  # Show top 5
            price = h.get('price', h) if isinstance(h, dict) else h
            strength = h.get('strength', 'N/A') if isinstance(h, dict) else 'N/A'
            print(f"    ${price:.4f} (strength: {strength})")

    if lows:
        print(f"\n  LOWS (support/buy liquidity):")
        for l in lows[:5]:  # Show top 5
            price = l.get('price', l) if isinstance(l, dict) else l
            strength = l.get('strength', 'N/A') if isinstance(l, dict) else 'N/A'
            print(f"    ${price:.4f} (strength: {strength})")

    # Show sweep features if available
    if 'sweep_features' in results:
        print(f"\n{'─'*70}")
        print("SWEEP FEATURES (ML model inputs):")
        print(f"{'─'*70}")
        sf = results['sweep_features']
        print(f"  Sweep Type: {'LOW' if results.get('is_low_sweep') else 'HIGH'}")
        print(f"  sweep_depth_atr: {sf.get('sweep_depth_atr', 0):.3f}")
        print(f"  sweep_wick_ratio: {sf.get('sweep_wick_ratio', 0):.3f}")
        print(f"  sweep_body_ratio: {sf.get('sweep_body_ratio', 0):.3f}")
        print(f"  candles_since_sweep: {sf.get('candles_since_sweep', 0)}")
        print(f"  sweep_volume_ratio: {sf.get('sweep_volume_ratio', 0):.2f}")
        print(f"  price_change_since_sweep: {sf.get('price_change_since_sweep', 0):.2f}%")
        print(f"  rule_price_aligned: {sf.get('rule_price_aligned', 0)}")
        print(f"  structure_lower_highs: {sf.get('structure_lower_highs', 0)}")
        print(f"  structure_higher_lows: {sf.get('structure_higher_lows', 0)}")
        print(f"  structure_trend_aligned: {sf.get('structure_trend_aligned', 0)}")
        # NEW features
        print(f"\n  --- NEW FEATURES (Jan 2026) ---")
        print(f"  follow_through_per_candle: {sf.get('follow_through_per_candle', 0):.3f}%/candle")
        print(f"  distance_from_sweep_pct: {sf.get('distance_from_sweep_pct', 0):.2f}%")
        print(f"  indecision_candle: {sf.get('indecision_candle', 0)} {'⚠️ CHOPPY!' if sf.get('indecision_candle', 0) == 1 else ''}")
        print(f"  rule_weak_follow_through: {sf.get('rule_weak_follow_through', 0)} {'⚠️ WEAK!' if sf.get('rule_weak_follow_through', 0) == 1 else ''}")
        print(f"  rule_close_to_sweep_level: {sf.get('rule_close_to_sweep_level', 0)} {'⚠️ RETEST RISK!' if sf.get('rule_close_to_sweep_level', 0) == 1 else ''}")

    print(f"\n{'─'*70}")
    print("ML PREDICTIONS:")
    print(f"{'─'*70}")

    if 'prediction_long' in results:
        pred_long = results['prediction_long']
        pred_short = results['prediction_short']

        print(f"\n  🟢 LONG Prediction:")
        print(f"     Probability: {pred_long.get('probability', 0):.1%}")
        print(f"     Decision: {pred_long.get('decision', 'N/A')}")
        print(f"     Take Trade: {pred_long.get('take_trade', False)}")

        print(f"\n  🔴 SHORT Prediction:")
        print(f"     Probability: {pred_short.get('probability', 0):.1%}")
        print(f"     Decision: {pred_short.get('decision', 'N/A')}")
        print(f"     Take Trade: {pred_short.get('take_trade', False)}")

        print(f"\n  📊 RECOMMENDED: {results.get('recommended', 'N/A')}")

        # Check actual outcome for BOTH directions
        print(f"\n{'─'*70}")
        print("ACTUAL OUTCOMES (comparing both directions):")
        print(f"{'─'*70}")

        # Check LONG outcome
        outcome_long = check_outcome(df, candle_idx, 'LONG', results['atr'])
        print(f"\n  🟢 IF LONG:")
        print(f"     Entry: ${outcome_long['entry_price']:.4f}")
        print(f"     Stop Loss: ${outcome_long['stop_loss']:.4f} ({(outcome_long['stop_loss']/outcome_long['entry_price']-1)*100:+.2f}%)")
        print(f"     Take Profit: ${outcome_long['take_profit']:.4f} ({(outcome_long['take_profit']/outcome_long['entry_price']-1)*100:+.2f}%)")
        print(f"     Max Favorable: +{outcome_long['max_favorable']:.2f}% (price went UP this much)")
        print(f"     Max Adverse: -{outcome_long['max_adverse']:.2f}% (drawdown)")
        print(f"     Result: {outcome_long['result']} {'✅' if outcome_long['result'] == 'WIN' else '❌' if outcome_long['result'] == 'LOSS' else '⏳'}")
        if outcome_long['exit_price']:
            print(f"     Exit After: {outcome_long['exit_candle']} candles ({outcome_long['exit_candle'] * 4}h)")

        # Check SHORT outcome
        outcome_short = check_outcome(df, candle_idx, 'SHORT', results['atr'])
        print(f"\n  🔴 IF SHORT:")
        print(f"     Entry: ${outcome_short['entry_price']:.4f}")
        print(f"     Stop Loss: ${outcome_short['stop_loss']:.4f} ({(outcome_short['stop_loss']/outcome_short['entry_price']-1)*100:+.2f}%)")
        print(f"     Take Profit: ${outcome_short['take_profit']:.4f} ({(outcome_short['take_profit']/outcome_short['entry_price']-1)*100:+.2f}%)")
        print(f"     Max Favorable: +{outcome_short['max_favorable']:.2f}% (price went DOWN this much)")
        print(f"     Max Adverse: -{outcome_short['max_adverse']:.2f}% (price went UP against us)")
        print(f"     Result: {outcome_short['result']} {'✅' if outcome_short['result'] == 'WIN' else '❌' if outcome_short['result'] == 'LOSS' else '⏳'}")
        if outcome_short['exit_price']:
            print(f"     Exit After: {outcome_short['exit_candle']} candles ({outcome_short['exit_candle'] * 4}h)")

        # Show price path for first few candles
        print(f"\n  📈 PRICE PATH (first 5 candles after entry):")
        for p in outcome_long['price_path'][:5]:
            pct_from_entry = (p['close'] - outcome_long['entry_price']) / outcome_long['entry_price'] * 100
            print(f"     Candle {p['candle']}: H=${p['high']:.4f} L=${p['low']:.4f} C=${p['close']:.4f} ({pct_from_entry:+.2f}%)")

        # Determine correct direction
        recommended = results.get('recommended', 'UNKNOWN')
        long_won = outcome_long['result'] == 'WIN'
        short_won = outcome_short['result'] == 'WIN'

        if long_won and not short_won:
            correct_dir = 'LONG'
        elif short_won and not long_won:
            correct_dir = 'SHORT'
        elif long_won and short_won:
            correct_dir = 'LONG (faster)' if outcome_long['exit_candle'] < outcome_short['exit_candle'] else 'SHORT (faster)'
        else:
            correct_dir = 'NEITHER'

        print(f"\n  📊 MODEL RECOMMENDED: {recommended}")
        print(f"  ✓ ACTUALLY CORRECT: {correct_dir}")

        model_correct = (recommended == 'LONG' and long_won) or (recommended == 'SHORT' and short_won)
        if model_correct:
            print(f"  ✅ MODEL WAS CORRECT!")
        elif correct_dir != 'NEITHER':
            print(f"  ❌ MODEL WAS WRONG - should have been {correct_dir}")

        results['outcome_long'] = outcome_long
        results['outcome_short'] = outcome_short
        results['correct_direction'] = correct_dir
        results['model_correct'] = model_correct
        results['outcome'] = outcome_long if recommended == 'LONG' else outcome_short
    else:
        print("  No sweep detected - no ML prediction available")

    return results


def main():
    """Run all test cases."""
    print("\n" + "="*70)
    print("QUALITY MODEL BACKTEST")
    print("="*70)
    print(f"Running {len(TEST_CASES)} test cases...")

    all_results = []
    wins = 0
    losses = 0
    pending = 0

    for test_case in TEST_CASES:
        result = run_backtest(test_case)
        if result:
            all_results.append(result)

            outcome = result.get('outcome', {})
            if outcome.get('result') == 'WIN':
                wins += 1
            elif outcome.get('result') == 'LOSS':
                losses += 1
            else:
                pending += 1

    # Summary
    print("\n" + "="*70)
    print("BACKTEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Sweeps Found: {len(all_results)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Pending/No Exit: {pending}")

    if wins + losses > 0:
        win_rate = wins / (wins + losses) * 100
        print(f"Win Rate: {win_rate:.1f}%")


if __name__ == '__main__':
    main()
