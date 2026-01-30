"""
Trade Optimizer Diagnostic - REAL DATA
=======================================

Pulls real candles from Binance, runs SMC detector, and shows
how the optimizer picks levels for each TP.

Usage:
    python diagnostic_optimizer.py BTCUSDT 15m LONG
    python diagnostic_optimizer.py ETHUSDT 1h SHORT
    python diagnostic_optimizer.py LITUSDT 15m LONG
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import math

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - Try to import from your app
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from core.data_fetcher import fetch_binance_klines
    print("✅ Imported fetch_binance_klines")
except ImportError:
    fetch_binance_klines = None
    print("⚠️ Could not import fetch_binance_klines - will use requests directly")

try:
    from core.smc_detector import detect_smc
    print("✅ Imported detect_smc")
except ImportError:
    detect_smc = None
    print("⚠️ Could not import detect_smc")

try:
    from core.trade_optimizer import (
        LevelCollector, TradeOptimizer, TradingMode, LevelType,
        PriceLevel, STRUCTURE_SCORES, MODE_CONSTRAINTS, get_optimized_trade,
        TIMEFRAME_MODE_MAP
    )
    print("✅ Imported trade_optimizer")
except ImportError:
    print("❌ Could not import trade_optimizer - this is required!")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK: Fetch candles directly from Binance if import fails
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_candles_fallback(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Fetch candles directly from Binance API"""
    import requests
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK: Basic SMC detection if import fails
# ═══════════════════════════════════════════════════════════════════════════════

def detect_smc_fallback(df: pd.DataFrame, trading_mode: str = 'day_trade') -> Dict:
    """Basic SMC detection fallback"""
    result = {
        'bullish_obs': [],
        'bearish_obs': [],
        'bullish_fvgs': [],
        'bearish_fvgs': [],
        'swing_high': 0,
        'swing_low': 0,
    }
    
    if df is None or len(df) < 20:
        return result
    
    # Find swing high/low
    lookback = min(50, len(df) - 1)
    result['swing_high'] = float(df['High'].tail(lookback).max())
    result['swing_low'] = float(df['Low'].tail(lookback).min())
    
    current_price = df['Close'].iloc[-1]
    
    for i in range(len(df) - 10, max(0, len(df) - 100), -1):
        candle = df.iloc[i]
        body = candle['Close'] - candle['Open']
        candle_range = candle['High'] - candle['Low']
        
        if candle_range == 0:
            continue
            
        body_pct = abs(body) / candle_range
        
        if body < 0 and body_pct > 0.6 and candle['High'] > current_price:
            result['bearish_obs'].append({
                'top': float(candle['High']),
                'bottom': float(candle['Open']),
            })
        
        if body > 0 and body_pct > 0.6 and candle['Low'] < current_price:
            result['bullish_obs'].append({
                'top': float(candle['Close']),
                'bottom': float(candle['Low']),
            })
    
    # Remove duplicates
    seen_bearish = set()
    unique_bearish = []
    for ob in result['bearish_obs']:
        key = (round(ob['top'], 2), round(ob['bottom'], 2))
        if key not in seen_bearish:
            seen_bearish.add(key)
            unique_bearish.append(ob)
    result['bearish_obs'] = unique_bearish[:10]
    
    seen_bullish = set()
    unique_bullish = []
    for ob in result['bullish_obs']:
        key = (round(ob['top'], 2), round(ob['bottom'], 2))
        if key not in seen_bullish:
            seen_bullish.add(key)
            unique_bullish.append(ob)
    result['bullish_obs'] = unique_bullish[:10]
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostic(symbol: str, timeframe: str, direction: str):
    """Run full diagnostic with real data"""
    
    print("\n" + "=" * 80)
    print(f"🔍 TRADE OPTIMIZER DIAGNOSTIC - REAL DATA")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Direction: {direction}")
    print("=" * 80)
    
    # STEP 1: Fetch real candles
    print("\n" + "─" * 80)
    print("📊 STEP 1: FETCHING REAL CANDLES FROM BINANCE")
    print("─" * 80)
    
    try:
        if fetch_binance_klines:
            df = fetch_binance_klines(symbol, timeframe, 200)
        else:
            df = fetch_candles_fallback(symbol, timeframe, 200)
        
        current_price = float(df['Close'].iloc[-1])
        print(f"   ✅ Fetched {len(df)} candles")
        print(f"   Current Price: ${current_price:,.4f}")
    except Exception as e:
        print(f"   ❌ Error fetching candles: {e}")
        print(f"\n   Running DEMO MODE instead...")
        run_demo()
        return
    
    # STEP 2: Run SMC Detection
    print("\n" + "─" * 80)
    print("📊 STEP 2: RUNNING SMC DETECTION (LTF + HTF)")
    print("─" * 80)
    
    mode = TIMEFRAME_MODE_MAP.get(timeframe, TradingMode.DAY_TRADE)
    
    # Map mode to string format that smc_detector expects
    mode_str_map = {
        TradingMode.SCALP: 'Scalp',
        TradingMode.DAY_TRADE: 'DayTrade',  # smc_detector uses 'Day Trade' or 'DayTrade'
        TradingMode.SWING: 'Swing',
        TradingMode.INVESTMENT: 'Investment',
    }
    trading_mode_str = mode_str_map.get(mode, 'DayTrade')
    print(f"   Mode: {mode.value} → trading_mode='{trading_mode_str}'")
    
    try:
        if detect_smc:
            smc_raw = detect_smc(df, trading_mode=trading_mode_str)
            
            # DEBUG: Show raw structure
            print(f"   DEBUG - Raw keys: {list(smc_raw.keys())}")
            if 'order_blocks' in smc_raw:
                ob_raw = smc_raw['order_blocks']
                ob_keys = list(ob_raw.keys()) if isinstance(ob_raw, dict) else 'not a dict'
                print(f"   DEBUG - order_blocks keys: {ob_keys}")
                
                # Show debug stats if available
                if isinstance(ob_raw, dict) and 'debug_stats' in ob_raw:
                    stats = ob_raw['debug_stats']
                    print(f"\n   📊 DEBUG STATS (why OBs might be missing):")
                    print(f"      Swing Lows found: {stats.get('swing_lows', 'N/A')}")
                    print(f"      Swing Highs found: {stats.get('swing_highs', 'N/A')}")
                    print(f"      Bullish OB candidates: {stats.get('bullish_candidates', 'N/A')}")
                    print(f"      Bullish OB mitigated: {stats.get('bullish_mitigated', 'N/A')}")
                    print(f"      Bullish OB valid: {stats.get('bullish_valid', 'N/A')}")
                    print(f"      ATR: ${stats.get('atr', 0):,.2f}")
                    print(f"      Min move threshold: {stats.get('atr', 0) * 0.4:,.2f} (ATR × 0.4)")
            
            # UNWRAP: SMC returns nested structure, OBs are under 'order_blocks' key
            ob_data = smc_raw.get('order_blocks', {})
            smc_data = {
                'bearish_obs': ob_data.get('bearish_obs', []),
                'bullish_obs': ob_data.get('bullish_obs', []),
                'bearish_ob_top': ob_data.get('bearish_ob_top', 0),
                'bearish_ob_bottom': ob_data.get('bearish_ob_bottom', 0),
                'bullish_ob_top': ob_data.get('bullish_ob_top', 0),
                'bullish_ob_bottom': ob_data.get('bullish_ob_bottom', 0),
                'swing_high': smc_raw.get('structure', {}).get('last_swing_high', 0),
                'swing_low': smc_raw.get('structure', {}).get('last_swing_low', 0),
                'fvg': smc_raw.get('fvg', {}),
            }
        else:
            smc_data = detect_smc_fallback(df, trading_mode=trading_mode_str)
        
        print(f"\n   ✅ SMC Detection complete")
        print(f"   Bearish OBs: {len(smc_data.get('bearish_obs', []))}")
        print(f"   Bullish OBs: {len(smc_data.get('bullish_obs', []))}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Show raw SMC data
    print(f"\n   📋 RAW SMC DATA:")
    
    if smc_data.get('bearish_obs'):
        print(f"\n   Bearish OBs (TP targets for LONG):")
        for i, ob in enumerate(smc_data['bearish_obs'][:10]):
            dist = (ob.get('bottom', 0) - current_price) / current_price * 100
            print(f"      {i+1}. ${ob.get('bottom', 0):,.4f} - ${ob.get('top', 0):,.4f} ({dist:+.2f}%)")
    
    if smc_data.get('bullish_obs'):
        print(f"\n   Bullish OBs (SL structure for LONG):")
        for i, ob in enumerate(smc_data['bullish_obs'][:10]):
            dist = (ob.get('top', 0) - current_price) / current_price * 100
            print(f"      {i+1}. ${ob.get('bottom', 0):,.4f} - ${ob.get('top', 0):,.4f} ({dist:+.2f}%)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2b: HTF (4H) SMC Detection
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n   📊 FETCHING HTF (4H) DATA...")
    
    htf_data = None
    htf_timeframe_map = {
        '1m': '15m',
        '5m': '1h',
        '15m': '4h',
        '1h': '4h',
        '4h': '1d',
        '1d': '1w',
    }
    htf_timeframe = htf_timeframe_map.get(timeframe, '4h')
    
    try:
        if fetch_binance_klines:
            htf_df = fetch_binance_klines(symbol, htf_timeframe, 100)
        else:
            htf_df = fetch_candles_fallback(symbol, htf_timeframe, 100)
        
        print(f"   ✅ Fetched {len(htf_df)} HTF ({htf_timeframe}) candles")
        
        # Run SMC on HTF
        if detect_smc:
            htf_smc_raw = detect_smc(htf_df, trading_mode='Swing')  # HTF uses Swing mode
            htf_ob_data = htf_smc_raw.get('order_blocks', {})
            htf_structure = htf_smc_raw.get('structure', {})
            
            htf_data = {
                'bearish_obs': htf_ob_data.get('bearish_obs', []),
                'bullish_obs': htf_ob_data.get('bullish_obs', []),
                'htf_swing_high': htf_structure.get('last_swing_high', 0),
                'htf_swing_low': htf_structure.get('last_swing_low', 0),
            }
            
            # Calculate HTF POC and VWAP
            try:
                htf_typical = (htf_df['High'] + htf_df['Low'] + htf_df['Close']) / 3
                htf_vwap = (htf_typical * htf_df['Volume']).sum() / htf_df['Volume'].sum()
                htf_data['htf_vwap'] = float(htf_vwap)
                
                # Simple POC calculation for HTF
                htf_price_min = htf_df['Low'].min()
                htf_price_max = htf_df['High'].max()
                htf_bins = {}
                bin_size = (htf_price_max - htf_price_min) / 30
                
                for idx in range(len(htf_df)):
                    row = htf_df.iloc[idx]
                    for b in range(30):
                        bin_mid = htf_price_min + (b + 0.5) * bin_size
                        if row['Low'] <= bin_mid <= row['High']:
                            htf_bins[bin_mid] = htf_bins.get(bin_mid, 0) + row['Volume']
                
                if htf_bins:
                    htf_poc = max(htf_bins.keys(), key=lambda x: htf_bins[x])
                    htf_data['htf_poc'] = float(htf_poc)
            except:
                pass
            
            print(f"   HTF Bearish OBs: {len(htf_data['bearish_obs'])}")
            print(f"   HTF Bullish OBs: {len(htf_data['bullish_obs'])}")
            print(f"   HTF Swing High: ${htf_data['htf_swing_high']:,.2f}")
            print(f"   HTF Swing Low: ${htf_data['htf_swing_low']:,.2f}")
            if htf_data.get('htf_poc'):
                print(f"   HTF POC: ${htf_data['htf_poc']:,.2f}")
            if htf_data.get('htf_vwap'):
                print(f"   HTF VWAP: ${htf_data['htf_vwap']:,.2f}")
            
            if htf_data.get('bearish_obs'):
                print(f"\n   🎯 HTF Bearish OBs (TP targets for LONG):")
                for i, ob in enumerate(htf_data['bearish_obs'][:5]):
                    dist = (ob.get('bottom', 0) - current_price) / current_price * 100
                    print(f"      {i+1}. ${ob.get('bottom', 0):,.2f} - ${ob.get('top', 0):,.2f} ({dist:+.2f}%)")
            
            if htf_data.get('bullish_obs'):
                print(f"\n   🛡️ HTF Bullish OBs (Strong SL zones):")
                for i, ob in enumerate(htf_data['bullish_obs'][:5]):
                    dist = (ob.get('top', 0) - current_price) / current_price * 100
                    print(f"      {i+1}. ${ob.get('bottom', 0):,.2f} - ${ob.get('top', 0):,.2f} ({dist:+.2f}%)")
        else:
            print(f"   ⚠️ detect_smc not available for HTF")
            
    except Exception as e:
        print(f"   ⚠️ HTF fetch failed: {e}")
    
    # STEP 3: Collect levels
    print("\n" + "─" * 80)
    print("📊 STEP 3: LEVEL COLLECTION")
    print("─" * 80)
    
    collector = LevelCollector(df=df, current_price=current_price, timeframe=timeframe)
    print(f"   ATR: ${collector.atr:,.4f} ({collector.atr/current_price*100:.2f}%)")
    
    collector.collect_from_smc(smc_data)
    print(f"   After SMC: {len(collector.levels)} levels")
    
    # HTF levels
    if htf_data:
        collector.collect_from_htf(htf_data)
        print(f"   After HTF: {len(collector.levels)} levels")
    
    # Volume profile - POC, VAH, VAL, VWAP
    collector.collect_volume_profile()
    print(f"   After Volume Profile: {len(collector.levels)} levels")
    
    # Previous day levels
    collector.collect_previous_day_levels()
    print(f"   After Prev Day: {len(collector.levels)} levels")
    
    collector.collect_from_historical()
    collector.collect_fibonacci()
    collector.collect_round_numbers()
    print(f"   Total levels: {len(collector.levels)}")
    
    # Show volume levels specifically
    volume_levels = [l for l in collector.levels if l.level_type.value in ['poc', 'vwap', 'vah', 'val']]
    if volume_levels:
        print(f"\n   📊 VOLUME LEVELS FOUND:")
        for level in volume_levels:
            dist = (level.price - current_price) / current_price * 100
            print(f"      {level.level_type.value.upper()}: ${level.price:,.4f} ({dist:+.2f}%) - {level.description}")
    
    # STEP 4: Categorize
    print("\n" + "─" * 80)
    print(f"📊 STEP 4: TP CANDIDATES FOR {direction}")
    print("─" * 80)
    
    tp_candidates = []
    sl_candidates = []
    
    for level in collector.levels:
        if direction == 'LONG':
            if level.price > current_price:
                tp_candidates.append(level)
            elif level.price < current_price:
                sl_candidates.append(level)
        else:
            if level.price < current_price:
                tp_candidates.append(level)
            elif level.price > current_price:
                sl_candidates.append(level)
    
    if direction == 'LONG':
        tp_candidates.sort(key=lambda x: x.price)
    else:
        tp_candidates.sort(key=lambda x: x.price, reverse=True)
    
    print(f"\n   {'#':>3} | {'Price':>14} | {'Dist %':>8} | {'Type':<20} | {'Description'}")
    print(f"   {'-'*3}-+-{'-'*14}-+-{'-'*8}-+-{'-'*20}-+-{'-'*30}")
    
    for i, level in enumerate(tp_candidates[:20]):
        dist_pct = abs(level.price - current_price) / current_price * 100
        print(f"   {i+1:>3} | ${level.price:>13,.4f} | {dist_pct:>7.2f}% | {level.level_type.value:<20} | {level.description}")
    
    # STEP 5: Scoring
    print("\n" + "─" * 80)
    print(f"📊 STEP 5: TP SCORING ({mode.value.upper()} MODE)")
    print("─" * 80)
    
    constraints = MODE_CONSTRAINTS[mode]
    print(f"   Ideal TP1: {constraints.ideal_tp1_pct}% | TP2: {constraints.ideal_tp2_pct}% | TP3: {constraints.ideal_tp3_pct}%")
    
    def score_level(level, role, entry, constraints):
        score = STRUCTURE_SCORES.get(level.level_type, 10)
        distance_pct = abs(level.price - entry) / entry * 100
        ideal = {'TP1': constraints.ideal_tp1_pct, 'TP2': constraints.ideal_tp2_pct, 'TP3': constraints.ideal_tp3_pct}.get(role, 2.0)
        if ideal > 0:
            deviation = abs(distance_pct - ideal) / ideal
            score += 50 * math.exp(-deviation ** 2)
        return score, distance_pct
    
    print(f"\n   Sorted by TP1 Score:")
    print(f"   {'#':>3} | {'Price':>14} | {'Dist%':>7} | {'Score':>6} | {'Type':<20}")
    print(f"   {'-'*3}-+-{'-'*14}-+-{'-'*7}-+-{'-'*6}-+-{'-'*20}")
    
    scored = [(l, *score_level(l, 'TP1', current_price, constraints)) for l in tp_candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    for i, (level, score, dist) in enumerate(scored[:15]):
        marker = " ⭐ BEST" if i == 0 else ""
        print(f"   {i+1:>3} | ${level.price:>13,.4f} | {dist:>6.2f}% | {score:>6.1f} | {level.level_type.value:<20}{marker}")
    
    # STEP 6: Run optimizer
    print("\n" + "─" * 80)
    print("🎯 STEP 6: OPTIMIZER RESULT")
    print("─" * 80)
    
    try:
        setup = get_optimized_trade(
            df=df,
            current_price=current_price,
            direction=direction,
            timeframe=timeframe,
            smc_data=smc_data,
            htf_data=htf_data,  # Include HTF OBs
        )
        
        tp1_pct = abs(setup.tp1 - current_price) / current_price * 100 if setup.tp1 else 0
        tp2_pct = abs(setup.tp2 - current_price) / current_price * 100 if setup.tp2 else 0
        tp3_pct = abs(setup.tp3 - current_price) / current_price * 100 if setup.tp3 else 0
        
        print(f"\n   Entry: ${setup.entry:,.4f}")
        print(f"   SL: ${setup.stop_loss:,.4f} ({setup.risk_pct:.2f}%) - {setup.sl_type}")
        print(f"\n   TP1: ${setup.tp1:,.4f} ({tp1_pct:.2f}%) R:R {setup.rr_tp1:.1f}:1 - {setup.tp1_type}")
        print(f"   TP2: ${setup.tp2:,.4f} ({tp2_pct:.2f}%) - {setup.tp2_type}")
        print(f"   TP3: ${setup.tp3:,.4f} ({tp3_pct:.2f}%) - {setup.tp3_type}")
        
        # Ordering check
        print(f"\n   ORDERING:")
        if direction == 'LONG':
            if setup.tp1 < setup.tp2 < setup.tp3:
                print(f"   ✅ TP1 < TP2 < TP3 - CORRECT!")
            else:
                print(f"   ❌ BUG! TP1={tp1_pct:.2f}%, TP2={tp2_pct:.2f}%, TP3={tp3_pct:.2f}%")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """Run with sample data when network unavailable"""
    print("\n" + "=" * 80)
    print("🔍 TRADE OPTIMIZER DIAGNOSTIC - DEMO MODE (Sample Data)")
    print("=" * 80)
    
    # Sample BTC data
    current_price = 94500.0
    
    # Create sample df
    np.random.seed(42)
    close_prices = [current_price + np.cumsum(np.random.randn(100) * 100)[-1] for _ in range(100)]
    close_prices[-1] = current_price
    
    df = pd.DataFrame({
        'Open': [p - 50 for p in close_prices],
        'High': [p + 200 for p in close_prices],
        'Low': [p - 200 for p in close_prices],
        'Close': close_prices,
        'Volume': [1000] * 100,
    })
    
    # Sample SMC data (what your real detector would return)
    smc_data = {
        'bearish_obs': [
            {'top': 95200, 'bottom': 95000},   # ~0.5%
            {'top': 96000, 'bottom': 95800},   # ~1.4%
            {'top': 96800, 'bottom': 96500},   # ~2.1%
            {'top': 98000, 'bottom': 97500},   # ~3.2%
        ],
        'bullish_obs': [
            {'top': 94200, 'bottom': 94000},   # -0.3%
            {'top': 93500, 'bottom': 93200},   # -1.1%
            {'top': 92800, 'bottom': 92500},   # -1.8%
        ],
        'swing_high': 96495,
        'swing_low': 92000,
    }
    
    print(f"\n   Current Price: ${current_price:,.2f}")
    print(f"\n   📋 SMC DATA (sample):")
    print(f"\n   Bearish OBs (TP targets):")
    for i, ob in enumerate(smc_data['bearish_obs']):
        dist = (ob['bottom'] - current_price) / current_price * 100
        print(f"      {i+1}. ${ob['bottom']:,.0f} - ${ob['top']:,.0f} ({dist:+.2f}%)")
    
    print(f"\n   Bullish OBs (SL structure):")
    for i, ob in enumerate(smc_data['bullish_obs']):
        dist = (ob['top'] - current_price) / current_price * 100
        print(f"      {i+1}. ${ob['bottom']:,.0f} - ${ob['top']:,.0f} ({dist:+.2f}%)")
    
    # Collect levels
    collector = LevelCollector(df=df, current_price=current_price, timeframe='15m')
    collector.collect_from_smc(smc_data)
    collector.collect_from_historical()
    collector.collect_fibonacci()
    collector.collect_round_numbers()
    
    # Categorize
    tp_candidates = [l for l in collector.levels if l.price > current_price]
    tp_candidates.sort(key=lambda x: x.price)
    
    print(f"\n" + "─" * 80)
    print(f"📊 ALL TP CANDIDATES ({len(tp_candidates)} levels)")
    print("─" * 80)
    print(f"\n   {'#':>3} | {'Price':>12} | {'Dist %':>8} | {'Type':<20} | {'Description'}")
    print(f"   {'-'*3}-+-{'-'*12}-+-{'-'*8}-+-{'-'*20}-+-{'-'*25}")
    
    for i, level in enumerate(tp_candidates[:20]):
        dist_pct = abs(level.price - current_price) / current_price * 100
        print(f"   {i+1:>3} | ${level.price:>11,.2f} | {dist_pct:>7.2f}% | {level.level_type.value:<20} | {level.description}")
    
    # Score for each TP role
    constraints = MODE_CONSTRAINTS[TradingMode.DAY_TRADE]
    
    print(f"\n" + "─" * 80)
    print(f"📊 SCORING (Day Trade: ideal TP1={constraints.ideal_tp1_pct}%, TP2={constraints.ideal_tp2_pct}%, TP3={constraints.ideal_tp3_pct}%)")
    print("─" * 80)
    
    def score_level(level, role, entry, constraints):
        struct_score = STRUCTURE_SCORES.get(level.level_type, 10)
        distance_pct = abs(level.price - entry) / entry * 100
        ideal = {'TP1': constraints.ideal_tp1_pct, 'TP2': constraints.ideal_tp2_pct, 'TP3': constraints.ideal_tp3_pct}.get(role, 2.0)
        mode_score = 50 * math.exp(-((distance_pct - ideal) / ideal) ** 2) if ideal > 0 else 0
        return struct_score + mode_score, struct_score, mode_score, distance_pct
    
    # Show scores for TP1, TP2, TP3
    for role in ['TP1', 'TP2', 'TP3']:
        ideal = {'TP1': constraints.ideal_tp1_pct, 'TP2': constraints.ideal_tp2_pct, 'TP3': constraints.ideal_tp3_pct}[role]
        print(f"\n   {role} Scoring (ideal = {ideal}%):")
        print(f"   {'Price':>12} | {'Dist%':>7} | {'Struct':>6} | {'Mode':>6} | {'TOTAL':>6} | {'Type':<18}")
        print(f"   {'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*18}")
        
        scored = [(l, *score_level(l, role, current_price, constraints)) for l in tp_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        for i, (level, total, struct, mode_s, dist) in enumerate(scored[:8]):
            marker = " ⭐" if i == 0 else ""
            print(f"   ${level.price:>11,.2f} | {dist:>6.2f}% | {struct:>6.0f} | {mode_s:>6.1f} | {total:>6.1f} | {level.level_type.value:<18}{marker}")
    
    # Run optimizer
    print(f"\n" + "─" * 80)
    print("🎯 OPTIMIZER RESULT")
    print("─" * 80)
    
    setup = get_optimized_trade(
        df=df,
        current_price=current_price,
        direction='LONG',
        timeframe='15m',
        smc_data=smc_data,
    )
    
    tp1_pct = abs(setup.tp1 - current_price) / current_price * 100
    tp2_pct = abs(setup.tp2 - current_price) / current_price * 100
    tp3_pct = abs(setup.tp3 - current_price) / current_price * 100
    
    print(f"\n   Entry: ${setup.entry:,.2f}")
    print(f"   SL: ${setup.stop_loss:,.2f} ({setup.risk_pct:.2f}%) - {setup.sl_type}")
    print(f"\n   TP1: ${setup.tp1:,.2f} ({tp1_pct:.2f}%) R:R {setup.rr_tp1:.1f}:1 - {setup.tp1_type}")
    print(f"   TP2: ${setup.tp2:,.2f} ({tp2_pct:.2f}%) - {setup.tp2_type}")
    print(f"   TP3: ${setup.tp3:,.2f} ({tp3_pct:.2f}%) - {setup.tp3_type}")
    
    print(f"\n   ORDERING: ", end="")
    if setup.tp1 < setup.tp2 < setup.tp3:
        print("✅ TP1 < TP2 < TP3 - CORRECT!")
    else:
        print(f"❌ BUG!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        symbol = sys.argv[1].upper()
        timeframe = sys.argv[2]
        direction = sys.argv[3].upper()
    elif len(sys.argv) >= 2:
        symbol = sys.argv[1].upper()
        timeframe = "15m"
        direction = "LONG"
    else:
        symbol = "LITUSDT"
        timeframe = "15m"
        direction = "LONG"
    
    # Try real data first, fall back to demo
    try:
        run_diagnostic(symbol, timeframe, direction)
    except Exception as e:
        print(f"\n⚠️ Could not fetch real data: {e}")
        print("Running demo mode with sample data...")
        run_demo()