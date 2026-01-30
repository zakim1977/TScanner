"""
DEBUG: Whale Delta Verification Script
======================================
Run this in your InvestorIQ directory to see the actual whale data.

Usage: python debug_whale_delta_local.py ZENUSDT
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_whale_delta(symbol='VANAUSDT'):
    print("=" * 70)
    print(f"🔍 WHALE DELTA DEBUG FOR: {symbol}")
    print("=" * 70)
    
    # Step 1: Try to get current live whale data
    print("\n📡 STEP 1: Current LIVE Whale Data (from Binance)")
    print("-" * 50)
    try:
        from core.whale_analysis import get_whale_analysis
        live_whale = get_whale_analysis(symbol)
        if live_whale:
            whale_pct = live_whale.get('whale_pct', 'N/A')
            retail_pct = live_whale.get('retail_pct', 'N/A')
            print(f"   Whale %: {whale_pct}")
            print(f"   Retail %: {retail_pct}")
            print(f"   Direction: {live_whale.get('direction', 'N/A')}")
            
            # Check raw data
            raw = live_whale.get('real_whale_data', {})
            if raw:
                print(f"\n   Raw API data:")
                print(f"   - Long Account: {raw.get('long_account', 'N/A')}%")
                print(f"   - Short Account: {raw.get('short_account', 'N/A')}%")
                print(f"   - Long/Short Ratio: {raw.get('long_short_ratio', 'N/A')}")
        else:
            print("   ❌ Could not fetch live whale data")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Step 2: Check historical database
    print("\n📊 STEP 2: Historical Database Records")
    print("-" * 50)
    try:
        from core.whale_data_store import get_whale_store
        store = get_whale_store()
        
        # Get all snapshots for this symbol
        all_snapshots = store.get_all_snapshots(lookback_days=7, max_records=1000)
        
        if all_snapshots:
            # Filter for this symbol
            symbol_snaps = [s for s in all_snapshots if s.get('symbol') == symbol]
            
            print(f"   Total snapshots in DB: {len(all_snapshots)}")
            print(f"   Snapshots for {symbol}: {len(symbol_snaps)}")
            
            if symbol_snaps:
                # Sort by timestamp
                symbol_snaps = sorted(symbol_snaps, key=lambda x: x.get('timestamp', ''), reverse=True)
                
                print(f"\n   📋 Records for {symbol} (newest first):")
                print("-" * 70)
                print(f"   {'Timestamp':<25} {'Whale%':>8} {'Retail%':>8} {'Source':<15}")
                print("-" * 70)
                
                for snap in symbol_snaps[:20]:  # Show last 20
                    ts = snap.get('timestamp', 'N/A')
                    
                    # Try different keys for whale %
                    whale_pct = snap.get('whale_long_pct') or snap.get('whale_pct')
                    retail_pct = snap.get('retail_long_pct') or snap.get('retail_pct')
                    
                    # Check nested structure
                    if whale_pct is None and 'whale_vs_retail' in snap:
                        whale_pct = snap['whale_vs_retail'].get('whale_pct')
                        retail_pct = snap['whale_vs_retail'].get('retail_pct')
                    
                    source = 'stored'
                    
                    whale_str = f"{float(whale_pct):.1f}" if whale_pct is not None else "N/A"
                    retail_str = f"{float(retail_pct):.1f}" if retail_pct is not None else "N/A"
                    
                    # Format timestamp
                    if ts and len(ts) > 19:
                        ts = ts[:19]
                    
                    print(f"   {ts:<25} {whale_str:>8}% {retail_str:>8}% {source:<15}")
                
                # Calculate actual delta
                if len(symbol_snaps) >= 2:
                    newest = symbol_snaps[0]
                    oldest = symbol_snaps[-1]
                    
                    new_whale = newest.get('whale_long_pct') or newest.get('whale_pct')
                    old_whale = oldest.get('whale_long_pct') or oldest.get('whale_pct')
                    
                    if new_whale is not None and old_whale is not None:
                        delta = float(new_whale) - float(old_whale)
                        print(f"\n   📈 DELTA CALCULATION:")
                        print(f"      Newest: {float(new_whale):.1f}% at {newest.get('timestamp', 'N/A')[:19]}")
                        print(f"      Oldest: {float(old_whale):.1f}% at {oldest.get('timestamp', 'N/A')[:19]}")
                        print(f"      Delta:  {delta:+.1f}%")
                        
                        if abs(delta) > 20:
                            print(f"\n   ⚠️ WARNING: {abs(delta):.1f}% delta is VERY large!")
                            print(f"      Possible causes:")
                            print(f"      - Data from different timeframes mixed?")
                            print(f"      - Stale data in DB?")
                            print(f"      - API returned bad data at some point?")
            else:
                print(f"   ❌ No snapshots found for {symbol}")
                
                # Show what symbols exist
                symbols_in_db = list(set(s.get('symbol', 'UNKNOWN') for s in all_snapshots))
                print(f"\n   Available symbols: {symbols_in_db[:15]}")
        else:
            print("   ❌ No snapshots in database")
            
    except Exception as e:
        import traceback
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
    
    # Step 3: Show what app.py would calculate
    print("\n🧮 STEP 3: What get_whale_delta() Would Return")
    print("-" * 50)
    try:
        # Import the function from app.py
        from app import get_whale_delta
        result = get_whale_delta(symbol, 'day_trade', final_call='LONG', phase='MARKUP')
        
        print(f"   whale_delta: {result.get('whale_delta')}")
        print(f"   retail_delta: {result.get('retail_delta')}")
        print(f"   lookback_label: {result.get('lookback_label')}")
        print(f"   data_available: {result.get('data_available')}")
        print(f"   whale_trend: {result.get('whale_trend')}")
        print(f"   insight: {result.get('insight')}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Done!")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'VANAUSDT'
    debug_whale_delta(symbol)