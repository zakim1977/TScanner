#!/usr/bin/env python3
"""
ONE-TIME Backtest for Historical Data Outcomes
===============================================

For existing data that doesn't have win/loss outcomes tracked.

NOTE: After running this once, outcomes are calculated AUTOMATICALLY:
- On app startup (processes 20 symbols per session)
- During historical import (processes each symbol as imported)

So you only need to run this script ONCE to process your existing 265K records.

For each historical snapshot, looks at price movement AFTER that point
to determine if it would have hit TP1 (win) or SL (loss).

Uses simple logic:
- LONG setup: If price goes UP 3%+ before going DOWN 2% = WIN
- SHORT setup: If price goes DOWN 3%+ before going UP 2% = WIN
- Neutral (WAIT): Use price direction to determine outcome

Run: python backtest_outcomes.py
"""

import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = "data/whale_history.db"

# Thresholds - MATCH SignalGenerator logic!
# SignalGenerator uses: SL = ATR * 2, TP1 = Risk * 1.5 (1.5:1 R:R)
SL_ATR_MULT = 2.0    # SL at 2x ATR (same as level_calculator)
TP_RR_RATIO = 1.5    # TP1 at 1.5:1 R:R (same as level_calculator)
MAX_BARS = 8         # Look ahead 8 bars (~32 hours for 4h) - reduced for realism
DEFAULT_ATR = 4.0    # Default ATR% for crypto (was 2% - too low!)


def backtest_outcomes():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    print("=" * 60)
    print("BACKTESTING HISTORICAL DATA FOR OUTCOMES")
    print("=" * 60)
    print(f"Using ATR-based thresholds (matches SignalGenerator):")
    print(f"  SL = ATR × {SL_ATR_MULT}  |  TP1 = SL × {TP_RR_RATIO} (1.5:1 R:R)  |  Lookahead: {MAX_BARS} bars")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # DIAGNOSTIC: Check whale_long_pct distribution
    cursor.execute("""
        SELECT 
            COUNT(CASE WHEN whale_long_pct >= 60 THEN 1 END) as bullish,
            COUNT(CASE WHEN whale_long_pct <= 40 THEN 1 END) as bearish,
            COUNT(CASE WHEN whale_long_pct > 40 AND whale_long_pct < 60 THEN 1 END) as neutral,
            AVG(whale_long_pct) as avg_whale
        FROM whale_snapshots
    """)
    diag = cursor.fetchone()
    print(f"\n📊 Data Distribution:")
    print(f"  Bullish (whale≥60%): {diag[0]:,}")
    print(f"  Bearish (whale≤40%): {diag[1]:,}")
    print(f"  Neutral (40-60%): {diag[2]:,}")
    print(f"  Average whale%: {diag[3]:.1f}%")
    
    if diag[2] > (diag[0] + diag[1]) * 2:
        print(f"\n⚠️  WARNING: Most data is NEUTRAL - win rates will be inflated!")
        print(f"   (Historical imports don't have real whale positioning)")
    
    # Get all symbols
    cursor.execute("SELECT DISTINCT symbol FROM whale_snapshots ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    print(f"\nFound {len(symbols)} symbols to process")
    
    total_processed = 0
    total_wins = 0
    total_losses = 0
    
    for idx, symbol in enumerate(symbols):
        print(f"\n[{idx+1}/{len(symbols)}] Processing {symbol}...", end=" ")
        
        # Get all records for this symbol, ordered by timestamp
        # INCLUDE atr_pct AND high/low for ATR calculation
        cursor.execute("""
            SELECT id, timestamp, price, whale_long_pct, signal_direction, atr_pct 
            FROM whale_snapshots 
            WHERE symbol = ? 
            ORDER BY timestamp ASC
        """, (symbol,))
        
        rows = cursor.fetchall()
        
        if len(rows) < MAX_BARS + 1:
            print(f"⏭️ Only {len(rows)} records (need {MAX_BARS+1}+)")
            continue
        
        prices = [row[2] for row in rows]
        sym_wins = 0
        sym_losses = 0
        updates = []
        
        # Pre-calculate ATR for each point using price volatility
        def calc_atr_at(idx):
            """Calculate ATR% from price changes over last 14 bars"""
            if idx < 14:
                return DEFAULT_ATR
            window = prices[idx-14:idx]
            if not window or min(p for p in window if p and p > 0) <= 0:
                return DEFAULT_ATR
            # Use absolute price changes as ATR proxy
            changes = []
            for i in range(1, len(window)):
                if window[i] and window[i-1] and window[i-1] > 0:
                    changes.append(abs(window[i] - window[i-1]) / window[i-1] * 100)
            return sum(changes) / len(changes) if changes else DEFAULT_ATR
        
        for i in range(len(rows) - MAX_BARS):
            record_id, timestamp, entry_price, whale_pct, signal_dir, atr_pct = rows[i]
            
            if entry_price <= 0:
                continue
            
            # CALCULATE ATR from price data if not stored
            if atr_pct and atr_pct > 0:
                atr = atr_pct
            else:
                atr = calc_atr_at(i)
            
            # MATCH SignalGenerator: SL = ATR * 2, TP = SL * 1.5
            sl_pct = atr * SL_ATR_MULT  # e.g., 4% * 2.0 = 8%
            tp_pct = sl_pct * TP_RR_RATIO  # e.g., 8% * 1.5 = 12%
            
            # Debug first symbol
            if idx == 0 and i < 3:
                print(f"\n  [DEBUG] ATR={atr:.2f}%, SL={sl_pct:.2f}%, TP={tp_pct:.2f}%")
            
            # Determine expected direction
            # Whale > 60% = expect LONG, Whale < 40% = expect SHORT
            if whale_pct >= 60:
                expected_dir = 'LONG'
            elif whale_pct <= 40:
                expected_dir = 'SHORT'
            else:
                expected_dir = 'NEUTRAL'
            
            # Look at future prices
            future_prices = [rows[j][2] for j in range(i+1, min(i+MAX_BARS+1, len(rows)))]
            
            if not future_prices:
                continue
            
            # Track max favorable and adverse excursion
            max_up = 0
            max_down = 0
            
            for fp in future_prices:
                if fp <= 0:
                    continue
                pct_change = ((fp - entry_price) / entry_price) * 100
                if pct_change > max_up:
                    max_up = pct_change
                if pct_change < max_down:
                    max_down = pct_change
            
            # Determine outcome using DYNAMIC thresholds
            hit_tp1 = 0
            hit_sl = 0
            candles_to_result = 0
            
            if expected_dir == 'LONG':
                # Check which came first - TP or SL
                for j, fp in enumerate(future_prices):
                    if fp <= 0:
                        continue
                    pct_change = ((fp - entry_price) / entry_price) * 100
                    if pct_change >= tp_pct:
                        hit_tp1 = 1
                        candles_to_result = j + 1
                        sym_wins += 1
                        break
                    elif pct_change <= -sl_pct:
                        hit_sl = 1
                        candles_to_result = j + 1
                        sym_losses += 1
                        break
                        
            elif expected_dir == 'SHORT':
                # For SHORT, down is good, up is bad
                for j, fp in enumerate(future_prices):
                    if fp <= 0:
                        continue
                    pct_change = ((fp - entry_price) / entry_price) * 100
                    if pct_change <= -tp_pct:  # Price went DOWN = win for short
                        hit_tp1 = 1
                        candles_to_result = j + 1
                        sym_wins += 1
                        break
                    elif pct_change >= sl_pct:  # Price went UP = loss for short
                        hit_sl = 1
                        candles_to_result = j + 1
                        sym_losses += 1
                        break
            else:
                # Neutral - check which significant move comes FIRST
                # If price drops by SL% first = loss, if rises by TP% first = win
                for j, fp in enumerate(future_prices):
                    if fp <= 0:
                        continue
                    pct_change = ((fp - entry_price) / entry_price) * 100
                    if pct_change >= tp_pct:  # Up move first = win
                        hit_tp1 = 1
                        candles_to_result = j + 1
                        sym_wins += 1
                        break
                    elif pct_change <= -sl_pct:  # Down move first = loss
                        hit_sl = 1
                        candles_to_result = j + 1
                        sym_losses += 1
                        break
            
            updates.append((
                hit_tp1, 
                hit_sl, 
                max_up,           # max_favorable_pct
                max_down,         # max_adverse_pct
                candles_to_result,
                record_id
            ))
        
        # Batch update
        if updates:
            cursor.executemany("""
                UPDATE whale_snapshots 
                SET hit_tp1 = ?, 
                    hit_sl = ?,
                    max_favorable_pct = ?,
                    max_adverse_pct = ?,
                    candles_to_result = ?
                WHERE id = ?
            """, updates)
            conn.commit()
            
            total_processed += len(updates)
            total_wins += sym_wins
            total_losses += sym_losses
            
            win_rate = (sym_wins / (sym_wins + sym_losses) * 100) if (sym_wins + sym_losses) > 0 else 0
            print(f"✅ {len(updates)} records | W:{sym_wins} L:{sym_losses} ({win_rate:.0f}%)")
        else:
            print("⚠️ No valid records")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ CRYPTO BACKTEST COMPLETE")
    print("=" * 60)
    print(f"Total processed: {total_processed:,}")
    print(f"Wins: {total_wins:,}  |  Losses: {total_losses:,}")
    if total_wins + total_losses > 0:
        overall_wr = (total_wins / (total_wins + total_losses)) * 100
        print(f"Overall Win Rate: {overall_wr:.1f}%")


def backtest_stock_outcomes():
    """Backtest stock/ETF historical data with detailed output"""
    
    print("\n" + "=" * 60)
    print("BACKTESTING STOCK/ETF DATA FOR OUTCOMES")
    print("=" * 60)
    
    try:
        from core.stock_data_store import get_stock_store
        store = get_stock_store()
        
        symbols = store.get_all_symbols()
        if not symbols:
            print("❌ No stock data found. Run Quiver import first.")
            return
        
        print(f"Found {len(symbols)} stocks/ETFs to process")
        print(f"Using: SL=2% | TP=3% (1.5:1 R:R) | Lookahead: 10 bars")
        
        # Stock thresholds
        SL_PCT = 2.0
        TP_PCT = 3.0  # 1.5:1 R:R
        MAX_BARS = 10
        
        total_wins = 0
        total_losses = 0
        total_processed = 0
        
        for idx, symbol in enumerate(symbols):
            print(f"\n[{idx+1}/{len(symbols)}] Processing {symbol}...", end=" ")
            
            data = store._load_symbol_data(symbol)
            if len(data) < MAX_BARS + 5:
                print(f"⏭️ Only {len(data)} records (need {MAX_BARS+5}+)")
                continue
            
            # Sort by timestamp
            data.sort(key=lambda x: x.get('timestamp', ''))
            prices = [d.get('price', 0) for d in data]
            
            sym_wins = 0
            sym_losses = 0
            updated = 0
            
            for i in range(len(data) - MAX_BARS):
                price = prices[i]
                if not price or price <= 0:
                    continue
                
                # Determine direction from institutional score
                inst_score = data[i].get('congress_score', 50) or data[i].get('combined_score', 50)
                if inst_score and inst_score >= 60:
                    expected_dir = 'LONG'
                elif inst_score and inst_score <= 40:
                    expected_dir = 'SHORT'
                else:
                    expected_dir = 'NEUTRAL'
                
                # Look ahead
                future_prices = [p for p in prices[i+1:i+MAX_BARS+1] if p and p > 0]
                if len(future_prices) < 3:
                    continue
                
                # Calculate max moves
                max_up = max((p - price) / price * 100 for p in future_prices)
                max_down = min((p - price) / price * 100 for p in future_prices)
                
                hit_tp1 = False
                hit_sl = False
                candles = 0
                
                # Check which comes first
                for j, fp in enumerate(future_prices):
                    pct = (fp - price) / price * 100
                    
                    if expected_dir == 'LONG':
                        if pct >= TP_PCT:
                            hit_tp1 = True
                            candles = j + 1
                            sym_wins += 1
                            break
                        elif pct <= -SL_PCT:
                            hit_sl = True
                            candles = j + 1
                            sym_losses += 1
                            break
                    elif expected_dir == 'SHORT':
                        if pct <= -TP_PCT:
                            hit_tp1 = True
                            candles = j + 1
                            sym_wins += 1
                            break
                        elif pct >= SL_PCT:
                            hit_sl = True
                            candles = j + 1
                            sym_losses += 1
                            break
                    else:
                        # Neutral
                        if pct >= TP_PCT:
                            hit_tp1 = True
                            candles = j + 1
                            sym_wins += 1
                            break
                        elif pct <= -SL_PCT:
                            hit_sl = True
                            candles = j + 1
                            sym_losses += 1
                            break
                
                # Update record
                data[i]['hit_tp1'] = hit_tp1
                data[i]['hit_sl'] = hit_sl
                data[i]['max_favorable_pct'] = max_up
                data[i]['max_adverse_pct'] = max_down
                data[i]['candles_to_result'] = candles
                data[i]['outcome_tracked'] = True
                updated += 1
            
            # Save and show results
            if updated > 0:
                store._save_symbol_data(symbol, data)
                total_processed += updated
                total_wins += sym_wins
                total_losses += sym_losses
                
                win_rate = (sym_wins / (sym_wins + sym_losses) * 100) if (sym_wins + sym_losses) > 0 else 0
                print(f"✅ {updated} records | W:{sym_wins} L:{sym_losses} ({win_rate:.0f}%)")
            else:
                print("⚠️ No valid records")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ STOCK BACKTEST COMPLETE")
        print("=" * 60)
        print(f"Total processed: {total_processed:,}")
        print(f"Wins: {total_wins:,}  |  Losses: {total_losses:,}")
        if total_wins + total_losses > 0:
            overall_wr = (total_wins / (total_wins + total_losses)) * 100
            print(f"Overall Win Rate: {overall_wr:.1f}%")
        
    except Exception as e:
        import traceback
        print(f"❌ Stock backtest error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--reset':
        # Reset all crypto outcomes to re-run backtest
        print("Resetting all CRYPTO outcomes...")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            UPDATE whale_snapshots 
            SET hit_tp1=NULL, hit_sl=NULL, candles_to_result=NULL,
                max_favorable_pct=NULL, max_adverse_pct=NULL
        """)
        conn.commit()
        count = conn.execute("SELECT changes()").fetchone()[0]
        conn.close()
        print(f"✅ Reset {count} records. Now run: python backtest_outcomes.py")
    
    elif len(sys.argv) > 1 and sys.argv[1] == '--reset-stocks':
        # Reset all stock outcomes
        print("Resetting all STOCK outcomes...")
        try:
            from core.stock_data_store import get_stock_store
            store = get_stock_store()
            symbols = store.get_all_symbols()
            reset_count = 0
            for symbol in symbols:
                data = store._load_symbol_data(symbol)
                for d in data:
                    d['outcome_tracked'] = False
                    d['hit_tp1'] = False
                    d['hit_sl'] = False
                    d['candles_to_result'] = 0
                    d['max_favorable_pct'] = 0
                    d['max_adverse_pct'] = 0
                store._save_symbol_data(symbol, data)
                reset_count += len(data)
            print(f"✅ Reset {reset_count} records across {len(symbols)} symbols.")
            print("Now run: python backtest_outcomes.py --stocks")
        except Exception as e:
            print(f"❌ Error: {e}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--stocks':
        backtest_stock_outcomes()
    elif len(sys.argv) > 1 and sys.argv[1] == '--all':
        backtest_outcomes()
        backtest_stock_outcomes()
    else:
        print("Usage:")
        print("  python backtest_outcomes.py               # Crypto only (default)")
        print("  python backtest_outcomes.py --stocks      # Stocks/ETFs only")
        print("  python backtest_outcomes.py --all         # Both crypto and stocks")
        print("  python backtest_outcomes.py --reset       # Clear crypto outcomes")
        print("  python backtest_outcomes.py --reset-stocks # Clear stock outcomes")
        print("")
        backtest_outcomes()
        print("\nNow historical validation will show actual win rates!")
        print("\n💡 Tip: Run with --stocks to backtest stock data too!")
