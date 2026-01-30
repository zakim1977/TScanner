#!/usr/bin/env python3
"""
Fix position_in_range in existing historical data.

The historical import had position_in_range hardcoded to 50 for all records.
This script calculates the actual position based on price relative to 20-bar high/low.

Run: python fix_position_range.py
"""

import sqlite3
import os
from datetime import datetime
from collections import defaultdict

DB_PATH = "data/whale_history.db"

def fix_position_in_range():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    print("=" * 60)
    print("FIXING position_in_range IN HISTORICAL DATA")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all symbols
    cursor.execute("SELECT DISTINCT symbol FROM whale_snapshots")
    symbols = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(symbols)} symbols to process")
    
    total_updated = 0
    
    for idx, symbol in enumerate(symbols):
        print(f"\n[{idx+1}/{len(symbols)}] Processing {symbol}...")
        
        # Get all records for this symbol, ordered by timestamp
        cursor.execute("""
            SELECT id, timestamp, price 
            FROM whale_snapshots 
            WHERE symbol = ? 
            ORDER BY timestamp ASC
        """, (symbol,))
        
        rows = cursor.fetchall()
        
        if len(rows) < 20:
            print(f"  ⏭️ Only {len(rows)} records - skipping (need 20+)")
            continue
        
        prices = [row[2] for row in rows]
        updates = []
        
        for i, (record_id, timestamp, price) in enumerate(rows):
            if price <= 0:
                continue
            
            # Calculate position based on last 20 bars
            if i >= 20:
                recent_prices = [p for p in prices[max(0, i-20):i+1] if p > 0]
                if recent_prices:
                    high_20 = max(recent_prices)
                    low_20 = min(recent_prices)
                    
                    if high_20 > low_20:
                        position = ((price - low_20) / (high_20 - low_20)) * 100
                        position = max(0, min(100, position))
                    else:
                        position = 50
                else:
                    position = 50
            else:
                # First 20 bars - use available data
                recent_prices = [p for p in prices[:i+1] if p > 0]
                if len(recent_prices) >= 5:
                    high = max(recent_prices)
                    low = min(recent_prices)
                    if high > low:
                        position = ((price - low) / (high - low)) * 100
                        position = max(0, min(100, position))
                    else:
                        position = 50
                else:
                    position = 50
            
            updates.append((position, record_id))
        
        # Batch update
        if updates:
            cursor.executemany("""
                UPDATE whale_snapshots 
                SET position_in_range = ? 
                WHERE id = ?
            """, updates)
            conn.commit()
            total_updated += len(updates)
            print(f"  ✅ Updated {len(updates)} records")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE: Updated {total_updated:,} records")
    print("=" * 60)
    print("\nNow position_in_range reflects actual price position!")
    print("Re-run your analysis to see historical validation results.")


if __name__ == "__main__":
    fix_position_range()
