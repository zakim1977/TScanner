#!/usr/bin/env python3
"""
Diagnose historical database issues.
Run: python diagnose_db.py
"""

import sqlite3
import os

DB_PATH = "data/whale_history.db"

def diagnose():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    print("=" * 60)
    print("DATABASE DIAGNOSTIC")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total records
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots")
    total = cursor.fetchone()[0]
    print(f"\n📊 Total records: {total:,}")
    
    # Date range
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM whale_snapshots")
    min_ts, max_ts = cursor.fetchone()
    print(f"📅 Date range: {min_ts} to {max_ts}")
    
    # Check whale_long_pct distribution
    print("\n🐋 Whale % Distribution:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN whale_long_pct IS NULL THEN 'NULL'
                WHEN whale_long_pct = 0 THEN '0'
                WHEN whale_long_pct < 30 THEN '<30'
                WHEN whale_long_pct < 50 THEN '30-50'
                WHEN whale_long_pct < 70 THEN '50-70'
                ELSE '70+'
            END as bucket,
            COUNT(*) as cnt
        FROM whale_snapshots
        GROUP BY bucket
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")
    
    # Check position_in_range distribution
    print("\n📍 Position % Distribution:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN position_in_range IS NULL THEN 'NULL'
                WHEN position_in_range = 50 THEN 'EXACTLY 50 (hardcoded)'
                WHEN position_in_range < 35 THEN 'EARLY (0-35)'
                WHEN position_in_range < 65 THEN 'MIDDLE (35-65)'
                ELSE 'LATE (65-100)'
            END as bucket,
            COUNT(*) as cnt
        FROM whale_snapshots
        GROUP BY bucket
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")
    
    # Check outcome tracking
    print("\n🎯 Outcome Tracking:")
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE hit_tp1 IS NOT NULL OR hit_sl IS NOT NULL")
    with_outcomes = cursor.fetchone()[0]
    print(f"  Records with outcomes: {with_outcomes:,} ({with_outcomes/total*100:.1f}%)")
    
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE hit_tp1 = 1")
    wins = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE hit_sl = 1")
    losses = cursor.fetchone()[0]
    print(f"  Wins: {wins:,}  |  Losses: {losses:,}")
    if wins + losses > 0:
        print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%")
    
    # Check OI change distribution
    print("\n📈 OI Change Distribution:")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN oi_change_24h IS NULL THEN 'NULL'
                WHEN oi_change_24h = 0 THEN 'ZERO'
                WHEN oi_change_24h > 0 THEN 'POSITIVE'
                ELSE 'NEGATIVE'
            END as bucket,
            COUNT(*) as cnt
        FROM whale_snapshots
        GROUP BY bucket
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")
    
    # Test query that SHOULD return results
    print("\n🔍 Test Query (whale 40-80%, no date filter):")
    cursor.execute("""
        SELECT COUNT(*) FROM whale_snapshots 
        WHERE whale_long_pct BETWEEN 40 AND 80
    """)
    test_count = cursor.fetchone()[0]
    print(f"  Matches: {test_count:,}")
    
    # Sample some actual records
    print("\n📝 Sample Records:")
    cursor.execute("""
        SELECT symbol, timestamp, whale_long_pct, position_in_range, oi_change_24h, hit_tp1, hit_sl
        FROM whale_snapshots
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if with_outcomes == 0:
        print("⚠️  No outcomes tracked! Run: python backtest_outcomes.py")
    
    # Check if all positions are 50
    cursor = sqlite3.connect(DB_PATH).cursor()
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE position_in_range = 50")
    all_50 = cursor.fetchone()[0]
    if all_50 == total:
        print("⚠️  All positions are 50 (hardcoded)! Run: python fix_position_range.py")


if __name__ == "__main__":
    diagnose()
