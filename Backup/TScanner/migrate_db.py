#!/usr/bin/env python
"""
Database Migration Script
=========================
Run this to fix existing databases that are missing columns.

Usage:
    python migrate_db.py
"""

import sqlite3
import os

DB_PATH = "data/whale_history.db"

def migrate():
    """Add missing columns to existing database"""
    
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        print("No migration needed - database will be created fresh.")
        return
    
    print(f"Migrating database: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(whale_snapshots)")
    columns = [col[1] for col in cursor.fetchall()]
    
    print(f"Current columns: {columns}")
    
    # Add data_source if missing
    if 'data_source' not in columns:
        print("Adding 'data_source' column...")
        cursor.execute("ALTER TABLE whale_snapshots ADD COLUMN data_source TEXT DEFAULT 'live'")
        print("✓ Added data_source column")
    else:
        print("✓ data_source column already exists")
    
    # Create index if missing
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_data_source 
        ON whale_snapshots(data_source)
    """)
    print("✓ Index created/verified")
    
    conn.commit()
    
    # Verify
    cursor.execute("PRAGMA table_info(whale_snapshots)")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"\nFinal columns: {columns}")
    
    # Show stats
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE data_source = 'live'")
    live = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE data_source = 'historical'")
    historical = cursor.fetchone()[0]
    
    print(f"\nDatabase stats:")
    print(f"  Total records: {total:,}")
    print(f"  Live records: {live:,}")
    print(f"  Historical records: {historical:,}")
    
    conn.close()
    print("\n✅ Migration complete!")

if __name__ == "__main__":
    migrate()
