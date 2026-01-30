import sqlite3
conn = sqlite3.connect("data/whale_history.db")
cursor = conn.cursor()

# Check ATR values
cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE atr_pct IS NOT NULL AND atr_pct > 0")
has_atr = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM whale_snapshots WHERE atr_pct IS NULL OR atr_pct = 0")
no_atr = cursor.fetchone()[0]

cursor.execute("SELECT AVG(atr_pct) FROM whale_snapshots WHERE atr_pct > 0")
avg_atr = cursor.fetchone()[0]

print(f"Records WITH ATR: {has_atr:,}")
print(f"Records WITHOUT ATR: {no_atr:,}")
print(f"Average ATR: {avg_atr}%")

# Sample some ATR values
cursor.execute("SELECT symbol, atr_pct FROM whale_snapshots WHERE atr_pct > 0 LIMIT 5")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}%")

conn.close()
