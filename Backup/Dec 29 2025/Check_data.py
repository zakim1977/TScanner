import sqlite3
conn = sqlite3.connect('data/whale_history.db')
c = conn.cursor()
c.execute("SELECT whale_long_pct, retail_long_pct FROM whale_snapshots WHERE symbol='BTCUSDT' AND data_source='historical' ORDER BY timestamp DESC LIMIT 5")
for row in c.fetchall():
    print(f"Whale: {row[0]}%, Retail: {row[1]}%")