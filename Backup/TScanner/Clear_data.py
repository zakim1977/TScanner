import sqlite3
conn = sqlite3.connect('data/whale_history.db')
conn.execute("DELETE FROM whale_snapshots WHERE data_source='historical'")
conn.commit()
print('Cleared old data!')