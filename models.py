import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            heatmap_path TEXT,
            result TEXT,
            pneumonia_prob REAL,
            normal_prob REAL,
            analyzed_at TEXT
        )
    ''')

    conn.commit()
    conn.close()
