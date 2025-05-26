import sqlite3

def insert_analysis(image_path, heatmap_path, result, pneumonia_prob, normal_prob, analyzed_at):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO analyses (
            image_path, heatmap_path, result, pneumonia_prob, normal_prob, analyzed_at
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (image_path, heatmap_path, result, pneumonia_prob, normal_prob, analyzed_at))

    conn.commit()
    conn.close()

def get_all_analyses():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM analyses ORDER BY analyzed_at DESC')
    rows = cursor.fetchall()

    conn.close()
    return rows
