# db_utils.py
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime

DB_PATH = Path("data") / "measurements.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_name TEXT,
        ph REAL,
        score REAL,
        notes TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_measurement(sample_name, ph=None, score=None, notes=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO measurements (sample_name, ph, score, notes, created_at) VALUES (?,?,?,?,?)",
                (sample_name, ph, score, notes, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def fetch_measurements(limit=1000):
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM measurements ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df

def delete_all_measurements():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM measurements")
    conn.commit()
    conn.close()
