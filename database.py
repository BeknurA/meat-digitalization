# Meat_Digitalization/database.py
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime

# ---------------------------
# Конфигурация базы данных
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_FILE = DATA_DIR / "measurements.db"

def get_conn():
    """
    Создает и возвращает соединение с базой данных SQLite.
    """
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    """
    Инициализирует базу данных, создавая таблицу 'measurements', если она не существует.
    """
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
    """
    Вставляет новое измерение в таблицу 'measurements'.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO measurements (sample_name, ph, score, notes, created_at) VALUES (?,?,?,?,?)",
                (sample_name, ph, score, notes, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def fetch_measurements(limit=5000):
    """
    Извлекает измерения из базы данных и возвращает их в виде DataFrame.
    """
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM measurements ORDER BY created_at DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    if not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'])
    return df

def delete_all_measurements():
    """
    Удаляет все записи из таблицы 'measurements'.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM measurements")
    conn.commit()
    conn.close()

# Инициализация базы данных при загрузке модуля
init_db()
