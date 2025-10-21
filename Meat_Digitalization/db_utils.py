# db_utils.py  (или db.py)
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime

# === Путь к базе данных ===
DB_PATH = Path("data") / "measurements.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Подключение к БД ===
def get_conn():
    """Создаёт и возвращает подключение к SQLite (без конфликтов потоков)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

# === Инициализация БД ===
def init_db():
    """Создаёт таблицу измерений, если её ещё нет."""
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

# === Вставка нового измерения ===
def insert_measurement(sample_name, ph=None, score=None, notes=None):
    """Добавляет новое измерение в таблицу."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO measurements (sample_name, ph, score, notes, created_at) VALUES (?,?,?,?,?)",
            (sample_name, ph, score, notes, datetime.utcnow().isoformat())
        )
        conn.commit()
    except Exception as e:
        print("Ошибка при добавлении записи в БД:", e)
    finally:
        conn.close()

# === Получение всех измерений ===
def fetch_measurements(limit=1000):
    """
    Возвращает DataFrame последних измерений.
    limit — ограничивает количество строк (по умолчанию 1000).
    """
    conn = get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM measurements ORDER BY created_at DESC LIMIT ?",
            conn,
            params=(limit,)
        )
        if not df.empty:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        return df
    except Exception as e:
        print("Ошибка при чтении из БД:", e)
        return pd.DataFrame()
    finally:
        conn.close()

# === Удаление всех измерений ===
def delete_all_measurements():
    """Полностью очищает таблицу измерений."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM measurements")
        conn.commit()
    except Exception as e:
        print("Ошибка при удалении:", e)
    finally:
        conn.close()

# === Проверка целостности и автоматическая инициализация ===
def ensure_db_ready():
    """
    Проверяет наличие таблицы и создаёт её при необходимости.
    Можно вызывать из app.py перед основными операциями.
    """
    try:
        init_db()
    except Exception as e:
        print("Ошибка инициализации БД:", e)
