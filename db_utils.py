import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np

# =================================================================
# === БЛОК 1: ФУНКЦИИ УПРАВЛЕНИЯ БАЗОЙ ДАННЫХ (db_utils.py) ===
# =================================================================

# === Путь к базе данных ===
DB_PATH = Path("data") / "measurements.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # Создаем папку 'data', если ее нет


# === Подключение к БД ===
def get_conn():
    """Создаёт и возвращает подключение к SQLite (без конфликтов потоков)."""
    # check_same_thread=False необходим для корректной работы в Streamlit
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


# === Вставка нового измерения (CREATE) ===
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
        st.error(f"Ошибка при добавлении записи в БД: {e}")
    finally:
        conn.close()


# === Получение всех измерений (READ) ===
@st.cache_data(ttl=60)  # Кэширование данных на 60 секунд для оптимизации
def fetch_measurements(limit=1000):
    """
    Возвращает DataFrame последних измерений.
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
        st.error(f"Ошибка при чтении из БД: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


# === Удаление всех измерений (DELETE) ===
def delete_all_measurements():
    """Полностью очищает таблицу измерений."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM measurements")
        conn.commit()
    except Exception as e:
        st.error(f"Ошибка при удалении: {e}")
    finally:
        conn.close()


# === Проверка целостности и автоматическая инициализация ===
def ensure_db_ready():
    """Проверяет наличие таблицы и создаёт её при необходимости."""
    try:
        init_db()
    except Exception as e:
        st.error(f"Критическая ошибка инициализации БД: {e}")


# =================================================================
# === БЛОК 2: ПРИЛОЖЕНИЕ STREAMLIT (Интерфейс) ===
# =================================================================

# --- Инициализация и настройка страницы ---
st.set_page_config(
    page_title="🧪 CRUD: Лабораторные Измерения Жая",
    page_icon="🧪",
    layout="wide"
)

# Убедимся, что база данных и таблица созданы перед началом работы
ensure_db_ready()


# --- Вспомогательная функция для обновления страницы ---
def refresh_page():
    """Перезапускает скрипт для обновления данных в таблице."""
    # Принудительная очистка кэша данных при обновлении
    st.cache_data.clear()
    st.rerun()


# --- Заголовок ---
st.title("🧪 Лабораторный Журнал: CRUD Операции")
st.subheader("Управление данными контроля качества для производства Жая")
st.markdown("---")

# --- 1. СЕКЦИЯ: C (CREATE) - Добавление нового измерения ---
st.header("➕ Добавить Новое Измерение")

with st.form(key='add_measurement_form'):
    col1, col2, col3 = st.columns(3)

    # Поля ввода
    with col1:
        sample_name = st.text_input("Название Пробы", help="Например: Партия-Жая-2025-001")

    with col2:
        ph = st.number_input("pH (Кислотность)", min_value=1.0, max_value=14.0, format="%.2f", step=0.01)

    with col3:
        score = st.number_input("Оценка (Score)", min_value=0.0, max_value=10.0, format="%.1f", step=0.1)

    notes = st.text_area("Дополнительные Заметки (Обязательно для аномалий)", height=50)

    # Кнопка отправки формы (размещена внизу формы по умолчанию)
    submit_button = st.form_submit_button(label='Сохранить Измерение ✅')

    # Логика отправки формы
    if submit_button:
        if sample_name:
            # Вызов функции вставки
            insert_measurement(sample_name, ph, score, notes)
            st.success(f"Измерение '{sample_name}' успешно добавлено.")
            # Перезапускаем приложение для обновления таблицы
            refresh_page()
        else:
            st.error("Пожалуйста, введите Название Пробы.")

st.markdown("---")

# --- 2. СЕКЦИЯ: R (READ) и D (DELETE) - Просмотр и управление данными ---
st.header("📋 Просмотр Журнала Измерений (READ)")

# Загружаем данные (благодаря кэшированию, запрос к БД будет выполнен только раз в 60с или при refresh_page)
df_measurements = fetch_measurements()

if df_measurements.empty:
    st.info("Журнал измерений пуст.")
else:
    # Отображение данных
    st.subheader(f"Актуальные записи (Всего: {len(df_measurements)})")

    # Используем st.dataframe с конфигурацией колонок
    st.dataframe(
        df_measurements,
        use_container_width=True,
        hide_index=True,
        column_order=('created_at', 'sample_name', 'ph', 'score', 'notes', 'id'),
        column_config={
            "created_at": st.column_config.DatetimeColumn("Дата Измерения", format="D MMM YY, HH:mm"),
            "sample_name": st.column_config.Column("Проба"),
            "ph": st.column_config.NumberColumn("pH", format="%.2f"),
            "score": st.column_config.NumberColumn("Оценка", format="%.1f"),
            "notes": st.column_config.Column("Заметки"),
            "id": st.column_config.Column("ID", disabled=True),
        }
    )

    st.markdown("---")

    # D (DELETE) - Очистка всех данных
    st.subheader("⚠️ Удаление Данных")

    # Кнопка удаления с подтверждением
    delete_col, _ = st.columns([0.3, 0.7])
    with delete_col:
        # Используем session_state для реализации двухэтапного подтверждения
        if st.button("🔴 Удалить ВСЕ Измерения (Очистить Журнал)", type="primary"):
            if st.session_state.get('confirm_delete', False):
                delete_all_measurements()
                st.success("Все записи успешно удалены!")
                del st.session_state['confirm_delete']  # Сброс состояния
                refresh_page()
            else:
                st.session_state['confirm_delete'] = True
                st.warning("Вы уверены? Это действие необратимо! Нажмите **красную кнопку еще раз** для подтверждения.")
        else:
            # Сброс состояния, если пользователь нажал в другом месте
            if 'confirm_delete' in st.session_state:
                del st.session_state['confirm_delete']