

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
import json
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import base64
import time
import os
from dict import LANG
# ---------------------------
# Конфигурация файлов/папок
# ---------------------------
# Используйте фиктивные файлы для демонстрации, если реальные не загружены
MEAT_XLSX = "meat_data.xlsx"
SHEET_NAME = "T6"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_FILE = DATA_DIR / "measurements.db"
MODEL_FILE = DATA_DIR / "model.pkl"
SCALER_FILE = DATA_DIR / "scaler.pkl"

# Если файлы не существуют, создаем фиктивные для работы кода
MEAT_DATA_XLSX = BASE_DIR / "meat_data.xlsx"
OPYTY_XLSX = BASE_DIR / "opyty.xlsx"
PRODUCTS_CSV = BASE_DIR / "Products.csv"
SAMPLES_CSV = BASE_DIR / "Samples.csv"
MEASUREMENTS_CSV = BASE_DIR / "Measurements.csv"

# Создание фиктивных данных, если их нет
if not OPYTY_XLSX.exists():
    temp_df = pd.DataFrame({
        'Time_h': [0, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120, 144],
        'pH_Control': [6.5, 6.4, 6.3, 6.1, 5.8, 5.5, 5.2, 5.1, 5.1, 5.15, 5.2, 5.2],
        'pH_Extract': [6.5, 6.45, 6.35, 6.2, 5.9, 5.6, 5.4, 5.3, 5.3, 5.35, 5.4, 5.4],
        'Salt_pct': [2.5] * 12,
        'Temp_C': [18] * 12
    })
    temp_df.to_excel(OPYTY_XLSX, index=False)

if not MEAT_DATA_XLSX.exists():
    temp_df_meat = pd.DataFrame({
        "BatchID": ["B001", "B002", "B003"],
        "mass_kg": [10.5, 12.0, 9.8],
        "T_initial_C": [2, 3, 2],
        "Salt_pct": [3.0, 3.5, 3.2],
        "Moisture_pct": [72, 70, 71],
        "StarterCFU": [1e6, 2e6, 1.5e6],
        "Extract_pct": [0.0, 3.0, 5.0]
    })
    with pd.ExcelWriter(MEAT_DATA_XLSX, engine='openpyxl') as writer:
        temp_df_meat.to_excel(writer, sheet_name=SHEET_NAME, index=False)

# ---------------------------
# Установки страницы 
# ---------------------------
st.set_page_config(page_title="Платформа Жая — расширенная", layout="wide")

# =================================================================
# 🎨 ENHANCED DESIGN AND ANIMATION - DARK THEME 
# =================================================================
st.markdown("""
<style>
/* 1. Global & Page Config */
.stApp {
    background-color: #111111; /* DARK/Black background */
    color: #f0f0f0; /* Light text for general readability */
}
/* ... (весь остальной CSS код ) ... */
/* Ensure all text within containers is readable */
.st-emotion-cache-1n76c1k, [data-testid="stSidebar"] div, div[data-testid="stForm"] > div > label > div {
    color: #f0f0f0 !important;
}
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
    color: #f0f0f0 !important;
}
/* 2. Fade-In Animation */
.fade-in {
  animation: fadeIn ease 0.5s;
}
@keyframes fadeIn {
  0% {opacity:0; transform:translateY(6px)}
  100% {opacity:1; transform:translateY(0)}
}
.small-muted {color: #a0a0a0; font-size:0.9em;}
/* 3. Title Animation */
.main-title-animation {
    animation: pulseTitle 1.5s infinite alternate;
    color: #4dc4ff; /* Accent Blue */
    text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
}
@keyframes pulseTitle {
    0% { transform: scale(1.0); opacity: 0.95; }
    100% { transform: scale(1.005); opacity: 1.0; }
}
/* 4. Customizing Sidebar */
[data-testid="stSidebar"] {
    background-color: #1f1f1f;
    box-shadow: 2px 0px 8px rgba(0,0,0,0.5);
}
/* 5. Metric Cards Styling */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stMetric"] {
    background-color: #2a2a2a;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    border-left: 5px solid #0d6efd;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    transition: all 0.3s;
}
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.6);
}
.stMetric .st-bd {
    font-size: 2em !important;
    font-weight: 700 !important;
    color: #0d6efd; /* Accent Blue */
}
/* --- NEW CARD STYLE for Key Findings --- */
.key-finding-card {
    background-color: #2a2a2a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    transition: transform 0.3s ease;
    border: 1px solid #333;
}
.key-finding-card:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.6);
}
.key-finding-card h4 {
    color: #4dc4ff !important;
    font-size: 1.2em;
    border-bottom: 2px solid #343a40;
    padding-bottom: 5px;
    margin-top: 0;
}
.key-value {
    font-size: 2.2em;
    font-weight: 700;
    color: #198754; /* Green for success */
}
/* 6. Step Buttons Styling */
.stButton button[key*="btn_"] {
    background-color: #495057;
    color: white;
    border-radius: 5px;
    border: none;
    transition: all 0.3s ease;
    padding: 10px 15px;
    font-weight: 600;
}
.stButton button[key*="btn_"]:hover {
    background-color: #6c757d;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
/* Highlight active button */
.stButton button[aria-pressed="true"] {
    background-color: #198754 !important;
    border-color: #198754 !important;
    box-shadow: 0 3px 6px rgba(25, 135, 84, 0.6) !important;
}
/* 7. pH Pulse Animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(25, 135, 84, 0); }
    100% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0); }
}
</style>
""", unsafe_allow_html=True)



def get_text(key: str, lang: str = "ru") -> str:
    """
    Return localized string for `key` in language `lang`.
    If missing, fallback to key.
    """
    try:
        # LANG айнымалысы lang_dict файлын импорттау арқылы қол жетімді
        return LANG.get(lang, LANG["ru"]).get(key, key)
    except Exception:
        return key

L = LANG
# Список кодов доступных языков (порядок — как в словаре LANG)
lang_codes = list(L.keys())
if not lang_codes:
    # На случай если LANG пуст — дефолтная защита
    lang_codes = ["ru"]

# Читаемые имена языков для селектора (если хочешь, можно расширить маппинг)
_lang_name_map = {
    "ru": "Русский",
    "en": "English",
    "kk": "Қазақша",
    # добавь другие удобочитаемые имена, если у тебя есть эти коды в LANG
}
lang_names = [ _lang_name_map.get(code, code) for code in lang_codes ]

# Выбираем дефолтный язык (предпочтение — 'ru', иначе первый в списке)
default_lang = "ru" if "ru" in lang_codes else lang_codes[0]

# Инициализация session_state для сохранения выбора между перезагрузками
if "lang_choice" not in st.session_state:
    st.session_state.lang_choice = default_lang

# Sidebar селектор языка (отображаем человекочитаемые имена)
# Используем index, чтобы выбрать текущий lang_choice корректно
try:
    current_index = lang_codes.index(st.session_state.lang_choice)
except ValueError:
    current_index = 0
selected_name = st.sidebar.selectbox("Язык / Language", lang_names, index=current_index)

# Преобразуем обратно имя -> код
selected_code = lang_codes[lang_names.index(selected_name)]
st.session_state.lang_choice = selected_code

# Локальная переменная для удобства в остальном коде
lang_choice = st.session_state.lang_choice

# Пример использования: заголовок в sidebar

# ---------------------------
# Функции Excel 
# ---------------------------
def safe_read_excel(path, sheet_name):
    if os.path.exists(path):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except ValueError:
            # Если листа нет — создаём новый
            st.warning(f"⚠️ Лист '{sheet_name}' не найден. Создаётся новый.")
            df = pd.DataFrame(
                columns=["BatchID", "mass_kg", "T_initial_C", "Salt_pct", "Moisture_pct", "StarterCFU", "Extract_pct"])
            with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        return df
    else:
        st.warning(f"⚠️ Файл {path} не найден. Создаётся новый.")
        df = pd.DataFrame(
            columns=["BatchID", "mass_kg", "T_initial_C", "Salt_pct", "Moisture_pct", "StarterCFU", "Extract_pct"])
        df.to_excel(path, index=False, sheet_name=sheet_name)
        return df


# --- Добавление новой строки ---
def append_row_excel(path, sheet_name, new_row):
    df = safe_read_excel(path, sheet_name)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


# ---------------------------
# DB Utility 
# ---------------------------
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
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


def fetch_measurements(limit=5000):
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


init_db()


# ---------------------------
# ML: 
# ---------------------------
class SimplePHModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        # try load
        if MODEL_FILE.exists() and SCALER_FILE.exists():
            try:
                self.model = joblib.load(MODEL_FILE)
                self.scaler = joblib.load(SCALER_FILE)
            except Exception:
                self.model, self.scaler = None, None

    def train(self, df, target='pH', feature_cols=None, test_size=0.2):
        df = df.copy()
        if target not in df.columns:
            raise ValueError("Target column not found")
        df = df.dropna(subset=[target])
        if feature_cols is None or len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target]
        # fallback single constant feature if none
        if len(feature_cols) == 0:
            df['_ones'] = 1.0
            feature_cols = ['_ones']
        X = df[feature_cols].astype(float).values
        y = df[target].astype(float).values
        # scale
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        # train
        self.model = LinearRegression()
        self.model.fit(Xs, y)
        # save
        joblib.dump(self.model, MODEL_FILE)
        joblib.dump(self.scaler, SCALER_FILE)
        # estimate metrics on training set (no holdout for simplicity)
        preds = self.model.predict(Xs)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        # r2:
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else 0.0
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"rmse": rmse, "r2": r2, "n": int(len(y)), "features": feature_cols}

    def predict(self, df, feature_cols=None):
        if self.model is None or self.scaler is None:
            # fallback
            n = len(df) if df is not None else 1
            return np.array([6.5] * n)
        if feature_cols is None or len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(feature_cols) == 0:
            X = np.ones((len(df), 1))
        else:
            X = df[feature_cols].astype(float).values
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        preds = np.clip(preds, 0.0, 14.0)  # realistic pH clamp
        return preds


ph_model = SimplePHModel()


# ---------------------------
# Utilities
# ---------------------------
# (safe_read_csv)
def safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    encodings = ['utf-8-sig', 'utf-8', 'windows-1251', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            try:
                return pd.read_csv(path, encoding=enc, engine='python')
            except Exception:
                continue
        except Exception:
            continue
    return pd.DataFrame()



def compute_score_from_ph(ph_value):
    if ph_value is None or (isinstance(ph_value, float) and np.isnan(ph_value)):
        return None
    return round(max(0.0, 10.0 - abs(ph_value - 6.5)), 2)


# (df_to_download_link )
def df_to_download_link(df, filename="export.csv", link_text="Скачать"):
    """
    Создает HTML-ссылку для скачивания DataFrame в виде CSV-файла.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    
    # Использование нового аргумента link_text
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'


# ---------------------------
# Load original data 
# ---------------------------
@st.cache_data
def load_all_data():
    data_sheets = {}
    try:
        if MEAT_DATA_XLSX.exists():
            xls = pd.ExcelFile(MEAT_DATA_XLSX)
            for sheet_name in xls.sheet_names:
                data_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
    except Exception as e:
        st.warning(f"Не удалось прочитать '{MEAT_DATA_XLSX.name}': {e}")

    df_ph = None
    if OPYTY_XLSX.exists():
        try:
            df_ph = pd.read_excel(OPYTY_XLSX, engine='openpyxl')
        except Exception as e:
            st.warning(f"Не удалось прочитать '{OPYTY_XLSX.name}': {e}")

    # Fake data if original files are missing
    if df_ph is None or df_ph.empty:
        df_ph = pd.DataFrame({
            'Time_h': [0, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120, 144],
            'pH_Control': [6.5, 6.4, 6.3, 6.1, 5.8, 5.5, 5.2, 5.1, 5.1, 5.15, 5.2, 5.2],
            'pH_Extract': [6.5, 6.45, 6.35, 6.2, 5.9, 5.6, 5.4, 5.3, 5.3, 5.35, 5.4, 5.4],
            'Salt_pct': [2.5] * 12,
            'Temp_C': [18] * 12
        })

    products_df = safe_read_csv(PRODUCTS_CSV)
    samples_df = safe_read_csv(SAMPLES_CSV)
    measurements_df = safe_read_csv(MEASUREMENTS_CSV)

    return data_sheets, df_ph, products_df, samples_df, measurements_df


all_meat_data, df_ph_raw, products, samples, measurements = load_all_data()

# Clean PH data for stability
if df_ph_raw is not None and not df_ph_raw.empty:
    df_ph_raw = df_ph_raw.rename(columns={
        df_ph_raw.columns[0]: 'Time_h',
        df_ph_raw.columns[1]: 'pH_Control',
        df_ph_raw.columns[2]: 'pH_Extract',
    }).iloc[:12]  # Возьмем только первые 12 строк для PH моделирования


# ---------------------------
# Original math functions 
# ---------------------------
def calculate_stability(pressure, viscosity):
    p, v = pressure, viscosity
    # Формула для прочности/стабильности (используется в странице Регрессионные модели качества)
    # Используем упрощенную формулу, близкую к квадратичной, как в исходном запросе.
    return 27.9 - 0.1 * p - 1.94 * v - 0.75 * p * v - 0.67 * p ** 2 - 2.5 * v ** 2


def get_ph_model(time_h, ph_obs):
    valid = ~np.isnan(time_h) & ~np.isnan(ph_obs)
    t, y = time_h[valid], ph_obs[valid]
    if len(t) < 3:
        return None, None, None, None
    try:
        coeffs = np.polyfit(t, y, 2)  # Квадратичная аппроксимация
        model_function = np.poly1d(coeffs)
        y_hat = model_function(t)
        r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
        rmse = np.sqrt(np.mean((y - y_hat) ** 2))
        return model_function, y_hat, rmse, r2
    except np.linalg.LinAlgError:
        return None, None, None, None


# ---------------------------
# UI: Main navigation 
# ---------------------------
sidebar_container = st.sidebar.container() 

# Используем контейнер для вывода статических элементов:

sidebar_container.markdown("<div class='fade-in'>", unsafe_allow_html=True)

# Заголовок и версия в боковом меню
sidebar_container.title(get_text("title", lang_choice))
sidebar_container.caption(get_text("version_note", lang_choice))

sidebar_container.markdown("</div>", unsafe_allow_html=True)
# Формирование пунктов меню
page_options = [
    get_text("menu_home", lang_choice),
    get_text("menu_production_process", lang_choice),
    get_text("menu_regression_models", lang_choice),
    get_text("menu_ph_modeling", lang_choice),
    get_text("menu_seabuckthorn_analysis", lang_choice),
    get_text("menu_data_exploration", lang_choice),
    get_text("menu_history_db", lang_choice),
    get_text("menu_ml_train_predict", lang_choice),
    get_text("menu_new_data_input", lang_choice),
]

# Радио-кнопки для выбора страницы
page = st.sidebar.radio(get_text("select_section", lang_choice), page_options, index=0)

# Session state initialization
if 'selected_product_id' not in st.session_state:
    st.session_state.selected_product_id = None
if 'selected_step' not in st.session_state:
    st.session_state.selected_step = None
if 'active_stage_clean' not in st.session_state:
    st.session_state['active_stage_clean'] = 'priemka'

# =================================================================
# PAGE: Главная / Home
# =================================================================
if page == get_text("menu_home", lang_choice):

    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='main-title-animation'>{get_text('full_title', lang_choice)}</h1>", unsafe_allow_html=True)
    st.subheader(get_text("home_desc", lang_choice))
    st.markdown("---")

    # 2. Ключевые функции платформы
    st.markdown(f"### {get_text('home_info', lang_choice)}")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            label="⚙️ " + get_text("menu_production_process", lang_choice),
            # stage_1: "1. Приемка" -> "1." + "control stages" (если stage_1 - это "1. Приемка")
            value= get_text("stage_control_suffix", lang_choice),
            delta=get_text("delta_production", lang_choice)
        )
        st.write(get_text("prod_subtitle", lang_choice))

    with col_b:
        st.metric(
            label="📈 " + get_text("menu_regression_models", lang_choice),
            # moisture_title: "Влажность (Moisture)" -> "pH и" + "Moisture"
            value="pH и " + get_text("moisture_title", lang_choice).split()[0],
            delta=get_text("delta_regression", lang_choice)
        )
        st.write(get_text("regression_subtitle", lang_choice))

    with col_c:
        st.metric(
            label="🔬 " + get_text("menu_seabuckthorn_analysis", lang_choice),
            value=get_text("seabuckthorn_value", lang_choice),
            delta=get_text("delta_seabuckthorn", lang_choice)
        )
        st.write(get_text("seabuck_desc", lang_choice))

    st.markdown("---")

    # 3. Основные научные достижения
    st.markdown(f"### {get_text('scientific_achievements', lang_choice)}") # ИСПРАВЛЕНО
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="key-finding-card">
            <h4>{get_text("wac_title", lang_choice)}</h4>
            <div class="small-muted">{get_text("wac_subtitle", lang_choice)}</div>
            <div class="key-value">75.6%</div>
            <div class="small-muted">{get_text("wac_note", lang_choice)}</div>
        </div>
        """, unsafe_allow_html=True) # ИСПРАВЛЕНО

    with col2:
        st.markdown(f"""
        <div class="key-finding-card">
            <h4>{get_text("shelf_life_title", lang_choice)}</h4>
            <div class="small-muted">{get_text("shelf_life_subtitle", lang_choice)}</div>
            <div class="key-value">60 {get_text("day_in_lang", lang_choice)}</div>
            <div class="small-muted">{get_text("shelf_life_note", lang_choice)}</div>
        </div>
        """, unsafe_allow_html=True) # ИСПРАВЛЕНО

    with col3:
        st.markdown(f"""
        <div class="key-finding-card">
            <h4>{get_text("optimal_conc_title", lang_choice)}</h4>
            <div class="small-muted">{get_text("optimal_conc_subtitle", lang_choice)}</div>
            <div class="key-value">3 – 5%</div>
            <div class="small-muted">{get_text("optimal_conc_note", lang_choice)}</div>
        </div>
        """, unsafe_allow_html=True) # ИСПРАВЛЕНО

    st.markdown("---")

    # 4. Окислительная стабильность
    st.subheader(get_text("oxidation_stability_title", lang_choice)) # ИСПРАВЛЕНО
    TBC_control = 2.80
    TBC_extract = 0.90
    reduction_pct = round((1 - (TBC_extract / TBC_control)) * 100)

    st.markdown(get_text("oxidation_goal", lang_choice)) # ИСПРАВЛЕНО
    st.progress(reduction_pct / 100,
                 text=f"**{reduction_pct}% {get_text('tba_reduction_text', lang_choice)}** "
                      f"{get_text('tba_caption_extract', lang_choice)}.") # ИСПРАВЛЕНО
    st.caption(f"{get_text('tba_caption', lang_choice)} {TBC_control} {get_text('mg_per_kg', lang_choice)} ({get_text('tba_caption_control', lang_choice)}) "
                f"{get_text('tba_caption_to', lang_choice)} {TBC_extract} {get_text('mg_per_kg', lang_choice)} (5% {get_text('tba_caption_extract', lang_choice)}).")
    st.success(get_text("oxidation_success", lang_choice)) # ИСПРАВЛЕНО

    st.markdown("---")

    # 5. Контроль pH
    st.subheader(get_text("ph_title", lang_choice))
    current_ph = 5.35
    ph_min = 5.1
    ph_max = 5.6

    st.markdown(f"""
        <div style='text-align:center; padding: 20px; background-color: #2a2a2a; border-radius: 10px; border: 2px solid #333;'>
            <h4 style='color:#f0f0f0;'>{get_text('predicted_ph', lang_choice)}:</h4>
            <h1 style='color:#198754; font-size: 3em; animation: pulse 1s infinite;'>{current_ph:.2f}</h1>
            <div class="small-muted">{get_text('ph_optimal', lang_choice)} <b>{ph_min:.1f} – {ph_max:.1f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    if ph_min <= current_ph <= ph_max:
        st.success(get_text("ph_optimal", lang_choice))
    else:
        st.warning(get_text("ph_insufficient", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)

# =================================================================
# =================================================================
# PAGE: Процесс производства Жая / Production Process
# =================================================================
elif page == get_text("menu_production_process", lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("jaya_process_title", lang_choice))
    st.markdown(get_text("jaya_process_subtitle", lang_choice))

    # Кнопки этапов (динамически локализованные)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(get_text("stage_priemka", lang_choice), key='btn_priemka'):
            st.session_state['active_stage_clean'] = 'priemka'
    with col2:
        if st.button(get_text("stage_posol", lang_choice), key='btn_posol'):
            st.session_state['active_stage_clean'] = 'posol'
    with col3:
        if st.button(get_text("stage_termo", lang_choice), key='btn_termo'):
            st.session_state['active_stage_clean'] = 'termokamera'
    with col4:
        if st.button(get_text("stage_upakovka", lang_choice), key='btn_upakovka'):
            st.session_state['active_stage_clean'] = 'upakovka'

    st.markdown("---")
    active_stage = st.session_state.get('active_stage_clean')

    # --------------------------
    # 1. Приемка
    # --------------------------
    if active_stage == 'priemka':
        st.header(get_text("stage_priemka_header", lang_choice))
        with st.expander(get_text("stage_priemka_expander", lang_choice), expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric(
            label=get_text("metric_mass", lang_choice), 
            value=f"1 {get_text('unit_kg', lang_choice)}",)
            col_p2.metric(label=get_text("metric_temp", lang_choice), value="0-3°С", help=get_text("help_temp", lang_choice))
            col_p3.metric(label=get_text("metric_ph", lang_choice), value="6.5-6.8", help=get_text("help_ph", lang_choice))
            st.markdown(get_text("tech_params_title", lang_choice))
            col_kpi_a, col_kpi_b, col_kpi_c = st.columns(3)
            col_kpi_a.metric(label=get_text("metric_yield", lang_choice), value="85%", delta=get_text("delta_gost", lang_choice))
            col_kpi_b.metric(label=get_text("metric_target_temp", lang_choice), value="74°С", delta=get_text("delta_inner", lang_choice))
            col_kpi_c.metric(
            label=get_text("metric_brine_loss", lang_choice), 
            value=f"100 {get_text('unit_g', lang_choice)}", # ИСПРАВЛЕНО
            delta_color="off",)
            st.markdown("---")
            st.info(get_text("digital_control_tip", lang_choice))

    # --------------------------
    # 2. Посол
    # --------------------------
    elif active_stage == 'posol':
        st.header(get_text("stage_posol_header", lang_choice))
        with st.expander(get_text("stage_posol_expander1", lang_choice), expanded=True):
            st.markdown(get_text("stage_posol_markdown1", lang_choice), unsafe_allow_html=True)

        with st.expander(get_text("stage_posol_expander2", lang_choice), expanded=False):
            st.markdown(get_text("stage_posol_markdown2", lang_choice))

    # --------------------------
    # 3. Термообработка
    # --------------------------
    elif active_stage == 'termokamera':
        st.header(get_text("stage_termo_header", lang_choice))
        st.info(get_text("stage_termo_info", lang_choice))

        termoparameters = [
            (get_text("termo_drying", lang_choice), "45°С", "20 мин", get_text("termo_drying_desc", lang_choice)),
            (get_text("termo_frying", lang_choice), "75-85°С", get_text("termo_frying_crit", lang_choice), get_text("termo_frying_desc", lang_choice)),
            (get_text("termo_steam", lang_choice), get_text("termo_steam_camtemp", lang_choice), get_text("termo_steam_inner", lang_choice),
             get_text("termo_steam_desc", lang_choice)),
            (get_text("termo_cool_dry", lang_choice), get_text("termo_cool_temp", lang_choice), "10 мин", get_text("termo_cool_desc", lang_choice)),
            (get_text("termo_smoke", lang_choice), "30-33°С", "1.5 ч", get_text("termo_smoke_desc", lang_choice))
        ]
        df_termo = pd.DataFrame(termoparameters, columns=[
            get_text("col_stage", lang_choice),
            get_text("col_temp", lang_choice),
            get_text("col_time", lang_choice),
            get_text("col_purpose", lang_choice)
        ])
        st.dataframe(df_termo.set_index(get_text("col_stage", lang_choice)), width=800)

        st.markdown("---")
        st.markdown(get_text("iot_monitoring_desc", lang_choice))

    # --------------------------
    # 4. Упаковка
    # --------------------------
    elif active_stage == 'upakovka':
        st.header(get_text("stage_upakovka_header", lang_choice))
        with st.expander(get_text("stage_upakovka_expander", lang_choice), expanded=True):
            st.markdown(get_text("stage_upakovka_markdown1", lang_choice))

        st.markdown("---")
        st.subheader(get_text("shelf_life_comparison", lang_choice))

        col_s1, col_s2 = st.columns(2)
        col_s1.metric(label=get_text("shelf_life_standard", lang_choice), value=get_text("shelf_life_std_value", lang_choice), delta_color="off")
        col_s2.metric(label=get_text("shelf_life_extract", lang_choice), value=get_text("shelf_life_ext_value", lang_choice), delta=get_text("shelf_life_delta_value", lang_choice))

        st.markdown(get_text("shelf_life_desc", lang_choice))
        st.info(get_text("storage_tip", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================================
# =================================================================
# PAGE: Регрессионные модели качества / Regression Quality Models
# =================================================================
elif page == get_text("menu_regression_models", lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("regression_title", lang_choice))
    st.markdown(get_text("regression_subtitle", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 1. Влажность конечного продукта
    # -------------------------------
    st.header(get_text("reg_w_title", lang_choice))
    st.latex(r"W = 65.0 + 0.12 \cdot T - 0.05 \cdot H + 0.5 \cdot E")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        T = st.slider(get_text("reg_w_T", lang_choice), min_value=20, max_value=35, value=25, step=1, key="w_T")
    with col_w2:
        H = st.slider(get_text("reg_w_H", lang_choice), min_value=1.0, max_value=10.0, value=5.0, step=0.5, key="w_H")
    with col_w3:
        E = st.slider(get_text("reg_w_E", lang_choice), min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="w_E_model1")

    W_predicted = 65.0 + 0.12 * T - 0.05 * H + 0.5 * E
    st.metric(label=get_text("reg_w_metric", lang_choice), value=f"{W_predicted:.2f}",
              delta=f"{get_text('reg_w_delta', lang_choice)} {W_predicted - 65.0:.2f} п.п.")
    st.info(get_text("reg_w_info", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 2. Активность воды
    # -------------------------------
    st.header(get_text("reg_aw_title", lang_choice))
    st.latex(r"A_w = 0.95 - 0.003 \cdot C - 0.005 \cdot T_s")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        C = st.slider(get_text("reg_aw_C", lang_choice), min_value=2.0, max_value=6.0, value=4.0, step=0.2, key="a_C")
    with col_a2:
        Ts = st.slider(get_text("reg_aw_Ts", lang_choice), min_value=1.0, max_value=7.0, value=3.0, step=0.5, key="a_Ts")

    Aw_predicted = 0.95 - 0.003 * C - 0.005 * Ts
    st.metric(label=get_text("reg_aw_metric", lang_choice), value=f"{Aw_predicted:.3f}",
              delta=(get_text("reg_aw_delta_high", lang_choice) if Aw_predicted > 0.90 else get_text("reg_aw_delta_ok", lang_choice)))
    st.success(get_text("reg_aw_info", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 3. Цветовая стабильность
    # -------------------------------
    st.header(get_text("reg_color_title", lang_choice))
    st.markdown(get_text("reg_color_desc", lang_choice))
    st.latex(r"\Delta E = 1.80 - 0.20 \cdot E + 0.05 \cdot H")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        E_color = st.slider(get_text("reg_color_E", lang_choice), min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="e_color")
    with col_c2:
        H_color = st.slider(get_text("reg_color_H", lang_choice), min_value=2.0, max_value=10.0, value=5.0, step=0.5, key="h_color")

    Delta_E_predicted = 1.80 - 0.20 * E_color + 0.05 * H_color
    st.metric(label=get_text("reg_color_metric", lang_choice), value=f"{Delta_E_predicted:.2f}",
              delta=get_text("reg_color_delta", lang_choice))

    if Delta_E_predicted < 1.5:
        st.success(get_text("reg_color_result_good", lang_choice))
    elif Delta_E_predicted < 2.5:
        st.warning(get_text("reg_color_result_warn", lang_choice))
    else:
        st.error(get_text("reg_color_result_bad", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 4. Окислительная стабильность (TBC)
    # -------------------------------
    st.header(get_text("reg_tbc_title", lang_choice))
    st.markdown(get_text("reg_tbc_desc", lang_choice))
    st.latex(r"\text{TBC}_{30\text{д}} = 2.80 - 0.35 \cdot E - 0.10 \cdot S")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        E_tbc = st.slider(get_text("reg_tbc_E", lang_choice), min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="e_tbc")
    with col_t2:
        S_tbc = st.slider(get_text("reg_tbc_S", lang_choice), min_value=2.0, max_value=5.0, value=3.5, step=0.1, key="s_tbc")

    TBC_predicted = 2.80 - 0.35 * E_tbc - 0.10 * S_tbc
    st.metric(label=get_text("reg_tbc_metric", lang_choice), value=f"{TBC_predicted:.2f}",
              delta=get_text("reg_tbc_delta", lang_choice))

    if TBC_predicted < 1.0:
        st.success(get_text("reg_tbc_result_good", lang_choice))
    elif TBC_predicted < 1.8:
        st.warning(get_text("reg_tbc_result_warn", lang_choice))
    else:
        st.error(get_text("reg_tbc_result_bad", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 5. Механическая прочность
    # -------------------------------
    st.header(get_text("reg_strength_title", lang_choice))
    st.info(get_text("reg_strength_info", lang_choice))

    with st.expander(get_text("reg_strength_expander", lang_choice), expanded=False):
        col_p_slider, col_v_slider = st.columns(2)
        with col_p_slider:
            P_input = st.slider(get_text("reg_strength_P", lang_choice), min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="p_pressure")
        with col_v_slider:
            V_input = st.slider(get_text("reg_strength_V", lang_choice), min_value=50, max_value=150, value=100, step=10, key="v_viscosity")

        Prochnost_score = calculate_stability(P_input, V_input / 100)
        st.metric(label=get_text("reg_strength_metric", lang_choice), value=f"{Prochnost_score:.2f}")

        if Prochnost_score >= 25:
            st.success(get_text("reg_strength_result_good", lang_choice))
        elif Prochnost_score >= 15:
            st.warning(get_text("reg_strength_result_warn", lang_choice))
        else:
            st.error(get_text("reg_strength_result_bad", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)



# =================================================================
# PAGE: Моделирование pH 
# =================================================================
# PAGE: Моделирование pH в процессе посола / pH Modeling
# =================================================================
elif page == get_text("menu_ph_modeling", lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("ph_title", lang_choice))
    st.markdown(f"### {get_text('ph_subtitle', lang_choice)}")

    with st.expander(get_text("ph_basis", lang_choice), expanded=True):
        st.write(get_text("ph_basis_text", lang_choice))

    st.markdown("---")
    st.subheader(get_text("ph_formula_title", lang_choice))
    st.latex(r"pH(t) = pH_0 - (pH_0 - pH_{\infty}) \cdot (1 - e^{-k \cdot t})")
    st.markdown(get_text("ph_formula_desc", lang_choice))
    st.warning(get_text("ph_formula_tip", lang_choice))
    st.markdown("---")

    # --- Функция модели ---
    def ph_model_func(t, pH0=6.6, pH_inf=4.6, k=0.03):
        t = np.array(t, dtype=float)
        ph = pH_inf + (pH0 - pH_inf) * np.exp(-k * t)
        ph = np.clip(ph, 0.0, 14.0)
        return ph

    # --- Интерактивный прогноз ---
    st.subheader(get_text("ph_forecast_title", lang_choice))
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        pH0 = st.number_input(get_text("ph_initial", lang_choice), value=6.6, format="%.2f")
    with col_b:
        pH_inf = st.number_input(get_text("ph_final", lang_choice), value=4.6, format="%.2f")
    with col_c:
        k = st.number_input(get_text("rate_constant", lang_choice), value=0.03, format="%.4f")

    t_input = st.slider(get_text("forecast_time", lang_choice),
                        min_value=1, max_value=240, value=48, step=1)
    pH_forecast = float(ph_model_func(t_input, pH0=pH0, pH_inf=pH_inf, k=k))

    st.metric(label=get_text("predicted_ph", lang_choice),
              value=f"{pH_forecast:.2f}",
              delta=f"{get_text('delta_target_ph', lang_choice)} {(pH_forecast - 5.6):.2f}",
              delta_color="inverse")

    # --- Классификация диапазона ---
    if pH_forecast < 4.8:
        st.error(get_text("ph_critical_low", lang_choice))
    elif 4.8 <= pH_forecast <= 5.6:
        st.success(get_text("ph_optimal", lang_choice))
    else:
        st.warning(get_text("ph_insufficient", lang_choice))

    st.markdown("---")
    st.subheader(get_text("ph_kinetics", lang_choice))

    times = np.linspace(0, 240, 300)
    pH_values = ph_model_func(times, pH0=pH0, pH_inf=pH_inf, k=k)

    fig = px.line(
        x=times,
        y=pH_values,
        labels={'x': get_text("time_hours", lang_choice), 'y': 'pH'},
        title=get_text("ph_plot_title", lang_choice)
    )
    fig.add_hrect(y0=4.8, y1=5.6, fillcolor="green", opacity=0.08, layer="below", line_width=0)
    fig.add_vline(x=t_input, line_dash="dash",
                  annotation_text=f"{t_input} {get_text('hours_short', lang_choice)}",
                  annotation_position="top right")
    fig.update_yaxes(range=[0, 8])
    st.plotly_chart(fig, use_container_width=True)


# =================================================================
# PAGE: Анализ с экстрактом облепихи 
# =================================================================
# PAGE: Анализ с экстрактом облепихи
# =================================================================
elif page == get_text("menu_seabuckthorn_analysis", lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("seabuck_title", lang_choice))
    st.write(get_text("seabuck_desc", lang_choice))
    st.markdown("---")

    # Таблица 1
    st.subheader(get_text("table1_title", lang_choice))
    table1_data = {
        get_text("indicator", lang_choice): [
            get_text("moisture", lang_choice),
            get_text("protein", lang_choice),
            get_text("fat", lang_choice),
            get_text("vus", lang_choice),
            get_text("tbch", lang_choice),
        ],
        get_text("control", lang_choice): [65.2, 21.2, 31.06, 60.2, 0.69],
        get_text("with_extract_5", lang_choice): [67.8, 25.44, 33.4, 67.4, 0.96],
    }
    df_table1 = pd.DataFrame(table1_data)
    st.dataframe(df_table1)

    # Таблица 2
    st.subheader(get_text("table2_title", lang_choice))
    table2_data = {
        get_text("indicator", lang_choice): [
            get_text("moisture", lang_choice),
            get_text("protein", lang_choice),
            get_text("fat", lang_choice),
            get_text("salt", lang_choice),
            get_text("ash", lang_choice),
        ],
        get_text("control", lang_choice): [68.96, 13.60, 11.03, 1.77, 2.96],
        get_text("with_extract_3", lang_choice): [70.08, 13.88, 8.51, 1.27, 2.22],
    }
    df_table2 = pd.DataFrame(table2_data)
    st.dataframe(df_table2)
    st.markdown("---")

    col1, col2 = st.columns(2)
    x_ticks = np.arange(0, 15.1, 2.5)

    with col1:
        st.subheader(get_text("fig1_title", lang_choice))
        x = np.array([0, 3, 5, 7, 9, 15])
        vlaga = np.array([65.2, 66.8, 68.9, 68.6, 67.8, 65.4])
        fig1 = px.line(x=x, y=vlaga, markers=True,
                       title=get_text("fig1_plot_title", lang_choice))
        fig1.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(get_text("fig3_title", lang_choice))
        VUS = np.array([60.2, 64.3, 67.4, 71.2, 73.5, 78.9])
        VSS = np.array([61.0, 65.5, 70.1, 73.8, 75.2, 77.4])
        ZhUS = np.array([60.0, 63.1, 66.8, 70.0, 72.5, 74.8])
        fig3 = px.line(x=x, y=VUS, markers=True,
                       title=get_text("fig3_plot_title", lang_choice))
        fig3.add_scatter(x=x, y=VUS, mode='lines+markers', name='ВУС, %')
        fig3.add_scatter(x=x, y=VSS, mode='lines+markers', name='ВСС, %')
        fig3.add_scatter(x=x, y=ZhUS, mode='lines+markers', name='ЖУС, %')
        fig3.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader(get_text("fig5_title", lang_choice))
        days2 = np.array([5, 10, 15])
        tbch_c2 = np.array([0.203, 0.284, 0.312])
        tbch_e2 = np.array([0.254, 0.366, 0.428])
        perox_c2 = np.array([13.27, 14.30, 15.21])
        perox_e2 = np.array([9.90, 10.80, 11.60])
        fig5 = px.line(title=get_text("fig5_plot_title", lang_choice))
        fig5.add_scatter(x=days2, y=tbch_c2, mode='lines+markers', name='ТБЧ контроль')
        fig5.add_scatter(x=days2, y=tbch_e2, mode='lines+markers', name='ТБЧ 3%')
        fig5.add_scatter(x=days2, y=perox_c2, mode='lines+markers', name='Перокс контроль')
        fig5.add_scatter(x=days2, y=perox_e2, mode='lines+markers', name='Перокс 3%')
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader(get_text("fig2_title", lang_choice))
        belok = np.array([21.2, 23.4, 25.4, 27.5, 29.8, 34.9])
        zhir = np.array([31.06, 32.4, 33.4, 37.1, 41.2, 45.0])
        fig2 = px.line(title=get_text("fig2_plot_title", lang_choice))
        fig2.add_scatter(x=x, y=belok, mode='lines+markers', name='Белок, %')
        fig2.add_scatter(x=x, y=zhir, mode='lines+markers', name='Жир, %')
        fig2.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(get_text("fig4_title", lang_choice))
        days = np.array([5, 10, 15])
        tbch_c = np.array([0.197, 0.376, 0.416])
        tbch_e = np.array([0.194, 0.361, 0.419])
        perox_c = np.array([17.96, 19.12, 20.25])
        perox_e = np.array([13.01, 14.40, 15.13])
        fig4 = px.line(title=get_text("fig4_plot_title", lang_choice))
        fig4.add_scatter(x=days, y=tbch_c, mode='lines+markers', name='ТБЧ контроль')
        fig4.add_scatter(x=days, y=tbch_e, mode='lines+markers', name='ТБЧ 3%')
        fig4.add_scatter(x=days, y=perox_c, mode='lines+markers', name='Перокс контроль')
        fig4.add_scatter(x=days, y=perox_e, mode='lines+markers', name='Перокс 3%')
        st.plotly_chart(fig4, use_container_width=True)


# =================================================================
# PAGE: Исследование данных 
# =================================================================
# PAGE: 🗂️ Исследование данных / Data Exploration
# =================================================================
elif page == get_text("menu_data_exploration", lang_choice):  # заменили L["menu"][5]
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("explore_title", lang_choice))
    st.write(get_text("explore_desc", lang_choice))

    # Используем глобальную переменную df_ph_raw
    df_to_use_for_ph = df_ph_raw

    if all_meat_data:
        available_tables = list(all_meat_data.keys())

        # Если есть данные по pH, добавляем как отдельный вариант
        if df_to_use_for_ph is not None:
            available_tables.append('opyty.xlsx')

        choice = st.selectbox(get_text("select_data", lang_choice), available_tables)
        st.markdown(f"**{get_text('viewing_data', lang_choice)} `{choice}`**")

        if choice == 'opyty.xlsx':
            if df_to_use_for_ph is not None:
                df_to_show = df_to_use_for_ph.copy()
            else:
                df_to_show = pd.DataFrame()
        else:
            df_to_show = all_meat_data.get(choice, pd.DataFrame()).copy()

        if 'Accuracy' in df_to_show.columns:
            df_to_show['Accuracy'] = pd.to_numeric(df_to_show['Accuracy'], errors='coerce')

        if not df_to_show.empty:
            st.dataframe(df_to_show)
        else:
            st.warning(f"{get_text('viewing_data', lang_choice)} '{choice}' — {get_text('data_empty_warning', lang_choice) if 'data_empty_warning' in L[lang_choice] else 'Данные не были загружены или пусты.'}")
    else:
        st.warning(get_text("data_load_error", lang_choice) if "data_load_error" in L[lang_choice] else "Не удалось загрузить данные для просмотра.")


# =================================================================
# PAGE: История / DB (Тарих / DB) 
# =================================================================
# PAGE: История / DB (Тарих / DB)
# =================================================================
elif page == get_text("menu_history_db", lang_choice):  # заменено с L["menu"][6]
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    # Заголовок и описание
    st.title(get_text("db_title", lang_choice))
    st.markdown(f"### {get_text('db_desc', lang_choice)}")

    df_db = fetch_measurements()

    if df_db.empty:
        st.info(get_text("history_empty", lang_choice))
    else:
        st.subheader(f"{get_text('total_records', lang_choice)} {len(df_db)}")
        st.dataframe(df_db, use_container_width=True)

        # 📈 График pH по времени
        fig_db = px.line(
            df_db.sort_values('created_at'),
            x='created_at',
            y='ph',
            color='sample_name',
            title=get_text("ph_over_time", lang_choice),
            template='plotly_dark'
        )
        st.plotly_chart(fig_db, use_container_width=True)

        # --- Экспорт и удаление
        st.markdown("---")
        col_dl, col_del = st.columns(2)

        with col_dl:
            st.markdown(df_to_download_link(df_db, "measurements_export.csv", get_text("export_all", lang_choice)),
                        unsafe_allow_html=True)

        with col_del:
            if st.button(f"❌ {get_text('clear_all', lang_choice)}", key="db_reset"):
                if st.session_state.get('confirm_reset', False):
                    delete_all_measurements()
                    st.session_state['confirm_reset'] = False
                    st.success(get_text("db_cleared", lang_choice))
                    st.experimental_rerun()
                else:
                    st.session_state['confirm_reset'] = True
                    st.warning(get_text("confirm_clear", lang_choice))
                    st.button(get_text("confirm_clear", lang_choice), key="confirm_btn")

    st.markdown("</div>", unsafe_allow_html=True)


# =================================================================
# PAGE: ML: Train / Predict 
# =================================================================
# PAGE: ML: Train / Predict
# =================================================================
elif page == get_text("menu_ml_train_predict", lang_choice):  # заменено с L["menu"][7]
    st.title(get_text("ml_title", lang_choice))
    st.markdown(get_text("ml_desc", lang_choice))

    tab1, tab2 = st.tabs([get_text("train_tab", lang_choice), get_text("predict_tab", lang_choice)])

    # --- TAB 1: TRAIN --------------------------------------------------------
    with tab1:
        st.subheader(get_text("train_subtitle", lang_choice))
        up = st.file_uploader(get_text("upload_train", lang_choice),
                              type=["csv", "xlsx", "xls"], key="train_up")

        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    df_train = pd.read_csv(up)
                else:
                    df_train = pd.read_excel(up)
            except Exception as e:
                st.error(f"{get_text('train_error', lang_choice)} {e}")
                df_train = pd.DataFrame()

            if df_train.empty:
                st.info(get_text("no_data", lang_choice))
            else:
                st.write(get_text("preview", lang_choice))
                st.dataframe(df_train.head(10))

                cols = df_train.columns.tolist()
                if 'pH' in cols:
                    target = 'pH'
                else:
                    target = st.selectbox(get_text("target_column", lang_choice), options=cols)

                features = st.multiselect(get_text("features", lang_choice), options=cols)

                if st.button(get_text("train_button", lang_choice)):
                    try:
                        metrics = ph_model.train(
                            df_train,
                            target=target,
                            feature_cols=features if features else None
                        )
                        st.success(get_text("train_success", lang_choice))
                        st.json(metrics)
                    except Exception as e:
                        st.error(f"{get_text('train_error', lang_choice)} {e}")

    # --- TAB 2: PREDICT -----------------------------------------------------
    with tab2:
        st.subheader(get_text("predict_subtitle", lang_choice))
        up2 = st.file_uploader(get_text("upload_predict", lang_choice),
                               type=["csv"], key="pred_up")

        if up2:
            try:
                df_pred = pd.read_csv(up2)
            except Exception:
                try:
                    df_pred = pd.read_excel(up2)
                except Exception as e:
                    st.error(f"Ошибка чтения: {e}")
                    df_pred = pd.DataFrame()

            if df_pred.empty:
                st.info(get_text("no_data", lang_choice))
            else:
                st.dataframe(df_pred.head(10))
                num_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
                st.write(f"{get_text('auto_features', lang_choice)} {num_cols}")

                if st.button(get_text("predict_button", lang_choice)):
                    preds = ph_model.predict(df_pred, feature_cols=num_cols)
                    df_pred['predicted_pH'] = np.round(preds, 3)
                    df_pred['score'] = df_pred['predicted_pH'].apply(compute_score_from_ph)

                    st.subheader(get_text("predict_results", lang_choice))
                    st.dataframe(df_pred.head(50))
                    st.markdown(df_to_download_link(df_pred, filename="predictions.csv"), unsafe_allow_html=True)

                    # --- Сохранение в базу, если есть sample_name
                    if 'sample_name' in df_pred.columns:
                        if st.button(get_text("save_to_db", lang_choice)):
                            saved = 0
                            for _, r in df_pred.iterrows():
                                insert_measurement(
                                    str(r.get('sample_name', 'sample')),
                                    float(r.get('predicted_pH', np.nan)),
                                    compute_score_from_ph(float(r.get('predicted_pH', np.nan))),
                                    notes="predicted"
                                )
                                saved += 1
                            st.success(f"{get_text('saved_records', lang_choice)} {saved}")


# =====================================================================
# PAGE: Ввод новых данных 
# =====================================================================
# PAGE: Ввод новых данных / Input new data
# =====================================================================
elif page == get_text("menu_new_data_input", lang_choice):
    st.title(get_text("input_title", lang_choice))
    st.markdown(f"### {get_text('input_subtitle', lang_choice)} ({MEAT_XLSX}, {get_text('sheet', lang_choice)} {SHEET_NAME})")

    # Загружаем текущие данные
    df_meat = safe_read_excel(MEAT_XLSX, SHEET_NAME)

    # Проверка наличия колонки BatchID
    if "BatchID" not in df_meat.columns:
        st.error(get_text("batchid_missing", lang_choice))
        st.stop()

    # Определяем следующий BatchID
    if len(df_meat) > 0 and df_meat["BatchID"].astype(str).str.match(r"^M\d+$").any():
        last_id_str = df_meat[df_meat["BatchID"].astype(str).str.match(r"^M\d+$")]["BatchID"].dropna().astype(str).iloc[-1]
        try:
            last_num = int(last_id_str[1:])
            next_id = f"M{last_num + 1}"
        except:
            next_id = "M1"
    else:
        next_id = "M1"

    # -------------------------------
    # Форма добавления данных
    # -------------------------------
    with st.form(key='batch_entry_form'):
        st.subheader(get_text("batch_params", lang_choice))

        st.text_input(get_text("batch_id", lang_choice), value=next_id, disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            mass_kg = st.number_input(get_text("mass", lang_choice), min_value=1.0, value=100.0, step=1.0)
            T_initial_C = st.number_input(get_text("initial_temp", lang_choice), min_value=-10.0, value=4.0, step=0.1)
            Salt_pct = st.number_input(get_text("salt_content", lang_choice), min_value=0.0, value=5.0, step=0.1)
        with col2:
            Moisture_pct = st.number_input(get_text("moisture", lang_choice), min_value=0.0, value=75.0, step=0.1)
            StarterCFU = st.number_input(get_text("starter_culture", lang_choice), min_value=0, value=1000000, step=10000)
            Extract_pct = st.number_input(get_text("extract_content", lang_choice), min_value=0.0, value=3.0, step=0.1)

        submitted = st.form_submit_button(get_text("save_data", lang_choice))

        if submitted:
            new_row = {
                "BatchID": next_id,
                "mass_kg": mass_kg,
                "T_initial_C": T_initial_C,
                "Salt_pct": Salt_pct,
                "Moisture_pct": Moisture_pct,
                "StarterCFU": StarterCFU,
                "Extract_pct": Extract_pct
            }
            try:
                append_row_excel(MEAT_XLSX, SHEET_NAME, new_row)
                st.success(f"{get_text('batch_added', lang_choice)}: '{next_id}'")
                st.cache_data.clear()
                load_all_data.clear()
            except Exception as e:
                st.error(f"{get_text('save_error', lang_choice)} {e}")

    st.markdown("---")
    st.subheader(get_text("current_data", lang_choice))
    st.dataframe(safe_read_excel(MEAT_XLSX, SHEET_NAME), use_container_width=True)


# ---------------------------
# Footer
# ---------------------------

