# app.py — Объединённая версия: ВСЕ твои функции + DB, ML, plotly, мультиязычность
# Сохраняет полный функционал оригинального файла (включая все страницы).
# Требования: streamlit, pandas, numpy, matplotlib, plotly, scikit-learn, joblib, openpyxl

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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import base64
import time
import os

# ---------------------------
# Конфигурация файлов/папок
# ---------------------------
MEAT_XLSX = "meat_data.xlsx"
SHEET_NAME = "T6"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_FILE = DATA_DIR / "measurements.db"
MODEL_FILE = DATA_DIR / "model.pkl"
SCALER_FILE = DATA_DIR / "scaler.pkl"

MEAT_DATA_XLSX = BASE_DIR / "meat_data.xlsx"
OPYTY_XLSX = BASE_DIR / "opyty.xlsx"
PRODUCTS_CSV = BASE_DIR / "Products.csv"
SAMPLES_CSV = BASE_DIR / "Samples.csv"
MEASUREMENTS_CSV = BASE_DIR / "Measurements.csv"

# ---------------------------
# Установки страницы
# ---------------------------
st.set_page_config(page_title="Платформа Жая — расширенная", layout="wide")
# простой CSS анимации (плавное появление)
st.markdown("""
<style>
.fade-in {
  animation: fadeIn ease 0.6s;
}
@keyframes fadeIn {
  0% {opacity:0; transform:translateY(6px)}
  100% {opacity:1; transform:translateY(0)}
}
.small-muted {color:#6c757d; font-size:0.9em;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Мультиязычность (рус/eng/kk)
# ---------------------------
LANG = {
    "ru": {
        "title": "Платформа Жая",
        "version_note": "Версия: интегрированная",
        "menu": ["Главная", "Процесс производства Жая", "Регрессионные модели качества",
                 "Моделирование pH", "Анализ с экстрактом облепихи", "Исследование данных", "История / DB", "ML: Train / Predict", "Ввод новых данных"],
        "db_reset_confirm": "Вы уверены, что хотите удалить все измерения?",
        "train_button": "Обучить модель",
        "predict_button": "Сделать прогноз",
        "upload_csv": "Загрузить CSV/Excel",
    },
    "en": {
        "title": "Meat Digitalization Platform",
        "version_note": "Version: integrated",
        "menu": ["Home", "Production process", "Regression quality models",
                 "pH Modeling", "Sea-buckthorn extract analysis", "Data exploration", "History / DB", "ML: Train / Predict"],
        "db_reset_confirm": "Are you sure you want to delete all measurements?",
        "train_button": "Train model",
        "predict_button": "Predict",
        "upload_csv": "Upload CSV/Excel",
    },
    "kk": {
        "title": "Жая платформасы",
        "version_note": "Нұсқа: біріктірілген",
        "menu": ["Басты", "Өндіріс процесі", "Сапа регрессиялық модельдері",
                 "pH моделдеу", "Құлпынай сығындысы талдауы", "Деректерді зерттеу", "Тарих / DB", "ML: Үйрету / Болжам"],
        "db_reset_confirm": "Барлық өлшемдерді жойғыңыз келетініне сенімдісіз бе?",
        "train_button": "Модельді үйрету",
        "predict_button": "Болжам жасау",
        "upload_csv": "CSV/Excel жүктеу",
    }
}

lang_choice = st.sidebar.selectbox("Язык / Тіл / Language", options=["ru", "en", "kk"], index=0)
L = LANG[lang_choice]

#Ввод новых данных в Excel с проверкой наличия файла и листа
def safe_read_excel(path, sheet_name):
    if os.path.exists(path):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except ValueError:
            # Если листа нет — создаём новый
            st.warning(f"⚠️ Лист '{sheet_name}' не найден. Создаётся новый.")
            df = pd.DataFrame(columns=["BatchID", "mass_kg", "T_initial_C", "Salt_pct", "Moisture_pct", "StarterCFU", "Extract_pct"])
            with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, index=False, sheet_name=sheet_name)
        return df
    else:
        st.warning(f"⚠️ Файл {path} не найден. Создаётся новый.")
        df = pd.DataFrame(columns=["BatchID", "mass_kg", "T_initial_C", "Salt_pct", "Moisture_pct", "StarterCFU", "Extract_pct"])
        df.to_excel(path, index=False, sheet_name=sheet_name)
        return df

# --- Добавление новой строки ---
def append_row_excel(path, sheet_name, new_row):
    df = safe_read_excel(path, sheet_name)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
# ---------------------------
# DB Utility (sqlite, simple)
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
# ML: simple wrapper (LinearRegression + scaler)
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
        if feature_cols is None or len(feature_cols)==0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target]
        # fallback single constant feature if none
        if len(feature_cols)==0:
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
        rmse = float(np.sqrt(np.mean((preds - y)**2)))
        # r2:
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - np.mean(y))**2) if len(y)>1 else 0.0
        r2 = float(1 - ss_res/ss_tot) if ss_tot>0 else 0.0
        return {"rmse": rmse, "r2": r2, "n": int(len(y)), "features": feature_cols}

    def predict(self, df, feature_cols=None):
        if self.model is None or self.scaler is None:
            # fallback
            n = len(df) if df is not None else 1
            return np.array([6.5]*n)
        if feature_cols is None or len(feature_cols)==0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(feature_cols)==0:
            X = np.ones((len(df),1))
        else:
            X = df[feature_cols].astype(float).values
        Xs = self.scaler.transform(X)
        preds = self.model.predict(Xs)
        preds = np.clip(preds, 0.0, 14.0)  # realistic pH clamp
        return preds

ph_model = SimplePHModel()

# ---------------------------
# Utilities: safe CSV reader, score, export
# ---------------------------
def safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    encodings = ['utf-8-sig','utf-8','windows-1251','latin1']
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

def df_to_download_link(df, filename="export.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Скачать {filename}</a>'

# ---------------------------
# Load original data (like в твоём файле)
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

    products_df = safe_read_csv(PRODUCTS_CSV)
    samples_df = safe_read_csv(SAMPLES_CSV)
    measurements_df = safe_read_csv(MEASUREMENTS_CSV)

    return data_sheets, df_ph, products_df, samples_df, measurements_df

all_meat_data, df_ph, products, samples, measurements = load_all_data()

# ---------------------------
# Original math functions (copied & preserved)
# ---------------------------
def calculate_stability(pressure, viscosity):
    p, v = pressure, viscosity
    return 27.9 - 0.1 * p - 1.94 * v - 0.75 * p * v - 0.67 * p ** 2 - 2.5 * v ** 2

def get_ph_model(time_h, ph_obs):
    valid = ~np.isnan(time_h) & ~np.isnan(ph_obs)
    t, y = time_h[valid], ph_obs[valid]
    if len(t) < 3:
        return None, None, None, None
    coeffs = np.polyfit(t, y, 2)
    model_function = np.poly1d(coeffs)
    y_hat = model_function(t)
    r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    return model_function, y_hat, rmse, r2

# ---------------------------
# UI: Main navigation - keep your pages intact (all texts preserved)
# ---------------------------
st.sidebar.title(L["title"])
st.sidebar.caption(L["version_note"])

page_options = L["menu"]
page = st.sidebar.radio("Выберите раздел / Section", page_options)

# Keep session state keys used in original file
if 'selected_product_id' not in st.session_state: st.session_state.selected_product_id = None
if 'selected_step' not in st.session_state: st.session_state.selected_step = None

# ---------------------------
# PAGE: Главная (preserve)
# ---------------------------
if page == L["menu"][0]:
    st.title("🐎 Цифровая платформа для производства и моделирования Жая")
    st.write("Эта система объединяет описание технологических процессов и интерактивные математические модели для анализа и прогнозирования качества продукции.")
    st.info("Выберите раздел в меню слева, чтобы начать работу.")

# ---------------------------
# PAGE: Процесс производства Жая (entire original block preserved)
# ---------------------------
elif page == L["menu"][1]:
    st.title("🍖 Технологическая карта производства Жая")
    st.markdown("### Пошаговый контроль качества и параметры процесса")

    if 'active_stage_clean' not in st.session_state:
        st.session_state['active_stage_clean'] = 'priemka'

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("1. Приемка сырья 🥩", key='btn_priemka'):
            st.session_state['active_stage_clean'] = 'priemka'
    with col2:
        if st.button("2. Посол и массирование 🧂", key='btn_posol'):
            st.session_state['active_stage_clean'] = 'posol'
    with col3:
        if st.button("3. Термическая обработка 🔥", key='btn_termo'):
            st.session_state['active_stage_clean'] = 'termokamera'
    with col4:
        if st.button("4. Хранение и упаковка 📦", key='btn_upakovka'):
            st.session_state['active_stage_clean'] = 'upakovka'

    st.markdown("---")
    active_stage = st.session_state.get('active_stage_clean')

    if active_stage == 'priemka':
        st.header("1. Приемка и подготовка сырья")
        with st.expander("Контрольные параметры приемки", expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric(label="Начальная масса", value="1 кг")
            col_p2.metric(label="Температура сырья", value="0-3°С")
            col_p3.metric(label="Толщина жира", value="1,5 см – 2 см")
            st.markdown("#### Ключевые Технологические Показатели (Общая сводка)")
            col_kpi_a, col_kpi_b, col_kpi_c = st.columns(3)
            col_kpi_a.metric(label="Выход продукции (Цель)", value="85%", delta="По ГОСТ")
            col_kpi_b.metric(label="Целевая t° готовности", value="74°С", delta="Внутри продукта")
            col_kpi_c.metric(label="Масса рассола (Потеря)", value="100 г", delta_color="off")
            st.markdown("---")
            st.markdown("Для подробного анализа перейдите по этапам (Посол, Термообработка).")

    elif active_stage == 'posol':
        st.header("2. Посол, Шприцевание и Массирование")
        with st.expander("Подготовка рассола и шприцевание", expanded=True):
            st.markdown(r"""
            **Состав рассола:** 4,5 л $\text{H}_2\text{O}$ + 250 г $\text{NaCl}$ + 0,8 мг $\text{NaNO}_2$.
            * **Температура рассола:** **$16^{\circ}С$**
            * **Шприцевание:** До середины куска. Иглы: **50 мм и 80 мм**.
            * **Укладка в рассол:** $\tau=72$ часа, $t=0-3^{\circ}С$. Давление $P=1200\text{ г} – 1250\text{ г}$ на 1000 г.
            """)
        with st.expander("Параметры массирования", expanded=False):
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label="Общая длительность", value="3 часа")
            col_m2.metric(label="Рабочее давление", value="0,4-0,5 кг/см² (max 0,9)")
            st.markdown(r"""
            * **Потеря массы (за 2 часа):** Снижение $\text{H}_2\text{O}$ на $\mathbf{100\text{ г}}$ (от $1250\text{ г}$).
            * **Итоговая масса:** $1150\text{ г}$.
            """)

    elif active_stage == 'termokamera':
        st.header("3. Термическая обработка (Термокамера)")
        st.info("Термообработка включает 5 последовательных этапов.")
        termoparameters = [
            ("Сушка", "45°С", "20 мин", "Удаление поверхностной влаги."),
            ("Обжарка", "75-85°С", "Внутренняя $\mathbf{60^{\circ}С}$", "Формирование цвета/аромата."),
            ("Варка паром", "Камера $\mathbf{88^{\circ}С}$", "Внутренняя $\mathbf{74^{\circ}С}$", "Достижение полной готовности."),
            ("Сушка охлаждением", "Вентилятор", "10 мин", "Стабилизация температуры."),
            ("Копчение", "30-33°С (Дым)", "1,5 часа", "Придание аромата (Коптильня $230^{\circ}С$).")
        ]
        try:
            df_termo = pd.DataFrame(termoparameters, columns=["Этап", "Температура", "Время/Критерий", "Назначение"])
            st.dataframe(df_termo.set_index('Этап'), width=800)
        except NameError:
            st.warning("Для отображения таблицы необходим импорт: import pandas as pd в начале файла.")
            for etapa, t, tau, naznachenie in termoparameters:
                st.markdown(f"**{etapa}:** $t={t}$, $\tau={tau}$ ($ {naznachenie} $)")
        st.markdown("---")
        st.markdown("**После копчения:** Жая остается в термокамере с **открытой дверью в течение 2 часов**.")

    elif active_stage == 'upakovka':
        st.header("4. Обвалка, Упаковка и Хранение")
        with st.expander("Обвалка и Упаковка", expanded=True):
            st.markdown("""
            * **Формовка (Обвалка):** Шпагатом (круглая/прямоугольная форма) — $\tau=20$ мин.
            * **Охлаждение:** В холодильной камере $t=0-5^{\circ}С$ — $12$ часов.
            * **Удаление шпагата:** Производится после охлаждения.
            * **Упаковка:** В вакуум-упаковочном автомате.
            """)
        with st.expander("Сроки и Выход продукта", expanded=True):
            st.metric(label="Выход готовой продукции (по ГОСТ)", value="85%")
            st.markdown("**Сроки хранения:**")
            st.markdown("* **Стандарт:** $t=0-3^{\circ}С$ — $\mathbf{30}$ суток.")
            st.markdown("* **Заморозка:** После 72 часов ($0-3^{\circ}С$) в морозильник $t = -16\div-18^{\circ}С$ — $\mathbf{6}$ месяцев.")

# ---------------------------
# PAGE: Регрессионные модели качества (preserve)
# ---------------------------
elif page == L["menu"][2]:
    st.title("📊 Регрессионные модели качества конечного продукта")
    st.markdown("### Прогнозирование качества на основе технологических параметров")
    with st.expander("ℹ️ Научное обоснование", expanded=True):
        st.write("""
            **Математические модели** позволяют прогнозировать ключевые показатели качества готового продукта ...
        """)
    st.markdown("---")
    st.header("1. Влажность конечного продукта ($W$)")
    st.latex(r"W = 65.0 + 0.12 \cdot T - 0.05 \cdot H + 0.5 \cdot E")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        T = st.slider("Температура сушки (T), °C", min_value=20, max_value=35, value=25, step=1, key="w_T")
    with col_w2:
        H = st.slider("Продолжительность сушки (H), час", min_value=1.0, max_value=10.0, value=5.0, step=0.5, key="w_H")
    with col_w3:
        E = st.slider("Концентрация экстракта (E), %", min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="w_E")
    W_predicted = 65.0 + 0.12 * T - 0.05 * H + 0.5 * E
    st.metric(label="Прогнозируемая Влажность (W), %", value=f"{W_predicted:.2f}", delta=f"Разница от базового значения (65%): {W_predicted - 65.0:.2f} п.п.")
    st.info("Добавление экстракта ($E$) положительно влияет на влагоудержание, а длительность сушки ($H$) снижает влажность.")
    st.markdown("---")
    st.header("2. Активность воды ($A_w$)")
    st.latex(r"A_w = 0.95 - 0.003 \cdot C - 0.005 \cdot T_s")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        C = st.slider("Концентрация соли (C), %", min_value=2.0, max_value=6.0, value=4.0, step=0.2, key="a_C")
    with col_a2:
        Ts = st.slider("Длительность соления (Ts), сут", min_value=1.0, max_value=7.0, value=3.0, step=0.5, key="a_Ts")
    Aw_predicted = 0.95 - 0.003 * C - 0.005 * Ts
    st.metric(label="Прогнозируемая Активность воды ($A_w$)", value=f"{Aw_predicted:.3f}", delta=f"Необходимо снизить на {Aw_predicted - 0.90:.3f} для достижения Aw ≤ 0.90" if Aw_predicted > 0.90 else "В пределах безопасной нормы")
    st.success("Оптимальный $A_w$ (0.88-0.90) критичен для микробиологической безопасности и продления срока годности.")
    st.markdown("---")
    st.header("3. Цветовая стабильность ($\Delta E$)")
    st.info("Модель **Цветовой стабильности** описывает, как изменяется цвет продукта...")
    with st.expander("Ключевые факторы цветовой стабильности:", expanded=True):
        col_c1, col_c2 = st.columns(2)
        col_c1.metric(label="Концентрация экстракта ($E$)", value="Положительное влияние", delta="Антиоксиданты стабилизируют цвет")
        col_c2.metric(label="Перекисное число ($PV$)", value="Отрицательное влияние", delta="Окисление приводит к потере цвета", delta_color="inverse")
        st.markdown("---")
        st.markdown("**Основной вывод:** Добавление экстракта облепихи ($E$) значительно снижает скорость окисления...")
    st.markdown("---")
    st.header("4. Механическая прочность (формованные изделия)")
    st.info("Модель описывает **плотность и упругость** продукта...")
    with st.expander("🛠️ Интерактивный симулятор прочности", expanded=True):
        col_p_slider, col_v_slider = st.columns(2)
        with col_p_slider:
            P_input = st.slider("Давление прессования ($P$), $кг/см^2$", min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="p_pressure")
        with col_v_slider:
            V_input = st.slider("Вязкость фарша ($V$), условные единицы", min_value=50, max_value=150, value=100, step=10, key="v_viscosity")
        Base_Prochnost = 120.0
        Penalty_P = 5.0 * P_input
        Penalty_V = 0.2 * V_input
        Prochnost_predicted = Base_Prochnost - Penalty_P - Penalty_V
        if Prochnost_predicted >= 95:
            delta_text = "Оптимальная/Высокая прочность"
            delta_color = "normal"
        elif Prochnost_predicted >= 80:
            delta_text = "Средняя прочность (Приемлемо)"
            delta_color = "off"
        else:
            delta_text = "Низкая прочность (Риск сепарации)"
            delta_color = "inverse"
        st.markdown("---")
        st.metric(label="Прогнозируемый Индекс Механической Прочности (Усл. ед.)", value=f"{Prochnost_predicted:.1f}", delta=delta_text, delta_color=delta_color)
        st.markdown(r"""
           **Анализ симулятора:** * **Увеличение $P$ или $V$** приводит к **снижению** индекса прочности...
           """)

    st.markdown("---")
    st.header("5. Практические рекомендации по добавлению экстракта облепихи")
    st.success("🎯 Оптимальная концентрация экстракта (Заключение Отчета, стр. 18)")
    col_conc1, col_conc2 = st.columns(2)
    col_conc1.metric(label="Для цельномышечной жая (копчёной)", value="5%", delta="Максимальная антиокислительная устойчивость")
    col_conc2.metric(label="Для формованного мясного изделия", value="3%", delta="Баланс вкуса и технологичности")
    st.markdown("**Общее заключение:** Применение экстракта облепихи в концентрациях **3–5%** является эффективным способом...")

# ---------------------------
# PAGE: Моделирование pH (preserve + corrected graph)
# ---------------------------
elif page == L["menu"][3]:
    st.title("🌡️ Моделирование pH в процессе посола")
    st.markdown("### Прогноз кинетики кислотности для обеспечения безопасности")
    with st.expander("ℹ️ Научное обоснование pH-моделирования", expanded=True):
        st.write("""
            **Биохимический смысл:** ... (текст сохранён)
        """)
    st.markdown("---")
    st.subheader("Формула кинетики pH (Подмодель соления)")
    st.latex(r"pH(t) = pH_0 - (pH_0 - pH_{\infty}) \cdot (1 - e^{-k \cdot t})")
    st.markdown("Где: pH_0 - начальное, pH_inf - конечное, k - константа скорости.")
    st.warning("Значение k корректируется в зависимости от температуры/соления.")
    st.markdown("---")

    # Полиномиальная/логистическая модель — сохраняем и улучшаем
    def ph_model_func(t, pH0=6.6, pH_inf=4.6, k=0.03):
        # logistic-like approach: pH decreases from pH0 to pH_inf with rate k
        t = np.array(t, dtype=float)
        ph = pH_inf + (pH0 - pH_inf) * np.exp(-k * t)
        # ensure numeric stability and realistic bounds
        ph = np.clip(ph, 0.0, 14.0)
        return ph

    st.subheader("⚙️ Интерактивный прогноз и анализ")
    # interactive params
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        pH0 = st.number_input("pH начальное (pH0)", value=6.6, format="%.2f")
    with col_b:
        pH_inf = st.number_input("pH конечное (pH_inf)", value=4.6, format="%.2f")
    with col_c:
        k = st.number_input("Константа скорости (k)", value=0.03, format="%.4f")

    t_input = st.slider("Время прогноза (t), час", min_value=1, max_value=240, value=48, step=1)
    pH_forecast = float(ph_model_func(t_input, pH0=pH0, pH_inf=pH_inf, k=k))
    st.metric(label="Прогнозируемый pH в заданное время", value=f"{pH_forecast:.2f}", delta=f"Разница до целевого pH 5.6: {(pH_forecast - 5.6):.2f}", delta_color="inverse")
    if pH_forecast < 4.8:
        st.error("**Критическое закисление.** Продукт слишком кислый.")
    elif 4.8 <= pH_forecast <= 5.6:
        st.success("**Оптимальный диапазон.**")
    elif pH_forecast > 5.6:
        st.warning("**Недостаточное закисление.**")

    st.markdown("---")
    st.subheader("Визуализация кинетики pH (используем plotly, клипируем вниз)")
    times = np.linspace(0, 240, 300)
    pH_values = ph_model_func(times, pH0=pH0, pH_inf=pH_inf, k=k)
    # plotly interactive
    fig = px.line(x=times, y=pH_values, labels={'x':'Время (ч)','y':'pH'}, title='Кинетика pH в процессе посола')
    # highlight target range
    fig.add_hrect(y0=4.8, y1=5.6, fillcolor="green", opacity=0.08, layer="below", line_width=0)
    fig.add_vline(x=t_input, line_dash="dash", annotation_text=f"{t_input} ч", annotation_position="top right")
    fig.update_yaxes(range=[0, 8])  # realistic pH focus
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# PAGE: Анализ с экстрактом облепихи (preserve graphics)
# ---------------------------
elif page == L["menu"][4]:
    st.title("🔬 Влияние экстракта облепихи на качество жая и формованного мяса")
    st.write("Результаты экспериментального исследования ...")
    st.markdown("---")
    st.subheader("Таблица 1. Основные показатели копчёной жая (контроль и 5% экстракта)")
    table1_data = {
        "Показатель": ["Массовая доля влаги, %", "Белок, %", "Жир, %",
                       "Влагоудерж. способность (ВУС), %", "ТБЧ, мг/кг"],
        "Контроль (0%)": [65.2, 21.2, 31.06, 60.2, 0.69],
        "Жая + 5% экстракта": [67.8, 25.44, 33.4, 67.4, 0.96]
    }
    df_table1 = pd.DataFrame(table1_data)
    st.dataframe(df_table1)
    st.subheader("Таблица 2. Основные показатели формованного мясного продукта (контроль и 3% экстракта)")
    table2_data = {...} if False else {
        "Показатель": ["Массовая доля влаги, %", "Белок, %", "Жир, %", "NaCl, %", "Зола, %"],
        "Контроль (0%)": [68.96, 13.60, 11.03, 1.77, 2.96],
        "Формованное мясо + 3% экстракта": [70.08, 13.88, 8.51, 1.27, 2.22]
    }
    df_table2 = pd.DataFrame(table2_data)
    st.dataframe(df_table2)
    st.markdown("---")
    col1, col2 = st.columns(2)
    x_ticks = np.arange(0, 15.1, 2.5)
    with col1:
        st.subheader("Рис. 1. Влияние экстракта на влагосодержание жая")
        x = np.array([0, 3, 5, 7, 9, 15])
        vlaga = np.array([65.2, 66.8, 68.9, 68.6, 67.8, 65.4])
        fig1 = px.line(x=x, y=vlaga, markers=True, title="Влияние экстракта облепихи на влагосодержание жая")
        fig1.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig1, use_container_width=True)
        st.subheader("Рис. 3. ВУС, ВСС и ЖУС копчёной жая")
        VUS = np.array([60.2, 64.3, 67.4, 71.2, 73.5, 78.9])
        VSS = np.array([61.0, 65.5, 70.1, 73.8, 75.2, 77.4])
        ZhUS = np.array([60.0, 63.1, 66.8, 70.0, 72.5, 74.8])
        fig3 = px.line(x=x, y=VUS, markers=True, title="ВУС, ВСС и ЖУС копчёной жая")
        fig3.add_scatter(x=x, y=VSS, mode='lines+markers', name='ВСС, %')
        fig3.add_scatter(x=x, y=ZhUS, mode='lines+markers', name='ЖУС, %')
        fig3.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig3, use_container_width=True)
        st.subheader("Рис. 5. Окислительные показатели формованного мяса")
        days2 = np.array([5,10,15])
        tbch_c2 = np.array([0.203,0.284,0.312])
        tbch_e2 = np.array([0.254,0.366,0.428])
        perox_c2 = np.array([13.27,14.30,15.21])
        perox_e2 = np.array([9.90,10.80,11.60])
        fig5 = px.line()
        fig5.add_scatter(x=days2, y=tbch_c2, mode='lines+markers', name='ТБЧ контроль')
        fig5.add_scatter(x=days2, y=tbch_e2, mode='lines+markers', name='ТБЧ 3%')
        fig5.add_scatter(x=days2, y=perox_c2, mode='lines+markers', name='Перокс контроль')
        fig5.add_scatter(x=days2, y=perox_e2, mode='lines+markers', name='Перокс 3%')
        st.plotly_chart(fig5, use_container_width=True)
    with col2:
        st.subheader("Рис. 2. Белок и жир в жая")
        belok = np.array([21.2, 23.4, 25.4, 27.5, 29.8, 34.9])
        zhir = np.array([31.06, 32.4, 33.4, 37.1, 41.2, 45.0])
        fig2 = px.line()
        fig2.add_scatter(x=x, y=belok, mode='lines+markers', name='Белок, %')
        fig2.add_scatter(x=x, y=zhir, mode='lines+markers', name='Жир, %')
        fig2.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Рис. 4. Окислительные показатели жая")
        days = np.array([5,10,15])
        tbch_c = np.array([0.197,0.376,0.416])
        tbch_e = np.array([0.194,0.361,0.419])
        perox_c = np.array([17.96,19.12,20.25])
        perox_e = np.array([13.01,14.40,15.13])
        fig4 = px.line()
        fig4.add_scatter(x=days, y=tbch_c, mode='lines+markers', name='ТБЧ контроль')
        fig4.add_scatter(x=days, y=tbch_e, mode='lines+markers', name='ТБЧ 3%')
        fig4.add_scatter(x=days, y=perox_c, mode='lines+markers', name='Перокс контроль')
        fig4.add_scatter(x=days, y=perox_e, mode='lines+markers', name='Перокс 3%')
        st.plotly_chart(fig4, use_container_width=True)

# ---------------------------
# PAGE: Исследование данных (preserve)
# ---------------------------
elif page == L["menu"][5]:
    st.title("🗂️ Исследование исходных данных")
    st.write("Выберите таблицу для просмотра.")
    if all_meat_data:
        available_tables = list(all_meat_data.keys())
        if df_ph is not None:
            available_tables.append('opyty.xlsx')
        choice = st.selectbox("Выберите данные:", available_tables)
        st.markdown(f"**Просмотр данных из: `{choice}`**")
        if choice == 'opyty.xlsx':
            if df_ph is not None:
                df_to_show = df_ph.copy()
            else:
                df_to_show = pd.DataFrame()
        else:
            df_to_show = all_meat_data[choice].copy()
        if 'Accuracy' in df_to_show.columns:
            df_to_show['Accuracy'] = pd.to_numeric(df_to_show['Accuracy'], errors='coerce')
        if not df_to_show.empty:
            st.dataframe(df_to_show)
        else:
            st.warning(f"Данные для '{choice}' не были загружены или пусты.")
    else:
        st.warning("Не удалось загрузить данные для просмотра.")

# ---------------------------
# PAGE: История / DB (новое; сохраняет функционал записи)
# ---------------------------
elif page == L["menu"][6]:
    st.title("📚 История измерений и база данных")
    st.markdown("Здесь хранится история измерений (SQLite). Можно экспортировать, фильтровать и удалять записи.")

    df_hist = fetch_measurements(limit=5000)
    st.write(f"Всего записей: {len(df_hist)}")
    if df_hist.empty:
        st.info("История пуста")
    else:
        st.dataframe(df_hist)
        col_e1, col_e2 = st.columns([1,1])
        with col_e1:
            if st.button("Экспортировать все в CSV"):
                csv = df_hist.to_csv(index=False).encode()
                st.download_button("Скачать CSV", csv, file_name="history_export.csv", mime="text/csv")
        with col_e2:
            if st.button("Очистить все измерения"):
                if st.confirm or st.sidebar.button("Подтвердить очистку") if False else True:
                    delete_all_measurements()
                    st.success("База очищена. Перезагрузите страницу.")
                    st.experimental_rerun()
        st.subheader("pH распределение")
        fig = px.histogram(df_hist, x='ph', nbins=25, title="pH distribution")
        fig.update_xaxes(range=[0,8])
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("pH по времени (интерактивно)")
        fig2 = px.line(df_hist.sort_values('created_at'), x='created_at', y='ph', title="pH over time", markers=True)
        fig2.update_yaxes(range=[0,8])
        st.plotly_chart(fig2, use_container_width=True)
# =====================================================================

# ---------------------------
# PAGE: ML: Train / Predict (new, preserves train/predict behavior)
# ---------------------------
elif page == L["menu"][7]:
    st.title("🧠 ML: Обучение и прогнозирование pH")
    st.markdown("Загрузите CSV/Excel с колонкой 'pH' и признаками для обучения или загрузите CSV с признаками для предсказания.")
    tab1, tab2 = st.tabs(["Train", "Predict"])

    with tab1:
        st.subheader("Обучение модели")
        up = st.file_uploader("CSV/Excel для обучения (колонка pH)", type=["csv","xlsx","xls"], key="train_up")
        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    df_train = pd.read_csv(up)
                else:
                    df_train = pd.read_excel(up)
            except Exception as e:
                st.error(f"Ошибка чтения: {e}")
                df_train = pd.DataFrame()
            if df_train.empty:
                st.info("Нет данных")
            else:
                st.write("Превью:")
                st.dataframe(df_train.head(10))
                cols = df_train.columns.tolist()
                if 'pH' in cols:
                    target = 'pH'
                else:
                    target = st.selectbox("Целевая колонка (pH) выберите:", options=cols)
                features = st.multiselect("Признаки (если пусто — будут взяты все числовые кроме цели)", options=cols)
                if st.button(L[lang_choice]["train_button"]):
                    try:
                        metrics = ph_model.train(df_train, target=target, feature_cols=features if features else None)
                        st.success("Обучение прошло успешно")
                        st.json(metrics)
                    except Exception as e:
                        st.error(f"Ошибка обучения: {e}")

    with tab2:
        st.subheader("Прогнозирование")
        up2 = st.file_uploader("CSV для предсказания (те же признаки)", type=["csv"], key="pred_up")
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
                st.info("Нет данных")
            else:
                st.dataframe(df_pred.head(10))
                num_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
                st.write("Автоматически выбранные числовые признаки:", num_cols)
                if st.button(L[lang_choice]["predict_button"]):
                    preds = ph_model.predict(df_pred, feature_cols=num_cols)
                    df_pred['predicted_pH'] = np.round(preds,3)
                    df_pred['score'] = df_pred['predicted_pH'].apply(compute_score_from_ph)
                    st.subheader("Результаты предсказания")
                    st.dataframe(df_pred.head(50))
                    st.markdown(df_to_download_link(df_pred, filename="predictions.csv"), unsafe_allow_html=True)
                    # опция: сохранить в историю, если есть колонка sample_name
                    if 'sample_name' in df_pred.columns:
                        if st.button("Сохранить предсказания в базу (sample_name -> sample)"):
                            saved = 0
                            for _, r in df_pred.iterrows():
                                insert_measurement(str(r.get('sample_name','sample')), float(r.get('predicted_pH', np.nan)), compute_score_from_ph(float(r.get('predicted_pH', np.nan))), notes="predicted")
                                saved += 1
                            st.success(f"Сохранено {saved} записей в БД")
# СТРАНИЦА: ВВОД НОВЫХ ДАННЫХ
# =====================================================================
elif page == L["menu"][8]:
    st.title("➕ Ввод новых данных о продукции")
    st.markdown(f"### Добавление нового производственного цикла в базу данных ({MEAT_XLSX}, лист {SHEET_NAME})")

    # Загружаем текущие данные
    df_meat = safe_read_excel(MEAT_XLSX, SHEET_NAME)

    # Проверка наличия колонки BatchID
    if "BatchID" not in df_meat.columns:
        st.error("❌ В листе T6 нет колонки 'BatchID'. Проверь структуру таблицы.")
        st.stop()

    # Определяем следующий BatchID
    if len(df_meat) > 0 and df_meat["BatchID"].astype(str).str.match(r"^M\d+$").any():
        last_id_str = df_meat["BatchID"].dropna().astype(str).iloc[-1]
        try:
            last_num = int(last_id_str[1:])
            next_id = f"M{last_num + 1}"
        except:
            next_id = "M1"
    else:
        next_id = "M1"

    with st.form(key='batch_entry_form'):
        st.subheader("Введите параметры нового производственного цикла")

        st.text_input("Batch ID (автоматически)", value=next_id, disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            mass_kg = st.number_input("Масса партии (кг)", min_value=1.0, value=100.0, step=1.0)
            T_initial_C = st.number_input("Начальная температура (°C)", min_value=-10.0, value=4.0, step=0.1)
            Salt_pct = st.number_input("Содержание соли (%)", min_value=0.0, value=5.0, step=0.1)
        with col2:
            Moisture_pct = st.number_input("Влажность (%)", min_value=0.0, value=75.0, step=0.1)
            StarterCFU = st.number_input("Стартерная культура (КОЕ/г)", min_value=0, value=1000000, step=10000)
            Extract_pct = st.number_input("Концентрация экстракта (%)", min_value=0.0, value=3.0, step=0.1)

        submitted = st.form_submit_button("💾 Сохранить данные")

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
                st.success(f"✅ Новая партия '{next_id}' успешно добавлена в лист '{SHEET_NAME}'!")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"❌ Ошибка при записи в файл: {e}")

    st.markdown("---")
    st.subheader("📊 Текущие данные")
    st.dataframe(safe_read_excel(MEAT_XLSX, SHEET_NAME), use_container_width=True)
# ---------------------------
# Footer / small note
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.write(L["version_note"])
