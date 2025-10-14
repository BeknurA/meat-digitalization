import json
import io
import zipfile
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# optional sklearn
try:
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    SKLEARN = True
except Exception:
    SKLEARN = False

# ---------------------------
# Настройки: путь к данным
# ---------------------------
# Предпочтение: взять папку, где лежит этот скрипт
DATA_DIR = Path(__file__).parent
# fallback: старая папка (если ты запускаешь не из папки)
fallback = Path(r"C:\Users\aidar\OneDrive\Desktop\МАДИНА\Milk_Digitalization")
if any(fallback.glob("*.csv")) and not any(DATA_DIR.glob("*.csv")):
    DATA_DIR = fallback

PRODUCTS_CSV = DATA_DIR / "Products.csv"
SAMPLES_CSV = DATA_DIR / "Samples.csv"
MEASUREMENTS_CSV = DATA_DIR / "Measurements.csv"
VITAMINS_CSV = DATA_DIR / "Vitamins_AminoAcids.csv"
STORAGE_CSV = DATA_DIR / "Storage_Conditions.csv"
NORMS_JSON = DATA_DIR / "process_norms.json"

# ---------------------------
# Утилиты: чтение/запись/парсинг
# ---------------------------
def safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def append_row_csv(path: Path, row: dict, cols_order=None):
    df_new = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    if cols_order:
        for c in cols_order:
            if c not in df_new.columns:
                df_new[c] = ""
        df_new = df_new[cols_order]
    df_new.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')

def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "" or "не обнаруж" in s.lower():
        return np.nan
    s = s.replace(',', '.').replace('×10^', 'e').replace('x10^', 'e').replace('×', '')
    if '±' in s:
        s = s.split('±')[0]
    cleaned = ''
    for ch in s:
        if ch.isdigit() or ch in '.-+eE':
            cleaned += ch
        else:
            break
    try:
        return float(cleaned)
    except Exception:
        return np.nan

def download_zip(paths, filename="Milk_Digitalization_all_csv.zip"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in paths:
            if Path(p).exists():
                z.write(p, arcname=Path(p).name)
    buf.seek(0)
    st.download_button("Скачать ZIP", data=buf, file_name=filename, mime="application/zip")

def embed_pdf(path: Path):
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode('utf-8')
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600"></iframe>'
    st.components.v1.html(html, height=600, scrolling=True)

# ---------------------------
# Загрузим данные (и нормализуем колонки если нужно)
# ---------------------------
products = safe_read_csv(PRODUCTS_CSV)
samples = safe_read_csv(SAMPLES_CSV)
measurements = safe_read_csv(MEASUREMENTS_CSV)
vitamins = safe_read_csv(VITAMINS_CSV)
storage = safe_read_csv(STORAGE_CSV)

# Нормализатор простых названий колонок
def ensure_col(df, candidates, new_name):
    if df.empty:
        return df, None
    for col in df.columns:
        for cand in candidates:
            if col.strip().lower() == cand.strip().lower():
                return df.rename(columns={col: new_name}), new_name
    return df, None

products, _ = ensure_col(products, ["product_id","id"], "product_id")
products, _ = ensure_col(products, ["name","product_name"], "name")
products, _ = ensure_col(products, ["type","category"], "type")
products, _ = ensure_col(products, ["source"], "source")
products, _ = ensure_col(products, ["description"], "description")

samples, _ = ensure_col(samples, ["sample_id","id"], "sample_id")
samples, _ = ensure_col(samples, ["product_id","product"], "product_id")
samples, _ = ensure_col(samples, ["reg_number"], "reg_number")
samples, _ = ensure_col(samples, ["date_received","date"], "date_received")
samples, _ = ensure_col(samples, ["storage_days","duration_days"], "storage_days")
samples, _ = ensure_col(samples, ["conditions"], "conditions")
samples, _ = ensure_col(samples, ["notes"], "notes")

measurements, _ = ensure_col(measurements, ["id"], "id")
measurements, _ = ensure_col(measurements, ["sample_id","sample"], "sample_id")
measurements, _ = ensure_col(measurements, ["parameter","param","indicator"], "parameter")
measurements, _ = ensure_col(measurements, ["actual_value","value","measurement"], "actual_value")
measurements, _ = ensure_col(measurements, ["unit"], "unit")
measurements, _ = ensure_col(measurements, ["method"], "method")

storage, _ = ensure_col(storage, ["sample_id"], "sample_id")
storage, _ = ensure_col(storage, ["temperature_C","temperature_c","temp"], "temperature_C")
storage, _ = ensure_col(storage, ["humidity_pct","humidity"], "humidity_pct")
storage, _ = ensure_col(storage, ["duration_days"], "duration_days")

# Преобразуем id в Int64 (без падения при NaN)
def to_intlike(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
    return df

products = to_intlike(products, "product_id")
samples = to_intlike(samples, "sample_id")
samples = to_intlike(samples, "product_id")
measurements = to_intlike(measurements, "sample_id")
storage = to_intlike(storage, "sample_id")

# Парсим числовые значения в measurements
if 'actual_value' in measurements.columns:
    measurements['actual_numeric'] = measurements['actual_value'].apply(parse_numeric)
else:
    measurements['actual_numeric'] = np.nan

# parse dates
if 'date_received' in samples.columns:
    samples['date_received'] = pd.to_datetime(samples['date_received'], errors='coerce')

# ---------------------------
# Нормы (файл JSON или дефолт)
# ---------------------------
default_norms = {
    "Пастеризация": {"min":72.0, "max":75.0, "unit":"°C", "note":"типовая пастеризация (из протоколов)"},
    "Охлаждение": {"min":2.0, "max":6.0, "unit":"°C", "note":"хранение/охлаждение"},
    "Ферментация": {"min":18.0, "max":42.0, "unit":"°C", "note":"вариативно в зависимости от рецептуры"}
}
if NORMS_JSON.exists():
    try:
        norms = json.loads(NORMS_JSON.read_text(encoding='utf-8'))
    except Exception:
        norms = default_norms
else:
    norms = default_norms

# ---------------------------
# UI: стили для цветных блоков
# ---------------------------
st.set_page_config(page_title="Milk Digitalization", layout="wide")
st.markdown("""
<style>
.card{background:#fff;padding:12px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.06);margin-bottom:12px}
.prod-title{font-weight:700;color:#0b4c86}
.step{border-radius:8px;padding:12px;margin-bottom:6px;color:white;font-weight:600}
.step-small{font-size:13px;color:#333}
</style>
""", unsafe_allow_html=True)

# Цвета для этапов (можно расширять)
STEP_COLORS = {
    "pasteurization":"#d9534f",  # red
    "cooling":"#0275d8",         # blue
    "fermentation":"#5cb85c",    # green
    "accept":"#5bc0de",          # cyan
    "normalization":"#f0ad4e",   # orange
    "homogenization":"#6f42c1",  # purple
    "inoculation":"#20c997",     # teal
    "coagulation":"#fd7e14",     # dark-orange
    "pressing":"#6c757d",        # gray
    "filtration":"#007bff",      # blue
    "storage":"#17a2b8"
}

def color_for_step(step_id):
    # try to pick by keyword
    sid = step_id.lower()
    for k,v in STEP_COLORS.items():
        if k in sid:
            return v
    # fallback color
    return "#0b4c86"

# ---------------------------
# Session state init
# ---------------------------
if 'page' not in st.session_state: st.session_state['page'] = 'Главная'
if 'selected_product' not in st.session_state: st.session_state['selected_product'] = None
if 'selected_step' not in st.session_state: st.session_state['selected_step'] = None

# Навигация
page = st.sidebar.radio("Меню", ["Главная","Продукт","Модели и аналитика"], index=["Главная","Продукт","Модели и аналитика"].index(st.session_state['page']) if st.session_state['page'] in ["Главная","Продукт","Модели и аналитика"] else 0)
st.session_state['page'] = page

def goto_product(pid):
    st.session_state['selected_product'] = int(pid)
    st.session_state['page'] = 'Продукт'
    st.session_state['selected_step'] = None
    st.rerun()

# ---------------------------
# Главная — 5 карточек
# ---------------------------
if st.session_state['page'] == 'Главная':
    st.title("🥛 Milk Digitalization")
    st.write("Платформа: карточки продуктов → нажми 'Подробнее' для открытия страницы продукта.")

    # фиксированные 5 продуктов (пользователь просил)
    fixed_products = [
        {"product_id":1,"name":"Молоко (коровье)","type":"молоко","source":"коровье","description":"Свежее молоко"},
        {"product_id":2,"name":"Молоко (козье)","type":"молоко","source":"козье","description":"Свежее молоко"},
        {"product_id":3,"name":"Сары ірімшік (коровье)","type":"сыр","source":"коровье","description":"Твёрдый сыр"},
        {"product_id":4,"name":"Сары ірімшік (козье)","type":"сыр","source":"козье","description":"Твёрдый сыр"},
        {"product_id":5,"name":"Айран","type":"кисломолочный","source":"коровье","description":"Кисломолочный продукт"}
    ]

    display_products = []
    for fp in fixed_products:
        # try to prefer data from CSV
        chosen = None
        if not products.empty:
            try:
                if 'product_id' in products.columns:
                    match = products[products['product_id'] == fp['product_id']]
                    if not match.empty:
                        chosen = match.iloc[0].to_dict()
            except Exception:
                chosen = None
        display_products.append(chosen if chosen is not None else fp)

    cols = st.columns(3)
    for i,p in enumerate(display_products):
        c = cols[i%3]
        with c:
            st.markdown(f"<div class='card'><div class='prod-title'>{p['name']}</div><div class='step-small'>Тип: {p.get('type','-')} • Источник: {p.get('source','-')}</div><div style='margin-top:8px'>{p.get('description','')}</div></div>", unsafe_allow_html=True)
            if st.button("Подробнее", key=f"open_{i}"):
                goto_product(p['product_id'])

    st.markdown("---")
    c1,c2 = st.columns(2)
    if c1.button("Обновить данные (reload CSV)"):
        # просто перезапускит приложение чтобы перечитать CSV
        st.experimental_rerun()
    if c2.button("Скачать CSV ZIP"):
        download_zip([PRODUCTS_CSV,SAMPLES_CSV,MEASUREMENTS_CSV,VITAMINS_CSV,STORAGE_CSV])

# ---------------------------
# Продукт — страница продукта + блок-схема
# ---------------------------
elif st.session_state['page'] == 'Продукт':
    pid = st.session_state.get('selected_product', None)
    if pid is None:
        st.info("Открой продукт с главной страницы.")
    else:
        # узнаём данные продукта
        prod = None
        if not products.empty and 'product_id' in products.columns:
            m = products[products['product_id'] == int(pid)]
            if not m.empty:
                prod = m.iloc[0].to_dict()
        if prod is None:
            # fallback на фиксированное имя
            names = {1:"Молоко (коровье)",2:"Молоко (козье)",3:"Сары ірімшік (коровье)",4:"Сары ірімшік (козье)",5:"Айран"}
            prod = {"product_id":pid,"name":names.get(pid,f"Продукт {pid}"),"type":"-","source":"-","description":""}

        st.header(prod['name'])
        st.write(f"**Тип:** {prod.get('type','-')}  •  **Источник:** {prod.get('source','-')}")
        if prod.get('description'):
            st.write(prod.get('description'))

        st.markdown("---")
        st.subheader("Процесс изготовления — нажми блок ниже, чтобы открыть деталь этапа")
        # определяем шаги для продукта (взяты из присланных тобой документов)
        name_low = str(prod['name']).lower()
        if "айран" in name_low:
            steps = [
                ("accept","Приемка и контроль сырья"),
                ("normalization","Нормализация состава"),
                ("pasteurization","Пастеризация (72–75°C)"),
                ("cooling_to_inoc","Охлаждение до заквашивания (~40–42°C)"),
                ("inoculation","Добавление закваски"),
                ("fermentation","Ферментация (контроль pH)"),
                ("final_cooling","Охлаждение и фасовка")
            ]
        elif "сары" in name_low or "ірімшік" in name_low:
            steps = [
                ("accept","Приемка и подготовка"),
                ("pasteurization","Пастеризация"),
                ("coagulation","Свертывание/коагуляция"),
                ("whey_removal","Отделение сыворотки"),
                ("pressing","Прессование"),
                ("salting","Посолка/обработка"),
                ("ripening","Выдержка / созревание")
            ]
        else:
            steps = [
                ("accept","Приемка и контроль сырья"),
                ("filtration","Фильтрация / Нормализация"),
                ("pasteurization","Пастеризация (72–75°C)"),
                ("cooling","Охлаждение (2–6°C)"),
                ("filling","Розлив / Упаковка"),
                ("storage","Хранение")
            ]

        # отрисовка блоков: colored div + кнопка под ним
        for sid,label in steps:
            color = color_for_step(sid)
            st.markdown(f"<div class='step' style='background:{color};'>{label}</div>", unsafe_allow_html=True)
            if st.button("Открыть этап", key=f"openstep_{pid}_{sid}"):
                st.session_state['selected_step'] = sid
                st.session_state['selected_step_label'] = label
                st.rerun()

        # показываем информацию о выбранном этапе
        if st.session_state.get('selected_step'):
            st.markdown("---")
            sel = st.session_state['selected_step']
            sel_label = st.session_state.get('selected_step_label', sel)
            st.subheader(f"Детали этапа: {sel_label}")

            # получаем норму (из json или дефолт)
            step_norm = None
            if NORMS_JSON.exists():
                try:
                    js = json.loads(NORMS_JSON.read_text(encoding='utf-8'))
                    # ищем ключ по id или по label
                    step_norm = js.get(sel) or js.get(sel_label) or None
                except:
                    step_norm = None
            if step_norm is None:
                # эвристика
                if "пастер" in sel_label.lower():
                    step_norm = norms.get("Пастеризация")
                elif "охлаж" in sel_label.lower() or "хран" in sel_label.lower():
                    step_norm = norms.get("Охлаждение")
                elif "фермент" in sel_label.lower():
                    step_norm = norms.get("Ферментация")
            if step_norm:
                st.write(f"**Норма:** {step_norm.get('min','-')} — {step_norm.get('max','-')} {step_norm.get('unit','')}")
                if step_norm.get('note'):
                    st.caption(step_norm.get('note'))
            else:
                st.info("Норма для этапа не найдена (могу создать process_norms.json по протоколам если хочешь).")

            # показать связанные измерения для партий этого продукта
            st.write("Связанные измерения (по партиям продукта):")
            if 'product_id' in samples.columns:
                prod_samples = samples[samples['product_id'] == int(pid)]
            else:
                prod_samples = pd.DataFrame()

            if prod_samples.empty:
                st.info("Партии не найдены (Samples.csv пуст или нет записей для этого продукта).")
            else:
                rel = measurements[measurements['sample_id'].isin(prod_samples['sample_id'])] if ('sample_id' in measurements.columns) else pd.DataFrame()
                if rel.empty:
                    st.info("Измерения для партий не найдены (Measurements.csv пуст или нет записей).")
                else:
                    rel_show = rel.copy()
                    if 'actual_numeric' not in rel_show.columns and 'actual_value' in rel_show.columns:
                        rel_show['actual_numeric'] = rel_show['actual_value'].apply(parse_numeric)
                    st.dataframe(rel_show.sort_values(['sample_id','parameter']).reset_index(drop=True))
                    # если есть temp measurements - сравнить с нормой
                    if step_norm and 'min' in step_norm and 'max' in step_norm:
                        temp_mask = rel_show['parameter'].astype(str).str.lower().str.contains('темпера|temp', na=False)
                        if temp_mask.any():
                            tmp = rel_show[temp_mask].copy()
                            tmp['num'] = tmp['actual_numeric']
                            tmp['ok'] = tmp['num'].apply(lambda x: True if pd.notna(x) and step_norm['min'] <= x <= step_norm['max'] else False)
                            st.write("Температурные измерения и соответствие норме:")
                            st.dataframe(tmp[['sample_id','parameter','actual_value','num','ok']].reset_index(drop=True))
                        else:
                            st.info("Температурных параметров в измерениях не обнаружено для этого продукта.")

            # форма добавления партии
            st.markdown("#### Добавить партию")
            with st.form(f"add_batch_{pid}", clear_on_submit=True):
                new_sample_id = int(samples['sample_id'].max())+1 if ('sample_id' in samples.columns and not samples.empty and samples['sample_id'].notna().any()) else 1
                reg = st.text_input("Регистрационный номер", value=f"{200+new_sample_id}")
                date_rcv = st.date_input("Дата поступления", value=datetime.now().date())
                storage_days = st.number_input("Срок хранения (дни)", min_value=0, value=0)
                temp = st.number_input("Температура (°C)", value=21.0, format="%.2f")
                humid = st.number_input("Влажность (%)", value=64)
                notes = st.text_area("Примечания")
                sub = st.form_submit_button("Сохранить партию")
            if sub:
                row = {"sample_id":int(new_sample_id),"product_id":int(pid),"reg_number":reg,"date_received":date_rcv.strftime("%Y-%m-%d"),"storage_days":int(storage_days),"conditions":f"{temp}°C, {humid}%","notes":notes}
                append_row_csv(SAMPLES_CSV, row, cols_order=["sample_id","product_id","reg_number","date_received","storage_days","conditions","notes"])
                st.success("Партия добавлена. Нажми 'Обновить данные' на главной, чтобы перезагрузить CSV в приложении.")

            # форма добавления измерения
            st.markdown("#### Добавить измерение")
            with st.form(f"add_meas_{pid}", clear_on_submit=True):
                sample_opts = prod_samples['sample_id'].tolist() if not prod_samples.empty else []
                sample_choice = st.selectbox("Sample ID", options=sample_opts) if sample_opts else None
                param = st.text_input("Параметр (например: pH, Белок, Жир, Температура)")
                val = st.text_input("Значение (например: 4.6 или 89.54±1.07)")
                unit = st.text_input("Единица", value="")
                method = st.text_input("Метод (ГОСТ ...)", value="")
                addm = st.form_submit_button("Добавить измерение")
            if addm:
                if sample_choice is None:
                    st.error("Сначала добавь партию (sample) для продукта.")
                else:
                    new_mid = int(measurements['id'].max())+1 if ('id' in measurements.columns and not measurements.empty and measurements['id'].notna().any()) else int(datetime.now().timestamp())
                    rowm = {"id": new_mid, "sample_id": int(sample_choice), "parameter": param, "unit": unit, "actual_value": val, "method": method}
                    append_row_csv(MEASUREMENTS_CSV, rowm, cols_order=["id","sample_id","parameter","unit","actual_value","method"])
                    st.success("Измерение добавлено. Нажми 'Обновить данные' на главной, чтобы увидеть изменения.")

# ---------------------------
# Модели и аналитика
# ---------------------------
elif st.session_state['page'] == 'Модели и аналитика':
    st.title("Модели и аналитика — регрессии и прогнозы")
    st.write("Подготовка данных → выбор target/feature → обучение → визуализация метрик")

    if measurements.empty or samples.empty:
        st.warning("Нет данных в Measurements.csv и/или Samples.csv.")
    else:
        meas = measurements.copy()
        if 'actual_numeric' not in meas.columns and 'actual_value' in meas.columns:
            meas['actual_numeric'] = meas['actual_value'].apply(parse_numeric)
        pivot = meas.pivot_table(index='sample_id', columns='parameter', values='actual_numeric', aggfunc='first').reset_index()
        df_all = samples.merge(pivot, on='sample_id', how='left')

        st.subheader("Preview данных")
        st.dataframe(df_all.head(30))

        ignore = ['sample_id','product_id','reg_number','date_received','storage_days','conditions','notes']
        possible = [c for c in df_all.columns if c not in ignore]
        if not possible:
            st.warning("Нет признаков для моделирования.")
        else:
            target = st.selectbox("Target (цель)", options=possible, index=0)
            features = st.multiselect("Features (признаки)", options=[c for c in possible if c!=target], default=[c for c in ['Белок','Жир','Влага','storage_days'] if c in df_all.columns][:3])
            if not features:
                st.info("Выберите признаки.")
            else:
                dataset = df_all[[target]+features].dropna()
                st.write("Доступных строк для обучения:", len(dataset))
                if len(dataset) < 5:
                    st.warning("Недостаточно данных. Нужны хотя бы 5 строк.")
                else:
                    X = dataset[features].astype(float).values
                    y = dataset[target].astype(float).values
                    test_size = st.slider("Тестовая доля", 0.1, 0.5, 0.3)

                    if SKLEARN:
                        alg = st.selectbox("Алгоритм", ["Linear","Ridge","Lasso"])
                        Model = LinearRegression if alg=="Linear" else (Ridge if alg=="Ridge" else Lasso)
                        model = Model()
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        rmse = mean_squared_error(y_test, y_pred, squared=False)
                        mae = mean_absolute_error(y_test, y_pred)
                        st.metric("R²", f"{r2:.4f}")
                        st.metric("RMSE", f"{rmse:.4f}")
                        st.metric("MAE", f"{mae:.4f}")

                        # coefficients
                        try:
                            coefs = dict(zip(features, model.coef_))
                            st.write("Коэффициенты:")
                            st.table(pd.DataFrame.from_dict(coefs, orient='index', columns=['coef']).round(6))
                        except Exception:
                            pass

                        # plot
                        fig, ax = plt.subplots(figsize=(8,4))
                        if len(features) == 1:
                            x_test = X_test.flatten()
                            idx = np.argsort(x_test)
                            ax.scatter(X_test, y_test, label="Actual", alpha=0.7)
                            ax.plot(x_test[idx], model.predict(X_test)[idx], color='red', label='Predict')
                            ax.set_xlabel(features[0]); ax.set_ylabel(target)
                            ax.legend()
                        else:
                            ax.scatter(y_test, y_pred, alpha=0.7)
                            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
                            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
                        st.pyplot(fig)
                    else:
                        # fallback: if single feature use numpy.polyfit
                        if len(features) == 1:
                            x = X.flatten()
                            coef = np.polyfit(x, y, 1)
                            slope, intercept = coef[0], coef[1]
                            st.write(f"Простая регрессия (numpy): {target} = {slope:.4f}*{features[0]} + {intercept:.4f}")
                            fig, ax = plt.subplots(figsize=(8,4))
                            idx = np.argsort(x)
                            ax.scatter(x, y, alpha=0.7)
                            ax.plot(x[idx], np.polyval(coef, x[idx]), color='red')
                            st.pyplot(fig)
                        else:
                            st.warning("Установи scikit-learn для обучения на нескольких признаках: pip install scikit-learn")
#streamlit run app1.py
# ---------------------------
# Конец file
# ---------------------------
