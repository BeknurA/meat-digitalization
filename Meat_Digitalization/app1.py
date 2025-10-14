# app.py — Полная комбинированная версия: Аналитика + Детализированный интерфейс + Новая модель pH
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import io
import json

# --- Проверка наличия 'openpyxl' ---
try:
    import openpyxl
except ImportError:
    st.error(
        "Критическая ошибка: 'openpyxl' не найдена. Установите ее, выполнив в терминале команду: pip install openpyxl")
    st.stop()

# ---------------------------
# Настройки страницы и путей к данным
# ---------------------------
st.set_page_config(page_title="Платформа Жая", layout="wide")
DATA_DIR = Path(__file__).parent
MEAT_DATA_XLSX = DATA_DIR / "meat_data.xlsx"
OPYTY_XLSX = DATA_DIR / "opyty.xlsx"
PRODUCTS_CSV = DATA_DIR / "Products.csv"
SAMPLES_CSV = DATA_DIR / "Samples.csv"
MEASUREMENTS_CSV = DATA_DIR / "Measurements.csv"


# ---------------------------
# Утилиты
# ---------------------------
def safe_read_csv(path: Path):
    if not path.exists(): return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="latin1")


def append_row_csv(path: Path, row: dict, cols_order=None):
    df_new = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    if cols_order:
        for c in cols_order:
            if c not in df_new.columns: df_new[c] = ""
        df_new = df_new[cols_order]
    df_new.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')


# ---------------------------
# Загрузка данных
# ---------------------------
@st.cache_data
def load_all_data():
    data_sheets = {}
    try:
        if MEAT_DATA_XLSX.exists():
            xls = pd.ExcelFile(MEAT_DATA_XLSX)
            for sheet_name in xls.sheet_names:
                data_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception as e:
        st.warning(f"Не удалось прочитать '{MEAT_DATA_XLSX.name}': {e}")

    df_ph = None
    if OPYTY_XLSX.exists():
        try:
            df_ph = pd.read_excel(OPYTY_XLSX)
        except Exception as e:
            st.warning(f"Не удалось прочитать '{OPYTY_XLSX.name}': {e}")

    products_df = safe_read_csv(PRODUCTS_CSV)
    samples_df = safe_read_csv(SAMPLES_CSV)
    measurements_df = safe_read_csv(MEASUREMENTS_CSV)

    return data_sheets, df_ph, products_df, samples_df, measurements_df


all_meat_data, df_ph, products, samples, measurements = load_all_data()


# ---------------------------
# Математические модели
# ---------------------------
def calculate_stability(pressure, viscosity):
    p, v = pressure, viscosity
    return 27.9 - 0.1 * p - 1.94 * v - 0.75 * p * v - 0.67 * p ** 2 - 2.5 * v ** 2


def get_ph_model(time_h, ph_obs):
    """Использует логарифмическую модель: pH = a*log(t) + b."""
    # Отфильтровываем некорректные значения (NaN и время <= 0)
    valid = ~np.isnan(time_h) & ~np.isnan(ph_obs) & (time_h > 0)
    t, y = time_h[valid], ph_obs[valid]

    # Проверяем, достаточно ли данных для построения модели
    if len(t) < 2:
        return None, None, None, None, None

    # Преобразуем время в логарифм и строим матрицу для регрессии
    log_t = np.log(t).reshape(-1, 1)
    X = np.hstack([log_t, np.ones(log_t.shape)])

    # Находим коэффициенты 'a' (наклон) и 'b' (смещение)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Создаем функцию, которая будет делать предсказания по нашей модели
    model_func = lambda t_in: coeffs[0] * np.log(t_in) + coeffs[1]

    # Получаем предсказанные значения для существующих точек
    y_hat = model_func(t)

    # Рассчитываем метрики точности
    r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))

    return model_func, y_hat, rmse, r2, coeffs


# ---------------------------
# Навигация и состояние сессии
# ---------------------------
st.sidebar.title("Меню платформы")
page_options = ["Главная", "Процесс производства Жая", "Анализ стабильности", "Моделирование pH", "Исследование данных"]
page = st.sidebar.radio("Выберите раздел:", page_options)

if 'selected_product_id' not in st.session_state: st.session_state.selected_product_id = None
if 'selected_step' not in st.session_state: st.session_state.selected_step = None

# ==========================================================================
# СТРАНИЦА 1: ГЛАВНАЯ
# ==========================================================================
if page == "Главная":
    st.title("🐎 Цифровая платформа для производства и моделирования Жая")
    st.write(
        "Эта система объединяет описание технологических процессов и интерактивные математические модели для анализа и прогнозирования качества продукции.")

    fixed_products = [
        {"id": 1, "name": "Партия №1 (Классический рецепт)",
         "description": "Контрольная партия, произведенная по стандартной технологии."},
        {"id": 2, "name": "Опыт №1 (Измененный посол)",
         "description": "Экспериментальная партия с новой рецептурой посолочной смеси."},
        {"id": 3, "name": "Опыт №2 (Ускоренная сушка)",
         "description": "Тестирование нового режима температуры и влажности в камере."},
        {"id": 4, "name": "Опыт №3 (Длительное созревание)",
         "description": "Партия с увеличенным сроком холодной ферментации."},
        {"id": 5, "name": "Опыт №4 (Новые стартовые культуры)",
         "description": "Использование альтернативных микроорганизмов для созревания."}
    ]

    st.markdown("---")
    st.header("Производственные партии и опыты")
    st.info("Выберите раздел в меню слева, чтобы начать работу с данными и моделями.")

    cols = st.columns(3)
    for i, p in enumerate(fixed_products):
        with cols[i % 3]:
            st.subheader(p['name'])
            st.write(p['description'])

# ==========================================================================
# СТРАНИЦА 2: ПРОЦЕСС ПРОИЗВОДСТВА
# ==========================================================================
elif page == "Процесс производства Жая":
    st.title("⚙️ Технологический процесс производства Жая")
    st.write("Интерактивное отслеживание этапов производства для выбранной партии.")

    if products.empty:
        products = pd.DataFrame([
            {"product_id": 1, "name": "Партия №1 (Классический рецепт)"},
            {"product_id": 2, "name": "Опыт №1 (Измененный посол)"},
        ])

    product_options = {row['name']: row['product_id'] for index, row in products.iterrows()}
    selected_product_name = st.selectbox("Выберите партию или опыт для просмотра:", options=product_options.keys())

    if selected_product_name:
        st.session_state.selected_product_id = product_options[selected_product_name]
        st.markdown(f"**Вы работаете с партией:** `{selected_product_name}`")

        steps = [
            ("receiving", "Приемка и подготовка сырья"),
            ("deboning", "Обвалка и жиловка"),
            ("curing", "Посол и добавление специй"),
            ("aging", "Созревание (ферментация)"),
            ("drying", "Сушка и вяление"),
            ("quality", "Контроль качества и упаковка")
        ]

        for sid, label in steps:
            if st.button(label, key=sid, use_container_width=True):
                st.session_state.selected_step = {"id": sid, "label": label}

        if st.session_state.selected_step:
            st.markdown("---")
            step_info = st.session_state.selected_step
            st.subheader(f"Детали этапа: {step_info['label']}")
            st.write("**Зарегистрированные измерения:**")
            pid = st.session_state.selected_product_id
            prod_samples = samples[samples['product_id'] == pid] if not samples.empty else pd.DataFrame()

            if prod_samples.empty:
                st.info("Для данной партии еще не зарегистрировано ни одного образца.")
            else:
                rel_measurements = measurements[measurements['sample_id'].isin(
                    prod_samples['sample_id'])] if not measurements.empty else pd.DataFrame()
                if rel_measurements.empty:
                    st.info("Для образцов этой партии еще нет измерений.")
                else:
                    st.dataframe(rel_measurements)

            col1, col2 = st.columns(2)
            with col1:
                with st.form(f"add_sample_form", clear_on_submit=True):
                    st.markdown("#### Добавить образец в партию")
                    reg_num = st.text_input("Регистрационный номер образца")
                    notes = st.text_area("Примечания")
                    if st.form_submit_button("Сохранить образец"):
                        new_id = samples['sample_id'].max() + 1 if not samples.empty and samples[
                            'sample_id'].notna().any() else 1
                        new_row = {"sample_id": new_id, "product_id": pid, "reg_number": reg_num, "notes": notes}
                        append_row_csv(SAMPLES_CSV, new_row,
                                       cols_order=["sample_id", "product_id", "reg_number", "date_received",
                                                   "storage_days", "conditions", "notes"])
                        st.success(f"Образец #{new_id} добавлен!")
                        st.rerun()

            with col2:
                with st.form(f"add_measurement_form", clear_on_submit=True):
                    st.markdown("#### Добавить измерение")
                    sample_opts = prod_samples['sample_id'].tolist() if not prod_samples.empty else []
                    sample_choice = st.selectbox("Для образца ID:", options=sample_opts)
                    param = st.text_input("Параметр (pH, влажность, aw)")
                    val = st.text_input("Значение")
                    if st.form_submit_button("Сохранить измерение"):
                        if sample_choice:
                            new_id = measurements['id'].max() + 1 if not measurements.empty and measurements[
                                'id'].notna().any() else 1
                            new_row = {"id": new_id, "sample_id": sample_choice, "parameter": param,
                                       "actual_value": val}
                            append_row_csv(MEASUREMENTS_CSV, new_row,
                                           cols_order=["id", "sample_id", "parameter", "unit", "actual_value",
                                                       "method"])
                            st.success(f"Измерение для образца #{sample_choice} добавлено!")
                            st.rerun()
                        else:
                            st.warning("Сначала добавьте хотя бы один образец.")

# ==========================================================================
# СТРАНИЦА 3: АНАЛИЗ СТАБИЛЬНОСТИ
# ==========================================================================
elif page == "Анализ стабильности":
    st.title("🧪 Анализ стабильности продукта")

    with st.expander("ℹ️ Зачем это нужно и какой результат лучший?"):
        st.write("""
            **Зачем это нужно?**
            Этот анализ помогает найти оптимальное **давление** и **вязкость** сырья для получения качественного продукта. Стабильность — это ключевой показатель, который говорит о том, насколько хорошо продукт будет сохранять свою структуру, не выделяя жир или влагу.

            **Какой результат лучший?**
            Цель — получить **максимально высокий** `Stability Index`. Чем выше это значение, тем более качественным, упругим и однородным будет конечный продукт. Используйте калькулятор ниже, чтобы найти "пик" на 3D-графике в пределах реальных технологических параметров.
        """)

    if all_meat_data and 'T2' in all_meat_data:
        df_stability = all_meat_data['T2']
        required_cols = ['Pressure_bar', 'Viscosity_mPa_s', 'StabilityIndex']
        if not all(col in df_stability.columns for col in required_cols):
            st.error(f"Ошибка в листе 'T2': отсутствуют столбцы {required_cols}.")
        else:
            st.subheader("Интерактивный калькулятор стабильности")
            col1, col2 = st.columns(2)
            pressure_val = col1.slider("Давление (Pressure, bar)", 0.0, 5.0, 2.5, 0.1)
            viscosity_val = col2.slider("Вязкость (Viscosity, mPa·s)", 0.0, 5.0, 2.5, 0.1)
            predicted_stability = calculate_stability(pressure_val, viscosity_val)
            st.metric(label="Прогнозируемый StabilityIndex", value=f"{predicted_stability:.4f}")

            st.markdown("---")
            st.subheader("Визуализация данных и модели")
            X1 = df_stability['Pressure_bar'].values;
            X2 = df_stability['Viscosity_mPa_s'].values;
            Y = df_stability['StabilityIndex'].values
            Z_model = calculate_stability(X1, X2)

            fig1, ax1 = plt.subplots();
            ax1.plot(Y, 'o-', label='Эксперимент');
            ax1.plot(Z_model, 's--', label='Модель');
            ax1.set_xlabel('BatchID');
            ax1.set_ylabel('Stability Index');
            ax1.set_title('Сравнение данных');
            ax1.legend();
            ax1.grid(True)

            X1g, X2g = np.meshgrid(np.linspace(min(X1), max(X1), 30), np.linspace(min(X2), max(X2), 30))
            Zg = calculate_stability(X1g, X2g)
            fig2 = plt.figure();
            ax2 = fig2.add_subplot(111, projection='3d');
            surf = ax2.plot_surface(X1g, X2g, Zg, cmap='turbo');
            ax2.set_xlabel('Pressure');
            ax2.set_ylabel('Viscosity');
            ax2.set_zlabel('Stability');
            ax2.set_title('3D поверхность отклика');
            fig2.colorbar(surf)

            col1_viz, col2_viz = st.columns(2)
            with col1_viz:
                st.pyplot(fig1)
            with col2_viz:
                st.pyplot(fig2)
    else:
        st.error("Данные для анализа стабильности (лист 'T2') не загружены.")

# ==========================================================================
# ==========================================================================
# СТРАНИЦА 4: МОДЕЛИРОВАНИЕ PH
# ==========================================================================
elif page == "Моделирование pH":
    st.title("🌡️ Моделирование pH в процессе посола")

    with st.expander("ℹ️ Зачем это нужно и какой результат лучший?"):
        st.write("""
            **Зачем это нужно?**
            Контроль pH (кислотности) — важнейший этап для обеспечения **безопасности** продукта и формирования правильного **вкуса и аромата**. Эта модель позволяет спрогнозировать, сколько времени потребуется для достижения нужного уровня pH.

            **Какой результат лучший?**
            Цель — найти время, необходимое для достижения целевого диапазона **pH от 4.8 до 5.3**.
        """)

    if df_ph is not None:
        required_cols = ['CuringTime_h', 'pH']
        if not all(col in df_ph.columns for col in required_cols):
            st.error(f"Ошибка в 'opyty.xlsx': отсутствуют столбцы {required_cols}.")
        else:
            time_h, ph_obs = df_ph['CuringTime_h'].values, df_ph['pH'].values

            model_func, ph_pred, rmse, r2, coeffs = get_ph_model(time_h, ph_obs)

            if model_func is None:
                st.warning("Недостаточно данных для построения модели pH.")
            else:
                st.markdown("---")
                st.subheader("Интерактивный прогноз pH (Логарифмическая модель)")
                st.markdown(f"**Модель: pH = {coeffs[0]:.4f} * log(t) + {coeffs[1]:.4f}**")

                col1, col2 = st.columns(2)
                col1.metric("R² (точность)", f"{r2:.3f}")
                col2.metric("RMSE (ошибка)", f"{rmse:.4f}")

                time_val = st.slider("Время посола (CuringTime, ч)",
                                     min_value=1.0,
                                     max_value=72.0,
                                     value=24.0)

                predicted_ph = model_func(time_val)
                st.metric(label=f"Прогнозируемый pH для {time_val:.1f} ч", value=f"{predicted_ph:.4f}")

                st.markdown("---")
                st.subheader("✅ Автоматический расчет оптимального времени")
                search_times = np.arange(1.0, 72.0, 0.1)
                predicted_phs = model_func(search_times)
                optimal_mask = (predicted_phs >= 4.8) & (predicted_phs <= 5.3)
                optimal_times = search_times[optimal_mask]

                if optimal_times.size > 0:
                    start_time = optimal_times.min()
                    end_time = optimal_times.max()
                    st.success(
                        f"Модель прогнозирует, что продукт будет находиться в целевом диапазоне pH (4.8-5.3) примерно **с {start_time:.1f} по {end_time:.1f} час**.")
                else:
                    st.warning(
                        "В пределах заданного времени (до 72ч) модель не прогнозирует попадания в целевой диапазон pH 4.8-5.3.")

                st.markdown("---")
                st.subheader("Визуализация данных и новой модели")
                valid_indices = ~np.isnan(time_h) & ~np.isnan(ph_obs) & (time_h > 0)
                fig1, ax1 = plt.subplots()

                if optimal_times.size > 0:
                    ax1.axvspan(start_time, end_time, color='green', alpha=0.2, label='Оптимальная зона (pH 4.8-5.3)')

                ax1.scatter(time_h[valid_indices], ph_obs[valid_indices], s=80, c='b', label='Эксперимент')
                t_smooth = np.linspace(min(time_h[valid_indices]), max(time_h[valid_indices]), 200)
                ph_model_smooth = model_func(t_smooth)
                ax1.plot(t_smooth, ph_model_smooth, 'r-', linewidth=2, label='Новая модель')
                ax1.set_xlabel('Время посола, ч');
                ax1.set_ylabel('pH');
                ax1.set_title('Зависимость pH от времени');
                ax1.legend();
                ax1.grid(True)

                fig2, ax2 = plt.subplots();
                ax2.scatter(ph_obs[valid_indices], ph_pred, s=80, c='b');
                ax2.plot([min(ph_obs[valid_indices]), max(ph_obs[valid_indices])],
                         [min(ph_obs[valid_indices]), max(ph_obs[valid_indices])], 'k--');
                ax2.set_xlabel('Наблюдаемый pH');
                ax2.set_ylabel('Предсказанный pH');
                ax2.set_title('Наблюдаемый vs Предсказанный');
                ax2.grid(True)

                col1_viz, col2_viz = st.columns(2)
                with col1_viz:
                    st.pyplot(fig1)
                with col2_viz:
                    st.pyplot(fig2)
    else:
        st.error("Данные для моделирования pH ('opyty.xlsx') не загружены.")
# ==========================================================================
# СТРАНИЦА 5: ИССЛЕДОВАНИЕ ДАННЫХ
# ==========================================================================
elif page == "Исследование данных":
    st.title("🗂️ Исследование исходных данных")
    st.write("Выберите таблицу для просмотра.")

    if all_meat_data:
        available_tables = list(all_meat_data.keys())
        if df_ph is not None:
            available_tables.append('opyty.xlsx')

        choice = st.selectbox("Выберите данные:", available_tables)

        st.markdown(f"**Просмотр данных из: `{choice}`**")

        if choice == 'opyty.xlsx':
            if df_ph is not None: st.dataframe(df_ph)
        else:
            st.dataframe(all_meat_data[choice])
    else:
        st.warning("Не удалось загрузить данные для просмотра.")
#streamlit run app.py