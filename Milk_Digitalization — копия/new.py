# app.py — Финальная объединенная версия с полиномиальной моделью pH, полным интерфейсом и новым разделом анализа
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
# CSV файлы для интерактивного интерфейса процесса
PRODUCTS_CSV = DATA_DIR / "Products.csv"
SAMPLES_CSV = DATA_DIR / "Samples.csv"
MEASUREMENTS_CSV = DATA_DIR / "Measurements.csv"


# ---------------------------
# Утилиты для чтения/записи CSV
# ---------------------------
# --- ИЗМЕНЕНИЕ: Более надежная функция чтения CSV ---
def safe_read_csv(path: Path):
    """
    Пытается прочитать CSV-файл, перебирая несколько кодировок и парсеров.
    """
    if not path.exists():
        return pd.DataFrame()

    encodings = ['utf-8-sig', 'utf-8', 'windows-1251', 'latin1']
    for enc in encodings:
        try:
            # Сначала пытаемся прочитать быстрым 'c' движком
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            # Если кодировка неверна, переходим к следующей
            continue
        except pd.errors.ParserError:
            # Если ошибка парсинга (как у вас), пробуем более гибкий 'python' движок
            try:
                st.warning(
                    f"Файл '{path.name}' имеет проблемы со структурой. Попытка чтения с помощью 'python' engine...")
                return pd.read_csv(path, encoding=enc, engine='python')
            except Exception as e:
                # Если и он не справился, проблема серьезная
                st.error(f"Не удалось прочитать файл '{path.name}' даже с 'python' engine. Ошибка: {e}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Произошла непредвиденная ошибка при чтении файла '{path.name}': {e}")
            return pd.DataFrame()

    # Если ни одна кодировка не подошла
    st.error(f"Не удалось определить кодировку и прочитать файл '{path.name}'.")
    return pd.DataFrame()


def append_row_csv(path: Path, row: dict, cols_order=None):
    df_new = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    if cols_order:
        for c in cols_order:
            if c not in df_new.columns: df_new[c] = ""
        df_new = df_new[cols_order]
    df_new.to_csv(path, mode='a', index=False, header=write_header, encoding='utf-8-sig')


# ---------------------------
# Функции для загрузки всех данных
# ---------------------------
@st.cache_data
def load_all_data():
    data_sheets = {}
    try:
        if MEAT_DATA_XLSX.exists():
            xls = pd.ExcelFile(MEAT_DATA_XLSX)
            for sheet_name in xls.sheet_names:
                # Используем openpyxl для чтения XLSX
                data_sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
    except Exception as e:
        st.warning(f"Не удалось прочитать '{MEAT_DATA_XLSX.name}': {e}")

    df_ph = None
    if OPYTY_XLSX.exists():
        try:
            # Используем openpyxl для чтения XLSX
            df_ph = pd.read_excel(OPYTY_XLSX, engine='openpyxl')
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
    """Использует полиномиальную модель 2-й степени: pH = at^2 + bt + c."""
    valid = ~np.isnan(time_h) & ~np.isnan(ph_obs)
    t, y = time_h[valid], ph_obs[valid]
    if len(t) < 3: return None, None, None, None
    coeffs = np.polyfit(t, y, 2)
    model_function = np.poly1d(coeffs)
    y_hat = model_function(t)
    r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
    rmse = np.sqrt(np.mean((y - y_hat) ** 2))
    return model_function, y_hat, rmse, r2


# ---------------------------
# Навигация и состояние сессии
# ---------------------------
st.sidebar.title("Меню платформы")
page_options = ["Главная", "Процесс производства Жая", "Анализ стабильности", "Моделирование pH",
                "Анализ с экстрактом облепихи", "Исследование данных"]
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
    st.info("Выберите раздел в меню слева, чтобы начать работу.")

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
            X1 = df_stability['Pressure_bar'].values
            X2 = df_stability['Viscosity_mPa_s'].values
            Y = df_stability['StabilityIndex'].values
            Z_model = calculate_stability(X1, X2)

            fig1, ax1 = plt.subplots()
            ax1.plot(Y, 'o-', label='Эксперимент')
            ax1.plot(Z_model, 's--', label='Модель')
            ax1.set_xlabel('BatchID')
            ax1.set_ylabel('Stability Index')
            ax1.set_title('Сравнение данных')
            ax1.legend()
            ax1.grid(True)

            X1g, X2g = np.meshgrid(np.linspace(min(X1), max(X1), 30), np.linspace(min(X2), max(X2), 30))
            Zg = calculate_stability(X1g, X2g)
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, projection='3d')
            surf = ax2.plot_surface(X1g, X2g, Zg, cmap='turbo')
            ax2.set_xlabel('Pressure')
            ax2.set_ylabel('Viscosity')
            ax2.set_zlabel('Stability')
            ax2.set_title('3D поверхность отклика')
            fig2.colorbar(surf)

            col1_viz, col2_viz = st.columns(2)
            with col1_viz:
                st.pyplot(fig1)
            with col2_viz:
                st.pyplot(fig2)
    else:
        st.error("Данные для анализа стабильности (лист 'T2') не загружены.")

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
            Цель — найти время, необходимое для достижения целевого диапазона **pH от 4.8 до 5.6**.
        """)

    if df_ph is not None:
        required_cols = ['CuringTime_h', 'pH']
        if not all(col in df_ph.columns for col in required_cols):
            st.error(f"Ошибка в 'opyty.xlsx': отсутствуют столбцы {required_cols}.")
        else:
            time_h, ph_obs = df_ph['CuringTime_h'].values, df_ph['pH'].values

            ph_model_func, ph_pred, rmse, r2 = get_ph_model(time_h, ph_obs)

            if ph_model_func is None:
                st.warning("Недостаточно данных для построения модели pH.")
            else:
                st.markdown("---")
                st.subheader("Интерактивный прогноз pH (Полиномиальная модель)")
                st.markdown(
                    f"**Модель: pH = {ph_model_func.coeffs[0]:.4f}*t² + {ph_model_func.coeffs[1]:.4f}*t + {ph_model_func.coeffs[2]:.4f}**")
                col1, col2 = st.columns(2)
                col1.metric("R² (точность)", f"{r2:.3f}")
                col2.metric("RMSE (ошибка)", f"{rmse:.4f}")

                time_val = st.slider("Время посола (CuringTime, ч)",
                                     min_value=float(np.nanmin(time_h)),
                                     max_value=float(np.nanmax(time_h)) * 1.5,
                                     value=float(np.nanmean(time_h[~np.isnan(time_h)])))

                predicted_ph = ph_model_func(time_val)
                st.metric(label=f"Прогнозируемый pH для {time_val:.1f} ч", value=f"{predicted_ph:.4f}")

                st.markdown("---")
                st.subheader("✅ Автоматический расчет оптимального времени")
                search_times = np.arange(1.0, 100.0, 0.1)
                predicted_phs = ph_model_func(search_times)
                optimal_mask = (predicted_phs >= 4.8) & (predicted_phs <= 5.6)
                optimal_times = search_times[optimal_mask]

                if optimal_times.size > 0:
                    start_time = optimal_times.min()
                    end_time = optimal_times.max()
                    st.success(
                        f"Модель прогнозирует, что продукт будет находиться в целевом диапазоне pH (4.8-5.6) примерно **с {start_time:.1f} по {end_time:.1f} час**.")
                else:
                    st.warning(
                        "В пределах заданного времени модель не прогнозирует попадания в целевой диапазон pH 4.8-5.3.")

                st.markdown("---")
                st.subheader("Визуализация данных и новой модели")
                valid_indices = ~np.isnan(time_h) & ~np.isnan(ph_obs)
                fig1, ax1 = plt.subplots()

                if optimal_times.size > 0:
                    ax1.axvspan(start_time, end_time, color='green', alpha=0.2, label='Оптимальная зона (pH 4.8-5.3)')

                ax1.scatter(time_h[valid_indices], ph_obs[valid_indices], s=80, c='b', label='Эксперимент')
                t_smooth = np.linspace(min(time_h[valid_indices]), max(time_h[valid_indices]), 200)
                ph_model_smooth = ph_model_func(t_smooth)
                ax1.plot(t_smooth, ph_model_smooth, 'r-', linewidth=2, label='Новая модель')
                ax1.set_xlabel('Время посола, ч')
                ax1.set_ylabel('pH')
                ax1.set_title('Зависимость pH от времени')
                ax1.legend()
                ax1.grid(True)

                fig2, ax2 = plt.subplots()
                ax2.scatter(ph_obs[valid_indices], ph_pred, s=80, c='b')
                ax2.plot([min(ph_obs[valid_indices]), max(ph_obs[valid_indices])],
                         [min(ph_obs[valid_indices]), max(ph_obs[valid_indices])], 'k--')
                ax2.set_xlabel('Наблюдаемый pH')
                ax2.set_ylabel('Предсказанный pH')
                ax2.set_title('Наблюдаемый vs Предсказанный')
                ax2.grid(True)

                col1_viz, col2_viz = st.columns(2)
                with col1_viz:
                    st.pyplot(fig1)
                with col2_viz:
                    st.pyplot(fig2)
    else:
        st.error("Данные для моделирования pH ('opyty.xlsx') не загружены.")

# ==========================================================================
# СТРАНИЦА 5: АНАЛИЗ С ЭКСТРАКТОМ ОБЛЕПИХИ (НОВЫЙ РАЗДЕЛ)
# ==========================================================================
elif page == "Анализ с экстрактом облепихи":
    st.title("🔬 Влияние экстракта облепихи на качество жая")
    st.write("Результаты экспериментального исследования по добавлению экстракта облепихи в различных концентрациях.")

    st.markdown("---")
    st.subheader("Таблица 1. Основные показатели копчёной жая (Контроль vs 5%)")
    data_t1 = {
        "Показатель": ["Влага, %", "Белок, %", "Жир, %", "ВУС, %", "ТБЧ, мг/кг"],
        "Контроль (0%)": [65.2, 21.2, 31.06, 60.2, 0.69],
        "Опыт (5%)": [67.8, 25.44, 33.4, 67.4, 0.96]
    }
    df_t1 = pd.DataFrame(data_t1)
    st.dataframe(df_t1)

    # --- Графики ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Рис. 1. Влияние на влагосодержание")
        x = np.array([0, 3, 5, 7, 9, 15])
        vlaga = np.array([65.2, 66.5, 67.8, 68.9, 67.8, 65.4])
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x, vlaga, 'o-b', linewidth=2, markersize=7)
        ax1.set_title("Влияние экстракта облепихи на влагосодержание жая")
        ax1.set_xlabel("Концентрация экстракта облепихи в рассоле, %")
        ax1.set_ylabel("Массовая доля влаги в жая, %")
        ax1.grid(True, linestyle=":")
        st.pyplot(fig1)

        st.subheader("Рис. 3. ВУС, ВСС, ЖУС копчёной жая")
        VUS = np.array([60.2, 64.3, 67.4, 71.2, 73.5, 78.9])
        VSS = np.array([61.0, 65.5, 70.1, 73.8, 75.2, 77.4])
        ZhUS = np.array([60.0, 63.1, 66.8, 70.0, 72.5, 74.8])
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(x, VUS, 'o-', color='orange', label='ВУС, %')
        ax3.plot(x, VSS, 'd-', color='gold', label='ВСС, %')
        ax3.plot(x, ZhUS, 's-', color='deeppink', label='ЖУС, %')
        ax3.set_title("ВУС, ВСС и ЖУС копчёной жая в зависимости от экстракта")
        ax3.set_xlabel("Концентрация экстракта облепихи, %")
        ax3.set_ylabel("Показатель, %")
        ax3.legend()
        ax3.grid(True, linestyle=":")
        st.pyplot(fig3)

        st.subheader("Рис. 5. Окислительные показатели формованного мяса")
        days2 = np.array([5, 10, 15])
        tbch_control2 = np.array([0.231, 0.284, 0.312])
        tbch_exp2 = np.array([0.254, 0.366, 0.428])
        perox_control2 = np.array([13.27, 14.30, 15.21])
        perox_exp2 = np.array([9.90, 10.80, 11.60])
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.plot(days2, tbch_control2, 'o--b', label='ТБЧ контроль')
        ax5.plot(days2, tbch_exp2, 's--c', label='ТБЧ 3% экстр')
        ax5.plot(days2, perox_control2, 'o-r', label='Перекисное ч. контроль')
        ax5.plot(days2, perox_exp2, 's-r', label='Перекисное ч. 3% экстр')
        ax5.set_title("Окислительные показатели формованного мяса при хранении (0–4°C)")
        ax5.set_xlabel("Время хранения, сут")
        ax5.set_ylabel("Значение показателя")
        ax5.legend()
        ax5.grid(True, linestyle=":")
        st.pyplot(fig5)

    with col2:
        st.subheader("Рис. 2. Белок и жир в жая")
        belok = np.array([21.2, 23.4, 25.4, 27.5, 29.8, 34.9])
        zhir = np.array([31.06, 32.4, 33.4, 37.1, 41.2, 45.0])
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(x, belok, 's-g', linewidth=2, markersize=7, label="Белок, %")
        ax2.plot(x, zhir, '^r-', linewidth=2, markersize=7, label="Жир, %")
        ax2.set_title("Белок и жир в жая при разных концентрациях экстракта")
        ax2.set_xlabel("Концентрация экстракта облепихи, %")
        ax2.set_ylabel("Массовая доля, %")
        ax2.legend()
        ax2.grid(True, linestyle=":")
        st.pyplot(fig2)

        st.subheader("Рис. 4. Окислительные показатели жая")
        days = np.array([5, 10, 15])
        tbch_control = np.array([0.322, 0.376, 0.416])
        tbch_exp = np.array([0.308, 0.361, 0.419])
        perox_control = np.array([17.96, 19.12, 20.25])
        perox_exp = np.array([13.01, 14.40, 15.13])
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.plot(days, tbch_control, 'o--b', label='ТБЧ контроль')
        ax4.plot(days, tbch_exp, 's--c', label='ТБЧ 3% экстр')
        ax4.plot(days, perox_control, 'o-r', label='Перекисное ч. контроль')
        ax4.plot(days, perox_exp, 's-r', label='Перекисное ч. 3% экстр')
        ax4.set_title("Окислительные показатели жая при хранении (0–3°C)")
        ax4.set_xlabel("Время хранения, сут")
        ax4.set_ylabel("Значение показателя")
        ax4.legend()
        ax4.grid(True, linestyle=":")
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("Сводная таблица эффекта экстракта облепихи:")
    summary = pd.DataFrame({
        "Показатель": ["Влага ↑", "Белок ↑", "Жир ↑", "ВУС ↑", "ТБЧ ↔", "Перекисное число ↓"],
        "Эффект экстракта облепихи": [
            "+2–3 п.п.", "+4 п.п.", "+2 п.п.",
            "+7 п.п.", "незначительное увеличение",
            "снижение на 25%"
        ]
    })
    st.dataframe(summary)

# ==========================================================================
# СТРАНИЦА 6: ИССЛЕДОВАНИЕ ДАННЫХ
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
#streamlit run new.py для запуска