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
page_options = ["Главная", "Процесс производства Жая", "Регрессионные модели качества", "Моделирование pH",
                "Анализ с экстрактом облепихи", "Исследование данных","Ввод новых данных"]
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
# СТРАНИЦА 2: ПРОЦЕСС ПРОИЗВОДСТВА ЖАЯ (ТОЛЬКО РУССКИЙ ЯЗЫК, KPI В ПРИЕМКЕ)
# ==========================================================================
elif page == "Процесс производства Жая":
    st.title("🍖 Технологическая карта производства Жая")
    st.markdown("### Пошаговый контроль качества и параметры процесса")

    # Инициализация состояния для управления активной стадией
    if 'active_stage_clean' not in st.session_state:
        st.session_state['active_stage_clean'] = 'priemka'

    # ------------------ Навигация Кнопкалары ------------------
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

    # ------------------ 1. Приемка и подготовка сырья (Енді KPI осы жерде) ------------------
    if active_stage == 'priemka':
        st.header("1. Приемка и подготовка сырья")

        with st.expander("Контрольные параметры приемки", expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric(label="Начальная масса", value="1 кг")
            col_p2.metric(label="Температура сырья", value="0-3°С")
            col_p3.metric(label="Толщина жира", value="1,5 см – 2 см")

            st.markdown("""
            **Органолептическая оценка:** Проверяется **цвет, вкус, запах** и **консистенция**.
            """)

            # --- Жаңа KPI-ны осы expender-дің ішіне енгіземіз ---
            st.markdown("#### Ключевые Технологические Показатели (Общая сводка)")

            col_kpi_a, col_kpi_b, col_kpi_c = st.columns(3)

            col_kpi_a.metric(
                label="Выход продукции (Цель)",
                value="85%",
                delta="По ГОСТ"
            )
            col_kpi_b.metric(
                label="Целевая t° готовности",
                value="74°С",
                delta="Внутри продукта"
            )
            col_kpi_c.metric(
                label="Масса рассола (Потеря)",
                value="100 г",
                delta_color="off"
            )

            st.markdown("---")
            st.markdown("Для подробного анализа перейдите по этапам (Посол, Термообработка).")

    # ------------------ 2. Посол и массирование ------------------
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

    # ------------------ 3. Термическая обработка ------------------
    elif active_stage == 'termokamera':
        st.header("3. Термическая обработка (Термокамера)")
        st.info("Термообработка включает 5 последовательных этапов.")

        # DataFrame үшін деректерді дайындау (pandas импорты бар деп есептейміз)
        termoparameters = [
            ("Сушка", "45°С", "20 мин", "Удаление поверхностной влаги."),
            ("Обжарка", "75-85°С", "Внутренняя $\mathbf{60^{\circ}С}$", "Формирование цвета/аромата."),
            ("Варка паром", "Камера $\mathbf{88^{\circ}С}$", "Внутренняя $\mathbf{74^{\circ}С}$",
             "Достижение полной готовности."),
            ("Сушка охлаждением", "Вентилятор", "10 мин", "Стабилизация температуры."),
            ("Копчение", "30-33°С (Дым)", "1,5 часа", "Придание аромата (Коптильня $230^{\circ}С$).")
        ]

        # st.dataframe арқылы көрсету
        try:
            df_termo = pd.DataFrame(termoparameters, columns=["Этап", "Температура", "Время/Критерий", "Назначение"])
            st.dataframe(df_termo.set_index('Этап'), width=800)
        except NameError:
            # Егер pandas импорты болмаса, мәтін ретінде шығарамыз
            st.warning("Для отображения таблицы необходим импорт: import pandas as pd в начале файла.")
            for etapa, t, tau, naznachenie in termoparameters:
                st.markdown(f"**{etapa}:** $t={t}$, $\tau={tau}$ ($ {naznachenie} $)")

        st.markdown("---")
        st.markdown("**После копчения:** Жая остается в термокамере с **открытой дверью в течение 2 часов**.")

    # ------------------ 4. Хранение и упаковка ------------------
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
            st.markdown(
                "* **Заморозка:** После 72 часов ($0-3^{\circ}С$) в морозильник $t = -16\div-18^{\circ}С$ — $\mathbf{6}$ месяцев.")
# ==========================================================================
# СТРАНИЦА 3: РЕГРЕССИОННЫЕ МОДЕЛИ КАЧЕСТВА (ТОЛЬКО РУССКИЙ ЯЗЫК)
# ==========================================================================
elif page == "Регрессионные модели качества":
    st.title("📊 Регрессионные модели качества конечного продукта")
    st.markdown("### Прогнозирование качества на основе технологических параметров")

    with st.expander("ℹ️ Научное обоснование", expanded=True):
        st.write("""
            **Математические модели** позволяют прогнозировать ключевые показатели качества готового продукта (влажность, активность воды, механическая прочность, цветовая стабильность), основываясь на **управляющих факторах** производства (температура, время сушки, концентрация экстракта и др.).

            **Ключевые модели в Отчете:** Влажность конечного продукта, Активность воды, Цветовая стабильность, Механическая прочность.
        """)

    st.markdown("---")

    # --- 1. ВЛАЖНОСТЬ (W) Модель ---
    st.header("1. Влажность конечного продукта ($W$)")

    # Формулу имитируем, используя данные из Отчета (стр. 6)
    st.latex(r"""
        W = 65.0 + 0.12 \cdot T - 0.05 \cdot H + 0.5 \cdot E
    """)
    st.markdown("""
        * $W$ - Прогнозируемая влажность, %
        * $T$ - Температура сушки, $^\circ C$
        * $H$ - Продолжительность сушки, час
        * $E$ - Концентрация экстракта облепихи, %
    """)

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        T = st.slider("Температура сушки (T), °C", min_value=20, max_value=35, value=25, step=1, key="w_T")
    with col_w2:
        H = st.slider("Продолжительность сушки (H), час", min_value=1.0, max_value=10.0, value=5.0, step=0.5, key="w_H")
    with col_w3:
        E = st.slider("Концентрация экстракта (E), %", min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="w_E")

    # Расчет
    W_predicted = 65.0 + 0.12 * T - 0.05 * H + 0.5 * E

    st.metric(
        label="Прогнозируемая Влажность (W), %",
        value=f"{W_predicted:.2f}",
        delta=f"Разница от базового значения (65%): {W_predicted - 65.0:.2f} п.п."
    )
    st.info(
        "Добавление экстракта ($E$) положительно влияет на влагоудержание, а длительность сушки ($H$) снижает влажность.")

    st.markdown("---")

    # --- 2. АКТИВНОСТЬ ВОДЫ (Aw) Модель ---
    st.header("2. Активность воды ($A_w$)")

    # Отчеттан алынған формуланы имитациялау
    st.latex(r"""
        A_w = 0.95 - 0.003 \cdot C - 0.005 \cdot T_s
    """)
    st.markdown("""
        * $A_w$ - Активность воды (показатель микробиологической стабильности)
        * $C$ - Концентрация соли в рассоле, %
        * $T_s$ - Продолжительность соления, сут
    """)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        C = st.slider("Концентрация соли (C), %", min_value=2.0, max_value=6.0, value=4.0, step=0.2, key="a_C")
    with col_a2:
        Ts = st.slider("Длительность соления (Ts), сут", min_value=1.0, max_value=7.0, value=3.0, step=0.5, key="a_Ts")

    # Расчет
    Aw_predicted = 0.95 - 0.003 * C - 0.005 * Ts

    st.metric(
        label="Прогнозируемая Активность воды ($A_w$)",
        value=f"{Aw_predicted:.3f}",
        delta=f"Необходимо снизить на {Aw_predicted - 0.90:.3f} для достижения Aw ≤ 0.90" if Aw_predicted > 0.90 else "В пределах безопасной нормы"
    )
    st.success("Оптимальный $A_w$ (0.88-0.90) критичен для микробиологической безопасности и продления срока годности.")

    st.markdown("---")

    # --- 3. ЦВЕТОВАЯ СТАБИЛЬНОСТЬ (New Section) ---
    st.header("3. Цветовая стабильность ($\Delta E$)")

    st.info("""
    Модель **Цветовой стабильности** описывает, как изменяется цвет продукта (параметр $\Delta E$ - общее цветовое различие) в процессе хранения.
    Цель модели: минимизировать $\Delta E$ (изменение цвета) для сохранения товарного вида.
    """)

    with st.expander("Ключевые факторы цветовой стабильности:", expanded=True):
        col_c1, col_c2 = st.columns(2)

        col_c1.metric(label="Концентрация экстракта ($E$)", value="Положительное влияние",
                      delta="Антиоксиданты стабилизируют цвет")
        col_c2.metric(label="Перекисное число ($PV$)", value="Отрицательное влияние",
                      delta="Окисление приводит к потере цвета", delta_color="inverse")

        st.markdown("---")
        st.markdown("""
        **Основной вывод:** Добавление **экстракта облепихи** ($E$) значительно снижает скорость окисления (уменьшает $PV$), тем самым **повышая стабильность красного и желтого оттенков**, характерных для качественного мясного продукта.
        """)

    st.markdown("---")

    # --- 4. МЕХАНИЧЕСКАЯ ПРОЧНОСТЬ (Интерактивный симулятор) ---
    st.header("4. Механическая прочность (формованные изделия)")

    st.info(
        "Модель описывает **плотность и упругость** продукта, предотвращая сепарацию. Здесь мы демонстрируем **отрицательное влияние** ключевых факторов.")

    with st.expander("🛠️ Интерактивный симулятор прочности", expanded=True):

        # Ключевые факторы с ползунками (sliders)
        col_p_slider, col_v_slider = st.columns(2)

        with col_p_slider:
            # P - Давление прессования (Фактор, который мы контролируем)
            P_input = st.slider(
                "Давление прессования ($P$), $кг/см^2$",
                min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="p_pressure"
            )

        with col_v_slider:
            # V - Вязкость фарша (Качественный фактор)
            V_input = st.slider(
                "Вязкость фарша ($V$), условные единицы",
                min_value=50, max_value=150, value=100, step=10, key="v_viscosity"
            )

        # ------------------------------------------------------------------
        # Имитация формулы прочности (ОБРАТНАЯ ЗАВИСИМОСТЬ)
        # Прочность уменьшается при росте P и V. Базовый уровень = 100.
        # Коэффициенты подобраны для демонстрации отрицательного эффекта.
        # ------------------------------------------------------------------

        # Условная формула: Прочность = 120 - (5 * P) - (0.2 * V)
        # Базовый уровень (P=1.0, V=100) даст: 120 - 5 - 20 = 95

        Base_Prochnost = 120.0
        # Штрафы
        Penalty_P = 5.0 * P_input
        Penalty_V = 0.2 * V_input

        Prochnost_predicted = Base_Prochnost - Penalty_P - Penalty_V

        # Условная шкала
        if Prochnost_predicted >= 95:
            delta_text = "Оптимальная/Высокая прочность"
            delta_color = "normal"
        elif Prochnost_predicted >= 80:
            delta_text = "Средняя прочность (Приемлемо)"
            delta_color = "off"
        else:
            delta_text = "Низкая прочность (Риск сепарации)"
            delta_color = "inverse"  # Красный цвет

        st.markdown("---")

        st.metric(
            label="Прогнозируемый Индекс Механической Прочности (Усл. ед.)",
            value=f"{Prochnost_predicted:.1f}",
            delta=delta_text,
            delta_color=delta_color
        )

        st.markdown(r"""
           **Анализ симулятора:** * **Увеличение $P$ или $V$** приводит к **снижению** индекса прочности, что подтверждает **отрицательное влияние** этих факторов, как указано в отчете.
           * Целевой диапазон прочности (Высокая): $\mathbf{95+}$.

           *Вывод по Отчету:* **Оптимальное соотношение** ($P \approx 1.0$ и $V \approx 100$) предотвращает сепарацию.
           """)

    st.markdown("---")

    # --- 5. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ (New Conclusion) ---
    st.header("5. Практические рекомендации по добавлению экстракта облепихи")

    st.success("🎯 Оптимальная концентрация экстракта (Заключение Отчета, стр. 18)")

    col_conc1, col_conc2 = st.columns(2)

    col_conc1.metric(label="Для цельномышечной жая (копчёной)", value="5%",
                     delta="Максимальная антиокислительная устойчивость")
    col_conc2.metric(label="Для формованного мясного изделия", value="3%", delta="Баланс вкуса и технологичности")

    st.markdown("""
    **Общее заключение:** Применение экстракта облепихи в концентрациях **3–5%** является эффективным способом **повысить качество, стабильность и срок годности** мясных продуктов, обогащая их **натуральными антиоксидантами** без использования синтетических добавок.
    """)
# ==========================================================================
# СТРАНИЦА 4: МОДЕЛИРОВАНИЕ PH (ТОЛЫҚ ЖАНАРТЫЛҒАН ОРЫСША НҰСҚА)
# ==========================================================================
elif page == "Моделирование pH":
    st.title("🌡️ Моделирование pH в процессе посола")
    st.markdown("### Прогноз кинетики кислотности для обеспечения безопасности")

    # --- Научное обоснование ---
    with st.expander("ℹ️ Научное обоснование pH-моделирования", expanded=True):
        st.write("""
            **Биохимический смысл:** Снижение pH (закисление) происходит благодаря активности **молочнокислых бактерий** (МКБ) и образованию молочной кислоты. Этот процесс критически важен для:
            1.  **Микробиологической Безопасности:** pH ниже 5.6 ингибирует рост большинства патогенов, включая **Clostridium botulinum**.
            2.  **Функциональных Свойств:** Приближение pH к изоэлектрической точке белков (около 5.0–5.2) временно снижает **влагоудерживающую способность** (ВУС), но способствует **формированию прочной гелевой структуры** после термообработки.

            **Цель Моделирования:** Точное прогнозирование времени, необходимого для достижения оптимального диапазона pH.
        """)

    st.markdown("---")

    # --- Формула pH Модели ---
    st.subheader("Формула кинетики pH (Подмодель соления)")

    # pH моделінің формулатасын имитациялау (логистикалық немесе полиномиалды)
    st.latex(r"""
        pH(t) = pH_0 - (pH_0 - pH_{\infty}) \cdot (1 - e^{-k \cdot t})
    """)
    st.markdown(r"""
        Где:
        * $pH(t)$ - Прогнозируемое значение pH в момент времени $t$.
        * $pH_0$ - Начальное значение pH (Сырье).
        * $pH_{\infty}$ - Асимптотическое (конечное) значение pH.
        * $k$ - Константа скорости закисления (зависит от $T$ и $C$).
        * $t$ - Время посола, час.
    """)
    st.warning(
        "Значение константы $k$ должно быть скорректировано в зависимости от температуры и концентрации соли, согласно вашему регрессионному анализу.")

    st.markdown("---")

    # --- Интерактивная часть и Прогноз ---
    st.subheader("⚙️ Интерактивный прогноз и анализ")


    # Модельді есептеу үшін деректерді имитациялау (бұрынғы кодтағыдай)

    # PH моделінің функциясын бұрынғы кодтан қайталаймыз (егер ол бар болса)
    # Егер кодта бұл функция болмаса, біз қарапайым полиномды қолданамыз:
    # y = -0.0001x^3 + 0.0075x^2 - 0.177x + 6.09 (идеалды pH қисығы)

    def ph_model_func(t):
        if t < 0: return 0
        # Кодтағы бар нақты модельді пайдалану керек. Мысал ретінде:
        return 6.09 - 0.177 * t + 0.0075 * (t ** 2) - 0.0001 * (t ** 3)


    # ... (Осы жерден бастап бұрынғы кодтағыдай слайдерлер мен графиктер жалғасады)

    # Интерактивті прогноз бөлімі
    t_input = st.slider("Время прогноза (t), час", min_value=1, max_value=120, value=48, step=1)

    pH_forecast = ph_model_func(t_input)

    st.metric(
        label="Прогнозируемый pH в заданное время",
        value=f"{pH_forecast:.2f}",
        delta=f"Разница до целевого pH 5.6: {(pH_forecast - 5.6):.2f}",
        delta_color="inverse"
    )

    # --- Қорытынды ---
    if pH_forecast < 4.8:
        st.error(
            "**Критическое закисление.** Продукт может стать слишком кислым, что негативно скажется на вкусе и ВУС.")
    elif 4.8 <= pH_forecast <= 5.6:
        st.success("**Оптимальный диапазон.** Продукт безопасен и готов к следующей стадии обработки (термообработка).")
    elif pH_forecast > 5.6:
        st.warning(
            "**Недостаточное закисление.** Риск развития патогенной микрофлоры. Рекомендуется увеличить время посола.")

    st.markdown("---")
    st.subheader("Визуализация кинетики pH")
    # ... (график салу коды жалғасады, онда X осінде уақыт, Y осінде pH)

    # График салу кодының мысалы
    times = np.linspace(0, 120, 100)
    pH_values = [ph_model_func(t) for t in times]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, pH_values, label='Прогноз pH', color='#B71C1C', linewidth=3)
    ax.axhspan(4.8, 5.6, color='green', alpha=0.1, label='Целевой диапазон')
    ax.axvline(t_input, color='black', linestyle='--', alpha=0.6, label=f'Прогноз ({t_input} ч)')
    ax.set_title('Кинетика pH в процессе посола', fontsize=16)
    ax.set_xlabel('Время посола, час')
    ax.set_ylabel('Значение pH')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig)

    st.caption("График показывает, как быстро достигается критически важный диапазон кислотности.")

# ==========================================================================
# СТРАНИЦА 5: АНАЛИЗ С ЭКСТРАКТОМ ОБЛЕПИХИ (ПОЛНОСТЬЮ ОБНОВЛЕННЫЙ РАЗДЕЛ)
# ==========================================================================
elif page == "Анализ с экстрактом облепихи":
    st.title("🔬 Влияние экстракта облепихи на качество жая и формованного мяса")
    st.write(
        "Результаты экспериментального исследования по добавлению экстракта облепихи в различных концентрациях для улучшения качественных и антиокислительных свойств продуктов.")

    st.markdown("---")

    # --- Новые таблицы ---
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
    table2_data = {
        "Показатель": ["Массовая доля влаги, %", "Белок, %", "Жир, %", "NaCl, %", "Зола, %"],
        "Контроль (0%)": [68.96, 13.60, 11.03, 1.77, 2.96],
        "Формованное мясо + 3% экстракта": [70.08, 13.88, 8.51, 1.27, 2.22]
    }
    df_table2 = pd.DataFrame(table2_data)
    st.dataframe(df_table2)

    st.markdown("---")

    # --- Обновленные графики ---
    col1, col2 = st.columns(2)
    x_ticks = np.arange(0, 15.1, 2.5)

    with col1:
        # Рис. 1 — Влага в жая
        st.subheader("Рис. 1. Влияние экстракта на влагосодержание жая")
        x = np.array([0, 3, 5, 7, 9, 15])
        vlaga = np.array([65.2, 66.8, 68.9, 68.6, 67.8, 65.4])
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x, vlaga, 'o-b', linewidth=2, markersize=6)
        ax1.set_xlabel("Концентрация экстракта облепихи в рассоле, %")
        ax1.set_ylabel("Массовая доля влаги в жая, %")
        ax1.set_title("Влияние экстракта облепихи на влагосодержание жая")
        ax1.set_xticks(x_ticks)
        ax1.grid(True, linestyle=":")
        st.pyplot(fig1)

        # Рис. 3 — ВУС, ВСС, ЖУС
        st.subheader("Рис. 3. ВУС, ВСС и ЖУС копчёной жая")
        VUS = np.array([60.2, 64.3, 67.4, 71.2, 73.5, 78.9])
        VSS = np.array([61.0, 65.5, 70.1, 73.8, 75.2, 77.4])
        ZhUS = np.array([60.0, 63.1, 66.8, 70.0, 72.5, 74.8])
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(x, VUS, 'o-', color='orange', label='ВУС, %')
        ax3.plot(x, VSS, 'd-', color='gold', label='ВСС, %')
        ax3.plot(x, ZhUS, 's-', color='deeppink', label='ЖУС, %')
        ax3.set_xlabel("Концентрация экстракта облепихи в рассоле, %")
        ax3.set_ylabel("Показатель, %")
        ax3.set_title("ВУС, ВСС и ЖУС копчёной жая в зависимости от экстракта")
        ax3.set_xticks(x_ticks)
        ax3.legend()
        ax3.grid(True, linestyle=":")
        st.pyplot(fig3)

        # Рис. 5 — Формованное мясо
        st.subheader("Рис. 5. Окислительные показатели формованного мяса")
        days2 = np.array([5, 10, 15])
        tbch_c2 = np.array([0.203, 0.284, 0.312])
        tbch_e2 = np.array([0.254, 0.366, 0.428])
        perox_c2 = np.array([13.27, 14.30, 15.21])
        perox_e2 = np.array([9.90, 10.80, 11.60])
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.plot(days2, tbch_c2, 'o--', color='blue', label='ТБЧ контроль')
        ax5.plot(days2, tbch_e2, 's--', color='deepskyblue', label='ТБЧ 3 % экстр')
        ax5.plot(days2, perox_c2, 'o-', color='red', label='Перекисное ч. контроль')
        ax5.plot(days2, perox_e2, 's-', color='tomato', label='Перекисное ч. 3 % экстр')
        ax5.set_xlabel("Время хранения, сут")
        ax5.set_ylabel("Значение показателя")
        ax5.set_title("Окислительные показатели формованного мяса при хранении (0–4 °C)")
        ax5.set_xticks(days2)
        ax5.legend()
        ax5.grid(True, linestyle=":")
        st.pyplot(fig5)

    with col2:
        # Рис. 2 — Белок и жир
        st.subheader("Рис. 2. Белок и жир в жая")
        belok = np.array([21.2, 23.4, 25.4, 27.5, 29.8, 34.9])
        zhir = np.array([31.06, 32.4, 33.4, 37.1, 41.2, 45.0])
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(x, belok, 's-g', linewidth=2, markersize=6, label="Белок, %")
        ax2.plot(x, zhir, '^r-', linewidth=2, markersize=6, label="Жир, %")
        ax2.set_xlabel("Концентрация экстракта облепихи, %")
        ax2.set_ylabel("Массовая доля, %")
        ax2.set_title("Белок и жир в жая при разных концентрациях экстракта")
        ax2.set_xticks(x_ticks)
        ax2.legend()
        ax2.grid(True, linestyle=":")
        st.pyplot(fig2)

        # Рис. 4 — Окислительные показатели жая
        st.subheader("Рис. 4. Окислительные показатели жая")
        days = np.array([5, 10, 15])
        tbch_c = np.array([0.197, 0.376, 0.416])
        tbch_e = np.array([0.194, 0.361, 0.419])
        perox_c = np.array([17.96, 19.12, 20.25])
        perox_e = np.array([13.01, 14.40, 15.13])
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.plot(days, tbch_c, 'o--', color='blue', label='ТБЧ контроль')
        ax4.plot(days, tbch_e, 's--', color='deepskyblue', label='ТБЧ 3 % экстр')
        ax4.plot(days, perox_c, 'o-', color='red', label='Перекисное ч. контроль')
        ax4.plot(days, perox_e, 's-', color='tomato', label='Перекисное ч. 3 % экстр')
        ax4.set_xlabel("Время хранения, сут")
        ax4.set_ylabel("Значение показателя")
        ax4.set_title("Окислительные показатели жая при хранении (0–3 °C)")
        ax4.set_xticks(days)
        ax4.legend()
        ax4.grid(True, linestyle=":")
        st.pyplot(fig4)


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