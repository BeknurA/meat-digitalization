# ui.py
import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import numpy as np
from typing import Optional

# ---------------------------
# Полный мультиязычный словарь (RU / EN / KK)
# ---------------------------
LANG = {
    # --------------------------------------------------------------------------
    # Русский (ru)
    # --------------------------------------------------------------------------
    "ru": {
        # Общие элементы
        "title": "Цифровая платформа — Meat Digitalization",
        "full_title": "Цифровая платформа для мясного деликатеса Жая",
        "version_note": "Версия: интегрированная",
        "select_section": "Выберите раздел",
        "db_reset_confirm": "Вы уверены, что хотите удалить все измерения?",
        "train_button": "Обучить модель",
        "predict_button": "Сделать прогноз",
        "upload_csv": "Загрузить CSV/Excel",
        "no_data": "Нет данных для отображения",
        "save": "Сохранить",
        "saved": "Сохранено",
        "download": "Скачать",

        #
        "unit_kg": "кг",
        "unit_g": "г",
        "moisture_title": "Влага",
        "jaya_process_title": "Технологическая карта производства Жая",
        "jaya_process_subtitle": "Пошаговый контроль качества и параметры процесса (с учетом Экстракта и IoT)",
        "stage_priemka": "1. Приемка сырья 🥩",
        "stage_posol": "2. Посол (Экстракт) 🧂",
        "stage_termo": "3. Термическая обработка 🔥",
        "stage_upakovka": "4. Хранение 📦",
        # Навигация (Заменены на отдельные ключи для прямого доступа)
        "menu_home": "Главная",
        "menu_production_process": "Процесс производства Жая",
        "menu_regression_models": "Регрессионные модели качества",
        "menu_ph_modeling": "Моделирование pH",
        "menu_seabuckthorn_analysis": "Анализ с экстрактом облепихи",
        "menu_data_exploration": "Исследование данных",
        "menu_history_db": "История / DB",
        "menu_ml_train_predict": "ML: Train / Predict",
        "menu_new_data_input": "Ввод новых данных",

        # Главная страница
        "home_title": "🐎 Цифровая платформа для производства и моделирования Жая",
        "home_desc": "Интеллектуальные решения для оптимизации производства и контроля качества",
        "home_info": "Выберите раздел в меню слева, чтобы начать работу.",
        "stage_control_suffix": "4 шага контроля",
        "delta_production": "От сырья до упаковки",
        "delta_regression": "На основе параметров засола/сушки",
        "seabuckthorn_value": "Экстракт Облепихи",
        "delta_seabuckthorn": "Увеличение стабильности/срока годности",

        # Ключи для раздела "Основные научные достижения"
        "scientific_achievements": "🏆 Основные научные достижения",
        "wac_title": "Влагоудерживающая способность (ВУС)",
        "wac_subtitle": "Рост ВУС при 5% экстракта",
        "wac_note": "Против 60.2% в контроле.",
        "shelf_life_title": "Срок годности (Прогноз)",
        "shelf_life_subtitle": "Максимальный срок хранения при 0–5°С",
        "shelf_life_note": "На 30 дней дольше стандарта (30 суток).",
        "optimal_conc_title": "Оптимальная концентрация",
        "optimal_conc_subtitle": "Рекомендованная дозировка экстракта",
        "optimal_conc_note": "Баланс вкуса и стабильности.",

        # Ключи для раздела "Окислительная стабильность"
        "oxidation_stability_title": "🧪 Окислительная стабильность: снижение перекисного числа (ТБЧ)",
        "oxidation_goal": "**Цель:** Снизить окисление после 30 дней хранения.",
        "tba_reduction_text": "снижение ТБЧ",
        "oxidation_success": "Высокая антиокислительная устойчивость продукта достигнута.",
        "tba_caption": "Снижение ТБЧ с",
        "tba_caption_to": "до",
        "tba_caption_control": "контроль",
        "tba_caption_extract": "(5%экстракт)",
        "mg_per_kg": "мг/кг",
        "day_in_lang": "суток",
        # Процесс производства
        "prod_title": "🍖 Технологическая карта производства Жая",
        "prod_subtitle": "Пошаговый контроль качества и параметры процесса",
        "stage_1": "1. Приемка сырья 🥩",
        "stage_2": "2. Посол и массирование 🧂",
        "stage_3": "3. Термическая обработка 🔥",
        "stage_4": "4. Хранение и упаковка 📦",
        "stage_priemka_header": "1. Приемка сырья 🥩",
        "stage_priemka_expander": "Контрольные параметры приемки",
        "metric_mass": "Начальная масса",
        "metric_temp": "Температура сырья",
        "metric_ph": "Начальный pH",
        "metric_yield": "Выход продукции",
        "metric_target_temp": "Целевая t° готовности",
        "metric_brine_loss": "Масса рассола (Потеря)",
        "tech_params_title": "Ключевые Технологические Показатели",
        "delta_gost": "По ГОСТу",
        "delta_inner": "Внутри продукта",
        "help_ph": "Важен для прогноза созревания",
        "help_temp": "Контроль с помощью IoT сенсоров в камере.",
        "digital_control_tip": "💡 Цифровой контроль: Автоматическая запись массы и температуры сырья.",
        "stage_posol_header": "2. Посол, Экстракт облепихи и Массирование🧂",
        "stage_posol_expander1": "Подготовка рассола и шприцевание",
        "stage_posol_markdown1": "Состав рассола: 4,5 л H₂O + 250 г NaCl + 0,8 мг NaNO₂.\n\n🌿 **Внедрение Экстракта Облепихи (Ключевой шаг)**\nОптимальная концентрация: 3% - 5% от массы рассола.\nРекомендация: Для цельномышечной Жая предпочтительно 5% (для максимальной антиокислительной защиты).\nФункция: Экстракт улучшает влагоудерживающую способность и действует как натуральный антиоксидант.\nТемпература рассола: 16°С\nУкладка в рассол: τ=72 часа, t=0−3°С. Давление P=1200 г–1250 г на 1000 г.",
        "stage_posol_expander2": "Контроль и Мониторинг",
        "stage_posol_markdown2": "* **Контроль соли:** Использование цифрового солемера (Солемер / Ареометр) для проверки концентрации extNaCl.\n* **Контроль pH:** Ежедневный замер pH в рассоле для отслеживания динамики созревания (см. раздел \"Моделирование pH\").",
        "iot_monitoring_desc": "🌡️ **IoT-Мониторинг:**\n\n* **Датчики:** Использование беспроводных термодатчиков (IoT-зонд) внутри продукта для постоянного контроля достижения 74°С.\n\n* **Управляющее воздействие:** Автоматическое отключение/переключение режима камеры при достижении заданной внутренней температуры.",
        "stage_termo_header": "3. Термическая обработка (IoT-контроль) 🔥",
        "stage_termo_info": "Термообработка включает 5 последовательных этапов. Критическая точка: внутренняя 74∘С",
        "stage1_title": "1. Приемка и подготовка сырья",
        "stage1_params": "Контрольные параметры приемки",
        "initial_mass": "Начальная масса",
        "raw_temp": "Температура сырья",
        "fat_thickness": "Толщина жира",
        "kpi_title": "Ключевые Технологические Показатели (Общая сводка)",
        "yield_target": "Выход продукции (Цель)",
        "target_temp": "Целевая t° готовности",
        "brine_loss": "Масса рассола (Потеря)",

        "stage2_title": "2. Посол, Шприцевание и Массирование",
        "brine_prep": "Подготовка рассола и шприцевание",
        "brine_composition": "Состав рассола",
        "brine_temp": "Температура рассола",
        "injection": "Шприцевание",
        "massage_params": "Параметры массирования",
        "total_duration": "Общая длительность",
        "working_pressure": "Рабочее давление",

        "stage3_title": "3. Термическая обработка (Термокамера)",
        "stage3_info": "Термообработка включает 5 последовательных этапов.",
        "drying": "Сушка",
        "roasting": "Обжарка",
        "steam_cooking": "Варка паром",
        "cooling": "Сушка охлаждением",
        "smoking": "Копчение",
        "col_stage": "Этап",
        "col_temp": "Температура (°C)",
        "col_time": "Время/критерий",
        "col_purpose": "Назначение",
        "termo_drying": "Сушка",
        "termo_frying": "Обжарка",
        "termo_steam": "Варка паром",
        "termo_cool_dry": "Сушка охлаждением",
        "termo_smoke": "Копчение",
        "termo_drying_desc": "Удаление поверхностной влаги",
        "termo_frying_desc": "Формирование цвета/аромата",
        "termo_steam_desc": "Достижение полной готовности",
        "termo_cool_desc": "Стабилизация температуры",
        "termo_smoke_desc": "Придание аромата ",
        "stage_upakovka_expander": "Обвалка, Упаковка и Хранение (Ключевые параметры)",
        "shelf_life_comparison": "Сравнение сроков годности:",
        "shelf_life_standard": "Срок годности (Стандарт, без экстракта)",
        "shelf_life_extract": "Срок годности (С 5% экстракта)",
        "shelf_life_desc": "Ключевой фактор: Экстракт облепихи снижает перекисное число (TBC), что замедляет окисление жиров и позволяет увеличить срок годности.",
        "storage_tip": "🔬 Критический контроль при хранении: Активность воды (Aw) 0.88–0.90 и температура должна быть в диапазоне 0–5°С",
        "stage_upakovka_markdown1": "Охлаждение: В холодильной камере t=0–5°С — 12 часов. Упаковка: В вакуум-упаковочном автомате.",
        "shelf_life_std_value": "30 суток",
        "shelf_life_ext_value": "60 суток",
        "shelf_life_delta_value": "+30 дней",

        "stage4_title": "4. Обвалка, Упаковка и Хранение",
        "stage_upakovka_header": "4. Упаковка и Срок Годности",
        "deboning_packaging": "Обвалка и Упаковка",
        "shelf_life": "Сроки и Выход продукта",
        "storage_standard": "Стандарт",
        "storage_freeze": "Заморозка",

        # Регрессионные модели
        "regression_title": "📊 Регрессионные модели качества конечного продукта",
        "regression_subtitle": "Прогнозирование качества на основе технологических параметров",

        "reg_w_title": "1. Влажность конечного продукта ($W$)",
        "reg_w_T": "Температура сушки (T), °C",
        "reg_w_H": "Продолжительность сушки (H), час",
        "reg_w_E": "Концентрация экстракта (E), %",
        "reg_w_metric": "Прогнозируемая Влажность (W), %",
        "reg_w_delta": "Разница от базового значения (65%):",
        "reg_w_info": "Добавление экстракта ($E$) положительно влияет на влагоудержание.",

        "reg_aw_title": "2. Активность воды ($A_w$)",
        "reg_aw_C": "Концентрация соли (C), %",
        "reg_aw_Ts": "Длительность соления (Ts), сут",
        "reg_aw_metric": "Прогнозируемая Активность воды ($A_w$)",
        "reg_aw_delta_high": "Необходимо снизить для достижения Aw ≤ 0.90",
        "reg_aw_delta_ok": "В пределах безопасной нормы",
        "reg_aw_info": "Оптимальный $A_w$ (0.88–0.90) критичен для микробиологической безопасности.",

        "reg_color_title": "3. Цветовая стабильность ($\\Delta E$)",
        "reg_color_desc": "Моделирование изменения цвета в зависимости от экстракта и сушки.",
        "reg_color_E": "Концентрация экстракта (E), %",
        "reg_color_H": "Продолжительность сушки (H), час",
        "reg_color_metric": "Прогнозируемое изменение цвета ($\\Delta E$)",
        "reg_color_delta": "Оптимальное значение $\\Delta E < 2.0$",
        "reg_color_result_good": "✅ Высокая цветовая стабильность.",
        "reg_color_result_warn": "⚠️ Цвет приемлемый, но может быть небольшое потемнение.",
        "reg_color_result_bad": "❌ Значительное изменение цвета. Слишком долгая сушка.",

        "reg_tbc_title": "4. Окислительная стабильность (Перекисное число - TBC)",
        "reg_tbc_desc": "Прогноз степени окисления продукта после 30 дней хранения.",
        "reg_tbc_E": "Концентрация экстракта (E), %",
        "reg_tbc_S": "Концентрация соли (S), %",
        "reg_tbc_metric": "Прогнозируемое TBC через 30 дней, мг/кг",
        "reg_tbc_delta": "Чем ниже, тем лучше (Цель TBC < 1.5)",
        "reg_tbc_result_good": "✅ Отличная устойчивость, срок годности до 60 дней.",
        "reg_tbc_result_warn": "⚠️ Хорошая стабильность, срок до 45 дней.",
        "reg_tbc_result_bad": "❌ Высокий риск окисления, срок ≤ 30 дней.",

        "reg_strength_title": "5. Механическая прочность (формованные изделия)",
        "reg_strength_info": "Модель описывает плотность и упругость продукта.",
        "reg_strength_expander": "🛠️ Интерактивный симулятор прочности",
        "reg_strength_P": "Давление прессования (P), кг/см²",
        "reg_strength_V": "Вязкость фарша (V), усл. ед.",
        "reg_strength_metric": "Индекс механической стабильности",
        "reg_strength_result_good": "✅ Оптимальная/Высокая прочность. Хорошее формование.",
        "reg_strength_result_warn": "⚠️ Средняя прочность. Требуется внимание к давлению.",
        "reg_strength_result_bad": "❌ Низкая прочность. Риск деформации продукта.",

        # pH моделирование
        "ph_title": "🌡️ Моделирование pH в процессе посола",
        "ph_subtitle": "Прогноз кинетики кислотности для обеспечения безопасности",
        "ph_basis": "ℹ️ Научное обоснование pH-моделирования",
        "ph_formula_title": "Формула кинетики pH (Подмодель соления)",
        "ph_initial": "pH начальное (pH0)",
        "ph_final": "pH конечное (pH_inf)",
        "rate_constant": "Константа скорости (k)",
        "forecast_time": "Время прогноза (t), час",
        "predicted_ph": "Прогнозируемый pH в заданное время",
        "ph_kinetics": "Визуализация кинетики pH",

        "ph_critical_low": "**Критическое закисление.** Продукт слишком кислый.",
        "ph_optimal": "Оптимальный диапазон.",
        "ph_insufficient": "**Недостаточное закисление.**",
        "menu_ph_modeling": "🌡️ Моделирование pH",
        "ph_basis_text": '''
        **Биохимический смысл:** Снижение pH (повышение кислотности) в процессе созревания мяса — ключевой фактор, влияющий на подавление нежелательной микрофлоры и формирование правильной текстуры и вкуса. Это происходит в основном за счет ферментации гликогена до молочной кислоты ферментами мяса и стартовыми культурами.

        **Почему это важно:**
        1. **Безопасность:** Быстрое достижение pH ниже 5.6-5.8 ингибирует рост патогенных бактерий (E.coli, Salmonella).
        2. **Качество:** Оптимальный конечный pH (4.8-5.4) способствует влагоудержанию, нежности и формированию цвета.
        3. **Контроль:** Модель позволяет предсказать, достигнет ли продукт целевого pH при данных условиях (температура, соль, стартеры).
        ''',
        "ph_formula_desc": "Где: pH₀ — начальное значение, pH_inf — конечное, k — константа скорости.",
        "ph_formula_tip": "Значение k корректируется в зависимости от температуры и солёности.",
        "ph_forecast_title": "⚙️ Интерактивный прогноз и анализ",
        "delta_target_ph": "Разница до целевого pH 5.6:",
        "time_hours": "Время (часы)",
        "hours_short": "ч",
        "ph_plot_title": "Кинетика pH в процессе посола",

        # Анализ облепихи
        "seabuck_title": "🔬 Влияние экстракта облепихи на качество жая и формованного мяса",
        "seabuck_desc": "Результаты экспериментального исследования",
        "table1_title": "Таблица 1. Основные показатели копчёной жая (контроль и 5% экстракта)",
        "table2_title": "Таблица 2. Основные показатели формованного мясного продукта (контроль и 3% экстракта)",
        "indicator": "Показатель",
        "control": "Контроль (0%)",
        "with_extract_5": "Жая + 5% экстракта",
        "with_extract_3": "Формованное мясо + 3% экстракта",
        "menu_seabuck_analysis": "🔬 Анализ с экстрактом облепихи",
        "moisture": "Массовая доля влаги, %",
        "protein": "Белок, %",
        "fat": "Жир, %",
        "vus": "Влагоудерж. способность (ВУС), %",
        "tbch": "ТБЧ, мг/кг",
        "salt": "NaCl, %",
        "ash": "Зола, %",
        "fig1_title": "Рис. 1. Влияние экстракта на влагосодержание жая",
        "fig1_plot_title": "Влияние экстракта облепихи на влагосодержание жая",
        "fig2_title": "Рис. 2. Белок и жир в жая",
        "fig2_plot_title": "Белок и жир в жая",
        "fig3_title": "Рис. 3. ВУС, ВСС и ЖУС копчёной жая",
        "fig3_plot_title": "ВУС, ВСС и ЖУС копчёной жая",
        "fig4_title": "Рис. 4. Окислительные показатели жая",
        "fig4_plot_title": "Окислительные показатели жая",
        "fig5_title": "Рис. 5. Окислительные показатели формованного мяса",
        "fig5_plot_title": "Окислительные показатели формованного мяса",

        # Исследование данных
        "explore_title": "🗂️ Исследование исходных данных",
        "explore_desc": "Выберите таблицу для просмотра.",
        "select_data": "Выберите данные:",
        "viewing_data": "Просмотр данных из:",
        "data_empty_warning": "Данные не были загружены или пусты.",
        "data_load_error": "Не удалось загрузить данные для просмотра.",

        # История / БД
        "db_title": "📚 История измерений и база данных",
        "db_desc": "Здесь хранится история измерений (SQLite). Можно экспортировать, фильтровать и удалять записи.",
        "total_records": "Всего записей:",
        "history_empty": "История пуста",
        "export_all": "Экспортировать все в CSV",
        "clear_all": "Очистить все измерения",
        "confirm_clear": "Подтвердить очистку",
        "db_cleared": "База очищена. Перезагрузите страницу.",
        "ph_distribution": "pH распределение",
        "ph_over_time": "pH по времени (интерактивно)",

        # ML страница
        "menu_ml": "ML: Обучение / Прогноз",
        "ml_title": "🧠 ML: Обучение и прогнозирование pH",
        "ml_desc": "Загрузите CSV/Excel с колонкой 'pH' и признаками для обучения или загрузите CSV с признаками для предсказания.",
        "train_tab": "Обучение",
        "predict_tab": "Прогноз",
        "train_subtitle": "Обучение модели",
        "upload_train": "CSV/Excel для обучения (колонка pH)",
        "preview": "Превью:",
        "target_column": "Целевая колонка (pH) выберите:",
        "features": "Признаки (если пусто — будут взяты все числовые кроме цели)",
        "train_success": "Обучение прошло успешно",
        "train_error": "Ошибка обучения:",
        "no_data": "Нет данных.",
        "predict_subtitle": "Прогнозирование",
        "upload_predict": "CSV для предсказания (те же признаки)",
        "auto_features": "Автоматически выбранные числовые признаки:",
        "predict_results": "Результаты предсказания",
        "save_to_db": "Сохранить предсказания в базу (sample_name -> sample)",
        "saved_records": "Сохранено записей в БД:",

        # Ввод данных
        "menu_input": "Ввод данных",
        "input_title": "➕ Ввод новых данных о продукции",
        "input_subtitle": "Добавление нового производственного цикла в базу данных",
        "sheet": "лист",
        "batch_params": "Введите параметры нового производственного цикла",
        "batch_id": "Batch ID (автоматически)",
        "mass": "Масса партии (кг)",
        "initial_temp": "Начальная температура (°C)",
        "salt_content": "Содержание соли (%)",
        "moisture": "Влажность (%)",
        "starter_culture": "Стартерная культура (КОЕ/г)",
        "extract_content": "Концентрация экстракта (%)",
        "save_data": "💾 Сохранить данные",
        "batch_added": "✅ Новая партия успешно добавлена",
        "save_error": "❌ Ошибка при записи в файл:",
        "current_data": "📊 Текущие данные",
        "batchid_missing": "❌ В листе нет колонки 'BatchID'. Проверь структуру таблицы.",

        # pH статусы
        "ph_in_normal": "pH в норме",
        "ph_too_low": "pH слишком низкий",
        "ph_too_high": "pH слишком высокий",
        "anim_good": "✅ Всё в порядке",
        "anim_bad": "⚠️ Требуется корректировка",
    },
    "en": {
        # General Elements
        "title": "Meat Digitalization Platform",
        "full_title": "Digital Platform for the Meat Delicacy 'Zhaya'",
        "version_note": "Version: merged",
        "select_section": "Select a section",
        "db_reset_confirm": "Are you sure you want to delete all measurements?",
        "train_button": "Train model",
        "predict_button": "Predict",
        "upload_csv": "Upload CSV/Excel",
        "no_data": "No data to display",
        "save": "Save",
        "saved": "Saved",
        "download": "Download",

        # Navigation

        "menu_home": "Home",
        "menu_production_process": "Jaya Production Process",
        "menu_regression_models": "Quality Regression Models",
        "menu_ph_modeling": "pH Modeling",
        "menu_seabuckthorn_analysis": "Analysis with Sea Buckthorn Extract",
        "menu_data_exploration": "Data Exploration",
        "menu_history_db": "History / DB",
        "menu_ml_train_predict": "ML: Train / Predict",
        "menu_new_data_input": "New Data Input",
        "jaya_process_title": "Technological Map of Jaya Production",
        "jaya_process_subtitle": "Step-by-step quality control and process parameters (with Extract and IoT)",
        "stage_priemka": "1. Raw Material Acceptance 🥩",
        "stage_posol": "2. Salting (Extract) 🧂",
        "stage_termo": "3. Thermal Processing 🔥",
        "stage_upakovka": "4. Storage 📦",

        # Home Page
        "moisture_title": "moisture",
        "unit_kg": "kg",
        "unit_g": "g",
        "home_title": "🐎 Digital Platform for Jaya Production and Modeling",
        "home_desc": "Intelligent solutions for optimizing production and quality control",
        "home_info": "Select a section from the menu on the left to start.",
        "stage_control_suffix": "4 control stages",
        "delta_production": "From raw materials to packaging",
        "delta_regression": "Based on curing/drying parameters",
        "seabuckthorn_value": "Sea Buckthorn Extract",
        "delta_seabuckthorn": "Increase stability/shelf life",

        "scientific_achievements": "🏆 Key Scientific Achievements",
        "wac_title": "Water-Holding Capacity (WHC)",
        "wac_subtitle": "WHC increase with 5% extract",
        "wac_note": "Against 60.2% in control.",
        "shelf_life_title": "Shelf Life (Prediction)",
        "shelf_life_subtitle": "Maximum storage period at 0–5°С",
        "shelf_life_note": "30 days longer than standard (30 days).",
        "optimal_conc_title": "Optimal Concentration",
        "optimal_conc_subtitle": "Recommended extract dosage",
        "optimal_conc_note": "Balance of taste and stability.",

        "oxidation_stability_title": "🧪 Oxidation Stability: Peroxide Value (TBA) Reduction",
        "oxidation_goal": "**Goal:** Reduce oxidation after 30 days of storage.",
        "tba_reduction_text": "TBA reduction",
        "oxidation_success": "High antioxidant stability of the product achieved.",
        "tba_caption": "TBA reduction from",
        "tba_caption_to": "to",
        "tba_caption_control": "control",
        "tba_caption_extract": "(5% extract)",
        "mg_per_kg": "mg/kg",
        "day_in_lang": "days",
        # Production Process
        "prod_title": "🍖 Technological Map of Jaya Production",
        "prod_subtitle": "Step-by-step quality control and process parameters",
        "stage_1": "1. Raw Material Acceptance 🥩",
        "stage_2": "2. Salting and Massaging 🧂",
        "stage_3": "3. Thermal Processing 🔥",
        "stage_4": "4. Storage and Packaging 📦",
        "stage_priemka_header": "1. Raw Material Acceptance 🥩",
        "stage_priemka_expander": "Acceptance Control Parameters",
        "metric_mass": "Initial Mass",
        "metric_temp": "Raw Material Temperature",
        "metric_ph": "Initial pH",
        "metric_yield": "Product Yield",
        "metric_target_temp": "Target t° of Readiness",
        "metric_brine_loss": "Brine Mass (Loss)",
        "tech_params_title": "Key Technological Indicators",
        "delta_gost": "According to GOST",
        "delta_inner": "In product",
        "help_ph": "Important for ripening forecast",
        "help_temp": "Control using IoT sensors in the camera.",
        "digital_control_tip": "💡 Digital control: Automatic recording of the mass and temperature of raw materials.",
        "stage_posol_header": "2. Salting, Sea Buckthorn Extract and Massage🧂",
        "stage_posol_expander1": "Brine Preparation and Injection",
        "stage_posol_markdown1": "Brine composition: 4.5 L H₂O + 250 g NaCl + 0.8 mg NaNO₂.\n\n🌿 **Introduction of Sea Buckthorn Extract (Key Step)**\nOptimal concentration: 3% - 5% of the brine mass.\nRecommendation: For whole-muscle Zhaya, 5% is preferable (for maximum antioxidant protection).\nFunction: The extract improves water-holding capacity and acts as a natural antioxidant.\nBrine temperature: 16°C\nImmersion in brine: τ=72 hours, t=0−3°C. Pressure P=1200 g–1250 g per 1000 g.",
        "stage_posol_expander2": "Control and Monitoring",
        "stage_posol_markdown2": "* **Salt control:** Using a digital salimeter (Salimeter / Areometer) to check the extNaCl concentration.\n* **pH control:** Daily pH measurement in the brine to track the maturation dynamics (see section \"pH Modeling\").",
        "iot_monitoring_desc": "🌡️ **IoT-Monitoring:**\n\n* **Sensors:** Using wireless thermal sensors (IoT-probe) inside the product for continuous monitoring to reach 74°C.\n\n* **Control action:** Automatic shutdown/switching of the chamber mode upon reaching the set internal temperature.",
        "stage_termo_header": "3. Thermal Processing (IoT Control) 🔥",
        "stage_termo_info": "Thermal processing includes 5 sequential stages. Critical point: internal 74°C",
        "stage1_title": "1. Raw Material Acceptance and Preparation",
        "stage1_params": "Acceptance Control Parameters",
        "initial_mass": "Initial Mass",
        "raw_temp": "Raw Material Temperature",
        "fat_thickness": "Fat Thickness",
        "kpi_title": "Key Technological Indicators (General Summary)",
        "yield_target": "Product Yield (Target)",
        "target_temp": "Target Cooking Temperature",
        "brine_loss": "Brine Mass (Loss)",

        "stage2_title": "2. Salting, Injection, and Massaging",
        "brine_prep": "Brine Preparation and Injection",
        "brine_composition": "Brine Composition",
        "brine_temp": "Brine Temperature",
        "injection": "Injection",
        "massage_params": "Massaging Parameters",
        "total_duration": "Total Duration",
        "working_pressure": "Working Pressure",

        "stage3_title": "3. Thermal Processing (Thermal Chamber)",
        "stage3_info": "Thermal processing includes 5 sequential stages.",
        "drying": "Drying",
        "roasting": "Roasting",
        "steam_cooking": "Steam Cooking",
        "cooling": "Cooling Drying",
        "smoking": "Smoking",

        "stage4_title": "4. Deboning, Packaging, and Storage",
        "stage_upakovka_header": "4. Packaging and Shelf Life",
        "deboning_packaging": "Deboning and Packaging",
        "shelf_life": "Shelf Life and Product Yield",
        "storage_standard": "Standard",
        "storage_freeze": "Freezing",
        "col_stage": "Stage",
        "col_temp": "Temperature (°C)",
        "col_time": "time/criteria",
        "col_purpose": "Appointment",
        "termo_drying": "Drying",
        "termo_frying": "Frying",
        "termo_steam": "Steam Cooking",
        "termo_cool_dry": "Cooling Drying",
        "termo_smoke": "Smoking",
        "termo_drying_desc": "Removing surface moisture",
        "termo_frying_desc": "Color/aroma formation",
        "termo_steam_desc": "Achieving full readiness",
        "termo_cool_desc": "Temperature stabilization",
        "termo_smoke_desc": "Adding aroma",
        "stage_upakovka_expander": "Deboning, Packaging and Storage (Key Parameters)",
        "shelf_life_comparison": "Shelf Life Comparison:",
        "shelf_life_standard": "Shelf life (Standard, without extract)",
        "shelf_life_extract": "Shelf life (With 5% extract)",
        "shelf_life_desc": "Key factor: Sea buckthorn extract reduces the Thiobarbituric Acid Reactive Substances (TBARS) number, which slows down fat oxidation and allows for an increase in shelf life.",
        "storage_tip": "🔬 Critical control during storage: Water activity (Aw) 0.88–0.90 and temperature must be in the range of 0–5°C",
        "stage_upakovka_markdown1": "Cooling: In a cold storage room t=0–5°C for 12 hours. Packaging: Using an automatic vacuum packaging machine.",
        "shelf_life_std_value": "30 days",
        "shelf_life_ext_value": "60 days",
        "shelf_life_delta_value": "+30 days",

        # Regression Models
        "regression_title": "📊 Regression Models of Final Product Quality",
        "regression_subtitle": "Predicting quality based on technological parameters",

        "reg_w_title": "1. Final Product Moisture ($W$)",
        "reg_w_T": "Drying Temperature (T), °C",
        "reg_w_H": "Drying Duration (H), hours",
        "reg_w_E": "Extract Concentration (E), %",
        "reg_w_metric": "Predicted Moisture (W), %",
        "reg_w_delta": "Difference from base value (65%):",
        "reg_w_info": "Adding extract ($E$) improves moisture retention.",

        "reg_aw_title": "2. Water Activity ($A_w$)",
        "reg_aw_C": "Salt Concentration (C), %",
        "reg_aw_Ts": "Salting Duration (Ts), days",
        "reg_aw_metric": "Predicted Water Activity ($A_w$)",
        "reg_aw_delta_high": "Needs to be reduced to achieve Aw ≤ 0.90",
        "reg_aw_delta_ok": "Within safe range",
        "reg_aw_info": "Optimal $A_w$ (0.88–0.90) is critical for microbiological safety.",

        "reg_color_title": "3. Color Stability ($\\Delta E$)",
        "reg_color_desc": "Modeling color change depending on extract and drying time.",
        "reg_color_E": "Extract Concentration (E), %",
        "reg_color_H": "Drying Duration (H), hours",
        "reg_color_metric": "Predicted Color Change ($\\Delta E$)",
        "reg_color_delta": "Optimal value $\\Delta E < 2.0$",
        "reg_color_result_good": "✅ High color stability.",
        "reg_color_result_warn": "⚠️ Acceptable color, slight darkening possible.",
        "reg_color_result_bad": "❌ Significant color change. Overdrying likely.",

        "reg_tbc_title": "4. Oxidative Stability (Peroxide Value - TBC)",
        "reg_tbc_desc": "Prediction of oxidation degree after 30 days of storage.",
        "reg_tbc_E": "Extract Concentration (E), %",
        "reg_tbc_S": "Salt Concentration (S), %",
        "reg_tbc_metric": "Predicted TBC after 30 days, mg/kg",
        "reg_tbc_delta": "The lower, the better (Target TBC < 1.5)",
        "reg_tbc_result_good": "✅ Excellent stability, shelf life up to 60 days.",
        "reg_tbc_result_warn": "⚠️ Good stability, shelf life up to 45 days.",
        "reg_tbc_result_bad": "❌ High oxidation risk, shelf life ≤ 30 days.",

        "reg_strength_title": "5. Mechanical Strength (Formed Products)",
        "reg_strength_info": "Model describes product density and elasticity.",
        "reg_strength_expander": "🛠️ Interactive Strength Simulator",
        "reg_strength_P": "Pressing Pressure (P), kg/cm²",
        "reg_strength_V": "Minced Meat Viscosity (V), units",
        "reg_strength_metric": "Mechanical Stability Index",
        "reg_strength_result_good": "✅ High strength. Good forming quality.",
        "reg_strength_result_warn": "⚠️ Medium strength. Attention to pressure required.",
        "reg_strength_result_bad": "❌ Low strength. Product deformation risk.",

        # pH Modeling
        "ph_basis": "ℹ️ Scientific Basis of pH Modeling",
        "ph_formula_title": "pH Kinetics Formula (Salting Submodel)",
        "ph_initial": "Initial pH (pH0)",
        "ph_final": "Final pH (pH_inf)",
        "rate_constant": "Rate Constant (k)",
        "forecast_time": "Forecast Time (t), hour",
        "predicted_ph": "Predicted pH at a given time",
        "ph_kinetics": "Visualization of pH Kinetics",

        "ph_critical_low": "**Critical acidification.** Product is too acidic.",
        "ph_optimal": "Optimal range.",
        "ph_insufficient": "**Insufficient acidification.**",
        "menu_ph_modeling": "🌡️ pH Modeling",
        "ph_title": "🌡️ pH Modeling During Salting",
        "ph_subtitle": "Prediction of acidity kinetics for safety assurance",
        "ph_basis_text": '''
        **Biochemical meaning:** The decrease in pH (increase in acidity) during meat ripening is a key factor affecting microbial inhibition and proper texture formation. It mainly occurs due to glycogen fermentation into lactic acid by starter cultures and meat enzymes.

        **Why it matters:**
        1. **Safety:** Rapid pH drop below 5.6–5.8 inhibits growth of pathogens (E.coli, Salmonella).
        2. **Quality:** Optimal final pH (4.8–5.4) improves tenderness, color, and water retention.
        3. **Control:** The model predicts if the product will reach the target pH under given conditions.
        ''',
        "ph_formula_title": "pH kinetics formula (Salting submodel)",
        "ph_formula_desc": "Where: pH₀ – initial, pH_inf – final, k – rate constant.",
        "ph_formula_tip": "k depends on temperature and salt concentration.",
        "ph_forecast_title": "⚙️ Interactive prediction and analysis",
        "ph_initial": "Initial pH (pH₀)",
        "rate_constant": "Rate constant (k)",
        "forecast_time": "Forecast time (t), hours",
        "predicted_ph": "Predicted pH at given time",
        "delta_target_ph": "Difference to target pH 5.6:",
        "ph_critical_low": "**Critical acidification.** Product too sour.",
        "ph_insufficient": "**Insufficient acidification.**",
        "ph_kinetics": "pH kinetics visualization",
        "time_hours": "Time (h)",
        "hours_short": "h",
        "ph_plot_title": "pH kinetics during salting",

        # Sea Buckthorn Analysis
        "menu_seabuck_analysis": "🔬 Analysis of the effect of sea buckthorn extract",
        "seabuck_title": "🔬 Effect of sea buckthorn extract on the quality of Zhaya and formed meat",
        "seabuck_desc": "Results of experimental study (based on report data).",
        "table1_title": "Table 1. Main indicators of smoked Zhaya (control and 5% extract)",
        "table2_title": "Table 2. Main indicators of formed meat product (control and 3% extract)",
        "indicator": "Indicator",
        "control": "Control (0%)",
        "with_extract_5": "Zhaya + 5% extract",
        "with_extract_3": "Formed meat + 3% extract",
        "moisture": "Moisture content, %",
        "protein": "Protein, %",
        "fat": "Fat, %",
        "vus": "Water-holding capacity (WHC), %",
        "tbch": "TBARs, mg/kg",
        "salt": "NaCl, %",
        "ash": "Ash, %",
        "fig1_title": "Fig. 1. Effect of extract on Zhaya moisture content",
        "fig1_plot_title": "Effect of sea buckthorn extract on Zhaya moisture content",
        "fig2_title": "Fig. 2. Protein and fat content in Zhaya",
        "fig2_plot_title": "Protein and fat in Zhaya",
        "fig3_title": "Fig. 3. WHC, WRC, and FRC of smoked Zhaya",
        "fig3_plot_title": "WHC, WRC, and FRC of smoked Zhaya",
        "fig4_title": "Fig. 4. Oxidative indicators of Zhaya",
        "fig4_plot_title": "Oxidative indicators of Zhaya",
        "fig5_title": "Fig. 5. Oxidative indicators of formed meat",
        "fig5_plot_title": "Oxidative indicators of formed meat",

        # Data Exploration
        "explore_title": "🗂️ Raw Data Exploration",
        "explore_desc": "Select a table to view.",
        "select_data": "Select Data:",
        "viewing_data": "Viewing data from:",
        "data_empty_warning": "Data not loaded or empty.",
        "data_load_error": "Failed to load data for viewing.",

        # History / DB
        "db_title": "📚 Measurement History and Database",
        "db_desc": "This stores the measurement history (SQLite). You can export, filter, and delete records.",
        "total_records": "Total records:",
        "history_empty": "History is empty",
        "export_all": "Export all to CSV",
        "clear_all": "Clear all measurements",
        "confirm_clear": "Confirm clear",
        "db_cleared": "Database cleared. Reload the page.",
        "ph_distribution": "pH Distribution",
        "ph_over_time": "pH over Time (Interactive)",

        # ML Page
        "menu_ml": "ML: Train / Predict",
        "ml_title": "🧠 ML: Training and pH Prediction",
        "ml_desc": "Upload a CSV/Excel file with 'pH' column and features for training, or a CSV with features for prediction.",
        "train_tab": "Train",
        "predict_tab": "Predict",
        "train_subtitle": "Model Training",
        "upload_train": "CSV/Excel for training (with 'pH' column)",
        "preview": "Preview:",
        "target_column": "Target column (pH):",
        "features": "Features (if empty — all numeric columns except target will be used)",
        "train_success": "Training completed successfully.",
        "train_error": "Training error:",
        "no_data": "No data.",
        "predict_subtitle": "Prediction",
        "upload_predict": "CSV for prediction (same features)",
        "auto_features": "Automatically selected numeric features:",
        "predict_results": "Prediction Results",
        "save_to_db": "Save predictions to database (sample_name -> sample)",
        "saved_records": "Records saved to DB:",

        # Data Input
        "menu_input": "Data Input",
        "input_title": "➕ Enter New Product Data",
        "input_subtitle": "Add a new production batch to the database",
        "sheet": "sheet",
        "batch_params": "Enter parameters of the new production batch",
        "batch_id": "Batch ID (auto-generated)",
        "mass": "Batch mass (kg)",
        "initial_temp": "Initial temperature (°C)",
        "salt_content": "Salt content (%)",
        "moisture": "Moisture (%)",
        "starter_culture": "Starter culture (CFU/g)",
        "extract_content": "Extract concentration (%)",
        "save_data": "💾 Save data",
        "batch_added": "✅ New batch successfully added",
        "save_error": "❌ Error writing to file:",
        "current_data": "📊 Current data",
        "batchid_missing": "❌ The sheet does not contain 'BatchID' column. Check the table structure.",

        # pH Statuses
        "ph_in_normal": "pH is normal",
        "ph_too_low": "pH is too low",
        "ph_too_high": "pH is too high",
        "anim_good": "✅ Everything is fine",
        "anim_bad": "⚠️ Correction needed",
    },
    "kk": {
        # Жалпы элементтер
        "title": "Сандық платформа — Meat Digitalization",
        "select_section": "Бөлімді таңдаңыз",
        "full_title": "«Жая» ет деликатесіне арналған цифрлық платформа",
        "version_note": "Нұсқа: біріктірілген",
        "db_reset_confirm": "Барлық өлшемдерді жойғыңыз келетініне сенімдісіз бе?",
        "train_button": "Модельді үйрету",
        "predict_button": "Болжам жасау",
        "upload_csv": "CSV/Excel жүктеу",
        "export": "CSV жүктеп алу",
        "no_data": "Көрсетуге деректер жоқ",
        "save": "Сақтау",
        "saved": "Сақталды",
        "download": "Жүктеп алу",

        # Навигация
        "menu_home": "Басты бет",
        "menu_production_process": "Жая өнімін өндіру процесі",
        "menu_regression_models": "Сапаның регрессиялық модельдері",
        "menu_ph_modeling": "pH модельдеу",
        "menu_seabuckthorn_analysis": "Шырғанақ сығындысымен талдау",
        "menu_data_exploration": "Деректерді зерттеу",
        "menu_history_db": "Тарих / ДБ",
        "menu_ml_train_predict": "ML: Оқыту / Болжау",
        "menu_new_data_input": "Жаңа деректерді енгізу",
        "jaya_process_title": "Жая өнімін өндірудің технологиялық картасы",
        "jaya_process_subtitle": "Қадамдық сапаны бақылау және процесс параметрлері (сығынды мен IoT ескеріледі)",
        "stage_priemka": "1. Шикізатты қабылдау 🥩",
        "stage_posol": "2. Тұздау (сығынды) 🧂",
        "stage_termo": "3. Термиялық өңдеу 🔥",
        "stage_upakovka": "4. Сақтау 📦",

        # Басты бет
        "unit_kg": "кг",
        "unit_g": "г",
        "moisture_title": "ылғал",
        "home_title": "🐎 Жай өнімін өндіру және модельдеуге арналған сандық платформа",
        "home_desc": "Өндірісті оңтайландыру және сапаны бақылау үшін Интеллектуалды шешімдер",
        "home_info": "Жұмысты бастау үшін сол жақтағы мәзірден бөлімді таңдаңыз.",
        "stage_control_suffix": "4 бақылау қадамдары",
        "delta_production": "Шикізаттан орауға дейін",
        "delta_regression": "Тұздау/кептіру параметрлеріне негізделген",
        "seabuckthorn_value": "Шырғанақ",
        "delta_seabuckthorn": "Тұрақтылықты/сақтау мерзімін арттыру",

        "scientific_achievements": "🏆 Негізгі ғылыми жетістіктер",
        "wac_title": "Нем ұстау қабілеті (НҰҚ)",
        "wac_subtitle": "5% сығындымен НҰҚ өсуі",
        "wac_note": "Бақылаудағы 60.2% -ға қарсы.",
        "shelf_life_title": "Жарамдылық мерзімі (Болжам)",
        "shelf_life_subtitle": "0–5°С кезіндегі максималды сақтау мерзімі",
        "shelf_life_note": "Стандарттан (30 тәулік) 30 күнге ұзағырақ.",
        "optimal_conc_title": "Оптималды концентрация",
        "optimal_conc_subtitle": "Ұсынылатын сығынды мөлшері",
        "optimal_conc_note": "Дәм мен тұрақтылық теңгерімі.",

        "oxidation_stability_title": "🧪 Тотығу тұрақтылығы: Асқын тотығу саны (ТБӘ) төмендеуі",
        "oxidation_goal": "**Мақсат:** 30 күн сақтаудан кейін тотығуды азайту.",
        "tba_reduction_text": "ТБӘ төмендеуі",
        "oxidation_success": "Өнімнің жоғары тотығуға қарсы тұрақтылығына қол жеткізілді.",
        "tba_caption": "ТБӘ төмендеуі",
        "tba_caption_to": "дейін",
        "tba_caption_control": "бақылау",
        "tba_caption_extract": "(5% сығынды)",
        "mg_per_kg": "мг/кг",
        # Өндіріс процесі
        "prod_title": "🍖 Жай өнімін өндірудің технологиялық картасы",
        "prod_subtitle": "Сапаны қадамдық бақылау және процесс параметрлері",
        "stage_1": "1. Шикізатты қабылдау 🥩",
        "stage_2": "2. Тұздау және массалау 🧂",
        "stage_3": "3. Термиялық өңдеу 🔥",
        "stage_4": "4. Сақтау және орау 📦",
        "stage_priemka_header": "1. Шикізатты қабылдау 🥩",
        "metric_mass": "Бастапқы масса",
        "metric_temp": "Шикізат температурасы",
        "metric_ph": "Бастапқы pH",
        "metric_yield": "Өнім шығысы",
        "metric_target_temp": "Мақсатты t°",
        "metric_brine_loss": "Тұздық массасы (Жоғалту)",
        "tech_params_title": "Негізгі технологиялық көрсеткіштер",
        "delta_gost": "ГОСТ бойынша",
        "delta_inner": "өнім ішінде",
        "help_ph": "Жетілу болжамы үшін маңызды",
        "help_temp": "Камерадағы IoT сенсорларын пайдаланып бақылау.",
        "digital_control_tip": "💡 Цифрлық басқару: Шикізаттың массасы мен температурасын автоматты түрде жазу.",
        "day_in_lang": "тәулік",
        "stage_posol_header": "2. Тұздау, Шырғанақ сығындысы және Массалау🧂",
        "stage_posol_expander1": "Тұздықты дайындау және енгізу",
        "stage_posol_markdown1": "Тұздықтың құрамы: 4,5 л H₂O + 250 г NaCl + 0,8 мг NaNO₂.\n\n🌿 **Шырғанақ сығындысын енгізу (Негізгі қадам)**\nОңтайлы концентрация: Тұздық массасының 3% - 5%.\nҰсыныс: Бүтін бұлшықетті Жая үшін 5% артықшылық беріледі (ең жоғары тотығуға қарсы қорғаныс үшін).\nФункциясы: Сығынды ылғал ұстау қабілетін жақсартады және табиғи антиоксидант ретінде әрекет етеді.\nТұздық температурасы: 16°C\nТұздыққа салу: τ=72 сағат, t=0−3°C. 1000 грамға P=1200 г–1250 г қысым.",
        "stage_posol_expander2": "Бақылау және мониторинг",
        "stage_posol_markdown2": "* **Тұзды бақылау:** extNaCl концентрациясын тексеру үшін цифрлық тұз өлшегішті (Солемер / Ареометр) қолдану.\n* **pH бақылау:** Пісіп-жетілу динамикасын бақылау үшін тұздықтағы pH мәнін күнделікті өлшеу (құжаттың \"pH модельдеу\" бөлімін қараңыз).",
        "iot_monitoring_desc": "🌡️ **IoT-Мониторинг:**\n\n* **Датчиктер:** Өнім ішінде 74°C-қа жетуді тұрақты бақылау үшін сымсыз термодатчиктерді (IoT-зонд) қолдану.\n\n* **Басқару әсері:** Белгіленген ішкі температураға жеткенде камера режимінің автоматты түрде өшірілуі/ауыстырылуы.",
        "stage_termo_header": "3. Термиялық өңдеу (IoT бақылауы) 🔥",
        "stage_termo_info": "Термиялық өңдеу 5 кезеңнен тұрады. Сындарлы нүкте: ішкі 74°C",

        "stage_priemka_expander": "Қабылдауды бақылау параметрлері",
        "stage1_title": "1. Шикізатты қабылдау және дайындау",
        "stage1_params": "Қабылдауды бақылау параметрлері",
        "initial_mass": "Бастапқы масса",
        "raw_temp": "Шикізат температурасы",
        "fat_thickness": "Майдың қалыңдығы",
        "kpi_title": "Негізгі Технологиялық Көрсеткіштер (Жалпы шолу)",
        "yield_target": "Өнім шығымы (Мақсат)",
        "target_temp": "Мақсатты t°",
        "brine_loss": "Тұздық массасы (Жоғалту)",

        "stage2_title": "2. Тұздау, Шприцтеу және Массалау",
        "brine_prep": "Тұздықты дайындау және шприцтеу",
        "brine_composition": "Тұздық құрамы",
        "brine_temp": "Тұздық температурасы",
        "injection": "Шприцтеу",
        "massage_params": "Массалау параметрлері",
        "total_duration": "Жалпы ұзақтығы",
        "working_pressure": "Жұмыс қысымы",

        "stage3_title": "3. Термиялық өңдеу (Термокамера)",
        "stage3_info": "Термиялық өңдеу 5 кезеңнен тұрады.",
        "drying": "Кептіру",
        "roasting": "Қуыру",
        "steam_cooking": "Бумен пісіру",
        "cooling": "Суытумен кептіру",
        "smoking": "Ыстау",
        "col_stage": "Кезең",
        "col_temp": "Температура (°C)",
        "col_time": "Уақыт/критерий",
        "col_purpose": "Мақсаты",
        "termo_drying": "Кептіру",
        "termo_frying": "Қуыру",
        "termo_steam": "Буда пісіру",
        "termo_cool_dry": "Суықпен кептіру",
        "termo_smoke": "Ыстау",
        "termo_drying_desc": "Беткі ылғалды кетіру",
        "termo_frying_desc": "Түсін/хош иісін қалыптастыру",
        "termo_steam_desc": "Толық дайындыққа жету",
        "termo_cool_desc": "Температураны тұрақтандыру",
        "termo_smoke_desc": "Хош иіс беру",
        "stage4_title": "4. Сүйектен айыру, Орау және Сақтау",
        "deboning_packaging": "Сүйектен айыру және орау",
        "shelf_life": "Сақтау мерзімі және өнім шығымы",
        "storage_standard": "Стандарт",
        "storage_freeze": "Мұздату",
        "stage_upakovka_header": "4. Орау және Жарамдылық Мерзімі",
        "stage_upakovka_expander": "Обвалка, Орау және Сақтау (Негізгі параметрлер)",
        "shelf_life_comparison": "Жарамдылық мерзімдерін салыстыру:",
        "shelf_life_standard": "Жарамдылық мерзімі (Стандарт, экстрактсыз)",
        "shelf_life_extract": "Жарамдылық мерзімі (5% экстрактпен)",
        "shelf_life_desc": "Негізгі фактор: Теңіз шырғанағының сығындысы тотығу санын (ТC) төмендетеді, бұл майлардың тотығуын бәсеңдетеді және жарамдылық мерзімін ұзартуға мүмкіндік береді.",
        "storage_tip": "🔬 Сақтау кезіндегі маңызды бақылау: Су белсенділігі (Aw) 0.88–0.90 және температура 0-5°C диапазонында болуы керек",
        "stage_upakovka_markdown1": "Салқындату: Тоңазытқыш камерада t=0–5°С — 12 сағат. Орау: Вакуумды-орау автоматында.",
        "shelf_life_std_value": "30 тәулік",
        "shelf_life_ext_value": "60 тәулік",
        "shelf_life_delta_value": "+30 күн",
        # Регрессиялық модельдер
        "regression_title": "📊 Дайын өнім сапасының регрессиялық модельдері",
        "regression_subtitle": "Технологиялық параметрлерге негізделген сапаны болжау",

        "reg_w_title": "1. Дайын өнімнің ылғалдылығы ($W$)",
        "reg_w_T": "Кептіру температурасы (T), °C",
        "reg_w_H": "Кептіру уақыты (H), сағат",
        "reg_w_E": "Экстракт концентрациясы (E), %",
        "reg_w_metric": "Болжанған ылғалдылық (W), %",
        "reg_w_delta": "Базалық мәннен айырмашылығы (65%):",
        "reg_w_info": "Экстракт қосу ($E$) өнімнің ылғал ұстау қабілетін арттырады.",

        "reg_aw_title": "2. Судың белсенділігі ($A_w$)",
        "reg_aw_C": "Тұз концентрациясы (C), %",
        "reg_aw_Ts": "Тұздау ұзақтығы (Ts), тәулік",
        "reg_aw_metric": "Болжанған судың белсенділігі ($A_w$)",
        "reg_aw_delta_high": "Aw ≤ 0.90 деңгейіне жету үшін төмендету қажет",
        "reg_aw_delta_ok": "Қауіпсіз норма шегінде",
        "reg_aw_info": "Оптималды $A_w$ (0.88–0.90) микробиологиялық қауіпсіздік үшін маңызды.",

        "reg_color_title": "3. Түстің тұрақтылығы ($\\Delta E$)",
        "reg_color_desc": "Экстракт пен кептіру уақытына байланысты түстің өзгерісін модельдеу.",
        "reg_color_E": "Экстракт концентрациясы (E), %",
        "reg_color_H": "Кептіру ұзақтығы (H), сағат",
        "reg_color_metric": "Болжанған түс өзгерісі ($\\Delta E$)",
        "reg_color_delta": "Оптималды мән $\\Delta E < 2.0$",
        "reg_color_result_good": "✅ Түстің жоғары тұрақтылығы.",
        "reg_color_result_warn": "⚠️ Түс қабылдауға жарамды, аздап күңгірттенуі мүмкін.",
        "reg_color_result_bad": "❌ Түстің айтарлықтай өзгеруі. Кептіру уақыты тым ұзақ.",

        "reg_tbc_title": "4. Тотықтыру тұрақтылығы (Перекис саны - TBC)",
        "reg_tbc_desc": "30 күн сақтау кезіндегі тотығу деңгейін болжау.",
        "reg_tbc_E": "Экстракт концентрациясы (E), %",
        "reg_tbc_S": "Тұз концентрациясы (S), %",
        "reg_tbc_metric": "30 күннен кейінгі болжанған TBC, мг/кг",
        "reg_tbc_delta": "Неғұрлым төмен болса, соғұрлым жақсы (мақсат TBC < 1.5)",
        "reg_tbc_result_good": "✅ Өнімнің жоғары тұрақтылығы, сақтау мерзімі 60 күнге дейін.",
        "reg_tbc_result_warn": "⚠️ Жақсы тұрақтылық, сақтау мерзімі 45 күнге дейін.",
        "reg_tbc_result_bad": "❌ Тотығу қаупі жоғары, сақтау мерзімі ≤ 30 күн.",

        "reg_strength_title": "5. Механикалық беріктік (қалыпталған өнімдер)",
        "reg_strength_info": "Модель өнімнің тығыздығы мен серпінділігін сипаттайды.",
        "reg_strength_expander": "🛠️ Механикалық беріктік симуляторы",
        "reg_strength_P": "Престеу қысымы (P), кг/см²",
        "reg_strength_V": "Фарш тұтқырлығы (V), шартты бірлік",
        "reg_strength_metric": "Механикалық тұрақтылық индексі",
        "reg_strength_result_good": "✅ Жоғары беріктік. Қалыптау сапалы.",
        "reg_strength_result_warn": "⚠️ Орташа беріктік. Қысымға назар аудару қажет.",
        "reg_strength_result_bad": "❌ Төмен беріктік. Өнім деформация қаупі бар.",

        # pH модельдеу
        "ph_title": "🌡️ Тұздау процесіндегі pH модельдеу",
        "ph_subtitle": "Қауіпсіздікті қамтамасыз ету үшін қышқылдықтың кинетикасын болжау",
        "ph_basis": "ℹ️ pH-модельдеудің ғылыми негіздемесі",
        "ph_formula_title": "pH кинетикасы формуласы (Тұздау кіші моделі)",
        "ph_initial": "Бастапқы pH (pH0)",
        "ph_final": "Соңғы pH (pH_inf)",
        "rate_constant": "Жылдамдық тұрақтысы (k)",
        "forecast_time": "Болжау уақыты (t), сағ",
        "predicted_ph": "Берілген уақыттағы болжанған pH",
        "ph_kinetics": "pH кинетикасын визуализациялау",

        "ph_critical_low": "**Сыни қышқылдану.** Өнім тым қышқыл.",
        "ph_optimal": "Оңтайлы диапазон.",
        "ph_insufficient": "**Жеткіліксіз қышқылдану.**",
        "menu_ph_modeling": "🌡️ pH моделдеуі",
        "ph_title": "🌡️ Тұздау кезіндегі pH моделдеуі",
        "ph_subtitle": "Қауіпсіздікке кепіл болу үшін қышқылдық кинетикасын болжау",
        "ph_basis": "ℹ️ pH моделдеуінің ғылыми негізі",
        "ph_basis_text": '''
        **Биохимиялық мәні:** Еттің жетілу процесінде pH төмендеуі (қышқылдықтың артуы) – қажетсіз микроорганизмдердің көбеюін тежеуге және дұрыс құрылым мен дәмнің қалыптасуына әсер ететін негізгі фактор. Бұл көбіне гликогеннің ет ферменттері мен стартерлік культуралар арқылы сүт қышқылына айналуынан болады.

        **Неліктен бұл маңызды:**
        1. **Қауіпсіздік:** pH 5.6–5.8-ден төмен деңгейге тез түсу патогенді бактериялардың (E.coli, Salmonella) өсуін тежейді.
        2. **Сапа:** Оптималды соңғы pH (4.8–5.4) еттің жұмсақтығын, түсін және ылғал ұстауын жақсартады.
        3. **Бақылау:** Модель ағымдағы жағдайларда (температура, тұз, стартерлер) өнімнің мақсатты pH-ге жететінін болжай алады.
        ''',
        "ph_formula_title": "pH кинетикасының формуласы (Тұздау ішкі моделі)",
        "ph_formula_desc": "Мұнда: pH₀ — бастапқы мән, pH_inf — соңғы мән, k — жылдамдық тұрақтысы.",
        "ph_formula_tip": "k мәні температура мен тұз мөлшеріне байланысты өзгереді.",
        "ph_forecast_title": "⚙️ Интерактивті болжау және талдау",
        "ph_initial": "Бастапқы pH (pH₀)",
        "ph_final": "Соңғы pH (pH_inf)",
        "rate_constant": "Жылдамдық тұрақтысы (k)",
        "forecast_time": "Болжау уақыты (t), сағат",
        "predicted_ph": "Берілген уақытта болжанған pH",
        "delta_target_ph": "Мақсатты pH 5.6 дейінгі айырмашылық:",
        "ph_critical_low": "**Критикалық қышқылдану.** Өнім тым қышқыл.",
        "ph_optimal": "Оптималды диапазон.",
        "ph_insufficient": "**Қышқылдану жеткіліксіз.**",
        "ph_kinetics": "pH кинетикасын визуализациялау",
        "time_hours": "Уақыт (сағат)",
        "hours_short": "сағ",
        "ph_plot_title": "Тұздау процесіндегі pH кинетикасы",

        # Шырғанақ талдауы
        "menu_seabuck_analysis": "🔬 Облепиха экстрактының әсерін талдау",
        "seabuck_title": "🔬 Облепиха экстрактының жая мен қалыпталған ет сапасына әсері",
        "seabuck_desc": "Эксперименттік зерттеу нәтижелері (Есеп деректері негізінде).",
        "table1_title": "1-кесте. Ысталған жаяның негізгі көрсеткіштері (бақылау және 5% экстракт)",
        "table2_title": "2-кесте. Қалыпталған ет өнімінің негізгі көрсеткіштері (бақылау және 3% экстракт)",
        "indicator": "Көрсеткіш",
        "control": "Бақылау (0%)",
        "with_extract_5": "Жая + 5% сығынды",
        "with_extract_3": "Қалыпталған ет + 3% сығынды",
        "moisture": "Ылғал мөлшері, %",
        "protein": "Ақуыз, %",
        "fat": "Май, %",
        "vus": "Ылғал ұстау қабілеті (ВУС), %",
        "tbch": "ТБЧ, мг/кг",
        "salt": "NaCl, %",
        "ash": "Күл, %",
        "fig1_title": "1-сурет. Экстракттың жая ылғалдылығына әсері",
        "fig1_plot_title": "Облепиха экстрактының жая ылғалдылығына әсері",
        "fig2_title": "2-сурет. Жаядағы ақуыз және май мөлшері",
        "fig2_plot_title": "Жаядағы ақуыз және май",
        "fig3_title": "3-сурет. Ысталған жаяның ВУС, ВСС және ЖУС көрсеткіштері",
        "fig3_plot_title": "Ысталған жаяның ВУС, ВСС және ЖУС көрсеткіштері",
        "fig4_title": "4-сурет. Жаяның тотығу көрсеткіштері",
        "fig4_plot_title": "Жаяның тотығу көрсеткіштері",
        "fig5_title": "5-сурет. Қалыпталған еттің тотығу көрсеткіштері",
        "fig5_plot_title": "Қалыпталған еттің тотығу көрсеткіштері",

        # Деректерді зерттеу
        "explore_title": "🗂️ Бастапқы деректерді зерттеу",
        "explore_desc": "Көру үшін кестені таңдаңыз.",
        "select_data": "Деректерді таңдаңыз:",
        "viewing_data": "Деректерді қарау:",
        "data_empty_warning": "Деректер жүктелмеген немесе бос.",
        "data_load_error": "Деректерді қарау үшін жүктеу мүмкін болмады.",

        # Тарих / ДБ
        "db_title": "📚 Өлшем тарихы және деректер базасы",
        "db_desc": "Мұнда өлшем тарихы сақталады (SQLite). Жазбаларды экспорттауға, сүзуге және жоюға болады.",
        "total_records": "Барлық жазбалар:",
        "history_empty": "Тарих бос",
        "export_all": "Барлығын CSV-ге экспорттау",
        "clear_all": "Барлық өлшемдерді тазалау",
        "confirm_clear": "Тазалауды растау",
        "db_cleared": "Деректер базасы тазартылды. Бетті қайта жүктеңіз.",
        "ph_distribution": "pH таралуы",
        "ph_over_time": "Уақыт бойынша pH (интерактивті)",

        # ML беті
        "menu_ml": "ML: Үйрету / Болжау",
        "ml_title": "🧠 ML: Үйрету және pH болжау",
        "ml_desc": "Үйрету үшін 'pH' бағаны бар CSV/Excel файлын немесе болжау үшін деректермен CSV файлын жүктеңіз.",
        "train_tab": "Үйрету",
        "predict_tab": "Болжау",
        "train_subtitle": "Модельді үйрету",
        "upload_train": "Үйретуге арналған CSV/Excel (pH бағанымен)",
        "preview": "Алдын ала қарау:",
        "target_column": "Мақсатты бағанды (pH) таңдаңыз:",
        "features": "Белгілер (егер бос болса — мақсаттан басқа барлық сандық бағандар алынады)",
        "train_success": "Үйрету сәтті аяқталды.",
        "train_error": "Үйрету қатесі:",
        "no_data": "Деректер жоқ.",
        "predict_subtitle": "Болжау",
        "upload_predict": "Болжауға арналған CSV (сол белгілермен)",
        "auto_features": "Автоматты түрде таңдалған сандық белгілер:",
        "predict_results": "Болжау нәтижелері",
        "save_to_db": "Болжауларды дерекқорға сақтау (sample_name -> sample)",
        "saved_records": "Дерекқорға сақталған жазбалар саны:",

        # Деректерді енгізу
        "menu_input": "Деректер енгізу",
        "input_title": "➕ Өнім туралы жаңа деректерді енгізу",
        "input_subtitle": "Жаңа өндірістік циклды дерекқорға қосу",
        "sheet": "парақ",
        "batch_params": "Жаңа өндірістік цикл параметрлерін енгізіңіз",
        "batch_id": "Batch ID (автоматты түрде)",
        "mass": "Партия массасы (кг)",
        "initial_temp": "Бастапқы температура (°C)",
        "salt_content": "Тұз мөлшері (%)",
        "moisture": "Ылғалдылық (%)",
        "starter_culture": "Стартер мәдениеті (КОЕ/г)",
        "extract_content": "Экстракт концентрациясы (%)",
        "save_data": "💾 Деректерді сақтау",
        "batch_added": "✅ Жаңа партия сәтті қосылды",
        "save_error": "❌ Файлға жазу кезінде қате:",
        "current_data": "📊 Ағымдағы деректер",
        "batchid_missing": "❌ Парақта 'BatchID' бағаны жоқ. Кесте құрылымын тексеріңіз.",

        # pH статус
        "ph_in_normal": "pH қалыпты",
        "ph_too_low": "pH тым төмен",
        "ph_too_high": "pH тым жоғары",
        "anim_good": "✅ Бәрі дұрыс",
        "anim_bad": "⚠️ Түзету қажет",
    }
}


def get_text(key: str, lang: str = "ru") -> str:
    """
    Return localized string for `key` in language `lang`.
    If missing, fallback to key.
    """
    try:
        return LANG.get(lang, LANG["ru"]).get(key, key)
    except Exception:
        return key


def df_to_download_link(df, filename="export.csv", link_text="Скачать"):
    """
    Создает HTML-ссылку для скачивания DataFrame в виде CSV-файла.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()

    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'


# ---------------------------
# pH timeseries plot (plotly)
# ---------------------------
def plot_ph_timeseries(df: pd.DataFrame, t_col: str = 'created_at', ph_col: str = 'ph', title: Optional[str] = None,
                       lang: str = "ru"):
    """
    Plot interactive pH timeseries using Plotly.
    - Clips y-axis to [0, 14] by default, but focuses on realistic range.
    - df must contain t_col and ph_col.
    """
    if df is None or df.empty:
        st.info(get_text("no_data", lang))
        return

    df = df.copy()
    if t_col in df.columns:
        try:
            df[t_col] = pd.to_datetime(df[t_col])
        except Exception:
            pass

    if title is None:
        title = get_text("ph_graph_title", lang)

    fig = px.line(df.sort_values(t_col), x=t_col, y=ph_col, title=title, markers=True)
    fig.update_yaxes(range=[0, 8], title="pH")
    fig.update_xaxes(title="Time")
    fig.update_layout(hovermode="x unified", template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# smoothing utility
# ---------------------------
def smooth_array(arr, window: int = 3):
    """
    Simple moving average smoothing for 1D numpy array or list.
    Returns numpy array.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr
    if window >= arr.size:
        return np.full_like(arr, arr.mean())
    return np.convolve(arr, np.ones(window) / window, mode='same')


# ---------------------------
# pH animation / CSS generator
# ---------------------------
def ph_animation_style(ph_value: float, lang: str = "ru", low_bound: float = 4.8, high_bound: float = 6.5) -> str:
    """
    Returns an HTML snippet with CSS animation depending on pH.
    - low_bound, high_bound define "optimal" range (customizable).
    - 'good' -> gentle green pulse + thumbs-up emoji
    - 'low' or 'high' -> red shake or orange warning pulse
    Use: st.markdown(ph_animation_style(ph, lang), unsafe_allow_html=True)
    """
    try:
        phv = float(ph_value)
    except Exception:
        phv = None

    normal_msg = get_text("ph_in_normal", lang)
    low_msg = get_text("ph_too_low", lang)
    high_msg = get_text("ph_too_high", lang)
    anim_good = get_text("anim_good", lang)
    anim_bad = get_text("anim_bad", lang)

    css_base = '''
    <style>
    .ph-card {
      border-radius: 12px;
      padding: 12px 18px;
      display: inline-block;
      font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      transition: transform 0.25s ease, box-shadow 0.25s ease;
      margin: 8px 0;
    }
    .good {
      background: linear-gradient(90deg, rgba(68,204,68,0.12), rgba(68,204,68,0.06));
      border: 1px solid rgba(68,204,68,0.18);
      animation: gentlePulse 1.6s infinite;
    }
    @keyframes gentlePulse {
      0% { box-shadow: 0 6px 18px rgba(68,204,68,0.06); transform: translateY(0px); }
      50% { box-shadow: 0 10px 26px rgba(68,204,68,0.12); transform: translateY(-4px); }
      100% { box-shadow: 0 6px 18px rgba(68,204,68,0.06); transform: translateY(0px); }
    }
    .warn {
      background: linear-gradient(90deg, rgba(255,170,0,0.12), rgba(255,170,0,0.04));
      border: 1px solid rgba(255,170,0,0.18);
      animation: warnPulse 1.1s infinite;
    }
    @keyframes warnPulse {
      0% { transform: translateY(0px); }
      50% { transform: translateY(-3px); }
      100% { transform: translateY(0px); }
    }
    .bad {
      background: linear-gradient(90deg, rgba(255,68,68,0.12), rgba(255,68,68,0.04));
      border: 1px solid rgba(255,68,68,0.18);
      animation: shake 0.7s infinite;
    }
    @keyframes shake {
      0% { transform: translateX(0px); }
      20% { transform: translateX(-5px); }
      40% { transform: translateX(5px); }
      60% { transform: translateX(-4px); }
      80% { transform: translateX(4px); }
      100% { transform: translateX(0px); }
    }
    .ph-value {
      font-weight: 700;
      font-size: 1.6rem;
    }
    .ph-emoji {
      font-size: 1.6rem;
      margin-right: 8px;
    }
    .ph-msg {
      font-size: 1rem;
      margin-top: 6px;
      color: #333;
    }
    </style>
    '''

    if phv is None:
        html = css_base + '''
        <div class="ph-card" style="background:#f4f4f4;border:1px solid #eee;">
            <div><span class="ph-value">—</span></div>
            <div class="ph-msg">No pH value</div>
        </div>
        '''
        return html

    if low_bound <= phv <= high_bound:
        emoji = "✅"
        state = "good"
        msg = f"{anim_good} — {normal_msg}"
        color = "#44cc44"
    elif phv < low_bound:
        emoji = "🛑"
        state = "bad"
        msg = f"{anim_bad} — {low_msg} ({phv:.2f})"
        color = "#ff4444"
    else:
        emoji = "⚠️"
        state = "warn"
        msg = f"{anim_bad} — {high_msg} ({phv:.2f})"
        color = "#ffaa00"

    html = css_base + f'''
    <div class="ph-card {state}">
        <div style="display:flex; align-items:center;">
            <div class="ph-emoji">{emoji}</div>
            <div>
                <div class="ph-value" style="color:{color};">{phv:.2f}</div>
                <div class="ph-msg">{msg}</div>
            </div>
        </div>
    </div>
    '''
    return html
