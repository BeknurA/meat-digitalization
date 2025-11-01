# ui_components.py
import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import numpy as np
from typing import Optional

# ui_components.py - Расширенный словарь для мультиязычности
import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import numpy as np
from typing import Optional

# ---------------------------
# Полный мультиязычный словарь (RU / EN / KK)
# ---------------------------

"""
Мультиязычный словарь (i18n) для цифровой платформы "Meat Digitalization".
Поддерживаемые языки:
- ru: Русский
- en: English
- kk: Қазақ (Kazakh)
"""

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
        "home_desc": "Эта система объединяет описание технологических процессов и интерактивные математические модели для анализа и прогнозирования качества продукции.",
        "home_info": "Выберите раздел в меню слева, чтобы начать работу.",
        
        # Процесс производства
        "prod_title": "🍖 Технологическая карта производства Жая",
        "prod_subtitle": "Пошаговый контроль качества и параметры процесса",
        "stage_1": "1. Приемка сырья 🥩",
        "stage_2": "2. Посол и массирование 🧂",
        "stage_3": "3. Термическая обработка 🔥",
        "stage_4": "4. Хранение и упаковка 📦",
        
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
        
        "stage4_title": "4. Обвалка, Упаковка и Хранение",
        "deboning_packaging": "Обвалка и Упаковка",
        "shelf_life": "Сроки и Выход продукта",
        "storage_standard": "Стандарт",
        "storage_freeze": "Заморозка",
        
        # Регрессионные модели
        "regression_title": "📊 Регрессионные модели качества конечного продукта",
        "regression_subtitle": "Прогнозирование качества на основе технологических параметров",
        "scientific_basis": "ℹ️ Научное обоснование",
        "scientific_desc": "Математические модели позволяют прогнозировать ключевые показатели качества готового продукта",
        
        "moisture_title": "1. Влажность конечного продукта (W)",
        "drying_temp": "Температура сушки (T), °C",
        "drying_duration": "Продолжительность сушки (H), час",
        "extract_conc": "Концентрация экстракта (E), %",
        "predicted_moisture": "Прогнозируемая Влажность (W), %",
        "moisture_info": "Добавление экстракта (E) положительно влияет на влагоудержание, а длительность сушки (H) снижает влажность.",
        
        "water_activity_title": "2. Активность воды (Aw)",
        "salt_conc": "Концентрация соли (C), %",
        "salting_duration": "Длительность соления (Ts), сут",
        "predicted_aw": "Прогнозируемая Активность воды (Aw)",
        "aw_info": "Оптимальный Aw (0.88-0.90) критичен для микробиологической безопасности и продления срока годности.",
        
        "color_title": "3. Цветовая стабильность (ΔE)",
        "color_info": "Модель Цветовой стабильности описывает, как изменяется цвет продукта",
        
        "strength_title": "4. Механическая прочность (формованные изделия)",
        "strength_info": "Модель описывает плотность и упругость продукта",
        "strength_simulator": "🛠️ Интерактивный симулятор прочности",
        "pressure": "Давление прессования (P), кг/см²",
        "viscosity": "Вязкость фарша (V), условные единицы",
        "strength_index": "Прогнозируемый Индекс Механической Прочности (Усл. ед.)",
        
        "recommendations_title": "5. Практические рекомендации по добавлению экстракта облепихи",
        "optimal_conc": "🎯 Оптимальная концентрация экстракта",
        "for_whole_muscle": "Для цельномышечной жая (копчёной)",
        "for_formed": "Для формованного мясного изделия",
        
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
        "ph_optimal": "**Оптимальный диапазон.**",
        "ph_insufficient": "**Недостаточное закисление.**",
        
        # Анализ облепихи
        "seabuck_title": "🔬 Влияние экстракта облепихи на качество жая и формованного мяса",
        "seabuck_desc": "Результаты экспериментального исследования",
        "table1_title": "Таблица 1. Основные показатели копчёной жая (контроль и 5% экстракта)",
        "table2_title": "Таблица 2. Основные показатели формованного мясного продукта (контроль и 3% экстракта)",
        "indicator": "Показатель",
        "control": "Контроль (0%)",
        "with_extract_5": "Жая + 5% экстракта",
        "with_extract_3": "Формованное мясо + 3% экстракта",
        
        # Исследование данных
        "explore_title": "🗂️ Исследование исходных данных",
        "explore_desc": "Выберите таблицу для просмотра.",
        "select_data": "Выберите данные:",
        "viewing_data": "Просмотр данных из:",
        
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
        "ml_title": "🧠 ML: Обучение и прогнозирование pH",
        "ml_desc": "Загрузите CSV/Excel с колонкой 'pH' и признаками для обучения или загрузите CSV с признаками для предсказания.",
        "train_tab": "Обучение",
        "predict_tab": "Прогноз",
        "train_subtitle": "Обучение модели",
        "upload_train": "CSV/Excel для обучения (колонка pH)",
        "preview": "Превью:",
        "target_column": "Целевая колонка (pH) выберите:",
        "features": "Признаки (если пусто — будут взяты все числовые кроме цели)",
        "train_button": "Обучить модель",
        "train_success": "Обучение прошло успешно",
        "train_error": "Ошибка обучения:",
        
        "predict_subtitle": "Прогнозирование",
        "upload_predict": "CSV для предсказания (те же признаки)",
        "auto_features": "Автоматически выбранные числовые признаки:",
        "predict_button": "Сделать прогноз",
        "predict_results": "Результаты предсказания",
        "save_to_db": "Сохранить предсказания в базу (sample_name -> sample)",
        "saved_records": "Сохранено записей в БД",
        
        # Ввод данных
        "input_title": "➕ Ввод новых данных о продукции",
        "input_subtitle": "Добавление нового производственного цикла в базу данных",
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
        
        # pH статусы
        "ph_in_normal": "pH в норме",
        "ph_too_low": "pH слишком низкий",
        "ph_too_high": "pH слишком высокий",
        "anim_good": "✅ Всё в порядке",
        "anim_bad": "⚠️ Требуется корректировка",
    },

    # --------------------------------------------------------------------------
    # Английский (en) - (unchanged)
    # --------------------------------------------------------------------------
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
        
        # Home Page
        "home_title": "🐎 Digital Platform for Jaya Production and Modeling",
        "home_desc": "This system combines the description of technological processes and interactive mathematical models for analyzing and predicting product quality.",
        "home_info": "Select a section from the menu on the left to start.",
        
        # Production Process
        "prod_title": "🍖 Technological Map of Jaya Production",
        "prod_subtitle": "Step-by-step quality control and process parameters",
        "stage_1": "1. Raw Material Acceptance 🥩",
        "stage_2": "2. Salting and Massaging 🧂",
        "stage_3": "3. Thermal Processing 🔥",
        "stage_4": "4. Storage and Packaging 📦",
        
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
        "deboning_packaging": "Deboning and Packaging",
        "shelf_life": "Shelf Life and Product Yield",
        "storage_standard": "Standard",
        "storage_freeze": "Freezing",
        
        # Regression Models
        "regression_title": "📊 Regression Models for Final Product Quality",
        "regression_subtitle": "Quality prediction based on technological parameters",
        "scientific_basis": "ℹ️ Scientific Basis",
        "scientific_desc": "Mathematical models allow predicting key quality indicators of the finished product",
        
        "moisture_title": "1. Final Product Moisture (W)",
        "drying_temp": "Drying Temperature (T), °C",
        "drying_duration": "Drying Duration (H), hour",
        "extract_conc": "Extract Concentration (E), %",
        "predicted_moisture": "Predicted Moisture (W), %",
        "moisture_info": "The addition of extract (E) positively affects water retention, while drying duration (H) reduces moisture.",
        
        "water_activity_title": "2. Water Activity (Aw)",
        "salt_conc": "Salt Concentration (C), %",
        "salting_duration": "Salting Duration (Ts), days",
        "predicted_aw": "Predicted Water Activity (Aw)",
        "aw_info": "Optimal Aw (0.88-0.90) is critical for microbiological safety and extending shelf life.",
        
        "color_title": "3. Color Stability (ΔE)",
        "color_info": "The Color Stability model describes how product color changes",
        
        "strength_title": "4. Mechanical Strength (Formed Products)",
        "strength_info": "The model describes the density and elasticity of the product",
        "strength_simulator": "🛠️ Interactive Strength Simulator",
        "pressure": "Pressing Pressure (P), kg/cm²",
        "viscosity": "Minced Meat Viscosity (V), conventional units",
        "strength_index": "Predicted Mechanical Strength Index (Conv. units)",
        
        "recommendations_title": "5. Practical Recommendations for Sea Buckthorn Extract Addition",
        "optimal_conc": "🎯 Optimal Extract Concentration",
        "for_whole_muscle": "For whole-muscle Jaya (smoked)",
        "for_formed": "For formed meat product",
        
        # pH Modeling
        "ph_title": "🌡️ pH Modeling during Salting",
        "ph_subtitle": "Prediction of acidity kinetics for safety assurance",
        "ph_basis": "ℹ️ Scientific Basis of pH Modeling",
        "ph_formula_title": "pH Kinetics Formula (Salting Submodel)",
        "ph_initial": "Initial pH (pH0)",
        "ph_final": "Final pH (pH_inf)",
        "rate_constant": "Rate Constant (k)",
        "forecast_time": "Forecast Time (t), hour",
        "predicted_ph": "Predicted pH at a given time",
        "ph_kinetics": "Visualization of pH Kinetics",
        
        "ph_critical_low": "**Critical acidification.** Product is too acidic.",
        "ph_optimal": "**Optimal range.**",
        "ph_insufficient": "**Insufficient acidification.**",
        
        # Sea Buckthorn Analysis
        "seabuck_title": "🔬 Influence of Sea Buckthorn Extract on Jaya and Formed Meat Quality",
        "seabuck_desc": "Results of experimental research",
        "table1_title": "Table 1. Main indicators of smoked Jaya (control and 5% extract)",
        "table2_title": "Table 2. Main indicators of formed meat product (control and 3% extract)",
        "indicator": "Indicator",
        "control": "Control (0%)",
        "with_extract_5": "Jaya + 5% extract",
        "with_extract_3": "Formed meat + 3% extract",
        
        # Data Exploration
        "explore_title": "🗂️ Raw Data Exploration",
        "explore_desc": "Select a table to view.",
        "select_data": "Select Data:",
        "viewing_data": "Viewing data from:",
        
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
        "ml_title": "🧠 ML: pH Training and Prediction",
        "ml_desc": "Upload CSV/Excel with a 'pH' column and features for training, or upload CSV with features for prediction.",
        "train_tab": "Training",
        "predict_tab": "Prediction",
        "train_subtitle": "Model Training",
        "upload_train": "CSV/Excel for Training (pH column)",
        "preview": "Preview:",
        "target_column": "Target Column (pH) select:",
        "features": "Features (if empty — all numerical except target will be used)",
        "train_button": "Train Model",
        "train_success": "Training successful",
        "train_error": "Training error:",
        
        "predict_subtitle": "Prediction",
        "upload_predict": "CSV for Prediction (same features)",
        "auto_features": "Automatically selected numerical features:",
        "predict_button": "Make Prediction",
        "predict_results": "Prediction Results",
        "save_to_db": "Save predictions to database (sample_name -> sample)",
        "saved_records": "Records saved to DB",
        
        # Data Input
        "input_title": "➕ New Product Data Input",
        "input_subtitle": "Adding a new production batch to the database",
        "batch_params": "Enter parameters for the new production batch",
        "batch_id": "Batch ID (automatic)",
        "mass": "Batch Mass (kg)",
        "initial_temp": "Initial Temperature (°C)",
        "salt_content": "Salt Content (%)",
        "moisture": "Moisture (%)",
        "starter_culture": "Starter Culture (CFU/g)",
        "extract_content": "Extract Concentration (%)",
        "save_data": "💾 Save Data",
        "batch_added": "✅ New batch successfully added",
        "save_error": "❌ Error writing to file:",
        "current_data": "📊 Current Data",
        
        # pH Statuses
        "ph_in_normal": "pH is normal",
        "ph_too_low": "pH is too low",
        "ph_too_high": "pH is too high",
        "anim_good": "✅ Everything is fine",
        "anim_bad": "⚠️ Correction needed",
    },

    # --------------------------------------------------------------------------
    # Казахский (kk)
    # --------------------------------------------------------------------------
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
        "menu_production_process": "Жай өнімін өндіру процесі",
        "menu_regression_models": "Сапаның регрессиялық модельдері",
        "menu_ph_modeling": "pH модельдеу",
        "menu_seabuckthorn_analysis": "Шырғанақ сығындысымен талдау",
        "menu_data_exploration": "Деректерді зерттеу",
        "menu_history_db": "Тарих / ДБ",
        "menu_ml_train_predict": "ML: Оқыту / Болжау",
        "menu_new_data_input": "Жаңа деректерді енгізу",
        
        # Басты бет
        "home_title": "🐎 Жай өнімін өндіру және модельдеуге арналған сандық платформа",
        "home_desc": "Бұл жүйе технологиялық процестерді сипаттауды және өнім сапасын талдау мен болжауға арналған интерактивті математикалық модельдерді біріктіреді.",
        "home_info": "Жұмысты бастау үшін сол жақтағы мәзірден бөлімді таңдаңыз.",
        
        # Өндіріс процесі
        "prod_title": "🍖 Жай өнімін өндірудің технологиялық картасы",
        "prod_subtitle": "Сапаны қадамдық бақылау және процесс параметрлері",
        "stage_1": "1. Шикізатты қабылдау 🥩",
        "stage_2": "2. Тұздау және массалау 🧂",
        "stage_3": "3. Термиялық өңдеу 🔥",
        "stage_4": "4. Сақтау және орау 📦",
        
        "stage1_title": "1. Шикізатты қабылдау және дайындау",
        "stage1_params": "Қабылдауды бақылау параметрлері",
        "initial_mass": "Бастапқы масса",
        "raw_temp": "Шикізат температурасы",
        "fat_thickness": "Майдың қалыңдығы",
        "kpi_title": "Негізгі Технологиялық Көрсеткіштер (Жалпы шолу)",
        "yield_target": "Өнім шығымы (Мақсат)",
        "target_temp": "Дайындықтың мақсатты t°",
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
        
        "stage4_title": "4. Сүйектен айыру, Орау және Сақтау",
        "deboning_packaging": "Сүйектен айыру және орау",
        "shelf_life": "Сақтау мерзімі және өнім шығымы",
        "storage_standard": "Стандарт",
        "storage_freeze": "Мұздату",
        
        # Регрессиялық модельдер
        "regression_title": "📊 Соңғы өнім сапасының регрессиялық модельдері",
        "regression_subtitle": "Технологиялық параметрлер негізінде сапаны болжау",
        "scientific_basis": "ℹ️ Ғылыми негіздеме",
        "scientific_desc": "Математикалық модельдер дайын өнімнің негізгі сапа көрсеткіштерін болжауға мүмкіндік береді",
        
        "moisture_title": "1. Соңғы өнімнің ылғалдылығы (W)",
        "drying_temp": "Кептіру температурасы (T), °C",
        "drying_duration": "Кептіру ұзақтығы (H), сағ",
        "extract_conc": "Сығынды концентрациясы (E), %",
        "predicted_moisture": "Болжанған ылғалдылық (W), %",
        "moisture_info": "Сығындыны (E) қосу ылғал ұстауға оң әсер етеді, ал кептіру ұзақтығы (H) ылғалдылықты төмендетеді.",
        
        "water_activity_title": "2. Су белсенділігі (Aw)",
        "salt_conc": "Тұз концентрациясы (C), %",
        "salting_duration": "Тұздау ұзақтығы (Ts), тәулік",
        "predicted_aw": "Болжанған Су белсенділігі (Aw)",
        "aw_info": "Оңтайлы Aw (0.88-0.90) микробиологиялық қауіпсіздік және сақтау мерзімін ұзарту үшін өте маңызды.",
        
        "color_title": "3. Түс тұрақтылығы (ΔE)",
        "color_info": "Түс тұрақтылығы моделі өнім түсінің қалай өзгеретінін сипаттайды",
        
        "strength_title": "4. Механикалық беріктік (қалыптасқан өнімдер)",
        "strength_info": "Модель өнімнің тығыздығы мен серпімділігін сипаттайды",
        "strength_simulator": "🛠️ Интерактивті беріктік симуляторы",
        "pressure": "Басу қысымы (P), кг/см²",
        "viscosity": "Фарш тұтқырлығы (V), шартты бірліктер",
        "strength_index": "Болжанған Механикалық Беріктік Индексі (Шартты бірл.)",
        
        "recommendations_title": "5. Шырғанақ сығындысын қосу бойынша практикалық ұсыныстар",
        "optimal_conc": "🎯 Сығындының оңтайлы концентрациясы",
        "for_whole_muscle": "Бүтін бұлшықетті жай үшін (ысталған)",
        "for_formed": "Қалыптасқан ет өнімі үшін",
        
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
        "ph_optimal": "**Оңтайлы диапазон.**",
        "ph_insufficient": "**Жеткіліксіз қышқылдану.**",
        
        # Шырғанақ талдауы
        "seabuck_title": "🔬 Шырғанақ сығындысының жай және қалыптасқан ет сапасына әсері",
        "seabuck_desc": "Эксперименттік зерттеу нәтижелері",
        "table1_title": "Кесте 1. Ысталған жайдың негізгі көрсеткіштері (бақылау және 5% сығынды)",
        "table2_title": "Кесте 2. Қалыптасқан ет өнімінің негізгі көрсеткіштері (бақылау және 3% сығынды)",
        "indicator": "Көрсеткіш",
        "control": "Бақылау (0%)",
        "with_extract_5": "Жай + 5% сығынды",
        "with_extract_3": "Қалыптасқан ет + 3% сығынды",
        
        # Деректерді зерттеу
        "explore_title": "🗂️ Бастапқы деректерді зерттеу",
        "explore_desc": "Көру үшін кестені таңдаңыз.",
        "select_data": "Деректерді таңдаңыз:",
        "viewing_data": "Деректерді қарау:",
        
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
        "ml_title": "🧠 ML: pH оқыту және болжау",
        "ml_desc": "Оқыту үшін 'pH' бағаны және белгілері бар CSV/Excel жүктеңіз немесе болжау үшін белгілері бар CSV жүктеңіз.",
        "train_tab": "Оқыту",
        "predict_tab": "Болжам",
        "train_subtitle": "Модельді оқыту",
        "upload_train": "Оқытуға арналған CSV/Excel (pH бағаны)",
        "preview": "Алдын ала қарау:",
        "target_column": "Мақсатты баған (pH) таңдаңыз:",
        "features": "Белгілер (егер бос болса — мақсаттан басқа барлық сандық белгілер алынады)",
        "train_button": "Модельді оқыту",
        "train_success": "Оқыту сәтті өтті",
        "train_error": "Оқыту қатесі:",
        
        "predict_subtitle": "Болжау",
        "upload_predict": "Болжауға арналған CSV (бірдей белгілер)",
        "auto_features": "Автоматты түрде таңдалған сандық белгілер:",
        "predict_button": "Болжам жасау",
        "predict_results": "Болжам нәтижелері",
        "save_to_db": "Болжамдарды деректер базасына сақтау (sample_name -> sample)",
        "saved_records": "ДБ-да сақталған жазбалар",
        
        # Деректерді енгізу
        "input_title": "➕ Жаңа өнім деректерін енгізу",
        "input_subtitle": "Деректер базасына жаңа өндіріс партиясын қосу",
        "batch_params": "Жаңа өндіріс партиясының параметрлерін енгізіңіз",
        "batch_id": "Партия ID (автоматты)",
        "mass": "Партия массасы (кг)",
        "initial_temp": "Бастапқы температура (°C)",
        "salt_content": "Тұз құрамы (%)",
        "moisture": "Ылғалдылық (%)",
        "starter_culture": "Стартерлік дақыл (КОЕ/г)",
        "extract_content": "Сығынды концентрациясы (%)",
        "save_data": "💾 Деректерді сақтау",
        "batch_added": "✅ Жаңа партия сәтті қосылды",
        "save_error": "❌ Файлға жазу қатесі:",
        "current_data": "📊 Ағымдағы деректер",
        
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

# ---------------------------
# Download / export helper
# ---------------------------
def download_link(df: pd.DataFrame, filename: str = "export.csv", label: Optional[str] = None):
    """
    Creates an HTML download link for a DataFrame (UTF-8-sig).
    Returns HTML string that can be used with st.markdown(..., unsafe_allow_html=True)
    """
    if df is None or df.empty:
        if label is None:
            label = "Empty"
        return f"<span>{label}</span>"
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    if label is None:
        label = f"Download {filename}"
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

# ---------------------------
# pH timeseries plot (plotly)
# ---------------------------
def plot_ph_timeseries(df: pd.DataFrame, t_col: str = 'created_at', ph_col: str = 'ph', title: Optional[str] = None, lang: str = "ru"):
    """
    Plot interactive pH timeseries using Plotly.
    - Clips y-axis to [0, 14] by default, but focuses on realistic range.
    - df must contain t_col and ph_col.
    """
    if df is None or df.empty:
        st.info(get_text("no_data", lang))
        return

    # Ensure datetime for x-axis
    df = df.copy()
    if t_col in df.columns:
        try:
            df[t_col] = pd.to_datetime(df[t_col])
        except Exception:
            pass

    if title is None:
        title = get_text("ph_graph_title", lang)

    fig = px.line(df.sort_values(t_col), x=t_col, y=ph_col, title=title, markers=True)
    # Focus on 0..8 range for meat pH usually, but allow expand
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

    # default messages
    normal_msg = get_text("ph_in_normal", lang)
    low_msg = get_text("ph_too_low", lang)
    high_msg = get_text("ph_too_high", lang)
    anim_good = get_text("anim_good", lang)
    anim_bad = get_text("anim_bad", lang)

    # CSS + HTML templates
    css_base = """
    <style>
    .ph-card {{
      border-radius: 12px;
      padding: 12px 18px;
      display: inline-block;
      font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      transition: transform 0.25s ease, box-shadow 0.25s ease;
      margin: 8px 0;
    }}
    /* good pulse */
    .good {{
      background: linear-gradient(90deg, rgba(68,204,68,0.12), rgba(68,204,68,0.06));
      border: 1px solid rgba(68,204,68,0.18);
      animation: gentlePulse 1.6s infinite;
    }}
    @keyframes gentlePulse {{
      0% {{ box-shadow: 0 6px 18px rgba(68,204,68,0.06); transform: translateY(0px); }}
      50% {{ box-shadow: 0 10px 26px rgba(68,204,68,0.12); transform: translateY(-4px); }}
      100% {{ box-shadow: 0 6px 18px rgba(68,204,68,0.06); transform: translateY(0px); }}
    }}
    /* warn pulse */
    .warn {{
      background: linear-gradient(90deg, rgba(255,170,0,0.12), rgba(255,170,0,0.04));
      border: 1px solid rgba(255,170,0,0.18);
      animation: warnPulse 1.1s infinite;
    }}
    @keyframes warnPulse {{
      0% {{ transform: translateY(0px); }}
      50% {{ transform: translateY(-3px); }}
      100% {{ transform: translateY(0px); }}
    }}
    /* bad shake */
    .bad {{
      background: linear-gradient(90deg, rgba(255,68,68,0.12), rgba(255,68,68,0.04));
      border: 1px solid rgba(255,68,68,0.18);
      animation: shake 0.7s infinite;
    }}
    @keyframes shake {{
      0% {{ transform: translateX(0px); }}
      20% {{ transform: translateX(-5px); }}
      40% {{ transform: translateX(5px); }}
      60% {{ transform: translateX(-4px); }}
      80% {{ transform: translateX(4px); }}
      100% {{ transform: translateX(0px); }}
    }}
    .ph-value {{
      font-weight: 700;
      font-size: 1.6rem;
    }}
    .ph-emoji {{
      font-size: 1.6rem;
      margin-right: 8px;
    }}
    .ph-msg {{
      font-size: 1rem;
      margin-top: 6px;
      color: #333;
    }}
    </style>
    """

    # If phv is None, return neutral box
    if phv is None:
        html = css_base + f"""
        <div class="ph-card" style="background:#f4f4f4;border:1px solid #eee;">
            <div><span class="ph-value">—</span></div>
            <div class="ph-msg">No pH value</div>
        </div>
        """
        return html

    # Determine state
    if low_bound <= phv <= high_bound:
        # good
        emoji = "✅"
        state = "good"
        msg = f"{anim_good} — {normal_msg}"
        color = "#44cc44"
    elif phv < low_bound:
        # too low -> bad
        emoji = "🛑"
        state = "bad"
        msg = f"{anim_bad} — {low_msg} ({phv:.2f})"
        color = "#ff4444"
    else:
        # too high -> warn
        emoji = "⚠️"
        state = "warn"
        msg = f"{anim_bad} — {high_msg} ({phv:.2f})"
        color = "#ffaa00"

    html = css_base + f"""
    <div class="ph-card {state}">
        <div style="display:flex; align-items:center;">
            <div class="ph-emoji">{emoji}</div>
            <div>
                <div class="ph-value" style="color:{color};">{phv:.2f}</div>
                <div class="ph-msg">{msg}</div>
            </div>
        </div>
    </div>
    """
    return html

# ---------------------------
# Optional: small helper to render slider with animated label (Streamlit limitation)
# Note: Streamlit's native slider cannot be animated via CSS directly.
# ph_animation_style should be shown near the slider.
# ---------------------------

# ---------------------------
# End of ui_components.py
# ---------------------------
