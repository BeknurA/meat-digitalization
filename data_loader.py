# Meat_Digitalization/data_loader.py
import pandas as pd
from pathlib import Path
import streamlit as st
import os

# ---------------------------
# Конфигурация файлов/папок
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
MEAT_DATA_XLSX = BASE_DIR / "meat_data.xlsx"
OPYTY_XLSX = BASE_DIR / "opyty.xlsx"
PRODUCTS_CSV = BASE_DIR / "Products.csv"
SAMPLES_CSV = BASE_DIR / "Samples.csv"
MEASUREMENTS_CSV = BASE_DIR / "Measurements.csv"
SHEET_NAME = "T6"


def safe_read_excel(path, sheet_name):
    """
    Безопасно читает лист Excel. Если файл или лист не существуют, создает новый DataFrame.
    """
    if os.path.exists(path):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except ValueError:
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


def append_row_excel(path, sheet_name, new_row):
    """
    Добавляет новую строку в указанный лист Excel.
    """
    df = safe_read_excel(path, sheet_name)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def safe_read_csv(path: Path):
    """
    Безопасно читает CSV-файл, пробуя различные кодировки.
    """
    if not path.exists():
        return pd.DataFrame()
    encodings = ['utf-8-sig', 'utf-8', 'windows-1251', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.DataFrame()


@st.cache_data
def load_all_data():
    """
    Загружает все данные из файлов Excel и CSV.
    Кэширует результат для производительности.
    """
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

    data_sheets = {}
    try:
        if MEAT_DATA_XLSX.exists():
            xls = pd.ExcelFile(MEAT_DATA_XLSX)
            for sheet in xls.sheet_names:
                data_sheets[sheet] = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
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
