# ui_components.py
import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import numpy as np

LANG = {
    "ru": {
        "title": "Цифровая платформа — Meat Digitalization",
        "upload_csv": "Загрузить CSV/Excel",
        "train": "Обучить модель",
        "predict": "Сделать прогноз",
        "history": "История измерений",
        "export": "Скачать CSV",
        "no_data": "Нет данных для отображения",
        "train_success": "Модель обучена. Метрики:",
        "train_fail": "Ошибка обучения",
        "predict_result": "Результаты предсказания"
    },
    "en": {
        "title": "Meat Digitalization Platform",
        "upload_csv": "Upload CSV/Excel",
        "train": "Train model",
        "predict": "Predict pH",
        "history": "Measurement history",
        "export": "Download CSV",
        "no_data": "No data to display",
        "train_success": "Model trained. Metrics:",
        "train_fail": "Train error",
        "predict_result": "Prediction results"
    },
    "kk": {
        "title": "Meat Digitalization платформа",
        "upload_csv": "CSV/Excel жүктеу",
        "train": "Модельді үйрету",
        "predict": "pH болжамы",
        "history": "Өлшемдер тарихы",
        "export": "CSV жүктеу",
        "no_data": "Көрсетуге деректер жоқ",
        "train_success": "Модель үйретілді. Метрикалар:",
        "train_fail": "Үйретуде қате",
        "predict_result": "Болжам нәтижелері"
    }
}

def download_link(df, filename="export.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Скачать {filename}</a>'

def plot_ph_timeseries(df, t_col='created_at', ph_col='ph', title="pH over time"):
    if df is None or df.empty:
        st.info("Нет данных для графика")
        return
    fig = px.line(df.sort_values(t_col), x=t_col, y=ph_col, title=title, markers=True)
    fig.update_yaxes(range=[0, 8])  # realistic pH window; adjust if needed
    st.plotly_chart(fig, use_container_width=True)

def smooth_array(arr, window=3):
    if len(arr) < window: return arr
    return np.convolve(arr, np.ones(window)/window, mode='same')
