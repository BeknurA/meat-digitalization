# ui_components.py
import streamlit as st
import pandas as pd
import base64
import plotly.express as px
import numpy as np
from typing import Optional

# ---------------------------
# Multilanguage strings (RU / EN / KK)
# keys are designed to be reused across app
# ---------------------------
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
        "predict_result": "Результаты предсказания",
        "ph_in_normal": "pH в норме",
        "ph_too_low": "pH слишком низкий",
        "ph_too_high": "pH слишком высокий",
        "save": "Сохранить",
        "saved": "Сохранено",
        "download": "Скачать",
        "ph_graph_title": "Динамика pH",
        "history_title": "История измерений",
        "no_history": "История пуста",
        "slider_label_ph": "Уровень pH",
        "slider_hint_low": "Нижняя граница безопасного диапазона",
        "slider_hint_high": "Верхняя граница безопасного диапазона",
        "anim_good": "✅ Всё в порядке",
        "anim_bad": "⚠️ Требуется корректировка",
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
        "predict_result": "Prediction results",
        "ph_in_normal": "pH is normal",
        "ph_too_low": "pH too low",
        "ph_too_high": "pH too high",
        "save": "Save",
        "saved": "Saved",
        "download": "Download",
        "ph_graph_title": "pH Time Series",
        "history_title": "Measurement history",
        "no_history": "History is empty",
        "slider_label_ph": "pH level",
        "slider_hint_low": "Lower safe bound",
        "slider_hint_high": "Upper safe bound",
        "anim_good": "✅ All good",
        "anim_bad": "⚠️ Adjustment needed",
    },
    "kk": {
        "title": "Meat Digitalization платформасы",
        "upload_csv": "CSV/Excel жүктеу",
        "train": "Модельді үйрету",
        "predict": "pH болжамы",
        "history": "Өлшемдер тарихы",
        "export": "CSV жүктеу",
        "no_data": "Көрсетуге деректер жоқ",
        "train_success": "Модель үйретілді. Метрикалар:",
        "train_fail": "Үйретуде қате",
        "predict_result": "Болжам нәтижелері",
        "ph_in_normal": "pH қалыпты",
        "ph_too_low": "pH тым төмен",
        "ph_too_high": "pH тым жоғары",
        "save": "Сақтау",
        "saved": "Сақталды",
        "download": "Жүктеу",
        "ph_graph_title": "pH уақыттық динамикасы",
        "history_title": "Өлшемдер тарихы",
        "no_history": "Тарих бос",
        "slider_label_ph": "pH деңгейі",
        "slider_hint_low": "Қауіпсіз төменгі шекара",
        "slider_hint_high": "Қауіпсіз жоғарғы шекара",
        "anim_good": "✅ Барлығы жақсы",
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
