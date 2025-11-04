# app.py
import streamlit as st
from ui import get_text, LANG
from pages.home   import show_home
from pages.production  import show_production_process
from pages.regression import show_regression_models
from pages.ph_modeling import show_ph_modeling
from pages.seabuckthorn import show_seabuckthorn_analysis
from pages.data_exploration import show_data_exploration
from pages.history_db import show_history_db
from pages.ml_training import show_ml_train_predict
from pages.new_data_input import show_new_data_input

# ---------------------------
# Установки страницы
# ---------------------------
# Используем layout="wide" для лучшего вида, но главное - CSS
st.set_page_config(page_title="Платформа Жая — расширенная", layout="wide")

# =================================================================
# 🎨 ENHANCED DESIGN AND ANIMATION - DARK THEME
# =================================================================
st.markdown("""
<style>
/* 1. Global & Page Config */
.stApp {
    background-color: #111111; /* DARK/Black background */
    color: #f0f0f0; /* Light text for general readability */
}
/* ... (весь остальной CSS код ) ... */
/* Ensure all text within containers is readable */
.st-emotion-cache-1n76c1k, [data-testid="stSidebar"] div, div[data-testid="stForm"] > div > label > div {
    color: #f0f0f0 !important;
}
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
    color: #f0f0f0 !important;
}
/* 2. Fade-In Animation */
.fade-in {
  animation: fadeIn ease 0.5s;
}
@keyframes fadeIn {
  0% {opacity:0; transform:translateY(6px)}
  100% {opacity:1; transform:translateY(0)}
}
.small-muted {color: #a0a0a0; font-size:0.9em;}
/* 3. Title Animation */
.main-title-animation {
    animation: pulseTitle 1.5s infinite alternate;
    color: #4dc4ff; /* Accent Blue */
    text-shadow: 1px 1px 4px rgba(0,0,0,0.4);
}
@keyframes pulseTitle {
    0% { transform: scale(1.0); opacity: 0.95; }
    100% { transform: scale(1.005); opacity: 1.0; }
}
/* 4. Customizing Sidebar */
[data-testid="stSidebar"] {
    background-color: #1f1f1f;
    box-shadow: 2px 0px 8px rgba(0,0,0,0.5);
}
/* 5. Metric Cards Styling */
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stMetric"] {
    background-color: #2a2a2a;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    border-left: 5px solid #0d6efd;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    transition: all 0.3s;
}
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.6);
}
.stMetric .st-bd {
    font-size: 2em !important;
    font-weight: 700 !important;
    color: #0d6efd; /* Accent Blue */
}
/* --- NEW CARD STYLE for Key Findings --- */
.key-finding-card {
    background-color: #2a2a2a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    transition: transform 0.3s ease;
    border: 1px solid #333;
}
.key-finding-card:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.6);
}
.key-finding-card h4 {
    color: #4dc4ff !important;
    font-size: 1.2em;
    border-bottom: 2px solid #343a40;
    padding-bottom: 5px;
    margin-top: 0;
}
.key-value {
    font-size: 2.2em;
    font-weight: 700;
    color: #198754; /* Green for success */
}
/* 6. Step Buttons Styling */
.stButton button[key*="btn_"] {
    background-color: #495057;
    color: white;
    border-radius: 5px;
    border: none;
    transition: all 0.3s ease;
    padding: 10px 15px;
    font-weight: 600;
}
.stButton button[key*="btn_"]:hover {
    background-color: #6c757d;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
}
/* Highlight active button */
.stButton button[aria-pressed="true"] {
    background-color: #198754 !important;
    border-color: #198754 !important;
    box-shadow: 0 3px 6px rgba(25, 135, 84, 0.6) !important;
}
/* 7. pH Pulse Animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(25, 135, 84, 0); }
    100% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0); }
}

/* 8. НОВЫЙ CSS: СКРЫТЬ АВТОМАТИЧЕСКОЕ МЕНЮ СТРАНИЦ */
[data-testid="stSidebarNav"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# UI: Main navigation
# ---------------------------
L = LANG
lang_codes = list(L.keys())
if not lang_codes:
    lang_codes = ["ru"]

_lang_name_map = {
    "ru": "Русский",
    "en": "English",
    "kk": "Қазақша",
}
lang_names = [ _lang_name_map.get(code, code) for code in lang_codes ]

default_lang = "ru" if "ru" in lang_codes else lang_codes[0]

if "lang_choice" not in st.session_state:
    st.session_state.lang_choice = default_lang

try:
    current_index = lang_codes.index(st.session_state.lang_choice)
except ValueError:
    current_index = 0

# Элементы, которые ВЫ хотите видеть на боковой панели:
# 1. Выбор языка
selected_name = st.sidebar.selectbox("Язык / Language", lang_names, index=current_index)

selected_code = lang_codes[lang_names.index(selected_name)]
st.session_state.lang_choice = selected_code
lang_choice = st.session_state.lang_choice

# 2. Заголовок и версия
sidebar_container = st.sidebar.container()
sidebar_container.markdown("<div class='fade-in'>", unsafe_allow_html=True)
sidebar_container.title(get_text("title", lang_choice))
sidebar_container.caption(get_text("version_note", lang_choice))
sidebar_container.markdown("</div>", unsafe_allow_html=True)

# 3. Выбор раздела (Ваша кастомная навигация)
page_options = [
    get_text("menu_home", lang_choice),
    get_text("menu_production_process", lang_choice),
    get_text("menu_regression_models", lang_choice),
    get_text("menu_ph_modeling", lang_choice),
    get_text("menu_seabuckthorn_analysis", lang_choice),
    get_text("menu_data_exploration", lang_choice),
    get_text("menu_history_db", lang_choice),
    get_text("menu_ml_train_predict", lang_choice),
    get_text("menu_new_data_input", lang_choice),
]

page = st.sidebar.radio(get_text("select_section", lang_choice), page_options, index=0)

# (Остальной session_state и роутинг остаются без изменений)
if 'selected_product_id' not in st.session_state:
    st.session_state.selected_product_id = None
if 'selected_step' not in st.session_state:
    st.session_state.selected_step = None
if 'active_stage_clean' not in st.session_state:
    st.session_state['active_stage_clean'] = 'priemka'

# =================================================================
# Page Routing
# =================================================================
if page == get_text("menu_home", lang_choice):
    show_home(lang_choice)
elif page == get_text("menu_production_process", lang_choice):
    show_production_process(lang_choice)
elif page == get_text("menu_regression_models", lang_choice):
    show_regression_models(lang_choice)
elif page == get_text("menu_ph_modeling", lang_choice):
    show_ph_modeling(lang_choice)
elif page == get_text("menu_seabuckthorn_analysis", lang_choice):
    show_seabuckthorn_analysis(lang_choice)
elif page == get_text("menu_data_exploration", lang_choice):
    show_data_exploration(lang_choice)
elif page == get_text("menu_history_db", lang_choice):
    show_history_db(lang_choice)
elif page == get_text("menu_ml_train_predict", lang_choice):
    show_ml_train_predict(lang_choice)
elif page == get_text("menu_new_data_input", lang_choice):
    show_new_data_input(lang_choice)