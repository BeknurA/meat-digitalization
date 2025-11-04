# Meat_Digitalization/pages/data_exploration.py
import streamlit as st
import pandas as pd
from ui import get_text
from data_loader import load_all_data


def show_data_exploration(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("explore_title", lang_choice))
    st.write(get_text("explore_desc", lang_choice))

    all_meat_data, df_ph_raw, _, _, _ = load_all_data()

    df_to_use_for_ph = df_ph_raw

    if all_meat_data:
        available_tables = list(all_meat_data.keys())

        if df_to_use_for_ph is not None:
            available_tables.append('opyty.xlsx')

        choice = st.selectbox(get_text("select_data", lang_choice), available_tables)
        st.markdown(f"**{get_text('viewing_data', lang_choice)} `{choice}`**")

        if choice == 'opyty.xlsx':
            if df_to_use_for_ph is not None:
                df_to_show = df_to_use_for_ph.copy()
            else:
                df_to_show = pd.DataFrame()
        else:
            df_to_show = all_meat_data.get(choice, pd.DataFrame()).copy()

        if 'Accuracy' in df_to_show.columns:
            df_to_show['Accuracy'] = pd.to_numeric(df_to_show['Accuracy'], errors='coerce')

        if not df_to_show.empty:
            st.dataframe(df_to_show)
        else:
            st.warning(
                f"{get_text('viewing_data', lang_choice)} '{choice}' — {get_text('data_empty_warning', lang_choice)}")
    else:
        st.warning(get_text("data_load_error", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)
