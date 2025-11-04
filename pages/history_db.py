# Meat_Digitalization/pages/history_db.py
import streamlit as st
import plotly.express as px
from ui import get_text, df_to_download_link
from database import fetch_measurements, delete_all_measurements

def show_history_db(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    st.title(get_text("db_title", lang_choice))
    st.markdown(f"### {get_text('db_desc', lang_choice)}")

    df_db = fetch_measurements()

    if df_db.empty:
        st.info(get_text("history_empty", lang_choice))
    else:
        st.subheader(f"{get_text('total_records', lang_choice)} {len(df_db)}")
        st.dataframe(df_db, use_container_width=True)

        fig_db = px.line(
            df_db.sort_values('created_at'),
            x='created_at',
            y='ph',
            color='sample_name',
            title=get_text("ph_over_time", lang_choice),
            template='plotly_dark'
        )
        st.plotly_chart(fig_db, use_container_width=True)

        st.markdown("---")
        col_dl, col_del = st.columns(2)

        with col_dl:
            st.markdown(df_to_download_link(df_db, "measurements_export.csv", get_text("export_all", lang_choice)),
                        unsafe_allow_html=True)

        with col_del:
            if st.button(f"❌ {get_text('clear_all', lang_choice)}", key="db_reset"):
                if st.session_state.get('confirm_reset', False):
                    delete_all_measurements()
                    st.session_state['confirm_reset'] = False
                    st.success(get_text("db_cleared", lang_choice))
                    st.experimental_rerun()
                else:
                    st.session_state['confirm_reset'] = True
                    st.warning(get_text("confirm_clear", lang_choice))
                    st.button(get_text("confirm_clear", lang_choice), key="confirm_btn")

    st.markdown("</div>", unsafe_allow_html=True)
