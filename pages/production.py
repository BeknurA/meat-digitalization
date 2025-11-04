# Meat_Digitalization/pages/production.py
import streamlit as st
import pandas as pd
from ui import get_text

def show_production_process(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("jaya_process_title", lang_choice))
    st.markdown(get_text("jaya_process_subtitle", lang_choice))

    # Кнопки этапов (динамически локализованные)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(get_text("stage_priemka", lang_choice), key='btn_priemka'):
            st.session_state['active_stage_clean'] = 'priemka'
    with col2:
        if st.button(get_text("stage_posol", lang_choice), key='btn_posol'):
            st.session_state['active_stage_clean'] = 'posol'
    with col3:
        if st.button(get_text("stage_termo", lang_choice), key='btn_termo'):
            st.session_state['active_stage_clean'] = 'termokamera'
    with col4:
        if st.button(get_text("stage_upakovka", lang_choice), key='btn_upakovka'):
            st.session_state['active_stage_clean'] = 'upakovka'

    st.markdown("---")
    active_stage = st.session_state.get('active_stage_clean', 'priemka')

    # --------------------------
    # 1. Приемка
    # --------------------------
    if active_stage == 'priemka':
        st.header(get_text("stage_priemka_header", lang_choice))
        with st.expander(get_text("stage_priemka_expander", lang_choice), expanded=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric(
            label=get_text("metric_mass", lang_choice),
            value=f"1 {get_text('unit_kg', lang_choice)}",)
            col_p2.metric(label=get_text("metric_temp", lang_choice), value="0-3°С", help=get_text("help_temp", lang_choice))
            col_p3.metric(label=get_text("metric_ph", lang_choice), value="6.5-6.8", help=get_text("help_ph", lang_choice))
            st.markdown(get_text("tech_params_title", lang_choice))
            col_kpi_a, col_kpi_b, col_kpi_c = st.columns(3)
            col_kpi_a.metric(label=get_text("metric_yield", lang_choice), value="85%", delta=get_text("delta_gost", lang_choice))
            col_kpi_b.metric(label=get_text("metric_target_temp", lang_choice), value="74°С", delta=get_text("delta_inner", lang_choice))
            col_kpi_c.metric(
            label=get_text("metric_brine_loss", lang_choice),
            value=f"100 {get_text('unit_g', lang_choice)}",
            delta_color="off",)
            st.markdown("---")
            st.info(get_text("digital_control_tip", lang_choice))

    # --------------------------
    # 2. Посол
    # --------------------------
    elif active_stage == 'posol':
        st.header(get_text("stage_posol_header", lang_choice))
        with st.expander(get_text("stage_posol_expander1", lang_choice), expanded=True):
            st.markdown(get_text("stage_posol_markdown1", lang_choice), unsafe_allow_html=True)

        with st.expander(get_text("stage_posol_expander2", lang_choice), expanded=False):
            st.markdown(get_text("stage_posol_markdown2", lang_choice))

    # --------------------------
    # 3. Термообработка
    # --------------------------
    elif active_stage == 'termokamera':
        st.header(get_text("stage_termo_header", lang_choice))
        st.info(get_text("stage_termo_info", lang_choice))

        termoparameters = [
            (get_text("termo_drying", lang_choice), "45°С", "20 мин", get_text("termo_drying_desc", lang_choice)),
            (get_text("termo_frying", lang_choice), "75-85°С", get_text("termo_frying_crit", lang_choice), get_text("termo_frying_desc", lang_choice)),
            (get_text("termo_steam", lang_choice), get_text("termo_steam_camtemp", lang_choice), get_text("termo_steam_inner", lang_choice),
             get_text("termo_steam_desc", lang_choice)),
            (get_text("termo_cool_dry", lang_choice), get_text("termo_cool_temp", lang_choice), "10 мин", get_text("termo_cool_desc", lang_choice)),
            (get_text("termo_smoke", lang_choice), "30-33°С", "1.5 ч", get_text("termo_smoke_desc", lang_choice))
        ]
        df_termo = pd.DataFrame(termoparameters, columns=[
            get_text("col_stage", lang_choice),
            get_text("col_temp", lang_choice),
            get_text("col_time", lang_choice),
            get_text("col_purpose", lang_choice)
        ])
        st.dataframe(df_termo.set_index(get_text("col_stage", lang_choice)), width=800)

        st.markdown("---")
        st.markdown(get_text("iot_monitoring_desc", lang_choice))

    # --------------------------
    # 4. Упаковка
    # --------------------------
    elif active_stage == 'upakovka':
        st.header(get_text("stage_upakovka_header", lang_choice))
        with st.expander(get_text("stage_upakovka_expander", lang_choice), expanded=True):
            st.markdown(get_text("stage_upakovka_markdown1", lang_choice))

        st.markdown("---")
        st.subheader(get_text("shelf_life_comparison", lang_choice))

        col_s1, col_s2 = st.columns(2)
        col_s1.metric(label=get_text("shelf_life_standard", lang_choice), value=get_text("shelf_life_std_value", lang_choice), delta_color="off")
        col_s2.metric(label=get_text("shelf_life_extract", lang_choice), value=get_text("shelf_life_ext_value", lang_choice), delta=get_text("shelf_life_delta_value", lang_choice))

        st.markdown(get_text("shelf_life_desc", lang_choice))
        st.info(get_text("storage_tip", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)
