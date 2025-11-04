# Meat_Digitalization/pages/regression.py
import streamlit as st
from ui import get_text

def calculate_stability(pressure, viscosity):
    p, v = pressure, viscosity
    return 27.9 - 0.1 * p - 1.94 * v - 0.75 * p * v - 0.67 * p ** 2 - 2.5 * v ** 2

def show_regression_models(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("regression_title", lang_choice))
    st.markdown(get_text("regression_subtitle", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 1. Влажность конечного продукта
    # -------------------------------
    st.header(get_text("reg_w_title", lang_choice))
    st.latex(r"W = 65.0 + 0.12 \cdot T - 0.05 \cdot H + 0.5 \cdot E")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        T = st.slider(get_text("reg_w_T", lang_choice), min_value=20, max_value=35, value=25, step=1, key="w_T")
    with col_w2:
        H = st.slider(get_text("reg_w_H", lang_choice), min_value=1.0, max_value=10.0, value=5.0, step=0.5, key="w_H")
    with col_w3:
        E = st.slider(get_text("reg_w_E", lang_choice), min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="w_E_model1")

    W_predicted = 65.0 + 0.12 * T - 0.05 * H + 0.5 * E
    st.metric(label=get_text("reg_w_metric", lang_choice), value=f"{W_predicted:.2f}",
              delta=f"{get_text('reg_w_delta', lang_choice)} {W_predicted - 65.0:.2f} п.п.")
    st.info(get_text("reg_w_info", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 2. Активность воды
    # -------------------------------
    st.header(get_text("reg_aw_title", lang_choice))
    st.latex(r"A_w = 0.95 - 0.003 \cdot C - 0.005 \cdot T_s")

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        C = st.slider(get_text("reg_aw_C", lang_choice), min_value=2.0, max_value=6.0, value=4.0, step=0.2, key="a_C")
    with col_a2:
        Ts = st.slider(get_text("reg_aw_Ts", lang_choice), min_value=1.0, max_value=7.0, value=3.0, step=0.5, key="a_Ts")

    Aw_predicted = 0.95 - 0.003 * C - 0.005 * Ts
    st.metric(label=get_text("reg_aw_metric", lang_choice), value=f"{Aw_predicted:.3f}",
              delta=(get_text("reg_aw_delta_high", lang_choice) if Aw_predicted > 0.90 else get_text("reg_aw_delta_ok", lang_choice)))
    st.success(get_text("reg_aw_info", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 3. Цветовая стабильность
    # -------------------------------
    st.header(get_text("reg_color_title", lang_choice))
    st.markdown(get_text("reg_color_desc", lang_choice))
    st.latex(r"\Delta E = 1.80 - 0.20 \cdot E + 0.05 \cdot H")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        E_color = st.slider(get_text("reg_color_E", lang_choice), min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="e_color")
    with col_c2:
        H_color = st.slider(get_text("reg_color_H", lang_choice), min_value=2.0, max_value=10.0, value=5.0, step=0.5, key="h_color")

    Delta_E_predicted = 1.80 - 0.20 * E_color + 0.05 * H_color
    st.metric(label=get_text("reg_color_metric", lang_choice), value=f"{Delta_E_predicted:.2f}",
              delta=get_text("reg_color_delta", lang_choice))

    if Delta_E_predicted < 1.5:
        st.success(get_text("reg_color_result_good", lang_choice))
    elif Delta_E_predicted < 2.5:
        st.warning(get_text("reg_color_result_warn", lang_choice))
    else:
        st.error(get_text("reg_color_result_bad", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 4. Окислительная стабильность (TBC)
    # -------------------------------
    st.header(get_text("reg_tbc_title", lang_choice))
    st.markdown(get_text("reg_tbc_desc", lang_choice))
    st.latex(r"\text{TBC}_{30\text{д}} = 2.80 - 0.35 \cdot E - 0.10 \cdot S")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        E_tbc = st.slider(get_text("reg_tbc_E", lang_choice), min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="e_tbc")
    with col_t2:
        S_tbc = st.slider(get_text("reg_tbc_S", lang_choice), min_value=2.0, max_value=5.0, value=3.5, step=0.1, key="s_tbc")

    TBC_predicted = 2.80 - 0.35 * E_tbc - 0.10 * S_tbc
    st.metric(label=get_text("reg_tbc_metric", lang_choice), value=f"{TBC_predicted:.2f}",
              delta=get_text("reg_tbc_delta", lang_choice))

    if TBC_predicted < 1.0:
        st.success(get_text("reg_tbc_result_good", lang_choice))
    elif TBC_predicted < 1.8:
        st.warning(get_text("reg_tbc_result_warn", lang_choice))
    else:
        st.error(get_text("reg_tbc_result_bad", lang_choice))
    st.markdown("---")

    # -------------------------------
    # 5. Механическая прочность
    # -------------------------------
    st.header(get_text("reg_strength_title", lang_choice))
    st.info(get_text("reg_strength_info", lang_choice))

    with st.expander(get_text("reg_strength_expander", lang_choice), expanded=False):
        col_p_slider, col_v_slider = st.columns(2)
        with col_p_slider:
            P_input = st.slider(get_text("reg_strength_P", lang_choice), min_value=0.5, max_value=2.0, value=1.0, step=0.1, key="p_pressure")
        with col_v_slider:
            V_input = st.slider(get_text("reg_strength_V", lang_choice), min_value=50, max_value=150, value=100, step=10, key="v_viscosity")

        Prochnost_score = calculate_stability(P_input, V_input / 100)
        st.metric(label=get_text("reg_strength_metric", lang_choice), value=f"{Prochnost_score:.2f}")

        if Prochnost_score >= 25:
            st.success(get_text("reg_strength_result_good", lang_choice))
        elif Prochnost_score >= 15:
            st.warning(get_text("reg_strength_result_warn", lang_choice))
        else:
            st.error(get_text("reg_strength_result_bad", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)
