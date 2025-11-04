# Meat_Digitalization/pages/ph_modeling.py
import streamlit as st
import numpy as np
import plotly.express as px
from ui import get_text

def ph_model_func(t, pH0=6.6, pH_inf=4.6, k=0.03):
    t = np.array(t, dtype=float)
    ph = pH_inf + (pH0 - pH_inf) * np.exp(-k * t)
    ph = np.clip(ph, 0.0, 14.0)
    return ph

def show_ph_modeling(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("ph_title", lang_choice))
    st.markdown(f"### {get_text('ph_subtitle', lang_choice)}")

    with st.expander(get_text("ph_basis", lang_choice), expanded=True):
        st.write(get_text("ph_basis_text", lang_choice))

    st.markdown("---")
    st.subheader(get_text("ph_formula_title", lang_choice))
    st.latex(r"pH(t) = pH_0 - (pH_0 - pH_{\infty}) \cdot (1 - e^{-k \cdot t})")
    st.markdown(get_text("ph_formula_desc", lang_choice))
    st.warning(get_text("ph_formula_tip", lang_choice))
    st.markdown("---")

    # --- Интерактивный прогноз ---
    st.subheader(get_text("ph_forecast_title", lang_choice))
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        pH0 = st.number_input(get_text("ph_initial", lang_choice), value=6.6, format="%.2f")
    with col_b:
        pH_inf = st.number_input(get_text("ph_final", lang_choice), value=4.6, format="%.2f")
    with col_c:
        k = st.number_input(get_text("rate_constant", lang_choice), value=0.03, format="%.4f")

    t_input = st.slider(get_text("forecast_time", lang_choice),
                        min_value=1, max_value=240, value=48, step=1)
    pH_forecast = float(ph_model_func(t_input, pH0=pH0, pH_inf=pH_inf, k=k))

    st.metric(label=get_text("predicted_ph", lang_choice),
              value=f"{pH_forecast:.2f}",
              delta=f"{get_text('delta_target_ph', lang_choice)} {(pH_forecast - 5.6):.2f}",
              delta_color="inverse")

    # --- Классификация диапазона ---
    if pH_forecast < 4.8:
        st.error(get_text("ph_critical_low", lang_choice))
    elif 4.8 <= pH_forecast <= 5.6:
        st.success(get_text("ph_optimal", lang_choice))
    else:
        st.warning(get_text("ph_insufficient", lang_choice))

    st.markdown("---")
    st.subheader(get_text("ph_kinetics", lang_choice))

    times = np.linspace(0, 240, 300)
    pH_values = ph_model_func(times, pH0=pH0, pH_inf=pH_inf, k=k)

    fig = px.line(
        x=times,
        y=pH_values,
        labels={'x': get_text("time_hours", lang_choice), 'y': 'pH'},
        title=get_text("ph_plot_title", lang_choice)
    )
    fig.add_hrect(y0=4.8, y1=5.6, fillcolor="green", opacity=0.08, layer="below", line_width=0)
    fig.add_vline(x=t_input, line_dash="dash",
                  annotation_text=f"{t_input} {get_text('hours_short', lang_choice)}",
                  annotation_position="top right")
    fig.update_yaxes(range=[0, 8])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
