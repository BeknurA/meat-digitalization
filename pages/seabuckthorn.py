# Meat_Digitalization/pages/seabuckthorn.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from ui import get_text

def show_seabuckthorn_analysis(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title(get_text("seabuck_title", lang_choice))
    st.write(get_text("seabuck_desc", lang_choice))
    st.markdown("---")

    # Таблица 1
    st.subheader(get_text("table1_title", lang_choice))
    table1_data = {
        get_text("indicator", lang_choice): [
            get_text("moisture", lang_choice),
            get_text("protein", lang_choice),
            get_text("fat", lang_choice),
            get_text("vus", lang_choice),
            get_text("tbch", lang_choice),
        ],
        get_text("control", lang_choice): [65.2, 21.2, 31.06, 60.2, 0.69],
        get_text("with_extract_5", lang_choice): [67.8, 25.44, 33.4, 67.4, 0.96],
    }
    df_table1 = pd.DataFrame(table1_data)
    st.dataframe(df_table1)

    # Таблица 2
    st.subheader(get_text("table2_title", lang_choice))
    table2_data = {
        get_text("indicator", lang_choice): [
            get_text("moisture", lang_choice),
            get_text("protein", lang_choice),
            get_text("fat", lang_choice),
            get_text("salt", lang_choice),
            get_text("ash", lang_choice),
        ],
        get_text("control", lang_choice): [68.96, 13.60, 11.03, 1.77, 2.96],
        get_text("with_extract_3", lang_choice): [70.08, 13.88, 8.51, 1.27, 2.22],
    }
    df_table2 = pd.DataFrame(table2_data)
    st.dataframe(df_table2)
    st.markdown("---")

    col1, col2 = st.columns(2)
    x_ticks = np.arange(0, 15.1, 2.5)

    with col1:
        st.subheader(get_text("fig1_title", lang_choice))
        x = np.array([0, 3, 5, 7, 9, 15])
        vlaga = np.array([65.2, 66.8, 68.9, 68.6, 67.8, 65.4])
        fig1 = px.line(x=x, y=vlaga, markers=True,
                       title=get_text("fig1_plot_title", lang_choice))
        fig1.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(get_text("fig3_title", lang_choice))
        VUS = np.array([60.2, 64.3, 67.4, 71.2, 73.5, 78.9])
        VSS = np.array([61.0, 65.5, 70.1, 73.8, 75.2, 77.4])
        ZhUS = np.array([60.0, 63.1, 66.8, 70.0, 72.5, 74.8])
        fig3 = px.line(x=x, y=VUS, markers=True,
                       title=get_text("fig3_plot_title", lang_choice))
        fig3.add_scatter(x=x, y=VUS, mode='lines+markers', name='ВУС, %')
        fig3.add_scatter(x=x, y=VSS, mode='lines+markers', name='ВСС, %')
        fig3.add_scatter(x=x, y=ZhUS, mode='lines+markers', name='ЖУС, %')
        fig3.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader(get_text("fig5_title", lang_choice))
        days2 = np.array([5, 10, 15])
        tbch_c2 = np.array([0.203, 0.284, 0.312])
        tbch_e2 = np.array([0.254, 0.366, 0.428])
        perox_c2 = np.array([13.27, 14.30, 15.21])
        perox_e2 = np.array([9.90, 10.80, 11.60])
        fig5 = px.line(title=get_text("fig5_plot_title", lang_choice))
        fig5.add_scatter(x=days2, y=tbch_c2, mode='lines+markers', name='ТБЧ контроль')
        fig5.add_scatter(x=days2, y=tbch_e2, mode='lines+markers', name='ТБЧ 3%')
        fig5.add_scatter(x=days2, y=perox_c2, mode='lines+markers', name='Перокс контроль')
        fig5.add_scatter(x=days2, y=perox_e2, mode='lines+markers', name='Перокс 3%')
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader(get_text("fig2_title", lang_choice))
        belok = np.array([21.2, 23.4, 25.4, 27.5, 29.8, 34.9])
        zhir = np.array([31.06, 32.4, 33.4, 37.1, 41.2, 45.0])
        fig2 = px.line(title=get_text("fig2_plot_title", lang_choice))
        fig2.add_scatter(x=x, y=belok, mode='lines+markers', name='Белок, %')
        fig2.add_scatter(x=x, y=zhir, mode='lines+markers', name='Жир, %')
        fig2.update_xaxes(tickvals=x_ticks)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader(get_text("fig4_title", lang_choice))
        days = np.array([5, 10, 15])
        tbch_c = np.array([0.197, 0.376, 0.416])
        tbch_e = np.array([0.194, 0.361, 0.419])
        perox_c = np.array([17.96, 19.12, 20.25])
        perox_e = np.array([13.01, 14.40, 15.13])
        fig4 = px.line(title=get_text("fig4_plot_title", lang_choice))
        fig4.add_scatter(x=days, y=tbch_c, mode='lines+markers', name='ТБЧ контроль')
        fig4.add_scatter(x=days, y=tbch_e, mode='lines+markers', name='ТБЧ 3%')
        fig4.add_scatter(x=days, y=perox_c, mode='lines+markers', name='Перокс контроль')
        fig4.add_scatter(x=days, y=perox_e, mode='lines+markers', name='Перокс 3%')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
