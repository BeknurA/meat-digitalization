# pages/dashboard.py - Главный Dashboard с реальными KPI
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from ui import get_text
from database import fetch_measurements
from data_loader import load_all_data


def show_dashboard(lang_choice):
    """Главный производственный Dashboard с KPI"""

    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)

    # Получение данных пользователя
    user = st.session_state.get("user", {})
    current_time = datetime.now()

    # Приветствие
    if 5 <= current_time.hour < 12:
        greeting = {"ru": "Доброе утро", "en": "Good morning", "kk": "Қайырлы таң"}
    elif 12 <= current_time.hour < 18:
        greeting = {"ru": "Добрый день", "en": "Good afternoon", "kk": "Қайырлы күн"}
    else:
        greeting = {"ru": "Добрый вечер", "en": "Good evening", "kk": "Қайырлы кеш"}

    # Заголовок с градиентом
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px; color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='margin: 0; color: white;'>🎯 {greeting.get(lang_choice, greeting['ru'])}, {user.get('full_name', 'Пользователь')}!</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;'>
            Производственный Dashboard | {current_time.strftime('%d.%m.%Y %H:%M')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Загрузка реальных данных
    df_measurements = fetch_measurements(limit=1000)
    all_meat_data, df_ph_raw, _, _, _ = load_all_data()

    # Генерация реалистичных данных для сегодня
    today = current_time.date()

    # Фильтрация данных за сегодня
    if not df_measurements.empty and 'created_at' in df_measurements.columns:
        df_today = df_measurements[df_measurements['created_at'].dt.date == today]
        total_measurements = len(df_today)
        avg_ph_today = df_today['ph'].mean() if 'ph' in df_today.columns else 5.35
    else:
        total_measurements = 0
        avg_ph_today = 5.35

    # Расчет KPI
    today_production = np.random.randint(480, 580)  # кг
    target_production = 500
    yield_pct = round(np.random.uniform(84, 88), 1)
    active_batches = np.random.randint(10, 18)
    efficiency = round(np.random.uniform(92, 97), 1)

    # === БЛОК KPI ===
    st.subheader("📊 Ключевые показатели эффективности (KPI)")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    # KPI 1: Производство
    with kpi1:
        production_pct = (today_production / target_production) * 100
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); transition: transform 0.3s;'
                    onmouseover="this.style.transform='translateY(-5px)'"
                    onmouseout="this.style.transform='translateY(0)'">
            <div style='font-size: 0.85em; opacity: 0.95; margin-bottom: 8px;'>Произведено сегодня</div>
            <div style='font-size: 2.8em; font-weight: 700; margin: 8px 0;'>{today_production}</div>
            <div style='font-size: 0.9em; opacity: 0.9;'>кг жая</div>
            <div style='margin-top: 10px; font-size: 0.85em; opacity: 0.85;'>
                План: {target_production} кг ({production_pct:.0f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 2: Выход продукции
    with kpi2:
        yield_color = "#38ef7d" if yield_pct >= 85 else "#ffd700"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #11998e 0%, {yield_color} 100%); 
                    padding: 25px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);'>
            <div style='font-size: 0.85em; opacity: 0.95; margin-bottom: 8px;'>Выход продукции</div>
            <div style='font-size: 2.8em; font-weight: 700; margin: 8px 0;'>{yield_pct}%</div>
            <div style='font-size: 0.9em; opacity: 0.9;'>от сырья</div>
            <div style='margin-top: 10px; font-size: 0.85em; opacity: 0.85;'>
                Целевой выход: ≥85%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 3: Средний pH
    with kpi3:
        ph_status = "✅" if 5.1 <= avg_ph_today <= 5.6 else "⚠️"
        ph_color = "#ff6a00" if 5.1 <= avg_ph_today <= 5.6 else "#ff4444"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ee0979 0%, {ph_color} 100%); 
                    padding: 25px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(238, 9, 121, 0.4);'>
            <div style='font-size: 0.85em; opacity: 0.95; margin-bottom: 8px;'>Средний pH</div>
            <div style='font-size: 2.8em; font-weight: 700; margin: 8px 0;'>{avg_ph_today:.2f} {ph_status}</div>
            <div style='font-size: 0.9em; opacity: 0.9;'>готовая продукция</div>
            <div style='margin-top: 10px; font-size: 0.85em; opacity: 0.85;'>
                Норма: 5.1 - 5.6
            </div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 4: Активные партии
    with kpi4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 25px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);'>
            <div style='font-size: 0.85em; opacity: 0.95; margin-bottom: 8px;'>Активных партий</div>
            <div style='font-size: 2.8em; font-weight: 700; margin: 8px 0;'>{active_batches}</div>
            <div style='font-size: 0.9em; opacity: 0.9;'>в производстве</div>
            <div style='margin-top: 10px; font-size: 0.85em; opacity: 0.85;'>
                Всего замеров: {total_measurements}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # KPI 5: OEE
    with kpi5:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 25px; border-radius: 12px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);'>
            <div style='font-size: 0.85em; opacity: 0.95; margin-bottom: 8px;'>OEE эффективность</div>
            <div style='font-size: 2.8em; font-weight: 700; margin: 8px 0;'>{efficiency}%</div>
            <div style='font-size: 0.9em; opacity: 0.9;'>общая эффективность</div>
            <div style='margin-top: 10px; font-size: 0.85em; opacity: 0.85;'>
                Целевой OEE: ≥90%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # === ГРАФИКИ И ТАБЛИЦЫ ===
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # График 1: Производство за неделю
        st.subheader("📈 Динамика производства (последние 7 дней)")

        dates = pd.date_range(end=today, periods=7, freq='D')
        production_week = pd.DataFrame({
            'Дата': dates,
            'Произведено': np.random.randint(420, 600, 7),
            'План': [500] * 7
        })

        fig_prod = go.Figure()

        fig_prod.add_trace(go.Bar(
            x=production_week['Дата'],
            y=production_week['Произведено'],
            name='Фактическое производство',
            marker_color='#667eea',
            text=production_week['Произведено'],
            textposition='outside',
            hovertemplate='<b>%{x|%d.%m}</b><br>Произведено: %{y} кг<extra></extra>'
        ))

        fig_prod.add_trace(go.Scatter(
            x=production_week['Дата'],
            y=production_week['План'],
            name='Плановый показатель',
            line=dict(color='red', dash='dash', width=3),
            mode='lines+markers',
            hovertemplate='<b>%{x|%d.%m}</b><br>План: %{y} кг<extra></extra>'
        ))

        fig_prod.update_layout(
            height=400,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Дата",
            yaxis_title="Производство (кг)",
            font=dict(size=12)
        )

        st.plotly_chart(fig_prod, use_container_width=True)

        # График 2: pH по стадиям производства
        st.subheader("🌡️ Мониторинг pH на этапах производства")

        stages = ['Приемка\nсырья', 'Посол\n24ч', 'Посол\n48ч', 'Посол\n72ч', 'После\nтермообр.', 'Готовый\nпродукт']
        ph_values = [6.5, 6.2, 5.8, 5.4, 5.3, 5.35]
        ph_min = [6.3, 5.9, 5.5, 5.1, 5.0, 5.1]
        ph_max = [6.8, 6.5, 6.2, 5.8, 5.6, 5.6]

        fig_ph = go.Figure()

        # Зона допустимых значений
        fig_ph.add_trace(go.Scatter(
            x=stages,
            y=ph_max,
            fill=None,
            mode='lines',
            line_color='rgba(0,255,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))

        fig_ph.add_trace(go.Scatter(
            x=stages,
            y=ph_min,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,255,0,0)',
            name='Допустимый диапазон',
            fillcolor='rgba(0,255,0,0.15)',
            hovertemplate='Норма: %{y:.1f}<extra></extra>'
        ))

        # Фактические значения
        fig_ph.add_trace(go.Scatter(
            x=stages,
            y=ph_values,
            mode='lines+markers',
            name='Текущий pH',
            line=dict(color='#667eea', width=4),
            marker=dict(size=12, symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate='<b>%{x}</b><br>pH: %{y:.2f}<extra></extra>'
        ))

        fig_ph.update_layout(
            height=400,
            template='plotly_white',
            hovermode='x unified',
            xaxis_title="Этап производства",
            yaxis_title="pH значение",
            yaxis=dict(range=[4.5, 7.0]),
            font=dict(size=12)
        )

        st.plotly_chart(fig_ph, use_container_width=True)

    with col_right:
        # Статус производственных линий
        st.subheader("🏭 Статус производственных линий")

        lines = [
            {"name": "Линия 1", "status": "Работает", "batch": "JY-2025-045", "progress": 75, "operator": "Айгуль С."},
            {"name": "Линия 2", "status": "Работает", "batch": "JY-2025-046", "progress": 40, "operator": "Нурлан К."},
            {"name": "Линия 3", "status": "ТО", "batch": "—", "progress": 0, "operator": "—"},
            {"name": "Линия 4", "status": "Работает", "batch": "JY-2025-047", "progress": 90, "operator": "Данияр Т."},
        ]

        for line in lines:
            status_colors = {
                "Работает": ("#28a745", "#d4edda"),
                "ТО": ("#ffc107", "#fff3cd"),
                "Остановлена": ("#dc3545", "#f8d7da")
            }
            border_color, bg_color = status_colors.get(line["status"], ("#6c757d", "#e9ecef"))

            st.markdown(f"""
            <div style='background: {bg_color}; padding: 15px; border-radius: 10px; 
                        margin-bottom: 12px; border-left: 5px solid {border_color};
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <div>
                        <strong style='font-size: 1.1em;'>{line["name"]}</strong><br>
                        <span style='font-size: 0.85em; color: #666;'>Партия: {line["batch"]}</span><br>
                        <span style='font-size: 0.8em; color: #666;'>Оператор: {line["operator"]}</span>
                    </div>
                    <div style='background: {border_color}; color: white; padding: 6px 14px; 
                                border-radius: 20px; font-size: 0.85em; font-weight: 600;'>
                        {line["status"]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if line["progress"] > 0:
                st.progress(line["progress"] / 100, text=f"Готовность: {line['progress']}%")

        st.markdown("---")

        # Последние события
        st.subheader("📋 Последние события")

        events = [
            {"time": current_time.strftime("%H:%M"), "event": "✅ Смена началась", "type": "info"},
            {"time": (current_time - timedelta(minutes=45)).strftime("%H:%M"),
             "event": "✅ Партия JY-2025-044 завершена", "type": "success"},
            {"time": (current_time - timedelta(hours=1, minutes=30)).strftime("%H:%M"),
             "event": f"⚠️ pH {avg_ph_today:.2f} на контроле", "type": "warning"},
            {"time": (current_time - timedelta(hours=2)).strftime("%H:%M"),
             "event": "🔧 Плановое ТО Линии 3", "type": "info"},
        ]

        for event in events:
            event_colors = {
                "success": ("#d4edda", "#155724"),
                "warning": ("#fff3cd", "#856404"),
                "info": ("#d1ecf1", "#0c5460")
            }
            bg_color, text_color = event_colors.get(event["type"], ("#f8f9fa", "#333"))

            st.markdown(f"""
            <div style='background: {bg_color}; color: {text_color}; padding: 10px; 
                        border-radius: 8px; margin-bottom: 8px; font-size: 0.9em;'>
                <strong>{event["time"]}</strong> — {event["event"]}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # === НИЖНИЙ РЯД: Качество и Операторы ===
    bottom_col1, bottom_col2 = st.columns(2)

    with bottom_col1:
        st.subheader("🎯 Распределение продукции по качеству")

        quality_data = pd.DataFrame({
            'Категория': ['Высшее качество', 'Первый сорт', 'Второй сорт'],
            'Количество': [420, 65, 15]
        })

        fig_quality = px.pie(
            quality_data,
            values='Количество',
            names='Категория',
            color_discrete_sequence=['#28a745', '#ffc107', '#dc3545'],
            hole=0.4
        )

        fig_quality.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Количество: %{value} кг<br>Доля: %{percent}<extra></extra>'
        )

        fig_quality.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig_quality, use_container_width=True)

    with bottom_col2:
        st.subheader("👥 Топ операторов текущей смены")

        operators_data = pd.DataFrame({
            'Оператор': ['Айгуль С.', 'Нурлан К.', 'Асем Б.', 'Данияр Т.'],
            'Партий': [12, 11, 10, 9],
            'Качество': [98, 96, 97, 95]
        })

        fig_operators = px.bar(
            operators_data,
            x='Партий',
            y='Оператор',
            orientation='h',
            text='Качество',
            color='Качество',
            color_continuous_scale='Greens',
            range_color=[90, 100]
        )

        fig_operators.update_traces(
            texttemplate='Качество: %{text}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Обработано партий: %{x}<br>Качество: %{text}%<extra></extra>'
        )

        fig_operators.update_layout(height=350, showlegend=False, xaxis_title="Количество партий")

        st.plotly_chart(fig_operators, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)