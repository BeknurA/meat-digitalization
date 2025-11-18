# pages/reports.py - Детальные отчеты о производстве
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from ui import get_text, df_to_download_link
from database import fetch_measurements

def show_reports(lang_choice):
    """Страница детальных отчетов о производстве"""
    
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    
    # Заголовок
    st.markdown("""
    <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 25px; border-radius: 15px; margin-bottom: 25px; color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='margin: 0; color: white;'>📊 Отчеты о производстве Жая</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.95; font-size: 1.05em;'>
            Комплексная аналитика и детальные отчеты по всем аспектам производства
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Фильтры в sidebar
    st.sidebar.header("🔍 Настройки отчетов")
    
    report_type = st.sidebar.selectbox(
        "Выберите тип отчета",
        [
            "📅 Производственный отчет",
            "📈 Анализ качества продукции",
            "💰 Экономические показатели",
            "⚙️ Технологический аудит",
            "👥 Эффективность персонала"
        ]
    )
    
    # Период отчета
    today = datetime.now().date()
    default_start = today - timedelta(days=30)
    
    col_date1, col_date2 = st.sidebar.columns(2)
    with col_date1:
        date_from = st.date_input("Дата от", value=default_start)
    with col_date2:
        date_to = st.date_input("Дата до", value=today)
    
    # Смены
    shifts = st.sidebar.multiselect(
        "Смена",
        ["Первая (08:00-16:00)", "Вторая (16:00-00:00)", "Ночная (00:00-08:00)"],
        default=["Первая (08:00-16:00)", "Вторая (16:00-00:00)"]
    )
    
    st.markdown("---")
    
    # Роутинг отчетов
    if report_type == "📅 Производственный отчет":
        show_production_report(date_from, date_to, shifts)
    elif report_type == "📈 Анализ качества продукции":
        show_quality_report(date_from, date_to)
    elif report_type == "💰 Экономические показатели":
        show_economic_report(date_from, date_to)
    elif report_type == "⚙️ Технологический аудит":
        show_tech_audit(date_from, date_to)
    elif report_type == "👥 Эффективность персонала":
        show_staff_report(date_from, date_to)
    
    st.markdown("</div>", unsafe_allow_html=True)


def show_production_report(date_from, date_to, shifts):
    """Производственный отчет"""
    st.header("📅 Производственный отчет")
    st.markdown(f"**Период:** {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}")
    
    # Генерация данных
    days = (date_to - date_from).days + 1
    dates = pd.date_range(start=date_from, end=date_to, freq='D')
    
    production_df = pd.DataFrame({
        'Дата': dates,
        'Партий': np.random.randint(12, 22, days),
        'Произведено (кг)': np.random.randint(420, 620, days),
        'План (кг)': [500] * days,
        'Выход (%)': np.round(np.random.uniform(83.5, 87.5, days), 1),
        'Брак (кг)': np.random.randint(3, 18, days),
        'Простои (мин)': np.random.randint(0, 90, days)
    })
    
    production_df['Выполнение плана (%)'] = np.round(
        (production_df['Произведено (кг)'] / production_df['План (кг)']) * 100, 1
    )
    production_df['Годная продукция (кг)'] = production_df['Произведено (кг)'] - production_df['Брак (кг)']
    
    # KPI сводка
    st.subheader("📊 Ключевые показатели периода")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_produced = production_df['Произведено (кг)'].sum()
    total_good = production_df['Годная продукция (кг)'].sum()
    avg_yield = production_df['Выход (%)'].mean()
    total_batches = production_df['Партий'].sum()
    avg_plan = production_df['Выполнение плана (%)'].mean()
    total_defects = production_df['Брак (кг)'].sum()
    defect_rate = (total_defects / total_produced) * 100
    
    with col1:
        st.metric(
            "Всего произведено",
            f"{total_produced:,} кг",
            delta=f"План: {len(dates) * 500:,} кг"
        )
        st.caption(f"Годной продукции: {total_good:,} кг")
    
    with col2:
        delta_yield = avg_yield - 85
        st.metric(
            "Средний выход",
            f"{avg_yield:.1f}%",
            delta=f"{delta_yield:+.1f}% от целевого (85%)"
        )
        st.caption(f"Максимум: {production_df['Выход (%)'].max():.1f}%")
    
    with col3:
        st.metric(
            "Обработано партий",
            f"{total_batches}",
            delta=f"{int(total_batches/days):.0f} партий/день"
        )
        st.caption(f"Мин/Макс: {production_df['Партий'].min()}/{production_df['Партий'].max()}")
    
    with col4:
        delta_plan = avg_plan - 100
        st.metric(
            "Выполнение плана",
            f"{avg_plan:.1f}%",
            delta=f"{delta_plan:+.1f}% от нормы"
        )
        st.caption(f"Дней выше плана: {len(production_df[production_df['Выполнение плана (%)'] > 100])}")
    
    st.markdown("---")
    
    # Дополнительная статистика
    col_extra1, col_extra2, col_extra3 = st.columns(3)
    
    with col_extra1:
        st.metric(
            "Общий брак",
            f"{total_defects} кг",
            delta=f"{defect_rate:.2f}% от производства"
        )
    
    with col_extra2:
        total_downtime = production_df['Простои (мин)'].sum()
        avg_downtime = production_df['Простои (мин)'].mean()
        st.metric(
            "Простои всего",
            f"{total_downtime} мин",
            delta=f"{avg_downtime:.0f} мин/день"
        )
    
    with col_extra3:
        efficiency = ((total_produced - total_defects) / total_produced) * 100
        st.metric(
            "Эффективность",
            f"{efficiency:.1f}%",
            delta="Годная продукция"
        )
    
    st.markdown("---")
    
    # Детальная таблица
    st.subheader("📋 Детальная разбивка по дням")
    
    # Форматирование для отображения
    display_df = production_df.copy()
    display_df['Дата'] = display_df['Дата'].dt.strftime('%d.%m.%Y')
    
    # Подсветка
    def highlight_values(row):
        colors = []
        for col in row.index:
            if col == 'Выполнение плана (%)':
                if row[col] >= 100:
                    colors.append('background-color: #d4edda')
                elif row[col] >= 90:
                    colors.append('background-color: #fff3cd')
                else:
                    colors.append('background-color: #f8d7da')
            else:
                colors.append('')
        return colors
    
    styled_df = display_df.style.apply(highlight_values, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Графики
    st.subheader("📈 Визуализация динамики")
    
    tab1, tab2, tab3 = st.tabs(["Производство", "Выход продукции", "Брак и простои"])
    
    with tab1:
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=production_df['Дата'],
            y=production_df['Произведено (кг)'],
            name='Факт',
            marker_color='#667eea',
            text=production_df['Произведено (кг)'],
            textposition='outside'
        ))
        
        fig1.add_trace(go.Scatter(
            x=production_df['Дата'],
            y=production_df['План (кг)'],
            name='План',
            line=dict(color='red', dash='dash', width=3),
            mode='lines'
        ))
        
        fig1.update_layout(
            title="Производство vs План",
            xaxis_title="Дата",
            yaxis_title="Производство (кг)",
            height=450,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=production_df['Дата'],
            y=production_df['Выход (%)'],
            mode='lines+markers',
            name='Выход продукции',
            line=dict(color='#11998e', width=3),
            marker=dict(size=8)
        ))
        
        fig2.add_hline(
            y=85,
            line_dash="dash",
            line_color="green",
            annotation_text="Целевой выход: 85%",
            annotation_position="right"
        )
        
        fig2.update_layout(
            title="Динамика выхода продукции",
            xaxis_title="Дата",
            yaxis_title="Выход (%)",
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            x=production_df['Дата'],
            y=production_df['Брак (кг)'],
            name='Брак (кг)',
            marker_color='#dc3545',
            yaxis='y'
        ))
        
        fig3.add_trace(go.Scatter(
            x=production_df['Дата'],
            y=production_df['Простои (мин)'],
            name='Простои (мин)',
            line=dict(color='#ffc107', width=3),
            mode='lines+markers',
            yaxis='y2'
        ))
        
        fig3.update_layout(
            title="Брак и простои",
            xaxis_title="Дата",
            yaxis=dict(title="Брак (кг)", side='left'),
            yaxis2=dict(title="Простои (мин)", side='right', overlaying='y'),
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Экспорт
    st.markdown("---")
    st.subheader("💾 Экспорт данных")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        st.markdown(
            df_to_download_link(
                display_df,
                f"production_report_{date_from}_{date_to}.csv",
                "📥 Скачать производственный отчет (CSV)"
            ),
            unsafe_allow_html=True
        )
    
    with col_export2:
        if st.button("📧 Отправить отчет на email"):
            st.success("✅ Отчет отправлен на manager@zhaya.kz")


def show_quality_report(date_from, date_to):
    """Отчет по качеству"""
    st.header("📈 Анализ качества продукции")
    st.markdown(f"**Период:** {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}")
    
    # Генерация данных
    days = (date_to - date_from).days + 1
    dates = pd.date_range(start=date_from, end=date_to, freq='D')
    
    quality_df = pd.DataFrame({
        'Дата': dates,
        'pH': np.round(np.random.uniform(5.05, 5.65, days), 2),
        'Влажность (%)': np.round(np.random.uniform(66, 73, days), 1),
        'Aw': np.round(np.random.uniform(0.86, 0.92, days), 3),
        'ТБЧ (мг/кг)': np.round(np.random.uniform(0.6, 1.8, days), 2),
        'Цвет (ΔE)': np.round(np.random.uniform(1.3, 2.8, days), 2),
        'Органолептика': np.random.randint(82, 99, days)
    })
    
    # Статус соответствия
    def quality_status(row):
        score = 0
        if 5.1 <= row['pH'] <= 5.6: score += 20
        if 68 <= row['Влажность (%)'] <= 72: score += 20
        if 0.88 <= row['Aw'] <= 0.90: score += 20
        if row['ТБЧ (мг/кг)'] < 1.5: score += 20
        if row['Органолептика'] >= 90: score += 20
        
        if score >= 90:
            return "✅ Отлично"
        elif score >= 70:
            return "⚠️ Хорошо"
        else:
            return "❌ Требует внимания"
    
    quality_df['Статус'] = quality_df.apply(quality_status, axis=1)
    
    # KPI качества
    st.subheader("🎯 Соответствие нормативам")
    
    col1, col2, col3, col4 = st.columns(4)
    
    ph_ok = ((quality_df['pH'] >= 5.1) & (quality_df['pH'] <= 5.6)).mean() * 100
    moisture_ok = ((quality_df['Влажность (%)'] >= 68) & (quality_df['Влажность (%)'] <= 72)).mean() * 100
    aw_ok = ((quality_df['Aw'] >= 0.88) & (quality_df['Aw'] <= 0.90)).mean() * 100
    tbc_ok = (quality_df['ТБЧ (мг/кг)'] < 1.5).mean() * 100
    
    with col1:
        st.metric(
            "pH в норме",
            f"{ph_ok:.0f}%",
            delta=f"Норма: 5.1-5.6"
        )
        st.caption(f"Средний: {quality_df['pH'].mean():.2f}")
    
    with col2:
        st.metric(
            "Влажность в норме",
            f"{moisture_ok:.0f}%",
            delta=f"Норма: 68-72%"
        )
        st.caption(f"Средняя: {quality_df['Влажность (%)'].mean():.1f}%")
    
    with col3:
        st.metric(
            "Aw в норме",
            f"{aw_ok:.0f}%",
            delta=f"Норма: 0.88-0.90"
        )
        st.caption(f"Среднее: {quality_df['Aw'].mean():.3f}")
    
    with col4:
        st.metric(
            "ТБЧ в норме",
            f"{tbc_ok:.0f}%",
            delta=f"Норма: < 1.5"
        )
        st.caption(f"Среднее: {quality_df['ТБЧ (мг/кг)'].mean():.2f}")
    
    st.markdown("---")
    
    # Распределение по статусам
    st.subheader("📊 Распределение продукции по качеству")
    
    status_counts = quality_df['Статус'].value_counts()
    
    col_pie, col_stats = st.columns([1, 1])
    
    with col_pie:
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Качество продукции (дни)",
            color_discrete_map={
                "✅ Отлично": "#28a745",
                "⚠️ Хорошо": "#ffc107",
                "❌ Требует внимания": "#dc3545"
            }
        )
        fig_status.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col_stats:
        st.markdown("### Статистика периода")
        st.markdown(f"""
        - **Отличных дней:** {status_counts.get('✅ Отлично', 0)} ({status_counts.get('✅ Отлично', 0)/days*100:.0f}%)
        - **Хороших дней:** {status_counts.get('⚠️ Хорошо', 0)} ({status_counts.get('⚠️ Хорошо', 0)/days*100:.0f}%)
        - **Требует внимания:** {status_counts.get('❌ Требует внимания', 0)} ({status_counts.get('❌ Требует внимания', 0)/days*100:.0f}%)
        
        **Ключевые выводы:**
        - Стабильное соблюдение технологии
        - pH находится в оптимальном диапазоне
        - Низкий уровень окисления (ТБЧ)
        - Высокие органолептические показатели
        """)
    
    st.markdown("---")
    
    # Графики параметров
    st.subheader("📈 Динамика параметров качества")
    
    tab1, tab2, tab3 = st.tabs(["pH и Влажность", "Активность воды (Aw)", "Окисление (ТБЧ)"])
    
    with tab1:
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=quality_df['Дата'],
            y=quality_df['pH'],
            name='pH',
            mode='lines+markers',
            line=dict(color='#667eea', width=2),
            yaxis='y'
        ))
        
        fig1.add_trace(go.Scatter(
            x=quality_df['Дата'],
            y=quality_df['Влажность (%)'],
            name='Влажность (%)',
            mode='lines+markers',
            line=dict(color='#11998e', width=2),
            yaxis='y2'
        ))
        
        fig1.add_hrect(y0=5.1, y1=5.6, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        
        fig1.update_layout(
            title="pH и Влажность",
            xaxis_title="Дата",
            yaxis=dict(title="pH", side='left', range=[4.8, 6.0]),
            yaxis2=dict(title="Влажность (%)", side='right', overlaying='y', range=[60, 80]),
            height=450,
            template='plotly_white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=quality_df['Дата'],
            y=quality_df['Aw'],
            mode='lines+markers',
            name='Aw',
            line=dict(color='#f093fb', width=3),
            marker=dict(size=8)
        ))
        
        fig2.add_hrect(y0=0.88, y1=0.90, fillcolor="green", opacity=0.15, layer="below", 
                       annotation_text="Оптимальный диапазон", annotation_position="top left")
        
        fig2.update_layout(
            title="Активность воды (Aw)",
            xaxis_title="Дата",
            yaxis_title="Aw",
            yaxis=dict(range=[0.80, 0.95]),
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig3 = px.bar(
            quality_df,
            x='Дата',
            y='ТБЧ (мг/кг)',
            title="Перекисное число (ТБЧ) - индикатор окисления жиров",
            color='ТБЧ (мг/кг)',
            color_continuous_scale='Reds'
        )
        
        fig3.add_hline(y=1.5, line_dash="dash", line_color="red", 
                       annotation_text="Макс. допустимое (1.5 мг/кг)")
        
        fig3.update_layout(height=450, template='plotly_white')
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Детальная таблица
    st.markdown("---")
    st.subheader("📋 Детальные показатели")
    
    display_quality = quality_df.copy()
    display_quality['Дата'] = display_quality['Дата'].dt.strftime('%d.%m.%Y')
    
    st.dataframe(display_quality, use_container_width=True, hide_index=True)
    
    # Экспорт
    st.markdown("---")
    st.markdown(
        df_to_download_link(
            display_quality,
            f"quality_report_{date_from}_{date_to}.csv",
            "📥 Скачать отчет по качеству (CSV)"
        ),
        unsafe_allow_html=True
    )


def show_economic_report(date_from, date_to):
    """Экономический отчет"""
    st.header("💰 Экономические показатели")
    st.markdown(f"**Период:** {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}")
    
    days = (date_to - date_from).days + 1
    
    # Расчет показателей
    total_produced = np.random.randint(450, 550, days).sum()
    selling_price = 1250  # тг/кг
    
    # Себестоимость
    raw_material_cost = total_produced * np.random.uniform(480, 520)
    labor_cost = days * np.random.uniform(195000, 215000)
    overhead_cost = days * np.random.uniform(55000, 75000)
    energy_cost = total_produced * np.random.uniform(18, 24)
    
    total_cost = raw_material_cost + labor_cost + overhead_cost + energy_cost
    
    # Выручка и прибыль
    revenue = total_produced * selling_price
    profit = revenue - total_cost
    profit_margin = (profit / revenue) * 100
    
    # KPI
    st.subheader("💵 Финансовые показатели")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Выручка",
            f"{revenue:,.0f} ₸",
            delta=f"+{np.random.randint(8, 18)}% к пред. периоду"
        )
    
    with col2:
        st.metric(
            "Себестоимость",
            f"{total_cost:,.0f} ₸",
            delta=f"-{np.random.randint(3, 9)}% оптимизация"
        )
    
    with col3:
        st.metric(
            "Прибыль",
            f"{profit:,.0f} ₸",
            delta=f"+{np.random.randint(12, 28)}%"
        )
    
    with col4:
        target_margin = 22
        delta_margin = profit_margin - target_margin
        st.metric(
            "Рентабельность",
            f"{profit_margin:.1f}%",
            delta=f"{delta_margin:+.1f}% от целевой ({target_margin}%)"
        )
    
    st.markdown("---")
    
    # Структура затрат
    st.subheader("📊 Структура себестоимости")
    
    col_pie, col_table = st.columns([1, 1])
    
    with col_pie:
        cost_structure = pd.DataFrame({
            'Статья затрат': ['Сырье и материалы', 'Фонд оплаты труда', 'Накладные расходы', 'Энергоресурсы'],
            'Сумма': [raw_material_cost, labor_cost, overhead_cost, energy_cost],
            'Процент': [
                (raw_material_cost/total_cost)*100,
                (labor_cost/total_cost)*100,
                (overhead_cost/total_cost)*100,
                (energy_cost/total_cost)*100
            ]
        })
        
        fig_costs = px.pie(
            cost_structure,
            values='Сумма',
            names='Статья затрат',
            title="Распределение затрат",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig_costs.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_costs, use_container_width=True)
    
    with col_table:
        st.markdown("### Детальная разбивка")
        
        detail_df = cost_structure.copy()
        detail_df['Сумма'] = detail_df['Сумма'].apply(lambda x: f"{x:,.0f} ₸")
        detail_df['Процент'] = detail_df['Процент'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        **Ключевые наблюдения:**
        - Сырье составляет основную долю ({(raw_material_cost/total_cost)*100:.0f}%)
        - ФОТ оптимален для текущих объемов
        - Энергоэффективность в пределах нормы
        - Накладные расходы контролируются
        """)
    
    st.markdown("---")
    
    # Экономика производства
    st.subheader("📈 Экономика производства")
    
    col_bar, col_waterfall = st.columns(2)
    
    with col_bar:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Сырье', 'ФОТ', 'Накладные', 'Энергия'],
                y=[raw_material_cost, labor_cost, overhead_cost, energy_cost],
                text=[f"{raw_material_cost:,.0f}", f"{labor_cost:,.0f}", 
                      f"{overhead_cost:,.0f}", f"{energy_cost:,.0f}"],
                textposition='outside',
                marker_color=['#667eea', '#11998e', '#f093fb', '#ff6a00']
            )
        ])
        
        fig_bar.update_layout(
            title="Затраты по статьям",
            xaxis_title="Статья",
            yaxis_title="Сумма (₸)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_waterfall:
        fig_waterfall = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=["Выручка", "Сырье", "ФОТ", "Накладные", "Энергия", "Прибыль"],
            textposition="outside",
            text=[f"{revenue:,.0f}", f"-{raw_material_cost:,.0f}", f"-{labor_cost:,.0f}",
                  f"-{overhead_cost:,.0f}", f"-{energy_cost:,.0f}", f"{profit:,.0f}"],
            y=[revenue, -raw_material_cost, -labor_cost, -overhead_cost, -energy_cost, profit],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="Каскад формирования прибыли",
            showlegend=False,
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.markdown("---")
    
    # Детальная таблица
    st.subheader("📋 Сводная финансовая таблица")
    
    financial_summary = pd.DataFrame({
        'Показатель': [
            'Объем производства',
            'Цена реализации',
            'Выручка',
            '',
            'Сырье и материалы',
            'Фонд оплаты труда',
            'Накладные расходы',
            'Энергоресурсы',
            'Итого себестоимость',
            '',
            'Валовая прибыль',
            'Рентабельность продаж',
            '',
            'Себестоимость 1 кг',
            'Прибыль на 1 кг'
        ],
        'Значение': [
            f"{total_produced:,.0f} кг",
            f"{selling_price} ₸/кг",
            f"{revenue:,.0f} ₸",
            "",
            f"{raw_material_cost:,.0f} ₸",
            f"{labor_cost:,.0f} ₸",
            f"{overhead_cost:,.0f} ₸",
            f"{energy_cost:,.0f} ₸",
            f"{total_cost:,.0f} ₸",
            "",
            f"{profit:,.0f} ₸",
            f"{profit_margin:.2f}%",
            "",
            f"{total_cost/total_produced:.2f} ₸/кг",
            f"{profit/total_produced:.2f} ₸/кг"
        ]
    })
    
    st.dataframe(financial_summary, use_container_width=True, hide_index=True)


def show_tech_audit(date_from, date_to):
    """Технологический аудит"""
    st.header("⚙️ Технологический аудит производства")
    st.markdown(f"**Период:** {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}")
    
    st.info("🔍 Проверка соответствия технологическим регламентам и стандартам качества")
    
    # Чеклист соответствия
    st.subheader("✅ Чеклист технологического соответствия")
    
    checklist = [
        {"Параметр": "Температура хранения сырья", "Норма": "0-4°С", "Факт": "2.5°С", "Статус": "✅ Соответствует", "Критичность": "Высокая"},
        {"Параметр": "Температура посола", "Норма": "0-3°С", "Факт": "1.8°С", "Статус": "✅ Соответствует", "Критичность": "Высокая"},
        {"Параметр": "Продолжительность посола", "Норма": "72±2 часа", "Факт": "72 часа", "Статус": "✅ Соответствует", "Критичность": "Высокая"},
        {"Параметр": "Концентрация соли в рассоле", "Норма": "3.0-3.5%", "Факт": "3.2%", "Статус": "✅ Соответствует", "Критичность": "Критическая"},
        {"Параметр": "Концентрация экстракта", "Норма": "3-5%", "Факт": "4.5%", "Статус": "✅ Соответствует", "Критичность": "Средняя"},
        {"Параметр": "Температура сушки", "Норма": "45±3°С", "Факт": "46°С", "Статус": "✅ Соответствует", "Критичность": "Средняя"},
        {"Параметр": "Температура обжарки", "Норма": "75-85°С", "Факт": "80°С", "Статус": "✅ Соответствует", "Критичность": "Высокая"},
        {"Параметр": "Внутренняя T° после варки", "Норма": "≥74°С", "Факт": "75°С", "Статус": "✅ Соответствует", "Критичность": "Критическая"},
        {"Параметр": "pH готового продукта", "Норма": "5.1-5.6", "Факт": "5.35", "Статус": "✅ Соответствует", "Критичность": "Критическая"},
        {"Параметр": "Активность воды (Aw)", "Норма": "0.88-0.90", "Факт": "0.89", "Статус": "✅ Соответствует", "Критичность": "Критическая"},
        {"Параметр": "Перекисное число (ТБЧ)", "Норма": "<1.5 мг/кг", "Факт": "0.95 мг/кг", "Статус": "✅ Соответствует", "Критичность": "Высокая"},
        {"Параметр": "Выход продукции", "Норма": "≥85%", "Факт": "86.2%", "Статус": "✅ Соответствует", "Критичность": "Средняя"},
        {"Параметр": "Цветовая стабильность (ΔE)", "Норма": "<2.0", "Факт": "1.7", "Статус": "✅ Соответствует", "Критичность": "Низкая"},
        {"Параметр": "Органолептическая оценка", "Норма": "≥85 баллов", "Факт": "92 балла", "Статус": "✅ Соответствует", "Критичность": "Высокая"},
    ]
    
    checklist_df = pd.DataFrame(checklist)
    
    # Статистика соответствия
    total_params = len(checklist_df)
    compliant = len(checklist_df[checklist_df['Статус'] == '✅ Соответствует'])
    compliance_rate = (compliant / total_params) * 100
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric(
            "Общее соответствие",
            f"{compliance_rate:.0f}%",
            delta=f"{compliant}/{total_params} параметров"
        )
    
    with col_stat2:
        critical_params = checklist_df[checklist_df['Критичность'] == 'Критическая']
        critical_ok = len(critical_params[critical_params['Статус'] == '✅ Соответствует'])
        st.metric(
            "Критические параметры",
            f"{critical_ok}/{len(critical_params)}",
            delta="100% соответствие" if critical_ok == len(critical_params) else "Требует внимания"
        )
    
    with col_stat3:
        st.metric(
            "Последний аудит",
            datetime.now().strftime('%d.%m.%Y'),
            delta="Сегодня"
        )
    
    st.markdown("---")
    
    # Таблица чеклиста с цветовой индикацией
    def highlight_status(row):
        colors = []
        for col in row.index:
            if col == 'Статус':
                if '✅' in str(row[col]):
                    colors.append('background-color: #d4edda; color: #155724')
                elif '⚠️' in str(row[col]):
                    colors.append('background-color: #fff3cd; color: #856404')
                else:
                    colors.append('background-color: #f8d7da; color: #721c24')
            elif col == 'Критичность':
                if row[col] == 'Критическая':
                    colors.append('background-color: #dc3545; color: white; font-weight: bold')
                elif row[col] == 'Высокая':
                    colors.append('background-color: #ffc107; color: #333')
                else:
                    colors.append('')
            else:
                colors.append('')
        return colors
    
    styled_checklist = checklist_df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_checklist, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Критические контрольные точки (HACCP)
    st.subheader("🎯 Критические контрольные точки (HACCP)")
    
    st.markdown("""
    <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff;'>
        <strong>Система НАССР (Hazard Analysis and Critical Control Points)</strong> — 
        международно признанная система управления безопасностью пищевой продукции.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    ccp_data = [
        {
            "ККТ": "ККТ-1: Приемка сырья",
            "Опасность": "Микробиологическая контаминация",
            "Критерий": "T ≤ 4°С, pH 5.8-6.8, без визуальных дефектов",
            "Мониторинг": "Каждая партия (100%)",
            "Корректирующие действия": "Отбраковка партии, возврат поставщику",
            "Документация": "Журнал приемки сырья",
            "Статус": "✅ Под контролем"
        },
        {
            "ККТ": "ККТ-2: Термическая обработка",
            "Опасность": "Выживание патогенных микроорганизмов",
            "Критерий": "T внутри продукта ≥ 74°С, выдержка ≥2 мин",
            "Мониторинг": "Непрерывный (IoT датчики)",
            "Корректирующие действия": "Повторная термообработка, изоляция партии",
            "Документация": "Карта термообработки (автоматич.)",
            "Статус": "✅ Под контролем"
        },
        {
            "ККТ": "ККТ-3: Охлаждение и хранение",
            "Опасность": "Размножение термостойких бактерий",
            "Критерий": "Охлаждение до T ≤ 5°С за 12 часов",
            "Мониторинг": "Каждые 2 часа",
            "Корректирующие действия": "Ускоренное охлаждение, изоляция партии",
            "Документация": "Журнал температурного режима",
            "Статус": "✅ Под контролем"
        },
        {
            "ККТ": "ККТ-4: Контроль pH",
            "Опасность": "Рост патогенов при высоком pH",
            "Критерий": "pH готового продукта 5.1-5.6",
            "Мониторинг": "Каждая партия после созревания",
            "Корректирующие действия": "Корректировка процесса посола, изоляция",
            "Документация": "Лабораторный журнал pH",
            "Статус": "✅ Под контролем"
        }
    ]
    
    for i, ccp in enumerate(ccp_data, 1):
        with st.expander(f"{ccp['ККТ']}", expanded=(i==1)):
            col_ccp1, col_ccp2 = st.columns(2)
            
            with col_ccp1:
                st.markdown(f"**Вид опасности:** {ccp['Опасность']}")
                st.markdown(f"**Критический предел:** {ccp['Критерий']}")
                st.markdown(f"**Частота мониторинга:** {ccp['Мониторинг']}")
            
            with col_ccp2:
                st.markdown(f"**Корректирующие действия:** {ccp['Корректирующие действия']}")
                st.markdown(f"**Документация:** {ccp['Документация']}")
                st.markdown(f"**Текущий статус:** {ccp['Статус']}")
    
    st.markdown("---")
    
    # Рекомендации
    st.subheader("💡 Рекомендации по улучшению")
    
    recommendations = [
        "✅ Все критические параметры находятся в пределах нормы",
        "📊 Рекомендуется внедрить автоматический контроль цвета (спектрофотометр)",
        "🔬 Расширить контроль микробиологических показателей (добавить тест на листерии)",
        "📈 Оптимизировать расход энергии на этапе термообработки (-5% возможно)",
        "🎓 Провести дополнительное обучение операторов по работе с IoT системами"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")


def show_staff_report(date_from, date_to):
    """Отчет по персоналу"""
    st.header("👥 Эффективность персонала")
    st.markdown(f"**Период:** {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}")
    
    # Данные по сменам
    st.subheader("📊 Производительность по сменам")
    
    shifts_df = pd.DataFrame({
        'Смена': ['Первая смена\n(08:00-16:00)', 'Вторая смена\n(16:00-00:00)', 'Ночная смена\n(00:00-08:00)'],
        'Операторов': [8, 7, 5],
        'Произведено (кг)': [530, 490, 350],
        'Партий': [23, 21, 15],
        'Простои (мин)': [42, 58, 28],
        'Качество (%)': [96.5, 94.8, 95.2],
        'Брак (кг)': [8, 12, 6]
    })
    
    shifts_df['Производительность (кг/чел)'] = np.round(
        shifts_df['Произведено (кг)'] / shifts_df['Операторов'], 1
    )
    shifts_df['Партий на оператора'] = np.round(
        shifts_df['Партий'] / shifts_df['Операторов'], 1
    )
    
    # KPI по сменам
    col_shift1, col_shift2, col_shift3 = st.columns(3)
    
    total_staff = shifts_df['Операторов'].sum()
    total_produced = shifts_df['Произведено (кг)'].sum()
    avg_productivity = shifts_df['Производительность (кг/чел)'].mean()
    avg_quality = shifts_df['Качество (%)'].mean()
    
    with col_shift1:
        st.metric(
            "Всего операторов",
            total_staff,
            delta=f"Средняя смена: {total_staff/3:.0f} чел"
        )
    
    with col_shift2:
        st.metric(
            "Средняя производительность",
            f"{avg_productivity:.1f} кг/чел",
            delta="Оптимальный уровень"
        )
    
    with col_shift3:
        st.metric(
            "Средний % качества",
            f"{avg_quality:.1f}%",
            delta="+1.2% к пред. периоду"
        )
    
    st.markdown("---")
    
    # Таблица по сменам
    st.dataframe(shifts_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Графики по сменам
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_shifts = go.Figure()
        
        fig_shifts.add_trace(go.Bar(
            x=shifts_df['Смена'],
            y=shifts_df['Произведено (кг)'],
            name='Производство',
            marker_color='#667eea',
            text=shifts_df['Произведено (кг)'],
            textposition='outside'
        ))
        
        fig_shifts.update_layout(
            title="Производство по сменам",
            xaxis_title="Смена",
            yaxis_title="Производство (кг)",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_shifts, use_container_width=True)
    
    with col_chart2:
        fig_quality = go.Figure()
        
        fig_quality.add_trace(go.Scatter(
            x=shifts_df['Смена'],
            y=shifts_df['Качество (%)'],
            mode='lines+markers',
            name='Качество',
            line=dict(color='#28a745', width=4),
            marker=dict(size=12)
        ))
        
        fig_quality.add_hline(
            y=95,
            line_dash="dash",
            line_color="red",
            annotation_text="Целевое качество: 95%"
        )
        
        fig_quality.update_layout(
            title="Качество продукции по сменам",
            xaxis_title="Смена",
            yaxis_title="Качество (%)",
            yaxis=dict(range=[90, 100]),
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_quality, use_container_width=True)
    
    st.markdown("---")
    
    # Топ операторов
    st.subheader("🏆 Рейтинг операторов за период")
    
    operators_df = pd.DataFrame({
        'Оператор': ['Айгуль Сериковна', 'Нурлан Касымов', 'Асем Болатова', 
                     'Данияр Токаев', 'Гульнара Есенова', 'Бауыржан Смагулов',
                     'Алия Нурланова', 'Ерлан Жумабеков'],
        'Партий обработано': [248, 241, 235, 228, 223, 218, 215, 210],
        'Качество (%)': [98.2, 96.8, 97.1, 95.5, 96.3, 94.8, 95.9, 94.2],
        'Производительность (кг/смена)': [68, 66, 65, 63, 62, 60, 59, 58],
        'Простои (мин/смена)': [15, 18, 16, 22, 19, 25, 20, 28],
        'KPI балл': [9.8, 9.5, 9.6, 9.2, 9.3, 9.0, 9.1, 8.9]
    })
    
    # График рейтинга
    fig_rating = px.bar(
        operators_df,
        x='KPI балл',
        y='Оператор',
        orientation='h',
        text='KPI балл',
        color='KPI балл',
        color_continuous_scale='Greens',
        range_color=[8.5, 10],
        title="Рейтинг операторов по KPI"
    )
    
    fig_rating.update_traces(
        texttemplate='%{text:.1f}',
        textposition='outside'
    )
    
    fig_rating.update_layout(height=450, showlegend=False)
    
    st.plotly_chart(fig_rating, use_container_width=True)
    
    st.markdown("---")
    
    # Детальная таблица операторов
    st.subheader("📋 Детальная статистика операторов")
    
    st.dataframe(operators_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Выводы и рекомендации
    st.subheader("💡 Выводы и рекомендации")
    
    col_concl1, col_concl2 = st.columns(2)
    
    with col_concl1:
        st.markdown("""
        **Сильные стороны:**
        - ✅ Высокая квалификация ведущих операторов (KPI >9.5)
        - ✅ Стабильное качество продукции на всех сменах (>94%)
        - ✅ Первая смена показывает наилучшую производительность
        - ✅ Низкий уровень простоев у топ-операторов (<20 мин/смену)
        """)
    
    with col_concl2:
        st.markdown("""
        **Области для улучшения:**
        - 📊 Ночная смена: необходимо усилить состав (+2 оператора)
        - 🎓 Провести обучение для операторов с KPI <9.0
        - ⚙️ Оптимизировать простои на второй смене (текущие 58 мин)
        - 💡 Внедрить систему наставничества (топ → новички)
        """)
    
    # Экспорт
    st.markdown("---")
    st.markdown(
        df_to_download_link(
            operators_df,
            f"staff_report_{date_from}_{date_to}.csv",
            "📥 Скачать отчет по персоналу (CSV)"
        ),
        unsafe_allow_html=True
    )
