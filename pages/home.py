# Meat_Digitalization/pages/home.py
import streamlit as st
from ui import get_text

def show_home(lang_choice):
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='main-title-animation'>{get_text('full_title', lang_choice)}</h1>", unsafe_allow_html=True)
    st.subheader(get_text("home_desc", lang_choice))
    st.markdown("---")

    # 2. Ключевые функции платформы
    st.markdown(f"### {get_text('home_info', lang_choice)}")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric(
            label="⚙️ " + get_text("menu_production_process", lang_choice),
            value=get_text("stage_control_suffix", lang_choice),
            delta=get_text("delta_production", lang_choice)
        )
        st.write(get_text("prod_subtitle", lang_choice))

    with col_b:
        st.metric(
            label="📈 " + get_text("menu_regression_models", lang_choice),
            value="pH и " + get_text("moisture_title", lang_choice).split()[0],
            delta=get_text("delta_regression", lang_choice)
        )
        st.write(get_text("regression_subtitle", lang_choice))

    with col_c:
        st.metric(
            label="🔬 " + get_text("menu_seabuckthorn_analysis", lang_choice),
            value=get_text("seabuckthorn_value", lang_choice),
            delta=get_text("delta_seabuckthorn", lang_choice)
        )
        st.write(get_text("seabuck_desc", lang_choice))

    st.markdown("---")

    # 3. Основные научные достижения
    st.markdown(f"### {get_text('scientific_achievements', lang_choice)}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'''
        <div class="key-finding-card">
            <h4>{get_text("wac_title", lang_choice)}</h4>
            <div class="small-muted">{get_text("wac_subtitle", lang_choice)}</div>
            <div class="key-value">75.6%</div>
            <div class="small-muted">{get_text("wac_note", lang_choice)}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.markdown(f'''
        <div class="key-finding-card">
            <h4>{get_text("shelf_life_title", lang_choice)}</h4>
            <div class="small-muted">{get_text("shelf_life_subtitle", lang_choice)}</div>
            <div class="key-value">60 {get_text("day_in_lang", lang_choice)}</div>
            <div class="small-muted">{get_text("shelf_life_note", lang_choice)}</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        st.markdown(f'''
        <div class="key-finding-card">
            <h4>{get_text("optimal_conc_title", lang_choice)}</h4>
            <div class="small-muted">{get_text("optimal_conc_subtitle", lang_choice)}</div>
            <div class="key-value">3 – 5%</div>
            <div class="small-muted">{get_text("optimal_conc_note", lang_choice)}</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # 4. Окислительная стабильность
    st.subheader(get_text("oxidation_stability_title", lang_choice))
    TBC_control = 2.80
    TBC_extract = 0.90
    reduction_pct = round((1 - (TBC_extract / TBC_control)) * 100)

    st.markdown(get_text("oxidation_goal", lang_choice))
    st.progress(reduction_pct / 100,
                 text=f"**{reduction_pct}% {get_text('tba_reduction_text', lang_choice)}** "
                      f"{get_text('tba_caption_extract', lang_choice)}.")
    st.caption(f"{get_text('tba_caption', lang_choice)} {TBC_control} {get_text('mg_per_kg', lang_choice)} ({get_text('tba_caption_control', lang_choice)}) "
                f"{get_text('tba_caption_to', lang_choice)} {TBC_extract} {get_text('mg_per_kg', lang_choice)} (5% {get_text('tba_caption_extract', lang_choice)}).")
    st.success(get_text("oxidation_success", lang_choice))

    st.markdown("---")

    # 5. Контроль pH
    st.subheader(get_text("ph_title", lang_choice))
    current_ph = 5.35
    ph_min = 5.1
    ph_max = 5.6

    st.markdown(f'''
        <div style='text-align:center; padding: 20px; background-color: #2a2a2a; border-radius: 10px; border: 2px solid #333;'>
            <h4 style='color:#f0f0f0;'>{get_text('predicted_ph', lang_choice)}:</h4>
            <h1 style='color:#198754; font-size: 3em; animation: pulse 1s infinite;'>{current_ph:.2f}</h1>
            <div class="small-muted">{get_text('ph_optimal', lang_choice)} <b>{ph_min:.1f} – {ph_max:.1f}</b></div>
        </div>
        ''', unsafe_allow_html=True)

    if ph_min <= current_ph <= ph_max:
        st.success(get_text("ph_optimal", lang_choice))
    else:
        st.warning(get_text("ph_insufficient", lang_choice))

    st.markdown("</div>", unsafe_allow_html=True)
