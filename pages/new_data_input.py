# Meat_Digitalization/pages/new_data_input.py
import streamlit as st
from ui import get_text
from data_loader import safe_read_excel, append_row_excel, MEAT_DATA_XLSX, SHEET_NAME, load_all_data

def show_new_data_input(lang_choice):
    st.title(get_text("input_title", lang_choice))
    st.markdown(f"### {get_text('input_subtitle', lang_choice)} ({MEAT_DATA_XLSX}, {get_text('sheet', lang_choice)} {SHEET_NAME})")

    df_meat = safe_read_excel(MEAT_DATA_XLSX, SHEET_NAME)

    if "BatchID" not in df_meat.columns:
        st.error(get_text("batchid_missing", lang_choice))
        st.stop()

    if len(df_meat) > 0 and df_meat["BatchID"].astype(str).str.match(r"^M\d+$").any():
        last_id_str = df_meat[df_meat["BatchID"].astype(str).str.match(r"^M\d+$")]["BatchID"].dropna().astype(str).iloc[-1]
        try:
            last_num = int(last_id_str[1:])
            next_id = f"M{last_num + 1}"
        except:
            next_id = "M1"
    else:
        next_id = "M1"

    with st.form(key='batch_entry_form'):
        st.subheader(get_text("batch_params", lang_choice))

        st.text_input(get_text("batch_id", lang_choice), value=next_id, disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            mass_kg = st.number_input(get_text("mass", lang_choice), min_value=1.0, value=100.0, step=1.0)
            T_initial_C = st.number_input(get_text("initial_temp", lang_choice), min_value=-10.0, value=4.0, step=0.1)
            Salt_pct = st.number_input(get_text("salt_content", lang_choice), min_value=0.0, value=5.0, step=0.1)
        with col2:
            Moisture_pct = st.number_input(get_text("moisture", lang_choice), min_value=0.0, value=75.0, step=0.1)
            StarterCFU = st.number_input(get_text("starter_culture", lang_choice), min_value=0, value=1000000, step=10000)
            Extract_pct = st.number_input(get_text("extract_content", lang_choice), min_value=0.0, value=3.0, step=0.1)

        submitted = st.form_submit_button(get_text("save_data", lang_choice))

        if submitted:
            new_row = {
                "BatchID": next_id,
                "mass_kg": mass_kg,
                "T_initial_C": T_initial_C,
                "Salt_pct": Salt_pct,
                "Moisture_pct": Moisture_pct,
                "StarterCFU": StarterCFU,
                "Extract_pct": Extract_pct
            }
            try:
                append_row_excel(MEAT_DATA_XLSX, SHEET_NAME, new_row)
                st.success(f"{get_text('batch_added', lang_choice)}: '{next_id}'")
                st.cache_data.clear()
                load_all_data.clear()
            except Exception as e:
                st.error(f"{get_text('save_error', lang_choice)} {e}")

    st.markdown("---")
    st.subheader(get_text("current_data", lang_choice))
    st.dataframe(safe_read_excel(MEAT_DATA_XLSX, SHEET_NAME), use_container_width=True)
