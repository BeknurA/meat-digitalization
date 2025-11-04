# Meat_Digitalization/pages/ml_training.py
import streamlit as st
import pandas as pd
import numpy as np
from ui import get_text, df_to_download_link
from ml import ph_model
from database import insert_measurement

def compute_score_from_ph(ph_value):
    if ph_value is None or (isinstance(ph_value, float) and np.isnan(ph_value)):
        return None
    return round(max(0.0, 10.0 - abs(ph_value - 6.5)), 2)

def show_ml_train_predict(lang_choice):
    st.title(get_text("ml_title", lang_choice))
    st.markdown(get_text("ml_desc", lang_choice))

    tab1, tab2 = st.tabs([get_text("train_tab", lang_choice), get_text("predict_tab", lang_choice)])

    with tab1:
        st.subheader(get_text("train_subtitle", lang_choice))
        up = st.file_uploader(get_text("upload_train", lang_choice),
                              type=["csv", "xlsx", "xls"], key="train_up")

        if up:
            try:
                if up.name.lower().endswith(".csv"):
                    df_train = pd.read_csv(up)
                else:
                    df_train = pd.read_excel(up)
            except Exception as e:
                st.error(f"{get_text('train_error', lang_choice)} {e}")
                df_train = pd.DataFrame()

            if df_train.empty:
                st.info(get_text("no_data", lang_choice))
            else:
                st.write(get_text("preview", lang_choice))
                st.dataframe(df_train.head(10))

                cols = df_train.columns.tolist()
                if 'pH' in cols:
                    target = 'pH'
                else:
                    target = st.selectbox(get_text("target_column", lang_choice), options=cols)

                features = st.multiselect(get_text("features", lang_choice), options=cols)

                if st.button(get_text("train_button", lang_choice)):
                    try:
                        metrics = ph_model.train(
                            df_train,
                            target=target,
                            feature_cols=features if features else None
                        )
                        st.success(get_text("train_success", lang_choice))
                        st.json(metrics)
                    except Exception as e:
                        st.error(f"{get_text('train_error', lang_choice)} {e}")

    with tab2:
        st.subheader(get_text("predict_subtitle", lang_choice))
        up2 = st.file_uploader(get_text("upload_predict", lang_choice),
                               type=["csv", "xlsx", "xls"], key="pred_up")

        if up2:
            try:
                if up2.name.lower().endswith(".csv"):
                    df_pred = pd.read_csv(up2)
                else:
                    df_pred = pd.read_excel(up2)
            except Exception as e:
                st.error(f"Ошибка чтения: {e}")
                df_pred = pd.DataFrame()

            if df_pred.empty:
                st.info(get_text("no_data", lang_choice))
            else:
                st.dataframe(df_pred.head(10))
                num_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
                st.write(f"{get_text('auto_features', lang_choice)} {num_cols}")

                if st.button(get_text("predict_button", lang_choice)):
                    preds = ph_model.predict(df_pred, feature_cols=num_cols)
                    df_pred['predicted_pH'] = np.round(preds, 3)
                    df_pred['score'] = df_pred['predicted_pH'].apply(compute_score_from_ph)

                    st.subheader(get_text("predict_results", lang_choice))
                    st.dataframe(df_pred.head(50))
                    st.markdown(df_to_download_link(df_pred, filename="predictions.csv"), unsafe_allow_html=True)

                    if 'sample_name' in df_pred.columns:
                        if st.button(get_text("save_to_db", lang_choice)):
                            saved = 0
                            for _, r in df_pred.iterrows():
                                insert_measurement(
                                    str(r.get('sample_name', 'sample')),
                                    float(r.get('predicted_pH', np.nan)),
                                    compute_score_from_ph(float(r.get('predicted_pH', np.nan))),
                                    notes="predicted"
                                )
                                saved += 1
                            st.success(f"{get_text('saved_records', lang_choice)} {saved}")
