import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System (NSL-KDD)")

# ==========================================================
# SAFE LOAD ALL MODELS
# ==========================================================
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading {name}: {e}")
        st.stop()

model = safe_load("xgb_model.pkl", "xgb_model.pkl")
scaler = safe_load("scaler.pkl", "scaler.pkl")
selector = safe_load("selector.pkl", "selector.pkl")
label_encoders = safe_load("label_map.pkl", "label_map.pkl")

if not isinstance(label_encoders, dict):
    st.error("❌ label_map.pkl should contain a dictionary of label encoders!")
    st.stop()

st.success("✅ All model files loaded successfully!")

# ==========================================================
# ORIGINAL NSL-KDD COLUMN NAMES
# ==========================================================
col_names = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

# ==========================================================
# FILE UPLOADER
# ==========================================================
st.subheader("📂 Upload Raw NSL-KDD Test File (.txt or .csv)")

uploaded = st.file_uploader("Upload dataset", type=["txt", "csv"])

if uploaded is not None:
    st.info("File uploaded. Processing...")

    # Load input file
    try:
        if uploaded.name.endswith(".txt"):
            df = pd.read_csv(uploaded, names=col_names)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    # Drop unused columns
    df = df.drop(columns=["label", "difficulty"], errors="ignore")

    # ======================================================
    # APPLY LABEL ENCODERS
    # ======================================================
    categorical_cols = ["protocol_type", "service", "flag"]

    for col in categorical_cols:
        if col not in df.columns:
            st.error(f"❌ Missing categorical column: {col}")
            st.stop()

        if col not in label_encoders:
            st.error(f"❌ No label encoder found for {col}.")
            st.stop()

        le = label_encoders[col]

        try:
            df[col] = le.transform(df[col])
        except Exception:
            st.error(f"❌ Unknown category in column {col}. Clean your data.")
            st.stop()

    # ======================================================
    # SCALE INPUT
    # ======================================================
    try:
        X_scaled = scaler.transform(df)
    except Exception as e:
        st.error(f"❌ Scaling error: {e}")
        st.stop()

    # ======================================================
    # FEATURE SELECTION
    # ======================================================
    try:
        X_sel = selector.transform(X_scaled)
    except Exception as e:
        st.error(f"❌ Feature selection error: {e}")
        st.stop()

    # ======================================================
    # PREDICT
    # ======================================================
    try:
        preds = model.predict(X_sel)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # ======================================================
    # MAP PREDICTED LABELS
    # ======================================================
    reverse_map = {
        0: "dos",
        1: "normal",
        2: "probe",
        3: "r2l",
        4: "u2r"
    }

    pred_labels = [reverse_map.get(p, "unknown") for p in preds]

    # Display results
    st.subheader("📊 Predictions")
    results_df = pd.DataFrame({"Prediction": pred_labels})
    st.dataframe(results_df)

    st.subheader("📌 Summary")
    st.write(results_df["Prediction"].value_counts())

    # Download button
    csv_download = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Predictions CSV", csv_download, "predictions.csv", "text/csv")
