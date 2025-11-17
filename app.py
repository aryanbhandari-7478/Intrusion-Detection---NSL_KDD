import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# LOAD MODEL + PREPROCESSORS
# -----------------------------
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
label_encoders = joblib.load("label_map.pkl")   # dict: protocol, service, flag

# ------------------------------------
# Official NSL-KDD 42 feature columns
# ------------------------------------
nsl_kdd_columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"
]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔐 Intrusion Detection System (NSL-KDD)")
st.write("Upload dataset (TXT or CSV) and get predictions.")

uploaded_file = st.file_uploader("Upload NSL-KDD file", type=["txt", "csv"])

if uploaded_file:
    try:
        # -----------------------------
        # Load TXT file (assign headers)
        # -----------------------------
        if uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, header=None)
            if df.shape[1] == 42:
                df.columns = nsl_kdd_columns
            else:
                st.error(f"TXT file must have 42 columns. Found: {df.shape[1]}")
                st.stop()

        # -----------------------------
        # Load CSV normally
        # -----------------------------
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("📌 Preview of Uploaded Data")
        st.dataframe(df.head())

        # -----------------------------
        # Drop label column if exists
        # -----------------------------
        if "label" in df.columns:
            df = df.drop(columns=["label"])

        # -----------------------------
        # Encode categorical columns
        # -----------------------------
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])

        # -----------------------------
        # Feature Selection
        # -----------------------------
        df_selected = selector.transform(df)

        # -----------------------------
        # Scaling
        # -----------------------------
        df_scaled = scaler.transform(df_selected)

        # -----------------------------
        # Prediction
        # -----------------------------
        preds = model.predict(df_scaled)

        # Optional mapping (your choice)
        mapping = {
            0: "normal",
            1: "dos",
            2: "probe",
            3: "r2l",
            4: "u2r"
        }
        pred_labels = [mapping[p] for p in preds]

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("🔎 Prediction Results")
        result_df = pd.DataFrame({
            "prediction": pred_labels
        })

        st.dataframe(result_df)

        st.success("Prediction completed successfully.")

    except Exception as e:
        st.error(f"Error while processing: {str(e)}")
