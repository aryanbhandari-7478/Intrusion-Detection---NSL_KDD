import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# -------------------------------------------------------------
# AUTO-REBUILD ENCODERS FROM label_map.pkl (no external files!)
# -------------------------------------------------------------

def build_encoders(label_map):
    encoders = {}
    for col, mapping in label_map.items():
        le = LabelEncoder()
        classes = list(mapping.keys())
        le.fit(classes)
        encoders[col] = le
    return encoders


# -------------------------------------------------------------
# LOAD MODELS + REBUILD ENCODERS
# -------------------------------------------------------------
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
label_map = joblib.load("label_map.pkl")

encoders = build_encoders(label_map)

st.title("🚨 Intrusion Detection Dashboard (NSL-KDD)")

st.write("Upload **CSV or TXT** containing raw NSL-KDD data. The app will auto-process it.")

uploaded = st.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])

if uploaded:
    try:
        # -------------------------------------------------------------
        # HANDLE TXT OR CSV AUTOMATICALLY
        # -------------------------------------------------------------
        if uploaded.name.endswith(".txt"):
            df = pd.read_csv(uploaded, header=None)
        else:
            df = pd.read_csv(uploaded)

        st.write("### Raw Input Preview")
        st.dataframe(df.head())

        # -------------------------------------------------------------
        # SET COLUMN NAMES IF MISSING (NSL-KDD has fixed 41 features)
        # -------------------------------------------------------------
        expected_columns = [
            "duration","protocol_type","service","flag","src_bytes","dst_bytes",
            "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
            "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
            "num_shells","num_access_files","num_outbound_cmds","is_host_login",
            "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
            "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
            "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate"
        ]

        if df.shape[1] >= 41:
            df = df.iloc[:, :41]
            df.columns = expected_columns
        else:
            st.error(f"❌ Your file has only {df.shape[1]} columns. Expected 41.")
            st.stop()

        # -------------------------------------------------------------
        # APPLY LABEL ENCODING SAFELY
        # -------------------------------------------------------------
        for col in ["protocol_type", "service", "flag"]:
            if col not in df.columns:
                st.error(f"❌ Missing categorical column: {col}")
                st.stop()

            le = encoders[col]

            df[col] = df[col].astype(str)

            # Handle unseen categories
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else None)

            if df[col].isna().sum() > 0:
                st.warning(f"⚠ Some unseen categories found in {col}. Marked as NaN and filled with most frequent.")
                df[col].fillna(df[col].mode()[0], inplace=True)

            df[col] = le.transform(df[col])

        # -------------------------------------------------------------
        # SCALE + SELECT FEATURES + PREDICT
        # -------------------------------------------------------------
        X_scaled = scaler.transform(df)
        X_selected = selector.transform(X_scaled)
        y_pred = model.predict(X_selected)

        df["prediction"] = y_pred

        st.write("### ✅ Prediction Output")
        st.dataframe(df[["prediction"]].head())

        # -------------------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # -------------------------------------------------------------
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Output CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error while processing: {e}")
