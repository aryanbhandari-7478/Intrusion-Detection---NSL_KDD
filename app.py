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
        obj = joblib.load(path)
        return obj
    except Exception as e:
        st.error(f"❌ Could not load {name}: {e}")
        st.stop()


model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
label_encoders = joblib.load("label_map.pkl")   # dict: protocol, service, flag

if not isinstance(encoders, dict):
    st.error("❌ encoders.joblib exists but is not a dictionary!")
    st.stop()

st.success("✅ All model files loaded successfully!")

# ==========================================================
# COLUMN NAMES (same as training)
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
st.subheader("📂 Upload a raw NSL-KDD .txt or .csv file")

uploaded = st.file_uploader("Upload dataset (.txt or .csv)", type=["txt", "csv"])

if uploaded is not None:
    st.info("File uploaded. Processing...")

    # ======================================================
    # READ RAW FILE
    # ======================================================
    try:
        if uploaded.name.endswith(".txt"):
            df = pd.read_csv(uploaded, names=col_names)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()

    if "label" not in df.columns:
        st.warning("⚠️ No label column found — treating as unlabeled test data.")

    # Remove unused columns
    for c in ["label", "difficulty"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # ======================================================
    # APPLY LABEL ENCODERS
    # ======================================================
    cat_cols = ["protocol_type", "service", "flag"]

    for c in cat_cols:
        if c not in df.columns:
            st.error(f"❌ Missing categorical column: {c}")
            st.stop()

        le = encoders.get(c)
        if le is None:
            st.error(f"❌ Encoder missing for column: {c}")
            st.stop()

        try:
            df[c] = le.transform(df[c])
        except:
            st.error(f"❌ Unknown category value encountered in {c}. Clean your input file.")
            st.stop()

    # ======================================================
    # SCALING
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
        st.error(f"❌ Feature selection failed: {e}")
        st.stop()

    # ======================================================
    # PREDICT
    # ======================================================
    try:
        preds = model.predict(X_sel)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.stop()

    # Reverse label map
    label_map = {
        0: "dos",
        1: "normal",
        2: "probe",
        3: "r2l",
        4: "u2r"
    }

    df_results = pd.DataFrame({
        "Prediction": [label_map.get(p, "unknown") for p in preds]
    })

    st.subheader("📊 Predictions")
    st.dataframe(df_results)

    # Summary
    st.subheader("📌 Summary")
    st.write(df_results["Prediction"].value_counts())


st.markdown("---")
st.caption("Created for NSL-KDD Intrusion Detection System")
