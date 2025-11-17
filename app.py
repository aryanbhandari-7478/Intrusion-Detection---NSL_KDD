import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# LOAD MODEL + PREPROCESSORS
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")
label_encoders = joblib.load("label_encoders.pkl")   # dict: protocol, service, flag

# -----------------------------
# ATTACK LABEL MAP
# -----------------------------
inv_label_map = {0: "dos", 1: "normal", 2: "probe", 3: "r2l", 4: "u2r"}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")

st.title("🚨 Intrusion Detection System (IDS) Dashboard")
st.write("Upload sample network traffic data and the model will predict attack categories.")

st.markdown("---")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.write("### 📌 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        # -----------------------------
        # APPLY SAME PREPROCESSING
        # -----------------------------

        # Label encoding for 3 columns
        for col in ["protocol_type", "service", "flag"]:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

        # Scale features
        X_scaled = scaler.transform(df)

        # Select features
        X_selected = selector.transform(X_scaled)

        # Predict
        preds = model.predict(X_selected)

        df["predicted_class"] = preds
        df["predicted_label"] = df["predicted_class"].map(inv_label_map)

        st.markdown("---")
        st.write("### ✅ Predictions")
        st.dataframe(df[["predicted_label"]].head(20))

        # -----------------------------
        # COUNTS + SIMPLE VISUALIZATION
        # -----------------------------
        st.markdown("### 📊 Prediction Distribution")
        counts = df["predicted_label"].value_counts()

        st.bar_chart(counts)

        # -----------------------------
        # DOWNLOAD RESULTS
        # -----------------------------
        st.download_button(
            label="📥 Download Predictions CSV",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error while processing: {str(e)}")
else:
    st.info("Upload a CSV file to get predictions.")
