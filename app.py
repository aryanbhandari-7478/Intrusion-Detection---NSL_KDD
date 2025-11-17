import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")
st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h1 style='margin: 0;'>🔐 Intrusion Detection Dashboard (NSL-KDD)</h1>
        <h3 style='margin: 0; opacity: 0.7;'>Aryan</h3>
    </div>
""", unsafe_allow_html=True)


# -------------------------
# Load core artifacts (must exist)
# -------------------------
def safe_load(path, name):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load {name}: {e}")
        st.stop()

model = safe_load("xgb_model.pkl", "xgb_model.pkl")
scaler = safe_load("scaler.pkl", "scaler.pkl")
selector = safe_load("selector.pkl", "selector.pkl")
label_map = safe_load("label_map.pkl", "label_map.pkl")  # may be target-map or dict-of-maps

st.success("✅ Core artifacts loaded (model/scaler/selector/label_map).")

# expected raw NSL-KDD columns (41 features; label/difficulty optional)
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

cat_cols = ["protocol_type", "service", "flag"]

# -------------------------
# Helper to build encoders from label_map (if it contains maps)
# -------------------------
def build_encoders_from_label_map(label_map_obj):
    encs = {}
    # Case A: label_map is a dict mapping column->(map value->int)
    try:
        for col in cat_cols:
            mapping = label_map_obj.get(col, None)
            if mapping and isinstance(mapping, dict):
                # mapping keys are category strings
                le = LabelEncoder()
                classes = list(mapping.keys())
                le.fit(classes)
                encs[col] = le
        if len(encs) == len(cat_cols):
            return encs
    except Exception:
        pass
    return None

# -------------------------
# Helper to build encoders by fitting on uploaded df
# -------------------------
def build_encoders_from_data(df):
    encs = {}
    for col in cat_cols:
        le = LabelEncoder()
        vals = df[col].astype(str).unique().tolist()
        le.fit(vals)
        encs[col] = le
    return encs

# -------------------------
# File upload
# -------------------------
uploaded = st.file_uploader("Upload NSL-KDD raw file (.txt or .csv)", type=["txt", "csv"])
if uploaded is None:
    st.info("Upload a .txt or .csv file in NSL-KDD format (raw).")
    st.stop()

# -------------------------
# Read file (try comma, else whitespace)
# -------------------------
try:
    uploaded.seek(0)
    df_try = pd.read_csv(uploaded, header=None)
    if df_try.shape[1] != 42 and df_try.shape[1] != 41 and df_try.shape[1] != 43:
        uploaded.seek(0)
        df_try = pd.read_csv(uploaded, header=None, delimiter=r"\s+")
    df = df_try.copy()
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Assign expected columns if unnamed (most raw NSL-KDD txts have no header)
# Many raw files have 42 columns (41 features + label) or 43 (features + label + difficulty).
if df.shape[1] >= 41:
    # Trim/pad to first 41 feature columns + optional label/difficulty
    if df.shape[1] >= 43:
        # assume columns: 41 features + label + difficulty
        df = df.iloc[:, :43]
        cols = expected_columns + ["label", "difficulty"]
        df.columns = cols
    elif df.shape[1] == 42:
        # assume 41 features + label
        df = df.iloc[:, :42]
        cols = expected_columns + ["label"]
        df.columns = cols
    elif df.shape[1] == 41:
        df = df.iloc[:, :41]
        df.columns = expected_columns
else:
    st.error(f"Unexpected number of columns: {df.shape[1]}. Expected at least 41.")
    st.stop()

st.subheader("Preview (first 5 rows)")
st.dataframe(df.head())

# Drop label/difficulty for prediction
for c in ["label", "difficulty"]:
    if c in df.columns:
        # keep a copy if label exists for evaluation, but remove before prediction input
        df = df.drop(columns=[c])

# Ensure categorical columns exist
for c in cat_cols:
    if c not in df.columns:
        st.error(f"Missing required column: {c}")
        st.stop()

# -------------------------
# Build encoders: prefer label_map data, else fit on uploaded data
# -------------------------
encoders = build_encoders_from_label_map(label_map)
if encoders is None:
    st.warning("label_map.pkl does not contain categorical maps. Encoders will be built from uploaded file values. "
               "This may cause mismatch with training encodings — proceed with caution.")
    encoders = build_encoders_from_data(df)

# Confirm encoder classes for debugging (optional)
st.write("Encoder classes preview:")
for k, le in encoders.items():
    st.write(f"{k}: {len(le.classes_)} classes sample -> {le.classes_[:5]}")

# -------------------------
# Apply encoders safely (handle unseen by filling with mode)
# -------------------------
for col in cat_cols:
    df[col] = df[col].astype(str)
    le = encoders[col]

    # Mark unseen categories as NaN
    df[col] = df[col].apply(lambda x: x if x in le.classes_ else None)
    if df[col].isna().any():
        # replace unseen with most frequent value in column (from uploaded data)
        fill_val = df[col].mode().dropna().values
        if len(fill_val) == 0:
            # fallback to first class known by encoder
            fill_val = [le.classes_[0]]
        else:
            fill_val = [fill_val[0]]
        df[col].fillna(fill_val[0], inplace=True)

    # transform
    df[col] = le.transform(df[col])

# -------------------------
# Scale, select and predict
# -------------------------
try:
    X_scaled = scaler.transform(df)
except Exception as e:
    st.error(f"Scaling failed: {e}")
    st.stop()

try:
    X_sel = selector.transform(X_scaled)
except Exception as e:
    st.error(f"Feature selection failed: {e}")
    st.stop()

try:
    preds = model.predict(X_sel)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# reverse label_map if label_map contains mapping for target
# Many users have label_map as {'dos':0, ...} or {0:'dos'}. We handle both.
inv_label = None
if isinstance(label_map, dict):
    # if values are ints (target mapping), invert
    vals = list(label_map.values())
    if all(isinstance(x, int) for x in vals):
        inv_label = {v:k for k,v in label_map.items()}

if inv_label is None:
    # fallback to default mapping
    inv_label = {0: "dos", 1: "normal", 2: "probe", 3: "r2l", 4: "u2r"}

pred_labels = [inv_label.get(int(p), "unknown") for p in preds]

out_df = pd.DataFrame(df)  # include input columns
out_df["prediction"] = pred_labels

st.subheader("Prediction results (first 20 rows)")
st.dataframe(out_df[["prediction"]].head(20))

st.subheader("Summary counts")
st.write(out_df["prediction"].value_counts())

# allow download
st.download_button("Download predictions CSV", out_df.to_csv(index=False).encode("utf-8"), "predictions.csv")


