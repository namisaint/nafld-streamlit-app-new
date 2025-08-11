# app.py
import os
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import gridfs
import certifi
from bson import ObjectId

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="NAFLD Predictor", page_icon="🤖", layout="wide")
st.title("🤖 NAFLD Lifestyle Risk Predictor")

st.caption("Build: dynamic inputs / MongoDB GridFS model loader / strict feature ordering")

# -----------------------
# EXACT 21 FEATURES (order matters)
# -----------------------
EXPECTED_COLS = [
    'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR',
    'ALQ111', 'ALQ121', 'ALQ142', 'ALQ151', 'ALQ170',
    'Is_Smoker_Cat',
    'SLQ050', 'SLQ120', 'SLD012',
    'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TSUGR', 'DR1TFIBE', 'DR1TTFAT',
    'PAQ620', 'BMXBMI'
]

RIDRETH3_CODE_MAP = {
    "Mexican American": 1,
    "Other Hispanic": 2,
    "Non-Hispanic White": 3,
    "Non-Hispanic Black": 4,
    "Non-Hispanic Asian": 6,
    "Other Race": 7
}

# -----------------------
# Load model from MongoDB GridFS (uses Streamlit secrets)
# st.secrets["mongo"] expected keys:
#   connection_string, db_name, bucket_name, model_file_id
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model_from_mongo():
    try:
        mongo_uri   = st.secrets["mongo"]["connection_string"]
        db_name     = st.secrets["mongo"]["db_name"]
        bucket_name = st.secrets["mongo"]["bucket_name"]
        file_id_str = st.secrets["mongo"]["model_file_id"]

        client = MongoClient(
            mongo_uri,
            server_api=ServerApi('1'),
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=8000
        )
        # Verify connectivity
        client.admin.command("ping")

        db = client[db_name]
        fs = gridfs.GridFS(db, collection=bucket_name)

        gridout = fs.get(ObjectId(file_id_str))
        model = joblib.load(BytesIO(gridout.read()))
        return model

    except Exception as e:
        # Best-effort diagnostics listing what files exist in the GridFS bucket
        diag = ""
        try:
            mongo_uri   = st.secrets["mongo"]["connection_string"]
            db_name     = st.secrets["mongo"]["db_name"]
            bucket_name = st.secrets["mongo"]["bucket_name"]
            client2 = MongoClient(mongo_uri, server_api=ServerApi('1'), tls=True, tlsCAFile=certifi.where())
            db2 = client2[db_name]
            names = [doc.get("filename") for doc in db2[f"{bucket_name}.files"].find({}, {"filename": 1}).limit(50)]
            diag = f"\nVisible files in {db_name}.{bucket_name}.files: {names}"
        except Exception:
            pass

        st.error(f"❌ Could not load model from MongoDB: {e}{diag}")
        st.stop()

with st.spinner("Loading model from MongoDB…"):
    model = load_model_from_mongo()
st.success("✅ Model loaded from MongoDB")

# -----------------------
# Sidebar (debug/info)
# -----------------------
with st.sidebar:
    st.markdown("### Expected features (app)")
    st.write(EXPECTED_COLS)
    try:
        model_cols = list(getattr(model, "feature_names_in_", []))
        if model_cols:
            st.markdown("### Features in model")
            st.write(model_cols)
            if model_cols != EXPECTED_COLS:
                st.warning("⚠ Model feature list/order differs from the app's EXPECTED_COLS.")
        else:
            st.info("Model does not expose feature_names_in_ (OK if it's a pipeline).")
    except Exception:
        st.info("Model does not expose feature_names_in_ (OK if it's a pipeline).")

# -----------------------
# Inputs (aligned to EXPECTED_COLS)
# -----------------------
st.header("Inputs")

c1, c2, c3 = st.columns(3)

with c1:
    gender_ui = st.selectbox("Gender (NHANES: 1=Male, 2=Female) — RIAGENDR", [1, 2], index=0)
    age_ui = st.slider("Age in years — RIDAGEYR", 18, 85, 40, 1)
    race_ui = st.selectbox("Race/Ethnicity — RIDRETH3", list(RIDRETH3_CODE_MAP.keys()), index=2)
    income_ui = st.slider("Family income ratio — INDFMPIR", 0.0, 10.0, 2.0, 0.1)

    # 0=Never, 1=Former, 2=Current (adjust if your training used a different mapping)
    smoker_ui = st.selectbox("Smoking status (Is_Smoker_Cat)", [0, 1, 2], index=0,
                             help="0=Never, 1=Former, 2=Current")

with c2:
    # SLQ050: Sleep disorder diagnosed? (0/1)
    sleep_disorder_ui = st.selectbox("Sleep disorder diagnosed? — SLQ050", [0, 1], index=0)

    # SLQ120: Hours usually sleep (float)
    sleep_hours_ui = st.slider("Sleep duration (hours/day) — SLQ120", 0.0, 24.0, 8.0, 0.25)

    pa_mins_ui = st.slider("Physical activity (minutes/day) — PAQ620", 0, 1440, 30, 5)
    bmi_ui = st.slider("BMI — BMXBMI", 10.0, 60.0, 25.0, 0.1)

with c3:
    alq111_ui = st.slider("Alcohol days/week — ALQ111", 0, 7, 0, 1)
    alq121_ui = st.slider("Alcohol drinks/day — ALQ121", 0, 50, 0, 1)
    alq142_ui = st.slider("Days drank in past year — ALQ142", 0, 366, 0, 1)
    alq151_ui = st.slider("Max drinks on any day — ALQ151", 0, 50, 0, 1)
    alq170_ui = st.slider("Intake freq (drinks/day) — ALQ170", 0.0, 50.0, 0.0, 0.1)

st.subheader("Nutrition (24h recall)")
n1, n2, n3 = st.columns(3)
with n1:
    kcal_ui = st.slider("Total calories (kcal) — DR1TKCAL", 0, 10000, 2000, 50)
    prot_ui = st.slider("Protein (g) — DR1TPROT", 0, 500, 60, 5)
with n2:
    carb_ui = st.slider("Carbohydrate (g) — DR1TCARB", 0, 1000, 250, 5)
    sug_ui = st.slider("Sugar (g) — DR1TSUGR", 0, 1000, 40, 5)
with n3:
    fib_ui = st.slider("Fibre (g) — DR1TFIBE", 0, 500, 30, 1)
    fat_ui = st.slider("Total fat (g) — DR1TTFAT", 0, 500, 70, 1)

# SLD012 is included in EXPECTED_COLS. If your training used a specific definition/coding,
# adjust this widget accordingly. Here we treat it as a simple binary indicator.
sld012_ui = st.selectbox("SLD012 (binary-coded)", [0, 1], index=0,
                         help="Set to match your training encoding. If unused in training, remove from EXPECTED_COLS.")

# -----------------------
# Build row (exact order)
# -----------------------
row = {
    'RIAGENDR': int(gender_ui),
    'RIDAGEYR': int(age_ui),
    'RIDRETH3': int(RIDRETH3_CODE_MAP[race_ui]),
    'INDFMPIR': float(income_ui),

    'ALQ111': int(alq111_ui),
    'ALQ121': int(alq121_ui),
    'ALQ142': int(alq142_ui),
    'ALQ151': int(alq151_ui),
    'ALQ170': float(alq170_ui),

    'Is_Smoker_Cat': int(smoker_ui),

    'SLQ050': int(sleep_disorder_ui),    # 0/1
    'SLQ120': float(sleep_hours_ui),     # hours
    'SLD012': int(sld012_ui),            # binary (adjust if needed)

    'DR1TKCAL': int(kcal_ui),
    'DR1TPROT': int(prot_ui),
    'DR1TCARB': int(carb_ui),
    'DR1TSUGR': int(sug_ui),
    'DR1TFIBE': int(fib_ui),
    'DR1TTFAT': int(fat_ui),

    'PAQ620': int(pa_mins_ui),
    'BMXBMI': float(bmi_ui),
}
X = pd.DataFrame([row], columns=EXPECTED_COLS)

# -----------------------
# Predict
# -----------------------
st.header("Prediction")

def _pos_idx(classes):
    """
    Return the index of the positive class for predict_proba.
    Tries common cases: 1, True, '1', or max numeric.
    """
    try:
        classes_list = list(classes)
        if 1 in classes_list: return classes_list.index(1)
        if True in classes_list: return classes_list.index(True)
        if "1" in classes_list: return classes_list.index("1")
        # Fallback: choose the class with the max numeric value
        nums = [float(c) for c in classes_list]
        return int(np.argmax(nums))
    except Exception:
        return 1 if classes is not None and len(classes) > 1 else 0

try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = getattr(model, "classes_", [0, 1])
        proba = float(probs[0][_pos_idx(classes)])
        yhat = int(model.predict(X)[0])
    else:
        yhat = int(model.predict(X)[0])
        proba = float(yhat)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.write("Inputs sent to model (ordered):", X.T.rename(columns={0: 'value'}))
    st.stop()

risk_pct = max(0.0, min(100.0, proba * 100.0))
risk_label = "High Risk" if risk_pct >= 70 else "Moderate Risk" if risk_pct >= 30 else "Low Risk"

st.markdown(f"### Predicted NAFLD Risk: **{risk_pct:.2f}% ({risk_label})**")
st.progress(risk_pct / 100.0)

# Debug / transparency
st.caption(f"Run timestamp: {datetime.utcnow().isoformat(timespec='seconds')}Z")
st.write("**Inputs sent to the model (verify they change as you move sliders):**")
st.dataframe(X.T.rename(columns={0: 'value'}), use_container_width=True)

st.divider()
st.markdown(
    "If predictions seem unresponsive, verify that your trained model's feature order "
    "matches `EXPECTED_COLS` exactly, and that categorical encodings (e.g., `Is_Smoker_Cat`) "
    "are the same as in training."
)
