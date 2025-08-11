# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pymongo import MongoClient
import gridfs
import certifi
from io import BytesIO

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="NAFLD Predictor", page_icon="🤖", layout="wide")
st.title("🤖 NAFLD Lifestyle Risk Predictor")

# -----------------------
# Constants: EXACT 21 FEATURES (order matters)
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
# Load model from MongoDB GridFS
# -----------------------
@st.cache_resource
def load_model_from_mongo():
    try:
        mongo_uri = st.secrets["mongo"]["connection_string"]
        db_name = st.secrets["mongo"]["db_name"]
        bucket_name = st.secrets["mongo"]["bucket_name"]
        file_id = st.secrets["mongo"]["model_file_id"]

        client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
        db = client[db_name]
        fs = gridfs.GridFS(db, bucket_name)

        model_file = fs.get(file_id)
        model_bytes = BytesIO(model_file.read())
        model = joblib.load(model_bytes)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model from MongoDB: {e}")
        st.stop()

model = load_model_from_mongo()

# -----------------------
# Sidebar info
# -----------------------
with st.sidebar:
    st.markdown("### Expected features (app)")
    st.write(EXPECTED_COLS)
    try:
        model_cols = list(model.feature_names_in_)
        st.markdown("### Features in model")
        st.write(model_cols)
        if model_cols != EXPECTED_COLS:
            st.warning("⚠ Model feature list differs from app's features.")
    except Exception:
        st.info("Model does not expose feature_names_in_ (ok).")

# -----------------------
# UI: Collect inputs
# -----------------------
st.header("Inputs")

c1, c2, c3 = st.columns(3)

with c1:
    gender_ui = st.selectbox("Gender", ["Male", "Female"], index=0)
    age_ui = st.slider("Age in years", 0, 120, 40, 1)
    race_ui = st.selectbox("Race/Ethnicity", list(RIDRETH3_CODE_MAP.keys()), index=2)
    income_ui = st.slider("Family income ratio", 0.0, 10.0, 2.0, 0.1)
    smoker_ui = st.selectbox("Smoking status", ["No", "Yes"], index=0)

with c2:
    sleep_hours_ui = st.slider("Sleep duration hours/day", 0.0, 24.0, 8.0, 0.25)
    work_hours_ui = st.slider("Work schedule duration hours", 0, 24, 8, 1)
    sleep_disorder_ui = st.selectbox("Sleep Disorder Status", ["No", "Yes"], index=0)
    pa_mins_ui = st.slider("Physical activity minutes/day", 0, 1440, 30, 5)
    bmi_ui = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)

with c3:
    alq111_ui = st.slider("ALQ111: Alcohol days/week", 0, 7, 0, 1)
    alq121_ui = st.slider("ALQ121: Alcohol drinks/day", 0, 50, 0, 1)
    alq142_ui = st.slider("ALQ142: Days drank in past year", 0, 366, 0, 1)
    alq151_ui = st.slider("ALQ151: Max drinks on any day", 0, 50, 0, 1)
    alq170_ui = st.slider("ALQ170: Intake freq (drinks/day)", 0.0, 50.0, 0.0, 0.1)

st.subheader("Nutrition")
n1, n2, n3 = st.columns(3)
with n1:
    kcal_ui = st.slider("Total kcal", 0, 10000, 2000, 50)
    prot_ui = st.slider("Protein (g)", 0, 500, 60, 5)
with n2:
    carb_ui = st.slider("Carbs (g)", 0, 1000, 250, 5)
    sug_ui = st.slider("Sugar (g)", 0, 1000, 40, 5)
with n3:
    fib_ui = st.slider("Fiber (g)", 0, 500, 30, 1)
    fat_ui = st.slider("Fat (g)", 0, 500, 70, 1)

# -----------------------
# Build row for prediction
# -----------------------
row = {
    'RIAGENDR': 1 if gender_ui == "Male" else 2,
    'RIDAGEYR': int(age_ui),
    'RIDRETH3': int(RIDRETH3_CODE_MAP[race_ui]),
    'INDFMPIR': float(income_ui),
    'ALQ111': int(alq111_ui),
    'ALQ121': int(alq121_ui),
    'ALQ142': int(alq142_ui),
    'ALQ151': int(alq151_ui),
    'ALQ170': float(alq170_ui),
    'Is_Smoker_Cat': 1 if smoker_ui == "Yes" else 0,
    'SLQ050': float(sleep_hours_ui),
    'SLQ120': int(work_hours_ui),
    'SLD012': 1 if sleep_disorder_ui == "Yes" else 0,
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
# Prediction
# -----------------------
try:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        pos_idx = 1  # assuming 0 = No, 1 = Yes
        proba = float(probs[0][pos_idx])
        yhat = int(model.predict(X)[0])
    else:
        yhat = int(model.predict(X)[0])
        proba = float(yhat)
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.write(X)
    st.stop()

risk_pct = proba * 100
risk_label = "High Risk" if risk_pct >= 70 else "Moderate Risk" if risk_pct >= 30 else "Low Risk"
risk_color = "red" if risk_pct >= 70 else "orange" if risk_pct >= 30 else "green"

st.markdown(
    f"### Predicted NAFLD Risk: **<span style='color:{risk_color}'>{risk_pct:.2f}% ({risk_label})</span>**",
    unsafe_allow_html=True
)
st.progress(risk_pct / 100)

st.write("**Inputs sent to the model:**")
st.dataframe(X.T.rename(columns={0: "value"}))
