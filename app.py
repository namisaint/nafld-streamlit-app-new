# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from datetime import datetime

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
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model_from_mongo():
    """
    Loads a joblib sklearn pipeline from MongoDB GridFS using secrets:
      [mongo]
      connection_string = "...mongodb+srv://..."
      db_name = "NAFLD_Models"
      bucket_name = "fs"
      model_file_id = "68999a69058cb88ad9dc1ea0"   # example
    """
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
        client.admin.command("ping")

        db = client[db_name]
        fs = gridfs.GridFS(db, collection=bucket_name)

        # Convert string to ObjectId
        gridout = fs.get(ObjectId(file_id_str))
        model = joblib.load(BytesIO(gridout.read()))
        return model

    except Exception as e:
        # Helpful diagnostics: list available files in the bucket
        try:
            names = [f.get("filename") for f in db[f"{bucket_name}.files"].find({}, {"filename":1})]
            st.error(f"❌ Could not load model: {e}\n\nVisible files in {db_name}.{bucket_name}.files: {names}")
        except Exception:
            st.error(f"❌ Could not load model from MongoDB: {e}")
        st.stop()

with st.spinner("Loading model from MongoDB…"):
    model = load_model_from_mongo()
st.success("✅ Model loaded from MongoDB")

# -----------------------
# Sidebar (optional info)
# -----------------------
with st.sidebar:
    st.markdown("### Expected features (app)")
    st.write(EXPECTED_COLS)
    try:
        model_cols = list(model.feature_names_in_)
        st.markdown("### Features in model")
        st.write(model_cols)
        if model_cols != EXPECTED_COLS:
            st.warning("⚠ Model feature list differs from the app's 21 features.")
    except Exception:
        st.info("Model does not expose feature_names_in_ (ok if saved as a pipeline).")

# -----------------------
# Inputs
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
    sleep_hours_ui = st.slider("Sleep duration (hours/day)", 0.0, 24.0, 8.0, 0.25)
    work_hours_ui = st.slider("Work schedule duration (hours)", 0, 24, 8, 1)
    sleep_disorder_ui = st.selectbox("Sleep Disorder Status", ["No", "Yes"], index=0)
    pa_mins_ui = st.slider("Physical activity (minutes/day)", 0, 1440, 30, 5)
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
    kcal_ui = st.slider("Total calorie intake (kcal)", 0, 10000, 2000, 50)
    prot_ui = st.slider("Total protein intake (grams)", 0, 500, 60, 5)
with n2:
    carb_ui = st.slider("Total carbohydrate intake (grams)", 0, 1000, 250, 5)
    sug_ui = st.slider("Total sugar intake (grams)", 0, 1000, 40, 5)
with n3:
    fib_ui = st.slider("Total fiber intake (grams)", 0, 500, 30, 1)
    fat_ui = st.slider("Total fat intake (grams)", 0, 500, 70, 1)

# -----------------------
# Build row (exact order)
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
# Predict
# -----------------------
st.header("Prediction")

def _pos_idx(classes):
    try:
        if 1 in classes: return list(classes).index(1)
        if True in classes: return list(classes).index(True)
        if "1" in classes: return list(classes).index("1")
        nums = [float(c) for c in classes]
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
    st.write("Inputs sent to model:", X.T.rename(columns={0: 'value'}))
    st.stop()

risk_pct = max(0.0, min(100.0, proba * 100.0))
risk_label = "High Risk" if risk_pct >= 70 else "Moderate Risk" if risk_pct >= 30 else "Low Risk"

st.markdown(f"### Predicted NAFLD Risk: **{risk_pct:.2f}% ({risk_label})**")
st.progress(risk_pct / 100.0)

st.write("**Inputs sent to the model:**")
st.dataframe(X.T.rename(columns={0: 'value'}))
