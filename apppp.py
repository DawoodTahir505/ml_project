import streamlit as st
import pandas as pd
import joblib

# ── Load YOUR uploaded pkl files ───────────────────────────────────────────
model            = joblib.load("KNN_heart_project.pkl")
scaler           = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ── Page Setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.markdown("Fill in the patient details below and click **Predict**.")

# ── User Inputs (exact columns: age,sex,cp,trestbps,chol,fbs,
#                restecg,thalach,exang,oldpeak,slope,ca,thal) ───────────────
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 50)

    sex = st.selectbox("Sex",
                       options=[1, 0],
                       format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")

    cp = st.selectbox("Chest Pain Type (cp)",
                      options=[0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "0 – Typical Angina",
                          1: "1 – Atypical Angina",
                          2: "2 – Non-Anginal Pain",
                          3: "3 – Asymptomatic"
                      }[x])

    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)

    chol = st.number_input("Cholesterol (chol, mg/dL)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL (fbs)",
                       options=[0, 1],
                       format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    restecg = st.selectbox("Resting ECG (restecg)",
                           options=[0, 1, 2],
                           format_func=lambda x: {
                               0: "0 – Normal",
                               1: "1 – ST-T Abnormality",
                               2: "2 – LV Hypertrophy"
                           }[x])

    thalach = st.slider("Max Heart Rate (thalach)", 60, 220, 150)

    exang = st.selectbox("Exercise-Induced Angina (exang)",
                         options=[0, 1],
                         format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

with col3:
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

    slope = st.selectbox("Slope of ST Segment (slope)",
                         options=[0, 1, 2],
                         format_func=lambda x: {
                             0: "0 – Upsloping",
                             1: "1 – Flat",
                             2: "2 – Downsloping"
                         }[x])

    ca = st.selectbox("Major Vessels Colored (ca)", options=[0, 1, 2, 3, 4])

    thal = st.selectbox("Thalassemia (thal)",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: {
                            0: "0 – Normal",
                            1: "1 – Fixed Defect",
                            2: "2 – Reversible Defect",
                            3: "3 – Unknown"
                        }[x])

# ── Predict Button ─────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔮 Predict", use_container_width=True, type="primary"):

    # Build DataFrame with exact column names from columns.pkl
    input_data = pd.DataFrame([{
        'age':      age,
        'sex':      sex,
        'cp':       cp,
        'trestbps': trestbps,
        'chol':     chol,
        'fbs':      fbs,
        'restecg':  restecg,
        'thalach':  thalach,
        'exang':    exang,
        'oldpeak':  oldpeak,
        'slope':    slope,
        'ca':       ca,
        'thal':     thal,
    }])

    # Ensure correct column order from columns.pkl
    input_data = input_data[expected_columns]

    # Scale using your scaler.pkl
    scaled_input = scaler.transform(input_data)

    # Predict using your KNN_heart_project.pkl
    prediction = model.predict(scaled_input)[0]
    proba      = model.predict_proba(scaled_input)[0]

    # ── Result ─────────────────────────────────────────────────────────────
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** — Confidence: {proba[1]*100:.1f}%")
    else:
        st.success(f"✅ **Low Risk of Heart Disease** — Confidence: {proba[0]*100:.1f}%")

    c1, c2 = st.columns(2)
    c1.metric("P(No Disease)",    f"{proba[0]*100:.1f}%")
    c2.metric("P(Heart Disease)", f"{proba[1]*100:.1f}%")