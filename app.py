"""
app.py
------
Streamlit web interface for the Heart Disease (10-Year CHD Risk) Predictor.

Run with:
    streamlit run app.py
"""

import pandas as pd
import joblib
import streamlit as st

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# -----------------------------------------------------------
# Load the trained model (cached so it only loads once)
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file 'heart_disease_model.pkl' not found.\n\n"
        "Run `python train_model.py` first to train and save the model "
        "(make sure `framingham.csv` is in this folder)."
    )
    st.stop()

# -----------------------------------------------------------
# Header
# -----------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "This tool estimates a person's **10-year risk of Coronary Heart "
    "Disease (CHD)** based on the Framingham Heart Study dataset. "
    "Fill in the fields below and click **Predict**."
)

st.divider()

# -----------------------------------------------------------
# Input form
# -----------------------------------------------------------
with st.form("chd_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        male = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    with col2:
        education = st.selectbox(
            "Education level",
            options=[1.0, 2.0, 3.0, 4.0],
            format_func=lambda x: f"Level {int(x)}",
        )

    st.subheader("Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        currentSmoker = st.selectbox("Current smoker?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col4:
        cigsPerDay = st.number_input("Cigarettes per day", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    st.subheader("Medical History")
    col5, col6, col7 = st.columns(3)
    with col5:
        BPMeds = st.selectbox("On BP medication?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col6:
        prevalentStroke = st.selectbox("History of stroke?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col7:
        prevalentHyp = st.selectbox("Hypertension?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    diabetes = st.selectbox("Diabetes?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    st.subheader("Clinical Measurements")
    col8, col9 = st.columns(2)
    with col8:
        totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, max_value=700.0, value=200.0, step=1.0)
        sysBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        diaBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
    with col9:
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        heartRate = st.number_input("Heart Rate (bpm)", min_value=0.0, max_value=250.0, value=75.0, step=1.0)
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0, value=80.0, step=1.0)

    submitted = st.form_submit_button("🔍 Predict CHD Risk", use_container_width=True)

# -----------------------------------------------------------
# Prediction
# -----------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        "male": male,
        "age": age,
        "education": education,
        "currentSmoker": currentSmoker,
        "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds,
        "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp,
        "diabetes": diabetes,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose,
    }])

    prediction = model.predict(input_data)[0]

    # Probability, if the underlying model supports it
    try:
        probability = model.predict_proba(input_data)[0][1]
    except AttributeError:
        probability = None

    st.divider()
    if prediction == 1:
        st.error("⚠️ **High Risk** of 10-Year Coronary Heart Disease")
    else:
        st.success("✅ **Low Risk** of 10-Year Coronary Heart Disease")

    if probability is not None:
        st.metric("Estimated Risk Probability", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    st.caption(
        "⚠️ This tool is for educational purposes only and is **not** a "
        "substitute for professional medical advice, diagnosis, or treatment."
    )
