import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1230 100%);
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff5f6d, #ffc371);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #b8b8c8;
        font-size: 1.05rem;
        max-width: 620px;
        margin: 0 auto;
    }

    /* Section cards */
    .section-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffc371;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff5f6d, #ffc371);
        color: #1a1230;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        border-radius: 14px;
        padding: 0.9rem 0;
        margin-top: 0.5rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 95, 109, 0.35);
        color: #1a1230;
    }

    /* Result boxes */
    .result-box {
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .high-risk {
        background: rgba(255, 70, 70, 0.12);
        border: 1px solid rgba(255, 70, 70, 0.45);
        color: #ff6b6b;
    }
    .low-risk {
        background: rgba(60, 220, 150, 0.12);
        border: 1px solid rgba(60, 220, 150, 0.45);
        color: #3ddc97;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
    }

    /* Labels */
    label, .stMarkdown, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #e6e6f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load model artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('KNN_heart.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('columns.pkl')
    return model, scaler, expected_columns

model, scaler, expected_columns = load_artifacts()

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p>Enter your clinical details below and get an instant, ML-powered estimate
    of your heart disease risk. This tool is for informational purposes only and
    is not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar - quick info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this tool")
    st.write(
        "This predictor uses a **K-Nearest Neighbors (KNN)** model trained on "
        "clinical heart health data to estimate risk of heart disease."
    )
    st.markdown("---")
    st.markdown("### 🩺 How it works")
    st.write(
        "1. Fill in your health details\n"
        "2. Click **Predict**\n"
        "3. View your risk result instantly"
    )
    st.markdown("---")
    st.caption("⚠️ Not a medical diagnosis. Always consult a doctor.")

# ----------------------------------------------------------------------------
# Input form, organized into sections
# ----------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧍 Personal Details</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🫀 Cardiac Indicators</div>', unsafe_allow_html=True)
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Predict button + result
# ----------------------------------------------------------------------------
predict_clicked = st.button("🔍 Predict My Risk")

if predict_clicked:
    with st.spinner("Analyzing your data..."):

        # Create a raw input dictionary
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Try to get probability if the model supports it
        risk_pct = None
        if hasattr(model, "predict_proba"):
            try:
                risk_pct = model.predict_proba(scaled_input)[0][1] * 100
            except Exception:
                risk_pct = None

    st.markdown("---")

    if prediction == 1:
        st.markdown(
            f"""<div class="result-box high-risk">
                ⚠️ High Risk of Heart Disease
                {f"<br><span style='font-size:1rem;font-weight:400;'>Estimated risk score: {risk_pct:.1f}%</span>" if risk_pct is not None else ""}
            </div>""",
            unsafe_allow_html=True
        )
        st.warning("Please consider consulting a cardiologist for a thorough evaluation.")
    else:
        st.markdown(
            f"""<div class="result-box low-risk">
                ✅ Low Risk of Heart Disease
                {f"<br><span style='font-size:1rem;font-weight:400;'>Estimated risk score: {risk_pct:.1f}%</span>" if risk_pct is not None else ""}
            </div>""",
            unsafe_allow_html=True
        )
        st.balloons()

    if risk_pct is not None:
        st.progress(min(int(risk_pct), 100))