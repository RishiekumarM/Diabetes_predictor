import streamlit as st
import numpy as np
import pickle

# 🎯 Load trained LightGBM model
model = pickle.load(open("model.pkl", "rb"))

# 🖼️ Page config
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

# 🌙 Dark mode styling
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #ffffff;
    }
    .gradient-bar {
        background: linear-gradient(to right, #8A2BE2, #4B0082);
        height: 20px;
        border-radius: 5px;
    }
    .bar-container {
        background-color: #333;
        border-radius: 5px;
        padding: 3px;
    }
    </style>
""", unsafe_allow_html=True)

# 🧪 Sidebar: Model Settings
st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.298)
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.write("- Accuracy: **77.23%**")
st.sidebar.write("- F1 Score: **0.736**")
st.sidebar.write("- ROC-AUC: **0.8398**")
st.sidebar.markdown("Model: **LightGBM**  \nTuned via **Optuna**")

# 🩺 App Title
st.title("🩺 Diabetes Prediction App")
st.markdown("Use patient health metrics to predict diabetes risk with a tuned LightGBM model.")

# 📥 Input Section
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.slider("Pregnancies", 0, 20, 3)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 40, 120, 80)
    skin_thickness = st.slider("Skin Thickness", 0, 99, 20)

with col2:
    insulin = st.slider("Insulin", 0.0, 900.0, 80.0)
    bmi = st.slider("Body Mass Index (BMI)", 10.0, 50.0, 25.0)
    pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.3)
    age = st.slider("Age", 1, 100, 30)

# 🔮 Gradient progress bar
def gradient_bar(prob):
    st.markdown(f"""
        <div class="bar-container">
            <div class="gradient-bar" style="width:{prob*100:.1f}%"></div>
        </div>
        <p style="text-align:center; font-weight:bold; color:#8A2BE2;">Probability: {prob:.4f}</p>
    """, unsafe_allow_html=True)

# 🧮 Predict button
if st.button("🔍 Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, pedigree, age]])
    prob = model.predict_proba(input_data)[0][1]
    prediction = int(prob >= threshold)

    # 📊 Results
    st.subheader("📈 Prediction Result")
    st.metric(label="Prediction Probability", value=f"{prob:.4f}")
    st.metric(label="Threshold", value=f"{threshold:.3f}")
    st.metric(label="Outcome", value="Diabetic 🚨" if prediction else "Non-Diabetic ✅")

    # 🌡️ Gradient bar
    gradient_bar(prob)

    # 🧾 Diagnostic note
    st.write("### 🧠 Diagnostic Note")
    if prediction:
        st.warning("⚠️ High risk of diabetes. Please consult a medical professional.")
    else:
        st.success("✅ Low risk of diabetes. Keep up the healthy lifestyle!")

    # 📘 Explanation of probability
    st.write("### 📘 What Does This Probability Mean?")
    st.markdown(f"""
    The model estimates a **{prob:.2%} chance** that the patient is diabetic based on the input values.
    This probability reflects how closely the patient's data matches patterns seen in diabetic cases during training.
    If the probability exceeds the decision threshold (**{threshold:.2f}**), the model classifies the patient as diabetic.
    """)

# 📎 Footer
st.markdown("---")
st.markdown("#### 📢 Disclaimer")
st.info("This model achieves an accuracy of **77.23%** on the test set. Predictions are based on statistical patterns and should not be considered a substitute for professional medical advice.")
st.caption("Built using LightGBM, Optuna, and Streamlit")
