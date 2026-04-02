import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('heart_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("❤️ Cardiovascular Disease Prediction App")
st.write("Enter your health details to check your heart disease risk.")

# Inputs
age = st.number_input("Age (years)", 20, 100, 40)

gender = st.selectbox("Gender", ["Male", "Female"])
height = st.number_input("Height (cm)", 100, 220, 165)
weight = st.number_input("Weight (kg)", 30, 200, 70)

ap_hi = st.number_input("Systolic Blood Pressure", 80, 200, 120)
ap_lo = st.number_input("Diastolic Blood Pressure", 50, 130, 80)

# Better UI labels
chol = st.selectbox(
    "Cholesterol Level",
    [1, 2, 3],
    format_func=lambda x: "Normal" if x==1 else ("High" if x==2 else "Very High")
)

gluc = st.selectbox(
    "Glucose Level",
    [1, 2, 3],
    format_func=lambda x: "Normal" if x==1 else ("High" if x==2 else "Very High")
)

smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
active = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

# Convert values
gender = 2 if gender == "Male" else 1

# IMPORTANT: same as training (age in days)
age = age * 365

# Feature array (correct order)
features = np.array([[age, gender, height, weight,
                      ap_hi, ap_lo, chol, gluc,
                      smoke, alco, active]])

# Scale
features_scaled = scaler.transform(features)

# Predict
if st.button("🔍 Predict"):
    result = model.predict(features_scaled)

    if result[0] == 1:
        st.error("⚠️ High Risk: You may have cardiovascular disease.")
    else:
        st.success("✅ Low Risk: You are likely healthy.")

# Footer
st.write("---")
st.caption("Developed by Komal | BSc CS | AI Project")