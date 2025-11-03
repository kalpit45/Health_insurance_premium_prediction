import streamlit as st
import joblib
import numpy as np
import os

# Load model from same folder
model_path = os.path.join(os.path.dirname(__file__), "insurance_model.pkl")
model = joblib.load(model_path)

st.title("Health Insurance Annual Premium Predictor")

st.write("Enter the details below to estimate the **yearly (annual)** health insurance premium.")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", 10.0, 60.0)
smoker = st.selectbox("Do you Smoke?", ["No", "Yes"])

# Encoding
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

if st.button("Predict Annual Premium"):
    features = np.array([[age, sex, bmi, smoker]])
    prediction = model.predict(features)[0]

    st.success(f"✅ Predicted **Annual Premium**: ₹{prediction:.2f}")

    st.info("This prediction represents the **yearly insurance premium**, not monthly.")
