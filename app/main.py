import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.model_loader import load_trained_model
from app.utils import preprocess_input

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to check if it's fraudulent.")

# Example fields based on the credit card dataset
features = {
    "scaled_time": 0.0,
    "scaled_amount": 0.0,
    "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0,
    "V6": 0.0, "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0,
    "V11": 0.0, "V12": 0.0, "V13": 0.0, "V14": 0.0, "V15": 0.0,
    "V16": 0.0, "V17": 0.0, "V18": 0.0, "V19": 0.0, "V20": 0.0,
    "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0, "V25": 0.0,
    "V26": 0.0, "V27": 0.0, "V28": 0.0,
}

user_input = {}
st.subheader("📝 Input Transaction Details")

# Generate numeric input boxes
for key in features.keys():
    user_input[key] = st.number_input(f"{key}", value=0.0)

if st.button("Predict"):
    st.write("🔍 Evaluating transaction...")
    input_df = preprocess_input(user_input)
    model = load_trained_model()

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! Probability: {proba:.2f}")
    else:
        st.success(f"✅ Transaction is legitimate. Probability of fraud: {proba:.2f}")
