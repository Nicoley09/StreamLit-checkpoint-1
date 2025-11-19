
import streamlit as st
import pandas as pd
import joblib

# Load trained model + feature list
model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

st.title("📱 Expresso Churn Prediction App")
st.write("Predict whether a telecom customer will churn based on their behavior data.")

# Collect user input dynamically
inputs = {}
for col in features:
    inputs[col] = st.number_input(f"Enter {col}", value=0.0)

# Convert to DataFrame when button is clicked
if st.button("Predict Churn"):
    input_df = pd.DataFrame([inputs])

    # Ensure correct data type
    input_df = input_df.astype(float)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is LIKELY to churn")
    else:
        st.success("✅ Customer is NOT likely to churn")
