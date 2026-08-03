import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# Load files
# -----------------------------

with open("churn_model.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl","rb") as f:
    feature_names = pickle.load(f)

with open("metadata.pkl","rb") as f:
    metadata = pickle.load(f)

st.set_page_config(page_title="Customer Churn Prediction",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Telco Customer Churn Prediction")

st.write("Predict whether a telecom customer is likely to churn.")

st.sidebar.header("Model Information")

st.sidebar.write(metadata)

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------

input_data = {}

for feature in feature_names:

    value = st.number_input(
        feature,
        value=0.0,
        step=0.1
    )

    input_data[feature] = value

if st.button("Predict Churn"):

    df = pd.DataFrame([input_data])

    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    st.markdown("---")

    if prediction == 1:

        st.error("⚠ Customer is likely to CHURN")

    else:

        st.success("✅ Customer is NOT likely to churn")

    st.metric("Churn Probability", f"{probability:.2%}")
