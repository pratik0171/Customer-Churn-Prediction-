import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load files
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.title(" Customer Churn Prediction System")

st.write("Enter Customer Information")

# ========== USER INPUTS ==========

tenure = st.number_input("Tenure (Months)", 0, 100)
monthly = st.number_input("Monthly Charges", 0.0, 500.0)
total = st.number_input("Total Charges", 0.0, 10000.0)

partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])
paperless = st.selectbox("Paperless Billing", ["Yes","No"])

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month","One year","Two year"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# ========== PREDICT ==========

if st.button("Predict"):

    # Create empty row with correct columns
    input_df = pd.DataFrame(0, index=[0], columns=features)

    # Numeric
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly
    input_df["TotalCharges"] = total

    # Binary
    input_df["Partner"] = 1 if partner=="Yes" else 0
    input_df["Dependents"] = 1 if dependents=="Yes" else 0
    input_df["PaperlessBilling"] = 1 if paperless=="Yes" else 0

    # Contract
    if contract != "Month-to-month":
        col = f"Contract_{contract}"
        if col in input_df.columns:
            input_df[col] = 1

    # Internet
    if internet != "No":
        col = f"InternetService_{internet}"
        if col in input_df.columns:
            input_df[col] = 1

    # Payment
    col = f"PaymentMethod_{payment}"
    if col in input_df.columns:
        input_df[col] = 1

    # Scale
    scaled = scaler.transform(input_df)

    # Predict
    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)[0][1]

    if pred[0] == 1:
        st.error(f" Customer Will Churn (Prob: {prob*100:.1f}%)")
    else:
        st.success(f" Customer Will Not Churn (Prob: {prob*100:.1f}%)")