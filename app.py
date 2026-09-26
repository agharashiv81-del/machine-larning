import streamlit as st
import joblib

model = joblib.load("model.pkl")
contract_encoder = joblib.load("contract_encoder.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer information to predict whether the customer may churn.")

tenure = st.number_input(
    "Customer Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charge = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=500.0,
    value=50.0
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

if st.button("Predict Churn"):

    contract_encoded = contract_encoder.transform([contract])[0]

    input_data = [[
        tenure,
        monthly_charge,
        contract_encoded
    ]]

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")
