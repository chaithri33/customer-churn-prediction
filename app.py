import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Customer Churn Prediction')

st.write("""
Enter customer details to predict if they are likely to churn.
""")

# Example inputs (customize these!)
gender = st.selectbox('Gender', ['Male', 'Female'])
SeniorCitizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
Partner = st.selectbox('Has Partner?', ['Yes', 'No'])
Dependents = st.selectbox('Has Dependents?', ['Yes', 'No'])
tenure = st.slider('Tenure (Months)', 0, 72, 12)
PhoneService = st.selectbox('Phone Service?', ['Yes', 'No'])
PaperlessBilling = st.selectbox('Paperless Billing?', ['Yes', 'No'])
MonthlyCharges = st.number_input('Monthly Charges', 0.0, 200.0, 70.0)
TotalCharges = st.number_input('Total Charges', 0.0, 10000.0, 2500.0)
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox('Payment Method', [
    'Electronic check',
    'Mailed check',
    'Bank transfer (automatic)',
    'Credit card (automatic)'
])
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

# Binary columns
Partner = 1 if Partner == 'Yes' else 0
Dependents = 1 if Dependents == 'Yes' else 0
PhoneService = 1 if PhoneService == 'Yes' else 0
PaperlessBilling = 1 if PaperlessBilling == 'Yes' else 0
SeniorCitizen = 1 if SeniorCitizen == 'Yes' else 0

# One-hot encoding for multi-class
gender_Male = 1 if gender == 'Male' else 0

MultipleLines_No_phone_service = 0
MultipleLines_Yes = 0
# Simple example: assume no multiple lines
# You can expand this input if needed

InternetService_Fiber_optic = 1 if InternetService == 'Fiber optic' else 0
InternetService_No = 1 if InternetService == 'No' else 0

Contract_One_year = 1 if Contract == 'One year' else 0
Contract_Two_year = 1 if Contract == 'Two year' else 0

PaymentMethod_Credit_card = 1 if PaymentMethod == 'Credit card (automatic)' else 0
PaymentMethod_Electronic_check = 1 if PaymentMethod == 'Electronic check' else 0
PaymentMethod_Mailed_check = 1 if PaymentMethod == 'Mailed check' else 0

# Scale numeric
scaled_nums = scaler.transform([[tenure, MonthlyCharges, TotalCharges]])[0]

# Final feature order must match training!
input_data = np.array([SeniorCitizen, Partner, Dependents, scaled_nums[0],
                       PhoneService, PaperlessBilling, scaled_nums[1], scaled_nums[2],
                       gender_Male,
                       MultipleLines_No_phone_service, MultipleLines_Yes,
                       InternetService_Fiber_optic, InternetService_No,
                       Contract_One_year, Contract_Two_year,
                       PaymentMethod_Credit_card, PaymentMethod_Electronic_check,
                       PaymentMethod_Mailed_check])

input_df = pd.DataFrame([input_data])

# Predict
if st.button('Predict Churn'):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write(f'**Churn Prediction:** {"Yes" if prediction == 1 else "No"}')
    st.write(f'**Churn Probability:** {probability:.2f}')
