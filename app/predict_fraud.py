import streamlit as st
import pandas as pd
import joblib

# ==== Load model artifacts ====
MODEL_PATH = "D:/Unified Mentors/fraud_detection/fraud_detection/saved_models/model.pkl"
FEATURES_PATH = "D:/Unified Mentors/fraud_detection/fraud_detection/saved_models/features.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# ==== Streamlit UI ====
st.set_page_config(page_title="💳 Fraud Detection")
st.title("💳 Fraud Transaction Detection")
st.markdown("Enter transaction details to check if it's **fraudulent**.")

with st.form("predict_form"):
    tx_amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    customer_tx_count = st.number_input("Customer Transaction Count", min_value=1, value=10)
    customer_avg = st.number_input("Customer Avg Amount", min_value=0.0, value=100.0)
    customer_std = st.number_input("Customer Std Dev", min_value=0.0, value=20.0)
    tx_diff = st.number_input("TX Amount - Avg", min_value=-1000.0, max_value=1000.0, value=0.0)
    tx_hour = st.slider("Transaction Hour", min_value=0, max_value=23, value=12)
    tx_day = st.slider("Transaction Day", min_value=1, max_value=31, value=15)
    customer_id = st.number_input("Customer ID (encoded)", min_value=0, value=100)
    terminal_id = st.number_input("Terminal ID (encoded)", min_value=0, value=50)
    submit = st.form_submit_button("🔍 Detect Fraud")

# ==== Prediction ====
if submit:
    input_data = pd.DataFrame([{
        "TX_AMOUNT": tx_amount,
        "CUSTOMER_TX_COUNT": customer_tx_count,
        "CUSTOMER_AVG_AMOUNT": customer_avg,
        "CUSTOMER_STD_AMOUNT": customer_std,
        "TX_AMOUNT_DIFF": tx_diff,
        "TX_HOUR": tx_hour,
        "TX_DAY": tx_day,
        "CUSTOMER_ID": customer_id,
        "TERMINAL_ID": terminal_id
    }])

    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.markdown(f"### 🎯 Fraud Probability: `{prob*100:.2f}%`")
    if pred == 1:
        st.error("🚨 This transaction is predicted to be FRAUDULENT!")
    else:
        st.success("✅ This transaction appears to be LEGITIMATE.")
