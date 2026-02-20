import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Retail Profit Predictor",
    page_icon="📊",
    layout="centered"
)

# -----------------------------------
# Load Model (Safe Loading)
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("rf_pipeline.pkl")

model = load_model()

# -----------------------------------
# Title
# -----------------------------------
st.title("📊 Retail Profit Prediction")
st.write("Predict whether a retail order will be **Profitable** or **Loss-Making**.")

st.divider()

# -----------------------------------
# User Inputs
# -----------------------------------
sales = st.number_input("Sales", min_value=0.0, value=100.0)
discount = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.1)

category = st.selectbox(
    "Category",
    ["Technology", "Furniture", "Office Supplies"]
)

sub_category = st.selectbox(
    "Sub-Category",
    ["Chairs", "Tables", "Phones"]
)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🚀 Predict", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Sales": [sales],
        "Discount": [discount],
        "Category": [category],
        "Sub-Category": [sub_category]
    })

    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        profit_prob = probability[0][1] * 100

        st.divider()

        if prediction[0] == 1:
            st.success("✅ This Order is Profitable")
        else:
            st.error("⚠️ This Order is Likely Loss-Making")

        st.metric("Profit Probability", f"{profit_prob:.2f}%")

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check input values.")
