import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
model = pickle.load(open("rf_pipeline.pkl", "rb"))

st.set_page_config(page_title="Retail Profit Predictor")

st.title("🛒 Retail Profit Prediction App")
st.write("Predict whether a retail order will be Profitable or Loss-Making.")

# User Inputs
sales = st.number_input("Sales", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1)
discount = st.slider("Discount", 0.0, 1.0)

category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
region = st.selectbox("Region", ["Central", "East", "South", "West"])
segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])

sub_category = st.text_input("Sub-Category (Example: Tables, Chairs, Phones)")

# Create input dataframe
input_data = pd.DataFrame({
    "Sales": [sales],
    "Quantity": [quantity],
    "Discount": [discount],
    "Category": [category],
    "Sub-Category": [sub_category],
    "Region": [region],
    "Segment": [segment]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ This order is likely Profitable")
    else:
        st.error("⚠ This order is likely Loss-Making")
