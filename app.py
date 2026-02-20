import streamlit as st

st.set_page_config(
    page_title="Retail Profit Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        📊 Retail Profit Prediction App
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; font-size:18px;'>
        Predict whether a retail order will be <b>Profitable</b> or <b>Loss-Making</b>
    </p>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    sales = st.number_input("Sales", min_value=0.0)
    discount = st.number_input("Discount", min_value=0.0, max_value=1.0)

with col2:
    category = st.selectbox("Category", ["Technology", "Furniture", "Office Supplies"])
    sub_category = st.selectbox("Sub-Category", [...])
    
predict_button = st.button("🚀 Predict Profitability", use_container_width=True)

if predict_button:
    input_data = pd.DataFrame({
        "Sales": [sales],
        "Discount": [discount],
        "Category": [category],
        "Sub-Category": [sub_category]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    profit_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.success("✅ This Order is Profitable!")
    else:
        st.error("⚠️ This Order is Likely Loss-Making!")

    st.metric("Profit Probability", f"{profit_prob:.2f}%")


st.sidebar.header("📌 About Project")
st.sidebar.write("""
This ML model predicts whether a retail order will result in profit or loss.
Built using Random Forest and deployed on Streamlit Cloud.
""")

st.sidebar.write("Author: Harsh Slathia")

st.markdown("""
    <style>
    .stApp {
        background-color: #F8F9F9;
    }
    </style>
""", unsafe_allow_html=True)





