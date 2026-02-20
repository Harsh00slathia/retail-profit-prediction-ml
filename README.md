📊 Retail Profit Prediction using Machine Learning
📌 Project Overview

This project analyzes retail sales data and builds a machine learning model to predict whether an order will be profitable or loss-making.

The solution is implemented as a production-ready ML pipeline and deployed using Streamlit Cloud.

The objective is to help businesses identify high-risk transactions and optimize pricing strategies using data-driven insights.

🎯 Business Objectives

Identify loss-making orders

Analyze the impact of discount on profitability

Support data-driven pricing decisions

Reduce financial risk through predictive modeling

🛠 Tech Stack

Python

pandas

numpy

matplotlib / seaborn

scikit-learn

Logistic Regression

Random Forest

Streamlit (Deployment)

joblib (Model Serialization)

📊 Exploratory Data Analysis (EDA)

Key Insights:

Technology category generated the highest overall sales.

Tables sub-category contributed to the highest losses.

Higher discounts strongly correlated with negative profit.

Excessive discounting significantly increases loss probability.

These insights guided feature engineering and model selection.

🤖 Machine Learning Approach
1️⃣ Logistic Regression

Accuracy: 92%

Recall (Loss Class): 95%

Precision (Loss Class): 71%

Strong recall but lower precision → More false positives.

2️⃣ Random Forest (Final Model)

Accuracy: 95%

Recall (Loss Class): 82%

Precision (Loss Class): 89%

Selected due to better balance between precision and recall, making it more suitable for real-world decision-making.

⚙️ ML Pipeline Architecture

The final model is built using:

ColumnTransformer

Numerical features → passthrough

Categorical features → OneHotEncoder (drop='first', handle_unknown='ignore')

Random Forest Classifier

End-to-end Pipeline serialized using joblib

The model handles unseen categories in production and is fully deployable.

🚀 Deployment

The model is deployed using Streamlit Cloud.

Production considerations handled:

Version compatibility issues

Model serialization using joblib

Handling unseen categorical variables

Cloud debugging and redeployment

🔗 Live App: https://retail-profit-prediction-ml-djbny7appx733irq94u4uuj.streamlit.app/
