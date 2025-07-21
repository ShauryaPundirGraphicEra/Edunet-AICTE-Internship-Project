import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Load Keras model
@st.cache_resource
def load_model_h5():
    return load_model("mlp_model.h5")

model = load_model_h5()

# === Categorical variables & encoders === #
categorical_features = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "gender": ["Male", "Female"],
    "native-country": ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India"]  # Example subset
}

encoders = {}
for col, classes in categorical_features.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    encoders[col] = le

# === Sidebar Inputs === #
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 90, 30)
workclass = st.sidebar.selectbox("Workclass", categorical_features["workclass"])
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000)
educational_num = st.sidebar.slider("Education Number", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", categorical_features["marital-status"])
occupation = st.sidebar.selectbox("Occupation", categorical_features["occupation"])
relationship = st.sidebar.selectbox("Relationship", categorical_features["relationship"])
race = st.sidebar.selectbox("Race", categorical_features["race"])
gender = st.sidebar.selectbox("Gender", categorical_features["gender"])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", categorical_features["native-country"])

# === Build input DataFrame === #
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# === Preprocess === #
def preprocess(df):
    df = df.copy()
    for col in categorical_features:
        df[col] = encoders[col].transform(df[col])
    return df.astype(np.float32)

# === Predict === #
if st.button("Predict Salary Class"):
    processed_input = preprocess(input_df)
    prediction = model.predict(processed_input)
    pred_class = (prediction[0][0] > 0.5).astype(int)
    result = ">50K" if pred_class == 1 else "≤50K"
    st.success(f"✅ Prediction: {result}")
