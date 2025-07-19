import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("best_model.pkl")
scaler = joblib.load("adult_scaler.pkl")
expected_columns = joblib.load("adult_columns.pkl")
encoders = joblib.load("adult_encoders.pkl")  # <-- Add this during training and save

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar input
st.header("Input Employee Details")
age = st.slider("Age", 18, 65, 30)
work_class = st.selectbox("Work class", encoders["workclass"].classes_)
education = st.selectbox("Education Level", encoders['education'].classes_)
occupation = st.selectbox("Job Role", encoders['occupation'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.slider("Capital Gain", 0, 10000, 0)
capital_loss = st.slider("Capital Loss", 0, 10000, 0)
hours_per_week = st.slider("Hours per week", 1, 80, 40)
native_country = st.selectbox("Native Country", encoders['native-country'].classes_)
experience = st.slider("Years of Experience", 0, 40, 5)

# Prepare input
input_dict = {
    'age': [age],
    'workclass': [work_class],
    'education': [education],
    'occupation': [occupation],
    'gender': [gender],
    'capital_gain': [capital_gain],
    'capital_loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'experience': [experience]
}
input_df = pd.DataFrame(input_dict)
input_data = pd.DataFrame(input_dict)

# Encode all categorical features
for col in ['education', 'occupation', 'gender', 'workclass', 'native-country']:
    encoder = encoders[col]
    input_df[col] = encoder.transform(input_df[col])


# Ensure all expected columns are present
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_columns]
scaled_input = scaler.transform(input_df)

st.write("### 🔎 Input Data")
st.write(input_data)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(scaled_input)
    st.success(f"✅ Prediction: {prediction[0]}")


