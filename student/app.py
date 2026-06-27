import streamlit as st
import joblib
import pandas as pd

# load model
model = joblib.load("model.pkl")

st.title("📊 Student Performance Prediction App")

st.write("Enter study hours to predict marks")

hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Predict Marks"):
    input_data = pd.DataFrame([[hours]], columns=["hours"])
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Marks: {round(prediction[0], 2)}")