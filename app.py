import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Stock Price Prediction")

file = st.file_uploader("Upload CSV file for prediction")

if file:
    data = pd.read_csv(file)
    st.write("Uploaded Data", data.head())

    if st.button("Predict"):
        prediction = model.predict(data)
        st.write("Predicted Prices:", prediction)
