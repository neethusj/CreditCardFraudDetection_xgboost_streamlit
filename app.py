import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Load model and scaler
model=joblib.load("models/xgb_model.pkl")
scaler=joblib.load("models/scaler.pkl")

#Load Sample transactions stored from the data set-Test data
legit_samples=pd.read_csv("data/sample_legit.csv")
fraud_samples=pd.read_csv("data/sample_fraud.csv")

#Add dropdown for sample test data
st.title("Credit Card Fraud detection using xgboost")
st.write("Select a sample transaction to test the model")
option=st.selectbox("Choose a sample transaction",["None"]+[f"Legit Example {i+1}" for i in range(len(legit_samples))]+
                    [f"Fraud Example {i+1}" for i in range(len(fraud_samples))])
if option.startswith("Legit"):
    index=int(option.split()[-1])-1
    selected_sample=legit_samples.iloc[index]
elif option.startswith("Fraud"):
    index=int(option.split()[-1])-1
    selected_sample=fraud_samples.iloc[index]
else:
    selected_sample=None

#Run Prediction
if selected_sample is not None:
    st.write("Loaded Transaction Details:")
    st.write(selected_sample)
    
    
    input_data=selected_sample.drop("Class").to_frame().T
    

    #Scale time and amount
    input_data[['Time','Amount']]=scaler.transform(input_data[['Time','Amount']])

    #Predict Probability
    prob=model.predict_proba(input_data.values)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Fraud Likelihood: {prob*100:.2f}%")
    st.write("Classification:","Fraud" if prob>0.4 else "Legit")
