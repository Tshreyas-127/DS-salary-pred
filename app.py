# app.py 
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and reference data
pipe = pickle.load(open('pipe.pkl', 'rb'))
ds_salaries = pickle.load(open('ds_salaries.pkl', 'rb'))

# Conversion rate from USD to INR
USD_TO_INR = 83  # Update this value if needed

st.title("💼 Data Science Salary Predictor")

# User Inputs
experience = st.selectbox("Experience Level", ds_salaries['experience_level'].unique())
employment = st.selectbox("Employment Type", ds_salaries['employment_type'].unique())
job = st.selectbox("Job Title", ds_salaries['job_title'].unique())
company_size = st.selectbox("Company Size", ds_salaries['company_size'].unique())
residence = st.selectbox("Employee Residence", ds_salaries['employee_residence'].unique())

# Predict Button
if st.button("Predict Salary"):
    input_data = pd.DataFrame([[experience, employment, job,
                                residence, company_size]],
                              columns=['experience_level', 'employment_type', 'job_title',
                                       'employee_residence', 'company_size'])
    
    salary_usd = np.exp(pipe.predict(input_data)[0])

    salary_inr = salary_usd * USD_TO_INR

    st.success(f"💲 Estimated Salary (USD): ${int(salary_usd):,}")
    st.success(f"🇮🇳 Estimated Salary (INR): ₹{int(salary_inr):,}")


