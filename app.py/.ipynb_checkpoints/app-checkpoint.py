import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model_1","rb"))

st.title("Loanlens : Loan Eligibility Model")

applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
employment_status = st.selectbox("Employment Status",["Salaried", "Self-Employed","Contract","Unemployed"])
age = st.number_input("Age")
marital_status = st.selectbox("Marital Status",["Single", "Married"])
dependents  = st.number_input("Dependents")
credit_score = st.number_input("Credit Score")
existing_loans = st.number_input("Existing loans")
dti_ratio = st.number_input("DTI Ratio")
savings = st.number_input("Savings")
collateral_value = st.number_input("Collateral Value")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
loan_purpose = st.selectbox("Loan Purpose",["Personal", "Car", "Business", "Home", "Education"])
property_area = st.selectbox("Property Area",["Urban", "Semiurban","Rural"])
edu_level = st.selectbox("Education Level",["Graduate","Not Graduate"])
gender = st.selectbox("Gender",["Male", "Female"])
emp_category = st.selectbox("Employer Category",["Private", "Government", "MNC", "Business" ,"Unemployed"])


input_data = np.array([[applicant_income, coapplicant_income, employment_status, 
                        age, marital_status, dependents, credit_score, existing_loans, dti_ratio , 
                        savings, collateral_value , loan_amount ,loan_term , loan_purpose, property_area
                        , edu_level, gender, emp_category
                       ]])

prediction = model.predict(input_data)

if prediction[0] == 1 :
    st.success("Loan Approved")
else :
    st.error("Loan Rejected")    