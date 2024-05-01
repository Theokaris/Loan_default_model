import pandas as pd
import pickle as pk
import streamlit as st
from loan_approval_function import preprocess_and_predict

trained_model = pk.load(open("loan_approval_model.pkl", "rb"))
data = pk.load(open("loan_approval_data.pkl", "rb"))


# steamlit title
st.title("Loan Approval Model")

# Input form
LOAN = st.number_input("Loan Amount", min_value=0.0)
MORTDUE = st.number_input("Mortgage Due", min_value=0.0)
VALUE = st.number_input("Property Value", min_value=0.0)
REASON = st.selectbox("Reason for Loan", ["HomeImp", "DebtCon"])
JOB = st.selectbox("Job Type", ["Other", "Office", "Sales", "Mgr", "ProfExe", "Self"])
YOJ = st.number_input("Years on Job", min_value=0.0)
DEROG = st.number_input("Number of Derogatory Reports", min_value=0.0)
DELINQ = st.number_input("Number of Delinquencies", min_value=0.0)
CLAGE = st.number_input("Age of oldest credit amount in months", min_value=0.0)
NINQ = st.number_input("Number of recent credit requests", min_value=0.0)
CLNO = st.number_input("Total credit accounts", min_value=0)
DEBTINC = st.number_input("Debt to Income Ratio", min_value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC]],
                              columns=["LOAN", "MORTDUE", "VALUE", "REASON", "JOB", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]) 
    prediction = preprocess_and_predict(input_data, trained_model)
    if prediction[0] == 1:
        st.write(f"{prediction[0]}: **Likely to default on Loan**")
    else:
        st.write(f"{prediction[0]}: **Unlikely to default on Loan**")