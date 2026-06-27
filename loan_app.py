import streamlit as st
import pandas as pd
import pickle

with open(
    "loan_pipeline.pkl",
    "rb"
) as f:

    pipeline = pickle.load(f)

st.set_page_config(
    page_title="Loan Approval System",
    layout="wide"
)

st.title(
    "🏦 Loan Eligibilty"
)

col1, col2 = st.columns(2)

with col1:

    # age = st.slider(
    #     "Age",
    #     21,
    #     60,
    #     30
    # )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "Postgraduate",
            "HighSchool"
        ]
    )

with col2:

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Salaried",
            "Business",
            "SelfEmployed"
        ]
    )

    years_employed = st.slider(
        "Years Employed",
        0,
        30,
        5
    )

st.subheader(
    "Financial Information"
)

applicant_income = st.number_input(
    "Applicant Income",
    placeholder="Enter Income"
)

co_income = st.number_input(
    "Co Applicant Income",
    value=0
)

cibil_score = st.slider(
    "CIBIL Score",
    300,
    900,
    750
)

existing_debt = st.number_input(
    "Existing Debt",
    placeholder="Enter existing Debt"
)

property_value = st.number_input(
    "Property Value",
    placeholder="Enter Property Values"
)

loan_amount = st.number_input(
    "Loan Amount",
    placeholder="Enter loan amount"
)

loan_term = st.selectbox(
    "Loan Term",
    [60, 120, 180, 240, 360]
)

if st.button("Predict"):
    if applicant_income <= 1000:
        st.error("Applicant Income must be greater than 1000")
        st.stop()
    if loan_amount <= 0:
        st.error("Loan Amount must be greater than 0")
        st.stop()


    user_data = pd.DataFrame({
        # "Age":[age],
        "Gender":[gender],
        "Marital_Status":[marital_status],
        "Education":[education],
        "Employment_Type":[employment_type],
        "Years_Employed":[years_employed],
        "Applicant_Income":[applicant_income],
        "CoApplicant_Income":[co_income],
        "CIBIL_Score":[cibil_score],
        "Existing_Debt":[existing_debt],
        "Property_Value":[property_value],
        "Loan_Amount":[loan_amount],
        "Loan_Term":[loan_term]
    })

    prediction = pipeline.predict(
        user_data
    )[0]

    probability = pipeline.predict_proba(
        user_data
    )[0][1]

    if prediction == 1:
        st.success("✅ YES, YOU'RE ELIGIBLE TO TAKE LOAN")
    else:
        st.error("❌ SORRY, YOU'RE NOT ELIGIBLE TO TAKE LOAN")

    st.metric(
        "Approval Probability",
        f"{probability*100:.2f}%"
    )

    total_income = (
        applicant_income +
        co_income
    )

    debt_ratio = round(
        existing_debt /
        (total_income + 1),
        2
    )

    st.subheader(
        "Applicant Summary"
    )

    summary = pd.DataFrame({
        "Metric":[
            "Total Income",
            "Loan Amount",
            "Property Value",
            "Existing Debt",
            "Debt Ratio",
            "CIBIL Score"
        ],
        "Value":[
            total_income,
            loan_amount,
            property_value,
            existing_debt,
            debt_ratio,
            cibil_score
        ]
    })

    st.dataframe(summary)
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px;'>
        Disclaimer: This AI-based prediction tool is for educational purposes only.
        Actual loan approval decisions are determined by financial institutions.
        They should not be considered financial,legal, or lending advice.
    </div>
    """,
    unsafe_allow_html=True
)





