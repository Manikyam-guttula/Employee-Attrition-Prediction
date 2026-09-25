import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import streamlit as st
import pandas as pd
from predict import predict_attrition

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(DATA_DIR, "prediction_history.csv")

# TEAL & SLATE GRAY THEME
st.markdown(
    """
    <style>
    /* Main App Light Slate Background */
    .stApp {
        background-color: #f1f5f9;
        color: #0f172a;
    }
    
    /* Sidebar Pure White Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1;
    }
    [data-testid="stSidebar"] label {
        color: #334155 !important;
        font-weight: 600;
    }
    
    /* Inputs Styling */
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    
    /* Ocean Teal Sliders & Buttons */
    div[data-baseweb="slider"] div {
        background-color: #0d9488 !important;
    }
    .stButton > button {
        background-color: #0d9488 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #0f766e !important;
        color: #ffffff !important;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 Employee Attrition Prediction System")
st.write("Evaluate attrition risk using key organizational metrics.")

# SIDEBAR INPUTS
st.sidebar.header("Employee Attributes")
emp_id = st.sidebar.text_input("EmployeeID", value="EMP_001")
age = st.sidebar.slider("Age", 18, 65, 35)
department = st.sidebar.selectbox("Department", ["Sales", "Engineering", "HR", "Marketing", "Finance"])
monthly_income = st.sidebar.number_input("Monthly Income (₹)", min_value=15000, max_value=200000, value=50000, step=1000)
years_at_company = st.sidebar.slider("Years at Company", 0, 30, 5)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-5)", 1, 5, 3)
work_life_balance = st.sidebar.slider("Work Life Balance (1-5)", 1, 5, 3)
performance_rating = st.sidebar.slider("Performance Rating (1-5)", 1, 5, 3)
overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])
distance_from_home = st.sidebar.slider("Distance From Home (km)", 1, 50, 10)

input_data = {
    "Age": age,
    "Department": department,
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "PerformanceRating": performance_rating,
    "OverTime": overtime,
    "DistanceFromHome": distance_from_home
}

# PREDICT ACTION
if st.sidebar.button("Predict Attrition", use_container_width=True) or st.button("Predict Attrition"):
    try:
        prediction, probability = predict_attrition(input_data)
        
        new_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "EmployeeID": emp_id,
            "Predicted Target (Attrition)": prediction,
            "Attrition Risk Score": f"{probability * 100:.1f}%",
            "Age": age,
            "Department": department,
            "Monthly Income (₹)": monthly_income,
            "Years at Company": years_at_company,
            "Job Satisfaction": job_satisfaction,
            "Work Life Balance": work_life_balance,
            "Performance Rating": performance_rating,
            "OverTime": overtime,
            "Distance From Home (km)": distance_from_home
        }
        
        new_df = pd.DataFrame([new_record])
        if not os.path.exists(HISTORY_FILE):
            new_df.to_csv(HISTORY_FILE, index=False)
        else:
            new_df.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
            
        st.session_state["latest_eval"] = (emp_id, prediction, probability)
    except Exception as e:
        st.error(f"Error: {e}")

# RESULT DISPLAY
if "latest_eval" in st.session_state:
    current_emp, prediction, probability = st.session_state["latest_eval"]
    st.markdown("---")
    st.subheader(f"📊 Results for {current_emp}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Attrition Risk Score", value=f"{probability * 100:.1f}%")
    with col2:
        st.metric(label="Predicted Attrition", value=prediction)
        
    if prediction == "Yes":
        st.error(f"🚨 **High Risk of Attrition**: Employee {current_emp} is flagged as likely to leave.")
    else:
        st.success(f"✅ **Low Risk of Attrition**: Employee {current_emp} is likely to remain.")

st.markdown("---")

# HISTORICAL LOGS
st.subheader("📜 Historical Attrition Logs")

if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
    if not history_df.empty:
        st.dataframe(history_df.iloc[::-1], use_container_width=True)
    else:
        st.info("No records stored in historical log.")
else:
    st.info("No past history recorded yet.")
