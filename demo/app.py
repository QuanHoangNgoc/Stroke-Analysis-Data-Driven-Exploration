import streamlit as st
from control import *
import pandas as pd

st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")
st.title("Demo Final Project CS313")
st.markdown("### Predict your chances of having a stroke 🧠")

st.markdown("#### Information")
col1, col2, col3 = st.columns(3)
with col1:
    name = st.text_input("Your name")
    gender = st.selectbox("Gender", ["Male", "Female"], index=1)
    married = st.radio("Married ?", ["Yes", "No"], index=0)
    age = st.slider("Age", min_value=0, max_value=120, value=43, step=1)
    

with col2:
    
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Government", "Never worked", "Children"], index=0)
    work_type_map = {
        'Private': 'private',
        'Self-employed': 'self-employed',
        'Government': 'govt_job',
        'Children': 'children',
        'Never worked': 'never_worked',
    }
    work_type = work_type_map[work_type]

    residence = st.selectbox("Residence Type", ["Rural", "Urban"], index=1)

    smoking_status = st.selectbox("Smoking Status", ["Unknown", "Never", "Formerly", "Smokes"], index=1)
    
    smoking_map = {
        'Never': 'never smoked',
        'Formerly': 'formerly smoked',
        'Smokes': 'smokes',
        'Unknown': 'unknown'
    }

    smoking_status = smoking_map[smoking_status]

with col3:
    hypertension = st.radio("Hypertension", ["Yes", "No"], index=1)
    heart_disease = st.radio("Heart disease", ["Yes", "No"], index=1)
    bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=28.0, step=0.1)

    glucose_level = st.slider("Average glucose level", min_value=50.0, max_value=300.0, value=89.0, step=0.1)



st.markdown("### ")
predict_button = st.button("Predict", help="Click to predict your stroke chances")

st.markdown("#### Result")
result_placeholder = st.empty()
if predict_button:
    user_info = pd.DataFrame({
        'gender': [gender],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [married],
        'work_type': [work_type],
        'residence_type': [residence],
        'smoking_status': [smoking_status],
        'age': [age],
        'avg_glucose_level': [glucose_level],
        'bmi': [bmi],
        'stroke': ['yes']
    })
    result_placeholder.info(f"Hello {name}, based on the provided information, your stroke risk will be predicted here.")
    pred, advice = process(user_info)
    
    if pred is not None:
        risk_percentage = pred[1] * 100
        if risk_percentage < 20:
            result_color = "lightgreen"
        elif risk_percentage < 50:
            result_color = "orange"
        else:
            result_color = "red"
            
        # Phần kết quả
        result_placeholder.markdown(
            f"""
            <div style='background-color: #333; color: {result_color}; padding: 20px; border-radius: 15px;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2); text-align: center;'>
                <h3 style='margin: 0;'>Prediction Result</h3>
                <h2 style='margin: 10px 0; font-size: 28px;'>Your stroke risk is estimated to be <b>{risk_percentage:.2f}%</b>!</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("#### Advice")
        st.markdown(
            f"""
            <div style='background: linear-gradient(113deg, rgba(2,0,36,1) 0%, rgba(57,119,157,1) 87%, rgba(1,85,130,1) 100%); padding: 20px; border-radius: 15px;
                        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);'>
                <p style='color: white; font-size: 18px; line-height: 1.8; margin: 0; text-align: justify;'>
                    {advice}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        result_placeholder.error("Unable to generate prediction. Please check your input and try again.")




st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: #F5F5F5;
        }
        .stTextInput, .stSelectbox, .stRadio, .stSlider {
            color: #F5F5F5 !important;
        }
        div[data-testid="stMarkdownContainer"] > p {
            font-size: 18px;
            color: #F5F5F5;
        }       
        .stButton>button {      
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45A049;
        }
        .css-18e3th9 {
            background-color: #333333;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
