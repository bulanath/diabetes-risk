import streamlit as st
import pandas as pd
import joblib

# Set up the Streamlit page configuration
st.set_page_config(page_title="Diabetes Risk Quiz", layout="wide", initial_sidebar_state="collapsed")

# Function to cache the model loading
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None
    
# Load the pre-trained model
model = load_model('diabetes_model.joblib')

st.title("Diabetes Risk Assessment Quiz")
st.markdown(
    "Answer the questions below to receive a risk assessment based on a machine learning model from the patterns found in the [BRFSS 2015 dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv). "
    "This quiz is for **educational purposes** only."
)

# Mappings
age_map = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80 or older"
}
education_map = {
    1: "Never attended school or kindergarten", 2: "Grades 1-8 (Elementary)",
    3: "Grades 9-11 (Some high school)", 4: "Grade 12 or GED (High school graduate)",
    5: "Some college or technical school", 6: "College graduate"
}
income_map = {
    1: "< $10,000 (~ < Rp 160 Juta)", 2: "< $15,000 (~ < Rp 240 Juta)",
    3: "< $20,000 (~ < Rp 320 Juta)", 4: "< $25,000 (~ < Rp 400 Juta)",
    5: "< $35,000 (~ < Rp 560 Juta)", 6: "< $50,000 (~ < Rp 800 Juta)",
    7: "< $75,000 (~ < Rp 1.2 Miliar)", 8: ">= $75,000 (~ >= Rp 1.2 Miliar)"
}

# Input form
with st.form(key='quiz_form'):
    st.header("👤 Personal and Health Information")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", ("Female", "Male"), key="sex")
        age_val_label = st.selectbox("Age category", list(age_map.values()), index=6, key="age")
        age_val = [k for k, v in age_map.items() if v == age_val_label][0]
        bmi = st.number_input("Body Mass Index (weight (kg) / [height (m)]²)", min_value=12.0, max_value=98.0, value=28.0, step=0.1, key="bmi")
        education_val_label = st.selectbox("Education level", list(education_map.values()), index=3, key="education")
        education_val = [k for k, v in education_map.items() if v == education_val_label][0]
        income_val_label = st.selectbox("Income level (per year)", list(income_map.values()), index=4, key="income")
        income_val = [k for k, v in income_map.items() if v == income_val_label][0]

    with col2:  
        any_healthcare = st.selectbox("Do you have health care coverage?", ("No", "Yes"), key="any_healthcare")
        no_doc_bc_cost = st.selectbox("In the past year, have you been unable to see a doctor due to cost?", ("No", "Yes"), key="no_doc")
        gen_hlth = st.slider("General health (1=excellent, 2=very good, 3=good, 4=fair, 5=poor)", 1, 5, key="gen_hlth")
        phys_hlth = st.slider("How many days of poor physical health in the last 30 days?", 0, 30, 0, step=1, key="phys_hlth")
        ment_hlth = st.slider("How many days of poor mental health in the last 30 days?", 0, 30, 0, step=1, key="ment_hlth")

    st.markdown("---")
    st.header("🩺 Medical History")
    col3, col4 = st.columns(2)
    with col3:
        high_bp = st.selectbox("Do you have high blood pressure?", ("No", "Yes"), key="high_bp")
        high_chol = st.selectbox("Do you have high cholesterol?", ("No", "Yes"), key="high_chol")
        chol_check = st.selectbox("Have you had a cholesterol check in the last 5 years?", ("No", "Yes"), key="chol_check")
    with col4:
        stroke = st.selectbox("Have you ever had a stroke?", ("No", "Yes"), key="stroke")
        heart_disease_attack = st.selectbox("Have you had a heart disease or a heart Attack?", ("No", "Yes"), key="heart_disease")
        diff_walk = st.selectbox("Do you have any difficulty walking or climbing up the stairs?", ("No", "Yes"), key="diff_walk")

    st.markdown("---")
    st.header("🚶‍➡️ Lifestyle Habits")
    col5, col6 = st.columns(2)
    with col5:
        phys_activity = st.selectbox("Do you have any physical activity in past 30 days?", ("No", "Yes"), key="phys_activity")
        fruits = st.selectbox("Do you consume fruit at least more than 1 daily?", ("No", "Yes"), key="fruits")
        veggies = st.selectbox("Do you consume vegetable at least more than 1 daily?", ("No", "Yes"), key="veggies")
    with col6:
        smoker = st.selectbox("Have you smoked at least 100 cigarettes?", ("No", "Yes"), key="smoker")
        hvy_alcohol_consump = st.selectbox("Are you considered a heavy drinker? (Men > 14 drinks/week, Women > 7 drinks/week)", ("No", "Yes"), key="hvy_alcohol")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label='Check My Risk', use_container_width=True)

# Logic
if submit_button:
    input_data = {
        'HighBP': 1 if high_bp == "Yes" else 0, 'HighChol': 1 if high_chol == "Yes" else 0,
        'CholCheck': 1 if chol_check == "Yes" else 0, 'BMI': bmi,
        'Smoker': 1 if smoker == "Yes" else 0, 'Stroke': 1 if stroke == "Yes" else 0,
        'HeartDiseaseorAttack': 1 if heart_disease_attack == "Yes" else 0,
        'PhysActivity': 1 if phys_activity == "Yes" else 0, 'Fruits': 1 if fruits == "Yes" else 0,
        'Veggies': 1 if veggies == "Yes" else 0, 'HvyAlcoholConsump': 1 if hvy_alcohol_consump == "Yes" else 0,
        'AnyHealthcare': 1 if any_healthcare == "Yes" else 0, 'NoDocbcCost': 1 if no_doc_bc_cost == "Yes" else 0,
        'GenHlth': gen_hlth, 'MentHlth': ment_hlth, 'PhysHlth': phys_hlth,
        'DiffWalk': 1 if diff_walk == "Yes" else 0, 'Sex': 1 if sex == "Male" else 0,
        'Age': age_val, 'Education': education_val, 'Income': income_val
    }

    # DataFrame for model input
    feature_order = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    input_df = pd.DataFrame([input_data])[feature_order]

    # Binary prediction
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.header("Your Result")

    if prediction == 1:
        st.error("**⚠️ High Risk Detected.** Based on your answers, the model indicates you have a **higher risk** of having diabetes or prediabetes.")
    else:
        st.success("**✅ Lower/No Risk Detected.** Based on your answers, the model indicates you have a **lower or no significant risk** of having diabetes at this time.")