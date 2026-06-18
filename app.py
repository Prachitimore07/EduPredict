import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🎓 EduPredict")
st.subheader("Early Warning System for Student Performance")

# Inputs
hours_studied = st.number_input("Hours Studied", 0.0, 24.0, 5.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0)

parental_involvement = st.selectbox(
    "Parental Involvement",
    [0, 1, 2]
)

access_to_resources = st.selectbox(
    "Access to Resources",
    [0, 1, 2]
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    [0, 1]
)

sleep_hours = st.number_input(
    "Sleep Hours",
    0.0,
    12.0,
    7.0
)

previous_scores = st.number_input(
    "Previous Scores",
    0.0,
    100.0,
    70.0
)

motivation_level = st.selectbox(
    "Motivation Level",
    [0, 1, 2]
)

internet_access = st.selectbox(
    "Internet Access",
    [0, 1]
)

tutoring_sessions = st.number_input(
    "Tutoring Sessions",
    0,
    20,
    2
)

family_income = st.selectbox(
    "Family Income",
    [0, 1, 2]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    [0, 1, 2]
)

school_type = st.selectbox(
    "School Type",
    [0, 1]
)

peer_influence = st.selectbox(
    "Peer Influence",
    [0, 1, 2]
)

physical_activity = st.number_input(
    "Physical Activity",
    0.0,
    10.0,
    1.0
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    [0, 1]
)

parental_education = st.selectbox(
    "Parental Education Level",
    [0, 1, 2]
)

distance_from_home = st.selectbox(
    "Distance From Home",
    [0, 1, 2]
)

gender = st.selectbox(
    "Gender",
    [0, 1]
)

# Prediction
if st.button("Predict Risk"):

    student_data = pd.DataFrame({
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Parental_Involvement": [parental_involvement],
        "Access_to_Resources": [access_to_resources],
        "Extracurricular_Activities": [extracurricular],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_scores],
        "Motivation_Level": [motivation_level],
        "Internet_Access": [internet_access],
        "Tutoring_Sessions": [tutoring_sessions],
        "Family_Income": [family_income],
        "Teacher_Quality": [teacher_quality],
        "School_Type": [school_type],
        "Peer_Influence": [peer_influence],
        "Physical_Activity": [physical_activity],
        "Learning_Disabilities": [learning_disabilities],
        "Parental_Education_Level": [parental_education],
        "Distance_from_Home": [distance_from_home],
        "Gender": [gender]
    })

    prediction = model.predict(student_data)

    risk_map = {
        0: "High Risk",
        1: "Low Risk",
        2: "Medium Risk"
    }

    st.success(
        f"Predicted Risk Level: {risk_map[prediction[0]]}"
    )