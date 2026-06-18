import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

print("\n===== Student Risk Prediction =====\n")

hours_studied = float(input("Hours Studied: "))
attendance = float(input("Attendance (%): "))
parental_involvement = int(input("Parental Involvement (Low=1, Medium=2, High=3): "))
access_to_resources = int(input("Access to Resources (Low=1, Medium=2, High=3): "))
extracurricular = int(input("Extracurricular Activities (No=0, Yes=1): "))
sleep_hours = float(input("Sleep Hours: "))
previous_scores = float(input("Previous Scores: "))
motivation_level = int(input("Motivation Level (Low=1, Medium=2, High=3): "))
internet_access = int(input("Internet Access (No=0, Yes=1): "))
tutoring_sessions = float(input("Tutoring Sessions: "))
family_income = int(input("Family Income (Low=1, Medium=2, High=3): "))
teacher_quality = int(input("Teacher Quality (Low=1, Medium=2, High=3): "))
school_type = int(input("School Type (Public=0, Private=1): "))
peer_influence = int(input("Peer Influence (Negative=0, Neutral=1, Positive=2): "))
physical_activity = float(input("Physical Activity Hours: "))
learning_disabilities = int(input("Learning Disabilities (No=0, Yes=1): "))
parental_education = int(input("Parental Education (High School=0, College=1, Postgraduate=2): "))
distance_from_home = int(input("Distance From Home (Near=0, Moderate=1, Far=2): "))
gender = int(input("Gender (Male=0, Female=1): "))

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

print("\nPredicted Risk Level:", risk_map[prediction[0]])