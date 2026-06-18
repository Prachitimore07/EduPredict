import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("data/student_data.csv")

def risk_level(score):

    if score >= 80:
        return "Low Risk"

    elif score >= 60:
        return "Medium Risk"

    else:
        return "High Risk"

df["Risk_Level"] = df["Exam_Score"].apply(risk_level)

df = df.drop("Exam_Score", axis=1)

for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

X = df.drop("Risk_Level", axis=1)

y = LabelEncoder().fit_transform(df["Risk_Level"])

model = XGBClassifier()

model.fit(X, y)

plt.figure(figsize=(10,6))

plt.barh(
    X.columns,
    model.feature_importances_
)

plt.xlabel("Importance")

plt.title("Feature Importance")

plt.show()