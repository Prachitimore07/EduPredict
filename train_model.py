import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv("data/student_data.csv")

# Create Risk Levels
def risk_level(score):

    if score >= 80:
        return "Low Risk"

    elif score >= 60:
        return "Medium Risk"

    else:
        return "High Risk"

df["Risk_Level"] = df["Exam_Score"].apply(risk_level)

# Drop Target Column
df = df.drop("Exam_Score", axis=1)

# Convert Text Columns
label_encoders = {}

for column in df.columns:

    if df[column].dtype == "object":

        le = LabelEncoder()

        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le

# Features
X = df.drop("Risk_Level", axis=1)

# Target
y = df["Risk_Level"]

# Encode Target
target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train XGBoost
model = XGBClassifier()

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Save Model
joblib.dump(model, "model.pkl")

print("Model Saved Successfully!")