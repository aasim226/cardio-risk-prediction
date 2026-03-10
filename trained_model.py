import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("cardiovascular_risk_dataset.csv")

df = df.drop(["Patient_ID", "heart_disease_risk_score"], axis=1)

target_column = "risk_category"

le_target = LabelEncoder()
df[target_column] = le_target.fit_transform(df[target_column])

categorical_columns = ["smoking_status", "family_history_heart_disease"]

categorical_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    categorical_encoders[col] = le

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump({
    "model": model,
    "scaler": scaler,
    "feature_order": X.columns.tolist(),
    "target_encoder": le_target,
    "categorical_encoders": categorical_encoders
}, "model.pkl")

print("\nModel trained and saved successfully.")