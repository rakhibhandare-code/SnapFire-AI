import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "firedata.csv"
MODEL_PATH = BASE_DIR / "models" / "sensor_model.pkl"


# Load dataset
data = pd.read_csv(DATA_PATH)

print("\n===== SnapFire AI: Sensor Model Training =====")
print(f"Dataset: {DATA_PATH}")
print(f"Samples: {len(data)}")

# Features and target
features = ["Temperature", "Smoke", "Flame", "Humidity"]

X = data[features]
y = data["Label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Save model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to:")
print(MODEL_PATH)

print("\n===== Training Complete =====")