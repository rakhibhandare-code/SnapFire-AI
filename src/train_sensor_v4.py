import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

DATASET = r"F:\SnapFire-AI\data\smoke_detection_iot.csv"

MODEL_DIR = r"F:\SnapFire-AI\models\sensor_v4"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sensor_model_v4.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# PHYSICAL SENSOR FEATURES
# ============================================================

FEATURES = [
    "Temperature[C]",
    "Humidity[%]",
    "TVOC[ppb]",
    "eCO2[ppm]",
    "Raw H2",
    "Raw Ethanol",
    "Pressure[hPa]",
    "PM1.0",
    "PM2.5",
    "NC0.5",
    "NC1.0",
    "NC2.5"
]

TARGET = "Fire Alarm"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SNAPFIRE AI - SENSOR MODEL V4")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print("Original shape:", df.shape)


# ============================================================
# SORT BY REAL UTC TIME
# ============================================================

df = df.sort_values(
    "UTC"
).reset_index(drop=True)

# ============================================================
# CREATE REAL RECORDING SESSIONS
#
# Normal sampling interval = approximately 1 second.
# A gap > 1 second means a new recording session.
# ============================================================

df["UTC_diff"] = df["UTC"].diff()

df["Session_ID"] = (
    df["UTC_diff"] > 1
).cumsum() + 1


# ============================================================
# DISPLAY SESSION INFORMATION
# ============================================================

sessions = (
    df.groupby("Session_ID")
      .agg(
          Samples=("Fire Alarm", "size"),
          Fire_Samples=("Fire Alarm", "sum")
      )
      .reset_index()
)

sessions["No_Fire_Samples"] = (
    sessions["Samples"] -
    sessions["Fire_Samples"]
)

print("\nRecording sessions:")
print(
    sessions.to_string(index=False)
)


# ============================================================
# TRAIN / TEST SESSION SPLIT
#
# Train:
#   Sessions 1, 2, 3, 5
#
# Test:
#   Session 4
#
# Session 4 contains 1,121 Fire samples and 33 No Fire
# samples, giving us a meaningful independent fire event.
# ============================================================

TRAIN_SESSIONS = [1, 2, 3, 5]
TEST_SESSIONS = [4]

train_df = df[
    df["Session_ID"].isin(TRAIN_SESSIONS)
].copy()

test_df = df[
    df["Session_ID"].isin(TEST_SESSIONS)
].copy()


print("\n" + "=" * 70)
print("SESSION-BASED SPLIT")
print("=" * 70)

print(
    "Training sessions:",
    TRAIN_SESSIONS
)

print(
    "Testing sessions :",
    TEST_SESSIONS
)

print(
    "\nTraining samples:",
    len(train_df)
)

print(
    "Testing samples :",
    len(test_df)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")
print(
    train_df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTesting class distribution:")
print(
    test_df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("SENSOR MODEL V4 RESULTS")
print("=" * 70)

print(
    f"\nAccuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Fire",
            "Fire"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nSensor Model V4 training complete.")

print("=" * 70)