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

MODEL_DIR = r"F:\SnapFire-AI\models\sensor_v3"
MODEL_PATH = os.path.join(MODEL_DIR, "sensor_model_v3.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# PHYSICAL SENSOR FEATURES ONLY
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
# LOAD DATASET
# ============================================================

print("=" * 65)
print("SNAPFIRE AI - SENSOR MODEL V3")
print("=" * 65)

print("\nLoading sensor dataset...")

df = pd.read_csv(DATASET)

print("Original dataset shape:", df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing columns:")
    for col in missing_columns:
        print(" -", col)

    raise SystemExit


# ============================================================
# KEEP ONLY PHYSICAL FEATURES + TARGET
# ============================================================

df = df[required_columns].copy()

df = df.dropna()

print("Rows after removing missing values:", len(df))


# ============================================================
# CREATE EVENT GROUPS
#
# A new event is created whenever Fire Alarm changes:
#
# 0 0 0 0 1 1 1 0 0
#       ^       ^
#
# Each continuous block stays together.
# ============================================================

df["Event_ID"] = (
    df[TARGET].ne(df[TARGET].shift())
    .cumsum()
)

print("\nNumber of detected events:", df["Event_ID"].nunique())


# ============================================================
# SHOW EVENT INFORMATION
# ============================================================

event_info = (
    df.groupby("Event_ID")[TARGET]
    .agg(["count", "first"])
    .reset_index()
)

event_info["Class"] = event_info["first"].map({
    0: "No Fire",
    1: "Fire"
})

print("\nDetected event blocks:")
print(event_info.to_string(index=False))


# ============================================================
# EVENT-AWARE TEST SET
#
# We intentionally hold out:
#
#   - one substantial FIRE event
#   - one substantial NO-FIRE event
#
# The selected events are chosen from the largest events
# of each class.
# ============================================================

event_sizes = (
    df.groupby(["Event_ID", TARGET])
    .size()
    .reset_index(name="Samples")
)

fire_events = event_sizes[
    event_sizes[TARGET] == 1
].sort_values("Samples", ascending=False)

normal_events = event_sizes[
    event_sizes[TARGET] == 0
].sort_values("Samples", ascending=False)


if fire_events.empty or normal_events.empty:
    raise RuntimeError(
        "Could not find both Fire and No Fire event groups."
    )


# Select a substantial event from each class.
#
# We don't use the very first/last rows as a test set.
# Entire events are held out.

test_fire_event = int(
    fire_events.iloc[0]["Event_ID"]
)

test_normal_event = int(
    normal_events.iloc[0]["Event_ID"]
)

test_events = [
    test_fire_event,
    test_normal_event
]


print("\n============================================================")
print("EVENT-AWARE SPLIT")
print("============================================================")

print("Held-out Fire event ID    :", test_fire_event)
print("Held-out No Fire event ID :", test_normal_event)


# ============================================================
# TRAIN / TEST DATA
# ============================================================

test_df = df[
    df["Event_ID"].isin(test_events)
].copy()

train_df = df[
    ~df["Event_ID"].isin(test_events)
].copy()


print("\nTraining samples:", len(train_df))
print("Testing samples :", len(test_df))


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
# PREPARE X / Y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Sensor Model V3...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


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


print("\n" + "=" * 60)
print("SENSOR MODEL V3 RESULTS")
print("=" * 60)

print(f"\nAccuracy           : {accuracy * 100:.2f}%")
print(f"Balanced Accuracy  : {balanced_accuracy * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Fire", "Fire"],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
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
    importance.to_string(index=False)
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

print("\nSensor Model V3 training complete.")
print("=" * 65)