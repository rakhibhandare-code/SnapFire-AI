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

MODEL_DIR = r"F:\SnapFire-AI\models\sensor_v2"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sensor_model_v2.pkl"
)

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
# LOAD DATA
# ============================================================

print("\nLoading sensor dataset...")

df = pd.read_csv(DATASET)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col for col in FEATURES + [TARGET]
    if col not in df.columns
]

if missing_columns:

    print("\nMissing columns:")
    for col in missing_columns:
        print("-", col)

    raise SystemExit(
        "\nDataset column names do not match the model."
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[FEATURES + [TARGET]].copy()

df = df.dropna()

print(
    f"Rows after removing missing values: {len(df)}"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nClass distribution:")

print(
    df[TARGET].value_counts()
)


# ============================================================
# CREATE EVENT-BASED SPLIT
# ============================================================
#
# We deliberately avoid a normal random split.
#
# The dataset contains long continuous fire/no-fire
# sequences. We reserve the last portion of the dataset
# for evaluation.
#
# IMPORTANT:
# This is a chronological holdout, not an independent
# real-world fire experiment.
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()

test_df = df.iloc[split_index:].copy()


X_train = train_df[FEATURES]

y_train = train_df[TARGET]

X_test = test_df[FEATURES]

y_test = test_df[TARGET]


print("\nTraining samples:", len(train_df))

print("Testing samples :", len(test_df))

print("\nTraining class distribution:")

print(y_train.value_counts())

print("\nTesting class distribution:")

print(y_test.value_counts())


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Sensor Model V2...")

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

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION
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

print("SENSOR MODEL V2 RESULTS")

print("=" * 60)

print(
    f"\nAccuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy  : {balanced_accuracy * 100:.2f}%"
)


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


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


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

print("\nSensor Model V2 training complete.")