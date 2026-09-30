import pandas as pd
import numpy as np

DATASET = r"F:\SnapFire-AI\data\smoke_detection_iot.csv"

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

print("=" * 70)
print("SNAPFIRE AI - SENSOR DATASET ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATASET)

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df[FEATURES + [TARGET]].isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# ------------------------------------------------------------
# EVENT ANALYSIS
# ------------------------------------------------------------

df["Event_ID"] = df[TARGET].ne(df[TARGET].shift()).cumsum()

events = (
    df.groupby("Event_ID")
      .agg(
          Samples=(TARGET, "size"),
          Label=(TARGET, "first")
      )
      .reset_index()
)

events["Class"] = events["Label"].map({
    0: "No Fire",
    1: "Fire"
})

print("\n" + "=" * 70)
print("EVENT STRUCTURE")
print("=" * 70)

print(events.to_string(index=False))

# ------------------------------------------------------------
# BASIC CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(df[TARGET].value_counts())

# ------------------------------------------------------------
# SENSOR STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SENSOR STATISTICS")
print("=" * 70)

stats = df.groupby(TARGET)[FEATURES].agg(
    ["mean", "median", "std", "min", "max"]
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print(stats)

# ------------------------------------------------------------
# FIRE VS NO-FIRE MEAN COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRE vs NO-FIRE MEAN VALUES")
print("=" * 70)

means = df.groupby(TARGET)[FEATURES].mean().T

means.columns = ["No Fire", "Fire"]

means["Difference"] = (
    means["Fire"] - means["No Fire"]
)

means["Absolute Difference"] = (
    means["Difference"].abs()
)

print(
    means.sort_values(
        "Absolute Difference",
        ascending=False
    ).to_string()
)

# ------------------------------------------------------------
# EVENT-LEVEL SENSOR MEANS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EVENT-LEVEL SENSOR MEANS")
print("=" * 70)

event_means = (
    df.groupby("Event_ID")[FEATURES]
      .mean()
)

event_labels = (
    df.groupby("Event_ID")[TARGET]
      .first()
)

event_summary = event_means.copy()
event_summary["Fire Alarm"] = event_labels
event_summary["Class"] = event_labels.map({
    0: "No Fire",
    1: "Fire"
})

print(event_summary.to_string())

# ------------------------------------------------------------
# CHECK UTC
# ------------------------------------------------------------

if "UTC" in df.columns:

    print("\n" + "=" * 70)
    print("UTC INFORMATION")
    print("=" * 70)

    print("UTC data type:")
    print(df["UTC"].dtype)

    print("\nFirst 10 UTC values:")
    print(df["UTC"].head(10).to_string(index=False))

    print("\nLast 10 UTC values:")
    print(df["UTC"].tail(10).to_string(index=False))

    print("\nUnique UTC values:")
    print(df["UTC"].nunique())

# ------------------------------------------------------------
# CHECK CNT
# ------------------------------------------------------------

if "CNT" in df.columns:

    print("\n" + "=" * 70)
    print("CNT INFORMATION")
    print("=" * 70)

    print("CNT unique values:", df["CNT"].nunique())

    print("\nCNT range:")
    print(
        "Min:", df["CNT"].min(),
        "Max:", df["CNT"].max()
    )

# ------------------------------------------------------------
# CORRELATION WITH TARGET
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION WITH FIRE ALARM")
print("=" * 70)

correlation = (
    df[FEATURES + [TARGET]]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(
        key=lambda x: x.abs(),
        ascending=False
    )
)

print(correlation)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)