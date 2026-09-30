import pandas as pd
import numpy as np

# ============================================================
# PATH
# ============================================================

DATASET = r"F:\SnapFire-AI\data\smoke_detection_iot.csv"


# ============================================================
# FEATURES
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

print("=" * 75)
print("SNAPFIRE AI - SENSOR SESSION COMPARISON")
print("=" * 75)

df = pd.read_csv(DATASET)

print("\nOriginal dataset:", df.shape)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values("UTC").reset_index(drop=True)

df["UTC_diff"] = df["UTC"].diff()

df["Session_ID"] = (
    df["UTC_diff"] > 1
).cumsum() + 1


# ============================================================
# SESSION SUMMARY
# ============================================================

sessions = (
    df.groupby("Session_ID")
      .agg(
          Samples=(TARGET, "size"),
          Fire_Samples=(TARGET, "sum")
      )
      .reset_index()
)

sessions["No_Fire_Samples"] = (
    sessions["Samples"] -
    sessions["Fire_Samples"]
)

print("\n" + "=" * 75)
print("SESSION SUMMARY")
print("=" * 75)

print(
    sessions.to_string(index=False)
)


# ============================================================
# MEAN VALUES BY SESSION
# ============================================================

print("\n" + "=" * 75)
print("MEAN SENSOR VALUES BY SESSION")
print("=" * 75)

session_means = (
    df.groupby("Session_ID")[FEATURES]
      .mean()
)

print(
    session_means.to_string()
)


# ============================================================
# FIRE-ONLY MEANS
# ============================================================

print("\n" + "=" * 75)
print("FIRE-ONLY SENSOR MEANS BY SESSION")
print("=" * 75)

fire_only = df[
    df[TARGET] == 1
]

fire_means = (
    fire_only
    .groupby("Session_ID")[FEATURES]
    .mean()
)

print(
    fire_means.to_string()
)


# ============================================================
# NO-FIRE-ONLY MEANS
# ============================================================

print("\n" + "=" * 75)
print("NO-FIRE SENSOR MEANS BY SESSION")
print("=" * 75)

no_fire_only = df[
    df[TARGET] == 0
]

no_fire_means = (
    no_fire_only
    .groupby("Session_ID")[FEATURES]
    .mean()
)

print(
    no_fire_means.to_string()
)


# ============================================================
# COMPARE FIRE SESSION 4 AGAINST FIRE SESSIONS 2 AND 3
# ============================================================

print("\n" + "=" * 75)
print("FIRE SESSION COMPARISON")
print("=" * 75)

fire_sessions = [2, 3, 4]

comparison = fire_means.loc[
    fire_means.index.intersection(
        fire_sessions
    )
].T

print(
    comparison.to_string()
)


# ============================================================
# PERCENTAGE DIFFERENCE
#
# Compare Session 4 against average of Sessions 2 and 3.
# ============================================================

reference_sessions = fire_means.loc[
    fire_means.index.intersection([2, 3])
]

reference_mean = reference_sessions.mean()

session4 = fire_means.loc[4]

comparison_table = pd.DataFrame({
    "Sessions_2_3_Average": reference_mean,
    "Session_4": session4
})

comparison_table["Absolute_Difference"] = (
    comparison_table["Session_4"]
    -
    comparison_table["Sessions_2_3_Average"]
)

comparison_table["Percentage_Difference"] = (
    comparison_table["Absolute_Difference"]
    /
    comparison_table["Sessions_2_3_Average"].replace(
        0,
        np.nan
    )
    * 100
)

print("\nSession 4 vs average of Sessions 2 & 3:")

print(
    comparison_table
    .sort_values(
        "Percentage_Difference",
        key=lambda x: x.abs(),
        ascending=False
    )
    .to_string()
)


# ============================================================
# MEDIAN COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("MEDIAN VALUES - FIRE SESSIONS")
print("=" * 75)

fire_medians = (
    fire_only
    .groupby("Session_ID")[FEATURES]
    .median()
)

print(
    fire_medians.loc[
        fire_medians.index.intersection(
            fire_sessions
        )
    ].to_string()
)


# ============================================================
# STANDARD DEVIATION
# ============================================================

print("\n" + "=" * 75)
print("STANDARD DEVIATION - FIRE SESSIONS")
print("=" * 75)

fire_std = (
    fire_only
    .groupby("Session_ID")[FEATURES]
    .std()
)

print(
    fire_std.loc[
        fire_std.index.intersection(
            fire_sessions
        )
    ].to_string()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("SESSION COMPARISON COMPLETE")
print("=" * 75)