import pandas as pd
import numpy as np

DATASET = r"F:\SnapFire-AI\data\smoke_detection_iot.csv"

df = pd.read_csv(DATASET)

print("=" * 70)
print("SNAPFIRE AI - RECORDING SESSION ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# UTC analysis
# ------------------------------------------------------------

df["UTC_datetime"] = pd.to_datetime(
    df["UTC"],
    unit="s"
)

df["UTC_diff"] = df["UTC"].diff()

print("\nUTC range:")
print("Start:", df["UTC_datetime"].min())
print("End  :", df["UTC_datetime"].max())

print("\nUTC difference statistics:")
print(df["UTC_diff"].describe())

# ------------------------------------------------------------
# Find gaps in recording
# ------------------------------------------------------------

gaps = df[df["UTC_diff"] > 1].copy()

print("\n" + "=" * 70)
print("RECORDING GAPS")
print("=" * 70)

print("Number of gaps > 1 second:", len(gaps))

if len(gaps) > 0:

    print("\nLargest gaps:")

    largest_gaps = (
        gaps[
            ["UTC_datetime", "UTC_diff", "Fire Alarm"]
        ]
        .sort_values(
            "UTC_diff",
            ascending=False
        )
        .head(30)
    )

    print(
        largest_gaps.to_string(index=False)
    )

# ------------------------------------------------------------
# Create recording sessions
# ------------------------------------------------------------

df["Session_ID"] = (
    df["UTC_diff"] > 1
).cumsum() + 1

sessions = (
    df.groupby("Session_ID")
      .agg(
          Samples=("Fire Alarm", "size"),
          Start=("UTC_datetime", "min"),
          End=("UTC_datetime", "max"),
          Fire_Samples=("Fire Alarm", "sum")
      )
      .reset_index()
)

sessions["No_Fire_Samples"] = (
    sessions["Samples"] -
    sessions["Fire_Samples"]
)

sessions["Fire_Percentage"] = (
    sessions["Fire_Samples"] /
    sessions["Samples"] * 100
)

print("\n" + "=" * 70)
print("RECORDING SESSIONS")
print("=" * 70)

print(
    sessions.to_string(index=False)
)

# ------------------------------------------------------------
# Event transitions
# ------------------------------------------------------------

df["Event_ID"] = (
    df["Fire Alarm"]
    .ne(df["Fire Alarm"].shift())
    .cumsum()
)

events = (
    df.groupby("Event_ID")
      .agg(
          Samples=("Fire Alarm", "size"),
          Label=("Fire Alarm", "first"),
          Session=("Session_ID", "first"),
          Start=("UTC_datetime", "min"),
          End=("UTC_datetime", "max")
      )
      .reset_index()
)

events["Class"] = events["Label"].map({
    0: "No Fire",
    1: "Fire"
})

print("\n" + "=" * 70)
print("EVENTS WITH SESSION INFORMATION")
print("=" * 70)

print(
    events.to_string(index=False)
)

# ------------------------------------------------------------
# Session/class relationship
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SESSION × FIRE ALARM")
print("=" * 70)

cross = pd.crosstab(
    df["Session_ID"],
    df["Fire Alarm"]
)

cross.columns = [
    "No Fire" if c == 0 else "Fire"
    for c in cross.columns
]

print(cross)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)