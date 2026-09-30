import pandas as pd

DATASET = r"F:\SnapFire-AI\data\smoke_detection_iot.csv"

print("=" * 70)
print("SNAPFIRE AI - CHRONOLOGICAL SESSION ANALYSIS V2")
print("=" * 70)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATASET)

print("\nOriginal shape:", df.shape)

# Convert UTC to datetime
df["UTC_datetime"] = pd.to_datetime(
    df["UTC"],
    unit="s"
)

# IMPORTANT:
# Sort by the actual timestamp BEFORE detecting sessions.
df = df.sort_values(
    "UTC"
).reset_index(drop=True)

# ------------------------------------------------------------
# UTC DIFFERENCES AFTER SORTING
# ------------------------------------------------------------

df["UTC_diff"] = df["UTC"].diff()

print("\nChronological UTC range:")
print("Start:", df["UTC_datetime"].min())
print("End  :", df["UTC_datetime"].max())

print("\nUTC difference statistics:")
print(df["UTC_diff"].describe())

# ------------------------------------------------------------
# FIND REAL RECORDING GAPS
#
# A normal sample is approximately 1 second apart.
# A gap > 1 second indicates a possible new recording session.
# ------------------------------------------------------------

gaps = df[
    df["UTC_diff"] > 1
].copy()

print("\n" + "=" * 70)
print("CHRONOLOGICAL RECORDING GAPS")
print("=" * 70)

print(
    "Number of gaps > 1 second:",
    len(gaps)
)

if len(gaps) > 0:

    gaps_display = gaps[
        [
            "UTC_datetime",
            "UTC_diff",
            "Fire Alarm"
        ]
    ].copy()

    print(
        gaps_display.to_string(
            index=False
        )
    )

# ------------------------------------------------------------
# CREATE CHRONOLOGICAL SESSIONS
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
print("CHRONOLOGICAL RECORDING SESSIONS")
print("=" * 70)

print(
    sessions.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# CREATE FIRE/NO-FIRE EVENTS AFTER SORTING
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
          Session_ID=("Session_ID", "first"),
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
print("CHRONOLOGICAL FIRE / NO-FIRE EVENTS")
print("=" * 70)

print(
    events.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# SESSION × CLASS
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

# ------------------------------------------------------------
# CHECK WHETHER UTC IS NOW STRICTLY INCREASING
# ------------------------------------------------------------

negative_diffs = (
    df["UTC_diff"] < 0
).sum()

zero_diffs = (
    df["UTC_diff"] == 0
).sum()

print("\n" + "=" * 70)
print("TIMESTAMP VALIDATION")
print("=" * 70)

print(
    "Negative UTC differences:",
    negative_diffs
)

print(
    "Zero UTC differences:",
    zero_diffs
)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)