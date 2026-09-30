import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sensor_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_fire_risk(temperature, smoke, flame, humidity):
    """
    Predict fire risk from sensor readings.
    """

    sensor_data = pd.DataFrame(
    [[temperature, smoke, flame, humidity]],
    columns=["Temperature", "Smoke", "Flame", "Humidity"]
)

    prediction = model.predict(sensor_data)[0]

    # Probability of each class
    probabilities = model.predict_proba(sensor_data)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    confidence = class_probabilities[prediction] * 100

    return prediction, confidence, class_probabilities


# --------------------------------------------------
# Test cases
# --------------------------------------------------

test_cases = [
    {
        "name": "Normal Environment",
        "temperature": 25,
        "smoke": 15,
        "flame": 0,
        "humidity": 50
    },
    {
        "name": "Possible Warning",
        "temperature": 33,
        "smoke": 45,
        "flame": 0,
        "humidity": 35
    },
    {
        "name": "Fire Condition",
        "temperature": 40,
        "smoke": 80,
        "flame": 1,
        "humidity": 22
    },
    {
        "name": "Smoke Without Flame",
        "temperature": 28,
        "smoke": 90,
        "flame": 0,
        "humidity": 42
    }
]


print("\n======================================")
print("       SNAPFIRE AI SENSOR ENGINE")
print("======================================")

for case in test_cases:

    prediction, confidence, probabilities = predict_fire_risk(
        case["temperature"],
        case["smoke"],
        case["flame"],
        case["humidity"]
    )

    print(f"\nScenario: {case['name']}")

    print(
        f"Temperature : {case['temperature']} °C"
    )

    print(
        f"Smoke       : {case['smoke']}"
    )

    print(
        f"Flame       : {case['flame']}"
    )

    print(
        f"Humidity    : {case['humidity']} %"
    )

    print(f"Prediction  : {prediction}")
    print(f"Confidence  : {confidence:.2f}%")

    print("Class probabilities:")

    for label, probability in probabilities.items():
        print(
            f"  {label}: {probability * 100:.2f}%"
        )


print("\n======================================")