import cv2
import joblib
import pandas as pd
from ultralytics import YOLO

from fusion_engine import fuse_predictions


# ============================================================
# SNAPFIRE AI - SENSOR + VISION INTEGRATED SYSTEM
# ============================================================

SENSOR_MODEL_PATH = (
    r"F:\SnapFire-AI\models\sensor_model.pkl"
)

VISION_MODEL_PATH = (
    r"F:\SnapFire-AI\models\fire_smoke_yolo_small"
    r"\weights\best.pt"
)

# Camera 1 was confirmed working on your laptop
CAMERA_INDEX = 1

# YOLO detection threshold
VISION_CONFIDENCE_THRESHOLD = 0.10


# ============================================================
# SENSOR TEST SCENARIOS
# ============================================================

SCENARIOS = {

    "1": {
        "name": "Normal Environment",
        "temperature": 25,
        "smoke": 15,
        "flame": 0,
        "humidity": 50
    },

    "2": {
        "name": "Possible Warning",
        "temperature": 33,
        "smoke": 45,
        "flame": 0,
        "humidity": 35
    },

    "3": {
        "name": "Fire Condition",
        "temperature": 40,
        "smoke": 80,
        "flame": 1,
        "humidity": 22
    },

    "4": {
        "name": "Smoke Without Flame",
        "temperature": 28,
        "smoke": 90,
        "flame": 0,
        "humidity": 42
    }
}


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("          SNAPFIRE AI - INTEGRATED AI SYSTEM")
print("=" * 70)

print("\n[1/3] Loading sensor ML model...")

try:

    sensor_model = joblib.load(
        SENSOR_MODEL_PATH
    )

    print("Sensor model loaded successfully.")

except Exception as error:

    print("\nERROR: Could not load sensor model.")
    print(error)
    exit()


print("\n[2/3] Loading YOLO vision model...")

try:

    vision_model = YOLO(
        VISION_MODEL_PATH
    )

    print("Vision model loaded successfully.")

except Exception as error:

    print("\nERROR: Could not load YOLO model.")
    print(error)
    exit()


# ============================================================
# SENSOR PREDICTION FUNCTION
# ============================================================

def get_sensor_prediction(scenario):

    sensor_data = pd.DataFrame(
        [[
            scenario["temperature"],
            scenario["smoke"],
            scenario["flame"],
            scenario["humidity"]
        ]],
        columns=[
            "Temperature",
            "Smoke",
            "Flame",
            "Humidity"
        ]
    )

    prediction = sensor_model.predict(
        sensor_data
    )[0]

    probabilities = sensor_model.predict_proba(
        sensor_data
    )[0]

    classes = sensor_model.classes_

    probability_dict = dict(
        zip(
            classes,
            probabilities
        )
    )

    confidence = float(
        max(probabilities)
    )

    return (
        prediction,
        confidence,
        probability_dict
    )


# ============================================================
# GET VISION PREDICTION
# ============================================================

def get_vision_prediction(frame):

    results = vision_model(
        frame,
        conf=VISION_CONFIDENCE_THRESHOLD,
        verbose=False
    )

    annotated_frame = results[0].plot()

    vision_prediction = "None"
    vision_confidence = 0.0

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = vision_model.names[
                class_id
            ]

            detections.append(
                (
                    class_name,
                    confidence
                )
            )

            # Keep strongest detection
            if confidence > vision_confidence:

                vision_prediction = (
                    class_name.capitalize()
                )

                vision_confidence = confidence

    return (
        annotated_frame,
        vision_prediction,
        vision_confidence,
        detections
    )


# ============================================================
# GET DISPLAY STATUS
# ============================================================

def get_status_text(final_decision):

    if final_decision == "FIRE":

        return "ALARM: FIRE CONDITION"

    elif final_decision == "WARNING":

        return "WARNING: VERIFY CONDITION"

    elif final_decision == "FALSE ALARM":

        return "FALSE ALARM: VERIFY"

    else:

        return "SYSTEM STATUS: NORMAL"


# ============================================================
# START CAMERA
# ============================================================

print("\n[3/3] Starting camera...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("\nERROR: Could not open camera.")
    print("Camera index currently set to:", CAMERA_INDEX)
    print("Try CAMERA_INDEX = 2 if required.")
    exit()

print("Camera started successfully.")


# ============================================================
# DEFAULT SENSOR SCENARIO
# ============================================================

current_scenario = SCENARIOS["1"]


# ============================================================
# CONTROLS
# ============================================================

print("\n")
print("=" * 70)
print("                         CONTROLS")
print("=" * 70)

print("1  → Normal Environment")
print("2  → Possible Warning")
print("3  → Fire Condition")
print("4  → Smoke Without Flame")
print("Q  → Quit")

print("\nIntegrated AI system started.")
print("")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print(
            "\nERROR: Could not read camera frame."
        )

        break


    # --------------------------------------------------------
    # SENSOR AI
    # --------------------------------------------------------

    (
        sensor_prediction,
        sensor_confidence,
        sensor_probabilities
    ) = get_sensor_prediction(
        current_scenario
    )


    # --------------------------------------------------------
    # VISION AI
    # --------------------------------------------------------

    (
        annotated_frame,
        vision_prediction,
        vision_confidence,
        detections
    ) = get_vision_prediction(
        frame
    )


    # --------------------------------------------------------
    # FUSION ENGINE
    # --------------------------------------------------------

    fusion_result = fuse_predictions(

        sensor_prediction,
        sensor_confidence,

        vision_prediction,
        vision_confidence
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    # Sensor prediction
    cv2.putText(
        annotated_frame,

        f"Sensor: {sensor_prediction} "
        f"({sensor_confidence * 100:.1f}%)",

        (20, 30),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    # Vision prediction
    cv2.putText(
        annotated_frame,

        f"Vision: {vision_prediction} "
        f"({vision_confidence * 100:.1f}%)",

        (20, 60),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    # Final decision
    cv2.putText(
        annotated_frame,

        f"FINAL: {fusion_result.final_decision}",

        (20, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.90,

        (255, 255, 255),

        3
    )


    # Current sensor scenario
    cv2.putText(
        annotated_frame,

        f"Scenario: {current_scenario['name']}",

        (20, 135),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )


    # Status message
    status_text = get_status_text(
        fusion_result.final_decision
    )

    cv2.putText(
        annotated_frame,

        status_text,

        (20, 170),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.60,

        (255, 255, 255),

        2
    )


    # Sensor values
    sensor_info = (
        f"T:{current_scenario['temperature']}C  "
        f"S:{current_scenario['smoke']}  "
        f"F:{current_scenario['flame']}  "
        f"H:{current_scenario['humidity']}%"
    )

    cv2.putText(
        annotated_frame,

        sensor_info,

        (20, 205),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.50,

        (255, 255, 255),

        1
    )


    # --------------------------------------------------------
    # DISPLAY WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "SnapFire AI - Integrated System",
        annotated_frame
    )


    # --------------------------------------------------------
    # KEYBOARD INPUT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # Normal
    if key == ord("1"):

        current_scenario = SCENARIOS["1"]

        print(
            "\nSensor scenario → Normal Environment"
        )


    # Warning
    elif key == ord("2"):

        current_scenario = SCENARIOS["2"]

        print(
            "\nSensor scenario → Possible Warning"
        )


    # Fire
    elif key == ord("3"):

        current_scenario = SCENARIOS["3"]

        print(
            "\nSensor scenario → Fire Condition"
        )


    # Smoke without flame
    elif key == ord("4"):

        current_scenario = SCENARIOS["4"]

        print(
            "\nSensor scenario → Smoke Without Flame"
        )


    # Quit
    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print("\n")
print("=" * 70)
print("             SNAPFIRE AI - SYSTEM STOPPED")
print("=" * 70)