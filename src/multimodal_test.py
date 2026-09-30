import cv2
from ultralytics import YOLO
from collections import deque

from sensor_evidence import evaluate_sensors
from fusion_engine_v2 import fuse_evidence


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"F:\SnapFire-AI\models\fire_smoke_yolo_v2\weights\best.pt"

CAMERA_INDEX = 1

CONF_THRESHOLD = 0.25

WINDOW_SIZE = 10
MIN_POSITIVE_FRAMES = 6


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("SNAPFIRE AI - MULTIMODAL INTEGRATION TEST")
print("=" * 70)

print("\nLoading YOLO V2...")

model = YOLO(MODEL_PATH)

print("YOLO loaded.")


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    raise SystemExit


# ============================================================
# TEMPORAL HISTORY
# ============================================================

fire_history = deque(
    maxlen=WINDOW_SIZE
)

smoke_history = deque(
    maxlen=WINDOW_SIZE
)


# ============================================================
# DEMO SENSOR VALUES
#
# These are temporary simulated sensor values.
# Later we will replace them with real RS485/Modbus data.
# ============================================================

sensor_values = {
    "temperature": 25,
    "humidity": 50,
    "tvoc": 100,
    "eco2": 450,
    "pm1": 2,
    "pm25": 3,
    "nc05": 10,
    "nc10": 2,
    "nc25": 1
}


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # ========================================================
    # VISION
    # ========================================================

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    fire_detected = False
    smoke_detected = False

    fire_conf = 0.0
    smoke_conf = 0.0


    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            if class_id == 0:

                fire_detected = True

                fire_conf = max(
                    fire_conf,
                    confidence
                )

            elif class_id == 1:

                smoke_detected = True

                smoke_conf = max(
                    smoke_conf,
                    confidence
                )


    # ========================================================
    # TEMPORAL HISTORY
    # ========================================================

    fire_history.append(
        fire_detected
    )

    smoke_history.append(
        smoke_detected
    )


    fire_confirmed = (
        len(fire_history) >= WINDOW_SIZE
        and sum(fire_history) >= MIN_POSITIVE_FRAMES
    )

    smoke_confirmed = (
        len(smoke_history) >= WINDOW_SIZE
        and sum(smoke_history) >= MIN_POSITIVE_FRAMES
    )


    # ========================================================
    # VISION STATUS
    # ========================================================

    if fire_confirmed:

        vision_status = "FIRE"

    elif smoke_confirmed:

        vision_status = "SMOKE"

    else:

        vision_status = "NORMAL"


    # ========================================================
    # SENSOR EVIDENCE
    # ========================================================

    sensor_result = evaluate_sensors(
        **sensor_values
    )


    # ========================================================
    # FUSION
    # ========================================================

    fusion_result = fuse_evidence(

        vision_status=vision_status,

        fire_confidence=fire_conf,

        smoke_confidence=smoke_conf,

        sensor_status=sensor_result["status"],

        sensor_confidence=sensor_result["confidence"],

        vision_confirmed=(
            fire_confirmed
            or smoke_confirmed
        )
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display = results[0].plot()


    cv2.putText(
        display,
        f"VISION: {vision_status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"SENSOR: {sensor_result['status']}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"FINAL: {fusion_result['status']}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Fire: {fire_conf:.2f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Sensor score: {sensor_result['confidence']:.2f}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Reason: {fusion_result['reason']}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )


    cv2.imshow(
        "SnapFire AI - Multimodal Test",
        display
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nMultimodal test stopped.")