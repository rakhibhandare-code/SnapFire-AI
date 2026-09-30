import cv2
from ultralytics import YOLO
from collections import deque

# ============================================================
# SNAPFIRE AI - VISION DETECTION V2
# YOLO + TEMPORAL CONFIRMATION
# ============================================================

MODEL_PATH = r"F:\SnapFire-AI\models\fire_smoke_yolo_v2\weights\best.pt"

CAMERA_INDEX = 1

# Minimum confidence for considering a detection
CONF_THRESHOLD = 0.25

# Temporal confirmation
WINDOW_SIZE = 10
MIN_POSITIVE_FRAMES = 6

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("SNAPFIRE AI - VISION DETECTION V2")
print("=" * 65)

print("\nLoading YOLO V2...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("\nERROR: Could not open camera.")
    raise SystemExit

print("Camera opened.")

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
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    fire_detected = False
    smoke_detected = False

    best_fire_conf = 0.0
    best_smoke_conf = 0.0

    # --------------------------------------------------------
    # READ DETECTIONS
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            # Class 0 = fire
            if cls_id == 0:

                fire_detected = True

                best_fire_conf = max(
                    best_fire_conf,
                    confidence
                )

            # Class 1 = smoke
            elif cls_id == 1:

                smoke_detected = True

                best_smoke_conf = max(
                    best_smoke_conf,
                    confidence
                )

    # --------------------------------------------------------
    # UPDATE TEMPORAL HISTORY
    # --------------------------------------------------------

    fire_history.append(
        fire_detected
    )

    smoke_history.append(
        smoke_detected
    )

    # --------------------------------------------------------
    # TEMPORAL CONFIRMATION
    # --------------------------------------------------------

    fire_count = sum(
        fire_history
    )

    smoke_count = sum(
        smoke_history
    )

    fire_confirmed = (
        len(fire_history) >= WINDOW_SIZE
        and fire_count >= MIN_POSITIVE_FRAMES
    )

    smoke_confirmed = (
        len(smoke_history) >= WINDOW_SIZE
        and smoke_count >= MIN_POSITIVE_FRAMES
    )

    # --------------------------------------------------------
    # FINAL VISION STATUS
    # --------------------------------------------------------

    if fire_confirmed:

        vision_status = "FIRE"

    elif smoke_confirmed:

        vision_status = "SMOKE"

    else:

        vision_status = "NORMAL"

    # --------------------------------------------------------
    # DRAW YOLO BOXES
    # --------------------------------------------------------

    annotated_frame = results[0].plot()

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    cv2.putText(
        annotated_frame,
        f"Vision: {vision_status}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Fire conf: {best_fire_conf:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Smoke conf: {best_smoke_conf:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Fire history: {fire_count}/{len(fire_history)}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Smoke history: {smoke_count}/{len(smoke_history)}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "SnapFire AI - Vision V2",
        annotated_frame
    )

    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nVision detection stopped.")