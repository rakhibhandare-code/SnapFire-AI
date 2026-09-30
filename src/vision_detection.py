from ultralytics import YOLO
import cv2

# ============================================================
# SNAPFIRE AI - REAL-TIME VISION ENGINE
# ============================================================

MODEL_PATH = r"F:\SnapFire-AI\models\fire_smoke_yolo_small\weights\best.pt"

print("=" * 60)
print("        SNAPFIRE AI - VISION ENGINE")
print("=" * 60)

print("\nLoading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Starting camera...")

# Camera 1 worked on your PC earlier
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open camera 1.")
    print("Try changing VideoCapture(1) to VideoCapture(2).")
    exit()

print("Camera started.")
print("\nPress Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Run YOLO detection
    results = model(
        frame,
        conf=0.10,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Get detected objects
    detected = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detected.append(
                (class_name, confidence)
            )

    # Display status
    if detected:

        highest = max(
            detected,
            key=lambda x: x[1]
        )

        class_name, confidence = highest

        print(
            f"\rDetection: {class_name.upper():<8} "
            f"Confidence: {confidence * 100:5.1f}%",
            end=""
        )

    else:

        print(
            "\rDetection: NONE       Confidence: 0.0%",
            end=""
        )

    cv2.imshow(
        "SnapFire AI - Vision Engine",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\n\nVision engine stopped.")