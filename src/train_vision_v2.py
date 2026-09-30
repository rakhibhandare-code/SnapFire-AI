from ultralytics import YOLO
import os

# ============================================================
# PATHS
# ============================================================

DATASET = r"F:\SnapFire-AI\vision_dataset\data.yaml"

MODEL_DIR = r"F:\SnapFire-AI\models"

# New model - does NOT overwrite the old one
RUN_NAME = "fire_smoke_yolo_v2"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=" * 70)
print("SNAPFIRE AI - VISION MODEL V2")
print("=" * 70)

print("\nLoading YOLO11n...")

model = YOLO("yolo11n.pt")


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")
print("Dataset:", DATASET)
print("Epochs : 20")
print("Image  : 640")
print("Batch  : 8")
print("Device : CPU")

results = model.train(
    data=DATASET,

    epochs=20,

    imgsz=640,

    batch=8,

    workers=2,

    device="cpu",

    project=MODEL_DIR,

    name=RUN_NAME,

    pretrained=True,

    patience=7,

    cache=False,

    plots=True,

    verbose=True
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING VISION MODEL V2")
print("=" * 70)

metrics = model.val(
    data=DATASET,
    imgsz=640,
    batch=8,
    device="cpu"
)

print("\nValidation complete.")

print("\nModel directory:")
print(
    os.path.join(
        MODEL_DIR,
        RUN_NAME
    )
)

print("\nBest model should be at:")
print(
    os.path.join(
        MODEL_DIR,
        RUN_NAME,
        "weights",
        "best.pt"
    )
)

print("\n" + "=" * 70)
print("VISION MODEL V2 TRAINING COMPLETE")
print("=" * 70)