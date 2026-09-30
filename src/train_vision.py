from ultralytics import YOLO

# ============================================================
# SNAPFIRE AI - VISION MODEL TRAINING
# ============================================================

DATASET = r"F:\SnapFire-AI\vision_dataset\data.yaml"

print("=" * 60)
print("       SNAPFIRE AI VISION TRAINING")
print("=" * 60)

print("\nDataset:", DATASET)
print("Classes: Fire, Smoke")
print("Training images: 2000")
print("Validation images: 500")
print("Device: CPU")
print("Epochs: 5")

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data=DATASET,
    epochs=5,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=2,
    project=r"F:\SnapFire-AI\models",
    name="fire_smoke_yolo_small",
    exist_ok=True
)

print("\n" + "=" * 60)
print("       VISION TRAINING COMPLETE")
print("=" * 60)

print("\nModel location:")
print(
    r"F:\SnapFire-AI\models\fire_smoke_yolo_small\weights\best.pt"
)