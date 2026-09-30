import random
import shutil
from pathlib import Path

# ============================================================
# SNAPFIRE AI - CREATE SMALL VISION DATASET
# ============================================================

SOURCE = Path(r"F:\Fire and Smoke Dataset")
DEST = Path(r"F:\SnapFire-AI\vision_dataset")

TRAIN_COUNT = 2000
VALID_COUNT = 500

random.seed(42)


def is_detection_label(label_file):
    """
    Accept only YOLO object-detection labels.
    Each valid detection line must contain 5 values:
    class_id x_center y_center width height
    """

    if not label_file.exists():
        return False

    try:
        lines = [
            line.strip()
            for line in label_file.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        for line in lines:
            values = line.split()

            # Detection format = exactly 5 values
            if len(values) != 5:
                return False

            # Make sure all values are numeric
            [float(v) for v in values]

        return True

    except Exception:
        return False


def collect_pairs(split):
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    pairs = []

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for image in image_dir.iterdir():

        if image.suffix.lower() not in extensions:
            continue

        label = label_dir / f"{image.stem}.txt"

        if is_detection_label(label):
            pairs.append((image, label))

    return pairs


def copy_pairs(pairs, split, count):

    random.shuffle(pairs)

    selected = pairs[:min(count, len(pairs))]

    image_dest = DEST / split / "images"
    label_dest = DEST / split / "labels"

    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    for image, label in selected:

        shutil.copy2(
            image,
            image_dest / image.name
        )

        shutil.copy2(
            label,
            label_dest / label.name
        )

    return len(selected)


print("=" * 60)
print("       SNAPFIRE AI - VISION DATASET PREPARATION")
print("=" * 60)

print("\nOriginal dataset:")
print(SOURCE)

print("\nNew dataset:")
print(DEST)

# ------------------------------------------------------------
# Collect valid detection pairs
# ------------------------------------------------------------

print("\nScanning training dataset...")

train_pairs = collect_pairs("train")

print(f"Valid detection training pairs found: {len(train_pairs)}")

print("\nScanning validation dataset...")

valid_pairs = collect_pairs("valid")

print(f"Valid detection validation pairs found: {len(valid_pairs)}")

# ------------------------------------------------------------
# Create subset
# ------------------------------------------------------------

print("\nCreating training subset...")

train_selected = copy_pairs(
    train_pairs,
    "train",
    TRAIN_COUNT
)

print(f"Training images copied: {train_selected}")

print("\nCreating validation subset...")

valid_selected = copy_pairs(
    valid_pairs,
    "valid",
    VALID_COUNT
)

print(f"Validation images copied: {valid_selected}")

# ------------------------------------------------------------
# Create data.yaml
# ------------------------------------------------------------

yaml_content = f"""path: {DEST.as_posix()}
train: train/images
val: valid/images

nc: 2
names: ['fire', 'smoke']
"""

yaml_file = DEST / "data.yaml"

yaml_file.write_text(yaml_content)

print("\nCreated:")
print(yaml_file)

print("\n" + "=" * 60)
print("       DATASET PREPARATION COMPLETE")
print("=" * 60)

print("\nDataset structure:")

print(f"""
{DEST}
├── train
│   ├── images
│   └── labels
├── valid
│   ├── images
│   └── labels
└── data.yaml
""")

print(f"Training images : {train_selected}")
print(f"Validation images : {valid_selected}")

print("\nOriginal dataset was NOT modified.")