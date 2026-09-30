import os
import shutil

SOURCE = r"F:\Fire and Smoke Dataset\test"
DEST = r"F:\SnapFire-AI\vision_dataset\test"

SOURCE_IMAGES = os.path.join(SOURCE, "images")
SOURCE_LABELS = os.path.join(SOURCE, "labels")

DEST_IMAGES = os.path.join(DEST, "images")
DEST_LABELS = os.path.join(DEST, "labels")

os.makedirs(DEST_IMAGES, exist_ok=True)
os.makedirs(DEST_LABELS, exist_ok=True)

print("Creating clean test subset...")

valid_pairs = 0
skipped = 0

for filename in os.listdir(SOURCE_LABELS):

    if not filename.endswith(".txt"):
        continue

    label_path = os.path.join(
        SOURCE_LABELS,
        filename
    )

    with open(label_path, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Keep only standard detection labels:
    # class x_center y_center width height
    valid_lines = []

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            values = [float(x) for x in parts]

            class_id = int(values[0])

            if class_id not in [0, 1]:
                continue

            valid_lines.append(line)

        except ValueError:
            continue

    if not valid_lines:
        skipped += 1
        continue

    image_name = os.path.splitext(filename)[0]

    image_path = None

    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:

        candidate = os.path.join(
            SOURCE_IMAGES,
            image_name + ext
        )

        if os.path.exists(candidate):
            image_path = candidate
            break

    if image_path is None:
        skipped += 1
        continue

    # Copy image
    shutil.copy2(
        image_path,
        os.path.join(
            DEST_IMAGES,
            os.path.basename(image_path)
        )
    )

    # Write cleaned label
    with open(
        os.path.join(
            DEST_LABELS,
            filename
        ),
        "w"
    ) as f:

        f.write(
            "\n".join(valid_lines)
        )

    valid_pairs += 1

print()
print("=" * 60)
print("CLEAN TEST SUBSET COMPLETE")
print("=" * 60)

print("Valid image/label pairs:", valid_pairs)
print("Skipped:", skipped)

print()
print("Test images:")
print(DEST_IMAGES)

print()
print("Test labels:")
print(DEST_LABELS)