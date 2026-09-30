import os

IMAGE_DIR = r"F:\Fire and Smoke Dataset\test\images"
LABEL_DIR = r"F:\Fire and Smoke Dataset\test\labels"

print("=" * 70)
print("TEST IMAGE / LABEL MATCHING DIAGNOSTIC")
print("=" * 70)

# ------------------------------------------------------------
# LIST FILES
# ------------------------------------------------------------

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
]

label_files = [
    f for f in os.listdir(LABEL_DIR)
    if f.lower().endswith(".txt")
]

print("\nImages found :", len(image_files))
print("Labels found :", len(label_files))

# ------------------------------------------------------------
# CREATE STEM SETS
# ------------------------------------------------------------

image_stems = {
    os.path.splitext(f)[0]
    for f in image_files
}

label_stems = {
    os.path.splitext(f)[0]
    for f in label_files
}

matched = image_stems.intersection(label_stems)

labels_without_images = label_stems - image_stems
images_without_labels = image_stems - label_stems

print("\nMatched image/label pairs:", len(matched))

print(
    "Labels without matching images:",
    len(labels_without_images)
)

print(
    "Images without matching labels:",
    len(images_without_labels)
)

# ------------------------------------------------------------
# SHOW EXAMPLES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXAMPLE IMAGE FILES")
print("=" * 70)

for f in image_files[:15]:
    print(f)

print("\n" + "=" * 70)
print("EXAMPLE LABEL FILES")
print("=" * 70)

for f in label_files[:15]:
    print(f)

# ------------------------------------------------------------
# SHOW UNMATCHED LABELS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNMATCHED LABEL EXAMPLES")
print("=" * 70)

for stem in list(labels_without_images)[:15]:
    print(stem)

# ------------------------------------------------------------
# SHOW UNMATCHED IMAGES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNMATCHED IMAGE EXAMPLES")
print("=" * 70)

for stem in list(images_without_labels)[:15]:
    print(stem)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)