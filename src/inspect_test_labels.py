import os

LABEL_DIR = r"F:\Fire and Smoke Dataset\test\labels"

print("=" * 70)
print("INSPECTING ORIGINAL TEST LABEL FORMAT")
print("=" * 70)

files = [
    f for f in os.listdir(LABEL_DIR)
    if f.endswith(".txt")
]

print("\nTotal label files:", len(files))

shown = 0

for filename in files:

    path = os.path.join(LABEL_DIR, filename)

    with open(path, "r", errors="ignore") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if not lines:
        print("\nEMPTY:", filename)
        shown += 1

    else:
        print("\nFILE:", filename)

        for line in lines[:5]:
            parts = line.split()

            print(
                "  Values:", len(parts),
                "|", line[:250]
            )

        shown += 1

    if shown >= 15:
        break

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)