from pathlib import Path
from PIL import Image

print("Checking images...\n")

errors = []
sizes = []

for split in ["train", "val", "test"]:
    for label in ["low_ch4", "high_ch4"]:
        folder = Path("data/processed") / split / label
        images = list(folder.iterdir())
        for img_path in images:
            try:
                img = Image.open(img_path)
                sizes.append(img.size)
                img.close()
            except Exception as e:
                errors.append((img_path, e))

print(f"Total images checked: {len(sizes)}")
print(f"Corrupt images found: {len(errors)}")

widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
print(f"\nWidth  range: {min(widths)}px to {max(widths)}px")
print(f"Height range: {min(heights)}px to {max(heights)}px")
print(f"\nAll images look good!" if not errors else f"\nProblematic files: {errors}")
