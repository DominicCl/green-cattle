import os
import shutil
from pathlib import Path
import random

random.seed(42)

HIGH_CH4 = ["Holstein Friesian cattle", "Brown Swiss cattle", "Ayrshire cattle"]
LOW_CH4  = ["Jersey cattle", "Red Dane cattle"]

RAW_DIR  = Path("data/raw/cattle-breeds/Cattle Breeds")
OUT_DIR  = Path("data/processed")

SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

def get_images(breed_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return [f for f in breed_dir.iterdir() if f.suffix.lower() in exts]

all_low, all_high = [], []

for breed in HIGH_CH4:
    imgs = get_images(RAW_DIR / breed)
    all_high.extend(imgs)
    print(f"  high_ch4  {breed}: {len(imgs)} images")

for breed in LOW_CH4:
    imgs = get_images(RAW_DIR / breed)
    all_low.extend(imgs)
    print(f"  low_ch4   {breed}: {len(imgs)} images")

def split_and_copy(images, label):
    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    buckets = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split, files in buckets.items():
        dest = OUT_DIR / split / label
        for f in files:
            shutil.copy(f, dest / f.name)
        print(f"  {split}/{label}: {len(files)} images copied")

print("\nCopying high_ch4 images...")
split_and_copy(all_high, "high_ch4")

print("\nCopying low_ch4 images...")
split_and_copy(all_low, "low_ch4")

print("\nDone! Final counts:")
for split in ["train", "val", "test"]:
    for label in ["low_ch4", "high_ch4"]:
        n = len(list((OUT_DIR / split / label).iterdir()))
        print(f"  {split}/{label}: {n}")
