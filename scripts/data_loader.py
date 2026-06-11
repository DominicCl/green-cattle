from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── 1. Define transforms ────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training transforms — augmentation applied here only
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Val/test transforms — no augmentation, just resize and normalize
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── 2. Load datasets from folders ───────────────────────────────────
DATA_DIR = Path("data/processed")

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transforms)
val_dataset   = datasets.ImageFolder(DATA_DIR / "val",   transform=eval_transforms)
test_dataset  = datasets.ImageFolder(DATA_DIR / "test",  transform=eval_transforms)

# ── 3. Create DataLoaders ────────────────────────────────────────────
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

# ── 4. Print a summary ───────────────────────────────────────────────
print("Dataset summary")
print("───────────────────────────────")
print(f"Classes : {train_dataset.classes}")
print(f"  train : {len(train_dataset)} images")
print(f"  val   : {len(val_dataset)} images")
print(f"  test  : {len(test_dataset)} images")
print(f"\nBatch size : 32")
print(f"Train batches per epoch : {len(train_loader)}")
print(f"\nData loaders are ready!")