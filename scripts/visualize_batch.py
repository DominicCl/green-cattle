import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load images WITHOUT normalization so they look normal to human eyes
# (normalized images look washed out and weird when displayed)
viz_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
])

DATA_DIR = Path("data/processed")
dataset  = datasets.ImageFolder(DATA_DIR / "train", transform=viz_transforms)
loader   = DataLoader(dataset, batch_size=16, shuffle=True)

# Grab one batch of 16 images
images, labels = next(iter(loader))
classes = dataset.classes

# Plot them in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Sample training images (after augmentation)", fontsize=14)

for i, ax in enumerate(axes.flat):
    # Convert from PyTorch tensor format to displayable format
    img = images[i].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(classes[labels[i]], fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("data/sample_batch.png", dpi=150)
print("Saved to data/sample_batch.png — open it in Finder to view!")