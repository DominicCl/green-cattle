import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import sys
import os

# Add scripts/ to path so we can import our own modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from breed_records import get_formula_vector_for_breed

# ── What this file does ───────────────────────────────────────────
# Trains the Green Cattle multimodal model in two stages:
#
# Stage 1: Freeze ResNet backbone, train only formula stream
#          and fusion layers. Fast, safe, establishes baseline.
#
# Stage 2: Unfreeze everything, train at very low learning rate.
#          Deep layers subtly adjust to cattle-specific features.
#
# This script also:
#   - Tracks train and val loss every epoch
#   - Saves the best model whenever val loss improves
#   - Prints clear progress so you can watch the model learn
# ─────────────────────────────────────────────────────────────────

# ── Configuration ────────────────────────────────────────────────
DATA_DIR    = Path("data/processed")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Training hyperparameters
# A hyperparameter is a setting YOU choose before training starts
# (as opposed to parameters like weights which the model learns)
BATCH_SIZE    = 32
STAGE1_EPOCHS = 15    # epochs with frozen backbone
STAGE2_EPOCHS = 15    # epochs with unfrozen backbone
STAGE1_LR     = 0.001 # learning rate for stage 1
STAGE2_LR     = 0.0001 # 10x smaller for stage 2 — careful fine-tuning

# ImageNet normalization constants (same as data_loader.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Custom Dataset ────────────────────────────────────────────────
# PyTorch's built-in ImageFolder only handles images.
# We need a custom Dataset class that returns BOTH an image
# AND a formula vector for each cow.
#
# What is a Dataset class?
# It's a Python class that tells PyTorch how to load one item
# from your data. PyTorch calls __len__ to know how many items
# there are, and __getitem__ to get item number i.
# The DataLoader then batches these items together automatically.

class CowDataset(Dataset):
    """
    Custom dataset that returns (image, formula_vector, label)
    for each cow photo.

    The formula vector is generated from breed-average farm records
    with random noise to simulate individual cow variation.
    """

    def __init__(self, split, transform):
        """
        Args:
            split: 'train', 'val', or 'test'
            transform: image transforms to apply
        """
        self.transform = transform
        self.samples = []

        # Walk through the folder structure and collect
        # (image_path, formula_vector, label) for every image
        split_dir = DATA_DIR / split

        # class_to_idx: {'high_ch4': 0, 'low_ch4': 1}
        # alphabetical order — same as ImageFolder would assign
        classes = sorted(os.listdir(split_dir))
        classes = [c for c in classes if not c.startswith('.')]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # Map each class folder to its possible breeds
        # so we can generate appropriate formula vectors
        from breed_records import FOLDER_TO_BREEDS

        for class_name in classes:
            class_dir = split_dir / class_name
            label = self.class_to_idx[class_name]
            breeds = FOLDER_TO_BREEDS[class_name]

            img_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in img_extensions:
                    # Randomly assign one of the breeds for this class
                    # For high_ch4: randomly Holstein, Brown Swiss, or Ayrshire
                    # For low_ch4: randomly Jersey or Red Dane
                    # This adds realistic variation across the dataset
                    breed = np.random.choice(breeds)
                    self.samples.append((img_path, breed, label))

        print(f"  {split}: {len(self.samples)} samples loaded")

    def __len__(self):
        """How many items in this dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get one item by index.
        Called automatically by DataLoader.

        Returns:
            image: tensor of shape (3, 224, 224)
            formula_vec: tensor of shape (23,)
            label: integer 0 or 1
        """
        img_path, breed, label = self.samples[idx]

        # Load and transform image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Generate formula vector for this breed
        # noise=True adds individual variation during training
        formula_vec = get_formula_vector_for_breed(
            breed,
            noise=(self.transform is not None)
        )
        formula_vec = torch.tensor(formula_vec, dtype=torch.float32)

        return image, formula_vec, label


# ── Transforms ───────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Training Functions ────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one complete pass through the training data.

    Args:
        model: our GreenCattleModel
        loader: DataLoader serving batches
        criterion: loss function
        optimizer: weight updater
        device: 'cpu' or 'mps' (Mac GPU)

    Returns:
        average loss for this epoch
    """
    # model.train() tells PyTorch we are training
    # This enables Dropout (randomly drops neurons)
    # and BatchNorm in training mode
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, formula_vecs, labels) in enumerate(loader):
        # Move data to the right device (CPU or GPU)
        images      = images.to(device)
        formula_vecs = formula_vecs.to(device)
        labels      = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────
        # Pass both images and formula vectors through the model
        # Get predictions — shape: (batch_size, 2) for 2 classes
        predictions = model(images, formula_vecs)

        # ── Calculate loss ────────────────────────────────────────
        # How wrong were our predictions?
        loss = criterion(predictions, labels)

        # ── Backward pass ─────────────────────────────────────────
        # optimizer.zero_grad() clears gradients from the last batch
        # If we didn't do this, gradients would accumulate
        optimizer.zero_grad()

        # loss.backward() calculates gradients for all parameters
        # using the chain rule (backpropagation)
        loss.backward()

        # optimizer.step() updates all trainable weights
        # by nudging them in the direction that reduces loss
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        _, predicted = predictions.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)} "
                  f"loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on val or test data.
    No weight updates happen here.
    """
    # model.eval() disables Dropout and sets BatchNorm to eval mode
    # This gives deterministic predictions
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad() tells PyTorch not to track gradients
    # This saves memory and speeds up evaluation significantly
    with torch.no_grad():
        for images, formula_vecs, labels in loader:
            images       = images.to(device)
            formula_vecs = formula_vecs.to(device)
            labels       = labels.to(device)

            predictions = model(images, formula_vecs)
            loss = criterion(predictions, labels)

            total_loss += loss.item()
            _, predicted = predictions.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def run_stage(model, train_loader, val_loader, criterion,
              optimizer, scheduler, device, epochs, stage_name,
              best_val_loss, save_path):
    """
    Run a complete training stage (Stage 1 or Stage 2).
    Saves the model whenever val loss improves.
    """
    print(f"\n{'='*50}")
    print(f"  {stage_name}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        # ReduceLROnPlateau scheduler checks if val loss improved
        # If it hasn't improved for 'patience' epochs, it reduces
        # the learning rate by a factor to help escape plateaus
        scheduler.step(val_loss)

        print(f"\n  Train loss: {train_loss:.4f} | "
              f"Train acc: {train_acc:.1f}%")
        print(f"  Val loss:   {val_loss:.4f} | "
              f"Val acc:   {val_acc:.1f}%")

        # Save model if this is the best val loss we've seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved! "
                  f"(val loss: {val_loss:.4f})")

    return best_val_loss


# ── Main Training Script ──────────────────────────────────────────
if __name__ == "__main__":

    # Detect best available device
    # MPS = Metal Performance Shaders = Mac GPU acceleration
    # Much faster than CPU for matrix operations
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac GPU (MPS) ✓")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ── Load datasets ─────────────────────────────────────────────
    print("\nLoading datasets...")
    train_dataset = CowDataset("train", train_transforms)
    val_dataset   = CowDataset("val",   eval_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    # ── Build model ───────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(freeze_backbone=True)
    model = model.to(device)

    # ── Loss function ─────────────────────────────────────────────
    # CrossEntropyLoss is standard for classification problems
    # It measures how far our predicted probabilities are from
    # the true one-hot labels
    #
    # class_weight handles our imbalance:
    # high_ch4 (class 0): 752 images
    # low_ch4  (class 1): 456 images
    # We weight the minority class higher so the model pays
    # more attention to getting low_ch4 right
    class_weights = torch.tensor([1.0, 1.65]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Stage 1 ───────────────────────────────────────────────────
    # Only train formula stream and fusion layers
    # Backbone is frozen — only 1.15M parameters update
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR
    )

    # ReduceLROnPlateau: if val loss doesn't improve for
    # 3 epochs in a row, reduce lr by 50%
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    save_path = MODELS_DIR / "green_cattle_best.pth"
    best_val_loss = float('inf')  # infinity — any real loss is better

    best_val_loss = run_stage(
        model, train_loader, val_loader, criterion,
        optimizer, scheduler, device,
        epochs=STAGE1_EPOCHS,
        stage_name="STAGE 1 — Training fusion layers (backbone frozen)",
        best_val_loss=best_val_loss,
        save_path=save_path
    )

    # ── Stage 2 ───────────────────────────────────────────────────
    # Unfreeze the entire backbone
    # Now all 24.6M parameters can update
    print("\nUnfreezing backbone for Stage 2...")
    for param in model.parameters():
        param.requires_grad = True

    # Much lower learning rate — we don't want to destroy the
    # pretrained visual knowledge, just gently fine-tune it
    optimizer = Adam(model.parameters(), lr=STAGE2_LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_loss = run_stage(
        model, train_loader, val_loader, criterion,
        optimizer, scheduler, device,
        epochs=STAGE2_EPOCHS,
        stage_name="STAGE 2 — Fine-tuning full network",
        best_val_loss=best_val_loss,
        save_path=save_path
    )

    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Best model saved to: {save_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*50}")