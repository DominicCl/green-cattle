import torch
import torch.nn as nn
from torchvision import models

# ── What this file does ───────────────────────────────────────────
# Defines the Green Cattle multimodal model architecture.
#
# It combines two completely different types of input:
#   1. A cow photo → processed by ResNet50 → 2048 visual features
#   2. Farm records → processed by 23 formulas → 23 CH4 estimates
#
# These two streams are concatenated and fed into a small fusion
# network that learns to combine them into one final prediction:
#   → CH4 production in grams per day
#
# This is a REGRESSION task — we're predicting a continuous number
# (grams of methane), not a binary low/high category like before.
# This is more powerful because it gives the reproduction tree
# actual scores to rank and optimize, not just two buckets.
# ─────────────────────────────────────────────────────────────────

class GreenCattleModel(nn.Module):
    """
    Multimodal model combining visual features from a cow photo
    with formula-based CH4 estimates from farm records.

    What is nn.Module?
    PyTorch requires all neural network models to inherit from
    nn.Module. This gives us automatic parameter tracking,
    GPU support, saving/loading, and the forward() method
    that defines how data flows through the network.
    """

    def __init__(self, num_formulas=23, freeze_backbone=True):
        """
        __init__ is called when you create the model object.
        It defines all the layers — but doesn't run any data yet.
        Think of it as assembling the machine before turning it on.

        Args:
            num_formulas: how many formula predictions we feed in (23)
            freeze_backbone: whether to freeze ResNet's early layers
        """
        super(GreenCattleModel, self).__init__()

        # ── VISUAL STREAM ─────────────────────────────────────────
        # Load ResNet50 with pretrained ImageNet weights
        # This gives us 25 million weights already trained on
        # 1.2 million photos — we get all that knowledge for free
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        # Freeze all layers if requested — they won't update
        # during training, preserving the pretrained knowledge
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # Remove ResNet's final classification layer
        # We want the 2048 features BEFORE the final decision
        # nn.Sequential(*list(...)[:-1]) means: take all layers
        # except the last one and chain them together
        self.visual_backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )
        # visual_backbone output shape: (batch_size, 2048, 1, 1)
        # The 1,1 at the end is a spatial dimension we'll flatten

        # ── FORMULA STREAM ────────────────────────────────────────
        # The 23 formula predictions are already meaningful numbers
        # (CH4 estimates in g/day). We pass them through a small
        # neural network to learn non-linear combinations of them.
        # This is better than just averaging them directly.
        #
        # What is nn.Linear?
        # A fully connected layer — every input connects to every
        # output. nn.Linear(23, 64) takes 23 numbers and produces
        # 64 numbers using learned weights.
        #
        # What is nn.ReLU?
        # The activation function we discussed — adds non-linearity
        # so the network can learn complex patterns.
        #
        # What is nn.BatchNorm1d?
        # Normalizes the numbers flowing through the layer to have
        # mean~0 and std~1. This stabilizes training significantly —
        # without it, the formula values (300-500 g/day) would be
        # on a very different scale from the visual features,
        # making it hard for the fusion layer to combine them.
        self.formula_stream = nn.Sequential(
            nn.BatchNorm1d(num_formulas),   # normalize formula inputs
            nn.Linear(num_formulas, 64),    # 23 → 64 features
            nn.ReLU(),
            nn.Linear(64, 64),              # 64 → 64 features
            nn.ReLU(),
        )
        # formula_stream output shape: (batch_size, 64)

        # ── FUSION LAYER ─────────────────────────────────────────
        # Concatenate visual (2048) + formula (64) = 2112 features
        # Then compress down to our final prediction
        #
        # What is nn.Dropout?
        # During training, randomly sets some neurons to zero.
        # This prevents the model from relying too heavily on any
        # single feature — it's forced to learn redundant paths.
        # Acts as a regularizer to reduce overfitting.
        # p=0.3 means 30% of neurons are randomly dropped each step.
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 64, 512),      # 2112 → 512
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),            # 512 → 128
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),              # 128 → 2 classes (low_ch4, high_ch4)
        )
        # Final output: one number — predicted CH4 in g/day

    def forward(self, image, formula_vector):
        """
        forward() defines how data actually flows through the model.
        PyTorch calls this automatically when you pass data in.

        Args:
            image: batch of cow photos
                   shape: (batch_size, 3, 224, 224)
            formula_vector: batch of formula predictions
                           shape: (batch_size, 23)

        Returns:
            CH4 prediction in g/day
            shape: (batch_size, 1)
        """

        # ── Visual stream ─────────────────────────────────────────
        # Pass image through ResNet50's frozen layers
        visual_features = self.visual_backbone(image)
        # Shape: (batch_size, 2048, 1, 1)

        # Flatten the spatial dimensions
        # view(batch_size, -1) reshapes to (batch_size, 2048)
        # -1 means "figure out this dimension automatically"
        visual_features = visual_features.view(
            visual_features.size(0), -1
        )
        # Shape: (batch_size, 2048)

        # ── Formula stream ────────────────────────────────────────
        # Pass formula predictions through the formula network
        formula_features = self.formula_stream(formula_vector)
        # Shape: (batch_size, 64)

        # ── Fusion ───────────────────────────────────────────────
        # Concatenate both feature vectors along dimension 1
        # torch.cat joins tensors end-to-end on the specified axis
        combined = torch.cat([visual_features, formula_features], dim=1)
        # Shape: (batch_size, 2048 + 64) = (batch_size, 2112)

        # Pass through fusion layers to get final prediction
        output = self.fusion(combined)
        # Shape: (batch_size, 1)

        return output


def build_model(freeze_backbone=True):
    """Convenience function to create the model"""
    return GreenCattleModel(
        num_formulas=23,
        freeze_backbone=freeze_backbone
    )


def count_parameters(model):
    """Print a summary of trainable vs frozen parameters"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    frozen    = total - trainable

    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters   : {frozen:,}")
    print(f"Frozen percentage   : {100*frozen/total:.1f}%")


if __name__ == "__main__":
    print("Building Green Cattle multimodal model...\n")

    model = build_model(freeze_backbone=True)
    count_parameters(model)

    # Test with dummy data to verify shapes work correctly
    # This is standard practice — always test your model
    # with fake data before touching real data
    print("\nRunning shape test with dummy data...")

    batch_size = 4

    # Fake batch of 4 cow images (3 channels, 224x224)
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    # Fake batch of 4 formula vectors (23 predictions each)
    dummy_formulas = torch.randn(batch_size, 23)

    # Forward pass
    with torch.no_grad():  # no_grad = don't track gradients
        output = model(dummy_images, dummy_formulas)

    print(f"Input image shape   : {dummy_images.shape}")
    print(f"Input formula shape : {dummy_formulas.shape}")
    print(f"Output shape        : {output.shape}")
    print(f"Sample predictions  : {output.squeeze().tolist()}")
    print("\nModel is ready!")