import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from train import CowDataset, eval_transforms, DATA_DIR, MODELS_DIR

# ── What this file does ───────────────────────────────────────────
# Loads the trained model and tests it on the TEST set — 183 images
# that were never used in training OR validation. This is our
# most honest measure of performance.
#
# IMPORTANT CONTEXT — read before trusting these numbers:
# This model was trained on breed-based proxy labels and SIMULATED
# farm records, not real individual cow methane measurements.
# High accuracy here means the model successfully learned to
# distinguish breed-typical visual and formula patterns — it does
# NOT mean the model can predict any individual cow's true methane
# output. See the printed disclaimer below for full context.
# ─────────────────────────────────────────────────────────────────

def build_confusion_matrix(model, loader, device):
    """
    Run the model on every test image and build a confusion matrix.

    A confusion matrix is a table showing:
        - how many high_ch4 cows were correctly called high_ch4
        - how many high_ch4 cows were wrongly called low_ch4
        - how many low_ch4 cows were correctly called low_ch4
        - how many low_ch4 cows were wrongly called high_ch4

    This tells us not just "how often is the model right" but
    exactly WHERE it makes mistakes.
    """
    model.eval()

    # confusion[true_label][predicted_label] = count
    confusion = np.zeros((2, 2), dtype=int)

    all_correct = []
    all_confidences = []

    with torch.no_grad():
        for images, formula_vecs, labels in loader:
            images       = images.to(device)
            formula_vecs = formula_vecs.to(device)
            labels       = labels.to(device)

            outputs = model(images, formula_vecs)

            # Convert raw model outputs to probabilities using softmax
            # Softmax squashes numbers into probabilities that sum to 1
            # e.g. [2.1, -0.5] becomes [0.88, 0.12]
            probs = torch.softmax(outputs, dim=1)

            # Get the predicted class — whichever has higher probability
            confidences, predicted = probs.max(1)

            for true_label, pred_label, conf in zip(
                labels.cpu().numpy(),
                predicted.cpu().numpy(),
                confidences.cpu().numpy()
            ):
                confusion[true_label][pred_label] += 1
                all_correct.append(true_label == pred_label)
                all_confidences.append(conf)

    return confusion, all_correct, all_confidences


def print_results(confusion, class_names):
    """Print a clear, readable confusion matrix and metrics"""

    print("\n" + "=" * 55)
    print("  CONFUSION MATRIX")
    print("=" * 55)
    print(f"\n{'':20}{'Predicted ' + class_names[0]:>18}"
          f"{'Predicted ' + class_names[1]:>18}")

    for i, true_class in enumerate(class_names):
        print(f"{'Actually ' + true_class:20}"
              f"{confusion[i][0]:>18}{confusion[i][1]:>18}")

    # Calculate per-class metrics
    total = confusion.sum()
    correct = confusion[0][0] + confusion[1][1]
    accuracy = 100.0 * correct / total

    print(f"\nOverall test accuracy: {accuracy:.1f}%  "
          f"({correct}/{total} correct)")

    # Precision: of all images predicted as class X, how many
    # actually were class X?
    # Recall: of all images that ARE class X, how many did we catch?
    for i, class_name in enumerate(class_names):
        true_positive  = confusion[i][i]
        false_positive = confusion[:, i].sum() - true_positive
        false_negative = confusion[i, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive + 1e-9)
        recall    = true_positive / (true_positive + false_negative + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"\n{class_name}:")
        print(f"  Precision: {precision*100:.1f}%  "
              f"(when model says {class_name}, how often is it right)")
        print(f"  Recall:    {recall*100:.1f}%  "
              f"(of all real {class_name} cows, how many did it catch)")
        print(f"  F1 score:  {f1*100:.1f}%  "
              f"(balance of precision and recall)")


def print_disclaimer():
    """Print the honest context for interpreting these results"""
    print("\n" + "=" * 55)
    print("  IMPORTANT CONTEXT FOR INTERPRETING THESE RESULTS")
    print("=" * 55)
    print("""
This model was trained on:
  - Breed-based proxy labels (not real individual measurements)
  - Simulated farm records sampled around breed averages

High accuracy here means the model successfully learned to
recognise BREED-TYPICAL visual and formula patterns associated
with published research on methane emissions.

It does NOT mean the model can predict any individual cow's
true measured methane output, since no cow in this dataset
was ever connected to a real CH4 sensor.

This is a scientifically grounded PROTOTYPE. The architecture,
formulas, and approach are sound. The next step toward a
production-ready system is retraining on real sensor-measured
data (e.g. GreenFeed system data, Wallace/Difford datasets).
""")


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load test dataset — images NEVER seen during training or val
    print("\nLoading test dataset...")
    test_dataset = CowDataset("test", eval_transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    # Load the best saved model
    print("\nLoading trained model...")
    model = build_model(freeze_backbone=False)
    model.load_state_dict(
        torch.load(MODELS_DIR / "green_cattle_best.pth", map_location=device)
    )
    model = model.to(device)
    print("Model loaded successfully!")

    # Class names in alphabetical order (how ImageFolder assigns them)
    class_names = ["high_ch4", "low_ch4"]

    # Run evaluation
    print("\nRunning evaluation on test set...")
    confusion, all_correct, all_confidences = build_confusion_matrix(
        model, test_loader, device
    )

    print_results(confusion, class_names)

    avg_confidence = np.mean(all_confidences) * 100
    print(f"\nAverage prediction confidence: {avg_confidence:.1f}%")

    print_disclaimer()