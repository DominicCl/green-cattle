import torch
import numpy as np
from pathlib import Path
import sys
import os
import itertools

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_model
from train import eval_transforms, DATA_DIR, MODELS_DIR
from breed_records import sample_records, get_formula_vector_for_breed, FOLDER_TO_BREEDS

# ── What this file does ───────────────────────────────────────────
# Takes a herd of cows (with photos and breeds), scores each one
# using the trained model, then recommends breeding pairs that
# minimize predicted offspring methane.
#
# KEY ASSUMPTION:
# Offspring CH4 risk ≈ average of both parents' scores
#                       + small random genetic variation
#
# This is a simplification of quantitative genetics, but it's
# grounded in research showing heritability of methane-linked
# traits like stature (genetic correlation r=0.43).
# ─────────────────────────────────────────────────────────────────

class Cow:
    """Represents one cow in the herd with her predicted CH4 score"""

    def __init__(self, cow_id, breed, image_path, ch4_score, confidence):
        self.cow_id = cow_id
        self.breed = breed
        self.image_path = image_path
        self.ch4_score = ch4_score      # 0 = high_ch4, 1 = low_ch4 (probability of low)
        self.confidence = confidence

    def __repr__(self):
        return f"Cow({self.cow_id}, {self.breed}, low_ch4_prob={self.ch4_score:.2f})"


def score_herd(model, device, num_cows=20):
    """
    Run the model on a sample of test images to build a 'herd'
    of scored cows. In a real deployment, this would run on
    a farmer's actual cow photos.
    """
    model.eval()

    cows = []
    cow_id = 0

    test_dir = DATA_DIR / "test"
    classes = sorted([c for c in os.listdir(test_dir) if not c.startswith('.')])

    from PIL import Image

    for class_name in classes:
        class_dir = test_dir / class_name
        breeds = FOLDER_TO_BREEDS[class_name]
        img_paths = list(class_dir.iterdir())[:num_cows // len(classes)]

        for img_path in img_paths:
            breed = np.random.choice(breeds)

            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image_tensor = eval_transforms(image).unsqueeze(0).to(device)

            # Generate formula vector for this breed (no noise — use averages)
            formula_vec = get_formula_vector_for_breed(breed, noise=False)
            formula_tensor = torch.tensor(
                formula_vec, dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Get model prediction
            with torch.no_grad():
                output = model(image_tensor, formula_tensor)
                probs = torch.softmax(output, dim=1)

            # probs[0][0] = probability of high_ch4
            # probs[0][1] = probability of low_ch4
            low_ch4_prob = probs[0][1].item()
            confidence = probs.max().item()

            cow = Cow(
                cow_id=f"cow_{cow_id:03d}",
                breed=breed,
                image_path=img_path,
                ch4_score=low_ch4_prob,  # higher = better (more likely low emitter)
                confidence=confidence
            )
            cows.append(cow)
            cow_id += 1

    return cows


def predict_offspring_score(parent1, parent2, genetic_noise_std=0.05):
    """
    Predict the offspring's expected CH4 score from two parents.

    The core assumption: offspring traits are roughly the average
    of both parents (additive genetic inheritance), with some
    random variation representing genetic recombination and
    environmental factors.

    Args:
        parent1, parent2: Cow objects
        genetic_noise_std: standard deviation of random variation
                           added to simulate genetic randomness

    Returns:
        predicted offspring low_ch4 probability (higher = better)
    """
    # Average of both parents' scores
    base_prediction = (parent1.ch4_score + parent2.ch4_score) / 2

    # Add small random genetic variation
    # np.random.normal centered at 0 simulates the natural
    # variation in how traits combine during reproduction
    noise = np.random.normal(0, genetic_noise_std)

    offspring_score = base_prediction + noise

    # Clip to valid probability range [0, 1]
    offspring_score = np.clip(offspring_score, 0, 1)

    return offspring_score


def find_best_breeding_pairs(cows, top_n=10):
    """
    Evaluate all possible breeding pairs and rank them by
    predicted offspring CH4 score.

    Args:
        cows: list of Cow objects
        top_n: how many top pairs to return

    Returns:
        list of (parent1, parent2, predicted_offspring_score) tuples,
        sorted from best (lowest emission risk) to worst
    """
    pairs = []

    # itertools.combinations generates every unique pair without
    # repeating (cow_A, cow_B) and (cow_B, cow_A) separately,
    # and without pairing a cow with herself
    for parent1, parent2 in itertools.combinations(cows, 2):
        offspring_score = predict_offspring_score(parent1, parent2)
        pairs.append((parent1, parent2, offspring_score))

    # Sort by offspring score descending — higher score means
    # higher probability of being a LOW emitter (better outcome)
    pairs.sort(key=lambda x: x[2], reverse=True)

    return pairs[:top_n]


def print_herd_summary(cows):
    """Print a summary of the scored herd"""
    print(f"\n{'='*60}")
    print(f"  HERD SUMMARY — {len(cows)} cows scored")
    print(f"{'='*60}\n")

    # Sort cows by their individual score, best first
    sorted_cows = sorted(cows, key=lambda c: c.ch4_score, reverse=True)

    print(f"{'Cow ID':<12}{'Breed':<25}{'Low CH4 prob':>14}{'Confidence':>14}")
    print("-" * 65)
    for cow in sorted_cows:
        print(f"{cow.cow_id:<12}{cow.breed:<25}"
              f"{cow.ch4_score*100:>13.1f}%{cow.confidence*100:>13.1f}%")


def print_breeding_recommendations(pairs):
    """Print the top recommended breeding pairs"""
    print(f"\n{'='*60}")
    print(f"  TOP {len(pairs)} RECOMMENDED BREEDING PAIRS")
    print(f"{'='*60}\n")
    print("Ranked by predicted offspring low-CH4 probability\n")

    for rank, (p1, p2, score) in enumerate(pairs, 1):
        print(f"#{rank}  {p1.cow_id} ({p1.breed}) x "
              f"{p2.cow_id} ({p2.breed})")
        print(f"     Parent scores: {p1.ch4_score*100:.1f}% / "
              f"{p2.ch4_score*100:.1f}%  →  "
              f"Predicted offspring: {score*100:.1f}% low-CH4 probability")
        print()


def print_disclaimer():
    print(f"\n{'='*60}")
    print("  IMPORTANT CONTEXT")
    print(f"{'='*60}")
    print("""
This breeding recommendation uses a SIMPLIFIED additive genetic
model (offspring score = average of parents + small random noise).

Real quantitative genetics is far more complex — it involves:
  - Multiple genes with varying dominance patterns
  - Non-additive genetic interactions (epistasis)
  - Environmental and maternal effects
  - Actual heritability coefficients specific to each trait

This is a reasonable FIRST APPROXIMATION for demonstrating the
breeding selection concept, grounded in published heritability
research (e.g. stature-methane genetic correlation r=0.43).

For real on-farm breeding decisions, consult a livestock genetics
specialist and use validated breeding value estimation tools.
""")


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load trained model
    print("\nLoading trained model...")
    model = build_model(freeze_backbone=False)
    model.load_state_dict(
        torch.load(MODELS_DIR / "green_cattle_best.pth", map_location=device)
    )
    model = model.to(device)
    print("Model loaded!")

    # Score a sample herd
    print("\nScoring herd from test set images...")
    cows = score_herd(model, device, num_cows=20)
    print(f"Scored {len(cows)} cows")

    # Print herd summary
    print_herd_summary(cows)

    # Find and print best breeding pairs
    best_pairs = find_best_breeding_pairs(cows, top_n=10)
    print_breeding_recommendations(best_pairs)

    print_disclaimer()