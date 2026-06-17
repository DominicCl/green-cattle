import numpy as np

# ── What this file does ───────────────────────────────────────────
# Assigns realistic average farm record values per breed based on
# published livestock science literature.
#
# Since our Kaggle dataset has images but no farm records, we
# simulate breed-average values. This is scientifically grounded —
# breed averages are well documented in dairy science research.
#
# These values feed into the 23 CH4 formulas in formulas.py,
# producing a formula vector for each image based on its breed.
#
# Sources:
#   - NRC Nutrient Requirements of Dairy Cattle (2001)
#   - AFBI Northern Irish dairy cattle database
#   - Ross et al. (2026) descriptive statistics
# ─────────────────────────────────────────────────────────────────

# Breed average farm records
# Each value has a mean and standard deviation so we can add
# realistic random variation — real cows vary around the average
BREED_RECORDS = {
    "Holstein Friesian cattle": {
        "DMI":  {"mean": 22.0, "std": 2.0},   # kg/day
        "GEI":  {"mean": 400.0, "std": 30.0},  # MJ/day
        "MEI":  {"mean": 280.0, "std": 20.0},  # MJ/day
        "NDF":  {"mean": 380.0, "std": 40.0},  # g/kg DMI
        "NDFI": {"mean": 8.5,   "std": 1.0},   # kg/day
        "NDFP": {"mean": 40.0,  "std": 4.0},   # %
        "FP":   {"mean": 0.55,  "std": 0.05},  # proportion
        "ECMY": {"mean": 30.0,  "std": 4.0},   # kg/day
        "LWT":  {"mean": 650.0, "std": 50.0},  # kg
    },
    "Brown Swiss cattle": {
        "DMI":  {"mean": 20.0, "std": 2.0},
        "GEI":  {"mean": 370.0, "std": 28.0},
        "MEI":  {"mean": 260.0, "std": 18.0},
        "NDF":  {"mean": 370.0, "std": 38.0},
        "NDFI": {"mean": 7.8,   "std": 0.9},
        "NDFP": {"mean": 39.0,  "std": 4.0},
        "FP":   {"mean": 0.53,  "std": 0.05},
        "ECMY": {"mean": 25.0,  "std": 3.5},
        "LWT":  {"mean": 700.0, "std": 55.0},
    },
    "Ayrshire cattle": {
        "DMI":  {"mean": 17.0, "std": 1.8},
        "GEI":  {"mean": 315.0, "std": 25.0},
        "MEI":  {"mean": 220.0, "std": 16.0},
        "NDF":  {"mean": 360.0, "std": 36.0},
        "NDFI": {"mean": 6.8,   "std": 0.8},
        "NDFP": {"mean": 38.0,  "std": 3.5},
        "FP":   {"mean": 0.52,  "std": 0.05},
        "ECMY": {"mean": 22.0,  "std": 3.0},
        "LWT":  {"mean": 550.0, "std": 45.0},
    },
    "Jersey cattle": {
        "DMI":  {"mean": 14.0, "std": 1.5},
        "GEI":  {"mean": 260.0, "std": 22.0},
        "MEI":  {"mean": 182.0, "std": 14.0},
        "NDF":  {"mean": 340.0, "std": 34.0},
        "NDFI": {"mean": 5.2,   "std": 0.7},
        "NDFP": {"mean": 36.0,  "std": 3.5},
        "FP":   {"mean": 0.50,  "std": 0.05},
        "ECMY": {"mean": 18.0,  "std": 2.5},
        "LWT":  {"mean": 430.0, "std": 35.0},
    },
    "Red Dane cattle": {
        "DMI":  {"mean": 15.0, "std": 1.6},
        "GEI":  {"mean": 278.0, "std": 23.0},
        "MEI":  {"mean": 195.0, "std": 15.0},
        "NDF":  {"mean": 345.0, "std": 35.0},
        "NDFI": {"mean": 5.6,   "std": 0.7},
        "NDFP": {"mean": 37.0,  "std": 3.5},
        "FP":   {"mean": 0.51,  "std": 0.05},
        "ECMY": {"mean": 19.0,  "std": 2.8},
        "LWT":  {"mean": 460.0, "std": 38.0},
    },
}

# Map from folder name to breed name
# ImageFolder assigns labels alphabetically:
# high_ch4 = 0, low_ch4 = 1
# We need to map image paths back to breeds
FOLDER_TO_BREEDS = {
    "high_ch4": [
        "Holstein Friesian cattle",
        "Brown Swiss cattle",
        "Ayrshire cattle"
    ],
    "low_ch4": [
        "Jersey cattle",
        "Red Dane cattle"
    ],
}


def sample_records(breed_name, noise=True):
    """
    Sample farm records for a given breed.

    Args:
        breed_name: one of the five breed names above
        noise: if True, adds random variation around the mean
               this simulates individual cow variation within a breed

    Returns:
        dict of farm record values
    """
    records = BREED_RECORDS[breed_name]
    sampled = {}

    for var, stats in records.items():
        if noise:
            # np.random.normal samples from a normal distribution
            # centered at mean with spread std
            # This means most values fall within 1-2 std of the mean
            val = np.random.normal(stats["mean"], stats["std"])
            # Clip to reasonable biological limits — no negative intake
            val = max(val, stats["mean"] * 0.5)
        else:
            val = stats["mean"]
        sampled[var] = float(val)

    return sampled


def get_formula_vector_for_breed(breed_name, noise=True):
    """
    Get the full 23-formula vector for a breed.
    This is what gets fed into the model's formula stream.

    Args:
        breed_name: one of the five breed names
        noise: whether to add individual variation

    Returns:
        numpy array of shape (23,)
    """
    from formulas import get_formula_vector

    records = sample_records(breed_name, noise=noise)

    return get_formula_vector(
        DMI=records["DMI"],
        GEI=records["GEI"],
        MEI=records["MEI"],
        NDF=records["NDF"],
        NDFI=records["NDFI"],
        NDFP=records["NDFP"],
        FP=records["FP"],
        ECMY=records["ECMY"],
        LWT=records["LWT"],
    )


if __name__ == "__main__":
    print("Breed average CH4 predictions (mean of 23 formulas)\n")
    print(f"{'Breed':<30} {'Mean DMI':>10} {'CH4 est.':>10}")
    print("─" * 52)

    for breed, records in BREED_RECORDS.items():
        vec = get_formula_vector_for_breed(breed, noise=False)
        mean_ch4 = vec.mean()
        dmi = records["DMI"]["mean"]
        print(f"{breed:<30} {dmi:>10.1f} {mean_ch4:>10.1f} g/day")

    print("\nFormula vectors ready for fusion layer!")