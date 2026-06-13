import numpy as np

# ── What this file does ───────────────────────────────────────────
# Implements all 23 published dairy cattle CH4 prediction formulas
# from Ross et al. (2026), Journal of Dairy Science.
#
# Each formula takes farm measurements as input and returns
# CH4 Production (CHP) in grams per day.
#
# Input variables:
#   DMI  = Dry Matter Intake (kg/day) — how much dry feed the cow eats
#   GEI  = Gross Energy Intake (MJ/day) — energy content of that feed
#   MEI  = Metabolisable Energy Intake (MJ/day) — energy cow can use
#   NDF  = Neutral Detergent Fiber content (g/kg DMI) — fiber in feed
#   NDFI = NDF Intake (kg/day) — actual kg of fiber eaten
#   NDFP = NDF Proportion (%) — percentage of diet that is fiber
#   FP   = Forage Proportion (%) — percentage of diet that is forage
#   ECMY = Energy Corrected Milk Yield (kg/day) — milk output
#   LWT  = Live Weight (kg) — body weight of the cow
#
# Conversion constants:
#   1 MJ CH4 = 55.2176 g CH4  (energy to mass)
#   1 MJ CH4 = 1000 kJ        (unit conversion)
# ─────────────────────────────────────────────────────────────────

MJ_TO_G = 1000 / 55.2176

def che_to_chp(che_mj):
    """Convert CH4 Energy (MJ/day) to CH4 Production (g/day)"""
    return che_mj * MJ_TO_G

def predict_all(DMI, GEI=None, MEI=None, NDF=None, NDFI=None,
                NDFP=None, FP=None, ECMY=None, LWT=None):
    """
    Run all 23 formulas and return predictions in g/day.
    Returns a dictionary: {formula_name: prediction_g_per_day}
    """
    predictions = {}

    # ── Niu et al. 2021 ───────────────────────────────────────────
    if DMI is not None and NDF is not None:
        che = (-26.0 + (15.3 * DMI) + (3.42 * (NDF / 10))) * 0.05565
        predictions["Niu2021_1"] = che_to_chp(che)

    if DMI is not None:
        che = (107 + (14.5 * DMI)) * 0.05565
        predictions["Niu2021_2"] = che_to_chp(che)

    # ── Niu et al. 2018 ───────────────────────────────────────────
    if DMI is not None and NDFP is not None:
        che = (33.2 + (13.6 * DMI) + (2.43 * NDFP)) * (55.65 / 1000)
        predictions["Niu2018_1"] = che_to_chp(che)

    # ── Charmley et al. 2016 ─────────────────────────────────────
    # Outputs directly in g/day — no conversion needed
    if DMI is not None:
        predictions["Charmley2016_1"] = 38.0 + (19.22 * DMI)

    if GEI is not None:
        predictions["Charmley2016_2"] = (2.14 + (0.058 * GEI)) / 0.05565

    # ── Ramin & Huhtanen 2013 ─────────────────────────────────────
    if DMI is not None:
        che = ((20 + (35.8 * DMI) - (0.5 * (DMI ** 2))) * 0.716) * 0.05565
        predictions["Ramin2013_1"] = che_to_chp(che)

    # Outputs in MJ/day — needs conversion
    if DMI is not None:
        # Output is in L/day, convert to g using CH4 density (1L = 0.716g)
        ch4_litres = (62 + (25 * DMI)) * 0.714
        predictions["Ramin2013_2"] = ch4_litres * 0.716

    # ── Mills et al. 2003 ─────────────────────────────────────────
    if DMI is not None:
        predictions["Mills2003_1"] = che_to_chp(5.93 + (0.92 * DMI))

    if MEI is not None:
        predictions["Mills2003_2"] = che_to_chp(8.25 + (0.07 * MEI))

    if FP is not None and DMI is not None:
        predictions["Mills2003_3"] = che_to_chp(
            1.06 + (10.27 * FP) + (0.87 * DMI)
        )

    # BEST individual model from Ross et al. — nonlinear Mitscherlich
    # Biologically correct: zero emissions at zero intake
    if DMI is not None:
        che = 56.27 - ((56.27 + 0) * (np.e ** (-0.028 * DMI)))
        predictions["Mills2003_NL4"] = che_to_chp(che)

    if MEI is not None:
        che = 45.98 - ((45.98 + 0) * (np.e ** (-0.003 * MEI)))
        predictions["Mills2003_NL5"] = che_to_chp(che)

    # ── Storlien et al. 2014 ─────────────────────────────────────
    if DMI is not None:
        predictions["Storlien2014_1"] = che_to_chp(-1.47 + (1.28 * DMI))

    # Outputs directly in g/day — no conversion needed
    if NDFI is not None:
        # Output is in MJ/day — convert to g/day
        predictions["Storlien2014_2"] = che_to_chp(-2.76 + (3.74 * NDFI))

    # ── Hristov et al. 2003 ───────────────────────────────────────
    # Outputs directly in g/day — no conversion needed
    if DMI is not None:
        predictions["Hristov2003_1"] = 2.54 + (19.14 * DMI)

    # ── Dong et al. 2022 ──────────────────────────────────────────
    if DMI is not None:
        predictions["Dong2022_1"] = che_to_chp(7.89 + (0.628 * DMI))

    if NDFI is not None:
        predictions["Dong2022_2"] = che_to_chp(8.97 + (1.522 * NDFI))

    # ── Nielsen et al. 2013 ───────────────────────────────────────
    if DMI is not None:
        predictions["Nielsen2013_1"] = (0 + (1.26 * DMI)) / 0.05565

    # ── Congio et al. 2022 ────────────────────────────────────────
    # All output directly in g/day — no conversion needed
    if DMI is not None:
        predictions["Congio2022_1"] = 40.7 + (18.0 * DMI)

    if GEI is not None:
        predictions["Congio2022_2"] = 42.1 + (1.00 * GEI)

    if DMI is not None and ECMY is not None:
        predictions["Congio2022_3"] = 30.6 + (16.3 * DMI) + (2.04 * ECMY)

    # ── IPCC Tier 2 ───────────────────────────────────────────────
    # International standard emission factors used by governments
    if GEI is not None:
        predictions["IPCC2006"] = (0 + (0.065 * GEI)) / 0.05565
        predictions["IPCC1997"] = (0 + (0.060 * GEI)) / 0.05565

    return predictions


def get_formula_vector(DMI, GEI=None, MEI=None, NDF=None, NDFI=None,
                       NDFP=None, FP=None, ECMY=None, LWT=None):
    """
    Returns a fixed-length numpy array of 23 formula predictions.
    Missing predictions are filled with the mean of available ones.
    This is what gets fed into the fusion layer of the model.
    """
    preds = predict_all(DMI, GEI, MEI, NDF, NDFI, NDFP, FP, ECMY, LWT)

    formula_names = [
        "Niu2021_1", "Niu2021_2", "Niu2018_1",
        "Charmley2016_1", "Charmley2016_2",
        "Ramin2013_1", "Ramin2013_2",
        "Mills2003_1", "Mills2003_2", "Mills2003_3",
        "Mills2003_NL4", "Mills2003_NL5",
        "Storlien2014_1", "Storlien2014_2",
        "Hristov2003_1",
        "Dong2022_1", "Dong2022_2",
        "Nielsen2013_1",
        "Congio2022_1", "Congio2022_2", "Congio2022_3",
        "IPCC2006", "IPCC1997"
    ]

    values = []
    for name in formula_names:
        if name in preds and preds[name] is not None:
            values.append(preds[name])
        else:
            values.append(np.nan)

    arr = np.array(values, dtype=np.float32)
    mean_val = np.nanmean(arr)
    arr = np.where(np.isnan(arr), mean_val, arr)

    return arr


if __name__ == "__main__":
    print("Testing formulas on a typical Holstein cow...")
    print("DMI=22 kg/day, GEI=400 MJ/day, LWT=650 kg\n")

    preds = predict_all(
        DMI=22.0,
        GEI=400.0,
        MEI=280.0,
        NDF=380.0,
        NDFI=8.5,
        NDFP=40.0,
        FP=0.55,
        ECMY=30.0,
        LWT=650.0
    )

    values = [v for v in preds.values() if v is not None]

    print(f"{'Formula':<20} {'CH4 (g/day)':>12}")
    print("─" * 34)
    for name, val in preds.items():
        print(f"{name:<20} {val:>12.1f}")

    print("─" * 34)
    print(f"{'Mean prediction':<20} {np.mean(values):>12.1f}")
    print(f"{'Min prediction':<20} {np.min(values):>12.1f}")
    print(f"{'Max prediction':<20} {np.max(values):>12.1f}")

    vec = get_formula_vector(DMI=22.0, GEI=400.0, MEI=280.0)
    print(f"\nFormula vector shape: {vec.shape}")
    print("Ready to feed into the fusion layer!")