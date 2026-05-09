"""
material_classifier_3.py  —  Two-stage capacitive material classifier.

Two-stage decision logic
------------------------
Stage 1 — Saturation test (metal detection):
    Metal objects short the AD7746 sense plates at the excitation frequency,
    driving the ADC to its rail on the majority of samples.  If more than
    SATURATION_THRESHOLD (80%) of the 100 samples lie within SATURATION_MARGIN
    counts of either ADC rail (0x000000 or 0xFFFFFF), the object is classified
    as Metal without further analysis.

Stage 2 — Mean-capacitance threshold (dielectric discrimination):
    If the saturation rate is below 80%, the corrected mean capacitance is
    compared against two midpoint thresholds derived from calibration data
    (see Section 4.3 of the project report):

        mean > METAL_GLASS_AVG_THRESHOLD  ->  Glass
        mean > GLASS_PAPER_AVG_THRESHOLD  ->  Paper
        else                              ->  Paper

Usage
-----
Standalone:
    python material_classifier_3.py <path_to_results_file>
    python material_classifier_3.py --selftest

Imported by pick_and_place_12.py:
    from material_classifier_3 import parse_file, extract_features, classify
"""

import os
import sys

import numpy as np

# -- Conversion constants -----------------------------------------------------
# The AD7746 GUI applies internal corrections not fully documented in the
# datasheet.  A linear correction is fitted to two anchor measurements
# (Paper and Metal) taken directly from the GUI display:
#
#   Reference averages from GUI (pF):   Paper = -1.932979 | Metal = -1.530241
#   Raw hex averages from files  (pF):  Paper = -3.776252 | Metal = -2.904855

RAW_PAPER = -3.776252
RAW_METAL = -2.904855
GUI_PAPER = -1.932979
GUI_METAL = -1.530241

SCALE  = (GUI_METAL - GUI_PAPER) / (RAW_METAL - RAW_PAPER)
OFFSET = GUI_PAPER - SCALE * RAW_PAPER

# -- Stage 1 constants  (see Section 4.3) ------------------------------------
# Any sample within SATURATION_MARGIN counts of either ADC rail is considered
# saturated.  500 counts = ~0.003% of the 24-bit full scale.
ADC_MAX              = 0xFFFFFF   # 16 777 215
ADC_MIN              = 0x000000
SATURATION_MARGIN    = 500        # ADC counts
SATURATION_THRESHOLD = 0.80       # fraction of samples; >80% -> Metal

# -- Stage 2 constants  (see Section 4.3) ------------------------------------
# Midpoints between per-class calibration means:
#   Metal ~= -1.530 pF | Glass ~= -1.819 pF | Paper ~= -1.933 pF
METAL_GLASS_AVG_THRESHOLD = -1.675  # pF  (Metal/Glass boundary)
GLASS_PAPER_AVG_THRESHOLD = -1.876  # pF  (Glass/Paper boundary)


# -- Conversion helpers -------------------------------------------------------

def hex_to_raw(hex_str):
    """Convert 24-bit hex code (string) to raw capacitance in pF."""
    raw = int(hex_str, 16)
    return ((raw / 0x800000) - 1.0) * 8.192


def raw_to_corrected(raw_val):
    """Apply linear correction to match AD7746 GUI calibration."""
    return raw_val * SCALE + OFFSET


# -- File parsing -------------------------------------------------------------

def parse_file(filepath):
    """
    Parse a tab-delimited AD7746 results file.

    Returns
    -------
    corrected_readings : list[float]  calibrated pF values
    raw_int_samples    : list[int]    raw ADC counts (needed for Stage 1)

    Each data line must be: <6-char hex>\\t<anything>
    Lines that do not match are silently skipped.
    """
    corrected = []
    raw_ints  = []
    with open(filepath, 'r') as fh:
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) >= 1 and len(parts[0]) == 6:
                try:
                    raw_int   = int(parts[0], 16)
                    raw_float = hex_to_raw(parts[0])
                    corrected.append(raw_to_corrected(raw_float))
                    raw_ints.append(raw_int)
                except ValueError:
                    continue
    return corrected, raw_ints


# -- Feature extraction -------------------------------------------------------

def extract_features(readings, raw_ints):
    """
    Compute features from a measurement run.

    Parameters
    ----------
    readings  : list[float]  corrected pF values from parse_file
    raw_ints  : list[int]    raw ADC counts from parse_file

    Returns
    -------
    average              : float  corrected mean capacitance (pF)
    rms_noise            : float  RMS deviation across the run (pF)
    saturation_fraction  : float  fraction of samples at the ADC rail
    """
    arr = np.array(readings)
    average   = float(np.mean(arr))
    rms_noise = float(np.std(arr))

    saturated = sum(
        1 for s in raw_ints
        if s <= ADC_MIN + SATURATION_MARGIN
        or s >= ADC_MAX - SATURATION_MARGIN
    )
    saturation_fraction = saturated / len(raw_ints) if raw_ints else 0.0

    return average, rms_noise, saturation_fraction


# -- Classifier ---------------------------------------------------------------

def classify(average, rms_noise, saturation_fraction):
    """
    Two-stage material classifier.

    Stage 1: if >80% of samples are at the ADC rail, return "Metal"
    immediately (sense plates shorted by metal body).

    Stage 2: discriminate Glass from Paper using midpoint thresholds
    derived from calibration data (Section 4.3).

    Parameters
    ----------
    average              : float
    rms_noise            : float  (computed but reserved for future use)
    saturation_fraction  : float

    Returns
    -------
    str : "Metal", "Glass", or "Paper"
    """
    # Stage 1 -- saturation test
    if saturation_fraction > SATURATION_THRESHOLD:
        return "Metal"

    # Stage 2 -- mean-capacitance thresholds
    if average > METAL_GLASS_AVG_THRESHOLD:
        return "Glass"
    elif average > GLASS_PAPER_AVG_THRESHOLD:
        return "Paper"
    else:
        return "Paper"


# -- Convenience wrapper ------------------------------------------------------

def classify_file(filepath):
    """Parse filepath and return (label, average, rms_noise, sat_frac)."""
    readings, raw_ints = parse_file(filepath)
    avg, noise, sat = extract_features(readings, raw_ints)
    label = classify(avg, noise, sat)
    return label, avg, noise, sat


# -- Standalone entry point ---------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python material_classifier_3.py <results_file.txt>")
        print("       python material_classifier_3.py --selftest")
        sys.exit(1)

    if sys.argv[1] == "--selftest":
        # Place paper_results.txt / glass_results.txt / metal_results.txt
        # in a data/ subfolder next to this script, then run:
        #   python material_classifier_3.py --selftest
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir   = os.path.join(script_dir, "data")
        test_files = {
            "Paper": os.path.join(data_dir, "paper_results.txt"),
            "Glass": os.path.join(data_dir, "glass_results.txt"),
            "Metal": os.path.join(data_dir, "metal_results.txt"),
        }
        print("\nAD7746 Material Classifier -- self-test")
        print("========================================")
        print(f"SCALE={SCALE:.6f}  OFFSET={OFFSET:.6f}")
        correct = total = 0
        for true_label, fpath in test_files.items():
            if not os.path.isfile(fpath):
                print(f"  SKIP {true_label}: not found ({fpath})")
                continue
            label, avg, noise, sat = classify_file(fpath)
            total += 1
            ok = (label == true_label)
            correct += int(ok)
            tick = "OK" if ok else "FAIL"
            print(f"  [{tick}] {true_label:6s}  avg={avg:.4f} pF  "
                  f"sat={sat:.1%}  -> {label}")
        if total:
            print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.0f}%)"
                  f"  [training-set fit -- not held-out validation]")
        return

    # Single-file mode
    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    label, avg, noise, sat = classify_file(filepath)
    print(f"\nFile              : {filepath}")
    print(f"Saturation frac.  : {sat:.1%}  (threshold {SATURATION_THRESHOLD:.0%})")
    print(f"Mean capacitance  : {avg:.4f} pF")
    print(f"RMS noise         : {noise:.6f} pF")
    print(f"Predicted material: {label}")


if __name__ == "__main__":
    main()
