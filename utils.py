from pathlib import Path
import pandas as pd
import numpy as np

ROI_MIN = 8.5
ROI_MAX = 12
TRESHOLD = 0.9

def fix_dt16_folder_structure(root_dir="PAKA_AI"):
    root = Path(root_dir)

    for dt_folder in root.iterdir():

        if not dt_folder.is_dir():
            continue

        if not dt_folder.name.startswith("DT16"):
            continue

        subfolders = {f.name: f for f in dt_folder.iterdir() if f.is_dir()}

        if "02_CH1_X_SLOW" in subfolders and "02_CH2_Y_FAST" in subfolders:

            folder_x = subfolders["02_CH1_X_SLOW"]
            folder_y = subfolders["02_CH2_Y_FAST"]

            print(f"Fixing folder: {dt_folder}")

            # tymczasowe nazwy (żeby uniknąć konfliktu)
            tmp_x = dt_folder / "TMP_X"
            tmp_y = dt_folder / "TMP_Y"

            folder_x.rename(tmp_x)
            folder_y.rename(tmp_y)

            # docelowe nazwy
            new_x = dt_folder / "02_CH2_X_SLOW"
            new_y = dt_folder / "02_CH1_Y_FAST"

            tmp_x.rename(new_x)
            tmp_y.rename(new_y)

            # --- zmiana nazw plików ---
            for folder in [new_x, new_y]:
                for file in folder.glob("*.txt"):

                    name = file.name

                    if "Switch_channel_1" in name:
                        new_name = name.replace("Switch_channel_1", "Switch_channel_2")

                    elif "Switch_channel_2" in name:
                        new_name = name.replace("Switch_channel_2", "Switch_channel_1")

                    else:
                        continue

                    new_path = file.with_name(new_name)
                    file.rename(new_path)

            print(f"✔ Fixed: {dt_folder}\n")

def compute_mean_roi_from_file(df, roi_min=ROI_MIN, roi_max=ROI_MAX):
    roi = df[(df["length"] >= roi_min) & (df["length"] <= roi_max)]
    return roi["value"].mean()

def read_measurement_file(file_path):
    """
    Wczytuje plik .txt i zwraca DataFrame z kolumnami:
    length, value
    """
    data_started = False
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Szukamy początku tabeli
            if line.startswith("Length"):
                data_started = True
                continue

            if not data_started:
                continue

            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                length = float(parts[0])
                value = float(parts[1])
                rows.append((length, value))
            except ValueError:
                continue

    return pd.DataFrame(rows, columns=["length", "value"])

def clean_roi(signal, roi_min=ROI_MIN, roi_max=ROI_MAX):
    
    roi = signal[(signal["length"] >= roi_min) & (signal["length"] <= roi_max)]

    if len(roi) < 10:
        return None, False

    # obcięcie brzegów
    roi = roi.iloc[2:-2]

    original_len = len(roi)

    # usuń NaN
    roi = roi.dropna()

    if len(roi) == 0:
        return None, False

    values = roi["value"].values

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        return roi, False

    z = 0.6745 * (values - median) / mad

    mask = np.abs(z) <= 3.5

    clean = roi[mask]

    quality = len(clean) >= TRESHOLD * original_len

    return clean, quality, len(clean)/original_len

def compute_signal_delta(y_df, y_ref_df):

    y_clean, q1, l1 = clean_roi(y_df)
    y_ref_clean, q2, l2 = clean_roi(y_ref_df)

    if y_clean is None or y_ref_clean is None:
        return None, False

    min_len = min(len(y_clean), len(y_ref_clean))

    y_vals = y_clean["value"].values[:min_len]
    y_ref_vals = y_ref_clean["value"].values[:min_len]

    delta = y_vals - y_ref_vals

    return delta, (q1 and q2), l1, l2

def quality_report(df, threshold = 0.9):

    # przelicz low_quality na podstawie threshold
    low_quality_recomputed = (
        (df["x_quality"] <= threshold) |
        (df["x_quality_ref"] <= threshold) |
        (df["y_quality"] <= threshold) |
        (df["y_quality_ref"] <= threshold)
    )

    report = {
        "total": len(df),

        # stare vs nowe
        "low_quality_original": df["low_quality"].sum(),
        "low_quality_recomputed": low_quality_recomputed.sum(),

        # pojedyncze warunki
        "x_good": (df["x_quality"] > threshold).sum(),
        "x_ref_good": (df["x_quality_ref"] > threshold).sum(),
        "y_good": (df["y_quality"] > threshold).sum(),
        "y_ref_good": (df["y_quality_ref"] > threshold).sum(),
    }

    # wszystkie warunki jednocześnie
    all_good_mask = (
        (df["x_quality"] > threshold) &
        (df["x_quality_ref"] > threshold) &
        (df["y_quality"] > threshold) &
        (df["y_quality_ref"] > threshold)
    )

    report["all_good"] = all_good_mask.sum()

    return pd.Series(report)

