from pathlib import Path
import re
import pandas as pd

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

def compute_mean_roi_from_file(df, roi_min=7.0, roi_max=12):
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

