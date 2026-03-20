from pathlib import Path
import pandas as pd

def build_dataframe(root_path="PAKA_AI"):
    # Column names in table
    columns = [
        "source_path",
        "temp_folder_label",
        "deltaT_label",
        "T_plate_label",
        "series_id",
        "channel_raw",
        "channel_norm",
        "pressure_raw_label"
    ]

    root_dir = Path("PAKA_AI")                              # Root folder
    files = list(root_dir.rglob("*.txt"))                   # Find every file ended with .txt
    print("Number of .txt files found: ",len(files))        # Number of .txt files

    rows = []

    for f in files:
        parts = f.parts

        # === Full Path ===
        source_path = str(f)

        # === Temperature folder ===
        temp_folder = parts[-3]  # np. DT0_T23_15

        # Split: DT0_T23_15
        temp_parts = temp_folder.split("_")

        deltaT = float(temp_parts[0].replace("DT", ""))   # 0
        T_plate = int(temp_parts[1].replace("T", ""))   # 23
        series_id = temp_parts[2]     # 15

        # === Subfolders ===
        subfolder = parts[-2]         # np. 15_CH2_X_SLOW
        sub_parts = subfolder.split("_")

        channel_raw = sub_parts[1]    # CH2

        # channel normalization
        if "X" in sub_parts:
            channel_norm = "X"
        elif "Y" in sub_parts:
            channel_norm = "Y"
        else:
            channel_norm = None

        # === Pressure ===
        filename = parts[-1]
        # np. Switch_channel_2-0.5MPa-1.txt
        try:
            pressure_raw = filename.split("-")[1]  # 0.5MPa
        except IndexError:
            pressure_raw = None

        # === Add row ===
        rows.append({
            "source_path": source_path,
            "temp_folder_label": temp_folder,
            "deltaT_label": deltaT,
            "T_plate_label": T_plate,
            "series_id": series_id,
            "channel_raw": channel_raw,
            "channel_norm": channel_norm,
            "pressure_raw_label": pressure_raw
        })

    # Create DataFrame
    df = pd.DataFrame(rows,columns=columns)
    return df
