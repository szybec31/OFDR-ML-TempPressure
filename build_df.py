from pathlib import Path
import pandas as pd

def build_dataframe():
    # Column names in table
    columns = [
        "source_path",
        "temp_folder_label",
        "deltaT_label",
        "T_plate_label",
        "series_id",
        "channel_raw",
        "channel_norm",
        "pressure_raw_label",
        "pressure_corr_mpa",
        "point_type",
        "pair_id",
        #"role",
        "is_complete_pair",
        "reference_zero_id",
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
            pressure_corr = float(pressure_raw.split("MPa")[0])
            point_type = None if pressure_raw == None else "zero_start" if pressure_corr == 0 else "zero_end" if pressure_corr == 11 or pressure_corr == 0.01 else "pressure_point"
            pressure_corr = None if pressure_raw == None else 0 if pressure_corr == 0 or pressure_corr == 11 or pressure_corr == 0.01 else pressure_corr

        except IndexError:
            pressure_raw = None
            pressure_corr = None
            point_type = None

        # === Add row ===
        rows.append({
            "source_path": source_path,
            "temp_folder_label": temp_folder,
            "deltaT_label": deltaT,
            "T_plate_label": T_plate,
            "series_id": series_id,
            "channel_raw": channel_raw,
            "channel_norm": channel_norm,
            "pressure_raw_label": pressure_raw,
            "pressure_corr_mpa": pressure_corr,
            "point_type": point_type,
            "pair_id": None,
            #"role": None,
            "is_complete_pair": None,
            "reference_zero_id": None,
        })

    # Create DataFrame
    df = pd.DataFrame(rows,columns=columns)

    # folder bazowy (do DT0_T23_15)
    df["base_folder"] = df["source_path"].str.extract(r'^(PAKA_AI\\[^\\]+)')

    # końcówka pliku
    df["file_suffix"] = df["source_path"].str.extract(r'-(\d+\.?\d*MPa-\d+\.txt)$')

    df["pair_key"] = df["base_folder"] + "\\" + df["file_suffix"]

    counts = df.groupby("pair_key")["source_path"].transform("count")
    df["is_complete_pair"] = counts == 2

    df["pair_id"] = df.groupby("pair_key", group_keys=False).apply(get_pair)

    df = df.drop(columns=["base_folder", "file_suffix", "pair_key"], errors="ignore")

    ## Ref ID:

    # =========================
# Reference mapping
# =========================

    df["idx"] = df["source_path"].str.extract(r'-(\d+)\.txt$')

    df["base"] = df["source_path"].str.extract(
        r'^(.*Switch_channel_\d-)'
    )

    df["ref_key"] = df["base"] + df["idx"]

    ref_map = (
        df[df["point_type"] == "zero_start"]
        .drop_duplicates("ref_key")
        .set_index("ref_key")["source_path"]
    )

    df["reference_zero_id"] = df["ref_key"].map(ref_map)

    df = df.drop(
        columns=["idx", "base", "ref_key"],
        errors="ignore"
    )

    # conditions = {
    #     "temp_calibration;": (
    #         (df["pressure_corr_mpa"] == 0) &
    #         (df["point_type"] == "zero_start") &
    #         (df["pair_id"].notna())
    #     ),
    #     "pressure_calibration;": (
    #         (df["deltaT_label"] == 0) &
    #         (df["pressure_corr_mpa"].between(0, 10)) &
    #         (df["point_type"] != "zero_end") &
    #         (df["pair_id"].notna())
    #     ),
    #     "joint_regression;": (
    #         (df["pressure_corr_mpa"].between(0, 10)) &
    #         (df["pair_id"].notna())
    #     ),
    #     "repeatability_test;": (
    #         df["point_type"] == "zero_end"
    #     )
    # }

    # df["role"] = "EDA;"

    # for label, mask in conditions.items():
    #     df.loc[mask, "role"] += label

    # ======================
    # ROLE FLAGS
    # ======================

    df["is_temp_calibration"] = (
        (df["pressure_corr_mpa"] == 0) &
        (df["point_type"] == "zero_start") &
        (df["pair_id"].notna())
    )

    df["is_pressure_calibration"] = (
        (df["deltaT_label"] == 0) &
        (df["pressure_corr_mpa"].between(0, 10)) &
        (df["point_type"] != "zero_end") &
        (df["pair_id"].notna())
    )

    df["is_joint_regression"] = (
        (df["pressure_corr_mpa"].between(0, 10)) &
        (df["pair_id"].notna())
    )

    df["is_repeatability_test"] = (
        df["point_type"] == "zero_end"
    )

    return df



def get_pair(group):
    paths = group["source_path"].tolist()
    if len(paths) == 2:
        return pd.Series(paths[::-1], index=group.index)
    return pd.Series([None]*len(group), index=group.index)