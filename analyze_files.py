import numpy as np
import pandas as pd
from build_df import build_dataframe
from utils import read_measurement_file, compute_mean_roi_from_file
from tqdm import tqdm

def build_folder_summary(df_meta):

    records = []
    file_cache = {}

    def get_file(path):
        if path is None or pd.isna(path):
            return None
        if path not in file_cache:
            file_cache[path] = read_measurement_file(path)
        return file_cache[path]

    df_y = df_meta[df_meta["channel_norm"] == "Y"]

    meta_map = df_meta.set_index("source_path")

    for _, row in tqdm(df_y.iterrows(), total=len(df_y), desc="Building summary"):

        if pd.isna(row["pair_id"]):
            continue

        y = get_file(row["source_path"])
        x = get_file(row["pair_id"])
        y_ref = get_file(row["reference_zero_id"])

        # X ref
        x_ref = None
        if pd.notna(row["pair_id"]):
            row2 = meta_map.loc[row["pair_id"]]
            x_ref_path = row2["reference_zero_id"]
            if pd.notna(x_ref_path):
                x_ref = get_file(x_ref_path)
            else:
                x_ref = None

        try:
            y_mean = compute_mean_roi_from_file(y)
            x_mean = compute_mean_roi_from_file(x)

            y_ref_mean = compute_mean_roi_from_file(y_ref) if y_ref is not None else 0
            x_ref_mean = compute_mean_roi_from_file(x_ref) if x_ref is not None else 0

            records.append({
                "folder": row["temp_folder_label"],
                "pressure": row["pressure_corr_mpa"],
                "y_signal": y_mean - y_ref_mean,
                "x_signal": x_mean - x_ref_mean
            })

        except Exception as e:
            print("Error:", e)

    return pd.DataFrame(records)

def detect_channel_swap(df_summary):

    folder_results = []

    for folder in tqdm(df_summary["folder"].unique(), desc="Detecting swap"):

        df_f = df_summary[df_summary["folder"] == folder]

        df_f = df_f.dropna()

        if len(df_f) < 3:
            continue

        try:
            slope_y = np.polyfit(df_f["pressure"], df_f["y_signal"], 1)[0]
            slope_x = np.polyfit(df_f["pressure"], df_f["x_signal"], 1)[0]

            folder_results.append({
                "folder": folder,
                "slope_y": slope_y,
                "slope_x": slope_x,
                "opposite_sign": np.sign(slope_y) != np.sign(slope_x)
            })

        except Exception as e:
            print("Regression error:", e)

    df_result = pd.DataFrame(folder_results)

    threshold = 0.05

    df_result["valid"] = (
        (np.abs(df_result["slope_y"]) > threshold) &
        (np.abs(df_result["slope_x"]) > threshold)
    )

    df_result["swapped"] = (
        df_result["valid"] &
        (~df_result["opposite_sign"])
    )

    return df_result

# ======================
# GŁÓWNA ANALIZA
# ======================

df_meta = build_dataframe()

df_summary = build_folder_summary(df_meta)

df_result = detect_channel_swap(df_summary)

print(df_result)