import numpy as np
import pandas as pd
from build_df import build_dataframe
from utils import read_measurement_file, compute_signal_delta
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

        if y is None or x is None or y_ref is None:
            continue

        # X ref
        x_ref = None
        if pd.notna(row["pair_id"]):
            row2 = meta_map.loc[row["pair_id"]]
            x_ref_path = row2["reference_zero_id"]
            if pd.notna(x_ref_path):
                x_ref = get_file(x_ref_path)
            else:
                x_ref = None
        
        if x_ref is None:
            continue

        try:
            y_signal, y_quality, yq1, yq2 = compute_signal_delta(y, y_ref)
            x_signal, x_quality, xq1, xq2 = compute_signal_delta(x, x_ref)

            if y_signal is None or x_signal is None:
                continue

            y75, y25 = np.percentile(y_signal, [75, 25])
            x75, x25 = np.percentile(x_signal, [75, 25])
            irq_Y = y75 - y25
            irq_X = x75 - x25

            records.append({
                "folder": row["temp_folder_label"],
                "pressure": row["pressure_corr_mpa"],
                "dT": row["deltaT_label"],
                "Tp": row["T_plate_label"],
                "mu_Y": np.mean(y_signal),
                "mu_X": np.mean(x_signal),
                "std_Y": np.std(y_signal),
                "std_X": np.std(x_signal),
                "irq_Y": irq_Y,
                "irq_X": irq_X,
                #"role": row["role"],
                "is_temp_calibration": row["is_temp_calibration"],
                "is_pressure_calibration": row["is_pressure_calibration"],
                "is_joint_regression": row["is_joint_regression"],
                "is_repeatability_test": row["is_repeatability_test"],
                "path_y": row["source_path"],
                "path_x": row["pair_id"],
                "ref": row["reference_zero_id"],
                "low_quality": not (y_quality and x_quality),
                "y_quality": yq1,
                "y_quality_ref": yq2,
                "x_quality": xq1,
                "x_quality_ref": xq2,
                "series_id": row["series_id"],
            })

        except Exception as e:
            print("Error:", e)

    df = pd.DataFrame(records)
    df["diff_XY"] = df["mu_X"] - df["mu_Y"]
    df["mean_XY"] = (df["mu_X"] + df["mu_Y"]) / 2

    return df

def detect_channel_swap(df_summary):

    folder_results = []

    for folder in tqdm(df_summary["folder"].unique(), desc="Detecting swap"):

        df_f = df_summary[df_summary["folder"] == folder]

        df_f = df_f.dropna()

        if len(df_f) < 3:
            continue

        try:
            slope_y = np.polyfit(df_f["pressure"], df_f["mu_Y"], 1)[0]
            slope_x = np.polyfit(df_f["pressure"], df_f["mu_X"], 1)[0]

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


if __name__ == "__main__":
    
    df_meta = build_dataframe()
    df_summary = build_folder_summary(df_meta)
    df_result = detect_channel_swap(df_summary)
    print(df_result)