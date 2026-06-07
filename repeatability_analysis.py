import os

import numpy as np
import pandas as pd


def compute_repeatability_metrics(
    input_path="Output_files/training_dataset.csv",
    output_path="Output_files/repeatability_results.csv",
    summary_path="Output_files/repeatability_summary_by_dt.csv",
):
    df = pd.read_csv(input_path)

    zero_start = df[
        (df["is_temp_calibration"] == True)
        & (df["is_repeatability_test"] == False)
    ].copy()

    zero_end = df[df["is_repeatability_test"] == True].copy()

    print("zero_start all:", len(zero_start))
    print("zero_end all:", len(zero_end))

    merged = zero_end.merge(
        zero_start[
            [
                "path_y",
                "mu_X",
                "mu_Y",
                "std_X",
                "std_Y",
                "dT",
                "Tp",
                "series_id",
                "low_quality",
            ]
        ],
        left_on="ref",
        right_on="path_y",
        suffixes=("_end", "_start"),
        how="left",
    )

    missing_refs = merged["mu_X_start"].isna().sum()

    print("merged:", len(merged))
    print("missing reference:", missing_refs)

    merged["pair_low_quality"] = (
        (merged["low_quality_end"] == True)
        | (merged["low_quality_start"] == True)
        | (merged["mu_X_start"].isna())
        | (merged["mu_Y_start"].isna())
    )

    good_pairs = merged[merged["pair_low_quality"] == False].copy()

    print("good repeatability pairs:", len(good_pairs))

    for data in [merged, good_pairs]:
        data["RD_X"] = data["mu_X_end"] - data["mu_X_start"]
        data["RD_Y"] = data["mu_Y_end"] - data["mu_Y_start"]
        data["RD_norm"] = np.sqrt(data["RD_X"] ** 2 + data["RD_Y"] ** 2)
        data["abs_RD_X"] = data["RD_X"].abs()
        data["abs_RD_Y"] = data["RD_Y"].abs()

    summary = good_pairs[
        ["RD_X", "RD_Y", "RD_norm", "abs_RD_X", "abs_RD_Y"]
    ].describe()

    by_dt = (
        good_pairs
        .groupby("dT_end")[["RD_X", "RD_Y", "RD_norm", "abs_RD_X", "abs_RD_Y"]]
        .agg(["mean", "std", "max"])
    )

    print("\nRD SUMMARY, good pairs only:")
    print(summary)

    print("\nRD BY dT, good pairs only:")
    print(by_dt)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    merged.to_csv(output_path, index=False)
    by_dt.to_csv(summary_path)

    print(f"\nSaved repeatability results to: {output_path}")
    print(f"Saved repeatability summary to: {summary_path}")

    return merged, good_pairs, summary, by_dt


if __name__ == "__main__":
    compute_repeatability_metrics()
