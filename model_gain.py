import os

import numpy as np
import pandas as pd


def compute_model_gain(
    input_path="Output_files/res_corr_temperature.csv",
    output_path="Output_files/model_gain_vs_baseline.csv",
    baseline_model="AN-BL",
):
    df = pd.read_csv(input_path)

    if baseline_model not in df["Model"].values:
        raise ValueError(f"Baseline model {baseline_model} not found in {input_path}")

    baseline = df[df["Model"] == baseline_model].iloc[0]

    pressure_base_rmse = float(baseline["PRESSURE_RMSE_avg"])
    dt_base_rmse = float(baseline["DT_RMSE_avg"])

    rows = []

    for _, row in df.iterrows():
        model = row["Model"]

        if model == baseline_model:
            continue

        pressure_rmse = float(row["PRESSURE_RMSE_avg"])
        dt_rmse = float(row["DT_RMSE_avg"])

        pressure_gain = (
            (pressure_base_rmse - pressure_rmse) / pressure_base_rmse
            if pressure_base_rmse != 0
            else np.nan
        )

        dt_gain = (
            (dt_base_rmse - dt_rmse) / dt_base_rmse
            if dt_base_rmse != 0
            else np.nan
        )

        mean_gain = np.nanmean([pressure_gain, dt_gain])

        rows.append({
            "model": model,
            "baseline_model": baseline_model,
            "pressure_rmse_baseline": pressure_base_rmse,
            "pressure_rmse_model": pressure_rmse,
            "pressure_gain": pressure_gain,
            "pressure_gain_percent": pressure_gain * 100,
            "pressure_gain_ge_15_percent": pressure_gain >= 0.15,
            "dT_rmse_baseline": dt_base_rmse,
            "dT_rmse_model": dt_rmse,
            "dT_gain": dt_gain,
            "dT_gain_percent": dt_gain * 100,
            "dT_gain_ge_15_percent": dt_gain >= 0.15,
            "mean_gain": mean_gain,
            "mean_gain_percent": mean_gain * 100,
        })

    result = pd.DataFrame(rows)

    print("\nMODEL GAIN VS BASELINE")
    print("=" * 100)
    print(f"Input file: {input_path}")
    print(f"Baseline: {baseline_model}")
    print(f"Baseline PRESSURE_RMSE: {pressure_base_rmse:.3f}")
    print(f"Baseline DT_RMSE: {dt_base_rmse:.3f}")

    display_cols = [
        "model",
        "pressure_rmse_model",
        "pressure_gain_percent",
        "pressure_gain_ge_15_percent",
        "dT_rmse_model",
        "dT_gain_percent",
        "dT_gain_ge_15_percent",
        "mean_gain_percent",
    ]

    print("\n")
    print(result[display_cols].to_string(index=False))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"\nSaved model gain table to: {output_path}")

    return result


if __name__ == "__main__":
    compute_model_gain()