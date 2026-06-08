import os

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from baselines.utils.build_groups import build_groups


def analyze_split(df, groups, name):
    y = df[["pressure", "dT"]]
    logo = LeaveOneGroupOut()

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(df, y, groups=groups),
        start=1,
    ):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        train_series = set(train["series_id"])
        test_series = set(test["series_id"])

        train_dt = set(train["dT"])
        test_dt = set(test["dT"])

        train_pressure = set(train["pressure"])
        test_pressure = set(test["pressure"])

        rows.append(
            {
                "strategy": name,
                "fold": fold,
                "test_group": groups.iloc[test_idx].unique()[0],
                "test_size": len(test),
                "test_series_count": len(test_series),
                "shared_series_count": len(train_series & test_series),
                "shared_series_ratio": len(train_series & test_series)
                / max(len(test_series), 1),
                "shared_dT_count": len(train_dt & test_dt),
                "shared_pressure_count": len(train_pressure & test_pressure),
            }
        )

    return pd.DataFrame(rows)


def run_validation_diagnostics(
    input_path="Output_files/paired_features.csv",
    output_path="Output_files/validation_diagnostics.csv",
    summary_path="Output_files/validation_diagnostics_summary.csv",
):
    df = pd.read_csv(input_path)
    df = df[df["low_quality"] == False].copy()

    groups_temp = build_groups(df, leave_one_condition_out=False)
    groups_condition = build_groups(df, leave_one_condition_out=True)

    temp_results = analyze_split(
        df=df,
        groups=groups_temp,
        name="leave_one_temperature_out",
    )

    condition_results = analyze_split(
        df=df,
        groups=groups_condition,
        name="leave_one_condition_out",
    )

    results = pd.concat([temp_results, condition_results], ignore_index=True)

    summary = (
        results.groupby("strategy")
        [
            [
                "test_size",
                "test_series_count",
                "shared_series_count",
                "shared_series_ratio",
                "shared_dT_count",
                "shared_pressure_count",
            ]
        ]
        .agg(["count", "mean", "std", "min", "max"])
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    summary.to_csv(summary_path)

    print("\nVALIDATION DIAGNOSTICS SUMMARY")
    print("=" * 80)
    print(summary)

    print("\nFOLDS WITH SHARED SERIES")
    print("=" * 80)
    for strategy, group in results.groupby("strategy"):
        shared = (group["shared_series_count"] > 0).sum()
        total = len(group)
        print(f"{strategy}: {shared} / {total}")

    print("\nINTERPRETATION")
    print("=" * 80)
    print(
        "leave_one_temperature_out: no shared series between train and test; "
        "recommended as the main generalization test."
    )
    print(
        "leave_one_condition_out: train and test share the same series_id and dT; "
        "use as an interpolation/diagnostic test, not as the main generalization test."
    )

    print(f"\nSaved diagnostics to: {output_path}")
    print(f"Saved summary to: {summary_path}")

    return results, summary


if __name__ == "__main__":
    run_validation_diagnostics()
