import os

import numpy as np
import pandas as pd

from baselines.models.linear import train_linear
from baselines.models.poly2_ridge import train_poly2_ridge
from baselines.models.svr_rbf import train_svr_rbf


DEFAULT_FEATURES = [
    "mu_X",
    "mu_Y",
    "std_X",
    "std_Y",
    "irq_X",
    "irq_Y",
    "diff_XY",
    "mean_XY",
]


def _available_features(df, features):
    return [feature for feature in features if feature in df.columns]


def _train_prediction_models(train_df, features):
    """
    Trains diagnostic models only on valid joint_regression samples.
    zero_end samples are not used for training.
    """
    X_train = train_df[features]
    y_train = train_df[["pressure", "dT"]]

    models = {
        "MO-LR": train_linear(X_train, y_train),
        "POLY2-RIDGE": train_poly2_ridge(X_train, y_train, {}),
        "SVR-RBF": train_svr_rbf(X_train, y_train, {}),
    }

    return models


def _add_signal_drift_metrics(df):
    """
    Signal-level repeatability drift:
    delta_mu_X = mu_X(zero_end) - mu_X(zero_start)
    delta_mu_Y = mu_Y(zero_end) - mu_Y(zero_start)
    """
    df["delta_mu_X"] = df["mu_X_end"] - df["mu_X_start"]
    df["delta_mu_Y"] = df["mu_Y_end"] - df["mu_Y_start"]
    df["delta_mu_norm"] = np.sqrt(df["delta_mu_X"] ** 2 + df["delta_mu_Y"] ** 2)
    df["abs_delta_mu_X"] = df["delta_mu_X"].abs()
    df["abs_delta_mu_Y"] = df["delta_mu_Y"].abs()

    # Backward-compatible aliases used in the previous report version.
    # These describe signal drift, not prediction-level Residual Drift.
    df["RD_X"] = df["delta_mu_X"]
    df["RD_Y"] = df["delta_mu_Y"]
    df["RD_norm"] = df["delta_mu_norm"]
    df["abs_RD_X"] = df["abs_delta_mu_X"]
    df["abs_RD_Y"] = df["abs_delta_mu_Y"]

    return df


def _add_prediction_residual_drift(df, models, features):
    """
    Prediction-level Residual Drift:
    RD = sqrt((T_hat_end - T_start)^2 + (P_hat_end - P_start)^2)

    The model predicts pressure and dT for zero_end samples.
    The expected reference state is the paired zero_start sample.
    """
    feature_end_cols = [f"{feature}_end" for feature in features]

    X_end = df[feature_end_cols].copy()
    X_end.columns = features

    for model_name, model in models.items():
        pred = model.predict(X_end)

        pressure_pred = pred[:, 0]
        dT_pred = pred[:, 1]

        df[f"{model_name}_pressure_pred_end"] = pressure_pred
        df[f"{model_name}_dT_pred_end"] = dT_pred

        df[f"{model_name}_RD_pressure"] = pressure_pred - df["pressure_start"]
        df[f"{model_name}_RD_dT"] = dT_pred - df["dT_start"]

        df[f"{model_name}_prediction_RD"] = np.sqrt(
            df[f"{model_name}_RD_pressure"] ** 2
            + df[f"{model_name}_RD_dT"] ** 2
        )

    return df


def compute_repeatability_metrics(
    input_path="Output_files/training_dataset.csv",
    output_path="Output_files/repeatability_results.csv",
    signal_summary_path="Output_files/repeatability_signal_drift_by_dt.csv",
    prediction_summary_path="Output_files/repeatability_prediction_rd_by_dt.csv",
):
    df = pd.read_csv(input_path)

    features = _available_features(df, DEFAULT_FEATURES)

    zero_start = df[
        (df["is_temp_calibration"] == True)
        & (df["is_repeatability_test"] == False)
    ].copy()

    zero_end = df[df["is_repeatability_test"] == True].copy()

    print("zero_start all:", len(zero_start))
    print("zero_end all:", len(zero_end))
    print("features used:", features)

    start_cols = [
        "path_y",
        "pressure",
        "dT",
        "Tp",
        "series_id",
        "low_quality",
    ] + features

    merged = zero_end.merge(
        zero_start[start_cols],
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

    merged = _add_signal_drift_metrics(merged)
    good_pairs = _add_signal_drift_metrics(good_pairs)

    train_df = df[
        (df["is_joint_regression"] == True)
        & (df["is_repeatability_test"] == False)
        & (df["low_quality"] == False)
    ].copy()

    print("prediction train samples:", len(train_df))

    models = _train_prediction_models(train_df, features)
    good_pairs = _add_prediction_residual_drift(good_pairs, models, features)

    signal_cols = [
        "delta_mu_X",
        "delta_mu_Y",
        "delta_mu_norm",
        "abs_delta_mu_X",
        "abs_delta_mu_Y",
    ]

    prediction_rd_cols = [
        f"{model_name}_prediction_RD"
        for model_name in models.keys()
    ]

    signal_summary = good_pairs[signal_cols].describe()

    signal_by_dt = (
        good_pairs
        .groupby("dT_end")[signal_cols]
        .agg(["mean", "std", "max"])
    )

    prediction_summary = good_pairs[prediction_rd_cols].describe()

    prediction_by_dt = (
        good_pairs
        .groupby("dT_end")[prediction_rd_cols]
        .agg(["mean", "std", "max"])
    )

    print("\nSIGNAL DRIFT SUMMARY, good pairs only:")
    print(signal_summary)

    print("\nSIGNAL DRIFT BY dT, good pairs only:")
    print(signal_by_dt)

    print("\nPREDICTION RESIDUAL DRIFT SUMMARY, good pairs only:")
    print(prediction_summary)

    print("\nPREDICTION RESIDUAL DRIFT BY dT, good pairs only:")
    print(prediction_by_dt)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    good_pairs.to_csv(output_path, index=False)
    signal_by_dt.to_csv(signal_summary_path)
    prediction_by_dt.to_csv(prediction_summary_path)

    print(f"\nSaved repeatability results to: {output_path}")
    print(f"Saved signal drift summary to: {signal_summary_path}")
    print(f"Saved prediction RD summary to: {prediction_summary_path}")

    return (
        good_pairs,
        signal_summary,
        signal_by_dt,
        prediction_summary,
        prediction_by_dt,
    )


if __name__ == "__main__":
    compute_repeatability_metrics()