import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "Output_files/folds.csv"
OUTPUT_DIR = "Output_files/plots"

LOCAL_ONLY = True

MODEL_ORDER = [
    "AN-BL",
    "MO-LR",
    "POLY2-RIDGE",
    "SVR-RBF",
    "NYSTROEM-SVR",
    "KRR-RBF",
    "GPR",
    "RF",
    "HGBR",
    "GBR",
]

METRICS = {
    "pressure_rmse": "Pressure RMSE",
    "dT_rmse": "Temperature RMSE",
    "pressure_mae": "Pressure MAE",
    "dT_mae": "Temperature MAE",
}


def load_data():
    df = pd.read_csv(INPUT_PATH)

    if LOCAL_ONLY:
        df = df[df["local"] == True].copy()

    return df


def build_summary_table(df):
    rows = []

    for model in MODEL_ORDER:
        df_model = df[df["model"] == model]

        if df_model.empty:
            continue

        row = {"model": model}

        for metric in METRICS:
            mean_value = df_model[metric].mean()
            std_value = df_model[metric].std(ddof=1)

            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            row[f"{metric}_formatted"] = f"{mean_value:.3f} ({std_value:.3f})"

        rows.append(row)

    return pd.DataFrame(rows)


def save_summary_tables(summary):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    numeric_path = os.path.join(OUTPUT_DIR, "model_results_summary_numeric.csv")
    formatted_path = os.path.join(OUTPUT_DIR, "model_results_summary_formatted.csv")

    summary.to_csv(numeric_path, index=False)

    formatted_cols = ["model"] + [f"{metric}_formatted" for metric in METRICS]
    formatted = summary[formatted_cols].copy()

    formatted = formatted.rename(columns={
        "pressure_rmse_formatted": "PRESSURE_RMSE",
        "dT_rmse_formatted": "DT_RMSE",
        "pressure_mae_formatted": "PRESSURE_MAE",
        "dT_mae_formatted": "DT_MAE",
    })

    formatted.to_csv(formatted_path, index=False)

    print(f"Saved numeric summary to: {numeric_path}")
    print(f"Saved formatted summary to: {formatted_path}")


def plot_metric(summary, metric, title, output_filename):
    means = summary[f"{metric}_mean"]
    stds = summary[f"{metric}_std"]
    models = summary["model"]

    plt.figure(figsize=(10, 6))
    plt.barh(models, means, xerr=stds, capsize=4)
    plt.xlabel(title)
    plt.ylabel("Model")
    plt.title(f"{title} by model, mean ± std")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    summary = build_summary_table(df)

    save_summary_tables(summary)

    plot_metric(
        summary,
        metric="pressure_rmse",
        title="Pressure RMSE",
        output_filename="pressure_rmse_comparison.png",
    )

    plot_metric(
        summary,
        metric="dT_rmse",
        title="Temperature RMSE",
        output_filename="dt_rmse_comparison.png",
    )

    plot_metric(
        summary,
        metric="pressure_mae",
        title="Pressure MAE",
        output_filename="pressure_mae_comparison.png",
    )

    plot_metric(
        summary,
        metric="dT_mae",
        title="Temperature MAE",
        output_filename="dt_mae_comparison.png",
    )

    print("\nSUMMARY TABLE, mean (std):")
    formatted_cols = [
        "model",
        "pressure_rmse_formatted",
        "dT_rmse_formatted",
        "pressure_mae_formatted",
        "dT_mae_formatted",
    ]
    print(summary[formatted_cols].to_string(index=False))


if __name__ == "__main__":
    main()