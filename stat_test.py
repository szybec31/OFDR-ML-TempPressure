import sys
import os
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


ALPHA = 0.05
DEFAULT_CSV_PATH = "Output_files/folds.csv"
DEFAULT_OUTPUT_PATH = "Output_files/wilcoxon_results.csv"

BASELINE_MODEL = "AN-BL"

DEFAULT_METRICS = [
    "pressure_rmse",
    "dT_rmse",
]


METRIC_ALIASES = {
    "pressure_mae": "pressure_mae",
    "pressure_rmse": "pressure_rmse",
    "pressure_maxae": "pressure_maxae",
    "pressure_r2": "pressure_r2",
    "dt_mae": "dT_mae",
    "dt_rmse": "dT_rmse",
    "dt_maxae": "dT_maxae",
    "dt_r2": "dT_r2",
}


def normalize_metric(metric_arg: str) -> str:
    metric_key = metric_arg.lower()

    if metric_key not in METRIC_ALIASES:
        raise ValueError(
            f"Unknown metric: {metric_arg}. "
            f"Allowed metrics: {list(METRIC_ALIASES.keys())}"
        )

    return METRIC_ALIASES[metric_key]


def load_results(csv_path: str, local_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if local_only:
        if "local" not in df.columns:
            raise ValueError("Column 'local' not found in folds.csv.")

        df = df[df["local"] == True].copy()

    return df


def select_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    selected = df[df["model"] == model_name].copy()

    if selected.empty:
        raise ValueError(f"Model not found in folds.csv: {model_name}")

    return selected.sort_values("fold")


def prepare_paired_samples(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str
):
    merged = pd.merge(
        df_a[["fold", metric]],
        df_b[["fold", metric]],
        on="fold",
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        raise ValueError("No shared folds found between compared models.")

    scores_a = merged[f"{metric}_a"].to_numpy()
    scores_b = merged[f"{metric}_b"].to_numpy()

    return merged, scores_a, scores_b


def run_wilcoxon(scores_a, scores_b):
    differences = scores_a - scores_b

    if np.allclose(differences, 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
        }

    statistic, p_value = wilcoxon(scores_a, scores_b)

    return {
        "statistic": statistic,
        "p_value": p_value,
    }


def compare_models(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
):
    df_a = select_model(df, model_a)
    df_b = select_model(df, model_b)

    merged, scores_a, scores_b = prepare_paired_samples(
        df_a,
        df_b,
        metric,
    )

    test_result = run_wilcoxon(scores_a, scores_b)

    mean_a = float(np.mean(scores_a))
    std_a = float(np.std(scores_a, ddof=1))

    mean_b = float(np.mean(scores_b))
    std_b = float(np.std(scores_b, ddof=1))

    mean_diff = mean_a - mean_b

    if mean_a != 0:
        relative_gain_percent = 100.0 * mean_diff / mean_a
    else:
        relative_gain_percent = np.nan

    if mean_b < mean_a:
        better_model = model_b
    elif mean_b > mean_a:
        better_model = model_a
    else:
        better_model = "tie"

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_folds": len(merged),
        "model_a_mean": mean_a,
        "model_a_std": std_a,
        "model_b_mean": mean_b,
        "model_b_std": std_b,
        "mean_diff_a_minus_b": mean_diff,
        "relative_gain_percent_b_vs_a": relative_gain_percent,
        "better_model_by_mean": better_model,
        "test": "Wilcoxon signed-rank test",
        "statistic": float(test_result["statistic"]),
        "p_value": float(test_result["p_value"]),
        "significant_alpha_0_05": bool(test_result["p_value"] < ALPHA),
    }


def run_all_wilcoxon(
    csv_path: str = DEFAULT_CSV_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    local_only: bool = True,
):
    df = load_results(csv_path, local_only=local_only)

    models = sorted(df["model"].unique().tolist())

    if BASELINE_MODEL not in models:
        raise ValueError(f"Baseline model not found: {BASELINE_MODEL}")

    compared_models = [m for m in models if m != BASELINE_MODEL]

    rows = []

    for metric in DEFAULT_METRICS:
        for model in compared_models:
            row = compare_models(
                df=df,
                model_a=BASELINE_MODEL,
                model_b=model,
                metric=metric,
            )
            rows.append(row)

    results_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nWILCOXON TESTS VS BASELINE")
    print("=" * 100)
    print(f"Input file: {csv_path}")
    print(f"Baseline: {BASELINE_MODEL}")
    print(f"Local only: {local_only}")
    print(f"Output file: {output_path}")
    print("=" * 100)

    display_cols = [
        "metric",
        "model_a",
        "model_b",
        "n_folds",
        "model_a_mean",
        "model_a_std",
        "model_b_mean",
        "model_b_std",
        "relative_gain_percent_b_vs_a",
        "p_value",
        "significant_alpha_0_05",
        "better_model_by_mean",
    ]

    print(results_df[display_cols].to_string(index=False))

    return results_df


def run_single_comparison(argv):
    if len(argv) not in [3, 4]:
        raise ValueError(
            "Use: python stat_test.py MODEL_A MODEL_B METRIC [local|full]\n"
            "Example: python stat_test.py AN-BL MO-LR PRESSURE_RMSE local"
        )

    model_a = argv[0].upper()
    model_b = argv[1].upper()
    metric = normalize_metric(argv[2])

    local_only = True
    if len(argv) == 4:
        mode = argv[3].lower()
        if mode == "local":
            local_only = True
        elif mode == "full":
            local_only = False
        else:
            raise ValueError("Fourth argument must be 'local' or 'full'.")

    df = load_results(DEFAULT_CSV_PATH, local_only=local_only)

    result = compare_models(
        df=df,
        model_a=model_a,
        model_b=model_b,
        metric=metric,
    )

    print("\nWILCOXON MODEL COMPARISON")
    print("=" * 80)
    print(f"Input file: {DEFAULT_CSV_PATH}")
    print(f"Local only: {local_only}")
    print(f"Model A: {model_a}")
    print(f"Model B: {model_b}")
    print(f"Metric: {metric}")
    print("-" * 80)

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) == 1 and argv[0].lower() == "help":
        print("Usage:")
        print("  python stat_test.py all")
        print("  python stat_test.py all_full")
        print("  python stat_test.py MODEL_A MODEL_B METRIC [local|full]")
        print("")
        print("Examples:")
        print("  python stat_test.py all")
        print("  python stat_test.py all_full")
        print("  python stat_test.py AN-BL MO-LR PRESSURE_RMSE local")
        print("  python stat_test.py AN-BL POLY2-RIDGE DT_RMSE local")
        exit()

    if len(argv) == 1 and argv[0].lower() == "all":
        run_all_wilcoxon(local_only=True)

    elif len(argv) == 1 and argv[0].lower() == "all_full":
        run_all_wilcoxon(
            output_path="Output_files/wilcoxon_results_all_folds.csv",
            local_only=False,
        )

    else:
        run_single_comparison(argv)