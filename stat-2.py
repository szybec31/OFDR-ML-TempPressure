import pandas as pd
import numpy as np
import sys

from scipy.stats import (
    shapiro,
    ttest_rel,
    wilcoxon
)

# =========================================================
# KONFIGURACJA
# =========================================================

CSV_PATH = "Output_files/folds.csv"
ALPHA = 0.05

# Dostępne metryki:
# f1_micro
# f1_macro
# b1
# recall_micro
# hamming
# avg_labels_true
# avg_labels_pred


# =========================================================
# WCZYTANIE DANYCH
# =========================================================

def load_results(csv_path: str, local_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if local_only:
        return df[df["local"]]
    return df


# =========================================================
# WYBÓR MODELU
# =========================================================

def select_model(
    df: pd.DataFrame,
    model_config: dict
) -> pd.DataFrame:

    filtered = df.copy()

    for key, value in model_config.items():
        filtered = filtered[filtered[key] == value]

    return filtered.sort_values("fold")


# =========================================================
# PRZYGOTOWANIE PAR
# =========================================================

def prepare_paired_samples(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str
):

    merged = pd.merge(
        df_a[["fold", metric]],
        df_b[["fold", metric]],
        on="fold",
        suffixes=("_A", "_B")
    )

    if len(merged) == 0:
        raise ValueError(
            f"Oczekiwano X foldów, znaleziono {len(merged)}"
        )

    scores_a = merged[f"{metric}_A"].values
    scores_b = merged[f"{metric}_B"].values

    return scores_a, scores_b


# =========================================================
# TEST NORMALNOŚCI
# =========================================================

def check_normality(
    scores_a,
    scores_b
):

    differences = scores_a - scores_b

    stat, p_value = shapiro(differences)

    return {
        "statistic": stat,
        "p_value": p_value,
        "normal": p_value > ALPHA
    }


# =========================================================
# EFFECT SIZE
# =========================================================

def cohens_d_paired(scores_a, scores_b):

    diff = scores_a - scores_b

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    return mean_diff / std_diff


# =========================================================
# TEST STATYSTYCZNY
# =========================================================

def run_statistical_test(
    scores_a,
    scores_b,
    used_test: str | None = None
):

    normality = check_normality(scores_a, scores_b)

    if (normality["normal"] and used_test == None) or used_test == "t-test":

        test_name = "Paired t-test"

        stat, p_value = ttest_rel(scores_a, scores_b)

        effect_size = cohens_d_paired(scores_a, scores_b)

    elif used_test == "wilcoxon" or (not normality["normal"] and used_test == None):

        test_name = "Wilcoxon signed-rank test"

        stat, p_value = wilcoxon(scores_a, scores_b)

        effect_size = None

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "normality": normality
    }


# =========================================================
# RAPORT
# =========================================================

def print_report(
    model_a,
    model_b,
    metric,
    scores_a,
    scores_b,
    results
):

    print("\n" + "=" * 60)
    print("PORÓWNANIE MODELI")
    print("=" * 60)

    print("\nMODEL A")
    for k, v in model_a.items():
        print(f"{k}: {v}")

    print("\nMODEL B")
    for k, v in model_b.items():
        print(f"{k}: {v}")

    print("\nMETRYKA:", metric)

    print("\nWYNIKI FOLDÓW")
    for i, (a, b) in enumerate(zip(scores_a, scores_b), start=1):
        print(f"Fold {i}: A={a:.4f} | B={b:.4f}")

    print("\n" + "-" * 60)
    print("TEST NORMALNOŚCI (Shapiro-Wilk)")
    print("-" * 60)

    print(
        f"Statistic = {results['normality']['statistic']:.6f}"
    )
    print(
        f"p-value   = {results['normality']['p_value']:.6f}"
    )

    if results["normality"]["normal"]:
        print("Wniosek: rozkład różnic jest normalny")
    else:
        print("Wniosek: brak normalności rozkładu różnic")

    print("\n" + "-" * 60)
    print("TEST STATYSTYCZNY")
    print("-" * 60)

    print(f"Test: {results['test']}")
    print(f"Statistic = {results['statistic']:.6f}")
    print(f"p-value   = {results['p_value']:.6f}")

    if results["effect_size"] is not None:
        print(f"Cohen's d = {results['effect_size']:.6f}")

    print("\n" + "-" * 60)

    if results["p_value"] < ALPHA:
        print(
            f"WYNIK ISTOTNY STATYSTYCZNIE "
            f"(p < {ALPHA})"
        )
    else:
        print(
            f"BRAK ISTOTNOŚCI STATYSTYCZNEJ "
            f"(p >= {ALPHA})"
        )

    print("=" * 60)


# =========================================================
# GŁÓWNA FUNKCJA
# =========================================================

def compare_models(
    csv_path,
    model_a,
    model_b,
    metric,
    local_only,
    used_test
):

    df = load_results(csv_path, local_only)

    df_a = select_model(df, model_a)
    df_b = select_model(df, model_b)

    if df_a.empty:
        raise ValueError("Nie znaleziono Modelu A")

    if df_b.empty:
        raise ValueError("Nie znaleziono Modelu B")

    scores_a, scores_b = prepare_paired_samples(
        df_a,
        df_b,
        metric
    )

    results = run_statistical_test(
        scores_a,
        scores_b,
        used_test
    )

    print_report(
        model_a,
        model_b,
        metric,
        scores_a,
        scores_b,
        results
    )

    return results


def build_comparison_table(
    csv_path,
    models,
    metric: str = "f1_samples",
    alpha: float = 0.05,
    output_csv: str = "pairwise_comparison.csv",
    local_only: bool = True,
    used_test: str | None = "wilcoxon"
):

    df = load_results(csv_path)

    names = list(models.keys())

    wins = {n: 0 for n in names}
    losses = {n: 0 for n in names}
    equals = {n: 0 for n in names}

    matrix = pd.DataFrame(
        "-",
        index=names,
        columns=names
    )

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            name_a = names[i]
            name_b = names[j]

            result = compare_models(
                csv_path=csv_path,
                model_a=models[name_a],
                model_b=models[name_b],
                metric=metric,
                local_only=local_only,
                used_test=used_test
            )

            p_value = result["p_value"]

            scores_a, scores_b = prepare_paired_samples(
                select_model(df, models[name_a]),
                select_model(df, models[name_b]),
                metric
            )

            mean_a = scores_a.mean()
            mean_b = scores_b.mean()

            # ===================================
            # Significant difference
            # ===================================

            if p_value < alpha:

                if mean_a > mean_b:

                    matrix.loc[name_a, name_b] = "↑"
                    matrix.loc[name_b, name_a] = "↓"

                    wins[name_a] += 1
                    losses[name_b] += 1

                else:

                    matrix.loc[name_a, name_b] = "↓"
                    matrix.loc[name_b, name_a] = "↑"

                    wins[name_b] += 1
                    losses[name_a] += 1

            # ===================================
            # No significant difference
            # ===================================

            else:

                matrix.loc[name_a, name_b] = "="
                matrix.loc[name_b, name_a] = "="

                equals[name_a] += 1
                equals[name_b] += 1

    # ==========================================
    # Add summary columns
    # ==========================================

    matrix["Wins"] = [wins[n] for n in names]
    matrix["Loses"] = [losses[n] for n in names]
    matrix["Draws"] = [equals[n] for n in names]
    matrix["Score"] = matrix["Wins"] - matrix["Loses"]

    # matrix = matrix.sort_values(
    #     by=["Score", "W"],
    #     ascending=False
    # )

    matrix.to_csv(output_csv)

    print(f"\nSaved comparison table to: {output_csv}")

    return matrix

# ==========================================
# MODEL NAMES FOR TABLE
# ==========================================

MODELS = {
    "AN-BL": {
        "model": "AN-BL"
    },
    "MO-LR": {
        "model": "MO-LR"
    },
    "MO-LR": {
        "model": "MO-LR"
    },
    "RF": {
        "model": "RF"
    },
    "POLY2-RIDGE": {
        "model": "POLY2-RIDGE"
    },
    "SVR-RBF": {
        "model": "SVR-RBF"
    },
}



if __name__ == "__main__":
    argv = sys.argv
    argv.pop(0)

    metric_aliases = {
        "pressure_mae": "pressure_mae",
        "pressure_rmse": "pressure_rmse",
        "pressure_maxae": "pressure_maxae",
        "pressure_r2": "pressure_r2",
        "dt_mae": "dT_mae",
        "dt_rmse": "dT_rmse",
        "dt_maxae": "dT_maxae",
        "dt_r2": "dT_r2",
    }

    ## Info for running from console
    if len(argv) == 1 and argv[0] == "help":
        print("Use 3+ argv: ")
        print("1st and 2nd argv must be values from [AN-BL, MO-LR, RF, POLY2-RIDGE, SVR-RBF]")
        print("3rd argv must be values from [PRESSURE_MAE, PRESSURE_RMSE, PRESSURE_MAXAE, PRESSURE_R2, DT_MAE, DT_RMSE, DT_MAXAE, DT_R2]")
        print("4th (opt) You may choose 0 or 1 - to use only corrected folds")
        print("5th (opt) You may decide which test you choose (wilcoxon or t-test)")
        print("You may use lower or upper case..")
        print("Also for 2nd option you may use 2+ argv using all arv")
        print("1st argv: all - which mean you compare all models")
        print("2nd - metric")
        print("3rd (opt) - use corrected folds (0 or 1)")
        print("4th (opt) - stat test")
        exit()

    if not (len(argv) >= 3 or (len(argv) >= 2 and argv[0] == "all")):
        ### If you running code without console you must to edit line below to change function argv
        argv = [
            "MO-LR", # AN-BL, MO-LR, RF, POLY2-RIDGE, SVR-RBF
            "AN-BL", # AN-BL, MO-LR, RF, POLY2-RIDGE, SVR-RBF
            "PRESSURE_MAE" # PRESSURE_MAE, PRESSURE_RMSE, PRESSURE_MAXAE, PRESSURE_R2, DT_MAE, DT_RMSE, DT_MAXAE, DT_R2,
        ]

        # argv = ["all", "PRESSURE_MAE"] # optional: ["all", "PRESSURE_MAE", 1, "wilcoxon"]


    if argv[0] == "all":

        metric_arg = argv[1].lower()

        if metric_arg not in metric_aliases:
            raise ValueError(
                f"Unknown metric: {argv[1]}. "
                f"Allowed metrics: {list(metric_aliases.keys())}"
            )

        table = build_comparison_table(
            csv_path = CSV_PATH,
            models = MODELS,
            metric = metric_aliases[metric_arg],
            output_csv = f"Output_files/stats_results_{metric_aliases[metric_arg]}.csv",
            local_only = argv[2] if len(argv) > 2 else 1,
            used_test = argv[3] if len(argv) > 3 else None
        )

        print(f"Used test: {argv[3] if len(argv) > 3 else "Default"}")
        print(table)
        exit()


    model_A = {
        "model": argv[0].upper(), 
    }

    model_B = {
        "model": argv[1].upper(), 
    }

    metric_arg = argv[2].lower()

    if metric_arg not in metric_aliases:
        raise ValueError(
            f"Unknown metric: {argv[2]}. "
            f"Allowed metrics: {list(metric_aliases.keys())}"
        )

    metric = metric_aliases[metric_arg]

    compare_models(
        csv_path="Output_files/folds.csv",
        model_a=model_A,
        model_b=model_B,
        metric=metric,
        local_only = argv[3] if len(argv) > 3 else 1,
        used_test = argv[4] if len(argv) > 4 else None
    )