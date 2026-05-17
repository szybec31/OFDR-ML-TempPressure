from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv, run_ablation_test

def print_and_save(avg_results, std_results, info, filename):
    # ==========================================
    # Wyświetlenie wyników w konsoli
    # ==========================================

    metrics = list(avg_results[0].keys())

    print("\n" + info)
    print("\n" + "=" * 190)

    header = f"{'Model':<25} | " + " | ".join(
        [f"{m.upper():<15}" for m in metrics]
    )
    print(header)

    print("-" * 190)

    for i, (avg, std) in enumerate(zip(avg_results, std_results)):
        metric_values = " | ".join(
            [f"{avg[m]:.3f} ({std[m]:.3f})".ljust(15) for m in metrics]
        )

        print(f"{models[i]:<25} | {metric_values}")

    print("=" * 190)

    # ==========================================
    # Zapis wyników do results_main.csv
    # ==========================================

    rows = []

    for i, (avg, std) in enumerate(zip(avg_results, std_results)):

        row = {"Model": models[i]}

        for metric in avg.keys():
            row[metric.upper()] = f"{avg[metric]:.3f} ({std[metric]:.3f})"

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join("Output_files", filename),
        index=False
    )

    print("\nResults saved to:")
    print(os.path.join("Output_files", filename))

def save_gain_vs_baseline(avg_results, models, filename):

    rows = []

    baseline_idx = models.index("AN-BL")
    baseline = avg_results[baseline_idx]

    for model_name, model_result in zip(models, avg_results):

        if model_name == "AN-BL":
            continue

        g_pressure = (
            (baseline["pressure_rmse"] - model_result["pressure_rmse"])
            / baseline["pressure_rmse"]
        )

        g_dT = (
            (baseline["dT_rmse"] - model_result["dT_rmse"])
            / baseline["dT_rmse"]
        )

        rows.append({
            "Model": model_name,
            "G_pressure_rmse": g_pressure,
            "G_pressure_rmse_percent": 100 * g_pressure,
            "G_dT_rmse": g_dT,
            "G_dT_rmse_percent": 100 * g_dT
        })

    df_gain = pd.DataFrame(rows)
    df_gain.to_csv(os.path.join("Output_files", filename), index=False)

    print("\nGain vs baseline saved to:")
    print(os.path.join("Output_files", filename))

def print_ablation_test(ablation_results):
    print("\n" + "=" * 150)
    print(
        f"{'Ablation scenario':<35} | "
        f"{'P_MAE (std)':<20} | "
        f"{'T_MAE (std)':<20} | "
        f"{'P_R2 (std)':<20} | "
        f"{'RD_MEAN (std)':<20} | "
        f"{'RD_MAX (std)':<20}"
    )
    print("-" * 150)

    rows = []

    for name, res in ablation_results.items():
        avg, std = res[0], res[1]

        p_mae = f"{avg['pressure_mae']:.3f} ({std['pressure_mae']:.3f})"
        t_mae = f"{avg['dT_mae']:.3f} ({std['dT_mae']:.3f})"
        p_r2 = f"{avg['pressure_r2']:.3f} ({std['pressure_r2']:.3f})"

        rd_mean_val = avg.get("rd_mean", float("nan"))
        rd_mean_std = std.get("rd_mean", float("nan"))
        rd_max_val = avg.get("rd_max", float("nan"))
        rd_max_std = std.get("rd_max", float("nan"))

        rd_mean = f"{rd_mean_val:.3f} ({rd_mean_std:.3f})"
        rd_max = f"{rd_max_val:.3f} ({rd_max_std:.3f})"

        print(
            f"{name:<35} | "
            f"{p_mae:<20} | "
            f"{t_mae:<20} | "
            f"{p_r2:<20} | "
            f"{rd_mean:<20} | "
            f"{rd_max:<20}"
        )

        rows.append({
            "Scenario": name,
            "pressure_mae": avg["pressure_mae"],
            "pressure_mae_std": std["pressure_mae"],
            "dT_mae": avg["dT_mae"],
            "dT_mae_std": std["dT_mae"],
            "pressure_r2": avg["pressure_r2"],
            "pressure_r2_std": std["pressure_r2"],
            "rd_mean": rd_mean_val,
            "rd_mean_std": rd_mean_std,
            "rd_max": rd_max_val,
            "rd_max_std": rd_max_std,
        })

    print("=" * 150)

    df_ablation = pd.DataFrame(rows)
    df_ablation.to_csv(
        os.path.join("Output_files", "results_ablations.csv"),
        index=False
    )

    print("\nAblation results saved to:")
    print(os.path.join("Output_files", "results_ablations.csv"))

if __name__ == "__main__":

    type = "info" # "prepare", "run", "ablations", "prepare_broken" lub "info"

    if type == "prepare":

        fix_dt16_folder_structure()

        df = build_dataframe()

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

        df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)

        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)

        df_base_for_training = df_summary[["series_id", "point_type", "pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X",
                                           "iqr_Y", "iqr_X", "Xinter", "Pdir", "diff_XY", "mean_XY", "is_temp_calibration", "is_pressure_calibration",
                                           "is_joint_regression", "is_repeatability_test", "low_quality","zero_end_shift_X", "zero_end_shift_Y"]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'paired_features.csv'), index=False)

        print(quality_report(df_summary, 0.9))

        exit()

        df_train = build_stupid_dataset(df)

        df_train.to_csv("Output_files/stupid_dataset.csv", index=False)

    elif type == "prepare_broken":
        df = build_dataframe(broken_data=True)

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

        df.to_csv(os.path.join(output_dir, 'inventory_broken.csv'), index=False)

        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset_broken.csv'), index=False)

        df_base_for_training = df_summary[["series_id", "point_type", "pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X",
                                           "iqr_Y", "iqr_X", "Xinter", "Pdir", "diff_XY", "mean_XY", "is_temp_calibration", "is_pressure_calibration",
                                           "is_joint_regression", "is_repeatability_test", "low_quality"]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'paired_features_broken.csv'), index=False)

    elif type == "run":
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        df = df_full[df_full["low_quality"] == False].copy()
        y = df[["pressure", "dT"]]
        groups = df["dT"]

        features = ["mu_Y", "mu_X", "diff_XY", "mean_XY", "std_Y", "std_X", "iqr_Y", "iqr_X", "Xinter", "Pdir"]
        models = ["AN-BL", "MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF"]
        avg_results, std_results, avg_results_wo_f1, std_results_wo_f1 = run_cv(
            df=df,
            y=y,
            models=models,
            df_value=features,
            groups=groups
        )

        print_and_save(avg_results, std_results, "All Results:", "results_main.csv")
        print_and_save(avg_results_wo_f1, std_results_wo_f1, "Results without fold 1:","results_main_without_fold1.csv")

        save_gain_vs_baseline(avg_results, models, "ml_vs_baseline_gain.csv")
        save_gain_vs_baseline(avg_results_wo_f1, models, "ml_vs_baseline_gain_without_fold1.csv")

    elif type == "ablations":
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')
        file_path_broken = os.path.join(output_dir, 'paired_features_broken.csv')

        df_full = pd.read_csv(file_path)
        df_broken = pd.read_csv(file_path_broken)
        df_full["diff_XY"] = df_full["mu_X"] - df_full["mu_Y"]
        df_full["mean_XY"] = (df_full["mu_X"] + df_full["mu_Y"]) / 2.0
        df_broken["diff_XY"] = df_broken["mu_X"] - df_broken["mu_Y"]
        df_broken["mean_XY"] = (df_broken["mu_X"] + df_broken["mu_Y"]) / 2.0

        ablation_results = {}

        # Ablacja 1: Cechy bazowe (4) vs Rozszerzone (8)
        df_clean = df_full[df_full["low_quality"] == False].copy()
        df_cleaner = df_clean[df_clean["is_repeatability_test"] == False].copy()
        df_broken = df_broken[(df_broken["low_quality"] == False)].copy()
        feat_4 = ["mu_X", "mu_Y", "diff_XY", "mean_XY"]
        feat_8 = ["mu_X", "mu_Y", "diff_XY", "mean_XY", "std_X", "std_Y", "iqr_X", "iqr_Y"]

        ablation_results["A1_4_Features"] = run_ablation_test("4 Features", df_cleaner, feat_4)
        ablation_results["A1_8_Features"] = run_ablation_test("8 Features", df_cleaner, feat_8)

        # Ablacja 2: Jeden kanał vs Dwa kanały
        feat_x = ["mu_X", "std_X", "iqr_X"]
        feat_y = ["mu_Y", "std_Y", "iqr_Y"]
        feat_xy = ["mu_X", "mu_Y", "std_X", "std_Y", "iqr_X", "iqr_Y", "diff_XY", "mean_XY"]

        ablation_results["A2_X_Only"] = run_ablation_test("X Channel Only", df_cleaner, feat_x)
        ablation_results["A2_Y_Only"] = run_ablation_test("Y Channel Only", df_cleaner, feat_y)
        ablation_results["A2_XY_Full"] = ablation_results["A1_8_Features"]

        # Ablacja 3: Wpływ korekty etykiet
        # Symulacja błędu etykiet: 10 MPa -> 11 MPa i 0 MPa -> 0.01 MPa
        ablation_results["A3_Bad_Labels"] = run_ablation_test(
            "Bad Labels (11MPa/0.01MPa)",
            df_broken,
            feat_8,
        )

        # Ablacja 4: Wpływ zero_end w treningu
        ablation_results["A4_With_Zero_End"] = run_ablation_test(
            "Including Zero_End",
            df_clean,
            feat_8,
            include_zero_end_train=True
        )

        print_ablation_test(ablation_results)

    elif type == "info":
        output_dir = 'Output_files'
        df = pd.read_csv(os.path.join(output_dir, 'paired_features.csv'))

        print(f"is_temp_calibration: {df['is_temp_calibration'].sum()}")
        print(f"is_pressure_calibration: {df['is_pressure_calibration'].sum()}")
        print(f"is_joint_regression: {df['is_joint_regression'].sum()}")
        print(f"is_repeatability_test: {df['is_repeatability_test'].sum()}")

        print("zero_end in joint_regression:")
        print(((df["is_repeatability_test"] == True) & (df["is_joint_regression"] == True)).sum())

        print("---------- Info ----------")
        print(df.info())
        
        print(f"pressure: {df['pressure'].unique()}")
        print(f"columns: {df.columns.tolist()}")
