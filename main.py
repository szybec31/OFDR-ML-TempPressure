from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv, run_ablation_test

def print_and_save(avg_results, std_results, info):
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
        "Output_files/results_main.csv",
        index=False
    )

    print("\nResults saved to:")
    print("Output_files/results_main.csv")

if __name__ == "__main__":

    type = "run" # "prepare", "run", "ablations" lub "info"

    if type == "prepare":

        fix_dt16_folder_structure()

        df = build_dataframe()

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

        df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)

        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)

        df_base_for_training = df_summary[["series_id", "pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X",
                                           "irq_Y", "irq_X", "is_temp_calibration", "is_pressure_calibration",
                                           "is_joint_regression", "is_repeatability_test", "low_quality"]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'paired_features.csv'), index=False)

        print(quality_report(df_summary, 0.9))

        exit()

        df_train = build_stupid_dataset(df)

        df_train.to_csv("Output_files/stupid_dataset.csv", index=False)


    elif type == "run":
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        df = df_full[df_full["low_quality"] == False].copy()
        y = df[["pressure", "dT"]]
        groups = df["Tp"]

        features = ["mu_Y", "mu_X", "std_Y", "std_X", "irq_Y", "irq_X"]
        models = ["AN-BL", "MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF"]
        avg_results, std_results, avg_results_wo_f1, std_results_wo_f1 = run_cv(
            df=df,
            y=y,
            models=models,
            df_value=features,
            groups=groups
        )

        print_and_save(avg_results, std_results, "All Results:")
        print_and_save(avg_results_wo_f1, std_results_wo_f1, "Results without fold 1:")

    elif type == "ablations":
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        df_full["diff_XY"] = df_full["mu_X"] - df_full["mu_Y"]
        df_full["mean_XY"] = (df_full["mu_X"] + df_full["mu_Y"]) / 2.0

        ablation_results = {}

        # Ablacja 1: Cechy bazowe (4) vs Rozszerzone (8)
        df_clean = df_full[df_full["low_quality"] == False].copy()
        feat_4 = ["mu_X", "mu_Y", "std_X", "std_Y"]
        feat_8 = ["mu_X", "mu_Y", "std_X", "std_Y", "irq_X", "irq_Y", "diff_XY", "mean_XY"]

        ablation_results["A1_4_Features"] = run_ablation_test("4 Features", df_clean, feat_4)
        ablation_results["A1_8_Features"] = run_ablation_test("8 Features", df_clean, feat_8)

        # Ablacja 2: Jeden kanał vs Dwa kanały
        feat_x = ["mu_X", "std_X", "irq_X"]
        feat_y = ["mu_Y", "std_Y", "irq_Y"]
        feat_xy = ["mu_X", "mu_Y", "std_X", "std_Y", "irq_X", "irq_Y", "diff_XY", "mean_XY"]

        ablation_results["A2_X_Only"] = run_ablation_test("X Channel Only", df_clean, feat_x)
        ablation_results["A2_Y_Only"] = run_ablation_test("Y Channel Only", df_clean, feat_y)
        ablation_results["A2_XY_Full"] = ablation_results["A1_8_Features"]

        # Ablacja 3: Wpływ korekty etykiet
        # Symulacja błędu etykiet: 10 MPa -> 11 MPa i 0 MPa -> 0.01 MPa
        ablation_results["A3_Bad_Labels"] = run_ablation_test(
            "Bad Labels (11MPa/0.01MPa)",
            df_clean,
            feat_8,
            corrupt_train_labels=True
        )

        # Ablacja 4: Wpływ zero_end w treningu
        ablation_results["A4_With_Zero_End"] = run_ablation_test(
            "Including Zero_End",
            df_clean,
            feat_8,
            include_zero_end_train=True
        )

        print("\n" + "=" * 80)
        print(f"{'Ablation scenario':<35} | {'P_MAE':<10} | {'T_MAE':<10} | {'P_R2':<10}")
        print("-" * 80)

        for name, res in ablation_results.items():
            print(f"{name:<35} | {res['pressure_mae']:<10.3f} | {res['dT_mae']:<10.3f} | {res['pressure_r2']:<10.3f}")
        print("=" * 80)

    elif type == "info":
        output_dir = 'Output_files'
        df = pd.read_csv(os.path.join(output_dir, 'paired_features.csv'))

        print(f"is_temp_calibration: {df["is_temp_calibration"].sum()}")
        print(f"is_pressure_calibration: {df["is_pressure_calibration"].sum()}")
        print(f"is_joint_regression: {df["is_joint_regression"].sum()}")
        print(f"is_repeatability_test: {df["is_repeatability_test"].sum()}")

        print("---------- Info ----------")
        print(df.info())
        
        print(f"pressure: {df["pressure"].unique()}")
