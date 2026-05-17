from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
import sys
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv, run_ablation_test

def print_and_save(models, avg_results, std_results, info):
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

def print_ablation_test(ablation_results):
    print("\n" + "=" * 110)
    print(
        f"{'Ablation scenario':<35} | "
        f"{'P_MAE (std)':<20} | "
        f"{'T_MAE (std)':<20} | "
        f"{'P_R2 (std)':<20}"
    )
    print("-" * 110)

    for name, res in ablation_results.items():
        p_mae = f"{res[0]['pressure_mae']:.3f} ({res[1]['pressure_mae']:.3f})"
        t_mae = f"{res[0]['dT_mae']:.3f} ({res[1]['dT_mae']:.3f})"
        p_r2 = f"{res[0]['pressure_r2']:.3f} ({res[1]['pressure_r2']:.3f})"

        print(
            f"{name:<35} | "
            f"{p_mae:<20} | "
            f"{t_mae:<20} | "
            f"{p_r2:<20}"
        )

    print("=" * 110)

def main(type = "prepare", broken = False):

    if type in ["prepare", "p"]:

        fix_dt16_folder_structure()

        df = build_dataframe(broken_data = broken)

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

        df.to_csv(os.path.join(output_dir, 'inventory.csv' if not broken else 'inventory_broken.csv'), index=False)

        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset.csv' if not broken else 'training_dataset_broken.csv'), index=False)

        df_base_for_training = df_summary[["series_id", "pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X",
                                           "irq_Y", "irq_X", "diff_XY", "mean_XY",  "is_temp_calibration", "is_pressure_calibration",
                                           "is_joint_regression", "is_repeatability_test", "low_quality"]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'paired_features.csv' if not broken else 'paired_features_broken.csv'), index=False)

        print(quality_report(df_summary, 0.9))

    elif type in ["run", "r"]:
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        df = df_full[df_full["low_quality"] == False].copy()
        y = df[["pressure", "dT"]]
        groups = df["dT"]

        features = ["mu_Y", "mu_X", "std_Y", "std_X", "irq_Y", "irq_X"]
        models = ["AN-BL", "MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF"]
        avg_results, std_results, avg_results_wo_f1, std_results_wo_f1 = run_cv(
            df=df,
            y=y,
            models=models,
            df_value=features,
            groups=groups
        )

        print_and_save(models, avg_results, std_results, "All Results:")
        print_and_save(models, avg_results_wo_f1, std_results_wo_f1, "Results without fold 1:")

    elif type in ["ablations", "a"]:
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        
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
        df_broken = pd.read_csv(os.path.join(output_dir, 'paired_features_broken.csv'))
        df_broken = df_broken[(df_broken["low_quality"] == False)].copy()
        ablation_results["A3_Bad_Labels"] = run_ablation_test(
            "Bad Labels (11MPa/0.01MPa)",
            df_broken,
            feat_8,
            include_zero_end_train=True
        )

        # Ablacja 4: Wpływ zero_end w treningu
        ablation_results["A4_With_Zero_End"] = run_ablation_test(
            "Including Zero_End",
            df_clean,
            feat_8,
            include_zero_end_train=True
        )

        print_ablation_test(ablation_results)

    elif type in ["info", "i"]:
        output_dir = 'Output_files'
        df = pd.read_csv(os.path.join(output_dir, 'paired_features.csv'))

        print(f"is_temp_calibration: {df["is_temp_calibration"].sum()}")
        print(f"is_pressure_calibration: {df["is_pressure_calibration"].sum()}")
        print(f"is_joint_regression: {df["is_joint_regression"].sum()}")
        print(f"is_repeatability_test: {df["is_repeatability_test"].sum()}")

        print("---------- Info ----------")
        print(df.info())
        
        print(f"pressure: {df["pressure"].unique()}")

if __name__ == "__main__":
    # type = "info" # "prepare", "run", "ablations" lub "info"
    # broken = False
    # main(type, broken)
    # exit()

    argv = sys.argv
    argv.pop(0)

    print(len(argv))

    while(len(argv) >= 1):
        arg = argv.pop(0)
        if arg in ["setup", "s"]:
            main("p")
            main("p", True)
            main("i")
            break
        else:
            if arg in ["prepare_broken", "pb"]:
                main("p", True)
            else:
                main(arg)
    
    