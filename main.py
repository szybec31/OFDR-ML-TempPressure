from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv


if __name__ == "__main__":

    type = "run" # "prepare", "run" lub "info"

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
        avg_results, std_results = run_cv(
            df=df,
            y=y,
            models=models,
            df_value=features,
            groups=groups
        )
        print("\nResults for nested leave-one-temperature-level-out\n")
        for i, (avg, std) in enumerate(zip(avg_results, std_results)):
            print(f"\nModel: {models[i]}\n")

            for metric in avg.keys():
                print(f"{metric.upper():<6} {avg[metric]:.6f} ± {std[metric]:.6f}")

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