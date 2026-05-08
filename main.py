from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv


if __name__ == "__main__":

    type = "run" # "prepare"

    if type == "prepare":

        fix_dt16_folder_structure()

        df = build_dataframe()

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

        df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)

        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)

        df_base_for_training = df_summary[["pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X", "irq_Y", "irq_X", "role", "low_quality"]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'paired_features.csv'), index=False)

        print(quality_report(df_summary, 0.9))

        exit()

        df_train = build_stupid_dataset(df)

        df_train.to_csv("Output_files/stupid_dataset.csv", index=False)

    elif type == "run":
        output_dir = 'Output_files'
        df = pd.read_csv(os.path.join(output_dir, 'paired_features.csv'))

        y = df[["pressure", "dT"]] # "dT, Tp"
        X = df.drop(columns = ["pressure", "dT", "Tp"])

        # ["MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF", "AN-BL", "MLP"]
        models = ["MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF"]
        avg_results, std_results = run_cv(X, y, 5, models = models, df_value = ["mu_Y", "mu_X"])

        print("\n===== CROSS VALIDATION RESULTS =====\n")

        for i, (avg, std) in enumerate(zip(avg_results, std_results)):
            print("-" * 30)
            print(f"Model {models[i]}")
            print("-" * 30)

            for metric in avg.keys():
                print(
                    f"{metric.upper():<6} "
                    f"{avg[metric]:.6f} "
                    f"± {std[metric]:.6f}"
                )