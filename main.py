from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv

if __name__ == "__main__":

    type = "run"  # "prepare" lub "run"

    if type == "prepare":
        fix_dt16_folder_structure()
        df = build_dataframe()

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)

        df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)
        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)

        # UWAGA: Dodano "series_id" oraz "role", aby run_cv wiedział jak dzielić dane i co filtrować
        df_base_for_training = df_summary[[
            "pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X",
            "irq_Y", "irq_X", "role", "low_quality", "series_id"
        ]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)

        print(quality_report(df_summary, 0.9))
        exit()

        df_train = build_stupid_dataset(df)

        df_train.to_csv("Output_files/stupid_dataset.csv", index=False)

    elif type == "run":
        output_dir = 'Output_files'
        df_all = pd.read_csv(os.path.join(output_dir, 'training_dataset.csv'))

        # Zgodnie z instrukcją: Używamy tylko danych joint_regression i odrzucamy low_quality [cite: 156]
        # Bezwzględnie odrzucamy zero_end z treningu głównego [cite: 170]
        df = df_all[df_all["role"].str.contains("joint_regression") & (df_all["low_quality"] == False)].copy()

        # Definicja etykiet (y) i cech (X)
        y = df[["pressure", "dT"]]
        X = df.drop(columns=["pressure", "dT", "Tp", "role", "low_quality", "series_id"])

        # Wyciągamy grupy potrzebne do Nested CV [cite: 160, 161]
        #groups_outer = df["series_id"]  # Dla testów
        groups_outer = df["dT"]  # Outer: Leave-One-Temperature-Level-Out
        groups_inner = df["series_id"]  # Inner: GroupKFold po seriach

        models = ["AN-BL", "MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF"]
        # Wywołujemy CV z nowymi parametrami grup
        avg_results, std_results = run_cv(X, y, groups_outer, groups_inner, models=models, df_value=[
            "mu_Y",
            "mu_X",
            "std_Y",
            "std_X",
            "irq_Y",
            "irq_X",
        ])

        print("\nNested leave-one-temperature-level-out results")
        for i, (avg, std) in enumerate(zip(avg_results, std_results)):
            print(f"\n\nModel {models[i]}:\n")
            for metric in avg.keys():
                print(f"{metric.upper():<10} {avg[metric]:.6f} ± {std[metric]:.6f}")


