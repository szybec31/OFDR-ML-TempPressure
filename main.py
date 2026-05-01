from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary

if __name__ == "__main__":

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