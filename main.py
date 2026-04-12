from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure
from analyze_files import build_folder_summary

if __name__ == "__main__":

    fix_dt16_folder_structure()

    df = build_dataframe()

    output_dir = 'Output_files'
    os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

    df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)

    df_summary = build_folder_summary(df)

    df_summary[df_summary["ref"].notna()][["pressure", "dT", "Tp", "y_signal", "x_signal", "role", "low_quality"]].to_csv(os.path.join(output_dir, 'training_dataset.csv'), index=False)

    exit()

    df_train = build_stupid_dataset(df)

    df_train.to_csv("Output_files/stupid_dataset.csv", index=False)