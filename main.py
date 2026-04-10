from build_df import build_dataframe
from build_global_db import build_training_dataset
import pandas as pd
import os
from utils import fix_dt16_folder_structure

fix_dt16_folder_structure()

df = build_dataframe()

output_dir = 'Output_files'
os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)

exit()

df_train = build_training_dataset(df)

df_train.to_csv("Output_files/training_dataset.csv", index=False)