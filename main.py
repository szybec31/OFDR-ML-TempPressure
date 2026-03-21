from build_df import build_dataframe
import pandas as pd
import os

df = build_dataframe()

output_dir = 'Output_files'
os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

df.to_csv(os.path.join(output_dir, 'inventory.csv'), index=False)