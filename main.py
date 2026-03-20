from build_df import build_dataframe
import pandas as pd

df = build_dataframe()
df.to_csv('Output_files/inventory.csv', index=False)