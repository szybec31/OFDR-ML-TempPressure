import pandas as pd
from tqdm import tqdm

def build_training_dataset(df):
    """
    Tworzy dataset do ML:
    length, y_value, y_ref_value, dT, Tp, pcm, role, x_value, x_ref_value
    """

    records = []
    file_cache = {}  # 🔥 cache

    def get_file(path):
        if path is None or pd.isna(path):
            return None

        if path not in file_cache:
            file_cache[path] = read_measurement_file(path)

        return file_cache[path]

    df_y = df[df["channel_norm"] == "Y"].copy()

    print(f"Processing {len(df_y)} Y-channel files...")

    for _, row in tqdm(df_y.iterrows(), total=len(df_y)):

        y_path = row["source_path"]
        x_path = row["pair_id"]
        y_ref_path = row["reference_zero_id"]

        if pd.isna(x_path):
            continue

        try:
            y_df = get_file(y_path)
            x_df = get_file(x_path)
            y_ref_df = get_file(y_ref_path)

            # X reference
            x_ref_df = None
            if pd.notna(y_ref_path):
                ref_row = df[df["source_path"] == y_ref_path]
                if not ref_row.empty:
                    x_ref_path = ref_row.iloc[0]["pair_id"]
                    x_ref_df = get_file(x_ref_path)

            # minimalna długość
            min_len = min(len(y_df), len(x_df))

            if y_ref_df is not None:
                min_len = min(min_len, len(y_ref_df))

            if x_ref_df is not None:
                min_len = min(min_len, len(x_ref_df))

            # zapis rekordów
            for i in range(min_len):
                records.append({
                    "length": y_df.iloc[i]["length"],
                    "y_value": y_df.iloc[i]["value"],
                    "x_value": x_df.iloc[i]["value"],
                    "y_ref_value": y_ref_df.iloc[i]["value"] if y_ref_df is not None else None,
                    "x_ref_value": x_ref_df.iloc[i]["value"] if x_ref_df is not None else None,
                    "dT": row["deltaT_label"],
                    "Tp": row["T_plate_label"],
                    "pcm": row["pressure_corr_mpa"],
                    "role": row["role"],
                })

        except Exception as e:
            print(f"\nError processing {y_path}: {e}")

    print(f"\nUnique files read: {len(file_cache)}")

    return pd.DataFrame(records)