import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from .run_experiment import run_experiment
from .utils.metrics import evaluate


def run_cv(df, y, models, df_value, groups, include_zero_end_train=False, prediction_file=False):
    logo = LeaveOneGroupOut()
    all_fold_results = {m: [] for m in models}
    all_y_true = []
    all_y_preds = {m: [] for m in models}

    # do zapisu do predictions.csv
    fold_metrics_rows = []
    prediction_rows = []
    fold_to_remove = []
    stored_fold_ids = []

    flags = ["is_temp_calibration", "is_pressure_calibration", "is_joint_regression", "is_repeatability_test"]
    X_full = df[df_value + flags + ["series_id"]]

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_full, y, groups=groups)):
        test_temp = groups.iloc[test_idx].unique()[0]
        print(f"\n>>> FOLD {fold + 1}: Test na dT = {test_temp}C")

        X_train_fold = X_full.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()

        X_test_fold = X_full.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        joint_mask = X_test_fold["is_joint_regression"] == True
        if not np.any(joint_mask):
            print(f"Pominięto fold - brak danych joint_regression")
            continue

        istc = sum(X_train_fold["is_temp_calibration"] == True)
        ispc = sum(X_train_fold["is_pressure_calibration"] == True)
        print("\n is_temp_calibration: ", istc)
        print("\n is_pressure_calibration: ", ispc)
        if ispc * istc == 0:
            fold_to_remove.append(fold)

        X_test_filtered = X_test_fold[joint_mask]
        y_test_filtered = y_test_fold[joint_mask]

        res_list, pred_list = run_experiment(
            X_train=X_train_fold,
            y_train=y_train_fold,
            X_test=X_test_filtered,
            y_test=y_test_filtered,
            models=models,
            include_zero_end_train=include_zero_end_train
        )

        stored_fold_ids.append(fold)

        all_y_true.append(y_test_filtered.values)
        for i, m_name in enumerate(models):
            print(res_list[i])
            all_fold_results[m_name].append(res_list[i])
            all_y_preds[m_name].append(pred_list[i])

            if prediction_file == True:
                if fold == 0:
                    continue
                # ==========================================
                # ZAPIS METRYK PER FOLD
                # ==========================================

                fold_row = {
                    "fold": fold + 1,
                    "test_dT": test_temp,
                    "model": m_name
                }

                fold_row.update(res_list[i])

                fold_metrics_rows.append(fold_row)

                # ==========================================
                # ZAPIS PREDYKCJI PER SAMPLE
                # ==========================================

                preds = pred_list[i]

                for j in range(len(preds)):
                    prediction_rows.append({
                        "fold": fold + 1,
                        "test_dT": test_temp,
                        "model": m_name,

                        "pressure_true":
                            y_test_filtered.iloc[j]["pressure"],

                        "pressure_pred":
                            preds[j, 0],

                        "dT_true":
                            y_test_filtered.iloc[j]["dT"],

                        "dT_pred":
                            preds[j, 1]
                    })

    final_avg = []
    final_std = []
    final_avg_without_fold_1 = []
    final_std_without_fold_1 = []

    if not all_y_true:
        print("Błąd: Nie zebrano żadnych wyników!")
        return [], []

    y_true_stacked = np.vstack(all_y_true)
    y_true_df = pd.DataFrame(y_true_stacked, columns=["pressure", "dT"])

    folds_to_save = []

    def aggregate_fold_metrics(m_folds):
        avg = {}
        std = {}

        for k in m_folds[0]:
            if k.endswith("_r2"):
                avg[k] = np.nan
                std[k] = 0.0
            else:
                values = [f[k] for f in m_folds]
                avg[k] = np.nanmean(values)
                std[k] = np.nanstd(values)

        return avg, std

    def compute_global_metrics(model_name, allowed_fold_ids=None):
        if allowed_fold_ids is None:
            y_true_parts = all_y_true
            y_pred_parts = all_y_preds[model_name]
        else:
            y_true_parts = [
                yt for yt, fid in zip(all_y_true, stored_fold_ids)
                if fid in allowed_fold_ids
            ]
            y_pred_parts = [
                yp for yp, fid in zip(all_y_preds[model_name], stored_fold_ids)
                if fid in allowed_fold_ids
            ]

        y_true_global_df = pd.DataFrame(
            np.vstack(y_true_parts),
            columns=["pressure", "dT"]
        )
        y_pred_global = np.vstack(y_pred_parts)

        return evaluate(y_true_global_df, y_pred_global)

    def attach_global_r2(avg, std, global_metrics):
        avg["pressure_r2"] = global_metrics["pressure_r2"]
        avg["dT_r2"] = global_metrics["dT_r2"]
        std["pressure_r2"] = 0.0
        std["dT_r2"] = 0.0

    for m_name in models:
        m_folds = all_fold_results[m_name]

        for i, f in enumerate(m_folds):
            fold_id = stored_fold_ids[i]

            row = {
                "model": m_name,
                "fold": fold_id + 1,
                "local": not (fold_id in fold_to_remove)
            }
            row.update(f)
            folds_to_save.append(row)

        avg, std = aggregate_fold_metrics(m_folds)

        global_metrics = compute_global_metrics(m_name)
        attach_global_r2(avg, std, global_metrics)

        final_avg.append(avg)
        final_std.append(std)

    pd.DataFrame(folds_to_save).to_csv("Output_files/folds.csv", index=False)

    for m_name in models:
        m_folds = [
            f for f, fid in zip(all_fold_results[m_name], stored_fold_ids)
            if fid not in fold_to_remove
        ]

        avg, std = aggregate_fold_metrics(m_folds)

        valid_fold_ids = [
            fid for fid in stored_fold_ids
            if fid not in fold_to_remove
        ]

        global_metrics = compute_global_metrics(m_name, valid_fold_ids)
        attach_global_r2(avg, std, global_metrics)

        final_avg_without_fold_1.append(avg)
        final_std_without_fold_1.append(std)

    if prediction_file == True:
        # ==========================================
        # SAVE CSV
        # ==========================================

        fold_metrics_df = pd.DataFrame(fold_metrics_rows)

        predictions_df = pd.DataFrame(prediction_rows)

        fold_metrics_df.to_csv(
            "Output_files/metrics_per_fold.csv",
            index=False
        )

        predictions_df.to_csv(
            "Output_files/predictions.csv",
            index=False
        )

    return final_avg, final_std, final_avg_without_fold_1, final_std_without_fold_1, fold_to_remove


def run_ablation_test(name, dataframe, features_list, groups, include_zero_end_train=False):
    print(f"\nTest ablacji \"{name}\"")
    y_local = dataframe[["pressure", "dT"]]

    avg, std, _, _, _ = run_cv(
        df=dataframe,
        y=y_local,
        models=["MO-LR"],
        df_value=features_list,
        groups=groups,
        include_zero_end_train=include_zero_end_train
    )
    return [avg[0], std[0]]

