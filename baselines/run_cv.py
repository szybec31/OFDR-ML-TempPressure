import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from .run_experiment import run_experiment
from .utils.metrics import evaluate


def run_cv(df, y, models, df_value, groups,include_zero_end_train=False, prediction_file=True):
    logo = LeaveOneGroupOut()
    all_fold_results = {m: [] for m in models}
    all_y_true = []
    all_y_preds = {m: [] for m in models}

    # do zapisu do predictions.csv
    fold_metrics_rows = []
    prediction_rows = []

    flags = ["is_temp_calibration", "is_pressure_calibration", "is_joint_regression", "is_repeatability_test"]
    X_full = df[df_value + flags + ["series_id"]]

    for fold, (train_idx, test_idx) in enumerate(logo.split(X_full, y, groups=groups)):
#        if fold == 0:
#            continue
        test_temp = groups.iloc[test_idx].unique()[0]
        print(f"\n>>> FOLD {fold + 1}: Test na dT = {test_temp}C")

        X_train_fold = X_full.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()

        X_test_fold = X_full.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        joint_mask = X_test_fold["is_joint_regression"] == True
        if not np.any(joint_mask):
            print(f"  Pominięto fold - brak danych joint_regression")
            continue

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

    for m_name in models:
        m_folds = all_fold_results[m_name]

        avg = {k: np.nanmean([f[k] for f in m_folds]) for k in m_folds[0]}
        std = {k: np.nanstd([f[k] for f in m_folds]) for k in m_folds[0]}

        m_preds_stacked = np.vstack(all_y_preds[m_name])
        global_metrics = evaluate(y_true_df, m_preds_stacked)

        avg["dT_r2"] = global_metrics["dT_r2"]
        avg["pressure_r2"] = global_metrics["pressure_r2"]
        std["dT_r2"] = 0.0

        final_avg.append(avg)
        final_std.append(std)

    for m_name in models:
        m_folds = all_fold_results[m_name].copy()
        m_folds.pop(0)

        avg = {k: np.nanmean([f[k] for f in m_folds]) for k in m_folds[0]}
        std = {k: np.nanstd([f[k] for f in m_folds]) for k in m_folds[0]}

        m_preds_stacked = np.vstack(all_y_preds[m_name])
        global_metrics = evaluate(y_true_df, m_preds_stacked)

        avg["dT_r2"] = global_metrics["dT_r2"]
        avg["pressure_r2"] = global_metrics["pressure_r2"]
        std["dT_r2"] = 0.0

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

    return final_avg, final_std, final_avg_without_fold_1, final_std_without_fold_1


def run_ablation_test(name, dataframe, features_list, include_zero_end_train=False):
    print(f"\nTest ablacji \"{name}\"")
    y_local = dataframe[["pressure", "dT"]]
    groups_local = dataframe["dT"]

    avg, std, _, _ = run_cv(
        df=dataframe,
        y=y_local,
        models=["MO-LR"],
        df_value=features_list,
        groups=groups_local,
        include_zero_end_train=include_zero_end_train
    )
    return [avg[0], std[0]]

