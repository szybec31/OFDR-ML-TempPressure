from build_df import build_dataframe
from build_global_db import build_stupid_dataset
import pandas as pd
import os
import sys
from utils import fix_dt16_folder_structure, quality_report
from analyze_files import build_folder_summary
from baselines.run_cv import run_cv, run_ablation_test
from baselines.utils.build_groups import build_groups
from baselines.utils.print_and_save import print_and_save, print_ablation_test
from repeatability_analysis import compute_repeatability_metrics
from validation_diagnostics import run_validation_diagnostics
from model_gain import compute_model_gain

def main(type = "prepare", arg1 = False, arg2 = False):

    if type in ["prepare", "p"]:

        broken = arg1

        fix_dt16_folder_structure()

        df = build_dataframe(broken_data = broken)

        output_dir = 'Output_files'
        os.makedirs(output_dir, exist_ok=True)  # create folder if it doesn't exist

        df.to_csv(os.path.join(output_dir, 'inventory.csv' if not broken else 'inventory_broken.csv'), index=False)

        df_summary = build_folder_summary(df)

        df_summary = df_summary[df_summary["ref"].notna()]
        df_summary.to_csv(os.path.join(output_dir, 'training_dataset.csv' if not broken else 'training_dataset_broken.csv'), index=False)

        df_base_for_training = df_summary[["series_id", "pressure", "dT", "Tp", "mu_Y", "mu_X", "std_Y", "std_X",
                                           "irq_Y", "irq_X", "diff_XY", "mean_XY", "Xinter", "Pdir",
                                           "is_temp_calibration", "is_pressure_calibration",
                                           "is_joint_regression", "is_repeatability_test", "low_quality"]]
        df_base_for_training.to_csv(os.path.join(output_dir, 'paired_features.csv' if not broken else 'paired_features_broken.csv'), index=False)

        print(quality_report(df_summary, 0.9))

    elif type in ["run", "r"]:

        leave_one_condition_out = arg1

        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        df = df_full[df_full["low_quality"] == False].copy()
        y = df[["pressure", "dT"]]
        groups = build_groups(df, leave_one_condition_out)

        features = ["mu_Y", "mu_X", "std_Y", "std_X", "irq_Y", "irq_X"]
        models = ["AN-BL", "MO-LR", "RF", "POLY2-RIDGE", "SVR-RBF", "NYSTROEM-SVR", "HGBR", "GPR", "KRR-RBF", "GBR", "XGBOOST"]
        avg_results, std_results, avg_results_wo_f1, std_results_wo_f1, fold_to_remove = run_cv(
            df=df,
            y=y,
            models=models,
            df_value=features,
            groups=groups,
            prediction_file=arg2
        )

        print_and_save(models, avg_results, std_results, "All Results:", f"res_all_{("condition" if leave_one_condition_out else "temperature")}.csv")
        print_and_save(models, avg_results_wo_f1, std_results_wo_f1, f"Results without folds {str(fold_to_remove)}:", f"res_corr_{("condition" if leave_one_condition_out else "temperature")}.csv")

    elif type in ["ablations", "a"]:
        output_dir = 'Output_files'
        file_path = os.path.join(output_dir, 'paired_features.csv')

        df_full = pd.read_csv(file_path)
        
        ablation_results = {}
        leave_one_condition_out = arg1

        # Ablacja 1: Cechy bazowe (4) vs Rozszerzone (8)
        df_clean = df_full[df_full["low_quality"] == False].copy()
        groups = build_groups(df_clean, leave_one_condition_out)

        feat_4 = ["mu_X", "mu_Y", "std_X", "std_Y"]
        feat_8 = ["mu_X", "mu_Y", "std_X", "std_Y", "irq_X", "irq_Y", "diff_XY", "mean_XY"]
        feat_10 = ["mu_X", "mu_Y", "std_X", "std_Y", "irq_X", "irq_Y", "diff_XY", "mean_XY", "Xinter", "Pdir"]

        ablation_results["A1_4_Features"] = run_ablation_test("4 Features", df_clean, feat_4, groups)
        ablation_results["A1_8_Features"] = run_ablation_test("8 Features", df_clean, feat_8, groups)
        ablation_results["A1_10_Features_Xinter_Pdir"] = run_ablation_test("10 Features + Xinter/Pdir", df_clean, feat_10, groups)

        # Ablacja 2: Jeden kanał vs Dwa kanały
        feat_x = ["mu_X", "std_X", "irq_X"]
        feat_y = ["mu_Y", "std_Y", "irq_Y"]
        feat_xy = ["mu_X", "mu_Y", "std_X", "std_Y", "irq_X", "irq_Y", "diff_XY", "mean_XY"]

        ablation_results["A2_X_Only"] = run_ablation_test("X Channel Only", df_clean, feat_x, groups)
        ablation_results["A2_Y_Only"] = run_ablation_test("Y Channel Only", df_clean, feat_y, groups)
        ablation_results["A2_XY_Full"] = ablation_results["A1_8_Features"]

        # Ablacja 3: Wpływ korekty etykiet
        # Symulacja błędu etykiet: 10 MPa -> 11 MPa i 0 MPa -> 0.01 MPa
        df_broken = pd.read_csv(os.path.join(output_dir, 'paired_features_broken.csv'))
        df_broken = df_broken[(df_broken["low_quality"] == False)].copy()
        groups_broken = build_groups(df_broken, leave_one_condition_out)
        ablation_results["A3_Bad_Labels"] = run_ablation_test(
            "Bad Labels (11MPa/0.01MPa)",
            df_broken,
            feat_8,
            groups_broken,
            include_zero_end_train=True
        )

        # Ablacja 4: Wpływ zero_end w treningu
        ablation_results["A4_With_Zero_End"] = run_ablation_test(
            "Including Zero_End",
            df_clean,
            feat_8,
            groups,
            include_zero_end_train=True
        )

        print_ablation_test(ablation_results,csv_path="Output_files/ablation_results.csv")

    elif type in ["repeatability", "rep", "rd"]:
        compute_repeatability_metrics()

    elif type in ["validation_diagnostics", "valdiag", "vd"]:
        run_validation_diagnostics()

    elif type in ["model_gain", "gain", "g"]:
        compute_model_gain()

    elif type in ["info", "i"]:
        output_dir = 'Output_files'
        df = pd.read_csv(os.path.join(output_dir, 'paired_features.csv'))

        print(f"is_temp_calibration: {df["is_temp_calibration"].sum()}")
        print(f"is_pressure_calibration: {df["is_pressure_calibration"].sum()}")
        print(f"is_joint_regression: {df["is_joint_regression"].sum()}")
        print(f"is_repeatability_test: {df["is_repeatability_test"].sum()}")

        print("---------- Info ----------")
        print(df.info())
        
        print(f"pressure: {df["pressure"].unique()}")

if __name__ == "__main__":
    argv = sys.argv
    argv.pop(0)

    # Console Help Section / Also important for non-console people:
    if len(argv) == 1 and argv[0] == "help":
        print("If you run script first time use `setup`")
        print("You may use also other (or multiple) options:") 
        print("`prepare`, `p`")
        print("`prepare_broken`, `pb`")
        print("`run`, `run_temp`, `r`, `rt`")
        print("`run_condition`, `rc`")
        print("`ablations`, `ablations_temp`, `a`, `at`")
        print("`ablations_condition`, `ac`")
        print("`repeatability`, `rep`, `rd`")
        print("`validation_diagnostics`, `valdiag`, `vd`")
        print("`model_gain`, `gain`, `g`")
        print("`info`, `i`")
        print("`setup`, `s`")
        exit()

    if len(argv) == 0:
        # If you not using console argv, feel fres to changes argv below, you may use as many argv as you want
        # Correct options are written above in console help section
        argv = ["run"]




    while(len(argv) >= 1):
        arg = argv.pop(0)
        if arg in ["setup", "s"]:
            main("p")
            main("p", True)
            main("i")
            break
        else:
            if arg in ["prepare_broken", "pb"]:
                main("p", True)
            elif arg in ["run", "run_temp", "r", "rt"]:
                main("r", False)
            elif arg in ["run_condition", "rc"]:
                main("r", True)
            elif arg in ["ablations", "ablations_temp", "a", "at"]:
                main("a", False)
            elif arg in ["ablations_condition", "ac"]:
                main("a", True)
            elif arg in ["repeatability", "rep", "rd"]:
                main("repeatability")
            elif arg in ["validation_diagnostics", "valdiag", "vd"]:
                main("validation_diagnostics")
            elif arg in ["model_gain", "gain", "g"]:
                main("model_gain")
            else:
                main(arg)
    
    