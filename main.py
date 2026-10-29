import os
import sys

import pandas as pd

from analyze_files import build_folder_summary
from build_df import build_dataframe
from utils import fix_dt16_folder_structure, quality_report

OUTPUT_DIR = "Output_files"

# kolumny zapisywane do zbioru modelowego paired_features.csv
PAIRED_COLUMNS = [
    "sample_id", "ref", "series_id", "rep_idx", "point_type",
    "pressure", "dT", "Tp",
    "mu_Y", "mu_X", "std_Y", "std_X", "iqr_Y", "iqr_X", "m", "d", "Xinter",
    "is_temp_calibration", "is_pressure_calibration", "is_joint_regression", "is_repeatability_test",
    "low_quality",
]

# skróty poleceń -> plik konfiguracyjny eksperymentu
CONFIG_ALIASES = {
    "run": "configs/main_condition.json",
    "r": "configs/main_condition.json",
    "run_temp": "configs/main_temperature.json",
    "rt": "configs/main_temperature.json",
    "run_series": "configs/leave_series_out.json",
    "rso": "configs/leave_series_out.json",
    "run_condition": "configs/main_condition.json",
    "rc": "configs/main_condition.json",
    "run_pressure": "configs/leave_pressure_out.json",
    "rp": "configs/leave_pressure_out.json",
    "run_block": "configs/block_holdout.json",
    "rb": "configs/block_holdout.json",
    "run_reduced": "configs/reduced_calibration.json",
    "rr": "configs/reduced_calibration.json",
    "run_roi": "configs/roi_windows.json",
    "rw": "configs/roi_windows.json",
    "run_spatial": "configs/spatial_windows.json",
    "rs": "configs/spatial_windows.json",
    "run_drift": "configs/drift_zero_end.json",
    "rd": "configs/drift_zero_end.json",
    "run_hybrid": "configs/hybrid_extrapolation.json",
    "rh": "configs/hybrid_extrapolation.json",
    "ablations": "configs/ablations.json",
    "a": "configs/ablations.json",
}


def prepare(broken=False):
    fix_dt16_folder_structure()

    df = build_dataframe(broken_data=broken)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "_broken" if broken else ""

    df.to_csv(os.path.join(OUTPUT_DIR, f"inventory{suffix}.csv"), index=False)

    df_summary = build_folder_summary(df)
    df_summary = df_summary[df_summary["ref"].notna()].copy()
    df_summary["sample_id"] = df_summary["path_y"]
    df_summary.to_csv(os.path.join(OUTPUT_DIR, f"training_dataset{suffix}.csv"), index=False)

    df_summary[PAIRED_COLUMNS].to_csv(os.path.join(OUTPUT_DIR, f"paired_features{suffix}.csv"), index=False)

    print(quality_report(df_summary, 0.9))


def info():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "paired_features.csv"))
    for flag in ["is_temp_calibration", "is_pressure_calibration", "is_joint_regression", "is_repeatability_test"]:
        print(f"{flag}: {df[flag].sum()} (po usunięciu low_quality: {df.loc[~df['low_quality'], flag].sum()})")
    print("---------- Info ----------")
    df.info()
    print(f"pressure: {sorted(df['pressure'].unique())}")


def experiment(config_path):
    from pipeline.runner import run_from_config
    return run_from_config(config_path)


HELP = """Użycie: python main.py POLECENIE [POLECENIE ...]

  setup, s                  przygotowanie danych (prepare + prepare_broken + info) - przy pierwszym uruchomieniu
  prepare, p                inventory.csv, training_dataset.csv, paired_features.csv
  prepare_broken, pb        jak wyżej, ale z nieskorygowanymi etykietami 11 / 0.01 MPa (ablacja 3)
  info, i                   liczności ról w paired_features.csv
  run, r, run_condition, rc eksperyment główny: configs/main_condition.json (leave-one-condition-out)
  run_temp, rt              configs/main_temperature.json (leave-one-temperature-out)
  run_series, rso           configs/leave_series_out.json (leave-one-series-out)
  run_pressure, rp          configs/leave_pressure_out.json (leave-one-pressure-level-out)
  run_block, rb             configs/block_holdout.json (usunięcie fragmentów przestrzeni T, p)
  run_reduced, rr           configs/reduced_calibration.json (pkt 6: ograniczenie liczby stanów kalibracyjnych)
  run_spatial, rs           configs/spatial_windows.json (pkt 7: okna przestrzenne wzdłuż ROI)
  run_roi, rw               configs/roi_windows.json (szerokość ROI: okna 1-100 cm i pełne ROI jako dane uczące)
  run_drift, rd             configs/drift_zero_end.json (zad. 9: dryft zero_end, wskaźnik jakości)
  run_hybrid, rh            configs/hybrid_extrapolation.json (modele hybrydowe; interpolacja vs ekstrapolacja)
  ablations, a              configs/ablations.json
  experiment PLIK.json      dowolny plik konfiguracyjny

Każdy eksperyment zapisuje komplet wyników w Output_files/runs/<data>_<nazwa>/
(ścieżka ostatniego uruchomienia: Output_files/runs/LATEST.txt)."""


if __name__ == "__main__":
    argv = sys.argv[1:]

    if not argv or argv[0] in ["help", "-h", "--help"]:
        print(HELP)
        sys.exit()

    while argv:
        arg = argv.pop(0)
        if arg in ["setup", "s"]:
            prepare()
            prepare(broken=True)
            info()
        elif arg in ["prepare", "p"]:
            prepare()
        elif arg in ["prepare_broken", "pb"]:
            prepare(broken=True)
        elif arg in ["info", "i"]:
            info()
        elif arg in CONFIG_ALIASES:
            experiment(CONFIG_ALIASES[arg])
        elif arg in ["experiment", "e"]:
            if not argv:
                sys.exit("Podaj ścieżkę do pliku konfiguracyjnego, np. python main.py experiment configs/main_condition.json")
            experiment(argv.pop(0))
        else:
            sys.exit(f"Nieznane polecenie: {arg}\n\n{HELP}")
