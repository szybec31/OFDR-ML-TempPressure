"""
Uruchomienie eksperymentu z pliku konfiguracyjnego JSON.

    python main.py experiment configs/main_condition.json

Struktura konfiguracji (wszystkie klucze poza "name" mają wartości domyślne z DEFAULTS):
    name, description, seed, data, drop_low_quality, temperature_target, temperature_reference,
    split, blocks, inner_cv, n_jobs, features, models, baseline, grids, include_zero_end_train,
    variants: [ {name, ...dowolne nadpisania powyższych kluczy...}, ... ]   (opcjonalne)

Klucz "experiment" wybiera rodzaj eksperymentu:
    "cv"                  - walidacja krzyżowa modeli (domyślnie),
    "reduced_calibration" - punkt 6: ograniczanie liczby stanów kalibracyjnych (pipeline/reduced_calibration.py),
    "spatial"             - punkt 7: okna przestrzenne wzdłuż ROI (pipeline/spatial.py).

Bez "variants" wykonywany jest jeden wariant. Z "variants" każdy wariant trafia do własnego
podkatalogu, a dodatkowo powstaje zbiorcza tabela variants_summary.
"""
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from .cv import run_cv
from .models import get_model_spec
from .reduced_calibration import run_reduced_calibration
from .report import build_report, compact_summary, write_table
from .spatial import run_spatial
from .metrics import TEMPERATURE_TARGETS, get_targets, temperature_target
from .splits import condition_labels, split_diagnostics

EXPERIMENTS = {"reduced_calibration": run_reduced_calibration, "spatial": run_spatial}

DEFAULTS = {
    "experiment": "cv",           # cv | reduced_calibration | spatial
    "description": "",
    "seed": 42,
    "data": "Output_files/paired_features.csv",
    "drop_low_quality": True,
    "temperature_target": "dT",   # cel temperaturowy: dT (deltaT_label) | Tp (T_plate_label)
    "temperature_reference": "auto",  # T_ref dla modeli fizycznych (liczą w T - T_ref); auto: patrz niżej
    "split": "condition",         # condition (główny) | temperature | series | pressure | block
    "blocks": None,               # tylko dla split = "block": lista prostokątów w (T, p)
    "inner_cv": {"n_splits": 3, "group": "series_id"},   # kolumna grupująca strojenie: series_id | condition_id
    "n_jobs": -1,
    "features": ["mu_X", "mu_Y", "m", "d", "std_X", "std_Y", "iqr_X", "iqr_Y"],
    "models": ["AN-BL", "MO-LR"],
    "baseline": "AN-BL",
    "grids": {},
    "include_zero_end_train": False,
    "exclude_test_groups": [],    # foldy pomijane (reduced_calibration, spatial), np. ["dT=0.0"]
    "training_dataset": "Output_files/training_dataset.csv",
    "bin_cache": "Output_files/spatial_bins_{base}cm.npz",
    "runs_dir": "Output_files/runs",
}


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions():
    import scipy
    import sklearn
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
    }


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    full = copy.deepcopy(DEFAULTS)
    full.update(cfg)
    if "name" not in full:
        full["name"] = os.path.splitext(os.path.basename(path))[0]
    return full


def load_data(cfg):
    df = pd.read_csv(cfg["data"])
    required = {"sample_id", "ref", "point_type", "rep_idx"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"W {cfg['data']} brakuje kolumn {sorted(missing)} - uruchom ponownie `python main.py setup`."
        )
    t_col = temperature_target(cfg)
    if t_col not in df.columns:
        raise RuntimeError(f"W {cfg['data']} brak kolumny celu temperaturowego {t_col!r}.")
    if cfg["drop_low_quality"]:
        df = df[df["low_quality"] == False]
    # identyfikator warunku pomiarowego (T, p) - grupy dla leave-one-condition-out i strojenia wewnętrznego
    df = df.assign(condition_id=condition_labels(df, t_col))
    return df.reset_index(drop=True)


def resolve_temperature_reference(df, cfg):
    """
    Temperatura referencyjna celu temperaturowego (w jego jednostkach), od której modele fizyczne
    liczą T - T_ref (ich wzór nie ma wyrazu wolnego: sygnał = 0 w stanie referencyjnym).
        liczba - użyta wprost,
        "auto" - dT: 0; inny cel: jego wartość w rekordach kalibracji ciśnieniowej
                 (seria referencyjna dT = 0), wymagana jedna, jednoznaczna wartość.
    """
    ref = cfg.get("temperature_reference", "auto")
    if isinstance(ref, (int, float)):
        return float(ref)
    if ref != "auto":
        raise ValueError(f"temperature_reference = {ref!r}; dozwolone: liczba lub 'auto'")
    t_col = temperature_target(cfg)
    if t_col == "dT":
        return 0.0
    vals = df.loc[df["is_pressure_calibration"] == True, t_col].unique()
    if len(vals) != 1:
        raise ValueError(f"temperature_reference='auto': {t_col} w serii referencyjnej ma wartości {vals} - "
                         "podaj temperature_reference jawnie w konfiguracji.")
    return float(vals[0])


def run_variant(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg["seed"])

    df = load_data(cfg)
    cfg["temperature_reference"] = resolve_temperature_reference(df, cfg)
    leaked = sorted(set(cfg["features"]) & ({"pressure", *TEMPERATURE_TARGETS}))
    if leaked:
        raise RuntimeError(f"Etykiety wśród cech (przeciek celu): {leaked}")
    with open(os.path.join(out_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Cel temperaturowy: {temperature_target(cfg)}, T_ref = {cfg['temperature_reference']:g}")

    if cfg["experiment"] in EXPERIMENTS:
        EXPERIMENTS[cfg["experiment"]](cfg, out_dir, df)
        return None

    needed = set(cfg["features"])
    for m in cfg["models"]:
        needed |= set(get_model_spec(m).features or [])
    missing = sorted(c for c in needed if c not in df.columns)
    if missing:
        raise RuntimeError(f"Brak cech w danych: {missing}")

    result = run_cv(df, cfg)
    result["split_diagnostics"] = split_diagnostics(df, result["fold_defs"], temperature_target(cfg))
    info = build_report(result, df, cfg, out_dir)

    print(f"\n[{cfg['name']}] wspólne foldy: {info['common_folds']}, wykluczone: {info['excluded_folds']}")
    print(compact_summary(info["summary_common"], get_targets(cfg)).to_string(index=False))
    return info


def run_from_config(path):
    base = load_config(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base["runs_dir"], f"{stamp}_{base['name']}")
    os.makedirs(run_dir, exist_ok=True)

    # ciche ostrzeżenia o zbieżności także w procesach roboczych GridSearchCV
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    warnings.filterwarnings("ignore")

    meta = {
        "config_path": path,
        "started": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "data_sha256": {},
        "versions": _versions(),
    }
    with open(path, encoding="utf-8") as f:
        raw_cfg = f.read()
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(raw_cfg)

    variants = base.pop("variants", None)
    if not variants:
        variants = [{}]

    rows = []
    for v in variants:
        cfg = copy.deepcopy(base)
        cfg.update(v)
        if v:
            cfg["name"] = f"{base['name']}/{v['name']}"
        out_dir = run_dir if len(variants) == 1 else os.path.join(run_dir, v["name"])
        meta["data_sha256"][cfg["data"]] = _sha256(cfg["data"])

        print("\n" + "=" * 100 + f"\nWARIANT: {cfg['name']}\n" + "=" * 100)
        info = run_variant(cfg, out_dir)
        if info is None:
            continue

        s = info["summary_common"].copy()
        s.insert(0, "variant", v.get("name", base["name"]))
        s.insert(2, "common_folds", str(info["common_folds"]))
        rows.append(s)

    if len(variants) > 1 and rows:
        summary = pd.concat(rows, ignore_index=True)
        write_table(summary, run_dir, "variants_summary")
        cols = ["variant", "model", "n_folds"] + [c for c in summary.columns
                                                 if c.endswith("_mean") and ("mae" in c or "rmse" in c)]
        write_table(summary[cols], run_dir, "variants_table", "Porównanie wariantów (średnie po wspólnych foldach)")

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(base["runs_dir"], "LATEST.txt"), "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")

    print(f"\nWyniki zapisane w: {run_dir}")
    return run_dir
