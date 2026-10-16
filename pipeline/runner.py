"""
Uruchomienie eksperymentu z pliku konfiguracyjnego JSON.

    python main.py experiment configs/main_temperature.json

Struktura konfiguracji (wszystkie klucze poza "name" mają wartości domyślne z DEFAULTS):
    name, description, seed, data, drop_low_quality, split, inner_cv, n_jobs,
    features, models, baseline, grids, include_zero_end_train,
    variants: [ {name, ...dowolne nadpisania powyższych kluczy...}, ... ]   (opcjonalne)

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
from .report import build_report, compact_summary, write_table
from .splits import make_groups, split_diagnostics

DEFAULTS = {
    "description": "",
    "seed": 42,
    "data": "Output_files/paired_features.csv",
    "drop_low_quality": True,
    "split": "temperature",
    "inner_cv": {"n_splits": 3, "group": "series_id"},
    "n_jobs": -1,
    "features": ["mu_X", "mu_Y", "std_X", "std_Y", "iqr_X", "iqr_Y"],
    "models": ["AN-BL", "MO-LR"],
    "baseline": "AN-BL",
    "grids": {},
    "include_zero_end_train": False,
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
    if cfg["drop_low_quality"]:
        df = df[df["low_quality"] == False]
    return df.reset_index(drop=True)


def run_variant(cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(cfg["seed"])
    with open(os.path.join(out_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    df = load_data(cfg)
    missing = [c for c in cfg["features"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Brak cech w danych: {missing}")

    result = run_cv(df, cfg)
    result["split_diagnostics"] = split_diagnostics(df, make_groups(df, cfg["split"]))
    info = build_report(result, df, cfg, out_dir)

    print(f"\n[{cfg['name']}] wspólne foldy: {info['common_folds']}, wykluczone: {info['excluded_folds']}")
    print(compact_summary(info["summary_common"]).to_string(index=False))
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

        s = info["summary_common"].copy()
        s.insert(0, "variant", v.get("name", base["name"]))
        s.insert(2, "common_folds", str(info["common_folds"]))
        rows.append(s)

    summary = pd.concat(rows, ignore_index=True)
    if len(variants) > 1:
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
