"""Metryki regresji dla obu celów (ciśnienie, zmiana temperatury)."""
import numpy as np
from sklearn.metrics import r2_score

TARGETS = ["pressure", "dT"]          # kolejność kolumn w y i w predykcjach
TARGET_UNITS = {"pressure": "MPa", "dT": "°C"}


def target_metrics(y_true, y_pred):
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "maxae": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),          # średni błąd ze znakiem (pred - true)
    }
    # R^2 nie ma sensu, gdy cel jest stały w zbiorze (np. jedna temperatura w foldzie)
    out["r2"] = float(r2_score(y_true, y_pred)) if np.std(y_true) > 1e-9 else np.nan
    return out


def evaluate(y_true, y_pred):
    """y_true: DataFrame z kolumnami TARGETS, y_pred: tablica (n, 2) w kolejności TARGETS."""
    y_pred = np.asarray(y_pred)
    res = {}
    for i, t in enumerate(TARGETS):
        for k, v in target_metrics(y_true[t].to_numpy(), y_pred[:, i]).items():
            res[f"{t}_{k}"] = v
    return res


METRIC_NAMES = [f"{t}_{m}" for t in TARGETS for m in ["mae", "rmse", "maxae", "bias", "r2"]]
