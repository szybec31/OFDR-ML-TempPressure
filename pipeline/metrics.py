"""
Metryki regresji dla obu celów (ciśnienie, temperatura).

Cel temperaturowy wybiera klucz konfiguracji "temperature_target" (kolumna danych):
    "dT" - zmiana temperatury względem referencji (deltaT_label, domyślnie),
    "Tp" - temperatura płyty (T_plate_label).
Nazwy metryk i kolumn wynikowych pochodzą od nazwy celu (np. dT_rmse / Tp_rmse, dT_pred / Tp_pred),
więc z każdej tabeli widać, który cel był użyty. Kolejność celów: [ciśnienie, temperatura].
"""
import numpy as np
from sklearn.metrics import r2_score

PRESSURE = "pressure"
DEFAULT_TEMPERATURE = "dT"
TEMPERATURE_TARGETS = ("dT", "Tp")
METRICS = ["mae", "rmse", "maxae", "bias", "r2"]
TARGET_UNITS = {"pressure": "MPa", "dT": "°C", "Tp": "°C"}


def temperature_target(cfg):
    t = cfg.get("temperature_target", DEFAULT_TEMPERATURE)
    if t not in TEMPERATURE_TARGETS:
        raise ValueError(f"temperature_target = {t!r}; dostępne: {TEMPERATURE_TARGETS}")
    return t


def get_targets(cfg):
    """[kolumna ciśnienia, kolumna temperatury] - kolejność kolumn w y i w predykcjach."""
    return [PRESSURE, temperature_target(cfg)]


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


def evaluate(y_true, y_pred, targets):
    """y_true: DataFrame z kolumnami targets, y_pred: tablica (n, 2) w kolejności targets."""
    y_pred = np.asarray(y_pred)
    res = {}
    for i, t in enumerate(targets):
        for k, v in target_metrics(y_true[t].to_numpy(), y_pred[:, i]).items():
            res[f"{t}_{k}"] = v
    return res


def metric_names(targets):
    return [f"{t}_{m}" for t in targets for m in METRICS]
