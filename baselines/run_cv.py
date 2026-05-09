from .run_experiment import run_experiment
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np

def run_cv(X, y, groups_outer, groups_inner, **config):
    """
    Realizuje Nested Cross-Validation (Zadanie 16).
    Outer: Leave-One-Temperature-Level-Out.
    Inner: Zarządzane wewnątrz run_experiment przez GridSearchCV.
    """
    logo = LeaveOneGroupOut()
    all_results = []


    # Outer loop - iteracja po poziomach temperatury
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups_outer)):
        current_temp = groups_outer.iloc[test_idx].unique()[0]
        print(f"--- Outer Fold {fold+1} (Test Temp: {current_temp}°C) ---")
        results_list = run_experiment(
            X, y,
            split=(train_idx, test_idx),
            groups_inner=groups_inner.iloc[train_idx], # Przekazujemy serie tylko dla zbioru treningowego
            **config
        )
        all_results.append(results_list)

    # Agregacja wyników per model
    n_models = len(all_results[0])
    avg_all, std_all = [], []

    for m in range(n_models):
        model_results = [fold[m] for fold in all_results]
        avg = {k: np.mean([r[k] for r in model_results]) for k in model_results[0]}
        std = {k: np.std([r[k] for r in model_results]) for k in model_results[0]}
        avg_all.append(avg)
        std_all.append(std)

    return avg_all, std_all

