from .run_experiment import run_experiment
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

def run_cv(df, y, n_splits=5, **config):
    # ALL DESCRIPTIONS IN run_experiment.py

    X = df # nie ma znaczenia kolumna, gdyz StratifiedKFold.split zwraca i tak tylko id's

    skf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X)):
        print(f"Fold {fold+1}/{n_splits}")

        results_list = run_experiment(
            df,
            y,
            split=(train_idx, test_idx),
            **config
        )

        all_results.append(results_list)

    # ========================
    # AGREGACJA PER MODEL
    # ========================

    n_models = len(all_results[0])  # np. 3 (model1, model2, fusion)

    avg_all = []
    std_all = []

    for m in range(n_models):
        model_results = [fold[m] for fold in all_results]

        avg = {
            key: np.mean([r[key] for r in model_results])
            for key in model_results[0]
        }

        std = {
            key: np.std([r[key] for r in model_results])
            for key in model_results[0]
        }

        avg_all.append(avg)
        std_all.append(std)

    return avg_all, std_all