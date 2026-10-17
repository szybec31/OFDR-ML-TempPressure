"""
Zewnętrzna pętla walidacji (LeaveOneGroupOut) + wewnętrzne strojenie (GridSearchCV z GroupKFold).

Zasady:
- zero_end nigdy nie trafia do testu głównego; jest przewidywany przez model folda, w którym
  jego grupa jest testowa (predykcja out-of-fold do analizy powtarzalności),
- zero_end trafia do treningu modeli ML tylko gdy include_zero_end_train = true (ablacja),
- model, którego nie da się skalibrować w danym foldzie, dostaje status "not_identifiable"
  (żadnych wartości domyślnych),
- skalowanie / strojenie odbywa się wyłącznie na danych treningowych folda.
"""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold

from baselines.models.physical_model import NotIdentifiableError
from .metrics import get_targets, evaluate
from .models import build_estimator, get_model_spec
from .splits import make_folds


def make_joint_scorer(sd_p, sd_t):
    """S = 0.5*MAE_p/SD(p_train) + 0.5*MAE_T/SD(T_train) (pkt 12.4 specyfikacji); zwracamy -S."""
    def scorer(estimator, X, y):
        y = np.asarray(y)
        pred = np.asarray(estimator.predict(X))
        mae_p = np.mean(np.abs(pred[:, 0] - y[:, 0]))
        mae_t = np.mean(np.abs(pred[:, 1] - y[:, 1]))
        return -(0.5 * mae_p / sd_p + 0.5 * mae_t / sd_t)
    return scorer


def _fit_ml(spec, rows, features, cfg, seed):
    targets = get_targets(cfg)
    X, y = rows[features], rows[targets]
    est = build_estimator(spec, seed, cfg)

    inner = cfg["inner_cv"]
    n_groups = rows[inner["group"]].nunique()
    n_splits = min(inner["n_splits"], n_groups)

    if not spec.grid or n_splits < 2:
        est.fit(X, y)
        return est, ({} if not spec.grid else {"_untuned": "za mało grup do strojenia"})

    sd_p = float(np.std(y[targets[0]])) or 1.0
    sd_t = float(np.std(y[targets[1]])) or 1.0
    search = GridSearchCV(
        estimator=est,
        param_grid=spec.grid,
        cv=GroupKFold(n_splits=n_splits),
        scoring=make_joint_scorer(sd_p, sd_t),
        n_jobs=cfg.get("n_jobs", -1),
        refit=True,
        error_score=np.nan,
    )
    search.fit(X, y, groups=rows[inner["group"]])
    return search.best_estimator_, dict(search.best_params_)


def run_cv(df, cfg):
    seed = cfg["seed"]
    features = cfg["features"]
    targets = get_targets(cfg)
    t_col = targets[1]
    folds = make_folds(df, cfg)
    specs = [get_model_spec(m, cfg.get("grids", {}).get(m)) for m in cfg["models"]]

    is_joint = df["is_joint_regression"] == True
    is_zero_end = df["is_repeatability_test"] == True

    fold_rows, pred_rows, coef_rows = [], [], []
    # dT i Tp zawsze zapisywane jako metadane; prawdziwa wartość celu jest w kolumnie t_col
    meta_cols = ["sample_id", "series_id", "rep_idx", "dT", "Tp", "pressure", "point_type"]

    n_folds = len(folds)

    for fold_id, (g, mask) in enumerate(folds, start=1):
        in_test = pd.Series(mask, index=df.index)
        train = df[~in_test]
        test_main = df[in_test & is_joint]
        test_zero_end = df[in_test & is_zero_end]

        print(f"\n>>> FOLD {fold_id}/{n_folds}: test = {g}  (test={len(test_main)}, zero_end={len(test_zero_end)})")

        ml_train = train[is_joint[~in_test] | (cfg.get("include_zero_end_train", False) & is_zero_end[~in_test])]

        for spec in specs:
            t0 = time.time()
            row = {"fold": fold_id, "test_group": g, "model": spec.name, "n_test": len(test_main)}
            try:
                if spec.kind == "physical":
                    model = build_estimator(spec, seed, cfg).fit(train, train[targets])
                    best_params = {}
                    coef_rows.append({"fold": fold_id, "test_group": g, "model": spec.name, **model.coefficients()})
                    predict = lambda d: model.predict(d)
                    row["n_train"] = int((train["is_temp_calibration"] | train["is_pressure_calibration"]).sum())
                else:
                    feats = spec.features or features
                    model, best_params = _fit_ml(spec, ml_train, feats, cfg, seed)
                    predict = lambda d, feats=feats: model.predict(d[feats])
                    row["n_train"] = len(ml_train)
            except NotIdentifiableError as e:
                row.update({"status": "not_identifiable", "note": str(e)})
                fold_rows.append(row)
                print(f"  {spec.name:<15} N/A ({e})")
                continue

            row["status"] = "ok"
            row["best_params"] = str(best_params) if best_params else ""

            for subset, part in [("test", test_main), ("zero_end", test_zero_end)]:
                if part.empty:
                    continue
                pred = np.asarray(predict(part))
                if subset == "test":
                    row.update(evaluate(part[targets], pred, targets))
                block = part[meta_cols].copy()
                block.insert(0, "subset", subset)
                block.insert(0, "model", spec.name)
                block.insert(0, "test_group", g)
                block.insert(0, "fold", fold_id)
                block[f"{targets[0]}_pred"] = pred[:, 0]
                block[f"{t_col}_pred"] = pred[:, 1]
                pred_rows.append(block)

            row["fit_seconds"] = round(time.time() - t0, 3)
            fold_rows.append(row)
            if "pressure_rmse" in row:
                print(f"  {spec.name:<15} p_RMSE={row['pressure_rmse']:.3f}  {t_col}_RMSE={row[f'{t_col}_rmse']:.3f}"
                      f"  ({row['fit_seconds']:.1f}s)")
            else:
                print(f"  {spec.name:<15} tylko zero_end ({row['fit_seconds']:.1f}s)")

    return {
        "folds": pd.DataFrame(fold_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame(),
        "coefficients": pd.DataFrame(coef_rows),
        "fold_defs": folds,
    }
