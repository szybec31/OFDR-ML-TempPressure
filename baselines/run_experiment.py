import numpy as np
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from .utils.metrics import evaluate
from .models.physical_model import train_physical_model
from .models.linear import train_linear
from .models.poly2_ridge import train_poly2_ridge
from .models.svr_rbf import train_svr_rbf
from .models.randomforest import train_random_forest

def joint_score(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae_t = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_p = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

    sd_t = np.std(y_true[:, 0])
    sd_p = np.std(y_true[:, 1])

    score = (
        0.5 * (mae_t / sd_t)
        + 0.5 * (mae_p / sd_p)
    )

    return -score

def compute_rd(y_repeat, y_pred_repeat):
    """
    Residual Drift:
    RD = sqrt((T_pred_end - T_start)^2 + (P_pred_end - P_start)^2)

    For zero_end:
    P_start = 0,
    T_start = real dT of the sample.
    """

    if y_repeat is None or y_pred_repeat is None or len(y_repeat) == 0:
        return {
            "rd_mean": np.nan,
            "rd_max": np.nan
        }

    p_true = y_repeat["pressure"].values
    t_true = y_repeat["dT"].values

    p_pred = y_pred_repeat[:, 0]
    t_pred = y_pred_repeat[:, 1]

    rd = np.sqrt((t_pred - t_true) ** 2 + (p_pred - p_true) ** 2)

    return {
        "rd_mean": np.mean(rd),
        "rd_max": np.max(rd)
    }

def run_experiment(X_train, y_train, X_test, y_test, models, X_repeat=None, y_repeat=None):
    train_series = X_train["series_id"]
    ml_features = [c for c in X_train.columns if
                   c not in ["series_id", "is_temp_calibration", "is_pressure_calibration", "is_joint_regression", "is_repeatability_test"]]

    ml_train_mask = X_train["is_joint_regression"] == True

    X_train_ml = X_train.loc[ml_train_mask, ml_features]
    y_train_ml = y_train.loc[X_train_ml.index]
    groups_ml = train_series.loc[X_train_ml.index]

    evaluations = []
    predictions = []

    grids = {
        "POLY2-RIDGE": {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10]},
        "SVR-RBF": {
            "nystroem__n_components": [100, 300, 500],
            "nystroem__gamma": [0.01, 0.1, 1.0],
            "linear_svr__estimator__C": [1, 10, 100],
            "linear_svr__estimator__epsilon": [0.01, 0.05, 0.1],
        },
        "RF": {
            "n_estimators": [300],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.8],
        }
    }

    for model_name in models:
        print(f"  Traning: {model_name}", end=" ", flush=True)

        if model_name == "AN-BL":
            best_model = train_physical_model(X_train, y_train)

        elif model_name == "MO-LR":
            best_model = train_linear(X_train_ml, y_train_ml)

        else:
            if model_name == "POLY2-RIDGE":
                base = train_poly2_ridge(X_train_ml, y_train_ml, {})
            elif model_name == "SVR-RBF":
                base = train_svr_rbf(X_train_ml, y_train_ml, {})
            elif model_name == "RF":
                base = train_random_forest(X_train_ml, y_train_ml, {})

            inner_cv = GroupKFold(n_splits=3)
            joint_scorer = make_scorer(joint_score, greater_is_better=True)
            grid_search = GridSearchCV(
                estimator=base,
                param_grid=grids[model_name],
                cv=inner_cv,
                scoring=joint_scorer,
                n_jobs=-1
            )
            grid_search.fit(X_train_ml, y_train_ml, groups=groups_ml)
            print(f"\n    Best params: {grid_search.best_params_}")
            print(f"    Best CV MAE: {-grid_search.best_score_:.4f}")
            best_model = grid_search.best_estimator_

        if model_name == "AN-BL":
            y_pred = best_model.predict(X_test)
        else:
            y_pred = best_model.predict(X_test[ml_features])

        eval_result = evaluate(y_test, y_pred)

        if X_repeat is not None and y_repeat is not None and len(X_repeat) > 0:
            if model_name == "AN-BL":
                y_pred_repeat = best_model.predict(X_repeat)
            else:
                y_pred_repeat = best_model.predict(X_repeat[ml_features])

            eval_result.update(compute_rd(y_repeat, y_pred_repeat))
        else:
            eval_result.update({
                "rd_mean": np.nan,
                "rd_max": np.nan
            })

        evaluations.append(eval_result)
        predictions.append(y_pred)
        print("- OK")

    return evaluations, predictions
