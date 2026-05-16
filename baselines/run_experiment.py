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


def run_experiment(X_train, y_train, X_test, y_test, models):
    train_series = X_train["series_id"]
    ml_features = [c for c in X_train.columns if
                   c not in ["series_id", "is_temp_calibration", "is_pressure_calibration", "is_joint_regression"]]

    evaluations = []
    predictions = []

    grids = {
        "POLY2-RIDGE": {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10]},
        "SVR-RBF": {
            "svr__estimator__C": [1, 10, 100],
            "svr__estimator__epsilon": [0.01, 0.05, 0.1],
            "svr__estimator__gamma": ["scale", 0.1, 1]
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
            best_model = train_linear(X_train[ml_features], y_train)

        else:
            if model_name == "POLY2-RIDGE":
                base = train_poly2_ridge(X_train[ml_features], y_train, {})
            elif model_name == "SVR-RBF":
                base = train_svr_rbf(X_train[ml_features], y_train, {})
            elif model_name == "RF":
                base = train_random_forest(X_train[ml_features], y_train, {})

            inner_cv = GroupKFold(n_splits=3)
            joint_scorer = make_scorer(joint_score, greater_is_better=True)
            grid_search = GridSearchCV(
                estimator=base,
                param_grid=grids[model_name],
                cv=inner_cv,
                scoring=joint_scorer,
                n_jobs=-1
            )
            grid_search.fit(X_train[ml_features], y_train, groups=train_series)
            print(f"\n    Best params: {grid_search.best_params_}")
            print(f"    Best CV MAE: {-grid_search.best_score_:.4f}")
            best_model = grid_search.best_estimator_

        if model_name == "AN-BL":
            y_pred = best_model.predict(X_test)
        else:
            y_pred = best_model.predict(X_test[ml_features])

        evaluations.append(evaluate(y_test, y_pred))
        predictions.append(y_pred)
        print("- OK")

    return evaluations, predictions

