import numpy as np
from sklearn.model_selection import GridSearchCV, GroupKFold
from .utils.metrics import evaluate
from .models.physical_model import train_physical_model
from .models.linear import train_linear
from .models.poly2_ridge import train_poly2_ridge
from .models.svr_rbf import train_svr_rbf
from .models.randomforest import train_random_forest


def run_experiment(X_train, y_train, X_test, y_test, models):
    train_series = X_train["series_id"]
    ml_features = [c for c in X_train.columns if
                   c not in ["series_id", "is_temp_calibration", "is_pressure_calibration", "is_joint_regression"]]

    evaluations = []
    predictions = []

    grids = {
        "POLY2-RIDGE": {"ridge__alpha": [0.1, 1.0, 10.0]},
        "SVR-RBF": {
            "svr__estimator__C": [1, 10, 100],
            "svr__estimator__epsilon": [0.01, 0.05, 0.1]
        },
        "RF": {
            "n_estimators": [300],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 2]
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
            grid_search = GridSearchCV(
                estimator=base,
                param_grid=grids[model_name],
                cv=inner_cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X_train[ml_features], y_train, groups=train_series)
            best_model = grid_search.best_estimator_

        if model_name == "AN-BL":
            y_pred = best_model.predict(X_test)
        else:
            y_pred = best_model.predict(X_test[ml_features])

        evaluations.append(evaluate(y_test, y_pred))
        predictions.append(y_pred)
        print("- OK")

    return evaluations, predictions

