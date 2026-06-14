import numpy as np
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from .utils.metrics import evaluate
from .models.physical_model import train_physical_model
from .models.linear import train_linear
from .models.poly2_ridge import train_poly2_ridge
from .models.svr_rbf import train_svr_rbf
from .models.nystroem_svr import train_nystroem_svr
from .models.randomforest import train_random_forest
from .models.hist_gradient_boosting import train_hist_gradient_boosting
from .models.gaussian_process import train_gaussian_process
from .models.kernel_ridge import train_kernel_ridge
from .models.gradient_boosting import train_gradient_boosting
from .models.xgboost_model import train_xgboost
from .models.mlp import train_mlp
import time

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

def getMLFeatures(X_train):
    return [c for c in X_train.columns if c not in ["series_id", "is_temp_calibration", "is_pressure_calibration", "is_joint_regression", "is_repeatability_test"]]

def filter_dataset(X, y, req_col=None, req_value=None):

    if req_col is not None:
        mask = X[req_col] == req_value

        X = X.loc[mask]
        y = y.loc[mask]

    return X, y

def run_experiment(X_train, y_train, X_test, y_test, models, include_zero_end_train = False):
    ml_features = getMLFeatures(X_train)

    print("Test shape: ",X_test.shape)
    evaluations = []
    predictions = []

    grids = {
        "POLY2-RIDGE": {
            "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
            "ridge__solver": ["auto", "saga"]
        },
        "SVR-RBF": {
            "svr__estimator__C": [0.1, 1, 10, 100, 1000],
            "svr__estimator__epsilon": [0.001, 0.01, 0.05, 0.1, 0.2],
            "svr__estimator__gamma": ["scale", "auto", 0.01, 0.1, 1]
        },
        "NYSTROEM-SVR": {
            "nystroem__n_components": [100, 300, 500, 1000],
            "nystroem__gamma": [0.01, 0.1, 1],
            "svr__estimator__C": [0.1, 1, 10],
            "svr__estimator__epsilon": [0.01, 0.05, 0.1],
        },
        "RF": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2", 0.8],
        },
        "HGBR": {
            "hgb__estimator__learning_rate": [0.01, 0.03, 0.1, 0.2],
            "hgb__estimator__max_leaf_nodes": [15, 31, 63],
            "hgb__estimator__l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "hgb__estimator__min_samples_leaf": [10, 20, 40],
        },
        "GPR": {
            "gpr__alpha": [1e-10, 1e-6, 1e-4, 1e-2, 1e-1],
            "gpr__n_restarts_optimizer": [0, 3, 5]
        },
        "KRR-RBF": {
            "krr__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "krr__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
        },
        "GBR": {
            "gbr__estimator__n_estimators": [100, 300, 500],
            "gbr__estimator__learning_rate": [0.01, 0.03, 0.1],
            "gbr__estimator__max_depth": [2, 3, 5],
            "gbr__estimator__subsample": [0.7, 0.8, 1.0],
        },
        "XGBOOST": {
            "xgb__estimator__n_estimators": [100, 200, 300, 500],
            "xgb__estimator__max_depth": [2, 3, 5, 7],
            "xgb__estimator__learning_rate": [0.01, 0.03, 0.1, 0.2],
            "xgb__estimator__subsample": [0.6, 0.8, 1.0],
            "xgb__estimator__colsample_bytree": [0.6, 0.8, 1.0],
        },
        "MLP": {
            "mlp__estimator__hidden_layer_sizes": [
                (16,),
                (32,),
                (32,16),
                (64,32)
            ],

            "mlp__estimator__activation": [
                "relu",
                "tanh"
            ],

            "mlp__estimator__alpha": [
                1e-4,
                1e-3,
                1e-2
            ],

            "mlp__estimator__learning_rate_init": [
                1e-3,
                1e-2
            ]
        }
    }

    for model_name in models:
        X_train_local = X_train.copy()
        y_train_local = y_train.copy()

        print(f"  Traning: {model_name}", end=" ", flush=True)
        start = time.time()

        if model_name == "AN-BL":
            best_model = train_physical_model(X_train_local, y_train_local)

        elif model_name == "MO-LR":
            if not include_zero_end_train:
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
            best_model = train_linear(X_train_local[ml_features], y_train_local)

        else:
            if model_name == "POLY2-RIDGE":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_poly2_ridge(X_train_local[ml_features], y_train_local, {})
            elif model_name == "SVR-RBF":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_svr_rbf(X_train_local[ml_features], y_train_local, {})
            elif model_name == "NYSTROEM-SVR":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_nystroem_svr(X_train_local[ml_features], y_train_local, {})
            elif model_name == "RF":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_random_forest(X_train_local[ml_features], y_train_local, {})
            elif model_name == "HGBR":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_hist_gradient_boosting(X_train_local[ml_features], y_train_local, {})
            elif model_name == "GPR":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_gaussian_process(X_train_local[ml_features], y_train_local, {})
            elif model_name == "KRR-RBF":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_kernel_ridge(X_train_local[ml_features], y_train_local, {})
            elif model_name == "GBR":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_gradient_boosting(X_train_local[ml_features], y_train_local, {})
            elif model_name == "XGBOOST":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_xgboost(X_train_local[ml_features], y_train_local, {})
            elif model_name == "MLP":
                X_train_local, y_train_local = filter_dataset(X_train_local, y_train_local, "is_joint_regression", True)
                base = train_mlp(X_train_local[ml_features], y_train_local, {})


            inner_cv = GroupKFold(n_splits=3)
            joint_scorer = make_scorer(joint_score, greater_is_better=True)
            grid_search = GridSearchCV(
                estimator=base,
                param_grid=grids[model_name],
                cv=inner_cv,
                scoring=joint_scorer,
                n_jobs=-1
            )
            train_series = X_train_local["series_id"]
            grid_search.fit(X_train_local[ml_features], y_train_local, groups=train_series)
            #print(f"\n    Best params: {grid_search.best_params_}")
            #print(f"    Best CV MAE: {-grid_search.best_score_:.4f}")
            best_model = grid_search.best_estimator_

        if model_name == "AN-BL":
            y_pred = best_model.predict(X_test)
        else:
            y_pred = best_model.predict(X_test[ml_features])

        ev = evaluate(y_test, y_pred)
        if model_name not in ["AN-BL", "MO-LR"]:
            ev["params"] = grid_search.best_params_
        evaluations.append(ev)
        predictions.append(y_pred)
        print("- OK", f"{(time.time() - start):.3f}")

    return evaluations, predictions

