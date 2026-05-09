import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from .utils.metrics import evaluate


def custom_combined_scorer(y_true, y_pred):
    """Kryterium wyboru hiperparametrów S z sekcji 12.4 [cite: 172, 173]"""
    mae_p = mean_absolute_error(y_true.iloc[:, 0], y_pred[:, 0])  # Ciśnienie
    mae_t = mean_absolute_error(y_true.iloc[:, 1], y_pred[:, 1])  # Temperatura
    return 0.5 * mae_t + 0.5 * mae_p


def run_experiment(X_all, y_all, split, groups_inner, **config):
    # Split danych na Outer Train i Outer Test
    train_idx, test_idx = split
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y_all.iloc[train_idx], y_all.iloc[test_idx]

    # Tylko wybrane cechy (np. mu_X, mu_Y)
    X_train_sub = X_train[config["df_value"]]
    X_test_sub = X_test[config["df_value"]]

    evaluations = []

    # Inner CV: GroupKFold po seriach [cite: 161]
    inner_cv = GroupKFold(n_splits=3)
    # Nasz scorer (GridSearch minimalizuje, więc greater_is_better=False)
    scorer = make_scorer(custom_combined_scorer, greater_is_better=False)

    for model_name in config["models"]:
        print(f"  Training Model: {model_name}...")

        if model_name == "MO-LR":
            from .models.linear import train_linear
            # Regresja liniowa nie wymaga strojenia w tym zadaniu [cite: 135]
            model = train_linear(X_train_sub, y_train)

        elif model_name == "POLY2-RIDGE":
            from .models.poly2_ridge import train_poly2_ridge
            # Definicja siatki parametrów [cite: 142]
            param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10]}
            # train_poly2_ridge musi zwracać Pipeline ze StandardScaler i PolynomialFeatures
            base_pipeline = train_poly2_ridge(None, None, config, return_pipeline=True)
            grid = GridSearchCV(base_pipeline, param_grid, cv=inner_cv, scoring=scorer)
            grid.fit(X_train_sub, y_train, groups=groups_inner)
            model = grid.best_estimator_

        elif model_name == "SVR-RBF":
            from .models.svr_rbf import train_svr_rbf
            # Parametry ze słownika [cite: 143]
            param_grid = {
                'svr__estimator__C': [1, 10, 100],
                'svr__estimator__epsilon': [0.01, 0.05, 0.1],
                'svr__estimator__gamma': ['scale', 0.1, 1]
            }
            base_pipeline = train_svr_rbf(None, None, config, return_pipeline=True)
            grid = GridSearchCV(base_pipeline, param_grid, cv=inner_cv, scoring=scorer)
            grid.fit(X_train_sub, y_train, groups=groups_inner)
            model = grid.best_estimator_

        elif model_name == "RF":
            from .models.randomforest import train_random_forest
            # Parametry RF [cite: 144, 145]
            param_grid = {
                'max_depth': [3, 5, None],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 0.8]
            }
            # Uwaga: RF nie wymaga skalowania, więc train_random_forest może zwracać sam model
            base_model = train_random_forest(None, None, config, return_base=True)
            grid = GridSearchCV(base_model, param_grid, cv=inner_cv, scoring=scorer)
            grid.fit(X_train_sub, y_train, groups=groups_inner)
            model = grid.best_estimator_
        elif model_name == "AN-BL":
            from baselines.models.physical_model import PhysicalModel

            model = PhysicalModel()

            # Physical model używa TYLKO mu_X i mu_Y
            model.fit(
                X_train[["mu_Y", "mu_X"]],
                y_train
            )

            y_pred = model.predict(
                X_test[["mu_Y", "mu_X"]]
            )

            evaluations.append(evaluate(y_test, y_pred))
            continue

        else:
            raise ValueError(f"Model {model_name} nieobsługiwany.")

        # Predykcja na niewidzianym poziomie temperatury (Outer Test)
        y_pred = model.predict(X_test_sub)
        evaluations.append(evaluate(y_test, y_pred))

    return evaluations


