from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

def train_svr_rbf(X_train, y_train, config, return_pipeline=False):
    base_model = SVR(
        kernel="rbf",
        C=config.get("C", 10),
        epsilon=config.get("epsilon", 0.05),
        gamma=config.get("gamma", "scale")
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(base_model))
    ])

    if return_pipeline:
        return model

    model.fit(X_train, y_train)
    return model


