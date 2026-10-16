from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_svr_rbf(seed=42, C=10, epsilon=0.05, gamma="scale"):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))),
    ])


def train_svr_rbf(X_train, y_train, config):
    return build_svr_rbf(
        C=config.get("C", 10),
        epsilon=config.get("epsilon", 0.05),
        gamma=config.get("gamma", "scale"),
    ).fit(X_train, y_train)
