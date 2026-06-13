from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_kernel_ridge(X_train, y_train, params=None):
    if params is None:
        params = {}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(
            kernel="rbf",
            **params
        ))
    ])

    model.fit(X_train, y_train)
    return model