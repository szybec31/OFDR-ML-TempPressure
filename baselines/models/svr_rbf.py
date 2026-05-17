from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVR

def train_svr_rbf(X_train, y_train, config):

    random_state = config.get("random_state", 42)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(
            kernel="rbf",
            gamma=config.get("gamma", 0.1),
            n_components=config.get("n_components", 300),
            random_state=random_state
        )),
        ("linear_svr", MultiOutputRegressor(
            LinearSVR(
                C=config.get("C", 10),
                epsilon=config.get("epsilon", 0.05),
                random_state=random_state,
                max_iter=100000,
                tol=1e-3
            )
        ))
    ])

    model.fit(X_train, y_train)

    return model
