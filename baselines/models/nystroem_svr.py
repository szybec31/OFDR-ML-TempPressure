from sklearn.kernel_approximation import Nystroem
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


def train_nystroem_svr(X_train, y_train, config):
    """
    Scalable RBF approximation:
    StandardScaler -> Nystroem(RBF) -> MultiOutput LinearSVR

    random_state is fixed to make the kernel approximation reproducible.
    """

    random_state = config.get("random_state", 42)

    base_model = LinearSVR(
        C=config.get("C", 10),
        epsilon=config.get("epsilon", 0.05),
        max_iter=config.get("max_iter", 20000),
        random_state=random_state,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(
            kernel="rbf",
            gamma=config.get("gamma", 0.1),
            n_components=config.get("n_components", 300),
            random_state=random_state,
        )),
        ("svr", MultiOutputRegressor(base_model)),
    ])

    model.fit(X_train, y_train)

    return model