from sklearn.kernel_approximation import Nystroem
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR


def build_nystroem_svr(seed=42, C=10, epsilon=0.05, gamma=0.1, n_components=300, max_iter=20000):
    """
    Skalowalna aproksymacja jądra RBF: StandardScaler -> Nystroem(RBF) -> MultiOutput LinearSVR.
    random_state jest ustalony, więc aproksymacja jądra jest odtwarzalna.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=seed)),
        ("svr", MultiOutputRegressor(
            LinearSVR(C=C, epsilon=epsilon, max_iter=max_iter, random_state=seed)
        )),
    ])


def train_nystroem_svr(X_train, y_train, config):
    return build_nystroem_svr(seed=config.get("random_state", 42)).fit(X_train, y_train)
