from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MultiOutputRegressor(MLPRegressor(
            random_state=seed,
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
        ))),
    ])
