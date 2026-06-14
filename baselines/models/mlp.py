from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_mlp(X, y, params):

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MultiOutputRegressor(
                MLPRegressor(
                    random_state=42,
                    max_iter=5000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    **params
                )
            )
        )
    ])

    model.fit(X, y)

    return model