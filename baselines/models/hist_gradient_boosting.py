from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_hist_gradient_boosting(X_train, y_train, params=None):
    if params is None:
        params = {}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("hgb", MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=42,
                max_iter=200,
                **params
            )
        ))
    ])

    model.fit(X_train, y_train)
    return model