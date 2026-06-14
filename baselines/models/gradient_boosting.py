from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_gradient_boosting(X_train, y_train, params=None):
    if params is None:
        params = {}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", MultiOutputRegressor(
            GradientBoostingRegressor(
                random_state=42,
                **params
            )
        ))
    ])

    model.fit(X_train, y_train)
    return model