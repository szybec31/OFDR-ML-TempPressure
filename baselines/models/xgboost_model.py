from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **params
            )
        ))
    ])

    model.fit(X_train, y_train)
    return model