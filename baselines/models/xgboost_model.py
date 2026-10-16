from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_xgboost(seed=42):
    # import lokalny: xgboost jest zależnością opcjonalną
    from xgboost import XGBRegressor

    return Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", MultiOutputRegressor(XGBRegressor(
            objective="reg:squarederror", random_state=seed, n_jobs=-1, verbosity=0
        ))),
    ])
