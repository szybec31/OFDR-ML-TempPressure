from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_gradient_boosting(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", MultiOutputRegressor(GradientBoostingRegressor(random_state=seed))),
    ])
