from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_hist_gradient_boosting(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("hgb", MultiOutputRegressor(HistGradientBoostingRegressor(random_state=seed, max_iter=200))),
    ])
