from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_poly2_ridge(seed=42, alpha=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("ridge", Ridge(alpha=alpha, random_state=seed)),
    ])


def train_poly2_ridge(X_train, y_train, config):
    return build_poly2_ridge(alpha=config.get("alpha", 1.0)).fit(X_train, y_train)
