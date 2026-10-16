from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression()),
    ])


def train_linear(X_train, y_train):
    return build_linear().fit(X_train, y_train)
