from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

def train_poly2_ridge(X_train, y_train, config):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("ridge", Ridge(
            alpha=config.get("alpha", 1.0)
        ))
    ])

    model.fit(X_train, y_train)

    return model