from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_linear(X_train, y_train):
    # MO-LR często lepiej działa na znormalizowanych danych
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model


