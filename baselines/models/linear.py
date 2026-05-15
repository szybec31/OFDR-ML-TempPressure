# models/linear.py
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_linear(X_train, y_train):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    model.fit(X_train, y_train)
    return model