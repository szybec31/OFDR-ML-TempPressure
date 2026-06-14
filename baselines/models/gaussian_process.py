from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_gaussian_process(X_train, y_train, params=None):
    if params is None:
        params = {}

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            optimizer=None,
            random_state=42,
            **params
        ))
    ])

    model.fit(X_train, y_train)
    return model