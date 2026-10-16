from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_kernel_ridge(seed=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf")),
    ])
