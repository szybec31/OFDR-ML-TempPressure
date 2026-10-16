"""Rejestr modeli: nazwa -> sposób budowy, siatka hiperparametrów i rodzaj danych treningowych."""
from dataclasses import dataclass, field
from typing import Callable, Optional

from baselines.models.gaussian_process import build_gaussian_process
from baselines.models.gradient_boosting import build_gradient_boosting
from baselines.models.hist_gradient_boosting import build_hist_gradient_boosting
from baselines.models.kernel_ridge import build_kernel_ridge
from baselines.models.linear import build_linear
from baselines.models.mlp import build_mlp
from baselines.models.nystroem_svr import build_nystroem_svr
from baselines.models.physical_model import PhysicalModel
from baselines.models.poly2_ridge import build_poly2_ridge
from baselines.models.randomforest import build_random_forest
from baselines.models.svr_rbf import build_svr_rbf
from baselines.models.xgboost_model import build_xgboost


@dataclass
class ModelSpec:
    name: str
    build: Callable            # build(seed) -> nienauczony estymator
    kind: str = "ml"           # "ml": uczony na cechach; "physical": korzysta z flag kalibracyjnych
    grid: Optional[dict] = None
    description: str = ""
    extra: dict = field(default_factory=dict)


# Siatki = stan z końca semestru 1 (commit 2f4e5c7 "fix grid, add mlp").
MODEL_REGISTRY = {
    "AN-BL": ModelSpec(
        "AN-BL", lambda seed: PhysicalModel(pressure_calibration="strict"), kind="physical",
        description="Model analityczny z artykułu, kalibracja sekwencyjna; brak dT=0 w treningu -> N/A",
    ),
    "AN-BL-NEAREST": ModelSpec(
        "AN-BL-NEAREST", lambda seed: PhysicalModel(pressure_calibration="nearest_temperature"),
        kind="physical",
        description="Jak AN-BL, ale bez dT=0 kalibracja ciśnieniowa z najniższej dostępnej temperatury",
    ),
    "MO-LR": ModelSpec("MO-LR", build_linear, description="Wielowyjściowa regresja liniowa"),
    "POLY2-RIDGE": ModelSpec(
        "POLY2-RIDGE", build_poly2_ridge,
        grid={"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100], "ridge__solver": ["auto", "saga"]},
    ),
    "SVR-RBF": ModelSpec(
        "SVR-RBF", build_svr_rbf,
        grid={
            "svr__estimator__C": [0.1, 1, 10, 100, 1000],
            "svr__estimator__epsilon": [0.001, 0.01, 0.05, 0.1, 0.2],
            "svr__estimator__gamma": ["scale", "auto", 0.01, 0.1, 1],
        },
    ),
    "NYSTROEM-SVR": ModelSpec(
        "NYSTROEM-SVR", build_nystroem_svr,
        grid={
            "nystroem__n_components": [100, 300, 500, 1000],
            "nystroem__gamma": [0.01, 0.1, 1],
            "svr__estimator__C": [0.1, 1, 10],
            "svr__estimator__epsilon": [0.01, 0.05, 0.1],
        },
    ),
    "RF": ModelSpec(
        "RF", build_random_forest,
        grid={
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2", 0.8],
        },
    ),
    "HGBR": ModelSpec(
        "HGBR", build_hist_gradient_boosting,
        grid={
            "hgb__estimator__learning_rate": [0.01, 0.03, 0.1, 0.2],
            "hgb__estimator__max_leaf_nodes": [15, 31, 63],
            "hgb__estimator__l2_regularization": [0.0, 0.1, 1.0, 10.0],
            "hgb__estimator__min_samples_leaf": [10, 20, 40],
        },
    ),
    "GPR": ModelSpec(
        "GPR", build_gaussian_process,
        grid={"gpr__alpha": [1e-10, 1e-6, 1e-4, 1e-2, 1e-1], "gpr__n_restarts_optimizer": [0, 3, 5]},
    ),
    "KRR-RBF": ModelSpec(
        "KRR-RBF", build_kernel_ridge,
        grid={"krr__alpha": [0.001, 0.01, 0.1, 1.0, 10.0], "krr__gamma": [0.001, 0.01, 0.1, 1.0]},
    ),
    "GBR": ModelSpec(
        "GBR", build_gradient_boosting,
        grid={
            "gbr__estimator__n_estimators": [100, 300, 500],
            "gbr__estimator__learning_rate": [0.01, 0.03, 0.1],
            "gbr__estimator__max_depth": [2, 3, 5],
            "gbr__estimator__subsample": [0.7, 0.8, 1.0],
        },
    ),
    "XGBOOST": ModelSpec(
        "XGBOOST", build_xgboost,
        grid={
            "xgb__estimator__n_estimators": [100, 200, 300, 500],
            "xgb__estimator__max_depth": [2, 3, 5, 7],
            "xgb__estimator__learning_rate": [0.01, 0.03, 0.1, 0.2],
            "xgb__estimator__subsample": [0.6, 0.8, 1.0],
            "xgb__estimator__colsample_bytree": [0.6, 0.8, 1.0],
        },
    ),
    "MLP": ModelSpec(
        "MLP", build_mlp,
        grid={
            "mlp__estimator__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
            "mlp__estimator__activation": ["relu", "tanh"],
            "mlp__estimator__alpha": [1e-4, 1e-3, 1e-2],
            "mlp__estimator__learning_rate_init": [1e-3, 1e-2],
        },
    ),
}


def get_model_spec(name, grid_override=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Nieznany model: {name}. Dostępne: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[name]
    if grid_override is not None:
        spec = ModelSpec(spec.name, spec.build, spec.kind, _parse_grid(grid_override), spec.description)
    return spec


def _parse_grid(grid):
    """JSON nie ma krotek ani None w listach typu (32, 16) - konwersja list zagnieżdżonych na krotki."""
    return {
        k: [tuple(v) if isinstance(v, list) else v for v in values]
        for k, values in grid.items()
    }
