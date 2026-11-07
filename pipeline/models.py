"""Rejestr modeli: nazwa -> sposób budowy, siatka hiperparametrów i rodzaj danych treningowych."""
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from baselines.models.hybrid import HybridPhysicalModel
from baselines.models.linear import build_linear
from baselines.models.nystroem_svr import build_nystroem_svr
from baselines.models.physical_model import JointPhysicalModel, PhysicalModel
from baselines.models.poly2_ridge import build_poly2_ridge
from baselines.models.randomforest import build_random_forest
from baselines.models.svr_rbf import build_svr_rbf


@dataclass
class ModelSpec:
    name: str
    build: Callable            # build(seed) -> nienauczony estymator
    kind: str = "ml"           # "ml": uczony na cechach; "physical": korzysta z flag kalibracyjnych
    grid: Optional[dict] = None
    description: str = ""
    extra: dict = field(default_factory=dict)
    features: Optional[list] = None   # stały zestaw cech modelu (None -> cechy z konfiguracji)
    min_states: int = 3               # minimalna liczba stanów (T, p) potrzebna do kalibracji


# Modele hybrydowe: mu_X, mu_Y -> część fizyczna (AN-JOINT), reszta -> model resztowy.
# m pominięte: to (mu_X + mu_Y) / 2, a temperaturę niesie już predykcja fizyczna.
HYBRID_FEATURES = ["mu_X", "mu_Y", "d", "std_X", "std_Y", "iqr_X", "iqr_Y"]

# Siatki = stan z końca semestru 1 (commit 2f4e5c7).
MODEL_REGISTRY = {
    "AN-BL": ModelSpec(
        "AN-BL", lambda seed: PhysicalModel(pressure_calibration="strict", temperature_calibration="strict"),
        kind="physical",
        description="Model analityczny z artykułu, kalibracja sekwencyjna; brak punktów kalibracyjnych "
                    "(domyślnie dT=0 / p=0) w treningu -> N/A",
    ),
    "AN-BL-NEAREST": ModelSpec(
        "AN-BL-NEAREST",
        lambda seed: PhysicalModel(pressure_calibration="nearest_temperature",
                                   temperature_calibration="nearest_pressure"),
        kind="physical",
        description="Jak AN-BL, ale gdy w treningu brak punktów kalibracyjnych: C z najbliższej dostępnej "
                    "temperatury, A i B z najbliższego dostępnego poziomu ciśnienia",
    ),
    "AN-JOINT": ModelSpec(
        "AN-JOINT", lambda seed: JointPhysicalModel(), features=["mu_X", "mu_Y"], min_states=3,
        description="Model fizyczny AN-BL z kalibracją łączną (MNK na wszystkich stanach T, p)",
    ),
    "CD-LIN": ModelSpec(
        "CD-LIN", build_linear, features=["m", "d"], min_states=3,
        description="Regresja liniowa na składowej wspólnej m i różnicowej d",
    ),
    "CD-POLY": ModelSpec(
        "CD-POLY", build_poly2_ridge, features=["m", "d"], min_states=6,
        grid={"ridge__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]},
        description="Wielomian 2. stopnia (ridge) na m i d",
    ),
    "HYB-JOINT-RIDGE": ModelSpec(
        "HYB-JOINT-RIDGE", lambda seed: HybridPhysicalModel(residual="ridge", random_state=seed),
        features=HYBRID_FEATURES, min_states=3,
        grid={"alpha": [0.1, 1, 10, 100, 1000, 10000, 100000]},
        description="Hybryda: AN-JOINT + Ridge na resztach (cechy d, std, IQR i predykcje fizyczne); "
                    "duże alpha = sam AN-JOINT",
    ),
    "HYB-JOINT-GP": ModelSpec(
        "HYB-JOINT-GP", lambda seed: HybridPhysicalModel(residual="gp", random_state=seed),
        features=HYBRID_FEATURES, min_states=3,
        description="Hybryda: AN-JOINT + proces gaussowski (RBF ARD + szum) na resztach; "
                    "daleko od danych poprawka -> 0, czyli powrót do fizyki",
    ),
    "MO-LR": ModelSpec("MO-LR", build_linear, description="Wielowyjściowa regresja liniowa"),
    "POLY2-RIDGE": ModelSpec(
        "POLY2-RIDGE", build_poly2_ridge, min_states=6,
        grid={"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100], "ridge__solver": ["auto", "saga"]},
    ),
    "SVR-RBF": ModelSpec(
        "SVR-RBF", build_svr_rbf, min_states=6,
        grid={
            "svr__estimator__C": [0.1, 1, 10, 100, 1000],
            "svr__estimator__epsilon": [0.001, 0.01, 0.05, 0.1, 0.2],
            "svr__estimator__gamma": ["scale", "auto", 0.01, 0.1, 1],
        },
    ),
    "NYSTROEM-SVR": ModelSpec(
        "NYSTROEM-SVR", build_nystroem_svr, min_states=6,
        grid={
            "nystroem__n_components": [100, 300, 500, 1000],
            "nystroem__gamma": [0.01, 0.1, 1],
            "svr__estimator__C": [0.1, 1, 10],
            "svr__estimator__epsilon": [0.01, 0.05, 0.1],
        },
    ),
    "RF": ModelSpec(
        "RF", build_random_forest, min_states=6,
        grid={
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 10, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2", 0.8],
        },
    ),
}


def build_estimator(spec, seed, cfg):
    """
    Nienauczony estymator; modele fizyczne dostają temperaturę referencyjną celu (t_ref).
    cfg["model_params"][nazwa] - nadpisania parametrów konstruktora (np. dla AN-BL / AN-BL-NEAREST:
    calibration_temperature, calibration_pressure, pressure_calibration, temperature_calibration).
    """
    est = spec.build(seed)
    params = (cfg.get("model_params") or {}).get(spec.name) or {}
    unknown = sorted(set(params) - set(est.get_params(deep=False)))
    if unknown:
        raise ValueError(f"model_params[{spec.name!r}]: nieznane parametry {unknown}; "
                         f"dostępne: {sorted(est.get_params(deep=False))}")
    if params:
        est.set_params(**params)
    if "t_ref" in est.get_params(deep=False):
        t_ref = cfg.get("temperature_reference", 0.0)
        if not isinstance(t_ref, (int, float)):
            raise ValueError(f"temperature_reference = {t_ref!r} nie zostało rozwiązane do liczby "
                             "(pipeline.runner.resolve_temperature_reference)")
        est.set_params(t_ref=float(t_ref))
    return est


def get_model_spec(name, grid_override=None):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Nieznany model: {name}. Dostępne: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[name]
    if grid_override is not None:
        spec = replace(spec, grid=_parse_grid(grid_override))
    return spec


def _parse_grid(grid):
    """JSON nie ma krotek - listy zagnieżdżone w siatce zamieniane są na krotki."""
    return {
        k: [tuple(v) if isinstance(v, list) else v for v in values]
        for k, values in grid.items()
    }


# ---------------------------------------------------------------- liczba parametrów, współczynniki

def count_parameters(est):
    """
    Liczba parametrów dopasowywanych na danych, używanych w predykcji (bez standaryzacji cech).
    Zwraca (liczba, rodzaj): "parametric" - liczba stała dla danego zestawu cech,
    "nonparametric" - rośnie z danymi (SVR: współczynniki dualne, GP: wagi alpha_, RF: węzły drzew).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.kernel_approximation import Nystroem
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVR

    if isinstance(est, PhysicalModel):
        return 6, "parametric"
    if isinstance(est, JointPhysicalModel):
        return (8 if est.fit_intercept else 6), "parametric"
    if isinstance(est, HybridPhysicalModel):
        n, kind = count_parameters(est.physical_)
        for m in est.residual_models_:
            k, kk = count_parameters(m)
            n += k
            kind = "nonparametric" if kk == "nonparametric" else kind
        return n, kind
    if isinstance(est, Pipeline):
        n, kind = 0, "parametric"
        for _, step in est.steps:
            k, kk = count_parameters(step)
            n += k
            kind = "nonparametric" if kk == "nonparametric" else kind
        return n, kind
    if isinstance(est, MultiOutputRegressor):
        parts = [count_parameters(e) for e in est.estimators_]
        return sum(k for k, _ in parts), ("nonparametric" if any(kk == "nonparametric" for _, kk in parts)
                                          else "parametric")
    if isinstance(est, SVR):
        return int(est.dual_coef_.size + est.intercept_.size), "nonparametric"
    if isinstance(est, GaussianProcessRegressor):
        return int(est.alpha_.size + est.kernel_.theta.size), "nonparametric"
    if isinstance(est, RandomForestRegressor):
        return int(sum(t.tree_.node_count for t in est.estimators_)), "nonparametric"
    if isinstance(est, Nystroem):
        return 0, "parametric"     # punkty bazowe to próbka danych; wagi liczy LinearSVR
    if hasattr(est, "coef_"):
        return int(np.size(est.coef_) + np.size(getattr(est, "intercept_", 0))), "parametric"
    return 0, "parametric"    # przekształcenia bez parametrów (StandardScaler, PolynomialFeatures)


def model_coefficients(model, features, targets):
    """
    Współczynniki modelu w formacie długim [{target, term, value, space}, ...]:
        modele fizyczne (AN-*)                  - A, B, C (D) dla kanałów X, Y (target = "signal"),
        Pipeline(scaler -> linear/ridge)        - przeliczone na jednostki oryginalnych cech (space = "raw"),
        Pipeline(scaler -> poly -> ridge)       - w przestrzeni cech standaryzowanych (space = "standardized").
    Dla pozostałych modeli (SVR, RF, GP, hybrydy) - pusta lista.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if isinstance(model, (PhysicalModel, JointPhysicalModel)):
        # tylko współczynniki modelu (bez t_ref i metadanych kalibracji); D tylko z wyrazem wolnym
        keys = ["Ax", "Bx", "Cx", "Ay", "By", "Cy"]
        if getattr(model, "fit_intercept", False):
            keys += ["Dx", "Dy"]
        c = model.coefficients()
        return [{"target": "signal", "term": k, "value": float(c[k]), "space": "raw"} for k in keys]
    if not isinstance(model, Pipeline):
        return []
    steps = dict(model.steps)
    final = model.steps[-1][1]
    if not hasattr(final, "coef_"):
        return []
    coef = np.atleast_2d(final.coef_)
    intercept = np.broadcast_to(np.asarray(final.intercept_, dtype=float), (coef.shape[0],))
    scaler = steps.get("scaler")
    poly = next((s for s in steps.values() if isinstance(s, PolynomialFeatures)), None)
    rows = []
    if poly is None and isinstance(scaler, StandardScaler):
        raw = coef / scaler.scale_
        raw_int = intercept - raw @ scaler.mean_
        names, space = list(features), "raw"
        coef, intercept = raw, raw_int
    else:
        names = list(poly.get_feature_names_out(features)) if poly is not None else list(features)
        space = "standardized"
    for j, t in enumerate(targets[:coef.shape[0]]):
        rows.append({"target": t, "term": "intercept", "value": float(intercept[j]), "space": space})
        rows += [{"target": t, "term": n, "value": float(c), "space": space} for n, c in zip(names, coef[j])]
    return rows
