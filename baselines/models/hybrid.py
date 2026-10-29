import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .physical_model import JointPhysicalModel


class HybridPhysicalModel(BaseEstimator, RegressorMixin):
    """
    Model hybrydowy: fizyka + ML na resztach.

        [p, T] = AN-JOINT(mu_X, mu_Y) + r(z),     z = [pozostałe cechy, p_phys, T_phys]

    1. AN-JOINT (mu_k = A_k T^2 + B_k T + C_k p) kalibrowany na danych treningowych folda daje
       predykcję fizyczną p_phys, T_phys - to ona odpowiada za ekstrapolację (krzywizna T^2).
    2. Model resztowy r uczy się błędu fizyki (y - predykcja fizyczna) z cech, których fizyka
       nie używa (d, std, IQR) oraz z samych predykcji fizycznych.
       residual = "ridge": Ridge na standaryzowanych cechach, osobny dla p i T; duże alpha -> r = stała
                  (średnia reszta), czyli AN-JOINT ze stałą poprawką (strojenie wewnętrzne może
                  wybrać "prawie samą fizykę").
       residual = "gp":    proces gaussowski z jądrem RBF (ARD) + szum, osobny dla p i T;
                  hiperparametry jądra z maksymalizacji wiarygodności. Granice jądra (skala >= 0,3
                  odchylenia cechy, szum >= 1e-3 wariancji reszt) nie pozwalają zapamiętać offsetów
                  poszczególnych serii (10 szybkich powtórzeń jest prawie identycznych). Daleko od danych
                  treningowych poprawka maleje do zera, więc model wraca do fizyki.

    Reszty liczone są w próbie treningowej: AN-JOINT ma 6 współczynników przy ~1000 rekordach,
    więc optymizm reszt jest pomijalny.

    Wejście X: DataFrame z kolumnami mu_X, mu_Y (dla fizyki) i dowolnymi innymi cechami (dla r).
    y: [ciśnienie, temperatura]. t_ref: jak w JointPhysicalModel.
    """

    def __init__(self, residual="ridge", alpha=1.0, use_physical_predictions=True,
                 gp_restarts=0, t_ref=0.0, random_state=42):
        self.residual = residual
        self.alpha = alpha
        self.use_physical_predictions = use_physical_predictions
        self.gp_restarts = gp_restarts
        self.t_ref = t_ref
        self.random_state = random_state

    def _residual_inputs(self, X, base):
        other = X.drop(columns=["mu_X", "mu_Y"]).to_numpy(dtype=float)
        if self.use_physical_predictions:
            other = np.column_stack([other, base])
        if other.shape[1] == 0:
            raise ValueError("HybridPhysicalModel: brak cech dla modelu resztowego")
        return other

    def _build_residual(self, n_features):
        if self.residual == "ridge":
            return Pipeline([("scaler", StandardScaler()),
                             ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state))])
        if self.residual == "gp":
            kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                      * RBF(length_scale=np.ones(n_features), length_scale_bounds=(0.3, 1e3))
                      + WhiteKernel(1e-1, (1e-3, 1e1)))
            return Pipeline([("scaler", StandardScaler()),
                             ("gp", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                             n_restarts_optimizer=self.gp_restarts,
                                                             random_state=self.random_state))])
        raise ValueError(f"residual = {self.residual!r}; dozwolone: 'ridge', 'gp'")

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HybridPhysicalModel wymaga DataFrame z kolumnami mu_X, mu_Y")
        y = np.asarray(y, dtype=float)
        self.physical_ = JointPhysicalModel(t_ref=self.t_ref).fit(X[["mu_X", "mu_Y"]], y)
        base = self.physical_.predict(X[["mu_X", "mu_Y"]])
        Z = self._residual_inputs(X, base)
        resid = y - base
        # osobny model dla każdego celu: reszty p i T mają inną skalę i strukturę
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)    # skale jądra na granicy (cechy nieistotne)
            self.residual_models_ = [self._build_residual(Z.shape[1]).fit(Z, resid[:, j])
                                     for j in range(y.shape[1])]
        return self

    def predict(self, X):
        base = self.physical_.predict(X[["mu_X", "mu_Y"]])
        Z = self._residual_inputs(X, base)
        corr = np.column_stack([m.predict(Z) for m in self.residual_models_])
        return base + corr
