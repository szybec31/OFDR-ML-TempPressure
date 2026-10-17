import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class NotIdentifiableError(RuntimeError):
    """Model nie może zostać skalibrowany na danych treningowych tego folda."""


class PhysicalModel(BaseEstimator, RegressorMixin):
    """
    Baseline analityczny z artykułu (Bednarek i in., 2026):

        dv_k = A_k * dT^2 + B_k * dT + C_k * p,   k in {X, Y}

    A_k, B_k - kalibracja temperaturowa na rekordach is_temp_calibration (p = 0),
    C_k      - kalibracja ciśnieniowa na rekordach is_pressure_calibration (dT = 0).

    Brak wartości domyślnych: jeżeli w danych treningowych brakuje rekordów kalibracyjnych,
    fit() zgłasza NotIdentifiableError, a fold jest raportowany jako N/A.

    y: [ciśnienie, temperatura] (po pozycji - kolumną temperatury może być dT albo Tp).
    t_ref: temperatura referencyjna celu temperaturowego; model liczy w T - t_ref (dla dT: 0,
           dla Tp: temperatura płyty w pomiarze referencyjnym). Predykcje wracają w jednostkach celu.

    pressure_calibration:
        "strict"              - tylko rekordy dT = 0 (dokładnie jak w artykule),
        "nearest_temperature" - gdy brak dT = 0, używa punktów ciśnieniowych z najniższej
                                dostępnej temperatury treningowej i odejmuje od nich część
                                temperaturową wyznaczoną z kalibracji temperaturowej.
    """

    def __init__(self, pressure_calibration="strict", t_ref=0.0):
        self.pressure_calibration = pressure_calibration
        self.t_ref = t_ref

    def fit(self, X, y):
        sig_x = X["mu_X"].to_numpy()
        sig_y = X["mu_Y"].to_numpy()
        y = np.asarray(y, dtype=float)
        p_true = y[:, 0]
        t_true = y[:, 1] - self.t_ref

        mask_temp = (X["is_temp_calibration"] == True).to_numpy()
        if np.unique(t_true[mask_temp]).size < 2:
            raise NotIdentifiableError("za mało poziomów temperatury do kalibracji A, B")

        T = t_true[mask_temp]
        feat_T = np.column_stack([T ** 2, T])
        self.Ax_, self.Bx_ = LinearRegression(fit_intercept=False).fit(feat_T, sig_x[mask_temp]).coef_
        self.Ay_, self.By_ = LinearRegression(fit_intercept=False).fit(feat_T, sig_y[mask_temp]).coef_

        mask_press = (X["is_pressure_calibration"] == True).to_numpy()
        self.pressure_calibration_T_ = 0.0

        if not np.any(mask_press):
            if self.pressure_calibration != "nearest_temperature":
                raise NotIdentifiableError("brak rekordów kalibracji ciśnieniowej (dT = 0)")

            joint = (X["is_joint_regression"] == True).to_numpy()
            if not np.any(joint):
                raise NotIdentifiableError("brak rekordów ciśnieniowych w danych treningowych")
            t_low = np.min(t_true[joint])
            mask_press = joint & (t_true == t_low)
            self.pressure_calibration_T_ = float(t_low)

        P = p_true[mask_press]
        if np.unique(P).size < 2:
            raise NotIdentifiableError("za mało poziomów ciśnienia do kalibracji C")

        T_p = t_true[mask_press]
        # odejmujemy część temperaturową (dla dT = 0 jest równa zeru)
        res_x = sig_x[mask_press] - (self.Ax_ * T_p ** 2 + self.Bx_ * T_p)
        res_y = sig_y[mask_press] - (self.Ay_ * T_p ** 2 + self.By_ * T_p)
        self.Cx_ = LinearRegression(fit_intercept=False).fit(P.reshape(-1, 1), res_x).coef_[0]
        self.Cy_ = LinearRegression(fit_intercept=False).fit(P.reshape(-1, 1), res_y).coef_[0]

        self.t_train_range_ = (float(np.min(t_true)), float(np.max(t_true)))
        return self

    def coefficients(self):
        return {
            "Ax": self.Ax_, "Bx": self.Bx_, "Cx": self.Cx_,
            "Ay": self.Ay_, "By": self.By_, "Cy": self.Cy_,
            "t_ref": self.t_ref,
            # temperatura (T - t_ref) punktów użytych do kalibracji ciśnieniowej
            "pressure_calibration_T": self.pressure_calibration_T_,
        }

    def predict(self, X):
        p, T, self.n_no_real_root_ = invert_quadratic(
            X["mu_X"].to_numpy(), X["mu_Y"].to_numpy(),
            self.Ax_, self.Bx_, self.Cx_, self.Ay_, self.By_, self.Cy_,
            t_mid=0.5 * sum(self.t_train_range_),
        )
        return np.column_stack([p, T + self.t_ref])


def invert_quadratic(mu_X, mu_Y, Ax, Bx, Cx, Ay, By, Cy, t_mid):
    """
    Odwrócenie układu  mu_k = A_k T^2 + B_k T + C_k p  (k = X, Y) względem (p, T).

    Działa elementowo z broadcastingiem (np. współczynniki per okno przestrzenne o kształcie (W,)
    i sygnały (n, W)). Zwraca (p, T, liczba przypadków bez pierwiastka rzeczywistego).
    """
    # eliminacja p:  mu_x/Cx - mu_y/Cy = a*T^2 + b*T
    a = Ax / Cx - Ay / Cy
    b = Bx / Cx - By / Cy
    c = -(mu_X / Cx - mu_Y / Cy)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = b ** 2 - 4 * a * c
        sqrt_d = np.sqrt(np.clip(delta, 0, None))
        T1 = (-b + sqrt_d) / (2 * a)
        T2 = (-b - sqrt_d) / (2 * a)
        # wybór pierwiastka bliższego środkowi zakresu temperatur treningowych
        T = np.where(np.abs(T1 - t_mid) <= np.abs(T2 - t_mid), T1, T2)
        # brak pierwiastków rzeczywistych -> wierzchołek paraboli (najbliższe rozwiązanie)
        T = np.where(delta < 0, -b / (2 * a), T)
        # człon kwadratowy pomijalny -> równanie liniowe
        T = np.where(np.abs(a) < 1e-15, -c / b, T)
        p = (mu_X - Ax * T ** 2 - Bx * T) / Cx
    n_no_root = int(np.sum((delta < 0) & (np.abs(a) >= 1e-15)))
    return p, T, n_no_root


class JointPhysicalModel(BaseEstimator, RegressorMixin):
    """
    AN-JOINT: ten sam model fizyczny co AN-BL,

        mu_k = A_k * dT^2 + B_k * dT + C_k * p  (+ D_k gdy fit_intercept),   k in {X, Y},

    ale wszystkie współczynniki dopasowane JEDNOCZEŚNIE metodą najmniejszych kwadratów na
    dowolnym zbiorze stanów kalibracyjnych (T, p) - nie wymaga osobnych serii p = 0 i dT = 0.
    Minimalny zestaw: 3 stany (4 z wyrazem wolnym) dające macierz [dT^2, dT, p] pełnego rzędu.

    Wejście X: kolumny [mu_X, mu_Y] (w tej kolejności), y: [ciśnienie, temperatura].
    t_ref: jak w PhysicalModel - model liczy w T - t_ref.
    """

    def __init__(self, fit_intercept=False, t_ref=0.0):
        self.fit_intercept = fit_intercept
        self.t_ref = t_ref

    def _design(self, T, p):
        cols = [T ** 2, T, p] + ([np.ones_like(T)] if self.fit_intercept else [])
        return np.column_stack(cols)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        p, T = y[:, 0], y[:, 1] - self.t_ref
        F = self._design(T, p)
        if np.linalg.matrix_rank(F) < F.shape[1]:
            raise NotIdentifiableError("stany kalibracyjne nie wyznaczają współczynników A, B, C")
        coef, *_ = np.linalg.lstsq(F, X[:, :2], rcond=None)      # (k, 2): kolumny X, Y
        self.Ax_, self.Bx_, self.Cx_ = coef[:3, 0]
        self.Ay_, self.By_, self.Cy_ = coef[:3, 1]
        self.Dx_, self.Dy_ = (coef[3, 0], coef[3, 1]) if self.fit_intercept else (0.0, 0.0)
        self.t_train_range_ = (float(np.min(T)), float(np.max(T)))
        return self

    def coefficients(self):
        return {"Ax": self.Ax_, "Bx": self.Bx_, "Cx": self.Cx_, "Dx": self.Dx_,
                "Ay": self.Ay_, "By": self.By_, "Cy": self.Cy_, "Dy": self.Dy_, "t_ref": self.t_ref}

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        p, T, self.n_no_real_root_ = invert_quadratic(
            X[:, 0] - self.Dx_, X[:, 1] - self.Dy_,
            self.Ax_, self.Bx_, self.Cx_, self.Ay_, self.By_, self.Cy_,
            t_mid=0.5 * sum(self.t_train_range_),
        )
        return np.column_stack([p, T + self.t_ref])


def train_physical_model(X, y, pressure_calibration="strict", t_ref=0.0):
    return PhysicalModel(pressure_calibration=pressure_calibration, t_ref=t_ref).fit(X, y)
