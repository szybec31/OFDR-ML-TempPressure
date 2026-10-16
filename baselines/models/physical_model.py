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

    pressure_calibration:
        "strict"              - tylko rekordy dT = 0 (dokładnie jak w artykule),
        "nearest_temperature" - gdy brak dT = 0, używa punktów ciśnieniowych z najniższej
                                dostępnej temperatury treningowej i odejmuje od nich część
                                temperaturową wyznaczoną z kalibracji temperaturowej.
    """

    def __init__(self, pressure_calibration="strict"):
        self.pressure_calibration = pressure_calibration

    def fit(self, X, y):
        sig_x = X["mu_X"].to_numpy()
        sig_y = X["mu_Y"].to_numpy()
        p_true = y["pressure"].to_numpy()
        t_true = y["dT"].to_numpy()

        mask_temp = (X["is_temp_calibration"] == True).to_numpy()
        if np.unique(t_true[mask_temp]).size < 2:
            raise NotIdentifiableError("za mało poziomów temperatury do kalibracji A, B")

        T = t_true[mask_temp]
        feat_T = np.column_stack([T ** 2, T])
        self.Ax_, self.Bx_ = LinearRegression(fit_intercept=False).fit(feat_T, sig_x[mask_temp]).coef_
        self.Ay_, self.By_ = LinearRegression(fit_intercept=False).fit(feat_T, sig_y[mask_temp]).coef_

        mask_press = (X["is_pressure_calibration"] == True).to_numpy()
        self.pressure_calibration_dT_ = 0.0

        if not np.any(mask_press):
            if self.pressure_calibration != "nearest_temperature":
                raise NotIdentifiableError("brak rekordów kalibracji ciśnieniowej (dT = 0)")

            joint = (X["is_joint_regression"] == True).to_numpy()
            if not np.any(joint):
                raise NotIdentifiableError("brak rekordów ciśnieniowych w danych treningowych")
            t_ref = np.min(t_true[joint])
            mask_press = joint & (t_true == t_ref)
            self.pressure_calibration_dT_ = float(t_ref)

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
            "pressure_calibration_dT": self.pressure_calibration_dT_,
        }

    def predict(self, X):
        mu_X = X["mu_X"].to_numpy()
        mu_Y = X["mu_Y"].to_numpy()

        # eliminacja p:  dv_x/Cx - dv_y/Cy = a*T^2 + b*T
        a = self.Ax_ / self.Cx_ - self.Ay_ / self.Cy_
        b = self.Bx_ / self.Cx_ - self.By_ / self.Cy_
        c = -(mu_X / self.Cx_ - mu_Y / self.Cy_)

        t_lo, t_hi = self.t_train_range_
        t_mid = 0.5 * (t_lo + t_hi)

        if abs(a) < 1e-15:
            T = -c / b
        else:
            delta = b ** 2 - 4 * a * c
            sqrt_d = np.sqrt(np.clip(delta, 0, None))
            T1 = (-b + sqrt_d) / (2 * a)
            T2 = (-b - sqrt_d) / (2 * a)
            # wybór pierwiastka bliższego zakresowi temperatur treningowych
            T = np.where(np.abs(T1 - t_mid) <= np.abs(T2 - t_mid), T1, T2)
            # brak pierwiastków rzeczywistych -> wierzchołek paraboli (najbliższe rozwiązanie)
            T = np.where(delta < 0, -b / (2 * a), T)
            self.n_no_real_root_ = int(np.sum(delta < 0))

        p = (mu_X - self.Ax_ * T ** 2 - self.Bx_ * T) / self.Cx_
        return np.column_stack([p, T])


def train_physical_model(X, y, pressure_calibration="strict"):
    return PhysicalModel(pressure_calibration=pressure_calibration).fit(X, y)
