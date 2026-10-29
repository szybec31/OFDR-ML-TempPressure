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

    A_k, B_k - kalibracja temperaturowa na punktach o ciśnieniu kalibracyjnym (domyślnie p = 0),
    C_k      - kalibracja ciśnieniowa na punktach o temperaturze kalibracyjnej (domyślnie dT = 0).
    Punkty kalibracyjne wybierane są spośród rekordów is_joint_regression (zero_end nigdy);
    dla wartości domyślnych są to dokładnie rekordy is_temp_calibration / is_pressure_calibration.

    Brak wartości domyślnych współczynników: jeżeli w danych treningowych brakuje punktów
    kalibracyjnych (i nie włączono reguły zastępczej), fit() zgłasza NotIdentifiableError,
    a fold jest raportowany jako N/A.

    y: [ciśnienie, temperatura] (po pozycji - kolumną temperatury może być dT albo Tp).
    t_ref: temperatura referencyjna celu temperaturowego; model liczy w T - t_ref (dla dT: 0,
           dla Tp: temperatura płyty w pomiarze referencyjnym). Predykcje wracają w jednostkach celu.

    calibration_temperature: temperatura (w jednostkach celu) serii kalibracji ciśnieniowej;
                             None -> t_ref (dT = 0, jak w artykule).
    calibration_pressure:    ciśnienie [MPa] punktów kalibracji temperaturowej (domyślnie 0).

    pressure_calibration - gdy w treningu brak calibration_temperature:
        "strict"              - N/A (dokładnie jak w artykule),
        "nearest_temperature" - kalibracja C z najbliższej dostępnej temperatury treningowej.
    temperature_calibration - gdy w treningu brak calibration_pressure:
        "strict"              - N/A,
        "nearest_pressure"    - kalibracja A, B z najbliższego dostępnego poziomu ciśnienia.

    Kolejność kalibracji (część niepochodząca od kalibrowanego członu jest odejmowana):
        p_cal = 0            -> A, B (bez wyrazu wolnego), potem C po odjęciu A*T^2 + B*T,
        T_cal = t_ref        -> C, potem A, B (bez wyrazu wolnego) po odjęciu C*p_cal,
        oba różne od zera    -> A, B z wyrazem wolnym (pochłania C*p_cal), potem C jak wyżej.
    """

    def __init__(self, pressure_calibration="strict", temperature_calibration="strict",
                 calibration_temperature=None, calibration_pressure=0.0, t_ref=0.0):
        self.pressure_calibration = pressure_calibration
        self.temperature_calibration = temperature_calibration
        self.calibration_temperature = calibration_temperature
        self.calibration_pressure = calibration_pressure
        self.t_ref = t_ref

    @staticmethod
    def _pick_level(values, other, target, allow_nearest, min_other, what):
        """
        Poziom `values` użyty do kalibracji: target, jeżeli występuje w nim >= min_other różnych
        wartości `other`; w przeciwnym razie (allow_nearest) najbliższy targetowi poziom spełniający
        ten warunek (remis -> niższy poziom).
        """
        def usable(level):
            return np.unique(other[np.isclose(values, level)]).size >= min_other

        if usable(target):
            return float(target)
        if not allow_nearest:
            raise NotIdentifiableError(f"brak punktów kalibracji {what} = {target:g} w danych treningowych")
        for level in sorted(np.unique(values), key=lambda v: (abs(v - target), v)):
            if usable(level):
                return float(level)
        raise NotIdentifiableError(f"brak poziomu {what} pozwalającego na kalibrację")

    def fit(self, X, y):
        if self.pressure_calibration not in ("strict", "nearest_temperature"):
            raise ValueError(f"pressure_calibration = {self.pressure_calibration!r}")
        if self.temperature_calibration not in ("strict", "nearest_pressure"):
            raise ValueError(f"temperature_calibration = {self.temperature_calibration!r}")

        sig_x = X["mu_X"].to_numpy()
        sig_y = X["mu_Y"].to_numpy()
        y = np.asarray(y, dtype=float)
        p_true = y[:, 0]
        t_true = y[:, 1] - self.t_ref

        joint = (X["is_joint_regression"] == True).to_numpy()
        if not np.any(joint):
            raise NotIdentifiableError("brak rekordów kalibracyjnych w danych treningowych")
        t_j, p_j = t_true[joint], p_true[joint]

        t_cal = 0.0 if self.calibration_temperature is None else float(self.calibration_temperature) - self.t_ref
        p_cal = float(self.calibration_pressure)

        # A, B bez wyrazu wolnego: 2 poziomy temperatury; z wyrazem wolnym (p != 0): 3
        p_used = self._pick_level(p_j, t_j, p_cal, self.temperature_calibration == "nearest_pressure",
                                  2 if np.isclose(p_cal, 0) else 3, "ciśnienia p")
        t_used = self._pick_level(t_j, p_j, t_cal, self.pressure_calibration == "nearest_temperature",
                                  2, "temperatury T - t_ref")
        if not np.isclose(p_used, 0) and np.unique(t_j[np.isclose(p_j, p_used)]).size < 3:
            raise NotIdentifiableError("za mało poziomów temperatury do kalibracji A, B")

        mask_temp = joint & np.isclose(p_true, p_used)
        mask_press = joint & np.isclose(t_true, t_used)
        T, P_t = t_true[mask_temp], p_true[mask_temp]
        T_p, P = t_true[mask_press], p_true[mask_press]

        def fit_ab(target_x, target_y, intercept):
            feat = np.column_stack([T ** 2, T])
            return (LinearRegression(fit_intercept=intercept).fit(feat, target_x).coef_,
                    LinearRegression(fit_intercept=intercept).fit(feat, target_y).coef_)

        def fit_c(temp_part_x, temp_part_y):
            P2 = P.reshape(-1, 1)
            return (LinearRegression(fit_intercept=False).fit(P2, sig_x[mask_press] - temp_part_x).coef_[0],
                    LinearRegression(fit_intercept=False).fit(P2, sig_y[mask_press] - temp_part_y).coef_[0])

        def temp_part():
            return self.Ax_ * T_p ** 2 + self.Bx_ * T_p, self.Ay_ * T_p ** 2 + self.By_ * T_p

        if np.isclose(p_used, 0):
            (self.Ax_, self.Bx_), (self.Ay_, self.By_) = fit_ab(sig_x[mask_temp], sig_y[mask_temp], False)
            self.Cx_, self.Cy_ = fit_c(*temp_part())
        elif np.isclose(t_used, 0):
            self.Cx_, self.Cy_ = fit_c(0.0, 0.0)
            (self.Ax_, self.Bx_), (self.Ay_, self.By_) = fit_ab(
                sig_x[mask_temp] - self.Cx_ * P_t, sig_y[mask_temp] - self.Cy_ * P_t, False)
        else:
            (self.Ax_, self.Bx_), (self.Ay_, self.By_) = fit_ab(sig_x[mask_temp], sig_y[mask_temp], True)
            self.Cx_, self.Cy_ = fit_c(*temp_part())

        self.pressure_calibration_T_ = float(t_used)
        self.temperature_calibration_p_ = float(p_used)
        self.t_train_range_ = (float(np.min(t_true)), float(np.max(t_true)))
        return self

    def coefficients(self):
        return {
            "Ax": self.Ax_, "Bx": self.Bx_, "Cx": self.Cx_,
            "Ay": self.Ay_, "By": self.By_, "Cy": self.Cy_,
            "t_ref": self.t_ref,
            # temperatura (T - t_ref) punktów użytych do kalibracji ciśnieniowej (C)
            "pressure_calibration_T": self.pressure_calibration_T_,
            # ciśnienie punktów użytych do kalibracji temperaturowej (A, B)
            "temperature_calibration_p": self.temperature_calibration_p_,
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


def train_physical_model(X, y, pressure_calibration="strict", t_ref=0.0, **kwargs):
    return PhysicalModel(pressure_calibration=pressure_calibration, t_ref=t_ref, **kwargs).fit(X, y)
