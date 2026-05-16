import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class PhysicalModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        # Domyślne wartości czułości (typowe dla K-SHF), żeby nie było dzielenia przez zero, jeśli brakuje
        # kalibracji w foldzie
        self.Ax = -0.0008698317120730003
        self.Bx = -1.401584526610069
        self.Ay = -0.0007269860998043833
        self.By = -1.3960459119883708
        self.Cx = 0.708356118866369
        self.Cy = -1.9765915670555956

    def fit(self, X, y):
        sig_x = X["mu_X"].values
        sig_y = X["mu_Y"].values
        p_true = y.iloc[:, 0].values
        t_true = y.iloc[:, 1].values

        mask_temp = X["is_temp_calibration"] == True
        print("is_temp_calibration",sum(mask_temp))

        if np.any(mask_temp):
            T = t_true[mask_temp]
            feat_T = np.column_stack([T ** 2, T])
            self.Ax, self.Bx = LinearRegression(fit_intercept=False).fit(feat_T, sig_x[mask_temp]).coef_
            self.Ay, self.By = LinearRegression(fit_intercept=False).fit(feat_T, sig_y[mask_temp]).coef_

        mask_press = X["is_pressure_calibration"] == True
        print("is_pressure_calibration",sum(mask_press))
        if np.any(mask_press):
            P = p_true[mask_press].reshape(-1, 1)
            self.Cx = LinearRegression(fit_intercept=False).fit(P, sig_x[mask_press]).coef_[0]
            self.Cy = LinearRegression(fit_intercept=False).fit(P, sig_y[mask_press]).coef_[0]

        return self

    def predict(self, X):

        print("Parametry Modelu analitycznego:")
        print(self.Ax, self.Ay)
        print(self.Bx, self.By)
        print(self.Cx, self.Cy)

        if isinstance(X, pd.DataFrame):
            mu_X = X["mu_X"].values
            mu_Y = X["mu_Y"].values
        else:
            mu_X = X[:, 1]
            mu_Y = X[:, 0]

        preds = []
        for mx, my in zip(mu_X, mu_Y):
            denom_x = self.Cx if abs(self.Cx) > 1e-12 else 0.708356118866369
            denom_y = self.Cy if abs(self.Cy) > 1e-12 else -1.9765915670555956

            a = (self.Ay / denom_y) - (self.Ax / denom_x)
            b = (self.By / denom_y) - (self.Bx / denom_x)
            c = (mx / denom_x) - (my / denom_y)

            delta = b ** 2 - 4 * a * c
            if delta < 0:
                T = 0
            else:
                T1 = (-b + np.sqrt(delta)) / (2 * a)
                T2 = (-b - np.sqrt(delta)) / (2 * a)
                candidates = [T1, T2]
                valid = [t for t in candidates if -50 <= t <= 100]

                if len(valid) == 0:
                    T = T1
                else:
                    T = valid[0]

            p = (mx - self.Ax * T ** 2 - self.Bx * T) / denom_x
            preds.append([p, T])

        return np.array(preds)


def train_physical_model(X, y):
    return PhysicalModel().fit(X, y)