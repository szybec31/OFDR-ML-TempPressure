import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

# calculate_A_B -> zbiór treningowy w foldzie gdzie
# calculate_C -> zbiór treningowy w foldzie
# calculate_DT_p -> zbiór testowy w foldzie

'''
na przyszłość
    # PODZBIORY tylko z TRAIN
    df_temp = df_train[df_train["pressure"] < 0.05]
    df_pressure = df_train[abs(df_train["dT"]) < 0.1]

    model.calculate_A_B(df_temp)
    model.calculate_C(df_pressure)

    # TEST → tylko predykcja
    for row in df_test:
        model.calculate_DT_p(...)
'''

class PhysicalModel():

    def __init__(self):
        self.Ax = None
        self.Bx = None
        self.Ay = None
        self.By = None
        self.Cx = None
        self.Cy = None

    # ======================
    # 1. TEMPERATURA
    # ======================
    def calculate_A_B(self, df_temp):

        # df_temp: tylko pressure ≈ 0

        T = df_temp["dT"].values
        X = np.column_stack([T**2, T])

        # X channel
        model_x = LinearRegression().fit(X, df_temp["x_signal"])
        self.Ax, self.Bx = model_x.coef_

        # Y channel
        model_y = LinearRegression().fit(X, df_temp["y_signal"])
        self.Ay, self.By = model_y.coef_

    # ======================
    # 2. CIŚNIENIE
    # ======================
    def calculate_C(self, df_pressure):

        # df_pressure: tylko ΔT ≈ 0

        p = df_pressure["pressure"].values.reshape(-1, 1)

        model_x = LinearRegression().fit(p, df_pressure["x_signal"])
        self.Cx = model_x.coef_[0]

        model_y = LinearRegression().fit(p, df_pressure["y_signal"])
        self.Cy = model_y.coef_[0]

    # ======================
    # 3. ROZWIĄZANIE UKŁADU
    # ======================
    def calculate_DT_p(self, mu_X, mu_Y):

        # współczynniki równania kwadratowego
        a = (self.Ay / self.Cy) - (self.Ax / self.Cx)
        b = (self.By / self.Cy) - (self.Bx / self.Cx)
        c = (mu_Y / self.Cy) - (mu_X / self.Cx)

        # Δ = b² - 4ac
        delta = b**2 - 4*a*c

        if delta < 0:
            return None, None

        # dwa rozwiązania
        T1 = (-b + np.sqrt(delta)) / (2*a)
        T2 = (-b - np.sqrt(delta)) / (2*a)

        # wybierz sensowne (np. bliższe 0)
        T = T1 if abs(T1) < abs(T2) else T2

        # policz p
        p = (mu_X - self.Ax*T**2 - self.Bx*T) / self.Cx

        return T, p

    def fit(self, X, y):
        # Zakładamy: X ma kolumny [mu_Y, mu_X], y ma [pressure, dT]
        # Tworzymy tymczasowy DataFrame dla łatwego filtrowania
        data = pd.DataFrame({
            'mu_Y': X.iloc[:, 0], 'mu_X': X.iloc[:, 1],
            'pressure': y.iloc[:, 0], 'dT': y.iloc[:, 1]
        })

        # 1. Kalibracja Temperatury (A, B): bierzemy punkty gdzie ciśnienie ≈ 0
        df_temp = data[np.abs(data["pressure"]) == np.min(np.abs(data["pressure"]))]
        T = df_temp["dT"].values
        X_poly = np.column_stack([T ** 2, T])

        self.Ax, self.Bx = LinearRegression().fit(X_poly, df_temp["mu_X"]).coef_
        self.Ay, self.By = LinearRegression().fit(X_poly, df_temp["mu_Y"]).coef_

        # 2. Kalibracja Ciśnienia (C): bierzemy punkty gdzie zmiana temperatury ≈ 0
        df_press = data[np.abs(data["dT"]) == np.min(np.abs(data["dT"]))]
        P = df_press["pressure"].values.reshape(-1, 1)

        self.Cx = LinearRegression().fit(P, df_press["mu_X"]).coef_[0]
        self.Cy = LinearRegression().fit(P, df_press["mu_Y"]).coef_[0]

        return self

    def predict(self, X):
        results = []
        for _, row in X.iterrows():
            mu_y, mu_x = row.iloc[0], row.iloc[1]

            # Układ równań kwadratowych dla dT
            a_coeff = (self.Ay / self.Cy) - (self.Ax / self.Cx)
            b_coeff = (self.By / self.Cy) - (self.Bx / self.Cx)
            c_coeff = (mu_y / self.Cy) - (mu_x / self.Cx)

            delta = b_coeff ** 2 - 4 * a_coeff * c_coeff
            if delta < 0:
                dT = -b_coeff / (2 * a_coeff)  # Przybliżenie jeśli brak pierwiastków
            else:
                t1 = (-b_coeff + np.sqrt(delta)) / (2 * a_coeff)
                t2 = (-b_coeff - np.sqrt(delta)) / (2 * a_coeff)
                dT = t1 if abs(t1) < abs(t2) else t2

            # Wyznaczenie ciśnienia p
            p = (mu_x - self.Ax * dT ** 2 - self.Bx * dT) / self.Cx
            results.append([p, dT])

        return np.array(results)
