import numpy as np
from sklearn.linear_model import LinearRegression

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