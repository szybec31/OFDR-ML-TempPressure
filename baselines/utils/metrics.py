from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np

def evaluate(y_true, y_pred):

    results = {}

    target_names = ["pressure", "dT"]

    for i, name in enumerate(target_names):

        yt = y_true.iloc[:, i]
        yp = y_pred[:, i]

        mse = mean_squared_error(yt, yp)

        results[f"{name}_mae"] = mean_absolute_error(yt, yp)
        results[f"{name}_rmse"] = np.sqrt(mse)
        results[f"{name}_r2"] = r2_score(yt, yp)

    return results