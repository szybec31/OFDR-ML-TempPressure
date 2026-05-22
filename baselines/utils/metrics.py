from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
import numpy as np

def evaluate(y_true, y_pred):
    results = {}
    target_names = ["pressure", "dT"]

    for i, name in enumerate(target_names):
        yt = y_true.iloc[:, i].values
        yp = y_pred[:, i]

        errors = yt - yp
        mse = mean_squared_error(yt, yp)
        results[f"{name}_mae"] = mean_absolute_error(yt, yp)
        results[f"{name}_rmse"] = np.sqrt(mse)
        results[f"{name}_maxae"] = np.max(np.abs(errors))
        #results[f"{name}_medae"] = median_absolute_error(yt, yp)
        #abs_err = np.abs(yt - yp)
        #results[f"{name}_p95ae"] = np.percentile(abs_err, 95)
        #results[f"{name}_bias"] = np.mean(yp - yt)
        if np.std(yt) < 1e-6:
            results[f"{name}_r2"] = np.nan
        else:
            results[f"{name}_r2"] = r2_score(yt, yp)

    return results

