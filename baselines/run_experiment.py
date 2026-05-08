import numpy as np
from sklearn.metrics import f1_score
from .utils.remake_config import clean_model_config
from .utils.metrics import evaluate

def run_experiment(df, y, split, **config):
	# ========================
	# model: str = "logistic" or "svm" or "random_forest" or "mlp"
	# models: list = up to 2 models from list above (only for "late-fusion")
	# n_estimators: int = base 200, for random_forest
    # max_depth: int = base 20, for random_forest
	# max_features_rf: str = base 'sqrt', for random_forest
	# ========================


	# ========================
	# VALIDATE all data
    # ========================
	if "models" not in config:
		if "model" not in config:
			raise ValueError("Choose model")
		config["models"] = [config["model"]]
	

	## LISTS:
	X = df[config["df_value"]]

    # ========================
    # SPLIT
    # ========================
	train_idx, test_idx = split

	X_train = X.iloc[train_idx]
	X_test  = X.iloc[test_idx]

	y_train = y.iloc[train_idx]
	y_test  = y.iloc[test_idx]

    # ========================
    # MODELS
    # ========================
	evaluations = []

	for model_name in config["models"]:

		print(f"Model: {model_name}")

		if model_name == "MO-LR":
			from .models.linear import train_linear
			model = train_linear(X_train, y_train)

		elif model_name == "POLY2-RIDGE":
			from .models.poly2_ridge import train_poly2_ridge
			model = train_poly2_ridge(X_train, y_train, config)

		elif model_name == "SVR-RBF":
			from .models.svr_rbf import train_svr_rbf
			model = train_svr_rbf(X_train, y_train, config)

		elif model_name == "RF":
			from .models.randomforest import train_random_forest
			model = train_random_forest(X_train, y_train, config)

		# elif model_name == "AN-BL":
		# 	from .models.physical_model import AnalyticalBaseline
		# 	model = AnalyticalBaseline()
		# 	model.fit(X_train, y_train)

		else:
			raise ValueError("Unknown model")

		y_pred = model.predict(X_test)

		evaluations.append(
			evaluate(y_test, y_pred)
        )

	return evaluations