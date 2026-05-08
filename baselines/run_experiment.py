import numpy as np
from sklearn.metrics import f1_score
from .utils.remake_config import clean_model_config
from .utils.metrics import evaluate

def run_experiment(df, y, split, **config):
	# ========================
	# model: str = "logistic" or "svm" or "random_forest" or "mlp"
	# models: list = up to 2 models from list above (only for "late-fusion")
	# balanced: bool = True or False
	# balanced_list: list[bool] = list of using balanced params in models (only for "late-fusion")
	# threshold: float = 0.2, 0.3, 0.5 (only for tfidf vectorizer), base value = 0.5 for tfidf or None (for other vect)
	# thresholds: list[float] = list of thresholds for late-fusion when using min one model based on tfidf vectorizer
	# n_estimators: int = base 200, for random_forest
    # max_depth: int = base 20, for random_forest
	# max_features_rf: str = base 'sqrt', for random_forest
	# hidden_layer_sizes: tuple = base (256, 128), for mlp
    # max_iter: int = base 20, for mlp
    # batch_size: int = base 64, for mlp
    # learning_rate_init: float = base 0.001, for mlp
	# ========================


	# ========================
	# VALIDATE all data
    # ========================
	if "models" not in config:
		if "model" not in config:
			raise ValueError("Choose model")
		config["models"] = [config["model"]]
	
	if "thresholds" not in config:
		config["thresholds"] = []
		if "threshold" in config:
			config["thresholds"] = [config["threshold"]]
		else:
			config["thresholds"] = [None, None]
			
	if "balanced_list" not in config:
		if "balanced" in config:
			config["balanced_list"] = [config["balanced"]]
		else:
			config["balanced_list"] = [False, False]

	## LISTS:
	X = df[config["df_value"]]

    # ========================
    # SPLIT
    # ========================
	train_idx, test_idx = split

	X_train = X.iloc[train_idx]
	X_test  = X.iloc[test_idx]

	y_train = y[train_idx]
	y_test  = y[test_idx]

    # ========================
    # MODELS AND PREDICTIONS
    # ========================
	
	preds = []
	evaluations = []
	info = []

	for i, model_name in enumerate(config["models"]):
		print(f"Model: {model_name}")

		Xtr = X_train
		Xte = X_test

		if model_name == "linear":
			from .models.linear import train_linear
			model = train_linear(Xtr, y_train)
			y_pred = model.predict(Xte)

		elif model_name == "svm":
			from .models.svm import train_svm
			model = train_svm(Xtr, y_train, config["balanced_list"][i], **clean_model_config(config, ["balanced"]))
			y_pred = model.predict(Xte)

		elif model_name == "random_forest":
			from .models.randomforest import train_random_forest
			print(f"Balanced: {config["balanced_list"][i]}")
			model = train_random_forest(Xtr, y_train, balanced=config["balanced_list"][i], **clean_model_config(config, ["balanced"]))
			y_pred = model.predict(Xte)
			y_proba = model.predict_proba(Xte)

		elif model_name == "mlp":
			from .models.mlp import train_mlp
			model = train_mlp(Xtr, y_train, **config)
			y_pred = model.predict(Xte)
			y_proba = model.predict_proba(Xte)

		else:
			raise ValueError("Unknown model")
		
		info.append(f"Model: {model_name}")
		evaluations.append(evaluate(y_test, y_pred))
		preds.append(y_pred)

	return evaluations