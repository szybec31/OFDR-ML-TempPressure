from sklearn.ensemble import RandomForestRegressor


def build_random_forest(seed=42):
    return RandomForestRegressor(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=seed)


def train_random_forest(X_train, y_train, config):
    return build_random_forest().set_params(**config).fit(X_train, y_train)
