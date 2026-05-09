from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, config):

    model = RandomForestRegressor(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", None),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        max_features=config.get("max_features", "sqrt"),
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model