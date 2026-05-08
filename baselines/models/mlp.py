from sklearn.neural_network import MLPClassifier

def train_mlp(X_train, y_train, **config):
    model = MLPClassifier(
        hidden_layer_sizes = config["hidden_layer_sizes"] if "hidden_layer_sizes" in config else (256, 128),
        activation = 'relu',
        solver = 'adam',
        max_iter = config["max_iter"] if "max_iter" in config else 20,
        batch_size = config["batch_size"] if "batch_size" in config else 64,
        learning_rate_init = config["learning_rate_init"] if "learning_rate_init" in config else 0.001,
        early_stopping = True,
        random_state = 42
    )

    model.fit(X_train, y_train)
    return model