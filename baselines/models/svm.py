from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

def train_svm(X_train, y_train, balanced=True, **config):
    model = OneVsRestClassifier(
        LinearSVC(
            class_weight='balanced' if balanced else None,
            max_iter=config.get("max_iter_svm", 1000)
        )
    )
    model.fit(X_train, y_train)
    return model