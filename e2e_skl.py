from tune_sklearn import TuneGridSearchCV

# Other imports
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

# Set training and validation sets
X, y = make_classification(
    n_samples=11000,
    n_features=1000,
    n_informative=50,
    n_redundant=0,
    n_classes=10,
    class_sep=2.5,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000)


def main(mode):
    # we can't scale sklearn out of the box.
    clf = SGDClassifier()

    if mode == "train":
        clf.fit(X_train, y_train)

    elif mode == "tune":
        # Example parameters to tune from SGDClassifier
        parameters = {"alpha": [1e-4, 1e-1, 1], "epsilon": [0.01, 0.1]}

        tune_search = TuneGridSearchCV(
            clf, parameters, early_stopping="MedianStoppingRule", max_iters=10
        )

        tune_search.fit(X_train, y_train)

    elif mode == "serve":
        pass


if __name__ == "__main__":
    import typer

    typer.run(main)
