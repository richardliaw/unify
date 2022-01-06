from tune_sklearn import TuneGridSearchCV

# Other imports
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier


def create_dataset():
    train_x, train_y = load_breast_cancer(return_X_y=True)
    train_set = RayDMatrix(train_x, train_y)
    return train_set


def main(mode):
    raytrainer = XGBoostTrainer(,
        args=[
            {"objective": "binary:logistic", "eval_metric": ["logloss", "error"],},
        ],
        kwargs=dict(
            evals_result=evals_result,
            # evals=[(train_set, "train")],  # TODO: CHECK IF THIS IS A GOOD IDEA?
            verbose_eval=False,
            # ray_params=RayParams(
            #     num_actors=2, cpus_per_actor=1
            # ),  # Number of remote actors
            # callbacks=callback,
        ),
        num_workers=10,
        num_gpus=20,
    )
    ds = create_dataset()

    if mode == "train":
        raytrainer.fit(dataset=ds, labels="label")
        model = raytrainer.fitted_model

    elif mode == "tune":
        parameters = {
            "alpha": tune.choice([1e-4, 1e-1, 1]),
            "epsilon": tune.choice([0.01, 0.1]),
        }

        ## ideally:
        # tuner = tune.Tuner(ray_trainer, config, ...)
        # tuner.fit(dataset, labels)

        tune.run(raytrainer, config=parameters, dataset=ds, labels="label")

    elif mode == "serve":
        pass


if __name__ == "__main__":
    import typer

    typer.run(main)
