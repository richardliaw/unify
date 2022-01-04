from ray.tune.integration.xgboost import TuneReportCheckpointCallback

from xgboost_ray import RayDMatrix, RayParams, train
from sklearn.datasets import load_breast_cancer


def train_model(config=None, callback=None):
    train_x, train_y = load_breast_cancer(return_X_y=True)
    train_set = RayDMatrix(train_x, train_y)

    evals_result = {}
    bst = train(
        {"objective": "binary:logistic", "eval_metric": ["logloss", "error"],},
        train_set,
        evals_result=evals_result,
        evals=[(train_set, "train")],
        verbose_eval=False,
        ray_params=RayParams(num_actors=2, cpus_per_actor=1),  # Number of remote actors
        callbacks=callback,
    )

    bst.save_model("model.xgb")


def tune_model():
    analysis = tune.run(
        lambda cfg: train_model(
            cfg, callback=TuneReportCheckpointCallback(filename="model.xgb")
        )
    )
    print("best config", analysis.best_config)
    print("best checkpoint", analysis.best_checkpoint)
    return result


def serve_model():
    pass
    # import requests

    # import ray
    # from ray import serve

    # serve.start()

    # @serve.deployment
    # class Counter:
    #     def __init__(self):
    #         self.count = 0

    #     def __call__(self, *args):
    #         self.count += 1
    #         return {"count": self.count}

    # # Deploy our class.
    # Counter.deploy()

    # # Query our endpoint in two different ways: from HTTP and from Python.
    # assert requests.get("http://127.0.0.1:8000/Counter").json() == {"count": 1}
    # assert ray.get(Counter.get_handle().remote()) == {"count": 2}


def main(mode):
    if mode == "train":
        train_model()

    if mode == "tune":
        tune_model()

    if mode == "serve":
        serve_model()


if __name__ == "__main__":
    import typer

    typer.run(main)
