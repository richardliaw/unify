from e2e_pt import *


def main(mode):
    trainer = DLTrainer(
        train_func=train_func,
        backend="torch",
        config={"lr": 1e-3},
        num_workers=10,
        use_gpu=True,
    )
    ds = create_dataset()

    if mode == "train":
        results = trainer.fit(
            dataset=ds,
            labels="label",
            callbacks=[JsonLoggerCallback()]
        )
        model = trainer.fitted_model

    elif mode == "tune":
        parameters = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([4, 16, 32]),
            "epochs": 3,
        }

        ## ideally:
        # tuner = tune.Tuner(ray_trainer, config, ...)
        # tuner.fit(dataset, labels)

        results = tune.run(trainer, config=parameters, dataset=ds, labels="label")
        model = results.best_model

    elif mode == "serve":
        pass


if __name__ == "__main__":
    import typer

    typer.run(main)
