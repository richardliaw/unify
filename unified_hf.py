def get_ray_trainer(
    a=0, b=0, double_output=False, train_len=64, eval_len=64, pretrained=True, **kwargs
):
    compute_metrics = kwargs.pop("compute_metrics", None)
    data_collator = kwargs.pop("data_collator", None)
    optimizers = kwargs.pop("optimizers", (None, None))
    output_dir = kwargs.pop("output_dir", "./regression")

    trainer_config = dict(
        data_collator=data_collator,
        # train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        model_init=model_init,
    )

    args = RegressionTrainingArguments(output_dir, a=a, b=b, **kwargs)

    if pretrained:
        config = RegressionModelConfig(a=a, b=b, double_output=double_output)
        model = RegressionPreTrainedModel(config)
    else:
        model = RegressionModel(a=a, b=b, double_output=double_output)

    return HFTrainer(
        model, args, trainer_config, num_workers=4, use_gpu=True)

def main(mode):
    trainer = get_ray_trainer()
    ds = create_dataset()

    if mode == "train":
        results = trainer.fit(
            train_dataset=ds,
            eval_dataset=ds2,
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