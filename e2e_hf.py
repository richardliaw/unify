def get_regression_trainer(
    a=0, b=0, double_output=False, train_len=64, eval_len=64, pretrained=True, **kwargs
):
    label_names = kwargs.get("label_names", None)
    train_dataset = RegressionDataset(length=train_len, label_names=label_names)
    eval_dataset = RegressionDataset(length=eval_len, label_names=label_names)

    model_init = kwargs.pop("model_init", None)
    if model_init is not None:
        model = None
    else:
        if pretrained:
            config = RegressionModelConfig(a=a, b=b, double_output=double_output)
            model = RegressionPreTrainedModel(config)
        else:
            model = RegressionModel(a=a, b=b, double_output=double_output)

    compute_metrics = kwargs.pop("compute_metrics", None)
    data_collator = kwargs.pop("data_collator", None)
    optimizers = kwargs.pop("optimizers", (None, None))
    output_dir = kwargs.pop("output_dir", "./regression")

    args = RegressionTrainingArguments(output_dir, a=a, b=b, **kwargs)
    return Trainer(
        model,
        args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        model_init=model_init,
    )


def main(mode):
    trainer = get_regression_trainer()

    if mode == "train":
        trainer.train()

    if mode == "tune":
        trainer.hyperparameter_search(backend="ray")

    if mode == "serve":
        serve_model()


if __name__ == "__main__":
    import typer

    typer.run(main)
