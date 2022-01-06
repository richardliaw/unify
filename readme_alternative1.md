## Training alternative
```python

raytrainer = train.XGBoostTrainer(
    args=[
        {"objective": "binary:logistic", "eval_metric": ["logloss", "error"],},
    ],
    kwargs=dict(verbose_eval=False),
    num_workers=10,
    use_gpus=True,
)

pipeline = ray.train.Preprocessor(
    stages=[
        ("vectorizer", Vectorizer()), 
        ("onehot", OneHotEncoding())
        ("trainer", raytrainer)
    ]
)

results = pipeline.fit( 
    train_dataset=trainset, 
    test_dataset=testset,
    target_columns=["labels"],
    callbacks=...)
model: XGBoostBooster = pipeline.fitted_model
```

## Hyperparameter tuning
```python
tuner = tune.Tuner(pipeline, tune_config)

analysis = tuner.fit(
    hp_space={...},
    preprocessor=pipeline, 
    data=train_set,
    target_columns=["labels"],
    cv=5,
    callbacks=...,
)

model: XGBoostBooster = analysis.best_model

```

## Batch processing

```python
ds = ray.data.read_parquet()
xgboost_predictor = ray.train.to_predictor(model)
pipeline.pop(-1)  # remove the trainer
inference_pipeline = pipeline.chain(xgboost_predictor)
ds = inference_pipeline.transform(ds)
```

## Serve the model
```
inference_pipeline.deploy()
```


