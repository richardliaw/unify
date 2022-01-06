# External library unification

We want to make sure there's one common path for users to use Ray ML libraries, regardless of the framework.

In this repository, we prototype an end-to-end framework / API for Ray ML -- data processing, training, tuning, serving.

## Data preprocessing
```python
import ray

ds = ray.data.read_parquet("...")

trainset, testset = ray.data.train_test_split(ds, percentage=0.2)

pipeline = ray.train.Preprocessor(
    stages=[
    ("vectorizer", Vectorizer()), 
    ("onehot", OneHotEncoding())]
)

pipeline = pipeline.fit(trainset)  # updates the stateful parameters
pipeline.save("file.json")
```

## Training

```python
raytrainer = train.XGBoostTrainer(
    args=[
        {"objective": "binary:logistic", "eval_metric": ["logloss", "error"],},
    ],
    kwargs=dict(verbose_eval=False),
    num_workers=10,  # Consider making this a second level set of arguments (RayParams)
    use_gpu=True,
)

# Note: We don't need to assume that user always want to split into train/test.
# Similarly, perhaps consider a X, y interface.

results = train.fit(  # should also consider trainer.fit, given feedback
    raytrainer, 
    pipeline, 
    train_dataset=trainset, 
    test_dataset=testset,
    target_columns=["labels"],
    callbacks=...)
model: XGBoost = raytrainer.fitted_model
model: nn.Module = raytrainer.fitted_model
model: svensFavorite = raytrainer.fitted_model
```

## Hyperparameter tuning
```python
data_config = dict(
    train_pipeline=pipeline,  # we need each pipeline to be fit on the training slice of the cv.
    train_dataset=trainset,   # can also set cv=5 or something here.
    test_dataset=testset,     
    target_columns=["labels"],
)
analysis = tune.run(
    raytrainer, 
    data_config=data_config,
    config=parameters)

model = analysis.best_model

```

## Batch processing

```python
ds = ray.data.read_parquet("...")
ds.map_batches(lambda batch: pipeline.transform(batch))
ds.map_batches(lambda batch: model.predict(batch), use_gpu=True)


## ideally, we can do:
# ds = ray.data.read_parquet()
# inference_pipeline = pipeline.chain(model)
# ds = inference_pipeline.transform(ds)
```

## Serving

```python
model_deployment = ray.serve.to_deployment(model: XGBoostBooster)
model_deployment.deploy()
```

# Advanced use cases




## Libraries to consider


- [x] hf
- [x] lightgbm / xgboost
- [x] pytorch / horovod
- ptl (won't do? people can just use regular pytorch.)
- rllib (should we tie rllib in? )


