class Results:
    ...

@dataclass
class RayConfig:
    use_gpu: bool
    num_workers: int

@dataclass
class DataConfig:
    data_pipeline: Union[ray.train.Pipeline, ray.DAG]
    train_dataset: ray.data.Dataset
    test_dataset: Optional[ray.data.Dataset]
    target_columns: List[str]

class TrainerBase:
    def __init__(self, ray_config: RayConfig):

    def fit(self, data_config, **tune_config) -> Results:
        pass

    @property
    def fitted_model(self):
        return self._fitted_model



class Predictor:
    def predict(self, inputs: Arraylike) -> Arraylike:
        pass

    def save(self, checkpoint_path) -> Path:
        pass

    def load(self, checkpoint_path):
        pass

    def to_deployment(self) -> ServeDeployment:
        pass