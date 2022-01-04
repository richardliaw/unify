import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pl_bolts.datamodules import MNISTDatamodule


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


def train_model(num_workers=2, callback=None):
    ray_plugin = RayPlugin(
        num_workers=num_workers, num_cpus_per_worker=1, use_gpu=False
    )
    trainer = Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        # logger=TensorBoardLogger(
        #     save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        plugins=[ray_plugin],
        callbacks=callback,
    )

    mnist_dm = MNISTDatamodule()
    model = LitMNIST(num_classes=mnist_dm.num_classes)

    trainer.fit(model, mnist_dm)


def _tune_trial(config, checkpoint_dir=None):
    callback = [
        TuneReportCheckpointCallback(
            metrics={"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"},
            filename="checkpoint",
            on="validation_end",
        )
    ]
    train_model(num_workers=config["num_workers"], callback=callback)


def tune_model():
    analysis = tune.run(_tune_trial, config={"num_workers": 2})
    print("best config", analysis.best_config)
    print("best checkpoint", analysis.best_checkpoint)
    return result


def serve_model():
    pass


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
