from typing import Any

import torch
from lightning import Callback
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT
import plotly.express as px
import sklearn


class DigitsReconstruction(Callback):
    def __init__(self, num_digits: int = 8):
        super().__init__()

        self.num_digits = num_digits

    def on_fit_start(self, trainer, pl_module):
        batch = next(iter(trainer.datamodule.val_dataloader()))
        self.x = batch[0][: self.num_digits]
        self.labels = batch[1][: self.num_digits]

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        reconstructions = pl_module(self.x.to(pl_module.device))[0].cpu()

        image_tensors = torch.cat([self.x, reconstructions], dim=-1).unbind(0)
        captions = [
            f"top: original, bottom: reconstruction, label: {s}" for s in self.labels
        ]

        images = [wandb.Image(t, caption=s) for (t, s) in zip(image_tensors, captions)]

        trainer.logger.experiment.log({"reconstructions": images}, step = trainer.global_step)


class LatentSpace(Callback):
    def __init__(self, num_batches: int = 10):
        super().__init__()

        self.num_batches = num_batches

    def on_fit_start(self, trainer, pl_module):
        self.encodings = []
        self.labels = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx <= self.num_batches:
            encoding = outputs["encoding"]
            self.encodings.append(encoding.cpu())

            _, labels = batch
            self.labels.append(labels)

    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.encodings = []
        self.labels = []

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        points = torch.cat(self.encodings, dim=0).cpu()
        labels = [str(l.item()) for l in torch.cat(self.labels, dim=0).cpu()]

        # points = sklearn.decomposition.PCA().fit_transform(points)

        fig = px.scatter( x=points[:,0], y=points[:,1], color=labels )
        fig.update_traces(marker=dict(size=10, opacity=.5))

        # fig.show()
        trainer.logger.experiment.log({"latent_space": fig}, step =trainer.global_step)
