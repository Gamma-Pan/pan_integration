from typing import Any
import torch

from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import plotly.express as px
import io
from PIL import Image
from sympy.physics.secondquant import wicks


class AutoEncoderViz(Callback):
    def __init__(self, batches_for_latent:int =10):
        super().__init__()
        self.batches_for_latent = batches_for_latent
        self.encodings = []
        self.labels = []

    def on_fit_start(self, trainer, pl_module ):
        self.batch = next(iter(trainer.datamodule.val_dataloader()))

    def on_validation_start(self, trainer , pl_module ):
        x, y = self.batch
        x_hat = pl_module(x.to(pl_module.device))
        x_hat = x_hat.cpu()

        imgs = [torch.cat([x_i, x_hat_i], dim=-1)[0] for (x_i,x_hat_i) in  zip(x[:9], x_hat[:9])]

        trainer.logger.log_image(key="reconstruction_samples", images=imgs )

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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
        if batch_idx < self.batches_for_latent:
            encodings, label = outputs
            self.encodings.append(encodings.cpu())
            self.labels.append(label.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        points = torch.cat(self.encodings, dim=0)
        labels = [str(x.item()) for x in torch.cat(self.labels, dim=0)]

        fig = px.scatter( x=points[:,0], y=points[:,1], color=labels )
        fig.update_traces(marker=dict(size=10, opacity=0.6))
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img = Image.open(io.BytesIO(img_bytes))
        trainer.logger.log_image(key="latent_space", images=[img])
