from typing import Any
import torch

from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import plotly.express as px
import io
from PIL import Image
from sklearn.manifold import TSNE


class AutoEncoderViz(Callback):
    def __init__(self, batches_for_latent:int =10):
        super().__init__()
        self.batches_for_latent = batches_for_latent
        self.encodings = []
        self.labels = []

    def on_fit_start(self, trainer, pl_module ):
        self.batch = next(iter(trainer.datamodule.val_dataloader()))

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        self.encodings_in = []
        self.labels = []
        self.encodings_out =  []

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
            (node_in, node_out), label = outputs
            self.encodings_in.append(node_in.cpu())
            self.encodings_out.append(node_out.cpu())
            self.labels.append(label.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        points_in = torch.cat(self.encodings_in, dim=0)
        points_out = torch.cat(self.encodings_out, dim=0)
        labels = [str(x.item()) for x in torch.cat(self.labels, dim=0)]

        embeds_in = TSNE(n_components=2).fit_transform(points_in)
        embeds_out = TSNE(n_components=2).fit_transform(points_out)

        fig_1 = px.scatter( x=embeds_in[:,0], y=points_in[:,1], color=labels )
        fig_1.update_traces(marker=dict(size=10, opacity=0.6))
        img_bytes = fig_1.to_image(format="png", width=1200, height=1000)
        img = Image.open(io.BytesIO(img_bytes))
        trainer.logger.log_image(key="latent_in", images=[img])

        fig_2 = px.scatter( x=embeds_out[:,0], y=points_out[:,1], color=labels )
        fig_2.update_traces(marker=dict(size=10, opacity=0.6))

        img_bytes = fig_2.to_image(format="png", width=1200, height=1000)
        img = Image.open(io.BytesIO(img_bytes))
        trainer.logger.log_image(key="latent_out", images=[img])
