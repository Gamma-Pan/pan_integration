import numpy as np
import torch, torchvision
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from torch.utils.data import RandomSampler, DataLoader

from nde_squared.data import mnist_dataloaders
from nde_squared.models.auto_encoders import Encoder, Decoder

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List

import plotly.express as px

mpl.use("TkAgg")


class LitVAE(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = torch.nn.BCELoss()
        self.lr = 3e-4

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        h = self.encoder(x)
        x_hat = self.decoder(h)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


class VizEmbeddings(pl.Callback):
    def __init__(self, plot_interval=20):
        self.plot_interval = plot_interval
        super().__init__()

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.num_classes = 10
        pl_module.indices = np.empty((pl_module.num_classes,), dtype=torch.Tensor)
        # initialize a figure to plot the 2d embedings of images
        colors = [
            "#5733FF",
            "#FF5733",
            "#33FF57",
            "#FF33A6",
            "#A6FF33",
            "#33A6FF",
            "#FF3366",
            "#3366FF",
            "#EDD605",
            "#FF6633",
        ]
        pl_module.fig_emb, pl_module.axes_emb = plt.subplots()
        pl_module.scatters = np.empty(
            pl_module.num_classes, dtype=mpl.collections.PathCollection
        )

        # get a 1000 random datapoint that will be plotted each epoch/step, they will be more or less evenly distributed
        dataset = trainer.train_dataloader.dataset
        random_sampler = RandomSampler(dataset)
        random_dataloader = DataLoader(dataset, batch_size=1000, sampler=random_sampler)
        pl_module.imgs, pl_module.classes = next(iter(random_dataloader))

        pl_module.axes_emb.set_xlim([-1, 1])
        pl_module.axes_emb.set_ylim([-1, 1])

        for idx in range(pl_module.num_classes):
            # init artists
            pl_module.scatters[idx] = pl_module.axes_emb.scatter(
                [], [], c=colors[idx], label=idx
            )
            # get indices for each class
            pl_module.indices[idx] = (pl_module.classes == idx).nonzero().squeeze()

        pl_module.axes_emb.legend(loc="upper left")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx,
    ) -> None:
        num_classes, encoder, imgs, indices, scatters = (
            pl_module.num_classes,
            pl_module.encoder,
            pl_module.imgs,
            pl_module.indices,
            pl_module.scatters,
        )
        if not batch_idx % self.plot_interval:
            with torch.no_grad():
                max_steps = len(trainer.train_dataloader)
                pl_module.fig_emb.suptitle(
                    f'step: {batch_idx}/{max_steps} epoch: {trainer.current_epoch} loss: {trainer.callback_metrics["train_loss"]:2.2f}'
                )
                # get the 2D embedding of saved images
                encoder.eval()
                embeddings = encoder(imgs.cuda()).cpu()
                encoder.train()

                # max = torch.max(embeddings)
                # min = torch.min(embeddings)
                # pl_module.axes_emb.set_xlim([min, max])
                # pl_module.axes_emb.set_ylim([min, max])

                for idx in range(num_classes):
                    # print(embeddings[indices[idx]][:2, ...])
                    scatters[idx].set_offsets(embeddings[indices[idx]])

                plt.pause(1 / 160)


class VizComparison(pl.Callback):
    def __init__(self, interval: int = 20):
        super().__init__()
        self.plot_interval = interval

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # init figure and axes
        pl_module.fig, pl_module.axes = plt.subplots(3, 3)
        pl_module.images = np.empty_like(pl_module.axes)

        for idx in range(9):
            pl_module.axes[idx // 3, idx % 3].xaxis.set_major_locator(
                ticker.NullLocator()
            )
            pl_module.axes[idx // 3, idx % 3].yaxis.set_major_locator(
                ticker.NullLocator()
            )

            pl_module.images[idx // 3][idx % 3] = pl_module.axes[
                idx // 3, idx % 3
            ].imshow(
                np.zeros((28, 56, 1), dtype=np.float32),
                vmin=0,
                vmax=1.0,
                cmap="Greys",
            )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # get 9 random images from testing trainer
        # TODO: use a variable amount of plotted images
        dataset = trainer.train_dataloader.dataset
        random_sampler = RandomSampler(dataset)
        random_dataloader = DataLoader(dataset, batch_size=10, sampler=random_sampler)
        pl_module.img_tensor = next(iter(random_dataloader))[0].detach()


    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        with torch.no_grad():
            if not batch_idx % self.plot_interval:
                max_steps = len(trainer.train_dataloader)
                pl_module.fig.suptitle(
                    f'step: {batch_idx}/{max_steps} epoch: {trainer.current_epoch} loss: {trainer.callback_metrics["train_loss"]:2.2f}'
                )
                pl_module.eval()
                reconstructions = pl_module.forward(pl_module.img_tensor.cuda()).cpu()
                pl_module.train()
                appended_imgs = torch.concat(
                    [reconstructions, pl_module.img_tensor], dim=3
                )

                for idx in range(9):
                    out_image = torch.permute(appended_imgs[idx, ...], (1, 2, 0))
                    pl_module.images[idx // 3][idx % 3].set_data(out_image)

                plt.pause(1 / 160)


if __name__ == "__main__":
    # plt.close("all")
    train_loader, validation_loader, _ = mnist_dataloaders(
        train=True, val=True, num_workers=1
    )
    encoder = Encoder()
    decoder = Decoder()
    lit_vae = LitVAE(encoder, decoder)
    callbacks = [VizEmbeddings(20)] # , VizComparison(20)]
    trainer = pl.Trainer(max_epochs=50, callbacks=callbacks, fast_dev_run=False)

    # tuner = Tuner(trainer)
    # tuner.lr_find(lit_vae, train_dataloaders=train_loader)

    trainer.fit(lit_vae, train_dataloaders=train_loader)
