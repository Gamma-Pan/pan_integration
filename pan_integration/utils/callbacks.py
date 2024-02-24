import lightning.pytorch as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader


class VizEmbeddings(pl.Callback):
    def __init__(self, plot_interval=20):
        self.plot_interval = plot_interval
        super().__init__()

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.num_classes = 10
        pl_module.indices = np.empty((pl_module.num_classes,), dtype=torch.Tensor)
        # initialize a figure to plot the 2d embeddings of images
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
                    f"step: {batch_idx}/{max_steps} "
                    f" epoch: {trainer.current_epoch} "
                    f'loss: {trainer.callback_metrics["train_loss"]:2.2f}'
                )
                # get the 2D embedding of saved images
                pl_module.model.eval()
                embeddings = encoder(imgs.cuda()).cpu()
                pl_module.model.train()

                # update axes to fit data
                max = 1  #torch.max(embeddings)
                min = -1 #torch.min(embeddings)

                pl_module.axes_emb.set_xlim([min, max])
                pl_module.axes_emb.set_ylim([min, max])

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
                    f"step: {batch_idx}/{max_steps} "
                    f"epoch: {trainer.current_epoch} "
                    f'loss: {trainer.callback_metrics["train_loss"]:2.2f}'
                )
                pl_module.model.eval()
                reconstructions = pl_module.forward(pl_module.img_tensor.cuda()).cpu()
                pl_module.model.train()
                appended_imgs = torch.concat(
                    [reconstructions, pl_module.img_tensor], dim=3
                )

                for idx in range(9):
                    out_image = torch.permute(appended_imgs[idx, ...], (1, 2, 0))
                    pl_module.images[idx // 3][idx % 3].set_data(out_image)

                plt.pause(1 / 160)
