import os, itertools
import numpy as np
import torch.utils.data
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
import torchmetrics
import matplotlib as mpl

mpl.use("TkAgg", force=True)
import matplotlib.pyplot as plt


def mnist_dataloaders(
    path: str = "/data/mnist",
    train: bool = False,
    val: bool = False,
    test: bool = False,
) -> list:
    assert (val and train) or not val, "validation loader only if train loader"
    dataset_path = os.getcwd() + path
    # if already downloaded, don't download
    download_dataset = not os.path.exists(dataset_path)
    out = [None, None, None]

    # create test dataloader if test == true
    if test:
        test_dataset = MNIST(
            dataset_path,
            download=download_dataset,
            train=False,
            transform=ToTensor(),
        )
        test_loader = DataLoader(
            test_dataset, num_workers=4, batch_size=64, shuffle=True
        )
        out[2] = test_loader

    if train:
        train_dataset = MNIST(
            dataset_path, download=download_dataset, train=False, transform=ToTensor()
        )

        if not val:
            train_loader = DataLoader(
                train_dataset, num_workers=4, batch_size=64, shuffle=True
            )
            out[0] = train_loader
        else:
            # create subsets for train and validation
            val_percent = 0.1
            train_dataset_size = int(len(train_dataset) * (1 - val_percent))
            val_dataset_size = int(len(train_dataset) * val_percent)
            generator = torch.Generator().manual_seed(42)

            train_subset, val_subset = utils.data.random_split(
                train_dataset,
                [train_dataset_size, val_dataset_size],
                generator=generator,
            )

            train_loader = DataLoader(train_subset, num_workers=4, batch_size=64)
            val_loader = DataLoader(
                val_subset, num_workers=4, batch_size=64, shuffle=False
            )
            out[0] = train_loader
            out[1] = val_loader

    return out


seq_model = nn.Sequential(
    nn.Conv2d(1, 64, 5, 1, 2),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(64),
    nn.Conv2d(64, 128, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.BatchNorm2d(128),
    nn.Flatten(),
    nn.Linear(128 * 7 * 7, 32),
    nn.ReLU(),
    nn.Linear(32, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.ln1 = nn.Linear(128 * 7 * 7, 100)
        self.ln2 = nn.Linear(100, 100)
        self.ln2_1 = nn.Linear(100, 100)
        self.ln3 = nn.Linear(100, 32)
        self.ln4 = nn.Linear(32, 10)

    def forward(self, x):
        # first res block
        x = self.bn1(F.max_pool2d(F.relu(self.conv1(x)), 2, 2))
        x = x + self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.max_pool2d((F.relu(self.conv3(x))), 2, 2))
        x = torch.flatten(x, start_dim=1)  # maintain batch dimension
        x = F.relu(self.ln1(x))
        x = x + F.relu(self.ln2(x))
        x = x + F.relu(self.ln2_1(x))
        x = F.relu(self.ln3(x))
        x = self.ln4(x)
        return x


class LitConv(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        # variables to hold image tensors for logging
        self.epoch_images = {
            "best": None,
            "worst": None,
            "random": None,
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.loss(logits, y)
        acc = self.acc_metric(logits, y)

        self.log("train_acc", acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # keep individual losses to compare
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits, y)

        # get scalar loss by mean over batch
        loss = torch.mean(losses)
        acc = self.acc_metric(logits, y)
        self.log("val_acc", acc)
        self.log("val_loss", loss)

        return x, y, logits, losses

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.acc_metric(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-4)


class ImageLog(pl.callbacks.Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        x, y, logits, losses = outputs
        x = x.cpu()
        y = y.cpu()
        logits = logits.cpu()
        probabilities = F.softmax(logits)
        losses = losses.cpu()

        # append the previous losses
        all_losses = torch.cat(
            (
                losses,
                pl_module.epoch_images["best"]["loss"],
                pl_module.epoch_images["worst"]["loss"],
            ),
            dim=0,
        )
        all_x = torch.cat(
            (
                x,
                pl_module.epoch_images["best"]["x"],
                pl_module.epoch_images["worst"]["x"],
            ),
            dim=0,
        )
        all_probabilities = torch.cat(
            (
                probabilities,
                pl_module.epoch_images["best"]["probabilities"],
                pl_module.epoch_images["worst"]["probabilities"],
            ),
            dim=0,
        )
        all_y = torch.cat(
            (
                y,
                pl_module.epoch_images["best"]["y"],
                pl_module.epoch_images["worst"]["y"],
            ),
            dim=0,
        )

        # find top 3 losses to get indices
        _, top_indices = torch.topk(all_losses, 3,)

        # find bottom 3 losses
        _, bottom_indices = torch.topk(all_losses, 3, largest=False)

        pl_module.epoch_images["best"] = {
            "x": all_x[top_indices, ...],
            "y": all_y[top_indices, ...],
            "probabilities": all_probabilities[top_indices, ...],
            "loss": all_losses[top_indices, ...],
        }

        pl_module.epoch_images["worst"] = {
            "x": all_x[bottom_indices, ...],
            "y": all_y[bottom_indices, ...],
            "probabilities": all_probabilities[bottom_indices, ...],
            "loss": all_losses[bottom_indices, ...],
        }

    def on_validation_epoch_start(self, trainer, pl_module):
        # get a batch from the dataloader
        x, y = next(iter(trainer.val_dataloaders))
        # move x to device to run inference
        x = x[0:3, ...].to(pl_module.device)  # keep first three examples
        y = y[0:3, ...].cpu()
        # make prediction for first 3 and move to cpu
        logits = pl_module.forward(x[0:3, ...]).cpu()
        probabilities = F.softmax(logits)
        # keep individual losses to compare
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fn(logits, y)

        image = {
            "x": x[0:3, ...].cpu(),
            "y": y[0:3, ...],
            "probabilities" : probabilities,
            "loss": losses,
        }

        pl_module.epoch_images = {"best": image, "worst": image, "random": image}

    def on_validation_epoch_end(self, trainer, pl_module):
        logger = trainer.logger
        epoch_images = pl_module.epoch_images

        # create 9 row, one for each image
        columns = list(epoch_images['best'].keys())
        data = []
        for image_dict in epoch_images.values():
            for idx in range(3):
                row = [ tensor[idx, ...] for tensor in image_dict.values()]
                row[0] = wandb.Image(row[0])
                data.append(row)

        logger.log_table(key='top_bottom_random', columns=columns, data=data)


def train_model(model, epochs: int, log: bool = False) -> None:
    # run = wandb.init(project="Testing")

    loaders = mnist_dataloaders(train=True, val=True, test=True)
    train_loader, val_loader, test_loader = loaders

    # lightning module
    my_lit_module = LitConv(model)
    wandb_logger = WandbLogger(project="Testing") if log else None

    # callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
    )
    progress_bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=10)
    draw_on_epoch_end = ImageLog()

    # trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, progress_bar_callback, draw_on_epoch_end],
    )
    trainer.fit(
        model=my_lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    trainer.test(model=my_lit_module, dataloaders=test_loader)

    wandb.finish()


if __name__ == "__main__":
    my_model = MyModel()
    train_model(my_model, epochs=50, log=True)
