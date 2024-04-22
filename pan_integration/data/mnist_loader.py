import torch.utils.data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch import utils
from torch.utils.data import DataLoader, random_split
import os
import lightning as L
from math import floor


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = "./data/mnist", batch_size: int = 64, num_workers: int = 1
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [transforms.ToTensor(),  transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            dataset_sz = len(mnist_full)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [floor(dataset_sz * 0.9), floor(dataset_sz * 0.1)],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)

