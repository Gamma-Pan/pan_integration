import torch.utils.data
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import utils
from torch.utils.data import DataLoader, random_split
import lightning as L
from math import floor

from sklearn.model_selection import train_test_split


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./data/mnist",
            batch_size: int = 64,
            num_workers: int = 1,
            pin_memory=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
                ),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=False, download=True)

        self.labels = []
        for sample in CIFAR10(self.data_dir, train=True, download=True):
            self.labels.append(sample[1])

    def setup(self, stage: str):
        if stage == "fit":
            mnist_train = CIFAR10(
                self.data_dir, train=True, transform=self.train_transform
            )
            mnist_val = CIFAR10(self.data_dir, train=True, transform=self.test_transform)

            dataset_sz = len(mnist_train)
            (
                ind_train,
                ind_val,
            ) = train_test_split(
                torch.arange(dataset_sz),
                stratify=self.labels,
                test_size=0.08,
                random_state=42,
            )
            self.mnist_train = torch.utils.data.Subset(mnist_train, ind_train)
            self.mnist_val = torch.utils.data.Subset(mnist_val, ind_val)

        if stage == "test":
            self.mnist_test = CIFAR10(
                self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
