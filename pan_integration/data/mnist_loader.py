import torch.utils.data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import utils
from torch.utils.data import DataLoader
import os


def mnist_dataloaders(
        path: str = "/data/mnist",
        train: bool = False,
        val: bool = False,
        test: bool = False,
        num_workers: int = 1,
        batch_size: int = 64
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
            test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True
        )
        out[2] = test_loader

    if train:
        train_dataset = MNIST(
            dataset_path, download=download_dataset, train=True, transform=ToTensor()
        )

        if not val:
            train_loader = DataLoader(
                train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True
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

            train_loader = DataLoader(train_subset, num_workers=num_workers, batch_size=batch_size)
            val_loader = DataLoader(
                val_subset, num_workers=num_workers, batch_size=batch_size, shuffle=False
            )
            out[0] = train_loader
            out[1] = val_loader

    return out
