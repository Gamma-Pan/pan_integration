import torch
from torch import nn
from typing import List


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 X 28 X 28 -> 64 X 12 X 12
        self.encoder_conv = nn.Sequential(
            # convolutional
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.encoder_linear = nn.Sequential(
            # fully connected
            nn.Linear(12 * 12 * 64, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.encoder_linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_linear = nn.Sequential(
            # fully connected
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, 64 * 12 * 12),
            nn.Dropout(),
        )

        self.decoder_conv_trans = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 5),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decoder_linear(x)
        x = torch.reshape(x, (-1, 64, 12, 12))
        x = self.decoder_conv_trans(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        width: int = 28,
        height: int = 28,
        in_channels: int = 1,
        latent_dim: int = 2,
        conv_channels: List = None,
        kernel_sizes: List = None,
        linear_sizes=None,
    ):
        super().__init__()
        if linear_sizes is None:
            linear_sizes = [2048, 128, 32]
        if conv_channels is None:
            conv_channels = [8, 16, 32]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        assert len(conv_channels) == len(
            kernel_sizes
        ), "conv_channels must have same length with kernel_sizes"

        channels = [in_channels] + conv_channels
        self.last_channel = channels[-1]
        self.encoder_conv = nn.Sequential(
            *[
                self._conv_block(channels[idx], channels[idx + 1], kernel_sizes[idx])
                for idx in range(len(conv_channels))
            ]
        )

        # size of the output from the convolutional layers
        self.conv_out_size = width - sum(kernel_sizes) + len(kernel_sizes)

        all_linear_sizes = (
            [self.conv_out_size**2 * channels[-1]] + linear_sizes + [latent_dim]
        )
        self.encoder_linear = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(all_linear_sizes[idx], all_linear_sizes[idx + 1]),
                    nn.Dropout(),
                    nn.ReLU(),
                )
                for idx in range(len(all_linear_sizes) - 2)
            ],
            nn.Linear(all_linear_sizes[-2], all_linear_sizes[-1]),
            nn.Dropout(),
            nn.Tanh(),
        )

        self.decoder_linear = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(all_linear_sizes[idx], all_linear_sizes[idx - 1]),
                    nn.Dropout(),
                    nn.ReLU(),
                )
                for idx in range(len(all_linear_sizes) - 1, 0, -1)
            ]
        )

        self.decoder_conv = nn.Sequential(
            *[
                self._convT_block(
                    channels[idx], channels[idx - 1], kernel_sizes[idx - 1]
                )
                for idx in range(len(channels) - 1, 1, -1)
            ],
            nn.ConvTranspose2d(
                channels[1], channels[0], kernel_size=kernel_sizes[0], stride=1, padding=0
            ),  # size is reduced by 2
            nn.BatchNorm2d(channels[0]),
            nn.Sigmoid(),
        )

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0
            ),  # size is reduced by 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    @staticmethod
    def _convT_block(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0
            ),  # size is reduced by 2
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def encoder(self, x):
        x = self.encoder_conv(x)
        x = torch.flatten(x, start_dim=1)
        h = self.encoder_linear(x)
        return h

    def decoder(self, h):
        x_hat = self.decoder_linear(h)
        x_hat = torch.reshape(
            x_hat, (-1, self.last_channel, self.conv_out_size, self.conv_out_size)
        )
        x_hat = self.decoder_conv(x_hat)
        return x_hat

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat


if __name__ == "__main__":
    ae = Autoencoder()
    x = torch.rand(64, 1, 28, 28)
    print(ae(x).shape)
