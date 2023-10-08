import torch
from torch import nn


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
