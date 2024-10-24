import functools
from functools import reduce
from math import pow, ceil, floor
from typing import Any

import torch
from lightning import LightningModule
from torch import nn, exp
from torchdyn.core import NeuralODE

from torchmetrics import Accuracy
from torchmetrics.functional import kl_divergence


class ConvEncoder(nn.Module):
    def __init__(self, channels, latent_dims, input_size):
        super().__init__()

        in_channels, height, width = input_size
        side = height

        enc_modules = []
        # calculate the size of each encoder layer output
        for ch_in, ch_out in zip([in_channels] + channels[:-1], channels[:-1]):
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        ch_in, ch_out, kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                ),
            )
            side = ceil(side / 2)

        # last layer is output channels
        enc_modules.append(
            nn.Sequential(
                nn.Conv2d(
                    channels[-2], channels[-1], kernel_size=3, padding=1, stride=2
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
        )
        side = floor(side / 2)

        self.conv = nn.Sequential(*enc_modules)

        conv_out_sz = int(channels[-1] * side**2)
        self.fc = nn.Linear(conv_out_sz, latent_dims)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class VarConvEncoder(nn.Module):
    def __init__(self, channels, latent_dims, input_size):
        super().__init__()

        in_channels, height, width = input_size
        side = height

        enc_modules = []
        # calculate the size of each encoder layer output
        for ch_in, ch_out in zip([in_channels] + channels[:-1], channels[:-1]):
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        ch_in, ch_out, kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                ),
            )
            side = ceil(side / 2)

        # last layer is output channels
        enc_modules.append(
            nn.Sequential(
                nn.Conv2d(
                    channels[-2], channels[-1], kernel_size=3, padding=1, stride=2
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
        )
        side = floor(side / 2)

        self.conv = nn.Sequential(*enc_modules)

        conv_out_sz = int(channels[-1] * side**2)
        self.fc_mu = nn.Linear(conv_out_sz, latent_dims)
        self.fc_sigma = nn.Linear(conv_out_sz, latent_dims)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma


class ConvDecoder(nn.Module):
    def __init__(self, channels, latent_dims, input_size):
        super().__init__()

        out_channels, height, width = input_size
        side = height
        paddings = []

        for _ in channels:
            paddings.append(1 - (side % 2))
            side = ceil(side / 2)

        self.conv_in_sz = (channels[0], side, side)  # calculate the output paddings

        self.fc = nn.Linear(
            latent_dims, functools.reduce(lambda a, b: a * b, self.conv_in_sz)
        )

        dec_modules = []
        for pad, ch_in, ch_out in zip(paddings[::-1], channels, channels[1:]):
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        output_padding=pad,
                        bias=False,
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                ),
            )

        dec_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels[-1],
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    output_padding=paddings[0],
                    bias=False,
                ),
                nn.Sigmoid(),
            ),
        )

        self.conv = nn.Sequential(*dec_modules)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, *self.conv_in_sz)
        return self.conv(x)


class AutoEncoder(LightningModule):
    def __init__(self, channels, latent_dims, output_size):
        super().__init__()
        self.encoder = ConvEncoder(channels, latent_dims, output_size)
        self.decoder = ConvDecoder(channels[::-1], latent_dims, output_size)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = 1e-3

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=0.9, last_epoch=-1
            ),
            "interval": "epoch",
            "frequency": 8,
        }
        out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        return out

    def _common_step(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)

        return decoding, encoding

    def forward(self, x):
        return self._common_step(x)

    def training_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        return {"deconding": decoding, "encoding": encoding}

    def test_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("test_loss", loss, on_step=False, on_epoch=True)


def var_loss(x, x_hat, mu, sigma):
    reconstruct_loss = torch.nn.functional.mse_loss(x, x_hat, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    return reconstruct_loss, kl_divergence


class VarAutoEncoder(LightningModule):
    def __init__(self, channels, latent_dims, output_size):
        super().__init__()
        self.encoder = VarConvEncoder(channels, latent_dims, output_size)
        self.mu_fc = nn.Linear(latent_dims, latent_dims)
        self.sigma_fc = nn.Linear(latent_dims, latent_dims)

        self.decoder = ConvDecoder(channels[::-1], latent_dims, output_size)
        self.loss_fn = var_loss
        self.learning_rate = 1e-3

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=0.9, last_epoch=-1
            ),
            "interval": "epoch",
            "frequency": 8,
        }
        out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        return out

    def _common_step(self, x):
        mu, sigma = self.encoder(x)

        # reparametrization
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        encoding = mu + eps * std

        decoding = self.decoder(encoding)

        return decoding, encoding, mu, sigma

    def forward(self, x):
        decoding, encoding, mu, singma = self._common_step(x)
        return decoding, encoding

    def training_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, mu, sigma = self._common_step(x)
        reconstruction_loss, kl_divergence = self.loss_fn(decoding, x, mu, sigma)
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("kl_divergence", kl_divergence)

        loss = reconstruction_loss + kl_divergence

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, mu, sigma = self._common_step(x)
        r, k = self.loss_fn(decoding, x, mu, sigma)
        loss = r + k

        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        return {"deconding": decoding, "encoding": encoding}

    def test_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, mu, sigma = self._common_step(x)
        r, k = self.loss_fn(decoding, x, mu, sigma)
        loss = r + k
        self.log("test_loss", loss, on_step=False, on_epoch=True)


class ODEAutoEncoderFC(LightningModule):
    def __init__(self, latent_dim, input_size):

        self.channels, self.height, self.width = input_size
        self.flat_size = self.channels * self.height * self.width

        super().__init__()
        encoder_vf = nn.Sequential(
            nn.Linear(self.flat_size, self.flat_size),
            nn.Softplus(),
        )
        decoder_vf = nn.Sequential(
            nn.Linear(self.flat_size, self.flat_size),
            nn.Softplus(),
        )

        self.latent_dim = latent_dim

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, latent_dim),
        )

        self.deprojector = nn.Sequential(
            nn.Linear(latent_dim, self.flat_size), nn.Dropout(0.1)
        )
        self.encoder = NeuralODE(encoder_vf, return_t_eval=False)
        self.decoder = NeuralODE(decoder_vf, return_t_eval=False)

        self.learning_rate = 1e-3
        self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=5, cooldown=3, min_lr=5e-7
            ),
            "monitor": "training_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        return out

    def _common_step(self, x):
        encoding_big = self.encoder(x.reshape(x.size(0), -1))[-1]

        encoding = self.projector(encoding_big)
        encoding_out = self.deprojector(encoding)

        decoding = self.decoder(encoding_out)[-1].reshape(
            x.size(0), self.channels, self.width, self.height
        )

        return decoding, encoding

    def forward(self, x):
        return self._common_step(x)

    def training_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        return {"deconding": decoding, "encoding": encoding}

    def test_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("test_loss", loss, on_step=False, on_epoch=True)


class ODEAutoEncoderConv(LightningModule):
    def __init__(self, channels, aug_dim, latent_dim, input_size):

        self.in_channels, self.height, self.width = input_size
        self.channels = channels

        self.aug_dim = aug_dim
        self.latent_dim = latent_dim
        super().__init__()

        encoder_vf = nn.Sequential(
            nn.BatchNorm2d(aug_dim),
            nn.Conv2d(aug_dim, aug_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.BatchNorm2d(aug_dim),
            nn.Conv2d(aug_dim, aug_dim, kernel_size=3, padding=1),
        )
        self.encoder = NeuralODE(encoder_vf, return_t_eval=False)

        self.project = nn.Sequential(
            nn.Conv2d(aug_dim, 1, 1, padding=0),
            nn.Flatten(),
            nn.Linear(self.height * self.width, latent_dim),
        )

        self.restore = nn.Linear(latent_dim, self.height * self.width)

        decoder_vf = nn.Sequential(
            nn.BatchNorm2d(aug_dim),
            nn.Conv2d(aug_dim, aug_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.BatchNorm2d(aug_dim),
            nn.Conv2d(aug_dim, aug_dim, kernel_size=3, padding=1),
        )

        self.decoder = NeuralODE(decoder_vf, return_t_eval=False)
        self.final = nn.Conv2d(aug_dim, self.in_channels, kernel_size=3, padding=1)

        self.learning_rate = 1e-3
        self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=0.5, patience=5, cooldown=3, min_lr=5e-7
            ),
            "monitor": "training_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        return out

    def _common_step(self, x):
        padded_input = torch.cat(
            [
                x,
                torch.zeros(
                    (x.size(0), self.aug_dim - x.size(1), x.size(2), x.size(3)),
                    device=self.device,
                ),
            ],
            dim=1
        )
        encoder_out = self.encoder(padded_input)[-1]
        encoding = self.project(encoder_out)

        decoder_in = torch.cat(
            [
                self.restore(encoding).reshape(-1, 1, self.height, self.width),
                torch.zeros(
                    (x.size(0), self.aug_dim - 1, x.size(2), x.size(3)),
                    device=self.device,
                ),
            ],
            dim=1
        )
        decoder_out = self.decoder(decoder_in)[-1]
        decoding = self.final(decoder_out)

        return decoding, encoding

    def forward(self, x):
        return self._common_step(x)

    def training_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("validation_loss", loss, on_step=False, on_epoch=True)

        return {"deconding": decoding, "encoding": encoding}

    def test_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding = self._common_step(x)
        loss = self.loss_fn(decoding, x)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
