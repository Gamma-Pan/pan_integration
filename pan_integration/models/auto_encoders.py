from functools import reduce
from math import pow, ceil
from typing import Any

import torch
from lightning import LightningModule
from torch import nn, exp
from torchdyn.core import NeuralODE

from torchmetrics import Accuracy


class Encoder(nn.Module):
    def __init__(self, channels, latent_dims, input_size):
        super().__init__()

        in_channels, height, width = input_size
        side = height

        enc_modules = []
        # calculate the size of each encoder layer
        for ch_in, ch_out in zip([in_channels] + channels, channels):
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                ),
            )
            side = int(side / 2)

        enc_modules.append(
            nn.Sequential(
                nn.Conv2d(
                    channels[-2], channels[-1], kernel_size=3, padding=1, stride=2
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
        )
        side = int(side / 2)

        # conv_out_sz = int(channels[-1] * (height / pow(2, len(channels))) ** 2)
        conv_out_sz = int(channels[-1] * side**2)

        self.fc = nn.Linear(conv_out_sz, latent_dims)

        self.conv = nn.Sequential(*enc_modules)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class VarEncoder(Encoder):
    def __init__(self, channels, latent_dims, input_size):
        super().__init__(channels, latent_dims, input_size)

        in_channels, height, width = input_size
        self.fc = None
        # conv_out_sz = int(channels[-1] * (height / pow(2, len(channels))) ** 2)
        side = height
        for _ in channels:
            side = int(side / 2)
        side = int(side / 2)
        conv_out_sz = int(channels[-1] * side ** 2)

        self.fc_mu = nn.Linear(conv_out_sz, latent_dims)
        self.fc_logvar = nn.Linear(conv_out_sz, latent_dims)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # reparametrization trick
        z = mu + exp(0.5 * logvar) * torch.randn_like(mu)

        return z, (mu, logvar)


class Decoder(nn.Module):
    def __init__(self, channels, latent_dims, output_size=(1, 32, 32)):
        super().__init__()
        out_channels, height, width = output_size

        side = int(height / pow(2, len(channels)))

        self.conv_in_size = [channels[0], side, side]
        self.fc = nn.Linear(
            latent_dims, int(reduce(lambda a, b: a * b, self.conv_in_size))
        )

        dec_modules = []
        # calculate the size of each encoder layer
        for ch_in, ch_out in zip(channels, channels[1:]):
            dec_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        ch_in,
                        ch_out,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                ),
            )
        dec_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    channels[-1],
                    out_channels,
                    kernel_size=2,
                    stride=2,
                ),
                nn.Sigmoid(),
            )
        )
        self.conv = nn.Sequential(*dec_modules)

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, *self.conv_in_size)
        return self.conv(x)


class AutoEncoder(LightningModule):
    def __init__(self, channels, latent_dim, input_size):
        super().__init__()
        self.encoder = Encoder(channels, latent_dim, input_size)
        self.decoder = Decoder(channels[::-1], latent_dim, input_size)

        self.learning_rate = 1e-3
        self.loss_fn = torch.nn.BCELoss()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=0.9, last_epoch=-1
            ),
            "interval": "epoch",
            "frequency": 5,
        }
        out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        return out

    def _common_step(self, x):
        enc = self.encoder(x)
        x_hat = self.decoder(enc)
        return x_hat, enc

    def forward(self, x):
        x_hat, _ = self._common_step(x)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, label = batch
        x_hat, enc = self._common_step(x)
        loss = self.loss_fn(x_hat, x)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        x_hat, encoding = self._common_step(x)
        loss = self.loss_fn(x_hat, x)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return encoding, label

    def test_step(self, batch, batch_idx):
        x, label = batch
        x_hat, encoding = self._common_step(x)
        loss = self.loss_fn(x_hat, x)

        self.val_mean_loss(loss)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            prog_bar=True,
        )


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, channels, latent_dim, input_size):
        super().__init__(channels, latent_dim, input_size)
        self.encoder = VarEncoder(channels, latent_dim, input_size)

    def _common_step(self, x):
        z, (mu, log_var) = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z, (mu, log_var)

    def forward(self, x):
        x_hat, z, _ = self._common_step(x)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, label = batch
        x_hat, enc, (mu, log_sigma) = self._common_step(x)
        reconstruct_loss = torch.nn.functional.binary_cross_entropy(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        loss = reconstruct_loss + kl_divergence

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        x_hat, encoding, (mu, log_sigma) = self._common_step(x)

        reconstruct_loss = torch.nn.functional.binary_cross_entropy(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        loss = reconstruct_loss + kl_divergence

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return encoding, label

    def test_step(self, batch, batch_idx):
        x, label = batch
        x_hat, encoding, (mu, log_sigma) = self._common_step(x)

        reconstruct_loss = torch.nn.BCELoss(x_hat, x)
        kl_divergence = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        loss = reconstruct_loss + kl_divergence

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )


class ODEAutoEncoder(AutoEncoder):
    def __init__(self, channels, latent_dim, input_size):
        super().__init__(channels, latent_dim, input_size)

        vf = nn.Linear(latent_dim, latent_dim)
        self.ode = NeuralODE(vf, return_t_eval=False)

    def _common_step(self, x):
        enc = self.encoder(x)
        enc_hat = self.ode(enc)[-1]
        x_hat = self.decoder(enc_hat)
        return x_hat, enc



class VEC(LightningModule):
    def __init__(self, channels, latent_dim, input_size):
        super().__init__()

        self.learning_rate = 10e-4
        self.accuracy = Accuracy('multiclass', num_classes=10)

        in_channels, height, width = input_size
        side = height

        enc_modules = []
        # calculate the size of each encoder layer
        for ch_in, ch_out in zip([in_channels] + channels, channels):
            enc_modules.append(
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                ),
            )
            side = max(1,int(side / 2))

        enc_modules.append(
            nn.Sequential(
                nn.Conv2d(
                    channels[-2], channels[-1], kernel_size=3, padding=1, stride=2
                ),
                nn.ReLU(),
                nn.Flatten(),
            )
        )
        side = max(1, int(side / 2))

        # conv_out_sz = int(channels[-1] * (height / pow(2, len(channels))) ** 2)
        conv_out_sz = int(channels[-1] * side**2)

        self.fc_mu = nn.Linear(conv_out_sz, latent_dim)
        self.fc_logvar = nn.Linear(conv_out_sz, latent_dim)

        self.conv = nn.Sequential(*enc_modules)

        vf = nn.Sequential( nn.Linear(latent_dim, latent_dim) )
        self.node = NeuralODE(vf, return_t_eval=False)

        self.classifier = nn.Linear(latent_dim, 10)

    def _common_step(self, x):
        x = self.conv(x)
        node_in = self.fc_mu(x)
        node_out = self.node(node_in)[-1]

        # logvar = self.fc_logvar(x)

        # reparametrization trick
        # z = mu + exp(0.5 * logvar) * torch.randn_like(mu)
        z = node_out #+ exp(0.5 * logvar) * torch.randn_like(mu)

        logit = self.classifier(z)
        return logit, (node_in, node_out )

    # def forward(self, x):
    #     logit, z, _ = self._common_step(x)
    #     return logit

    def training_step(self, batch, batch_idx):
        x, label = batch
        logit, _ = self._common_step(x)

        classification_loss = torch.nn.functional.cross_entropy(logit, label)
        # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        self.log("class_loss", classification_loss)
        # self.log("kld", kl_divergence)

        loss = classification_loss #+ kl_divergence

        acc = self.accuracy(logit, label)

        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return loss


    def validation_step(self, batch, batch_idx):
        x, label = batch
        logit,(node_in, node_out) = self._common_step(x)

        classification_loss = torch.nn.functional.cross_entropy(logit, label)
        # kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # self.log("kld", kl_divergence)

        loss = classification_loss #+ kl_divergence

        acc = self.accuracy(logit, label)

        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
        )
        return (node_in, node_out), label

    # def test_step(self, batch, batch_idx):
    #     x, label = batch
    #     x_hat, encoding, (mu, log_sigma) = self._common_step(x)
    #
    #     reconstruct_loss = torch.nn.BCELoss(x_hat, x)
    #     kl_divergence = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    #     loss = reconstruct_loss + kl_divergence
    #
    #     self.log(
    #         "test_loss",
    #         loss,
    #         on_epoch=True,
    #         prog_bar=True,
    #     )

    def configure_optimizers(self):
        # opt = torch.optim.AdamW(
        #     self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4
        # )
        # lr_scheduler_config = {
        #     "scheduler": torch.optim.lr_scheduler.ExponentialLR(
        #         opt, gamma=0.9, last_epoch=-1
        #     ),
        #     "interval": "epoch",
        #     "frequency": 10,
        # }
        # out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        # return out
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)










