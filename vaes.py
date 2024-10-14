from typing import Any

import lightning
import torch
import torch.nn.functional as F
from click import progressbar
from torch import nn

from lightning import LightningModule, Trainer

import torchmetrics as metrics

import plotly.express as px

from math import pow, ceil

from torchmetrics import MeanMetric

from pan_integration.data import MNISTDataModule

from torchdyn.core import NeuralODE


# %% plot callback
class PlotCallback(lightning.Callback):
    def on_test_end(self, trainer, lit_module):
        ds = trainer.test_dataloaders.dataset
        generator = torch.Generator().manual_seed(torch.randint(0, 10_000, (1,)).item())
        sampler = torch.utils.data.RandomSampler(ds, num_samples=9, generator=generator)
        plot_loader = torch.utils.data.DataLoader(ds, batch_size=9, sampler=sampler)
        imgs, labels = next(iter(plot_loader))
        imgs = imgs.to(lit_module.device)

        y_hat = lit_module(imgs)

        fig = px.imshow(
            imgs[:9, 0, ...].cpu(),
            facet_col=0,
            facet_col_wrap=3,
            binary_string=True,
            facet_col_spacing=0.001,
        )

        def labelizer(a):
            idx = int(a.text.split("=")[1])
            text1 = f"gred = {y_hat[idx].argmax().item()}"
            text2 = f"true= {labels[idx]}"
            text = text1 + "|" + text2
            a.update(text=text)

        fig.for_each_annotation(labelizer)
        fig.show()


# %% vanilla vae module


class Augmenter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x


class VF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.nfe = 0

        self.norm1 = nn.GroupNorm(channels, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(channels, channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        x = self.norm1(x)
        x = self.conv1(x)
        x = F.softplus(x)
        x = self.conv2(x)
        x = F.softplus(x)
        x = self.norm2(x)
        x = self.conv3(x)
        return x


class Classifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, 1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(6 * 16, 10)

    def forward(self, x):
        # x = self.drop(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin1(x)
        return x


class LitOdeClassifier(LightningModule):
    def __init__(
        self,
        t_span,
        embedding: nn.Module,
        ode_model: nn.Module,
        classifier: nn.Module,
    ):
        super().__init__()
        self.t_span = t_span
        self.ode_model = ode_model
        self.embedding = embedding
        self.classifier = classifier
        self.metric = metrics.classification.Accuracy("multiclass", num_classes=10)
        self.test_mean_acc = MeanMetric()

    def _common_step(self, x):
        x_em = self.embedding(x)

        _, y_hat = self.ode_model(x_em, t_span=self.t_span)

        logits = self.classifier(y_hat[-1])
        return logits

    def forward(self, x):
        return self._common_step(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self._common_step(x)
        loss = F.cross_entropy(logits, y)
        self.log("loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._common_step(x)
        test_acc = self.metric(y_hat, y)
        self.test_mean_acc(test_acc)

    def on_test_epoch_end(self):
        avg_acc = self.test_mean_acc.compute()
        self.log("test_acc", avg_acc, prog_bar=True)


# %% node vae module
class NODEVae(LightningModule):
    def __init__(self, in_dims=(1, 28, 28), latent_dim=10, channels: list = None):
        super().__init__()

        enc_modules = dict()
        enc_modules["conv"] = []
        enc_modules["fc"] = []

        # dec_modules = dict()
        # dec_modules["fc"] = []
        # dec_modules["conv"] = []

        in_channels, height, width = in_dims
        output_paddings = []

        for ch_in, ch_out in zip([in_channels] + channels, channels):
            enc_modules["conv"].append(
                nn.Sequential(
                    nn.Conv2d(
                        ch_in,
                        ch_out,
                        stride=2,
                        kernel_size=3,
                        padding=1,
                        bias=False if ch_out == channels[-1] else True,
                    ),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(),
                )
            )
            output_padding = int(not height % 2)
            height = ceil(height / 2)
            output_paddings.append(output_padding)

        # mu
        enc_modules["fc"].append(nn.Linear(height**2 * channels[-1], latent_dim))
        # sigma^2
        enc_modules["fc"].append(nn.Linear(height**2 * channels[-1], latent_dim))
        self.enc_conv = nn.Sequential(*enc_modules["conv"])
        self.enc_fcs = nn.ModuleList(enc_modules["fc"])

        # self.final_height = height
        # self.dec_fc = nn.Linear(latent_dim, height**2 * channels[-1])
        # for ch_in, ch_out, output_padding in zip(
        #     channels[::-1], channels[::-1][1:] + [in_channels], output_paddings[::-1]
        # ):
        #     dec_modules["conv"].append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(
        #                 ch_in,
        #                 ch_out,
        #                 kernel_size=3,
        #                 stride=2,
        #                 padding=1,
        #                 bias=False if ch_out == channels[-1] else True,
        #                 output_padding=output_padding,
        #             ),
        #             nn.BatchNorm2d(ch_out),
        #             nn.ReLU(),
        #         )
        #     )
        #
        # self.dec_conv = nn.Sequential(*dec_modules["conv"])

        neural_f = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus(),
            nn.Linear(latent_dim, latent_dim),
            nn.Softplus(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.node = NeuralODE(neural_f, return_t_eval=False)

        self.classifier = nn.Sequential(nn.Dropout(0.05), nn.Linear(latent_dim, 10))

        self.metric = metrics.classification.Accuracy("multiclass", num_classes=10)
        self.test_mean_acc = MeanMetric()

    def encoder(self, x):
        x = self.enc_conv(x)
        x = x.reshape(x.size(0), -1)
        mu = self.enc_fcs[0](x)
        var = self.enc_fcs[1](x)
        return mu, var

    def forward(self, x):
        y_hat, _ = self._common_step(x)
        return y_hat

    def _common_step(self, x):
        mu, log_var = self.encoder(x)

        # reparametrization trick
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(log_var * 0.5)

        y_hat = self.classifier(self.node(z)[-1])

        return y_hat, (mu, log_var)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (mu, log_var) = self._common_step(x)

        y_oh = torch.nn.functional.one_hot(y, 10).float()

        classification_loss = nn.functional.binary_cross_entropy_with_logits(
            y_hat, y_oh, reduction="sum"
        )
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - torch.exp(log_var))

        loss = classification_loss + kl_divergence
        self.log("train_loss", loss, prog_bar=True)
        self.log("classification_loss", classification_loss, prog_bar=True)
        self.log("kl_divergence", kl_divergence, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, (mu, log_var) = self._common_step(x)
        test_acc = self.metric(y_hat, y)
        self.test_mean_acc(test_acc)

    def on_test_epoch_end(self):
        avg_acc = self.test_mean_acc.compute()
        self.log("test_acc", avg_acc, prog_bar=True)


# %% train and test vae
# dmodule = MNISTDataModule(batch_size=64, num_workers=8)
# CHANNELS = [16, 16, 16]
# vae = NODEVae(latent_dim=50, channels=CHANNELS)
# trainer = Trainer(fast_dev_run=False, max_epochs=2, callbacks=[PlotCallback()])
# trainer.fit(vae, datamodule=dmodule)
# trainer.test(vae, datamodule=dmodule)

# %% train and test vanilla
dmodule = MNISTDataModule(batch_size=64, num_workers=8)
CHANNELS = 64
augmenter = Augmenter(CHANNELS)
vf = VF(64)
node = NeuralODE(vector_field=vf)
classifier = Classifier(CHANNELS)
il_ode = LitOdeClassifier(
    torch.tensor([0.0, 1.0]), embedding=augmenter, ode_model=node, classifier=classifier
)
trainer = Trainer(fast_dev_run=False, max_epochs=2, callbacks=[PlotCallback()])
trainer.fit(il_ode, datamodule=dmodule)
trainer.test(il_ode, datamodule=dmodule)
