import torch
from torch import nn
from torch.linalg import inv, pinv
import torch.nn.functional as F

from torchdyn.core import NeuralODE

from math import prod

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import multiprocessing

from pan_integration.utils.callbacks import *
from pan_integration.data.mnist_loader import MNISTDataModule



class ILAugmenter(nn.Module):
    def __init__(self, in_ch:int=1, out_ch:int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1,1)

    def forward(self, x):
        return self.conv1(x)

class VF(nn.Module):
    def __init__(self, channels: int=16):
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

class NODEWrapper(nn.Module):
    def __init__(self, vf):
        super().__init__()
        self.node = NeuralODE(vf, return_t_eval=False)

    def forward(self, x):
        return self.node(x)[-1]


class SSAutoEncoder(LightningModule):
    def __init__(self, input_size:tuple= (1, 28, 28), latent_dim:int=2, channels:int=16):
        super().__init__()

        self.augmenter = ILAugmenter(out_ch=channels)
        self.output_layer = nn.Sequential( nn.Conv2d(channels, input_size[0], 1, 1), nn.Sigmoid())

        self.encoder = NODEWrapper(VF())
        self.decoder = NODEWrapper(VF())

        self.ss_basis = nn.Parameter(torch.randn( prod(input_size[1:])*channels ,latent_dim), requires_grad=True)

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

        x = self.augmenter(x)
        enc_out = self.encoder(x)
        # ??? how do I project tensor?
        z = enc_out.view(x.size(0), -1 )
        encoding =  (self.ss_basis @ ( inv( self.ss_basis.mT @  self.ss_basis) @ (self.ss_basis.mT @ z[...,None]) ))[...,0]

        decoding = self.decoder(encoding.reshape(*x.shape))
        decoding = self.output_layer(decoding)

        return decoding, encoding, z

    def forward(self, x):
        decoding, encoding, z= self._common_step(x)
        coordinates =  (pinv(self.ss_basis) @ encoding[...,None])[...,0]

        return decoding, coordinates

    def training_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, z = self._common_step(x)

        reconstruction_loss = F.mse_loss(decoding, x, reduction='sum')
        projection_loss = F.mse_loss(encoding, z, reduction='sum')

        self.log("reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("projection_loss", projection_loss, prog_bar=True)

        loss = reconstruction_loss + projection_loss

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, z = self._common_step(x)

        reconstruction_loss = F.mse_loss(decoding, x, reduction='sum')
        projection_loss = F.mse_loss(encoding, z, reduction='sum')

        loss = reconstruction_loss + projection_loss

        # find the subspace coordinates

        coordinates =  (pinv(self.ss_basis) @ encoding[...,None])[...,0]

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"deconding": decoding, "encoding": coordinates}

    def test_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, z = self._common_step(x)

        reconstruction_loss = F.mse_loss(decoding, x, reduction='sum')
        projection_loss = F.mse_loss(encoding, z, reduction='sum')

        loss = reconstruction_loss + projection_loss

        self.log("test_loss", loss, on_step=False, on_epoch=True)


class SubSpaceNODEProj(LightningModule):
    def __init__(self, input_size:tuple= (1, 28, 28), latent_dim:int=2, channels:int=16):
        super().__init__()

        self.augmenter = ILAugmenter(out_ch=channels)
        self.output_layer = nn.Sequential( nn.Conv2d(channels, input_size[0], 1, 1), nn.Sigmoid())

        vf = VF()
        self.encoder = NODEWrapper(vf)

        self.ss_basis = nn.Parameter(torch.randn( prod(input_size[1:])*channels ,latent_dim), requires_grad=True)

        self.learning_rate = 1e-3


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

        x = self.augmenter(x)
        enc_out = self.encoder(x)
        # ??? how do I project tensor?
        z = enc_out.view(x.size(0), -1 )
        encoding =  (self.ss_basis @ ( inv( self.ss_basis.mT @  self.ss_basis) @ (self.ss_basis.mT @ z[...,None]) ))[...,0]

        return x, encoding, z

    def forward(self, x):
        decoding, encoding, z= self._common_step(x)
        coordinates =  (pinv(self.ss_basis) @ encoding[...,None])[...,0]

        return decoding, coordinates

    def training_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, z = self._common_step(x)

        # reconstruction_loss = F.mse_loss(decoding, x, reduction='mean')
        projection_loss = F.mse_loss(encoding, z, reduction='mean')

        # self.log("reconstruction_loss", reconstruction_loss, prog_bar=True)
        self.log("projection_loss", projection_loss, prog_bar=True)

        loss = projection_loss

        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, z = self._common_step(x)

        # reconstruction_loss = F.mse_loss(decoding, x, reduction='sum')
        projection_loss = F.mse_loss(encoding, z, reduction='sum')

        loss = projection_loss

        # find the subspace coordinates

        coordinates =  (pinv(self.ss_basis) @ encoding[...,None])[...,0]

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"deconding": decoding, "encoding": coordinates}

    def test_step(self, batch, batch_idx):
        x, label = batch
        decoding, encoding, z = self._common_step(x)

        reconstruction_loss = F.mse_loss(decoding, x, reduction='sum')
        projection_loss = F.mse_loss(encoding, z, reduction='sum')

        loss = reconstruction_loss + projection_loss

        self.log("test_loss", loss, on_step=False, on_epoch=True)


if __name__ == '__main__':

    num_workers = multiprocessing.cpu_count()

    mnist_module = MNISTDataModule(num_workers=num_workers, batch_size=128)
    wandb_logger = WandbLogger(project="ode_encoder")
    callbacks = [
        # DigitsReconstruction(num_digits=24),
        LearningRateMonitor(logging_interval="epoch"),
        LatentSpace()
    ]

    # model = SSAutoEncoder(input_size=(1, 32, 32), latent_dim=2, channels=16)
    model = SubSpaceNODEProj(input_size=(1,32,32), latent_dim=2, channels=16)

    trainer = Trainer(
        fast_dev_run=False,
        max_epochs=100,
        enable_checkpointing=False,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=mnist_module)





