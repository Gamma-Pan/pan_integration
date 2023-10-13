import torch
import lightning.pytorch as pl
from nde_squared.data import mnist_dataloaders
from nde_squared.models.auto_encoders import Encoder, Decoder, Autoencoder
from nde_squared.utils import VizEmbeddings, VizComparison
import matplotlib as mpl
mpl.use("TkAgg")


class LitVAE(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.loss_fn = torch.nn.BCELoss()
        self.lr = 3e-4

    def forward(self, x):
        return model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)


if __name__ == "__main__":
    train_loader, validation_loader, _ = mnist_dataloaders(
        train=True, val=True, num_workers=1, batch_size=64
    )
    model = Autoencoder(conv_channels=[])
    lit_vae = LitVAE(model)
    callbacks = [VizEmbeddings(50), VizComparison(50)]

    trainer = pl.Trainer(max_epochs=50, callbacks=callbacks, fast_dev_run=False)
    trainer.fit(lit_vae, train_dataloaders=train_loader)
