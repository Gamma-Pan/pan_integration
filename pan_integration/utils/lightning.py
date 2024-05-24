from typing import Any

from lightning import LightningModule
import torch
from torch import nn, Tensor
from torch.profiler import profile, record_function, ProfilerActivity

class LitOdeClassifier(LightningModule):
    def __init__(
            self,
            embedding: nn.Module,
            ode_model: nn.Module,
            classifier: nn.Module,
    ):
        super().__init__()
        self.ode_model = ode_model
        self.embedding = embedding
        self.classifier = classifier
        self.automatic_optimization = False

        self.nfe = 0
        # self.save_hyperparameters()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x_em = self.embedding(x)
        _, y_hat = self.ode_model(x_em)

        logits = self.classifier(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        self.log(f"nfe_forward", float(self.ode_model.vf.nfe), prog_bar=True)
        self.ode_model.vf.nfe = 0

        _, preds = torch.max(logits, dim=1)
        acc = torch.sum(preds == y) / y.shape[0]

        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        # return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        # return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.ode_model.parameters(), lr=0.0005)
