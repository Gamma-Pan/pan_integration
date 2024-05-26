from typing import Any

from lightning import LightningModule
from lightning.pytorch.callbacks import Callback

import torch
from torch import nn, Tensor
from torch.profiler import profile, record_function, ProfilerActivity


class NfeMetrics(Callback):
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        nfes = float(pl_module.ode_model.vf.nfe)
        pl_module.log(f"nfe_forward_train", nfes, prog_bar=True)
        pl_module.ode_model.vf.nfe = 0

    def on_after_backward(self, trainer, pl_module):
        nfes = float(pl_module.ode_model.vf.nfe)
        pl_module.log(f"nfe_backward_train", nfes, prog_bar=True)
        pl_module.ode_model.vf.nfe = 0

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        nfes = float(pl_module.ode_model.vf.nfe)
        pl_module.log(f"nfe_test", nfes, prog_bar=True)
        pl_module.ode_model.vf.nfe = 0


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

        self.nfe = 0

        self.learning_rate = 1e-3
        # self.save_hyperparameters()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x_em = self.embedding(x)
        _, y_hat = self.ode_model(x_em, t_span=self.t_span)

        logits = self.classifier(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

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
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # TODO: use LR callback
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=5e-5
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt),
            "monitor": "val_loss",
            "interval": "step",
            "frequency": 10,
            # "patience": 10,
        }
        out = {'optimizer': opt, 'lr_scheduler': lr_scheduler_config}
        return out


def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
    loss.backward()

    self.log("nfe_backward", float(self.ode_model.vf.nfe), prog_bar=True)
    self.ode_model.vf.nfe = 0
