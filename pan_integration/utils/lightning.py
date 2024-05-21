from lightning import LightningModule
import torch
from torch import nn
from ..core.ode import PanODE


class LitOdeClassifier(LightningModule):
    def __init__(
            self,
            embedding: nn.Module,
            ode_model: nn.Module,
            classifier: nn.Module,
            t_span: torch.Tensor = torch.linspace(0, 1, 10),
    ):
        super().__init__()
        self.t_span, self.ode_model = t_span, ode_model
        self.embedding = embedding
        self.classifier = classifier
        self.B_prev = None
        # self.save_hyperparameters()

    def _common_step(self, batch, batch_idx, log=True, update_B=False):
        x, y = batch
        x_em = self.embedding(x)
        if isinstance(self.ode_model, PanODE):
            t_eval, y_hat, metrics, B = self.ode_model(x_em, self.t_span, B_init=self.B_prev)
            if update_B:
                with torch.no_grad():
                    dims = B.shape
                    self.B_prev = torch.mean(B, 0, keepdim=True).expand(*dims)[...,2:]
            if log:
                self.log("zero_B_delta", float(metrics[0]), prog_bar=True)
                self.log("zero_iters", float(metrics[1]), prog_bar=True)
        else:
            t_eval, y_hat = self.ode_model(x_em, self.t_span)

        logits = self.classifier(y_hat[-1])
        loss = nn.CrossEntropyLoss()(logits, y)

        _, preds = torch.max(logits, dim=1)
        acc = torch.sum(preds == y) / y.shape[0]
        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx, update_B=True)
        nfe = self.ode_model.vf.nfe

        self.log("loss", loss, prog_bar=True)
        self.log("nfe", nfe, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx, log=False)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.ode_model.parameters(), lr=0.0005)
