from typing import Any

from lightning import LightningModule
from lightning.pytorch.callbacks import Callback

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, Tensor
from torch.profiler import profile, record_function, ProfilerActivity


class NfeMetrics(Callback):
    def __init__(self):
        self.running = 0


    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        nfes = float(pl_module.ode_model.vf.nfe)
        self.running += nfes
        pl_module.log(f"nfe_forward_train", nfes, prog_bar=True)
        pl_module.log(f"total_nfe_forward_train", self.running, prog_bar=True)
        pl_module.ode_model.vf.nfe = 0

    # def on_after_backward(self, trainer, pl_module):
    #     nfes = float(pl_module.ode_model.vf.nfe)
    #     pl_module.log(f"nfe_backward_train", nfes, prog_bar=True)
    #     pl_module.ode_model.vf.nfe = 0

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        nfes = float(pl_module.ode_model.vf.nfe)
        pl_module.log(f"nfe_test", nfes, prog_bar=True)
        pl_module.ode_model.vf.nfe = 0


class ProfilerCallback(Callback):
    def __init__(self, schedule=None, epoch=1):
        super().__init__()
        if schedule is None:
            schedule = torch.profiler.schedule(
                skip_first=5,
                wait=5,
                warmup=3,
                active=2,
                repeat=1
            )
        self.epoch = epoch

        self.profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=self.ready,
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            # record_shapes=True,
            # profile_memory=True,
        )

    def ready(self, profiler ):
        profiler.export_chrome_trace('./trace.json')

    def on_train_start(self, trainer, pl_module):
        if trainer.current_epoch == self.epoch:
            self.profiler.start()

    def on_train_end(self, trainer, pl_module ) -> None:
        if trainer.current_epoch == self.epoch:
            self.profiler.stop()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.profiler.step()


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
        opt = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate,
        )
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt),
            "monitor": "loss",
            "interval": "step",
            "frequency": 10,
            # "patience": 10,
        }
        out = {"optimizer": opt, "lr_scheduler": lr_scheduler_config}
        return out

    # def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
    #     loss.backward()
    #
    #     self.log("nfe_backward", float(self.ode_model.vf.nfe), prog_bar=True)
    #     self.ode_model.vf.nfe = 0