from typing import Any

from lightning import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
from torch.profiler import profile, record_function, ProfilerActivity

import wandb

import torchdyn


class NfeMetrics(Callback):
    def __init__(self):
        self.running_fwd = 0
        self.running_back = 0

        self.step_fwd = 0

        self.epoch_start_fwd = 0
        self.epoch_start_back = 0

    def on_before_backward(self, trainer, pl_module, optimizer):
        fwd_nfes = float(pl_module.ode_model.vf.nfe)
        self.step_fwd = fwd_nfes
        self.running_fwd += fwd_nfes

        pl_module.log("nfe_train_fwd", fwd_nfes, prog_bar=True)
        pl_module.log("nfe_train_fwd_cum", self.running_fwd)

        pl_module.ode_model.vf.nfe = 0

    def on_after_backward(self, trainer, pl_module):
        back_nfes = float(pl_module.ode_model.vf.nfe)
        self.running_back += back_nfes

        pl_module.log("nfe_train_back", back_nfes, prog_bar=True)
        pl_module.log("nfe_train_back_cum", self.running_back)

        step_total = self.step_fwd + back_nfes
        pl_module.log("nfe_train_total", step_total)
        pl_module.log("nfe_train_total_cum", self.running_fwd + self.running_back)

        pl_module.ode_model.vf.nfe = 0

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.epoch_start_fwd = self.running_fwd
        self.epoch_start_back = self.running_back

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        nfes_epoch_fwd = self.running_fwd - self.epoch_start_fwd
        nfes_epoch_back = self.running_back - self.epoch_start_back

        pl_module.log("nfe_train_fwd_epoch", nfes_epoch_fwd)
        pl_module.log("nfe_train_back_epoch", nfes_epoch_back)
        pl_module.log("nfe_train_total_epoch", nfes_epoch_fwd + nfes_epoch_back)

        pl_module.log("nfe_train_fwd_epoch_cum", self.running_fwd)
        pl_module.log("nfe_train_back_epoch_cum", self.running_back)
        pl_module.log("nfe_train_total_epoch_cum", self.running_fwd + self.running_back)

    def on_test_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        nfes = float(pl_module.ode_model.vf.nfe)
        pl_module.log(f"nfe_test_step", nfes, prog_bar=True)
        pl_module.ode_model.vf.nfe = 0


class ProfilerCallback(Callback):
    def __init__(self, schedule=None, epoch=1, fname="trace"):
        super().__init__()
        if schedule is None:
            schedule = torch.profiler.schedule(
                skip_first=5, wait=5, warmup=3, active=2, repeat=1
            )
        self.epoch = epoch

        self.fname = fname

        self.profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=self.ready,
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            # record_shapes=True,
            profile_memory=True,
        )

    def ready(self, profiler):
        profiler.export_chrome_trace(f"{self.fname}.json")

    def on_train_start(self, trainer, pl_module):
        if trainer.current_epoch == self.epoch:
            self.profiler.start()

    def on_train_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch == self.epoch:
            self.profiler.stop()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.profiler.step()


class PlotTrajectories(Callback):

    def __init__(self, f, freq=10):
        super().__init__()
        self.f = f
        self.run = None
        self.freq = freq

    def on_train_start(self, trainer, pl_module) -> None:
        self.run = trainer.logger.experiment
        self.table = wandb.Table(
            columns=["fwd_traj","epoch" ] #"batch_acc", "batch_loss", "nfes"]
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx & self.freq != 0 : return
        x, y = next(iter(trainer.val_dataloaders))
        x = x.cuda()
        x_em = pl_module.embedding(x)
        t_span = torch.linspace(pl_module.t_span[0], pl_module.t_span[-1], 100).cuda()
        _, y_hat_pan = pl_module.ode_model(x_em, t_span=t_span)
        _, y_hat_true = torchdyn.numerics.odeint(
            self.f, t_span=t_span, x=x_em, solver="tsit5", atol=1e-9
        )
        fig, axes = plt.subplots(3, 3)
        dims = y_hat_pan[0].shape
        for idx, ax in enumerate(axes.reshape(-1)):
            rand_idx = [torch.randint(0, d, (1,)).item() for d in dims]
            ax.plot(
                t_span.cpu(),
                y_hat_pan[:, *rand_idx].cpu(),
                "r-",
                label="pan",
            )
            ax.plot(
                t_span.cpu(),
                y_hat_true[:, *rand_idx].cpu(),
                "g--",
                label="true",
            )
        # plt.show()
        # self.table
        if self.run is not None:
            self.run.log({'chart': fig})

        # logits = pl_module.classifier(y_hat_pan[-1])
        # loss = nn.CrossEntropyLoss()(logits, y)
        #
        # _, preds = torch.max(logits, dim=1)
        # acc = torch.sum(preds == y) / y.shape[0]


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
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.val_acc = 0
        self.val_batches = 0

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)
        self.val_acc += acc
        self.val_batches += 1
        return loss

    def on_validation_epoch_end(self):
        val_acc_epoch = self.val_acc / self.val_batches
        self.log("val_acc", val_acc_epoch, prog_bar=True)

    def on_test_start(self):
        self.test_acc = 0
        self.test_batches = 0

    def test_step(self, batch, batch_idx):
        loss, preds, acc = self._common_step(batch, batch_idx)
        self.test_batches += 1
        self.test_acc += acc

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        test_acc = self.test_acc / self.test_batches
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        # TODO: use LR callback
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

    # def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
    #     loss.backward()
    #
    #     self.log("nfe_backward", float(self.ode_model.vf.nfe), prog_bar=True)
    #     self.ode_model.vf.nfe = 0
