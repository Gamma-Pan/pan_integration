from torchvision.models import resnet18
from torch import nn
from torch.optim import Adam
import lightning.pytorch as pl


class ResNet18Img(pl.LightningModule):
    def __init__(self, img_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        # by default resnet18 accepts an input with three channels
        self.model.conv1 = nn.Conv2d(img_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

