import lightning.pytorch as pl
from nde_squared.models import ResNet18Img
from nde_squared.data import load_mnist


my_resnet = ResNet18Img()
train_loader, val_loader, test_loader = load_mnist()

trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
trainer.fit(model=my_resnet,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader)
