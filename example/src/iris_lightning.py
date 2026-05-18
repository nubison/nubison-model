"""IrisLightning — Lightning module used by ``train_lightning.ipynb``.

Lives in ``src/`` for the same reason as ``iris_net.py`` —
``pickle.load`` in ``infer_lightning.ipynb`` needs to resolve the
class by ``src.iris_lightning.IrisLightning``.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class IrisLightning(pl.LightningModule):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.fc = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_accuracy", acc, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
