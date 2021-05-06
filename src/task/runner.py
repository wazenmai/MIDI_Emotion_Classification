import torch
import torch.nn as nn
import numpy as np 
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..metric import accuracy

class Runner(LightningModule):
    def __init__(self, model, runner_config: DictConfig):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.hparams.update(runner_config.optimizer.params)
        self.hparams.update(runner_config.scheduler.params)
        self.hparams.update(runner_config.trainer.params)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov,
        )

        scheduler = ReduceLROnPlateau(
            optimizer=opt,
            factor=self.hparams.factor,
            patience=self.hparams.patience,
            verbose=self.hparams.verbose,
            mode=self.hparams.mode,
        )
        lr_scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss' # Metric to monitor
        }

        return [opt], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        audio, label, _ = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label.long())
        acc = accuracy(prediction, label.long())
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,    
            },
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    
    def validation_step(self, batch, batch_idx):
        audio, label, _ = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label.long())
        acc = accuracy(prediction, label.long())
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_acc = np.mean(np.array([output["val_acc"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {
                "val_loss": val_loss,
                "val_acc": val_acc,
                }

    def test_step(self, batch, batch_idx):
        audio, label, fname = batch
        prediction = self.forward(audio)
        loss = self.criterion(prediction, label.long())
        acc = accuracy(prediction, label.long())
        return {"val_loss": loss, "val_acc": acc, "prediction": prediction, "label":label, "fname":fname}

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_acc = np.mean(np.array([output["val_acc"] for output in outputs]))
        predictions = [output["prediction"] for output in outputs]
        labels = [output["label"] for output in outputs]
        fnames = [output["fname"] for output in outputs]
        result = {"val_loss": val_loss,"val_acc": val_acc}
        pred = {"predictions": predictions,"labels": labels, "fnames": fnames}
        torch.save(pred, "predictions.pt")
        return result
        