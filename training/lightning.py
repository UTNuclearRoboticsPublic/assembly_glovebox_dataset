import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from dataloader import AssemblyDataset
from torch.utils.data import DataLoader
from UNET import UNET
import torch.nn.functional as F
from torch.utils.data import random_split
from argparse import ArgumentParser
import os
from datamodule import AssemblyDataModule
from pytorch_lightning import loggers as pl_loggers
import torchvision

class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.model = UNET(in_channels=3, out_channels=3)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, raw_preds= self._common_set(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            prog_bar=True
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, raw_preds= self._common_set(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
            },
            prog_bar=True
        )

        # for every two batches
        if batch_idx % 2 == 0:
            self._make_grid(x, "val_images")
            predictions = torch.argmax(raw_preds, dim=1)

            # predictions only on 0, 1 (most likely because we need more epochs) -> check that dim is correct, though
            self._make_grid(predictions, "val_preds")

    def test_step(self, batch, batch_idx):
        # is this working properly? barely any metric data.
        loss, raw_preds = self._common_set(batch, batch_idx)

        # automatically averages these values across the epoch
        self.log_dict(
            {
                "test_loss": loss,
            },
            prog_bar=True
        )

    def predict_step(self, batch, batch_idx):
        # used only when loading a model from a checkpoint
        # use argmax here
        x, y = batch
        raw_preds = self.model(x)
        probs = F.softmax(raw_preds, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def _common_set(self, batch, batch_idx):
        x, y = batch
        raw_preds = self.model(x)
        loss = F.cross_entropy(raw_preds, y.long())
        return loss, raw_preds
    
    # creates a grid of images and respective predictions in the validation set
    def _make_grid(self, values, name):
        grid = torchvision.utils.make_grid(values[0])
        self.logger.experiment.add_image(
            name,
            grid,
            self.global_step
        )



if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    model = LitModel()
    dm = AssemblyDataModule()
    # using multiple GPUs when on TACC

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./logs')

    # why no checkpoint added?
    trainer = pl.Trainer(default_root_dir='./checkpoints/', 
                         accelerator="gpu", max_epochs=15, logger=tensorboard, fast_dev_run=False)

    # loading a model
    # model = LitModel.load_from_checkpoint('./checkpoints/checkpoint.ckpt')
    # predictions = trainer.predict(model, dm)

    trainer.fit(model, dm)
    trainer.test(model, dm)
    # look into debugging so you can test beforehand    