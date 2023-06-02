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
from pytorch_lightning.profilers import PyTorchProfiler
import torchvision
import torchmetrics
from metrics import *
import numpy as np
from pytorch_lightning.cli import LightningCLI

class LitModel(pl.LightningModule):
    def __init__(self, droprate=0):
        super(LitModel, self).__init__()
        self.model = UNET(in_channels=3, out_channels=3, droprate=0)
        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

        #only use hyperparameters if you need it for instantiating the model
        # otherwise, use it from the CLI only for simplicity
        # self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        loss, raw_preds= self._common_set(batch, batch_idx)

        # defualt on epoch=False, which is why it was not showing earlier
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        # why was no training loss recorded?
        # what is hp metric

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, raw_preds= self._common_set(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_iou": self.iou(raw_preds, y.to(torch.int32))
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
        x, y = batch
        # is this working properly? barely any metric data.
        loss, raw_preds = self._common_set(batch, batch_idx)

        # automatically averages these values across the epoch
        self.log_dict(
            {
                "test_loss": loss,
                "test_iou": self.iou(raw_preds, y.to(torch.int32)),
                "test_ace": adaptive_calibration_error(raw_preds, y),
                "test_entropy": predictive_entropy(raw_preds)
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
        # grid = torchvision.utils.make_grid(values[0])

        if name == "val_preds":
            
            values = values.cpu().numpy()

            final_imgs = np.zeros((values.shape[0], 3, values.shape[1], values.shape[2]))

            for idx, preds in enumerate(values):
                color_map = {
                    0: (0, 0, 0),
                    1: (0, 255, 0),
                    2: (0, 0, 255)
                }

                final_image = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.uint8)
                for index, color in color_map.items():
                    final_image[preds == index] = color

                preds = np.transpose(final_image, (2, 0, 1))

                final_imgs[idx, :, :, :] = preds
            values = final_imgs




        self.logger.experiment.add_images(
            name,
            values[:3],
            self.global_step,
        )

if __name__ == "__main__":


    torch.set_float32_matmul_precision('medium')
    model = LitModel()
    dm = AssemblyDataModule(
        fit_query= ['Test_Subject_1', 'ood', 'J', 'Top_View'],
        test_query= ['Test_Subject_1', 'ood', 'TB', 'Side_View']
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./logs')

    trainer = pl.Trainer(default_root_dir='./checkpoints/', 
                         accelerator="gpu", max_epochs=150, logger=tensorboard, fast_dev_run=False,
                         profiler="pytorch")


    trainer.fit(model, dm, ckpt_path='./logs/lightning_logs/version_11/checkpoints/epoch=114-step=115.ckpt')

    trainer.test(model, dm)