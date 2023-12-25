import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import lightning.pytorch as pl
import torchmetrics
import numpy as np
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
import time

from ..metrics import *
from ..models.UNET import UNET

import os
import csv

class LitModel(pl.LightningModule):
    def __init__(self, droprate=0, learning_rate=0.001, test_dropout=False):
        super(LitModel, self).__init__()
        self.model = UNET(in_channels=3, out_channels=3, droprate=droprate)
        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)
        self.learning_rate = learning_rate

        self.entropy_outputs = []

        self.avg_pred_time = 0.0

        self.test_dropout = test_dropout



        #only use hyperparameters if you need it for instantiating the model
        # otherwise, use it from the CLI only for simplicity
        # self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        loss, raw_preds= self._common_set(batch, batch_idx)

        # defualt on epoch=False, which is why it was not showing earlier
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # why was no training loss recorded?
        # what is hp metric

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, raw_preds= self._common_set(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_iou": self.get_avg_iou(raw_preds, y)
            },
            prog_bar=True,
            sync_dist=True
        )

        # for every two batches
        if batch_idx % 2 == 0:
            self._make_grid(x, "val_images")
            predictions = torch.argmax(raw_preds, dim=1)

            # predictions only on 0, 1 (most likely because we need more epochs) -> check that dim is correct, though
            self._make_grid(predictions, "val_preds")
    
    def get_avg_iou(self, raw_preds, y):

        iou1 = self.iou(raw_preds, y[0].to(torch.int32))
        iou2 = self.iou(raw_preds, y[1].to(torch.int32))
        return (iou1+iou2) / 2

    def test_step(self, batch, batch_idx):
        # add matplotlib figure - https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_figure

        if self.test_dropout:
            print("....we made it to dropout")
            self.model.train()

        x, y = batch
        # is this working properly? barely any metric data.
        loss, raw_preds = self._common_set(batch, batch_idx)

        def get_avg_ace(raw_preds, y):
            ace1 = adaptive_calibration_error(y_pred=raw_preds, y_true=y[0])
            ace2 = adaptive_calibration_error(y_pred=raw_preds, y_true=y[1])
            return (ace1+ace2) / 2
        
        # because bisenet return multiple logits in train mode
        if isinstance(raw_preds, tuple):
            raw_preds = raw_preds[0]

        # automatically averages these values across the epoch
        self.log_dict(
            {
                "test_loss": loss,
                "test_iou": self.get_avg_iou(raw_preds, y),
                "test_ace": get_avg_ace(raw_preds, y), # [4, 3, 161, 161] and [4, 161, 161] (two targets though)
                "avg_frame_reference_time": self.avg_pred_time
                # "test_entropy": predictive_entropy(raw_preds)
            },
            prog_bar=True,
            sync_dist=True
        )

        return_ent = predictive_entropy(raw_preds)

        self.entropy_outputs.extend(return_ent)
    
    def on_test_epoch_end(self):
        entropy_values = self.entropy_outputs
        print(f".............the entropy values are {entropy_values}")

        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(entropy_values)

        self.logger.experiment.add_figure(
            tag="predictive entropy", figure=fig
            )
        

        if not self.trainer.fast_dev_run:
            csv_path = os.path.join(self.trainer.log_dir, "raw_entropy.csv")

            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                for item in entropy_values:
                    csv_writer.writerow([item])
            

    
    def predict_step(self, batch, batch_idx):
        # used only when loading a model from a checkpoint
        # use argmax here
        x, y = batch
        raw_preds = self.model(x)
        probs = F.softmax(raw_preds, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_loss(self, raw_preds, y):
        y1, y2 = y
        loss_1 = F.cross_entropy(raw_preds, y1.long())
        loss_2 = F.cross_entropy(raw_preds, y2.long())
        loss = (loss_1 + loss_2) / 2
        return loss 


    def _common_set(self, batch, batch_idx):
        x, y = batch
        start_time = time.time()

        raw_preds = self.model(x)

        end_time = time.time()
        pred_time = end_time - start_time
        print(f".....the pred time is {pred_time} and {x.shape}")
        self.avg_pred_time = pred_time / x.shape[0]
        # print(f"shape of input x is {x.shape}")
        loss = self.get_loss(raw_preds, y)
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

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./logs', name='UNET')

    trainer = pl.Trainer(default_root_dir='./checkpoints/', 
                         accelerator="gpu", max_epochs=150, logger=tensorboard, fast_dev_run=True,
                         profiler="pytorch")

    # trainer.fit(model, dm, ckpt_path='./logs/lightning_logs/version_11/checkpoints/epoch=114-step=115.ckpt')
    trainer.fit(model, dm)

    trainer.test(model, dm)

    # deep ensemble testing