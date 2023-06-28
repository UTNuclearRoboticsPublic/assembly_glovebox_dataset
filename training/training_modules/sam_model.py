import pytorch_lightning as pl
import torch
from UNET import UNET
import torch.nn.functional as F
import os
from datamodule import AssemblyDataModule
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from metrics import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
import sys
from tqdm import  tqdm
import torchvision.transforms as transforms
from torch.optim import Adam
from statistics import mean
from PIL import Image
from transformers import SamMaskDecoderConfig, SamProcessor, SamModel


class SamModel(pl.LightningModule):
    def __init__(self):
        super(SamModel, self).__init__()


        dec_config = SamMaskDecoderConfig(num_multimask_outputs=3, iou_head_depth=3)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base", ignore_mismatched_sizes=True)
        self.model = SamModel.from_pretrained("facebook/sam-vit-base", ignore_mismatched_sizes=True, mask_decoder_config = dec_config)
        
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)


    def training_step(self, batch, batch_idx):
        loss, predicted_masks= self._common_set(batch, batch_idx)

        # defualt on epoch=False, which is why it was not showing earlier
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, predicted_masks= self._common_set(batch, batch_idx)

        self.log_dict(
            {
                "val_loss": loss,
                "val_iou": self.iou(predicted_masks, y.to(torch.int32))
            },
            prog_bar=True
        )

        # for every two batches
        if batch_idx % 2 == 0:
            self._make_grid(x, "val_images")

            # here you need to adjust the outputed masks to have shape [1, height, width]
            predictions = torch.argmax(predicted_masks, dim=1)
            # predictions only on 0, 1 (most likely because we need more epochs) -> check that dim is correct, though
            self._make_grid(predicted_masks, "val_preds")

    def test_step(self, batch, batch_idx):
        x, y = batch
        # is this working properly? barely any metric data.
        loss, predicted_masks = self._common_set(batch, batch_idx)

        # automatically averages these values across the epoch
        self.log_dict(
            {
                "test_loss": loss,
                "test_iou": self.iou(predicted_masks, y.to(torch.int32)),
                "test_ace": adaptive_calibration_error(predicted_masks, y),
                "test_entropy": predictive_entropy(predicted_masks)
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
        optimizer = torch.optim.Adam(self.model.mask_decoder.parameters(), lr=0.001)
        return optimizer
    
    def _common_set(self, batch, batch_idx):
        x, y = batch

        y = torch.unsqueeze(y, 0)

        input_box = torch.from_numpy(
                        np.array(
                            [
                              [0, 0, x.shape[1], x.shape[0]]
                            ]
                          )
                    ).float()
        input_boxes = torch.repeat(x.shape[0], 1, 1)
        
        inputs = self.processor(x, input_boxes = input_boxes, return_tensors="pt")
        raw_preds = self.model(**inputs, multimask_output=True)

        predicted_masks = raw_preds.pred_masks.squeeze(1) # result shape -> [1, 3, 256, 256]

        loss = F.cross_entropy(predicted_masks, y.long())

        return loss, predicted_masks
    
    # creates a grid of images and respective predictions in the validation set
    def _make_grid(self, values, name):
        # grid = torchvision.utils.make_grid(values[0])

        if name == "val_preds":
            
            values = values.cpu().numpy()

            final_imgs = np.zeros((values.shape[0], 3, values.shape[1], values.shape[2]))

            for idx, preds in enumerate(values[:3]):
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

        # logs 3 of whatever was sent through the method (images or predictions)
        self.logger.experiment.add_images(
            name,
            values[:3],
            self.global_step,
        )



if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    model = SamModel()
    dm = AssemblyDataModule(
        fit_query= ['Test_Subject_1', 'ood', 'J', 'Top_View'],
        test_query= ['Test_Subject_1', 'ood', 'TB', 'Side_View']
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./tb_logs', name="sam")
    # this is where checkpoints will be stored
    # look under root_dir here -> https://pytorch-lightning.readthedocs.io/en/0.10.0/logging.html


    trainer = pl.Trainer(accelerator="gpu", max_epochs=150, logger=tensorboard, fast_dev_run=True,
                         profiler="pytorch")

    # trainer.fit(model, dm, ckpt_path='./logs/lightning_logs/version_11/checkpoints/epoch=114-step=115.ckpt')
    trainer.fit(model, dm)

    trainer.test(model, dm)