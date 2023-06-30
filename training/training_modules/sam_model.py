import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from metrics import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
from tqdm import  tqdm
import torchvision.transforms as transforms
from torch.optim import Adam
from statistics import mean
from PIL import Image
from transformers import SamMaskDecoderConfig, SamProcessor, SamModel

from dataloaders.datamodule import AssemblyDataModule
from models import UNET

from lightning_model import LitModel

class SamModel(LitModel):
    def __init__(self):
        super(SamModel, self).__init__()


        dec_config = SamMaskDecoderConfig(num_multimask_outputs=3, iou_head_depth=3)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base", ignore_mismatched_sizes=True)
        self.model = SamModel.from_pretrained("facebook/sam-vit-base", ignore_mismatched_sizes=True, mask_decoder_config = dec_config)
        
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

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

        input_box = torch.from_numpy(
                        np.array(
                            [
                              [0, 0, x.shape[2], x.shape[3]]
                            ]
                          )
                    ).float()
        input_boxes = torch.repeat(input_box.shape[0], 1, 1)
        
        inputs = self.processor(x, input_boxes = input_boxes, return_tensors="pt")
        raw_preds = self.model(**inputs, multimask_output=True)

        predicted_masks = raw_preds.pred_masks.squeeze(1) # result shape -> [1, 3, 256, 256]

        loss = F.cross_entropy(predicted_masks, y.long())

        # adjust 0 if this is not working
        return loss, predicted_masks[0]

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