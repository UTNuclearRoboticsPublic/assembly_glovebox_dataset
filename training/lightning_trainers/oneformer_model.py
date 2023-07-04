import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from metrics import *
import numpy as np
from transformers import OneFormerProcessor, OneFormerModel, TrainingArguments, Trainer, OneFormerForUniversalSegmentation

from dataloaders.datamodule import AssemblyDataModule
from lightning_model import LitModel


class OneFormerLitModel(LitModel):
    def __init__(self, droprate=0):
        super(OneFormerLitModel, self).__init__()

        id2label = {
            0: "background",
            1: "left_hand",
            2: "right_hand"
        }

        self.model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny", id2label=id2label, label2id = {v: k for k, v in id2label.items()},
                                                          num_classes=3, ignore_mismatched_sizes=True, num_queries=3)
        self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny",metadata=id2label, num_labels=3, class_names=["background", "left_hand", "right_hand"], do_reduce_labels=True,
                                                                size=644, ignore_mismatched_sizes=True) 
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def _common_set(self, batch, batch_idx):
        x, y = batch
        inputs = self.processor(x, ["semantic"], return_tensors="pt")
        raw_preds = self.model(**inputs)
        loss = F.cross_entropy(raw_preds, y.long())
        return loss, raw_preds.transformer_decoder_mask_predictions
    
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
    model = OneFormerLitModel()
    dm = AssemblyDataModule(
        fit_query= ['Test_Subject_1', 'ood', 'J', 'Top_View'],
        test_query= ['Test_Subject_1', 'ood', 'TB', 'Side_View']
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir='./logs', name='oneformer')

    trainer = pl.Trainer(default_root_dir='./checkpoints/', 
                         accelerator="gpu", max_epochs=150, logger=tensorboard, fast_dev_run=True,
                         profiler="pytorch")

    # trainer.fit(model, dm, ckpt_path='./logs/lightning_logs/version_11/checkpoints/epoch=114-step=115.ckpt')
    trainer.fit(model, dm)

    trainer.test(model, dm)