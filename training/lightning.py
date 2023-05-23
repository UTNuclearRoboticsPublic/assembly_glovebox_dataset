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

class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.model = UNET(in_channels=3, out_channels=3)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, raw_preds, y = self._common_set(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
            },
            prog_bar=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, raw_preds, y = self._common_set(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
            },
            prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, raw_preds, y = self._common_set(batch, batch_idx)

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
        x = x.reshape(x.ize(0), -1)
        raw_preds = self.model(x)
        probs = F.softmax(raw_preds, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # predictions you can make after each batch in the epoch
        # save the image predictions here
        torch.save(predictions, os.path.join(f'./pred{trainer.global_rank}.pt'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step_end(self):
        # use this with softmax also need for val and test?
        pass
    
    def _common_set(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1) # why this?
        raw_preds = self.model(x)
        loss = F.cross_entropy(raw_preds, y)
        return loss, raw_preds, y



if __name__ == "__main__":

    # # arg parsing with python - a bit inefficient
    # parser = ArgumentParser()
    # parser.add_argument("--devices", type=int, default=1)
    # parser.add_argument("--epochs", type=int, default=150)

    # # parse the args
    # args = parser.parse_args()

    model = LitModel()
    dm = AssemblyDataModule()

    # look into this method
    # using multiple GPUs when on TACC
    trainer = pl.Trainer(default_root_dir='./checkpoints/', 
                         accelerator="gpu", min_epochs=1, max_epochs=150)

    # loading a model
    # model = LitModel.load_from_checkpoint('./checkpoints/checkpoint.ckpt')
    # predictions = trainer.predict(model, dm)

    trainer.fit(model, dm)
    trainer.test(model, dm)
    # use trainer.tune to find optimal hyperparameters
    # look into debugging so you can test beforehand