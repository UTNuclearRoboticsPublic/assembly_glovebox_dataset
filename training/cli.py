from training.lightning_trainers.lightning_model import LitModel
from training.lightning_trainers.oneformer_model import OneFormerLitModel
from training.lightning_trainers.sam_model import SamLitModel
from training.lightning_trainers.mobilesam_model import MobileSamLitModel
from training.lightning_trainers.bisenetv2_model import BiSeNetV2Model

from training.dataloaders.datamodule import AssemblyDataModule
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping
import lightning

from lightning.pytorch.callbacks import ModelCheckpoint

import lightning.pytorch as pl


## call - python cli.py fit --config config.yml
## adjust for which directory the config file has been placed in

def cli_main():
    # cli = LightningCLI(LitModel, AssemblyDataModule)
    cli = LightningCLI(datamodule_class=AssemblyDataModule)


if __name__ == '__main__':

    # here are some helpful args--
    # --trainer.accelerator "gpu" --trainer.max_epochs 150 --trainer.ckpt_path (resume training from here)
    # -- trainer.logger (defaults to tensorboard)
    # -- trainer.profiler

    cli_main()