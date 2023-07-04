from lightning_trainers.lightning_model import LitModel
from lightning_trainers.oneformer_model import OneFormerLitModel
from lightning_trainers.sam_model import SamModel

from dataloaders.datamodule import AssemblyDataModule
from lightning.pytorch.cli import LightningCLI

## call - python cli.py fit --config config.yml
## adjust for which directory the config file has been placed in

def cli_main():
    # cli = LightningCLI(LitModel, AssemblyDataModule)
    cli = LightningCLI(AssemblyDataModule)


if __name__ == '__main__':

    # here are some helpful args--
    # --trainer.accelerator "gpu" --trainer.max_epochs 150 --trainer.ckpt_path (resume training from here)
    # -- trainer.logger (defaults to tensorboard)
    # -- trainer.profiler

    cli_main()