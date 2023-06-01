from lightning_model import LitModel
from datamodule import AssemblyDataModule
from pytorch_lightning.cli import LightningCLI

## call - python cli.py fit --config config.yml
## adjust for which directory the config file has been placed in

def cli_main():
    cli = LightningCLI(LitModel, AssemblyDataModule)

if __name__ == '__main__':

    # --trainer.accelerator "gpu" --trainer.max_epochs 150 --trainer.ckpt_path (resume training from here)
    # -- trainer.logger (defaults to tensorboard)
    # -- trainer.profiler

    # 
    cli_main()