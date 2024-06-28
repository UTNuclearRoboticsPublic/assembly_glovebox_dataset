from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# Datasets
from transfer_learning.tl_dataloaders.bin_datamodule import BinAssemblyDataModule
from transfer_learning.tl_dataloaders.HaDR_datamodule import BinAssemblyHaDRDataModule
from transfer_learning.tl_dataloaders.HRC_datamodule import HRCDataModule
from transfer_learning.tl_dataloaders.WH_datamodule import BinAssemblWHyDataModule
# Model
from transfer_learning.tl_lightning_trainers.lightning_model import LitModel

def main(args):
    pt_ds_name = args.dataset
    chkpt_path = args.chkpt_path
    data_kwargs = {
        "fit_query": {
            "participants": [
                "Test_Subject_1",
                "Test_Subject_3",
                "Test_Subject_4",
                "Test_Subject_5",
                "Test_Subject_6",
                "Test_Subject_7",
                "Test_Subject_8",
                "Test_Subject_9",
                "Test_Subject_10",

            ],
            "distribution": ["id"],
            "task": ["Jenga_task", "Toolbox_task"],
            "view": ["Top_View", "Side_View"],
        },
        "test_query": {
            "participants": [
                "Test_Subject_2",
            ],
            "distribution": ["id"],
            "task": ["Jenga_task", "Toolbox_task"],
            "view": ["Top_View", "Side_View"],
        },
        "batch_size": 32,
        "img_size": 256,
    }

    hags_module = BinAssemblyDataModule(**data_kwargs)
    stage1_dm = None
    if pt_ds_name == "HADR":
        stage1_dm = BinAssemblyHaDRDataModule(**data_kwargs)
    elif pt_ds_name == "HRC":
        stage1_dm = HRCDataModule(**data_kwargs)
    elif pt_ds_name == "WH":
        stage1_dm = BinAssemblWHyDataModule(**data_kwargs)

    model_kwargs = {
        "learning_rate": 1e-3,
        "droprate": 0.1,
        "test_dropout": False,
    }

    model = LitModel(**model_kwargs)

    stage_1_trainer_kwargs = {
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": [7],
        "callbacks": [EarlyStopping(monitor="val_loss", mode="min", patience=7),
                      ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=2)],
        "num_nodes": 1,
        "precision": "32-true",
        "check_val_every_n_epoch": 1,
        "enable_checkpointing": True,
        "accumulate_grad_batches": 1,
        "use_distributed_sampler": True,
    }
    
    stage_1_trainer = Trainer(**stage_1_trainer_kwargs)
    stage_1_trainer.fit(model, datamodule=stage1_dm)

    stage_2_trainer_kwargs = {
    "max_epochs": 20,
    "accelerator": "gpu",
    "devices": [7],
    "callbacks": [EarlyStopping(monitor="val_loss", mode="min", patience=7),
                    ModelCheckpoint(monitor="val_iou", mode="max", save_top_k=2)],
    "num_nodes": 1,
    "precision": "32-true",
    "check_val_every_n_epoch": 1,
    "enable_checkpointing": True,
    "accumulate_grad_batches": 1,
    "use_distributed_sampler": True,
    }

    stage_2_trainer = Trainer(**stage_2_trainer_kwargs)
    stage_2_trainer.fit(model, datamodule=hags_module)
    stage_2_trainer.save_checkpoint(chkpt_path)  #"example.ckpt")
    return 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="None", type=str, required=True)
    parser.add_argument("--chkpt_path", default="None", type=str, required=True)
    args = parser.parse_args()

    main(args)