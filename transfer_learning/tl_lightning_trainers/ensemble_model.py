import torch
import torch.nn.functional as F
import torchmetrics

from training.lightning_trainers.lightning_model import LitModel
from training.lightning_trainers.mobilesam_model import MobileSamLitModel
from training.lightning_trainers.bisenetv2_model import BiSeNetV2Model

from lightning.pytorch import loggers as pl_loggers
# from training.dataloaders.datamodule import AssemblyDataModule
from transfer_learning.tl_dataloaders.bin_datamodule import BinAssemblyDataModule
import lightning.pytorch as pl

import time
import os


class EnsembleModel(LitModel):
    def __init__(self, model_type, test_dropout=False):
        super(EnsembleModel, self).__init__()

        self.model_type = model_type

        #TODO: instead of using LitModel, use other one if not using UNET

        if model_type == "unet":
            path_1 = "/home/slwanna/FIRETEAM/saved_models_tb/lightning_logs/ft_unet_wh_0/checkpoints/epoch=31-step=33788.ckpt"
            model_1 = LitModel.load_from_checkpoint(path_1, map_location=torch.device('cuda:0'), test_dropout=test_dropout)

            path_2 = "/home/slwanna/FIRETEAM/saved_models_tb/lightning_logs/ft_unet_wh_1/checkpoints/epoch=99-step=16052.ckpt"
            model_2 = LitModel.load_from_checkpoint(path_2, map_location=torch.device('cuda:0'), test_dropout=test_dropout)

            path_3 = "/home/slwanna/FIRETEAM/saved_models_tb/lightning_logs/ft_unet_wh_2/checkpoints/epoch=101-step=16404.ckpt"
            model_3 = LitModel.load_from_checkpoint(path_3, map_location=torch.device('cuda:0'), test_dropout=test_dropout)

        if model_type == "bisenetv2":
            path_1="/home/slwanna/FIRETEAM/saved_models_tb/lightning_logs/ft_bisenet_wh_0/checkpoints/epoch=35-step=1264.ckpt"
            model_1 = BiSeNetV2Model.load_from_checkpoint(path_1, map_location=torch.device('cuda:1'), test_dropout=test_dropout)

            path_2="/home/slwanna/FIRETEAM/saved_models_tb/lightning_logs/ft_bisenet_wh_1/checkpoints/epoch=36-step=31564.ckpt"
            model_2 = BiSeNetV2Model.load_from_checkpoint(path_2, map_location=torch.device('cuda:1'), test_dropout=test_dropout)

            path_3="/home/slwanna/FIRETEAM/saved_models_tb/lightning_logs/ft_bisenet_wh_2/checkpoints/epoch=36-step=1268.ckpt"
            model_3 = BiSeNetV2Model.load_from_checkpoint(path_3, map_location=torch.device('cuda:1'), test_dropout=test_dropout)

        if model_type == "mobilesam":
            
            path_1="./lightning_logs/version_80/checkpoints/epoch=3027-step=24224.ckpt"
            model_1 = MobileSamLitModel.load_from_checkpoint(path_1, map_location=torch.device('cuda:2'), test_dropout=test_dropout)
            
            path_2="./lightning_logs/version_75/checkpoints/epoch=1849-step=29600.ckpt"
            model_2 = MobileSamLitModel.load_from_checkpoint(path_2, map_location=torch.device('cuda:2'), test_dropout=test_dropout)

            path_3="./lightning_logs/version_69/checkpoints/epoch=1312-step=10704.ckpt"
            model_3 = MobileSamLitModel.load_from_checkpoint(path_3, map_location=torch.device('cuda:2'), test_dropout=test_dropout)

        self.models = [model_1, model_2, model_3]



    def _common_set(self, batch, batch_idx):

        # TODO: checkwhy the IoU is almost 0

        x, y = batch
        print("....inside the new common set")
        #TODO: average time
        predictions = []
        times = []
        for model in self.models:
            start_time = time.perf_counter()

            # use the _common_set method from the model instead if MobileSAM
            if self.model_type=="mobilesam":
                _, output = model._common_set(batch, batch_idx)
            else:
                output = model.model(x) # access the model in the method

             # because bisenet return multiple logits in train mode
            if isinstance(output, tuple):
                output = output[0]

            end_time = time.perf_counter()
            pred_time = end_time - start_time
            times.append(pred_time)

            print(f" Time: {pred_time / x.shape[0]}")
            predictions.append(output)
        self.avg_pred_time = sum(times) / x.shape[0] # should just be sum because you are using all of the models
        print(f"average time is {self.avg_pred_time}")
        super()._set_time(self.avg_pred_time)
        predictions = torch.stack(predictions, dim=-1)
        averaged_predictions = torch.mean(predictions, dim=-1)
        loss = self.get_loss(averaged_predictions, y)
        return loss, averaged_predictions

if __name__ == "__main__":

    # TODO: make this a command line argument
    model_type =  "unet" #"bisenetv2" #"mobilesam"
    fast_dev_run = False
    
    # TODO: make this a command line argument
    sets = ["id", "ood", "gs", "ood+gs"]
    # Step 1. Load all relevant models in the ensemble.
    # Be consistent, either chose the latest or best performing epoch. Don't combine like I think I did.

    if model_type == "bisenetv2":
        batch_size = 128
        gpu = 1
    elif model_type == "unet":
        batch_size = 64
        gpu = 0
    elif model_type == "mobilesam":
        batch_size = 64
        gpu = 2
    else:
        batch_size = 0

    data = {
        "fit_query": {
            "participants": [
                "Test_Subject_1",
                "Test_Subject_2",
                "Test_Subject_3",
                "Test_Subject_4",
                # "Test_Subject_6",
                "Test_Subject_7",
                "Test_Subject_9",
                "Test_Subject_10",
                "Test_Subject_11",
                "Test_Subject_12"
            ],
            "distribution": ["id"],
            "task": ["J", "TB"],
            "view": ["Top_View", "Side_View"]
        },
        "test_query": {
            "participants": [
                "Test_Subject_1",
                "Test_Subject_2",
                "Test_Subject_3",
                "Test_Subject_4",
                # "Test_Subject_6",
                "Test_Subject_7",
                "Test_Subject_9",
                "Test_Subject_10",
                "Test_Subject_11",
                "Test_Subject_12"
            ],
            "distribution": ["id"],
            "task": ["J", "TB"],
            "view": ["Top_View", "Side_View"]
        },
    }

    dm = BinAssemblyDataModule(
        test_query = data.get("test_query"),
        fit_query = data.get("fit_query"),
        batch_size=batch_size,
        img_size=256
    )

    for active_set in sets:
        for dropout in [False, True]:
            model = EnsembleModel(model_type = model_type, test_dropout=dropout)
            used_data = data.copy()
            if active_set=="ood":
                used_data['test_query']['distribution'] = ["ood"]
            if active_set=="id":
                used_data['test_query']['distribution'] = ["id"]
                # used_data['test_query']['participants'] = ["Test_Subject_2"]
            if active_set=="gs":
                used_data['test_query']['distribution'] = ["replaced_green_screen"]
            if active_set=="ood+gs":
                used_data['test_query']['distribution'] = ["ood", "replaced_green_screen"]

            target_path = f"./lightning_logs/{active_set}_{model_type}_dropout_{dropout}"

            if not os.path.exists(target_path):
                os.makedirs(target_path)

            tb_logger = pl_loggers.TensorBoardLogger(save_dir="./lightning_logs", name=f"NEW_{active_set}_{model_type}_dropout_{dropout}")

            trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=tb_logger, devices=[gpu], accelerator="gpu")

            trainer.test(model, dm)