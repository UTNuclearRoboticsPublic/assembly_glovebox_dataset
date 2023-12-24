import torch
import torch.nn.functional as F
import torchmetrics

from training.lightning_trainers.lightning_model import LitModel
from lightning.pytorch import loggers as pl_loggers
from training.dataloaders.datamodule import AssemblyDataModule
import lightning.pytorch as pl


class EnsembleModel(LitModel):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def _common_set_(self, batch, batch_idx):
        x, y = batch
        #TODO: average time
        predictions = []
        for model in self.models:
            output = model(x)
            predictions.append(output)
        predictions = torch.stack(predictions, dim=-1)
        averaged_predictions = torch.mean(predictions, dim=-1)
        loss = self.get_loss(averaged_predictions, y)
        return loss, averaged_predictions

if __name__ == "__main__":

    # Step 1. Load all relevant models in the ensemble.
    # Be consistent, either chose the latest or best performing epoch. Don't combine like I think I did.

    path_1 = "./lightning_logs/version_73/checkpoints/epoch=1011-step=8096.ckpt"
    unet_model_1 = LitModel.load_from_checkpoint(path_1, map_location=torch.device('cuda'))

    path_2 = "./lightning_logs/version_76/checkpoints/epoch=1856-step=29712.ckpt"
    unet_model_2 = LitModel.load_from_checkpoint(path_2, map_location=torch.device('cuda'))

    path_3 = "./lightning_logs/version_78/checkpoints/epoch=3223-step=51584.ckpt"
    unet_model_3 = LitModel.load_from_checkpoint(path_3, map_location=torch.device('cuda'))

    model = EnsembleModel(models = [unet_model_1, unet_model_2, unet_model_3])


    data = {
        "fit_query": {
            "participants": [
                "Test_Subject_1",
                "Test_Subject_3",
                "Test_Subject_4",
                "Test_Subject_6",
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
                "Test_Subject_3",
                "Test_Subject_4",
                "Test_Subject_6",
                "Test_Subject_7",
                "Test_Subject_9",
                "Test_Subject_10",
                "Test_Subject_12"
            ],
            "distribution": ["id"],
            "task": ["J", "TB"],
            "view": ["Top_View", "Side_View"]
        },
    }

    dm = AssemblyDataModule(
        test_query = data.get("test_query"),
        fit_query = data.get("fit_query"),
        batch_size=64,
        img_size=256
    )

    # tensorboard = pl_loggers.TensorBoardLogger()

    trainer = pl.Trainer(fast_dev_run=True)

    trainer.test(model, dm)

