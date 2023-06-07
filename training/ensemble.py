# for using deep ensembles with PyTorch lighning
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from datamodule import AssemblyDataModule
import torchmetrics
from metrics import *
from lightning_model import LitModel
from torch.utils.tensorboard import SummaryWriter


def ensemble_predict(models, dm):

    dm.setup("test")

    writer = SummaryWriter()

    test_data = dm.test_dataloader()
    for i, batch in enumerate(test_data):
        x, y = batch
        raw_preds = []
        for model in models:
                pred = model.model(x)
                raw_preds.append(pred)
        preds_stack = torch.stack(raw_preds)
        avg_preds = torch.mean(preds_stack, dim=0)

        loss = F.cross_entropy(avg_preds, y.long())

        iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)
        test_iou = iou(avg_preds, y.to(torch.int32))

        record = {
             'loss': loss,
             'iou': iou,
        }

        writer.add_scalars('run', record, i)
    writer.close()

def ensemble_run(path_to_models, dm):
    models = []
    
    for path in path_to_models:

        torch.set_float32_matmul_precision('medium')
        model = LitModel.load_from_checkpoint(path)
        model.eval()

        models.append(model)

    ensemble_predict(models, dm)


if __name__ == '__main__':
    dm = AssemblyDataModule(
            fit_query= ['Test_Subject_1', 'ood', 'J', 'Top_View'],
            test_query= ['Test_Subject_1', 'ood', 'TB', 'Side_View']
        )
    
    path_to_models = [
         './logs/model1'
         './logs/model2'
    ]
    
    ensemble_run(path_to_models=path_to_models, dm=dm)