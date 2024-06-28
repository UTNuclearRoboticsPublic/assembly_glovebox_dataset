from training.lightning_trainers.lightning_model import LitModel
from ..tl_models.bisenet_v2 import BiSeNetV2
import torchmetrics
import torch.nn.functional as F
import torch


# like UNet, just check the declarations
class BiSeNetV2Model(LitModel):
    def __init__(self, learning_rate=0.001, weight_decay=0.1, test_dropout=False):
        super(BiSeNetV2Model, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.iou = torchmetrics.classification.BinaryJaccardIndex()


        self.model = BiSeNetV2(n_classes=1)

        self.test_dropout = test_dropout

    def get_loss(self, raw_preds, y):
        y1, y2 = y
        criterion = torch.nn.BCEWithLogitsLoss()
        losses_1 = []
        losses_2 = []
        loss = None
        if self.model.training:
            for preds in raw_preds:
                # print(f"the shape of preds is {preds.shape} and the shape of y1 is {y1.shape}") # [64,3,160,160] [64, 160, 160]
                preds = torch.squeeze(preds,1)
                loss_1 = criterion(preds, y1.float())
                loss_2 = criterion(preds, y2.float())
                losses_1.append(loss_1)
                losses_2.append(loss_2)
            loss = (sum(losses_1) + sum(losses_2)) / 2

        else:
            raw_preds = torch.squeeze(raw_preds,1)

            loss_1 = criterion(raw_preds, y1.float())
            loss_2 = criterion(raw_preds, y2.float())
            loss = (loss_1 + loss_2) / 2
        return loss 