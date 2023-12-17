from training.lightning_trainers.lightning_model import LitModel
from ..models.bisenet_v2 import BiSeNetV2
import torchmetrics
import torch.nn.functional as F



# like UNet, just check the declarations
class BiSeNetV2Model(LitModel):
    def __init__(self, learning_rate=0.001, weight_decay=0.1, test_dropout=False):
        super(BiSeNetV2Model, self).__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

        self.model = BiSeNetV2(n_classes=3)

        self.test_dropout = test_dropout

    def get_loss(self, raw_preds, y):
        y1, y2 = y

        losses_1 = []
        losses_2 = []
        loss = None

        if self.model.training:
            for preds in raw_preds:
                print(f"the shape of preds is {preds.shape} and the shape of y1 is {y1.shape}") # [64,3,160,160] [64, 160, 160]
                loss_1 = F.cross_entropy(preds, y1.long())
                loss_2 = F.cross_entropy(preds, y2.long())
                losses_1.append(loss_1)
                losses_2.append(loss_2)
            loss = (sum(losses_1) + sum(losses_2)) / 2

        else:
            loss_1 = F.cross_entropy(raw_preds, y1.long())
            loss_2 = F.cross_entropy(raw_preds, y2.long())
            loss = (loss_1 + loss_2) / 2
        return loss 