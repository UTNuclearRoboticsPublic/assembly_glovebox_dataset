from scipy.stats import entropy
import numpy as np
from torchmetrics import CalibrationError
from torch.nn import functional as F
import torch
from netcal.metrics.confidence import ECE, ACE 

def predictive_entropy(preds, mean=True):
    # base defaults to natural log
    print(f".....preds shape is {preds.shape}")
    probs = F.softmax(preds, dim=1)

    probs = probs.permute(0, 2, 3, 1).reshape(probs.shape[0], -1, probs.shape[1])


    pred_entropy = entropy(probs.cpu().numpy(), base=2, axis=-1)

    print(f"........the shape of entropy is {np.shape(pred_entropy)}")
    return_ent = pred_entropy
    return list(np.mean(return_ent, axis=1)) if mean else pred_entropy