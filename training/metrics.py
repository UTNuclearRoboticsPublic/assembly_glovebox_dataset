from scipy.stats import entropy
import numpy as np
from torchmetrics import CalibrationError
from torch.nn import functional as F

def predictive_entropy(preds, mean=False):
    # base defaults to natural log
    probs = F.softmax(preds, dim=1)
    entropy = np.apply_along_axis(entropy, 1, probs)
    return np.mean(entropy) if mean else entropy

def adaptive_calibration_error(preds, targets):
    num_bins = 15
    b = np.linspace(start=0, stop=1, num=num_bins)
    b = np.quantile(preds, b)
    b = np.unique(b)
    num_bins = len(b)

    # norm 'l1' coresponds with expected calibration error
    ece = CalibrationError(task='multiclass', num_classes=3, n_bins=b, norm='l1')
    ace = ece(preds, int(targets))
    return ace