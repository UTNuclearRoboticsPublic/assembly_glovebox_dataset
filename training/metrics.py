from scipy.stats import entropy
import numpy as np
from torchmetrics import CalibrationError
from torch.nn import functional as F
import torch

def predictive_entropy(preds, mean=True):
    # base defaults to natural log
    probs = F.softmax(preds, dim=1)
    pred_entropy = np.apply_along_axis(entropy, 1, probs.cpu().numpy())
    return np.mean(pred_entropy) if mean else pred_entropy

def expected_calibration_error(preds, targets):
    num_bins = 15
    b = np.linspace(start=0, stop=1, num=num_bins)
    b = np.quantile(preds, b)
    b = np.unique(b)
    num_bins = len(b)

    # norm 'l1' coresponds with expected calibration error
    ece = CalibrationError(task='multiclass', num_classes=3, n_bins=b, norm='l1')
    ace = ece(preds, int(targets))
    return ace

def adaptive_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)


    b = np.linspace(start=0, stop=1.0, num=num_bins)
    b = np.quantile(prob_y, b)
    b = np.unique(b)
    num_bins = len(b)
    bins = np.digitize(prob_y, bins=b, right=True)


    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]

# def adaptive_calibration_error2(y, raw_preds, ranges=15):

#     predictions = torch.argmax(raw_preds, dim=1) # shape [B, N, W, H]
#     probs = torch.softmax(raw_preds, dim=1)
#     # y - shape [B, N, W, H]

#     # 1/KR
#     classes = raw_preds.shape[1]
#     factor = 1 / (15 * classes)

#     # [N/R] for each image
#     range_idx = torch.numel(predictions) / torch.shape[0]

#     for img in range(raw_preds.shape[0]):
#     # for each class
#         for c in range(classes):
#             preds = predictions[img, c, :, :]
#             one_dim_preds = torch.flatten(preds)
            
#             # figure out how to do bins here...
#             N = torch.numel(predictions[img, c, :, :])
#             range_iter = torch.linspace(N, ranges)

#             # for each bin
#             for r in range_iter:
#                 one_dim_preds[]
#                 pass

