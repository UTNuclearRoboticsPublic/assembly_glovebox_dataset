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

def adaptive_calibration_error(y_pred, y_true, num_bins=15):
    """
    Overall structure

    Flatten and sort values in each class array out of the softmax -> [2, 3, 900]
    Flatten and sort the values in the ground truth tensor as well -> [2, 3, 900]
    Flatten and sort the values in the argmaxes predictions -> [2, 3, 900]
    ^ all three should have matching values from the input image for accuracy predictions

    bin these values into 15 bins, so resulting shape for each is [2, 3, 15, 60] where each
    indice of the 15 bins contains the respective sorted probabilities for that bin (do the same for the target and argmaxed tensor)

    calculate confidences for each bin
    calculate accuracy for each bin
    sum across classes, then sum across bins, then across num_examples, and divide by (num_ranges * num_examples * num_classes)
    """

    """## Initializing Inputs"""

    probs = torch.softmax(y_pred, dim=1)
    confidences, y_hats = torch.max(probs, dim=1) # respective values (max value along each row), indices (argmax return)

    num_classes = y_pred.shape[1]

    """## Sorting softmax prediction and ground truth values (BEFORE BINNING)"""

    num_ranges = 23

    flattened = torch.flatten(probs, start_dim=2)
    sorted_pred, sort_indices = torch.sort(flattened, dim=1)

    """Doing this for the ground truth, using the indices from sorting the probabilities to sort this correctly"""

    ground = torch.nn.functional.one_hot(y_true.long(), num_classes=3)
    ground_truth = ground.permute(0,-1, 1, 2)

    flat_gt = torch.flatten(ground_truth, start_dim=2)

    sorted_gt = torch.gather(flat_gt, dim=1, index=sort_indices)
    sorted_gt.shape

    """Doing this for the argmaxed predictions (using indices from sorted probabilities)"""

    max_ground = torch.nn.functional.one_hot(y_hats.long(), num_classes=3)
    ground_arg = max_ground.permute(0,-1, 1, 2)
    flat_arg = torch.flatten(ground_arg, start_dim=2)
    sorted_arg = torch.gather(flat_arg, dim=1, index=sort_indices)
    sorted_arg.shape

    """## Binning softmax, ground truth values"""

    chunked_preds = torch.chunk(sorted_pred, num_ranges, dim=2)
    binned_preds = torch.stack(chunked_preds, dim=2)


    chunked_gt = torch.chunk(sorted_gt, num_ranges, dim=2)
    binned_gt = torch.stack(chunked_gt, dim=2)
    binned_gt.shape

    chunked_arg = torch.chunk(sorted_arg, num_ranges, dim=2)
    binned_arg = torch.stack(chunked_arg, dim=2)
    

    """## Calculating confidences"""

    # calculate the mean confidence for each binning range for each
    confidences = torch.mean(binned_preds, dim=3)
    

    """## Calculating accuracy for each binning range"""

    differences = torch.abs(binned_arg - binned_gt)
    nonzero_per_bin = torch.count_nonzero(differences, dim=3)

    acc = nonzero_per_bin / binned_gt.shape[-1]
    

    """## Difference between accuracy and confidences"""

    error = torch.abs(confidences - acc)

    """## Summing across ranges, classes, and examples"""

    range_sum = torch.sum(error, dim=2)

    class_sum = torch.sum(range_sum, dim=1)

    num_sum = torch.sum(class_sum, dim = 0)

    ace = num_sum / (2 * num_classes * num_ranges)
    return ace


