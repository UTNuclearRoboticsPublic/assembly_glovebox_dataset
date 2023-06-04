import torch
import numpy as np
from torchmetrics import CalibrationError

def get_batch(batch_idx, y_true, y_hat, confidences):
    "Returns batch from given index for values"
    y = y_true[batch_idx, :, :]
    y_hat = y_hat[batch_idx, :, :] 
    confidence = confidences[batch_idx, :, :]

    return y, y_hat, confidence

def sort_probs(confidence):
    "Return a 1D tensor of sorted probabilities from given confidence"

    flat = torch.flatten(confidence)
    sorted, indices = torch.sort(flat)

    return sorted
 
def get_accuracy(class_idx, upper, lower, y, confidence_values, pred_values, sort):

    # find upper and lower range probability bounds from indexes
    lower = sort[int(lower.item())]
    upper = sort[int(upper.item())]
    
    # filter confidence tensor for prediction probabilities only for elements of the current class
    # then, return the indices where these filtered values are in the matrix
    fil_confidence = torch.where(y == class_idx, confidence_values, 0)
    fil_idxs = torch.where((fil_confidence >= lower) & (fil_confidence <= upper))

    # get values of the ground truth and predicted values only
    # for what was filtered above
    fil_truth = y[fil_idxs]
    fil_pred_values = pred_values[fil_idxs]

    # calculate the accuracy for the given class for this bin
    diff = fil_pred_values - fil_truth
    num_correct = diff.shape[0] - torch.count_nonzero(diff).item()
    acc = torch.abs(torch.tensor(num_correct / fil_truth.shape[0]))

    return acc

def ace(y_true, num_ranges,confidences, y_hats, num_classes):
    """
    Params:
        - y_true -> [N, H, W] (ground truth labels)
        - num_ranges -> (number of bins requested)
        - confidences -> [N, H, W] (predicted probabilities in same shape as y_hats)
        - y_hats -> [N, H, W] (predicted class indices after argmax)

    Returns:
        - ace -> adapative calibration error averaged across the batches 
    """
    
    ace = 0
    batch_size = y_true.shape[0]

    for batch_idx in range(batch_size):
        y, y_hat, confidence = get_batch(batch_idx, y_true, y_hats, confidences)

        # total number of values for each respective class = (W * H)
        num_vals = y.shape[-2] * y.shape[-1]

        # total sum of calibration errors
        sum = 0

        for class_idx in range(num_classes):
            
            # find distribution of probabilities for creating adaptive ranges
            sort = sort_probs(confidence)

            # elements per range
            el_per_r = (int)(num_vals/num_ranges)

            # find whole range endpoints
            ranges = torch.linspace(el_per_r, num_vals-1, num_ranges)

            prev_r = torch.tensor(0)
            for r in ranges:
                # find range index bounds
                upper = r
                lower = prev_r

                # calculate average confidence of bin and accuracy
                acc = get_accuracy(class_idx, upper, lower, y, confidence, y_hat, sort)
                conf = torch.mean(sort)

                # find calibration error just for this range for this class
                # then, add to sum of cal_errors
                cal_error = torch.abs(acc - conf)
                sum+=cal_error.item()
                
                prev_r = r

        # calculate ACE after multiplying sum by 1/KR
        average = 1 / (num_ranges * num_classes)
        ace += average * sum

    # return average ACE for all the batches
    return ace/batch_size

def main(raw_preds, y_true, num_ranges):
    """
    Params:
        - raw_preds -> [N, C, H, W] (predictions straight from the model, before softmax)
        - y_true -> [N, H, W] (ground truth labels)

    Returns:
        - ace -> adapative calibration error averaged across the batches 
    """

    probs = torch.softmax(raw_preds, dim=1)

    # y_hats are predictions after being sent through argmax
    # confidences are the probabilities of the respective predictions (same shape as y_hats)
    confidences, y_hats = torch.max(probs, dim=1)

    num_classes = raw_preds.shape[1]

    pred_ace = ace(
                y_true = y_true, 
                num_ranges = num_ranges, 
                confidences = confidences, 
                y_hats = y_hats, 
                num_classes = num_classes
            )

    return pred_ace



if __name__ == '__main__':

    # creating random tensor of demo input size for debugging (raw_preds and ground truth)
    raw_preds = torch.rand(2, 3, 25, 25)
    y_true = torch.argmax(torch.rand(2, 3, 25, 25), dim=1)
    
    ace = main(raw_preds, y_true, num_ranges=15)
    
    print(ace)

    # this is for cross-referencing what the ece looks like
    # ece = CalibrationError(task='multiclass', num_classes=3, n_bins=15, norm='l1')
    # print(ece(torch.softmax(raw_preds, dim=1), y_true))
