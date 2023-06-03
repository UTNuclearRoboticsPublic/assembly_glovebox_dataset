import torch
import numpy as np


raw_preds = torch.rand(2, 3, 25, 25)
y_true = torch.argmax(torch.rand(2, 3, 25, 25), dim=1)
num_ranges = 15


probs = torch.softmax(raw_preds, dim=1)

# predictions are argmax predictions
# confidences are the probabilities of the respective predictions
confidences, y_hat = torch.max(probs, dim=1)

classes = raw_preds.shape[1]


# 
def get_batch(batch_idx, y_true, y_hat, confidences):
    y = y_true[batch_idx, :, :]
    y_hat = y_hat[batch_idx, :, :] 
    confidence = confidences[batch_idx, :, :]

    return y, y_hat, confidence

# 
def sort_probs(confidence):
    flat = torch.flatten(confidence)
    sorted_one_hot, indices = torch.sort(flat)

    return sorted_one_hot

# 
def get_accuracy(class_idx, upper, lower, y, confidence_values, pred_values):
    
    # FIND PREDICTED VALUES IN THE RANGE
    # filter confidence and argmaxed predictions for values in the range of the bin
    # also, return the indices where these filtered values are in the matrix
    fil_confidence = torch.where(y == class_idx, confidence_values, 0)

    # problem here is the empty tensor for otherwise
    fil_pred_values = torch.where((fil_confidence >= lower) & (fil_confidence <= upper), pred_values, torch.tensor([]))
    fil_idxs = fil_confidence.where((fil_confidence >= lower) & (fil_confidence <= upper))

    # FIND GROUND TRUTH VALUES IN THE RANGE
    fil_truth = y[fil_idxs]

    # CALCULATE THE ACCURACY FOR THIS RANGE FOR THE GIVEN CLASS
    diff = fil_pred_values - fil_truth
    num_correct = diff.size() - torch.count_nonzero(diff)
    acc = torch.abs(num_correct / fil_truth.size())


    return acc

batch_size = raw_preds.shape[0]
# make handling batches happen in a function
for batch_idx in range(batch_size):
    y, y_hat, confidence = get_batch(batch_idx, y_true, y_hat, confidences)

    # number of values for each class
    num_vals = y.shape[-2] * y.shape[-1]

    # sum of calibration errors for every range for every class
    sum = 0

    for class_idx in range(classes):
        # find total range of probabilities of class in image and sort
        sort = sort_probs(confidence)

        # split into ranges
        ind = (int)(num_vals/num_ranges)
        ranges = torch.linspace(ind, num_vals, ind)

        prev_r = 0
        for r in ranges:
            upper = r
            lower = prev_r

            # calculate confidence and accuracy
            acc = get_accuracy(class_idx, upper, lower, y, confidence, y_hat)
            conf = torch.mean(sort)

            # difference for this range
            cal_error = torch.abs(acc - conf)
            sum+=cal_error
            
            prev_r = r