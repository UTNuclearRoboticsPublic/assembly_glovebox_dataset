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

# def adaptive_calibration_error(y_pred, y_true, num_bins=15):
#     """
#     Overall structure

#     Flatten and sort values in each class array out of the softmax -> [2, 3, 900]
#     Flatten and sort the values in the ground truth tensor as well -> [2, 3, 900]
#     Flatten and sort the values in the argmaxes predictions -> [2, 3, 900]
#     ^ all three should have matching values from the input image for accuracy predictions

#     bin these values into 15 bins, so resulting shape for each is [2, 3, 15, 60] where each
#     indice of the 15 bins contains the respective sorted probabilities for that bin (do the same for the target and argmaxed tensor)

#     calculate confidences for each bin
#     calculate accuracy for each bin
#     sum across classes, then sum across bins, then across num_examples, and divide by (num_ranges * num_examples * num_classes)
#     """

#     """## Initializing Inputs"""

#     print(f"the shape of y pred is {y_pred.shape} and true is {y_true.shape}")

#     probs = torch.softmax(y_pred, dim=1)
#     confidences, y_hats = torch.max(probs, dim=1) # respective values (max value along each row), indices (argmax return)

#     num_classes = y_pred.shape[1]

#     """## Sorting softmax prediction and ground truth values (BEFORE BINNING)"""

#     num_ranges = 16 # has to be divisible by the image size (no workaround yet -> img size is 256 so we are using 16 as a factor)

#     flattened = torch.flatten(probs, start_dim=2)
#     sorted_pred, sort_indices = torch.sort(flattened, dim=1)

#     """Doing this for the ground truth, using the indices from sorting the probabilities to sort this correctly"""

#     ground = torch.nn.functional.one_hot(y_true.long(), num_classes=3)
#     ground_truth = ground.permute(0,-1, 1, 2)

#     flat_gt = torch.flatten(ground_truth, start_dim=2)

#     sorted_gt = torch.gather(flat_gt, dim=1, index=sort_indices)
#     sorted_gt.shape

#     """Doing this for the argmaxed predictions (using indices from sorted probabilities)"""

#     max_ground = torch.nn.functional.one_hot(y_hats.long(), num_classes=3)
#     ground_arg = max_ground.permute(0,-1, 1, 2)
#     flat_arg = torch.flatten(ground_arg, start_dim=2)
#     sorted_arg = torch.gather(flat_arg, dim=1, index=sort_indices)
#     sorted_arg.shape

#     """## Binning softmax, ground truth values"""

#     chunked_preds = torch.chunk(sorted_pred, num_ranges, dim=2)
#     binned_preds = torch.stack(chunked_preds, dim=2)


#     chunked_gt = torch.chunk(sorted_gt, num_ranges, dim=2)
#     binned_gt = torch.stack(chunked_gt, dim=2)
#     binned_gt.shape

#     chunked_arg = torch.chunk(sorted_arg, num_ranges, dim=2)
#     binned_arg = torch.stack(chunked_arg, dim=2)
    

#     """## Calculating confidences"""

#     # calculate the mean confidence for each binning range for each
#     confidences = torch.mean(binned_preds, dim=3)
    

#     """## Calculating accuracy for each binning range"""

#     differences = torch.abs(binned_arg - binned_gt)
#     nonzero_per_bin = torch.count_nonzero(differences, dim=3)

#     acc = nonzero_per_bin / binned_gt.shape[-1]
    

#     """## Difference between accuracy and confidences"""

#     error = torch.abs(confidences - acc)

#     """## Summing across ranges, classes, and examples"""

#     range_sum = torch.sum(error, dim=2)

#     class_sum = torch.sum(range_sum, dim=1)

#     num_sum = torch.sum(class_sum, dim = 0)

#     ace = num_sum / (2 * num_classes * num_ranges)
#     return ace

# def adaptive_calibration_error_2(y_pred, y_true, num_bins=15):
#     bs = 128
#     num_classes = 3
#     img_h = img_w = 256

#     print(f"the shape of y pred is {y_pred.shape} and true is {y_true.shape}")
#     # the shape of y pred is torch.Size([128, 3, 256, 256]) and true is torch.Size([128, 256, 256])


#     # y_pred = torch.rand(bs, num_classes, img_h, img_w)
#     y_pred = y_pred.to(torch.float32).cpu()

#     print(y_pred.dtype, y_pred.shape)
#     flattened_y_pred = torch.flatten(y_pred, start_dim=2) # dims: [bs, num_classes, img_resolution (img_h*img_w)]

#     # y_gt = torch.round((num_classes-1) * torch.rand(bs, img_h, img_w))
#     # y_gt = torch.round((num_classes-1) * torch.rand(bs, img_h, img_w))
#     y_gt = y_true.to(torch.float32).cpu()
#     flattened_y_gt = torch.flatten(y_gt, start_dim=1) # dims: [bs, num_classes, img_resolution (img_h*img_w)]

#     print(y_gt.dtype, y_gt.shape)

#     # torch.float32 torch.Size([128, 3, 256, 256])
#     # torch.float32 torch.Size([128, 256, 256])

#     # ace_class = ACE(bins=20, equal_intervals=False)
#     ece_class = ECE(bins=15)

#     avg_ace_over_batch = 0
#     avg_ece_over_batch = 0
#     for item in range(flattened_y_pred.shape[0]): # iterate through batches
#         curr_pred = flattened_y_pred[item] # initially num_classes, 4096
#         print(curr_pred.shape)
#         test = torch.transpose(curr_pred, 0, 1).numpy()
#         print(test.shape)
#         print(test.dtype)

#         curr_gt = flattened_y_gt[item].numpy()
#         print(curr_gt.shape)
#         print(curr_gt.dtype)

#     # avg_ace_over_batch += ace_class.measure(test, curr_gt)
#     avg_ece_over_batch += ece_class.measure(test, curr_gt)
#     print(f"avg ace: {avg_ace_over_batch/flattened_y_pred.shape[0]}")
#     print(f"avg ece: {avg_ece_over_batch/flattened_y_pred.shape[0]}")

#     return avg_ece_over_batch/flattened_y_pred.shape[0]