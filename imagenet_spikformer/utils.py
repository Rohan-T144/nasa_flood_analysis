import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        # shape (8, 1, 224, 224)
        
        # pred is expected to be logits, so apply sigmoid/softmax
        pred_prob = torch.sigmoid(pred_logits)

        # Flatten both tensors
        pred_prob = pred_prob.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred_prob * target).sum()
        dice = (2. * intersection + self.smooth) / (pred_prob.sum() + target.sum() + self.smooth)
        
        return 1 - dice

# def calculate_iou(pred, target, num_classes):
#     iou_scores = []
#     # Convert predictions to class indices
#     pred = torch.argmax(pred, dim=1)
#     pred = pred.view(-1)
#     target = target.view(-1)

#     for cls in range(num_classes):
#         pred_inds = (pred == cls)
#         target_inds = (target == cls)
#         intersection = (pred_inds[target_inds]).long().sum().item()
#         union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
#         if union == 0:
#             # If there is no ground truth or prediction, it's a perfect match
#             iou_scores.append(float('nan')) 
#         else:
#             iou_scores.append(intersection / union)
#     return torch.tensor(np.nanmean(iou_scores)) # Return mean IoU

def calculate_iou(pred_logits, target, num_classes, threshold=0.5):
    """
    Calculates IoU for binary segmentation.
    Expects raw logits as input.
    """
    # Apply sigmoid and threshold to get binary predictions
    pred_prob = torch.sigmoid(pred_logits)
    pred_mask = (pred_prob > threshold).long() # Convert to 0s and 1s

    pred_mask = pred_mask.view(-1)
    target = target.view(-1)

    # For binary case (num_classes=1), we calculate IoU for the positive class (class 1)
    # This assumes the target mask is also 0s and 1s.
    intersection = (pred_mask & target).sum().item()
    union = (pred_mask | target).sum().item()
    
    if union == 0:
        # If there is no ground truth or prediction for the positive class,
        # it can be considered a perfect match (IoU=1) or nan.
        # Returning nan is safer to indicate the case.
        iou = float('nan')
    else:
        iou = intersection / union
        
    # Since it's binary, we return the single IoU score.
    # The original loop was for multi-class, which is not needed here.
    return torch.tensor(iou)

