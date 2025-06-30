import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # shape (8, 1, 224, 224)
        
        # pred is expected to be logits, so apply sigmoid/softmax
        pred = torch.softmax(pred, dim=1)
        # Flatten both tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

def calculate_iou(pred, target, num_classes):
    iou_scores = []
    # Convert predictions to class indices
    pred = torch.argmax(pred, dim=1)
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            # If there is no ground truth or prediction, it's a perfect match
            iou_scores.append(float('nan')) 
        else:
            iou_scores.append(intersection / union)
    return torch.tensor(np.nanmean(iou_scores)) # Return mean IoU
